import json
from enum import Enum
import cv2
import numpy as np
import torch
import quaternion

import rospy

from geometry_msgs.msg import Point, Pose, Quaternion

from utils import OPENCV_TO_OPENGL
from scripts.nodes import GetDatasetConfigResponse

class PoseDataType(Enum):
    C2W_OPENCV = 'C2W_OPENCV'
    C2W_OPENGL = 'C2W_OPENGL'
    W2C_OPENCV = 'W2C_OPENCV'
    W2C_OPENGL = 'W2C_OPENGL'
    
class HeightDirection(Enum):
    X_NEGATIVE = 0
    X_POSITIVE = 1
    Y_NEGATIVE = 2
    Y_POSITIVE = 3
    Z_NEGATIVE = 4
    Z_POSITIVE = 5

class DatasetFormats(Enum):
    STRCAM = 'strcam'
    SIMCAM = 'simcam'

CV_FILENODE_TYPE = {
    cv2.FILE_NODE_NONE: 'FILE_NODE_NONE',
    cv2.FILE_NODE_INT: 'FILE_NODE_INT',
    cv2.FILE_NODE_REAL: 'FILE_NODE_REAL',
    cv2.FILE_NODE_STRING: 'FILE_NODE_STRING',
    cv2.FILE_NODE_SEQ: 'FILE_NODE_SEQ',
    cv2.FILE_NODE_MAP: 'FILE_NODE_MAP',
    cv2.FILE_NODE_TYPE_MASK: 'FILE_NODE_TYPE_MASK',
    cv2.FILE_NODE_FLOAT: 'FILE_NODE_FLOAT',
    cv2.FILE_NODE_FLOW: 'FILE_NODE_FLOW',
    cv2.FILE_NODE_STR: 'FILE_NODE_STR',
    cv2.FILE_NODE_UNIFORM: 'FILE_NODE_UNIFORM',
    cv2.FILE_NODE_EMPTY: 'FILE_NODE_EMPTY',
    cv2.FILE_NODE_NAMED: 'FILE_NODE_NAMED'
}
print(json.dumps(CV_FILENODE_TYPE, indent=4))

def readSeqFileNode(file_node:cv2.FileNode):
    assert(file_node.isSeq())
    res = []
    for i in range(file_node.size()):
        res.append(file_node.at(i).real())
    return res

def readMapFileNode(file_root:cv2.FileNode, deep=False):
    res = dict()
    for key in file_root.keys():
        file_node = file_root.getNode(key)
        file_node_type = file_node.type()
        rospy.logdebug("%s %s :", key, CV_FILENODE_TYPE[file_node_type])
        if file_node_type == cv2.FILE_NODE_INT:
            res[key] = int(file_node.real())
        elif file_node_type == cv2.FILE_NODE_REAL:
            res[key] = file_node.real()
        elif file_node_type == cv2.FILE_NODE_STRING:
            res[key] = file_node.string()
        elif file_node_type == cv2.FILE_NODE_SEQ:
            res[key] = readSeqFileNode(file_node)
        elif file_node_type == cv2.FILE_NODE_MAP:
            res[key] = readMapFileNode(file_node)
            if not deep:
                res[key] = np.reshape(np.array(res[key]["data"]), (int(res[key]["rows"]), int(res[key]["cols"])))
        elif file_node_type == cv2.FILE_NODE_TYPE_MASK:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_FLOAT:
            res[key] = file_node.real()
        elif file_node_type == cv2.FILE_NODE_FLOW:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_STR:
            res[key] = file_node.string()
        elif file_node_type == cv2.FILE_NODE_UNIFORM:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_EMPTY:
            raise NotImplementedError
        elif file_node_type == cv2.FILE_NODE_NAMED:
            raise NotImplementedError
        else:
            raise NotImplementedError
        rospy.logdebug(res[key])
    return res

def readCameraCfg(yamlpath):
    cv_file = cv2.FileStorage(yamlpath, cv2.FILE_STORAGE_READ)
    res = readMapFileNode(cv_file.root())
    return res

class RGBDSensor:
    
    def __init__(self,
                 height:int,
                 width:int,
                 fx:float,
                 fy:float,
                 cx:float,
                 cy:float,
                 dist:np.ndarray,
                 depth_min:float,
                 depth_max:float,
                 depth_scale:float,
                 position:np.ndarray,
                 downsample_factor:float=1.0):
        '''
        depth_max: unit in meters
        depth_min: unit in meters
        '''
        if downsample_factor > 1.0:
            self.height = int(np.ceil(height / downsample_factor))
            self.width = int(np.ceil(width / downsample_factor))
            self.fx = fx * self.width / width
            self.fy = fy * self.height / height
            self.cx = cx * self.width / width
            self.cy = cy * self.height / height
        elif downsample_factor == 1.0:
            self.height = int(height)
            self.width = int(width)
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
        else:
            raise ValueError(f"Invalid downsample factor: {downsample_factor}")
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.depth_scale = depth_scale
        self.position = position
        self.downsample_factor = downsample_factor
        self.directions = get_camera_rays(self.height, self.width, self.fx, self.fy, self.cx, self.cy)
        self.intrinsics = np.array([self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3)
        self.hfov = 2 * np.arctan(self.width / (2 * self.fx))
        self.vfov = 2 * np.arctan(self.height / (2 * self.fy))
        self.dist = dist
        
    def point2img(self, pt:np.ndarray):
        """ 3D point to image coordinate """
        img_pt = np.array([round((pt[0] / pt[2]) * self.fx + self.cx), round((pt[1] / pt[2]) * self.fy + self.cy)])
        return img_pt
    
    def points2img(self, points):
        # points: shape (N, 3)
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        u = (fx * x / z) + cx
        v = (fy * y / z) + cy
        img_pts = np.vstack((u, v)).T  # shape: (N, 2)
        return img_pts
    
    def is_in_fov(self, img_pt:np.ndarray):
        if img_pt[0] < 0 or img_pt[1] < 0 or img_pt[0] >= self.width or img_pt[1] >= self.height:
            return False
        else:
            return True
        
# ROS conversion functions

def dataset_config_to_ros(dataset_config:dict) -> GetDatasetConfigResponse:
    dataset_config_ros = dataset_config.copy()
    for key, value in dataset_config.items():
        if issubclass(type(value), (int, float, str)):
            pass
        elif isinstance(value, np.ndarray):
            if value.shape == (3, ):
                dataset_config_ros[key] = Point(*value)
            elif value.shape == (4, 4):
                value_ros = Pose()
                value_ros.position = Point(*value[:3, 3])
                value_ros.orientation = Quaternion(
                    *np.roll(
                        quaternion.as_float_array(quaternion.from_rotation_matrix(value[:3, :3])),
                        -1))
                dataset_config_ros[key] = value_ros
            elif value.shape == (5, 1) or value.shape == (5, ):
                # camera distortion
                dataset_config_ros[key] = [value[0], value[1], value[2], value[3], value[4]]
            else:
                raise ValueError(f'Invalid shape of {key}, get {type(value)}')
        else:
            raise ValueError(f'Invalid type of {key}, get {type(value)}')
    return GetDatasetConfigResponse(**dataset_config_ros)

def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type == 'OpenGL':
        dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
    elif type == 'OpenCV':
        dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d

def compute_intrinsics(width, height, hfov_rad, vfov_rad=None):
    fx = 0.5 * width / np.tan(hfov_rad / 2.)
    if vfov_rad is None:
        # NOTE: In habitat, fx is equal to fy
        fy = fx
    else:
        fy = 0.5 * height / np.tan(vfov_rad / 2.)
    cx = width / 2 - 1
    cy = height / 2 - 1
    return fx, fy, cx, cy
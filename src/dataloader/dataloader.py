import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
import threading
import time
from typing import Tuple, Union, Dict
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import open3d as o3d
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf

from utils import OPENCV_TO_GAZEBO
from utils.gui_utils import pose_to_matrix
from utils.pc_utils import depth_to_pointcloud
from src.dataloader import RGBDSensor, DatasetFormats, PoseDataType, HeightDirection, readCameraCfg
from utils.ros_utils import call_capture_service, call_get_camera_pose_service, call_get_turntable_pose_service

class RobotScanDataset(Dataset):
    
    __pose_data_type = PoseDataType.C2W_OPENCV
    __height_direction = HeightDirection.Z_POSITIVE
    
    def __init__(self, config:dict, object_id:str, remark:str) -> None:
        self.scene_bbox = np.array(config['dataset']['bbox'])
        self.sensor_config_url = config['sensor']['config']
        self.depth_scale = config['dataset']['depth_scale']
        self.capture_num = config['dataset']['capture_num']
        self.rgbd_sensor_downsample_factor = config['dataset']['downsample']
        self.with_arm_turntable = config['dataset']['with_arm_turntable']
        self.object_id = object_id
        self.remark = remark
        self.frame_id = 0
        self.capture_times = 0
        self.finished_flag = False
        self.cur_data_path = ''
        self.auto_stoped = False
        
        dataset_format = DatasetFormats(config['dataset']['format'])
        
        results_dir_name = time.strftime('%Y-%m-%d_%H-%M-%S') + f'_{dataset_format.value}_{self.object_id}'
        if self.remark != 'None':
            results_dir_name += f'_{self.remark}'
        self.results_dir = os.path.join(PACKAGE_PATH, 'results', results_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)
        self.config = config
        
    def setup(self) -> Dict[str, Union[float, int, str, np.ndarray]]:
        sensor_config = readCameraCfg(self.sensor_config_url)
        assert sensor_config['DepthMapFactor'] == self.depth_scale, f"Depth scale mismatch: {sensor_config['DepthMapFactor']} != {self.depth_scale}"
        dataset_config = {
            'rgbd_height':             int(sensor_config['eRgbCamSize'][1]),
            'rgbd_width':              int(sensor_config['eRgbCamSize'][0]),
            'rgbd_fx':                 sensor_config['eRgbCamK'][0][0],
            'rgbd_fy':                 sensor_config['eRgbCamK'][1][1],
            'rgbd_cx':                 sensor_config['eRgbCamK'][0][2],
            'rgbd_cy':                 sensor_config['eRgbCamK'][1][2],
            'rgbd_dist':               sensor_config['eRgbCamDist'],
            'rgbd_depth_max':          sensor_config['DepthMax'],
            'rgbd_depth_min':          sensor_config['DepthMin'],
            'rgbd_depth_scale':        sensor_config['DepthMapFactor'],
            'rgbd_downsample_factor':  self.rgbd_sensor_downsample_factor,
            'focal_length':            sensor_config['eRgbFocalLength'],
            'scene_bound_min':         np.array(self.scene_bbox[0]),
            'scene_bound_max':         np.array(self.scene_bbox[1]),
            'pose_data_type':          self.__pose_data_type.value,
            'height_direction':        int(self.__height_direction.value),
            'results_dir':             self.results_dir
        }
        return dataset_config
        
    def get_frame(self) -> dict:
        success, cur_data_path = call_capture_service()
        if success and cur_data_path != '':
            rospy.loginfo(f'Capture success! cur_data_path: {cur_data_path}')
            rospy.sleep(1)
            
            # Read RGB and depth data
            scene_rgb_np = cv2.imread(os.path.join(cur_data_path, "out_rgb.png"))
            scene_rgb_np = cv2.cvtColor(scene_rgb_np, cv2.COLOR_BGR2RGB)
            scene_pcd_path = os.path.join(cur_data_path, "scene.pcd")
            pcd = o3d.io.read_point_cloud(scene_pcd_path)
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)
            
            # Get poses
            cur_cam_pose = call_get_camera_pose_service()
            cur_turntable_pose, _ = call_get_turntable_pose_service()
            cam2base = pose_to_matrix(cur_cam_pose)
            turntable2base = pose_to_matrix(cur_turntable_pose)
            cam2turntable = np.linalg.inv(turntable2base) @ cam2base
            
            self.cur_data_path = cur_data_path
        else:
            rospy.logerr('Capture failed!')
            return {}, ''
        
        ret = {
            'frame_id': self.frame_id,
            'c2w': cam2turntable.astype(np.float32),
            'scene_rgb': scene_rgb_np,
            'scene_pcd_points': pcd_points,
            'scene_pcd_colors': pcd_colors
        }
        self.frame_id += 1
        
        return ret, cur_data_path
    
    def extract_object_point_cloud(self, rgbd_sensor: RGBDSensor, cur_frame_np: Dict[str, Union[np.ndarray, torch.Tensor]], masks_image: np.ndarray):
        mask_height, mask_width = masks_image.shape


        points = cur_frame_np['scene_pcd_points']
        colors = cur_frame_np['scene_pcd_colors']
        rgb_image = cur_frame_np['scene_rgb']

        img_pts = rgbd_sensor.points2img(points)  # shape: (N, 2)
        img_x = img_pts[:, 0].astype(np.int32)
        img_y = img_pts[:, 1].astype(np.int32)

        valid_mask = (img_x >= 0) & (img_x < mask_width) & \
                    (img_y >= 0) & (img_y < mask_height)

        valid_points = points[valid_mask]
        valid_colors = colors[valid_mask]
        valid_img_x = img_x[valid_mask]
        valid_img_y = img_y[valid_mask]
        # scene_depth_np[valid_img_y, valid_img_x] = valid_points[:, 2]
        object_ids = masks_image[valid_img_y, valid_img_x]

        object_mask = object_ids > 0
        object_ids = object_ids[object_mask]
        valid_points = valid_points[object_mask]
        valid_colors = valid_colors[object_mask]
        valid_img_x = valid_img_x[object_mask]
        valid_img_y = valid_img_y[object_mask]

        unique_object_ids = np.unique(object_ids)

        objs_pc = {}
        objs_depth_np = {}
        objs_rgb_np = {}
        objs_mask = {}

        for obj_id in unique_object_ids:
            objs_depth_np[obj_id] = np.zeros((mask_height, mask_width), dtype=np.float32)
            objs_rgb_np[obj_id] = np.ones_like(rgb_image).astype(np.uint8) * 255
            objs_mask[obj_id] = np.zeros((mask_height, mask_width), dtype=np.uint8)

        depth_kernel = np.ones((3, 3))
        for obj_id in unique_object_ids:
            obj_mask = object_ids == obj_id
            objs_pc[obj_id] = {
                'points': valid_points[obj_mask],
                'colors': valid_colors[obj_mask]
            }
            objs_depth_np[obj_id][valid_img_y[obj_mask], valid_img_x[obj_mask]] = valid_points[obj_mask][:, 2] / rgbd_sensor.depth_scale # m
            # depth completion
            objs_depth_np[obj_id] = cv2.medianBlur(objs_depth_np[obj_id], 5)
            objs_depth_np[obj_id] = cv2.dilate(objs_depth_np[obj_id], depth_kernel, iterations=1)
            
            mask_idxs = (masks_image == obj_id)
            objs_rgb_np[obj_id][mask_idxs] = rgb_image[mask_idxs]
            objs_mask[obj_id] = (masks_image == obj_id).astype(np.uint8) * 255

        return objs_pc, objs_depth_np, objs_rgb_np, objs_mask
        
    def update(self) -> bool:
        if self.capture_times >= self.capture_num or self.auto_stoped:
            self.finished_flag = True
            self.config['dataset']['capture_num'] = self.capture_times
            with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
            return False
        # self.capture_times += 1
        return True
    
    def close(self):
        # TODO: close the dataset
        pass
    
    def is_finished(self) -> bool:
        return self.finished_flag
    
    def get_capture_info(self) -> Tuple[int, int]:
        return self.capture_times, self.capture_num
    
    def get_object_id(self) -> str:
        return self.object_id

class SimRobotScanDataset(RobotScanDataset):
    
    __pose_data_type = PoseDataType.C2W_OPENCV
    __height_direction = HeightDirection.Z_POSITIVE
    
    def __init__(self, config:dict, object_id:str, remark:str) -> None:
        super().__init__(config, object_id, remark)
        self.color_topic = config['dataset']['color_topic']
        self.depth_topic = config['dataset']['depth_topic']
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self._sync_lock = threading.Lock()
        self._sync_seq = 0
        self._sync_pair = (None, None)
        self._sync_ready = threading.Event()

        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self._rgbd_sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=30, slop=0.05)
        self._rgbd_sync.registerCallback(self._store_synced_rgbd)

    def _store_synced_rgbd(self, color_msg, depth_msg):
        with self._sync_lock:
            self._sync_pair = (color_msg, depth_msg)
            self._sync_seq += 1
        self._sync_ready.set()

    def _wait_synced_rgbd(self, timeout=5.0):
        """Return latest synced pair after discarding one buffered frame."""
        with self._sync_lock:
            seq = self._sync_seq
        deadline = time.time() + timeout
        pair = None
        for _ in range(2):
            while time.time() < deadline:
                if self._sync_ready.wait(timeout=deadline - time.time()):
                    self._sync_ready.clear()
                    with self._sync_lock:
                        if self._sync_seq > seq:
                            pair = self._sync_pair
                            seq = self._sync_seq
                            break
            else:
                raise rospy.ROSException("Timed out waiting for synchronized RGB-D frames")
        return pair

    def get_frame(self) -> dict:
        try:
            color_msg, depth_msg = self._wait_synced_rgbd()
            rospy.loginfo(f'Capture success!')
            
            if self.with_arm_turntable:
                scene_rgb_np = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")
                scene_rgb_np = cv2.cvtColor(scene_rgb_np, cv2.COLOR_BGR2RGB)
                scene_depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                # calibrate poses
                cur_cam_pose = call_get_camera_pose_service()
                cur_turntable_pose, _ = call_get_turntable_pose_service()
                cam2base = pose_to_matrix(cur_cam_pose)
                turntable2base = pose_to_matrix(cur_turntable_pose)
                cam2turntable_old = np.linalg.inv(turntable2base) @ cam2base
                # gt poses
                try:
                    self.listener.waitForTransform('/turntable_support_link', '/sim_depth_frame', rospy.Time(0), rospy.Duration(4.0))
                    (trans, rot) = self.listener.lookupTransform('/turntable_support_link', '/sim_depth_frame', rospy.Time(0))
                    cam2turntable = tf.transformations.quaternion_matrix(rot)
                    cam2turntable[:3, 3] = trans
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    raise Exception("TF lookup failed")
            else:
                scene_rgb_np = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")
                scene_rgb_np = cv2.cvtColor(scene_rgb_np, cv2.COLOR_BGR2RGB)
                scene_depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                if scene_depth_np.dtype == np.uint16:
                    scene_depth_np = scene_depth_np.astype(np.float32) / 1000.0 # convert to m
                # gt poses
                from gazebo_msgs.srv import GetModelState
                rospy.wait_for_service('/gazebo/get_model_state')
                get_model_state = rospy.ServiceProxy(
                    '/gazebo/get_model_state',
                    GetModelState
                )
                resp = get_model_state('realsense', '')
                cam2turntable = pose_to_matrix(resp.pose) @ np.linalg.inv(OPENCV_TO_GAZEBO)
                cam2turntable_old = cam2turntable # use gt
            
            ret = {
                'frame_id': self.frame_id,
                "c2w": cam2turntable_old.astype(np.float32),
                'gt_c2w': cam2turntable.astype(np.float32),
                'scene_rgb': scene_rgb_np,
                'scene_depth_np': scene_depth_np,
            }
            self.frame_id += 1
            return ret, ''
        
        except Exception as e:
            rospy.logerr(f"Error while capturing: {e}")
            return {}, ''
    
    def extract_object_point_cloud(self, rgbd_sensor: RGBDSensor, cur_frame_np: Dict[str, Union[np.ndarray, torch.Tensor]], masks_image: np.ndarray):
        unique_object_ids = np.unique(masks_image)
        scene_rgb_np = cur_frame_np['scene_rgb']
        scene_depth_np = cur_frame_np['scene_depth_np']
        
        objs_pc = {}
        objs_depth_np = {}
        objs_rgb_np = {}
        objs_mask = {}

        for obj_id in unique_object_ids:
            objs_depth_np[obj_id] = np.zeros((masks_image.shape[0], masks_image.shape[1]), dtype=np.float32)
            objs_rgb_np[obj_id] = np.ones_like(cur_frame_np['scene_rgb']).astype(np.uint8) * 255
            objs_mask[obj_id] = np.zeros((masks_image.shape[0], masks_image.shape[1]), dtype=np.uint8)
        for obj_id in unique_object_ids:
            mask_idxs = (masks_image == obj_id)
            objs_depth_np[obj_id][mask_idxs] = scene_depth_np[mask_idxs] / rgbd_sensor.depth_scale # m
            invalid_mask = np.isnan(objs_depth_np[obj_id]) | np.isinf(objs_depth_np[obj_id])
            objs_depth_np[obj_id][invalid_mask] = 0 # NOTE: set invalid depth to 0
            objs_rgb_np[obj_id][mask_idxs] = scene_rgb_np[mask_idxs]
            objs_mask[obj_id] = (masks_image == obj_id).astype(np.uint8) * 255
            # objs_pc[obj_id] = None # NOTE: not use point cloud in simulation
            # TODO: get object point cloud
            points, colors = depth_to_pointcloud(
                objs_rgb_np[obj_id], 
                objs_depth_np[obj_id],
                objs_mask[obj_id] > 0, 
                rgbd_sensor.intrinsics,
                1000.0
            )
            objs_pc[obj_id] = {
                'points': points,
                'colors': colors
            }
            
        return objs_pc, objs_depth_np, objs_rgb_np, objs_mask
    
    def setup(self) -> Dict[str, Union[float, int, str, np.ndarray]]:
        sensor_config = readCameraCfg(self.sensor_config_url)
        assert sensor_config['DepthMapFactor'] == self.depth_scale, f"Depth scale mismatch: {sensor_config['DepthMapFactor']} != {self.depth_scale}"
        
        dataset_config = {
            'rgbd_height':             sensor_config['Camera.height'],
            'rgbd_width':              sensor_config['Camera.width'],
            'rgbd_fx':                 sensor_config['Camera.fx'],
            'rgbd_fy':                 sensor_config['Camera.fy'],
            'rgbd_cx':                 sensor_config['Camera.cx'],
            'rgbd_cy':                 sensor_config['Camera.cy'],
            'rgbd_dist':               np.zeros((5, 1)),
            'rgbd_depth_max':          sensor_config['Camera.depth_max'],
            'rgbd_depth_min':          sensor_config['Camera.depth_min'],
            'rgbd_depth_scale':        sensor_config['DepthMapFactor'],
            'rgbd_downsample_factor':  self.rgbd_sensor_downsample_factor,
            'focal_length':            0.4,
            'scene_bound_min':         np.array(self.scene_bbox[0]),
            'scene_bound_max':         np.array(self.scene_bbox[1]),
            'pose_data_type':          self.__pose_data_type.value,
            'height_direction':        int(self.__height_direction.value),
            'results_dir':             self.results_dir
        }
        return dataset_config

def get_dataset(config:dict, object_id:str, remark:str='None') -> Union[RobotScanDataset, None]:
    dataset_format = DatasetFormats(config['dataset']['format'])
    if object_id != 'Eval':
        capture_num = rospy.get_param('capture_num', -1)
        config['dataset']['capture_num'] = config['dataset']['capture_num'] if capture_num == -1 else capture_num
    if dataset_format == DatasetFormats.STRCAM:
        return RobotScanDataset(config, object_id, remark)
    elif dataset_format == DatasetFormats.SIMCAM:
        return SimRobotScanDataset(config, object_id, remark)
    else:
        raise NotImplementedError(f'Dataset format {dataset_format.name} not support.')
    
        
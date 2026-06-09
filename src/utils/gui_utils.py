from enum import Enum
import numpy as np
import open3d as o3d
import quaternion
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Pose
from utils import OPENCV_TO_OPENGL

class PoseChangeType(Enum):
    NONE = 0
    TRANSLATION = 1
    ROTATION = 2
    BOTH = 3

class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -10.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up

def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum

class Gaussians:
    def __init__(self, gaussians):
        self.means = gaussians.get_xyz.detach().clone()
        self.scales = gaussians.get_scaling.detach().clone()
        self.rotations = gaussians.get_rotation.detach().clone()
        self.opacities = gaussians.get_opacity.detach().clone()
        self.harmonics = gaussians.get_features.detach().clone()
        self.normals = gaussians.get_normal.detach().clone()
        self.background_color = gaussians.background_color.clone()

class GaussianPacket:
    def __init__(
        self,
        gaussians=None,
        current_frame=None,
        kf_window=None,
        keyframes=None,
        keyframe_colors=None,
        iteration=None,
    ):
        self.has_gaussians = False
        self.current_frame = current_frame
        self.kf_window = kf_window
        self.keyframes = keyframes
        self.keyframe_colors = keyframe_colors
        self.iteration = iteration
        
        if gaussians is not None:
            self.has_gaussians = True
            self.gaussians = Gaussians(gaussians)
        
def vfov_to_hfov(vfov_deg, height, width):
    # http://paulbourke.net/miscellaneous/lens/
    return np.rad2deg(
        2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
    )

def rgbd_to_pointcloud(
    rgb_data:np.ndarray,
    depth_data:np.ndarray,
    pose_data:np.ndarray,
    camera_intrinsics_tensor:o3d.core.Tensor,
    depth_scale:float,
    depth_max:float,
    device:o3d.core.Device) -> o3d.t.geometry.PointCloud:
    if rgb_data.dtype == np.float32:
        rgb_data_uint8 = (rgb_data * 255).astype(np.uint8)
    elif rgb_data.dtype == np.uint8:
        rgb_data_uint8 = rgb_data
    else:
        raise ValueError(f"Invalid rgb_data dtype: {rgb_data.dtype}")
    if depth_data.dtype == np.float32:
        depth_data_uint16 = (depth_data * 1000).astype(np.uint16)
    elif depth_data.dtype == np.uint16:
        depth_data_uint16 = depth_data
    else:
        raise ValueError(f"Invalid depth_data dtype: {depth_data.dtype}")
    rgb_image = o3d.t.geometry.Image(o3d.core.Tensor(rgb_data_uint8, device=device))
    depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth_data_uint16, device=device))
    rgbd_image = o3d.t.geometry.RGBDImage(rgb_image, depth_image)
    current_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics_tensor,
        o3d.core.Tensor(np.linalg.inv(OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL), device=device),
        depth_scale=depth_scale,
        depth_max=depth_max)
    return current_pcd.cpu()

def pose_to_matrix(pose:Pose) -> np.ndarray:
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    transform_matrix[:3, :3] = quaternion.as_rotation_matrix(
        quaternion.from_float_array([
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z]))
    return transform_matrix

def matrix_to_pose(matrix: np.ndarray) -> Pose:
    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 transformation matrix")

    pose = Pose()
    pose.position.x = matrix[0, 3]
    pose.position.y = matrix[1, 3]
    pose.position.z = matrix[2, 3]
    rot_matrix = matrix[:3, :3]
    q = quaternion.from_rotation_matrix(rot_matrix)
    pose.orientation.w = q.w
    pose.orientation.x = q.x
    pose.orientation.y = q.y
    pose.orientation.z = q.z
    return pose

def is_pose_changed(
    frame_c2w_old:np.ndarray,
    frame_c2w_new:np.ndarray,
    translation_threshold:float,
    rotation_threshold:float) -> PoseChangeType:
    assert frame_c2w_old is not None, "frame_c2w_old is None"
    assert frame_c2w_new is not None, "frame_c2w_new is None"
    frame_c2w_diff_translation = np.linalg.norm(frame_c2w_new[:3, 3] - frame_c2w_old[:3, 3])
    frame_c2w_diff_rotation = np.dot(frame_c2w_new[:3, :3], np.linalg.inv(frame_c2w_old[:3, :3]))
    frame_c2w_diff_rotation = np.arccos((np.trace(frame_c2w_diff_rotation) - 1) / 2)
    frame_c2w_diff_rotation = np.degrees(frame_c2w_diff_rotation)
    if frame_c2w_diff_translation > translation_threshold and frame_c2w_diff_rotation > rotation_threshold:
        rospy.logdebug(f'Get new c2w\nc2w_diff_translation: {frame_c2w_diff_translation}\nc2w_diff_rotation: {frame_c2w_diff_rotation}')
        return PoseChangeType.BOTH
    elif frame_c2w_diff_translation > translation_threshold:
        rospy.logdebug(f'Get new c2w\nc2w_diff_translation: {frame_c2w_diff_translation}\nc2w_diff_rotation: {frame_c2w_diff_rotation}')
        return PoseChangeType.TRANSLATION
    elif frame_c2w_diff_rotation > rotation_threshold:
        rospy.logdebug(f'Get new c2w\nc2w_diff_translation: {frame_c2w_diff_translation}\nc2w_diff_rotation: {frame_c2w_diff_rotation}')
        return PoseChangeType.ROTATION
    else:
        return PoseChangeType.NONE
    
def update_traj(trajectory:list, color_name:str='cool')->o3d.geometry.LineSet:
    points = []
    lines = []
    colors = []
    line_colormap = plt.get_cmap(color_name)

    for i in range(len(trajectory)):
        points.append(trajectory[i])
        if i < len(trajectory)-1:
            lines.append([i, i+1])
            # Normalize the step index to the range [0, 1]
            color_value = i / (len(trajectory) - 1)
            # Get the color from the colormap
            color = line_colormap(color_value)[:3]  # Take only RGB values
            colors.append(color)
    
    points = np.array(points).astype(np.float32)
    points = points.reshape(-1, 3)
    lines = np.array(lines).reshape(-1, 2)
    
    cam_traj = o3d.geometry.LineSet()
    cam_traj.points = o3d.utility.Vector3dVector(points)
    cam_traj.lines = o3d.utility.Vector2iVector(lines)
    cam_traj.colors = o3d.utility.Vector3dVector(colors)
    
    return cam_traj
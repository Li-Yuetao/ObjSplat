import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from enum import Enum
import numpy as np
from copy import deepcopy
from queue import Queue
import torch
import threading
import time
from typing import Dict, List, Union
import quaternion
from imgviz import depth2rgb
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm
import open3d as o3d
from open3d.visualization import rendering, gui
import glfw
from OpenGL import GL as gl
from tqdm import trange
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Pose, PoseStamped

from dataloader import RGBDSensor, PoseDataType, dataset_config_to_ros
from mapper import get_mapper, MapperState, GaussianColorType, MapperType
from mapper.gsmap.gaussian_surfels.utils.graphics_utils import fov2focal
from utils.gui_utils import GaussianPacket, PoseChangeType, vfov_to_hfov, create_frustum, rgbd_to_pointcloud, is_pose_changed, update_traj, pose_to_matrix, matrix_to_pose
from utils.logging_utils import Log
from utils.pose_utils import compute_pose_error, compute_path_length
from utils.ros_utils import call_move_turntable_service, call_get_turntable_pose_service, call_move_camera_service, call_move_arm_service
from utils import PROJECT_NAME, OPENCV_TO_OPENGL, OPENCV_TO_GAZEBO, CURRENT_FRUSTUM, BEFORE_REFINE_FRUSTUM, GT_FRUSTUM, CURRENT_HORIZON, CANDIDATE_FRUSTUM, start_timing, end_timing, GlobalState, ArmGroupState
from dataloader.dataloader import RobotScanDataset, SimRobotScanDataset
from visualizer.gl_render import util, util_gau
from visualizer.gl_render.render_ogl import OpenGLRenderer
from scripts.nodes import ImageSeg, ImageSegRequest, ImageSegResponse,\
                GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
                    ApplyCapture, ApplyCaptureRequest, ApplyCaptureResponse,\
                        SetPlannerState, SetPlannerStateRequest, SetPlannerStateResponse,\
                                GetReconInfo, GetReconInfoRequest, GetReconInfoResponse,\
                                    ObjectAlignment, ObjectAlignmentRequest, ObjectAlignmentResponse,\
                                        GetUncertainty, GetUncertaintyRequest, GetUncertaintyResponse
            
class Visualizer:
    
    class LocalDatasetState(Enum):
        INITIALIZING = 0
        INITIALIZED = 1
        RUNNING = 2
    
    class QueryUncertaintyFlag(Enum):
        NONE = 0
        RUNNING = 1
    
    def __init__(self,
                mapper_type:MapperType,
                config:Dict,
                init_state:GlobalState,
                font_id:int,
                device:torch.device,
                local_dataset:Union[RobotScanDataset, SimRobotScanDataset],
                hide_windows:bool,
                save_runtime_data:bool):
        self.__mapper_type = mapper_type
        self.__device = device
        self.__hide_windows = hide_windows
        self.__global_states_selectable = [GlobalState.AUTO_PLANNING, GlobalState.MANUAL_CONTROL, GlobalState.PAUSE]
        self.__arm_group_states_selectable = [ArmGroupState.ready]
        self.__local_dataset:Union[RobotScanDataset, SimRobotScanDataset] = local_dataset
        self.mapping_finished = False
        self.render_img = None
        self.__with_arm_turntable = config['dataset']['with_arm_turntable']
            
        os.chdir(PACKAGE_PATH)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        
        config['hide_windows'] = hide_windows
        capture_num = config['dataset']['capture_num']
        self.__refine_pose_flag = config['mapper']['refine_pose']
        self.__auto_stop = config['dataset']['auto_stop']
        self.object_id = 'None'
        self.__global_state = init_state
        self.__arm_group_state = ArmGroupState.ready
        self.__frame_update_translation_threshold = config['mapper']['pose']['update_threshold']['translation']
        self.__frame_update_rotation_threshold = config['mapper']['pose']['update_threshold']['rotation']
        self.__traj_info = dict()
        self.__traj_info['cam_centers'] = []
        self.__traj_info['line_colormap'] = plt.get_cmap('cool')
        self.__traj_info['norm_factor'] = 0.5
        self.__traj_info['length'] = 0.0
        self.only_quality_uncertainty_flag = False
        self.eval_visual_quality = config['system']['eval']['visual_quality']
        self.save_gaussians_data_every = config['system']['eval']['save_gaussians_data_every']
        self.__mapping_condition = threading.Condition()
        
        if self.__local_dataset is not None:
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZING
            self.__local_dataset_condition = threading.Condition()
            self.__local_dataset_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=1)
            self.__local_dataset_pose_ros = None
            self.__local_dataset_thread = threading.Thread(
                target=self.__update_dataset,
                name='UpdateDataset',
                daemon=True)
            self.__local_dataset_thread.start()
            self.__local_dataset_label = gui.Label('')
            self.__local_dataset_label.font_id = font_id
            _, capture_num = self.__local_dataset.get_capture_info()
            self.object_id = self.__local_dataset.get_object_id()
        
        self.__update_main_thread = threading.Thread(
            target=self.__update_main,
            name='UpdateMain',
            daemon=True)
        
        self.__init_dataset(save_runtime_data)
        self.__init_o3d_elements()
        
        frame_first = self.__frames_cache.get()
        c2w = frame_first['c2w'].detach().cpu().numpy()
        
        # add first pose
        pose_data_o3d = OPENCV_TO_OPENGL @ c2w @ OPENCV_TO_OPENGL
        latest_location = pose_data_o3d[:3, 3].copy()
        self.__traj_info['cam_centers'].append(latest_location)
        
        if self.__frames_cache.empty(): self.__frames_cache.put(frame_first)
        
        self.q_main2vis = Queue(maxsize=1)
        self.frustum_dict = {}
        self.__use_gaussian_condition = threading.Condition()
        self.__collect_data_flag = False
        rospy.Service('set_global_state', SetPlannerState, self.__set_global_state)
        self.__collect_data_mode = 'none'
        
        Mapper = get_mapper(self.__mapper_type)
        self.__mapper = Mapper(
            config,
            self.__rgbd_sensor,
            self.__device,
            self.q_main2vis,
            self.__results_dir,
            capture_num)
        
        self.__init_window(
            config['mapper']['interval_max_ratio'])
        
        if not self.__hide_windows:
            self.set_opengl_gs()
        
        self.__update_main_thread.start()
    
    # NOTE: initialization functions
    
    def __init_dataset(self, save_runtime_data) -> o3d.geometry.TriangleMesh:
        self.__frames_cache:Queue[Dict[str, Union[int, torch.Tensor]]] = Queue(maxsize=1)
        self.__frame_c2w_last = None
        
        self.__movement_flag_pub = rospy.Publisher('movement_flag', String, queue_size=1)
        self.__apply_capture_service = rospy.Service('apply_capture', ApplyCapture, self.__apply_capture)
        self.__get_recon_info_service = rospy.Service('get_recon_info', GetReconInfo, self.__get_recon_info)
        self.__get_uncertainty_service = rospy.Service('get_uncertainty', GetUncertainty, self.__get_uncertainty)
        self.__get_uncertainty_condition = threading.Condition()
        self.__get_uncertainty_flag = self.QueryUncertaintyFlag.NONE
        self.__object_alignment_service = rospy.ServiceProxy('/object_align_node/kissmatcher', ObjectAlignment) 
        
        if self.__local_dataset is None:
            # TODO
            raise Exception('Local dataset is None.')
        else:
            self.__local_dataset_condition.acquire()
            if self.__local_dataset_state == self.LocalDatasetState.INITIALIZING:
                self.__local_dataset_condition.wait()
                
        Trc = np.eye(4)
        Trc[:3, 3] = np.array([
            self.__dataset_config.rgbd_position.x,
            self.__dataset_config.rgbd_position.y,
            self.__dataset_config.rgbd_position.z])
        self.__Tcr = np.linalg.inv(Trc)
        
        self.__results_dir = self.__dataset_config.results_dir
        os.makedirs(self.__results_dir, exist_ok=True)
        # NOTE: Save runtime data
        self.__save_runtime_data = save_runtime_data
        self.__runtime_data_info = None
        if self.__save_runtime_data:
            self.__runtime_data_info = {"render_use_time": []}
            self.render_rgbd_dir = os.path.join(self.__results_dir, 'render_rgbd')
            if not os.path.exists(self.render_rgbd_dir): os.makedirs(self.render_rgbd_dir)
            self.__init_runtime_data_info()
        self.__pose_data_type = PoseDataType(self.__dataset_config.pose_data_type)
        self.__height_direction = (self.__dataset_config.height_direction // 2, (self.__dataset_config.height_direction % 2) * 2 - 1)
        
        self.__rgbd_sensor = RGBDSensor(
            height=self.__dataset_config.rgbd_height,
            width=self.__dataset_config.rgbd_width,
            fx=self.__dataset_config.rgbd_fx,
            fy=self.__dataset_config.rgbd_fy,
            cx=self.__dataset_config.rgbd_cx,
            cy=self.__dataset_config.rgbd_cy,
            dist=self.__dataset_config.rgbd_dist,
            depth_min=self.__dataset_config.rgbd_depth_min,
            depth_max=self.__dataset_config.rgbd_depth_max,
            depth_scale=self.__dataset_config.rgbd_depth_scale,
            position=np.array([
                self.__dataset_config.rgbd_position.x,
                self.__dataset_config.rgbd_position.y,
                self.__dataset_config.rgbd_position.z]),
            downsample_factor=self.__dataset_config.rgbd_downsample_factor)
        
        if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
            self.__local_dataset_condition.wait()
        self.__local_dataset_condition.notify_all()
        self.__local_dataset_condition.release()
            
        self.__bbox_visualize = np.array([
            [self.__dataset_config.scene_bound_min.x, self.__dataset_config.scene_bound_max.x],
            [self.__dataset_config.scene_bound_min.y, self.__dataset_config.scene_bound_max.y],
            [self.__dataset_config.scene_bound_min.z, self.__dataset_config.scene_bound_max.z]])
        
    def __init_o3d_elements(self):
        self.__device_o3c = o3d.core.Device(self.__device.type, self.__device.index)
        
        self.__o3d_pcd:Dict[str, o3d.t.geometry.PointCloud] = {
            'current_pcd': None,
        }
        
        self.__o3d_const_camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.__rgbd_sensor.width,
            self.__rgbd_sensor.height,
            self.__rgbd_sensor.fx,
            self.__rgbd_sensor.fy,
            self.__rgbd_sensor.cx,
            self.__rgbd_sensor.cy)
        self.__o3d_const_camera_intrinsics_o3c = o3d.core.Tensor(self.__o3d_const_camera_intrinsics.intrinsic_matrix, device=self.__device_o3c)
        
        self.__o3d_materials:Dict[str, rendering.MaterialRecord] = {
            'lit_mat': None,
            'lit_mat_transparency': None,
            'unlit_mat': None,
            'unlit_line_mat': None,
            'unlit_line_mat_slim': None,
        }
        
        if not self.__hide_windows:
            self.__o3d_materials['lit_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['lit_mat'].shader = 'defaultLit'
            self.__o3d_materials['lit_mat_transparency'] = rendering.MaterialRecord()
            self.__o3d_materials['lit_mat_transparency'].shader = 'defaultLitTransparency'
            self.__o3d_materials['lit_mat_transparency'].has_alpha = True
            self.__o3d_materials['lit_mat_transparency'].base_color = [1.0, 1.0, 1.0, 0.9]
            self.__o3d_materials['unlit_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_mat'].shader = 'defaultUnlit'
            self.__o3d_materials['unlit_mat'].sRGB_color = True
            self.__o3d_materials['unlit_line_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat'].line_width = 5.0
            self.__o3d_materials['unlit_line_mat_slim'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat_slim'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat_slim'].line_width = 3.0
            self.__o3d_materials['unlit_line_mat_thick'] = rendering.MaterialRecord()
            self.__o3d_materials['unlit_line_mat_thick'].shader = 'unlitLine'
            self.__o3d_materials['unlit_line_mat_thick'].line_width = 15.0
            self.__o3d_materials['gaussian_mat'] = rendering.MaterialRecord()
            self.__o3d_materials['gaussian_mat'].shader = 'defaultUnlit'
        
    def __init_window(self, interval_max_ratio:float):
        kf_every = self.__mapper.get_kf_every()
        assert 0 < kf_every, f'Invalid keyframe every: {kf_every}'
        map_every = self.__mapper.get_map_every()
        mapping_iters = self.__mapper.get_mapping_iters()
        assert 0 < map_every, f'Invalid map every: {map_every}'
        update_interval_max = int(max(interval_max_ratio * max(kf_every, map_every), mapping_iters))
        assert interval_max_ratio >= 1.0, f'Invalid interval ratio: {interval_max_ratio}'
        
        self.__set_planner_state_service = rospy.ServiceProxy('set_planner_state', SetPlannerState)
        rospy.wait_for_service('set_planner_state')
        
        if self.__hide_windows:
            self.__global_state_callback(self.__global_state.value, None)
            return
        else:
            # NOTE: Initialize GUI
            self.__window:gui.Window = gui.Application.instance.create_window(PROJECT_NAME, 1920, 1080)
            self.__window.show(False)

            em = self.__window.theme.font_size
            margin = 0.5 * em
            spacing = int(np.round(0.25 * em))
            vspacing = int(np.round(0.5 * em))
            
            margins = gui.Margins(vspacing)
            self.__panel_control = gui.Vert(spacing, margins)
            self.__panel_visualize = gui.Vert(spacing, margins)
            
            self.__widget_3d = gui.SceneWidget()
            self.__widget_3d.scene = rendering.Open3DScene(self.__window.renderer)
            self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
            self.__widget_3d.scene.scene.set_sun_light([-0.2, 1.0 ,0.2], [1.0, 1.0, 1.0], 70000)
            self.__widget_3d.scene.scene.enable_sun_light(True)
            self.__widget_3d.set_on_key(self.__widget_3d_on_key)
            
            # NOTE: Widgets for control panel
            global_state_vgrid = gui.VGrid(2, spacing, gui.Margins(0, 0, em, 0))

            global_state_vgrid.add_child(gui.Label('Global State'))
            self.__global_state_combobox = gui.Combobox()
            for global_state in self.__global_states_selectable:
                self.__global_state_combobox.add_item(global_state.value)
            self.__global_state_combobox.selected_text = self.__global_state.value
            self.__global_state_combobox.set_on_selection_changed(self.__global_state_callback)
            global_state_vgrid.add_child(self.__global_state_combobox)
            self.__panel_control.add_child(global_state_vgrid)
            self.__global_state_callback(self.__global_state_combobox.selected_text, None)
            
            self.__panel_control.add_fixed(vspacing)
            if self.__local_dataset is not None:
                self.__panel_control.add_child(gui.Label('Local Dataset Info'))
                self.__panel_control.add_child(self.__local_dataset_label)
            
            self.__panel_control.add_child(gui.Label('3D Visualization Settings'))
            
            panel_control_vgrid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))
            
            panel_control_vgrid.add_child(gui.Label('    Mapper Configurations'))
            panel_control_vgrid.add_child(gui.Label(''))

            panel_control_vgrid.add_child(gui.Label('        Map Every'))
            self.__map_every_slider = gui.Slider(gui.Slider.INT)
            self.__map_every_slider.set_limits(1, update_interval_max)
            self.__map_every_slider.int_value = map_every
            self.__map_every_slider.set_on_value_changed(lambda value: self.__mapper.set_map_every(value))
            panel_control_vgrid.add_child(self.__map_every_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframe Every'))
            self.__kf_every_slider = gui.Slider(gui.Slider.INT)
            self.__kf_every_slider.set_limits(1, update_interval_max)
            self.__kf_every_slider.int_value = kf_every
            self.__kf_every_slider.set_on_value_changed(lambda value: self.__mapper.set_kf_every(value))
            panel_control_vgrid.add_child(self.__kf_every_slider)
            

            panel_control_vgrid.add_child(gui.Label('    Global Status'))
            panel_control_vgrid.add_child(gui.Label(''))
            
            view_gs_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            view_gs_grid.add_child(gui.Label('        View Gaussians'))
            self.__view_gaussians_box = gui.Checkbox('')
            self.__view_gaussians_box.checked = True
            def view_gaussians_callback(checked:bool):
                self.followcam_chbox.enabled = checked
                self.staybehind_chbox.enabled = checked
                if checked is False:
                    self.__widget_3d.scene.set_background([1.0, 1.0, 1.0, 1.0])
                return gui.Checkbox.HANDLED
            self.__view_gaussians_box.set_on_checked(view_gaussians_callback)
            view_gs_grid.add_child(self.__view_gaussians_box)
            
            panel_control_vgrid.add_child(view_gs_grid)
            panel_control_vgrid.add_child(gui.Label(''))
            
            # Note: 3DGS Visualization, as referenced in MonoGS.
            panel_control_vgrid.add_child(gui.Label("        Viewing options"))
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            self.followcam_chbox = gui.Checkbox("Follow Camera")
            self.followcam_chbox.checked = False
            chbox_tile.add_child(self.followcam_chbox)
            def followcam_chbox_callback(checked:bool):
                if checked:
                    self.staybehind_chbox.checked = True
                    self.__cam_traj_box.checked = False
                    # self.__height_direction_lower_bound_slider.double_value -= 0.5
                else:
                    self.staybehind_chbox.checked = False
                    self.__cam_traj_box.checked = True
                    # self.__height_direction_lower_bound_slider.double_value += 0.5
                    self.__widget_3d.look_at(self.__init_look_at['center'], self.__init_look_at['eye'], self.__init_look_at['up'])
                return gui.Checkbox.HANDLED
            self.followcam_chbox.set_on_checked(followcam_chbox_callback)
            panel_control_vgrid.add_child(chbox_tile)
            
            panel_control_vgrid.add_child(gui.Label(''))
            
            self.staybehind_chbox = gui.Checkbox("From Behind")
            self.staybehind_chbox.checked = False
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            chbox_tile.add_child(self.staybehind_chbox)
            panel_control_vgrid.add_child(chbox_tile)
            
            panel_control_vgrid.add_child(gui.Label(''))
            
            self.frontonly_chbox = gui.Checkbox("Front Only")
            self.frontonly_chbox.checked = True
            chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
            chbox_tile.add_child(self.frontonly_chbox)
            panel_control_vgrid.add_child(chbox_tile)
            
            panel_control_vgrid.add_child(gui.Label("        Rendering options"))
            self.__gaussian_color_combobox = gui.Combobox()
            self.__gaussian_color_type = None
            for gaussian_color_type in GaussianColorType:
                self.__gaussian_color_combobox.add_item(gaussian_color_type.value)
            self.__gaussian_color_combobox.selected_text = GaussianColorType.Color.value
            def gaussian_color_combobox_callback(color_type_name:str, color_type_index:int):
                self.__gaussian_color_type = GaussianColorType(color_type_name)
                return gui.Combobox.HANDLED
            self.__gaussian_color_combobox.set_on_selection_changed(gaussian_color_combobox_callback)
            gaussian_color_combobox_callback(self.__gaussian_color_combobox.selected_text, None)
            panel_control_vgrid.add_child(self.__gaussian_color_combobox)
            
            panel_control_vgrid.add_child(gui.Label('        Gaussian Scale (0-1)'))
            self.__gaussian_scale_slider = gui.Slider(gui.Slider.DOUBLE)
            self.__gaussian_scale_slider.set_limits(0.001, 1.0)
            self.__gaussian_scale_slider.double_value = 1.0
            panel_control_vgrid.add_child(self.__gaussian_scale_slider)
            
            panel_control_vgrid.add_child(gui.Label('        Origin Mesh'))
            origin_mesh_box = gui.Checkbox('')
            origin_mesh_box.checked = False
            origin_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            self.__widget_3d.scene.add_geometry('origin_mesh', origin_mesh, self.__o3d_materials['lit_mat'])
            self.__widget_3d.scene.show_geometry("origin_mesh", origin_mesh_box.checked)
            def origin_mesh_chbox_callback(checked:bool):
                name = "origin_mesh"
                self.__widget_3d.scene.show_geometry(name, checked)
            origin_mesh_box.set_on_checked(origin_mesh_chbox_callback)
            panel_control_vgrid.add_child(origin_mesh_box)
            
            panel_control_vgrid.add_child(gui.Label('        Viewpoint list'))
            self.__kf_combobox = gui.Combobox()
            self.__kf_combobox.set_on_selection_changed(self.keyframe_combobox_callback)
            panel_control_vgrid.add_child(self.__kf_combobox)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframe Views'))
            self.__kf_viewpoints_box = gui.Checkbox('')
            self.__kf_viewpoints_box.checked = True
            self.__kf_viewpoints_box.set_on_checked(self.__on_cameras_chbox)
            panel_control_vgrid.add_child(self.__kf_viewpoints_box)
            
            panel_control_vgrid.add_child(gui.Label('        Keyframe Views GT'))
            self.__kf_viewpoints_gt_box = gui.Checkbox('')
            self.__kf_viewpoints_gt_box.checked = True
            self.__kf_viewpoints_gt_box.set_on_checked(self.__on_gt_cameras_chbox)
            panel_control_vgrid.add_child(self.__kf_viewpoints_gt_box)
            
            panel_control_vgrid.add_child(gui.Label('        Candidate Views'))
            self.__candidate_frustums_box = gui.Checkbox('')
            self.__candidate_frustums_box.checked = True
            self.__candidate_frustums_box.set_on_checked(self.__on_candidate_cameras_chbox)
            panel_control_vgrid.add_child(self.__candidate_frustums_box)
            
            panel_control_vgrid.add_child(gui.Label('    Local Status'))
            panel_control_vgrid.add_child(gui.Label(''))
            
            panel_control_vgrid.add_child(gui.Label('        Current window'))
            self.__kf_window_box = gui.Checkbox('')
            self.__kf_window_box.checked = False
            self.__kf_window_box.set_on_checked(self.__on_kf_window_chbox)
            panel_control_vgrid.add_child(self.__kf_window_box)
            self.kf_window = None
            
            panel_control_vgrid.add_child(gui.Label('        Current Frustum'))
            self.__current_frustum_box = gui.Checkbox('')
            self.__current_frustum_box.checked = True
            self.__current_frustum_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_frustum', checked))
            panel_control_vgrid.add_child(self.__current_frustum_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current Horizon'))
            self.__current_horizon_box = gui.Checkbox('')
            self.__current_horizon_box.checked = False
            self.__current_horizon_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_horizon', checked))
            panel_control_vgrid.add_child(self.__current_horizon_box)
            
            panel_control_vgrid.add_child(gui.Label('        Current PCD'))
            self.__current_pcd_box = gui.Checkbox('')
            self.__current_pcd_box.checked = False
            self.__current_pcd_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('current_pcd', checked))
            panel_control_vgrid.add_child(self.__current_pcd_box)
            
            panel_control_vgrid.add_child(gui.Label('        Camera Trajectory'))
            self.__cam_traj_box = gui.Checkbox('')
            self.__cam_traj_box.checked = True
            self.__cam_traj_box.set_on_checked(lambda checked: self.__widget_3d.scene.show_geometry('cam_traj', checked))
            panel_control_vgrid.add_child(self.__cam_traj_box)
            
            panel_control_vgrid.add_child(gui.Label('    Control interface'))
            self.__arm_group_state_combobox = gui.Combobox()
            for arm_group_state in self.__arm_group_states_selectable:
                self.__arm_group_state_combobox.add_item(arm_group_state.value)
            self.__arm_group_state_combobox.selected_text = self.__global_state.value
            self.__arm_group_state_combobox.set_on_selection_changed(self.__arm_group_state_callback)
            panel_control_vgrid.add_child(self.__arm_group_state_combobox)
            self.__global_state_callback(self.__global_state_combobox.selected_text, None)
            self.__move_arm_group_state_button = gui.Button('Execute')
            self.__move_arm_group_state = lambda: self.__move_arm_group_state_callback()
            self.__move_arm_group_state_button.set_on_clicked(self.__move_arm_group_state)
            panel_control_vgrid.add_child(self.__move_arm_group_state_button)
            self.__turntable2uicam_button = gui.Button('Turn to UI Camera')
            self.__turntable2uicam = lambda: self.__turn2uicam_callback()
            self.__turntable2uicam_button.set_on_clicked(self.__turntable2uicam)
            panel_control_vgrid.add_child(self.__turntable2uicam_button)
            
            self.__panel_control.add_child(panel_control_vgrid)
            
            panel_visualize_tabs = gui.TabControl()
            panel_visualize_tab_margin = gui.Margins(0, int(np.round(0.5 * em)), em, em)
            
            tab_live_view = gui.ScrollableVert(0, panel_visualize_tab_margin)
            
            image_placeholder_numpy = np.zeros((self.__rgbd_sensor.height, self.__rgbd_sensor.width * 2, 3), dtype=np.uint8)
            image_placeholder = o3d.geometry.Image(image_placeholder_numpy)
            
            if self.__save_runtime_data:
                save_current_data_button = gui.Button('Save Current Data')
                self.__save_current_data = lambda: self.__save_current_data_callback(-1)
                save_current_data_button.set_on_clicked(self.__save_current_data)
                tab_live_view.add_child(save_current_data_button)
                tab_live_view.add_fixed(vspacing)

            self.screenshot_button = gui.Button("Screenshot")
            self._on_screenshot_button = lambda: self.__on_screenshot_callback()
            self.screenshot_button.set_on_clicked(self._on_screenshot_button)
            tab_live_view.add_child(self.screenshot_button)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('  RGBD Live Image'))
            self.__rgbd_live_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_live_image)
            tab_live_view.add_fixed(vspacing)
            
            render_grid = gui.VGrid(2, spacing, gui.Margins(0, 0, 0, 0))
            render_grid.add_child(gui.Label('  Rendered RGBD Image'))
            self.__render_box = gui.Checkbox('')
            self.__render_box.checked = True
            def render_box_callback(checked:bool):
                self.__render_every_slider.enabled = checked
                return gui.Checkbox.HANDLED
            self.__render_box.set_on_checked(render_box_callback)
            render_grid.add_child(self.__render_box)
            
            render_grid.add_child(gui.Label('    Render Every'))
            self.__render_every_slider = gui.Slider(gui.Slider.INT)
            self.__render_every_slider.set_limits(1, update_interval_max)
            self.__render_every_slider.int_value = 1
            self.__render_every_slider.enabled = self.__render_box.checked
            render_grid.add_child(self.__render_every_slider)
            
            tab_live_view.add_child(render_grid)
            
            self.__rgbd_render_image = gui.ImageWidget()
            tab_live_view.add_child(self.__rgbd_render_image)
            tab_live_view.add_fixed(vspacing)
            
            tab_live_view.add_child(gui.Label('  Uncertainty'))
            self.__uncertainty_render_image = gui.ImageWidget()
            tab_live_view.add_child(self.__uncertainty_render_image)
            tab_live_view.add_fixed(vspacing)
            
            # NOTE: Information
            tab_live_info = gui.ScrollableVert(0, panel_visualize_tab_margin)
            self.num_gaussians_info = gui.Label(" Number of Gaussians: ")
            tab_live_info.add_child(self.num_gaussians_info)
            self.offline_iterations_info = gui.Label(" Offline Iterations: ")
            tab_live_info.add_child(self.offline_iterations_info)
            self.cam_pose_info = gui.Label("Camera Pose: ")
            tab_live_info.add_child(self.cam_pose_info)
            self.ui_cam_pose_info = gui.Label("UI Camera Pose: ")
            tab_live_info.add_child(self.ui_cam_pose_info)
            self.render_use_time_info = gui.Label("Render Use Time: ")
            tab_live_info.add_child(self.render_use_time_info)
            
            panel_visualize_tabs.add_tab('Live View', tab_live_view)
            panel_visualize_tabs.add_tab('Live Info', tab_live_info)
            
            self.__panel_visualize.add_child(panel_visualize_tabs)
            
            self.__window.add_child(self.__panel_control)
            self.__window.add_child(self.__widget_3d)
            self.__window.add_child(self.__panel_visualize)
            
            self.__window.set_on_layout(self.__window_on_layout)
            self.__window.set_on_close(self.__window_on_close)
            
            # NOTE: Setup the UI Camera
            center = np.average(self.__bbox_visualize, axis=1)
            center[self.__height_direction[0]] = 0
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.__bbox_visualize[:, 0], self.__bbox_visualize[:, 1])
            self.__widget_3d.setup_camera(60.0, bbox, center)
            height_location = np.max(np.ptp(self.__bbox_visualize, axis=1))
            center_bias = np.zeros(3)
            use_topdown_view = False
            if use_topdown_view:
                center_bias[self.__height_direction[0]] = -self.__height_direction[1] * (height_location - 1)
            else:
                center_bias[self.__height_direction[0]] = -self.__height_direction[1] * (height_location - 1) * 1.5
                center_bias[(self.__height_direction[0] + 1) % 3] = 2 * self.__height_direction[1]
            up_vector = np.zeros(3)
            up_vector[(self.__height_direction[0] + 1) % 3] = -self.__height_direction[1]
            center_bias *= 0.4 # more zoom in
            self.__init_look_at = {'center': center, 'eye': center + center_bias, 'up': up_vector}
            self.__widget_3d.look_at(center, center + center_bias, up_vector)
            self.__window.show(True)
    
    def __init_runtime_data_info(self):
        self.__get_debug_data_flag = False

        directories = {
            'runtime_data_dir': os.path.join(self.__results_dir, 'runtime_data'),
            'current_vis_data_dir': os.path.join(self.__results_dir, 'runtime_data', 'current_vis_data')
        }

        for key, path in directories.items():
            if not os.path.exists(path):
                os.makedirs(path)
            self.__runtime_data_info[key] = path

        self.__runtime_data_info['current_vis_data'] = dict()
    
    # NOTE: main function
    
    def __update_main(self):
        frame_id = None
        frame_last_received = None
        rendering_rgbd_last_frame_id = -np.inf
        self.__gaussian_for_render = None
        self.__gaussian_packet = None
        self.gaussians_num = 0
        self.offline_iterations = 0
        self.__trigger_update_viewpoints_flag = False
        self.candidate_frustums = []
        self.last_save_frame_id = -1
        self.max_normal_uncertainty = np.inf
        self.timing_online_mapping = start_timing()
        self.online_mapping_time = 0.0
        
        while self.__global_state not in [GlobalState.POST_PROCESSING, GlobalState.QUIT]:
            if self.__global_state in [GlobalState.AUTO_PLANNING, GlobalState.MANUAL_CONTROL, GlobalState.COLLECT_DATA]:
                # NOTE: Get observation
                if self.__frames_cache.empty() and (frame_id is not None):
                    frame_current = None
                else:
                    frame_current = self.__frames_cache.get()
                    if frame_id is None:
                        frame_id = 0
                        self.__update_ui_frame(frame_current)
                    else:
                        frame_id += 1
                    frame_current['frame_id'] = frame_id
                    frame_last_received = frame_current.copy()
                assert frame_id is not None, 'Initialize failed'
            else:
                frame_current = None
            if self.__global_state in [GlobalState.AUTO_PLANNING, GlobalState.MANUAL_CONTROL]:
                # NOTE: Train model
                mapper_state = self.__mapper.run(frame_current)
                if self.save_gaussians_data_every > 0 and (mapper_state is not MapperState.ON_MAPPING) and (frame_id % self.save_gaussians_data_every == 0):
                    self.__mapper.gs.save_gaussian_data(frame_id)
                self.__traj_info["capture_times"] = frame_id + 1
                
                # publish current pose to planner
                if mapper_state is not MapperState.ON_MAPPING:
                    cur_pose = matrix_to_pose(self.__mapper.get_current_pose())
                    pose_ros = PoseStamped()
                    pose_ros.header.stamp = rospy.Time.now()
                    pose_ros.header.frame_id = 'world'
                    pose_ros.pose = cur_pose
                    self.__local_dataset_pose_ros = pose_ros 
                    self.__local_dataset_pose_pub.publish(self.__local_dataset_pose_ros)
                
                with self.__mapping_condition:
                    self.__mapping_condition.notify_all()
            elif self.__global_state == GlobalState.COLLECT_DATA:
                self.__mapper.collect_data(frame_current, mode=self.__collect_data_mode)
                with self.__mapping_condition:
                    self.__mapping_condition.notify_all()
            else:
                mapper_state = MapperState.IDLE
                
            # NOTE: output some information
            if self.__save_runtime_data and (frame_id % 5 == 0) and frame_id != self.last_save_frame_id:
                self.__save_current_data_callback(frame_id)
                self.__on_screenshot_callback()
            
            # NOTE: Render RGBD image
            rerender_rgbd_flag = False
            if not self.__hide_windows or self.__save_runtime_data:
                if (frame_last_received is not None and\
                    frame_id > rendering_rgbd_last_frame_id):
                    color_vis, depth_vis = self.__mapper.render_rgbd(frame_last_received, scale_modifier=1.0)
                    rgbd = np.hstack((color_vis, depth_vis))
                    if self.__save_runtime_data:
                        cv2.imwrite(str(self.render_rgbd_dir) + f'/{frame_id}.png', cv2.cvtColor(rgbd, cv2.COLOR_BGR2RGB))
                        self.__runtime_data_info['current_vis_data']['rgb_render'] = color_vis
                        self.__runtime_data_info['current_vis_data']['depth_render'] = depth_vis
                    if (not self.__hide_windows and self.__render_box.checked and\
                            frame_id % self.__render_every_slider.int_value == 0):
                        self.__o3d_cache_render_rgbd = o3d.geometry.Image(rgbd)
                    rendering_rgbd_last_frame_id = frame_id
                    rerender_rgbd_flag = True
            
            # NOTE: Update uncertainty
            rerender_uncertainty_flag = False
            if self.__get_uncertainty_flag in [self.QueryUncertaintyFlag.RUNNING]:
                max_full_uncertainty = -np.inf
                max_quality_uncertainty = -np.inf
                max_normal_uncertainty = -np.inf
                max_full_index = max_full_index = -1
                for idx in trange(len(self.__mapper.candidate_views), desc="Computing Uncertainty"):
                    # timing_get_uncertainty = start_timing()
                    view = self.__mapper.candidate_views[idx]
                    c2w = view['c2w']
                    view['full_uncertainty'], view['quality_uncertainty'], normal_uncertainty, _ = self.__mapper.get_view_uncertainty(c2w)
                    if view['full_uncertainty'] > max_full_uncertainty:
                        max_full_uncertainty = view['full_uncertainty']
                        max_full_index = idx
                    if view['quality_uncertainty'] > max_quality_uncertainty:
                        max_quality_uncertainty = view['quality_uncertainty']
                        max_quality_index = idx
                    if normal_uncertainty > max_normal_uncertainty:
                        max_normal_uncertainty = normal_uncertainty
                    # Log(f'Get view-{idx} uncertainty used {end_timing(*timing_get_uncertainty):.2f} ms. uncertainty: {view["uncertainty"]}')
                reduction = self.max_normal_uncertainty - max_normal_uncertainty
                self.max_normal_uncertainty = max_normal_uncertainty
                Log(f'Max full uncertainty: {max_full_uncertainty}, max quality uncertainty: {max_quality_uncertainty}, max normal uncertainty: {max_normal_uncertainty} (reduction: {reduction})')
                if not self.only_quality_uncertainty_flag and max_normal_uncertainty > 0.55 and not (reduction > 0 and reduction < 0.12):
                    max_index = max_full_index
                else:
                    # NOTE: If max_normal_uncertainty is small enough, choose to auto stop early
                    if self.__auto_stop and max_normal_uncertainty <= 0.05:
                        self.__local_dataset.auto_stoped = True
                        Log(f'Trigger auto stop for data collection, max normal uncertainty: {max_normal_uncertainty}')
                    # NOTE: Just use quality uncertainty
                    max_index = max_quality_index
                    self.only_quality_uncertainty_flag = True
                _, _, _, uncertainty_vis = self.__mapper.get_view_uncertainty(self.__mapper.candidate_views[max_index]['c2w'], show_image=True, only_quality=self.only_quality_uncertainty_flag)
                if (not self.__hide_windows and self.__render_box.checked and\
                        frame_id % self.__render_every_slider.int_value == 0):
                    self.__o3d_cache_render_uncertainty = o3d.geometry.Image(uncertainty_vis)
                if (self.__get_uncertainty_flag == self.QueryUncertaintyFlag.RUNNING):
                    self.__get_uncertainty_flag = self.QueryUncertaintyFlag.NONE
                    with self.__get_uncertainty_condition:
                        self.__get_uncertainty_condition.notify_all()
                        self.__get_uncertainty_condition.wait()
                    self.__trigger_update_viewpoints_flag = True
                    rerender_uncertainty_flag = True
            
            # NOTE: Update viewpoints
            rerender_candidate_frustums_flag = False
            if self.__trigger_update_viewpoints_flag:
                candidate_views = self.__mapper.candidate_views
                if not self.__hide_windows:
                    # generate frustums to visualize
                    self.candidate_frustums = self.__update_candidate_frustums(candidate_views)
                    rerender_candidate_frustums_flag = True
                self.__trigger_update_viewpoints_flag = False
            
            # NOTE: Update GUI
            self.__update_ui_mapper(
                frame_current,
                rerender_rgbd_flag,
                rerender_candidate_frustums_flag,
                rerender_uncertainty_flag)
            
            time.sleep(0.05)
        
        # NOTE: Post process
        post_process_time = 0.0
        if self.__global_state == GlobalState.POST_PROCESSING:
            self.__mapper.post_process_flag = True
            timing_post_process = start_timing()
            while not self.__mapper.post_process_finished:
            # while self.__global_state != GlobalState.QUIT:
                if self.mapping_finished:
                    print(f'Exiting post process thread')
                    break
                # NOTE: Update GUI
                self.__update_ui_mapper(
                    frame_current,
                    rerender_rgbd_flag,
                    rerender_candidate_frustums_flag,
                    rerender_uncertainty_flag)
                time.sleep(0.05)
            post_process_time = end_timing(*timing_post_process)
            Log(f'Post process finished, used {post_process_time:.2f} ms')
        
        # NOTE: Save data
        path_length = self.__traj_info['length']
        results_url = os.path.join(self.__results_dir, 'gaussians_data', 'results.txt')
        with open(results_url, 'a') as f:
            f.write(f'\n###Reconstruction Effitiency###\n')
            f.write(f'Capture Views: {self.__traj_info["capture_times"]}, ')
            f.write(f'Path Length: {path_length} m, ')
            f.write(f'Gaussians Num: {self.gaussians_num}, ')
            f.write(f'Offline Iterations: {self.offline_iterations}, ')
            f.write(f'Time cost(online): {self.online_mapping_time/1000:.2f} s, ')
            f.write(f'Time cost(offline): {post_process_time/1000:.2f} s.\n')
        
        Log(f'Move to initial pose [{self.__arm_group_states_selectable[0].value}]')
        set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(GlobalState.QUIT.value))
        self.__close_all()
    
    def __update_candidate_frustums(self, candidate_views:List[Pose]):
        candidate_frustums = []
        if self.only_quality_uncertainty_flag:
            uncertainties = [view['quality_uncertainty'] for view in candidate_views]
        else:
            uncertainties = [view['full_uncertainty'] for view in candidate_views]
        if uncertainties:
            min_uncertainty = min(uncertainties)
            max_uncertainty = max(uncertainties)
            uncertainty_range = max_uncertainty - min_uncertainty if max_uncertainty > min_uncertainty else 1.0

        for idx, view in enumerate(candidate_views):
            T_wc = view['c2w'] @ OPENCV_TO_OPENGL  # let z_axis is backward
            pose_data_o3d = OPENCV_TO_OPENGL @ T_wc @ OPENCV_TO_OPENGL
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                CANDIDATE_FRUSTUM['scale'])
            
            # Normalize uncertainty and map to a color
            if self.only_quality_uncertainty_flag:
                normalized_uncertainty = (view['quality_uncertainty'] - min_uncertainty) / uncertainty_range
            else:
                normalized_uncertainty = (view['full_uncertainty'] - min_uncertainty) / uncertainty_range
            color = matplotlib.cm.get_cmap('coolwarm')(normalized_uncertainty)[:3]  # Normalize and get RGB
            frustum.paint_uniform_color(color)
            
            candidate_frustums.append(frustum)
        return candidate_frustums
    
    def __update_ui_mapper(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]],
                        rerender_rgbd_flag:bool,
                        rerender_candidate_frustums_flag:bool,
                        rerender_uncertainty_flag:bool):
        self.receive_data(self.q_main2vis)
        if not self.__hide_windows:
            if self.__view_gaussians_box.checked == True:
                self.render_gaussian()
                    
        if not self.__hide_windows:
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_mapper(
                    rerender_rgbd_flag,
                    rerender_candidate_frustums_flag,
                    rerender_uncertainty_flag))
        
        return
    
    def __update_main_thread_ui_mapper(self,
                        rerender_rgbd_flag:bool,
                        rerender_candidate_frustums_flag:bool,
                        rerender_uncertainty_flag:bool):
        self.num_gaussians_info.text = " Number of Gaussians: {}".format(self.gaussians_num)
        self.offline_iterations_info.text = " Offline Iterations: {}".format(self.offline_iterations)
        if rerender_rgbd_flag:
            if self.__o3d_cache_gt_rgbd is not None:
                self.__rgbd_live_image.update_image(self.__o3d_cache_gt_rgbd)
            self.__rgbd_render_image.update_image(self.__o3d_cache_render_rgbd)
        if rerender_uncertainty_flag:
            self.__uncertainty_render_image.update_image(self.__o3d_cache_render_uncertainty)
        
        if rerender_candidate_frustums_flag:
            i = 0
            while True:
                if (not self.__widget_3d.scene.has_geometry(f"candidate_frustum_{i}")):
                    break
                else:
                    self.__widget_3d.scene.remove_geometry(f"candidate_frustum_{i}")
                i += 1
        
        if rerender_candidate_frustums_flag:
            for i, target_frustum in enumerate(self.candidate_frustums):
                if (target_frustum is not None):
                    self.__widget_3d.scene.add_geometry(f"candidate_frustum_{i}", target_frustum, self.__o3d_materials[CANDIDATE_FRUSTUM['material']])
        
        return

    def __update_ui_frame(self,
                        frame_current:Union[None, Dict[str, Union[torch.Tensor, int]]]):
        if not self.__update_main_thread.is_alive():
            return
        
        current_frustum = None
        current_pcd = None
        current_horizon = None
        cam_traj = None
        current_frustum_before = None
        self.__o3d_cache_gt_rgbd = None
        
        if frame_current is not None:
            rgb_data:np.ndarray = frame_current['rgb'].detach().cpu().numpy()
            depth_data:np.ndarray = frame_current['depth'].detach().cpu().numpy()
            
            pose_data:Union[np.ndarray, torch.Tensor] = frame_current['c2w']
            if isinstance(pose_data, torch.Tensor):
                pose_data = pose_data.detach().cpu().numpy()
                
            pose_quat = quaternion.from_rotation_matrix(pose_data[:3, :3])
            pose_rotation_vector_ = quaternion.as_rotation_vector(pose_quat)
            pose_rotation_vector = np.zeros(3)
            pose_rotation_vector[self.__height_direction[0]] = pose_rotation_vector_[self.__height_direction[0]]
            pose_quat_only_yaw = quaternion.from_rotation_vector(pose_rotation_vector)
            pose_data_only_yaw = deepcopy(pose_data)
            pose_data_only_yaw[:3, :3] = quaternion.as_rotation_matrix(pose_quat_only_yaw)
            pose_data_agent:np.ndarray = pose_data_only_yaw @ self.__Tcr
            pose_data_agent[:3, :3] = pose_data[:3, :3]
            
            pose_data_o3d = OPENCV_TO_OPENGL @ pose_data @ OPENCV_TO_OPENGL
            current_frustum = o3d.geometry.LineSet.create_camera_visualization(
                self.__o3d_const_camera_intrinsics,
                np.linalg.inv(pose_data_o3d),
                CURRENT_FRUSTUM['scale'])
            current_frustum.paint_uniform_color(CURRENT_FRUSTUM['color'])
            
            if 'c2w_before' in frame_current:
                if not torch.equal(frame_current['c2w'], frame_current['c2w_before']):
                    pose_data_before = frame_current['c2w_before']
                    if isinstance(pose_data_before, torch.Tensor):
                        pose_data_before = pose_data_before.detach().cpu().numpy()
                    pose_data_before_o3d = OPENCV_TO_OPENGL @ pose_data_before @ OPENCV_TO_OPENGL
                    current_frustum_before = o3d.geometry.LineSet.create_camera_visualization(
                        self.__o3d_const_camera_intrinsics,
                        np.linalg.inv(pose_data_before_o3d),
                        BEFORE_REFINE_FRUSTUM['scale'])
                    current_frustum_before.paint_uniform_color(BEFORE_REFINE_FRUSTUM['color'])
            current_gt_frustum = None
            
            if rgb_data.dtype == np.float32:
                rgb_vis = np.uint8(rgb_data * 255)
            elif rgb_data.dtype == np.uint8:
                rgb_vis = rgb_data
            else:
                raise Exception('Invalid rgb data type')
            depth_vis = depth2rgb(depth_data, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max)
            rgbd_vis = np.hstack((rgb_vis, depth_vis))
            self.__o3d_cache_gt_rgbd = o3d.geometry.Image(rgbd_vis)
            
            if self.__save_runtime_data:
                self.__runtime_data_info['current_vis_data']['rgb'] = rgb_vis
                self.__runtime_data_info['current_vis_data']['depth'] = depth_vis
            
            if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)) or np.all(depth_data == 0):
                rospy.logwarn('Depth contains NaN, Inf or all 0')
                self.__valid_depth_flag = False
            else:
                self.__o3d_pcd['current_pcd'] = rgbd_to_pointcloud(
                    rgb_vis,
                    depth_data,
                    pose_data,
                    self.__o3d_const_camera_intrinsics_o3c,
                    1000,
                    self.__rgbd_sensor.depth_max,
                    self.__device_o3c
                )
                current_pcd:o3d.t.geometry.PointCloud = self.__update_pcd(
                    'current_pcd',
                    False if self.__hide_windows else self.__current_pcd_box.checked,
                    self.__o3d_materials['unlit_mat'],
                    False)
                current_pcd_legacy:o3d.geometry.PointCloud = current_pcd.to_legacy()
                current_horizon:o3d.geometry.AxisAlignedBoundingBox = current_pcd_legacy.get_axis_aligned_bounding_box()
                current_horizon.color = CURRENT_HORIZON['color']
            
            if self.__global_state in [GlobalState.AUTO_PLANNING, GlobalState.MANUAL_CONTROL]:
                latest_location = pose_data_o3d[:3, 3].copy()
                if not self.__hide_windows:
                    # NOTE: show the trajectory
                    if len(self.__traj_info['cam_centers']) > 1:
                        if np.linalg.norm(latest_location - self.__traj_info['cam_centers'][-1]) > 0.01:
                            self.__traj_info['cam_centers'].append(latest_location)
                            self.__traj_info['length'] = compute_path_length(np.array(self.__traj_info['cam_centers']))
                            cam_traj = update_traj(self.__traj_info['cam_centers'], color_name='cool')
                    else:
                        self.__traj_info['cam_centers'].append(latest_location)
                    # NOTE: show information
                    if frame_current is not None:
                        c2w = pose_data.copy()
                        c2w = c2w @ OPENCV_TO_OPENGL # z-axis facing forward
                        log_info = ' Current cam pose(in opencv): \n{}'.format(c2w.round(3))
                        path_length = self.__traj_info['length']
                        log_info += '\n Path lengh: {:.2f} m'.format(path_length)
                        self.cam_pose_info.text = log_info
                else:
                    self.__traj_info['cam_centers'].append(latest_location)
                    self.__traj_info['length'] = compute_path_length(np.array(self.__traj_info['cam_centers']))

        if not self.__hide_windows:
            timing_update_render = start_timing()
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__update_main_thread_ui_frame(
                    current_frustum,
                    current_frustum_before,
                    current_gt_frustum,
                    current_pcd,
                    current_horizon,
                    cam_traj))
            Log(f'Update ui of frame used {end_timing(*timing_update_render):.2f} ms', tag='GUI')
            
    def __update_main_thread_ui_frame(self,
                        current_frustum:o3d.geometry.LineSet,
                        current_frustum_before:o3d.geometry.LineSet,
                        current_gt_frustum:o3d.geometry.LineSet,
                        current_pcd:o3d.geometry.PointCloud,
                        current_horizon:o3d.geometry.AxisAlignedBoundingBox,
                        cam_traj:o3d.geometry.LineSet):
        if current_frustum is not None:
            self.__widget_3d.scene.remove_geometry('current_frustum')
            self.__widget_3d.scene.add_geometry('current_frustum', current_frustum, self.__o3d_materials[CURRENT_FRUSTUM['material']])
            self.__widget_3d.scene.show_geometry('current_frustum', self.__current_frustum_box.checked)
        
        if current_frustum_before is not None:
            self.__widget_3d.scene.remove_geometry('current_frustum_before')
            self.__widget_3d.scene.add_geometry('current_frustum_before', current_frustum_before, self.__o3d_materials[BEFORE_REFINE_FRUSTUM['material']])
            self.__widget_3d.scene.show_geometry('current_frustum_before', self.__current_frustum_box.checked)
        
        if current_gt_frustum is not None:
            self.__widget_3d.scene.remove_geometry('current_gt_frustum')
            self.__widget_3d.scene.add_geometry('current_gt_frustum', current_gt_frustum, self.__o3d_materials[GT_FRUSTUM['material']])
            self.__widget_3d.scene.show_geometry('current_gt_frustum', self.__current_frustum_box.checked)
            
        if current_pcd is not None:
            self.__widget_3d.scene.remove_geometry('current_pcd')
            self.__widget_3d.scene.add_geometry(
                'current_pcd',
                current_pcd,
                self.__o3d_materials['unlit_mat'])
            self.__widget_3d.scene.show_geometry('current_pcd', self.__current_pcd_box.checked)
            
        if current_horizon is not None:
            self.__widget_3d.scene.remove_geometry('current_horizon')
            self.__widget_3d.scene.add_geometry('current_horizon', current_horizon, self.__o3d_materials['unlit_line_mat'])
            self.__widget_3d.scene.show_geometry('current_horizon', self.__current_horizon_box.checked)
        
        if self.__cam_traj_box.checked:
            if cam_traj is not None:
                self.__widget_3d.scene.remove_geometry("cam_traj")
                self.__widget_3d.scene.add_geometry("cam_traj", cam_traj, self.__o3d_materials['unlit_line_mat'])
        else:
            self.__widget_3d.scene.remove_geometry("cam_traj")
        
        return
            
    def __update_pcd(self, pointcloud_name:str, show:bool, material:Union[str, o3d.visualization.rendering.MaterialRecord]=None, update:bool=True) -> o3d.t.geometry.PointCloud:
        if isinstance(material, str):
            material = self.__o3d_materials[material]
        elif isinstance(material, o3d.visualization.rendering.MaterialRecord):
            pass
        else:
            material = self.__o3d_materials['unlit_mat']
        pcd = o3d.t.geometry.PointCloud(self.__o3d_pcd[pointcloud_name])
        if update:
            if self.__widget_3d.scene.has_geometry(pointcloud_name):
                self.__widget_3d.scene.scene.update_geometry(pointcloud_name,
                                                            pcd,
                                                            rendering.Scene.UPDATE_POINTS_FLAG +\
                                                                rendering.Scene.UPDATE_COLORS_FLAG +\
                                                                    rendering.Scene.UPDATE_NORMALS_FLAG +\
                                                                        rendering.Scene.UPDATE_UV0_FLAG)
            else:
                self.__widget_3d.scene.add_geometry(pointcloud_name, pcd, material)
            self.__widget_3d.scene.show_geometry(pointcloud_name, show)
        return pcd
    
    def set_opengl_gs(self):
        self.widget3d_width_ratio = 0.7
        self.window_w = self.__window.size.width
        self.window_h = self.__window.size.height
        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)
    
    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window
    
    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.__gaussian_scale_slider.double_value)
        self.g_renderer.set_render_mod(-4)
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)
        
    def render_gaussian(self):
        if self.__gaussian_for_render is None:
            return
        
        current_cam = self.__get_current_cam()
        
        timing_render_gaussian = start_timing()
        if self.__gaussian_color_type == GaussianColorType.Elipsoid:
            self.__render_request = {
                'cam': current_cam,
                'ready': threading.Event()
            }
            
            gui.Application.instance.post_to_main_thread(
                self.__window,
                lambda: self.__render_gaussian_main_thread(current_cam)
            )
            self.__render_request['ready'].wait()
            self.render_img = self.__render_request['result']
        else:
            self.render_img = self.__mapper.render_o3d_image(
                self.__gaussian_for_render, 
                current_cam, 
                self.__gaussian_scale_slider.double_value, 
                self.__gaussian_color_type,
                front_only=self.frontonly_chbox.checked
            )
        
        self.__widget_3d.scene.set_background([0, 0, 0, 1], o3d.geometry.Image(self.render_img))
        self.render_use_time_info.text = f' Render gaussians used {end_timing(*timing_render_gaussian):.2f} ms'

    def __render_gaussian_main_thread(self, current_cam):
        # TODO: It can render Ellipsoid, but threading issues occur when dragging and dropping.
        try:
            glfw.make_context_current(self.window_gl)
            
            glfw.poll_events()
            gl.glClearColor(0, 0, 0, 1.0)
            gl.glClear(
                gl.GL_COLOR_BUFFER_BIT
                | gl.GL_DEPTH_BUFFER_BIT
                | gl.GL_STENCIL_BUFFER_BIT
            )

            w = int(self.__window.size.width * self.widget3d_width_ratio)
            glfw.set_window_size(self.window_gl, w, self.__window.size.height)
            self.g_camera.fovy = current_cam.fovy
            self.g_camera.update_resolution(self.__window.size.height, w)
            self.g_renderer.set_render_reso(w, self.__window.size.height)
            frustum = create_frustum(
                np.linalg.inv(OPENCV_TO_OPENGL @ self.__widget_3d.scene.camera.get_view_matrix() @ OPENCV_TO_OPENGL)
            )

            self.g_camera.position = frustum.eye.astype(np.float32)
            self.g_camera.target = frustum.center.astype(np.float32)
            self.g_camera.up = frustum.up.astype(np.float32)

            self.gaussians_gl.xyz = self.__gaussian_for_render.means.cpu().numpy()
            self.gaussians_gl.opacity = self.__gaussian_for_render.opacities.cpu().numpy()
            self.gaussians_gl.scale = self.__gaussian_for_render.scales.cpu().numpy()
            self.gaussians_gl.rot = self.__gaussian_for_render.rotations.cpu().numpy()
            self.gaussians_gl.sh = self.__gaussian_for_render.harmonics.cpu().numpy()[:, 0, :]
            
            self.update_activated_renderer_state(self.gaussians_gl)
            self.g_renderer.sort_and_update(self.g_camera)
            width, height = glfw.get_framebuffer_size(self.window_gl)
            self.g_renderer.draw()
            bufferdata = gl.glReadPixels(
                0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
            )
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3).copy()
            cv2.flip(img, 0, img)
            
            glfw.swap_buffers(self.window_gl)
            
            self.__render_request['result'] = img
            self.__render_request['ready'].set()
            
        except Exception as e:
            rospy.logerr(f"Render error: {e}")
            self.__render_request['result'] = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
            self.__render_request['ready'].set()
                
    def add_camera(self, c2w:np.ndarray, name, color=[0, 1, 0], size=0.025):
        if self.__mapper_type == MapperType.GSMap:
            C2W = OPENCV_TO_OPENGL @ c2w
        frustum = create_frustum(C2W, color, size=size)
        if name not in self.frustum_dict.keys():
            frustum = create_frustum(C2W, color, size=size)
            self.__kf_combobox.add_item(name)
            self.frustum_dict[name] = frustum
            self.__widget_3d.scene.add_geometry(name, frustum.line_set, self.__o3d_materials['unlit_line_mat'])
        frustum = self.frustum_dict[name]
        frustum.update_pose(C2W)
        self.__widget_3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.__widget_3d.scene.show_geometry(name, self.__kf_viewpoints_box.checked)
        return frustum
                
    def receive_data(self, q:Queue):
        if q is None or q.empty():
            pass
        else:
            # update gaussian scene
            self.__gaussian_packet:GaussianPacket = q.get()
        
        if self.__gaussian_packet is None:
            return None
        
        if self.__gaussian_packet.iteration is not None:
            self.offline_iterations = self.__gaussian_packet.iteration

        if self.__gaussian_packet.has_gaussians:
            with self.__use_gaussian_condition:
                self.__use_gaussian_condition.acquire()
                if self.__mapper_type == MapperType.GSMap:
                    self.__gaussian_for_render = self.__gaussian_packet.gaussians
                    self.gaussians_num = self.__gaussian_for_render.means.shape[0]
                if self.__save_runtime_data and self.__get_debug_data_flag == True:
                    with open(self.__runtime_data_info['f_num_gaussians'], 'a') as f:
                        f.write(f'{self.gaussians_num}\n')
                    self.__get_debug_data_flag = False
                self.__use_gaussian_condition.release()

        if not self.__hide_windows and self.__gaussian_packet.current_frame is not None:
            self.__gaussian_packet.current_frame = self.__gaussian_packet.current_frame 
            frustum = self.add_camera(
                self.__gaussian_packet.current_frame['extrinsic'].detach().cpu().numpy(), name="current", color=[0, 1, 0]
            )
            if self.__view_gaussians_box.checked and self.followcam_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.__widget_3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
            self.__gaussian_packet.current_frame = None
        
        if not self.__hide_windows and self.__gaussian_packet.keyframes is not None:
            for iid,keyframe in enumerate(self.__gaussian_packet.keyframes):
                if self.__gaussian_packet.keyframe_colors is not None:
                    color = self.__gaussian_packet.keyframe_colors[iid]
                else:
                    color = [0, 0, 1]
                name = "keyframe_{}".format(keyframe['id'])
                c2w_np = keyframe['extrinsic'].detach().cpu().numpy()
                gt_c2w_np = keyframe['gt_extrinsic'].detach().cpu().numpy()
                self.add_camera(c2w_np, name=name, color=color)
                gt_name = "keyframe_{}_gt".format(keyframe['id'])
                self.add_camera(gt_c2w_np, name=gt_name, color=GT_FRUSTUM['color'])
            self.__gaussian_packet.keyframes = None

        if not self.__hide_windows and self.__gaussian_packet.kf_window is not None:
            self.kf_window = self.__gaussian_packet.kf_window
            self.__on_kf_window_chbox(is_checked=self.__kf_window_box.checked)
            self.__gaussian_packet.kf_window = None

    def __close_all(self):
        while not self.q_main2vis.empty():
            self.q_main2vis.get()
        self.q_main2vis = None
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        if self.__mapper is not None:
            self.__mapper.post_process_thread.join()
        if self.__hide_windows:
            rospy.signal_shutdown('Quit')
        else:
            gui.Application.instance.quit()
        Log(f'Exit main update thread', tag='ObjSplat')   

    def __get_current_cam(self):
        w2c = OPENCV_TO_OPENGL @ self.__widget_3d.scene.camera.get_view_matrix()

        H, W = int(self.__window.size.height), int(self.__widget_3d_width)
        vfov_deg = self.__widget_3d.scene.camera.get_field_of_view()
        hfov_deg = vfov_to_hfov(vfov_deg, H, W)
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, W)
        fy = fov2focal(FoVy, H)
        cx = W // 2
        cy = H // 2
        T = torch.from_numpy(w2c)
        w2c = w2c @ OPENCV_TO_OPENGL # z-axis facing forward
        
        if self.__mapper_type == MapperType.GSMap:
            from mapper.gsmap.gaussian_surfels.cameras import Camera
            c2w = np.linalg.inv(w2c)
            extrinsic = torch.from_numpy(c2w).float()
            intrinsic = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
            ).float()
            current_cam = Camera.init_from_gui(
                -1, extrinsic, intrinsic, H=H, W=W, fovx=FoVx, fovy=FoVy
            )
        
        # update ui camera pose info
        self.uicam2turntable = np.linalg.inv(w2c) # in opencv frame
        log_info = ' Current UI cam pose(in opencv): \n{}'.format(self.uicam2turntable.round(3))
        self.ui_cam_pose_info.text = log_info
        
        return current_cam
    
    def __update_dataset(self):
        with self.__local_dataset_condition:
            dataset_config = self.__local_dataset.setup()
            self.__dataset_config:GetDatasetConfigResponse = dataset_config_to_ros(dataset_config)
            rospy.Service('get_dataset_config', GetDatasetConfig, self.__get_dataset_config)
            self.__local_dataset_state = self.LocalDatasetState.INITIALIZED
            self.__local_dataset_condition.notify_all()
            self.__last_frame_np = None
            
            if self.__with_arm_turntable:
                # Initialize, move to the default arm group state
                call_move_arm_service(mode=2, group_state=self.__arm_group_states_selectable[0].value)
                call_move_turntable_service(mode=2, group_state=self.__arm_group_states_selectable[0].value)
            else:
                # just move camera to the initial pose
                # - Translation: [0.017, -0.376, 0.599]
                # - Rotation: in Quaternion [0.946, 0.008, -0.003, -0.323] # x, y, z, w
                T_tu = np.eye(4)
                T_tu[:3, 3] = np.array([0.017, -0.376, 0.599])
                T_tu[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_float_array([-0.323, 0.946, 0.008, -0.003])) # w, x, y, z
                cam_pose = matrix_to_pose(T_tu @ OPENCV_TO_GAZEBO)
                call_move_camera_service(cam_pose)
            
            while self.__global_state != GlobalState.QUIT:
                if self.__local_dataset_state == self.LocalDatasetState.INITIALIZED:
                    self.__local_dataset_state = self.LocalDatasetState.RUNNING
                    self.__local_dataset_condition.notify_all()
                if self.__local_dataset.is_finished():
                    self.__global_state = GlobalState.POST_PROCESSING
                else:
                    self.__local_dataset_condition.wait()
                if self.__global_state in [GlobalState.POST_PROCESSING, GlobalState.QUIT]:
                    break
                
                if self.__global_state != GlobalState.COLLECT_DATA:
                    update_result = self.__local_dataset.update()
                    if not update_result:
                        # online mapping time cost
                        self.online_mapping_time = end_timing(*self.timing_online_mapping)
                        Log(f'Online mapping finished, used {self.online_mapping_time:.2f} ms')
                        if self.eval_visual_quality:
                            # NOTE: Collect test views for evaluation
                            set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(GlobalState.COLLECT_DATA.value))
                            self.__global_state = GlobalState.COLLECT_DATA
                            self.__local_dataset.capture_times = 0
                            self.__local_dataset.finished_flag = False
                            with self.__local_dataset_condition:
                                self.__local_dataset_condition.notify_all()
                            continue
                        else:
                            # NOTE: Directly do post-processing
                            rospy.logwarn('Object Reconstruction is finished. Quitting...')
                            self.__global_state = GlobalState.POST_PROCESSING
                            break
                capture_times, capture_num = self.__local_dataset.get_capture_info()
                object_id = self.__local_dataset.get_object_id()
                self.__local_dataset_label.text = f'Object ID: {object_id}\nCapture: {capture_times + 1}/{capture_num}'
                Log(f"Object ID: {object_id}, Capture: {capture_times + 1}/{capture_num}")
                cur_frame_np, self.__cur_data_path = self.__local_dataset.get_frame()
                
                # NOTE: Segmentation
                result, masks_image = self.call_seg_service(object_id)
                if self.__rgbd_sensor.downsample_factor > 1:
                    # resize cur_frame_np['scene_rgb'] and cur_frame_np['scene_depth_np']
                    cur_frame_np['scene_rgb'] = cv2.resize(cur_frame_np['scene_rgb'], (self.__rgbd_sensor.width, self.__rgbd_sensor.height), interpolation=cv2.INTER_LINEAR)
                    cur_frame_np['scene_depth_np'] = cv2.resize(cur_frame_np['scene_depth_np'], (self.__rgbd_sensor.width, self.__rgbd_sensor.height), interpolation=cv2.INTER_NEAREST)
                    masks_image = cv2.resize(masks_image, (self.__rgbd_sensor.width, self.__rgbd_sensor.height), interpolation=cv2.INTER_NEAREST)
                if masks_image is not None:
                    objs_pc, objs_depth_np, objs_rgb_np, objs_mask = self.__local_dataset.extract_object_point_cloud(self.__rgbd_sensor, cur_frame_np, masks_image)
                    cur_frame_np['object_pc'] = objs_pc[1]
                    cur_frame_np['object_mask'] = objs_mask[1]
                    cur_frame_np['object_rgb'] = objs_rgb_np[1]
                    cur_frame_np['object_depth'] = objs_depth_np[1]
                    if self.__last_frame_np is not None and self.__refine_pose_flag:
                        # error_transformation = self.__refine_frame_pose(cur_frame_np, self.__last_frame_np)
                        error_transformation = self.__refine_frame_pose_use_kissmatcher(cur_frame_np, self.__last_frame_np)
                        cur_frame_np['c2w_before'] = cur_frame_np['c2w']
                        if not np.allclose(error_transformation, np.identity(4)):
                            cur_frame_np['c2w'] = error_transformation @ cur_frame_np['c2w']
                            if 'gt_c2w' in cur_frame_np:
                                before_translation_error, before_rotation_error = compute_pose_error(cur_frame_np['c2w_before'], cur_frame_np['gt_c2w'])
                                refined_translation_error, refined_rotation_error = compute_pose_error(cur_frame_np['c2w'], cur_frame_np['gt_c2w'])
                                Log(
                                    f'Pose refinement: \n'
                                    f'translation error: {before_translation_error * 1000:.3f} mm -> {refined_translation_error * 1000:.3f} mm,\n'
                                    f'rotation error: {before_rotation_error:.3f} deg -> {refined_rotation_error:.3f} deg',
                                    tag='ObjSplat'
                                )
                                # print(f'error transformation:\n{error_transformation}')
                    else:
                        # use gt_c2w as c2w
                        if 'gt_c2w' in cur_frame_np:
                            cur_frame_np['c2w'] = cur_frame_np['gt_c2w']
                else:
                    rospy.logwarn(f'Segmentation failed.')
                    continue
                
                frame_c2w = cur_frame_np['c2w'] @ OPENCV_TO_OPENGL
                pose_change_type = self.__is_pose_changed(frame_c2w)
                if pose_change_type != PoseChangeType.NONE:
                    self.__frame_c2w_last = frame_c2w
                    frame_torch = {
                        'scene_rgb': torch.from_numpy(cur_frame_np['scene_rgb']),
                        'rgb': torch.from_numpy(cur_frame_np['object_rgb']),
                        'depth': torch.from_numpy(cur_frame_np['object_depth']),
                        'mask': torch.from_numpy(cur_frame_np['object_mask']),
                        'c2w': torch.from_numpy(frame_c2w)}
                    if 'c2w_before' in cur_frame_np:
                        frame_c2w_before = cur_frame_np['c2w_before'] @ OPENCV_TO_OPENGL
                        frame_torch['c2w_before'] = torch.from_numpy(frame_c2w_before)
                    if 'gt_c2w' in cur_frame_np:
                        gt_c2w = cur_frame_np['gt_c2w'] @ OPENCV_TO_OPENGL
                        frame_torch['gt_c2w'] = torch.from_numpy(gt_c2w)
                    self.__last_frame_np = deepcopy(cur_frame_np)
                    self.__update_ui_frame(frame_torch)
                    self.__local_dataset.capture_times += 1 # successful capture
                    self.__frames_cache.put(frame_torch) # NOTE: Block and wait for mapper to finish
                else:
                    rospy.logwarn('Moved but not pose changed.')
                self.__local_dataset_condition.notify_all()
            self.__local_dataset.close()
    
    def __refine_frame_pose_use_kissmatcher(self, cur_frame_np, last_frame_np):
        # Save two point clouds in `.ply` format
        # use self.__local_dataset.results_dir to save the temporary point clouds
        cur_frame_path = self.__local_dataset.results_dir + '/cur_frame.ply'
        last_frame_path = self.__local_dataset.results_dir + '/last_frame.ply'
        
        # Extract point cloud data
        cur_frame_points = cur_frame_np['object_pc']["points"]
        last_frame_points = last_frame_np['object_pc']["points"]
        cur_pose = cur_frame_np['c2w'].copy()
        cur_pose[:3, 3] = cur_pose[:3, 3] * 1000.0 # mm
        last_pose = last_frame_np['c2w'].copy()
        last_pose[:3, 3] = last_pose[:3, 3] * 1000.0 # mm
        
        # Create Open3D point cloud objects
        cur_frame_pcd = o3d.geometry.PointCloud()
        cur_frame_pcd.points = o3d.utility.Vector3dVector(cur_frame_points)
        cur_frame_pcd.transform(cur_pose)
        last_frame_pcd = o3d.geometry.PointCloud()
        last_frame_pcd.points = o3d.utility.Vector3dVector(last_frame_points)
        last_frame_pcd.transform(last_pose)
        
        # Save point clouds to `.ply` files
        o3d.io.write_point_cloud(cur_frame_path, cur_frame_pcd)
        o3d.io.write_point_cloud(last_frame_path, last_frame_pcd)
        
        # Call the object alignment service
        req: ObjectAlignmentRequest = ObjectAlignmentRequest()
        req.src_data_path = cur_frame_path
        req.target_data_path = last_frame_path
        req.resolution = 1.0
        rep: ObjectAlignmentResponse = self.__object_alignment_service(req)
        
        # Process the response
        if rep.result:
            T = pose_to_matrix(rep.pose)
            T[:3, 3] = T[:3, 3] / 1000.0 # mm -> m
        else:
            rospy.logwarn('Object alignment failed.')
            T = np.identity(4)
        return T
    
    # NOTE: callback functions for GUI
            
    def __window_on_layout(self, ctx:gui.LayoutContext):
        em = ctx.theme.font_size

        panel_width = 23 * em
        rect:gui.Rect = self.__window.content_rect

        self.__panel_control.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)
        x = self.__panel_control.frame.get_right()
        
        # 3D widget width
        self.__widget_3d_width = rect.width - 2*panel_width

        self.__widget_3d.frame = gui.Rect(x, rect.y, rect.get_right() - 2*panel_width, rect.height)
        self.__panel_visualize.frame = gui.Rect(self.__widget_3d.frame.get_right(), rect.y, panel_width, rect.height)

        return
        
    def __window_on_close(self) -> bool:
        self.__global_state = GlobalState.QUIT
        self.__mapper.mapping_finished = True
        if self.__local_dataset is not None:
            with self.__local_dataset_condition:
                self.__local_dataset_condition.notify_all()
            self.__local_dataset_thread.join()
        self.__update_main_thread.join()
        gui.Application.instance.quit()
        return True

    def __widget_3d_on_key(self, event:gui.KeyEvent):
        movement_flag = 'None'
        if self.__global_state in [GlobalState.MANUAL_CONTROL] and event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.UP:
                movement_flag = 'next_scan'
            elif event.key == gui.KeyName.RIGHT:
                movement_flag = 'only_scan'
            else:
                return gui.Widget.IGNORED
            self.__movement_flag_pub.publish(movement_flag)
            return gui.Widget.HANDLED
        return gui.Widget.IGNORED
    
    def __on_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            parts = name.split('_')
            if parts[0] == "ui":
                continue # skip ui cameras
            if len(parts) == 2:
                self.__widget_3d.scene.show_geometry(name, is_checked)
        
    def __on_gt_cameras_chbox(self, is_checked, name=None):
        names = self.frustum_dict.keys() if name is None else [name]
        for name in names:
            parts = name.split('_')
            if len(parts) == 3 and parts[2] == "gt":
                self.__widget_3d.scene.show_geometry(name, is_checked)
    
    def __on_candidate_cameras_chbox(self, is_checked):
        i = 0
        while True:
            geom_name = f"candidate_frustum_{i}"
            if not self.__widget_3d.scene.has_geometry(geom_name):
                break
            self.__widget_3d.scene.show_geometry(geom_name, is_checked)
            i += 1
    
    def __on_kf_window_chbox(self, is_checked):
        if self.kf_window is None:
            return
        edge_cnt = 0
        for key in self.kf_window.keys():
            for kf_idx in self.kf_window[key]:
                name = "kf_edge_{}".format(edge_cnt)
                edge_cnt += 1
                if "keyframe_{}".format(key) not in self.frustum_dict.keys():
                    continue
                test1 = self.frustum_dict["keyframe_{}".format(key)].view_dir[1]
                kf = self.frustum_dict["keyframe_{}".format(kf_idx)].view_dir[1]
                points = [test1, kf]
                lines = [[0, 1]]
                colors = [[0, 1, 0]]

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)

                if is_checked:
                    self.__widget_3d.scene.remove_geometry(name)
                    self.__widget_3d.scene.add_geometry(name, line_set, self.__o3d_materials['unlit_line_mat'])
                else:
                    self.__widget_3d.scene.remove_geometry(name)

    def keyframe_combobox_callback(self, new_val, new_idx):
        frustum = self.frustum_dict[new_val]
        viewpoint = frustum.view_dir
        self.__widget_3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
        return gui.Combobox.HANDLED
    
    def __apply_capture(self, req:ApplyCaptureRequest) -> ApplyCaptureResponse:
        rep = ApplyCaptureResponse()
        self.__collect_data_mode = req.mode
        if self.__local_dataset is None:
            rospy.logwarn('Local dataset is None.')
            rep.success = False
        else:
            if not req.views:
                rospy.logdebug('Apply capture directly.')
                with self.__local_dataset_condition:
                    self.__local_dataset_condition.notify_all()
                    self.__local_dataset_condition.wait()
            else:
                for view in req.views:
                    if (not self.__collect_data_flag) and self.__global_state in [GlobalState.COLLECT_DATA, GlobalState.POST_PROCESSING, GlobalState.QUIT]:
                        if self.eval_visual_quality and self.__global_state == GlobalState.COLLECT_DATA:
                            self.__collect_data_flag = True
                        else:
                            Log(f'Stop applying capture', tag='ObjSplat')
                            break
                    flag = self.__move_to_pose(view)
                    if not flag:
                        rospy.logwarn('Failed to move to pose: {}'.format(view))
                        rep.fail_views.append(view)
                        continue
                    with self.__local_dataset_condition:
                        self.__local_dataset_condition.notify_all()
                        self.__local_dataset_condition.wait()
                    with self.__mapping_condition:
                        self.__mapping_condition.wait()
            rep.success = True
        return rep
    
    def __set_global_state(self, request:SetPlannerStateRequest) -> SetPlannerStateResponse:
        rospy.loginfo(f'Set global state: {request.global_state}')
        self.__global_state = GlobalState(request.global_state)
        with self.__local_dataset_condition:
            self.__local_dataset_condition.notify_all()
        return SetPlannerStateResponse()
    
    def __move_to_pose(self, T_tu):
        if isinstance(T_tu, Pose):
            T_tu = pose_to_matrix(T_tu)
        
        if not self.__with_arm_turntable:
            # Directly control camera pose without using robotic arms and turntables
            cam_pose = matrix_to_pose(T_tu @ OPENCV_TO_GAZEBO)
            apply_movement_flag = call_move_camera_service(cam_pose)
            return apply_movement_flag
        
        cur_turntable_pose, cur_joint_values = call_get_turntable_pose_service()
        if cur_turntable_pose is None:
            return False
        
        T_bt = pose_to_matrix(cur_turntable_pose)
        T_tb = np.linalg.inv(T_bt)
        
        def project_to_plane(v, n):
            n = n / np.linalg.norm(n)  # Normalize the normal vector
            return v - np.dot(v, n) * n
        z_axis = np.array([0, 0, 1])
        v1 = project_to_plane(T_tu[:3, 3], z_axis)
        v2 = project_to_plane(T_tb[:3, 3], z_axis)
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        cross = np.cross(v1, v2)
        if cross[2] < 0:
            theta = -theta
        theta += np.pi / 5 # add bias angle to better view the object 
        Log(f'Truntable to UI camera pose angle: {np.rad2deg(theta):.2f} degree', tag='ObjSplat')
        
        if np.abs(theta) > np.pi / 12:
            target_joint_value = (cur_joint_values[0] + theta + np.pi) % (2 * np.pi) - np.pi # -pi ~ pi
            call_move_turntable_service(mode=1, joint_value=target_joint_value)
            
            # update turntable pose
            cur_turntable_pose, _ = call_get_turntable_pose_service()
            T_bt = pose_to_matrix(cur_turntable_pose)
        
        T_bu = T_bt @ T_tu
        cam_pose = matrix_to_pose(T_bu)
        apply_movement_flag = call_move_camera_service(cam_pose)
        return apply_movement_flag
    
    def __get_recon_info(self, req:GetReconInfoRequest) -> GetReconInfoResponse:
        rep = GetReconInfoResponse()
        object_bound = self.__mapper.get_object_bound()
        rep.object_bound_min = object_bound[0].flatten().tolist()
        rep.object_bound_max = object_bound[1].flatten().tolist()
        rep.current_pose = matrix_to_pose(self.__mapper.get_current_pose())
        return rep

    def __get_uncertainty(self, req:GetUncertaintyRequest) -> GetUncertaintyResponse:
        with self.__get_uncertainty_condition:
            cadidate_views = [{'id':idx, 'c2w': pose_to_matrix(pose)} for idx, pose in enumerate(req.candidate_views)]
            self.__mapper.set_candidate_views(cadidate_views)
            self.__get_uncertainty_flag = self.QueryUncertaintyFlag.RUNNING
            self.__get_uncertainty_condition.wait()
            if self.only_quality_uncertainty_flag:
                uncertainties = [view['quality_uncertainty'] for view in self.__mapper.candidate_views]
            else:
                uncertainties = [view['full_uncertainty'] for view in self.__mapper.candidate_views]
            self.__get_uncertainty_condition.notify_all()
        
        rep = GetUncertaintyResponse()
        rep.candidate_views_uncertainty = uncertainties
        return rep
    
    # NOTE: Common Funtions
    
    def __is_pose_changed(self, frame_c2w:np.ndarray) -> PoseChangeType:
        if self.__frame_c2w_last is None:
            self.__frame_c2w_last = frame_c2w
            return PoseChangeType.BOTH
        else:
            return is_pose_changed(
                self.__frame_c2w_last,
                frame_c2w,
                self.__frame_update_translation_threshold,
                self.__frame_update_rotation_threshold)
    
    def __save_current_data_callback(self, frame_id):
        current_vis_data_dir = self.__runtime_data_info['current_vis_data_dir'] + f'/capture_{self.__traj_info["capture_times"]}'
        os.makedirs(current_vis_data_dir, exist_ok=True)
        if self.__runtime_data_info['current_vis_data'] is not None:
            for key, value in self.__runtime_data_info['current_vis_data'].items():
                if value.shape[2] == 3:
                    value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{str(current_vis_data_dir)}/{key}.png', value)
        Log(f'Save current data done', tag='GUI')
        self.last_save_frame_id = frame_id
        return
    
    def __on_screenshot_callback(self):
        if self.render_img is None:
            return
        from datetime import datetime
        dt = datetime.now().strftime("%H-%M-%S")
        height = self.__window.size.height
        width = self.__widget_3d_width
        app = gui.Application.instance
        img = np.asarray(app.render_to_image(self.__widget_3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{self.__results_dir}/cam-{dt}.png", img)
        
        img = np.asarray(self.render_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{self.__results_dir}/{dt}.png", img)
    
    def __move_arm_group_state_callback(self):
        arm_group_state = ArmGroupState(self.__arm_group_state_combobox.selected_text)
        call_move_arm_service(mode=2, group_state=arm_group_state.value)
        return gui.Combobox.HANDLED
    
    def __turn2uicam_callback(self):
        if self.uicam2turntable is not None:
            T_tu = self.uicam2turntable.copy()
            req = ApplyCaptureRequest()
            req.views = [T_tu]
            self.__apply_capture(req)
        return gui.Combobox.HANDLED
    
    # NOTE: callback functions for ROS
    
    def __global_state_callback(self, global_state_str:str, global_state_index:int):
        global_state = GlobalState(global_state_str)
        set_planner_state_response:SetPlannerStateResponse = self.__set_planner_state_service(SetPlannerStateRequest(global_state_str))
        if global_state == self.__global_state:
            return gui.Combobox.HANDLED
        self.__global_state = global_state
        return gui.Combobox.HANDLED
    
    def __arm_group_state_callback(self, arm_group_state_str:str, arm_group_state_index:int):
        arm_group_state = ArmGroupState(arm_group_state_str)
        if arm_group_state == self.__arm_group_state:
            return gui.Combobox.HANDLED
        self.__arm_group_state = arm_group_state
        return gui.Combobox.HANDLED
    
    def __get_dataset_config(self, req:GetDatasetConfigRequest) -> GetDatasetConfigResponse:
        return self.__dataset_config
    
    def call_seg_service(self, object_id):
        rospy.wait_for_service('grounded_sam2')
        try:
            seg_service = rospy.ServiceProxy('grounded_sam2', ImageSeg)
            request = ImageSegRequest()
            request.object_name = object_id
            request.cur_data_path = self.__cur_data_path
            response: ImageSegResponse = seg_service(request)
            if response.masks_image.data:
                br = CvBridge()
                masks_image = br.imgmsg_to_cv2(response.masks_image, desired_encoding='mono8')
                return response.result, masks_image
            else:
                return response.result, None
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False, None
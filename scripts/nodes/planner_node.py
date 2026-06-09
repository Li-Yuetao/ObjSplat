#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import argparse
import threading
import json
from typing import List
from enum import Enum
import faulthandler
import torch
import numpy as np
import random
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose

from utils import PROJECT_NAME, GlobalState, start_timing, end_timing
from utils.logging_utils import Log
from utils.gui_utils import matrix_to_pose, pose_to_matrix
from planner.planner import generate_candidate_views, select_path_poses_networkx_multiobjective, interpolate_pose, generate_circular_candidate_views
from scripts.nodes import \
    GetDatasetConfig, GetDatasetConfigResponse, GetDatasetConfigRequest,\
        SetPlannerState, SetPlannerStateResponse, SetPlannerStateRequest,\
            ApplyCapture, ApplyCaptureRequest, ApplyCaptureResponse,\
                GetReconInfo, GetReconInfoRequest, GetReconInfoResponse,\
                        GetUncertainty, GetUncertaintyRequest, GetUncertaintyResponse

class ViewStrategyType(Enum):
    Random = 'Random'
    Circle = 'Circle'
    NBV = 'NBV'
    NBV_1_NBP = 'NBV_1_NBP'
    NBP_R = 'NBP_R'
    NBP_P = 'NBP_P'

class PlannerNode:
    
    def __init__(
        self,
        config_url:str,
        hide_windows:bool,
        view_strategy: str="") -> None:
        self.__hide_windows = hide_windows
        os.chdir(PACKAGE_PATH)
        rospy.loginfo(f'Current working directory: {os.getcwd()}')
        with open(config_url) as f:
            config = json.load(f)
        
        capture_num = rospy.get_param('capture_num', -1)
        self.capture_num = config['dataset']['capture_num'] if capture_num == -1 else capture_num
        self.__view_strategy = ViewStrategyType(config['planner']['view_strategy'] if view_strategy == "None" else view_strategy)
        self.__capture_times = 0
        self.candidate_views = []

        self.__global_state = None
        self.__global_state_condition = threading.Condition()
        rospy.Service('set_planner_state', SetPlannerState, self.__set_planner_state)
        with self.__global_state_condition:
            self.__global_state_condition.wait()
        
        self.__get_dataset_config_service = rospy.ServiceProxy('get_dataset_config', GetDatasetConfig)
        rospy.wait_for_service('get_dataset_config')
        self.__setup_for_episode()
        self.__apply_capture_service = rospy.ServiceProxy('apply_capture', ApplyCapture)
        rospy.wait_for_service('apply_capture')
        self.__get_recon_info_service = rospy.ServiceProxy('get_recon_info', GetReconInfo)
        rospy.wait_for_service('get_recon_info')
        self.__trigger_update_viewpoints_pub = rospy.Publisher('trigger_update_viewpoints', Bool, queue_size=1)
        self.__get_viewpoints_uncertainty_service = rospy.ServiceProxy('get_uncertainty', GetUncertainty)
        rospy.wait_for_service('get_uncertainty')
        rospy.Subscriber('current_pose', PoseStamped, self.__current_pose_callback)
        self.__set_global_state_service = rospy.ServiceProxy('set_global_state', SetPlannerState)
        
        while not rospy.is_shutdown() and self.__global_state != GlobalState.QUIT:
            if self.__global_state is not GlobalState.AUTO_PLANNING:
                if self.__global_state == GlobalState.QUIT:
                    break
                elif self.__global_state == GlobalState.COLLECT_DATA:
                    # NOTE: generate candidate views
                    self.__generate_candidate_views(object_bound_min, object_bound_max)
                    self.__run_evaluation_capture()
                    self.__set_global_state_service(SetPlannerStateRequest(GlobalState.POST_PROCESSING.value))
                with self.__global_state_condition:
                    self.__global_state_condition.wait()
                continue
            else:
                if self.__viewpoint_planning_flag:
                    # NOTE: get newest bbox of the object
                    get_recon_info_response:GetReconInfoResponse = self.__get_recon_info_service(GetReconInfoRequest())
                    object_bound_min = np.array(get_recon_info_response.object_bound_min)
                    object_bound_max = np.array(get_recon_info_response.object_bound_max)
                    current_pose = pose_to_matrix(get_recon_info_response.current_pose)
                    if np.all(object_bound_min == object_bound_max):
                        continue
                    # NOTE: generate candidate views
                    candidate_points_arr, center, sphere_radius = self.__generate_candidate_views(object_bound_min, object_bound_max)
                    
                    self.__trigger_update_viewpoints_pub.publish(Bool(True))
                    
                    if self.__view_strategy == ViewStrategyType.Random:
                        if not self.candidate_views:
                            print("No candidate views available for Random strategy.")
                            next_path = []
                        else:
                            random_index = random.randint(0, len(self.candidate_views) - 1)
                            next_path = [self.candidate_views[random_index]]
                    else:
                        # NOTE: get uncertainty of candidate views
                        req = GetUncertaintyRequest()
                        req.candidate_views = self.candidate_views
                        rep:GetUncertaintyResponse = self.__get_viewpoints_uncertainty_service(req)
                        candidate_views_uncertainty = np.array(rep.candidate_views_uncertainty)
                        Log(f'Uncertainty mean: {candidate_views_uncertainty.mean():.2f}, max: {candidate_views_uncertainty.max():.2f}, min: {candidate_views_uncertainty.min():.2f}')
                        candidate_views_uncertainty = candidate_views_uncertainty / np.max(candidate_views_uncertainty) # normalize to [0, 1]
                        max_uncertainty_index = np.argmax(candidate_views_uncertainty)
                        
                        timing_select_path = start_timing()
                        if self.__view_strategy == ViewStrategyType.NBV or self.__view_strategy == ViewStrategyType.NBV_1_NBP:
                            # NOTE: use max uncertainty pose
                            target_view_pose = self.candidate_views[max_uncertainty_index]
                            next_path = [target_view_pose]
                            if self.__view_strategy == ViewStrategyType.NBV_1_NBP:
                                self.__view_strategy = ViewStrategyType.NBP_P # switch to NBP_P after first NBV
                        elif self.__view_strategy == ViewStrategyType.NBP_P or self.__view_strategy == ViewStrategyType.NBP_R:
                            graph_path_poses = select_path_poses_networkx_multiobjective(
                                candidate_points_arr=candidate_points_arr,
                                candidate_view_poses=self.candidate_views,
                                candidate_views_uncertainty=candidate_views_uncertainty,
                                current_pose=current_pose,
                                target_index=max_uncertainty_index,
                                alpha=0.1,
                                lambda_weight=0.5,
                                max_paths=20,
                                min_nodes=2
                            )
                            if self.__view_strategy == ViewStrategyType.NBP_R:
                                # NOTE: just select the first pose in the path
                                next_path = [graph_path_poses[1]]
                            else:
                                next_path = graph_path_poses[1:] # NOTE: remove the pose near the current pose
                            # next_path = graph_path_poses[1] # NOTE: select the first pose in NBP
                        elif self.__view_strategy == ViewStrategyType.Circle:
                            # NOTE: generate circular path around the object
                            graph_path_poses = generate_circular_candidate_views(
                                center=center,
                                sphere_radius=0.4,
                                latitude_deg=45.0,
                                num_views=self.capture_num,
                            )
                            next_path = graph_path_poses
                                
                        rospy.logdebug(f'Select path used {end_timing(*timing_select_path):.2f} ms')
                    
                    req = ApplyCaptureRequest()
                    req.views = next_path
                    try:
                        apply_capture_response:ApplyCaptureResponse = self.__apply_capture_service(req)
                        if len(apply_capture_response.fail_views) > 0:
                            self.__visited_poses.extend(apply_capture_response.fail_views) # TODO: add failed views
                        if apply_capture_response.success:
                            self.__capture_times += 1
                    except rospy.ServiceException as e:
                        rospy.logerr(f'Apply capture service call failed: {e}')
                        self.__global_state = GlobalState.QUIT
                        break
                    Log(f'Capture success!, move to next pose.')
                else:
                    Log(f'Planner has finished the scan.')
        Log(f'Planner node finished.')
    
    def __generate_candidate_views(self, object_bound_min, object_bound_max, num_candidate_points=120, latitude_lower_deg=10.0, latitude_upper_deg=85.0):
        timing_generate_candidate_views = start_timing()
        candidate_points_arr, self.candidate_views, center, sphere_radius = generate_candidate_views(
                                object_bound_min, 
                                object_bound_max, 
                                self.__focal_length,
                                num_candidate_points=num_candidate_points,
                                latitude_lower_deg=latitude_lower_deg,
                                latitude_upper_deg=latitude_upper_deg,
                                visited_poses=self.__visited_poses)
        rospy.logdebug(f'Generate candidate views used {end_timing(*timing_generate_candidate_views):.2f} ms')
        return candidate_points_arr, center, sphere_radius
    
    def __setup_for_episode(self) -> None:
        self.__dataset_config:GetDatasetConfigResponse = self.__get_dataset_config_service(GetDatasetConfigRequest())
        self.__results_dir = self.__dataset_config.results_dir
        self.__focal_length = self.__dataset_config.focal_length
        os.makedirs(self.__results_dir, exist_ok=True)
        
        self.__visited_poses: List[Pose] = []
        if self.__global_state == GlobalState.AUTO_PLANNING:
            self.__viewpoint_planning_flag = True
        else:
            self.__viewpoint_planning_flag = False
    
    def __run_evaluation_capture(self, n_samples: int = 10) -> None:
        Log("Starting evaluative data acquisition phase...")

        # 1. Generate in-trajectory views
        in_trajectory_views = self.__generate_in_trajectory_views(self.__visited_poses, interpolate_num=1)
        if n_samples < len(in_trajectory_views):
            in_trajectory_views = random.sample(in_trajectory_views, n_samples) # randomly sample

        # 2. Generate novel views
        if n_samples > len(self.candidate_views):
            Log(f'Warning: Requested number of novel views > candidate views. Reducing from {n_samples} to {self.candidate_views}.', tag='War')
            n_samples = len(self.candidate_views)
        novel_views = random.sample(self.candidate_views, n_samples) # randomly sample
        
        # 3. Execute capture
        Log("Capturing in-trajectory views...")
        # self.__execute_capture_for_evaluation(in_trajectory_views, mode='traj')
        Log("Capturing novel views...")
        self.__execute_capture_for_evaluation(novel_views, mode='novel')

        Log("Evaluative data acquisition finished.")

    def __generate_in_trajectory_views(self, visited_poses: list, interpolate_num: int = 1):
        if len(visited_poses) < 2:
            Log("Not enough poses to interpolate in-trajectory views.", tag='War')
            return []

        interpolated_views = []

        for i in range(len(visited_poses) - 1):
            pose1, pose2 = visited_poses[i], visited_poses[i + 1]
            T_i = pose_to_matrix(pose1)
            T_j = pose_to_matrix(pose2)

            # NOTE: Interpolate n_samples poses between pose1 and pose2
            interpolated_poses = [
                interpolate_pose(T_i, T_j, lam)
                for lam in np.linspace(0, 1, interpolate_num + 2)[1:-1]  # Exclude 0 and 1
            ]
            # NOTE: Convert interpolated poses back to Pose format
            interpolated_views.extend(matrix_to_pose(pose) for pose in interpolated_poses)

        Log(f"Generated {len(interpolated_views)} in-trajectory views from {len(visited_poses)} keyframes.")
        return interpolated_views
    
    def __execute_capture_for_evaluation(self, views_to_capture, mode='traj'):
        req = ApplyCaptureRequest()
        req.views = views_to_capture
        req.mode = mode
        try:
            Log(f'Executing capture for {len(views_to_capture)} views in mode: {mode}.')
            apply_capture_response:ApplyCaptureResponse = self.__apply_capture_service(req)
            if apply_capture_response.success:
                self.__capture_times += 1
                return True
        except rospy.ServiceException as e:
            rospy.logerr(f'Apply capture service call failed: {e}')
            self.__global_state = GlobalState.QUIT
            return False
        
    # NOTE: ros callback functions
    
    def __set_planner_state(self, request:SetPlannerStateRequest) -> SetPlannerStateResponse:
        rospy.loginfo(f'Set planner state: {request.global_state}')
        if self.__global_state is None:
            self.__global_state = GlobalState(request.global_state)
            with self.__global_state_condition:
                self.__global_state_condition.notify_all()
        else:
            global_state_old = GlobalState(self.__global_state)
            self.__global_state = GlobalState(request.global_state)
            if (self.__global_state == GlobalState.AUTO_PLANNING) and (global_state_old is not GlobalState.AUTO_PLANNING):
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
            if self.__global_state == GlobalState.COLLECT_DATA or self.__global_state == GlobalState.QUIT:
                with self.__global_state_condition:
                    self.__global_state_condition.notify_all()
        return SetPlannerStateResponse()
    
    def __current_pose_callback(self, cur_pose:PoseStamped) -> None:
        self.__visited_poses.append(cur_pose.pose)
        return

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} planner node.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--hide_windows',
                        type=int,
                        required=True,
                        help='Disable windows.')
    parser.add_argument('--debug',
                        type=int,
                        required=False,
                        help='Debug mode, output more logs.')
    parser.add_argument('--view_strategy',
                        type=str,
                        required=False,
                        default="None",
                        help='View planning strategy, override config file if set.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.DEBUG if bool(args.debug) else rospy.INFO)
    
    PlannerNode(args.config, bool(args.hide_windows), view_strategy=args.view_strategy)
    
    rospy.loginfo(f'{PROJECT_NAME} planner node finished.')
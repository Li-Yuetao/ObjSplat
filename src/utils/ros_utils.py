from typing import Union
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from scripts.nodes import StrCam_Capture, StrCam_CaptureRequest, StrCam_CaptureResponse,\
    MoveCam, MoveCamRequest, MoveCamResponse,\
        GetCamPose, GetCamPoseRequest, GetCamPoseResponse,\
            GetTurntablePose, GetTurntablePoseRequest, GetTurntablePoseResponse,\
                MoveTurntable, MoveTurntableRequest, MoveTurntableResponse,\
                    MoveArm, MoveArmRequest, MoveArmResponse,\
                        MoveCam, MoveCamRequest, MoveCamResponse

# NOTE: ROS service call
def call_capture_service():
    rospy.wait_for_service('/zk_camera_node/capture_once')
    try:
        move_arm_service = rospy.ServiceProxy('/zk_camera_node/capture_once', StrCam_Capture)
        response:StrCam_CaptureResponse = move_arm_service()
        return response.result, response.cur_data_path
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False, ''
    
def call_move_camera_service(pose:Pose) -> bool:
    rospy.wait_for_service('/robot_server/move_camera')
    try:
        move_arm_service = rospy.ServiceProxy('/robot_server/move_camera', MoveCam)
        req:MoveCamRequest = MoveCamRequest()
        req.cam_pose = pose
        response:MoveCamResponse = move_arm_service(req)
        return response.result
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False
    
def call_move_arm_service(mode:int, joint_values:np.ndarray=np.zeros(6), group_state:str='ready') -> bool:
    rospy.wait_for_service('/robot_server/move_arm')
    try:
        move_arm_service = rospy.ServiceProxy('/robot_server/move_arm', MoveArm)
        req:MoveArmRequest = MoveArmRequest()
        req.mode = mode # 0: arm pose, 1: joint values 2: group state name
        req.joint_values = joint_values.tolist()
        req.group_state = group_state
        response:MoveArmResponse = move_arm_service(req)
        return response.result
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False
    
def call_get_camera_pose_service() -> Pose:
    rospy.wait_for_service('/robot_server/get_cam_pose')
    try:
        get_camera_pose_service = rospy.ServiceProxy('/robot_server/get_cam_pose', GetCamPose)
        response:GetCamPoseResponse = get_camera_pose_service()
        return response.cam_pose
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return None
    
def call_get_turntable_pose_service() -> Union[Pose, float]:
    rospy.wait_for_service('/robot_server/get_turntable_pose')
    try:
        get_turntable_pose_service = rospy.ServiceProxy('/robot_server/get_turntable_pose', GetTurntablePose)
        response:GetTurntablePoseResponse = get_turntable_pose_service()
        return response.turntable_pose, response.joint_values
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return None, 0.0
    
def call_move_turntable_service(mode:int, joint_value:float=0.0, group_state:str='ready') -> bool:
    rospy.wait_for_service('/robot_server/move_turntable')
    try:
        move_turntable_service = rospy.ServiceProxy('/robot_server/move_turntable', MoveTurntable)
        req:MoveTurntableRequest = MoveTurntableRequest()
        req.mode = mode
        req.joint_value = joint_value
        req.group_state = group_state
        response:MoveTurntableResponse = move_turntable_service(req)
        return response.result
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)
        return False
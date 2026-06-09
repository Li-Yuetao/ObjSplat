#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
os.chdir(PACKAGE_PATH)
from utils import GSO_OBJECTS
import rospy
from objsplat_robot_msgs.srv import ObjectControl, ObjectControlRequest

PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

def main():
    rospy.init_node('batch_object_loader_client')
    print("Waiting for /object_manager service...")
    rospy.wait_for_service('/object_manager')
    print("/object_manager service is available.")
    try:
        object_manager = rospy.ServiceProxy('/object_manager', ObjectControl)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        return
    
    config_file_name = 'simcam_gsmap.json'
    
    # Test1: Varying type of objects
    # capture_nums = np.arange(2, 31, 2, dtype=int) # >= 2
    # random_z = False
    # z_angle = 0.0
    
    # Test2: Varying random_z in one object
    capture_nums = [30]
    random_z = False
    z_angle = 0.0
    with_arm_turntable = False
    
    for capture_num in capture_nums:
        # if capture_num % 5 == 0:
        #     continue
        print(f"Running experiments with capture_num={capture_num}...")
    
        for obj_name, obj_org_name in GSO_OBJECTS.items():
            req = ObjectControlRequest()
            req.model_type = 'GSO'
            req.model_name = obj_org_name
            req.ref_frame = 'world' if not with_arm_turntable else 'turntable_support_link'
            req.pose = "0 0 0.01 0 0 0 1"
            req.random_z = random_z
            req.z_angle = z_angle
            req.action = 'add'

            try:
                resp = object_manager(req)
                if resp.success:
                    print(f"Added model {obj_org_name}: {resp.message}")
                else:
                    print(f"Failed to add model {obj_org_name}: {resp.message}")
            except rospy.ServiceException as e:
                print(f"Service call exception: {e}")
            
            config_file_url = os.path.join(PACKAGE_PATH, 'config', 'datasets', config_file_name)
            os.system(f'roslaunch objsplat sim.launch \
                            mapper:=GSMap \
                            config:={config_file_url} \
                            serial_number:=1 \
                            object_id:={obj_name} \
                            debug:=0 \
                            hide_planner_windows:=0 \
                            hide_mapper_windows:=1 \
                            save_runtime_data:=0 \
                            mode:=AUTO_PLANNING \
                            capture_num:={capture_num} \
                            with_arm_turntable:={int(with_arm_turntable)}')
            
            # delete the object after adding
            req.action = 'delete'
            try:
                resp = object_manager(req)
                if resp.success:
                    print(f"Deleted model {obj_org_name}: {resp.message}")
                else:
                    print(f"Failed to delete model {obj_org_name}: {resp.message}")
            except rospy.ServiceException as e:
                print(f"Service call exception: {e}")
    
    # evaluate the geometry
    print("Evaluating geometry...")
    os.system(f'rosrun objsplat eval_geometry.py \
                    --results_dir {os.path.join(PACKAGE_PATH, "results")} \
                    --iteration -1')
if __name__ == '__main__':
    main()
#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
UTILS_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src', 'mapper', 'gsmap', 'gaussian_surfels', 'utils'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
sys.path.append(UTILS_PATH)
import json
import argparse
from typing import Union

import faulthandler
import torch
import numpy as np
from open3d.visualization import gui
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import rospy
from mapper import MapperType
from utils import PROJECT_NAME, GlobalState
from dataloader.dataloader import get_dataset, RobotScanDataset, SimRobotScanDataset
from visualizer.visualizer import Visualizer

if __name__ == '__main__':
    faulthandler.enable()
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} mapper node.')
    parser.add_argument('--mapper',
                        type=str,
                        choices=list(MapperType.__members__),
                        required=True,
                        help='Specify the mapper type.')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='Input config url (*.json).')
    parser.add_argument('--serial_number',
                        type=str,
                        required=True,
                        help='Specify sensor serial number.')
    parser.add_argument('--object_id',
                        type=str,
                        required=True,
                        help='Specify test object id.')
    parser.add_argument('--gpu_id',
                        type=int,
                        required=True,
                        help='Specify gpu id.')
    parser.add_argument('--mode',
                        type=str,
                        choices=list(GlobalState.__members__)[:-1],
                        required=True,
                        help='Specify the mode to start with.')
    parser.add_argument('--hide_windows',
                        type=int,
                        required=True,
                        help='Disable windows.')
    parser.add_argument('--save_runtime_data',
                        type=int,
                        required=True,
                        help='Save runtime data.')
    parser.add_argument('--debug',
                        type=int,
                        default=0,
                        help='Debug mode, save evaluation results.')
    parser.add_argument('--remark',
                        type=str,
                        default='None',
                        help='remark info.')
    parser.add_argument('--view_strategy',
                        type=str,
                        required=False,
                        default="None",
                        help='View planning strategy, override config file if set.')
    parser.add_argument('--with_arm_turntable',
                        type=int,
                        default=1,
                        help='Whether to use robotic arms and turntables, if not, directly control camera pose.')
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    rospy.init_node(ros_args['__name'], anonymous=True, log_level=rospy.DEBUG if bool(args.debug) else rospy.INFO)
    
    if torch.cuda.is_available():
        device = torch.device('cuda', args.gpu_id)
    else:
        rospy.logwarn('No GPU available.')
        device = torch.device('cpu')
        
    os.chdir(PACKAGE_PATH)
    with open(args.config) as f:
        config = json.load(f)
        if 'sensor' in config:
            config['sensor']['config'] = os.path.abspath(
                os.path.join(os.path.dirname(args.config), os.pardir, os.pardir, config['sensor']['config'], f'{args.serial_number}.yaml'))
        
        if args.view_strategy != "None":
            config['planner']['view_strategy'] = args.view_strategy
        
        if args.with_arm_turntable == 0:
            config['dataset']['with_arm_turntable'] = False
        
        dataset:Union[RobotScanDataset, SimRobotScanDataset] = get_dataset(config, args.object_id, args.remark)
    
    hide_windows = bool(args.hide_windows)
    if not hide_windows:
        app = gui.Application.instance
        app.initialize()
    w = Visualizer(
        MapperType(args.mapper),
        config,
        GlobalState(args.mode),
        1 if hide_windows else app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE)),
        device,
        dataset,
        hide_windows,
        bool(args.save_runtime_data))
    if hide_windows:
        rospy.spin()
    else:
        app.run()
    
    rospy.loginfo(f'{PROJECT_NAME} mapper node finished.')
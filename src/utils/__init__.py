import time
from enum import Enum
from typing import Tuple, Union

import torch
import numpy as np

PROJECT_NAME = 'ObjSplat'

OPENCV_TO_OPENGL = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
)

OPENCV_TO_GAZEBO = np.array(
    [
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]
    ]
)

CURRENT_FRUSTUM = {
    'color': [0.961, 0.475, 0.000], # orange
    'scale': 0.05,
    'material': 'unlit_line_mat',
}
GT_FRUSTUM = {
    'color': [0.475, 0.961, 0.000], # green
    'scale': 0.05,
    'material': 'unlit_line_mat',
}
BEFORE_REFINE_FRUSTUM = {
    'color': [1.0, 0.4, 0.7], # pink
    'scale': 0.05,
    'material': 'unlit_line_mat',
}
CURRENT_HORIZON = {
    'color': [0.0, 1.0, 0.0],
}
CANDIDATE_FRUSTUM = {
    'color': [0.5, 0.5, 0.5],
    'scale': 0.05,
    'material': 'unlit_line_mat',
}

# NOTE: Functions for TIMING
def start_timing(use_cuda:bool=True) -> Tuple[Union[torch.cuda.Event, float], Union[torch.cuda.Event, None]]:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
        end = None
    return start, end

def end_timing(start:Union[torch.cuda.Event, float], end:Union[torch.cuda.Event, None], use_cuda:bool=True) -> float:
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        end = time.perf_counter()
        elapsed_time = end - start
        # Convert to milliseconds to have the same units
        # as torch.cuda.Event.elapsed_time
        elapsed_time = elapsed_time * 1000
    return elapsed_time

class GlobalState(Enum):
    MANUAL_CONTROL = 'MANUAL_CONTROL'
    AUTO_PLANNING = 'AUTO_PLANNING'
    PAUSE = 'PAUSE'
    COLLECT_DATA = 'COLLECT_DATA'
    POST_PROCESSING = 'POST_PROCESSING'
    QUIT = 'QUIT'
    
class ArmGroupState(Enum):
    # reference the ROS Moveit srdf file
    ready = 'ready'

# 16 objects
GSO_OBJECTS = {
    'BUNNY-RACER': 'BUNNY_RACER',
    'CHICKEN-RACER': 'CHICKEN_RACER',
    'Turtle': 'Vtech_Roll_Learn_Turtle',
    'Elephant': 'Sootheze_Cold_Therapy_Elephant',
    'Yellow-toy': 'Ortho_Forward_Facing',
    'Mario': 'Nintendo_Mario_Action_Figure',
    'Toy': 'Nintendo_Yoshi_Action_Figure',
    'Dino': 'Dino_3',
    'Horse': 'Breyer_Horse_Of_The_Year_2015',
    'Eagle': 'Schleich_Bald_Eagle',
    'Dragon': 'Animal_Planet_Foam_2Headed_Dragon',
    'shoe': 'Womens_Bluefish_2Eye_Boat_Shoe_in_Tan',
    'Candy-box': 'Nips_Hard_Candy_Rich_Creamy_Butter_Rum_4_oz_1133_g',
    'Cup': 'ACE_Coffee_Mug_Kristen_16_oz_cup',
    'Backpack': 'Olive_Kids_Trains_Planes_Trucks_Bogo_Backpack',
    'STACKING-BEAR': 'STACKING_BEAR',
}
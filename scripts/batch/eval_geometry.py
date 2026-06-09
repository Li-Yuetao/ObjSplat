#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
os.chdir(PACKAGE_PATH)
import argparse
from tqdm import tqdm

from utils import PROJECT_NAME, GSO_OBJECTS
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} results geometry evaluator.')
    parser.add_argument('--results_dir',
                        type=str,
                        default=os.path.join(PACKAGE_PATH, 'results'),
                        help='Results directory.')
    parser.add_argument('--force',
                        action='store_true',
                        help='Specify whether to force evaluation.')
    parser.add_argument('--img',
                        action='store_true',
                        help='Specify whether to generate images during evaluation.')
    parser.add_argument('--iteration',
                        type=int,
                        default=-1,
                        help='Specify the iteration to evaluate, -1 means the latest iteration.')
    
    args = parser.parse_args()
    
    GENERATE_MESH_SCRIPT_URL = os.path.join(PACKAGE_PATH, 'scripts', 'nodes', 'mesh_generation_node.py')
    EVAL_MESH_SCRIPT_URL = os.path.join(PACKAGE_PATH, 'scripts', 'evaluation', 'eval_mesh.py')
    
    results_dir = args.results_dir
    force = args.force
    img = args.img
    iteration = args.iteration
    for result_dir in tqdm(os.listdir(results_dir)):
        result_dir_url = os.path.join(results_dir, result_dir)
        if not os.path.isdir(result_dir_url):
            continue
        config_url = os.path.join(result_dir_url, 'config.json')
        object_name = result_dir.split('_')[3]
        pointcloud_dir_url = os.path.join(result_dir_url, 'gaussians_data', 'point_cloud')
        if os.path.exists(config_url) and os.path.exists(pointcloud_dir_url):
            # mesh_url = os.path.join(result_dir_url, 'gaussians_data', 'poisson_mesh_9_pruned.ply')
            mesh_url = os.path.join(result_dir_url, 'gaussians_data', 'tsdf_mesh.ply') # use tsdf mesh for evaluation
            dataset_base = os.path.abspath(os.path.join(PACKAGE_PATH, "..", "objsplat_robot", "objsplat_robot_gazebo", "object_models", "GSO"))
            print(f'dataset_base: {dataset_base}')
            if os.path.exists(mesh_url) and not force:
                print(f'{mesh_url} already exists, skip generate mesh.')
            else:
                os.system(f'python {GENERATE_MESH_SCRIPT_URL} --experiment {result_dir_url} --iteration {iteration } {"--img" if img else ""} --poisson_depth 9 {"--force" if force else ""}')
            evaluation_results = os.path.join(result_dir_url, 'gaussians_data', 'evaluation_results')
            if object_name not in GSO_OBJECTS:
                print(f'Object {object_name} not in GSO_OBJECTS, skip.')
                continue
            if os.path.exists(evaluation_results):
                print(f'{evaluation_results} already exists, skip evaluation.')
                continue
            os.system(f'python {EVAL_MESH_SCRIPT_URL} --dataset gso --dataset_base {dataset_base} --object_name {GSO_OBJECTS[object_name]} --mesh_path {mesh_url} {"--force" if force else ""}')
#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import numpy as np
from utils.gui_utils import matrix_to_pose
from typing import List
from geometry_msgs.msg import Pose
import networkx as nx
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_pose(pose1: np.ndarray, pose2: np.ndarray, lam: float) -> np.ndarray:
    """
    Interpolates between two 4x4 poses.
    lam ∈ [0, 1]
    """
    # Linear interpolate translation
    t1, t2 = pose1[:3, 3], pose2[:3, 3]
    t_interp = (1 - lam) * t1 + lam * t2

    # Spherical linear interpolation (SLERP) for rotation using Slerp
    rots = R.from_matrix([pose1[:3, :3], pose2[:3, :3]])
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([lam]).as_matrix()[0]

    # Combine back to 4x4 pose
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp
    pose_interp[:3, 3] = t_interp
    return pose_interp

def generate_candidate_views(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    camera_focal_length: float,
    num_candidate_points: int,
    latitude_lower_deg: float = 10.0,
    latitude_upper_deg: float = 90.0,
    center_bias: np.ndarray = np.array([0, 0, 0]),
    visited_poses: List[Pose] = [],
    min_distance: float = 0.05
):
    if not (0 <= latitude_lower_deg <= 90 and 0 <= latitude_upper_deg <= 90):
        print("Warning: Latitude bounds are typically within [0, 90] degrees.")
    if latitude_lower_deg >= latitude_upper_deg:
        raise ValueError("latitude_lower_deg must be less than latitude_upper_deg.")
    if num_candidate_points <= 0:
        return np.array([]), [], np.zeros(3), 0.0

    if np.all(bbox_max == bbox_min):
        obj_half_diagonal = 0.15
        sphere_radius = 0.15 + camera_focal_length
        center = bbox_min + center_bias
    else:
        obj_half_diagonal = np.linalg.norm(bbox_max - bbox_min) / 2.0
        sphere_radius = obj_half_diagonal + camera_focal_length
        center = (bbox_max + bbox_min) / 2.0 + center_bias

    # Generate points using Vogel spiral distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(num_candidate_points)
    sample_fractions = np.ones(num_candidate_points) * 0.5 if num_candidate_points == 1 else \
                      (indices + 0.5) / num_candidate_points
    sin_lat_min = np.sin(np.deg2rad(latitude_lower_deg))
    sin_lat_max = np.sin(np.deg2rad(latitude_upper_deg))
    sin_lat_values = np.clip(sin_lat_min + (sin_lat_max - sin_lat_min) * sample_fractions, -1.0, 1.0)
    latitudes_rad = np.arcsin(sin_lat_values)
    longitudes_rad = (indices * golden_angle) % (2.0 * np.pi)
    
    cos_latitudes = np.cos(latitudes_rad)
    positions = np.column_stack([
        sphere_radius * cos_latitudes * np.cos(longitudes_rad),
        sphere_radius * cos_latitudes * np.sin(longitudes_rad),
        sphere_radius * sin_lat_values
    ]) + center

    # Remove visited poses
    if visited_poses and len(visited_poses) > 0:
        visited_positions = np.array([get_position_from_pose(p) for p in visited_poses])
        if len(visited_positions) > 0:
            visited_tree = KDTree(visited_positions)
            dists, _ = visited_tree.query(positions, k=1)
            valid_mask = (dists >= min_distance).flatten()
            positions = positions[valid_mask]
            longitudes_rad = longitudes_rad[valid_mask]
            indices = indices[valid_mask]
    
    if len(positions) == 0:
        return np.array([]), [], center, sphere_radius
    
    world_z_axis = np.array([0., 0., 1.])
    candidate_view_poses = []
    
    directions = center - positions
    directions_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    z_axes = directions / directions_norm
    
    # Efficiently compute rotation and transformation matrices
    for i, pos in enumerate(positions):
        z_axis = z_axes[i]
        x_axis = np.cross(z_axis, world_z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-10:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        transform_wc = np.eye(4)
        transform_wc[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        transform_wc[:3, 3] = pos
        
        candidate_view_poses.append(matrix_to_pose(transform_wc))
    
    return positions, candidate_view_poses, center, sphere_radius

def generate_circular_candidate_views(
    center: np.ndarray,
    sphere_radius: float,
    latitude_deg: float,
    num_views: int
):
    if num_views <= 0:
        return np.array([]), [], center, sphere_radius

    # Convert latitude to radians
    latitude_rad = np.deg2rad(latitude_deg)
    
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    positions = np.column_stack([
        sphere_radius * np.cos(angles),  # X
        sphere_radius * np.sin(angles),  # Y
        np.full(num_views, np.sin(latitude_rad) * sphere_radius)  # Z
    ]) + center

    world_z_axis = np.array([0., 0., 1.])
    circular_view_poses = []
    
    directions = center - positions
    directions_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    z_axes = directions / directions_norm
    
    # Efficiently compute rotation and transformation matrices
    for i, pos in enumerate(positions):
        z_axis = z_axes[i]
        x_axis = np.cross(z_axis, world_z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-10:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        transform_wc = np.eye(4)
        transform_wc[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        transform_wc[:3, 3] = pos
        
        circular_view_poses.append(matrix_to_pose(transform_wc))
    
    return circular_view_poses

def get_position_from_pose(pose: np.ndarray) -> np.ndarray:
    """Extracts the 3D position (translation) from a 4x4 pose matrix."""
    # check if pose is Pose or matrix
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return pose[:3, 3]
    elif isinstance(pose, Pose):
        return np.array([pose.position.x, pose.position.y, pose.position.z])

def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.dot(v0, v1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot) * t
    rel_vec = v1 - v0 * dot
    rel_vec = rel_vec / np.linalg.norm(rel_vec)
    return v0 * np.cos(theta) + rel_vec * np.sin(theta)

def select_path_poses_networkx_multiobjective(
    candidate_points_arr: np.ndarray,
    candidate_view_poses: List[Pose],
    candidate_views_uncertainty: np.ndarray,
    current_pose: np.ndarray,
    target_index: int,
    alpha: float = 0.1,
    k_connect: int = 10,
    lambda_weight: float = 0.5,
    max_paths: int = 20,
    min_nodes: int = 4
) -> List[Pose]:
    # precompute arrays
    pts = candidate_points_arr
    unc = candidate_views_uncertainty
    N = len(pts)
    current_pos = current_pose[:3, 3]

    # build k-NN graph and edge lookup
    nbrs = NearestNeighbors(n_neighbors=k_connect+1).fit(pts)
    dists, idxs = nbrs.kneighbors(pts)
    edge_list = []
    dist_map = {}
    for i in range(N):
        for j in range(1, k_connect+1):
            v = idxs[i, j]
            d = dists[i, j]
            w = d / (alpha + 0.5*(unc[i] + unc[v]))
            edge_list.append((i, v, w))
            dist_map[(i, v)] = dist_map[(v, i)] = d

    # connect current pose as node -1
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    d2c = np.linalg.norm(pts - current_pos, axis=1)
    for v in np.argsort(d2c)[:k_connect]:
        d = d2c[v]
        w = d / (alpha + unc[v])
        G.add_edge(-1, v, weight=w)
        dist_map[(-1, v)] = dist_map[(v, -1)] = d

    # search paths
    try:
        paths = nx.shortest_simple_paths(G, source=-1, target=target_index, weight='weight')
        best_score, best_path = float('inf'), None
        for count, path in enumerate(paths):
            if count >= max_paths:
                break
            if len(path) < min_nodes:
                continue
            pl = sum(dist_map[(path[k], path[k+1])] for k in range(len(path)-1))
            tu = sum(unc[i] for i in path if i >= 0)
            score = lambda_weight * pl - (1 - lambda_weight) * tu # lambda_weight -> 1, pl, lambda_weight -> 0, tu
            if score < best_score:
                best_score, best_path = score, path
        if best_path is None:
            print("No qualified path found.")
            return []
        print(f"Selected path with {len(best_path)} points, final cost {best_score:.3f}")
        return [matrix_to_pose(current_pose) if i == -1 else candidate_view_poses[i] for i in best_path]
    except nx.NetworkXNoPath:
        print("No path found.")
        return []
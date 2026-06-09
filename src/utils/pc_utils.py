import numpy as np
import open3d as o3d

def depth_to_pointcloud(rgb_img, depth_img, mask, intrinsic_matrix, depth_scale=1000.0):
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    v, u = np.where(mask)
    z = depth_img[v, u]
    
    valid = (z > 0) & np.isfinite(z)
    if not np.any(valid):
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    u, v, z = u[valid], v[valid], z[valid]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack((x, y, z), axis=1) * depth_scale  # mm
    colors = rgb_img[v, u] / 255.0  # normalize to [0, 1]
    
    return points, colors

def preprocess_point_cloud(pcd, voxel_size, camera_location=None):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))

    if camera_location is not None:
        pcd_down.orient_normals_towards_camera_location(
            camera_location=camera_location)

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, max_retries=3):
    distance_threshold = voxel_size * 1.5
    fitness_threshold = 0.3
    rmse_threshold = voxel_size * 2.0
    
    best_result = None
    best_fitness = 0.0
    
    for attempt in range(max_retries):
        max_iter = 4000000 + attempt * 2000000
        
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 500))
        
        print(f"[Attempt {attempt+1}/{max_retries}] Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.6f}")
        
        if result.fitness > best_fitness:
            best_fitness = result.fitness
            best_result = result
        
        if result.fitness >= fitness_threshold and result.inlier_rmse <= rmse_threshold:
            print(f"[SUCCESS] Registration succeeded")
            return result
        
        distance_threshold *= 1.2 # Relax threshold for next attempt
    
    if best_fitness > 0.1:
        print(f"[WARNING] Using best result (fitness={best_fitness:.4f})")
        return best_result
    
    print("[FAILED] Registration failed, returning identity")
    failed_result = o3d.pipelines.registration.RegistrationResult()
    failed_result.transformation = np.identity(4)
    failed_result.fitness = 0.0
    failed_result.inlier_rmse = float('inf')
    return failed_result

def refine_registration(source, target, voxel_size, init_trans):
    distance_threshold = voxel_size * 0.4
    print(f"[INFO] ICP refinement: voxel_size = {voxel_size}, distance_threshold = {distance_threshold:.4f}")
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
    return result

def save_two_pointclouds_in_one_ply(pred_pcd, gt_pcd, out_file):
    pred_colors = np.tile(np.array([[1, 0, 0]]), (np.asarray(pred_pcd.points).shape[0], 1))
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)
    gt_colors = np.tile(np.array([[0, 0, 1]]), (np.asarray(gt_pcd.points).shape[0], 1))
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
    combined_pcd = pred_pcd + gt_pcd
    o3d.io.write_point_cloud(out_file, combined_pcd)

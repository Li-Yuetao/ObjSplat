import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
os.chdir(PACKAGE_PATH)
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import trimesh
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree as KDTree

from src.utils import GSO_OBJECTS
from src.utils.logging_utils import Log
from src.utils.pc_utils import preprocess_point_cloud, execute_global_registration, refine_registration, save_two_pointclouds_in_one_ply
from scripts.evaluation.eval_fscore import eval_recon

'''
reconstruction evaluation tools
modified from:
- https://github.com/cvg/nice-slam/blob/master/src/tools/eval_recon.py
- https://github.com/jzhangbs/DTUeval-python
'''

def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp

def calc_3d_mesh_metric(mesh_rec, mesh_gt, distance_thresh=0.01):
    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=distance_thresh)
    accuracy_rec *= 1000  # convert to mm
    completion_rec *= 1000  # convert to mm
    completion_ratio_rec *= 100  # convert to %

    return {'acc': accuracy_rec, 'comp': completion_rec, 'comp%': completion_ratio_rec}

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)
    
def icp_with_global_registration(source_pcd, target_pcd, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    result_global = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_icp = refine_registration(source_pcd, target_pcd, voxel_size, result_global.transformation)
    return result_icp
    
def color_mesh_error(mesh, error_per_vertex, max_error, color_low=[1,1,1], color_high=[1,0,0]):
    normalized = np.clip(error_per_vertex / max_error, 0, 1)
    colors = np.outer(1 - normalized, color_low) + np.outer(normalized, color_high)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def eval_simple(mesh_path, gt_mesh_path, scale=1e3, gt_scale_factor=1.0, distance_thresh=0.01, num_sample_points=200000):
    mp.freeze_support()
    vis_out_dir = os.path.dirname(mesh_path)
    tqdm.write(f"[INFO] Evaluation output directory: {vis_out_dir}")

    with tqdm(total=8, desc="Evaluating Mesh", unit="step") as pbar:
        pred_mesh = o3d.io.read_triangle_mesh(mesh_path)
        pred_mesh.remove_unreferenced_vertices()
        pbar.update(1)

        gt_mesh_dir = os.path.dirname(gt_mesh_path)
        gt_mesh_scaled_path = os.path.join(gt_mesh_dir, f'model-scale-{gt_scale_factor:.2f}.ply')
        if os.path.exists(gt_mesh_scaled_path):
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_scaled_path)
        else:
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
            gt_mesh.remove_unreferenced_vertices()
            gt_mesh.scale(gt_scale_factor, center=gt_mesh.get_center())
            o3d.io.write_triangle_mesh(gt_mesh_scaled_path, gt_mesh)
            tqdm.write(f"[INFO] Saved scaled GT mesh to: {gt_mesh_scaled_path}")
        pbar.update(1)

        pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=num_sample_points)
        gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=num_sample_points)
        pbar.update(1)

        result_icp = icp_with_global_registration(pred_pcd, gt_pcd, voxel_size=0.003)
        pred_pcd.transform(result_icp.transformation)
        pred_mesh.transform(result_icp.transformation)
        
        o3d.io.write_triangle_mesh(mesh_path, pred_mesh)
        tqdm.write(f"[INFO] Saved aligned prediction mesh: {mesh_path}")

        save_two_pointclouds_in_one_ply(pred_pcd, gt_pcd, f'{vis_out_dir}/combined_aligned.ply')
        pbar.update(1)

        pred_trimesh = trimesh.Trimesh(vertices=np.asarray(pred_mesh.vertices), faces=np.asarray(pred_mesh.triangles))
        gt_trimesh = trimesh.Trimesh(vertices=np.asarray(gt_mesh.vertices), faces=np.asarray(gt_mesh.triangles))
        eval_result = calc_3d_mesh_metric(pred_trimesh, gt_trimesh, distance_thresh)
        pbar.update(1)

        data_pcd = np.asarray(pred_pcd.points)
        gt_points = np.asarray(gt_pcd.points)
        max_dist = distance_thresh

        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=max_dist, algorithm='kd_tree', n_jobs=-1)

        nn_engine.fit(gt_points)
        dist_d2s, _ = nn_engine.kneighbors(data_pcd, return_distance=True)
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        nn_engine.fit(data_pcd)
        dist_s2d, _ = nn_engine.kneighbors(gt_points, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
        chamfer_distance = (mean_d2s + mean_s2d) / 2 * scale
        pbar.update(1)
        

        def visualize_error(pcd_points, dist, out_path):
            R, G, W = np.array([[1, 0, 0]]), np.array([[0, 1, 0]]), np.array([[1, 1, 1]])
            alpha = dist.clip(max=max_dist) / max_dist
            color = R * alpha + W * (1 - alpha)
            color[dist[:, 0] >= max_dist] = G
            write_vis_pcd(out_path, pcd_points, color)
        visualize_error(data_pcd, dist_d2s, f'{vis_out_dir}/vis_d2s.ply')
        visualize_error(gt_points, dist_s2d, f'{vis_out_dir}/vis_s2d.ply')
        pbar.update(1)
        
        # F-score
        result = eval_recon(mesh_path, gt_mesh_scaled_path, eval_2d=False, eval_3d=True, distance_thresh=distance_thresh)
        # Geometry
        result['chamfer_distance'] = chamfer_distance
        # Completion
        result['accuracy'] = eval_result['acc']
        result['completion'] = eval_result['comp']
        result['completion_ratio'] = eval_result['comp%']
        pbar.update(1)
        return result

def parse_rescale_from_sdf(sdf_path):
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
        for scale_tag in root.iter('scale'):
            values = scale_tag.text.strip().split()
            if len(values) == 3:
                scale_values = list(map(float, values))
                return scale_values[0]
    except Exception as e:
        print(f"Parse SDF failed: {e}")
    return 1.0

def _truncate_from_marker(save_path: str, marker: str = '###Completeness###'):
    """delete previous results after marker in the save_path file"""
    if not os.path.exists(save_path):
        return
    with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    idx = content.find(marker)
    if idx != -1:
        kept = content[:idx].rstrip() + '\n'
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(kept)

def eval_gso(dataset_base, mesh_path, object_name, distance_thresh):
    if object_name == 'None':
        result_name = os.path.basename(os.path.dirname(os.path.dirname(mesh_path)))
        object_name = GSO_OBJECTS[result_name.split('_')[3]]

    gt_mesh_path = os.path.join(dataset_base, object_name, 'meshes', 'model.obj')
    sdf_path = os.path.join(dataset_base, object_name, 'model.sdf')
    rescale_factor = parse_rescale_from_sdf(sdf_path)
    print(f'Evaluating: {mesh_path}')
    result = eval_simple(mesh_path, gt_mesh_path, gt_scale_factor=rescale_factor, distance_thresh=distance_thresh)

    save_path = os.path.join(os.path.dirname(mesh_path), 'results.txt')

    # delete previous results after marker
    _truncate_from_marker(save_path, '###Completeness###')

    Log(f'Dataset: GSO, eval mesh: {mesh_path}')
    Log(f'Accuracy: {result["accuracy"]:.4f} mm, Completion: {result["completion"]:.4f} mm, Completion ratio: {result["completion_ratio"]:.2f} %')
    Log(f'Chamfer Distance: {result["chamfer_distance"]:.4f} mm, '
        f'F-score: {result["f-score"]:.4f}, '
        f'recall: {result["recall"]:.4f}, '
        f'mean precision: {result["mean precision"] * 1000:.4f} mm, '
        f'mean recall: {result["mean recall"] * 1000:.4f} mm')

    log_info = (
        f'\n###Completeness###\n'
        f'Completion ratio: {result["completion_ratio"]} %, '
        f'Completion: {result["completion"]} mm.\n'
        f'\n###Geometry Quality###\n'
        f'Accuracy: {result["accuracy"]} mm, '
        f'Chamfer Distance: {result["chamfer_distance"]} mm, '
        f'F1-score: {result["f-score"]}, '
        f'Prec: {result["precision"]}, '
        f'Recall: {result["recall"]}, '
        f'MeanPrec: {result["mean precision"] * 1000} mm, '
        f'MeanRecall: {result["mean recall"] * 1000} mm, '
        f'd_th: {result["dist_threshold"]} m'
    )
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(log_info + '\n')

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q

def eval(in_file, scene, dataset_dir, S, T):
    data_mesh = o3d.io.read_triangle_mesh(str(in_file))


    data_mesh.remove_unreferenced_vertices()

    mp.freeze_support()

    # default dtu values
    max_dist = 20
    patch = 60
    thresh = 0.2  # downsample density

    pbar = tqdm(total=9)
    pbar.set_description('read data mesh')

    vertices = np.asarray(data_mesh.vertices)

    vertices = vertices / S + T

    triangles = np.asarray(data_mesh.triangles)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description('sample pcd from mesh')
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    obs_mask_file = loadmat(f'{dataset_dir}/ObsMask/ObsMask{scene}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    pbar.update(1)
    pbar.set_description('read STL pcd')
    stl_pcd = o3d.io.read_point_cloud(f'{dataset_dir}/Points/stl/stl{scene:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    ground_plane = loadmat(f'{dataset_dir}/ObsMask/Plane{scene}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]
    # stl_above = stl

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    
    pbar.update(1)
    print(f'max_dist: {max_dist}')
    visualize_threshold = max_dist
    vis_out_dir = f'{os.path.dirname(in_file)}'
    pbar.set_description('visualize error')
    vis_dist = visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{vis_out_dir}/vis_d2s.ply', data_down, data_color)
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    write_vis_pcd(f'{vis_out_dir}/vis_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2

    return over_all

def eval_dtu(source_path, scanId, dtu_gt_path, in_mesh):
    scale_mat = np.load(f'{source_path}/cameras.npz')['scale_mat_0']
    S = np.linalg.inv(scale_mat[:3, :3])[0][0]
    T = scale_mat[:3, 3:].T
    print(f'Evaluating: {in_mesh}')
    cd = eval(in_mesh, scanId, dtu_gt_path, S, T)
    print('CD:', cd)
    return cd

if __name__ == '__main__':
    parser = ArgumentParser(description="eval script parameters")
    parser.add_argument("--dataset", choices=['dtu', 'gso'], type=str, required=True)
    parser.add_argument("--dataset_base", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--object_name", type=str, default='None')
    parser.add_argument("--distance_thresh", 
                        type=float, 
                        default=0.005, 
                        help="Distance threshold")
    args = parser.parse_args(sys.argv[1:])
    if args.dataset == 'dtu':
        eval_dtu(args.source_path, args.dtu_scanId, args.dtu_gt_path, args.mesh_path)
    elif args.dataset == 'gso':
        eval_gso(args.dataset_base, args.mesh_path, args.object_name, args.distance_thresh)
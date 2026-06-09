#!/usr/bin/env python
import os
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SRC_PATH = os.path.abspath(os.path.join(PACKAGE_PATH, 'src'))
import sys
sys.path.append(PACKAGE_PATH)
sys.path.append(SRC_PATH)
import numpy as np
import torch
import pickle
import json
import cv2
from munch import munchify # dict to object
import open3d as o3d
import argparse
from typing import List
from tqdm import tqdm
from torchvision.utils import save_image
import warnings
warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from mapper.gsmap.gaussian_surfels.cameras import Camera
from mapper.gsmap.gaussian_surfels.utils.general_utils import safe_state, poisson_mesh
from mapper.gsmap.gaussian_surfels.gaussian_renderer import render
from mapper.gsmap.gaussian_surfels.utils.image_utils import depth2rgb, normal2rgb, depth2normal, resample_points, grid_prune
from mapper.gsmap.gaussian_surfels.gaussian_model import GaussianModel
from mapper.gsmap.gaussian_surfels.utils.system_utils import searchForMaxIteration

def load_cameras_from_file(model_path):
    camera_file = os.path.join(model_path, "cameras.pkl")
    os.path.exists(camera_file)
    with open(camera_file, "rb") as pickle_file:
        camera_params = pickle.load(pickle_file)
    
    rgb_dir = os.path.join(model_path, "rgb")
    rgb_idxs = os.listdir(rgb_dir)
    rgb_idxs.sort(key=lambda x: int(x.split(".")[0]))
    viewpoints = []
    for i, rgb_idx in enumerate(rgb_idxs):
        
        rgb_path = os.path.join(rgb_dir, rgb_idx)
        depth_path = os.path.join(model_path, "depth", rgb_idx)
        mask_path = os.path.join(model_path, "mask", rgb_idx)
        # read image
        rgb_np = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
        depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # convert to tensor
        rgb = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
        depth = torch.from_numpy(depth_np.astype(np.float32) / 6553.5).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0).to(device) # (1, H, W)
        
        # extrinsic = np.array(camera_params[i][:16]).reshape(4, 4)
        # intrinsic = np.array(camera_params[i][16:]).reshape(3, 3)
        extrinsic = torch.tensor(camera_params[i][0:16], device=device).view(4, 4)
        intrinsic = torch.tensor(camera_params[i][16:], device=device).view(3, 3)
        
        viewpoint = Camera(
            id=int(rgb_idx.split(".")[0]),
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            resolution=(rgb.shape[1], rgb.shape[2]), # H, W
            rgb=rgb,
            depth=depth,
            mask=mask,
            device=device,
        )
        viewpoints.append(viewpoint)
    return viewpoints

@torch.no_grad()
def generate_mesh(config_url, model_path, iteration, write_image:bool, poisson_depth: int, name='train', use_tsdf=True, use_poisson=False):
    
    with open(config_url) as f:
        config = json.load(f)
    gaussian_map_cfg = munchify(config["mapper"]["gs_backend"]["gaussians"])
    
    gaussians = GaussianModel(gaussian_map_cfg, None, device)
    
    if iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    print("Loading trained model at iteration {}".format(loaded_iter))
    gaussians.load_ply(os.path.join(model_path,
                                    "point_cloud",
                                    "iteration_" + str(loaded_iter),
                                    "point_cloud.ply"))
    
    viewpoints:List[Camera] = load_cameras_from_file(model_path)
    
    # TSDF mesh extraction
    if use_tsdf:
        extract_tsdf_mesh(model_path, viewpoints, gaussians)

    if use_poisson:
        if name == 'train':
            bound = None
            occ_grid, grid_shift, grid_scale, grid_dim = gaussians.to_occ_grid(0.0, 512, bound)

        resampled = []
        for idx, view in enumerate(tqdm(viewpoints, desc="Rendering progress")):
            background = torch.zeros((3), dtype=torch.float32, device="cuda")
            render_pkg = render(view, gaussians, background, [float('inf'), float('inf')], front_only=True, device=gaussians.device)

            image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
                render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            mask_gt = view.get_gtMask()
            gt_image = view.get_gtImage(background).cuda()
            mask_vis = (opac.detach() > 1e-5)
            depth_range = [0, 20]
            mask_clip = (depth > depth_range[0]) * (depth < depth_range[1])

            normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
            d2n = depth2normal(depth, mask_vis, view.fovx, view.fovy)

            if name == 'train':
                pts = resample_points(view, depth, normal, image, mask_vis * mask_gt * mask_clip)
                grid_mask = grid_prune(occ_grid, grid_shift, grid_scale, grid_dim, pts[..., :3], thrsh=1)
                clean_mask = grid_mask #* mask_mask
                pts = pts[clean_mask]
                resampled.append(pts.cpu())

            if write_image:
                render_path = os.path.join(model_path, name, "ours_{}".format(loaded_iter), "renders")
                gts_path = os.path.join(model_path, name, "ours_{}".format(loaded_iter), "gt")
                info_path = os.path.join(model_path, name, "ours_{}".format(loaded_iter), "info")

                os.makedirs(render_path, exist_ok=True)
                os.makedirs(gts_path, exist_ok=True)
                os.makedirs(info_path, exist_ok=True)
                
                normal_wrt = normal2rgb(normal, mask_vis)
                depth_wrt = depth2rgb(depth, mask_vis)
                d2n_wrt = normal2rgb(d2n, mask_vis)
                normal_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
                depth_wrt += background [:, None, None]* (~mask_vis).expand_as(image) * mask_gt
                d2n_wrt += background[:, None, None] * (~mask_vis).expand_as(image) * mask_gt
                outofmask = mask_vis * (1 - mask_gt)
                mask_vis_wrt = outofmask * (opac - 1) + mask_vis
                img_wrt = torch.cat([gt_image, image, normal_wrt, d2n_wrt, depth_wrt], 2)
                wrt_mask = torch.cat([mask_gt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt, mask_vis_wrt], 2)
                img_wrt = torch.cat([img_wrt, wrt_mask], 0)
                save_image(img_wrt.cpu(), os.path.join(info_path, '{}'.format(view.id) + f".png"))
                save_image(image.cpu(), os.path.join(render_path, '{}'.format(view.id) + ".png"))
                save_image((torch.cat([gt_image, mask_gt], 0)).cpu(), os.path.join(gts_path, '{}'.format(view.id) + ".png"))

            view.to_cpu()

        if name == 'train':
            resampled = torch.cat(resampled, 0)
            mesh_path = f'{model_path}/poisson_mesh_{poisson_depth}'
            
            poisson_mesh(mesh_path, resampled[:, :3], resampled[:, 3:6], resampled[:, 6:], poisson_depth, 3 * 1e-5)

def extract_tsdf_mesh(model_path, viewpoints:List[Camera], gaussians:GaussianModel, 
                      voxel_length=0.0005, sdf_trunc=0.003, depth_trunc=5.0,
                      filter_cluster_min_tri=100, smooth_iterations=2):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print(f"Integrating {len(viewpoints)} views into TSDF volume...")
    for idx, viewpoint_cam in enumerate(tqdm(viewpoints, desc="TSDF Integration")):
        render_pkg = render(
            viewpoint_cam, gaussians, gaussians.background_color, 
            [float('inf'), float('inf')], front_only=True, device=gaussians.device
        )
        rgb = render_pkg["render"]
        depth = render_pkg["depth"]
        opac = render_pkg["opac"]

        rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
        depth_np = depth.squeeze(0).detach().cpu().numpy()
        opac_np = opac.squeeze(0).detach().cpu().numpy()

        valid_mask = (opac_np > 0.5) & (depth_np > 0.01) & (depth_np < depth_trunc)
        depth_filtered = depth_np.copy()
        depth_filtered[~valid_mask] = 0.0

        depth_filtered = cv2.bilateralFilter(
            depth_filtered.astype(np.float32), d=5, sigmaColor=0.01, sigmaSpace=5
        )
        depth_filtered[~valid_mask] = 0.0

        intrinsic_np = viewpoint_cam.intrinsic.detach().cpu().numpy()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            viewpoint_cam.image_width, 
            viewpoint_cam.image_height,
            float(intrinsic_np[0, 0]),  # fx
            float(intrinsic_np[1, 1]),  # fy
            float(intrinsic_np[0, 2]),  # cx
            float(intrinsic_np[1, 2])   # cy
        )

        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray((rgb_np * 255).astype(np.uint8)))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_filtered.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, 
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        extrinsic = np.linalg.inv(viewpoint_cam.extrinsic.detach().cpu().numpy())
        volume.integrate(rgbd, intrinsic, extrinsic)

    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    mesh = filter_isolated_vertices(mesh, filter_cluster_min_tri)
    print(f"After filtering: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if smooth_iterations > 0:
        print(f"Smoothing mesh with {smooth_iterations} iterations...")
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=smooth_iterations,
            lambda_filter=0.5
        )
    mesh.compute_vertex_normals()
    
    mesh_file = os.path.join(model_path, 'tsdf_mesh.ply')
    o3d.io.write_triangle_mesh(mesh_file, mesh, write_ascii=False, compressed=True)
    print(f"Saved TSDF mesh to {mesh_file}")
    
    return mesh

def filter_isolated_vertices(mesh, filter_cluster_min_tri=50):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    triangles_to_remove = (
        cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
    )
    mesh.remove_triangles_by_mask(triangles_to_remove)
    return mesh


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=f'Mesh Generation')
    parser.add_argument('--experiment',
                        type=str,
                        required=True,
                        help='v.')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--img", 
                        action="store_true",
                        help='Whether to save the image'
                        )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--poisson_depth", default=10, type=int)
    
    args, ros_args = parser.parse_known_args()
    
    ros_args = dict([arg.split(':=') for arg in ros_args])
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    experiment_path = os.path.abspath(args.experiment)
    result_name = args.experiment.split("/")[-1]
    print(f"Generatinig mesh for gaussian map {result_name}")
    model_path = f"{experiment_path}/gaussians_data"
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist")
        exit(1)
    
    config_url = os.path.join(experiment_path, "config.json")
    mesh = generate_mesh(config_url, model_path, iteration=args.iteration, write_image=args.img, poisson_depth=args.poisson_depth)
    

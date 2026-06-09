from queue import Queue
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Union
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision.utils import save_image
from munch import munchify # dict to object
from argparse import Namespace
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    # TENSORBOARD_FOUND = True
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

from utils.logging_utils import Log
from utils.slam_utils import to_se3_vec, get_pointcloud
from utils.gui_utils import GaussianPacket
from mapper.gsmap.gaussian_surfels.cameras import Camera
from mapper.gsmap.gaussian_surfels.gaussian_model import GaussianModel
from mapper.gsmap.gaussian_surfels.gaussian_renderer import render
from mapper.gsmap.gaussian_surfels.utils.image_utils import depth2rgb, normal2rgb, depth2normal, normal2curv
from mapper.gsmap.gaussian_surfels.utils.loss_utils import l1_loss, ssim, cos_loss, bce_loss
from mapper.gsmap.gaussian_surfels.utils.camera_utils import get_center_and_diag

from dataloader import RGBDSensor
from .utils import l1_loss_fc_mask, cal_psnr, cal_ssim, cal_lpips

class GSBackEnd(mp.Process):
    def __init__(
        self,
        config:dict,
        rgbd_sensor:RGBDSensor,
        save_dir:Path,
        q_main2vis:Union[Queue, None], 
        device:torch.device,
        densify_downscale_factor:int=1
    ):
        super().__init__()
        self.config = config
        self.rgbd_sensor = rgbd_sensor
        self.q_main2vis = q_main2vis
        self.device = device
        self.iteration_count = 0
        self.viewpoints:Dict[str, Camera] = {}
        self.test_viewpoints:Dict[str, Camera] = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.gaussian_map_cfg = munchify(config["gaussians"])
        self.window_size = self.gaussian_map_cfg.window_size
        self.densify_downscale_factor = densify_downscale_factor
        
        if self.gaussian_map_cfg.eval_plots:
            self.eval_plots_dir = self.save_dir.joinpath("eval_plots")
            self.eval_plots_dir.mkdir(parents=True, exist_ok=True)

        self.gaussians = GaussianModel(self.gaussian_map_cfg, self.viewpoints, self.save_dir, device)
        self.gaussians.training_setup()
        self.pool = torch.nn.MaxPool2d(9, stride=1, padding=4)

        self.cameras_extent = 6.0
        self.camera_params_list = []
        self.poses_cw = []
        # self.set_hyperparams()
    
    def process_track_data(self, packet):
        for i, idx in enumerate(packet['viz_idx']):
            idx = idx.item()
            # NOTE: choose the window_size high covisible neighbors
            if idx not in self.viewpoints:
                tstamp = packet['tstamp'][i].item()
                viewpoint = Camera.init_from_tracking(idx, packet["rgb"][i], 
                                                    packet["depth"][i], 
                                                    packet["depth_range"],
                                                    packet["mask"][i], 
                                                    packet["extrinsic"][i], 
                                                    packet["intrinsic"][i], 
                                                    tstamp,
                                                    camera_lr=self.gaussian_map_cfg.training_args.camera_lr,
                                                    gt_extrinsic=packet["gt_extrinsic"][i])
                self.viewpoints[idx] = viewpoint
                self.gaussians.add_gaussians(self.viewpoints[idx])
        
        if len(self.viewpoints) > 2:
            top_covisible_ids = self.select_high_covisibility_viewpoints(
                self.viewpoints, top_k=self.window_size - 1
            )
        else:
            top_covisible_ids = []
        
        # NOTE: local window 
        self.current_window = [idx] + [vid for vid in top_covisible_ids if vid != idx]
        # NOTE: global window
        other_ids = [
            vid
            for vid in self.viewpoints.keys()
            if vid not in self.current_window
        ]
        newest_id = max(self.viewpoints.keys())
        global_pool = other_ids
        global_weights = []
        for vid in global_pool:
            cam = self.viewpoints[vid]
            age = newest_id - vid
            weight = np.sqrt(age + 1) / (cam.online_iters + 1)
            global_weights.append(weight)
        global_weights = np.array(global_weights)
        global_weights /= global_weights.sum()

        self.train(self.current_window, global_pool, global_weights)
        
        current_window_dict = {}
        current_window_dict[self.current_window[0]] = self.current_window[1:]
        keyframes = [self.viewpoints[kf_idx].to_minimal_dict() for kf_idx in self.current_window]
        if self.q_main2vis is not None:
            self.q_main2vis.put(
                GaussianPacket(
                    gaussians=self.gaussians,
                    current_frame=self.viewpoints[idx].to_minimal_dict(),
                    keyframes=keyframes,
                    kf_window=current_window_dict,
                )
            )
    @torch.no_grad()
    def process_test_data(self, packet, mode='novel'):
        idx = len(self.test_viewpoints) # Add at the end
        name = f'{mode}_{idx}'
        viewpoint = Camera.init_from_tracking(name, packet["rgb"], 
                                            packet["depth"], 
                                            packet["depth_range"],
                                            packet["mask"], 
                                            packet["extrinsic"], 
                                            packet["intrinsic"], 
                                            packet['tstamp'],
                                            camera_lr=0.0,
                                            gt_extrinsic=packet["gt_extrinsic"])
        self.test_viewpoints[name] = viewpoint
    
    @torch.no_grad()
    def select_high_covisibility_viewpoints(
        self,
        viewpoints: Dict[str, Camera],
        top_k: int = 2,
        co_thres: float = 0.5,
        pixels: int = 1600,
        depth_thres: float = 0.005
    ):
        selected_ids = []
        cur_viewpoint = list(viewpoints.values())[-1]
        cur_viewpoint.to_device(self.device)
        H, W = cur_viewpoint.image_height, cur_viewpoint.image_width

        gt_depth = cur_viewpoint.depth
        intrinsics = cur_viewpoint.intrinsic
        c2w = cur_viewpoint.extrinsic

        valid_idx = torch.where(gt_depth[0] > 0)
        valid_idx = torch.stack(valid_idx, dim=1)
        if valid_idx.shape[0] == 0:
            return []
        if valid_idx.shape[0] > pixels:
            perm = torch.randperm(valid_idx.shape[0], device=valid_idx.device)[:pixels]
            sampled_idx = valid_idx[perm]
        else:
            sampled_idx = valid_idx

        # Back-project sampled pixels into world space
        pts_world = get_pointcloud(gt_depth, intrinsics, c2w, sampled_idx)

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(3, 1, 1)
        total_pixels = H * W
        flat_template = torch.full((total_pixels,), 0.0, device=pts_world.device)

        for vid, viewpoint in viewpoints.items():
            if vid == cur_viewpoint.id:
                continue

            R_w2c = viewpoint.extrinsic[:3,:3].t().to(pts_world.device)
            T_w2c = -R_w2c @ viewpoint.extrinsic[:3,3].to(pts_world.device)
            pts_cam_v = (R_w2c @ pts_world.t() + T_w2c.unsqueeze(1)).t()
            K_v = viewpoint.intrinsic.to(pts_world.device)
            zs = pts_cam_v[:, 2]
            us = (pts_cam_v[:, 0] * K_v[0, 0] / zs + K_v[0, 2]).round().long()
            vs = (pts_cam_v[:, 1] * K_v[1, 1] / zs + K_v[1, 2]).round().long()

            valid = (us >= 0) & (us < W) & (vs >= 0) & (vs < H) & (zs > 0)
            if valid.sum() == 0:
                continue

            us, vs, zs = us[valid], vs[valid], zs[valid]
            flat_idx = vs * W + us

            depth_proj = flat_template.clone()
            depth_proj[flat_idx] = zs
            depth_proj = depth_proj.view(1, H, W)

            # Render depth and normal maps
            render_pkg = render(
                viewpoint, self.gaussians, self.gaussians.background_color,
                [float('inf'), float('inf')], device=self.device
            )
            depth_render = render_pkg["depth"]
            normal_render = render_pkg["normal"]

            depth_ref = depth_render[0, vs, us]
            depth_proj_sampled = depth_proj[0, vs, us]
            invalid_depth = (depth_proj_sampled - depth_ref) > depth_thres

            cos_theta = F.cosine_similarity(normal_render, z_axis, dim=0)
            nz_mask = (normal_render.norm(dim=0) > 1e-6)
            cos_theta = torch.where(nz_mask, cos_theta, torch.zeros_like(cos_theta))
            invalid_normal = cos_theta > 0
            invalid_normal_pts = invalid_normal[vs, us]

            invalid = invalid_depth | invalid_normal_pts

            # NOTE: Compute covisibility rate
            cov_rate = 1.0 - invalid.float().mean()
            viewpoint.covisibility_rate = cov_rate

            if cov_rate > co_thres:
                selected_ids.append(vid)

        best_ids = sorted(
            selected_ids,
            key=lambda vid: viewpoints[vid].covisibility_rate,
            reverse=True
        )[:top_k]

        return best_ids

    
    def train(self, current_window: list = [], global_pool: list = [], global_weights: list = [], steps: int = None):
        if len(current_window) == 0:
            return
        
        torch.cuda.empty_cache()
        iterations = self.gaussians.online_iterations if steps is None else steps
        
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        for i in range(iterations):
            self.gaussians.update_learning_rate(i)
            # Global window
            num_global = min(2, len(global_pool))
            if num_global > 0:
                chosen_ids = np.random.choice(global_pool, size=num_global, replace=False, p=global_weights)
                extra_views = [self.viewpoints[idx] for idx in chosen_ids]
            else:
                extra_views = []
            
            train_views = viewpoint_stack + extra_views
            
            viewpoint_cam:Camera = train_views.pop(random.randint(0, len(train_views) - 1))
            background = torch.rand((3), dtype=torch.float32, device="cuda") if self.gaussians.random_background else self.gaussians.background_color
            patch_size = [float('inf'), float('inf')]
            
            render_pkg = render(viewpoint_cam, self.gaussians, background, patch_size, device=self.device)
            image, normal, depth, opac = render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"]
            
            mask_vis = (opac.detach() > 1e-3)
            mask_depth = (depth > 0.0)
            
            # Compute per-view masked L1 loss for RGB and depth
            gt_image = viewpoint_cam.get_gtImage(background)
            rgb_loss = l1_loss_fc_mask((torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b, gt_image, mask_vis)
            depth_loss = l1_loss_fc_mask(depth, viewpoint_cam.depth, mask_depth)
            # Compute bce loss for predicted opacity vs. GT mask (object vs. background)
            mask_gt = viewpoint_cam.get_gtMask()
            alpha_loss = bce_loss(opac, mask_gt)
            
            # Compute normal loss and curvature loss
            normal = torch.nn.functional.normalize(normal, dim=0) * mask_vis
            render_d2n = depth2normal(depth, mask_vis, viewpoint_cam.fovx, viewpoint_cam.fovy)
            normal_l1_loss = l1_loss_fc_mask(normal, viewpoint_cam.normal, mask_depth)
            surface_loss = normal_l1_loss+ cos_loss(normal, render_d2n)
            
            opac_ = self.gaussians.get_opacity
            opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
            opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
            opac_mask = opac_mask0 * 0.01 + opac_mask1
            loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()
            
            curv_n = normal2curv(normal, mask_vis)
            loss_curv = l1_loss(curv_n * 1, 0) #+ 1 * l1_loss(curv_d2n, 0)
            
            total_loss = (rgb_loss 
                          + 0.8 * depth_loss
                          + 0.1 * surface_loss
                          + 0.005 * loss_curv
                          + 0.1 * alpha_loss
                          + 0.01 * loss_opac)
            
            total_loss.backward()
            viewpoint_cam.online_iters += 1
            
            with torch.no_grad():
                # Backpropagate the loss and update the Gaussian parameters
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                viewpoint_cam.optimizer.step()
                viewpoint_cam.optimizer.zero_grad()
                if self.gaussian_map_cfg.eval_plots and (i == 0 or i == iterations - 1):
                    self.save_eval_plots(image, gt_image, 0, depth, viewpoint_cam.depth, 0, opac, viewpoint_cam.mask, normal, render_d2n, None, current_window[0], i, viewpoint_cam.id)
        
        self.post_processing(current_window)
        self.gaussians.is_init = True
    
    @torch.no_grad()
    def post_processing(self, current_window=[], c0=0.5, tau=0.1):
        # only use the latest view for confidence update
        cur_viewpoint = self.viewpoints[current_window[0]]
        patch_size = [float('inf'), float('inf')]
        render_pkg = render(cur_viewpoint, self.gaussians, self.gaussians.background_color, patch_size, device=self.device)

        update_mask = render_pkg["count"] >= 1.0
        
        gaussian_means = self.gaussians.get_xyz.detach()
        gaussian_normals = self.gaussians.get_normal.detach()
        camera_center = cur_viewpoint.extrinsic[:3, 3]
        view_directions = camera_center - gaussian_means
        view_distances = torch.linalg.norm(view_directions, dim=1)
        view_directions = (view_directions / (view_distances.unsqueeze(-1) + 1e-8))
        cosine_sim = torch.sum( gaussian_normals * view_directions, dim=1)
        valid_angle_mask = cosine_sim > 0

        final_mask = (update_mask & valid_angle_mask)
        self.gaussians.view_supports += (final_mask.float())
        delta = (view_directions[final_mask] - self.gaussians.view_means[final_mask])

        self.gaussians.view_means[final_mask] += (delta / self.gaussians.view_supports[final_mask].unsqueeze(-1))

        cosine_sim = torch.clamp(cosine_sim, min=0.0, max=1.0)
        distance_factor = torch.clamp(view_distances / 1.2, min=0.0, max=1.0) # Assuming 1.2m is the max distance for normalization
        front_weight = torch.sigmoid((cosine_sim - c0) / tau)
        score_increment = ((1.0 - distance_factor) * front_weight * cosine_sim)
        self.gaussians.view_scores[final_mask] += score_increment[final_mask]
    
    def finalize(self):
        self.offline_refinement(iterations=self.gaussian_map_cfg.offline_iterations)
        with torch.no_grad():
            for viewpoint in self.viewpoints.values():
                viewpoint.update()
                extrinsic = viewpoint.extrinsic.cpu().view(-1).numpy().tolist()
                intrinsic = viewpoint.intrinsic.cpu().view(-1).numpy().tolist()
                camera_params = extrinsic + intrinsic
                self.camera_params_list.append(camera_params)
                
                T_w2c = np.eye(4)
                T_w2c[0:3, 0:3] = viewpoint.extrinsic.cpu().numpy()[:3, :3]
                T_w2c[0:3, 3] = viewpoint.extrinsic.cpu().numpy()[:3, 3]
                self.poses_cw.append(np.hstack(([viewpoint.tstamp], to_se3_vec(T_w2c))))
                
            camera_pose_file = os.path.join(self.save_dir, f"cameras.pkl")
            with open(camera_pose_file, "wb") as pickle_file:
                pickle.dump(self.camera_params_list, pickle_file)
            self.poses_cw.sort(key=lambda x: x[0])
        return np.stack(self.poses_cw)
    
    @torch.no_grad()
    def change_resolution(self, scale:int=1):
        # NOTE: Reload viewpoint data to a different resolution
        width_before, height_before = self.viewpoints[0].image_width, self.viewpoints[0].image_height
        H, W = int(self.rgbd_sensor.width / scale), int(self.rgbd_sensor.height / scale)
        Log(f"Reloading viewpoint data from [{width_before}, {height_before}] to [{H}, {W}]... ")
        successful_reloads = 0
        for idx, viewpoint in tqdm(self.viewpoints.items(), desc="Reloading Camera Data"):
            if viewpoint.update_frame_data(model_path=self.save_dir, scale=scale):
                successful_reloads += 1
        Log(f"Successfully reloaded data for {successful_reloads}/{len(self.viewpoints)} cameras.")
    
    def offline_refinement(self, iterations:int, checkpoint_iterations=[]):
        Log("Starting offline refinement")
        
        torch.cuda.empty_cache()
        opt = self.gaussians.training_args
        self.gaussians.training_setup()
        tb_writer = self.prepare_output_and_logger(opt)
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        camera_centers = [cam.camera_center.detach().cpu().view(3, 1).numpy() for cam in self.viewpoints.values()]
        center, diagonal = get_center_and_diag(camera_centers)
        cameras_extent = diagonal * 1.1 # radius
        scale = self.densify_downscale_factor
        
        first_iter = 10
        ema_loss_for_log = 0.0
        train_views = []
        
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device=self.device)
        
        # --- Setup for higher SH degree ---
        target_sh_degree = self.gaussian_map_cfg.sh_degree # Or a specific degree for offline, e.g., 3
        current_num_gaussians = self.gaussians.get_xyz.shape[0]
        # Calculate the number of AC coefficients for the target SH degree
        num_ac_coeffs_target = (target_sh_degree + 1)**2 - 1

        if num_ac_coeffs_target > 0:
            current_features_rest = self.gaussians._features_rest.detach()
            num_channels = current_features_rest.shape[2] if current_features_rest.nelement() > 0 else 3 # Assuming 3 channels (RGB)

            # Create new _features_rest tensor, initialized to zeros
            new_features_rest_data = torch.zeros(
                (current_num_gaussians, num_ac_coeffs_target, num_channels), # Shape (N, num_coeffs, C)
                device=self.device,
                dtype=torch.float32
            )

            # If there were existing AC coefficients (e.g., from a lower SH degree), copy them over
            if current_features_rest.nelement() > 0:
                num_ac_coeffs_current = current_features_rest.shape[1]
                coeffs_to_copy = min(num_ac_coeffs_current, num_ac_coeffs_target)
                new_features_rest_data[:, :coeffs_to_copy, :] = current_features_rest[:, :coeffs_to_copy, :]
            
            self.gaussians._features_rest = torch.nn.Parameter(new_features_rest_data.requires_grad_(True))

            updated_params_dict = self.gaussians.replace_tensor_to_optimizer(self.gaussians._features_rest, "f_rest")
            if updated_params_dict and "f_rest" in updated_params_dict:
                self.gaussians._features_rest = updated_params_dict["f_rest"]
            
        progress_bar = tqdm(range(first_iter, iterations), desc="Offline Refinement")
        for iteration in range(first_iter, iterations + 2):
            iter_start.record()
            self.gaussians.update_learning_rate(iteration)
            
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()
            
            elif iteration - 1 == 500 + 1:
                scale = 2
                self.change_resolution(scale)
            elif iteration - 1 == 1000 + 1:
                scale = 1
                self.change_resolution(scale)
            
            if not train_views:
                train_views = list(self.viewpoints.values())
            viewpoint_cam:Camera = train_views.pop(random.randint(0, len(train_views) - 1))
            
            background = torch.rand((3), dtype=torch.float32, device="cuda") if self.gaussians.random_background else self.gaussians.background_color
            patch_size = [float('inf'), float('inf')]
            
            render_pkg = render(viewpoint_cam, self.gaussians, background, patch_size, front_only=True, device=self.device)
            image, normal, depth, opac, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
                render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            mask_gt = viewpoint_cam.get_gtMask()
            gt_image = viewpoint_cam.get_gtImage(background)
            mask_vis = (opac.detach() > 1e-5)
            mask_depth = (depth > 0.0)
            
            # Loss
            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b
            Ll1 = l1_loss(image, gt_image)
            loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            loss_mask = (opac * (1 - self.pool(mask_gt))).mean()
            
            normal_l1_loss = l1_loss_fc_mask(normal, viewpoint_cam.normal, mask_depth)
            render_d2n = depth2normal(depth, mask_vis, viewpoint_cam.fovx, viewpoint_cam.fovy)
            loss_surface = normal_l1_loss+ cos_loss(normal, render_d2n)

            opac_ = self.gaussians.get_opacity
            opac_mask0 = torch.gt(opac_, 0.01) * torch.le(opac_, 0.5)
            opac_mask1 = torch.gt(opac_, 0.5) * torch.le(opac_, 0.99)
            opac_mask = opac_mask0 * 0.01 + opac_mask1
            loss_opac = (torch.exp(-(opac_ - 0.5)**2 * 20) * opac_mask).mean()
            
            curv_n = normal2curv(normal, mask_vis)
            loss_curv = l1_loss(curv_n * 1, 0) #+ 1 * l1_loss(curv_d2n, 0)
            
            # gaussian anisotropy loss
            # loss_gaussian = gaussian_loss_fc(self.gaussians.get_scaling)
            
            loss = 1 * loss_rgb
            loss += 0.1 * loss_mask
            loss += (0.01 + 0.1 * min(2 * iteration / iterations, 1)) * loss_surface
            loss += 0.005 * loss_curv
            loss += 0.01 * loss_opac
            # loss += 1.0 * loss_gaussian

            loss.backward()
            iter_end.record()
            
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * Ll1.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"RGBL1": f"{ema_loss_for_log:.{7}f}, Pts={len(self.gaussians._xyz)}"})
                    progress_bar.update(10)
                if iteration == iterations:
                    progress_bar.close()

                # Log and save
                test_background = self.gaussians.background_color
                self.training_report(scale, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), opt.test_iterations, test_background)
                if (iteration in opt.save_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
                    self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

                # Densification
                if iteration > opt.densify_from_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    min_opac = 0.1
                    if iteration % opt.densification_interval == 0:
                        self.gaussians.adaptive_prune(min_opac, cameras_extent)
                        self.gaussians.adaptive_densify(opt.densify_grad_threshold, cameras_extent)
                    
                    if (iteration - 1) % opt.opacity_reset_interval == 0 and opt.opacity_lr > 0:
                        self.gaussians.reset_opacity(0.12, iteration)

                if (iteration - 1) % 1000 == 0:
                    normal_wrt = normal2rgb(normal, mask_vis)
                    d2n_wrt = normal2rgb(render_d2n, mask_vis)
                    depth_wrt = depth2rgb(depth, mask_vis)
                    img_wrt = torch.cat([gt_image, image, normal_wrt * opac, d2n_wrt * opac, depth_wrt * opac], 2)
                    save_image(img_wrt.cpu(), self.save_dir.joinpath('test.png'))
                    save_image(img_wrt.cpu(), self.save_dir.parents[1] / 'test.png')
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True) # set_to_none=True
                viewpoint_cam.optimizer.step()
                viewpoint_cam.optimizer.zero_grad(set_to_none=True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.gaussians.capture(), iteration), self.save_dir.joinpath("chkpnt", str(iteration), ".pth"))
                
            if self.q_main2vis is not None and iteration % 50 == 0:
                self.q_main2vis.put(
                    GaussianPacket(
                        gaussians=self.gaussians,
                        current_frame=None,
                        iteration=iteration,
                    )
                )

        Log("Map refinement done")
        
    def prepare_output_and_logger(self, args):    
        with open(self.save_dir.joinpath("cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.save_dir)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer

    @torch.no_grad()
    def training_report(self, scale, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, bg, test_interval=0):
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            tb_writer.add_scalar('iter_time', elapsed, iteration)

        if test_interval > 0:
            if iteration % test_interval != 0:
                return
        else:
            if iteration not in testing_iterations:
                return
        if scale != 1:
            self.change_resolution(1) # recover full resolution for evaluation

        torch.cuda.empty_cache()
        train_count = 0
        test_novel_count = 0
        test_traj_count = 0
        # training viewpoints
        train_metrics, train_count = self._evaluate_viewpoints(
            viewpoints=self.viewpoints,
            label='train',
            iteration=iteration,
            save_plot=self.gaussian_map_cfg.eval_plots
        )

        # test viewpoints
        test_intraj_metrics, test_novel_metrics = None, None
        if hasattr(self, 'test_viewpoints') and len(self.test_viewpoints) > 0:
            # ---- Split test viewpoints ----
            test_in_traj = {k: v for k, v in self.test_viewpoints.items() if 'traj' in k.lower() or 'intra' in k.lower()}
            test_novel = {k: v for k, v in self.test_viewpoints.items() if 'novel' in k.lower() or 'new' in k.lower()}

            if len(test_in_traj) > 0:
                test_intraj_metrics, test_traj_count = self._evaluate_viewpoints(
                    viewpoints=test_in_traj,
                    label='test-traj',
                    iteration=iteration,
                    save_plot=self.gaussian_map_cfg.eval_plots
                )

            if len(test_novel) > 0:
                test_novel_metrics, test_novel_count = self._evaluate_viewpoints(
                    viewpoints=test_novel,
                    label='test-novel',
                    iteration=iteration,
                    save_plot=self.gaussian_map_cfg.eval_plots,
                    save_images=False
                )

        # Logging & TensorBoard
        results_file = f'{self.save_dir}/results.txt'
        with open(results_file, 'a') as f:
            f.write(f'\n[ITER {iteration}] Training (N={train_count}): '
                    f'RGBL1={train_metrics["rgb_l1"]} DepthL1={train_metrics["depth_l1"]} '
                    f'PSNR={train_metrics["psnr"]} SSIM={train_metrics["ssim"]} LPIPS={train_metrics["lpips"]}\n')

            if test_intraj_metrics is not None:
                f.write(f'[ITER {iteration}] Test-InTrajectory (N={test_traj_count}): '
                        f'RGBL1={test_intraj_metrics["rgb_l1"]} DepthL1={test_intraj_metrics["depth_l1"]} '
                        f'PSNR={test_intraj_metrics["psnr"]} SSIM={test_intraj_metrics["ssim"]} LPIPS={test_intraj_metrics["lpips"]}\n')

            if test_novel_metrics is not None:
                f.write(f'[ITER {iteration}] Test-NovelView (N={test_novel_count}): '
                        f'RGBL1={test_novel_metrics["rgb_l1"]} DepthL1={test_novel_metrics["depth_l1"]} '
                        f'PSNR={test_novel_metrics["psnr"]} SSIM={test_novel_metrics["ssim"]} LPIPS={test_novel_metrics["lpips"]}\n')

        if tb_writer:
            for prefix, metrics in [
                ('train', train_metrics),
                ('test_in_traj', test_intraj_metrics or {}),
                ('test_novel', test_novel_metrics or {})
            ]:
                if metrics:
                    tb_writer.add_scalar(f'{prefix}/RGB_L1', metrics['rgb_l1'], iteration)
                    tb_writer.add_scalar(f'{prefix}/Depth_L1', metrics['depth_l1'], iteration)
                    tb_writer.add_scalar(f'{prefix}/PSNR', metrics['psnr'], iteration)
                    tb_writer.add_scalar(f'{prefix}/SSIM', metrics['ssim'], iteration)
                    tb_writer.add_scalar(f'{prefix}/LPIPS', metrics['lpips'], iteration)
            tb_writer.add_histogram("opacity_histogram", self.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', self.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()
        
        if scale != 1:
            self.change_resolution(scale)  # revert to previous training resolution
    
    @torch.no_grad()
    def _evaluate_viewpoints(self, viewpoints: Dict[str, Camera], label: str, iteration: int, save_plot: bool = False, save_images: bool = False) -> Dict[str, float]:
        rgb_l1_total = 0.0
        depth_l1_total = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        count = 0

        for key, viewpoint in viewpoints.items():
            count += 1
            viewpoint.to_device(self.device)

            patch_size = [float('inf'), float('inf')]
            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.gaussians.background_color,
                patch_size,
                device=self.device,
                front_only=True
            )

            rgb_render = torch.clamp(render_pkg["render"], 0.0, 1.0)
            depth_render = render_pkg["depth"]
            opac = render_pkg["opac"]
            normal = render_pkg["normal"]

            mask_vis = (opac.detach() > 1e-3)
            mask_depth = (viewpoint.depth > 0.0)
            mask_gt = viewpoint.get_gtMask()

            gt_image = viewpoint.get_gtImage(self.gaussians.background_color)
            image = (torch.exp(viewpoint.exposure_a)) * rgb_render + viewpoint.exposure_b

            # RGB Loss
            rgb_l1 = l1_loss_fc_mask(image, gt_image, mask_gt.bool())

            # Depth Loss
            depth_l1 = l1_loss_fc_mask(depth_render, viewpoint.depth, mask_depth)
            
            # Quality metrics
            psnr = cal_psnr(rgb_render, gt_image, mask_gt)
            ssim_val = cal_ssim(rgb_render.unsqueeze(0), gt_image.unsqueeze(0), mask_gt.unsqueeze(0))
            lpips_val = cal_lpips(rgb_render.unsqueeze(0), gt_image.unsqueeze(0), mask_gt.unsqueeze(0))

            rgb_l1_total += rgb_l1
            depth_l1_total += depth_l1
            psnr_total += psnr
            ssim_total += ssim_val
            lpips_total += lpips_val

            render_d2n = depth2normal(depth_render, mask_vis, viewpoint.fovx, viewpoint.fovy)
            if save_plot:
                self.save_eval_plots(
                    rgb_render, gt_image, psnr, depth_render,
                    viewpoint.depth, depth_l1, opac, viewpoint.mask,
                    normal, render_d2n, viewpoint.normal,
                    None, iteration, viewpoint.id
                )
            
            # save images
            if save_images:
                Log(f'RGBL1: {rgb_l1.item():.5f}, DepthL1: {depth_l1.item():.5f}, PSNR: {psnr:.2f}, SSIM: {ssim_val:.3f}, LPIPS: {lpips_val:.3f} for {key}')
                eval_dir = self.save_dir.joinpath('eval', key)
                eval_dir.mkdir(parents=True, exist_ok=True)
                
                mask_gt_depth = (viewpoint.depth > 0.0)
                
                normal_vis = normal2rgb(normal, mask_vis)
                d2n_vis = normal2rgb(render_d2n, mask_vis)
                depth_vis = depth2rgb(depth_render, mask_vis)
                gt_depth_vis = depth2rgb(viewpoint.depth, mask_gt_depth)
                
                # RGB render with alpha
                rgba_render = torch.cat([image, opac], dim=0)  # (4, H, W)
                save_image(rgba_render.cpu(), eval_dir / f'{key}_rgb_rgba.png')
                
                # GT with mask as alpha
                rgba_gt = torch.cat([gt_image, mask_gt.float()], dim=0)
                save_image(rgba_gt.cpu(), eval_dir / f'{key}_rgb_gt_rgba.png')
                
                # Normal with alpha
                rgba_normal = torch.cat([normal_vis, opac], dim=0)
                save_image(rgba_normal.cpu(), eval_dir / f'{key}_normal_rgba.png')
                
                # D2N with alpha
                rgba_d2n = torch.cat([d2n_vis, opac], dim=0)
                save_image(rgba_d2n.cpu(), eval_dir / f'{key}_d2n_rgba.png')
                
                # Depth with alpha
                rgba_depth = torch.cat([depth_vis, opac], dim=0)
                save_image(rgba_depth.cpu(), eval_dir / f'{key}_depth_rgba.png')
                
                # GT Depth with mask as alpha
                rgba_gt_depth = torch.cat([gt_depth_vis, mask_gt_depth.float()], dim=0)
                save_image(rgba_gt_depth.cpu(), eval_dir / f'{key}_depth_gt_rgba.png')
                
                # Alpha channel
                save_image(opac.cpu(), eval_dir / f'{key}_alpha.png')
                
                normal_black = normal_vis * opac
                d2n_black = d2n_vis * opac
                depth_black = depth_vis * opac
                gt_depth_black = gt_depth_vis * mask_gt_depth.float()
                
                preview = torch.cat([
                    gt_image, image, 
                    normal_black, d2n_black, 
                    gt_depth_black, depth_black
                ], dim=2)
                save_image(preview.cpu(), eval_dir / f'{key}_preview.png')
                
                # save eval results to txt
                with open(eval_dir.joinpath(f'eval_{key}_results.txt'), 'w') as f:
                    f.write(f'RGBL1: {rgb_l1.item():.5f} \n')
                    f.write(f'DepthL1: {depth_l1.item():.5f}\n')
                    f.write(f'PSNR: {psnr:.2f}\n')
                    f.write(f'SSIM: {ssim_val:.3f}\n')
                    f.write(f'LPIPS: {lpips_val:.3f}\n')

        # ---- Average metrics ----
        if count == 0:
            print(f'[WARN] No viewpoints found for {label}')
            return {"rgb_l1": 0.0, "depth_l1": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}

        metrics = {
            "rgb_l1": rgb_l1_total / count,
            "depth_l1": depth_l1_total / count,
            "psnr": psnr_total / count,
            "ssim": ssim_total / count,
            "lpips": lpips_total / count,
        }

        gaussians_count = self.gaussians._xyz.shape[0]
        print(f'\n[ITER {iteration}] Evaluating {label}: '
            f'RGBL1={metrics["rgb_l1"]:.5f} DepthL1={metrics["depth_l1"]:.5f} '
            # f'RGBL1={metrics["rgb_l1"]:.5f} DepthL1={metrics["depth_l1"] * 100:.3f} ' # cm
            f'PSNR={metrics["psnr"]:.2f} SSIM={metrics["ssim"]:.3f} LPIPS={metrics["lpips"]:.3f} '
            f'Gaussians={gaussians_count} (N={count})')
        return metrics, count
    
    @torch.no_grad()
    def save_eval_plots(self, rgb_preds, rgb_gt, psnr, depth_preds, depth_gt, depth_l1_loss, mask_preds, mask_gt, normal_preds, render_d2n, gt_d2n, cur_frame_id, iteration, id, depth_max=1.5):
        # RGB (convert to numpy float [0, 1] for imshow)
        image_np = rgb_preds.squeeze().permute(1,2,0).detach().cpu().numpy() # (H, W, 3)
        gt_image_np = rgb_gt.squeeze().permute(1,2,0).detach().cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        gt_image_np = np.clip(gt_image_np, 0, 1)
        # Depth (raw numpy float)
        depth_np = depth_preds.squeeze().detach().cpu().numpy()
        gt_depth_np = depth_gt.squeeze().detach().cpu().numpy()
        all_depths = np.concatenate([depth_np.flatten(), gt_depth_np.flatten()])
        all_depths = all_depths[all_depths > 0] # Exclude zero depth

        # Mask/Alpha (raw numpy float [0, 1])
        mask_np = mask_preds.squeeze().detach().cpu().numpy()
        gt_mask_np = mask_gt.squeeze().detach().cpu().numpy()
        # Normal (raw numpy float [-1, 1])
        rend_normal_np = normal_preds.squeeze().permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
        if render_d2n is not None:
            normal_gt = render_d2n
        elif gt_d2n is not None:
            normal_gt = gt_d2n
        if normal_gt is not None:
            # TODO if normal_gt is not None, use it
            surf_normal_np = normal_gt.squeeze().permute(1, 2, 0).detach().cpu().numpy() # (H, W, 3)
        else:
            surf_normal_np = np.zeros_like(rend_normal_np) # Placeholder if no normal supervision

        # --- Calculate Differences (for visualization) ---
        # Calculate differences only if GT is available
        rgb_diff = np.abs(image_np - gt_image_np) if rgb_gt is not None else np.zeros_like(image_np)
        depth_diff = np.abs(depth_np - gt_depth_np) if depth_gt is not None else np.zeros_like(depth_np)
        mask_diff = np.abs(mask_np - gt_mask_np) if mask_gt is not None else np.zeros_like(mask_np)
        # Normal diff: Euclidean distance [0, 2]
        normal_diff = np.linalg.norm(rend_normal_np - surf_normal_np, axis=-1) if normal_gt is not None else np.zeros_like(rend_normal_np[..., 0])

        # --- Plotting ---
        aspect_ratio = gt_image_np.shape[1] / gt_image_np.shape[0]
        fig_height = 12
        fig_width = fig_height # or fig_height * aspect_ratio
        fig, axs = plt.subplots(4, 3, figsize=(fig_width, fig_height))

        # Row 0: RGB
        axs[0, 0].imshow(image_np)
        axs[0, 0].set_title(f"Rendered RGB, PSNR: {psnr:.2f}")
        axs[0, 1].imshow(gt_image_np)
        axs[0, 1].set_title("Ground Truth RGB")
        axs[0, 2].imshow(rgb_diff, cmap='hot', vmin=0, vmax=rgb_diff.max())
        axs[0, 2].set_title("RGB Abs Diff")

        # Row 1: Depth
        axs[1, 0].imshow(depth_np, cmap='viridis', vmin=0, vmax=depth_max)
        axs[1, 0].set_title(f"Rendered Depth, L1 Loss: {depth_l1_loss * 100:.2f} cm")
        axs[1, 1].imshow(gt_depth_np, cmap='viridis', vmin=0, vmax=depth_max)
        axs[1, 1].set_title("Ground Truth Depth")
        axs[1, 2].imshow(depth_diff, cmap='hot', vmin=0, vmax=depth_max/5.0)
        axs[1, 2].set_title("Depth Abs Diff")

        # Row 2: Mask/Alpha
        axs[2, 0].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axs[2, 0].set_title("Rendered Alpha")
        axs[2, 1].imshow(gt_mask_np, cmap='gray', vmin=0, vmax=1)
        axs[2, 1].set_title("Ground Truth Alpha")
        axs[2, 2].imshow(mask_diff, cmap='hot', vmin=0, vmax=1)
        axs[2, 2].set_title("Alpha Abs Diff")

        # Row 3: Normal
        # Scale [-1, 1] normals to [0, 1] for imshow visualization
        axs[3, 0].imshow((rend_normal_np + 1) / 2.0)
        axs[3, 0].set_title("Rendered Normal")
        axs[3, 1].imshow((surf_normal_np + 1) / 2.0)
        if render_d2n is not None:
            axs[3, 1].set_title("Render d2n")
        else:
            axs[3, 1].set_title("Ground Truth d2n")
        axs[3, 2].imshow(normal_diff, cmap='hot', vmin=0, vmax=2)
        axs[3, 2].set_title("Normal Abs Diff (Euclidean)")

        # Turn off axes for all subplots
        for row in axs:
            for ax in row:
                ax.axis('off')
                ax.grid(False)

        # Save the figure
        if cur_frame_id is not None:
            fig.suptitle(f"Online idx {cur_frame_id} Iteration {iteration} Key Frame idx {id}", fontsize=20)
            save_path = os.path.join(self.eval_plots_dir, f'online_idx_{cur_frame_id}_kf_idx_{id}_iters_{iteration}.png')
        else:
            fig.suptitle(f"Offline Iteration {iteration} Key Frame idx {id}", fontsize=20)
            save_path = os.path.join(self.eval_plots_dir, f'offline_kf_idx_{id}_iters_{iteration}.png')

        # Adjust layout to prevent titles/subplots overlapping
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle

        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    
    def save_gaussian_data(self, frame_id:int):
        print("\n[SCAN {}] Saving Gaussians".format(frame_id))
        point_cloud_path = os.path.join(self.save_dir, "scans/scan_{}".format(frame_id))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
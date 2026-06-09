#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from imgviz import depth2rgb
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from torch.utils.cpp_extension import load
from torchvision.utils import save_image

from mapper.gsmap.gaussian_surfels.cameras import Camera
from mapper.gsmap.gaussian_surfels.utils.general_utils import build_rotation, build_scaling_rotation, get_expon_lr_func, inverse_sigmoid, normal2rotation, quaternion2rotmat, strip_symmetric
from mapper.gsmap.gaussian_surfels.utils.graphics_utils import BasicPointCloud
from mapper.gsmap.gaussian_surfels.utils.image_utils import world2scrn
from mapper.gsmap.gaussian_surfels.utils.system_utils import mkdir_p
from mapper.gsmap.utils import depth2normal, get_smooth_depth, get_world_rays, sample_image_grid

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, args, viewpoints:Dict[int, Camera], save_dir:Path, device="cuda:0"):
        self.device = device
        self.active_sh_degree = 0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.scale_gradient_accum = torch.empty(0)
        self.rot_gradient_accum = torch.empty(0)
        self.opac_gradient_accum = torch.empty(0)
        
        # non-trainable gaussian parameters for confidence
        self.view_scores = torch.empty(0)
        self.view_supports = torch.empty(0)
        self.view_means = torch.empty((0, 3))
        
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.viewpoints = viewpoints
        self.is_init = False
        self.save_dir = save_dir
        
        self.max_sh_degree = args.sh_degree  
        self.online_iterations = args.online_iterations
        self.training_args = args.training_args
        self.random_background = args.random_background
        self.background_color = torch.tensor(
            args.background, dtype=torch.float32
        ).to(self.device)
        self.rgb_error_thres = args.rgb_error_thres
        self.use_view_distribution = args.use_view_distribution                           
        try:
            self.config = [args.surface, args.normalize_depth, args.perpix_depth]
        except AttributeError:
            self.config = [True, True, True]
        self.config.append(self.training_args.camera_lr > 0)
        self.config = torch.tensor(self.config, dtype=torch.float32, device="cuda")
        self.setup_functions()
        self.utils_mod = load(name="cuda_utils", sources=["src/mapper/gsmap/gaussian_surfels/utils/ext.cpp", "src/mapper/gsmap/gaussian_surfels/utils/cuda_utils.cu"])
        self.opac_reset_record = [0, 0]
        

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.scale_gradient_accum,
            self.rot_gradient_accum,
            self.opac_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.config
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        scale_gradient_accum,
        rot_gradient_accum,
        opac_gradient_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.config) = model_args
        self.training_setup()
        self.xyz_gradient_accum = xyz_gradient_accum
        self.scale_gradient_accum = scale_gradient_accum
        self.rot_gradient_accum = rot_gradient_accum
        self.opac_gradient_accum = opac_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        # print(self._scaling)
        return self.scaling_activation(self._scaling)
        # scaling_2d = torch.cat([self._scaling[..., :2], torch.full_like(self._scaling[..., 2:], -1e10)], -1)
        # return self.scaling_activation(scaling_2d)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        # fix: shape is not consistent
        min_size = min(self._features_dc.shape[0], self._features_rest.shape[0])
        features_dc = self._features_dc[:min_size]
        features_rest = self._features_rest[:min_size]
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    @property
    def get_confidences(self):
        if self.use_view_distribution:
            view_var = self.view_means.norm(dim=-1)
            view_var[torch.isnan(view_var)] = 1.0
            view_variance_factor = torch.exp(1 - view_var)
            if view_variance_factor.shape[0] != self.view_scores.shape[0]:
                # TODO: UI and offline optimization may cause variable re-access issues.
                view_variance_factor = torch.ones_like(self.view_scores)
            confidences = torch.clamp(
                view_variance_factor * self.view_scores, min=0, max=1
            )
        else:
            confidences = torch.clamp(
                1 - 1 / torch.exp(self.view_supports), min=0, max=1
            )
        if confidences.numel() == 0:
            confidences = torch.ones(self._xyz.shape[0], device=self.device, dtype=self._xyz.dtype)
        return confidences
    
    @property
    def get_normal(self):
        return quaternion2rotmat(self.get_rotation)[..., 2]


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self):
        self.percent_dense = self.training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.rot_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.opac_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        self.config[3] = self.training_args.camera_lr > 0
        # self.optimizer = torch.optim.SGD(l)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.training_args.position_lr_delay_mult,
                                                    max_steps=self.training_args.position_lr_max_steps)
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def add_gaussians(self, viewpoint:Camera):
        rgb_gt = viewpoint.rgb.to(self.device)             # shape (3, H, W)
        depth_gt = viewpoint.depth.to(self.device)         # shape (1, H, W)
        # Smooth the depth map for more stable normal estimation
        depth_smooth = get_smooth_depth(depth_gt.squeeze(0).cpu().numpy())
        depth_smooth = torch.tensor(depth_smooth, device=self.device).unsqueeze(0)  # (1, H, W)
        normalized_intrinsic = viewpoint.normalized_intrinsic.to(self.device)  # (3, 3)
        extrinsic = viewpoint.extrinsic.to(self.device)  # (4, 4)
        valid_mask = (depth_gt > 0.0).view(-1)  # Flattened mask for all pixels
        
        _, H, W = rgb_gt.shape
        point_num = H * W
        # Generate ray directions for each pixel in the camera (image grid)
        xy_ray, _ = sample_image_grid((H, W), device=self.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")                  # reshape to (H*W, 1, 2)
        origins, directions = get_world_rays(xy_ray, extrinsic, normalized_intrinsic)
        # Compute 3D point cloud: origin + depth * direction for each pixel
        pcd = (origins + directions * depth_gt.view(-1, 1, 1)).squeeze(1)       # shape (H*W, 3)

        # Initialize normal vectors (world and camera) for each point
        pcd_normals     = torch.zeros(point_num, 3, device=self.device)
        pcd_normals_cam = torch.zeros(point_num, 3, device=self.device)
        pcd_normals[:, 2]     = 1.0  # default normal pointing along +Z (world)
        pcd_normals_cam[:, 2] = 1.0  # default normal pointing along camera +Z

        # Estimate surface normals from the depth map (in camera space)
        normals_cam = depth2normal(
            depth_smooth, valid_mask.view(1, H, W), fov=(np.pi / 3, np.pi / 3)
        )
        normals_cam = normals_cam.permute(1, 2, 0).view(-1, 3)
        valid_normal_mask = torch.sum(normals_cam**2, dim=-1) > 0.0
        valid_mask = valid_mask * valid_normal_mask  # only keep points with valid depth and normal

        # Transform camera-space normals to world-space normals
        normals_world = torch.matmul(extrinsic[:3, :3], normals_cam.T).T
        pcd_normals_cam[valid_mask] = normals_cam[valid_mask]
        pcd_normals[valid_mask]     = normals_world[valid_mask]
        directions_norm = torch.nn.functional.normalize(directions.squeeze(1), dim=1)
        cos_sim = torch.sum(directions_norm * pcd_normals, dim=-1)
        valid_normal_mask = cos_sim < -0.01
        valid_mask = valid_mask * valid_normal_mask

        if self.is_init:
            from mapper.gsmap.gaussian_surfels.gaussian_renderer import render
            patch_size = [float('inf'), float('inf')]
            render_pkg = render(viewpoint, self, self.background_color, patch_size, device=self.device)
            rgb_render, normal, depth_render, opac, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["normal"], render_pkg["depth"], render_pkg["opac"], \
                render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            global_render_results = {
                "rgb": rgb_render,
                "depth": depth_render.squeeze(1),
                "opacity": opac.squeeze(1),
                "normal": normal,
            }

        else:
            global_render_results = None

        means_new = pcd  # positions for new Gaussians (N_new, 3)
        rotations_new, _ = normal2rotation(pcd_normals)
        dist2 = torch.clamp_min(distCUDA2(pcd), 0.0000001)
        scales_new = torch.log(torch.sqrt(dist2 / 2))[...,None].repeat(1, 3)
        scales_new[..., -1] -= 1e10 # squeeze z scaling
        too_large_scale_mask = torch.any(scales_new > -6, dim=1)
        
        opacities_new = inverse_sigmoid(0.5 * torch.ones((point_num, 1), dtype=torch.float, device="cuda"))
        feat_dim_dc = 1   # one coefficient (l=0) for each color channel
        feat_dim_rest = 0
        if self.active_sh_degree > 0:
            total_coeff = (self.active_sh_degree + 1) ** 2
            feat_dim_rest = total_coeff - 1
        # Create feature tensors
        features_dc_new = torch.zeros(point_num, feat_dim_dc, 3, device=self.device)
        features_rest_new = torch.zeros(point_num, feat_dim_rest, 3, device=self.device) if feat_dim_rest > 0 else torch.empty(point_num, 0, 3, device=self.device)
        features_dc_new[:, 0, :] = rgb_gt.permute(1, 2, 0).view(-1, 3)
        
        view_scores_new = torch.zeros(point_num, device=self.device)
        view_supports_new = torch.zeros(point_num, device=self.device)
        view_means_new = torch.zeros((point_num, 3), device=self.device)

        nan_rotation_mask = torch.any(rotations_new.isnan(), dim=1)
        valid_mask = valid_mask * (~nan_rotation_mask) * (~too_large_scale_mask)
        select_mask = self.cal_mask(rgb_gt, depth_gt, global_render_results)
        select_mask = select_mask.to(self.device)
        select_mask = select_mask * valid_mask
        selected_idx = torch.nonzero(select_mask, as_tuple=False).flatten()
        
        new_count = selected_idx.numel()
        if new_count == 0:
            return  # no new points to add
        new_tensors = {
            "xyz":     means_new.float()[select_mask], 
            "f_dc":    features_dc_new.float()[select_mask],
            "f_rest":  features_rest_new.float()[select_mask],
            "opacity": opacities_new.float()[select_mask],
            "scaling": scales_new.float()[select_mask],
            "rotation": rotations_new.float()[select_mask],
            "view_scores": view_scores_new.float()[select_mask],
            "view_supports": view_supports_new.float()[select_mask],
            "view_means": view_means_new.float()[select_mask]
        }

        if self.optimizer is None:
            # If no optimizer is set yet (initial addition), just assign the tensors (mark as parameters for future optimization)
            self._xyz      = nn.Parameter(new_tensors["xyz"].clone().detach().requires_grad_(True))
            self._features_dc   = nn.Parameter(new_tensors["f_dc"].clone().detach().requires_grad_(True))
            self._features_rest = nn.Parameter(new_tensors["f_rest"].clone().detach().requires_grad_(True))
            self._opacity  = nn.Parameter(new_tensors["opacity"].clone().detach().requires_grad_(True))
            self._scaling  = nn.Parameter(new_tensors["scaling"].clone().detach().requires_grad_(True))
            self._rotation = nn.Parameter(new_tensors["rotation"].clone().detach().requires_grad_(True))
            self.view_scores = new_tensors["view_scores"].clone().detach().to(self.device)
            self.view_supports = new_tensors["view_supports"].clone().detach().to(self.device)
            self.view_means = new_tensors["view_means"].clone().detach().to(self.device)
            
            l = [
                {'params': [self._xyz], 'lr': self.training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': self.training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': self.training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': self.training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': self.training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': self.training_args.rotation_lr, "name": "rotation"}
            ]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            # Model already has an optimizer (adding points mid-training): concatenate new parameters and update optimizer state
            optim_tensors = self.cat_tensors_to_optimizer(new_tensors)
            self._xyz         = optim_tensors["xyz"]
            self._features_dc = optim_tensors["f_dc"]
            self._features_rest = optim_tensors["f_rest"]
            self._opacity     = optim_tensors["opacity"]
            self._scaling     = optim_tensors["scaling"]
            self._rotation    = optim_tensors["rotation"]
            self.view_scores = torch.cat((self.view_scores, new_tensors["view_scores"].detach()), dim=0)
            self.view_supports = torch.cat((self.view_supports, new_tensors["view_supports"].detach()), dim=0)
            self.view_means = torch.cat((self.view_means, new_tensors["view_means"].detach()), dim=0)
            if self.xyz_gradient_accum.numel() != 0:
                zeros_N = torch.zeros((new_count, 1), device=self.device)
                self.xyz_gradient_accum   = torch.cat((self.xyz_gradient_accum,   zeros_N), dim=0)
                self.scale_gradient_accum = torch.cat((self.scale_gradient_accum, zeros_N), dim=0)
                self.rot_gradient_accum   = torch.cat((self.rot_gradient_accum,   zeros_N), dim=0)
                self.opac_gradient_accum  = torch.cat((self.opac_gradient_accum,  zeros_N), dim=0)
                self.denom                = torch.cat((self.denom, zeros_N), dim=0)
            
    
    def cal_mask(self, rgb_gt, depth_gt, pred, save_image_flag=True):
        _, h, w = rgb_gt.shape
        device = rgb_gt.device
        if pred is not None:
            rgb = pred["rgb"].to(device)
            depth = pred["depth"].to(device)
            opacity = pred["opacity"].to(device)
            normal = pred["normal"].to(device)

            rgb_error = torch.mean((rgb_gt - rgb) ** 2, dim=0)
            rgb_error_mask = (rgb_error > self.rgb_error_thres).unsqueeze(0) # (1, h, w)
            mask = rgb_error_mask
            
            depth_error = torch.abs(depth_gt - depth) * (depth_gt > 0)
            valid_mask = (depth > 0) & (depth_error > 0)
            mean = depth_error[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=depth.device)
            non_presence_depth_mask = (depth > depth_gt) * (depth_error > 2*mean)
            mask = mask | non_presence_depth_mask
            mask = mask | (opacity < 0.5)
            
            # normal
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).view(3, 1, 1)
            cos_theta = F.cosine_similarity(normal, z_axis, dim=0)
            nz_mask = (normal.norm(dim=0) > 1e-6)
            cos_theta = torch.where(nz_mask, cos_theta, torch.zeros_like(cos_theta))
            invalid_normal_mask = (cos_theta > 0)
            mask = mask | invalid_normal_mask
            
            mask = mask.squeeze(0) # (h, w)
            if save_image_flag:
                rgb_error_np = rgb_error.detach().cpu().numpy()
                rgb_error_vis = depth2rgb(np.squeeze(rgb_error_np), min_value=rgb_error_np.min(), max_value=rgb_error_np.max(), colormap="jet")
                rgb_error_vis = torch.from_numpy(rgb_error_vis).permute(2, 0, 1).float() / 255.0

                H, W = rgb_error_vis.shape[1:]

                def prepare_mask(mask_np):
                    tensor = torch.from_numpy(mask_np).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
                    return F.interpolate(tensor.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

                rgb_error_mask_np = np.squeeze(rgb_error_mask.detach().cpu().numpy().squeeze(0)).astype(np.uint8) * 255
                rgb_error_mask_vis = prepare_mask(rgb_error_mask_np)

                mask_np = np.squeeze(mask.detach().cpu().numpy()).astype(np.uint8) * 255
                mask_vis = prepare_mask(mask_np)

                opacity_mask_np = np.squeeze((opacity < 0.5).detach().cpu().numpy()).astype(np.uint8) * 255
                opacity_mask_vis = prepare_mask(opacity_mask_np)

                non_presence_depth_mask_np = np.squeeze(non_presence_depth_mask.detach().cpu().numpy()).astype(np.uint8) * 255
                non_presence_depth_mask_vis = prepare_mask(non_presence_depth_mask_np)

                # === Render depth visualization ===
                depth_vis_np = depth.detach().cpu().numpy()
                depth_vis_img = depth2rgb(np.squeeze(depth_vis_np), min_value=depth_vis_np.min(), max_value=depth_vis_np.max(), colormap="viridis")
                depth_vis = torch.from_numpy(depth_vis_img).permute(2, 0, 1).float() / 255.0
                depth_vis = F.interpolate(depth_vis.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

                # === Normal visualization ===
                normal_vis = (normal.detach().cpu() + 1) / 2  # normalize to [0, 1]
                normal_vis = F.interpolate(normal_vis.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

                normal_mask_np = np.squeeze(invalid_normal_mask.detach().cpu().numpy()).astype(np.uint8) * 255
                normal_mask_vis = prepare_mask(normal_mask_np)

                # Combine images
                top_row = torch.cat([rgb.cpu(), rgb_error_vis, rgb_error_mask_vis], dim=2)
                middle_row = torch.cat([depth_vis, non_presence_depth_mask_vis, opacity_mask_vis], dim=2)
                bottom_row = torch.cat([normal_vis, normal_mask_vis, mask_vis], dim=2)

                max_width = max(top_row.shape[2], middle_row.shape[2], bottom_row.shape[2])
                def pad_to_width(tensor, target_width):
                    if tensor.shape[2] < target_width:
                        pad = target_width - tensor.shape[2]
                        return F.pad(tensor, (0, pad, 0, 0), mode='constant', value=0)
                    return tensor

                top_row = pad_to_width(top_row, max_width)
                middle_row = pad_to_width(middle_row, max_width)
                bottom_row = pad_to_width(bottom_row, max_width)

                combined_img = torch.cat([top_row, middle_row, bottom_row], dim=1)
                save_image(combined_img, self.save_dir.joinpath('add_gaussians_mask.png'))
                save_image(combined_img, self.save_dir.parents[1] / 'add_gaussians_mask.png')
                
        else:
            mask = torch.ones(h, w).to(device)

        return rearrange(mask.bool(), "h w -> (h w)")

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, ratio, iteration):
        # if len(self._xyz) < self.opac_reset_record[0] * 1.05 and iteration < self.opac_reset_record[1] + 3000:
        #     print(len(self._xyz), self.opac_reset_record, 'notreset')
        #     return
        # print(len(self._xyz), self.opac_reset_record, 'reset')
        # self.opac_reset_record = [len(self._xyz), iteration]

        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * ratio))
        opacities_new = inverse_sigmoid(self.get_opacity * ratio)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.scale_gradient_accum = self.scale_gradient_accum[valid_points_mask]
        self.rot_gradient_accum = self.rot_gradient_accum[valid_points_mask]
        self.opac_gradient_accum = self.opac_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.view_scores = self.view_scores[valid_points_mask]
        self.view_supports = self.view_supports[valid_points_mask]
        self.view_means = self.view_means[valid_points_mask]
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_view_scores, new_view_supports, new_view_means):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.scale_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.rot_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.opac_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.view_scores = torch.cat([self.view_scores, new_view_scores], dim=0)
        self.view_supports = torch.cat([self.view_supports, new_view_supports], dim=0)
        self.view_means = torch.cat([self.view_means, new_view_means], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, pre_mask=True):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        if self.config[0] > 0:
            new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_view_scores = self.view_scores[selected_pts_mask].repeat(N)
        new_view_supports = self.view_supports[selected_pts_mask].repeat(N)
        new_view_means = self.view_means[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_view_scores, new_view_supports, new_view_means)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, pre_mask=True):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        selected_pts_mask *= pre_mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_view_scores = self.view_scores[selected_pts_mask]
        new_view_supports = self.view_supports[selected_pts_mask]
        new_view_means = self.view_means[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_view_scores, new_view_supports, new_view_means)

    def adaptive_prune(self, min_opacity, extent):

        n_ori = len(self._xyz)

        # prune
        opac_temp = self.get_opacity
        prune_opac =  (opac_temp < min_opacity).squeeze()
        scale_min = self.get_scaling[:, :2].min(1).values
        scale_max = self.get_scaling[:, :2].max(1).values
        prune_scale = scale_max > 0.5 * extent
        prune_scale += (scale_min * scale_max) < (1e-8 * extent**2)
        
        prune_vis = (self.denom == 0).squeeze()
        prune = prune_opac + prune_scale + prune_vis
        self.prune_points(prune)

    def adaptive_densify(self, max_grad, extent):
        grad_pos = self.xyz_gradient_accum / self.denom
        grad_scale = self.scale_gradient_accum /self.denom
        grad_rot = self.rot_gradient_accum /self.denom
        grad_opac = self.opac_gradient_accum /self.denom
        grad_pos[grad_pos.isnan()] = 0.0
        grad_scale[grad_scale.isnan()] = 0.0
        grad_rot[grad_rot.isnan()] = 0.0
        grad_opac[grad_opac.isnan()] = 0.0

        # densify
        larger = torch.le(grad_scale, 1e-7)[:, 0] #if opac_lr == 0 else True
        denser = torch.le(grad_opac, 2)[:, 0]
        pre_mask = denser * larger
        
        self.densify_and_clone(grad_pos, max_grad, extent, pre_mask=pre_mask)
        self.densify_and_split(grad_pos, max_grad, extent)


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.scale_gradient_accum[update_filter] += self._scaling.grad[update_filter, :2].sum(1, True)
        self.rot_gradient_accum[update_filter] += torch.norm(self._rotation[update_filter], dim=-1, keepdim=True)
        self.opac_gradient_accum[update_filter] += self._opacity[update_filter]
        self.denom[update_filter] += 1

    def mask_prune(self, cams, pad=4):
        batch_size = 32
        batch_num = len(cams) // batch_size + int(len(cams) % batch_size != 0)
        cams_batch = [cams[i * batch_size : min(len(cams), (i + 1) * batch_size)] for i in range(batch_num)]
        for c in cams_batch:
            _, _, inMask, outView = world2scrn(self._xyz.detach(), c, pad)
            visible = inMask.all(0) * ~(outView.all(0))
            if list(visible.shape) != []:
                self.prune_points(~visible)

    def to_occ_grid(self, cutoff, grid_dim_max=512, bound_overwrite=None):
        if bound_overwrite is None:
            xyz_min = self._xyz.min(0)[0]
            xyz_max = self._xyz.max(0)[0]
            xyz_len = xyz_max - xyz_min
            xyz_min -= xyz_len * 0.1
            xyz_max += xyz_len * 0.1
        else:
            xyz_min, xyz_max = bound_overwrite
        xyz_len = xyz_max - xyz_min

        grid_len = xyz_len.max() / grid_dim_max
        grid_dim = (xyz_len / grid_len + 0.5).to(torch.int32)

        grid = self.utils_mod.gaussian2occgrid(xyz_min, xyz_max, grid_len, grid_dim,
                                               self.get_xyz, self.get_rotation, self.get_scaling, self.get_opacity,
                                               torch.tensor([cutoff]).to(torch.float32).cuda())
        return grid, -xyz_min, 1 / grid_len, grid_dim
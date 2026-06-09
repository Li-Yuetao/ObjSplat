#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(__file__))
import time
from typing import Dict, List, Union, Tuple
from queue import Queue
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from imgviz import depth2rgb
import resource
import threading
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from dataloader import RGBDSensor
from mapper import MapperState, GaussianColorType, MapperType
from utils import OPENCV_TO_OPENGL
from utils.logging_utils import Log
from mapper.gsmap.gaussian_surfels.cameras import Camera
from mapper.gsmap.gs_backend import GSBackEnd
from mapper.gsmap.frame_buffer import FrameBuffer
from mapper.gsmap.gaussian_surfels.gaussian_renderer import render
from mapper.gsmap.gaussian_surfels.utils.image_utils import depth2rgb as depth2rgb_colormap

class GSMap:
    def __init__(
        self,
        config:dict,
        rgbd_sensor:RGBDSensor, 
        device:torch.device, 
        q_main2vis: Union[Queue, None], 
        results_dir:str, 
        capture_num:int
        ):
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
        self.__device = device
        self.__rgbd_sensor = rgbd_sensor
        self.gaussians_data_dir = Path(os.path.join(results_dir, 'gaussians_data'))
        self.gaussians_data_dir.mkdir(parents=True, exist_ok=True)
        self.gt_depth_dir = self.gaussians_data_dir.joinpath('depth')
        self.gt_depth_dir.mkdir(parents=True, exist_ok=True)
        self.gt_rgb_dir = self.gaussians_data_dir.joinpath('rgb')
        self.gt_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.gt_scene_rgb_dir = self.gaussians_data_dir.joinpath('scene_rgb')
        self.gt_scene_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.gt_mask_dir = self.gaussians_data_dir.joinpath('mask')
        self.gt_mask_dir.mkdir(parents=True, exist_ok=True)
        
        self.__capture_num = capture_num
        self.__kf_every = config['mapper']['keyframe_every']
        self.__map_every = config['mapper']['map_every']
        self.__mapping_iters = config['mapper']['mapping_iters']
        self.__densify_downscale_factor = config['mapper']['densify_downscale_factor']
        self.__densification_image_width = int(self.__rgbd_sensor.width / self.__densify_downscale_factor)
        self.__densification_image_height = int(self.__rgbd_sensor.height / self.__densify_downscale_factor)
        
        self.__mapping_idx = None
        self.__in_trajectory_idx = 0
        self.__tracking_idx = 0
        self.__flag_mapper_finished = False
        self.post_process_finished = False
        
        # ref: hislam2
        self.__gsb_cfg = config['mapper']['gs_backend']
        self.__fb_cfg = config['mapper']['frame_buffer']
        # store images, depth, poses, intrinsics (shared between processes)
        image_size = (self.__densification_image_height, self.__densification_image_width)
        self.frame_buffer = FrameBuffer(self.__fb_cfg, image_size)
        
        # gaussian map
        self.gs = GSBackEnd(self.__gsb_cfg, rgbd_sensor, self.gaussians_data_dir, q_main2vis, device, self.__densify_downscale_factor)
        
        self.post_process_flag = False
        self.mapping_finished = False
        self.post_process_thread = threading.Thread(
            target=self.post_process,
            name='PostPcocess',
            daemon=True)
        self.post_process_thread.start()

    def run(self, batch:Union[Dict[str, torch.Tensor], None]) -> MapperState:
        if self.__mapping_idx is not None:
            mapping_idx_cur = self.__mapping_idx + 1
        
        if batch is None or self.__flag_mapper_finished == True:
            batch_copy = None
            mapper_state = MapperState.ON_MAPPING
            return mapper_state
        else:
            assert batch['frame_id'] == self.__tracking_idx, f'Frame id must be the same, but got {batch["frame_id"]} and {self.__tracking_idx}'
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.__device)
            batch_copy = batch.copy()
            self.__tracking_idx += 1
            if self.__mapping_idx is None:
                mapper_state = MapperState.MAPPING
                self.__mapping_idx = 0
            elif self.__tracking_idx > mapping_idx_cur and self.__tracking_idx <= self.__capture_num:
                self.__mapping_idx = mapping_idx_cur
                mapper_state = MapperState.MAPPING
            else:
                mapper_state = MapperState.IDLE
                
        if mapper_state == MapperState.MAPPING:
            self.__mapping(batch_copy, self.__mapping_idx)
        elif mapper_state == MapperState.IDLE:
            pass
        else:
            raise NotImplementedError(f'Unsupported mapper state: {mapper_state}')
        
        return mapper_state
    
    def collect_data(self, batch:Union[Dict[str, torch.Tensor], None], mode:str='traj') -> None:
        if batch is None:
            return
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.__device)
        batch_copy = batch.copy()
        
        (_, rgb_np, depth_np, mask_np, c2w, gt_c2w) = self.__get_np_data(batch, self.__in_trajectory_idx, save_flag=False)
        
        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0 # [N, 3, H, W]
        depth = torch.from_numpy(depth_np).unsqueeze(0).float() # [N, 1, H, W]
        mask = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0 # [N, 1, H, W]
        extrinsic = torch.from_numpy(c2w).float() # [N, 4, 4]
        gt_extrinsic = torch.from_numpy(gt_c2w).float() if gt_c2w is not None else None
        intrinsic_new = self.__rgbd_sensor.intrinsics.copy()
        intrinsic = torch.from_numpy(intrinsic_new).float() # [N, 3, 3]
        data = {
                'viz_idx':  self.__in_trajectory_idx,
                'tstamp':   self.__in_trajectory_idx,
                'rgb':   rgb,
                'depth':   depth,
                'mask':   mask,
                'extrinsic':    extrinsic,
                'gt_extrinsic':    gt_extrinsic,
                'intrinsic':   intrinsic,
                "depth_range": (self.__rgbd_sensor.depth_min, self.__rgbd_sensor.depth_max),}
        self.gs.process_test_data(data, mode)
        
        self.__in_trajectory_idx = self.__in_trajectory_idx + 1
        return
    
    def __get_np_data(self, batch:Dict[str, torch.Tensor], cur_frame_id=None, save_flag=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scene_rgb_np:np.ndarray = batch['scene_rgb'].detach().cpu().numpy()
        rgb_np:np.ndarray = batch['rgb'].detach().cpu().numpy()
        depth_np:np.ndarray = batch['depth'].detach().cpu().numpy()
        mask_np:np.ndarray = batch['mask'].detach().cpu().numpy()
        c2w:np.ndarray = batch['c2w'].detach().cpu().numpy()
        c2w = c2w @ OPENCV_TO_OPENGL # z-axis facing forward
        self.cur_pose = c2w.copy()
        gt_c2w = None
        if 'gt_c2w' in batch:
            gt_c2w = batch['gt_c2w'].detach().cpu().numpy()
            gt_c2w = gt_c2w @ OPENCV_TO_OPENGL # z-axis facing forward
        
        if save_flag and cur_frame_id is not None:
            # Save RGB Image
            image = np.asarray(rgb_np, dtype=np.uint8).reshape((self.__rgbd_sensor.height, self.__rgbd_sensor.width, 3))
            image_save_path = str(self.gt_rgb_dir.joinpath(f"{str(cur_frame_id).zfill(4)}.png"))
            cv2.imwrite(image_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save Scene RGB Image
            scene_rgb = np.asarray(scene_rgb_np, dtype=np.uint8).reshape((self.__rgbd_sensor.height, self.__rgbd_sensor.width, 3))
            scene_rgb_save_path = str(self.gt_scene_rgb_dir.joinpath(f"{str(cur_frame_id).zfill(4)}.png"))
            cv2.imwrite(scene_rgb_save_path, cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR))
            
            # Save Depth Image(16-bit PNG)
            save_depth = np.clip(depth_np * 6553.5, 0, 65535).astype(np.uint16)
            save_depth = cv2.resize(save_depth, dsize=(
                self.__rgbd_sensor.width, self.__rgbd_sensor.height), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(self.gt_depth_dir.joinpath(f"{str(cur_frame_id).zfill(4)}.png")), save_depth)
            
            # Mask
            mask = mask_np.astype(np.uint8).reshape((self.__rgbd_sensor.height, self.__rgbd_sensor.width))
            mask_save_path = str(self.gt_mask_dir.joinpath(f"{str(cur_frame_id).zfill(4)}.png"))
            cv2.imwrite(mask_save_path, mask)
        
        return scene_rgb_np, rgb_np, depth_np, mask_np, c2w, gt_c2w
    
    def __mapping(self, batch:Union[Dict[str, torch.Tensor], None], cur_frame_id:int=None):
        if self.post_process_flag:
            return
        (_, rgb_np, depth_np, mask_np, c2w, gt_c2w) = self.__get_np_data(batch, cur_frame_id, save_flag=True)
        
        if self.__densify_downscale_factor > 1:
            rgb_np = cv2.resize(rgb_np, (self.__densification_image_width, self.__densification_image_height), interpolation=cv2.INTER_LINEAR)
            depth_np = cv2.resize(depth_np, (self.__densification_image_width, self.__densification_image_height), interpolation=cv2.INTER_NEAREST)
            mask_np = cv2.resize(mask_np, (self.__densification_image_width, self.__densification_image_height), interpolation=cv2.INTER_NEAREST)

        with torch.no_grad():
            rgb = torch.from_numpy(rgb_np).to(device=self.__device).permute(2, 0, 1).float() / 255.0
            depth = torch.from_numpy(depth_np).to(device=self.__device).unsqueeze(0).float()
            mask = torch.from_numpy(mask_np).to(device=self.__device).unsqueeze(0).float() / 255.0
            mask = mask > 0.5
            extrinsic = torch.from_numpy(c2w).to(device=self.__device).float()
            gt_extrinsic = torch.from_numpy(gt_c2w).to(device=self.__device).float() if gt_c2w is not None else None
            intrinsic_new = self.__rgbd_sensor.intrinsics.copy()
            intrinsic_new[0, :] /= self.__densify_downscale_factor
            intrinsic_new[1, :] /= self.__densify_downscale_factor
            intrinsic = torch.from_numpy(intrinsic_new).to(device=self.__device).float()
            self.frame_buffer.append(cur_frame_id, rgb, depth, mask, extrinsic, gt_extrinsic, intrinsic)
            
            if cur_frame_id > 2:
                viz_idx = torch.arange(cur_frame_id-3, cur_frame_id+1, device='cuda') # TODO: use all frames
            else:
                viz_idx = torch.arange(cur_frame_id+1, device='cuda')
        
        self.call_gs(viz_idx)
    
    def call_gs(self, viz_idx):
        data = {'viz_idx':  viz_idx.to(device='cpu'),
                'tstamp':   self.frame_buffer.tstamp[viz_idx].to(device='cpu'),
                'rgb':   self.frame_buffer.rgbs[viz_idx.cpu()],
                'depth':   self.frame_buffer.depths[viz_idx.cpu()],
                'mask':   self.frame_buffer.masks[viz_idx.cpu()],
                'extrinsic':    self.frame_buffer.poses[viz_idx].to(device='cpu'),
                'gt_extrinsic':    self.frame_buffer.gt_poses[viz_idx].to(device='cpu'),
                'intrinsic':   self.frame_buffer.intrinsics[viz_idx].to(device='cpu'),
                "depth_range": (self.__rgbd_sensor.depth_min, self.__rgbd_sensor.depth_max),}
        self.gs.process_track_data(data)
        
        
    def post_process(self):
        while not self.post_process_flag:
            if self.mapping_finished:
                print(f'Exiting post process thread')
                return
            time.sleep(0.5)
        Log('Excute post process')
        self.terminate()
    
    def get_capture_num(self) -> int:
        return int(self.__capture_num)
    
    def get_kf_every(self) -> int:
        return int(self.__kf_every)

    def set_kf_every(self, kf_every:int) -> None:
        self.__kf_every = kf_every
    
    def get_map_every(self) -> int:
        return int(self.__map_every)
    
    def set_map_every(self, map_every:int) -> None:
        self.__map_every = map_every
    
    def get_mapping_iters(self) -> int:
        return int(self.__mapping_iters)

    def get_mapper_type(self) -> str:
        return MapperType.GSMap
    
    def set_candidate_views(self, candidate_views:List[Dict[str, None]]) -> None:
        self.candidate_views = candidate_views

    def get_object_bound(self) -> np.ndarray:
        if self.gs is not None:
            means = self.gs.gaussians.get_xyz
            if means.shape[0] == 0:
                return np.zeros((2, 3))
            xyz = means.cpu().detach().numpy()
            # Remove outliers, just choose 98 percent of points
            lower = np.percentile(xyz, 5, axis=0)
            upper = np.percentile(xyz, 95, axis=0)
            mask = np.all((xyz >= lower) & (xyz <= upper), axis=1)
            filtered_xyz = xyz[mask]
            if filtered_xyz.shape[0] == 0:
                return np.zeros((2, 3))
            min_bound = np.min(filtered_xyz, axis=0)
            max_bound = np.max(filtered_xyz, axis=0)
            return np.vstack((min_bound, max_bound))
        else:
            return np.zeros((2, 3))
    
    def get_current_pose(self) -> np.ndarray:
        return self.cur_pose if hasattr(self, 'cur_pose') else np.eye(4)
    
    def render_rgbd(self, batch:Dict[str, torch.Tensor], scale_modifier=1.0, front_only=True):
        _, _, _, _, view_c2w, _ = self.__get_np_data(batch)
        
        extrinsic = torch.from_numpy(view_c2w).float()
        intrinsic = torch.tensor(
            [[self.__rgbd_sensor.fx, 0.0, self.__rgbd_sensor.cx], [0.0, self.__rgbd_sensor.fy, self.__rgbd_sensor.cy], [0.0, 0.0, 1.0]]
        ).float()
        current_cam = Camera.init_from_gui(
            -1, extrinsic, intrinsic, H=self.__rgbd_sensor.height, W=self.__rgbd_sensor.width, fovx=self.__rgbd_sensor.hfov, fovy=self.__rgbd_sensor.vfov
        )
        patch_size = [float('inf'), float('inf')]
        render_pkg = render(current_cam, self.gs.gaussians, self.gs.gaussians.background_color, patch_size, scaling_modifier=scale_modifier, front_only=front_only, device=self.gs.device)
        color_vis:np.ndarray = torch.permute(torch.clamp(render_pkg["render"], min=0, max=1.0), (1, 2, 0)).detach().cpu().numpy()
        color_vis = (color_vis * 255).astype(np.uint8)
        depth_vis = render_pkg["depth"][0, :, :].detach().cpu().numpy()
        depth_vis = depth2rgb(depth_vis, min_value=self.__rgbd_sensor.depth_min, max_value=self.__rgbd_sensor.depth_max, colormap="jet")
        
        return color_vis, depth_vis
    
    
    @torch.no_grad()
    def render_o3d_image(self, gaussian_cur, current_cam, scale_modifier=1.0, gaussian_color_type:GaussianColorType=GaussianColorType.Color, use_d2n=False, front_only=False):
        
        patch_size = [float('inf'), float('inf')]
        render_pkg = render(current_cam, self.gs.gaussians, self.gs.gaussians.background_color, patch_size, scaling_modifier=scale_modifier, front_only=front_only, device=self.gs.device)
        
        # Choose the type of Gaussian to render
        if gaussian_color_type == GaussianColorType.Depth:
            depth, opacity = render_pkg["depth"], render_pkg["opac"]
            mask_vis = (opacity.detach() > 1e-5)
            depth = depth2rgb_colormap(depth, mask_vis) # 0-1, (3, H, W)
            depth_np = (
                (torch.clamp(depth * opacity, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            return  depth_np
        elif gaussian_color_type == GaussianColorType.Opacity:
            opacity = render_pkg["opac"]
            opacity = opacity[0, :, :].detach().cpu().numpy()
            opacity = depth2rgb(
                opacity, min_value=0.0, max_value=1.0, colormap="jet"
            )
            return opacity
        elif gaussian_color_type == GaussianColorType.Normal:
            normal, opac= render_pkg["normal"], render_pkg["opac"]
            valid_mask = (opac.squeeze(0).detach() > 1e-5)
            normal = torch.nn.functional.normalize(normal, dim=0)
            normal = 1 - torch.add(normal, 1.00000) / 2
            normal = (
                (torch.clamp(normal, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            return normal
        elif gaussian_color_type == GaussianColorType.Confidence:
            confidence = render_pkg["confidence"]
            confidence = confidence[0, :, :].cpu().numpy()
            confidence = depth2rgb(
                1 - confidence, min_value=0, max_value=1, colormap="jet"
            )
            confidence = torch.from_numpy(confidence)
            confidence = torch.permute(confidence, (2, 0, 1)).float()
            confidence = (confidence).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            
            return confidence
        elif gaussian_color_type == GaussianColorType.Uncertainty:
            normal, opac= render_pkg["normal"], render_pkg["opac"]
            valid_mask = (opac.squeeze(0).detach() > 1e-5)
            normal = torch.nn.functional.normalize(normal, dim=0)
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype).view(3, 1, 1)
            cos_theta = F.cosine_similarity(normal, z_axis, dim=0)  # [H, W]
            non_zero_mask = (normal.norm(dim=0) > 1e-6)
            valid_mask = valid_mask & non_zero_mask
            fill_value = torch.tensor(0.0, device=cos_theta.device, dtype=cos_theta.dtype)
            normal_uncertainty = torch.where(valid_mask, cos_theta, fill_value).clamp(0.0, 1.0)
            confidence = render_pkg["confidence"]
            confidence_uncertainty = 1 - confidence[0, :, :]
            angle_mask = cos_theta > 0
            
            uncertainty = torch.where(angle_mask, normal_uncertainty, confidence_uncertainty)
            uncertainty = uncertainty * valid_mask # only consider valid regions
            uncertainty_np = uncertainty.clamp(0.0, 1.0).detach().cpu().numpy()
            uncertainty_colored_np = depth2rgb(
                uncertainty_np, min_value=0.0, max_value=1.0, colormap="turbo"
            )
            return uncertainty_colored_np
        else:
            rgb = (
                (torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            return rgb
    
    def terminate(self):
        torch.cuda.empty_cache()
        self.gs.finalize()
        self.post_process_finished = True
    
    @torch.no_grad()
    def get_view_uncertainty(self, view_c2w:np.ndarray, show_image=False, only_quality=False):
        FoVx = self.__rgbd_sensor.hfov
        FoVy = self.__rgbd_sensor.vfov
        extrinsic = torch.from_numpy(view_c2w).float()
        intrinsic = torch.tensor(
            [[self.__rgbd_sensor.fx, 0.0, self.__rgbd_sensor.cx], [0.0, self.__rgbd_sensor.fy, self.__rgbd_sensor.cy], [0.0, 0.0, 1.0]]
        ).float()
        current_cam = Camera.init_from_gui(
            -1, extrinsic, intrinsic, H=self.__rgbd_sensor.height, W=self.__rgbd_sensor.width, fovx=FoVx, fovy=FoVy
        )
        
        patch_size = [float('inf'), float('inf')]
        render_pkg = render(current_cam, self.gs.gaussians, self.gs.gaussians.background_color, patch_size, device=self.gs.device)
        normal, opac= render_pkg["normal"], render_pkg["opac"]
    
        valid_mask = (opac.squeeze(0).detach() > 0.1)
        # normal & confidence
        normal = torch.nn.functional.normalize(normal, dim=0)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype).view(3, 1, 1)
        cos_theta = F.cosine_similarity(normal, z_axis, dim=0)  # [H, W]
        non_zero_mask = (normal.norm(dim=0) > 1e-6)
        valid_mask = valid_mask & non_zero_mask
        fill_value = torch.tensor(0.0, device=cos_theta.device, dtype=cos_theta.dtype)
        normal_uncertainty = torch.where(valid_mask, cos_theta, fill_value).clamp(0.0, 1.0)
        confidence = render_pkg["confidence"]
        confidence_uncertainty = 1 - confidence[0, :, :]
        angle_mask = cos_theta > 0
        
        uncertainty = torch.where(angle_mask, normal_uncertainty, confidence_uncertainty)
        
        # NOTE: change viewpoint evaluation mode
        full_uncertainty_sum = torch.sum(uncertainty) # Completeness + Quality
        quality_uncertainty_sum = torch.sum(uncertainty * valid_mask) # Quality
        normal_uncertainty_sum = torch.sum(normal_uncertainty)
        
        # NOTE: Normalize by all pixels
        total_pixels = valid_mask.sum().item()
        if total_pixels > 0:
            full_uncertainty_sum /= total_pixels
            quality_uncertainty_sum /= total_pixels
            normal_uncertainty_sum /= total_pixels
        
        # color uncertainty
        image_vis = None
        if show_image:
            if only_quality:
                uncertainty = uncertainty * valid_mask
            uncertainty_np = uncertainty.clamp(0.0, 1.0).detach().cpu().numpy()
            image_vis = depth2rgb(
                uncertainty_np, min_value=0.0, max_value=1.0, colormap="turbo"
            )
            # rgb + uncertainty
            color_vis:np.ndarray = torch.permute(torch.clamp(render_pkg["render"], min=0, max=1.0), (1, 2, 0)).detach().cpu().numpy()
            color_vis = (color_vis * 255).astype(np.uint8)
            image_vis = np.hstack((color_vis, image_vis))
        
        return full_uncertainty_sum.item(), quality_uncertainty_sum.item(), normal_uncertainty_sum.item(), image_vis
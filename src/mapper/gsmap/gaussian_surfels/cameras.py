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
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import cv2
from utils.logging_utils import Log
from mapper.gsmap.gaussian_surfels.utils.graphics_utils import getView2World, getProjectionMatrix, fov2focal, focal2fov
from mapper.gsmap.gaussian_surfels.utils.general_utils import rotmat2quaternion
from mapper.gsmap.gaussian_surfels.utils.image_utils import depth2normal

class Camera(nn.Module):
    def __init__(self, 
                id,
                extrinsic,
                intrinsic=None,
                resolution=None,
                fov=None,
                depth_range=None,
                rgb=None,
                depth=None,
                mask=None,
                tstamp=None,
                device="cuda",
                scene_scale=1,
                camera_lr=None,
                gt_extrinsic=None,
                normal=None):
        super(Camera, self).__init__()
        self.id = id
        try:
            self.device = torch.device(device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {device} failed, fallback to default cuda device" )
            self.device = torch.device("cuda")
        self.tstamp = tstamp
        if resolution is not None:
            H, W = resolution
        else:
            W = None
            H = None
        self.image_width = W if W is not None else self.rgb.shape[2]
        self.image_height = H if H is not None else self.rgb.shape[1]
        self.extrinsic = extrinsic.to(device)
        self.gt_extrinsic = gt_extrinsic if gt_extrinsic is not None else None
        self.intrinsic = intrinsic.to(device)
        self.normalized_intrinsic = self.intrinsic.clone()
        self.normalized_intrinsic[0, :] /= W # normalize
        self.normalized_intrinsic[1, :] /= H
        self.R = extrinsic[:3, :3]
        self.q = rotmat2quaternion(self.R[None], True)[0]
        self.T = extrinsic[:3, 3]
        self.covisibility_rate = 0.0
        self.online_iters = 0 # number of iterations
        
        if fov is not None:
            self.fovx, self.fovy = fov
        else:
            self.fovx = focal2fov(intrinsic[0, 0], resolution[1])
            self.fovy = focal2fov(intrinsic[1, 1], resolution[0])
        
        if depth_range is not None:
            self.znear, self.zfar = depth_range
        else:
            self.znear = 0.01
            self.zfar = 100.0
        self.rgb = rgb.clamp(0.0, 1.0) if rgb is not None else None
        if normal is not None:
            self.normal = F.normalize(normal, dim=0)
        else:
            if depth is not None:
                # compute normal from depth
                mask_depth = (depth > 0.0)
                self.normal = depth2normal(depth, mask_depth, self.fovx, self.fovy)
            else:
                self.normal = None
        self.depth = depth
        self.mask = mask
        self.scene_scale = scene_scale

        self.q = nn.Parameter(self.q.to(torch.float32).contiguous().requires_grad_(True))
        self.T = nn.Parameter(self.T.to(torch.float32).contiguous().requires_grad_(True))
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.to(device) # move q & T to device

        self.lr = camera_lr if camera_lr is not None else 0
        l = [
            {'params': [self.q], 'lr': self.lr * self.scene_scale},
            {'params': [self.T], 'lr': self.lr * self.scene_scale},
            {'params': [self.exposure_a], 'lr': 0.01, "name": "exposure_a_{}".format(id)},
            {'params': [self.exposure_b], 'lr': 0.01, "name": "exposure_b_{}".format(id)},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.prcppoint = torch.tensor([intrinsic[0, 2] / self.image_width, intrinsic[1, 2] / self.image_height], dtype=torch.float32, device=device)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovx, self.fovy, self.image_width, self.image_height, self.prcppoint).transpose(0,1).to(device)
        self.update()
        self.to_cpu()
    
    def to_minimal_dict(self):
        return {
            'id': self.id,
            'extrinsic': self.extrinsic.clone(),
            'gt_extrinsic': self.gt_extrinsic.clone()
        }
    
    def update(self):
        self.extrinsic = getView2World(self.q, self.T)
        self.world_view_transform = self.extrinsic.inverse().transpose(0, 1) # transpose of w2c
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = (-self.T[None]@self.world_view_transform[:3, :3].t())[0]
    
    def to_device(self, device=None):
        device = self.device if device is None else device
        attr_dict = vars(self)
        tensor_keys = [k for k, v in attr_dict.items() if type(v) == torch.Tensor]
        for k in tensor_keys:
            attr_dict[k] = attr_dict[k].to(device) # move tensors to device
        self.to(device) # move parameters to device
        self.device = device
        return self
    
    def to_cpu(self):
        attr_dict = vars(self)
        tensor_keys = [k for k, v in attr_dict.items() if type(v) == torch.Tensor]
        for k in tensor_keys:
            attr_dict[k] = attr_dict[k].cpu()
        self.cpu()
        self.device = torch.device('cpu')
        return self
            
    def get_feat(self, func=None):
        if func is None:
            return self.rgb[None]

        if self.feat is None:
            feat = func(self.rgb[None])
            feat = [i.detach() for i in feat]
            self.feat = feat
            
        return self.feat
    
    def get_intrinsic(self):
        fx = fov2focal(self.fovx, self.image_width)
        fy = fov2focal(self.fovy, self.image_height)
        cx = self.prcppoint[0] * self.image_width
        cy = self.prcppoint[1] * self.image_height
        return torch.tensor([fx, 0, cx,
                             0, fy, cy,
                             0,  0, 1]).reshape([3, 3])

    def get_gtMask(self, with_mask=True):
        if self.mask is None or not with_mask:
            self.mask = torch.ones_like(self.rgb[:1], device="cuda")
        return self.mask#.to(torch.bool)

    def get_gtImage(self, bg, with_mask=True, mask_overwrite=None):
        if self.mask is None or not with_mask:
            return self.rgb
        mask = self.get_gtMask(with_mask) if mask_overwrite is None else mask_overwrite
        return self.rgb * mask + bg[:, None, None] * (1 - mask)
    
    def random_patch(self, h_size=float('inf'), w_size=float('inf')):
        h = self.image_height
        w = self.image_width
        h_size = min(h_size, h)
        w_size = min(w_size, w)
        h0 = random.randint(0, h - h_size)
        w0 = random.randint(0, w - w_size)
        h1 = h0 + h_size
        w1 = w0 + w_size
        return torch.tensor([h0, w0, h1, w1]).to(torch.float32).to(self.device)
    
    def add_noise(self, s):
        T = self.T + (random.random() - 0.5) * s
        self.T = nn.Parameter(T.contiguous().requires_grad_(True))
        self.update()
    
    @classmethod
    def init_from_tracking(cls, id, rgb, depth, depth_range, mask, extrinsic, intrinsic, tstamp, camera_lr, gt_extrinsic):
        _, H, W = rgb.shape
        fovx = focal2fov(intrinsic[0, 0], W)
        fovy = focal2fov(intrinsic[1, 1], H)
        
        return cls(
            id,
            extrinsic,
            intrinsic,
            (H, W),
            (fovx, fovy),
            depth_range,
            rgb,
            depth,
            mask,
            tstamp,
            camera_lr=camera_lr,
            gt_extrinsic=gt_extrinsic,
        )

    @classmethod
    def init_from_gui(cls, id, extrinsic, intrinsic, H, W, fovx, fovy):
        return cls(id, extrinsic, intrinsic, (H, W), (fovx, fovy))
    
    def _update_camera_data(self, rgb_tensor, depth_tensor, mask_tensor):
        # Update the parameters according to the ratio of full image resolution to current image resolution
        H, W = rgb_tensor.shape[1], rgb_tensor.shape[2]
        height_scale = H / self.image_height
        width_scale = W / self.image_width
        self.intrinsic[0, :] *= width_scale
        self.intrinsic[1, :] *= height_scale
        self.image_height = H
        self.image_width = W
        
        self.rgb = rgb_tensor.clamp(0.0, 1.0).to(self.device) if rgb_tensor is not None else None
        self.depth = depth_tensor.to(self.device) if depth_tensor is not None else None
        self.mask = mask_tensor.to(self.device) if mask_tensor is not None else None
        
        # update normal
        if self.normal is not None and self.depth is not None:
            # compute normal from depth
            mask_depth = (self.depth > 0.0)
            self.normal = depth2normal(self.depth, mask_depth, self.fovx, self.fovy)
        
        self.prcppoint = torch.tensor([self.intrinsic[0, 2] / self.image_width, self.intrinsic[1, 2] / self.image_height], dtype=torch.float32, device=self.device)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.fovx, self.fovy, self.image_width, self.image_height, self.prcppoint).transpose(0,1).to(self.device)
        self.update()

    @torch.no_grad()
    def update_frame_data(self, model_path: str, scale:int=1):
        # self.to(data_device)
        rgb_fname = f"{int(self.id):04d}.png"
        rgb_path = os.path.join(model_path, "rgb", rgb_fname)
        depth_path = os.path.join(model_path, "depth", rgb_fname)
        mask_path = os.path.join(model_path, "mask", rgb_fname)

        if not os.path.exists(rgb_path):
            Log(f"Error: RGB file not found at {rgb_path}")
            return False
        if not os.path.exists(depth_path):
            Log(f"Warning: Depth file not found at {depth_path}")
        if not os.path.exists(mask_path):
            Log(f"Warning: Mask file not found at {mask_path}")

        rgb_np = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if rgb_np is None:
            Log(f"Error: Could not read RGB image {rgb_path}"); return False
        if rgb_np.shape[2] == 4: # BGRA
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGRA2RGB)
        else: # BGR
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)

        depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if scale > 1:
            H, W = int(rgb_np.shape[0] / scale), int(rgb_np.shape[1] / scale)
            rgb_np = cv2.resize(rgb_np, (W, H), interpolation=cv2.INTER_LINEAR)
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        
        new_rgb = torch.from_numpy(rgb_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        new_depth = None
        if depth_np is not None:
            new_depth = torch.from_numpy(depth_np.astype(np.float32) / 6553.5).unsqueeze(0)
        new_mask = None
        if mask_np is not None:
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            new_mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)

        self._update_camera_data(
            rgb_tensor=new_rgb,
            depth_tensor=new_depth,
            mask_tensor=new_mask
        )
        
        # self.to_cpu()
        return True
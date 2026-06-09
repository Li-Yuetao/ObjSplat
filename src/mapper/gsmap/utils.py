import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def l1_loss_fc_mask(network_output, gt, mask=None):
    '''
    network_output, gt: (C, H, W)
    mask: (1, H, W) 
    '''

    network_output = network_output.permute(1, 2, 0) # [H, W, C]
    gt = gt.permute(1, 2, 0) # [H, W, C]

    if mask is not None:
        mask = mask.squeeze(0) # [H, W]
        network_output = network_output[mask]
        gt = gt[mask]
    
    loss = ((torch.abs(network_output - gt))).mean()

    return loss

lpips_cal = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
    "cuda"
)

def cal_mse(pred, gt, mask=1.0):
    mse = (((pred - gt) * mask) ** 2).mean().cpu()
    return mse

def cal_psnr(rgb_pred, rgb_gt, mask=1.0):
    mse = cal_mse(rgb_pred, rgb_gt, mask)
    psnr = -10 * math.log10(mse + 1e-8)
    return psnr


def cal_ssim(rgb_pred, rgb_gt, mask=1.0):
    mask = mask.to(rgb_pred.device)
    rgb_pred_masked = rgb_pred * mask
    rgb_gt_masked = rgb_gt * mask
    ssim_cal = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim_cal(rgb_pred_masked.cpu(), rgb_gt_masked.cpu()).item()
    return ssim

def cal_lpips(rgb_pred, rgb_gt, mask=1.0):
    rgb_pred_masked = rgb_pred * mask
    rgb_gt_masked = rgb_gt * mask
    lpips = lpips_cal(rgb_pred_masked, rgb_gt_masked).item()
    return lpips


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def get_smooth_depth(depth, tolerance=0.5):
    invalid_mask = depth < 0.0
    valid_depth_image = np.copy(depth)
    valid_depth_image[invalid_mask] = np.nan
    filtered_depth = cv2.bilateralFilter(
        np.nan_to_num(valid_depth_image), 5, tolerance, 7
    )
    filtered_depth[invalid_mask] = -1.0
    return filtered_depth


def depth2normal(depth, mask, fov):
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(
        torch.arange(0, shape[0], device=device, dtype=torch.float32),
        torch.arange(0, shape[1], device=device, dtype=torch.float32),
        torch.arange(0, shape[2], device=device, dtype=torch.float32),
        indexing="ij",
    )
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= 0.5 * shape[1]
    p[..., 1:2] -= 0.5 * shape[0]
    p *= camD
    K00 = fov2focal(fov[0], shape[0])
    K11 = fov2focal(fov[1], shape[1])
    K = torch.tensor([K00, 0, 0, K11], device=device).reshape([2, 2])
    Kinv = torch.inverse(K)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    p_padded = torch.nn.functional.pad(
        camPos[None], [0, 0, 1, 1, 1, 1], mode="replicate"
    )
    mask_padded = torch.nn.functional.pad(
        mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode="replicate"
    ).to(torch.bool)

    p_c = p_padded[:, 1:-1, 1:-1, :] * mask_padded[:, 1:-1, 1:-1, :]
    p_u = (p_padded[:, :-2, 1:-1, :] - p_c) * mask_padded[:, :-2, 1:-1, :]
    p_l = (p_padded[:, 1:-1, :-2, :] - p_c) * mask_padded[:, 1:-1, :-2, :]
    p_b = (p_padded[:, 2:, 1:-1, :] - p_c) * mask_padded[:, 2:, 1:-1, :]
    p_r = (p_padded[:, 1:-1, 2:, :] - p_c) * mask_padded[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l, dim=-1)
    n_ur = torch.cross(p_r, p_u, dim=-1)
    n_br = torch.cross(p_b, p_r, dim=-1)
    n_bl = torch.cross(p_l, p_b, dim=-1)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    n = torch.nn.functional.normalize(n, dim=-1)

    n = (n * mask).permute([2, 0, 1])
    return n


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
):
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(
    points,
):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors,
):
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates,
    transformation,
):
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates,
    extrinsics,
):
    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(
    coordinates,
    z,
    intrinsics,
):
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates,
    extrinsics,
    intrinsics,
):
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )

    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions
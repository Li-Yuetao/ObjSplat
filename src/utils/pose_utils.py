import numpy as np

def compute_pose_error(T1: np.ndarray, T2: np.ndarray):
    assert T1.shape == (4, 4) and T2.shape == (4, 4)
    translation_error = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R_diff = R1 @ R2.T
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    rotation_error = np.degrees(angle_rad)
    return translation_error, rotation_error

def compute_path_length(cam_centers:np.ndarray) -> float:
    diffs = cam_centers[1:] - cam_centers[:-1]  # shape: (N-1, 3)
    segment_lengths = np.linalg.norm(diffs, axis=1)  # shape: (N-1,)
    return np.sum(segment_lengths)
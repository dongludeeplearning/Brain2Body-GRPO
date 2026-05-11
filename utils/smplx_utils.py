"""
SMPL-X utility helpers.
Converts VQ-VAE decoded motion features → joint positions for reward computation.
"""

from __future__ import annotations
import torch
import numpy as np


# SMPL-X joint indices (22-joint subset commonly used)
JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

LEFT_ANKLE_IDX  = 7
RIGHT_ANKLE_IDX = 8
LEFT_FOOT_IDX   = 10
RIGHT_FOOT_IDX  = 11


def decode_motion_to_joints(motion: torch.Tensor) -> torch.Tensor:
    """
    Lightweight joint extractor from motion feature vectors.
    Assumes motion is stored as (B, T, J*3) joint positions (flat).

    For a full system, integrate smplx library here.
    Returns (B, T, J, 3).
    """
    B, T, D = motion.shape
    J = D // 3
    return motion.reshape(B, T, J, 3)


def extract_foot_joints(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    joints: (B, T, J, 3)
    Returns:
        foot_joints:  (B, T, 2, 3)  [left_ankle, right_ankle]
        foot_contact: (B, T, 2)     binary contact (height < threshold)
    """
    foot = joints[:, :, [LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX], :]  # (B, T, 2, 3)
    contact = foot[..., 1] < 0.05                               # Y-axis height threshold
    return foot, contact


def compute_affordance(
    scene_pcd: np.ndarray,    # (N, 3)
    body_joints: np.ndarray,  # (J, 3) single frame
) -> np.ndarray:
    """
    Affordance map: L2 distance from each scene point to nearest body joint.
    Returns (N,) float array.
    """
    diff = scene_pcd[:, None, :] - body_joints[None, :, :]  # (N, J, 3)
    dists = np.linalg.norm(diff, axis=-1)                    # (N, J)
    return dists.min(axis=-1)                                # (N,)

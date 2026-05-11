"""
Geometry utilities: SDF computation, occupancy grids, scene change detection.
"""

from __future__ import annotations
import numpy as np
import torch


def voxelize(
    pcd: np.ndarray,          # (N, 3)
    resolution: int = 32,
    bounds: tuple = (-1.0, 1.0),
) -> np.ndarray:
    """Convert point cloud to binary 3D occupancy grid (resolution³)."""
    lo, hi = bounds
    idx = ((pcd - lo) / (hi - lo) * resolution).astype(int)
    idx = np.clip(idx, 0, resolution - 1)
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


def compute_scene_delta(
    grid_t: np.ndarray,       # (R, R, R)
    grid_prev: np.ndarray,    # (R, R, R)
) -> float:
    """Mean absolute change between two occupancy grids."""
    return float(np.abs(grid_t.astype(float) - grid_prev.astype(float)).mean())


def local_occupancy_grid(
    scene_pcd: np.ndarray,        # (N, 3)
    center: np.ndarray,           # (3,)  pelvis position
    half_extent: float = 0.6,
    resolution: int = 32,
) -> np.ndarray:
    """
    Extract a local occupancy grid centred at `center`.
    Range: [center - half_extent, center + half_extent] on each axis.
    Returns (resolution³,) flat array.
    """
    local = scene_pcd - center
    mask  = (np.abs(local) <= half_extent).all(axis=-1)
    local_pts = local[mask]
    return voxelize(local_pts, resolution=resolution, bounds=(-half_extent, half_extent)).flatten()


def approximate_sdf(
    query_pts: torch.Tensor,      # (B, V, 3)
    scene_pcd: torch.Tensor,      # (B, N, 3)
) -> torch.Tensor:
    """
    Approximate signed distance by negating the distance to nearest scene point.
    (Positive outside, negative inside — approximation only.)
    Returns (B, V).
    """
    # (B, V, N)
    diff  = query_pts.unsqueeze(2) - scene_pcd.unsqueeze(1)
    dists = diff.norm(dim=-1)
    return dists.min(dim=-1).values   # (B, V) — always positive (unsigned approx)

"""
Scene encoder: encodes a 3D scene point cloud into a fixed-size feature vector.
Uses a lightweight PointNet-style architecture.
Affordance map (L2 distance from scene points to body joints) is appended as a feature channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SharedMLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SceneEncoder(nn.Module):
    """
    Input: point cloud (B, N, 3+C) where C are optional per-point features
           affordance map (B, N) - L2 distance from each scene point to nearest body joint
    Output: scene feature (B, out_dim)
    """
    def __init__(
        self,
        in_channels: int = 4,       # xyz + affordance
        out_dim: int = 256,
        n_points: int = 2048,
    ):
        super().__init__()
        self.n_points = n_points

        # per-point feature extraction
        self.point_mlp = SharedMLP([in_channels, 64, 128, 256])

        # global pooling → scene token
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

        # lightweight change encoder for ΔS (dynamic scenes)
        self.change_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, xyz, affordance=None):
        """
        xyz:        (B, N, 3)
        affordance: (B, N) optional affordance values
        Returns: scene_feat (B, out_dim)
        """
        B, N, _ = xyz.shape

        if affordance is not None:
            pts = torch.cat([xyz, affordance.unsqueeze(-1)], dim=-1)  # (B, N, 4)
        else:
            pts = torch.cat([xyz, torch.zeros(B, N, 1, device=xyz.device)], dim=-1)

        pt_feats = self.point_mlp(pts)                  # (B, N, 256)
        global_feat = pt_feats.max(dim=1).values        # (B, 256)
        return self.global_mlp(global_feat)             # (B, out_dim)

    def encode_change(self, scene_feat_t, scene_feat_prev):
        """Encode scene difference ΔS = S_t - S_{t-1} as a change event feature."""
        delta = scene_feat_t - scene_feat_prev          # (B, out_dim)
        return self.change_mlp(delta)                   # (B, out_dim)

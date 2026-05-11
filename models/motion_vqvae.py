"""
Motion VQ-VAE: encodes SMPL-X motion sequences into discrete tokens.
Architecture follows MotionGPT (arXiv:2306.14795).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class MotionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 4, down_t: int = 2):
        super().__init__()
        # project input joints → hidden
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        blocks = []
        for i in range(n_layers):
            blocks.append(ResBlock1D(hidden_dim))
            if i < down_t:
                # temporal downsampling by 2
                blocks.append(nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1))
        blocks.append(nn.Conv1d(hidden_dim, latent_dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: (B, T, joint_dim) → (B, latent_dim, T//4)
        x = rearrange(x, 'b t d -> b d t')
        x = self.input_proj(x)
        return self.net(x)


class MotionDecoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 4, up_t: int = 2):
        super().__init__()
        blocks = [nn.Conv1d(latent_dim, hidden_dim, 1)]
        for i in range(n_layers):
            if i < up_t:
                # temporal upsampling by 2
                blocks.append(nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1))
            blocks.append(ResBlock1D(hidden_dim))
        blocks.append(nn.Conv1d(hidden_dim, output_dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: (B, latent_dim, T') → (B, T, joint_dim)
        x = self.net(x)
        return rearrange(x, 'b d t -> b t d')


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, z):
        # z: (B, latent_dim, T)
        B, D, T = z.shape
        z_flat = rearrange(z, 'b d t -> (b t) d')   # (B*T, D)

        # L2 distances to codebook entries
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(1)
        )  # (B*T, codebook_size)

        indices = dist.argmin(dim=1)                 # (B*T,)
        z_q_flat = self.codebook(indices)            # (B*T, D)
        z_q = rearrange(z_q_flat, '(b t) d -> b d t', b=B, t=T)

        # straight-through estimator
        loss_embed = F.mse_loss(z_q.detach(), z)
        loss_commit = self.commitment_cost * F.mse_loss(z_q, z.detach())
        z_q_st = z + (z_q - z).detach()

        indices_2d = indices.reshape(B, T)
        return z_q_st, indices_2d, loss_embed + loss_commit

    @torch.no_grad()
    def decode_indices(self, indices):
        # indices: (B, T) → (B, latent_dim, T)
        z_q = self.codebook(indices)
        return rearrange(z_q, 'b t d -> b d t')


class MotionVQVAE(nn.Module):
    """
    Encodes a motion sequence (SMPL-X joint positions) into discrete tokens.
    Token count = T // 4  (two stride-2 downsampling layers by default).
    """
    def __init__(
        self,
        input_dim: int = 205,       # SMPL-X: 22 joints × 3 + 22 joints × 6 + global = flexible
        hidden_dim: int = 512,
        latent_dim: int = 256,
        codebook_size: int = 512,
        n_layers: int = 4,
        down_t: int = 2,
    ):
        super().__init__()
        self.encoder = MotionEncoder(input_dim, hidden_dim, latent_dim, n_layers, down_t)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim)
        self.decoder = MotionDecoder(input_dim, hidden_dim, latent_dim, n_layers, down_t)

    def forward(self, x):
        # x: (B, T, joint_dim)
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        return recon, indices, recon_loss + vq_loss

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        _, indices, _ = self.quantizer(z)
        return indices                               # (B, T//4)

    @torch.no_grad()
    def decode(self, indices):
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q)                    # (B, T, joint_dim)

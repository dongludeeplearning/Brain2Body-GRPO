"""
HSI Dataset loader.
Expects data organized as:
  data_root/
    train/
      sample_000/
        instruction.txt
        scene_pcd.npy       (N, 3) point cloud
        scene_objects.txt   one object name per line
        root_positions.npy  (T, 3) absolute pelvis positions
        motion.npy          (T, joint_dim) SMPL-X joint features
    val/ ...
"""

from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .pseudo_plan import build_pseudo_supervision
from .tokenizer import Brain2BodyTokenizer


class HSIDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer: Brain2BodyTokenizer,
        vqvae,
        max_seq_len: int = 512,
        window_size: int = 48,
        fps: int = 30,
    ):
        self.tokenizer = tokenizer
        self.vqvae = vqvae
        self.max_seq_len = max_seq_len
        self.window_size = window_size

        split_dir = os.path.join(data_root, split)
        self.samples = sorted([
            os.path.join(split_dir, d)
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]

        with open(os.path.join(sample_dir, "instruction.txt")) as f:
            instruction = f.read().strip()

        with open(os.path.join(sample_dir, "scene_objects.txt")) as f:
            scene_objects = [l.strip() for l in f.readlines() if l.strip()]

        scene_pcd       = np.load(os.path.join(sample_dir, "scene_pcd.npy")).astype(np.float32)
        root_positions  = np.load(os.path.join(sample_dir, "root_positions.npy")).astype(np.float32)
        motion_np       = np.load(os.path.join(sample_dir, "motion.npy")).astype(np.float32)

        # random window clip
        T = motion_np.shape[0]
        if T > self.window_size:
            start = np.random.randint(0, T - self.window_size)
            motion_np      = motion_np[start: start + self.window_size]
            root_positions = root_positions[start: start + self.window_size]

        motion_tensor = torch.from_numpy(motion_np).unsqueeze(0)   # (1, T, D)

        # build pseudo supervision (plan + trajectory + motion tokens)
        sup = build_pseudo_supervision(
            instruction, scene_objects, root_positions, self.vqvae, motion_tensor
        )

        # tokenize full sequence
        encoded = self.tokenizer.encode_full(
            instruction, sup["plan"], sup["deltas"], sup["motion_indices"]
        )

        input_ids = torch.tensor(encoded["input_ids"][:self.max_seq_len], dtype=torch.long)
        labels    = torch.tensor(encoded["labels"][:self.max_seq_len], dtype=torch.long)

        # subsample point cloud to fixed size N=2048
        N = 2048
        if len(scene_pcd) >= N:
            idx_pts = np.random.choice(len(scene_pcd), N, replace=False)
        else:
            idx_pts = np.random.choice(len(scene_pcd), N, replace=True)
        scene_pcd_t = torch.from_numpy(scene_pcd[idx_pts])         # (N, 3)

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "scene_pcd":      scene_pcd_t,
            "instruction":    instruction,
            "scene_objects":  scene_objects,
            "root_positions": torch.from_numpy(root_positions),
            "motion":         torch.from_numpy(motion_np),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length sequences to batch maximum."""
    pad_id = 0
    out = {}
    for k in ("input_ids", "labels"):
        seqs    = [b[k] for b in batch]
        max_len = max(s.shape[0] for s in seqs)
        padded  = torch.stack([
            torch.nn.functional.pad(s, (0, max_len - s.shape[0]), value=pad_id)
            for s in seqs
        ])
        out[k]           = padded
        out[f"{k}_mask"] = (padded != pad_id).long()

    out["scene_pcd"]     = torch.stack([b["scene_pcd"] for b in batch])
    out["instruction"]   = [b["instruction"] for b in batch]
    out["scene_objects"] = [b["scene_objects"] for b in batch]
    return out


def build_dataloader(
    data_root: str,
    split: str,
    tokenizer: Brain2BodyTokenizer,
    vqvae,
    batch_size: int = 16,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    dataset = HSIDataset(data_root, split, tokenizer, vqvae, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

"""
Entry point for SFT training.
Usage: python scripts/train_sft.py --config configs/sft_config.yaml
"""

import argparse
import torch
from omegaconf import OmegaConf

from models.motion_vqvae import MotionVQVAE
from models.brain2body import Brain2BodyModel
from data.tokenizer import Brain2BodyTokenizer
from data.dataset import build_dataloader
from training.sft_trainer import SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device

    # 1. load or init VQ-VAE
    vqvae = MotionVQVAE(
        codebook_size=cfg.model.motion_codebook_size,
        latent_dim=cfg.model.motion_token_dim,
    ).to(device)
    if cfg.model.vqvae_ckpt:
        vqvae.load_state_dict(torch.load(cfg.model.vqvae_ckpt, map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    # 2. tokenizer (obj_names loaded from dataset in practice)
    tokenizer = Brain2BodyTokenizer(
        t5_model_name=cfg.model.t5_model_name,
        traj_bins=cfg.model.traj_bins,
        traj_range=cfg.model.traj_range,
        motion_codebook_size=cfg.model.motion_codebook_size,
    )

    # 3. model
    model = Brain2BodyModel(
        t5_model_name=cfg.model.t5_model_name,
        vocab_size=tokenizer.vocab_size,
        scene_feat_dim=cfg.model.scene_feat_dim,
    )

    # 4. dataloaders
    train_loader = build_dataloader(
        cfg.data.data_root, cfg.data.train_split,
        tokenizer, vqvae,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        window_size=cfg.data.window_size,
    )
    val_loader = build_dataloader(
        cfg.data.data_root, cfg.data.val_split,
        tokenizer, vqvae,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        window_size=cfg.data.window_size,
    )

    # 5. train
    trainer = SFTTrainer(model, tokenizer, train_loader, val_loader, cfg, device=device)
    trainer.train()


if __name__ == "__main__":
    main()

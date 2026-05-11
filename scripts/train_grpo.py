"""
Entry point for GRPO fine-tuning.
Usage: python scripts/train_grpo.py --config configs/grpo_config.yaml
"""

import argparse
import copy
import torch
from omegaconf import OmegaConf

from models.motion_vqvae import MotionVQVAE
from models.brain2body import Brain2BodyModel
from models.reward import GeometryReward
from data.tokenizer import Brain2BodyTokenizer
from data.dataset import build_dataloader
from training.grpo_trainer import GRPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device

    # 1. frozen VQ-VAE
    vqvae = MotionVQVAE(
        codebook_size=cfg.model.motion_codebook_size,
        latent_dim=cfg.model.motion_token_dim,
    ).to(device)
    vqvae.load_state_dict(torch.load(cfg.model.vqvae_ckpt, map_location=device))
    vqvae.eval()

    # 2. tokenizer
    tokenizer = Brain2BodyTokenizer(
        t5_model_name=cfg.model.t5_model_name,
        traj_bins=cfg.model.traj_bins,
        traj_range=cfg.model.traj_range,
        motion_codebook_size=cfg.model.motion_codebook_size,
    )

    # 3. policy model (loaded from SFT checkpoint)
    model = Brain2BodyModel(
        t5_model_name=cfg.model.t5_model_name,
        vocab_size=tokenizer.vocab_size,
        scene_feat_dim=cfg.model.scene_feat_dim,
    )
    ckpt = torch.load(cfg.model.sft_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 4. frozen reference model (deep copy of SFT model)
    ref_model = copy.deepcopy(model)

    # 5. reward function
    reward_fn = GeometryReward(
        w_goal=cfg.reward.w_goal,
        w_contact=cfg.reward.w_contact,
        w_penetration=cfg.reward.w_penetration,
        w_foot=cfg.reward.w_foot,
        w_smooth=cfg.reward.w_smooth,
        w_plan=cfg.reward.w_plan,
    )

    # 6. dataloader
    train_loader = build_dataloader(
        cfg.data.data_root, cfg.data.train_split,
        tokenizer, vqvae,
        batch_size=cfg.training.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        window_size=cfg.data.window_size,
    )

    # 7. train
    trainer = GRPOTrainer(
        model, ref_model, vqvae, reward_fn,
        tokenizer, train_loader, cfg, device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()

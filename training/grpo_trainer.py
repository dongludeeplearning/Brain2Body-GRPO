"""
GRPO Fine-tuning Trainer (Step 6 in diagram).

For each input, sample K complete sequences, decode motion, compute geometry rewards,
compute group-relative advantage, and update AR policy.

L_GRPO = -1/K Σ Aᵢ log πθ(Yᵢ|X) + β KL(πθ||π_ref)
where Aᵢ = (Rᵢ - mean(R)) / (std(R) + ε)
"""

from __future__ import annotations
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from models.brain2body import Brain2BodyModel
from models.motion_vqvae import MotionVQVAE
from models.reward import GeometryReward
from data.tokenizer import Brain2BodyTokenizer
from utils.smplx_utils import decode_motion_to_joints


class GRPOTrainer:
    def __init__(
        self,
        model: Brain2BodyModel,
        ref_model: Brain2BodyModel,      # frozen reference (π_ref)
        vqvae: MotionVQVAE,              # frozen motion decoder
        reward_fn: GeometryReward,
        tokenizer: Brain2BodyTokenizer,
        train_loader,
        cfg: DictConfig,
        device: str = "cuda",
    ):
        self.model      = model.to(device)
        self.ref_model  = ref_model.to(device)
        self.vqvae      = vqvae.to(device)
        self.reward_fn  = reward_fn.to(device)
        self.tokenizer  = tokenizer
        self.train_loader = train_loader
        self.cfg    = cfg
        self.device = device

        # freeze ref model and vqvae
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        for p in self.vqvae.parameters():
            p.requires_grad_(False)

        self.optimizer  = AdamW(model.parameters(), lr=cfg.training.lr)
        self.K          = cfg.grpo.K
        self.beta       = cfg.grpo.beta
        self.eps        = cfg.grpo.eps
        self.global_step = 0

        os.makedirs(cfg.training.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Parse generated token sequence → plan, traj, motion_indices
    # ------------------------------------------------------------------
    def _parse_output(self, token_ids: list[int]) -> dict:
        sid = self.tokenizer._section_ids
        mo  = self.tokenizer._motion_offset
        mc  = self.tokenizer.motion_codebook_size

        plan_ids, traj_ids, motion_ids = [], [], []
        cur = None
        for t in token_ids:
            if t == sid["<PLAN>"]:      cur = "plan"
            elif t == sid["<TRAJ>"]:    cur = "traj"
            elif t == sid["<MOTION>"]:  cur = "motion"
            elif t in (sid["</PLAN>"], sid["</TRAJ>"], sid["</MOTION>"]):
                cur = None
            elif cur == "plan":   plan_ids.append(t)
            elif cur == "traj":   traj_ids.append(t)
            elif cur == "motion": motion_ids.append(t)

        plan    = self.tokenizer.decode_plan(plan_ids)
        deltas  = self.tokenizer.decode_trajectory(traj_ids)
        m_idx   = [t - mo for t in motion_ids if mo <= t < mo + mc]
        return {"plan": plan, "deltas": deltas, "motion_indices": m_idx}

    # ------------------------------------------------------------------
    # GRPO step for a single sample
    # ------------------------------------------------------------------
    def _grpo_step(self, batch: dict) -> dict:
        # take first sample in batch (GRPO is per-sample)
        input_ids  = batch["input_ids"][[0]].to(self.device)       # (1, L)
        scene_pcd  = batch["scene_pcd"][[0]].to(self.device)       # (1, N, 3)
        goal_pos   = batch["root_positions"][[0], -1].to(self.device)  # (1, 3)
        scene_objs = batch["scene_objects"][0]

        # 1. sample K sequences
        with torch.no_grad():
            sampled = self.model.sample_K(
                input_ids, scene_pcd,
                K=self.K,
                temperature=self.cfg.grpo.temperature,
                top_p=self.cfg.grpo.top_p,
            )   # (K, L_out)

        # 2. decode motion from each sample and compute rewards
        rewards = []
        parsed_list = []
        for k in range(self.K):
            ids = sampled[k].tolist()
            parsed = self._parse_output(ids)
            parsed_list.append(parsed)

            m_idx = parsed["motion_indices"]
            if len(m_idx) == 0:
                rewards.append(-1.0)   # invalid output
                continue

            m_tensor = torch.tensor(m_idx, device=self.device).unsqueeze(0)   # (1, N)
            try:
                motion = self.vqvae.decode(m_tensor)    # (1, T, joint_dim)
                joints = decode_motion_to_joints(motion) # (1, T, J, 3)
            except Exception:
                rewards.append(-1.0)
                continue

            final_pos = joints[0, -1, 0]  # root joint of last frame

            # placeholder surface (real impl uses scene mesh)
            obj_surface  = scene_pcd[0].unsqueeze(0)    # (1, N, 3)
            foot_joints  = joints[:, :, [7, 8]]         # (1, T, 2, 3) L/R ankle
            foot_contact = (foot_joints[..., 1] < 0.05) # simple height threshold

            r_dict = self.reward_fn(
                final_pos=final_pos.unsqueeze(0),
                goal_pos=goal_pos,
                body_joints=joints,
                obj_surface=obj_surface,
                foot_joints=foot_joints,
                foot_contact=foot_contact,
                decoded_plan=[parsed["plan"]],
                scene_objects_list=[scene_objs],
            )
            rewards.append(r_dict["total"].item())

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (K,)

        # 3. group-relative advantage
        mean_r = rewards_t.mean()
        std_r  = rewards_t.std()
        adv    = (rewards_t - mean_r) / (std_r + self.eps)   # (K,)

        # 4. compute log-probs under current policy and reference policy
        input_ids_k   = input_ids.expand(self.K, -1)
        scene_pcd_k   = scene_pcd.expand(self.K, -1, -1)

        log_probs_cur = self.model.compute_log_probs(
            input_ids_k, scene_pcd_k, sampled
        )  # (K, L_out-1)

        with torch.no_grad():
            log_probs_ref = self.ref_model.compute_log_probs(
                input_ids_k, scene_pcd_k, sampled
            )  # (K, L_out-1)

        # sequence log-prob = sum over tokens
        seq_log_probs_cur = log_probs_cur.sum(dim=-1)   # (K,)
        seq_log_probs_ref = log_probs_ref.sum(dim=-1)   # (K,)

        # 5. GRPO loss
        policy_loss = -(adv * seq_log_probs_cur).mean()
        kl_penalty  = (seq_log_probs_cur - seq_log_probs_ref).mean()
        loss        = policy_loss + self.beta * kl_penalty

        return {
            "loss":          loss,
            "policy_loss":   policy_loss,
            "kl_penalty":    kl_penalty,
            "mean_reward":   mean_r,
            "max_reward":    rewards_t.max(),
            "reward_std":    std_r,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=dict(cfg))

        for epoch in range(cfg.training.num_epochs):
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"GRPO Epoch {epoch}"):
                metrics = self._grpo_step(batch)

                self.optimizer.zero_grad()
                metrics["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.grad_clip)
                self.optimizer.step()
                self.global_step += 1

                if self.global_step % cfg.training.log_every == 0:
                    wandb.log({
                        "grpo/loss":         metrics["loss"].item(),
                        "grpo/policy_loss":  metrics["policy_loss"].item(),
                        "grpo/kl_penalty":   metrics["kl_penalty"].item(),
                        "grpo/mean_reward":  metrics["mean_reward"].item(),
                        "grpo/max_reward":   metrics["max_reward"].item(),
                        "grpo/reward_std":   metrics["reward_std"].item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }, step=self.global_step)

                if self.global_step % cfg.training.save_every == 0:
                    self._save(f"grpo_step_{self.global_step}.pt")

        self._save("grpo_final.pt")
        wandb.finish()

    def _save(self, name: str):
        path = os.path.join(self.cfg.training.output_dir, name)
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step":          self.global_step,
        }, path)

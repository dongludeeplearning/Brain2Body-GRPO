"""
SFT Trainer (Step 3 in diagram).
Loss: L = L_plan + λ₁ L_traj + λ₂ L_motion
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

from models.brain2body import Brain2BodyModel
from data.tokenizer import Brain2BodyTokenizer, SECTION_TOKENS


def _build_section_masks(labels: torch.Tensor, tokenizer: Brain2BodyTokenizer):
    """
    Given label token ids, return boolean masks for plan/traj/motion positions.
    """
    sid = tokenizer._section_ids
    B, L = labels.shape
    device = labels.device

    plan_mask   = torch.zeros(B, L, dtype=torch.bool, device=device)
    traj_mask   = torch.zeros(B, L, dtype=torch.bool, device=device)
    motion_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    for b in range(B):
        cur_section = None
        for t in range(L):
            tok = labels[b, t].item()
            if tok == sid["<PLAN>"]:
                cur_section = "plan"
            elif tok == sid["<TRAJ>"]:
                cur_section = "traj"
            elif tok == sid["<MOTION>"]:
                cur_section = "motion"
            elif tok in (sid["</PLAN>"], sid["</TRAJ>"], sid["</MOTION>"]):
                cur_section = None
            elif cur_section == "plan":
                plan_mask[b, t] = True
            elif cur_section == "traj":
                traj_mask[b, t] = True
            elif cur_section == "motion":
                motion_mask[b, t] = True

    return plan_mask, traj_mask, motion_mask


class SFTTrainer:
    def __init__(
        self,
        model: Brain2BodyModel,
        tokenizer: Brain2BodyTokenizer,
        train_loader,
        val_loader,
        cfg: DictConfig,
        device: str = "cuda",
    ):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg    = cfg
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=cfg.training.lr)

        total_steps   = len(train_loader) * cfg.training.num_epochs
        warmup_steps  = cfg.training.warmup_steps
        warmup_sched  = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        decay_sched   = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1,
                                 total_iters=total_steps - warmup_steps)
        self.scheduler = SequentialLR(self.optimizer, [warmup_sched, decay_sched], milestones=[warmup_steps])

        self.lambda_traj   = cfg.training.lambda_traj
        self.lambda_motion = cfg.training.lambda_motion
        self.global_step   = 0
        self.best_val_loss = float("inf")

        os.makedirs(cfg.training.output_dir, exist_ok=True)

    def _step(self, batch: dict) -> dict:
        input_ids  = batch["input_ids"].to(self.device)
        labels     = batch["labels"].to(self.device)
        scene_pcd  = batch["scene_pcd"].to(self.device)
        attn_mask  = batch["input_ids_mask"].to(self.device)

        out = self.model(
            input_ids=input_ids,
            labels=labels,
            scene_pcd=scene_pcd,
            attention_mask=attn_mask,
        )

        # per-section losses
        plan_mask, traj_mask, motion_mask = _build_section_masks(labels, self.tokenizer)
        losses = self.model.compute_section_losses(
            out["logits"], labels,
            plan_mask, traj_mask, motion_mask,
            lambda_traj=self.lambda_traj,
            lambda_motion=self.lambda_motion,
        )
        return losses

    def train(self):
        cfg = self.cfg
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=dict(cfg))

        for epoch in range(cfg.training.num_epochs):
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                losses = self._step(batch)
                loss   = losses["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % cfg.training.log_every == 0:
                    wandb.log({
                        "train/loss":        losses["loss"].item(),
                        "train/loss_plan":   losses["loss_plan"].item(),
                        "train/loss_traj":   losses["loss_traj"].item(),
                        "train/loss_motion": losses["loss_motion"].item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }, step=self.global_step)

                if self.global_step % cfg.training.save_every == 0:
                    self._save(f"step_{self.global_step}.pt")

            val_loss = self._validate()
            wandb.log({"val/loss": val_loss}, step=self.global_step)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save("best.pt")

        wandb.finish()

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            losses = self._step(batch)
            total += losses["loss"].item()
            count += 1
        return total / max(count, 1)

    def _save(self, name: str):
        path = os.path.join(self.cfg.training.output_dir, name)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

"""
Brain2Body AR Policy (Step 3 in diagram).

T5-based autoregressive model that generates a single sequence:
  <PLAN> plan_tokens </PLAN>
  <TRAJ> traj_tokens </TRAJ>
  <MOTION> motion_tokens </MOTION>

conditioned on: instruction text + scene features (from SceneEncoder).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

from .scene_encoder import SceneEncoder


class Brain2BodyModel(nn.Module):
    """
    AR policy built on top of T5.
    Scene features are injected as a soft prompt prepended to the encoder output.
    """

    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-base",
        vocab_size: int | None = None,   # after adding special tokens
        scene_feat_dim: int = 256,
        n_scene_tokens: int = 4,         # number of soft scene prompt tokens
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_dim = self.t5.config.d_model

        # extend embedding table if vocabulary was expanded
        if vocab_size is not None and vocab_size > self.t5.config.vocab_size:
            self.t5.resize_token_embeddings(vocab_size)

        self.scene_encoder = SceneEncoder(in_channels=4, out_dim=scene_feat_dim)

        # project scene feature → n_scene_tokens soft prompt tokens
        self.scene_proj = nn.Linear(scene_feat_dim, n_scene_tokens * t5_dim)
        self.n_scene_tokens = n_scene_tokens
        self.t5_dim = t5_dim

    # ------------------------------------------------------------------
    # Forward pass (SFT training)
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,          # (B, L_text)
        labels: torch.Tensor,             # (B, L_out)
        scene_pcd: torch.Tensor,          # (B, N, 3)
        attention_mask: torch.Tensor | None = None,
        affordance: torch.Tensor | None = None,
    ) -> dict:
        B = input_ids.shape[0]

        # encode scene → soft prompt
        scene_feat  = self.scene_encoder(scene_pcd, affordance)           # (B, scene_feat_dim)
        scene_tokens = self.scene_proj(scene_feat).view(B, self.n_scene_tokens, self.t5_dim)

        # get text encoder embeddings
        text_embeds = self.t5.encoder.embed_tokens(input_ids)             # (B, L, d)

        # prepend scene tokens to text embeddings
        encoder_inputs = torch.cat([scene_tokens, text_embeds], dim=1)    # (B, n+L, d)

        # build extended attention mask
        scene_mask = torch.ones(B, self.n_scene_tokens, device=input_ids.device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        ext_mask = torch.cat([scene_mask, attention_mask], dim=1)

        # run T5 encoder manually, then pass to decoder
        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=ext_mask,
        )

        # decoder with labels → computes cross-entropy loss internally
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=ext_mask,
            labels=labels,
        )

        return {
            "loss":   outputs.loss,
            "logits": outputs.logits,
        }

    # ------------------------------------------------------------------
    # Per-section losses for SFT (L_plan + λ_traj * L_traj + λ_motion * L_motion)
    # ------------------------------------------------------------------
    def compute_section_losses(
        self,
        logits: torch.Tensor,     # (B, L_out, V)
        labels: torch.Tensor,     # (B, L_out)
        plan_mask: torch.Tensor,  # (B, L_out) bool — positions that are plan tokens
        traj_mask: torch.Tensor,
        motion_mask: torch.Tensor,
        lambda_traj: float = 0.5,
        lambda_motion: float = 1.0,
    ) -> dict:
        import torch.nn.functional as F
        log_probs = F.log_softmax(logits, dim=-1)       # (B, L, V)
        nll = -log_probs.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, L)
        valid = (labels >= 0).float()

        def masked_mean(nll, mask):
            m = mask.float() * valid
            return (nll * m).sum() / (m.sum() + 1e-8)

        l_plan   = masked_mean(nll, plan_mask)
        l_traj   = masked_mean(nll, traj_mask)
        l_motion = masked_mean(nll, motion_mask)
        total    = l_plan + lambda_traj * l_traj + lambda_motion * l_motion

        return {
            "loss":         total,
            "loss_plan":    l_plan,
            "loss_traj":    l_traj,
            "loss_motion":  l_motion,
        }

    # ------------------------------------------------------------------
    # Autoregressive generation (inference / GRPO sampling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        scene_pcd: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        affordance: torch.Tensor | None = None,
        max_new_tokens: int = 400,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        scene_feat   = self.scene_encoder(scene_pcd, affordance)
        scene_tokens = self.scene_proj(scene_feat).view(B, self.n_scene_tokens, self.t5_dim)

        text_embeds  = self.t5.encoder.embed_tokens(input_ids)
        encoder_inputs = torch.cat([scene_tokens, text_embeds], dim=1)

        scene_mask = torch.ones(B, self.n_scene_tokens, device=input_ids.device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        ext_mask = torch.cat([scene_mask, attention_mask], dim=1)

        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=ext_mask,
        )

        generated = self.t5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=ext_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )
        return generated   # (B * num_return_sequences, L_out)

    # ------------------------------------------------------------------
    # Sampling for GRPO (returns K sequences with log-probs)
    # ------------------------------------------------------------------
    def sample_K(
        self,
        input_ids: torch.Tensor,    # (1, L)  — single sample
        scene_pcd: torch.Tensor,    # (1, N, 3)
        K: int = 8,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_new_tokens: int = 400,
        affordance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns (K, L_out) sampled token sequences."""
        input_ids_rep  = input_ids.expand(K, -1)
        scene_pcd_rep  = scene_pcd.expand(K, -1, -1)
        aff_rep        = affordance.expand(K, -1) if affordance is not None else None

        return self.generate(
            input_ids_rep, scene_pcd_rep,
            affordance=aff_rep,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
        )   # (K, L_out)

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,    # (B, L_in)
        scene_pcd: torch.Tensor,    # (B, N, 3)
        generated_ids: torch.Tensor,  # (B, L_out)
        attention_mask: torch.Tensor | None = None,
        affordance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute per-token log-probs for given (input, output) pairs.
        Used in GRPO loss.
        Returns (B, L_out) log-probs.
        """
        B = input_ids.shape[0]
        scene_feat   = self.scene_encoder(scene_pcd, affordance)
        scene_tokens = self.scene_proj(scene_feat).view(B, self.n_scene_tokens, self.t5_dim)
        text_embeds  = self.t5.encoder.embed_tokens(input_ids)
        encoder_inputs = torch.cat([scene_tokens, text_embeds], dim=1)

        scene_mask = torch.ones(B, self.n_scene_tokens, device=input_ids.device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        ext_mask = torch.cat([scene_mask, attention_mask], dim=1)

        encoder_outputs = self.t5.encoder(
            inputs_embeds=encoder_inputs,
            attention_mask=ext_mask,
        )

        # shift labels: decoder input is generated_ids[:-1], target is generated_ids[1:]
        decoder_input  = generated_ids[:, :-1]
        target         = generated_ids[:, 1:]

        out = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=ext_mask,
            decoder_input_ids=decoder_input,
        )
        import torch.nn.functional as F
        log_probs = F.log_softmax(out.logits, dim=-1)           # (B, L-1, V)
        token_log_probs = log_probs.gather(
            2, target.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)                                           # (B, L-1)
        return token_log_probs

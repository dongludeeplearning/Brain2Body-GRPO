"""
Multi-modal tokenizer for Brain2Body-GRPO.

Extends T5 vocabulary with:
  - Plan tokens:    <APPROACH:X>, <AVOID:X>, <ALIGN:X>, <INTERACT:X>
  - Section tokens: <PLAN>, </PLAN>, <TRAJ>, </TRAJ>, <MOTION>, </MOTION>
  - Traj tokens:    <TRAJ_dx_{i}>, <TRAJ_dy_{i}>, <TRAJ_dz_{i}>  (quantized bins)
  - Motion tokens:  <MOTION_{i}>  (VQ-VAE codebook indices)
"""

from __future__ import annotations
import re
import numpy as np
import torch
from transformers import T5Tokenizer


PLAN_ACTIONS = ["APPROACH", "AVOID", "ALIGN", "INTERACT"]
SECTION_TOKENS = ["<PLAN>", "</PLAN>", "<TRAJ>", "</TRAJ>", "<MOTION>", "</MOTION>"]


def _build_special_tokens(
    obj_names: list[str],
    traj_bins: int,
    motion_codebook_size: int,
) -> list[str]:
    tokens = list(SECTION_TOKENS)
    for action in PLAN_ACTIONS:
        for obj in obj_names:
            tokens.append(f"<{action}:{obj}>")
    for axis in ["dx", "dy", "dz"]:
        for i in range(traj_bins):
            tokens.append(f"<TRAJ_{axis}_{i}>")
    for i in range(motion_codebook_size):
        tokens.append(f"<MOTION_{i}>")
    return tokens


class Brain2BodyTokenizer:
    """
    Wraps a T5Tokenizer and adds multi-modal special tokens.
    Provides helpers to encode/decode plan, trajectory, and motion tokens.
    """

    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-base",
        obj_names: list[str] | None = None,
        traj_bins: int = 64,
        traj_range: float = 2.0,
        motion_codebook_size: int = 512,
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.traj_bins = traj_bins
        self.traj_range = traj_range
        self.motion_codebook_size = motion_codebook_size
        self.obj_names = obj_names or []

        special = _build_special_tokens(self.obj_names, traj_bins, motion_codebook_size)
        self.tokenizer.add_special_tokens({"additional_special_tokens": special})

        # cache token ids for fast lookup
        self._section_ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in SECTION_TOKENS}
        self._traj_offset = {
            axis: self.tokenizer.convert_tokens_to_ids(f"<TRAJ_{axis}_0>")
            for axis in ["dx", "dy", "dz"]
        }
        self._motion_offset = self.tokenizer.convert_tokens_to_ids("<MOTION_0>")

    # ------------------------------------------------------------------ #
    # Plan encoding
    # ------------------------------------------------------------------ #
    def encode_plan(self, plan: list[tuple[str, str]]) -> list[int]:
        """
        plan: [("APPROACH", "sofa"), ("AVOID", "table"), ...]
        Returns list of token ids including section markers.
        """
        ids = [self._section_ids["<PLAN>"]]
        for action, obj in plan:
            tok = f"<{action}:{obj}>"
            ids.append(self.tokenizer.convert_tokens_to_ids(tok))
        ids.append(self._section_ids["</PLAN>"])
        return ids

    def decode_plan(self, ids: list[int]) -> list[tuple[str, str]]:
        result = []
        for tid in ids:
            if tid in (self._section_ids["<PLAN>"], self._section_ids["</PLAN>"]):
                continue
            tok = self.tokenizer.convert_ids_to_tokens(tid)
            m = re.match(r"<(\w+):(\w+)>", tok)
            if m:
                result.append((m.group(1), m.group(2)))
        return result

    # ------------------------------------------------------------------ #
    # Trajectory encoding  (Δpos per step)
    # ------------------------------------------------------------------ #
    def _quantize(self, val: float) -> int:
        val = np.clip(val, -self.traj_range, self.traj_range)
        return int((val + self.traj_range) / (2 * self.traj_range) * (self.traj_bins - 1))

    def _dequantize(self, idx: int) -> float:
        return idx / (self.traj_bins - 1) * 2 * self.traj_range - self.traj_range

    def encode_trajectory(self, deltas: np.ndarray) -> list[int]:
        """
        deltas: (T, 3) relative root position increments
        Returns token ids: <TRAJ> dx_0 dy_0 dz_0 dx_1 ... </TRAJ>
        """
        ids = [self._section_ids["<TRAJ>"]]
        for step in deltas:
            for axis_i, axis in enumerate(["dx", "dy", "dz"]):
                bin_idx = self._quantize(step[axis_i])
                ids.append(self._traj_offset[axis] + bin_idx)
        ids.append(self._section_ids["</TRAJ>"])
        return ids

    def decode_trajectory(self, ids: list[int]) -> np.ndarray:
        """Returns (T, 3) float array."""
        steps = []
        cur = []
        for tid in ids:
            if tid in (self._section_ids["<TRAJ>"], self._section_ids["</TRAJ>"]):
                continue
            for axis in ["dx", "dy", "dz"]:
                offset = self._traj_offset[axis]
                if offset <= tid < offset + self.traj_bins:
                    cur.append(self._dequantize(tid - offset))
                    break
            if len(cur) == 3:
                steps.append(cur)
                cur = []
        return np.array(steps, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Motion token encoding
    # ------------------------------------------------------------------ #
    def encode_motion(self, indices: torch.Tensor) -> list[int]:
        """
        indices: (N,) VQ-VAE codebook indices
        Returns token ids: <MOTION> m_0 m_1 ... </MOTION>
        """
        ids = [self._section_ids["<MOTION>"]]
        ids += [self._motion_offset + int(i) for i in indices]
        ids.append(self._section_ids["</MOTION>"])
        return ids

    def decode_motion_indices(self, ids: list[int]) -> list[int]:
        result = []
        for tid in ids:
            if tid in (self._section_ids["<MOTION>"], self._section_ids["</MOTION>"]):
                continue
            offset = self._motion_offset
            if offset <= tid < offset + self.motion_codebook_size:
                result.append(tid - offset)
        return result

    # ------------------------------------------------------------------ #
    # Full sequence
    # ------------------------------------------------------------------ #
    def encode_full(
        self,
        instruction: str,
        plan: list[tuple[str, str]],
        deltas: np.ndarray,
        motion_indices: torch.Tensor,
    ) -> dict:
        """Returns input_ids and target_ids for SFT training."""
        input_ids = self.tokenizer(instruction, return_tensors="pt").input_ids[0].tolist()
        target_ids = (
            self.encode_plan(plan)
            + self.encode_trajectory(deltas)
            + self.encode_motion(motion_indices)
            + [self.tokenizer.eos_token_id]
        )
        return {"input_ids": input_ids, "labels": target_ids}

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

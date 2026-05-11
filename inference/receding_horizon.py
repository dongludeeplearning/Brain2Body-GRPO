"""
Receding Horizon Execution (Step 7 in diagram).

At each timestep t:
  1. Observe Scene_t
  2. Generate Plan_t + ΔPOS_t + Motion_t
  3. Execute K frames (rollout)
  4. At t+K: check if scene changed → re-plan if yes, else continue
"""

from __future__ import annotations
import numpy as np
import torch

from models.brain2body import Brain2BodyModel
from models.motion_vqvae import MotionVQVAE
from models.scene_encoder import SceneEncoder
from data.tokenizer import Brain2BodyTokenizer
from utils.smplx_utils import decode_motion_to_joints


SCENE_CHANGE_THRESHOLD = 0.1   # mean feature distance to trigger re-plan


class RecedingHorizonExecutor:
    """
    Runs the Brain2Body policy in a dynamic environment loop.

    Supports:
      - Online Plan Update
      - Local Trajectory Update
      - GRPO Local Optimization (optional, for online adaptation)
    """

    def __init__(
        self,
        model: Brain2BodyModel,
        vqvae: MotionVQVAE,
        tokenizer: Brain2BodyTokenizer,
        execute_horizon: int = 16,      # frames to execute before re-checking
        max_steps: int = 300,
        device: str = "cuda",
    ):
        self.model     = model.to(device).eval()
        self.vqvae     = vqvae.to(device).eval()
        self.tokenizer = tokenizer
        self.execute_horizon = execute_horizon
        self.max_steps  = max_steps
        self.device     = device

        self._prev_scene_feat: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Scene change detection
    # ------------------------------------------------------------------
    def _scene_changed(self, scene_feat: torch.Tensor) -> bool:
        if self._prev_scene_feat is None:
            return True
        delta = (scene_feat - self._prev_scene_feat).norm().item()
        return delta > SCENE_CHANGE_THRESHOLD

    # ------------------------------------------------------------------
    # Single planning step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _plan(
        self,
        instruction: str,
        scene_pcd: np.ndarray,          # (N, 3)
        affordance: np.ndarray | None = None,   # (N,)
    ) -> dict:
        pcd_t = torch.from_numpy(scene_pcd).unsqueeze(0).float().to(self.device)  # (1, N, 3)
        aff_t = (
            torch.from_numpy(affordance).unsqueeze(0).float().to(self.device)
            if affordance is not None else None
        )

        input_ids = self.tokenizer.tokenizer(
            instruction, return_tensors="pt"
        ).input_ids.to(self.device)

        generated = self.model.generate(
            input_ids=input_ids,
            scene_pcd=pcd_t,
            affordance=aff_t,
            do_sample=False,
            max_new_tokens=400,
        )[0]   # (L_out,)

        # parse output
        ids = generated.tolist()
        sid = self.tokenizer._section_ids
        mo  = self.tokenizer._motion_offset
        mc  = self.tokenizer.motion_codebook_size

        plan_ids, traj_ids, motion_ids = [], [], []
        cur = None
        for t in ids:
            if t == sid["<PLAN>"]:      cur = "plan"
            elif t == sid["<TRAJ>"]:    cur = "traj"
            elif t == sid["<MOTION>"]:  cur = "motion"
            elif t in (sid["</PLAN>"], sid["</TRAJ>"], sid["</MOTION>"]):
                cur = None
            elif cur == "plan":   plan_ids.append(t)
            elif cur == "traj":   traj_ids.append(t)
            elif cur == "motion": motion_ids.append(t)

        plan   = self.tokenizer.decode_plan(plan_ids)
        deltas = self.tokenizer.decode_trajectory(traj_ids)
        m_idx  = [t - mo for t in motion_ids if mo <= t < mo + mc]

        motion_np = None
        if m_idx:
            m_tensor = torch.tensor(m_idx, device=self.device).unsqueeze(0)
            motion_np = self.vqvae.decode(m_tensor)[0].cpu().numpy()  # (T, joint_dim)

        # update cached scene feature for change detection
        self._prev_scene_feat = self.model.scene_encoder(pcd_t, aff_t).detach()

        return {"plan": plan, "deltas": deltas, "motion": motion_np}

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------
    def run(
        self,
        instruction: str,
        initial_scene_pcd: np.ndarray,
        scene_stream,      # callable(t) → (N, 3) point cloud at time t
        affordance_fn=None,  # callable(t, pcd) → (N,) affordance
        goal_pos: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Execute receding horizon loop.

        scene_stream: function(t) → current scene point cloud
        Returns: list of motion segments (each T×joint_dim numpy array)
        """
        all_motions = []
        t = 0
        current_plan = None

        while t < self.max_steps:
            pcd = scene_stream(t)
            aff = affordance_fn(t, pcd) if affordance_fn else None

            # check scene change → re-plan
            pcd_t = torch.from_numpy(pcd).unsqueeze(0).float().to(self.device)
            aff_t = (
                torch.from_numpy(aff).unsqueeze(0).float().to(self.device)
                if aff is not None else None
            )
            scene_feat = self.model.scene_encoder(pcd_t, aff_t).detach()

            if self._scene_changed(scene_feat) or current_plan is None:
                print(f"[t={t}] Scene changed — re-planning...")
                result = self._plan(instruction, pcd, aff)
                current_plan   = result["plan"]
                current_deltas = result["deltas"]
                current_motion = result["motion"]
                self._prev_scene_feat = scene_feat
            else:
                # continue with current plan, only update local trajectory
                current_motion = self._plan(instruction, pcd, aff)["motion"]

            if current_motion is not None:
                exec_len = min(self.execute_horizon, len(current_motion))
                all_motions.append(current_motion[:exec_len])
                t += exec_len
            else:
                t += self.execute_horizon

            # goal check
            if goal_pos is not None and current_motion is not None:
                last_root = decode_motion_to_joints(
                    torch.from_numpy(current_motion[-1:]).unsqueeze(0).to(self.device)
                )[0, 0, 0].cpu().numpy()
                if np.linalg.norm(last_root - goal_pos) < 0.3:
                    print(f"[t={t}] Goal reached.")
                    break

        return all_motions

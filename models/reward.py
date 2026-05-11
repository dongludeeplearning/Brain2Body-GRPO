"""
Geometry-based Reward Functions (Step 5 in diagram).

R_goal        — Task success: distance from final position to goal
R_contact     — Object contact: body surface proximity to target object
R_penetration — Scene penetration: body mesh vs scene mesh overlap
R_foot        — Foot sliding: foot velocity when foot should be planted
R_smooth      — Motion smoothness: joint acceleration
R_plan        — Plan validity: correct format and object references

Total: R = Σ wᵢ Rᵢ
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class GeometryReward(nn.Module):
    def __init__(
        self,
        w_goal: float        =  1.0,
        w_contact: float     =  0.5,
        w_penetration: float = -1.0,
        w_foot: float        = -0.3,
        w_smooth: float      =  0.2,
        w_plan: float        =  0.3,
        goal_threshold: float    = 0.3,    # metres
        contact_threshold: float = 0.1,
        penetration_threshold: float = 0.02,
    ):
        super().__init__()
        self.w_goal        = w_goal
        self.w_contact     = w_contact
        self.w_penetration = w_penetration
        self.w_foot        = w_foot
        self.w_smooth      = w_smooth
        self.w_plan        = w_plan
        self.goal_threshold        = goal_threshold
        self.contact_threshold     = contact_threshold
        self.penetration_threshold = penetration_threshold

    # ------------------------------------------------------------------
    # Individual reward components
    # ------------------------------------------------------------------

    def r_goal(self, final_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        """
        final_pos: (B, 3) predicted last root position
        goal_pos:  (B, 3) target position
        Returns (B,) — 1.0 if within threshold, else distance-based decay.
        """
        dist = (final_pos - goal_pos).norm(dim=-1)              # (B,)
        return torch.exp(-dist / self.goal_threshold)

    def r_contact(
        self,
        body_joints: torch.Tensor,     # (B, T, J, 3) joint positions over time
        obj_surface: torch.Tensor,     # (B, M, 3)  object surface points
    ) -> torch.Tensor:
        """
        Reward for body making contact with target object at any time step.
        Returns (B,) ∈ [0, 1].
        """
        # min distance from any joint to any surface point at each frame
        # body_joints: (B, T, J, 3) → (B, T*J, 3)
        B, T, J, _ = body_joints.shape
        joints_flat = body_joints.reshape(B, T * J, 3)

        # (B, T*J, M) pairwise distances
        diff = joints_flat.unsqueeze(2) - obj_surface.unsqueeze(1)   # (B, T*J, M, 3)
        dists = diff.norm(dim=-1)                                     # (B, T*J, M)
        min_dist = dists.min(dim=-1).values.min(dim=-1).values        # (B,)
        return torch.exp(-min_dist / self.contact_threshold)

    def r_penetration(
        self,
        body_verts: torch.Tensor,   # (B, T, V, 3) body mesh vertices
        scene_sdf: callable,        # function: (B, V, 3) → (B, V) signed distance
    ) -> torch.Tensor:
        """
        Penalty for body vertices inside scene geometry (SDF < 0).
        Returns (B,) — mean penetration depth.
        """
        B, T, V, _ = body_verts.shape
        verts_flat = body_verts.reshape(B * T, V, 3)
        sdf_vals   = scene_sdf(verts_flat).reshape(B, T, V)     # (B, T, V)
        penetration = torch.clamp(-sdf_vals, min=0)             # only negative SDF
        return -penetration.mean(dim=[1, 2])                    # (B,) negative reward

    def r_foot_sliding(
        self,
        foot_joints: torch.Tensor,  # (B, T, 2, 3) left & right foot positions
        foot_contact: torch.Tensor, # (B, T, 2)  binary contact labels
    ) -> torch.Tensor:
        """
        Penalty for foot velocity when contact label is True.
        Returns (B,) negative values.
        """
        vel   = (foot_joints[:, 1:] - foot_joints[:, :-1]).norm(dim=-1)  # (B, T-1, 2)
        mask  = foot_contact[:, :-1].float()                               # (B, T-1, 2)
        slide = (vel * mask).mean(dim=[1, 2])                              # (B,)
        return -slide

    def r_smooth(self, joints: torch.Tensor) -> torch.Tensor:
        """
        joints: (B, T, J, 3)
        Reward for low joint acceleration (second difference).
        Returns (B,).
        """
        vel   = joints[:, 1:] - joints[:, :-1]             # (B, T-1, J, 3)
        accel = (vel[:, 1:] - vel[:, :-1]).norm(dim=-1)    # (B, T-2, J)
        return -accel.mean(dim=[1, 2])                      # (B,)

    def r_plan_validity(
        self,
        decoded_plan: list[list[tuple[str, str]]],   # B × plan_steps
        scene_objects_list: list[list[str]],          # B × obj_names
        valid_actions: tuple = ("APPROACH", "AVOID", "ALIGN", "INTERACT"),
    ) -> torch.Tensor:
        """
        Reward 1.0 if plan tokens are well-formed and reference valid scene objects.
        Returns (B,) float tensor.
        """
        scores = []
        for plan, objs in zip(decoded_plan, scene_objects_list):
            if not plan:
                scores.append(0.0)
                continue
            obj_set = set(o.lower() for o in objs)
            valid = all(
                act in valid_actions and obj.lower() in obj_set
                for act, obj in plan
            )
            scores.append(1.0 if valid else 0.3)
        return torch.tensor(scores, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Aggregate reward
    # ------------------------------------------------------------------
    def forward(
        self,
        final_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        body_joints: torch.Tensor,
        obj_surface: torch.Tensor,
        foot_joints: torch.Tensor,
        foot_contact: torch.Tensor,
        decoded_plan: list,
        scene_objects_list: list,
        body_verts: torch.Tensor | None = None,
        scene_sdf: callable | None = None,
    ) -> dict:
        device = final_pos.device

        R_goal    = self.r_goal(final_pos, goal_pos)
        R_contact = self.r_contact(body_joints, obj_surface)
        R_foot    = self.r_foot_sliding(foot_joints, foot_contact)
        R_smooth  = self.r_smooth(body_joints)
        R_plan    = self.r_plan_validity(decoded_plan, scene_objects_list).to(device)

        total = (
            self.w_goal        * R_goal
            + self.w_contact   * R_contact
            + self.w_foot      * R_foot
            + self.w_smooth    * R_smooth
            + self.w_plan      * R_plan
        )

        if body_verts is not None and scene_sdf is not None:
            R_pen  = self.r_penetration(body_verts, scene_sdf)
            total += self.w_penetration * R_pen
        else:
            R_pen = torch.zeros_like(total)

        return {
            "total":       total,       # (B,)
            "r_goal":      R_goal,
            "r_contact":   R_contact,
            "r_penetration": R_pen,
            "r_foot":      R_foot,
            "r_smooth":    R_smooth,
            "r_plan":      R_plan,
        }

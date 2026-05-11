"""
Rule-based Pseudo Plan Extraction (Step 2 in diagram).

Given a scene, instruction, and ground-truth trajectory,
automatically generates:
  - Object-centric plan: [(action, object), ...]
  - Relative trajectory ΔPOS: (T, 3) root position increments
  - Motion token indices: encoded by VQ-VAE
"""

from __future__ import annotations
import re
import numpy as np


# Simple keyword → plan action mapping
_ACTION_KEYWORDS = {
    "APPROACH": ["walk to", "go to", "move to", "approach", "reach"],
    "AVOID":    ["avoid", "go around", "don't hit", "bypass"],
    "ALIGN":    ["face", "align", "turn toward", "face the"],
    "INTERACT": ["sit", "sit down", "pick up", "grab", "touch", "use", "open", "lie"],
}


def extract_plan_from_instruction(
    instruction: str,
    scene_objects: list[str],
) -> list[tuple[str, str]]:
    """
    Heuristic rule-based plan extraction.
    Returns a list of (action, object) pairs ordered by likely execution sequence.

    For a full system, this would use an LLM or structured scene parser.
    Here we use keyword matching as pseudo-supervision.
    """
    instruction_lower = instruction.lower()
    plan = []
    seen = set()

    # find target object mentioned in instruction
    target_obj = None
    for obj in scene_objects:
        if obj.lower() in instruction_lower:
            target_obj = obj
            break

    if target_obj is None and scene_objects:
        target_obj = scene_objects[0]   # fallback

    # parse action keywords
    for action, keywords in _ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in instruction_lower:
                key = (action, target_obj)
                if key not in seen:
                    plan.append(key)
                    seen.add(key)
                break

    # always prepend APPROACH if not present and there is a target
    if target_obj and ("APPROACH", target_obj) not in seen:
        plan.insert(0, ("APPROACH", target_obj))
        seen.add(("APPROACH", target_obj))

    # if navigating through obstacles, add AVOID for other objects
    for obj in scene_objects:
        if obj != target_obj and ("AVOID", obj) not in seen:
            # simple heuristic: add AVOID for objects between start and target
            # in a real system, this would use path planning
            plan.insert(1, ("AVOID", obj))
            seen.add(("AVOID", obj))
            break   # one avoid per plan for now

    return plan


def extract_trajectory_deltas(
    root_positions: np.ndarray,
) -> np.ndarray:
    """
    Compute relative root position increments ΔPOS from absolute positions.

    root_positions: (T, 3) absolute XYZ of pelvis
    Returns: (T-1, 3) increments
    """
    return np.diff(root_positions, axis=0).astype(np.float32)


def build_pseudo_supervision(
    instruction: str,
    scene_objects: list[str],
    root_positions: np.ndarray,
    vqvae,                          # MotionVQVAE, called with .encode()
    motion_seq: "torch.Tensor",     # (1, T, joint_dim)
) -> dict:
    """
    Full pseudo-supervision builder for one training sample.

    Returns:
        plan:     list of (action, object) tuples
        deltas:   (T-1, 3) trajectory increments
        motion_indices: (N,) VQ-VAE token indices
    """
    import torch
    plan = extract_plan_from_instruction(instruction, scene_objects)
    deltas = extract_trajectory_deltas(root_positions)

    with torch.no_grad():
        motion_indices = vqvae.encode(motion_seq).squeeze(0)  # (N,)

    return {"plan": plan, "deltas": deltas, "motion_indices": motion_indices}

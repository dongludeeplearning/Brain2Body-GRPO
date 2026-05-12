# Brain2Body-GRPO

**Program-Driven Human-Scene Interaction via Geometry-Grounded Preference Optimization**

Given a natural language instruction and a 3D scene, Brain2Body-GRPO generates physically plausible human motion that navigates and interacts with scene objects — including dynamically changing environments.

---

## Method Overview

The pipeline consists of four stages:

```
Scene + Instruction
       │
       ▼
┌─────────────────────────────┐
│  Stage 1 · Pseudo Labeling  │  Rule-based plan extraction:
│                             │  <APPROACH:sofa> <AVOID:table>
│  ΔP OS trajectory           │  Relative root position Δp₁..ΔpT
│  Motion tokens (VQ-VAE)     │  Discrete motion codes m₁..mN
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2 · SFT Training     │  T5 decoder learns to autoregressively
│                             │  generate a single unified sequence:
│  <PLAN> ... <TRAJ> ... <MOTION> ...
│                             │
│  Loss = L_plan + λ₁L_traj + λ₂L_motion
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3 · GRPO Fine-tuning │  Sample K outputs for each input.
│                             │  Decode motion → compute geometry rewards.
│  Aᵢ = (Rᵢ − mean(R))       │  Group-relative advantage normalization.
│       / (std(R) + ε)        │
│                             │
│  L_GRPO = −1/K Σ Aᵢ log π  │
│           + β KL(π ‖ π_ref) │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 4 · Execution        │  Receding horizon loop:
│                             │  Observe → Plan → Execute K frames
│                             │  Detect scene change → Re-plan
└─────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Motion VQ-VAE** | Encodes SMPL-X motion sequences into discrete tokens via 1D temporal conv encoder + vector quantization codebook (size 512) |
| **Multi-modal Tokenizer** | Extends T5 vocabulary with plan tokens `<APPROACH:obj>`, trajectory tokens `<TRAJ_dx_i>`, and motion tokens `<MOTION_i>` |
| **Brain2Body AR Model** | T5 decoder conditioned on scene point cloud (injected as soft prompt tokens). Generates `<PLAN>`, `<TRAJ>`, `<MOTION>` as a single autoregressive sequence |
| **Scene Encoder** | PointNet-style encoder for RGB point cloud + affordance map (L₂ distance from scene points to body joints) |
| **Geometry Rewards** | Six reward signals: R_goal (task success), R_contact (object contact), R_penetration (no body-scene overlap), R_foot (no foot sliding), R_smooth (motion smoothness), R_plan (plan validity) |
| **Receding Horizon** | At each step, observe scene → generate plan+motion → execute → check scene change → re-plan if needed |

---

## Installation

**Requirements**: Python 3.10+, CUDA 11.8+

```bash
git clone https://github.com/dongludeeplearning/cot-hsi.git
cd cot-hsi

conda create -n cot-hsi python=3.10 -y
conda activate cot-hsi

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Data Preparation

Organize your dataset in the following format:

```
data/
└── lingo/
    ├── train/
    │   ├── sample_000/
    │   │   ├── instruction.txt       # "Walk to the sofa and sit down."
    │   │   ├── scene_pcd.npy         # (N, 3) scene point cloud
    │   │   ├── scene_objects.txt     # one object name per line
    │   │   ├── root_positions.npy    # (T, 3) pelvis trajectory
    │   │   └── motion.npy            # (T, joint_dim) SMPL-X features
    │   └── sample_001/ ...
    └── val/ ...
```

Compatible datasets: [LINGO](https://github.com/LingomotionLab/LINGO), [TRUMANS](https://github.com/lijiaman/trumans).

---

## Training

### Step 1 — Pre-train the Motion VQ-VAE

Before SFT, train the motion tokenizer on your motion data:

```bash
python scripts/train_vqvae.py --config configs/vqvae_config.yaml
```

The checkpoint is saved to `checkpoints/vqvae/best.pt`.

### Step 2 — SFT Training (AR Policy)

```bash
python scripts/train_sft.py --config configs/sft_config.yaml
```

Key config options (`configs/sft_config.yaml`):

```yaml
model:
  t5_model_name: "google/flan-t5-base"  # or flan-t5-large
  motion_codebook_size: 512
  traj_bins: 64

training:
  batch_size: 16
  num_epochs: 100
  lr: 1e-4
  lambda_traj: 0.5        # weight for trajectory loss
  lambda_motion: 1.0      # weight for motion loss
  output_dir: "checkpoints/sft"
```

Training logs are sent to [Weights & Biases](https://wandb.ai). Set `WANDB_API_KEY` or run `wandb login` beforehand.

### Step 3 — GRPO Fine-tuning

Requires a completed SFT checkpoint:

```bash
python scripts/train_grpo.py --config configs/grpo_config.yaml
```

Key config options (`configs/grpo_config.yaml`):

```yaml
model:
  sft_ckpt: "checkpoints/sft/best.pt"

grpo:
  K: 8                # samples per input
  beta: 0.01          # KL penalty weight
  temperature: 1.0

reward:
  w_goal: 1.0
  w_contact: 0.5
  w_penetration: -1.0
  w_foot: -0.3
  w_smooth: 0.2
  w_plan: 0.3
```

---

## Inference

### Single-scene generation

```python
from models import Brain2BodyModel, MotionVQVAE
from data import Brain2BodyTokenizer
from inference import RecedingHorizonExecutor
import numpy as np
import torch

device = "cuda"

vqvae     = MotionVQVAE().to(device)
tokenizer = Brain2BodyTokenizer()
model     = Brain2BodyModel(vocab_size=tokenizer.vocab_size).to(device)

# load checkpoints
vqvae.load_state_dict(torch.load("checkpoints/vqvae/best.pt"))
model.load_state_dict(torch.load("checkpoints/grpo/grpo_final.pt")["model_state_dict"])

executor = RecedingHorizonExecutor(model, vqvae, tokenizer, device=device)

scene_pcd = np.load("data/lingo/val/sample_000/scene_pcd.npy")

motions = executor.run(
    instruction="Walk to the sofa and sit down.",
    initial_scene_pcd=scene_pcd,
    scene_stream=lambda t: scene_pcd,   # static scene
    goal_pos=np.array([2.0, 0.0, 1.5]),
)
```

### Dynamic scene (receding horizon)

```python
def dynamic_scene(t):
    # return updated point cloud at timestep t
    pcd = np.load(f"data/dynamic/frame_{t:04d}.npy")
    return pcd

motions = executor.run(
    instruction="Avoid the moving table and reach the door.",
    initial_scene_pcd=dynamic_scene(0),
    scene_stream=dynamic_scene,
    goal_pos=np.array([5.0, 0.0, 0.0]),
)
```

---

## Evaluation

### Quantitative metrics

```bash
python scripts/evaluate.py \
    --config configs/grpo_config.yaml \
    --split val \
    --ckpt checkpoints/grpo/grpo_final.pt
```

Reported metrics:

| Metric | Description |
|--------|-------------|
| **FID** | Fréchet Inception Distance — motion quality |
| **R-Precision** | Text-motion retrieval accuracy (Top-1/3) |
| **MM-Dist** | Multi-modal distance between text and motion |
| **Goal Error (m)** | L₂ distance from final position to goal |
| **Traj. Similarity (%)** | Trajectory overlap with ground truth |
| **Penetration Rate (%)** | Fraction of frames with body-scene penetration |
| **Foot Sliding** | Mean foot velocity during contact |

### Baselines

| Method | Scene-aware | Dynamic | LLM backbone |
|--------|-------------|---------|-------------|
| MotionGPT | ❌ | ❌ | ✅ |
| Dyn-HSI | ✅ | ✅ | ❌ |
| HSI-GPT | ✅ | ❌ | ✅ |
| **Brain2Body-GRPO** | ✅ | ✅ | ✅ |

---

## Project Structure

```
cot-hsi/
├── configs/
│   ├── sft_config.yaml         # SFT hyperparameters
│   └── grpo_config.yaml        # GRPO hyperparameters
├── data/
│   ├── tokenizer.py            # Multi-modal tokenizer (plan/traj/motion)
│   ├── pseudo_plan.py          # Rule-based pseudo supervision extraction
│   └── dataset.py              # HSI dataset loader
├── models/
│   ├── motion_vqvae.py         # Motion VQ-VAE encoder/decoder
│   ├── scene_encoder.py        # PointNet scene + affordance encoder
│   ├── brain2body.py           # T5-based AR policy model
│   └── reward.py               # Geometry-based reward functions
├── training/
│   ├── sft_trainer.py          # SFT training loop
│   └── grpo_trainer.py         # GRPO fine-tuning loop
├── inference/
│   └── receding_horizon.py     # Dynamic environment execution
├── utils/
│   ├── smplx_utils.py          # SMPL-X joint utilities
│   └── geometry.py             # SDF, occupancy grid, scene delta
└── scripts/
    ├── train_sft.py            # SFT entry point
    └── train_grpo.py           # GRPO entry point
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{brain2body_grpo,
  title   = {Brain2Body-GRPO: Program-Driven Human-Scene Interaction
             via Geometry-Grounded Preference Optimization},
  year    = {2026},
}
```

## Acknowledgements

This project builds on:
- [MotionGPT](https://github.com/OpenMotionLab/MotionGPT) — motion tokenization and T5 AR framework
- [Motion-R1](https://github.com/GigaAI-Research/Motion-R1) — GRPO training for motion generation
- [Dyn-HSI](https://arxiv.org/abs/2601.19484) — dynamic scene interaction and hierarchical memory
- [HSI-GPT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_HSI-GPT_A_General-Purpose_Large_Scene-Motion-Language_Model_for_Human_Scene_Interaction_CVPR_2025_paper.html) — scene-motion-language unified framework

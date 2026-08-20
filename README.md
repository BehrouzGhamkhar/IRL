# VR-Based Human Feedback for Interactive Reinforcement Learning

**R&D Project - Master of Autonomous Systems, Hochschule Bonn-Rhein-Sieg**  
**Author:** Behrouz Ghamkhar  
**Supervisors:** Prof. Dr. Teena Chakkalayil Hassan and Michal Stolarz  
**Submitted:** May 2026

---

## Overview

This project trains a Humanoid robot to perform contextually appropriate social greeting behaviours using Interactive Reinforcement Learning (IRL). The agent learns to select the correct action - **DoNothing**, **Talk**, **Look**, **Wave**, or **HandShake** — from a 12-dimensional body-signal observation space, without being given explicit task labels.

Three training conditions are implemented:

| Condition | Algorithm | Feedback source | Status                    |
|---|---|---|---------------------------|
| Autonomous PPO baseline | PPO (~150k steps) | Hand-crafted reward function | Completed                 |
| COACH + Keyboard | COACH (164 steps) | Human teacher via arrow keys | Completed                 |
| COACH + VR | COACH | Human teacher via Meta Quest 3 | Implemented, not executed |

The two COACH conditions independently load the same PPO checkpoint (config3, seed 42) and fine-tune without depending on each other.
 
> **VR note:** The VR COACH interface (controller buttons, head gestures, voice commands) is fully implemented in `VRMultimodalReward.cs` and `CommunicationManager.cs`. A physical body incompatibility between the NPC's body size and the VR person controller's size prevented finetuning a learned model.

---

## Key Results

| Metric | Autonomous PPO | COACH + Keyboard |
|---|---|---|
| Test accuracy (2,000 steps) | 74.14% | 72.37% |
| Action entropy (bits) | 1.39 | 1.79 |
| HandShake selections | 20 | 137 |
| Look selections | 1158 | 820 |
| Wave selections | 220 | 424 |
| HandShake accuracy | 85.0% | 81.8% |
| Talk selections | 0 | 0 |
| Feedback steps used | — | 164 |

164 COACH steps redistributed probability mass from Look toward HandShake (×6.85) and Wave (×1.9), without degrading per-action correctness.

---

## Hardware & Software Stack

| Component | Details                      |
|---|------------------------------|
| Unity | **6000.0.59f2**              |
| Unity ML-Agents (C# package) | 4.0.0                        |
| ML-Agents Python package | 1.1.0                        |
| Python | 3.10.12                      |
| PyTorch | 2.0.1 (CUDA 11.8)            |
| VR Headset | HTC Vive port3               |
| CPU | Intel Core i7-12700K         |
| GPU | NVIDIA RTX 3080 (10 GB VRAM) |
| RAM | 32 GB DDR4-3200              |
| OS | Windows 11               |

---

## System Architecture

```
┌───────────────────────────────────────┐                ┌──────────────────────────────────┐
│          Unity Environment            │  ──────────>  │         Python Trainer           │
│                                       │ observations  │                                  │
│  HumanAgent                           │ (12-dim vec)  │  config.yaml                     │
│  (NPCController / VRPersonController) │               │  train.py                        │
│            │                          │  <──────────  │    ALGORITHMS registry           │
│            v                          │  action (0–4) │    { 'ppo': PPOAgent,            │
│  CommunicationManager                 │               │      'coach': COACHAgent }       │
│  (episode coord, obs, routing)        │  ──────────>  │    SimplePPO network             │
│            │                          │  reward       │    (fc1->fc2->actor+critic)      │
│            v                          │               │    AccuracyTracker               │
│  IRewardProvider                      │  · · · · >    │    checkpoint .pt                │
│  ┌─────────────────────────────────┐  │  npc_seed     │    (EnvironmentParameters)       │
│  │ AutonomousRewardProvider        │  │  side-channel │                                  │
│  │   task–action match             │  │               └──────────────────────────────────┘
│  │   +1.0 / -0.2 / -0.0001         │  │
│  ├─────────────────────────────────┤  │
│  │ KeyboardRewardProvider          │ <── ↑ +1  ↓ -1
│  │   arrow keys                    │  │
│  ├─────────────────────────────────┤  │
│  │ VRMultimodalReward              │ <── button ±1.0
│  │   priority-based fusion         │  │  voice  ±0.75
│  │   cooldown gating               │  │  nod/shake ±0.5
│  └─────────────────────────────────┘  │
│            │                          │
│            v                          │
│  PepperAgent                          │
│  (CollectObservations, AddReward)     │
│  |r| > 0.05 -> COACH update           │
│            │                          │
│            v                          │
│  PepperController                     │
│  (ExecuteAction, animations)          │
│                                       │
│  SeedReceiver                         │
│  (npc_seed side-channel)              │
└───────────────────────────────────────┘
```

**Training mode is controlled by a single flag in `train.py`:**
- `VR_MODE = False` -> PPO sweep: discovers all YAMLs in `configs/`, runs in parallel via `multiprocessing.Pool`
- `VR_MODE = True` -> COACH/VR: reads from `configs/Finetune/`, runs sequentially, waits for Unity Editor

---

## Repository Structure

```
project/
│
├── Python/training/agents/
│   ├── train.py                    # Main loop; VR_MODE flag; RESUME_FROM path
│   ├── utils.py                    # load_config, find_config_files, AccuracyTracker,
│   │                               #   save/load rewards and accuracy CSVs
│   ├── analyse_runs.py             # Sweep analysis, convergence detection, plots, CSV export
│   │
│   └── algorithms/
│       ├── ppo.py                      # PPOAgent: act, store_transition, update, save, load
│       └── coach.py                    # COACHAgent: immediate per-event policy gradient update
│    
├── Python/configs/
│   ├── config1.yaml                # PPO baseline (lr=0.0003, hidden=128, rollout=256)
│   ├── config2.yaml                # lr = 0.0001
│   ├── config3.yaml                # lr = 0.001  ← best; used as COACH checkpoint source
│   ├── config4.yaml                # hidden = 256
│   ├── config5.yaml                # hidden = 64
│   ├── config6.yaml                # entropy high (start=0.3, end=0.05)
│   ├── config7.yaml                # entropy low  (start=0.05, end=0.005)
│   ├── config8.yaml                # rollout = 512
│   ├── config9.yaml                # rollout = 128
│   ├── config10.yaml               # epsilon = 0.1
│   └── Finetune/
│       └── coach_config.yaml       # COACH fine-tuning (keyboard + VR)
│
├── Test/IntegrationTests/
│   ├── test_compare_models.py      # Cross-condition evaluation (PPO vs COACH, 2000 steps)
│   └── test_pet_task_accuracy.py   # Per-action accuracy evaluation
│
├── Unity/Assets/Scripts (Some of the C# scripts)
│   ├── PepperAgent.cs              # ML-Agents Agent subclass: observations, actions, rewards
│   ├── PepperController.cs         # Action execution, animations, AgentAction enum
│   ├── CommunicationManager.cs     # Central coordinator: episode logic, obs, reward routing
│   ├── NPCController.cs            # NPC task selection (uniform random), NavMesh navigation
│   ├── VRPersonController.cs       # VR participant controller (replaces NPC in VR mode)
│   ├── IHumanAgent.cs              # Interface implemented by NPC and VR participant
│   ├── IRewardProvider.cs          # Interface: OnReward event -> scalar float
│   ├── AutonomousRewardProvider.cs # Task–action reward matching
│   ├── KeyboardRewardProvider.cs   # Arrow key feedback: up=+1, down=-1
│   ├── VRMultimodalReward.cs       # Priority-based fusion: button/voice/head gestures
│   ├── SeedReceiver.cs             # Reads npc_seed from EnvironmentParameters side-channel
│   ├── DataDisplay.cs              # In-scene training metrics HUD
│   └── Feedbacklogger.cs           # Logs feedback events to file
│
├── Python/training/agents/training_logs/
│    └── analysis/
│        ├── sweep_results.csv
│        ├── definitive_summary.csv
│        ├── seed_variance.csv
│        ├── irl_comparison.csv
│        ├── per_task_accuracy.csv       # COACH run action-level accuracy
│        ├── per_task_accuracy_baseline.csv
│        └── plots/
│            ├── reward_curve_config4.png
│            ├── accuracy_curve_config4.png
│            └── all_configs_comparison.png
│            
└── Data/Builds/
    ├── BaselineHeadless/  
    ├── Baseline/
    ├── KeyboardCOACH/ 
    └── VRCOACH/       
```

---

## Observation Space (12 dimensions)

| Idx | Feature | Range | Description |
|---|---|---|---|
| 0–4 | PepperState | one-hot (5) | Idle / Looking / Waving / Handshaking / PerformingAction |
| 5 | DistanceToHuman | [0, 1] | Normalised by 10 m; 0 = at Pepper, 1 = far away |
| 6 | HandshakeInProgress | {0, 1} | 1 when handshake sequence is active |
| 7 | CanHandshake | {0, 1} | 1 when 3 s cooldown has elapsed |
| 8 | WristHeight | [0, 1] | (y_wrist − y_floor) / h_person + noise |
| 9 | WristToCoreDistance | [0, 1] | ‖p_wrist − p_hip‖ / h_person + noise |
| 10 | BodyOrientation | [−1, 1] | Facing Pepper (+1) vs away (−1) |
| 11 | GazeDirection | [−1, 1] | Gaze toward Pepper (+1) vs away (−1) |

Features 8 and 9 are **body-relative**: both are divided by `h_person = max(0.1, head_y − floor_y)`, making them scale-invariant across the NPC rig and VR participants of differing heights. Hip position is approximated as 55% of standing height from the head transform (no separate hip bone needed). Additive noise η ~ U(−0.05, +0.05) is applied each step; observations are sampled after a per-task delay (default 1.0 s ± 0.2 s jitter) to capture mid-gesture state.

---

## Action Space (5 discrete actions)

| ID | Name | Correct context |
|---|---|---|
| 0 | DoNothing | NPC > 5 m away |
| 1 | Talk | Task ID 6 (talk request) |
| 2 | Look | NPC ≤ 5 m nearby |
| 3 | Wave | Task ID 7 (wave request) |
| 4 | HandShake | Task ID 2 (handshake request) |

---

## Reward Function (Autonomous baseline)

| Condition | Reward |
|---|---|
| Correct action (most tasks) | +1.0 |
| Wrong action | −0.2 |
| Correct DoNothing when far | +0.001 |
| Per step (always) | −0.0001 |

A **reward gate** in `CommunicationManager` ensures only the first action per task event receives a reward, preventing spamming. In IRL modes the autonomous reward function is bypassed entirely — `EvaluateReward()` returns immediately and all reward comes from the active `IRewardProvider`.

**Accuracy is derived purely from reward in `AccuracyTracker`:**
- `reward > 0.0` -> correct
- `reward < -0.05` -> wrong
- anything between (living penalty) -> ignored

---

## Hyperparameter Sweep

10 configurations × 3 seeds = 30 runs, each varying one group relative to the baseline. Configs run in parallel via `multiprocessing.Pool` with up to `MAX_PARALLEL = 10` simultaneous Unity instances.

| Config | Change | Final Acc. | Steps to 80% | Max Avg Reward |
|---|---|---|---|---|
| config1 | Baseline | 63.0% | 12k | 4.30 |
| config2 | lr = 0.0001 | 50.9% | 33k | 3.28 |
| **config3** | **lr = 0.001** | **65.2%** | **7k** | **4.69** |
| config4 | hidden = 256 | 64.9% | 8k | 4.52 |
| config5 | hidden = 64 | 57.1% | 23k | 3.97 |
| config6 | entropy high | 54.1% | 35k | 3.54 |
| config7 | entropy low | 63.5% | 12k | 4.40 |
| config8 | rollout = 512 | 57.7% | 19k | 4.08 |
| config9 | rollout = 128 | 64.4% | 10k | 4.44 |
| config10 | epsilon = 0.1 | 60.2% | 14k | 4.34 |

**config3** (lr=0.001, hidden=128) selected for COACH - best accuracy, fastest convergence, compact network reducing overfitting risk during the short fine-tuning phase.

---

## Running the Project

### Prerequisites

```bash
pip install torch==2.0.1 mlagents==1.1.0 numpy pyyaml matplotlib pandas
```

Unity project requires:
- Unity **6000.0.59f2**
- ML-Agents C# package **4.0.0** (`com.unity.ml-agents`)
- OpenXR plugin (VR condition)

---

### Stage 1 — PPO Hyperparameter Sweep

In `train.py`, set:
```python
VR_MODE = False
RESUME_FROM = None
```

Then run:
```bash
cd Python/training/agents
python train.py
# Discovers all YAMLs in configs/, runs up to MAX_PARALLEL=10 in parallel
# Each run saves to: training_logs/run_seed{N}_{config}_{fresh|resumed}_{timestamp}/
#   model_final.pt
#   rewards_final.csv
#   accuracy_final.csv
```

Analyse results:
```bash
python analyse_runs.py
# Outputs: training_logs/analysis/sweep_results.csv
#          training_logs/analysis/definitive_summary.csv
#          training_logs/analysis/plots/
```

---

### Stage 2a — COACH Keyboard Fine-Tuning

In `train.py`, set:
```python
VR_MODE = True   # reads from configs/Finetune/coach_config.yaml
RESUME_FROM = "training_logs/run_seed42_config3 (2)_fresh_.../model_final.pt"
```

Then:
1. Open Unity project, load the **Keyboard scene**  OR use the KeyboardCOACH Build in the Data/Builds folder
2. Press Play in the Unity Editor (if in Unity)
3. Run:
```bash
python train.py
# Script prints: "Waiting for Unity Editor - Press play now ..."
# Teacher uses ↑ (correct) and ↓ (wrong) arrow keys during live session
# Updates fire immediately when |reward| > 0.05
```

---

### Stage 2b — COACH VR Fine-Tuning

Same as Stage 2a but:
1. Open Unity project, load the **VR scene** (replaces NPC with VRPersonController) OR use the VRCOACH Build in the Data/Builds folder
2. Connect VR Headset
3. Same `train.py` command — teacher uses controller buttons, head nods/shakes, voice commands

---

### Evaluation

```bash
# Compare PPO baseline vs COACH checkpoint (2000 steps each)
python test_compare_models.py

# Per-action accuracy breakdown
python test_pet_task_accuracy.py
```

---

## COACH Algorithm

COACH performs an **immediate policy gradient update** after every non-trivial feedback event (`|r| > 0.05`), as implemented in `algorithms/coach.py`:

```
θ ← θ − α · ∇_θ [ −f_t · log π_θ(a_t | s_t) − c2 · H[π_θ(· | s_t)] ]
```

- `f_t` — scalar human feedback (+1 or −1)
- Positive reinforces the chosen action; negative suppresses it and redistributes mass
- Entropy term `c2 · H[π]` keeps the policy explorative during fine-tuning
- No rollout buffer, value function, or advantage estimation needed - one `(s, a, f)` triple per update
- Updates only fire when `|r| > 0.05` (threshold from `utils.REWARD_NEGATIVE_THRESHOLD`) so the −0.0001 per-step penalty is silently ignored

`COACHAgent` is fully interface-compatible with `train.py`'s PPO loop:

| `train.py` call | `COACHAgent` behaviour                                |
|---|-------------------------------------------------------|
| `act(obs)` | Returns `(action, 0.0, log_prob)` - fake critic value |
| `get_value(obs)` | Returns `0.0` - no critic needed                      |
| `store_transition(...)` | Fires `_coach_update()` immediately if `              |r| > 0.05` |
| `update(next_value)` | No-op - PPO rollout call, harmless                    |
| `save(path)` | Same `.pt` format as PPO, adds `coach_updates` key    |
| `load(path)` | Loads PPO weights, resets optimiser to COACH lr       |

Switching from PPO to COACH requires exactly **two changes**: set `algorithm: coach` in the config and point `RESUME_FROM` to the PPO checkpoint.

---

## VR Multimodal Feedback Fusion

`VRMultimodalReward.cs` implements priority-based fusion of three simultaneous modalities:

| Modality | Reward magnitude | Priority |
|---|---|---|
| Controller button | ±1.0 | Highest |
| Voice command ("good"/"bad") | ±0.75 | Medium |
| Head gesture (nod / shake) | +0.5 / −0.5 | Lowest |

When multiple signals arrive simultaneously the highest-priority modality wins. A cooldown window prevents the same gesture from firing multiple times. Only one signal is emitted per decision step.

---

## Key Design Decisions

**Why COACH instead of continued PPO for fine-tuning?**
PPO requires 256–2048 steps per gradient update. With ~164 feedback steps total, at most one PPO update is possible — far too noisy to shift a converged policy. COACH's per-event mechanism works with a single `(s, a, f)` triple and produces a useful gradient immediately.

**Why body-signal observations instead of task IDs?**
Explicit task labels would make the system trivial and non-generalisable to a real Pepper deployment. Body-signal features allow the agent to infer intent from posture - the same information a physical robot would have access to.

**Why body-relative normalisation?**
Fixed constants (e.g., divide wrist height by 2.5 m globally) break when a live human replaces the NPC in VR, because absolute wrist heights vary with each person's height. Dividing by `h_person = head_y − floor_y` makes features scale-invariant across participants without needing a separate hip bone in the VR rig.

**Why is the VR condition not executed?**
The NPC's capsule collider and the VR person controller's rigid-body configuration are physically incompatible — this causes collision artefacts that prevent stable NPC–participant interaction during live sessions. Fix: switch the NPC to a trigger collider or adjust the VR controller physics layer mask.

---

## Known Issues & Limitations

- **VR physical incompatibility:** Resolve collider conflict to enable live VR COACH sessions
- **Talk action still suppressed:** 164 steps shifted Look->HandShake/Wave but did not surface Talk; a longer session with ~50% Talk-biased task distribution is expected to complete the correction
- **Single teacher:** Inter-teacher variability in timing and consistency not assessed
- **Simulation only:** Transfer to a physical Pepper with RGB-D pose estimation not evaluated
- **Stale placeholder in methodology:** Section IV-H mentions "76.0% accuracy / 7.2% Talk decisions" — these are draft numbers that should read 74.14% / 0% per the actual results

---

## AI Assistance Disclosure

**DeepSeek** — initial code scaffolding, YAML templates, LaTeX table formatting, equation formatting.  
**Claude (Anthropic)** — COACH algorithm design, body-relative normalisation derivation, training pipeline architecture, code review.  
All AI-generated content was reviewed, tested, and modified by the author. All training runs and experimental results are entirely the author's own work.
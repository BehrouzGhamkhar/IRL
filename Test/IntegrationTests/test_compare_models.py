"""
Runs two models (e.g. Autonomous RL and IRL) sequentially against the Unity
environment and computes the metrics needed for the IRL comparison table.

Metrics computed per model:
  - Final task accuracy (%)         : correct / (correct + wrong) across all steps
  - Best episode reward              : max total reward seen in a single episode
  - Avg episode reward               : mean over all episodes
  - Action distribution entropy      : H = -sum(p * log2(p))  [bits]
  - HandShake task accuracy (%)      : per-action accuracy for action 4
  - Talk task accuracy (%)           : per-action accuracy for action 1
"""

import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

MODELS = {
    "Autonomous RL": {
        "path":        "../../Python/training/agents/training_logs/run_seed42_config3 (2)_fresh_20260529_202921/model_final.pt",
        "hidden_size": 128,
    },
    "IRL": {
        "path":        "../../Python/training/agents/training_logs/run_seed41_coach_config_resumed_20260530_170944/model_final.pt",
        "hidden_size": 128,
    },
}

UNITY_BUILD = "../../Data/Builds/BaselineHeadless/InteractiveRL.exe"
MAX_STEPS = 2000
TIME_SCALE = 100.0
OUTPUT_CSV = "../../Python/training/agents/training_logs/analysis/irl_comparison.csv"

# Must match training setup
REWARD_CORRECT_THRESHOLD = 0.0  # reward > this  -> correct
REWARD_WRONG_THRESHOLD = -0.05  # reward < this  -> wrong  (step penalty -0.0001 is ignored)


# Model  (identical architecture to training)
class SimplePPO(nn.Module):
    def __init__(self, obs_size=12, act_size=5, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, act_size)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.actor(x)


def load_model(model_path, hidden_size, device):
    model = SimplePPO(hidden_size=hidden_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"  Model loaded: {model_path}  (hidden={hidden_size})")
    return model


def pick_action(model, obs, device):
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_tensor)
        action = torch.argmax(logits, dim=-1).item()
    return action


# Evaluation loop  (runs one model, returns stats)
def evaluate_model(model_path, hidden_size, device):

    """Connect to Unity, run the model for MAX_STEPS, return a stats dict."""

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(
        time_scale=TIME_SCALE,
        target_frame_rate=60,
        capture_frame_rate=60,
    )
    env = UnityEnvironment(
        file_name=UNITY_BUILD,
        no_graphics=True,
        side_channels=[channel],
    )
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    model = load_model(model_path, hidden_size, device)

    correct_counts = defaultdict(int)  # action_id -> correct count
    wrong_counts = defaultdict(int)  # action_id -> wrong count
    action_counts = defaultdict(int)  # action_id -> total times chosen
    episode_rewards = []  # total reward per episode

    decision_steps, _ = env.get_steps(behavior_name)
    obs = decision_steps.obs[0][0]
    last_action = None
    episode_reward = 0.0
    total_steps = 0
    episodes = 0

    print(f"\n  Running {MAX_STEPS:,} steps...")

    try:
        while total_steps < MAX_STEPS:

            action = pick_action(model, obs, device)
            action_counts[action] += 1

            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.array([[action]]))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                reward = terminal_steps.reward[0]
                _record(last_action, reward, correct_counts, wrong_counts)
                episode_reward += reward
                episode_rewards.append(episode_reward)

                episodes += 1
                episode_reward = 0.0
                last_action = None

                env.reset()
                decision_steps, _ = env.get_steps(behavior_name)
                obs = decision_steps.obs[0][0]

            else:
                reward = decision_steps.reward[0]
                _record(last_action, reward, correct_counts, wrong_counts)
                episode_reward += reward
                obs = decision_steps.obs[0][0]

            last_action = action
            total_steps += 1

            if total_steps % 2000 == 0:
                print(f"    Step {total_steps:,} / {MAX_STEPS:,}  |  Episodes: {episodes}")

    except KeyboardInterrupt:
        print("  Stopped early.")
    finally:
        env.close()

    return _compute_stats(correct_counts, wrong_counts, action_counts, episode_rewards)


def _record(action, reward, correct_counts, wrong_counts):
    if action is None:
        return
    if reward > REWARD_CORRECT_THRESHOLD:
        correct_counts[action] += 1
    elif reward < REWARD_WRONG_THRESHOLD:
        wrong_counts[action] += 1


def _compute_stats(correct_counts, wrong_counts, action_counts, episode_rewards):
    """Derive all table metrics from the raw counters."""

    # Overall accuracy
    total_correct = sum(correct_counts.values())
    total_wrong = sum(wrong_counts.values())
    total_decided = total_correct + total_wrong
    overall_acc = (total_correct / total_decided * 100) if total_decided > 0 else None

    # Episode reward stats
    best_reward = max(episode_rewards) if episode_rewards else None
    avg_reward = np.mean(episode_rewards) if episode_rewards else None

    # Action distribution entropy (bits), formula is H = -sum(p * log2(p))
    total_actions = sum(action_counts.values())
    entropy = 0.0
    if total_actions > 0:
        for count in action_counts.values():
            p = count / total_actions
            if p > 0:
                entropy -= p * math.log2(p)

    # Per-action accuracy helpers
    def acc(action_id):
        c = correct_counts[action_id]
        w = wrong_counts[action_id]
        return (c / (c + w) * 100) if (c + w) > 0 else None

    return {
        "Final task accuracy (%)": overall_acc,
        "Best episode reward": best_reward,
        "Avg reward": avg_reward,
        "Action distribution entropy (bits)": entropy,
        "HandShake occurrences (n)": correct_counts[4] + wrong_counts[4],
        "HandShake task accuracy (%)": acc(4),
        "Talk occurrences (n)": correct_counts[1] + wrong_counts[1],
        "Talk task accuracy (%)": acc(1)
    }


# Print + save comparison table
def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_and_save(results: dict):
    model_names = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    # Console table
    col = 28
    print(f"\n{'-' * 75}")
    print(f"  {'Metric':<38} ", end="")
    for name in model_names:
        print(f"{name:>16}", end="  ")
    print(f"\n  {'-' * 38} ", end="")
    for _ in model_names:
        print(f"{'-' * 16}", end="  ")
    print()

    for metric in metrics:
        print(f"  {metric:<38} ", end="")
        for name in model_names:
            print(f"{_fmt(results[name][metric]):>16}", end="  ")
        print()

    print(f"{'-' * 75}\n")

    # CSV
    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for name in model_names:
            row[name] = _fmt(results[name][metric])
        rows.append(row)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Metric"] + model_names)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {OUTPUT_CSV}")


# Entry point
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}
    for model_name, cfg in MODELS.items():
        print(f"\n{'═' * 55}")
        print(f"  Evaluating: {model_name}")
        print(f"{'═' * 55}")
        results[model_name] = evaluate_model(cfg["path"], cfg["hidden_size"], device)

    print_and_save(results)

"""
Runs the trained model against the Unity environment for a fixed number of steps
and computes per-action accuracy from the reward signal alone.

How it works:
  - After each env.step(), the reward tells us if the last action was correct or wrong.
  - reward > 0        -> correct  (task reward: +1.0 or +0.001 for DoNothing)
  - reward < -0.05    -> wrong    (task penalty: -0.1 or -0.2)

Since each action maps to exactly one task, per-action accuracy = per-task accuracy
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

MODEL_PATH = "../../Python/training/agents/training_logs/run_seed41_coach_config_resumed_20260530_170944/model_final.pt"  # path to the saved model checkpoint
UNITY_BUILD = "../../Data/Builds/BaselineHeadless/InteractiveRL.exe"  # path to Unity .exe, or None to use the Editor
MAX_STEPS = 2000  # how many env steps to run
TIME_SCALE = 100.0  # simulation speed (higher = faster, less stable)
OUTPUT_CSV = "../../Python/training/agents/training_logs/analysis/per_task_accuracy.csv"  # where to save results

# Constants  (must match training setup)

REWARD_CORRECT_THRESHOLD = 0.0  # reward > this  → correct action
REWARD_WRONG_THRESHOLD = -0.05  # reward < this  → wrong action (ignores -0.0001 step penalty)

ACTION_NAMES = {
    0: "DoNothing  (Far > 5m)",
    1: "Talk       (Task ID 6)",
    2: "Look       (Nearby >= 5m)",
    3: "Wave       (Task ID 7)",
    4: "HandShake  (Task ID 2)",
}


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


def load_model(model_path, device):
    model = SimplePPO().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # ← add strict=False
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model


def pick_action(model, obs, device):
    """Greedy action selection (no exploration during evaluation)."""
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_tensor)
        action = torch.argmax(logits, dim=-1).item()
    return action


# Main evaluation loop

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
    model = load_model(MODEL_PATH, device)

    # Per-action counters
    correct_counts = defaultdict(int)
    wrong_counts = defaultdict(int)

    decision_steps, _ = env.get_steps(behavior_name)
    if len(decision_steps) == 0:
        print("No agents found after reset. Is the scene set up correctly?")
        env.close()
        return

    obs = decision_steps.obs[0][0]
    last_action = None
    total_steps = 0
    episodes = 0

    print(f"\nRunning for {MAX_STEPS:,} steps...\n")

    try:
        while total_steps < MAX_STEPS:

            # Pick and send action
            action = pick_action(model, obs, device)
            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.array([[action]]))
            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Check reward for the action we just took
            if len(terminal_steps) > 0:
                reward = terminal_steps.reward[0]
                _record(last_action, reward, correct_counts, wrong_counts)

                episodes += 1
                env.reset()
                decision_steps, _ = env.get_steps(behavior_name)
                obs = decision_steps.obs[0][0]
                last_action = None

            else:
                reward = decision_steps.reward[0]
                _record(last_action, reward, correct_counts, wrong_counts)
                obs = decision_steps.obs[0][0]

            last_action = action
            total_steps += 1

            if total_steps % 1000 == 0:
                print(f"  Step {total_steps:,} / {MAX_STEPS:,}  |  Episodes: {episodes}")

    except KeyboardInterrupt:
        print("\nStopped early by user.")

    finally:
        env.close()

    _print_and_save_results(correct_counts, wrong_counts, total_steps, episodes)


def _record(action, reward, correct_counts, wrong_counts):
    """Record the outcome of an action based on the reward signal."""
    if action is None:
        return  # no action sent yet (first step)
    if reward > REWARD_CORRECT_THRESHOLD:
        correct_counts[action] += 1
    elif reward < REWARD_WRONG_THRESHOLD:
        wrong_counts[action] += 1
    # else: living penalty only (-0.0001), ignore


def _print_and_save_results(correct_counts, wrong_counts, total_steps, episodes):
    """Print accuracy table to console and save to CSV."""

    print(f"\n{'-' * 65}")
    print(f"  Results after {total_steps:,} steps  ({episodes} episodes)")
    print(f"{'-' * 65}")
    print(f"  {'Action':<28} {'Correct':>8} {'Wrong':>8} {'Accuracy':>10}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 10}")

    rows = []
    for action_id, name in ACTION_NAMES.items():
        correct = correct_counts[action_id]
        wrong = wrong_counts[action_id]
        total = correct + wrong
        acc = (correct / total * 100) if total > 0 else None
        acc_str = f"{acc:.1f}%" if acc is not None else "N/A"

        print(f"  {name:<28} {correct:>8} {wrong:>8} {acc_str:>10}")
        rows.append({
            "Action ID": action_id,
            "Action Name": name.split()[0],  # short name only (e.g. "DoNothing")
            "Correct": correct,
            "Wrong": wrong,
            "Accuracy": acc_str,
        })

    print(f"{'-' * 65}\n")

    # Save to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()

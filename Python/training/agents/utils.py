import yaml
import numpy as np
import os
from pathlib import Path
from datetime import datetime

REWARD_POSITIVE_THRESHOLD = 0.0
REWARD_NEGATIVE_THRESHOLD = -0.05


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Config
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_config_files(config_dir='configs'):
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"Config directory not found: {config_dir}")
        return []
    return sorted(str(f) for f in config_path.glob('*.yaml'))


# Accuracy tracker
class AccuracyTracker:
    """
    Derives accuracy purely from the reward signal.
      reward > POSITIVE_THRESHOLD  -> correct
      reward < NEGATIVE_THRESHOLD  -> wrong
      anything in between          -> ignored (living penalty etc.)
    """

    def __init__(self):
        self._total_correct = 0
        self._total_wrong = 0
        self._ep_correct = 0
        self._ep_wrong = 0

    def record(self, reward: float) -> None:
        if reward > REWARD_POSITIVE_THRESHOLD:
            self._ep_correct += 1
            self._total_correct += 1
        elif reward < REWARD_NEGATIVE_THRESHOLD:
            self._ep_wrong += 1
            self._total_wrong += 1

    def episode_accuracy(self) -> float | None:
        total = self._ep_correct + self._ep_wrong
        return self._ep_correct / total if total > 0 else None

    def episode_counts(self) -> tuple[int, int]:
        return self._ep_correct, self._ep_wrong

    def lifetime_accuracy(self) -> float | None:
        total = self._total_correct + self._total_wrong
        return self._total_correct / total if total > 0 else None

    def lifetime_counts(self) -> tuple[int, int]:
        return self._total_correct, self._total_wrong

    def reset_episode(self) -> None:
        self._ep_correct = 0
        self._ep_wrong = 0


# CSV helpers
def save_rewards_to_file(all_rewards, filepath, total_steps=0):
    """
    all_rewards: list of (step, reward) tuples.
    """
    with open(filepath, 'w') as f:
        f.write("Step,Reward\n")
        for step, r in all_rewards:
            f.write(f"{step},{r:.4f}\n")

    meta_path = filepath + ".meta"
    with open(meta_path, 'w') as f:
        f.write(f"total_steps={total_steps}\n")
        f.write(f"total_episodes={len(all_rewards)}\n")

    print(f"Rewards saved -> {filepath} (episodes={len(all_rewards)}, steps={total_steps})")


def save_accuracy_to_file(accuracy_history, filepath):
    """
    accuracy_history: list of (step, correct, wrong) tuples.
    """
    if not accuracy_history:
        return
    with open(filepath, 'w') as f:
        f.write("Step,Correct,Wrong\n")
        for step, correct, wrong in accuracy_history:
            f.write(f"{step},{correct},{wrong}\n")
    print(f"Accuracy saved -> {filepath} (episodes={len(accuracy_history)})")


def load_rewards_from_csv(filepath):
    """
    Returns:
        rewards     : list of (step, reward) tuples
        total_steps : int from .meta file
        total_eps   : int (number of rows)
    """
    rewards = []
    if not os.path.exists(filepath):
        print(f"No CSV at {filepath} — starting fresh.")
        return rewards, 0, 0

    with open(filepath, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    rewards.append((int(parts[0]), float(parts[1])))
                except ValueError:
                    pass

    total_steps = 0
    meta_path = filepath + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                k, _, v = line.strip().partition('=')
                if k == 'total_steps':
                    total_steps = int(v)
    else:
        print("  No .meta file — step count starts from 0.")

    print(f"Loaded {len(rewards)} episodes from {filepath} (steps: {total_steps})")
    return rewards, total_steps, len(rewards)


def load_accuracy_from_csv(filepath):
    """
    Returns list of (step, correct, wrong) tuples.
    """
    history = []
    if not os.path.exists(filepath):
        print(f"No accuracy CSV at {filepath} — starting fresh.")
        return history

    with open(filepath, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    history.append((int(parts[0]), int(parts[1]), int(parts[2])))
                except ValueError:
                    pass

    print(f"Loaded {len(history)} accuracy episodes from {filepath}")
    return history
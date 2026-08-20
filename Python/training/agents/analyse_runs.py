"""
analyse_runs.py
===============
Post-processing script -- reads training_logs/ and produces:

  1. sweep_results.csv        - one row per config, metrics averaged across seeds
  2. seed_variance.csv        - mean ± std for configs 1, 4, 8
  3. definitive_summary.csv   - detailed metrics for the best config run
  4. Re-generated plots       - reward curve, accuracy curve, comparison plot
"""

import os
import re
import csv
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


CONVERGENCE_THRESHOLD = 0.80   # rolling avg must exceed this
CONVERGENCE_WINDOW = 50        # for this many consecutive episodes
ROLLING_WINDOW = 100           # episodes for rolling average
ACCURACY_SMOOTH_WINDOW = 50    # episodes for accuracy rolling average
FINAL_ACCURACY_FRACTION = 0.05
LOGS_DIR = "training_logs"  # Change this if needed


def load_rewards_csv(filepath: str) -> List[Tuple[int, float]]:
    """Returns list of (step, reward) tuples from rewards_final.csv."""
    rewards = []
    if not os.path.exists(filepath):
        return rewards

    with open(filepath, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    rewards.append((int(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return rewards


def load_accuracy_csv(filepath: str) -> List[Tuple[int, int, int]]:
    """Returns list of (step, correct, wrong) tuples from accuracy_final.csv."""
    history = []
    if not os.path.exists(filepath):
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
    return history


# Metric computation
def rolling_average(values: List[float], window: int) -> List[float]:
    """Returns list of rolling averages (same length as input)."""
    return [np.mean(values[max(0, i - window + 1): i + 1]) for i in range(len(values))]


def compute_accuracy(correct: int, wrong: int) -> Optional[float]:
    """Calculate accuracy percentage from correct and wrong counts."""
    total = correct + wrong
    return 100.0 * correct / total if total > 0 else None

def compute_final_accuracy(accuracy_data: List, recent_fraction:float = 0.1) -> Optional[float]:
    """Compute accuracy over most recent fraction of episodes."""
    if not accuracy_data:
        return None

    window = max(1, int(len(accuracy_data) * recent_fraction))
    recent_data = accuracy_data[-window:]

    total_correct = sum(c for _, c, w in recent_data)
    total_wrong = sum(w for _, c, w in recent_data)
    return compute_accuracy(total_correct, total_wrong)


def find_convergence(rolling_avgs: List[float], steps: List[int]) -> Tuple[Optional[int], Optional[int]]:
    """Find step and episode where convergence occurs."""
    consecutive = 0
    for i, avg in enumerate(rolling_avgs):
        if avg >= CONVERGENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= CONVERGENCE_WINDOW:
                conv_ep = i - CONVERGENCE_WINDOW + 1
                return steps[conv_ep], conv_ep
        else:
            consecutive = 0
    return None, None


def compute_metrics(rewards_data: List, accuracy_data: List) -> Optional[Dict[str, Any]]:
    """Compute all metrics for a single run."""
    if not rewards_data:
        return None

    steps = [s for s, _ in rewards_data]
    rewards = [r for _, r in rewards_data]
    rolling_avgs = rolling_average(rewards, ROLLING_WINDOW)

    final_accuracy = compute_final_accuracy(accuracy_data, recent_fraction=FINAL_ACCURACY_FRACTION)
    steps_to_80, convergence_episode = find_convergence(rolling_avgs, steps)

    post_convergence_std = None
    if convergence_episode is not None:
        post_avgs = rolling_avgs[convergence_episode:]
        if len(post_avgs) > 1:
            post_convergence_std = float(np.std(post_avgs))

    return {
        'final_accuracy': final_accuracy,
        'steps_to_80': steps_to_80,
        'convergence_episode': convergence_episode,
        'max_avg100_reward': float(np.max(rolling_avgs)),
        'post_convergence_std': post_convergence_std,
        'max_episode_reward': float(np.max(rewards)),
        'min_episode_reward': float(np.min(rewards)),
        'final_avg100_reward': float(rolling_avgs[-1]),
        'total_episodes': len(rewards),
        'total_steps': steps[-1] if steps else 0,
    }



# Folder discovery
FOLDER_PATTERN = re.compile(r'^run_seed(\d+)_(.+?)_(fresh|resumed)_\d{8}_\d{6}$')


def parse_run_folder(folder_name: str) -> Tuple[Optional[int], Optional[str]]:
    """Extracts (seed, config_name) from folder name."""
    match = FOLDER_PATTERN.match(folder_name)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def find_runs(logs_dir: str) -> Dict[str, List[Dict]]:
    """Scans logs_dir and returns runs grouped by config."""
    runs = defaultdict(list)
    logs_path = Path(logs_dir)

    if not logs_path.exists():
        print(f"ERROR: logs directory not found: {logs_dir}")
        return runs

    for folder in sorted(os.listdir(logs_dir)):
        full_path = logs_path / folder
        if not full_path.is_dir():
            continue

        seed, config = parse_run_folder(folder)
        if config is None:
            print(f"  Skipping (unrecognised folder name): {folder}")
            continue

        rewards_path = full_path / 'rewards_final.csv'
        accuracy_path = full_path / 'accuracy_final.csv'

        rewards_data = load_rewards_csv(str(rewards_path))
        accuracy_data = load_accuracy_csv(str(accuracy_path))

        if not rewards_data:
            print(f"  WARNING: no rewards data in {folder}")
            continue

        runs[config].append({
            'seed': seed,
            'folder': folder,
            'rewards_data': rewards_data,
            'accuracy_data': accuracy_data,
            'metrics': compute_metrics(rewards_data, accuracy_data),
        })
        print(f"  Loaded: {folder} (seed={seed}, episodes={len(rewards_data)})")

    return runs



# Aggregation across seeds
def aggregate_seeds(run_list: List[Dict]) -> Optional[Dict[str, Any]]:
    """Aggregate metrics across seeds for one config."""
    metrics_list = [r['metrics'] for r in run_list if r['metrics'] is not None]
    if not metrics_list:
        return None

    keys = ['final_accuracy', 'steps_to_80', 'max_avg100_reward', 'post_convergence_std']
    result = {}

    for key in keys:
        values = [m[key] for m in metrics_list if m[key] is not None]
        if values:
            result[f'{key}_mean'] = float(np.mean(values))
            result[f'{key}_std'] = float(np.std(values))
            result[f'{key}_n'] = len(values)
        else:
            result[f'{key}_mean'] = None
            result[f'{key}_std'] = None
            result[f'{key}_n'] = 0

    return result



# Formatting helpers

def fmt_number(val: Optional[float], decimals: int = 1, scale: float = 1.0, suffix: str = '') -> str:
    """Format a number or return 'N/A'."""
    if val is None:
        return 'N/A'
    return f"{val * scale:.{decimals}f}{suffix}"


def fmt_steps(val: Optional[int]) -> str:
    """Format steps as thousands (e.g., 167k)."""
    if val is None:
        return 'N/A'
    return f"{val // 1000}k"


def config_sort_key(name: str) -> int:
    """Extract numeric part from config name for sorting."""
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else 0



# CSV writers

def save_sweep_results(aggregated: Dict, output_path: str):
    """Saves the main sweep table (one row per config, means only)."""
    rows = []
    for config in sorted(aggregated.keys(), key=config_sort_key):
        agg = aggregated[config]
        if agg is None:
            continue

        rows.append({
            'Config': config,
            'Final Accuracy (%)': fmt_number(agg['final_accuracy_mean'], 1, suffix='%'),
            'Steps to 80%': fmt_steps(int(agg['steps_to_80_mean']) if agg['steps_to_80_mean'] else None),
            'Max Avg100 Reward': fmt_number(agg['max_avg100_reward_mean'], 2),
            'Post-conv Std (Sigma)': fmt_number(agg['post_convergence_std_mean'], 2),
            'N seeds': agg['final_accuracy_n'],
        })

    if not rows:
        print("No data to write to sweep_results.csv")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSweep results saved -> {output_path}")


def save_seed_variance(aggregated: Dict, configs_for_variance: List[str], output_path: str):
    """Saves mean ± std table for specified configs."""
    rows = []
    for config in sorted(configs_for_variance, key=config_sort_key):
        agg = aggregated.get(config)
        if agg is None:
            print(f"  WARNING: no data for {config} in seed variance table")
            continue

        def mean_std(key: str) -> str:
            mean_val = agg[f'{key}_mean']
            std_val = agg[f'{key}_std']
            if mean_val is None:
                return 'N/A'

            if key == 'final_accuracy':
                return f"{mean_val:.1f} +- {std_val:.1f}"
            elif key == 'steps_to_80':
                return f"{fmt_steps(int(mean_val))} +- {fmt_steps(int(std_val))}"
            else:
                return f"{mean_val:.2f} +- {std_val:.2f}"

        rows.append({
            'Config': config,
            'Final Accuracy (%)': mean_std('final_accuracy'),
            'Steps to 80%': mean_std('steps_to_80'),
            'Max Avg100 Reward': mean_std('max_avg100_reward'),
        })

    if not rows:
        print("No data to write to seed_variance.csv")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Seed variance saved -> {output_path}")


def save_definitive_summary(run_list: List[Dict], output_path: str):
    """Saves detailed metrics for the best run."""
    best_run = max(
        (r for r in run_list if r['metrics'] is not None),
        key=lambda r: r['metrics']['final_accuracy'] or 0
    )
    m = best_run['metrics']

    rows = [
        (f'Final task accuracy (last {FINAL_ACCURACY_FRACTION * 100}% of episodes)',
         fmt_number(m['final_accuracy'], 1, suffix='%')),
        ('Final rolling avg reward (last 100)', fmt_number(m['final_avg100_reward'], 2)),
        ('Steps to 80% accuracy', fmt_steps(m['steps_to_80'])),
        ('Episode of first 80% accuracy', str(m['convergence_episode']) if m['convergence_episode'] else 'N/A'),
        ('Max episode reward', fmt_number(m['max_episode_reward'], 2)),
        ('Min episode reward', fmt_number(m['min_episode_reward'], 2)),
        ('Post-convergence reward std dev', fmt_number(m['post_convergence_std'], 2)),
        ('Total episodes', str(m['total_episodes'])),
        ('Total steps', fmt_steps(m['total_steps'])),
        ('Seed', str(best_run['seed'])),
        ('Folder', best_run['folder']),
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerows(rows)
    print(f"Definitive summary saved -> {output_path}")

    print(f"\n{'=' * 55}")
    print(f"DEFINITIVE RUN SUMMARY  ({best_run['folder']})")
    print(f"{'=' * 55}")
    for metric, value in rows:
        print(f"  {metric:<45} {value}")
    print(f"{'=' * 55}\n")


# Plotting
def get_reward_series(rewards_data: List) -> Tuple[List[int], List[float], List[float]]:
    """Extract steps, raw rewards, and rolling averages."""
    steps = [s for s, _ in rewards_data]
    rewards = [r for _, r in rewards_data]
    avgs = rolling_average(rewards, ROLLING_WINDOW)
    return steps, rewards, avgs


def plot_definitive_reward(rewards_data: List, output_path: str):
    """Reward curve for the definitive run."""
    steps, rewards, avgs = get_reward_series(rewards_data)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, rewards, color='steelblue', alpha=0.3, linewidth=0.8,
            label='Episode reward (raw)')
    ax.plot(steps, avgs, color='crimson', linewidth=2,
            label=f'Rolling avg (last {ROLLING_WINDOW} eps)')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)

    ax.set_xlabel('Training step')
    ax.set_ylabel('Episode reward')
    ax.set_title('Training Reward Curve - Definitive Run')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reward curve saved -> {output_path}")


def plot_definitive_accuracy(accuracy_data: List, output_path: str):
    """Accuracy curve for the definitive run."""
    # Calculate accuracies from correct/wrong
    scored = [(step, compute_accuracy(c, w)) for step, c, w in accuracy_data if compute_accuracy(c, w) is not None]
    if not scored:
        print("No accuracy data for plot.")
        return

    steps = [s for s, _ in scored]
    accuracies = [a for _, a in scored]
    smoothed = rolling_average(accuracies, ACCURACY_SMOOTH_WINDOW)

    # Lifetime accuracy (cumulative)
    cumulative_correct = np.cumsum([c for _, c, w in accuracy_data])
    cumulative_total = np.cumsum([c + w for _, c, w in accuracy_data])
    lifetime_acc = np.where(cumulative_total > 0, 100.0 * cumulative_correct / cumulative_total, np.nan)  # Convert to percentage
    lt_steps = [s for s, _, _ in accuracy_data]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, accuracies, color='steelblue', alpha=0.3, linewidth=0.8,
            label='Episode accuracy (raw)')
    ax.plot(steps, smoothed, color='crimson', linewidth=2,
            label=f'Rolling avg (last {ACCURACY_SMOOTH_WINDOW} eps)')
    ax.plot(lt_steps, lifetime_acc, color='darkorange', linewidth=1.5,
            linestyle='--', label='Cumulative lifetime (for reference only)')
    ax.axhline(100, color='k', linewidth=0.5, linestyle='--', alpha=0.3)  # 100% line
    ax.axhline(CONVERGENCE_THRESHOLD * 100, color='green', linewidth=1,
               linestyle=':', alpha=0.7, label=f'{CONVERGENCE_THRESHOLD * 100}% threshold')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Curve - Definitive Run')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy curve saved -> {output_path}")


def plot_comparison_all_configs(runs_dict: Dict, output_path_prefix: str):
    """Comparison plot across all configs."""
    configs = sorted(runs_dict.keys(), key=config_sort_key)
    colors = plt.cm.tab10.colors

    fig, (ax_reward, ax_accuracy) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('All Configs - Performance Comparison', fontsize=14, fontweight='bold')

    # Reward comparison
    for i, config in enumerate(configs):
        color = colors[i % len(colors)]
        run_list = runs_dict[config]

        all_avgs = []
        all_steps = None
        for run in run_list:
            steps, _, avgs = get_reward_series(run['rewards_data'])
            all_avgs.append(avgs)
            if all_steps is None:
                all_steps = steps

        if all_avgs and all_steps:
            min_len = min(len(a) for a in all_avgs)
            mean_avg = np.mean([a[:min_len] for a in all_avgs], axis=0)
            ax_reward.plot(all_steps[:min_len], mean_avg, color=color,
                           linewidth=1.5, label=config)

    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel(f'Rolling avg reward (last {ROLLING_WINDOW} eps)')
    ax_reward.set_title('Smoothed Reward - All Configs')
    ax_reward.legend(fontsize=8, ncol=2)
    ax_reward.grid(True, alpha=0.3)

    # Accuracy comparison
    for i, config in enumerate(configs):
        color = colors[i % len(colors)]
        run_list = runs_dict[config]

        all_smoothed = []
        all_steps = None

        for run in run_list:
            # Extract accuracy as percentage (0-100)
            steps = []
            accuracies = []
            for step, correct, wrong in run['accuracy_data']:
                total = correct + wrong
                if total > 0:
                    steps.append(step)
                    accuracies.append(100.0 * correct / total)

            if not steps:
                print(f"  WARNING: No valid accuracy data for {config}")
                continue

            smooth_window = min(ACCURACY_SMOOTH_WINDOW, len(accuracies) // 2 or 1)
            if smooth_window > 1:
                smoothed = rolling_average(accuracies, smooth_window)
            else:
                smoothed = accuracies

            all_smoothed.append(smoothed)
            if all_steps is None:
                all_steps = steps

        if all_smoothed and all_steps:
            # Truncate to shortest run for averaging
            min_len = min(len(s) for s in all_smoothed)
            mean_acc = np.mean([s[:min_len] for s in all_smoothed], axis=0)
            ax_accuracy.plot(all_steps[:min_len], mean_acc, color=color,
                             linewidth=1.5, label=config)

    ax_accuracy.axhline(CONVERGENCE_THRESHOLD * 100, color='k', linewidth=0.8,
             linestyle=':', alpha=0.6, label=f'{CONVERGENCE_THRESHOLD * 100}% threshold')
    ax_accuracy.set_xlabel('Step')
    ax_accuracy.set_ylabel(f'Rolling avg accuracy (%) (window={ACCURACY_SMOOTH_WINDOW})')
    ax_accuracy.set_title('Smoothed Accuracy - All Configs')
    ax_accuracy.set_ylim(0, 105)
    ax_accuracy.legend(fontsize=8, ncol=2)
    ax_accuracy.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_path_prefix}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved -> {output_path}")

# Main
def main():
    print(f"\n{'=' * 60}")
    print(f"ANALYSING RUNS IN: {LOGS_DIR}")
    print(f"Convergence: rolling avg >= {CONVERGENCE_THRESHOLD} "
          f"for {CONVERGENCE_WINDOW} consecutive episodes")
    print(f"{'=' * 60}\n")

    output_dir = os.path.join(LOGS_DIR, 'analysis')
    plots_dir = os.path.join(output_dir, 'training_logs/plots')
    os.makedirs(plots_dir, exist_ok=True)

    runs_dict = find_runs(LOGS_DIR)
    if not runs_dict:
        print("No valid runs found. Check your logs directory and folder naming.")
        return

    print(f"\nFound {sum(len(v) for v in runs_dict.values())} runs "
          f"across {len(runs_dict)} configs.\n")

    # Aggregate across seeds
    aggregated = {config: aggregate_seeds(run_list) for config, run_list in runs_dict.items()}

    # Sweep results table
    save_sweep_results(aggregated, os.path.join(output_dir, 'sweep_results.csv'))

    # Print table to console
    print(f"\n{'Config':<12} {'Accuracy':>12} {'Steps-to-80':>12} "
          f"{'MaxAvg100':>12} {'Post-σ':>10} {'Seeds':>6}")
    print('-' * 68)
    for config in sorted(aggregated.keys(), key=config_sort_key):
        agg = aggregated[config]
        if agg is None:
            continue
        print(f"{config:<12} "
              f"{fmt_number(agg['final_accuracy_mean'], 1, suffix='%'):>12} "
              f"{fmt_steps(int(agg['steps_to_80_mean']) if agg['steps_to_80_mean'] else None):>12} "
              f"{fmt_number(agg['max_avg100_reward_mean'], 2):>12} "
              f"{fmt_number(agg['post_convergence_std_mean'], 2):>10} "
              f"{agg['final_accuracy_n']:>6}")

    valid = {c: agg for c, agg in aggregated.items() if agg and agg['final_accuracy_mean'] is not None}
    best_config = max(valid, key=lambda c: valid[c]['final_accuracy_mean']) if valid else None

    if best_config:
        print(f"\nBest config (highest mean final accuracy): {best_config}")

    variance_configs = sorted(valid, key=lambda c: valid[c]['final_accuracy_mean'], reverse=True)[:3]
    print(f"Seed variance table configs: {variance_configs}")
    save_seed_variance(aggregated, variance_configs, os.path.join(output_dir, 'seed_variance.csv'))

    if best_config and best_config in runs_dict:
        save_definitive_summary(runs_dict[best_config], os.path.join(output_dir, 'definitive_summary.csv'))

        best_run = max(runs_dict[best_config], key=lambda r: (r['metrics'] or {}).get('final_accuracy') or 0)
        plot_definitive_reward(best_run['rewards_data'], os.path.join(plots_dir, f'reward_curve_{best_config}.png'))
        plot_definitive_accuracy(best_run['accuracy_data'], os.path.join(plots_dir, f'accuracy_curve_{best_config}.png'))
    else:
        print("WARNING: best config not found - skipping definitive plots.")

    if len(runs_dict) > 1:
        plot_comparison_all_configs(runs_dict, os.path.join(plots_dir, 'all_configs'))

    print(f"\nAll outputs saved to: {output_dir}/")
    print("Done.\n")


if __name__ == '__main__':
    main()
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import yaml
from collections import deque, defaultdict
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path


# Config loader

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_config_files(config_dir='configs'):
    # Find all .yaml files in the config directory and return sorted list.
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"Config directory not found: {config_dir}")
        return []
    yaml_files = sorted(config_path.glob('*.yaml'))
    return [str(f) for f in yaml_files]


REWARD_POSITIVE_THRESHOLD = 0.0
REWARD_NEGATIVE_THRESHOLD = -0.05


class AccuracyTracker:
    """
    Derives accuracy purely from the reward signal

    Logic:
        reward > POSITIVE_THRESHOLD  → agent did the right thing  (correct)
        reward < NEGATIVE_THRESHOLD  → agent did the wrong thing  (wrong)
        anything in between          → step is skipped (living penalty etc.)
    """

    def __init__(self):
        # Lifetime counters
        self._total_correct = 0
        self._total_wrong = 0

        # Per-episode counters (reset each episode)
        self._ep_correct = 0
        self._ep_wrong = 0

    def record(self, reward: float) -> None:
        # Call once per step with the reward received that step
        if reward > REWARD_POSITIVE_THRESHOLD:
            self._ep_correct += 1
            self._total_correct += 1
        elif reward < REWARD_NEGATIVE_THRESHOLD:
            self._ep_wrong += 1
            self._total_wrong += 1
        # else: living penalty / neutral step — ignore

    def episode_accuracy(self) -> float | None:

        # Accuracy for the episode that just finished

        total = self._ep_correct + self._ep_wrong
        return self._ep_correct / total if total > 0 else None

    def episode_counts(self) -> tuple[int, int]:
        # Returns (correct, wrong) for the current episode
        return self._ep_correct, self._ep_wrong

    def lifetime_accuracy(self) -> float | None:
        # Overall accuracy across all episodes
        total = self._total_correct + self._total_wrong
        return self._total_correct / total if total > 0 else None

    def lifetime_counts(self) -> tuple[int, int]:
        # Returns (correct, wrong) lifetime totals
        return self._total_correct, self._total_wrong

    def reset_episode(self) -> None:
        self._ep_correct = 0
        self._ep_wrong = 0


# Model

class SimplePPO(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)


# Agent

class PPOAgent:
    def __init__(self, cfg, device='cuda'):
        behavior_cfg = cfg['behaviors']['PepperGreeting']
        hp = behavior_cfg['hyperparameters']
        net = behavior_cfg['network_settings']
        reward_cfg = behavior_cfg['reward_signals']['extrinsic']
        env_cfg = cfg['env_settings']

        self.device = device

        obs_size = env_cfg['obs_size']
        act_size = env_cfg['act_size']
        hidden_size = net['hidden_units']

        self.actor_critic = SimplePPO(obs_size, act_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=hp['learning_rate']
        )

        self.clip_param = hp['epsilon']
        self.ppo_epoch = hp['num_epoch']
        self.num_mini_batch = hp['num_mini_batch']
        self.value_loss_coef = hp['value_loss_coef']
        self.entropy_coef = hp['entropy_start']
        self.gamma = reward_cfg['gamma']
        self.tau = hp['lambd']
        self.max_grad_norm = behavior_cfg['max_grad_norm']

        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

        self.action_counts = np.zeros(act_size)

    def act(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.actor_critic(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
                log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        return action, value.item(), log_prob

    def store_transition(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(0.0 if done else 1.0)
        self.action_counts[action] += 1

    def compute_gae(self, next_value):
        gae = 0
        returns = []
        advs = []
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = (self.rewards[step]
                     + self.gamma * values[step + 1] * self.masks[step]
                     - values[step])
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advs

    def update(self, next_value):
        returns, advantages = self.compute_gae(next_value)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        observations = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.observations)
        mini_batch_size = max(1, n // self.num_mini_batch)

        for _ in range(self.ppo_epoch):
            indices = np.random.permutation(n)
            for start in range(0, n, mini_batch_size):
                idx = indices[start: start + mini_batch_size]

                logits, values_pred = self.actor_critic(observations[idx])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred.squeeze(), returns[idx])
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.clear_buffers()

    def clear_buffers(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

    def save(self, path, total_steps=0, episode_count=0):
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
            'total_steps': total_steps,
            'episode_count': episode_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'entropy_coef' in ckpt:
            self.entropy_coef = ckpt['entropy_coef']
        steps = ckpt.get('total_steps', 0)
        episodes = ckpt.get('episode_count', 0)
        return steps, episodes


# CSV / metadata helpers

def save_rewards_to_file(all_rewards, filepath, total_steps=0):
    """Write the full reward history to CSV (episode numbers are 1-based row index).
    Also writes a companion .meta file storing total_steps and total_episodes so
    resumed runs can continue all counters from exactly where we left off."""
    with open(filepath, 'w') as f:
        f.write("Episode,Reward,Cumulative Reward\n")
        cumulative = 0
        for i, r in enumerate(all_rewards, 1):
            cumulative += r
            f.write(f"{i},{r:.4f},{cumulative:.4f}\n")

    meta_path = filepath + ".meta"
    with open(meta_path, 'w') as f:
        f.write(f"total_steps={total_steps}\n")
        f.write(f"total_episodes={len(all_rewards)}\n")

    print(f"Rewards saved → {filepath}  (episodes={len(all_rewards)}, steps={total_steps})")


def save_accuracy_to_file(accuracy_history, filepath):
    """
    Writes per-episode accuracy to CSV.
    Format: Episode, Accuracy, Correct, Wrong
    """
    if not accuracy_history:
        return

    with open(filepath, 'w') as f:
        f.write("Episode,Accuracy,Correct,Wrong\n")
        for i, (acc, correct, wrong) in enumerate(accuracy_history, 1):
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            f.write(f"{i},{acc_str},{correct},{wrong}\n")

    print(f"Accuracy saved → {filepath}  (episodes={len(accuracy_history)})")


def load_rewards_from_csv(filepath):
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
                    rewards.append(float(parts[1]))
                except ValueError:
                    pass

    # Read companion meta file for step count
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

    print(f"Loaded {len(rewards)} episodes from {filepath}  (steps: {total_steps})")
    return rewards, total_steps, len(rewards)


def load_accuracy_from_csv(filepath):
    """Load accuracy history from a previous run's accuracy CSV.
    Returns a list of (accuracy, correct, wrong) tuples, one per episode."""
    history = []
    if not os.path.exists(filepath):
        print(f"No accuracy CSV at {filepath} — starting fresh.")
        return history

    with open(filepath, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    acc     = None if parts[1] == 'N/A' else float(parts[1])
                    correct = int(parts[2])
                    wrong   = int(parts[3])
                    history.append((acc, correct, wrong))
                except ValueError:
                    pass

    print(f"Loaded {len(history)} accuracy episodes from {filepath}")
    return history


# Plotting

def plot_training_results(episode_rewards, save_path=None, hyperparams=None):
    if not episode_rewards:
        print("No data to plot!")
        return

    episodes = list(range(1, len(episode_rewards) + 1))
    cumulative = np.cumsum(episode_rewards)
    window = min(100, len(episode_rewards))
    moving_avg = [np.mean(episode_rewards[max(0, i - window + 1): i + 1])
                  for i in range(len(episode_rewards))]

    # Add extra row at the bottom for the hyperparameter table if provided
    if hyperparams:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[5, 1.6], hspace=0.55, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[1, :])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(episodes, cumulative, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward vs Episodes')
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, episode_rewards, 'g-', alpha=0.4, linewidth=1,
             label='Episode Reward')
    ax2.plot(episodes, moving_avg, 'r-', linewidth=2,
             label=f'Moving Avg ({window} eps)')
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward vs Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Hyperparameter table
    if hyperparams:
        ax_info.axis('off')

        # Split params into two columns for a compact two-row table layout
        keys = list(hyperparams.keys())
        values = [str(v) for v in hyperparams.values()]
        mid = (len(keys) + 1) // 2
        col1_k, col1_v = keys[:mid], values[:mid]
        col2_k, col2_v = keys[mid:], values[mid:]
        # Pad shorter column so both have equal rows
        while len(col2_k) < len(col1_k):
            col2_k.append('');
            col2_v.append('')

        cell_text = [[f"{k}  =  {v}" if k else "" for k, v in zip(col1_k, col1_v)],
                     [f"{k}  =  {v}" if k else "" for k, v in zip(col2_k, col2_v)]]
        # Transpose so rows are parameter pairs, columns are the two groups
        cell_text = list(map(list, zip(*cell_text)))

        tbl = ax_info.table(
            cellText=cell_text,
            colWidths=[0.48, 0.48],
            loc='center',
            cellLoc='left',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)

        # Style: header-style top row, alternating shading
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#cccccc')
            cell.set_facecolor('#f5f5f5' if row % 2 == 0 else '#ffffff')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {save_path}")
    plt.close()


def plot_accuracy(accuracy_history, save_path=None):
    """Plot accuracy over episodes with a moving average."""
    # Filter to episodes that had meaningful feedback
    scored = [(i + 1, acc) for i, (acc, _, _) in enumerate(accuracy_history) if acc is not None]
    if not scored:
        print("No accuracy data to plot.")
        return

    episodes = [e for e, _ in scored]
    accuracies = [a for _, a in scored]

    window = min(50, len(accuracies))
    moving_avg = [np.mean(accuracies[max(0, i - window + 1): i + 1])
                  for i in range(len(accuracies))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, accuracies, 'b-', alpha=0.3, linewidth=1, label='Episode Accuracy')
    ax.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window} eps)')
    ax.axhline(1.0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title('Action Accuracy over Episodes\n(based on reward signal: positive = correct, negative = wrong)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Accuracy plot saved → {save_path}")
    plt.close()


# Training

def train(config_path='configs/config.yaml'):
    # RESUME_FROM         = "training_logs/run_20260410_044811_fresh/ppo_model_final.pt"
    # RESUME_CSV          = "training_logs/run_20260410_044811_fresh/rewards_final.csv"
    # RESUME_ACCURACY_CSV = "training_logs/run_20260410_044811_fresh/accuracy_final.csv"
    UNITY_BUILD_PATH = "../../../../Data/Build/InteractiveRL.exe"

    RESUME_FROM         = None
    RESUME_CSV          = None
    RESUME_ACCURACY_CSV = None
    # UNITY_BUILD_PATH = None

    cfg = load_config(config_path)
    behavior_cfg = cfg['behaviors']['PepperGreeting']
    hp = behavior_cfg['hyperparameters']
    engine_cfg = cfg['engine_settings']
    env_cfg = cfg['env_settings']

    max_total_steps = behavior_cfg['max_steps']
    ROLLOUT_STEPS = behavior_cfg['rollout_steps']
    ENTROPY_END = hp['entropy_end']

    # Connect to Unity
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=engine_cfg['time_scale'])
    env = UnityEnvironment(file_name=UNITY_BUILD_PATH, side_channels=[channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]

    # Init agent + accuracy tracker
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPOAgent(cfg, device=device)
    tracker = AccuracyTracker()

    # Load previous state
    prior_rewards = []
    step_offset = 0
    episode_offset = 0

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Loading checkpoint: {RESUME_FROM}")
        agent.load(RESUME_FROM)
        print(f"  Entropy restored: {agent.entropy_coef:.4f}")
    elif RESUME_FROM:
        print(f"WARNING: checkpoint not found at '{RESUME_FROM}' — starting fresh.")

    if RESUME_CSV:
        prior_rewards, step_offset, episode_offset = load_rewards_from_csv(RESUME_CSV)
        print(f"  Resuming from episode {episode_offset}, step {step_offset}")

    # Fresh run: reset entropy to configured start value
    if RESUME_FROM is None:
        agent.entropy_coef = hp['entropy_start']

    ENTROPY_START = agent.entropy_coef

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "resumed" if RESUME_FROM else "fresh"
    output_dir = f"training_logs/run_{timestamp}_{run_label}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'run_info.txt'), 'w') as f:
        f.write(f"Run started      : {timestamp}\n")
        f.write(f"Device           : {device}\n")
        f.write(f"Config file      : {config_path}\n")
        f.write(f"Resumed model    : {RESUME_FROM or 'scratch'}\n")
        f.write(f"Resumed CSV      : {RESUME_CSV or 'none'}\n")
        f.write(f"Step offset      : {step_offset}\n")
        f.write(f"Episode offset   : {episode_offset}\n")
        f.write(f"Max steps (run)  : {max_total_steps}\n")
        f.write(f"Entropy start    : {ENTROPY_START:.4f}\n")
        f.write(f"Accuracy method  : reward-based "
                f"(pos>{REWARD_POSITIVE_THRESHOLD}, neg<{REWARD_NEGATIVE_THRESHOLD})\n")

    print(f"\nTraining on {device} | logs → {output_dir}")
    print(f"Config: {config_path}")
    print(f"Episodes : {episode_offset} → ...")
    print(f"Steps    : {step_offset} → {step_offset + max_total_steps}\n")

    # Counters
    all_rewards = list(prior_rewards)
    this_run_rewards = []
    accuracy_history = load_accuracy_from_csv(RESUME_ACCURACY_CSV) if RESUME_ACCURACY_CSV else []  # restored from prior run
    episode_rewards_buffer = deque(maxlen=100)

    for r in prior_rewards[-100:]:
        episode_rewards_buffer.append(r)

    total_steps = step_offset
    episode_count = episode_offset
    episode_reward = 0.0
    rollout_step = 0

    # Initial observation
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) == 0:
        print("No agents found after reset!")
        env.close()
        return
    obs = decision_steps.obs[0][0]

    # Main loop
    try:
        while total_steps < step_offset + max_total_steps:

            while rollout_step < ROLLOUT_STEPS:
                action, value, log_prob = agent.act(obs)

                decision_steps, terminal_steps = env.get_steps(behavior_name)
                if len(decision_steps) > 0:
                    action_tuple = ActionTuple()
                    action_tuple.add_discrete(np.array([[action]]))
                    env.set_actions(behavior_name, action_tuple)

                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)

                if len(terminal_steps) > 0:
                    reward = terminal_steps.reward[0]
                    done = True
                    next_obs = (terminal_steps.obs[0][0]
                                if terminal_steps.obs else np.zeros(env_cfg['obs_size']))

                    # Record this step's reward for accuracy
                    tracker.record(reward)

                    next_value = 0.0

                    episode_reward += reward
                    all_rewards.append(episode_reward)
                    this_run_rewards.append(episode_reward)
                    episode_rewards_buffer.append(episode_reward)
                    avg100 = np.mean(episode_rewards_buffer)

                    # Episode accuracy summary
                    ep_acc = tracker.episode_accuracy()
                    ep_correct, ep_wrong = tracker.episode_counts()
                    accuracy_history.append((ep_acc, ep_correct, ep_wrong))
                    tracker.reset_episode()

                    acc_display = f"{ep_acc * 100:.1f}%" if ep_acc is not None else "N/A"
                    lt_acc = tracker.lifetime_accuracy()
                    lt_display = f"{lt_acc * 100:.1f}%" if lt_acc is not None else "N/A"

                    print(f"Episode {episode_count:5d} | "
                          f"Steps: {total_steps:7d} | "
                          f"Reward: {episode_reward:7.3f} | "
                          f"Avg(100): {avg100:7.3f} | "
                          f"Entropy: {agent.entropy_coef:.4f} | "
                          f"Acc: {acc_display} ({ep_correct}✓ {ep_wrong}✗) | "
                          f"Lifetime: {lt_display}")

                    episode_count += 1
                    episode_reward = 0.0

                    env.reset()
                    decision_steps, terminal_steps = env.get_steps(behavior_name)
                    next_obs = (decision_steps.obs[0][0]
                                if len(decision_steps) > 0
                                else np.zeros(env_cfg['obs_size']))

                else:
                    reward = decision_steps.reward[0]
                    done = False
                    next_obs = decision_steps.obs[0][0]
                    _, next_value, _ = agent.act(next_obs, deterministic=True)

                    # Record mid-episode steps too
                    tracker.record(reward)

                agent.store_transition(obs, action, reward, value, log_prob, done)

                obs = next_obs
                episode_reward += reward
                total_steps += 1
                rollout_step += 1

                if total_steps % 500 == 0:
                    total_acts = agent.action_counts.sum()
                    if total_acts > 0:
                        dist_str = " | ".join(
                            f"A{i}:{100 * c / total_acts:.0f}%"
                            for i, c in enumerate(agent.action_counts))
                        print(f"  [Step {total_steps}] Action dist: {dist_str}")

            # PPO update
            _, next_value, _ = agent.act(obs, deterministic=True)
            agent.update(next_value)
            rollout_step = 0

            # Entropy decays over this run's step budget only
            run_progress = (total_steps - step_offset) / max_total_steps
            agent.entropy_coef = ENTROPY_START + run_progress * (ENTROPY_END - ENTROPY_START)

            # Periodic checkpoint
            if episode_count % 100 == 0 and episode_count > episode_offset:
                save_rewards_to_file(all_rewards,
                                     os.path.join(output_dir, 'rewards.csv'),
                                     total_steps=total_steps)
                save_accuracy_to_file(accuracy_history,
                                      os.path.join(output_dir, 'accuracy.csv'))
                print(f"  Checkpoint saved.")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        import traceback
        print(f"\nTraining error: {e}")
        traceback.print_exc()
    finally:
        env.close()

        # Final saves
        final_model = os.path.join(output_dir, 'ppo_model_final.pt')
        agent.save(final_model, total_steps=total_steps, episode_count=episode_count)
        print(f"Final model → {final_model}")

        final_csv = os.path.join(output_dir, 'rewards_final.csv')
        save_rewards_to_file(all_rewards, final_csv, total_steps=total_steps)
        save_accuracy_to_file(accuracy_history,
                              os.path.join(output_dir, 'accuracy_final.csv'))

        # Print lifetime summary
        lt_correct, lt_wrong = tracker.lifetime_counts()
        lt_acc = tracker.lifetime_accuracy()
        if lt_acc is not None:
            bar = '█' * int(lt_acc * 30) + '░' * (30 - int(lt_acc * 30))
            print(f"\n── Lifetime Accuracy ─────────────────────────────")
            print(f"  {bar}  {lt_acc * 100:.1f}%")
            print(f"  Correct: {lt_correct}  |  Wrong: {lt_wrong}  |  "
                  f"Total scored: {lt_correct + lt_wrong}")
            print(f"──────────────────────────────────────────────────")

        net = behavior_cfg['network_settings']
        hyperparams = {
            'obs_size': env_cfg['obs_size'],
            'act_size': env_cfg['act_size'],
            'hidden_size': net['hidden_units'],
            'learning_rate': hp['learning_rate'],
            'gamma': agent.gamma,
            'tau (GAE λ)': agent.tau,
            'clip_param': agent.clip_param,
            'ppo_epoch': agent.ppo_epoch,
            'num_mini_batch': agent.num_mini_batch,
            'rollout_steps': ROLLOUT_STEPS,
            'value_loss_coef': agent.value_loss_coef,
            'entropy_start': ENTROPY_START,
            'entropy_end': ENTROPY_END,
            'max_grad_norm': agent.max_grad_norm,
            'max_steps': max_total_steps,
            'device': str(device),
        }

        plot_training_results(
            all_rewards,
            save_path=os.path.join(output_dir, 'training_plot_final.png'),
            hyperparams=hyperparams)

        plot_accuracy(accuracy_history,
            save_path=os.path.join(output_dir, 'accuracy_plot_final.png'))

        print("\n" + "=" * 55)
        print("TRAINING SUMMARY")
        print("=" * 55)
        print(f"Config file      : {config_path}")
        print(f"Steps this run   : {total_steps - step_offset}")
        print(f"Steps total      : {total_steps}")
        print(f"Episodes this run: {episode_count - episode_offset}")
        print(f"Episodes total   : {episode_count}")
        if this_run_rewards:
            print(f"Best  (this run) : {max(this_run_rewards):.3f}")
            print(f"Worst (this run) : {min(this_run_rewards):.3f}")
            print(f"Avg   (this run) : {np.mean(this_run_rewards):.3f}")
        if all_rewards:
            print(f"Best  (all time) : {max(all_rewards):.3f}")
            print(f"Avg   (all time) : {np.mean(all_rewards):.3f}")
        print(f"Last 100 avg     : {np.mean(list(episode_rewards_buffer)):.3f}")
        print(f"Output dir       : {output_dir}")
        print("=" * 55)

        return {
            'label': os.path.splitext(os.path.basename(config_path))[0],
            'rewards': all_rewards,
            'accuracy_history': accuracy_history,
        }

def plot_all_runs_comparison(run_results, save_dir='training_logs'):

    if not run_results:
        return

    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    # Reward comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('All Configs — Comparison', fontsize=14, fontweight='bold')

    for i, run in enumerate(run_results):
        color = colors[i % len(colors)]
        rewards = run['rewards']
        label = run['label']
        if not rewards:
            continue

        episodes = list(range(1, len(rewards) + 1))
        window = min(100, len(rewards))
        moving_avg = [np.mean(rewards[max(0, j - window + 1): j + 1])
                      for j in range(len(rewards))]
        cumulative = np.cumsum(rewards)

        ax1.plot(episodes, cumulative, color=color, linewidth=1.5, label=label)
        ax2.plot(episodes, moving_avg, color=color, linewidth=1.5,
                 label=f'{label} (avg{window})')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward (moving avg)')
    ax2.set_title('Smoothed Episode Reward')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison_rewards.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Comparison reward plot saved → {path}")
    plt.close()

    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle('All Configs — Accuracy Comparison', fontsize=14, fontweight='bold')

    for i, run in enumerate(run_results):
        color = colors[i % len(colors)]
        acc_history = run['accuracy_history']
        label = run['label']

        scored = [(j + 1, acc) for j, (acc, _, _) in enumerate(acc_history)
                  if acc is not None]
        if not scored:
            continue

        episodes = [e for e, _ in scored]
        accuracies = [a for _, a in scored]
        window = min(50, len(accuracies))
        moving_avg = [np.mean(accuracies[max(0, j - window + 1): j + 1])
                      for j in range(len(accuracies))]

        ax.plot(episodes, accuracies, color=color, alpha=0.2, linewidth=1)
        ax.plot(episodes, moving_avg, color=color, linewidth=2, label=label)

    ax.axhline(1.0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title('Action Accuracy (solid = moving avg, faint = raw)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Comparison accuracy plot saved → {path}")
    plt.close()

if __name__ == '__main__':
    config_files = find_config_files('../../../configs')

    if not config_files:
        print("No config files found in configs/ directory!")
        exit(1)

    print("\n" + "=" * 70)
    print("MULTI-CONFIG TRAINING")
    print("=" * 70)
    print(f"Found {len(config_files)} config files:")
    for i, cf in enumerate(config_files, 1):
        print(f"  {i}. {cf}")
    print("=" * 70 + "\n")

    all_run_results = []
    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'#' * 70}")
        print(f"# Running config {i}/{len(config_files)}")
        print(f"{'#' * 70}\n")
        result = train(config_file)
        if result is not None:
            all_run_results.append(result)

    # Final cross-run comparison plots
    if len(all_run_results) > 1:
        comparison_dir = '../training_logs'
        os.makedirs(comparison_dir, exist_ok=True)
        print(f"\n{'=' * 55}")
        print("Generating comparison plots across all configs...")
        plot_all_runs_comparison(all_run_results, save_dir=comparison_dir)
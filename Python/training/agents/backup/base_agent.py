import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import yaml
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from datetime import datetime


# Simple PPO Model (Actor-Critic)
class SimplePPO(nn.Module):
    def __init__(self, obs_size=18, act_size=5, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)  # raw logits for discrete actions
        value = self.critic(x)
        return logits, value


class PPOAgent:
    def __init__(self, obs_size=18, act_size=5, hidden_size=128, device='cuda'):
        self.device = device
        self.actor_critic = SimplePPO(obs_size, act_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)

        # PPO hyperparameters
        self.clip_param = 0.2  # standard PPO clip (was 0.1 — too tight)
        self.ppo_epoch = 4
        self.num_mini_batch = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.1  # start high for exploration, decayed in train()
        self.gamma = 0.95  # tune (0 for IRL)
        self.tau = 0.95  # GAE lambda
        self.max_grad_norm = 0.5

        # Experience buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

        # Action distribution tracking
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

        # Normalize advantages
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
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred.squeeze(), returns[idx])

                loss = (policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.clear_buffers()

    def clear_buffers(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

    def save(self, path):
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actor_critic.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'entropy_coef' in ckpt:
            self.entropy_coef = ckpt['entropy_coef']


# ── Plotting / saving ─────────────────────────────────────────────────────────

def plot_training_results(episode_rewards, save_path=None):
    if not episode_rewards:
        print("No data to plot!")
        return

    episodes = list(range(1, len(episode_rewards) + 1))
    cumulative = np.cumsum(episode_rewards)
    window = min(100, len(episode_rewards))
    moving_avg = [np.mean(episode_rewards[max(0, i - window + 1): i + 1])
                  for i in range(len(episode_rewards))]

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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.show()


def save_rewards_to_file(episode_rewards, filepath):
    with open(filepath, 'w') as f:
        f.write("Episode,Reward,Cumulative Reward\n")
        cumulative = 0
        for i, r in enumerate(episode_rewards, 1):
            cumulative += r
            f.write(f"{i},{r:.4f},{cumulative:.4f}\n")
    print(f"Rewards saved to: {filepath}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    max_total_steps = config['behaviors']['PepperGreeting']['max_steps']

    # ── Connect to Unity ───────────────────────────────────────────────────────
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]

    # Initialize agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPOAgent(obs_size=18, act_size=5, device=device)

    # Entropy schedule: start at 0.1, decay toward 0.01 over training
    ENTROPY_START = 0.1
    ENTROPY_END = 0.02
    agent.entropy_coef = ENTROPY_START

    # Rollout size: collect this many steps before each PPO update
    ROLLOUT_STEPS = 1024

    # ── Logging ────────────────────────────────────────────────────────────────
    episode_rewards = []
    episode_rewards_buffer = deque(maxlen=100)
    total_steps = 0
    episode_count = 0
    episode_reward = 0.0
    rollout_step = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_logs/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training on {device} | logs → {output_dir}")

    # ── Initial observation ────────────────────────────────────────────────────
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) == 0:
        print("No agents found after reset!")
        env.close()
        return
    obs = decision_steps.obs[0][0]

    # ── Main loop ──────────────────────────────────────────────────────────────
    try:
        while total_steps < max_total_steps:

            # ── Rollout collection ─────────────────────────────────────────────
            while rollout_step < ROLLOUT_STEPS:
                action, value, log_prob = agent.act(obs)

                # Guard: only send if agents are waiting for a decision
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
                                if len(terminal_steps.obs) > 0
                                else np.zeros(18))
                    next_value = 0.0

                    # Episode finished — log and reset
                    episode_reward += reward
                    episode_rewards.append(episode_reward)
                    episode_rewards_buffer.append(episode_reward)
                    avg100 = np.mean(episode_rewards_buffer)

                    print(f"Episode {episode_count:4d} | "
                          f"Steps: {total_steps:6d} | "
                          f"Ep reward: {episode_reward:7.3f} | "
                          f"Avg(100): {avg100:7.3f} | "
                          f"Entropy: {agent.entropy_coef:.4f}")

                    episode_count += 1
                    episode_reward = 0.0

                    env.reset()
                    decision_steps, terminal_steps = env.get_steps(behavior_name)
                    next_obs = (decision_steps.obs[0][0]
                                if len(decision_steps) > 0
                                else np.zeros(18))

                else:
                    reward = decision_steps.reward[0]
                    done = False
                    next_obs = decision_steps.obs[0][0]
                    _, next_value, _ = agent.act(next_obs, deterministic=True)

                agent.store_transition(obs, action, reward, value, log_prob, done)

                obs = next_obs
                episode_reward += reward
                total_steps += 1
                rollout_step += 1

                # Action distribution log every 500 steps
                if total_steps % 500 == 0:
                    total_acts = agent.action_counts.sum()
                    if total_acts > 0:
                        dist_str = " | ".join(
                            [f"A{i}:{100 * c / total_acts:.0f}%"
                             for i, c in enumerate(agent.action_counts)])
                        print(f"  [Step {total_steps}] Action dist: {dist_str}")

            # ── PPO update after each rollout ──────────────────────────────────
            _, next_value, _ = agent.act(obs, deterministic=True)
            agent.update(next_value)
            rollout_step = 0

            # Decay entropy coefficient linearly over training
            progress = total_steps / max_total_steps
            agent.entropy_coef = ENTROPY_START + progress * (ENTROPY_END - ENTROPY_START)

            # Save periodically
            if episode_count % 100 == 0 and episode_count > 0:
                model_path = os.path.join(output_dir,
                                          f'ppo_model_ep{episode_count}.pt')
                agent.save(model_path)
                save_rewards_to_file(
                    episode_rewards,
                    os.path.join(output_dir, 'rewards.csv'))
                print(f"  Model saved → {model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        import traceback
        print(f"\nTraining error: {e}")
        traceback.print_exc()
    finally:
        env.close()

        final_model = os.path.join(output_dir, 'ppo_model_final.pt')
        agent.save(final_model)
        print(f"Final model → {final_model}")

        rewards_file = os.path.join(output_dir, 'rewards_final.csv')
        save_rewards_to_file(episode_rewards, rewards_file)

        plot_file = os.path.join(output_dir, 'training_plot_final.png')
        plot_training_results(episode_rewards, plot_file)

        print("\n" + "=" * 55)
        print("TRAINING SUMMARY")
        print("=" * 55)
        print(f"Total episodes : {episode_count}")
        print(f"Total steps    : {total_steps}")
        if episode_rewards:
            print(f"Best reward    : {max(episode_rewards):.3f}")
            print(f"Worst reward   : {min(episode_rewards):.3f}")
            print(f"Average reward : {np.mean(episode_rewards):.3f}")
            print(f"Last 100 avg   : {np.mean(list(episode_rewards_buffer)):.3f}")
        print(f"Output dir     : {output_dir}")
        print("=" * 55)


if __name__ == '__main__':
    train()

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
import json
from datetime import datetime


# Simple PPO Model (Actor-Critic)
class SimplePPO(nn.Module):
    def __init__(self, obs_size=11, act_size=5, hidden_size=128):
        super().__init__()
        # Shared backbone
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Separate heads
        self.actor = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
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
    def __init__(self, obs_size=11, act_size=5, hidden_size=128, device='cuda'):
        self.device = device
        self.actor_critic = SimplePPO(obs_size, act_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)

        # PPO hyperparameters
        self.clip_param = 0.2
        self.ppo_epoch = 4
        self.num_mini_batch = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.tau = 0.95  # GAE parameter
        self.max_grad_norm = 0.5

        # Experience buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

    def act(self, obs, deterministic=False):
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.actor_critic(obs_tensor)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
                log_prob = 0  # not needed for deterministic
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
        self.masks.append(1.0 if not done else 0.0)

    def compute_gae(self, next_value):
        gae = 0
        returns = []
        advantages = []

        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return returns, advantages

    def update(self, next_value):
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        observations = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates
        for _ in range(self.ppo_epoch):
            indices = np.arange(len(self.observations))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.num_mini_batch):
                end = start + self.num_mini_batch
                idx = indices[start:end]

                # Get batch
                obs_batch = observations[idx]
                act_batch = actions[idx]
                return_batch = returns[idx]
                adv_batch = advantages[idx]
                old_log_prob_batch = old_log_probs[idx]
                old_value_batch = old_values[idx]

                # Forward pass
                logits, values = self.actor_critic(obs_batch)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # Calculate losses
                new_log_probs = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(), return_batch)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Clear buffers
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
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.full_log_dir = os.path.join(log_dir, f"run_{self.timestamp}")
        os.makedirs(self.full_log_dir, exist_ok=True)

        # Data storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.average_rewards = []
        self.total_steps_history = []

    def log_episode(self, episode_num, episode_reward, episode_length, total_steps, avg_reward_last_100):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.average_rewards.append(avg_reward_last_100)
        self.total_steps_history.append(total_steps)

        # Save to JSON file periodically
        if episode_num % 10 == 0:
            self.save_to_json()

    def save_to_json(self):
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'average_rewards': self.average_rewards,
            'total_steps_history': self.total_steps_history,
            'timestamp': self.timestamp,
            'total_episodes': len(self.episode_rewards)
        }

        json_path = os.path.join(self.full_log_dir, 'training_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def create_plots(self, show_plot=True, save_plot=True):
        if len(self.episode_rewards) == 0:
            print("No data to plot!")
            return

        episodes = list(range(1, len(self.episode_rewards) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Episode Rewards
        axes[0, 0].plot(episodes, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
        axes[0, 0].set_xlabel('Training Episodes')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: Moving Average (last 100 episodes)
        if len(self.average_rewards) > 0:
            axes[0, 1].plot(episodes, self.average_rewards, 'r-', linewidth=2, label='Moving Avg (100 episodes)')
            axes[0, 1].set_xlabel('Training Episodes')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].set_title('Moving Average Reward (Last 100 Episodes)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Plot 3: Cumulative Reward
        cumulative_rewards = np.cumsum(self.episode_rewards)
        axes[1, 0].plot(episodes, cumulative_rewards, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Training Episodes')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].set_title('Cumulative Total Reward')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Episode Lengths
        axes[1, 1].plot(episodes, self.episode_lengths, 'm-', alpha=0.6)
        axes[1, 1].set_xlabel('Training Episodes')
        axes[1, 1].set_ylabel('Episode Length (steps)')
        axes[1, 1].set_title('Episode Lengths Over Time')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'PPO Training Progress - {self.timestamp}', fontsize=16)
        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.full_log_dir, 'training_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")

        if show_plot:
            plt.show()

        # Create a separate detailed plot for cumulative reward
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, cumulative_rewards, 'b-', linewidth=2)
        plt.fill_between(episodes, cumulative_rewards, alpha=0.3, color='blue')
        plt.xlabel('Training Episodes', fontsize=12)
        plt.ylabel('Cumulative Total Reward', fontsize=12)
        plt.title('Total Feedback (Accumulated Reward) Over Training', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add trend line
        if len(episodes) > 1:
            z = np.polyfit(episodes, cumulative_rewards, 1)
            p = np.poly1d(z)
            plt.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            plt.legend()

        if save_plot:
            detailed_path = os.path.join(self.full_log_dir, 'cumulative_reward.png')
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')

        if show_plot:
            plt.show()


def train():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Connect to Unity
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20.0)  # Speed up training
    env = UnityEnvironment(file_name=None, side_channels=[channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    # Initialize agent and logger
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPOAgent(obs_size=11, act_size=5, device=device)
    logger = TrainingLogger()

    print(f"Starting PPO training on {device}...")
    print(f"Logs will be saved to: {logger.full_log_dir}")

    # Training statistics
    episode_rewards_buffer = deque(maxlen=100)
    total_steps = 0
    episode_count = 0

    # Get initial state
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(decision_steps) == 0:
        print("No agents found!")
        return

    obs = decision_steps.obs[0][0]

    try:
        while total_steps < config['behaviors']['PepperGreeting']['max_steps']:
            episode_reward = 0
            done = False
            step_count = 0

            while not done:
                # Act
                action, value, log_prob = agent.act(obs)

                # Send action to Unity
                action_tuple = ActionTuple()
                action_tuple.add_discrete(np.array([[action]]))
                env.set_actions(behavior_name, action_tuple)
                env.step()

                # Get result
                decision_steps, terminal_steps = env.get_steps(behavior_name)

                if len(terminal_steps) > 0:
                    reward = terminal_steps.reward[0]
                    done = True
                    next_obs = terminal_steps.obs[0][0] if len(terminal_steps.obs) > 0 else np.zeros(11)
                    next_value = 0  # terminal state
                else:
                    reward = decision_steps.reward[0]
                    next_obs = decision_steps.obs[0][0]
                    _, next_value, _ = agent.act(next_obs, deterministic=True)

                # Store transition
                agent.store_transition(obs, action, reward, value, log_prob, done)

                # Update
                obs = next_obs
                episode_reward += reward
                step_count += 1
                total_steps += 1

                # Logging
                if total_steps % 100 == 0:
                    print(f"Step {total_steps} | Episode {episode_count} | Reward: {episode_reward:.2f}")

                # Episode length limit
                if step_count >= 1000:
                    done = True

            # End of episode - perform PPO update
            if done:
                # Get value for last state if not terminal
                if len(terminal_steps) == 0:  # Episode ended due to length limit
                    _, next_value, _ = agent.act(obs, deterministic=True)
                else:
                    next_value = 0

                # Update if we have enough experience
                if len(agent.observations) > 0:
                    agent.update(next_value)

                # Track statistics
                episode_rewards_buffer.append(episode_reward)
                avg_reward_last_100 = np.mean(episode_rewards_buffer) if episode_rewards_buffer else 0

                # Log episode data
                logger.log_episode(episode_count, episode_reward, step_count, total_steps, avg_reward_last_100)

                print(f"Episode {episode_count} ended | Steps: {step_count} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward (last 100): {avg_reward_last_100:.2f}")

                episode_count += 1

                # Save model periodically
                if episode_count % 10 == 0:
                    model_path = os.path.join(logger.full_log_dir, f'ppo_model_ep{episode_count}.pt')
                    agent.save(model_path)
                    print(f"Model saved to: {model_path}")

                # Reset environment for next episode
                env.reset()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                if len(decision_steps) > 0:
                    obs = decision_steps.obs[0][0]
                else:
                    obs = np.zeros(11)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        env.close()

        # Save final model
        final_model_path = os.path.join(logger.full_log_dir, 'ppo_model_final.pt')
        agent.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Save all training data
        logger.save_to_json()

        # Create and display plots
        print("\n" + "=" * 50)
        print("Generating training plots...")
        print("=" * 50)
        logger.create_plots(show_plot=True, save_plot=True)

        # Print summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total episodes: {episode_count}")
        print(f"Total steps: {total_steps}")
        if len(logger.episode_rewards) > 0:
            print(f"Best episode reward: {max(logger.episode_rewards):.2f}")
            print(f"Worst episode reward: {min(logger.episode_rewards):.2f}")
            print(f"Average episode reward: {np.mean(logger.episode_rewards):.2f}")
            print(f"Total cumulative reward: {np.sum(logger.episode_rewards):.2f}")
        print(f"Logs directory: {logger.full_log_dir}")
        print("=" * 50)


if __name__ == '__main__':
    # Check if matplotlib is available
    try:
        import matplotlib

        # Use non-interactive backend if running in a headless environment
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not installed. Install with: pip install matplotlib")
        print("Running without plotting functionality...")

    train()
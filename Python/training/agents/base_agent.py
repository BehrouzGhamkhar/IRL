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
        self.gamma = 0.99 # todo: tune (should be 0 in IRL)
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

                # todo: reference the paper: https://github.com/openai/baselines/blob/master/baselines/ppo2/model.py
                # https://arxiv.org/abs/1707.06347
                # todo: also try to remove entropy to see the results
                # try fixing adjusting the reward
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


def plot_training_results(episode_rewards, save_path=None):

    if len(episode_rewards) == 0:
        print("No data to plot!")
        return

    episodes = list(range(1, len(episode_rewards) + 1))

    # Calculate cumulative rewards
    cumulative_rewards = np.cumsum(episode_rewards)

    # Calculate moving average (last 100 episodes)
    window_size = min(100, len(episode_rewards))
    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window_size + 1)
        moving_avg.append(np.mean(episode_rewards[start:i + 1]))

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cumulative Reward vs Episodes
    ax1.plot(episodes, cumulative_rewards, 'b-', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward vs Episodes')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average Reward vs Episodes
    ax2.plot(episodes, episode_rewards, 'g-', alpha=0.6, linewidth=1, label='Episode Reward')
    ax2.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size} eps)')
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
    """Save rewards to a simple text file"""
    with open(filepath, 'w') as f:
        f.write("Episode,Reward,Cumulative Reward\n")
        cumulative = 0
        for i, reward in enumerate(episode_rewards, 1):
            cumulative += reward
            f.write(f"{i},{reward:.2f},{cumulative:.2f}\n")
    print(f"Rewards saved to: {filepath}")


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

    # Initialize agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPOAgent(obs_size=11, act_size=5, device=device)

    # SIMPLIFIED: Just track episode rewards
    episode_rewards = []
    episode_rewards_buffer = deque(maxlen=100)
    total_steps = 0
    episode_count = 0

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_logs/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting PPO training on {device}...")
    print(f"Logs will be saved to: {output_dir}")

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

                # Logging every 100 steps
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
                episode_rewards.append(episode_reward)
                episode_rewards_buffer.append(episode_reward)
                avg_reward_last_100 = np.mean(episode_rewards_buffer) if episode_rewards_buffer else 0

                # Simple printout
                print(f"Episode {episode_count} | Steps: {step_count} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg (100): {avg_reward_last_100:.2f}")

                episode_count += 1

                # Save model periodically
                if episode_count % 100 == 0:
                    model_path = os.path.join(output_dir, f'ppo_model_ep{episode_count}.pt')
                    agent.save(model_path)
                    print(f"Model saved to: {model_path}")

                    # Also save rewards and plot periodically
                    rewards_file = os.path.join(output_dir, 'rewards.csv')
                    save_rewards_to_file(episode_rewards, rewards_file)

                    plot_file = os.path.join(output_dir, f'training_plot_ep{episode_count}.png')
                    plot_training_results(episode_rewards, plot_file)

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
        final_model_path = os.path.join(output_dir, 'ppo_model_final.pt')
        agent.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Save final rewards and plot
        rewards_file = os.path.join(output_dir, 'rewards_final.csv')
        save_rewards_to_file(episode_rewards, rewards_file)

        plot_file = os.path.join(output_dir, 'training_plot_final.png')
        plot_training_results(episode_rewards, plot_file)

        # Simple summary
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total episodes: {episode_count}")
        print(f"Total steps: {total_steps}")
        if len(episode_rewards) > 0:
            print(f"Best reward: {max(episode_rewards):.2f}")
            print(f"Worst reward: {min(episode_rewards):.2f}")
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
            print(f"Total cumulative reward: {np.sum(episode_rewards):.2f}")
        print(f"Output directory: {output_dir}")
        print("=" * 50)


if __name__ == '__main__':
    train()
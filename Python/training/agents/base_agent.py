import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import yaml
import time


# Simple PPO Model (Actor-Critic)
class SimplePPO(torch.nn.Module):
    def __init__(self, obs_size=16, act_size=5, hidden_size=128):
        super().__init__()
        # Shared backbone
        self.fc1 = torch.nn.Linear(obs_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)

        # Actor (policy)
        self.actor = torch.nn.Linear(hidden_size, act_size)

        # Critic (value)
        self.critic = torch.nn.Linear(hidden_size, 1)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)  # raw logits for discrete actions
        value = self.critic(x)
        return logits, value


# ────────────────────────────────────────────────
# Training Loop
# ────────────────────────────────────────────────
def train():
    # Load config (for hyperparams)
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    hp = config['behaviors']['PepperGreeting']['hyperparameters']

    # Connect to Unity
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name='D:\jozavat\MAS\Semester4\R&D\docs\git\Data\Build\InteractiveRL.exe', side_channels=[channel])  # None = connect to running Unity
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]  # 'PepperGreeting'
    spec = env.behavior_specs[behavior_name]

    # Model + optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePPO(obs_size=16, act_size=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])

    print(f"Starting training on {device}...")
    step = 0
    while step < config['behaviors']['PepperGreeting']['max_steps']:
        # Reset if done
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if step % 100 == 0:
            print(f"Step {step} | Active agents: {len(decision_steps)}")

        if len(terminal_steps) > 0:
            # Handle terminal (episode end) – add to buffer if using full PPO
            pass  # For simplicity, just continue

        # Collect observations
        if len(decision_steps) > 0:
            obs = decision_steps.obs[0][0]  # first agent, vector obs
            obs_tensor = torch.FloatTensor(obs).to(device).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                logits, value = model(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).item()

            # Send action to Unity
            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.array([[action]]))
            env.set_actions(behavior_name, action_tuple)

        # Step environment
        env.step()

        # Get reward/done from new steps
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        reward = decision_steps.reward[0] if len(decision_steps) > 0 else 0
        done = len(terminal_steps) > 0

        # PPO update (simplified – full PPO needs buffer + advantages)
        # we do on-policy update every step (not efficient, but works)
        if step % hp['batch_size'] == 0 and step > 0:
            # Placeholder for loss computation – in real PPO use ratios + clipping
            # simple policy gradient + value loss
            logits, value = model(obs_tensor)
            log_prob = torch.log_softmax(logits, dim=-1)[0, action]
            advantage = reward - value.item()  # crude advantage
            policy_loss = -log_prob * advantage
            value_loss = (reward - value) ** 2
            loss = policy_loss + 0.5 * value_loss  # simple

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        step += 1
        if step % 1000 == 0:
            print(f"Step {step} | Last reward: {reward}")

        if done:
            env.reset()

    env.close()
    torch.save(model.state_dict(), 'pepper_model.pt')


if __name__ == '__main__':
    train()
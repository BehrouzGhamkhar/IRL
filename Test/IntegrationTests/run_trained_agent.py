import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple


# Model Definition (same as training)
class SimplePPO(nn.Module):
    def __init__(self, obs_size=12, act_size=5, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# Agent for Inference Only
class PPOInferenceAgent:
    def __init__(self, model_path, obs_size=12, act_size=5, device='cuda'):

        self.device = device
        self.model = SimplePPO(obs_size, act_size).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        print("Model loaded successfully.")

    def act(self, obs):
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(logits, dim=-1).item()

        # Debug: print what the model sees and decides
        print(f"Obs: {obs[:5].round(2)} | Probs: {probs.cpu().numpy()[0].round(3)} | Action: {action}")

        return action


# Run Unity Environment
def run(model_path):

    # Speed up simulation
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=2.0)

    # If using editor:
    env = UnityEnvironment(file_name=None, side_channels=[channel])

    # If using build:
    # env = UnityEnvironment(file_name="BuildName", side_channels=[channel])

    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]

    agent = PPOInferenceAgent(model_path)

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(decision_steps) == 0:
        print("No agents detected.")
        return

    obs = decision_steps.obs[0][0]

    print("Running trained agent... Press Ctrl+C to stop.")

    try:
        while True:

            # Select action
            action = agent.act(obs)

            # Send to Unity
            action_tuple = ActionTuple()
            action_tuple.add_discrete(np.array([[action]]))
            env.set_actions(behavior_name, action_tuple)

            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Handle episode end
            if len(terminal_steps) > 0:
                print("Episode finished. Resetting environment.")
                env.reset()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                obs = decision_steps.obs[0][0]
            else:
                obs = decision_steps.obs[0][0]

    except KeyboardInterrupt:
        print("Stopped by user.")

    finally:
        env.close()



# Entry Point
if __name__ == "__main__":

    model_path = "../../Python/training/agents/training_logs/run_seed41_config3_fresh_20260528_003340/model_final.pt"
    run(model_path)

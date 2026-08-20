import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

        self.action_counts = np.zeros(act_size, dtype=np.int64)
        self.action_counts_recent = np.zeros(act_size, dtype=np.int64)

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

    def get_value(self, obs):
        """Returns the critic's value estimate for bootstrapping / GAE."""
        obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.actor_critic(obs_tensor)
        return value.item()

    def store_transition(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(0.0 if done else 1.0)
        self.action_counts[action] += 1
        self.action_counts_recent[action] += 1

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

        returns_t      = torch.FloatTensor(returns).to(self.device)
        advantages_t   = torch.FloatTensor(advantages).to(self.device)
        observations_t = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions_t      = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        # store old values so we can clip the value loss
        old_values_t   = torch.FloatTensor(self.values).to(self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(self.observations)
        mini_batch_size = max(1, n // self.num_mini_batch)

        for _ in range(self.ppo_epoch):
            indices = np.random.permutation(n)
            for start in range(0, n, mini_batch_size):
                idx = indices[start: start + mini_batch_size]

                logits, values_pred = self.actor_critic(observations_t[idx])
                values_pred = values_pred.squeeze()
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # clipped value loss (standard PPO)
                values_clipped = old_values_t[idx] + torch.clamp(
                    values_pred - old_values_t[idx],
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    F.mse_loss(values_pred.squeeze(), returns_t[idx].squeeze()),
                    F.mse_loss(values_clipped.squeeze(), returns_t[idx].squeeze())
                )

                loss = (policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
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
        """
        Returns (total_steps, episode_count) stored in the checkpoint so the
        caller can use them as offsets when no CSV resume file is provided.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'entropy_coef' in ckpt:
            self.entropy_coef = ckpt['entropy_coef']
        steps = ckpt.get('total_steps', 0)
        episodes = ckpt.get('episode_count', 0)
        return steps, episodes
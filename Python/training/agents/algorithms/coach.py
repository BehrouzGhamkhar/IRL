import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_


class SimplePPO(nn.Module):
    """
    critic head is kept so PPO checkpoint
    Only the actor head is used during COACH updates.
    """
    def __init__(self, obs_size, act_size, hidden_size):
        super().__init__()
        self.fc1    = nn.Linear(obs_size,   hidden_size)
        self.fc2    = nn.Linear(hidden_size, hidden_size)
        self.actor  = nn.Linear(hidden_size, act_size)
        self.critic = nn.Linear(hidden_size, 1)          # unused by COACH, kept for weight compat
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)


class COACHAgent:
    """
    Interface-compatible with PPOAgent:
        act()             -> returns (action, 0.0, log_prob)   0.0 = fake value
        get_value()       -> returns 0.0                        no critic needed
        store_transition()-> triggers immediate update on non-trivial feedback
        update()          -> no-op  (PPO rollout loop calls this; harmless)
        save() / load()   -> same checkpoint format as PPO

    train.py needs only ONE LINE changed (ALGORITHMS dict).
    The entire training loop runs unchanged.

    Feedback threshold: Only genuine feedback signals

    """

    FEEDBACK_THRESHOLD = 0.05   # rewards with |r| < this are step penalties, ignored

    def __init__(self, cfg, device='cpu'):
        behavior_cfg = cfg['behaviors']['PepperGreeting']
        hp           = behavior_cfg['hyperparameters']
        net          = behavior_cfg['network_settings']
        env_cfg      = cfg['env_settings']

        self.device = device

        obs_size    = env_cfg['obs_size']           # 12
        act_size    = env_cfg['act_size']           # 5
        hidden_size = net['hidden_units']           # must be 128 to match PPO checkpoint

        self.actor_critic = SimplePPO(obs_size, act_size, hidden_size).to(device)
        self.optimizer    = torch.optim.Adam(
            self.actor_critic.parameters(), lr=hp['learning_rate']
        )

        # entropy_coef read from COACH config
        self.entropy_coef  = hp.get('entropy_start', 0.05)
        self.max_grad_norm = behavior_cfg.get('max_grad_norm', 0.5)

        self.action_counts        = np.zeros(act_size, dtype=np.int64)
        self.action_counts_recent = np.zeros(act_size, dtype=np.int64)
        self._updates_done = 0   # for save() diagnostics                        #

    def act(self, obs, deterministic=False):
        """
        Sample an action from the current policy.
        Returns (action, fake_value, log_prob) - fake_value=0.0 keeps
        train.py's  'action, value, log_prob = agent.act(obs)'  working.
        """
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.actor_critic(obs_t)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action   = torch.argmax(probs, dim=-1).item()
                log_prob = 0.0
            else:
                dist     = torch.distributions.Categorical(probs)
                action   = dist.sample().item()
                log_prob = dist.log_prob(
                    torch.tensor(action, device=self.device)
                ).item()
        return action, 0.0, log_prob   # 0.0 = fake critic value

    def get_value(self, obs):
        # train.py calls this for PPO bootstrapping; COACH doesnt need
        return 0.0

    def store_transition(self, obs, action, reward, value, log_prob, done):
        """
        Called every step by train.py.
        Records action counts and fires an immediate COACH update whenever
        """
        self.action_counts[action]        += 1
        self.action_counts_recent[action] += 1

        if abs(reward) > self.FEEDBACK_THRESHOLD:
            self._coach_update(obs, action, reward)

    def update(self, next_value):
        """
        train.py calls this after every rollout.
        COACH updates already happened inside store_transition().
        """
        pass

    def clear_buffers(self):
        # COACH has no rollout buffer.
        pass

    #  COACH gradient step
    def _coach_update(self, obs, action, feedback):
        """
        Single-step policy gradient weighted by human feedback:

            loss = -feedback * log π(action | obs)  -  entropy_coef * H[pi(.|obs)]

        The entropy bonus discourages premature collapse during fine-tuning,
        which is especially important when feedback is sparse (< 200 steps total).
        Positive feedback reinforces the chosen action;
        negative feedback suppresses it and implicitly boosts alternatives.
        """
        obs_t    = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.long, device=self.device)

        logits, _ = self.actor_critic(obs_t)
        dist      = torch.distributions.Categorical(F.softmax(logits, dim=-1))

        log_prob = dist.log_prob(action_t)
        entropy  = dist.entropy()

        loss = -feedback * log_prob - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._updates_done += 1

    #  Checkpoint  (same format as PPOAgent for cross-compatibility)

    def save(self, path, total_steps=0, episode_count=0):
        torch.save({
            'model_state_dict':     self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef':         self.entropy_coef,
            'total_steps':          total_steps,
            'episode_count':        episode_count,
            'coach_updates':        self._updates_done,   # marker: "this is a COACH ckpt"
        }, path)
        print(f"  [COACH] Gradient updates performed: {self._updates_done}")

    def load(self, path):
        """
        Loads a checkpoint.  Two cases:

        1. Loading a PPO checkpoint for fine-tuning (no 'coach_updates' key):
             - Actor + critic weights are restored  (architecture is identical)
             - Optimizer state is NOT restored       (fresh COACH learning rate takes effect)
             - entropy_coef is NOT overwritten       (uses coach_config.yaml value)

        2. Resuming a previous COACH run ('coach_updates' key present):
             - Everything is restored, including optimizer state and entropy_coef.

        Returns (total_steps, episode_count) for train.py's offset tracking.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['model_state_dict'])

        is_coach_ckpt = 'coach_updates' in ckpt

        if is_coach_ckpt:
            # Resuming a COACH session — restore everything
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self._updates_done = ckpt.get('coach_updates', 0)
            if 'entropy_coef' in ckpt:
                self.entropy_coef = ckpt['entropy_coef']
            print(f"  [COACH] Resumed COACH checkpoint  "
                  f"(prior updates: {self._updates_done})")
        else:
            # Loading PPO weights for fine-tuning — fresh optimizer + COACH entropy
            print(f"  [COACH] Loaded PPO weights for fine-tuning  "
                  f"(optimizer reset, entropy_coef={self.entropy_coef:.4f})")

        steps    = ckpt.get('total_steps', 0)
        episodes = ckpt.get('episode_count', 0)
        return steps, episodes

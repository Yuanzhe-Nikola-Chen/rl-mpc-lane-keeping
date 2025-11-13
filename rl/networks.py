import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Simple actor-critic network for continuous actions.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def policy(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std = self.log_std.exp()
        return mean, std

    def value(self, obs: torch.Tensor):
        return self.critic(obs)

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rl.networks import ActorCritic


class PPOAgent:
    """
    Minimal PPO implementation for continuous control.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
    ):
        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size

    def select_action(self, obs_tensor: torch.Tensor):
        with torch.no_grad():
            mean, std = self.ac.policy(obs_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1)
            value = self.ac.value(obs_tensor).squeeze(-1)
        return action, logprob, value

    def compute_advantage(self, rewards, values, dones):
        T = len(rewards)
        adv = torch.zeros(T, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]
            delta = (
                rewards[t]
                + self.gamma * next_value * next_non_terminal
                - values[t]
            )
            last_gae = (
                delta + self.gamma * self.lam * next_non_terminal * last_gae
            )
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def update(self, buffer):
        obs = buffer.obs[: buffer.ptr]
        acts = buffer.acts[: buffer.ptr]
        rews = buffer.rews[: buffer.ptr]
        dones = buffer.dones[: buffer.ptr]
        old_logprobs = buffer.logprobs[: buffer.ptr]
        values = buffer.values[: buffer.ptr]

        adv, returns = self.compute_advantage(rews, values, dones)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                batch_obs = obs[batch_idx]
                batch_acts = acts[batch_idx]
                batch_adv = adv[batch_idx]
                batch_ret = returns[batch_idx]
                batch_old_logp = old_logprobs[batch_idx]

                mean, std = self.ac.policy(batch_obs)
                dist = Normal(mean, std)
                logp = dist.log_prob(batch_acts).sum(-1)
                ratio = torch.exp(logp - batch_old_logp)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.ac.value(batch_obs).squeeze(-1)
                value_loss = nn.functional.mse_loss(value_pred, batch_ret)

                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

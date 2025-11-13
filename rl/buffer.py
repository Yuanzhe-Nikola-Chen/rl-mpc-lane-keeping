import torch


class RolloutBuffer:
    """
    Stores rollouts for PPO updates.
    """

    def __init__(self, size: int, obs_dim: int, act_dim: int, device: str):
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rews = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = size
        self.device = device

    def store(self, obs, act, rew, done, logprob, value):
        i = self.ptr
        self.obs[i] = obs
        self.acts[i] = act
        self.rews[i] = rew
        self.dones[i] = float(done)
        self.logprobs[i] = logprob
        self.values[i] = value
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.size

    def reset(self):
        self.ptr = 0

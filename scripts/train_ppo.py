import torch
import numpy as np

from envs.lane_keeping_env import LaneKeepingEnv
from rl.buffer import RolloutBuffer
from rl.ppo_agent import PPOAgent


def train(
    total_steps: int = 200_000,
    steps_per_rollout: int = 4096,
    device: str = "cpu",
):
    env = LaneKeepingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim, act_dim, device=device)
    buffer = RolloutBuffer(steps_per_rollout, obs_dim, act_dim, device)

    obs, _ = env.reset()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    step = 0
    ep_reward = 0.0
    ep_len = 0

    while step < total_steps:
        buffer.reset()
        while not buffer.is_full():
            with torch.no_grad():
                action, logprob, value = agent.select_action(
                    obs_tensor.unsqueeze(0)
                )
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = env.step(action_np)

            buffer.store(
                obs_tensor,
                torch.as_tensor(
                    action_np, dtype=torch.float32, device=device
                ),
                torch.as_tensor(reward, dtype=torch.float32, device=device),
                torch.as_tensor(done or truncated, dtype=torch.float32, device=device),
                logprob.detach(),
                value.detach(),
            )

            ep_reward += reward
            ep_len += 1
            step += 1

            if done or truncated:
                print(f"Episode done: reward={ep_reward:.1f}, len={ep_len}")
                ep_reward = 0.0
                ep_len = 0
                next_obs, _ = env.reset()

            obs = next_obs
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            )

            if step >= total_steps:
                break

        print(f"Collected {buffer.ptr} steps, updating PPO...")
        agent.update(buffer)

    print("Training completed.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device)

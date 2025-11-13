from envs.lane_keeping_env import LaneKeepingEnv
from mpc.mpc_controller import SimpleMPCManeuver

# Note: For a full comparison, you would also load a trained PPO policy
# and run it here. For now, this script only runs the MPC baseline and
# serves as a placeholder for future RL vs MPC analysis.


def main():
    env = LaneKeepingEnv()
    mpc = SimpleMPCManeuver()

    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action = mpc.control(env.state)
        obs, r, done, truncated, info = env.step(action)
        total_reward += r

    print("MPC baseline total reward (for comparison):", total_reward)


if __name__ == "__main__":
    main()

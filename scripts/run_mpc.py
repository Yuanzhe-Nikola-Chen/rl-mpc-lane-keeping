from envs.lane_keeping_env import LaneKeepingEnv
from mpc.mpc_controller import SimpleMPCManeuver


def run_episode():
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

    print("MPC episode total reward:", total_reward)


if __name__ == "__main__":
    run_episode()

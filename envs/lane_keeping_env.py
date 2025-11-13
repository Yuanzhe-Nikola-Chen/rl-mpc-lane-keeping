import numpy as np
import gym
from gym import spaces

from models.kinematic_bicycle import KinematicBicycleModel


class LaneKeepingEnv(gym.Env):
    """
    Simple lane-keeping environment.

    - Road is along X-axis.
    - Desired lane center at Y = 0.
    - Observation: [y_error, yaw_error, speed].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dt: float = 0.05, episode_length: int = 400):
        super().__init__()

        self.dt = dt
        self.episode_length = episode_length
        self.model = KinematicBicycleModel(dt=dt)

        high_obs = np.array([5.0, np.pi, 35.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float32
        )

        max_steer = np.deg2rad(30.0)
        max_accel = 4.0
        self.action_space = spaces.Box(
            low=np.array([-max_steer, -max_accel], dtype=np.float32),
            high=np.array([max_steer, max_accel], dtype=np.float32),
            dtype=np.float32,
        )

        self.desired_y = 0.0
        self.desired_speed = 15.0

        self.state = None
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        X0 = 0.0
        Y0 = self.np_random.uniform(-1.0, 1.0)
        yaw0 = self.np_random.uniform(-0.1, 0.1)
        v0 = self.np_random.uniform(5.0, 20.0)
        self.state = np.array([X0, Y0, yaw0, v0], dtype=float)

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        X, Y, yaw, v = self.state
        y_err = Y - self.desired_y
        yaw_err = yaw
        return np.array([y_err, yaw_err, v], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        next_state = self.model.step(self.state, np.asarray(action, dtype=float))
        self.state = next_state

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._is_done(obs)
        truncated = self.step_count >= self.episode_length
        info = {}

        return obs, float(reward), done, truncated, info

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        y_err, yaw_err, v = obs
        steer, accel = action

        r_y = - (y_err ** 2)
        r_yaw = - (yaw_err ** 2)
        r_speed = - ((v - self.desired_speed) ** 2) * 0.01
        r_steer = - (steer ** 2) * 0.1
        r_accel = - (accel ** 2) * 0.01

        reward = r_y + r_yaw + r_speed + r_steer + r_accel

        if abs(y_err) > 3.0:
            reward -= 20.0

        return reward

    def _is_done(self, obs: np.ndarray) -> bool:
        y_err, yaw_err, v = obs
        if abs(y_err) > 4.0:
            return True
        return False

    def render(self, mode="human"):
        X, Y, yaw, v = self.state
        print(f"Step {self.step_count}: X={X:.1f}, Y={Y:.2f}, yaw={yaw:.2f}, v={v:.1f}")

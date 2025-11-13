import numpy as np


class KinematicBicycleModel:
    """
    Simple kinematic bicycle model for lane-keeping.

    State x = [X, Y, yaw, v] in world frame.
    Control u = [delta, a].
    """

    def __init__(self, wheelbase: float = 2.7, dt: float = 0.05):
        self.L = wheelbase
        self.dt = dt

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Integrate one step of the kinematic bicycle model.

        Args:
            state: np.array of shape (4,) -> [X, Y, yaw, v]
            control: np.array of shape (2,) -> [delta, a]

        Returns:
            next_state: np.array of shape (4,)
        """
        X, Y, yaw, v = state
        delta, a = control

        max_steer = np.deg2rad(30.0)
        delta = np.clip(delta, -max_steer, max_steer)
        a = np.clip(a, -4.0, 2.0)

        beta = 0.0  # no slip for simple model
        X_next = X + v * np.cos(yaw + beta) * self.dt
        Y_next = Y + v * np.sin(yaw + beta) * self.dt
        yaw_next = yaw + v / self.L * np.tan(delta) * self.dt
        v_next = v + a * self.dt

        v_next = np.clip(v_next, 0.0, 30.0)

        return np.array([X_next, Y_next, yaw_next, v_next], dtype=float)

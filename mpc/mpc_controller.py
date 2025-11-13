import numpy as np

from models.kinematic_bicycle import KinematicBicycleModel


class SimpleMPCManeuver:
    """
    Simple random-shooting MPC for lane-keeping:

    - At each step, sample multiple candidate control sequences.
    - Roll out the model and evaluate quadratic cost.
    - Apply the first control of the best sequence.
    """

    def __init__(self, horizon: int = 15, dt: float = 0.05, num_samples: int = 256):
        self.model = KinematicBicycleModel(dt=dt)
        self.horizon = horizon
        self.num_samples = num_samples

        self.max_steer = np.deg2rad(30.0)
        self.max_accel = 3.0

    def control(
        self,
        state: np.ndarray,
        desired_y: float = 0.0,
        desired_speed: float = 15.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        steer_seq = rng.uniform(
            -self.max_steer, self.max_steer, size=(self.num_samples, self.horizon)
        )
        accel_seq = rng.uniform(
            -1.0, 1.0, size=(self.num_samples, self.horizon)
        )

        best_cost = np.inf
        best_u0 = np.array([0.0, 0.0], dtype=float)

        for i in range(self.num_samples):
            s = state.copy()
            cost = 0.0
            for t in range(self.horizon):
                u = np.array([steer_seq[i, t], accel_seq[i, t]], dtype=float)
                s = self.model.step(s, u)
                X, Y, yaw, v = s
                y_err = Y - desired_y
                yaw_err = yaw

                cost += (
                    y_err ** 2
                    + yaw_err ** 2
                    + 0.01 * (v - desired_speed) ** 2
                    + 0.1 * (u[0] ** 2)
                    + 0.01 * (u[1] ** 2)
                )

                if abs(y_err) > 4.0:
                    cost += 100.0
                    break

            if cost < best_cost:
                best_cost = cost
                best_u0 = np.array([steer_seq[i, 0], accel_seq[i, 0]], dtype=float)

        return best_u0

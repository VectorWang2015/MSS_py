import math
import unittest

import numpy as np

from MSS_simulator_py import OSVDynamics, OSVEnvironment


class TestOSVDynamics(unittest.TestCase):
    def setUp(self):
        self.model = OSVDynamics()
        self.x0 = np.zeros(12)
        self.u0 = np.zeros(6)

    def test_derivative_shape_and_finite(self):
        xdot = self.model.derivatives(self.x0, self.u0, OSVEnvironment())
        self.assertEqual(xdot.shape, (12,))
        self.assertTrue(np.all(np.isfinite(xdot)))

    def test_zero_input_zero_disturbance_equilibrium(self):
        xdot = self.model.derivatives(self.x0, self.u0, OSVEnvironment())
        self.assertTrue(np.allclose(xdot, np.zeros(12), atol=1e-12))

    def test_ned_disturbance_changes_body_dynamics(self):
        x = self.x0.copy()
        x[11] = math.pi / 2.0
        env = OSVEnvironment(
            current_speed=0.0,
            current_direction=0.0,
            tau_env_ned=np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        xdot = self.model.derivatives(x, self.u0, env)
        self.assertGreater(abs(xdot[1]), 1e-8)

    def test_ned_wind_input_changes_dynamics(self):
        env = OSVEnvironment(
            current_speed=0.0,
            current_direction=0.0,
            tau_env_ned=np.zeros(6),
            wind_speed=8.0,
            wind_direction=0.0,
        )
        xdot = self.model.derivatives(self.x0, self.u0, env)
        self.assertGreater(abs(xdot[0]) + abs(xdot[1]) + abs(xdot[5]), 1e-8)

    def test_wind_force_scales_with_speed_squared(self):
        env1 = OSVEnvironment(wind_speed=4.0, wind_direction=0.0)
        env2 = OSVEnvironment(wind_speed=8.0, wind_direction=0.0)
        tau1 = self.model._tau_wind_body(self.x0, env1)
        tau2 = self.model._tau_wind_body(self.x0, env2)
        self.assertGreater(abs(tau1[0]), 0.0)
        self.assertAlmostEqual(tau2[0] / tau1[0], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

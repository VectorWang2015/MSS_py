import unittest

import numpy as np

from MSS_simulator_py.osv.model import OSVDynamics


class TestThrusterConventions(unittest.TestCase):
    def setUp(self):
        self.model = OSVDynamics()

    def test_n1_is_tunnel_thruster_not_surge_propulsor(self):
        tau = self.model._tau_thr(np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(tau[0], 0.0, places=9)
        self.assertGreater(abs(tau[1]), 1.0)

    def test_positive_azimuth_gives_positive_sway_component(self):
        tau = self.model._tau_thr(np.array([0.0, 0.0, 100.0, 0.0, 0.3, 0.0]))
        self.assertGreater(tau[1], 0.0)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import numpy as np

from MSS_simulator_py.osv.wind_simple import (
    OSVWindSimpleParams,
    load_osv_wind_simple_params,
    wind_velocity_ned,
)


class TestWindSimple(unittest.TestCase):
    def test_default_params_reasonable(self):
        p = load_osv_wind_simple_params()
        self.assertIsInstance(p, OSVWindSimpleParams)
        self.assertGreater(p.rho_air, 1.0)
        self.assertGreater(p.afw, 0.0)
        self.assertGreater(p.alw, p.afw)

    def test_ned_wind_velocity_components(self):
        vx = wind_velocity_ned(2.0, 0.0)
        vy = wind_velocity_ned(2.0, math.pi / 2.0)
        self.assertTrue(np.allclose(vx, np.array([2.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(vy, np.array([0.0, 2.0, 0.0]), atol=1e-12))


if __name__ == "__main__":
    unittest.main()

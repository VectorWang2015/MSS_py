import math
import unittest

import numpy as np

from demo.demo_controls import DemoConfig, DemoControlState, apply_action


class TestDemoControls(unittest.TestCase):
    def test_rpm_clipping(self):
        cfg = DemoConfig()
        s = DemoControlState()
        for _ in range(300):
            s = apply_action(s, cfg, "n1_up")
        self.assertAlmostEqual(s.rpm[0], cfg.rpm_limits[0])

    def test_azimuth_clipping(self):
        cfg = DemoConfig()
        s = DemoControlState()
        for _ in range(200):
            s = apply_action(s, cfg, "a1_left")
        self.assertAlmostEqual(s.azimuth[0], -cfg.azimuth_limit)

    def test_disturbance_ned_change(self):
        cfg = DemoConfig()
        s = DemoControlState()
        s = apply_action(s, cfg, "tau_x_plus")
        self.assertGreater(s.tau_env_ned[0], 0.0)

    def test_control_vector_layout(self):
        cfg = DemoConfig()
        s = DemoControlState(
            rpm=np.array([10.0, 20.0, 30.0, 40.0]),
            azimuth=np.array([math.radians(5.0), math.radians(-7.0)]),
        )
        u = s.to_control_vector(cfg)
        self.assertEqual(u.shape, (6,))
        self.assertTrue(np.allclose(u[:4], [10.0, 20.0, 30.0, 40.0]))

    def test_current_direction_key_semantics(self):
        cfg = DemoConfig()
        s = DemoControlState(current_direction=0.0)
        s = apply_action(s, cfg, "current_dir_left")
        self.assertGreater(s.current_direction, 0.0)
        s = apply_action(s, cfg, "current_dir_right")
        self.assertAlmostEqual(s.current_direction, 0.0)

    def test_wind_speed_and_direction_actions(self):
        cfg = DemoConfig()
        s = DemoControlState(wind_speed=0.0, wind_direction=0.0)
        s = apply_action(s, cfg, "wind_speed_up")
        self.assertGreater(s.wind_speed, 0.0)
        s = apply_action(s, cfg, "wind_dir_plus")
        self.assertGreater(s.wind_direction, 0.0)


if __name__ == "__main__":
    unittest.main()

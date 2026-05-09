import unittest

import numpy as np

from dp_env import ActionMaskWrapper, EnvConfig, VesselDPEnv


class TestDPActionWrapper(unittest.TestCase):
    def test_legacy_fixed_azimuth_maps_to_6d(self):
        env = ActionMaskWrapper(VesselDPEnv(EnvConfig()), mode="legacy_4d_fixed_azimuth")
        mapped = env._map_action(np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32))
        self.assertEqual(mapped.shape, (6,))
        self.assertTrue(np.allclose(mapped[:4], np.array([0.1, -0.2, 0.3, -0.4])))

    def test_legacy_mask_bow_sets_bow_zero(self):
        env = ActionMaskWrapper(VesselDPEnv(EnvConfig()), mode="legacy_4d_mask_bow")
        mapped = env._map_action(np.array([0.3, -0.5, 0.2, -0.1], dtype=np.float32))
        self.assertTrue(np.allclose(mapped, np.array([0.0, 0.0, 0.3, -0.5, 0.2, -0.1])))


if __name__ == "__main__":
    unittest.main()

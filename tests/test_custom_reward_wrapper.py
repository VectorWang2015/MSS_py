import sys
import unittest
from pathlib import Path

import numpy as np

from dp_env import ActionMaskWrapper, EnvConfig, VesselDPEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_gym"))
from my_reward import CustomRewardWrapper


class TestCustomRewardWrapper(unittest.TestCase):
    def test_prev_env_action_is_six_dimensional_in_4d_policy_mode(self):
        env = CustomRewardWrapper(
            ActionMaskWrapper(VesselDPEnv(EnvConfig()), mode="legacy_4d_fixed_azimuth")
        )
        env.reset(seed=0)
        self.assertEqual(env._prev_env_action.shape, (6,))

        action = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        self.assertEqual(info["env_action"].shape, (6,))
        self.assertTrue(np.allclose(env._prev_env_action, info["env_action"]))

    def test_custom_reward_receives_base_env(self):
        env = CustomRewardWrapper(
            ActionMaskWrapper(VesselDPEnv(EnvConfig()), mode="legacy_4d_mask_bow")
        )
        env.reset(seed=0)
        self.assertIs(env._base_env, env.unwrapped)
        self.assertTrue(hasattr(env._base_env, "ship_cfg"))
        self.assertTrue(hasattr(env._base_env, "task_cfg"))


if __name__ == "__main__":
    unittest.main()

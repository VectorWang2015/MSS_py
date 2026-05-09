import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ActionMaskWrapper(gym.Wrapper):
    """Wrapper for action masking/adaptation.

    Modes:
      - full_6d: pass-through [n1, n2, n3, n4, a1, a2]
      - legacy_4d_fixed_azimuth: [n1, n2, n3, n4], fix a1/a2 from cfg.ship.fixed_azimuth_deg
      - legacy_4d_mask_bow: [n3, n4, a1, a2], set n1=n2=0
    """

    def __init__(self, env: gym.Env, mode: str = "legacy_4d_fixed_azimuth"):
        super().__init__(env)
        self.mode = mode
        if self.mode == "full_6d":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        elif self.mode in ("legacy_4d_fixed_azimuth", "legacy_4d_mask_bow"):
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action wrapper mode: {self.mode}")

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.mode == "full_6d":
            return a.reshape(6)

        if self.mode == "legacy_4d_fixed_azimuth":
            a = a.reshape(4)
            az_max_deg = float(self.env.ship_cfg.azimuth_max_deg)
            az_fixed_deg = self.env.ship_cfg.fixed_azimuth_deg
            a1_norm = float(az_fixed_deg[0]) / az_max_deg
            a2_norm = float(az_fixed_deg[1]) / az_max_deg
            return np.array([a[0], a[1], a[2], a[3], a1_norm, a2_norm], dtype=float)

        # legacy_4d_mask_bow
        a = a.reshape(4)
        return np.array([0.0, 0.0, a[0], a[1], a[2], a[3]], dtype=float)

    def step(self, action):
        mapped = self._map_action(action)
        obs, reward, terminated, truncated, info = self.env.step(mapped)
        info["policy_action"] = np.asarray(action, dtype=float).copy()
        info["env_action"] = mapped.copy()
        return obs, reward, terminated, truncated, info

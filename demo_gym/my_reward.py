"""Custom reward function and Gymnasium wrapper.

Usage:
    from my_reward import CustomRewardWrapper
    env = CustomRewardWrapper(VesselDPEnv(cfg))

The wrapper intercepts env.step() and replaces the original reward
with compute_custom_reward(). Edit compute_custom_reward() to
experiment with different reward formulations without touching dp_env/.
"""
import math
import gymnasium as gym
import numpy as np


def compute_custom_reward(env, info, prev_action, terminated, truncated):
    """Custom DP reward: distance + yaw

    All state variables are extracted from info dict.  Edit this function
    freely --- the wrapper below always calls it.
    """
    w_distance = -5.0
    w_yaw = 1.0

    dL = info["distance_ship_lengths"]
    dpsi = info["dpsi_rad"]
    u, v, r = info["u"], info["v"], info["r"]

    # --- core terms ---
    dist = w_distance * dL
    yaw = w_yaw * (1.0 + math.cos(dpsi))

    """
    # 仅做展示，如何调取速度，动作来设计reward
    # --- speed penalty (encourage low-speed holding) ---
    speed = math.hypot(u, v)
    speed_penalty = -0.3 * speed

    # --- action smoothness penalty ---
    applied = info["applied_action"]
    dt = env.task_cfg.control_dt
    if len(prev_action) == 0:
        action_rate = 0.0
    else:
        delta = (applied - prev_action) / max(dt, 1e-6)
        action_rate = float(np.mean(delta ** 2))
    action_penalty = -0.001 * action_rate

    total = dist + yaw + speed_penalty + action_penalty
    terms = {
        "dist": dist,
        "yaw": yaw,
        "speed": speed_penalty,
        "action": action_penalty,
    }
    """
    total = dist + yaw
    terms = {
        "dist": dist,
        "yaw": yaw,
    }
    return float(total), terms


class CustomRewardWrapper(gym.Wrapper):
    """gym.Wrapper that replaces the reward using compute_custom_reward.

    Insertion:
        env = VesselDPEnv(cfg)
        env = CustomRewardWrapper(env)    # reward is now overridden

    The original reward and terms in info dict are overwritten.
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_action = np.zeros(4, dtype=np.float32)

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        r, terms = compute_custom_reward(
            self.env, info, self._prev_action.copy(), terminated, truncated
        )
        self._prev_action = info["applied_action"].copy()
        info["reward"] = float(r)
        info["reward_terms"] = terms
        return obs, float(r), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_action = np.zeros(4, dtype=np.float32)
        return obs, info

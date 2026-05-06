"""Default reward function.  Edit this file to tune.

Signature exposes full state so you can freely compose terms from
distance, yaw, velocities, accelerations, and actions.
"""
import math
import numpy as np


def compute_dp_reward(
    d_norm_L: float,
    dpsi: float,
    eta: float,
    u: float,
    v: float,
    r: float,
    du: float,
    dv: float,
    dr: float,
    applied_action: np.ndarray,
    prev_applied_action: np.ndarray,
    dt: float,
    timeout: bool,
) -> tuple[float, dict[str, float]]:
    w_distance = 1.0
    w_yaw = 1.0
    distance_penalty = -w_distance * d_norm_L
    yaw_reward = w_yaw * (1.0 + math.cos(dpsi))
    total = distance_penalty + yaw_reward
    terms = {
        "total": total,
        "distance_penalty": distance_penalty,
        "yaw_reward": yaw_reward,
    }
    return float(total), terms

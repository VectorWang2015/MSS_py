"""Quick demo: compare 6D and 4D action modes for 20 random steps.

Run:
    python demo_action_modes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from dp_env import ActionMaskWrapper, EnvConfig, VesselDPEnv

sys.path.insert(0, str(Path(__file__).resolve().parent / "demo_gym"))
from my_reward import CustomRewardWrapper


def run_rollout(env, label: str, steps: int = 20, seed: int = 7) -> None:
    obs, info = env.reset(seed=seed)
    print(f"\n=== {label} ===")
    print(f"action_space.shape = {env.action_space.shape}")
    print(
        f"t={info['time_s']:.0f}  dL={info['distance_ship_lengths']:.3f}  "
        f"dpsi={info['dpsi_deg']:.1f}deg"
    )

    rng = np.random.default_rng(seed)
    for k in range(1, steps + 1):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if "env_action" in info:
            env_action = np.asarray(info["env_action"], dtype=float)
            mapped = f" mapped={np.array2string(env_action, precision=2, suppress_small=True)}"
        else:
            mapped = ""

        print(
            f"step={k:02d} action={np.array2string(action, precision=2, suppress_small=True)}"
            f" reward={reward:+.3f} dL={info['distance_ship_lengths']:.3f}"
            f" dpsi={info['dpsi_deg']:+6.2f}deg"
            f"{mapped}"
        )

        if terminated or truncated:
            print(f"episode ended: terminated={terminated}, truncated={truncated}")
            break


def main() -> None:
    cfg = EnvConfig()
    cfg.task.timeout_s = 1000.0

    env_6d = CustomRewardWrapper(ActionMaskWrapper(VesselDPEnv(cfg), mode="full_6d"))
    run_rollout(env_6d, label="Full 6D Action Mode", steps=20, seed=11)

    env_4d_fixed = CustomRewardWrapper(
        ActionMaskWrapper(VesselDPEnv(cfg), mode="legacy_4d_fixed_azimuth")
    )
    run_rollout(env_4d_fixed, label="Legacy 4D Fixed-Azimuth Mode", steps=20, seed=11)

    env_4d_mask_bow = CustomRewardWrapper(
        ActionMaskWrapper(VesselDPEnv(cfg), mode="legacy_4d_mask_bow")
    )
    run_rollout(env_4d_mask_bow, label="Legacy 4D Mask-Bow Mode", steps=20, seed=11)


if __name__ == "__main__":
    main()

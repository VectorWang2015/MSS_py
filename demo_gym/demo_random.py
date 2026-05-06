"""Minimal demo: random actions on VesselDPEnv + CustomRewardWrapper."""
import numpy as np
from dp_env import VesselDPEnv, EnvConfig
from my_reward import CustomRewardWrapper

cfg = EnvConfig()
cfg.ship.current_speed = 0.3
cfg.task.timeout_s = 20.0

env = VesselDPEnv(cfg)
env = CustomRewardWrapper(env)
obs, info = env.reset(seed=42)

print(f"=== 初始状态 ===")
print(f"位置: N={info['north']:.1f}m  E={info['east']:.1f}m  Yaw={info['psi_deg']:.1f}°")
print(f"目标: N={info['target_north_m']:.1f}m  E={info['target_east_m']:.1f}m  Yaw={info['target_yaw_deg']:.1f}°")
print(f"距离: {info['distance_m']:.1f}m  艏向误差: {info['dpsi_deg']:.1f}°")
print(f"速度: u={info['u']:+.3f}  v={info['v']:+.3f}  r={info['r']:+.4f}")
print()
print(f"{'t':>4s}  {'action':>24s}  {'d_norm_L':>8s}  {'dpsi°':>7s}  {'u':>7s}  {'v':>7s}  {'reward':>8s}")
print("-" * 80)

total_reward = 0.0
while True:
    action = np.random.uniform(-1, 1, (4,)).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    act_str = f"[{action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f} {action[3]:+.2f}]"
    print(f"{info['time_s']:4.0f}  {act_str:>24s}  {obs[0]:8.4f}  {info['dpsi_deg']:7.1f}  "
          f"{info['u']:+7.3f}  {info['v']:+7.3f}  {reward:+8.4f}")

    assert not terminated
    if truncated:
        break

print("-" * 80)
print(f"总步数: {int(info['time_s'])}  总奖励: {total_reward:.3f}")
print(f"最终距离: {info['distance_m']:.1f}m  最终艏向误差: {info['dpsi_deg']:.1f}°")

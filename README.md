# MSS_simulator_py

这是一个面向 MSS（Marine Systems Simulator）的 Python 仿真仓库，目前已实现 **OSV（Offshore Supply Vessel）的动力学模型，可用于仿真** 。

## 目前包含什么

- `MSS_simulator_py/osv/`
  - OSV 动力学核心模型（参数沿用 MSS `osv.m`）
  -  `custom params`（推荐参数，修改了原参数左推进器，使左右对称）
  - 支持输入：`rpm + azimuth + current + wind + tau_env_ned`
  - 支持输出：`xdot` 与 RK4 单步积分状态
  - 风模型参数与 OSV 主参数隔离：`MSS_simulator_py/osv/wind_simple.py`
- `demo/`
  - `pygame` 交互 demo（放在模块外，便于后续扩展）
  - demo 默认使用 `custom params`
  - 可实时调节推进器、方位角、海流、NED 扰动力
- `tests/`
  - 动力学、坐标约定、推进器约定、demo 几何相关测试

## 快速使用

### 1) 运行测试

在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

### 2) 启动交互 demo（pygame）

在仓库根目录执行：

```bash
python -m demo.demo
```

如果使用你创建的 conda 环境（例如 `osv_demo`）：

```bash
conda run -n osv_demo python -m demo.demo
```

### 3) 在代码中加载参数

```python
from MSS_simulator_py.osv import (
    OSVDynamics,
    load_osv_params,
    load_osv_custom_params,
    load_osv_wind_simple_params,
)

base = load_osv_params()         # 原始参数（贴近 MSS osv.m）
custom = load_osv_custom_params()# 推荐，确保左右推进器对称，仅尾部推进器上限做对称化

wind = load_osv_wind_simple_params()  # OSV风参数（独立于OSV主参数）
model = OSVDynamics(params=custom, wind_params=wind)
```

## 代码结构建议

- 核心动力学扩展：优先在 `MSS_simulator_py/osv/` 内增加
- 新船型：建议新增平级子模块，如 `MSS_simulator_py/<vessel_name>/`
- 交互/可视化：优先放在 `demo/`，避免污染核心模块

## 开发说明（参数加载与逐步仿真调用）

### 参数加载

- 原始参数（贴近 MSS `osv.m`）：`load_osv_params()`
- 自定义参数（当前用于 demo，尾部两台推进器上限对称化）：`load_osv_custom_params()`

### 每步仿真接口

核心类：`OSVDynamics`

- `derivatives(state, control, env) -> xdot`
  - 输入当前状态/控制/环境
  - 返回连续时间导数 `xdot`
- `step_rk4(state, control, dt, env) -> state_next`
  - 输入当前状态/控制/步长/环境
  - 返回离散一步积分后的新状态

### 输入定义

- `state`：`shape=(12,)`
  - `[u, v, w, p, q, r, north, east, down, phi, theta, psi]`
- `control`：`shape=(6,)`
  - `[n1, n2, n3, n4, a1, a2]`
  - `n*` 单位是 RPM，`a*` 单位是 rad
- `env`：`OSVEnvironment`
  - `current_speed`：海流速度 `Vc`（m/s）
  - `current_direction`：海流方向 `beta`（rad, NED）
  - `wind_speed`：风速 `Vw`（m/s, NED）
  - `wind_direction`：风向 `betaVw`（rad, NED，从北顺时针）
  - `tau_env_ned`：外扰广义力 `shape=(6,)`，NED 系 `[X, Y, Z, K, M, N]`

### 输出定义

- `xdot`：`shape=(12,)`，状态导数
- `state_next`：`shape=(12,)`，一步更新状态

### 最小调用示例

```python
import numpy as np

from MSS_simulator_py.osv import OSVDynamics, OSVEnvironment, load_osv_custom_params

params = load_osv_custom_params()  # 或 load_osv_params()
model = OSVDynamics(params=params)

state = np.zeros(12)
control = np.array([0.0, 0.0, 120.0, 120.0, 0.0, 0.0])
env = OSVEnvironment(
    current_speed=0.4,
    current_direction=np.deg2rad(-140.0),
    wind_speed=8.0,
    wind_direction=np.deg2rad(30.0),
    tau_env_ned=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

dt = 0.02
for _ in range(500):
    state = model.step_rk4(state, control, dt, env)
```

## dp_env —— 动态定位 (DP) Gym 环境

`dp_env` 是一个基于 Gymnasium 的船舶动态定位强化学习环境。底层物理模型使用 `MSS_simulator_py` (6DOF OSV83 动力学)，环境输出 11 维观测。

DP 是一个持续控制任务，没有“到达即终止”的逻辑；环境在 `timeout_s` 后 `truncated=True`。

### 观测（11 维）

- `d_norm_L`：控制点到目标点的归一化距离（船长尺度）
- `cos(dpsi), sin(dpsi)`：艏向误差编码
- `cos(eta), sin(eta)`：目标方位角编码（船头坐标系）
- `u, v, r`：速度状态
- `du, dv, dr`：直接由动力学导数得到的加速度项

### 动作（6 维，原生）

- 动作空间：`Box([-1]*6, [1]*6)`
- 映射方式：
  - `action[0:4] -> n1..n4`（按 `n_max` 缩放为 rpm）
  - `action[4:6] -> a1,a2`（按 `azimuth_max_deg` 缩放为角度）

### 动作包装（兼容旧4维策略）

通过 `ActionMaskWrapper` 可选择是否屏蔽部分动作：

- `full_6d`：6维直通
- `legacy_4d_fixed_azimuth`：策略输出 4 维 `[n1,n2,n3,n4]`，`a1/a2` 固定为 `ShipConfig.fixed_azimuth_deg`
- `legacy_4d_mask_bow`：策略输出 4 维 `[n3,n4,a1,a2]`，自动屏蔽 `n1,n2=0`

### 时间步长

- `control_dt = 1.0 s`
- `integration_substeps = 8`
- 实际积分步长：`0.125 s`（RK4）

### 默认奖励

`dp_env/reward.py` 默认奖励为：

`R = -w_distance * d_norm_L + w_yaw * (1 + cos(dpsi))`

如果你要调整奖励，请**只通过** `demo_gym/my_reward.py` 里的 `CustomRewardWrapper` 覆盖奖励。
不建议直接改 `dp_env/reward.py`，该文件应保持为环境默认基线实现。

### 使用示例

```python
import numpy as np
from dp_env import VesselDPEnv, EnvConfig, ActionMaskWrapper
from demo_gym.my_reward import CustomRewardWrapper

cfg = EnvConfig()
env = VesselDPEnv(cfg)
env = ActionMaskWrapper(env, mode="legacy_4d_fixed_azimuth")
env = CustomRewardWrapper(env)

obs, info = env.reset(seed=0)
while True:
    action = np.random.uniform(-1, 1, (4,)).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    if truncated:
        break
```

### 关键配置（`dp_env/config.py`）

- `ShipConfig`：控制点偏移、推进器固定方位角、海流/风参数、观测裁剪
- `TaskConfig`：目标点、目标艏向、起点随机化、超时与积分参数
- `EnvConfig`：`ship + task` 顶层配置容器

### 文件结构

```text
dp_env/
├── __init__.py
├── config.py
├── env.py
└── reward.py
```

## demo_gym（训练与演示脚本）

- `demo_gym/demo_random.py`：随机动作跑环境，打印轨迹/奖励信息
- `demo_gym/my_reward.py`：奖励包装器（推荐且唯一的奖励修改入口）
- `demo_gym/train_ddpg.py`：基于 Tianshou 的 DDPG 训练脚本（默认通过 `ActionMaskWrapper` 保持4维策略接口）

```bash
python demo_gym/demo_random.py
python demo_gym/train_ddpg.py
```

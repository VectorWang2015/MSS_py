# MSS_simulator_py

这是一个面向 MSS（Marine Systems Simulator）的 Python 仿真仓库，目前已实现 **OSV（Offshore Supply Vessel）的动力学模型，可用于仿真** 。

## 目前包含什么

- `MSS_simulator_py/osv/`
  - OSV 动力学核心模型（参数沿用 MSS `osv.m`）
  -  `custom params`（推荐参数，修改了原参数左推进器，使左右对称）
  - 支持输入：`rpm + azimuth + current + tau_env_ned`
  - 支持输出：`xdot` 与 RK4 单步积分状态
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
from MSS_simulator_py.osv import OSVDynamics, load_osv_params, load_osv_custom_params

base = load_osv_params()         # 原始参数（贴近 MSS osv.m）
custom = load_osv_custom_params()# 推荐，确保左右推进器对称，仅尾部推进器上限做对称化

model = OSVDynamics(params=custom)
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
    tau_env_ned=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)

dt = 0.02
for _ in range(500):
    state = model.step_rk4(state, control, dt, env)
```

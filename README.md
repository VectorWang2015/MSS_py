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

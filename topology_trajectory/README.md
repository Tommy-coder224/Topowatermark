# 拓扑引导轨迹水印（Topology-Guided Trajectory Watermark）

水印编在**轨迹的持久表示 D̃(z)** 中；「拓扑引导」= 用目标 λ_u 优化 z，使 D̃(z) ∈ Λ_u。

- **理论**：见项目根目录 `THEORY_TOPOLOGY_GUIDED_TRAJECTORY_WATERMARK.md`
- **实验顺序与设计**：见 `TOPOLOGY_TRAJECTORY_EXPERIMENT_DESIGN.md`

## 实验顺序（先保证拓扑引导能训练）

1. **Phase 1**：可训练性（已跑通）→ `run_phase1.py`
2. **Phase 2**：稳定性/效率（多 seed + 计时/显存）→ `run_phase2_stability.py`
3. **Phase 3**：多用户溯源/可扩展（多 λ_u，最近邻归属）→ `run_phase3_multiuser.py`
4. **Phase 4（可选）**：KL + 鲁棒性（扰动、攻击、trade-off）

## 代码结构（可扩展）

| 模块 | 文件 | 职责 | 可替换为 |
|------|------|------|----------|
| TrajectoryGenerator | `interfaces.py` + `trajectory_diffusers.py` | z → γ(z) | 其他 scheduler/模型 |
| FiltrationBuilder | `interfaces.py` + `filtration_simple.py` | γ → 点云/序列 | 1D 函数、自定义点云 |
| PersistenceLayer | `interfaces.py` + `persistence_simple.py` | → D̃(z) | TopologyLayer、PersLay、持久图像 |
| Embedder | `embedder.py` | 优化 z 使 D̃(z)≈λ_target | 不同优化器、KL 正则 |
| Detector | `interfaces.py` | 表示 + in_region | 逆过程 + 阈值 |

## 运行 Phase 1–3

```bash
# Phase 1：可训练性
python -m topology_trajectory.run_phase1 --height 256 --width 256 --num_inference_steps 15 --steps 300

# Phase 2：稳定性/效率（多 seed）
python -m topology_trajectory.run_phase2_stability --height 256 --width 256 --num_inference_steps 15 --steps 200 --seeds 0 1 2

# Phase 3：多用户溯源（最近邻归属）
python -m topology_trajectory.run_phase3_multiuser --height 256 --width 256 --num_inference_steps 15 --steps 200 --users 8
```

输出默认写入：
- Phase 1 → `output_topology_phase1/`
- Phase 2 → `output_topology_phase2/`
- Phase 3 → `output_topology_phase3/`

## 依赖

- PyTorch
- 真实运行需：`diffusers`、`transformers`
- 可微持久层（可选）：TopologyLayer、torchperslay 等，替换 `persistence_simple.py` 中的 `SimplePersistenceProxy`

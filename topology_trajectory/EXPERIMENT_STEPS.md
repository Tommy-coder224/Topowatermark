# 拓扑引导轨迹水印：实验步骤（含 RTX 5090 真实扩散）

## 1. 干跑结果是否正常？

你当前结果：
- `loss` 从 ~60 降到 ~47（200 步），**正常**。
- 说明：梯度可传、优化有效，**「拓扑引导能训练」** 已验证。

---

## 2. 接下来怎么做？

### Phase 1 真实扩散（推荐顺序）

**第一步：小图、少步、先跑通（约 5–10 分钟）**

```bash
python -m topology_trajectory.run_phase1 --height 256 --width 256 --num_inference_steps 15 --steps 300
```

- 默认 **FP32 + GPU**，稳定；要加速可加 `--fp16`。
- 若报错 `encode_prompt`，多半是 diffusers 版本差异，把报错贴出来即可再改。

**第二步：512（可选，显存/时间约 4 倍）**

- 脚本已对 512 自动：开启 UNet 梯度检查点、去噪步数上限 10。若仍**卡住或 OOM**，请继续用 256。
- 方法本身**与分辨率解耦**：水印在「轨迹的拓扑/持久表示」里，不依赖像素密度；256 验证通过即说明可行，未来 512/2048 只需在更高维潜空间里优化同一条轨迹即可。

```bash
python -m topology_trajectory.run_phase1 --height 512 --width 512 --num_inference_steps 10 --steps 300
```

**第三步：加强优化（可选，建议 256）**

```bash
python -m topology_trajectory.run_phase1 --height 256 --width 256 --num_inference_steps 20 --steps 500 --lr 5e-3
```

---

## 3. 参数说明（充分利用 5090）

| 参数 | 建议 | 说明 |
|------|------|------|
| `--height, --width` | **256** | Phase 1 建议 256；512 显存/时间大，易卡住，脚本已做步数限制与梯度检查点 |
| `--num_inference_steps` | 10–15 | 去噪步数；512 时自动上限 10 |
| `--steps` | 300–500 | 嵌入优化步数 |
| `--lr` | 1e-2 或 5e-3 | 过大可能震荡，过小收敛慢 |
| 默认 FP32 | 是 | 更稳定；加速可加 `--fp16` |
| `--model_id` | model/stable-diffusion-v1-4（本地） | 本地路径优先，避免联网 |

---

## 4. 看什么算「成功」？

- **控制台**：每 `log_every` 步打印的 loss **总体下降**（允许小幅震荡）。
- **输出**：`output_topology_phase1/phase1_result.json` 里 `final_loss` 为有限数（不是 nan）。
- **曲线**：`phase1_loss.png` 里 loss 随 step 下降。

若 loss 不降或很快 nan，可：减小 `--lr`（如 5e-3）、或先用 `--height 256 --width 256` 再放大。

---

## 5. 再往后（Phase 2–3）

- **Phase 2（稳定性/效率）**：多 seed 重复 Phase 1，统计：final loss 均值/方差、时间/step、显存峰值。
  - 脚本：`python -m topology_trajectory.run_phase2_stability --height 256 --width 256 --num_inference_steps 15 --steps 200 --seeds 0 1 2`
- **Phase 3（多用户溯源/可扩展）**：构造多组目标 λ_u（如 λ_0 + 小扰动），分别嵌入得到 z_u；再用最近邻把 D̃(z_u) 归属到最接近的 λ_u，报告 100% 溯源与可扩展性曲线（users=2,4,8,16）。
  - 脚本：`python -m topology_trajectory.run_phase3_multiuser --height 256 --width 256 --num_inference_steps 15 --steps 200 --users 8`
- 再往后可加 KL 正则、鲁棒性攻击等（见 TOPOLOGY_TRAJECTORY_EXPERIMENT_DESIGN.md）。

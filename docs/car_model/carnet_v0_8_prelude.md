# CarNet v0.8 前奏：诊断先行

_撰写：2026-04-24。服务器恢复后按本文档执行，再决定 v0.8 主攻方向。_

---

## 0. 为什么要写这份文档

v0.1→v0.7 共 7 轮，我们基本走的都是 **"猜一个改动 → 跑 150 ep → 看 best metric"** 的循环。这在早期有效（v0.2 补数据、v0.4 加硬 corruption 都大赢），但到 v0.6/v0.7 阶段边际收益明显变小：

- v0.7 终点 `absolute recon chamfer = 0.1015`，和 v0.4 打平
- 距离 Monte-Carlo 噪声地板（~0.01-0.03）还有 3-5× 空间
- 但从 v0.4 → v0.7 的 loss / corruption / 宽度调整全部**没有把 recon 压下去**

继续"调权重、再跑一轮"大概率继续打平。**必须换做法**：用少量诊断实验先定位瓶颈在哪，再针对性地设计 v0.8。

---

## 1. 两个互斥（且都合理）的假设

### 假设 A — Identity-copy attractor 还没撞破
Residual decoder 的形式是 `output = corrupted + delta`。局部最优 `delta ≈ 0.5 × residual` 就能拿到不错的 chamfer，模型停在"只修一半"的位置，因为完全修好的 basin 在全局最优但梯度指向那里的信号被局部最优吸收掉。

**可证伪预期**：模型输出的 delta 方向正确，但幅度不够——`||clean - recon|| < ||clean - corrupted||` 在所有点上成立，但比例只有 ~0.5-0.7。

### 假设 B — Chamfer L1 奖励 mode-averaging
当 corruption 导致某区域（如缺个轮毂）有多种合理恢复，chamfer 的最优解是**条件均值**而不是任意一个具体模式。输出视觉上模糊、平均化。

**可证伪预期**：误差在输入 corruption 最重的区域（hole 中心、大 jitter 点）**不成比例地高**，且 recon 局部点密度偏低（mode-averaging 的特征）。

两者可以同时成立。需要诊断来分清。

---

## 2. 诊断实验清单（服务器恢复后按序执行）

所有实验使用 **v0.7 的 `best_composite.pt`** 作为被测模型。不重训，只重跑 eval 或用小 probe 脚本。

### 实验 ① — Per-corruption 逐一禁用 eval

**目的**：看每种 corruption 单独作用时模型修到什么程度，找出**最致命**的一种。

**协议**：复用 `scripts/car_model/eval_carnet_v0_7.sh`，每次跑前用一个 **临时 model yaml** 改 `corruptions:` 配置，只开一种 corruption（其余 `enabled: false`）。循环 6 次：
- `point_dropout only`（dropout_ratio=0.25）
- `gaussian_jitter only`（sigma=0.06）
- `normal_noise only`（sigma=0.08, flip=0.02）
- `local_hole_mask only`（max_holes=3, radius=0.22）
- `outlier_cluster only`（cluster_size=48）
- `density_imbalance only`

**关键**：eval 时也要关掉 dataloader 里的 SO(3) 旋转（`dynamic_corruption=False`），保证 6 次跑同分布。

**预算**：每次 eval ~2min（test=373 patches），6 次合计 ~15min + 写配置 ~15min = **~30min**

**读数**：`eval/recon_chamfer_l1` 和 `eval/visible_chamfer_l1` 横向对比。
- 如果单一 corruption 能让 chamfer 冲到 0.15 → 它是主要瓶颈
- 如果 6 种都在 0.05-0.07 但叠加时 0.10 → 问题在**叠加而非单项**，建议 iterative refinement

**产物**：`outputs/carnet/v0_7/eval_ablation/per_corruption_sweep.csv`，每行一条 corruption，列：`recon`, `visible`, `hidden`, `gain`

### 实验 ② — Zero-corruption sanity check

**目的**：测模型的"干净输入 → 干净输出"上限。这是一个**必须通过**的健康体检。

**协议**：跑 eval，但**所有 corruption 全部 `enabled: false`**。模型应该直接"复制"输入（因为输入就是 clean）。

**预算**：~2min

**读数**：
- `eval/recon_chamfer_l1` **应该 < 0.03**（MC 噪声地板附近）
- 若 > 0.05 → 模型有**本质缺陷**，residual decoder 没学到恒等映射。这极端情况下说明 encoder/decoder 架构本身不够表达力
- 若 ~0.01-0.03 → 架构健康，问题完全在 corruption→clean 这一步

**这一步 5 分钟就知道要不要重新考虑架构**。

### 实验 ③ — Residual 方向正确性可视化

**目的**：直接看模型 delta 的**方向和幅度**。

**协议**：写一个 probe 脚本（~30 行 Python），对 test 集前 20 个 patch：
1. 跑 v0.7 forward，拿到 `recon_points` 和 `corrupted_points`
2. 对每个点计算：
   - `r_in = ||clean - corrupted||`（per-point 输入误差）
   - `r_out = ||clean - recon||`（per-point 输出误差）
   - `ratio = r_out / max(r_in, 1e-6)`（修复比例：0=完美，1=没修，>1=修反了）
3. 画散点图：x=r_in, y=r_out，y=x 为对角线。  
   理想：所有点在 y=x 下方且靠近 y=0
4. 统计 `ratio` 的分布直方图

**预算**：脚本开发 ~30min + 跑 ~10min = **~40min**

**读数**：
- `ratio` 中位数 0.4-0.6 → 假设 A 成立（修一半）
- `ratio` 中位数 <0.2 → 修得其实不错，问题在少数难点拖累均值
- `ratio` 长尾 > 1 → 有反向修复的点，delta 方向学反了

**产物**：`outputs/carnet/v0_7/eval_ablation/residual_direction_analysis.png` + `.csv`

---

## 3. 诊断 → v0.8 的决策树

```
                        实验 ②: Zero-corruption chamfer
                         │
             < 0.03 ─────┼───── > 0.05
             (架构健康)        (架构有缺陷)
                │                │
                │                ▼
                │       ★★ v0.8 = 换架构（先换 decoder_hidden_dims
                │          [768,384]→[1024,512,256]，再考虑别的）
                │
                ▼
        实验 ① + ③ 结果

        ┌─────────── hole 主导 + ratio 中位数 ~0.5 ──────────┐
        │                                                      │
        │   假设 A + B 同时发生（hole 大区域 + 修一半）         │
        │                                                      │
        │   ★★★ v0.8 = Dense Occupancy Pipeline                │
        │   - 输出 128³ occupancy grid（ConvONet 范式）         │
        │   - 推理 marching cubes → resample 2048 pts           │
        │   - 彻底摆脱 residual decoder 和 chamfer mode-avg     │
        │   - 重训 150 ep ~6h，完整重改代码 ~1 周              │
        │                                                      │
        └──────────────────────────────────────────────────────┘

        ┌─────────── ratio 中位数 ~0.5 + 无单一致命 corruption ─┐
        │                                                        │
        │   假设 A 主导                                           │
        │                                                        │
        │   ★★ v0.8 = Iterative Refinement                       │
        │   - 训练时随机 k=1,2,3 步 self-recursion                │
        │   - 推理 k=2 or 3                                       │
        │   - residual decoder 保留，但每步只做保守修复           │
        │   - 累积修复 > 单步猛修                                  │
        │   - ~3 天开发，重训 1 次                                 │
        │                                                        │
        └────────────────────────────────────────────────────────┘

        ┌─────────── 误差集中在 hole 中心 + 局部点稀疏 ──────────┐
        │                                                        │
        │   假设 B 主导（mode-averaging）                         │
        │                                                        │
        │   ★★ v0.8 = + EMD loss                                  │
        │   - visible_chamfer : EMD (Sinkhorn) = 1 : 0.5          │
        │   - EMD 要求一对一匹配，禁止平均投降                     │
        │   - 训练速度降 20-30%                                   │
        │   - 1 天开发，重训 1 次                                  │
        │                                                        │
        └────────────────────────────────────────────────────────┘
```

---

## 4. 备选（不在诊断分支里但随时可开）

### B1 — 翻 `use_residual_reconstruction` 为 false
单行 config 改。风险：global decoder 表达力弱，visible 可能大退步；收益：彻底破 identity-copy。  
**适用时机**：诊断显示 ratio 长期卡在 0.5 附近且 EMD / iterative 都不管用时作为最后一招。

### B2 — 数据扩到 8-10k patches
Objaverse 的 LVIS "vehicle" 还有更多没下的（我们当前只拉了 ~800 个扩充）。扩到 8-10k 需要：
- 重跑 `materialize_ext.py` 放宽筛选条件
- 重跑 patch builder
- 合并到 `meshfleet_car_cache_v6`
- 重训  
**适用时机**：作为独立、缓慢、稳健的背景任务，不依赖诊断结果，和任何 v0.8 改动正交。

### B3 — Progressive curriculum refinement
不改架构，把 `severity_scale` 从 0 线性升到 1.0 跨 30 ep（而不是当前的 const=1）。  
**适用时机**：B1/B2 都不行时的保守选项。风险低，收益中等。

---

## 5. 时间表建议

| 阶段 | 内容 | 预算 |
|---|---|---|
| **D0（服务器恢复当天）** | 实验 ② (zero-corruption) | 2 min |
| **D0** | 实验 ① (per-corruption) | 30 min |
| **D0** | 实验 ③ (residual probe) | 40 min |
| **D0 晚** | 写诊断报告 + 按决策树选方向 | 1 h |
| **D1-D3** | 实现选中的 v0.8 方向（如 iterative refinement / EMD） | 1-3 天 |
| **D3-D5** | v0.8 训练 150 ep + eval | 5-7 h 训练 + eval |
| **D5 晚** | v0.8 eval 对比 v0.4/v0.7 绝对 chamfer，决定是否再往下走 | 1 h |

**如果 v0.8 选 Dense Occupancy Pipeline**（最大改动），时间表拉长为 1-2 周。

---

## 6. 自检清单：执行时不能犯的错

- [ ] **永远看 absolute chamfer，不看 gain**（v0.6 教训）
- [ ] **诊断前不改 corruption 难度**，否则实验 ① 的横向可比性崩
- [ ] **三个实验都完成之前不启动 v0.8 训练**，不重复"不诊断就改"的错误
- [ ] **v0.8 只做一处主改动**，v0.6 同时改了 corruption + loss 导致归因混乱的教训
- [ ] **实验 ② 不通过（> 0.05）就不要往后走**——架构缺陷是上游问题，调 loss 解决不了

---

_下一步行动：等到"服务器已恢复"的指令，立即执行实验 ②（最便宜的健康体检），再按顺序推进。_

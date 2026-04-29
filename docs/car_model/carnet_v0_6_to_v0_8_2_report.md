# CarNet v0.6 → v0.8.2 全程整合报告

_撰写：2026-04-29。整合此前的 progress report（v0.1-v0.7）、v0.8 prelude/diagnosis、v0.8/v0.8.1/v0.8.2 训练实测。所有数字来自 checkpoint `best_metrics` 与 `outputs/carnet/<ver>/full/history.json`（均为可复现来源）。_

---

## TL;DR

| 版本 | 范式 | best val_recon_chamfer_l1 ↓ | gain ↑ | 备注 |
|---|---|---|---|---|
| **v0.7** | residual decoder | **0.1015** @ ep104 | 0.0436 | 受控消融 baseline，仍是项目 SOTA |
| v0.8 | point-flow matching, K=4, EdgeConv off, pure noise | 0.1263 @ ep52 | 0.0188 | FM 范式首版，已退步 ~24% |
| v0.8.1 | + corrupted warm-start (`x_0=corrupted+0.15z`) + EdgeConv k=16 + K=8 | **0.2169** @ ep11 | **−0.0718** | 严重 regression：训练第一批就是最优，之后没在学 |
| v0.8.2 | 回退 init 为 pure noise，保留 EdgeConv k=16 + K=8 | 0.1200 @ ep76 | 0.0252 | warm-start 是 v0.8.1 的全部元凶 |

**核心结论**：

1. **residual decoder 在 chamfer 监督下塌缩为 ~0.04 的恒定低通滤波器**（v0.8 诊断证实），所以 v0.4/v0.7 都卡在 0.10。
2. **point-space conditional flow matching（FM）在本架构下天花板 ≈ 0.12**，整体落后 residual 范式约 20%。pf_loss 长期 plateau 在 0.4，velocity 回归没真正学好。
3. **EdgeConv k=16 + K=8 增益微弱**（0.128 → 0.120），不足以扭转范式劣势。
4. **corrupted-noise warm-start 是 SNR-崩溃陷阱**：把不可预测的 `−0.15z` 写进 FM 目标，令真正的 `clean−corrupted` 信号被淹没。在 v0.8.1 上单独造成 +0.09 chamfer 退步、−0.072 负 gain。
5. **v0.7 仍是 SOTA**。下一步方向需要超越"在两条 plateau 中选一条更好的"——见 §6。

---

## 1. 数据 / 架构基础（与之前各版本一致）

- **数据**：`meshfleet_car_cache_v5`（MeshFleet 1616 + Objaverse 车辆扩充 ~817 = 2433 patches）。所有 v0.x≥2 共用。
- **架构主干**：v11_latent_flow_hybrid（37M 参数，latent_dim=384，3-layer cross-attention 编码器，32 latent queries）。v0.8 起 decoder 替换为 v12_point_flow_hybrid（39.6M）。
- **辅助 head 不变**：occupancy / free-space / point-defect / corruption-score / intrinsic-difficulty / symmetry / retrieval / latent-flow。

---

## 2. v0.6 — 大胆推进，被 gain 指标骗了

**动机**：v0.4 把 corruption 从温和推到中等已经突破 identity-copy（gain 翻倍）。继续这条路，再加：
- **split chamfer**：`recon_chamfer 1.0 → 0.3`（降级为全局正则），新增 `visible_recon_chamfer 1.0`，保留 `hidden_completion_chamfer 2.0`。
- **更硬 corruption**：dropout 0.25→0.30，jitter 0.06→0.075，hole_radius 0.22→0.26。
- `recon_normal_loss 0.5→0.8`，`symmetry_consistency_loss 0.5→0.8`。

**checkpoint best_metrics**（v0.4 vs v0.6）：

| 指标 | v0.4 | v0.6 |
|---|---|---|
| best_recon ↓ | 0.1015 | 0.1070 |
| best_gain ↑ | 0.0436 | **0.0593 (+36%)** |
| best_paper ↑ | 0.2614 | **0.2884 (+10%)** |

看起来又是大胜——直到对 absolute chamfer：

| eval test set | v0.4 | v0.6 | Δ |
|---|---|---|---|
| recon_chamfer_l1 | 0.1017 | 0.1073 | **−5.5% 退步** |
| visible_chamfer_l1 | 0.1100 | 0.1152 | −4.7% 退步 |
| hidden_chamfer_l1 | 0.1675 | 0.1749 | −4.4% 退步 |

**真相**：corruption 加硬把 `chamfer_before` 从 0.147 推到 0.175，所以 `gain = before − after` 自然膨胀，但 `after` 反而每项都退。视觉上 v0.6 的修复件比 v0.4 明显更糙。

**方法论教训（项目最重要的一条）**：
> **`gain` 在 corruption 不同的实验间不可比。只有 `absolute chamfer_after` 可比。**

---

## 3. v0.7 — 受控消融，验证 split chamfer 不是伪进步

**动机**：v0.6 同时改了 corruption 和 loss 结构两个变量，归因混乱。v0.7 锁住 corruption = v0.4 水平，只保留 v0.6 的 loss 结构（split visible/hidden + symmetry=0.8）。

**结果（150 ep 完整训练，best_metrics）**：

| 指标 | v0.4 final | **v0.7 best @ ep104** |
|---|---|---|
| best_recon ↓ | 0.1015 | **0.1015** |
| best_gain ↑ | 0.0436 | **0.0436** |
| best_visibility ↑ | 0.8804 | 0.8744 |

**结论**：split chamfer 在 corruption 匹配时**精确打平 v0.4**——不是噪音也不是严格进步。v0.7 配置作为后续实验的 baseline 固定下来。

---

## 4. v0.7 诊断：为什么 0.10 是天花板

_完整 diagnosis：`docs/car_model/carnet_v0_8_diagnosis.md`。这里只摘要支撑后续决策的关键证据。_

对 `outputs/carnet/v0_7/full/checkpoints/best_composite.pt` 跑了三个实验：

### 4.1 Zero-corruption sanity（实验 ②）

clean → clean 的"复制题"，模型应该几乎零误差：

| 指标 | 数值 |
|---|---|
| chamfer_before | 0.0000 |
| **recon_chamfer_l1** | **0.0399** |
| visible_recon_chamfer | 0.0506 |

模型对**完全干净的输入仍引入 0.04 的 chamfer**——residual decoder 没学到恒等映射。这个 0.04 是后面所有指标的楼板。

### 4.2 Per-corruption sweep（实验 ①）

| profile | before | after | gain |
|---|---|---|---|
| zero | 0.0000 | 0.0399 | **−0.0399** |
| only_normal_noise | 0.0000 | 0.0398 | −0.0398 |
| only_outlier_cluster | 0.0074 | 0.0410 | −0.0336 |
| only_density_imbalance | 0.0096 | 0.0459 | −0.0362 |
| only_point_dropout | 0.0214 | 0.0537 | −0.0323 |
| only_local_hole_mask | 0.0291 | 0.0616 | −0.0325 |
| **only_gaussian_jitter** | 0.1188 | 0.0844 | **+0.0344** |
| default (all on) | 0.1462 | 0.1015 | +0.0448 |

**6/7 种单 corruption 让模型把输入修得更糟**。**只有 gaussian_jitter 一种**让模型起作用——因为 smoothing 本身就是去 gauss 噪。

### 4.3 Residual 方向 probe（实验 ③）

50 patch × 2048 点，统计 `ratio = ||recon−clean|| / ||corrupted−clean||`：

| 统计 | 值 |
|---|---|
| ratio 中位数 | 0.74 |
| pct(ratio < 0.2)（修得好）| 7.3% |
| pct(0.6 ≤ ratio < 1.0)（修一点点）| 39.8% |
| **pct(ratio ≥ 1.0)（反向修复）** | **23.1%** |

模型平均只去掉 26% 的输入误差，**近 1/4 的点被推得更远了**。

### 4.4 合成诊断

```
模型固有楼板（恒定 smoothing）       ~0.04
gaussian_jitter 残余                ~0.02
hole/dropout/outlier 结构性误差     ~0.04
─────────────────────────────────
total recon ≈ 0.10  ←  v0.7 上界
```

**residual + chamfer 的组合让 `delta=fixed smoothing` 成为局部最优盆地。** 不论再加多少 loss 或 corruption，这个盆地不会自动逃脱。

---

## 5. v0.8 家族 — 改成 point-space flow matching

诊断结论指向"换 decoder 范式，让 loss surface 不再有 identity 盆地"。v0.8 引入新 decoder `v12_point_flow_hybrid`：

- **去掉 residual 旁路**（`use_residual_reconstruction` 强制 false）。
- **decoder = per-point conditional flow matching**：从 `x_0 ~ N(0,I)` 出发，时间 `t∈[0,1]` 下用 K 步 Euler 积分到 `x_1 = clean`；线性插值 `x_t = (1−t)x_0 + t x_1`，目标速度 `v_target = x_1 − x_0`，loss = MSE(v_pred, v_target)。
- 编码器 / 全部辅助 head 不变。
- corruption / lr schedule 与 v0.7 一致。

### 5.1 v0.8 — FM K=4，pure noise，无 EdgeConv

**checkpoint best_metrics @ ep52**：

| 指标 | v0.7 | **v0.8** | Δ |
|---|---|---|---|
| best_recon ↓ | 0.1015 | **0.1263** | +24% 退步 |
| best_gain ↑ | 0.0436 | 0.0188 | −57% |
| best_visibility ↑ | 0.8744 | 0.8539 | −2.3% |
| best_paper ↑ | 0.2557 | 0.1541 | −40% |

**观察**：
- `occupancy_iou` 后续训练攀升超过 v0.7（0.85+）→ **encoder/latent 仍然学得好**。
- `recon_chamfer` plateau 在 0.128，**pf_loss 在 0.46 不再下降**——decoder 的 velocity 回归没收敛。
- 视觉上 recon 像"latent 强加在 noise 上的中性填充"，没有针对性结构。

**诊断**：把 decoder 从纯 N(0,I) 起步直接 generate 出 2048 个干净点，per-point MLP 难以表达精细几何。**问题从 identity 盆地变成了 capacity 不足**。

### 5.2 v0.8.1 — 三处叠加修复（结果：灾难性 regression）

按 v0.8 后的"route A"假设，叠加三个改动：

1. **`point_flow_init_mode: corrupted_noise`**（`x_0 = corrupted + 0.15·z`）— 让 flow 从 corrupted 几何附近出发，不再从纯 N(0,I)。
2. **`point_flow_use_edge_conv: true, k=16`** — DGCNN-style 邻居聚合，给 decoder 显式局部结构 context。
3. **`point_flow_steps: 4 → 8`** — 提高 Euler 分辨率。

**完整 150 ep 结果**：

| 指标 | v0.8 | **v0.8.1** | 备注 |
|---|---|---|---|
| best_recon ↓ | 0.1263 | **0.2169** @ ep11 | **退步 +71%** |
| 终点 recon | 0.128 | **0.2172** | 接近"什么都没学"水平 |
| best_gain ↑ | 0.0188 | **−0.0718** | 输出**比输入更糟** |
| val_pf_loss | ~0.46 | **0.25 (flat 整段训练)** | velocity 回归从未启动 |

`best_recon @ ep11` 意味着训练第一批之后就是最优，之后只有恶化。

**根因（写明在 v0.8.2 config prelude 里）**：当 `x_0 = corrupted + 0.15·z` 时，

```
v_target = x_1 − x_0
        = clean − corrupted − 0.15·z
                ↑              ↑
          真正的 signal     不可预测的随机项
```

`−0.15·z` 是 sample 时新抽的 N(0,I)，**不在任何 condition 中**——decoder 无法预测它。它每维方差 ≈ 0.0225，远大于 `clean − corrupted` 的 per-point 量级（典型 0.05²=0.0025）。**FM 目标的信噪比直接崩溃，loss 收敛到背景方差就停**——这正是观察到的 pf_loss=0.25 平。

注意：单独 `EdgeConv k=16` 或单独 `K=8` 不可能造成这个量级的失败（v0.8.2 后续证实），warm-start 是单点元凶。

### 5.3 v0.8.2 — 隔离实验：回退 warm-start，保留 EdgeConv 与 K=8

**目的**：定位 v0.8.1 三处改动里到底哪一个炸的。最便宜的 falsification 是只回退 init mode：

- `point_flow_init_mode: noise`（回退）
- 保留 `point_flow_use_edge_conv: true, k=16`
- 保留 `point_flow_steps: 8`

**150 ep 完整结果**：

| 指标 | v0.8 | **v0.8.2** | v0.8.1 |
|---|---|---|---|
| best_recon ↓ | 0.1263 | **0.1200 @ ep76** | 0.2169 |
| 终点 recon | 0.128 | 0.1231 | 0.2172 |
| best_gain ↑ | 0.0188 | **+0.0252** | −0.0718 |
| best_iou ↑ | ~0.85 | **0.9171** | 0.8886 |
| 终点 pf_loss | ~0.46 | 0.40~0.45 | 0.25 (flat) |

**判读**：
1. **回退 init 立刻让 chamfer 从 0.217 → 0.120，gain 从 −0.072 → +0.025** ✅ warm-start 是 v0.8.1 的 100% 元凶，假设证实。
2. **EdgeConv k=16 + K=8 净增益 ≈ 0.008**（v0.8 → v0.8.2 0.128→0.120），在噪声范围内，不算明显贡献。
3. **早期 plateau + 轻微 overfitting**：val_recon 在 ep76 触底 0.120 后，之后 70 ep 缓慢退化到 0.123；pf_loss 在 0.4 上下震荡。
4. `recon_to_corrupted_chamfer_l1`（新加的诊断指标）≈ 0.169 vs `corrupted_chamfer_l1` = 0.145——recon 离 corrupted **比 corrupted 离 clean 还远** → 模型不是抄写 corrupted（这点比 v0.7 强），但也没能把 recon 拉到 clean 几何上。

### 5.4 v0.8 家族小结

| 改动 | 单变量贡献 | 是否值得保留 |
|---|---|---|
| residual → FM | **−0.025** chamfer（即 +0.025 退步） | ❌ 在当前架构上 |
| EdgeConv k=16 | +0.008 ≈ 0 | 中性，可保留 |
| K=4 → K=8 | 与 EdgeConv 共同 +0.008 | 中性 |
| corrupted_noise warm-start (scale=0.15) | **−0.097**（灾难） | ❌ 必须删除 |

**FM 范式天花板 ≈ 0.12**，比 residual 范式的 0.10 系统性更差。pf_loss 长期 0.4（理想 < 0.15）说明 velocity 回归本身没学好，原因可能是：
- 从 N(0,I) 直接 generate 2048 点，per-point MLP 表达力不够（capacity bottleneck）。
- linear interpolant + simple v_target 在小 batch / small model 上 SNR 仍偏低。
- 没有 latent-conditional CFG / classifier-free guidance，condition 强度不足。

---

## 6. 关键方法论收获（v0.6→v0.8.2 累积）

1. **gain 不可跨 corruption 难度比较** — v0.6 教训。任何 corruption 设置改动后，唯一可信比较轴是 `absolute chamfer_after`（绝对 recon_chamfer_l1）。
2. **identity-copy 不会被"加 loss/加 head"打破** — 必须改 decoder 范式或 loss surface 形状。v0.6/v0.7 的 split chamfer 不能逃出 0.10 盆地。
3. **诊断 > 猜测**。v0.7 → v0.8 之间的 zero-corruption + per-corruption sweep + residual probe 给出了清晰的 0.04 楼板证据，避免了再花一周推 width / loss 调节。
4. **FM target 不能写入不可预测的项**。`v_target = x_1 − x_0` 中的 `x_0` 必须**完全由 condition 描述**或**完全由 condition 抽样**——任何 sample-time 新抽的随机项进入 target 都会 SNR 崩溃。warm-start 这种"在 corrupted 附近加噪"的直觉要么放进 condition，要么用 deterministic init。
5. **多变量同时改动会污染归因** — v0.6（corruption + loss 同时改）和 v0.8.1（init + EdgeConv + K 同时改）两次都付出代价。新版本必须保证**只有一个 hypothesis 在变**。
6. **early-kill 要慎重**。v0.3 在 ep50 看似死，ep54 ignite。但 v0.8.1 在 ep11 就 best、之后只退化——这是 **real death 不是 long ignition**，应该在 ep30 就 kill 而不是跑满 150。

---

## 7. 数字总表

### 7.1 best_metrics（checkpoint 自身保存的 best）

| version | epoch | best_recon ↓ | best_gain ↑ | best_visibility ↑ | best_composite ↑ | best_paper ↑ |
|---|---|---|---|---|---|---|
| v0.4 | 149 | 0.1015 | **0.0436** | 0.8804 | 0.2948 | 0.2614 |
| v0.6 | 149 | 0.1070 | 0.0593† | 0.8711 | 0.2957 | **0.2884†** |
| **v0.7** | 125 | **0.1015** | 0.0436 | 0.8744 | 0.2924 | 0.2557 |
| v0.8 | 52 | 0.1263 | 0.0188 | 0.8539 | 0.2187 | 0.1541 |
| v0.8.1 | 11 | 0.2169 | −0.0718 | 0.0000‡ | — | — |
| v0.8.2 | 76 | 0.1200 | 0.0252 | 0.9171 | — | — |

† v0.6 的高 gain 是 corruption 加硬带来的伪信号（见 §2）。
‡ v0.8.1 best_recon 在 ep11，那时 occupancy head 还在初始化阶段，IoU 记录为 0。

### 7.2 final-epoch validation（150 ep 跑完时）

| version | val_recon_chamfer_l1 ↓ | val_gain ↑ | val_iou_visible ↑ | val_pf_loss |
|---|---|---|---|---|
| v0.7 (ep125, killed) | ~0.10 | ~0.04 | ~0.92 | n/a |
| v0.8 (ep52, killed) | 0.128 | 0.019 | ~0.85 | ~0.46 |
| v0.8.1 (ep149) | 0.217 | −0.072 | 0.889 | 0.25 (flat) |
| v0.8.2 (ep149) | 0.123 | 0.022 | 0.906 | 0.43 |

### 7.3 v0.7 诊断楼板分解（重申，因为它是 §6 决策的根据）

```
模型固有楼板 (residual + chamfer collapse)   ~0.04
gaussian_jitter 残余                         ~0.02
hole/dropout/outlier 结构性误差              ~0.04
──────────────────────────────────────────────
total v0.7 recon ≈ 0.10
```

---

## 8. 下一步方向选择

### 排除（已被证伪或被证劣）

- ❌ **再调 v0.7 的 loss 权重 / corruption 难度** — 0.04 楼板是架构层 collapse，不是权重问题。
- ❌ **加宽 v0.7 架构** — v0.3/v0.5 两次独立失败。
- ❌ **保留 corrupted_noise warm-start 的当前形式** — SNR 崩溃是结构性问题。
- ❌ **继续在 v0.8 家族里推 K 步数 / EdgeConv 半径** — 在 0.12 plateau 上挪 ±0.01，story 性弱。

### 候选 A — 混合范式 v0.9（推荐）

**思路**：保留 v0.7 residual decoder 作主干（已知能到 0.10），把 FM 作为**轻量 stochastic refinement head**（K=2~4，权重 0.1~0.2），输入是 residual decoder 的输出而不是噪声。

**优点**：
- 拿住 0.10 baseline 不丢。
- FM 已经写好，作为 polish 模块复用现成基建。
- NeurIPS 故事保留生成式贡献。

**风险**：FM 在 residual 之上能否再压 0.01-0.02 chamfer 是开放问题。建议先做诊断 D 再下 commit。

### 候选 B — 先做诊断 D（建议先做）

对 `outputs/carnet/v0_8_2/full/checkpoints/best_composite.pt` 复用 §4 的 `diagnose_carnet` 工具：

1. **Per-corruption sweep on v0.8.2** — 看 FM 在哪类 corruption 上比 residual 强 / 弱。如果 FM 把某类（比如 hole）修得更好，hybrid 思路有据可依。
2. **Per-point error map** — 比较 v0.7 和 v0.8.2 在同一 patch 上的 per-point error 空间分布。如果误差分布**互补**（一个修结构、一个修密度），合并有意义。
3. **pf_loss vs t 分桶** — 看 velocity 回归是哪段 t 区间没学好（早期 / 晚期 / 中段）。如果集中在小 t（接近噪声），用 logit-normal t-sampling 有救。

预算：1 个工作半天。

### 候选 C — 直接回 residual 路线，做 v0.8 系的"反向修复"修复

诊断 §4.3 里 23% 的点被 v0.7 反向修复。如果能找出这部分点的特征（比如点位于 hole 边界 / outlier 周围），用一个轻量 mask 把它们排除在 chamfer 监督之外，理论上能压 0.10 楼板。

预算：1 周。风险中。

### 候选 D — 新数据 / 新 corruption 类型（背景任务）

`materialize_ext.py` 还能多拉 5-7k 个 Objaverse 车辆，得到 v6 cache。和任何代码改动正交，可以挂在 D1-D3 的训练间隙。

---

## 9. 推荐路径

**今天到本周**：候选 B（诊断）→ 根据诊断结果决定走 A（混合）还是 C（mask 修复）。
**下周开始**：A 或 C 的实现 + 训练，目标 absolute recon < 0.095（首次跌破 v0.7 楼板）。
**背景**：D 的数据扩充挂着跑。

---

## 附录 A：完整 config / 启动器索引

| version | model config | train config | launcher |
|---|---|---|---|
| v0.6 | `configs/ss3dm_prior/carnet_v0_6/model_carnet_v0_6.yaml` | `train_carnet_v0_6.yaml` | `scripts/car_model/train_carnet_v0_6.sh` |
| v0.7 | `configs/ss3dm_prior/carnet_v0_7/model_carnet_v0_7.yaml` | `train_carnet_v0_7.yaml` | `scripts/car_model/train_carnet_v0_7.sh` |
| v0.8 | `configs/ss3dm_prior/carnet_v0_8/model_carnet_v0_8.yaml` | `train_carnet_v0_8.yaml` | `scripts/car_model/train_carnet_v0_8.sh` |
| v0.8.1 | `configs/ss3dm_prior/carnet_v0_8_1/model_carnet_v0_8_1.yaml` | `train_carnet_v0_8_1.yaml` | `scripts/car_model/train_carnet_v0_8_1.sh` |
| v0.8.2 | `configs/ss3dm_prior/carnet_v0_8_2/model_carnet_v0_8_2.yaml` | `train_carnet_v0_8_2.yaml` | `scripts/car_model/train_carnet_v0_8_2.sh` |

## 附录 B：相关历史文档

- `docs/car_model/carnet_progress_report_v0_7.md` — v0.1→v0.7 完整 progress report（含视觉素材索引）。
- `docs/car_model/carnet_v0_8_prelude.md` — v0.7→v0.8 之间的诊断方案设计。
- `docs/car_model/carnet_v0_8_diagnosis.md` — v0.7 诊断完整数据（实验①②③ + per-corruption sweep CSV）。

## 附录 C：可复现 wandb / checkpoint

- wandb project: `carnet_v0_2`（共享 project）
- v0.8.2 run: <https://wandb.ai/karamazovaniki-university-of-southern-california/carnet_v0_2/runs/avzav2t2>
- checkpoint paths: `outputs/carnet/{v0_7, v0_8, v0_8_1, v0_8_2}/full/checkpoints/best_*.pt`
- history.json: `outputs/carnet/{v0_8_1, v0_8_2}/full/history.json`（v0.7/v0.8 是早期格式，best_metrics 在 checkpoint 内）

---

_报告结束。_

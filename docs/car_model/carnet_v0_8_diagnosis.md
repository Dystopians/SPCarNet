# CarNet v0.8 诊断报告

_执行时间：2026-04-24。被测：`outputs/carnet/v0_7/full/checkpoints/best_composite.pt`（v0.7 best_composite，150 ep 训练完成）。测试集：`meshfleet_car_cache_v5`，373 个 patch。_

---

## TL;DR

**v0.8 prelude 的两个候选假设都不对，真实情况更糟。**

实测结果表明：v0.7 的"residual decoder"实际上学到的是一个**固定幅度 ~0.04 的低通平滑滤波器**——它不会根据输入的 corruption 类型做针对性修复，而是对任何输入都一视同仁地"平滑一遍"。

- 这个滤波器**只对 Gaussian 噪声有帮助**（因为 smoothing 本来就去噪）
- 对 point_dropout / local_hole / outlier_cluster / density_imbalance 等**结构性 corruption**，它反而把输入搞更差
- 对干净输入（zero corruption），它也硬要平滑一下 → 0.04 的恒定噪声地板

**结论：residual + chamfer 这套架构已经触顶**。调权重、加 loss 不能解决 —— 架构根本没在学"修复"，它学到了最省力的局部最优：平均一下输入。

v0.8 必须**换架构**。prelude 决策树里的 Dense Occupancy Pipeline 是方向，但需要比 prelude 预想的更激进——**直接条件生成**，而不是继续在 residual 旁路里做修补。

---

## 1. 实验 ② — Zero-corruption sanity check

| 指标 | 数值 | Prelude 判据 |
|---|---|---|
| `chamfer_before` | 0.0000 | 预期 |
| `recon_chamfer_l1` | **0.0399** | < 0.03 健康 / 0.03-0.05 边界 / > 0.05 缺陷 |
| `visible_recon_chamfer` | 0.0506 | — |
| `hidden_completion_chamfer` | 0.1167 | — |

**结论**：**处在边界偏向缺陷**。模型对完全 clean 的输入引入了 ~0.04 的 chamfer 误差。这说明 residual decoder 无法保持恒等映射，即便训练时理论上该收敛到 `delta ≈ 0`。这 0.04 就是后面所有指标的**楼板误差**。

---

## 2. 实验 ① — Per-corruption sweep

单开每种 corruption 时的 recon 表现（其余 corruption 全关，LiDAR block 全关）：

| profile              | chamfer_before | chamfer_after | gain        | 评价 |
|----------------------|----------------|---------------|-------------|------|
| zero                 | 0.0000         | 0.0399        | **-0.0399** | 楼板 |
| only_normal_noise    | 0.0000         | 0.0398        | **-0.0398** | 楼板（法向量 corruption 对 xyz 无扰动）|
| only_outlier_cluster | 0.0074         | 0.0410        | **-0.0336** | corruption < 楼板，搞更差 |
| only_density_imbalance| 0.0096        | 0.0459        | **-0.0362** | 搞更差 |
| only_point_dropout   | 0.0214         | 0.0537        | **-0.0323** | 搞更差 |
| only_local_hole_mask | 0.0291         | 0.0616        | **-0.0325** | 搞更差 |
| only_gaussian_jitter | 0.1188         | 0.0844        | **+0.0344** | **唯一**起作用的 |
| **default (all on)** | **0.1462**     | **0.1015**    | **+0.0448** | 全量 |

**观察**：

1. **7 种 corruption 里只有 gaussian_jitter 一个让模型输出比输入好**。
2. **zero 和 only_normal_noise 的 recon 几乎完全一样**（0.0399 vs 0.0398），因为 normal_noise 不改 xyz。这证实了 0.04 是模型的**固有固定输出误差**，与 corruption 内容无关。
3. **点操作类 corruption（dropout / hole / outlier / density）的 chamfer_before 都 ≤ 0.03**，都**小于模型楼板 0.04**。模型直接把这些"容易"的输入搞到 0.04-0.06。
4. **default 全开时 gain 是正的（+0.0448）**，但这 0.045 的 gain 里，≥0.034 是来自 gaussian smoothing，剩下的差不多是楼板的副作用，**不是结构性修复**。

**结论**：v0.7 模型学到的"修复能力"主要是 **Gaussian smoothing**（低通滤波），而**不是**按 corruption 类型定向修复。这解释了为什么：

- 加 point_dropout 难度后，benchmark 上 gain 反而更差（dropout 不是 smoothing 能修的）
- 加 local_hole 后 visible_chamfer 下不去（smoothing 无法填洞）
- outlier 变多时指标飘（smoothing 不把 outlier 拉回来）

---

## 3. 实验 ③ — Residual direction probe（50 patch × 2048 point = 102400 点）

Per-point 分析 `r_in = ||corrupted - clean_NN||`，`r_out = ||recon - clean_NN||`，`ratio = r_out / r_in`。

| 统计 | 数值 |
|---|---|
| r_in mean / median  | 0.0546 / 0.0415 |
| r_out mean / median | 0.0299 / 0.0281 |
| **ratio median** | **0.743** |
| ratio mean | 0.761 |
| ratio q25 / q75 / q95 | 0.45 / 0.98 / 1.42 |
| pct(ratio < 0.2) | 7.3% |
| pct(0.2 ≤ ratio < 0.6) | 29.8% |
| pct(0.6 ≤ ratio < 1.0) | 39.8% |
| **pct(ratio ≥ 1.0)** | **23.1%** |

**观察**：

- **ratio 中位数 0.74**，比 prelude 假设 A 预期的 0.4-0.6 **更差**。模型平均只去掉 26% 的输入误差。
- **23% 的点 ratio ≥ 1.0**：将近四分之一的点被模型"反向修复"（输出比输入离 ground truth 更远）。这和 "固定 smoothing" 的解释一致：smoothing 会把**本来位置正确的点**推离真实位置。
- 只有 7.3% 的点 ratio < 0.2（"修得很好"），远少于"修得几乎没修"（40%）和"修反了"（23%）。

**结论**：delta 方向既不是"一致性地修半步"（prelude 假设 A），也不是"在孔洞中心出错"（prelude 假设 B）。它**就是 smoothing**——对任何点都施加一个向局部均值靠拢的小位移，导致原本正确的点变差、原本偏的点变好一点点。

---

## 4. 合成诊断 → 为什么卡 0.10

| 来源 | 贡献 chamfer |
|---|---|
| 模型固有楼板（smoothing 干净输入都会产生）| **~0.04** |
| Gaussian jitter 未被完全吃掉的残余 | ~0.02 |
| Hole / dropout / outlier 的结构性误差（smoothing 修不了）| ~0.04 |
| **总 recon_chamfer** | **≈ 0.10** |

这不是**"还差一点 fine-tune 能突破"**，是**架构性封顶**。继续调 loss 权重、加 corruption 难度、扩数据，最多把上面每一项压 10-20%，压到 0.08 已是极限，且 paper 意义上没有 novel story。

---

## 5. v0.8 方向选择

### 排除

- ❌ **再调权重**：prelude 里的 EMD / iterative / progressive curriculum 都假设模型"基本对只是不够好"，但实测是**模型根本没学会修复**。
- ❌ **继续 residual decoder**：`recon = corrupted + delta` 这个形式让 `delta=smoothing` 成为局部最优盆地，梯度绕不过去。
- ❌ **Dense Occupancy Pipeline 作为 residual 的旁路**：如果仍让 residual 参与 chamfer 监督，同样会塌缩到 smoothing。

### 推荐方向

**v0.8 = 去掉 residual decoder，改成纯条件生成（conditional point diffusion / flow matching）**

关键点：
1. **彻底移除 `recon = corrupted + delta` 这一支**。decoder 直接从 latent 生成 recon point cloud，不再 anchoring 到 corrupted 上。
2. **保留 encoder（看 corrupted + observed + queries），生成 latent condition**。
3. **decoder 用 flow matching / 小步 diffusion** 从 Gaussian prior 出发，在 latent condition 下生成 clean。
4. **监督用 Chamfer + NN-L1 + symmetry**，但 loss surface 里不再有"identity"盆地，因为没有 residual 旁路了。

次选（更保守）：
- **先复活 Dense Occupancy Pipeline**（128³ grid + MC），看看 voxel 路径能否打破 residual smoothing，但要**同时** `use_residual_reconstruction=false` 切掉 residual 旁路——不然仍然会塌回来。
- 如果 flow matching 风险太高，作为 v0.8.1 中间步骤先跑一版 `use_residual_reconstruction=false` 的 v0.7 配置：理论上 gain 会显著下降，但起码能验证 "去掉 residual" 后模型到底在学什么。

### 不推荐

- **EMD / iterative refinement**（prelude 里 ★★ 那两项）：都在假设 residual decoder 可用，前提错了。

---

## 6. 可验证性

本报告所有数据都可复现：

```
# 环境
CKPT=/data/peilincai/mesh-splatting/outputs/carnet/v0_7/full/checkpoints/best_composite.pt
CACHE=/data/peilincai/mesh-splatting/outputs/ss3dm_prior_car/meshfleet_car_cache_v5
SPLIT=$CACHE/split_meshfleet_car.yaml
OUT=/data/peilincai/mesh-splatting/outputs/carnet/v0_7/eval_ablation

# 实验 ②
CUDA_VISIBLE_DEVICES=1 python -m ss3dm_prior.tools.diagnose_carnet \
  --checkpoint $CKPT --patch_cache_dir $CACHE --split_config $SPLIT \
  --output_dir $OUT --eval_name zero_corruption --profile zero

# 实验 ① （循环 6 个 only_xxx + default）
for p in default only_point_dropout only_gaussian_jitter only_normal_noise \
         only_local_hole_mask only_outlier_cluster only_density_imbalance; do
  CUDA_VISIBLE_DEVICES=1 python -m ss3dm_prior.tools.diagnose_carnet \
    --checkpoint $CKPT --patch_cache_dir $CACHE --split_config $SPLIT \
    --output_dir $OUT --eval_name $p --profile $p
done

# 实验 ③
CUDA_VISIBLE_DEVICES=5 python -m ss3dm_prior.tools.diagnose_carnet \
  --checkpoint $CKPT --patch_cache_dir $CACHE --split_config $SPLIT \
  --output_dir $OUT --eval_name default_probe --profile default \
  --probe_points --probe_max_patches 50
python -m ss3dm_prior.tools.analyze_probe \
  --probe_npz $OUT/default_probe/probe_points.npz \
  --output_dir $OUT/default_probe
```

产物：
- `outputs/carnet/v0_7/eval_ablation/per_corruption_sweep.csv` — 8 个 profile 汇总
- `outputs/carnet/v0_7/eval_ablation/*/summary.json` — 每个 profile 详细
- `outputs/carnet/v0_7/eval_ablation/default_probe/residual_stats.json` — ratio 统计
- `outputs/carnet/v0_7/eval_ablation/default_probe/residual_ratio_histogram.png` — ratio 分布
- `outputs/carnet/v0_7/eval_ablation/default_probe/residual_scatter.png` — r_out vs r_in

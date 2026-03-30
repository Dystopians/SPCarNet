# PRISM-Prune 增量实验报告（2026-03-30）

## 1. 报告目的

本文汇总当前分支在引入 PRISM-Prune 后的**功能增量、创新点、关键代码、参数配置、实验设置、实验结果与问题反思**，用于后续提交给外部模型进行进一步诊断与改良方案设计。

核心验收目标（来自当前项目要求）：

1. FPS 明显优于 baseline  
2. Geometry 指标优于 baseline（Depth + Normal 代理）  
3. 视觉质量不出现明显退化

---

## 2. 本次功能增量总览

### 2.1 训练流程从单阶段变为多阶段 PRISM 状态机

新增 PRISM 流程控制逻辑，训练被拆分为：

1. `GEOMETRY_ACQUISITION`  
2. `STATS_COLLECTION`  
3. `DEAD_PRUNE_ROUND`  
4. `CANDIDATE_PRUNE_ROUND`  
5. `RECOVERY_FINE_TUNE`  
6. `FINAL_FINE_TUNE`

并通过 one-shot prune 调度避免同一轮重复重试。

关键文件：

- `utils/prism_pipeline.py`
- `train.py`

### 2.2 引入多证据三角形打分与状态分类

新增 PRISM scoring，将可见性、梯度敏感度、几何支持、视角多样性、边界信息融合为 `utility/risk/redundancy/prune_score`，并分类：

- `PROTECTED`
- `DEAD`
- `SUSPICIOUS`
- `CANDIDATE`

关键文件：

- `utils/triangle_stats.py`
- `utils/prism_scoring.py`
- `utils/triangle_structure_utils.py`
- `utils/triangle_sparse_support.py`

### 2.3 引入双重安全门与回滚机制

在候选剪枝之前执行 counterfactual gate；在 prune-recovery round 末尾执行全局 validation gate；触发阈值则回滚整轮快照。

关键文件：

- `utils/prism_counterfactual.py`
- `utils/prism_validation.py`
- `train.py`

### 2.4 评估协议升级（更严格）

新增可切换 split 策略：`llff` 与 `file`，并支持 out-of-train split；新增 COLMAP sparse 对齐的几何代理评估脚本（depth + normal proxy）。

关键文件：

- `arguments/__init__.py`
- `scene/__init__.py`
- `scene/dataset_readers.py`
- `create_colmap_outoftrain_split.py`
- `evaluate_geometry_colmap.py`

### 2.5 实验自动化与可复现产物

新增一整套 parking-ground 脚本，覆盖：

- 四组训练实验
- 统一 benchmark
- 定性面板导出
- 自动 4 卡并行调度

关键文件：

- `scripts/parking_ground/run_case.sh`
- `scripts/parking_ground/run_full_practice_suite.sh`
- `scripts/parking_ground/run_full_practice_suite_auto_gpu.sh`
- `scripts/parking_ground/benchmark_prism_runs.py`
- `scripts/parking_ground/make_qualitative_panels.py`

### 2.6 WandB 监控与开销优化

新增去重标量上报与可控日志频率，修复“关闭 fixed views 时 test/train 数值指标也被意外关闭”的回归问题。

关键文件：

- `train.py`（`_wandb_log_filtered` 与 `training_report` 路径）

---

## 3. 创新点（相对此前分支）

1. **多证据 + 风险约束的剪枝决策**：从“单阈值剪枝”升级为 utility/risk/redundancy 联合评分。  
2. **Counterfactual 先验验证**：剪枝前先模拟影响，减少一次性不可逆损伤。  
3. **全局验证回滚门**：将 round 视作事务，失败整轮回滚。  
4. **严格 out-of-train 评估范式**：避免 LLFF 近邻 holdout 带来的“看似泛化、实则近邻插值”。  
5. **工程化实验框架**：四组对照 + 自动 benchmark + 面板导出，降低实验管理成本。  
6. **线上监控可用性提升**：WandB 指标流更稳定，日志开销可控。

---

## 4. 关键代码与职责映射

### 4.1 参数与入口

- `arguments/__init__.py`
  - split 参数：`split_strategy`、`split_file`
  - PRISM 参数：`prism_*`
  - ground 参数：`ground_*`
  - WandB 参数：`wandb_*`

- `train.py`
  - PRISM 主循环集成
  - pruning 触发、round checkpoint、rollback
  - validation gate 调用
  - WandB 过滤上报

### 4.2 PRISM 核心

- `utils/prism_pipeline.py`：阶段调度状态机
- `utils/triangle_stats.py`：EMA 统计收集（visibility/gradient 等）
- `utils/prism_scoring.py`：utility/risk/redundancy 与分类
- `utils/prism_counterfactual.py`：校准集模拟与候选筛选
- `utils/prism_validation.py`：阶段最优比较与回滚判定

### 4.3 结构与稀疏几何支持

- `utils/triangle_structure_utils.py`：边界/非流形/二面角/共面度/QEM-like
- `utils/triangle_sparse_support.py`：COLMAP sparse 局部支持分数
- `utils/colmap_sparse_utils.py`：稀疏点抽取

### 4.4 数据划分与几何评估

- `create_colmap_outoftrain_split.py`：空间分离 train/test(+dropped) 生成
- `scene/dataset_readers.py`：`split_strategy=file` 解析与加载
- `evaluate_geometry_colmap.py`：Depth/Normal proxy 指标计算

### 4.5 实验脚本

- `scripts/parking_ground/run_case.sh`：统一 case 入口（no_prism/grounding/dead_only/full_prism/full_prism_ground_protect）
- `scripts/parking_ground/run_full_practice_suite.sh`：串行全流程
- `scripts/parking_ground/run_full_practice_suite_auto_gpu.sh`：自动选卡并行
- `scripts/parking_ground/benchmark_prism_runs.py`：统一汇总表生成

---

## 5. 关键参数（当前主实验口径）

### 5.1 数据与训练口径

- Scene：`/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix`
- Split：`--split_strategy file --split_file .../split_outoftrain_v1.json`
- Iterations：`30000`
- Test iterations：每 `1000` 一次

### 5.2 PRISM 主参数（来自 `run_case.sh`）

- `--enable_prism_pruning`
- `--prism_collect_stats`
- `--prism_collect_interval 20`
- `--prism_stats_warmup_iters 2000`
- `--prism_geometry_acq_until_iter 12000`
- `--prism_stats_collection_iters 800`
- `--prism_dead_rounds 1`
- `--prism_candidate_rounds 3`
- `--prism_candidate_prune_ratio_per_round 0.015`
- `--prism_dead_prune_ratio 0.005`
- `--prism_recovery_iters 400`
- `--prism_use_counterfactual_gate`
- `--prism_round_checkpoint`
- `--prism_validation_interval 1000`
- `--prism_validation_max_views 32`

`prism_ground` 额外：

- `--prism_use_ground_protect`
- `--prism_use_roi_protect`

### 5.3 回滚阈值默认值（参数层）

- `prism_rollback_absrel_rel_thresh = 0.01`
- `prism_rollback_mean_angle_thresh = 0.4`
- `prism_rollback_psnr_drop_thresh = 0.10`
- `prism_rollback_mae_increase_thresh = 0.003`

### 5.4 WandB 开销控制

- `--wandb_scalar_log_interval 10`
- `--wandb_image_log_interval 5000`
- 可选 `--wandb_disable_fixed_views`

---

## 6. 实验设置（本轮分析所用结果）

主要依据：

1. `benchmarks/prism_parking_ground/20260327_110529/benchmark_summary.md`
2. `benchmarks/prism_parking_ground/20260327_110529/benchmark_results.json`
3. `benchmarks/prism_parking_ground/iter20000_from_training_logs/iter20000_comparison.*`
4. 各 run 的 `wandb-summary.json`
5. 各 run 的 `geometry_eval_colmap/iter_30000_bench.json`
6. `prism_validation/validation_iter_029000.json`

对比组：

- baseline：`pfull_geom_first_no_prism`
- grounding：`pfull_geom_first_grounding`
- prism：`pfull_geom_first_full_prism`
- prism_ground：`pfull_geom_first_full_prism_ground_protect`

---

## 7. 实验结果

### 7.1 30000 最终 benchmark（统一口径）

| Run | PSNR | SSIM | LPIPS | Depth MAE | AbsRel | Delta<1.25 | MeanAngle | AbsCos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 15.1331 | 0.5315 | 0.4936 | 0.4290 | 0.0348 | 0.9679 | 36.7059 | 0.7372 |
| grounding | 15.1287 | 0.5271 | 0.4964 | 0.4978 | 0.0437 | 0.9557 | 40.9574 | 0.6902 |
| prism | 15.2789 | 0.5311 | 0.4943 | 0.4575 | 0.0349 | 0.9691 | 37.2540 | 0.7331 |
| prism_ground | 15.2694 | 0.5295 | 0.4936 | 0.4976 | 0.0364 | 0.9655 | 36.7715 | 0.7336 |

### 7.2 WandB（30000 收敛点）观察

- baseline: `test/psnr 15.2956`, `test/fps 33.41`
- prism: `test/psnr 15.4648`, `test/fps 35.99`
- prism_ground: `test/psnr 15.4772`, `test/fps 35.56`

趋势：PRISM 组在 `test/fps` 与 `test/psnr` 有提升，感知指标变化不大（LPIPS/SSIM 小幅波动）。

### 7.3 20000 中期日志（训练日志抽取）

相对 baseline：

- `prism`：`dPSNR +0.1652`, `dFPS +1.07`, `dL1 -0.00163`
- `prism_ground`：`dPSNR +0.2162`, `dFPS +0.16`, `dL1 -0.00184`

说明中期确有“效率 + 图像”收益趋势。

### 7.4 回滚与安全门行为

`prism` 与 `prism_ground` 均记录到 validation rollback 触发痕迹（benchmark 汇总中的 `RollbackByVal=1`，并有 pre/post round checkpoints 产物）。

---

## 8. 目标达成判定

按“FPS + Geometry + Visual”三重目标判定：

1. **FPS**：达成（PRISM 组优于 baseline）  
2. **Visual（全局）**：基本达成（PSNR 提升，SSIM/LPIPS整体无灾难性退化）  
3. **Geometry（关键）**：**未稳定达成**
   - `prism`：`AbsRel` 与 baseline 接近但略差；`MeanAngle` 更差；`DepthMAE` 更差  
   - `prism_ground`：几何指标整体弱于 baseline

综合：**当前版本是“速度与图像指标改良明显，但几何指标仍未超过 baseline”的部分达成状态。**

---

## 9. 当前主要问题定位（供外部模型重点分析）

### 9.1 Validation gate 的几何信号在部分阶段不可用

在 `validation_iter_029000.json` 中，`absrel` 与 `mean_angle` 为 `NaN`，导致回滚规则主要由 `psnr_drop/mae_increase` 决定，几何约束力度不足。

### 9.2 多目标优化冲突尚未解耦

当前 PRISM 倾向提升渲染效率与整体视觉指标，但对几何代理（尤其法向一致性）有轻微负迁移。

### 9.3 Grounding 分支在该场景下稳定性不如预期

`grounding` 与 `prism_ground` 未在 geometry 上带来正向增益，提示 ground 相关保护/正则在该数据分布上可能引入偏置或噪声。

### 9.4 中期最优未系统固化

此前保存点偏稀（已修正默认 `SAVE_ITERATIONS`），导致若最优出现在 `15000/16000/20000/21000` 附近，后续精评可复现性不足。

---

## 10. 反思与后续改良方向

### 10.1 先保证几何门可观测，再谈权重调优

优先确保 validation views 上可稳定得到 depth/normal 代理，避免 gate 退化为纯图像门。

### 10.2 将“剪枝收益”从全局改为 ROI 分层

对近场关键区（停车位区域）采用更保守 prune 与更严格几何门，远场放宽，提高“局部几何质量 + 全局效率”兼容性。

### 10.3 重新校准评分权重与保护策略

当前 `utility/risk/redundancy` 对法向/深度稳定性的显式约束可能不足；建议增加与几何代理一致的保护项或候选惩罚项。

### 10.4 调整阶段边界与恢复窗口

`geometry_acq_until_iter=12000`、`stats_collection_iters=800`、`recovery_iters=400` 可能过于固定；可做小矩阵搜索观察几何最佳窗口是否晚于当前剪枝时段。

### 10.5 强化“checkpoint 选择策略”

以 `15000/16000/20000/21000/30000` 固定保存并进行统一渲染+几何评估，避免仅依据最后迭代误判方法优劣。

---

## 11. 建议的最小增量消融矩阵（高信息密度）

建议只跑 3 组（控制成本）：

1. **PRISM-GeoGate**：保持当前 PRISM，强化几何可观测与 gate 生效  
2. **PRISM-ROIConservative**：近场更保守、远场正常剪枝  
3. **PRISM-LatePrune**：推迟 candidate prune 起点，观察几何是否恢复

统一评估：

- `15000/16000/20000/21000/30000` 全部 checkpoint
- 固定 split（file）
- 固定渲染与 geometry eval 脚本
- 对比 `PSNR/LPIPS/FPS + AbsRel/Delta1.25/MeanAngle`

---

## 12. 结论（一句话）

当前分支已实现 PRISM 的工程化闭环与明显效率收益，但关键瓶颈仍是“几何指标未稳定超过 baseline”；下一步应优先修复几何门可观测性并做 ROI/阶段化剪枝策略改良。


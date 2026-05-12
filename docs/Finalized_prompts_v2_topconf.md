# Finalized_prompts.md — MeshSplatOpt 下一阶段顶会冲刺执行稿

生成日期：2026-05-06  
面向对象：Codex Coding Agent，在 `Dystopians/SPCarNet` 仓库中逐步实现、验证、记录、提交。  
核心目标：把当前已经很强但仍偏“压缩 + 稀疏深度恢复 + 规则化自适应策略”的 MeshSplatOpt，推进为一个真正有顶会竞争力的、可解释、可验证、可扩展的 **Evidence-Sentinel Certified Mesh Surgery** 方法。

---

## 0. 当前判断：我们站在哪里，距离顶会还差什么

### 0.1 当前已经很强的部分

1. **工程和实验纪律已经明显超过普通研究原型。**  
   仓库已经形成 R-stage / F-stage 的失败记录、强负例、消融、W&B run、独立 `render.py + metrics.py` 与 `evaluate_geometry_colmap.py` 评估路径。F38/F39 gate ablation、F35/F36 no-freeze controls、F37 QEM matched baseline、F34 long continuation rejection 等负例构成了很好的 reviewer-facing honesty backbone。

2. **F82 是当前 accepted multiscene fixed adaptive policy baseline。**  
   F82 v5 在 bonsai / courtyard / room / counter 两个 seed 上共 8/8 all-metric clean wins，且 topology unchanged。它不是手写 per-scene table，而是固定 policy + 固定 recovery recipe，在小场景风险 cap 上从 F79/F80 失败中修正而来。

3. **F47/F48/F49 的 CSEF-family validation-budget package 已经能支撑一个“强 workshop / 中等 conference”级别故事。**  
   5/5 selected scenes 对 clean-long baseline 全部 PSNR、SSIM、LPIPS、AbsRel、Depth MAE 改善；但它依赖 validation-selected budget，因此不能声称 single universal hyperparameter。

4. **F75/F82 证明 adaptive policy 方向不是空想。**  
   F75 在 parking 上以同 topology 打败 F7；F82 在剩余场景完成两 seed 固定策略验证。前期 F65–F67 失败给出了重要启发：render-only evidence 不能当成几何 evidence；area / redundancy primary + render as risk/audit 才稳定。

5. **最新 F95 已经把 courtyard 的主要可见质量问题基本打通。**  
   F95 在 courtyard 上相对 F82 同时改善 PSNR、SSIM、LPIPS、每个固定 per-view PSNR 样本、normal mean angle；但 sparse depth / AbsRel 仍小幅退化，因此 strict parent-Pareto gate 仍拒绝。

### 0.2 当前核心瓶颈

现在真正卡住的不是“能不能训练出更好 RGB”，也不是“能不能避免 per-view PSNR 负样本”。最新瓶颈是：

> **dense render-space depth anchor、sparse COLMAP gate depth、teacher RGB recovery 三者的几何目标不一致。**

F95/F96 说明：

- vertex L2 anchor 不是解；强了会伤 LPIPS/per-view/normal。
- freeze vertices 不是解；depth 可能改善，但 normal / LPIPS 退化。
- global dense render-depth anchor 不是解；F96 增大 render-depth anchor 后反而没有修复 sparse depth。
- teacher render recovery 能改善 RGB/per-view，但会在少量 sparse correspondences 上制造 gate-critical regression。

换句话说，当前失败是 **局部的、correspondence-level 的、gate-specific 的**；继续扫全局 `lambda_sparse_colmap_depth`、`checkpoint_render_depth_anchor_lambda` 或全局 depth anchor，是低效且容易过拟合的。

### 0.3 距离真正顶会还有多远

一个保守评估：

- **工程可复现性 / 实验纪律：75–85%**。已经很强，但还要补更广 benchmark、seed、baselines、runtime、failure taxonomy。
- **顶会方法新意：45–60%**。当前 accepted evidence 更像“高质量 compact recovery policy + sparse-depth supervision + strict freeze”。如果只停在这里，容易被 reviewer 归类为 heuristic engineering 或 careful ablation paper。
- **解决最终问题的能力：50–65%**。compression/recovery 已经不错，但“bidirectional mesh surgery / snap / fill / hole repair”还没有成为 headline load-bearing mechanism；现有非 delete edits 多为 safe diagnostic，而非 full-budget quality win。
- **可投顶会强度：现在约 55–65%；完成本文件的 SCE-Repair + multiscene validation 后可接近 75–85%。**

要跨过门槛，需要做一次方法坐标系转换：

> 不再把优化单位主要看成 triangle、global loss、global prune ratio，而是把 **gate-critical evidence correspondence** 当成一等公民。让每个 sparse COLMAP correspondence、每个 view、每个 local cluster 都成为可以诊断、约束、回滚、解释的 sentinel。

这就是本执行稿提出的新主线：**SCE-Repair: Sentinel Correspondence Evidence Repair**。

---

## 1. 新方法总论：SCE-Repair / Evidence-Sentinel MeshSplatOpt

### 1.1 一句话贡献

**SCE-Repair converts MeshSplatOpt from global compact-recovery tuning into a sentinel-certified local evidence optimizer: every recovery/edit is accepted only if it improves rendering while satisfying one-sided parent-Pareto constraints on the exact sparse correspondences and local evidence clusters that certify geometry.**

中文表达：

> SCE-Repair 把 MeshSplatOpt 从“全局压缩 + 全局恢复 loss 调参”推进为“由证据哨兵驱动的局部几何-外观协同修复”：每个恢复或 edit 都必须在改善 RGB 的同时，对 gate-critical sparse correspondences 满足一侧 parent-Pareto 非退化约束。

### 1.2 方法核心

当前 F95 失败小而关键，因此下一阶段不应继续盲目大扫参数，而应实现：

1. **Sparse-depth regression analyzer**：比较 parent F82 与 candidate F95 在每个 COLMAP sparse correspondence 上的预测深度、相对误差、绝对误差、normal、alpha/coverage、render residual、boundary/occlusion proxy，并聚类出真正造成 gate rejection 的局部区域。

2. **Sentinel cache**：把训练/校准视角中的 high-risk sparse correspondences 固化成 sentinel cache，明确哪些点是“必须不能比 parent 更差”的几何哨兵。

3. **One-sided parent rollback loss**：训练时只惩罚 candidate 相对 parent 在 sentinel 上的退化，而不是把所有 depth 都拉向 parent，也不是把所有 sparse points 都强行拟合 GT。核心形式是：

   ```text
   loss = SmoothL1( ReLU( error_current - error_parent - margin ) )
   ```

   其中 error 可以是 AbsRel、MAE、或二者组合；只在 current 比 parent 明显更差时激活。

4. **Measured recovery policy**：先跑 F95-style short teacher + render-normal anchored recovery；如果 training/calibration sentinel 出现 sparse-depth regression，则自动开启 targeted rollback；如果 sentinel 与 held-out render 都通过，则早停并接受。

5. **Local surgery revival**：在 sentinel clusters 与 CSEF explanation debt 同时高的区域，才重新启用 snap / split / fill / appearance reset。非 delete edits 不再为了“证明能 edit”而 edit，而是由具体 evidence debt 和 gate-critical regression 触发。

### 1.3 顶会故事如何重写

不要把论文主线写成：

> 我们提出一个 CSEF heuristic selector，然后 sparse-depth recovery 让 compact mesh 变好。

应该写成：

> Existing mesh/GS pruning methods optimize primitive count or global reconstruction loss. MeshSplatOpt introduces evidence-sentinel certified mesh surgery: CSEF proposes compact or constructive local edits, while sparse correspondence sentinels enforce one-sided geometry non-regression under counterfactual rendering. This enables aggressive topology reduction and targeted repair without sacrificing sparse geometry certificates.

这能把 novelty 从“调参/规则”拉升到“证据约束形式 + localized certificate + one-sided recovery dynamics”。

---

## 2. 全局执行规则：Codex 每一步必须遵守

1. **每个 Prompt 单独执行、单独写报告、单独 commit。** 不要一次性改完整条链。
2. **不允许用 test split 的 sparse correspondences 参与训练 loss 或 policy 选择。** Test 只用于最终 independent gate/report。
3. **所有 GPU run 必须保留独立评估：** `render.py + metrics.py + evaluate_geometry_colmap.py`。
4. **所有 recovery 必须默认 strict topology freeze：** `--freeze_topology_updates --skip_restricted_delaunay`。
5. **不要覆盖 F82、F95、F96 等已存在结果。** 新输出统一使用 `final_stageSCE*` 或 `stageSCE*` 前缀。
6. **不要隐藏失败。** 如果某个 stage fail，写 failure report 并更新 research log。
7. **所有新增功能必须 opt-in。** 默认训练行为不能改变。
8. **所有训练 run 在 W&B 可用时使用 online logging。** run id 写入报告。
9. **所有 analyzer / gate / loss 都必须支持 smoke test，不依赖 GPU 的部分先用 synthetic tensors 测。**
10. **每次开始前运行：**

```bash
git status --short
python --version
python -m compileall scripts/car_model ss3dm_prior utils -q
```

11. **每次 GPU run 前记录：**

```bash
nvidia-smi
export WANDB_PROJECT=spcarnet_meshprior
export WANDB_MODE=online
```

12. **每个 stage 至少写：**

```text
docs/car_model/<stage_name>_design.md      # 如果涉及新机制
docs/car_model/<stage_name>_report.md      # 实现/实验/失败报告
docs/car_model/SPCarNet_research_log.md    # append-only 研究日志
outputs/carnet/meshsplatopt/<stage_name>/  # 机器可读 artifact
```

---

# Prompt SCE0 — 最新状态锁定与路径审计

```text
You are working inside Dystopians/SPCarNet on branch main or a new branch derived from main.

Mission:
Create a precise source-of-truth audit for the SCE-Repair line. Do not change training code in this stage.

Read and summarize, giving newer documents higher priority:
- README.md
- README.zh.md
- docs/car_model/final_stageF90_F96_lessons_and_bottleneck_report.md
- docs/car_model/final_stageF85_F89_repair_progress_report.md
- docs/car_model/final_stageF82_policy_v5_robustness_report.md
- docs/car_model/final_stageF75_adaptive_policy_reflection_report.md
- docs/car_model/final_stageF47_F48_csef_family_all_metric_repair_report.md
- docs/car_model/SPCarNet_research_log.md
- scripts/car_model/meshsplatopt_run_strict_compact_recovery.py
- evaluate_geometry_colmap.py
- utils/prism_geometry_proxy.py
- utils/prism_counterfactual.py
- utils/prism_scoring.py
- utils/prism_adaptive_policy.py
- ss3dm_prior/meshsplatopt/compact_selector.py
- train.py sections containing sparse COLMAP depth loss, teacher render loss, checkpoint geometry anchor, checkpoint render geometry anchor.

Run:
- git status --short
- git log --oneline -20
- python -m compileall scripts/car_model ss3dm_prior utils -q

Write:
- docs/car_model/final_stageSCE0_state_audit.md

The audit must include:
1. Accepted current baseline: F82 fixed adaptive policy v5.
2. Strongest rejected repair candidate: F95.
3. Exact remaining bottleneck: courtyard sparse depth / AbsRel parent-Pareto failure.
4. Why global lambda sweeps are now low-priority.
5. Code hooks already available:
   - `collect_view_sparse_depth_correspondences`
   - render package `surf_depth`
   - strict compact recovery wrapper
   - checkpoint render geometry anchor
   - optimizer LR overrides
6. Missing code hooks:
   - per-correspondence parent-vs-candidate regression analyzer
   - sentinel cache builder
   - one-sided parent rollback sparse-depth loss
   - sentinel-aware recovery policy and gate
7. Decision: `PROCEED_TO_SCE1` or `STOP`.

Append a short entry to docs/car_model/SPCarNet_research_log.md.
Commit and push if possible.

Gate:
- PASS only if the report explicitly says the next step is per-correspondence sparse-depth analysis before more full recovery runs.
```

---

# Prompt SCE1 — Sparse-depth parent-vs-candidate regression analyzer

```text
Stage SCE0 must be PASS.

Mission:
Build a sparse-depth regression analyzer that compares a parent checkpoint and a candidate checkpoint at the exact COLMAP sparse correspondence level.

This is the most important diagnostic stage. Do not launch new training runs before this analyzer exists.

Design doc:
- docs/car_model/final_stageSCE1_sparse_depth_regression_analyzer_design.md

Implementation files:
- utils/sparse_depth_regression.py
- scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py
- scripts/car_model/smoke_test_stageSCE1_sparse_depth_regression.py

Required CLI:
python scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py \
  --source_path <scene_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_model_path> \
  --parent_iteration 26000 \
  --candidate_model_path <F95_model_path> \
  --candidate_iteration 27000 \
  --split test \
  --max_points_per_view 500 \
  --point_error_max 2.0 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --seed 7 \
  --output_dir outputs/carnet/meshsplatopt/final_stageSCE1_sparse_depth_regression/<scene>

Implementation requirements:
1. Load parent and candidate TriangleModel checkpoints separately.
2. Build the same COLMAP proxy context used by `evaluate_geometry_colmap.py`.
3. For every selected view and every selected sparse correspondence, collect:
   - image_name
   - normalized image key
   - point3D_id if available; if current helper does not return it, extend it.
   - px, py in rendered image coordinates
   - gt_depth
   - parent_pred_depth
   - candidate_pred_depth
   - parent_abs_error
   - candidate_abs_error
   - parent_abs_rel
   - candidate_abs_rel
   - delta_abs_error = candidate_abs_error - parent_abs_error
   - delta_abs_rel = candidate_abs_rel - parent_abs_rel
   - parent_valid / candidate_valid
   - optional parent/candidate normal cosine if `rend_normal` is available
   - optional alpha / coverage if render package provides it
   - optional parent/candidate RGB residual at pixel if GT image is available
4. Aggregate by:
   - global
   - per-view
   - point3D_id
   - depth range bins
   - image boundary bins
   - error quantiles
   - simple 2D pixel connected clusters or k-means clusters for regression points
5. Define regression masks:
   - `regressed_abs`: delta_abs_error > margin_abs
   - `regressed_rel`: delta_abs_rel > margin_rel
   - `gate_critical`: contributes to candidate failing parent-Pareto; default union of top 10% positive delta_abs_rel and top 10% delta_abs_error
6. Do not use test split output for training in later stages. Mark every output with `split=test|train|calibration`.

Outputs:
- correspondence_regressions.csv
- correspondence_regressions.npz
- per_view_regression_summary.csv
- point_regression_summary.csv
- cluster_regression_summary.csv
- sentinel_candidate_mask.npz
- regression_report.md
- regression_summary.json

Smoke test:
- Create synthetic parent/candidate depth arrays and synthetic sparse correspondences.
- Verify deltas, masks, aggregates, and top-k regression clusters are correct.
- Verify invalid candidate depth is counted and reported.

Real diagnostic run:
- Run on courtyard F82 vs F95 if paths exist locally.
- If paths are missing, write exact command contract and mark real run as `PENDING_LOCAL_ARTIFACTS`, not PASS.

Gate:
- PASS if smoke test passes and, when local artifacts exist, the analyzer reproduces the known F95-vs-F82 direction: RGB/per-view not considered here, but sparse depth/AbsRel candidate worse than parent on courtyard.
```

---

# Prompt SCE2 — Train/calibration sentinel cache builder without test leakage

```text
Stage SCE1 must be PASS.

Mission:
Build a sentinel cache for training-time targeted geometry preservation. The cache selects sparse correspondences from train/calibration views only, never from test views.

Design doc:
- docs/car_model/final_stageSCE2_sentinel_cache_design.md

Implementation files:
- utils/sparse_depth_sentinel_cache.py
- scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py
- scripts/car_model/smoke_test_stageSCE2_sentinel_cache.py

Required CLI:
python scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py \
  --source_path <scene_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_model_path> \
  --parent_iteration 26000 \
  --candidate_model_path <optional_candidate_model_path> \
  --candidate_iteration <optional_candidate_iter> \
  --split train \
  --num_views 32 \
  --prefer_hard_views \
  --prefer_observable_views \
  --max_points_per_view 500 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --point_error_max 2.0 \
  --regression_report <optional SCE1 train regression json> \
  --output outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/<scene>/sentinel_cache.npz

Sentinel cache content:
- manifest JSON:
  - source_path
  - split
  - parent model path and iteration
  - candidate model path and iteration if provided
  - view selection reason
  - no_test_leakage=true
- per-camera entries:
  - camera key / image name
  - px, py
  - gt_depth
  - point3D_id
  - parent_pred_depth at cache creation time
  - parent_abs_error
  - parent_abs_rel
  - optional candidate_pred_depth and regression deltas
  - sentinel_weight
  - cluster_id if available
  - is_regressed_candidate if candidate was provided

Selection policy:
1. Always include enough low-COLMAP-error correspondences for stable geometry.
2. Oversample views with high parent/candidate depth risk if candidate regression report exists.
3. Use view-level hard examples from train/calibration only.
4. Cluster-balance the selected correspondences so one dense view or one object region does not dominate.
5. Store all random seeds and sampling parameters.

Smoke test:
- Synthetic views with point ids and known errors.
- Verify no test views are selected when split=train.
- Verify cluster balancing changes weights rather than silently dropping all hard points.

Outputs:
- sentinel_cache.npz
- sentinel_manifest.json
- sentinel_view_summary.csv
- sentinel_report.md

Gate:
- PASS only if cache generation is deterministic under fixed seed and explicitly records `no_test_leakage=true`.
```

---

# Prompt SCE3 — One-sided parent-Pareto sparse-depth rollback loss

```text
Stage SCE2 must be PASS.

Mission:
Implement an opt-in one-sided parent rollback loss in train.py. The loss must activate only when current rendered sparse depth is worse than the parent checkpoint at sentinel correspondences by a configured margin.

Design doc:
- docs/car_model/final_stageSCE3_parent_rollback_loss_design.md

Modify:
- arguments/__init__.py
- train.py
- scripts/car_model/meshsplatopt_run_strict_compact_recovery.py

New train.py flags:
--enable_sparse_depth_parent_rollback_loss
--sparse_depth_parent_rollback_cache <path>
--lambda_sparse_depth_parent_rollback <float>
--sparse_depth_parent_rollback_start_iter <int>
--sparse_depth_parent_rollback_warmup_iters <int>
--sparse_depth_parent_rollback_margin_abs <float>
--sparse_depth_parent_rollback_margin_rel <float>
--sparse_depth_parent_rollback_huber_delta <float>
--sparse_depth_parent_rollback_cluster_balance
--sparse_depth_parent_rollback_max_points_per_view <int>
--sparse_depth_parent_rollback_loss_space absrel|mae|combined

Loss definition:
For each sentinel correspondence i:
- parent_abs = abs(parent_pred_depth_i - gt_depth_i)
- current_abs = abs(current_pred_depth_i - gt_depth_i)
- parent_rel = parent_abs / max(gt_depth_i, eps)
- current_rel = current_abs / max(gt_depth_i, eps)

If loss_space == absrel:
  violation_i = ReLU(current_rel - parent_rel - margin_rel)
If loss_space == mae:
  violation_i = ReLU(current_abs - parent_abs - margin_abs)
If loss_space == combined:
  violation_i = ReLU(current_rel - parent_rel - margin_rel) + beta * ReLU(current_abs - parent_abs - margin_abs)

loss = lambda * weighted_smooth_l1(violation_i, 0)

Critical rules:
1. Do not pull current depth toward parent everywhere.
2. Do not penalize improvements over parent.
3. Do not use test sentinel cache for training; if manifest split is test, raise RuntimeError unless an explicit diagnostic-only override is set.
4. If a camera has no sentinels in cache, skip loss for that iteration.
5. If render package lacks `surf_depth`, log zero active points and do not crash unless strict flag is enabled.
6. Loss must be compatible with strict topology-frozen recovery and current sparse COLMAP depth loss.

Logging to W&B / stdout:
- loss_components/loss_sparse_parent_rollback
- loss_components/loss_sparse_parent_rollback_pure
- sparse_parent_rollback/lambda
- sparse_parent_rollback/active_points
- sparse_parent_rollback/active_fraction
- sparse_parent_rollback/mean_violation_rel
- sparse_parent_rollback/max_violation_rel
- sparse_parent_rollback/mean_violation_abs
- sparse_parent_rollback/max_violation_abs
- sparse_parent_rollback/cache_split

Wrapper changes:
Add corresponding args in `meshsplatopt_run_strict_compact_recovery.py`; include them in recovery_summary.json and exact_train_command.txt.

Smoke tests:
- Unit-test loss on tensors: improved current depth gives zero loss; equal parent gives zero; worse current gives positive; cluster weights affect mean.
- Test split manifest should raise RuntimeError.
- Missing camera key should skip cleanly.

Gate:
- PASS only if the new loss is opt-in, zero-cost when disabled, and unit tests prove it is one-sided.
```

---

# Prompt SCE4 — Sentinel-aware parent-Pareto gate

```text
Stage SCE3 must be PASS.

Mission:
Implement a parent-vs-candidate gate that reports both aggregate parent-Pareto metrics and sentinel-level non-regression before launching or accepting expensive full evaluations.

Design doc:
- docs/car_model/final_stageSCE4_sentinel_gate_design.md

Implementation files:
- scripts/car_model/meshsplatopt_sentinel_parent_pareto_gate.py
- utils/sentinel_parent_pareto_gate.py
- scripts/car_model/smoke_test_stageSCE4_sentinel_gate.py

Required CLI:
python scripts/car_model/meshsplatopt_sentinel_parent_pareto_gate.py \
  --source_path <scene_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <parent_path> \
  --parent_iteration <iter> \
  --candidate_model_path <candidate_path> \
  --candidate_iteration <iter> \
  --sentinel_cache <train_or_calibration_cache.npz> \
  --output_dir <out_dir>

Gate outputs:
- sentinel_parent_pareto_gate.json
- sentinel_per_view_summary.csv
- sentinel_cluster_summary.csv
- sentinel_gate_report.md

Gate checks:
1. Candidate mean sentinel AbsRel <= parent mean sentinel AbsRel + tolerance.
2. Candidate mean sentinel Depth MAE <= parent mean sentinel Depth MAE + tolerance.
3. Worst-view sentinel regression count <= threshold.
4. Top-regression cluster max delta <= threshold or cluster weight below threshold.
5. Optional normal non-regression if normal cache exists.
6. Mark split=train/calibration/test. Test gate is report-only and must not be used to select training hyperparameters.

Smoke test:
- Synthetic parent/candidate arrays with known pass/fail.
- Verify per-view and per-cluster aggregation.

Gate:
- PASS if synthetic pass/fail behavior is correct and real cache gate can run on at least one available local scene or produces exact pending command if artifacts are missing.
```

---

# Prompt SCE5 — F97 diagnostic package: F82 vs F95 correspondence failure map

```text
Stages SCE1-SCE4 must be PASS.

Mission:
Run the analyzer and sentinel gate on the current critical case: courtyard F82 parent vs F95 candidate. This stage is diagnostic only; do not train.

Inputs:
Use the local paths recorded in existing reports if they exist. Expected important paths include:
- F82 courtyard accepted model from final_stageF82 policy v5 evidence
- F95 courtyard render geometry anchor repair model
- outputs/carnet/meshsplatopt/final_stageF95_render_geometry_anchor_repair/courtyard_vs_f82_pareto_gate.json if available locally

Run:
1. SCE1 analyzer on test split for report-only diagnosis.
2. SCE1 analyzer on train/calibration split for training-safe sentinel selection.
3. SCE2 sentinel cache builder from train/calibration views.
4. SCE4 sentinel gate on F82 vs F95 using train/calibration cache.

Write:
- docs/car_model/final_stageSCE5_f82_f95_sparse_regression_diagnostic_report.md

The report must answer:
1. Which views dominate the F95 sparse depth regression?
2. Which correspondence clusters dominate the regression?
3. Are regressions near boundaries, low-alpha/coverage, far-depth regions, high COLMAP error, or occlusion/disocclusion areas?
4. Does the train/calibration sentinel signal predict the test gate failure direction?
5. Is the failure broad or localized?
6. Which loss-space is more aligned with the failure: AbsRel, MAE, or combined?
7. Recommended first rollback lambda and margins for SCE6.

Outputs:
- outputs/carnet/meshsplatopt/final_stageSCE5_f82_f95_sparse_regression_diagnostic/
  - test_regression/
  - train_regression/
  - sentinel_cache/
  - sentinel_gate/
  - SCE6_recommended_command.txt

Decision:
- `PROCEED_TO_SCE6` if regression is localized or sentinel-predictive.
- `STOP_AND_REDESIGN` if train/calibration sentinels do not correlate at all with test sparse-depth failure.

Gate:
- PASS only if the report identifies concrete failing views/clusters and gives one recommended targeted rollback configuration.
```

---

# Prompt SCE6 — First targeted rollback recovery run: F95 visual gains + sparse sentinel non-regression

```text
Stage SCE5 must be PASS and recommend a concrete configuration.

Mission:
Run a small, principled F97/F98 recovery experiment that keeps F95-style visual/normal gains but fixes sparse depth through one-sided parent rollback sentinels.

This is not a broad global sweep. Run at most three configs unless the first config has an obvious implementation error.

Base recipe:
- Parent: F82 courtyard accepted checkpoint, iteration 26000.
- Recovery horizon: 26000 -> 27000 first, not 28000.
- Strict topology freeze: required.
- Keep F95 useful pieces:
  - teacher render lambda around 0.001 if teacher cache exists.
  - sparse COLMAP depth lambda 0.001.
  - checkpoint render normal anchor 0.01.
  - checkpoint render depth anchor either 0.0 or the F95 value, depending on SCE5 diagnosis.
- Add SCE parent rollback loss from train/calibration sentinel cache.

Candidate configs:
1. combined loss, lambda from SCE5 recommendation, small margins.
2. absrel-only if AbsRel dominates failure.
3. mae-only if Depth MAE dominates failure.

For each run:
1. Use `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py` with new SCE args.
2. Render independent test views.
3. Run `metrics.py`.
4. Run `evaluate_geometry_colmap.py --max_points_per_view 500`.
5. Run SCE4 sentinel gate.
6. Run existing parent-Pareto comparison vs F82 if available; otherwise implement a small collector that compares metrics JSONs.

Write:
- docs/car_model/final_stageSCE6_targeted_rollback_recovery_report.md

Required table columns:
- run name
- W&B id
- rollback lambda / loss_space / margins
- PSNR / dPSNR vs F82
- SSIM / dSSIM vs F82
- LPIPS / dLPIPS vs F82
- per-view min dPSNR and negative view count
- AbsRel / dAbsRel vs F82
- Depth MAE / dDepth vs F82
- Normal angle / dNormal vs F82
- sentinel train/calibration pass/fail
- topology unchanged
- decision

Acceptance gate:
A candidate can supersede F95/F82 for courtyard only if:
- topology unchanged
- PSNR, SSIM, LPIPS all improve over F82
- per-view min dPSNR >= 0, or any negative view is smaller than an explicitly documented numerical-noise tolerance
- AbsRel <= F82 AbsRel + tolerance
- Depth MAE <= F82 Depth MAE + tolerance
- Normal angle <= F82 normal angle + tolerance
- sentinel gate passes

Decision:
- `SCE_COURTYARD_PARENT_PARETO_PASS` if a candidate passes.
- `SCE_TARGETED_ROLLBACK_PARTIAL` if visual/per-view pass but geometry still fails with diagnosis.
- `SCE_TARGETED_ROLLBACK_FAIL` if it loses F95 visual gains or does not improve sparse depth.

Gate:
- PASS only if at least one run completes with independent metrics and an honest decision report.
```

---

# Prompt SCE7 — Convert the manual fix into an automatic SCE recovery policy

```text
Stage SCE6 must be PASS or PARTIAL with useful evidence.

Mission:
Convert the successful or most promising SCE6 behavior into a reusable automatic policy. The policy must decide when to activate parent rollback from sentinel evidence, not from manual scene labels.

Design doc:
- docs/car_model/final_stageSCE7_automatic_sce_policy_design.md

Implementation files:
- utils/sce_recovery_policy.py
- scripts/car_model/meshsplatopt_run_sce_policy_recovery.py
- scripts/car_model/smoke_test_stageSCE7_sce_policy.py

Policy flow:
1. Start from compact/adaptive F82-style checkpoint.
2. Build or load train/calibration sentinel cache.
3. Run a short visual recovery probe with teacher/render-normal anchor for N_probe iterations, or evaluate an existing F95-style candidate if provided.
4. Run sentinel gate against parent.
5. If sentinel is non-degrading, continue visual recovery or early stop.
6. If sentinel degrades, activate one-sided parent rollback loss on regressed sentinel clusters only.
7. Monitor sentinel and RGB proxy every K iterations.
8. Early stop when RGB improves and sentinel non-regression holds.
9. Reject if sentinel improves only by destroying RGB/per-view metrics.

New CLI should support:
python scripts/car_model/meshsplatopt_run_sce_policy_recovery.py \
  --source_path ... \
  --output_path ... \
  --load_iteration 26000 \
  --final_iteration 27000 \
  --sentinel_cache ... \
  --parent_model_path ... \
  --policy sce_v1 \
  --execute

Policy config fields:
- visual_probe_iters
- teacher_render_lambda
- sparse_lambda
- lpips_lambda
- render_normal_anchor_lambda
- render_depth_anchor_lambda
- sentinel_check_interval
- rollback_activation_absrel_delta
- rollback_activation_depth_delta
- rollback_lambda_base
- rollback_loss_space
- rollback_cluster_top_k
- early_stop_patience

Outputs:
- sce_policy_decision.json
- exact_train_command.txt
- sentinel_gate_history.csv
- recovery_summary.json
- topology_audit.json
- policy_report.md

Smoke test:
- Synthetic metric history where sentinel degrades should activate rollback.
- Synthetic metric history where sentinel passes should not activate rollback.
- Early stop should pick the best parent-Pareto point, not the last point.

Gate:
- PASS only if the policy is scene-agnostic and stores every decision reason.
```

---

# Prompt SCE8 — Multiscene fixed-policy validation of SCE-Repair

```text
Stage SCE7 must be PASS.

Mission:
Run SCE policy v1 on the current fixed-policy validation suite without per-scene retuning.

Scenes:
- bonsai
- courtyard
- room
- counter
- parking_phone_tiny if local budget allows

Seeds:
- train_seed=0
- train_seed=1
- optional train_seed=2 for final paper margin

Baselines to compare:
1. Clean-long 22k baseline.
2. F48/F49 validation-budget CSEF-family selected row where available.
3. F75 parking headline where applicable.
4. F82 fixed adaptive policy v5.
5. F95/F96-style global render anchor candidate for courtyard if relevant.
6. Random same-count and QEM rows where already available.

Rules:
- No per-scene policy parameter changes.
- Scene-specific sentinel cache is allowed because it is built from each scene's own train/calibration evidence, but policy thresholds must remain fixed.
- Test split only for final independent report.

Write collector:
- scripts/car_model/final_collect_stageSCE8_multiscene_policy.py

Write report:
- docs/car_model/final_stageSCE8_multiscene_sce_policy_report.md

Required tables:
1. Per-scene, per-seed metrics vs clean-long.
2. Per-scene, per-seed metrics vs F82 parent.
3. Topology reduction and topology unchanged audit.
4. Sentinel activation statistics.
5. Failure table for any scene/seed that does not pass.
6. Runtime and memory overhead of SCE analyzer/cache/loss.

Decision labels:
- `SCE_POLICY_V1_TWO_SEED_PASS` if all selected scenes pass all metrics vs F82 or clean-long as defined.
- `SCE_POLICY_V1_RENDER_PASS_GEOMETRY_MIXED` if RGB improves but sparse geometry mixed.
- `SCE_POLICY_V1_FAIL` if not competitive.

Gate:
- PASS only if the report includes both successes and failures, and explicitly states whether SCE v1 supersedes F82 or remains a repair module for courtyard-like failures.
```

---

# Prompt SCE9 — Sentinel-guided local mesh surgery revival: snap / split / fill only where evidence demands it

```text
Stage SCE8 must be PASS or produce a clear localized failure taxonomy.

Mission:
Revive the bidirectional mesh surgery promise using SCE evidence. Do not force snap/fill globally. Trigger non-delete edits only where sentinel regression clusters and CSEF explanation debt agree.

Design doc:
- docs/car_model/final_stageSCE9_sentinel_guided_local_surgery_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/sce_local_surgery.py
- scripts/car_model/meshsplatopt_make_sce_local_surgery_proposals.py
- scripts/car_model/meshsplatopt_apply_sce_local_surgery.py
- scripts/car_model/smoke_test_stageSCE9_sce_local_surgery.py

Proposal trigger:
A region becomes eligible only if:
1. It contains a high-weight sentinel regression cluster or high CSEF explanation debt.
2. It has enough local evidence to avoid hallucination.
3. It passes free-space / boundary / sparse support pre-checks.
4. It is not simply solved by one-sided rollback recovery.

Allowed local operations:
- SNAP_VERTICES: for local depth/normal mismatch with supported plane or sparse surface.
- SPLIT_TRIANGLES: for under-resolved local geometry where sentinel depth varies inside large triangles.
- FILL_PATCH: for boundary/void debt with enough support.
- APPEARANCE_RESET: for regions where geometry is fixed but radiance is ghosted.
- PROTECT: for parent-good sparse geometry that recovery keeps trying to damage.

Every proposal must record:
- linked sentinel ids
- linked CSEF region ids
- evidence summary
- free-space risk
- prior-only flag
- expected topology cost
- rollback snapshot

Smoke tests:
1. Synthetic dented plane with sentinel depth regression -> SNAP proposal accepted by synthetic gate.
2. Large triangle with varying sparse depths -> SPLIT proposal lowers sentinel error.
3. Supported hole -> FILL proposal accepted.
4. Unknown unobserved void -> rejected in normal mode.
5. Harmful edit rolls back exactly.

Real diagnostic:
- Apply only to one scene/cluster identified by SCE5/SCE8.
- Run render-backed counterfactual gate before any recovery.
- Run short SCE recovery after accepted edit.

Report:
- docs/car_model/final_stageSCE9_local_surgery_report.md

Gate:
- PASS only if at least one non-delete edit is beneficial on synthetic smoke and at least one real-scene proposal is either honestly accepted or rejected with a useful certificate.
```

---

# Prompt SCE10 — Top-conference ablation and reviewer-risk package

```text
Stage SCE8 and, if possible, SCE9 must be complete.

Mission:
Build the final ablation package that can survive top-conference review.

Write collector scripts:
- scripts/car_model/final_collect_stageSCE10_ablation_package.py
- scripts/car_model/final_build_stageSCE10_tables.py
- scripts/car_model/final_build_stageSCE10_qualitative_gallery.py

Ablations required:
1. clean-long baseline
2. validation-budget CSEF-family F48/F49
3. F82 fixed adaptive policy v5
4. F95 render geometry anchor without sentinel rollback
5. F96 stronger global dense depth anchor
6. SCE rollback loss without teacher/render-normal anchor
7. SCE rollback loss with global all-sparse points, no sentinel selection
8. SCE rollback one-sided hinge vs symmetric parent depth L2
9. no sparse COLMAP depth loss
10. no LPIPS tiny loss
11. no strict topology freeze
12. random same-count compaction
13. QEM matched baseline where available
14. SCE local surgery disabled vs enabled where SCE9 exists

Metrics:
- PSNR
- SSIM
- LPIPS
- AbsRel
- Depth MAE
- Normal angle
- per-view min PSNR delta
- negative view count
- topology triangles / vertices
- reduction percentage
- runtime and memory
- W&B run id
- decision label

Reports:
- docs/car_model/final_stageSCE10_ablation_package_report.md
- docs/car_model/final_stageSCE10_reviewer_risk_checklist.md
- docs/car_model/final_stageSCE10_claims_and_limitations.md

Reviewer-risk checklist must explicitly answer:
1. Is the method only a heuristic policy? Explain SCE one-sided sentinel constraint as the principled component.
2. Does it leak test geometry into training? Show split manifests and no_test_leakage checks.
3. Is F82 already enough? Show where SCE improves or where it is a targeted repair module.
4. Does global depth anchoring solve the problem? Compare F96 and SCE.
5. Are non-delete edits actually load-bearing? If not, state they are optional diagnostics and do not overclaim.
6. Are selected scenes cherry-picked? Provide all attempted scenes and failures.
7. Are metrics independent? Confirm render.py + metrics.py + evaluate_geometry_colmap.py.
8. Does topology freeze just hide a limitation? Include no-freeze controls and explain topology contract.

Gate:
- PASS only if the report contains enough negative controls to make the final paper claim narrower but defensible.
```

---

# Prompt SCE11 — Final paper-facing method spec and release package

```text
Stage SCE10 must be PASS.

Mission:
Write the final method specification and release-ready package for paper drafting.

Write:
- docs/car_model/final_stageSCE11_method_spec.md
- docs/car_model/final_stageSCE11_experiment_protocol.md
- docs/car_model/final_stageSCE11_reproducibility_checklist.md
- docs/car_model/final_stageSCE11_paper_outline.md
- docs/car_model/final_stageSCE11_final_decision.md

Method spec must include:
1. CSEF field definition.
2. Sentinel correspondence definition.
3. Parent-Pareto one-sided rollback objective.
4. SCE recovery policy algorithm in pseudocode.
5. Counterfactual gate algorithm.
6. Optional local surgery algorithm.
7. Training/evaluation split rules.
8. Complexity/runtime discussion.
9. Failure modes.
10. Final claims that are supported by evidence only.

Paper outline:
- Abstract
- Motivation
- Related work positioning
- Method
- Experiments
- Ablations
- Limitations
- Conclusion

Final decision labels:
- `READY_FOR_TOP_CONFERENCE_DRAFT` if SCE policy provides multiscene improvement or a clear, novel repair capability with strong ablations.
- `READY_FOR_WORKSHOP_OR_ARXIV` if evidence is solid but novelty remains mostly recovery/heuristic.
- `NEEDS_MORE_METHOD_WORK` if SCE does not outperform F82 or does not produce a new capability.

Gate:
- PASS only if the final decision does not overclaim beyond results.
```

---

## 3. SCE run naming convention

Use consistent names:

```text
final_stageSCE1_sparse_depth_regression
final_stageSCE2_sentinel_cache
final_stageSCE3_parent_rollback_loss
final_stageSCE4_sentinel_gate
final_stageSCE5_f82_f95_sparse_regression_diagnostic
final_stageSCE6_targeted_rollback_recovery
final_stageSCE7_automatic_sce_policy
final_stageSCE8_multiscene_sce_policy
final_stageSCE9_sentinel_guided_local_surgery
final_stageSCE10_ablation_package
final_stageSCE11_method_spec
```

W&B group suggestions:

```text
finalSCE6_targeted_rollback
finalSCE7_policy_recovery
finalSCE8_multiscene_policy
finalSCE9_local_surgery
finalSCE10_ablation
```

---

## 4. Recommended first concrete command sequence

After Codex implements SCE1/SCE2/SCE4, run diagnostics before training:

```bash
python scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py \
  --source_path <courtyard_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_courtyard_model> \
  --parent_iteration 26000 \
  --candidate_model_path <F95_courtyard_model> \
  --candidate_iteration 27000 \
  --split test \
  --max_points_per_view 500 \
  --point_error_max 2.0 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --seed 7 \
  --output_dir outputs/carnet/meshsplatopt/final_stageSCE5_f82_f95_sparse_regression_diagnostic/courtyard/test_regression

python scripts/car_model/meshsplatopt_sparse_depth_regression_analyzer.py \
  --source_path <courtyard_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_courtyard_model> \
  --parent_iteration 26000 \
  --candidate_model_path <F95_courtyard_model> \
  --candidate_iteration 27000 \
  --split train \
  --max_points_per_view 500 \
  --point_error_max 2.0 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --seed 7 \
  --output_dir outputs/carnet/meshsplatopt/final_stageSCE5_f82_f95_sparse_regression_diagnostic/courtyard/train_regression

python scripts/car_model/meshsplatopt_build_sparse_depth_sentinel_cache.py \
  --source_path <courtyard_source> \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path <F82_courtyard_model> \
  --parent_iteration 26000 \
  --candidate_model_path <F95_courtyard_model> \
  --candidate_iteration 27000 \
  --split train \
  --num_views 32 \
  --prefer_hard_views \
  --prefer_observable_views \
  --max_points_per_view 500 \
  --sample_mode mixed_low_error \
  --low_error_fraction 0.5 \
  --point_error_max 2.0 \
  --output outputs/carnet/meshsplatopt/final_stageSCE5_f82_f95_sparse_regression_diagnostic/courtyard/sentinel_cache/sentinel_cache.npz
```

Only after the report identifies concrete clusters should SCE6 training begin.

---

## 5. Kill criteria and promotion criteria

### 5.1 Kill criteria

Stop or redesign if any of the following happens:

1. Train/calibration sentinels do not predict test sparse-depth failure at all.
2. One-sided rollback loss fixes sparse depth only by destroying F95 RGB/per-view gains.
3. SCE policy requires per-scene hand thresholds to pass.
4. The method cannot beat F82 on any meaningful scenario and cannot produce a new local repair capability.
5. Local surgery edits remain only synthetic and do not help any real-scene diagnostic cluster.

### 5.2 Promotion criteria

Promote SCE-Repair as the new paper-facing method if:

1. Courtyard F95 bottleneck is resolved: RGB/per-view/normal gains retained and sparse AbsRel/Depth non-regression achieved vs F82.
2. SCE policy works without per-scene retuning on at least the F82 four-scene suite, or clearly improves a documented failure case while preserving F82 elsewhere.
3. Ablations show global dense anchor F96 is weaker than sentinel one-sided rollback.
4. No-test-leakage manifests are clean.
5. At least one non-delete local surgery is either real-scene beneficial or honestly demoted as optional infrastructure without overclaiming.

---

## 6. Final intended paper claim after successful SCE stages

If SCE6-SCE10 pass, the paper claim should be:

> MeshSplatOpt introduces evidence-sentinel certified mesh surgery for mesh splatting. A Counterfactual Surface Evidence Field proposes compact or local repair edits, while sparse correspondence sentinels enforce one-sided parent-Pareto geometry constraints during recovery. This enables aggressive topology reduction and targeted visual improvement without sacrificing sparse geometry certificates. Unlike global depth anchoring or heuristic pruning, our method localizes evidence conflict at the exact correspondences that certify geometry and applies rollback only where a candidate regresses.

If SCE only passes courtyard but not broad multiscene validation, use a narrower claim:

> SCE-Repair is a targeted repair module for parent-Pareto bottlenecks in MeshSplatOpt, resolving cases where teacher-render recovery improves visual metrics but regresses sparse geometry.

If SCE fails, keep F82/F49 as the honest paper-facing method and report SCE as negative evidence.

---

# 7. 顶会创新性补强层：不要把 SCE 写成 F95 修补器

## 7.0 直接判断

原始 SCE0–SCE11 能解决当前最明确的技术瓶颈，尤其是 F95 在 courtyard 上的 sparse-depth / AbsRel parent-Pareto 失败。但如果只做到这些，论文仍可能被 reviewer 看成：

> 一个很聪明的、局部化的 depth regularizer / recovery gate，用来修复已有 compact policy 的 corner case。

这还不足以稳稳跨过顶会创新性鸿沟。顶会版本必须把 SCE 上升为**主方法理论对象**，并证明它不只是 courtyard/F95 repair，而是一个可迁移的 **certificate-carrying bidirectional mesh surgery framework**。

因此必须追加下面的 SCE12–SCE18。它们的目标不是继续堆实验，而是完成三件事：

1. **概念升维**：从 “sparse rollback loss” 升级为 “evidence conflict graph + certificate-carrying correspondences”。
2. **任务升维**：从 “compact recovery” 升级为 “真实/合成可控 defect 的 bidirectional repair benchmark”。
3. **审稿升维**：从 “指标都赢” 升级为 “为什么全局 depth anchor、QEM、delete-only、no-sentinel、no-certificate 都不能替代我们”。

如果时间只能做一件事，优先做 SCE12 + SCE13 + SCE14 + SCE16。这四个阶段决定顶会新意能不能成立。

---

## Prompt SCE12 — Evidence Conflict Graph: 把 SCE 从 loss 变成论文核心对象

```text
You are working inside Dystopians/SPCarNet.

Stage SCE0-SCE3 should exist or be in progress. This stage is a conceptual and implementation upgrade: do not treat sparse sentinel rollback as just a loss term. Turn it into an Evidence Conflict Graph, which is the paper-facing object that links views, sparse COLMAP points, rendered pixels, mesh faces, local edit clusters, and certificates.

Mission:
Implement and document an Evidence Conflict Graph (ECG) abstraction.

Why:
The current project risks being framed as compact mesh recovery with many heuristics. ECG makes the method look like a general evidence optimizer: it explicitly represents where render improvement conflicts with sparse geometry evidence, and it assigns local repair actions to those conflicts.

Write before code:
- docs/car_model/final_stageSCE12_evidence_conflict_graph_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/evidence_conflict_graph.py
- scripts/car_model/meshsplatopt_build_evidence_conflict_graph.py
- scripts/car_model/smoke_test_stageSCE12_evidence_conflict_graph.py

Graph nodes:
- `view_node`: train/test/calibration view, split tag, observability stats.
- `sparse_point_node`: COLMAP point id, 3D xyz, reprojection error, normal estimate if available.
- `pixel_sample_node`: rendered pixel coordinate, predicted parent depth, predicted candidate depth, GT sparse depth, residual, alpha/coverage if available.
- `mesh_cluster_node`: local face/vertex cluster near the projected sparse point or high residual region.
- `certificate_node`: depth non-regression, AbsRel non-regression, normal consistency, render improvement, changed-pixel safety, topology unchanged, free-space if available.
- `edit_action_node`: delete, snap, split, fill, appearance reset, rollback-only.

Graph edges:
- view observes sparse point.
- sparse point projects to pixel sample.
- pixel sample is explained by mesh cluster if renderer/nearest face information is available; otherwise use approximate local spatial support.
- certificate constrains pixel sample / sparse point / cluster.
- edit action targets cluster.

Required outputs:
- `evidence_conflict_graph.json` compact JSON with node/edge summaries.
- `evidence_conflict_graph.npz` arrays for large per-correspondence data.
- `ecg_cluster_summary.csv` ranking clusters by conflict severity.
- `ecg_report.md` with top-20 conflict clusters and suggested actions.

Conflict scores:
- `depth_conflict = max(0, candidate_abs_error - parent_abs_error - margin_abs)`
- `absrel_conflict = max(0, candidate_absrel - parent_absrel - margin_rel)`
- `render_gain = parent_rgb_error - candidate_rgb_error` if available.
- `bad_tradeoff = render_gain_positive AND depth_conflict_positive`.
- `certificate_pressure = weighted sum of active certificate violations`.
- `editability = CSEF debt + local redundancy - positive_surface_evidence - uncertainty`.

Smoke test:
Create synthetic data with:
1. one sparse point where candidate improves RGB and depth;
2. one sparse point where candidate improves RGB but worsens depth;
3. one sparse point where candidate worsens both;
4. one mesh cluster with high debt and low support;
5. one protected cluster.

Gate:
- PASS only if the graph correctly ranks the RGB-improved/depth-worsened point as the top conflict and does not recommend destructive edits on protected cluster.

Report must explicitly state:
- ECG is not a new metric for cherry-picking; it is built on train/calibration sentinels for policy and on held-out/test only for final audit.
- ECG is the bridge from CSEF proposal to SCE certificate.

Append research-log entry.
Commit and push.
```

---

## Prompt SCE13 — Certificate-Carrying Edit Planner: 从诊断图走向真正 bidirectional surgery

```text
Stage SCE12 must be PASS.

Mission:
Implement a certificate-carrying local edit planner that uses the Evidence Conflict Graph to choose between rollback-only, appearance-only, snap, split, fill, and delete actions.

This is the decisive novelty bridge. The method should not appear as a passive regularizer. It must actively decide which type of local mesh surgery is appropriate for a conflict cluster.

Write before code:
- docs/car_model/final_stageSCE13_certificate_carrying_edit_planner_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/certificate_edit_planner.py
- scripts/car_model/meshsplatopt_plan_certificate_edits.py
- scripts/car_model/smoke_test_stageSCE13_certificate_edit_planner.py

Input:
- ECG output from SCE12.
- CSEF signals if available.
- Existing defect_mining outputs if available.
- Parent/candidate checkpoint paths for metadata only; do not run heavy training here.

Allowed action classes:
1. `ROLLBACK_ONLY`: use SCE one-sided loss, no topology/edit change.
2. `APPEARANCE_ONLY`: freeze geometry, update features/weights only.
3. `SNAP_LOCAL`: move a local cluster toward parent/sparse/plane support.
4. `SPLIT_ALLOCATE`: add local degrees of freedom where render debt is high but geometry evidence supports a surface.
5. `FILL_PATCH_LOCAL`: fill certified local hole/void with boundary + plane/sparse support.
6. `DELETE_OR_COLLAPSE`: remove low-evidence redundant/unsafe primitive cluster.
7. `REJECT_UNOBSERVED`: do not edit; label hallucination risk.

Decision rules:
- If candidate improves render but worsens sparse depth at existing supported points: prefer `ROLLBACK_ONLY` or `SNAP_LOCAL`, not fill/delete.
- If render debt is high and sparse/plane support says surface should exist but mesh has no support: prefer `SPLIT_ALLOCATE` or `FILL_PATCH_LOCAL`.
- If a cluster has low positive evidence, high redundancy, and no debt: prefer `DELETE_OR_COLLAPSE`.
- If evidence is weak and prior-only flag would be true: choose `REJECT_UNOBSERVED`, unless explicit diagnostic mode.
- Every action must carry certificates it must satisfy before commit.

Output:
- `certificate_edit_plan.json`
- `certificate_edit_plan.csv`
- `certificate_edit_plan_report.md`

Each planned edit record must include:
- action type;
- target ECG cluster ids;
- target sparse point ids;
- required certificates;
- expected risk;
- expected benefit;
- whether it is allowed for headline evidence;
- whether it touches topology;
- recommended recovery flags.

Smoke test:
Construct 6 synthetic ECG clusters, one for each action class. Verify planner chooses the intended action and attaches the right certificates.

Gate:
- PASS only if at least snap/split/fill/delete/rollback-only are all represented in the planner output on synthetic ECG.

Append research-log entry.
Commit and push.
```

---

## Prompt SCE14 — Mesh Surgery Stress Test Benchmark: 顶会需要一个不是压缩表的任务

```text
Stage SCE13 must be PASS.

Mission:
Create a Mesh Surgery Stress Test benchmark that proves the method solves a broader downstream problem than compact recovery.

Why:
If the paper only reports clean-long vs compact-recovery metrics, the novelty remains vulnerable. The benchmark should evaluate whether a method can repair controlled mesh-splatting defects while preserving render and sparse geometry certificates.

Write before code:
- docs/car_model/final_stageSCE14_mesh_surgery_stress_test_design.md

Implementation files:
- ss3dm_prior/meshsplatopt/stress_test_defects.py
- scripts/car_model/meshsplatopt_make_stress_test_defects.py
- scripts/car_model/meshsplatopt_run_stress_test_suite.py
- scripts/car_model/meshsplatopt_collect_stress_test_results.py
- scripts/car_model/smoke_test_stageSCE14_stress_test_defects.py

Defect families:
1. `FLOATER_INSERTION`: insert unsupported triangles / splats near visible surfaces.
2. `SUPPORTED_SURFACE_DELETE`: delete a local supported patch.
3. `DENT_DEFORM`: push vertices inward/outward around a supported region.
4. `ROUGH_SURFACE_NOISE`: perturb local geometry normal/position.
5. `BOUNDARY_HOLE`: remove boundary-supported local patch.
6. `GROUND_VOID`: remove a planar/height-field ground region.
7. `APPEARANCE_GHOST`: corrupt features/weights without geometry change.
8. `OVERCOMPACT_CLUSTER`: delete/collapse a locally over-aggressive compact cluster.

Benchmark split:
- Use train/calibration views for repair policy and recovery.
- Use held-out/test views only for final evaluation.
- Keep a manifest proving no test leakage.

Methods to compare:
- clean corrupted checkpoint without repair;
- naive continuation;
- global sparse-depth recovery;
- global render-depth anchor;
- F82/F49/F75-style compact recovery where applicable;
- delete-only CSEF;
- QEM / simplification baseline for topology defects;
- SCE rollback-only;
- SCE + certificate edit planner.

Metrics:
- PSNR, SSIM, LPIPS;
- sparse AbsRel, Depth MAE, normal angle;
- defect-local render error;
- defect-local sparse error;
- topology delta;
- percentage of defects repaired without violating certificates;
- false repair rate on unknown/unobserved voids.

Outputs:
- `stress_test_manifest.json`
- `stress_test_results.json`
- `stress_test_results.csv`
- `stress_test_report.md`
- qualitative montage scripts for before/corrupted/repaired/error.

Smoke test:
Use a small synthetic mesh/checkpoint-like array and verify each defect generator is reversible and writes a manifest.

Gate:
- PASS if at least synthetic defect generation and metric collection work.
- PROMOTE only if SCE + planner beats naive/global baselines on at least 5/8 defect families, with no silent fill of unknown voids.

Append research-log entry.
Commit and push.
```

---

## Prompt SCE15 — Real-Scene Local Surgery Pilot: 至少让一个非 delete edit 成为 load-bearing 证据

```text
Stage SCE14 must be PASS.

Mission:
Run a real-scene local surgery pilot driven by SCE12/SCE13. The goal is to find at least one real checkpoint where snap/split/fill/appearance-reset is not merely safe, but improves a documented local failure under independent evaluation.

Preferred scenes:
- courtyard, because F95 exposes a localized sparse-depth conflict.
- parking_phone_tiny, because it has strong prior topology and large-scale geometry issues.
- bonsai/room/counter only if ECG identifies clear local conflict clusters.

Write before GPU work:
- docs/car_model/final_stageSCE15_real_scene_local_surgery_pilot_plan.md

Implementation may reuse existing edit infrastructure. Add wrappers if needed:
- scripts/car_model/meshsplatopt_materialize_certificate_edit_plan.py
- scripts/car_model/meshsplatopt_run_certificate_edit_recovery.py
- scripts/car_model/meshsplatopt_gate_certificate_edit_result.py

Required pilot protocol:
1. Build ECG on train/calibration split from parent vs candidate or corrupted checkpoint.
2. Plan certificate edits.
3. Materialize only top-1 or top-k low-risk edit packages.
4. Run short recovery with strict topology freeze unless split/fill requires controlled topology change.
5. Run independent render/metrics/geometry evaluation.
6. Run parent-Pareto gate and defect-local gate.
7. Compare to rollback-only SCE and global loss baselines.

Do not overclaim:
- If local surgery does not beat rollback-only, document it and demote local surgery to optional infrastructure.
- If local surgery creates topology changes, report topology delta and compare against same-topology rollback-only.

Required outputs:
- `real_surgery_plan.json`
- `real_surgery_materialization_report.md`
- `real_surgery_gate.json`
- `real_surgery_report.md`

Promotion criteria:
- At least one non-delete action improves a local defect metric and does not degrade global parent-Pareto metrics beyond threshold.
- Best if it also improves a headline metric, but local-certified improvement is acceptable for a supplemental claim.

Hard gate:
- If no non-delete edit passes on real scenes, do not claim bidirectional surgery as the main empirical win. Keep it as a framework component and make SCE certificate/recovery the paper core.

Append research-log entry.
Commit and push.
```

---

## Prompt SCE16 — Reviewer-Killer Ablation Matrix: 证明不是 global depth / QEM / delete-only / LPIPS trick

```text
Stage SCE6 or SCE15 should be available.

Mission:
Build a reviewer-facing ablation matrix that directly attacks the strongest alternative explanations.

Write:
- docs/car_model/final_stageSCE16_reviewer_killer_ablation_plan.md

Implementation files:
- scripts/car_model/meshsplatopt_collect_reviewer_killer_ablations.py
- scripts/car_model/meshsplatopt_make_ablation_latex_tables.py

Ablation axes:
1. `no_sentinel`: F95-style teacher + render-normal recovery without sparse sentinel rollback.
2. `global_sparse_only`: global sparse loss, same lambda/horizon.
3. `global_render_depth_anchor`: F96-style dense render depth anchor.
4. `vertex_anchor`: F90/F91 style parameter-space anchor.
5. `freeze_geometry`: F92/F93 style appearance-only or vertex/weight freeze.
6. `sentinel_all_points`: one-sided loss but not conflict-targeted; all train sparse points.
7. `sentinel_conflict_only`: proposed ECG/SCE targeted version.
8. `no_parent_one_sided`: ordinary supervised depth to sparse GT, no parent non-regression envelope.
9. `no_train_test_separation`: diagnostic only, must be marked invalid if run; prove main results do not use it.
10. `delete_only_csef`: no snap/split/fill planner.
11. `qem_or_decimation`: strong topology simplification baseline.
12. `lpips_heavy`: confirm perceptual loss alone regresses depth.

For each row report:
- render metrics;
- sparse geometry metrics;
- per-view min PSNR delta and negative view count;
- local sentinel pass rate;
- topology;
- runtime/memory;
- whether it passes parent-Pareto gate.

Required conclusion format:
- Which alternative matches RGB but fails sparse geometry?
- Which alternative matches geometry but fails RGB/per-view?
- Which alternative overfits training sentinels but fails test?
- Which component is load-bearing?

Gate:
- PASS if the table clearly shows that SCE targeted one-sided conflict loss is not replaceable by global dense depth anchor, vertex anchor, or freeze-only recovery.

Append research-log entry.
Commit and push.
```

---

## Prompt SCE17 — Paper Method Spec and Claim Lock: 防止写作时被拖回工程故事

```text
Stage SCE12 and at least one real result stage must be available.

Mission:
Write the final paper-facing method specification and claim lock. This is not the full paper, but it freezes what the paper is allowed to claim and what it must not claim.

Write:
- docs/car_model/final_stageSCE17_paper_method_spec_and_claim_lock.md

The document must include:

1. Method name options:
   - MeshSplatOpt-SCE
   - Evidence-Sentinel Mesh Surgery
   - Certificate-Carrying MeshSplatOpt
   Choose one primary and explain why.

2. Core contributions:
   - Counterfactual Surface Evidence Field for proposal/risk.
   - Evidence Conflict Graph for localizing evidence contradictions.
   - Certificate-carrying sparse correspondences / sentinels.
   - One-sided parent-Pareto rollback recovery.
   - Optional certificate edit planner for bidirectional surgery.

3. Formal-ish guarantee / proposition:
   State a limited proposition, not overclaiming:
   If optimization reaches zero sentinel rollback loss on a fixed sentinel set, then candidate sparse error on those sentinels is no worse than parent plus margin under the chosen error functional. This is not a global geometry guarantee, but it is a precise certificate on the measured evidence set.

4. What is not claimed:
   - Not a universal single prune ratio.
   - Not guaranteed dense geometry correctness.
   - Not allowed to use test correspondences for training.
   - Not claiming prior-only void hallucination as real repair.
   - Not claiming every non-delete edit improves every scene unless SCE15 proves it.

5. Main table candidates:
   - F49 CSEF-family 5-scene table.
   - F82 fixed adaptive policy two-seed table.
   - SCE courtyard/F95 bottleneck repair table.
   - Stress-test benchmark table if SCE14 passes.
   - Local surgery pilot if SCE15 passes.

6. Reviewer risk checklist:
   - Is it just pruning? Answer with stress-test + non-delete planner.
   - Is it just depth regularization? Answer with SCE16 ablations.
   - Is it overfit to selected scenes? Answer with two seeds + held-out split + failures.
   - Is sparse COLMAP proxy weak? Answer with transparency, normal proxy caveat, and independent render metrics.
   - Is validation-selected budget unfair? Separate F49 validation-budget claim from F82/SCE fixed-policy claim.

7. Final paper claim options:
   Provide three claim tiers:
   - Tier A: full top-conference claim if SCE14/SCE15 pass.
   - Tier B: strong method claim if SCE fixes F95 and transfers multiscene but local surgery remains optional.
   - Tier C: honest narrower claim if SCE only fixes courtyard.

Gate:
- PASS only if the document makes it impossible to accidentally overclaim in the paper.

Append research-log entry.
Commit and push.
```

---

## Prompt SCE18 — 最终 Go / No-Go 顶会裁决表

```text
Stage SCE16 and SCE17 should be available.

Mission:
Create a final top-conference readiness decision table. This table decides whether to submit as a top-conference full paper, workshop/short paper, or continue research.

Write:
- docs/car_model/final_stageSCE18_top_conference_readiness_decision.md

Rubric:
Score each 0-5:
1. Novel method object: CSEF + ECG + sentinels are clearly distinct from prior pruning/compression.
2. Load-bearing mechanism: ablations show SCE/ECG/certificates matter.
3. Real-scene evidence: not only synthetic; at least one difficult real bottleneck is solved.
4. Generality: multiscene or stress-test breadth.
5. Baseline strength: clean-long, QEM, random, delete-only, global-depth, no-freeze, no-gate covered.
6. Honesty: failures and limitations documented.
7. Reproducibility: commands, W&B, artifacts, no-test-leakage manifests.
8. Qualitative evidence: galleries show visibly meaningful improvements.
9. Runtime/memory practicality: topology reduction or repair cost is defensible.
10. Writing clarity: paper claim does not rely on hidden caveats.

Decision rules:
- 42-50: submit top conference full paper.
- 35-41: likely submit if deadline pressure is high, but strengthen writing and limitations.
- 28-34: workshop/short or continue experiments.
- below 28: do not submit as full top-conference method paper.

Required output:
- final score table.
- exact missing blockers.
- recommended paper title.
- recommended main figures.
- recommended main tables.
- one-paragraph abstract draft.
- final go/no-go decision.

Gate:
- PASS only if the decision is honest and actionable.

Append research-log entry.
Commit and push.
```

---

# 8. 修订后的最终判断

如果只执行 SCE0–SCE11：

- 可以大概率解决当前 F95/F82 的 sparse-depth bottleneck；
- 可以把方法从 global tuning 推到 localized certificate recovery；
- 但顶会创新性仍有被质疑为 “smart regularizer + careful engineering” 的风险。

如果执行 SCE0–SCE18，尤其 SCE12–SCE16：

- 方法核心会变成 **Evidence Conflict Graph + Certificate-Carrying Sentinels + Bidirectional Edit Planner**；
- 实验核心会从 “压缩恢复表” 扩展到 “mesh surgery stress test + real conflict repair”；
- reviewer 很难再简单归类为普通 pruning/compression 或 depth regularization；
- 这才真正有机会弥合当前工作与顶会工作之间的创新性鸿沟。

最终采用建议：

> 不要直接采用原始 SCE0–SCE11 作为最终执行稿；应采用本 V2，把 SCE12–SCE18 作为顶会创新性补强层。原始 SCE 是必要条件，V2 的 ECG / planner / stress-test / reviewer-killer ablation 才是冲顶会的充分性方向。


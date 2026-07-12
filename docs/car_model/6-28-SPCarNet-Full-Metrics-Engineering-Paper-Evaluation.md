# SPCarNet 完整评估报告：Metrics / 工程闭环 / 论文级可用性

Date: 2026-06-28

## 结论先行

当前结论是：**还没有达到“论文终局闭环”**。  
如果只看“是否存在一个本地 full9 上超过 clean MeshSplatting 的表示级版本”，答案是 **有**：当前最强、可验证的 baked representation 结果仍然是 **v106 POD-MoE base-preserve**，它在本地 selected full9 口径上超过 clean MeshSplatting。  
如果看“新一代 vNext certified residual surface texture 是否已经成为可推广、可写成论文主结果的方法”，答案是 **还没有**：vNext 的工程协议很强，但已完成 full9 metrics 低于 clean MeshSplatting 和 v106。最新 v165 flowers exact run 把 target changed pixels 从 v164 的 `860` 扩大到 `8324`，约 `9.68x`，但 PSNR 只提升 `+0.000051`，SSIM 基本不变且微降，LPIPS 仅改善 `-0.00000042`；这说明当前瓶颈已经从“完全改不到”转为“改到了但残差表示强度仍不足”。

一句话评价：

- **Metrics 层面**：v106 达到本地 baseline 超越；vNext 未达标。
- **工程层面**：vNext 的审计、manifest、strict no-target-GT apply、target-evidence verifier、fallback/no-op 和 W&B 记录已经接近论文级工程框架，但 runtime、存储稳定性和全场景 promotion 仍是明显短板。
- **论文层面**：目前可以讲一个“逐步走向可审计修复/压缩”的研究故事，但不能诚实宣称 vNext 已经全面胜出；paper-final 状态仍是 `NOT COMPLETE`。

## 当前最重要的数字

### Full9 汇总

| method | scenes | PSNR | SSIM | LPIPS | 相对 clean MeshSplatting | 当前角色 |
|---|---:|---:|---:|---:|---|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | 本地公平基线 |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 | 稳定表示级 anchor |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 | 当前最强已验证 baked representation |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | +1.329628 / +0.034657 / -0.063316 | 强 RGB endpoint/reference，不能直接混作 baked representation |
| vNext structure-aware shrink cleanup | 9 | 25.067699 | 0.741260 | 0.306689 | -0.083983 / -0.007758 / +0.019068 | 协议完整，但未推广 |
| vNext effective-margin gate | 9 | 25.067410 | 0.741259 | 0.306695 | -0.084272 / -0.007759 / +0.019074 | 更安全，但更接近 no-op |

解释：

- v106 相对本地 clean MeshSplatting 是明确正向：PSNR/SSIM 更高，LPIPS 更低。
- v106 相对 v104c 只有小幅增益：`+0.002181` PSNR、`+0.000103` SSIM、`-0.000112` LPIPS；它是稳定提升，但不是“大幅颠覆”。
- vNext full9 低于 clean MeshSplatting，也低于 v106；它目前应被视为“工程协议和瓶颈诊断路线”，不能作为最终质量主线。

## v106：当前最强可汇报表示级版本

v106 的核心方法是保留 v104c 风格的稳定 shrink view-affine residual field，再叠加两个保守的三角形残差专家：`detail` 与 `occlusion_boundary`。它不是简单参数扫描，而是把局部细节与遮挡边界作为两个不同专家，以 base-preserve 方式避免破坏原始 MeshSplatting 表示。

### v106 per-scene 结果

| scene | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.719175 | 0.675086 | 0.313405 | +0.001526 | +0.000115 | -0.000098 |
| flowers | 20.077723 | 0.531240 | 0.374393 | +0.001879 | +0.000163 | -0.000080 |
| garden | 25.790945 | 0.799382 | 0.174480 | +0.002851 | +0.000119 | -0.000104 |
| stump | 25.460457 | 0.714661 | 0.282135 | +0.001146 | +0.000061 | -0.000078 |
| treehill | 21.245092 | 0.578518 | 0.384177 | +0.001329 | +0.000099 | -0.000121 |
| room | 29.600351 | 0.891889 | 0.230616 | +0.002516 | +0.000051 | -0.000048 |
| counter | 27.499645 | 0.867521 | 0.238847 | +0.001577 | +0.000102 | -0.000139 |
| kitchen | 28.772043 | 0.881652 | 0.187815 | +0.001595 | +0.000062 | -0.000206 |
| bonsai | 30.316090 | 0.907520 | 0.230050 | +0.005213 | +0.000154 | -0.000136 |
| mean | 25.831280 | 0.760830 | 0.268435 | +0.002181 | +0.000103 | -0.000112 |

证据文件：

- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- `docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md`

评估：

- 这是当前最适合用于汇报的 **baked representation** 结果。
- 论文风险在于：增益方向稳定，但幅度偏小，视觉差异不容易在全图直接看出来。
- 它可以作为“当前最好版本”，但还不足以支撑“远超 MeshSplatting / 颠覆式提升”的强 claim。

## vNext：工程闭环很强，但质量未达标

vNext certified residual surface texture 的目标是把修复限制在可认证的三角形/UV/bin footprint 上，并用 policy-val、image gate、bin uncertainty guard、fallback/no-op 来避免 out-of-trajectory 崩塌。它的研究价值在于协议和可审计性，而不是当前指标。

### 已完成 full9 证据

| run | scenes | protocol pass | accepted nonzero | fallback/no-op | mean changed fraction | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| structure-aware shrink cleanup | 9 | 9 | 6 | 3 | 0.002756271 | 25.067699 | 0.741260 | 0.306689 |
| effective-margin gate | 9 | 9 | 1 | 8 | 0.001371507 | 25.067410 | 0.741259 | 0.306695 |

证据文件：

- `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`

评估：

- `structure-aware shrink cleanup` 说明 vNext 能在 full9 上完整跑通，并且有 6 个场景产生非零改动。
- `effective-margin gate` 说明更严格的 safety gate 可以抑制低效候选，但代价是 8/9 场景 fallback/no-op。
- 两者都低于 clean MeshSplatting 和 v106，所以不能推广为主结果。

## v162 / v163 / v164 / v165 flowers 诊断

flowers 是当前 vNext 短板诊断最清楚的场景。v162-v164 的核心发现是：不是 alpha 不够好，而是被认证允许修改的 target footprint 太小，导致全图指标和人眼视觉几乎不变。v165 进一步证明：仅把 target-visible footprint 放大仍不够，必须让写入的 train-only residual representation 本身更有表达力，否则会出现“改动区域扩大但指标几乎不动”的情况。

| version | 状态 | 核心机制 | accepted | alpha | changed pixels | allowed bins / faces | PSNR | SSIM | LPIPS | 诊断 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v162 | complete | sparse-selective bridge 语义修复 | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | 真实修复，但 footprint 极小 |
| v163 | complete | target-footprint residual-debt support expansion | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | support expansion 只找到 1 个 eligible face，未改善 |
| v164 | complete | target-visible connected region growth | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | connected growth 无 eligible bins，未扩大 footprint |
| v165 | complete | train-only target-impact residual basis | true | 0.1875 | 8324 | 1145 / 26 | 20.452848 | 0.549059 | 0.355544 | footprint 明显扩大，但指标增益只有噪声级 |

v162-v165 证据：

- v162 root: `/dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective`
- v163 root: `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion`
- v163 report: `docs/car_model/6-28-SPCarNet-Metrics-Engineering-Paper-Evaluation-v163.md`
- v164 root: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers`
- v164 manifest: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_v164_target_connected_exact_manifest.json`
- v164 audit: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v164 metrics: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_ours_26000_v164_target_connected_exact_flowers_test_results.json`
- v165 root: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers`
- v165 manifest: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v165 audit: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v165 metrics: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_ours_26000_v165_target_impact_exact_flowers_test_results.json`
- v165 render dir: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/renders`
- v165 GT dir: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/gt`

v164 已实现内容：

- 在 `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` 中加入 `target_connected_region_growth`。
- 在 `scripts/car_model/run_vnext_certified_residual_texture_scene.py` 中加入 runner/parser 参数转发。
- 新增接口包括 `--enable_sparse_materialization_target_connected_region_growth`、radius、min pixels/views、policy samples、positive view fraction、允许的负增益上限和 max extra bins。
- dry-run 通过，真实 exact run 完整跑通，输出 root 为 `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers`。

v164 exact run 完成状态：

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_v164_target_connected_exact_manifest.json`
- manifest status: `COMPLETE`
- manifest errors: `[]`
- log: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/logs/02_certified_texture.log`
- apply elapsed: `23702.957s`，populate eval GT elapsed: `41.668s`，evaluate elapsed: `43.504s`
- W&B offline run: `/dev/shm/peilincai_wandb_v164_target_connected_exact/wandb/offline-run-20260628_134505-6569eb6r`
- connected growth: `enabled=true`，`reason=no_eligible_connected_bins`，`seed_allowed_bin_count=40`，`candidate_bin_count=0`，`added_bin_count=0`，`added_target_pixels=0`，`final_allowed_bin_count=121`
- target-visible expansion: `original_allowed_bin_count=40`，`candidate_bin_count=81`，`added_bin_count=81`，`final_allowed_bin_count=121`，`added_target_pixels=479`
- target apply: `changed_pixels=860`，`png_quantized_changed_pixels=849`，`changed_fraction=2.3180093151630152e-05`
- 当前解释：v164 完整验证了 connected growth 这条补丁路线，但结论是负面的。它没有找到可安全加入的 connected bins，因此没有扩大 footprint，metrics 与 v162/v163 完全一致。这说明当前瓶颈不是“少一个邻域扩张开关”，而是 certified sparse bin 候选集本身过窄。

v165 exact run 完成状态：

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- manifest status: `COMPLETE`
- manifest errors: `[]`
- protocol audit: `passed=true`，`target_apply_leak=false`，`target_gt_visible_to_apply=false`，`target_gt_visible_to_selection=false`，`target_gt_visible_to_eval=true`，`target_forbidden_keys_stripped=true`
- commands: `strip_target_evidence_no_gt` elapsed `152.590s`，`apply_certified_residual_texture` elapsed `5415.726s`，`populate_eval_gt_from_target_evidence` elapsed `11.721s`，`evaluate_vnext_target` elapsed `43.088s`
- W&B offline run: `/dev/shm/peilincai_wandb_v165_target_impact_exact/wandb/offline-run-20260628_153357-ezjo72h3`
- target-impact residual basis: `candidate_bin_count=2600`，`added_bin_count=1024`，`added_policy_row_bin_count=732`，`added_without_policy_row_bin_count=292`，`added_target_pixels=9275`，`added_target_view_hits=2240`
- final certified footprint: `original_allowed_bin_count=121`，`original_allowed_face_count=13`，`final_allowed_bin_count=1145`，`final_allowed_face_count=26`
- target apply: `changed_pixels=8324`，`png_quantized_changed_pixels=7896`，`changed_fraction=0.00022436173883042952`
- metrics: PSNR `20.452848434448242`，SSIM `0.5490590929985046`，LPIPS `0.3555436134338379`
- delta vs v164: PSNR `+0.0000514984`，SSIM `-0.0000000596`，LPIPS `-0.0000004172`，changed pixels `+7464`
- 当前解释：v165 是重要的工程和诊断进展，因为它首次把 flowers 的 target footprint 从几百像素级扩大到数千像素级，并保持 no-target-GT apply 审计通过；但它不是质量突破，因为三指标提升几乎不可见，定性图也很难形成强展示。

## 工程级评估

### 已经达到论文级工程雏形的部分

- strict no-target-GT apply：target apply 阶段不直接读取 test GT。
- 独立 eval-GT population：最终评估阶段再补 GT，用于公平评价。
- command manifest：保存命令、路径、状态、return code、错误和输出位置。
- adapter audit：记录 accepted/fallback、alpha、target changed pixels、bin guard、sparse materialization、topology 等。
- fallback/no-op：候选不安全时显式回退，避免把坏结果硬写成方法输出。
- W&B offline：长程/中程实验已经按要求接入 offline logging，且在 `/data` 满盘时转移到 `/dev/shm`。
- 接口闭合：v164 的 connected growth 和 v165 的 train-only target-impact residual basis 都已经在 adapter 与 runner 两侧都有 CLI 参数和校验。
- target evidence verifier：新增 `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`，可以独立扫描 stripped target evidence 中是否仍残留 `rgb_gt`、`residual_rgb`、teacher residual 等 forbidden keys。
- strict guard：runner 现在强制所有 target-footprint apply path 必须启用 `--strict_no_target_gt_apply`，否则直接 parser error，避免把 target footprint 机制误跑成 target-GT 可见流程。
- audit 修复：v165 后已修正 target-impact / connected-growth footprint cache 共享隐患，并补上 target-impact added-sample 统计，后续重跑会得到更完整的 audit。

### 仍然不足的工程问题

- runtime 太慢：v162 flowers adapter 约 `5771.652s`，v163 flowers adapter 约 `8684.925s`；v164 exact apply 约 `23702.957s`；v165 exact apply 约 `5415.726s`。虽然 v165 比 v164 快，但仍远不适合作为高吞吐论文实验系统。
- GPU 利用率低：大量耗时在 CPU/IO/NumPy/Python evidence traversal，不是典型 GPU 训练瓶颈。
- `/data` 已满，`/dev/shm` 也接近满载；W&B、manifest 和长程实验存在失败风险。
- v164 虽完整跑通，但以 6.58 小时成本得到零 footprint 增量；v165 虽扩大 footprint，却没有实质质量增益。二者共同说明当前 verification cost / improvement ratio 不适合论文主系统。
- vNext 仍缺一个“固定策略 full9 promotion run”能同时击败 clean MeshSplatting 和 v106。
- 已完成的 v165 exact run 是在新增 verifier 集成前启动的，所以 manifest 里还没有 `verify_stripped_target_evidence_no_gt` 这一步；但补丁后的手动 verifier 已通过，后续 patched runner dry-run 会包含该 verify command。

## 论文级评估

### 可以诚实写进 PPT/讨论的 claim

1. 我们已经建立了本地 same-protocol clean MeshSplatting baseline，并能进行 full9 比较。
2. v106 POD-MoE base-preserve 是当前最强已验证 baked representation，full9 平均三指标超过 clean MeshSplatting。
3. vNext 是一个更严谨的 no-target-GT、可审计 residual surface texture 框架，能明确记录何时改、改哪里、为什么拒绝。
4. v162/v163/v164 的负面结果很有价值：它说明仅靠 support expansion、connected growth 或更严格 gate 不能解决 footprint 太小的问题。
5. v164 的失败把瓶颈定位得更明确：安全候选集不足，而不是 alpha、单个 face support 或邻域半径设置不足。
6. v165 进一步把瓶颈推进了一步：target-impact 机制能扩大 certified footprint，但现有 train-only residual basis 的表达力不足，无法转化为明显指标或视觉收益。

### 当前不能写成最终论文主张的 claim

1. 不能说 vNext 已全面超越 MeshSplatting。
2. 不能说当前方法有“人眼明显可见”的稳定视觉提升。
3. 不能说当前方法在几何、压缩、PSNR/SSIM/LPIPS、LPIPS 感知质量上全部全面胜出。
4. 不能把 v101/v102 强 RGB endpoint 与 baked representation 结果混成一个口径。
5. 不能把 v164 当成成功改进；它已经完成验证，但没有带来 footprint 或 metrics 增益。
6. 不能把 v165 当成成功质量改进；它是成功的 footprint/工程实验，但不是成功的 paper-quality result。

### 公平性与论文口径缺口

- 当前 clean MeshSplatting baseline 是本地 selected full9 口径，不等同于官方论文中完整 Mip-NeRF360 表格的绝对口径；它可以用于本地同口径比较，但不能直接宣称超过官方 paper number。
- v106 vs clean 的本地比较较公平，因为二者都用 selected full9 汇总；但仍需要补齐 per-scene clean checkpoint 选择规则、checkpoint snapshot、eval script hash 和失败场景说明。
- 当前报告的主要指标仍集中在 PSNR/SSIM/LPIPS；论文级“全面胜出”还需要三角形数量、mesh/texture/model size、渲染速度、训练/后处理时间、显存/存储占用、几何一致性和定性局部放大图。
- vNext 的 full9 promotion 结果还没有超过 clean/v106，因此不能把 vNext 与 v106 混成一个“统一已经全面胜出”的最终方法。

## 当前达标度判断

| 维度 | 达标度 | 说明 |
|---|---:|---|
| baseline 公平性 | 80% | 本地 clean selected full9 已有，但仍需和论文官方 Mip-NeRF360 口径继续对齐 |
| metrics 超越 | 55% | v106 超 clean，但 vNext 不超；提升幅度仍小 |
| 工程闭环 | 82% | manifest/audit/fallback/W&B/strict verifier 基本完整，v164/v165 exact run 已闭环，但 runtime 和存储仍弱 |
| 论文故事 | 62% | 有方法线、反思线和明确瓶颈推进，但缺强主结果 |
| 定性展示 | 45% | full-frame 视觉差异偏弱；v165 扩大 footprint 后仍缺可视化强收益 |
| 最终 paper-ready | 58% | 可做阶段性汇报，不宜宣称终局完成 |

综合判断：**当前约 58%-62% paper-loop 完成度**。  
它比最初盲目调参阶段强很多，已经有 baseline、full9、工程审计、W&B 长程记录、strict no-target-GT 防线和明确瓶颈；但离“顶会主结果闭环”仍有明显距离。

## 下一步优先级

1. 停止把 vNext 的主要希望放在同一套 sparse bin allowlist 的小半径扩张或 alpha 微调上；v164/v165 已经证明这条线最多解决 footprint，不能自然带来视觉质量突破。
2. 下一步应转向更强的 train-only representation：例如 face-local residual basis 的容量升级、target-visible residual field 的低秩/多专家表示、或以 policy-val certificate 约束的局部纹理优化。关键要求是继续保持 no-target-GT apply。
3. 新策略必须先在 flowers 做 footprint/visual diagnostic，再固定策略跑 full9，与 clean MeshSplatting、v106、vNext effective-margin gate 和 v165 做同口径比较。
4. 工程上优先缓存 policy-val reusable evidence、减少重复 atlas traversal，并把 patched verifier 纳入所有 exact run manifest，否则 vNext 难以作为可复现实验系统。
5. README/PPT 中必须明确区分三条线：Phase-J/endpoint reference、v106 baked representation、vNext certified representation route。当前最适合汇报的正结果是 v106；v165 是瓶颈诊断和工程可信度证据。

## Evidence Index

- v106 full9 assembled: `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- v106 full9 compare: `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- vNext structure-aware full9 summary: `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- vNext effective-margin full9 summary: `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`
- v163 detailed evaluation: `docs/car_model/6-28-SPCarNet-Metrics-Engineering-Paper-Evaluation-v163.md`
- v164 exact manifest: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_v164_target_connected_exact_manifest.json`
- v164 exact audit: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v164 exact metrics: `/dev/shm/peilincai_spcarnet_20260628_v164_target_connected_exact/flowers/reports/flowers_ours_26000_v164_target_connected_exact_flowers_test_results.json`
- v165 exact manifest: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- v165 exact audit: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- v165 exact metrics: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/flowers_ours_26000_v165_target_impact_exact_flowers_test_results.json`
- v165 no-GT verifier audit: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/reports/manual_target_apply_no_gt_verify_after_patch.json`
- v165 exact renders: `/dev/shm/peilincai_spcarnet_20260628_v165_target_impact_exact/flowers/model/test/ours_26000_v165_target_impact_exact_flowers/renders`
- v164 adapter implementation: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- v164 runner implementation: `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- no-GT target evidence verifier: `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`

## Final Status

Final status: NOT COMPLETE.

未完成项：

- vNext 尚未在 full9 上超过 clean MeshSplatting 或 v106。
- 定性优势仍不明显，当前 footprint 太小。
- 工程 runtime 和存储稳定性仍需修复。
- v164 target-connected growth 已完成但无增益；v165 target-impact footprint 扩大但指标几乎不动，说明下一步必须换更强的 train-only residual representation，而不是继续小修 policy 参数。

下一条最精确的继续方向是：

```bash
WANDB_DIR=/dev/shm/peilincai_wandb_next_residual_capacity \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<low_or_mid_load_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene flowers \
  --method_name ours_26000_next_residual_capacity_flowers \
  --strict_no_target_gt_apply \
  --enable_train_only_target_impact_residual_basis \
  --target_impact_max_extra_bins 1024 \
  --write_manifest \
  --output_root /dev/shm/peilincai_spcarnet_next_residual_capacity_flowers
```

这条命令只能作为下一轮 patched-run 基础；真正需要新增的是更强的 train-only residual capacity，而不只是重复 v165 的 footprint expansion。

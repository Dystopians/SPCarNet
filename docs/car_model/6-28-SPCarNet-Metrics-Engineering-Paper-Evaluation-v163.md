# SPCarNet Metrics / Engineering / Paper-Readiness Evaluation

Date: 2026-06-28

## 中文执行摘要

当前结论是：**还没有达到论文最终闭环，不能宣称 vNext 已经全面超越
MeshSplatting baseline**。

已经成立的部分是：

- 本地 clean MeshSplatting full9 baseline 已经可用，可以作为当前仓库的
  公平基线。
- 目前最强的、可作为 baked representation 口径结果的版本仍是 **v106
  POD-MoE base-preserve**。它在 full9 平均 PSNR/SSIM/LPIPS 上超过本地
  clean MeshSplatting。
- vNext 方向的工程协议已经比较扎实：严格 no-target-GT apply、独立
  eval-GT population、manifest、W&B offline、adapter audit、topology
  audit 和 fallback/no-op 都已具备。
- v162 修复了一个真实的方法语义问题：sparse-selective non-regression
  annotation 在 bin-uncertainty bridging 之后不应丢失。
- v163 已经完成 flowers 单场景验证，证明 support-expansion hook 可以跑通，
  但它只额外找到 1 个 eligible face，最终 changed pixels、allowlist 和
  PSNR/SSIM/LPIPS 都与 v162 相同。

没有成立的部分是：

- vNext 当前 full9 证据仍低于 clean MeshSplatting 和 v106。
- v163 不是里程碑式质量突破，而是一个清楚的负面诊断：现有策略无法把
  certified footprint 扩大到足以影响全图指标和人眼可见质量。
- 工程上已经具备论文级审计框架，但方法效果还不足以支撑论文主张。

下一步最应做的不是继续扫 alpha 或阈值，而是实现 **围绕已认证 sparse bins
的 target-visible connected region growth**，让认证区域在同 face/邻近 UV
上可控扩张，并继续用 policy-val post gate 保证不退化。

This report audits the current SPCarNet / MeshSplatting repair line from three
separate claim layers:

1. the strongest historical RGB endpoint;
2. the strongest verified MeshSplatting-compatible representation endpoint;
3. the newer vNext certified residual surface texture route.

These layers must not be mixed. A method can be strong as an RGB endpoint while
not yet being the cleanest baked representation claim, and a protocol-clean
representation experiment can still be below the quality bar.

## Executive Verdict

Current status: **NOT COMPLETE for a paper-final closed loop**.

What is already strong:

- The local clean MeshSplatting full9 baseline is available and should remain
  the fair baseline for this repo.
- The current strongest verified baked representation line is **v106 POD-MoE
  base-preserve**, which beats local clean MeshSplatting on aggregate PSNR,
  SSIM, and LPIPS.
- The vNext route now has a strong engineering protocol: strict no-target-GT
  apply, separate eval-GT population, command manifests, W&B offline logging,
  adapter audits, topology audits, and explicit fallback/no-op behavior.
- v162 fixed a real sparse certification semantics bug: sparse-selective
  non-regression annotation is preserved after bin-uncertainty bridging.

What is not yet strong enough:

- The completed vNext full9 results are below clean MeshSplatting and below
  v106.
- The latest completed v162 flowers result is accepted and protocol-clean, but
  only changes `860 / 37,100,800` target pixels, which is far too small to move
  full-image metrics or make visual improvement obvious.
- v163 support expansion completed as a diagnostic. It is protocol-clean and
  accepted, but it did **not** improve the v162 footprint or metrics.

## Full9 Quantitative Summary

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean | role |
|---|---:|---:|---:|---:|---|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | local fair baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 | stable representation anchor |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 | strongest verified baked representation |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | +1.329628 / +0.034657 / -0.063316 | strong RGB endpoint/reference |
| vNext structure-aware shrink cleanup | 9 | 25.067699 | 0.741260 | 0.306689 | -0.083983 / -0.007758 / +0.019068 | protocol-complete, not promoted |
| vNext effective-margin gate | 9 | 25.067410 | 0.741259 | 0.306695 | -0.084272 / -0.007759 / +0.019074 | safer gate, still not promoted |

Interpretation:

- v106 clearly beats the local clean MeshSplatting baseline on the selected
  full9 evaluator.
- v106's gain over v104c is positive but small: `+0.002181` PSNR,
  `+0.000103` SSIM, `-0.000112` LPIPS.
- vNext currently does not beat clean MeshSplatting or v106 on completed full9
  evidence. It should not be used as the headline result yet.

## v106 Per-Scene Evidence

Source: `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`.

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

The v106 direction is consistent; its weakness is effect size, not sign.

## vNext Completed Full9 Evidence

Completed vNext structure-aware cleanup:

- `9 / 9` scenes completed.
- `9 / 9` protocol audits passed.
- `6 / 9` scenes accepted nonzero residual output.
- `3 / 9` scenes fell back/no-op.
- Mean changed fraction: `0.002756271`.
- Mean metrics: `25.067699 / 0.741260 / 0.306689`.

Completed vNext effective-margin gate:

- `9 / 9` scenes completed.
- `9 / 9` protocol audits passed.
- `1 / 9` scene accepted nonzero residual output.
- `8 / 9` scenes fell back/no-op.
- Mean changed fraction: `0.001371507`.
- Mean metrics: `25.067410 / 0.741259 / 0.306695`.

Interpretation:

- The no-target-GT protocol and fallback machinery are credible.
- The current vNext representation does not yet have enough reliable target
  impact. It is safer, but not better than clean MeshSplatting or v106.

## v159-v163 Flowers Diagnostic Thread

| version | status | key mechanism | accepted | alpha | changed pixels | PSNR | SSIM | LPIPS | diagnosis |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| v159 | complete | sparse residual materialization with face-guard skip | true | 0.3750 | 466 | 20.452793 | 0.549059 | 0.355544 | proof of life, but extremely sparse |
| v161 | complete | bridge sparse profile through empty bin-guard intersection | true | 0.0625 | 860 | 20.452782 | 0.549059 | 0.355544 | protocol clean, but alpha collapsed |
| v162 | complete | preserve sparse-selective semantics after bridge | true | 0.3750 | 860 | 20.452797 | 0.549059 | 0.355544 | real semantics fix, footprint still tiny |
| v163 | complete | target-footprint residual-debt support expansion | true | 0.3750 | 860 | 20.452797 | 0.549059 | 0.355544 | support expansion found only one eligible extra face; final footprint and metrics match v162 |

v162 output root:

```text
/dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective
```

v163 output root:

```text
/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion
```

v163 completed facts:

- dry-run passed with command count `3`, errors `0`, protocol audit passed, and
  W&B offline logging enabled.
- real run completed on `flowers` with GPU `5`.
- manifest status: `COMPLETE`; manifest errors: `[]`.
- protocol audit passed:
  - target GT visible to apply: `false`;
  - target GT visible to eval: `true`;
  - selection uses test GT: `false`.
- runner command return codes:
  - `apply_certified_residual_texture`: `0`, `8684.925s`;
  - `populate_eval_gt_from_target_evidence`: `0`, `34.624s`;
  - `evaluate_vnext_target`: `0`, `46.127s`.
- the top-level runner process exited with code `1` only because W&B tried to
  create `/data/peilincai/mesh-splatting/wandb/offline-run-...` after the
  manifest was already complete and `/data` had no free space.
- a post-hoc W&B offline record was written successfully to:
  `/dev/shm/peilincai_wandb_v163_support_expansion/wandb/offline-run-20260628_065417-1bo6nrvn`.

v163 mechanism facts:

- support expansion mode: `target_footprint_residual_debt_ladder`;
- base faces: `342`;
- eligible extra faces: `1`;
- candidate planning after dominance pruning:
  - base candidate: `342` faces, accepted, `alpha=0.375`,
    train relative gain `0.031881346`;
  - support-expanded candidate: `343` faces, accepted, `alpha=0.375`,
    train relative gain `0.03186874`.
- the expanded candidate was slightly weaker than the base candidate and did not
  change the final sparse allowlist.
- final sparse materialization:
  - allowed bins: `121`;
  - allowed faces: `13`;
  - target changed pixels: `860`;
  - PNG-quantized changed pixels: `849`;
  - changed fraction: `2.318009315e-05`.

v163 quantitative result:

| method | PSNR | SSIM | LPIPS | changed pixels | png-changed pixels |
|---|---:|---:|---:|---:|---:|
| v162 sparse-selective bridge | 20.452797 | 0.549059 | 0.355544 | 860 | 849 |
| v163 support expansion | 20.452797 | 0.549059 | 0.355544 | 860 | 849 |

Interpretation: v163 proves that the support-expansion hook is real and
auditable, but the specific `target_footprint_residual_debt` policy is too weak
on `flowers`: it finds only one extra face and produces no measurable or visible
quality gain over v162.

## Engineering Evaluation

Strong engineering points:

- Train/eval protocol is explicit and auditable.
- The runner now separates no-GT target apply evidence from GT-bearing final
  eval evidence.
- W&B offline logging is enabled for milestone runs.
- Manifests preserve commands, settings, paths, protocol audit, and status.
- Adapter audits capture accepted/fallback state, target changed pixels,
  selected alpha, sparse materialization, bin guard behavior, and target apply
  summaries.
- v162 is a genuine code-level method fix, not just a hyperparameter change.

Current engineering weaknesses:

- Runtime is high. v162 flowers adapter elapsed time was `5771.652s`; v163
  flowers adapter elapsed time increased to `8684.925s`.
- GPU utilization is low during much of vNext; the bottleneck is CPU/IO-heavy
  evidence traversal and NumPy/Python policy validation, not conventional
  GPU-saturated training.
- The current vNext improvement thread is still single-scene diagnostic work,
  not a fixed-policy full9 promotion run.
- The biggest active method bottleneck is target-visible certified footprint,
  not merely alpha selection. v163 confirms this: support expansion added only
  one eligible face and the final allowlist stayed at `121` bins / `13` faces.
- W&B logging must be redirected away from `/data` when `/data` is full. The
  v163 manifest completed, but the top-level process returned `1` because W&B
  could not create a new offline run directory on `/data`.

## Paper-Readiness Evaluation

Paper-ready claims today:

- v106 can be presented as the current strongest local baked representation
  result against clean MeshSplatting on the selected full9 evaluator.
- vNext can be presented as a rigorous certified residual-surface-texture
  research route with strong protocol safeguards and clear failure diagnosis.
- v162 can be presented as a correctness milestone in sparse certification:
  sparse-selective non-regression semantics must survive post-guard bridging.

Claims that are not yet paper-ready:

- vNext does not yet beat clean MeshSplatting or v106 on completed full9
  quantitative evidence.
- v162/v163 cannot be promoted without a fixed-policy full9 run that beats the
  local clean baseline and v106. v163 is not ready for full9 promotion because
  it did not improve even the single-scene `flowers` diagnostic.
- The vNext qualitative advantage is not visually obvious yet because the
  modified target footprint remains tiny.

Current honest paper status:

```text
Engineering scaffold: strong.
Protocol fairness: strong.
v106 metric baseline: positive and usable.
vNext metric superiority: not yet achieved.
Qualitative vNext visual impact: not yet convincing.
Paper-final closed loop: NOT COMPLETE.
```

## Recommended Next Method Work

If v163 does not significantly enlarge target changed footprint, the next real
method change should not be another scalar threshold sweep. The best next change
is **target-visible connected region growth around certified sparse bins**:

- seed from bins already certified by sparse materialization;
- grow to same-face neighboring UV bins visible in target geometry;
- require train policy-val observations and no strong negative evidence;
- preserve post-materialization policy-val re-evaluation as the safety gate;
- record audit fields for seed bin, distance, target pixels/views, policy
  samples/views, and gain/risk.

This directly targets the current bottleneck: the method can certify tiny sparse
regions, but cannot yet materialize enough visible area to move full-image
metrics or visual quality.

## Evidence Index

Repo-local evidence:

- `README.md`
- `docs/car_model/6-28-SPCarNet-Current-Metrics-Engineering-Paper-Evaluation.md`
- `docs/car_model/6-28-v162-SparseSelectiveBridge-Log.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/`
- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

v163 live artifact evidence:

- manifest:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- adapter audit:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/model/surface_residual_region_texture_adapter_audit.json`
- metrics:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/reports/flowers_ours_26000_v163_support_expansion_flowers_test_results.json`
- per-view metrics:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/reports/flowers_ours_26000_v163_support_expansion_flowers_test_per_view.json`
- GT population audit:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/reports/flowers_ours_26000_v163_support_expansion_flowers_test_eval_gt_population_audit.json`
- qualitative renders:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/model/test/ours_26000_v163_support_expansion_flowers/renders/`
- qualitative GT:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/model/test/ours_26000_v163_support_expansion_flowers/gt/`
- logs:
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/logs/02_certified_texture.log`,
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/logs/02b_populate_eval_gt.log`,
  `/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/logs/03_eval.log`
- post-hoc W&B offline record:
  `/dev/shm/peilincai_wandb_v163_support_expansion/wandb/offline-run-20260628_065417-1bo6nrvn`

## Reproduction Notes

The v163 run is reproducible from the manifest command list. The most important
apply-stage command is recorded as the first command in:

```text
/dev/shm/peilincai_spcarnet_20260628_v163_support_expansion/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json
```

The exact runner-level intent was:

```bash
python scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene flowers \
  --output_root /dev/shm/peilincai_spcarnet_20260628_v163_support_expansion \
  --method_name ours_26000_v163_support_expansion_flowers \
  --gpu 5 \
  --strict_no_target_gt_apply \
  --wandb --wandb_mode offline \
  --wandb_project spcarnet_meshprior \
  --wandb_group v163_support_expansion \
  --wandb_name v163_support_expansion_flowers \
  --support_expansion_mode target_footprint_residual_debt \
  --support_expansion_max_extra_faces_candidates 2048,4096
```

Because `/data` was full at the end of the run, the runner process returned
`1` when W&B tried to create a new offline directory under the repo. The
manifest was already `COMPLETE`, all three core commands returned `0`, and a
post-hoc W&B record was written under `/dev/shm`.

## Final Checklist

| item | status | evidence |
|---|---|---|
| real method change in train/eval pipeline | done | v162 sparse-selective semantics fix; v163 support-expansion hook in adapter/runner |
| baseline available | done | local clean MeshSplatting full9 summary in this report |
| current method available | done | v106 full9 and vNext full9 summaries |
| improved-method run | partial | v163 flowers completed, but did not improve v162 |
| ablation | partial | v159/v161/v162/v163 flowers thread documents mechanism deltas |
| metrics saved | done | v163 JSON metrics and full9 summaries listed above |
| qualitative outputs saved | done | v163 render/GT folders listed above |
| commands/configs/result paths documented | done | manifest, logs, and report index listed above |
| final paper story written | partial | current story is honest, but vNext cannot be the headline yet |
| final review marks weaknesses | done | bottlenecks and NOT COMPLETE status recorded |

## Final Status

```text
Final status: NOT COMPLETE.
```

Reason: the best paper-usable result is still v106, which beats local clean
MeshSplatting but only with a small margin. The newer vNext route has strong
engineering and fairness controls, but v163 did not produce a quality gain over
v162 and remains far below the paper-final bar. The active bottleneck is not
another alpha or threshold setting; it is the inability to grow a target-visible
certified residual footprint large enough to affect full-image metrics and
human-visible quality.

Exact next engineering command/prompt:

```text
Implement target-visible connected region growth around certified sparse bins in
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py, expose
it through scripts/car_model/run_vnext_certified_residual_texture_scene.py, then
rerun flowers with W&B offline to /dev/shm. If and only if flowers improves over
v162 on PSNR/SSIM/LPIPS and changed footprint, promote the fixed policy to a
full9 run against clean MeshSplatting and v106.
```

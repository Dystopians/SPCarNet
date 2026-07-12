# v96/v97 Checkpoint-Baked Diagnostics Live Log

Date: 2026-06-25  
Owner: subagent D documentation integrator  
Scope: `counter` checkpoint-baked diagnostics only. Do not promote to paper result unless acceptance checklist passes.

## 1. Motivation

当前 paper-safe endpoint 仍是 Phase-J guarded adaptive ELA + compact checkpoint。它的 RGB/compactness 证据强，但最大收益来自 render-time adapter，不是一个 baked checkpoint。v96/v97 的目的不是再调一个 atlas，而是回答一个更硬的问题：

```text
Can the Phase-J/ELA repair signal be baked into a topology-frozen checkpoint
without losing held-out RGB quality or geometry safety?
```

如果 counter gate 不能过，停止扩展到 hard-triad/full9；把结果作为 representation-level negative evidence。

## 2. What v96 Changed

v96 changes method form from external region-texture/adapter repair to checkpoint-baked recovery:

```text
compact checkpoint @ 26000
  -> train-only Phase-J/ELA teacher renders
  -> parent render rollback
  -> checkpoint render depth/normal anchors
  -> train sparse-depth sentinel cache
  -> topology-frozen recovery checkpoint
```

Key implementation anchor:

```text
scripts/car_model/run_v96_checkpoint_baked_certified_repair_scene.py
```

Confirmed v96 fixcache run:

| Field | Value |
|---|---|
| scene | `counter` |
| load/final iteration | `26000 -> 30000` |
| teacher lambda | `0.02` |
| parent rollback lambda | `1.0` |
| depth/normal render anchor | `0.01 / 0.005` |
| sparse-depth rollback lambda | `0.02` |
| sentinel cache | `24` train views, `12000` sentinels, no test leakage |
| topology | unchanged: `9644247` triangles, `2478825` vertices |

## 3. Cache Bug / Fix Summary

Original v96 wrote the sentinel cache path with a `.json` suffix:

```text
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625/.../sparse_depth_parent_rollback_train_cache.json
```

Observed artifact was instead:

```text
sparse_depth_parent_rollback_train_cache.json.npz
```

The downstream strict recovery command still pointed at `.json`; the original run therefore had no completed final/topology result (`topology_unchanged: null`).

Fixcache rerun made the producer/consumer contract explicit with `.npz`:

```text
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/.../sparse_depth_parent_rollback_train_cache.npz
```

Fix status: complete enough for counter diagnostics; not a promotion.

## 4. Comparison Anchors

Counter promotion gate is strict and pre-declared:

| Anchor | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 counter anchor floor | `>26.7561378479` | `>0.8621263504` | `<0.2516906559` |
| v95 rejected region-texture run | `26.7500514984` | `0.8620513678` | `0.2519962788` |

Full9 context, not the counter promotion gate:

| Context | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| local selected clean MeshSplatting, full9 mean | `25.1517` | `0.7490` | `0.2876` |
| current Phase-J endpoint, full9 mean | `26.4828` | `0.7837` | `0.2243` |

## 5. v96 Fixcache Result

| Run | Iter | PSNR | SSIM | LPIPS | Gate vs v84/v86 |
|---|---:|---:|---:|---:|---|
| v96 fixcache checkpoint-baked | `30000` | `24.9729061127` | `0.7726505399` | `0.3565489948` | FAIL all RGB axes |

Delta vs v84/v86 anchor:

| Metric | Delta | Interpretation |
|---|---:|---|
| PSNR | `-1.7832317352` | below floor |
| SSIM | `-0.0894758105` | below floor |
| LPIPS | `+0.1048583389` | worse; should be lower |

Geometry diagnostic:

| Item | Value |
|---|---:|
| depth count | `15000` |
| depth MAE | `0.0598036154` |
| depth AbsRel | `0.0081289469` |
| normal mean angle deg | `21.5838501595` |
| normal median angle deg | `12.6424128150` |

Result paths:

```text
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/counter_v96_checkpoint_baked_certified_repair/recovery_model/results.json
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/counter_v96_checkpoint_baked_certified_repair/recovery_model/geometry_eval_colmap/iter_30000_max500.json
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/counter_v96_checkpoint_baked_certified_repair/contract/topology_audit.json
```

Conclusion: v96 fixcache is a valid negative diagnostic. It should not be expanded.

## 6. v96 Full Result Placeholders

Only fill these after the counter gate passes; current v96 fixcache does not pass.

| Stage | Required artifact | Status |
|---|---|---|
| counter RGB | `recovery_model/results.json` | filled; FAIL |
| counter geometry | `geometry_eval_colmap/iter_30000_max500.json` | filled; diagnostic only |
| counter topology | `contract/topology_audit.json` | filled; topology unchanged |
| hard-triad RGB | `counter,kitchen,bonsai` same protocol table | `PENDING / blocked by counter fail` |
| full9 RGB | selected clean comparison table | `PENDING / blocked by counter fail` |
| full9 geometry | depth/normal/sparse-depth table | `PENDING / blocked by counter fail` |
| runtime | baked render-only profile vs Phase-J v94 | `PENDING / only if promoted` |

## 7. No-Extra Ablation Placeholder

Purpose: check whether extra teacher/rollback/geometry/sparse-depth terms are helping or hurting. This is not a matched final v96 result because it stops at `28000` and disables the v96 safety terms.

| Run | Iter | Teacher | Rollback | Geo anchors | Sparse rollback | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v96 no-extra continuation | `28000` | `0.0` | `0.0` | `0.0 / 0.0` | `0.0` | `26.4047603607` | `0.8455374837` | `0.2774610221` |

Delta vs v84/v86 anchor:

```text
PSNR -0.3513774872, SSIM -0.0165888667, LPIPS +0.0257703662
```

Geometry placeholder:

| Item | Value |
|---|---:|
| depth MAE | `0.0588230047` |
| depth AbsRel | `0.0077802933` |
| normal mean angle deg | `26.8162771176` |
| topology unchanged | `true` |

Next matched ablation slot:

```text
TODO: same final_iteration=30000, same render/eval protocol, only one component removed at a time.
```

## 8. v97 Pending Placeholder

v97 is a safer-teacher/lower-LR diagnostic, not complete promotion evidence.

Observed partial run:

| Run | Iter | PSNR | SSIM | LPIPS | Status |
|---|---:|---:|---:|---:|---|
| v97 safe teacher LR | `28000` | `26.6566371918` | `0.8581178188` | `0.2615049779` | partial; below anchor |

Delta vs v84/v86 anchor:

```text
PSNR -0.0995006561, SSIM -0.0040085316, LPIPS +0.0098143220
```

Missing before any v97 decision:

| Artifact | Status |
|---|---|
| final `30000` checkpoint | missing |
| topology audit | missing |
| geometry eval | missing |
| hard-triad/full9 expansion | not allowed before counter pass |

## 9. Exact Commands / Result Path Slots

v96 original dry/run evidence:

```text
/dev/shm/spcarnet_v96_dryrun_20260625/counter_v96_checkpoint_baked_certified_repair/v96_manifest.json
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625/counter_v96_checkpoint_baked_certified_repair/v96_manifest.json
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625/counter_v96_checkpoint_baked_certified_repair/build_sparse_depth_sentinel_command.txt
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625/counter_v96_checkpoint_baked_certified_repair/strict_recovery_command.txt
```

v96 fixcache evidence:

```text
/dev/shm/spcarnet_v96_fixcache_dryrun_20260625/counter_v96_checkpoint_baked_certified_repair/v96_manifest.json
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/counter_v96_checkpoint_baked_certified_repair/v96_manifest.json
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/counter_v96_checkpoint_baked_certified_repair/build_sparse_depth_sentinel_command.txt
/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625_fixcache/counter_v96_checkpoint_baked_certified_repair/strict_recovery_command.txt
```

no-extra ablation:

```text
/dev/shm/peilincai_spcarnet_v96_ablation_noextra_20260625/counter_v96_noextra_continuation_ablation/v96_manifest.json
/dev/shm/peilincai_spcarnet_v96_ablation_noextra_20260625/counter_v96_noextra_continuation_ablation/recovery_model/results.json
```

v97 partial:

```text
/dev/shm/spcarnet_v97_dryrun_20260625/counter_v97_safe_teacher_lr/v96_manifest.json
/dev/shm/peilincai_spcarnet_v97_safe_teacher_20260625/counter_v97_safe_teacher_lr/v96_manifest.json
/dev/shm/peilincai_spcarnet_v97_safe_teacher_20260625/counter_v97_safe_teacher_lr/recovery_model/results.json
```

## 10. Acceptance Checklist

Counter gate:

- [x] v96 fixcache cache path contract fixed to `.npz`.
- [x] v96 fixcache counter RGB rendered and evaluated.
- [x] v96 fixcache geometry and topology diagnostics present.
- [ ] PSNR beats `26.7561378479`.
- [ ] SSIM beats `0.8621263504`.
- [ ] LPIPS beats `0.2516906559`.
- [ ] v97 has complete final checkpoint, topology audit, and geometry eval.

Promotion gate:

- [ ] counter passes all RGB axes.
- [ ] geometry/sparse-depth diagnostics do not reveal unacceptable regression.
- [ ] hard-triad `counter,kitchen,bonsai` passes same protocol.
- [ ] full9 table passes against selected clean MeshSplatting baseline.
- [ ] runtime/profile supports baked-checkpoint claim or runtime is excluded.
- [ ] final artifact manifest maps every paper claim to exact paths.

Current verdict:

```text
Do not promote v96/v97. Archive v96 fixcache as a negative checkpoint-baked diagnostic.
Keep v97 pending/partial until complete artifacts exist, but current partial RGB is still below anchor.
```

## 11. Honest Paper-Risk Statement

SPCarNet 的当前强 claim 仍然是 post-training evidence-certified repair/compaction，而不是已经解决 baked representation。v96 fixcache 说明把 teacher/rollback/geometry/sparse-depth 约束直接塞进 topology-frozen checkpoint 会明显损伤 held-out RGB；no-extra 和 v97 partial 虽更接近 anchor，但仍未过 gate，也缺少完整 acceptance artifacts。论文里可以诚实报告这条路线是 active diagnostic/negative evidence，但不能把 v96/v97 写成 surpassed MeshSplatting 或 deployed-speed solution。

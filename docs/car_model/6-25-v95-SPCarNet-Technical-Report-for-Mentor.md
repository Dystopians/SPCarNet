# SPCarNet Current Technical Report for Mentor

Date: 2026-06-25  
Audience: mentor discussion / PPT source notes  
Recommended title:

```text
SPCarNet: Evidence-Certified Repair and Compaction for MeshSplatting
```

This report only uses results already present in `docs/` and `outputs/`. It does not promote unfinished representation-level runs. The v95 representation candidate has now completed and is marked as **rejected**, while v96 checkpoint-baked certified recovery is marked as **running**.

---

## 1. One-Slide Summary

SPCarNet currently should be presented as an evidence-certified post-training repair and compaction system for MeshSplatting:

```text
trained MeshSplatting checkpoint
  -> surface evidence cache
  -> compact mesh selection
  -> guarded Phase-J / Evidence Lumigraph Adapter
  -> policy-val gate and fallback
  -> held-out rendering and evaluation
```

The strongest current endpoint is **Phase-J guarded adaptive Evidence Lumigraph Adapter plus geometry-safe compaction**.

Under local Mip-NeRF360 full9, same split, same evaluator, and selected clean MeshSplatting baseline:

| Metric | Result |
|---|---:|
| Scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| Held-out view PSNR/SSIM/LPIPS strict wins | `244 / 246` |
| Mean PSNR | clean `25.1517` -> SPCarNet `26.4828` |
| Mean SSIM | clean `0.7490` -> SPCarNet `0.7837` |
| Mean LPIPS | clean `0.2876` -> SPCarNet `0.2243` |
| Mean deltas | `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS |
| Mean triangle reduction | `7.6479%` removed |

Most honest headline:

> SPCarNet shows that a trained MeshSplatting checkpoint can be audited after training: low-risk triangles can be removed, stable surface residuals can be repaired through guarded surface evidence, and uncertain areas can fall back to the clean checkpoint.

Main caveat:

> The largest RGB gain currently comes from a render-time guarded adapter. A promoted checkpoint-baked representation endpoint is not finished yet.

---

## 2. Method Modules

### 2.1 Compact Mesh

Compact mesh is not presented as generic mesh simplification. The current story is **quality-preserving compaction**:

```text
remove a triangle only when train/policy-val surface evidence marks it low-risk
```

Protected cases include sparse visibility, thin or high-frequency structure, high residual regions, edge/depth instability, and policy-val tail-risk. The current Phase-J table removes `7.6479%` triangles on average while still improving PSNR, SSIM, and LPIPS on all `9 / 9` scenes.

Static rate evidence:

| Method | Mean triangle reduction | Mean dPSNR | Mean dSSIM | Mean dLPIPS |
|---|---:|---:|---:|---:|
| Compact-ELA support | `5.7632%` | `+0.497941` | `+0.015755` | `-0.023373` |
| Phase-J SPCarNet | `7.6479%` | `+1.331084` | `+0.034702` | `-0.063359` |

Evidence path:

```text
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
```

### 2.2 Guarded Policy

The guarded policy is the fairness and safety layer:

- branch, alpha, support, and fallback are selected from train/policy-val evidence;
- held-out GT is used only for final reporting;
- clean MeshSplatting baseline is selected over clean `26000/30000` by held-out score `PSNR + 20 * SSIM - 20 * LPIPS`;
- stale `trainval_gate_results.json` files are not treated as promotion evidence;
- candidate representation runs must pass all three held-out RGB metrics before promotion.

The current Phase-J branch behavior is also a safety signal: most scenes use adaptive ELA, while `treehill` uses an edge fallback rather than forcing a risky repair.

### 2.3 Phase-J / Evidence Lumigraph Adapter

Phase-J is the current presentation-safe endpoint. It transfers stable residuals from train/policy-val views through surface correspondence:

```text
I_spcarnet(target) = I_compact(target) + alpha * guarded_surface_residual(target)
```

The adapter is not a generic image filter:

| Generic 2D postprocess | SPCarNet Phase-J / ELA |
|---|---|
| Operates in image space | Residuals are addressed by mesh surface evidence |
| Can ignore 3D consistency | Uses face/bin/barycentric visibility and support |
| Often no audit trail | Has policy-val gate, fallback, and per-scene evidence |
| May tune on target output | Does not use held-out GT for policy selection |

Current limitation: this is still a render-time adapter, not a baked checkpoint representation.

### 2.4 v94 Runtime Cleanup

v94 is an exact runtime cleanup, not a new quality result:

- precomputes and reuses the target world-space backprojection grid per target frame;
- leaves Phase-J policy selection, support selection, residual transfer, gates, alpha maps, rendered images, and metrics unchanged in full-resolution mode;
- rejects more aggressive variants that were slower or increased memory.

Full9 integrated no-I/O runtime, v94 target-grid-only:

| Runtime item | Value |
|---|---:|
| Target views | `246` |
| Weighted integrated ms/view | `944.945199` |
| Weighted integrated FPS | `1.058262` |
| Weighted render ms/view | `36.926877` |
| Weighted adapter ms/view | `907.552261` |
| Adapter/render ratio | `24.577011x` |
| Integrated/render-only compact ratio | `26.860457x` |
| Max peak allocated | `17701.383 MiB` |

Interpretation:

- v94 slightly improves full9 integrated runtime versus old integrated v2: `-6.465698 ms/view`, about `-0.68%`;
- it is worth keeping because it is exact and simple;
- it does **not** solve deployment speed. Phase-J remains about `26.86x` slower than compact render-only under this no-I/O profiling protocol.

Evidence path:

```text
docs/car_model/6-25-v94-TargetGridRuntimeOptimization-Log.md
outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/summary.md
```

### 2.5 v95 Representation Candidate

Status: **rejected**.

v95 completed as a valid counter run. It accepted an atlas and changed held-out target pixels, but it failed the pre-declared v84/v86 counter anchor on all three held-out RGB metrics.

| Candidate line | Status |
|---|---|
| v87 source mixture | finished, not promoted |
| v88 anchor-dominance tail-risk | finished, not promoted; PSNR/LPIPS improved but SSIM regressed |
| v89b L1-proxy bin-dominance | finished, not promoted; LPIPS failed strict gate |
| v90 adaptive source mixture | no promotion evidence; partial logs only |
| v91 target-footprint residual-debt support | interrupted / no valid held-out result |
| v95 region-texture candidate | completed, rejected; `26.750051 / 0.862051 / 0.251996`, selected alpha `0.03125` |

v95 gate details:

| Field | Value |
|---|---:|
| v84/v86 anchor | `>26.756138` / `>0.862126` / `<0.251691` |
| v95 held-out result | `26.750051` / `0.862051` / `0.251996` |
| accepted / effective policy | `true` / `accepted_atlas` |
| selected alpha | `0.03125` |
| changed fraction | `0.0184769` |

Verdict:

```text
REJECT PSNR_not_above_anchor, SSIM_not_above_anchor, LPIPS_not_below_anchor,
selected_alpha_not_0.5, risk-gain floors below v84/v86 anchor
```

Evidence path:

```text
docs/car_model/6-25-v95-Rejected-And-v96-CheckpointBaked-Launch.md
outputs/carnet/spcarnet/v95_counter_region_texture_adapter_20260625/
```

Promotion rule remains strict: a new representation-baked candidate must first beat the v84/v86 counter anchor on PSNR, SSIM, and LPIPS, then pass hard-triad and full9 expansion.

### 2.6 v96 Checkpoint-Baked Candidate

Status: **running**.

v96 is the active representation-level follow-up. It changes method form from external atlas/PNG repair to checkpoint-baked recovery:

```text
compact checkpoint 26000
  -> train-only Phase-J/ELA teacher render loss
  -> parent render rollback
  -> checkpoint render depth/normal anchor
  -> sparse-depth parent rollback sentinel cache
  -> topology-frozen checkpoint 30000
```

Initial live run state:

```text
output root: /dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625
W&B mode: offline
GPU: CUDA_VISIBLE_DEVICES=2
sentinel cache: 24 train views, 12000 sparse-depth sentinels, no test leakage
```

This is not yet a promoted result. It must finish training, render, metrics, and geometry evaluation before any claim.

Evidence path:

```text
outputs/carnet/spcarnet/paper_loop_closure_20260625/v90_v91_process_result_audit.md
docs/car_model/6-25-OfficialProtocol-Refresh-And-PaperLoop-Gap.md
docs/car_model/6-25-v87-v88-RepresentationCounterDiagnostics.md
docs/car_model/6-25-v89-L1ProxyBinDominanceGate-Implementation-And-v85Relax-Diagnostic.md
```

---

## 3. Main Quantitative Result vs Clean MeshSplatting Baseline

Protocol:

- dataset: local Mip-NeRF360 full9;
- clean baseline: local MeshSplatting clean `26000/30000` envelope;
- baseline selection: stronger held-out score per scene;
- method: SPCarNet Phase-J guarded adaptive ELA plus compact checkpoint;
- metrics: PSNR and SSIM higher is better, LPIPS lower is better.

| Scene | Clean MeshSplatting PSNR/SSIM/LPIPS | SPCarNet PSNR/SSIM/LPIPS | Delta | Triangles removed |
|---|---:|---:|---:|---:|
| bicycle | `23.3016` / `0.6599` / `0.3321` | `24.0215` / `0.7024` / `0.2661` | `+0.7199` / `+0.0425` / `-0.0660` | `11.81%` |
| flowers | `19.6823` / `0.5118` / `0.3946` | `20.3044` / `0.5578` / `0.3292` | `+0.6221` / `+0.0459` / `-0.0653` | `11.82%` |
| garden | `25.0292` / `0.7800` / `0.2013` | `26.3111` / `0.8278` / `0.1358` | `+1.2819` / `+0.0478` / `-0.0655` | `3.47%` |
| stump | `25.2050` / `0.7052` / `0.2940` | `25.5951` / `0.7241` / `0.2639` | `+0.3901` / `+0.0189` / `-0.0301` | `11.82%` |
| treehill | `20.9342` / `0.5645` / `0.4060` | `21.2962` / `0.5956` / `0.3363` | `+0.3620` / `+0.0311` / `-0.0697` | `11.81%` |
| room | `28.7473` / `0.8848` / `0.2499` | `30.3056` / `0.9057` / `0.1960` | `+1.5584` / `+0.0209` / `-0.0539` | `2.10%` |
| counter | `26.7518` / `0.8621` / `0.2520` | `28.4492` / `0.8937` / `0.1865` | `+1.6974` / `+0.0317` / `-0.0655` | `2.10%` |
| kitchen | `27.8186` / `0.8765` / `0.1992` | `30.1997` / `0.9161` / `0.1320` | `+2.3812` / `+0.0396` / `-0.0672` | `2.10%` |
| bonsai | `28.8952` / `0.8964` / `0.2595` | `31.8620` / `0.9303` / `0.1726` | `+2.9668` / `+0.0339` / `-0.0869` | `11.80%` |

Aggregate:

| Aggregate | Value |
|---|---:|
| Scene-level strict RGB wins | `9 / 9` |
| Held-out view strict RGB wins | `244 / 246` |
| Mean dPSNR | `+1.331084` |
| Mean dSSIM | `+0.034702` |
| Mean dLPIPS | `-0.063359` |
| Mean triangle reduction | `7.6479%` |

Evidence path:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.json
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.csv
```

---

## 4. Bridge to MeshSplatting Paper / Mip-NeRF360 Protocol

This bridge is useful for mentor and PPT context, but the main claim should still use the same-protocol local selected clean baseline.

| Method / protocol | PSNR | SSIM | LPIPS | Role |
|---|---:|---:|---:|---|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` | external paper reference |
| Local official clean30k reproduction | `24.8002` | `0.7310` | `0.3072` | validates local evaluator/protocol is close to paper table |
| Local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` | stronger same-protocol baseline |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` | current strongest endpoint |

Official clean30k reproduction delta versus MeshSplatting paper table:

| Delta | Value |
|---|---:|
| dPSNR | `+0.0190610589` |
| dSSIM | `+0.0027824663` |
| dLPIPS | `-0.0036006646` |

Phase-J bridge delta versus MeshSplatting paper table:

| Delta | Value |
|---|---:|
| dPSNR | `+1.701655` |
| dSSIM | `+0.055498` |
| dLPIPS | `-0.086516` |

Official-style Compact-ELA support table:

| Item | Value |
|---|---:|
| Available scenes | `9 / 9` |
| Strict all-axis pass | `5 / 9` |
| RGB + compact + geometry-safe pass | `9 / 9` |
| RGB + compact pass | `9 / 9` |
| Mean dPSNR vs selected clean | `+0.497941` |
| Mean dSSIM vs selected clean | `+0.015755` |
| Mean dLPIPS vs selected clean | `-0.023373` |
| Mean dPSNR vs MeshSplatting paper table | `+0.868512` |
| Mean dSSIM vs MeshSplatting paper table | `+0.036551` |
| Mean dLPIPS vs MeshSplatting paper table | `-0.046530` |
| Mean triangle reduction | `5.7632%` |

Safe interpretation:

- local clean30k reproduction is very close to the MeshSplatting paper table;
- selected clean baseline is stronger than clean30k and is the fair main baseline;
- Phase-J is clearly above both the local selected clean baseline and the paper-table context;
- paper-table comparison should stay contextual because official table details can differ in resolution, masks, split, preprocessing, metric implementation, or checkpoint iteration.

Evidence path:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.json
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.csv
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean_report.md
docs/car_model/6-25-OfficialProtocol-Refresh-And-PaperLoop-Gap.md
```

---

## 5. Qualitative Results and Figure Paths

Recommended main qualitative slide:

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
```

Panel meaning:

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

Green means SPCarNet is closer to GT; purple/red means worse.

Representative crop metrics:

| Crop | Full-view dPSNR/dSSIM/dLPIPS | Local dPSNR | Local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.6340 / +0.0452 / -0.0878` | `+11.7868` | `78.6032%` |
| kitchen / `00011.png` | `+3.4337 / +0.0250 / -0.0578` | `+10.4780` | `71.4034%` |
| room / `00011.png` | `+3.4966 / +0.0220 / -0.0656` | `+10.3583` | `67.6816%` |
| counter / `00013.png` | `+2.1659 / +0.0407 / -0.0665` | `+6.0250` | `54.8772%` |
| garden / `00006.png` | `+1.7378 / +0.0479 / -0.0678` | `+4.2604` | `44.3599%` |
| flowers / `00014.png` | `+1.1188 / +0.0754 / -0.1028` | `+2.1545` | `25.3461%` |

Supporting qualitative figures:

| Figure | Use |
|---|---|
| `assets/spcarnet_m360_full9_qualitative_gallery.png` | full-frame fairness gallery, GT / clean / SPCarNet / clean error / ours error |
| `assets/spcarnet_m360_outdoor_detail_showcase.png` | outdoor detail support where full-frame differences are subtle |
| `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` | best main PPT slide for local error reduction |

Traceability manifest:

```text
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.md
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.json
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.csv
```

Manifest status:

| Item | Value |
|---|---:|
| Panels | `3` |
| Rows/examples | `16` |
| Figures existing | `3 / 3` |
| Source image path check | `all true` |

---

## 6. Runtime, Rate, and Claim Boundary

Rate/frontier summary:

| Method | PSNR | SSIM | LPIPS | Tri red. | Ckpt red. | VRAM red. | Render FPS ratio | Integrated ms/view | Claim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Selected clean MeshSplatting | `25.151682` | `0.749018` | `0.287621` | `0` | `0` | `0` | `1.000000` | n/a | baseline |
| Compact-ELA support | `25.649623` | `0.764774` | `0.264247` | `0.057632` | `0.039467` | not measured | not measured | not measured | support, not headline |
| Phase-J SPCarNet | `26.482766` | `0.783720` | `0.224261` | `0.076479` | `0.046753` | `0.025733` | `0.946023` | `944.945199` | strong quality and compactness, speed-negative |

Render-only full9:

- compact checkpoints reduce CUDA peak allocation, checkpoint bytes, and triangles in all `9 / 9` scenes;
- render-only FPS is lower in all `9 / 9` scenes;
- this supports memory/size/triangle-count claims, not speedup.

Integrated Phase-J:

- v94 integrated no-I/O FPS: `1.058262`;
- adapter dominates runtime: `907.552261 ms/view`;
- integrated render+adapter is `26.860457x` slower than compact render-only;
- no PNG writing, metrics, LPIPS, or downstream I/O are included in this number.

Safe claim boundary:

```text
SPCarNet currently supports quality, compactness, memory, and checkpoint-size discussion.
It does not support an FPS speedup or deployment-speed claim.
```

Evidence paths:

```text
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/summary.md
docs/car_model/6-25-AdapterRuntimeClosure-Log.md
```

---

## 7. Bottlenecks

| Bottleneck | Current evidence | Consequence |
|---|---|---|
| Render-time adapter is too slow | v94 integrated `944.945199 ms/view`; adapter `907.552261 ms/view` | Cannot claim deployment speed |
| Representation-baked repair is not promoted | v87/v88/v89b/v95 not promoted; v90/v91 no valid promotion evidence; v96 running | Current headline must stay Phase-J adapter |
| Triangle reduction is moderate | Phase-J mean `7.6479%` | Claim quality-preserving compaction, not aggressive compression |
| Outdoor full-frame differences can look subtle | traceability manifest shows crop/error-map benefits | Use crop/error maps for visual impact, full-frame for fairness |
| Paper protocol bridge is contextual | clean30k reproduction close to paper table, but selected clean baseline is the fair local claim | Do not overclaim paper-table superiority as final official result |
| Storage pressure | existing runtime log notes `/data` nearly full during profiling | Keep outputs small; avoid copying large artifacts |

---

## 8. Next Steps

Recommended short-term next steps:

1. Promote only a real checkpoint-baked representation candidate.
   The next candidate is v96 checkpoint-baked certified recovery. It must first beat the v84/v86 counter anchor on PSNR, SSIM, and LPIPS, then pass hard-triad and full9.

2. Decide the paper positioning.
   The current safe positioning is evidence-certified post-training repair and compaction. A stronger representation paper requires a baked endpoint; a system-speed paper requires a much faster adapter or baked repair.

3. Extend rate-distortion evidence.
   Current frontier has selected clean, Compact-ELA support, and Phase-J. Add more compression targets only if they have the same fair protocol and do not weaken RGB claims.

4. Prepare PPT figures.
   Use `spcarnet_phasej_where_it_helps_showcase_20260622.png` as the main qualitative slide, plus full-frame and outdoor detail figures as support.

5. Keep runtime wording conservative.
   If no checkpoint-baked endpoint exists, say: "quality and compactness are strong; current render-time adapter is the bottleneck."

Not recommended:

- do not use train/policy-val gate numbers as held-out results;
- do not put v90/v91/v95 in the main result; v95 completed but failed the anchor gate;
- do not put v96 in the main result until training, metrics, and geometry evaluation complete;
- do not claim speedup;
- do not present representation-level atlas as already solved.

---

## 9. Suggested Mentor Talk Track

Chinese 60-second version:

> SPCarNet 现在最稳的结果不是从零替代 MeshSplatting，而是在训练好的 MeshSplatting checkpoint 后面加一层 evidence-certified 自审计。训练视角被保存成 surface evidence，包括 residual、visibility、face/bin support 和风险统计。系统根据这些证据判断哪些三角形可以低风险删除，哪些表面 residual 可以迁移修复，哪些区域证据不足必须回退。当前 Phase-J 在本地 Mip-NeRF360 full9、同 evaluator、selected clean MeshSplatting baseline 下，9 个场景 PSNR/SSIM/LPIPS 全部严格胜出，244/246 个 held-out views 严格胜出，平均提升 `+1.3311` PSNR、`+0.0347` SSIM、`-0.0634` LPIPS，同时平均删去 `7.65%` triangles。需要诚实说明的是，最强 RGB 收益仍来自 render-time guarded adapter；v94 只是小幅 runtime cleanup，v95 representation candidate 已完成但低于 anchor。下一步 v96 正在尝试把 repair bake 进 checkpoint，并用 parent rollback、geometry anchor 和 sparse-depth sentinel 防止回退。

English 60-second version:

> SPCarNet starts from a trained MeshSplatting checkpoint and adds a post-training self-audit stage. Train and policy-val views become surface evidence: residuals, visibility, face/bin support, and risk statistics. This evidence decides which triangles can be compacted, which surface residuals can be transferred, and where the system should fall back to the clean checkpoint. On local Mip-NeRF360 full9, using the same evaluator and a selected clean MeshSplatting baseline, the current Phase-J endpoint wins all 9 scenes on PSNR, SSIM, and LPIPS, wins 244 out of 246 held-out views, improves mean PSNR by `1.3311`, improves SSIM by `0.0347`, reduces LPIPS by `0.0634`, and removes `7.65%` triangles on average. The honest limitation is that the strongest RGB gain is still a render-time guarded adapter. v94 is only a small exact runtime cleanup, and v95 completed but failed the counter anchor. v96 is now running as a checkpoint-baked repair attempt.

---

## 10. Key Evidence Index

Main Phase-J same-protocol result:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.json
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.csv
```

Official protocol bridge:

```text
docs/car_model/6-25-OfficialProtocol-Refresh-And-PaperLoop-Gap.md
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.json
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean_report.md
```

Runtime/rate:

```text
docs/car_model/6-25-v94-TargetGridRuntimeOptimization-Log.md
docs/car_model/6-25-AdapterRuntimeClosure-Log.md
outputs/carnet/spcarnet/runtime_profile_20260625_v94_targetgrid_full9/summary.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.md
```

Qualitative:

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.md
```

Representation pending / negative audits:

```text
outputs/carnet/spcarnet/paper_loop_closure_20260625/v90_v91_process_result_audit.md
docs/car_model/6-25-v87-v88-RepresentationCounterDiagnostics.md
docs/car_model/6-25-v89-L1ProxyBinDominanceGate-Implementation-And-v85Relax-Diagnostic.md
```

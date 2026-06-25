# 6-25 Official Protocol Refresh and Paper-Loop Gap Log

Date: 2026-06-25

This log records the non-training progress made while the v87/v88/v89b representation-level counter runs were being closed and archived.

The goal is to strengthen the paper-facing evidence loop without confusing transient train/policy-val gate artifacts with held-out test results.

---

## 1. Official Clean MeshSplatting Refresh

I refreshed the official-style MeshSplatting clean30k metric collection from existing full9 artifacts:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/collect_paper_m360_repro_metrics.py \
  --root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --scenes bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai \
  --iteration 30000 \
  --out-csv outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.csv \
  --out-json outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.json
```

Result:

| item | value |
|---|---:|
| completed scenes | `9 / 9` |
| mean PSNR | `24.8001721700` |
| mean SSIM | `0.7310046885` |
| mean LPIPS | `0.3071771132` |
| mean dPSNR vs MeshSplatting paper table | `+0.0190610589` |
| mean dSSIM vs MeshSplatting paper table | `+0.0027824663` |
| mean dLPIPS vs MeshSplatting paper table | `-0.0036006646` |

Interpretation:

- The local official clean30k reproduction is close to the MeshSplatting paper table (`24.78 / 0.728 / 0.310`).
- This supports using the local evaluator/protocol as a reasonable paper-facing comparison basis.
- The stronger selected-clean baseline used in the main SPCarNet report remains stricter than clean30k because it selects the better held-out row from clean `26000/30000`.

Artifacts:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.json
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper_iter30000_refresh_20260625.csv
```

---

## 2. Official-Style Compact-ELA Method Refresh

I also refreshed the official-style Compact-ELA method table using the existing `compact_ela_sor_adaptive_geo_26k` artifacts.

The first refresh attempt used the wrong policy tag and produced `0` rows; it is not used as evidence. The valid refresh used `--policy_tag sor_adaptive_geo`.

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/collect_paper_m360_compact_ela_policy_metrics.py \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --method_root outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --policy_tag sor_adaptive_geo \
  --method_name ours_26000_sor_adaptive_geo_compact_ela \
  --baseline_iterations 26000,30000 \
  --method_iteration 26000 \
  --out_dir outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct
```

Result summary:

| item | value |
|---|---:|
| available scenes | `9 / 9` |
| strict all-axis pass | `5 / 9` |
| RGB + compact + geometry-safe pass | `9 / 9` |
| RGB + compact pass | `9 / 9` |
| mean dPSNR vs selected clean | `+0.497941` |
| mean dSSIM vs selected clean | `+0.015755` |
| mean dLPIPS vs selected clean | `-0.023373` |
| mean dPSNR vs MeshSplatting paper table | `+0.868512` |
| mean dSSIM vs MeshSplatting paper table | `+0.036551` |
| mean dLPIPS vs MeshSplatting paper table | `-0.046530` |
| mean triangle reduction | `5.7632%` removed |

Artifacts:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean_report.md
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean.json
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean.csv
outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_clean_baseline_candidates.csv
```

Interpretation:

- This is a useful official-style paper support table.
- It is not the same as the strongest current Phase-J headline (`+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS, `7.6479%` triangle reduction).
- It is still valuable because it is a stricter full9 paper-protocol bridge: all scenes pass RGB + compact, and all scenes are geometry-safe or strictly better on geometry.

---

## 3. Current Claim Boundary

The main paper/PPT headline should remain:

```text
SPCarNet Phase-J guarded adaptive ELA + geometry-safe compaction
beats selected-clean MeshSplatting on local Mip-NeRF360 full9:
9/9 scene strict RGB wins, 244/246 view strict RGB wins,
mean +1.331084 PSNR, +0.034702 SSIM, -0.063359 LPIPS,
and 7.6479% triangle reduction.
```

The official-style Compact-ELA table should be used as supporting evidence:

```text
Under a paper-facing official clean30k reproduction and selected-clean full9 audit,
Compact-ELA has 9/9 RGB + compact passes and 5/9 strict all-axis passes,
with mean +0.497941 PSNR and 5.7632% triangle reduction.
```

Do not claim that the representation-baked atlas endpoint has reached the Phase-J effect size.

---

## 4. Active Representation-Level Runs

The following runs are still active in `/dev/shm` and are not promoted:

| run | status | note |
|---|---|---|
| `v87_source_mixture_20260625` | finished, not promoted | accepted edit but below v84/v86 anchor on PSNR, SSIM, and LPIPS |
| `v88_anchor_dominance_tailrisk_counter_20260625` | finished, not promoted | accepted edit; PSNR `+1.91e-5` and LPIPS `-2.62e-6` vs anchor, but SSIM is `-8.94e-7`, so it fails the strict all-three promotion gate |
| `v89b_l1proxy_counter_20260625` | finished, not promoted | accepted edit; held-out counter PSNR is `+1.91e-6` over anchor, SSIM ties, LPIPS is `+5.97e-8` worse |

Important correction:

- `trainval_gate_results.json` exists in these roots and has high numbers.
- Those numbers are train/policy-val gate evidence, not final held-out test metrics.
- They must not be used as promotion evidence.

The v89b promotion gate was:

```text
counter PSNR > 26.7561378479
counter SSIM > 0.8621263504
counter LPIPS < 0.2516906559
accepted atlas
target changed fraction >= 0.001
policy-val SSIM/L1 audit no weaker than v84/v86
```

v89b did not pass this strict gate, so it should not expand to hard-triad or full9.

---

## 5. Remaining Paper-Loop Gaps

## 5.1 Static Rate/Profile Artifact

I added a small reproducible collector for static rate/model-size evidence:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/collect_spcarnet_static_rate_profile.py \
  --out_dir outputs/carnet/spcarnet/static_rate_profile_20260625
```

Artifacts:

```text
scripts/car_model/collect_spcarnet_static_rate_profile.py
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.md
outputs/carnet/spcarnet/static_rate_profile_20260625/summary.json
outputs/carnet/spcarnet/static_rate_profile_20260625/per_scene.csv
```

Static summary:

| item | value |
|---|---:|
| scenes | `9` |
| mean Compact-ELA triangle reduction | `0.0576320303` |
| mean Phase-J triangle reduction | `0.0764793954` |
| mean Compact-ELA dPSNR / dSSIM / dLPIPS | `+0.4979406993 / +0.0157554150 / -0.0233734068` |
| mean Phase-J dPSNR / dSSIM / dLPIPS | `+1.3310835097 / +0.0347016388 / -0.0633594493` |
| FPS measured | `false` |
| peak VRAM measured | `false` |

This improves the rate/model-size evidence, but it does not close runtime profiling. A controlled render benchmark is still required before claiming FPS or deployment speed.

## 5.2 Render-Only Runtime Profiling

I added a controlled render-only profiler:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/benchmark_render_runtime_profile.py \
  -m <model_path> \
  --iteration <iter> \
  --split test \
  --max_views <N> \
  --repeats <R> \
  --out_json <out.json> \
  --out_md <out.md>
```

Artifacts:

```text
scripts/car_model/benchmark_render_runtime_profile.py
outputs/carnet/spcarnet/runtime_profile_20260625_smoke/clean30k_counter_test4.json
outputs/carnet/spcarnet/runtime_profile_20260625_smoke/clean30k_counter_test4.md
outputs/carnet/spcarnet/runtime_profile_20260625_smoke/phasej_compact_counter_test4.json
outputs/carnet/spcarnet/runtime_profile_20260625_smoke/phasej_compact_counter_test4.md
outputs/carnet/spcarnet/runtime_profile_20260625_counter_fulltest/clean30k_counter_fulltest.json
outputs/carnet/spcarnet/runtime_profile_20260625_counter_fulltest/clean30k_counter_fulltest.md
outputs/carnet/spcarnet/runtime_profile_20260625_counter_fulltest/phasej_compact_counter_fulltest.json
outputs/carnet/spcarnet/runtime_profile_20260625_counter_fulltest/phasej_compact_counter_fulltest.md
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_full9_renderonly/per_scene.csv
```

Full9 render-only summary on GPU5, test split, all views, `3` repeats:

| item | value |
|---|---:|
| scenes | `9` |
| mean clean FPS | `31.739752` |
| mean compact FPS | `30.075865` |
| mean FPS ratio | `0.946023` |
| mean dFPS | `-1.663887` |
| FPS win scenes | `0 / 9` |
| mean peak allocated reduction | `2.5733%` |
| peak allocated reduction scenes | `9 / 9` |
| mean checkpoint-byte reduction | `4.6753%` |
| checkpoint-byte reduction scenes | `9 / 9` |
| mean triangle reduction | `7.6479%` |

Counter row from the same full-test run:

| profile | FPS | ms/view | peak allocated MiB | peak reserved MiB | triangles | checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|
| clean30k counter | `24.049097` | `41.581638` | `12098.717` | `15152.000` | `9850919` | `764173855` |
| Phase-J compact counter checkpoint | `22.809441` | `43.842166` | `11876.499` | `14828.000` | `9644247` | `747061087` |

Interpretation:

- This benchmark records ms/view, FPS, CUDA peak memory, triangles, and checkpoint bytes without writing PNGs.
- The compact checkpoint reduces triangles, checkpoint bytes, and CUDA memory in all `9 / 9` scenes.
- It is slower in render-only FPS in all `9 / 9` scenes, so the current evidence supports a memory/size claim, not a speed claim.
- It is render-only and does not include `metrics.py`, LPIPS, disk I/O, or the Phase-J render-time Evidence Lumigraph Adapter post-process.
- The integrated no-I/O render+adapter table is now reported in Section 5.3b; a promoted checkpoint-baked candidate remains missing.

## 5.3 Phase-J Adapter Postprocess Runtime

I added and ran an isolated adapter postprocess profiler:

```text
scripts/car_model/benchmark_ela_postprocess_runtime.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_adapter_postprocess/per_scene.csv
```

Scope:

```text
adapt_frame only; no PNG writes; no metrics.py; no LPIPS; no renderer; no policy calibration
```

Full9 summary, test split, all `246` views, `2` repeats per scene:

| item | value |
|---|---:|
| scenes | `9` |
| weighted adapter ms/view | `1061.298183` |
| weighted adapter FPS | `0.942242` |
| max CUDA peak allocated | `4437.766 MiB` |
| render-only compact ms/view | `34.092124` |
| approx render + adapter ms/view | `1095.390307` |
| approx render + adapter FPS | `0.912917` |
| adapter/render time ratio | `31.130304x` |

Interpretation:

- This closes the missing isolated adapter-postprocess runtime evidence.
- The result is negative for speed: current Phase-J render-time adapter dominates runtime.
- This strengthens the paper boundary: Phase-J is a strong quality/compactness endpoint, not a deployable speed endpoint.
- The integrated no-I/O runner below replaces the earlier approximate render+adapter number for paper-facing runtime discussion.

## 5.3b Phase-J Integrated Render + Adapter Runtime

I added and ran a single-process integrated profiler:

```text
scripts/car_model/benchmark_phasej_integrated_runtime.py
scripts/car_model/summarize_phasej_runtime_profiles.py
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.md
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/summary.json
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/per_scene.csv
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/raw/
outputs/carnet/spcarnet/runtime_profile_20260625_phasej_integrated_v2/logs/
```

Scope:

```text
canonical renderer forward + Phase-J adapt_frame in one process; no PNG writes; no metrics.py; no LPIPS; no policy calibration
```

The v2 runner explicitly checks Scene/evidence frame name alignment and reuses one support `FrameLoader` per repeat, so support render/depth tensors are cached instead of being reloaded for every target view.

Full9 summary, test split, all `246` views, `2` repeats per scene:

| item | value |
|---|---:|
| scenes | `9` |
| weighted integrated ms/view | `951.410896` |
| weighted integrated FPS | `1.051071` |
| weighted render ms/view | `37.090434` |
| weighted adapter ms/view | `913.855245` |
| adapter/render time ratio | `24.638570x` |
| max CUDA peak allocated | `17703.596 MiB` |
| render-only compact ms/view | `35.179789` |
| integrated/render-only compact ms ratio | `27.044247x` |

Interpretation:

- This closes the exact no-I/O render+adapter runtime evidence gap.
- The result is still negative for speed: integrated SPCarNet is about `27x` slower than compact render-only under this protocol.
- The remaining runtime gap is no longer "we did not measure integrated runtime"; it is "we measured it and it is too slow unless repair is baked into the checkpoint or the adapter is substantially accelerated."

Additional closure artifacts:

```text
outputs/carnet/spcarnet/paper_loop_closure_20260625/rate_distortion_frontier_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/evidence_manifest_delta_20260625.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/runtime_adapter_gap_audit.md
outputs/carnet/spcarnet/paper_loop_closure_20260625/v90_v91_process_result_audit.md
```

## 5.4 Qualitative Traceability Manifest

I added a provenance manifest for the current presentation figures:

```text
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.md
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.json
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.csv
```

Summary:

| item | value |
|---|---:|
| panels | `3` |
| rows | `16` |
| figures existing | `3 / 3` |
| source image path check | `all true` |

Covered panels:

| panel | examples | role |
|---|---:|---|
| `phasej_where_it_helps` | `6` | main Phase-J local error-reduction qualitative figure |
| `compact_ela_outdoor_detail` | `5` | outdoor-detail support figure |
| `compact_ela_fullframe_gallery` | `5` | full-frame fairness/support figure |

This closes the immediate PPT provenance gap: each shown example is tied to scene, view, metric deltas, crop coordinates when applicable, and source image paths. A stronger paper appendix can still add regenerated per-crop PNGs and error-map thumbnails, but the current figure provenance is no longer missing.

## 5.5 Open Gaps

| priority | gap | required artifact |
|---:|---|---|
| P0 | decide whether representation-baked repair can be promoted | v87/v88/v89b are closed and not promoted; a new candidate must beat the v84/v86 counter anchor on all three metrics before hard-triad/full9 expansion |
| P1 | deployment-speed FPS/VRAM evidence | render-only, isolated adapter, and integrated no-I/O render+adapter tables exist; still need PNG/I/O/metrics deployment profile only if claiming deployment speed, plus any promoted checkpoint-baked candidate |
| P1 | qualitative appendix assets | provenance manifest exists; optional next step is regenerated per-crop PNG/error-map thumbnails for appendix |
| P1 | old Phase-S strict collector | either regenerate missing `treehill/counter` gates or explicitly deprecate Phase-S from the paper loop |
| P2 | evidence manifest refresh | new manifest after official protocol, qualitative traceability, profiling, and v89 diagnostic are closed |

---

## 6. Verdict

Progress improved today because the official-style comparison evidence is now clearer:

- clean30k reproduction is refreshed and close to the MeshSplatting paper table;
- Compact-ELA official-style full9 support table is refreshed and complete;
- static rate/model-size evidence is now reproducibly collected;
- render-only, isolated adapter, and integrated no-I/O runtime profilers are now available, with deployment-speed and checkpoint-baked runtime still open;
- qualitative figure provenance is now explicitly traced for the current presentation panels;
- the report boundary is sharper: Phase-J is the headline, official Compact-ELA is paper-protocol support, representation-baked atlas is still ongoing.

The full engineering + paper loop is still not complete.

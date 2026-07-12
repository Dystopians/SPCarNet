# v48 Auto-Support Surface Atlas Log

Date: 2026-06-23

## Summary

v48 extends the v47 auto-capacity guarded surface atlas with a train-only support-expansion policy. The goal is to address the main v47 bottleneck: the atlas can only change pixels whose target rays hit carrier-supported faces, and v47 support coverage was often below 2%.

The new policy adds candidate faces from fit-view residual evidence, then lets the existing train policy-val gate decide whether expanded support is safe. It is not a per-scene hand-picked parameter: the same candidate rule and guard are applied to all four validation scenes.

Result on `garden/room/counter/bonsai`:

| comparison | strict wins | nonregressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs no-op | 4 / 4 | 4 / 4 | +0.002506 | +0.00004284 | -0.00005950 |
| v48 vs v42 | 3 / 4 | 4 / 4 | +0.001527 | +0.00003043 | -0.00004843 |
| v48 vs v43 | 4 / 4 | 4 / 4 | +0.001727 | +0.00003681 | -0.00004688 |
| v48 vs v46 | 3 / 4 | 4 / 4 | +0.001388 | +0.00003016 | -0.00004649 |
| v48 vs v47 | 3 / 4 | 4 / 4 | +0.001340 | +0.00002900 | -0.00004260 |

The important qualitative interpretation is conservative: v48 is a real representation-level improvement over v47, but its absolute effect size is still much smaller than Phase-J render-time ELA. It should be used as a recent-method / ablation slide, not as the headline endpoint.

Full9 extension was completed after the four-scene run by adding `bicycle`,
`flowers`, `stump`, `treehill`, and `kitchen` under the same fixed v48 policy.
Because `/data` was nearly full, the large evidence/render outputs were staged
in `/dev/shm/peilincai_spcarnet_v48_full9_20260623`; only small audit/results
artifacts were copied back to `/data` for durable provenance.

Result on full9 versus same-evidence no-op compact baseline:

| comparison | strict wins | nonregressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v48 vs no-op full9 | 7 / 9 | 8 / 9 | +0.001462 | +0.00002774 | -0.00003953 |

The two non-strict cases are informative. `stump` was rejected by the train
policy-val gates and is reported as an effective no-op fallback. `treehill`
improves PSNR and SSIM but regresses LPIPS by `+0.00001916`, so it is not a
strict three-metric win. This makes v48 a safer, broader representation-level
evidence point than v47, but still not a Phase-J replacement.

## Method Change

Implementation:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
```

New interface:

```text
--support_expansion_mode {none,fit_residual_topk}
--support_expansion_max_extra_faces 2048
--support_expansion_min_face_samples 128
--support_expansion_min_mean_l1 0.003
```

The v48 policy:

1. Keep the original v47 `base_carrier` support as a safe candidate.
2. Scan fit views only, excluding policy-val views using the same `policy_val_stride`.
3. For non-carrier faces, accumulate valid residual evidence by face id.
4. Rank extra faces by `mean_l1 * log1p(samples)`.
5. Add up to `2048` extra faces if they pass sample and residual thresholds.
6. Evaluate both support candidates under the same train policy-val gate:
   - `base_carrier`
   - `fit_residual_topk`
7. Jointly select support mode, texture size, fill mode, and alpha with existing non-regression guards.
8. Fall back to conservative base behavior if expanded support is not policy-val safe.

The candidate grid used in this run:

```text
support mode: base_carrier, fit_residual_topk
texture_size: 8,16,24,32
fill mode: auto_policy over face_mean / nearest_observed
alpha grid: 0,0.015625,0.03125,0.0625,0.125
```

## Fixed Validation Command Pattern

The four-scene run used the same policy. Per-scene paths differ only by scene/evidence root:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
  --fit_evidence_dir <train-evidence-dir> \
  --target_evidence_dir <target-evidence-dir> \
  --region_carrier_json <policy-val-pruned-region-carriers.json> \
  --output_model outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/<scene>_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter \
  --target_split test \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --method_name ours_26000_<scene>_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter \
  --texture_size 16 \
  --texture_size_candidates 8,16,24,32 \
  --support_expansion_mode fit_residual_topk \
  --support_expansion_max_extra_faces 2048 \
  --support_expansion_min_face_samples 128 \
  --support_expansion_min_mean_l1 0.003 \
  --atlas_empty_bin_fill_mode auto_policy \
  --select_alpha_by_risk_gate \
  --enable_policy_val_image_ssim_gate \
  --force
```

Metrics were run with:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python metrics.py \
  -m outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/<scene>_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter
```

## Four-Scene Results

| scene | PSNR | SSIM | LPIPS | support | +faces | texture | fill | alpha | changed |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| garden | 24.742157 | 0.75409001 | 0.24797200 | fit_residual_topk | 977 | 16 | nearest_observed | 0.1250 | 1.8491% |
| room | 28.740660 | 0.88482928 | 0.24989747 | base_carrier | 0 | 16 | face_mean | 0.1250 | 1.0602% |
| counter | 26.753004 | 0.86208355 | 0.25191796 | fit_residual_topk | 2048 | 32 | nearest_observed | 0.1250 | 4.2312% |
| bonsai | 28.868425 | 0.89606690 | 0.25924534 | fit_residual_topk | 2048 | 16 | nearest_observed | 0.1250 | 2.0578% |

### vs no-op

| scene | dPSNR | dSSIM | dLPIPS | strict |
|---|---:|---:|---:|---:|
| garden | +0.001154 | +0.00004101 | -0.00005122 | 1 |
| room | +0.001656 | +0.00003928 | -0.00001849 | 1 |
| counter | +0.003168 | +0.00003421 | -0.00008002 | 1 |
| bonsai | +0.004045 | +0.00005686 | -0.00008827 | 1 |

### vs v47

| scene | dPSNR | dSSIM | dLPIPS | strict | interpretation |
|---|---:|---:|---:|---:|---|
| garden | +0.000895 | +0.00003618 | -0.00003994 | 1 | expanded support accepted |
| room | +0.000000 | +0.00000000 | +0.00000000 | 0 | safe tie / base-carrier fallback |
| counter | +0.001593 | +0.00002706 | -0.00005126 | 1 | expanded support accepted |
| bonsai | +0.002874 | +0.00005275 | -0.00007921 | 1 | expanded support accepted |

## Full9 Extension Results

The missing five scenes were run with the same fixed v48 policy and no
scene-specific parameter changes. `stump` is the only policy rejection and is
counted as fallback no-op in the effective full9 summary.

| scene | PSNR | SSIM | LPIPS | support | +faces | texture | fill | alpha | changed |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| bicycle | 23.294018 | 0.65965807 | 0.33226576 | fit_residual_topk | 1453 | 24 | nearest_observed | 0.1250 | 0.8947% |
| flowers | 19.668833 | 0.51168275 | 0.39478543 | fit_residual_topk | 1961 | 32 | nearest_observed | 0.0312 | 2.0957% |
| garden | 24.742157 | 0.75409001 | 0.24797200 | fit_residual_topk | 977 | 16 | nearest_observed | 0.1250 | 1.8491% |
| stump | 25.180920 | 0.70441955 | 0.29421365 | fallback no-op | 0 | 8 | face_mean | 0.1250 | 0.0000% |
| treehill | 20.923422 | 0.56422561 | 0.40612733 | fit_residual_topk | 933 | 24 | nearest_observed | 0.0312 | 1.3127% |
| room | 28.740660 | 0.88482928 | 0.24989747 | base_carrier | 0 | 16 | face_mean | 0.1250 | 1.0602% |
| counter | 26.753004 | 0.86208355 | 0.25191796 | fit_residual_topk | 2048 | 32 | nearest_observed | 0.1250 | 4.2312% |
| kitchen | 27.818651 | 0.87650901 | 0.19906622 | fit_residual_topk | 2048 | 24 | nearest_observed | 0.1250 | 2.9758% |
| bonsai | 28.868425 | 0.89606690 | 0.25924534 | fit_residual_topk | 2048 | 16 | nearest_observed | 0.1250 | 2.0578% |

### vs no-op full9

| scene | dPSNR | dSSIM | dLPIPS | strict | nonreg/tie |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.000536 | +0.00000691 | -0.00000879 | 1 | 1 |
| flowers | +0.000137 | +0.00000477 | -0.00000215 | 1 | 1 |
| garden | +0.001154 | +0.00004101 | -0.00005122 | 1 | 1 |
| stump | +0.000000 | +0.00000000 | +0.00000000 | 0 | 1 |
| treehill | +0.000195 | +0.00000209 | +0.00001916 | 0 | 0 |
| room | +0.001656 | +0.00003928 | -0.00001849 | 1 | 1 |
| counter | +0.003168 | +0.00003421 | -0.00008002 | 1 | 1 |
| kitchen | +0.002270 | +0.00006449 | -0.00012597 | 1 | 1 |
| bonsai | +0.004045 | +0.00005686 | -0.00008827 | 1 | 1 |

## Evidence Paths

| content | path |
|---|---|
| v48 summary MD | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_multiscene_summary.md` |
| v48 summary JSON | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_multiscene_summary.json` |
| v48 full9 summary MD | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.md` |
| v48 full9 summary JSON | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_autosupport_autocap_guarded_v42calib_full9_summary.json` |
| durable small-artifact archive | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v48_full9_missing_scene_small_artifacts_20260623` |
| temporary full evidence/render root | `/dev/shm/peilincai_spcarnet_v48_full9_20260623` |
| garden output | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter` |
| room output | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/room_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter` |
| counter output | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter` |
| bonsai output | `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_v48_autosupport_autocap_guarded_v42calib_region_texture_adapter` |
| implementation | `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` |
| multiscene summarizer | `scripts/car_model/summarize_surface_residual_atlas_multiscene.py` |

## Current Judgment

v48 is better than v47 as the latest representation-level story because it attacks the measured support bottleneck rather than only changing atlas capacity. It passes the same four-scene no-test-GT validation with strict positive results over no-op and safe non-regression versus v47. The full9 extension is weaker but still useful: it is positive on mean metrics and `7 / 9` strict versus no-op, with one deliberate no-op fallback and one LPIPS regression.

The limitation remains clear: the absolute gain is still small. For the mentor PPT, v48 should be framed as:

> Evidence that residual repair can be pushed from render-time ELA toward a train-only, surface-addressed, self-expanding representation.

It should not be framed as:

> A replacement for Phase-J or a final top-conference endpoint.

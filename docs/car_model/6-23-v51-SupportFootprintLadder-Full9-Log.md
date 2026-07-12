# v51 Support-Footprint Ladder Full9 Log

Date: 2026-06-23

Status: `NOT COMPLETE`. v51 is a real train/eval pipeline change and a useful capacity-bottleneck probe, but it is not a global replacement for v48.

## Motivation

v48 auto-support surface atlas is the current strongest representation-level full9 result, but `counter`, `kitchen`, and `bonsai` all hit the `+2048` extra-face support cap. That suggests the bottleneck is not just alpha, texture size, or fill mode. The atlas sometimes needs a larger train-evidence support footprint to cover high-residual surface regions.

v51 tests this hypothesis by replacing the single `topK=2048` support expansion with a fixed train-only support ladder.

## Code Changes

Implementation files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

Main additions:

- `rank_fit_residual_extra_faces(...)`: ranks non-carrier faces by fit-view residual evidence.
- `expanded_candidate_faces_from_ranked_rows(...)`: materializes support candidates from the ranked residual-face list.
- `--support_expansion_max_extra_faces_candidates`: evaluates a fixed ladder such as `2048,4096`.
- runner forwarding for `--texture_size_candidates` and `--atlas_empty_bin_fill_mode`.

The full9 fixed policy was intentionally small:

```text
support candidates: 2048,4096
texture size: 32
fill mode: nearest_observed
L1 positive-view gate: 0.5
target changed fraction gate: 0.0
```

## Failed Broad Grid Diagnostic

The first `counter` probe used a broad `4 support x 4 texture x 2 fill = 32` candidate grid:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --output_root /dev/shm/peilincai_spcarnet_v51_support_ladder_20260623 \
  --tag v51_support_ladder_l1pos05_trainpolicy_fairnoop_region_texture_adapter \
  --min_policy_val_l1_positive_view_fraction 0.5 \
  --support_expansion_max_extra_faces_candidates 1024,2048,4096 \
  --force
```

This run was manually interrupted after roughly 30 minutes. The lesson is that v51 should not be deployed as another wide parameter grid. It should be a fixed capacity probe or a fixed train-only meta-policy.

## Full9 Command Template

```bash
CUDA_VISIBLE_DEVICES=<gpu> python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene <scene> \
  --gpu <gpu> \
  --output_root /dev/shm/peilincai_spcarnet_v51_fast_support_ladder_20260623 \
  --tag v51_fast_support_ladder_tex32_nearest_l1pos05_region_texture_adapter \
  --min_policy_val_l1_positive_view_fraction 0.5 \
  --min_target_changed_fraction 0.0 \
  --support_expansion_max_extra_faces_candidates 2048,4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --force
```

## Full9 Summary

Evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v51_fast_support_ladder_small_artifacts_20260623
```

| comparison | strict scene wins | non-regressive/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|
| v51 vs same-evidence no-op | `6 / 9` | `6 / 9` | `+0.001380497` | `+0.000030067` | `-0.000051187` |
| v51 vs v48 | `3 / 9` | `3 / 9` | `-0.000081804` | `+0.000002331` | `-0.000011659` |
| v51 vs v50 | `5 / 9` | `8 / 9` | `+0.000116136` | `+0.000008331` | `-0.000017136` |

| scene | policy | support | +faces | changed | vs no-op | vs v48 | vs v50 |
|---|---|---|---:|---:|---|---|---|
| bicycle | accepted_atlas | fit_residual_topk_2048 | 1453 | 0.9063% | S | - | S |
| flowers | fallback_noop | base_carrier | 0 | 0.0000% | - | - | N |
| garden | accepted_atlas | base_carrier | 0 | 0.3515% | S | - | S |
| stump | fallback_noop | base_carrier | 0 | 0.0000% | - | - | N |
| treehill | fallback_noop | base_carrier | 0 | 0.0000% | - | - | N |
| room | accepted_atlas | base_carrier | 0 | 1.0474% | S | - | - |
| counter | accepted_atlas | fit_residual_topk_4096 | 4096 | 6.5362% | S | S | S |
| kitchen | accepted_atlas | fit_residual_topk_4096 | 2952 | 3.9361% | S | S | S |
| bonsai | accepted_atlas | fit_residual_topk_4096 | 2832 | 2.6786% | S | S | S |

`S` means strict PSNR/SSIM/LPIPS win. `N` means non-regressive/tie. `-` means at least one metric regresses.

## Interpretation

v51 validates the support-capacity hypothesis on the three cap-hit scenes:

- `counter`: v51 beats v48/v50/no-op on all three metrics.
- `kitchen`: v51 beats v48/v50/no-op on all three metrics.
- `bonsai`: v51 beats v48/v50/no-op on all three metrics.

However, v51-fast uses fixed `texture=32` and fixed `nearest_observed` fill. That removes v48's auto-capacity/auto-fill advantage and causes regressions on several non-cap-hit scenes. Therefore, v51 is not the current full9 headline method.

The next fair design should be a fixed scene-agnostic meta-policy:

```text
if train-only capacity diagnostics indicate support-cap bottleneck:
    enable v51 support ladder
else:
    keep v48 auto-support / auto-capacity / auto-fill policy
```

This is the correct next step because it uses v51 as a capacity-aware representation upgrade rather than another manual parameter game.

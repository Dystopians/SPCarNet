# v113c Frame Fallback and v114 OOF-Refit Log

Date: 2026-06-25

## Why This Step Exists

v113b made the strict gate safe by falling back to the v106 parent when the target render landed outside train/even support. That fixed the garden v110b regression, but it also proved the branch was safety-only: it did not create a better candidate than v106.

This update separates two issues:

1. v113c improves the gate policy from scene-level OOT fallback to frame-level OOT fallback.
2. v114 starts the next real quality attempt on the candidate side with OOF-refit POD-MoE.

## v113c Method

The v109 render-realized parent gate now supports:

```text
--oot_gate_mode frame_fallback
```

When OOT support fails, this mode disables only target frames whose camera centers are outside the source-support certificate. It does not read target GT. It writes the disabled frame list and post-fallback mask-weighted OOT exposure into the gate report.

Implementation:

```text
scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py
scripts/car_model/run_v113b_oot_tail_gate_replay_scene.py
scripts/car_model/smoke_test_v109_oot_gate.py
```

The replay runner report path was also fixed so different gate methods no longer overwrite the same orchestration report.

## v113c Garden Result

Artifacts:

```text
docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/garden/
docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md
docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.json
```

| method | PSNR | SSIM | LPIPS | note |
|---|---:|---:|---:|---|
| clean MeshSplatting | 25.029211 | 0.780035 | 0.201314 | local clean baseline |
| v106 parent | 25.790945 | 0.799382 | 0.174480 | current quality line |
| v110b | 25.430321 | 0.783703 | 0.186970 | strict train/even candidate gate |
| v113b scene fallback | 25.790945 | 0.799382 | 0.174480 | full scene fallback to v106 |
| v113c frame fallback | 25.499817 | 0.786888 | 0.184260 | frame-level OOT fallback |

v113c improves over v110b on garden:

```text
+0.069496 PSNR
+0.003185 SSIM
-0.002710 LPIPS
```

But it remains below v106:

```text
-0.291128 PSNR
-0.012494 SSIM
+0.009780 LPIPS
```

OOT detail:

```text
target_frame_fraction: 0.083333
mask_weighted_ood_fraction before frame fallback: 0.090031
mask_weighted_ood_fraction after frame fallback: 0.0
frame_fallback_count: 2
frame_fallback_images: 00020.png, 00021.png
```

Conclusion: v113c is a useful safety refinement and avoids over-conservative whole-scene fallback, but it still cannot solve the main quality gap.

## v114 Method

v114 is a candidate-side change:

```text
method_version = v114_oof_refit_pod_moe
```

It changes POD-MoE reliability instead of only changing the gate. Final coefficients are fit from train/all, but expert reliability is capped by out-of-fold gains:

```text
both folds positive -> use conservative min gain/scale
one fold positive -> allow the expert with half gain/scale
zero or nonpositive OOF gain -> zero reliability
joint two-expert descent lock remains active
```

This targets the observed strict-split failure: train/even-only candidates lose too much boundary/detail capacity, while old cross-fit requires both directions and suppresses sparse but valid evidence.

Implementation:

```text
scripts/car_model/build_v105_evidence_gated_mixture_field.py
scripts/car_model/smoke_test_v114_oof_refit.py
```

Smoke/static checks passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/build_v105_evidence_gated_mixture_field.py \
  scripts/car_model/smoke_test_v114_oof_refit.py \
  scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py \
  scripts/car_model/run_v113b_oot_tail_gate_replay_scene.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v114_oof_refit.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v109_oot_gate.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v113b_replay_runner_args.py
```

## v114 Running Experiment

Garden v114 field build is running:

```bash
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_v105_evidence_gated_mixture_field.py \
  --model_path /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/garden/detached_model \
  --delta_bank_path /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/garden/v102_preprojected_delta_bank_train.pt \
  --output_field /dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden/fields/ours_26000_v114_oof_refit_podmoe_garden_field.pt \
  --endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --iteration 26000 \
  --split train \
  --view_subset all \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --field_variant pod_moe \
  --method_version v114_oof_refit_pod_moe \
  --gate_source crossfit_risk \
  --view_gate_temperature 0.0 \
  --min_count 1 \
  --min_views 1 \
  --ridge 0.001 \
  --residual_clip 0.08 \
  --view_std_floor 0.0001 \
  --rank_rtol 1e-07 \
  --condition_max 100000000.0 \
  --gate_boost 0.5 \
  --chunk_pixels 262144
```

Local runtime root:

```text
/dev/shm/peilincai_spcarnet_v114_oof_refit_20260625/garden/
```

Status: running. It is not yet a completed quality result.

## Honest Assessment

v113c confirms that OOT granularity matters, but the remaining gap is not primarily the gate. The branch needs a stronger candidate. v114 is the first candidate-side attempt in this round that directly addresses the strict-split capacity loss without per-scene parameter scanning.

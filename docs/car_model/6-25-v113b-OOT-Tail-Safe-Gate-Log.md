# v113b OOT Tail-Safe Parent Gate Log

Date: 2026-06-25

## Motivation

The v110/v110b strict gate exposed a concrete weakness: a candidate can look strongly positive on train/odd calibration views and still regress on held-out test. Garden was the clearest example:

```text
v106 parent: 25.790945 / 0.799382 / 0.174480
v110b gate: 25.430321 / 0.783703 / 0.186970
```

The failure was not target GT leakage. The gate used no target GT for policy selection. The failure was that average calibration gain did not prove target-trajectory safety.

## Method Change

v113b upgrades `meshsplatopt_v109_render_realized_parent_gate.py` with two certificates.

### 1. Per-Metric Lower-Tail Certificate

The old gate already checked a 5th-percentile balanced score. That was insufficient because SSIM-weighted score could remain positive while a few calibration frames had negative PSNR gain.

v113b records and can constrain:

```text
p05_d_psnr
p05_d_ssim
p05_d_lpips
```

The v110/v111 runners now pass:

```text
--min_p05_psnr_gain=0.0
--min_p05_ssim_gain=-1e-06
--min_p05_lpips_gain=-1000000000.0
```

### 2. Target-GT-Free OOT Support Certificate

v113b also checks whether nonzero target/test masks lie within the camera support of the train/even source frames used to build the candidate field.

Inputs:

- `camera_index.json` from calibration and target render directories;
- candidate field `.manifest.json`, especially `selected_frame_keys`;
- parent/candidate renders for mask statistics.

Not used:

- target/test GT for gate decision.

Default v110/v111 mode is now:

```text
--oot_gate_mode scene_fallback
--oot_center_quantile 0.95
--oot_center_rel_margin 0.0
--oot_center_abs_margin 0.0
--oot_max_frame_fraction 0.10
--oot_max_mask_weighted_fraction 0.05
```

If the target mask is concentrated on views outside the empirical calibration-to-source support envelope, the gate overwrites the candidate output with the parent.

## Verification

Static/smoke checks passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  scripts/car_model/smoke_test_v109_gate_subset.py \
  scripts/car_model/smoke_test_v109_oot_gate.py \
  scripts/car_model/smoke_test_v110_strict_runner_args.py \
  scripts/car_model/smoke_test_v111_runner_args.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v109_gate_subset.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v109_oot_gate.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v110_strict_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v111_runner_args.py
```

W&B offline runs:

```text
flowers v113b: /data/peilincai/mesh-splatting/wandb/offline-run-20260625_222217-91exngmi
garden v113b:  /data/peilincai/mesh-splatting/wandb/offline-run-20260625_221816-ue3fm1ck
```

## Results

Durable result summary:

```text
docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md
```

| scene | v110b | v113b | result |
|---|---:|---:|---|
| flowers | 20.077723 / 0.531240 / 0.374393 | 20.077723 / 0.531240 / 0.374393 | remains parent-safe |
| garden | 25.430321 / 0.783703 / 0.186970 | 25.790945 / 0.799382 / 0.174480 | regression repaired back to v106 |

Garden OOT evidence:

```text
calib p95 center support: 0.757181
target p95 center distance: 0.806651
mask-weighted OOD fraction: 0.090031
threshold: 0.05
decision: scene fallback to parent
```

## Current Claim Boundary

Safe to claim:

- v113b is a real gate-method improvement over v110b for reliability.
- v113b fixes the observed garden strict-gate regression without using target GT.
- v113b keeps the strict branch non-regressive to the v106 parent on the two completed representative scenes.

Not safe to claim:

- v113b is a quality improvement over v106. It falls back to v106 on flowers and garden.
- v113b completes the full paper loop. Counter, bonsai, and v111 end-to-end strict runs are still unfinished.

## Next Work

The next research step is not more fallback. We need a candidate generator that can pass the v113b lower-tail and OOT certificates while improving beyond v106. Until then, v113b should be presented as a safety certificate and failure-mode repair, not as the final paper method.

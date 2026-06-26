# v110 Strict-Split Parent-Gated Evidence Log

Created: 2026-06-25 19:26 PDT

## Purpose

v106 POD-MoE remains the current stable representation anchor, but the earlier v106/v108 path was still a target-camera sidecar diagnostic. v109 added render-realized parent preservation, but flowers selected `mask=0` and only preserved v106. v110 is the first strict-split attempt to close that fairness gap:

- fit candidate field on `train/even`;
- calibrate the parent gate on held-out `train/odd`;
- evaluate only on `test`;
- keep v106 as immutable parent, so harmful candidates should fall back rather than regress.

This is a fairness and reliability milestone, not yet a confirmed quality breakthrough.

## Implemented Interfaces

1. `build_v105_evidence_gated_mixture_field.py`
   - Added `--view_subset {all,even,odd}`.
   - Preserves original split-local frame indices for delta-bank keys and camera metadata.
   - Writes `source_available_frames`, `source_target_frames`, `view_subset`, `selected_frame_indices`, and `selected_frame_keys` into payload/manifest.

2. `run_v102_preprojected_delta_scene.py`
   - Added `--target_split {test,train}`.
   - Train banks are written as `v102_preprojected_delta_bank_train.pt` to avoid overwriting historical test banks.
   - Train-bank verification now falls back to comparing build renders against fast preprojected renders when a separate train reference method is absent.

3. `meshsplatopt_v109_render_realized_parent_gate.py`
   - Added `--calib_view_subset {all,even,odd}`.
   - Filters calibration PNGs by numeric filename stem before sampling.
   - Reports `calib_view_subset`, `calib_candidate_count`, and `calib_selected_count`.

4. `run_v110_strict_split_parent_gate_scene.py`
   - New orchestration script.
   - Runs train-even field build, candidate train/test render, train-odd v109 gate, and test evaluation.
   - Writes commands, return codes, logs, paths, metrics, and gate report into one JSON/MD report.

## Smoke/Static Checks

Passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/build_v105_evidence_gated_mixture_field.py \
  scripts/car_model/smoke_test_v105_view_subset.py \
  scripts/car_model/meshsplatopt_v109_render_realized_parent_gate.py \
  scripts/car_model/smoke_test_v109_gate_subset.py \
  scripts/car_model/run_v102_preprojected_delta_scene.py \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  scripts/car_model/smoke_test_v110_strict_runner_args.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v105_view_subset.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v109_gate_subset.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v110_strict_runner_args.py
```

## Train-Bank Status

Output root:

```text
/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625
```

Completed and verified metadata:

| scene | train bank | split | frames/deltas | note |
|---|---|---:|---:|---|
| flowers | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/flowers/v102_preprojected_delta_bank_train.pt` | train | 151/151 | bank valid; old report marked false because train reference render was absent |
| garden | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/garden/v102_preprojected_delta_bank_train.pt` | train | 161/161 | bank valid; v106 train parent render was missing and is being generated |

Still running at log creation:

- counter train bank on GPU1;
- bonsai train bank on GPU5.

## Running Strict Experiments

### flowers v110 strict

Started on GPU2 with W&B offline logging:

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  --scene flowers \
  --gpu 2 \
  --merge_model_results \
  --wandb \
  --output_root /dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625
```

Current stage at log creation:

```text
build_train_even_candidate_field
log: /dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/logs/01_build_candidate_field.log
progress: train/even field build running, 76 selected views
```

Expected outputs:

```text
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/reports/flowers_v110_strict_split_parent_gate_report.json
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/reports/flowers_v110_strict_split_parent_gate_report.md
```

### garden preparation

Garden train bank exists, but v106 parent train renders were missing. They are being generated on GPU3:

```bash
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python render.py \
  -m /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/garden/detached_model \
  --iteration 26000 \
  --skip_test \
  --checkpoint_endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --checkpoint_endpoint_output_method ours_26000_v106_podmoe_basepreserve_garden \
  --checkpoint_endpoint_surface_field_path /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_field/garden/v106_podmoe_mc1_mv1_r1em03_clip8em02_vs1em04_rr1em07_cm1ep08_gb5em01_normal_equation_vgt0ep00_float16_s4_field.pt \
  --checkpoint_endpoint_require_surface_field \
  --checkpoint_endpoint_no_intermediate_outputs \
  --quiet
```

After this finishes, garden can run the same strict v110 orchestration.

## Current Risk

The v110 candidate is still based on the v108 normal-equation candidate family. Prior flowers test-sidecar evidence showed v108 was worse than v106 and v109 fell back to parent. Therefore the likely outcomes are:

- best case: held-out train-odd accepts a nonzero local mask and test improves over v106;
- safe case: v109 rejects the candidate and preserves v106;
- bad case: held-out train-odd accepts a candidate that hurts test, exposing a new reliability gap.

Only the first case is a true quality breakthrough. The second case is still useful evidence for reliability, but it is not enough for a paper-level final method.

## 2026-06-25 20:20 PDT Checkpoint

This checkpoint updates the initial running log without rewriting the earlier historical state.

### Subagent Review Outcome

Five parallel subagent tasks were completed:

| role | output | key result |
|---|---|---|
| repo/results mapping | read-only v110 artifact audit | v110 had no complete gate/eval result at audit time; flowers/garden/counter were still in field build; bonsai initially failed preflight due missing v106 parent train renders |
| method-gap analysis | code-level fairness review | v110 is strict for the new candidate/gate path, but the v106 parent is still a test-sidecar artifact, so v110 alone does not close end-to-end fairness |
| report aggregation | `scripts/car_model/collect_v110_strict_split_report.py` | added a reusable summary CLI; later patched to read clean MeshSplatting metrics from `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k` |
| mentor report draft | `docs/car_model/6-25-SPCarNet-v110-StrictSplit-Technical-Report-Draft.md` | written as a claim-boundary report: v106 is the quality line; v109/v110 are safety/fairness validation unless final reports prove otherwise |
| end-to-end strict runner | `scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py` | implemented v111 runner where the parent is also train-only (`train/all`), candidate is `train/even`, gate calibration is `train/odd`, and evaluation is `test` |

### Current Verified Inputs

All four representative train banks now exist and have train metadata:

| scene | train bank | split | frames/deltas |
|---|---|---:|---:|
| flowers | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/flowers/v102_preprojected_delta_bank_train.pt` | train | 151/151 |
| garden | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/garden/v102_preprojected_delta_bank_train.pt` | train | 161/161 |
| counter | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/counter/v102_preprojected_delta_bank_train.pt` | train | 210/210 |
| bonsai | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/bonsai/v102_preprojected_delta_bank_train.pt` | train | 255/255 |

All four v106 parent train render directories are now available, including the previously missing bonsai directory:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/train/ours_26000_v106_podmoe_basepreserve_<scene>/renders
```

### Correct Clean-Baseline Source

The clean MeshSplatting comparator should come from:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>/results.json
```

with key `ours_26000`. Do not use the v101 detached package result as "clean"; that is an endpoint package result and is a different method line.

Current four-scene clean vs v106 parent snapshot:

| scene | clean PSNR / SSIM / LPIPS | v106 PSNR / SSIM / LPIPS | v106 minus clean |
|---|---:|---:|---:|
| flowers | 19.682257 / 0.511822 / 0.394563 | 20.077723 / 0.531240 / 0.374393 | +0.395466 / +0.019418 / -0.020170 |
| garden | 25.029211 / 0.780035 / 0.201314 | 25.790945 / 0.799382 / 0.174480 | +0.761734 / +0.019347 / -0.026834 |
| counter | 26.751774 / 0.862055 / 0.252003 | 27.499645 / 0.867521 / 0.238847 | +0.747871 / +0.005466 / -0.013156 |
| bonsai | 28.895233 / 0.896400 / 0.259493 | 30.316090 / 0.907520 / 0.230050 | +1.420856 / +0.011120 / -0.029443 |

### Current v110 Runs

All four representative scenes are now launched with W&B offline logging:

| scene | GPU | current stage at checkpoint | log |
|---|---:|---|---|
| flowers | 2 | `build_train_even_candidate_field`, about `75/76` selected views | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/logs/01_build_candidate_field.log` |
| garden | 3 | `build_train_even_candidate_field`, about `42/81` selected views | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/garden/logs/01_build_candidate_field.log` |
| counter | 1 | `build_train_even_candidate_field`, about `11/105` selected views | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter/logs/01_build_candidate_field.log` |
| bonsai | 5 | `build_train_even_candidate_field`, about `1/128` selected views after preflight fix | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/bonsai/logs/01_build_candidate_field.log` |

The live summary file is:

```text
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/v110_strict_split_parent_gate_summary.md
```

At this checkpoint, v110 gated metrics are still `NA` because no scene has reached gate/eval completion yet.

### v111 End-to-End Strict Interface

Because method-gap review found that v106 parent itself is not end-to-end strict, a new runner was added:

```text
scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py
scripts/car_model/smoke_test_v111_runner_args.py
```

v111 protocol:

```text
train/all   -> build parent field
train+test  -> render parent
train/even  -> build candidate field
train+test  -> render candidate
train/odd   -> calibrate v109 parent gate
test        -> apply/evaluate gated output
```

Verification:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  scripts/car_model/smoke_test_v111_runner_args.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_v111_runner_args.py
```

Result: smoke passed. No real v111 GPU run has been launched yet because the four v110 field builds are still consuming GPUs and `/dev/shm` has limited headroom.

### Updated Claim Boundary

Safe:

- v106 beats the selected local clean MeshSplatting baseline on the four representative scenes above.
- v110 strict candidate/gate experiments are running for four representative scenes.
- v111 now exists as the proper end-to-end strict interface needed to address the parent fairness objection.

Unsafe:

- claiming v110 improves over v106 before a gate/eval report exists;
- claiming v110 closes end-to-end fairness, because its parent is still v106;
- claiming v111 works before a real GPU run completes.

## 2026-06-25 20:50 PDT Flowers Strict Result and Safety Patch

### Interface Bug Fixed

The first flowers v110 orchestration reached `gate_train_odd_to_test` but failed because negative float thresholds were passed as separate CLI tokens:

```text
--min_mean_ssim_gain -1e-06
```

`argparse` treated the negative exponent token as an option. The runners now pass negative-capable values with `--arg=value`:

```text
--frame_threshold_quantile=-1.0
--min_mean_ssim_gain=-1e-06
--min_mean_lpips_gain=-1000000000.0
--min_p05_score_gain=-0.0001
```

Patched files:

```text
scripts/car_model/run_v110_strict_split_parent_gate_scene.py
scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py
```

Checks passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_v110_strict_split_parent_gate_scene.py \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v110_strict_runner_args.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_v111_runner_args.py
```

### Default v110 Flowers Gate Is a Negative Result

Using the already-rendered flowers candidate, the corrected default gate was run manually:

```text
method: ours_26000_v110_strict_train_even_odd_parent_gate_flowers
report: /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v110_strict_train_even_odd_parent_gate_flowers/v109_render_realized_parent_gate_report.json
eval: /dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/reports/flowers_ours_26000_v110_strict_train_even_odd_parent_gate_flowers_test_results.json
wandb: /data/peilincai/mesh-splatting/wandb/offline-run-20260625_204153-5o7gfepm
```

The gate selected a nonzero policy from train-odd:

| item | value |
|---|---:|
| calibration candidate count | 151 |
| calibration selected count | 64 |
| calibration dPSNR | +0.382280 |
| calibration dSSIM | +0.025216 |
| target mean mask | 0.493082 |
| no target GT used for policy | true |

But held-out test regressed relative to the v106 parent:

| method | PSNR | SSIM | LPIPS | vs clean | vs v106 |
|---|---:|---:|---:|---:|---:|
| clean MeshSplatting | 19.682257 | 0.511822 | 0.394563 | - | - |
| v106 parent | 20.077723 | 0.531240 | 0.374393 | +0.395466 / +0.019418 / -0.020170 | - |
| default v110 gate | 19.966076 | 0.522843 | 0.380387 | +0.283819 / +0.011021 / -0.014176 | -0.111647 / -0.008397 / +0.005994 |

Conclusion: default v110 strict split is a false accept on flowers. It still beats clean MeshSplatting but is worse than the v106 parent, so it is not a valid promotion.

### v110b Gain-Margin Safety Fix

A fixed safety-margin ablation was tested without target/test GT policy selection:

```text
method: ours_26000_v110b_strict_gainmargin_parent_gate_flowers
change: --min_mean_psnr_gain 0.5
report: /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v110b_strict_gainmargin_parent_gate_flowers/v109_render_realized_parent_gate_report.json
eval: /dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/reports/flowers_ours_26000_v110b_strict_gainmargin_parent_gate_flowers_test_results.json
wandb: /data/peilincai/mesh-splatting/wandb/offline-run-20260625_204624-qymu25hs
```

The flowers train-odd candidate gain is below the new margin, so v110b falls back:

| item | value |
|---|---:|
| fallback to parent | 1 |
| target mean mask | 0.000000 |
| no target GT used for policy | true |

v110b test metrics equal v106:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v106 parent | 20.077723 | 0.531240 | 0.374393 |
| v110b gain-margin gate | 20.077723 | 0.531240 | 0.374393 |

The runner defaults were updated so future v110/v111 runs use `--min_mean_psnr_gain 0.5`. This is a safety correction, not a quality breakthrough. It restores parent preservation on flowers and prevents the observed false accept.

## 2026-06-25 21:55 PDT Garden v110b Manual Follow-Up

Garden's original v110 orchestration reached the gate step but failed because it was launched before the negative-argument CLI patch:

```text
error: argument --min_mean_ssim_gain: expected one argument
```

The already-built/rendered garden candidate was therefore reused with the fixed `--arg=value` syntax and the new default `--min_mean_psnr_gain 0.5`.

```text
method: ours_26000_v110b_strict_gainmargin_parent_gate_garden
report: /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/garden/detached_model/test/ours_26000_v110b_strict_gainmargin_parent_gate_garden/v109_render_realized_parent_gate_report.json
eval: /dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/garden/reports/garden_ours_26000_v110b_strict_gainmargin_parent_gate_garden_test_results.json
wandb: /data/peilincai/mesh-splatting/wandb/offline-run-20260625_215017-0on2skoq
```

The garden train-odd gate accepted a nonzero mask:

| item | value |
|---|---:|
| calibration candidate count | 161 |
| calibration selected count | 64 |
| calibration dPSNR | +0.799440 |
| calibration dSSIM | +0.038866 |
| fallback to parent | 0 |
| target mean mask | 0.699225 |
| no target GT used for policy | true |

Held-out test still regressed relative to the v106 parent:

| method | PSNR | SSIM | LPIPS | vs clean | vs v106 |
|---|---:|---:|---:|---:|---:|
| clean MeshSplatting | 25.029211 | 0.780035 | 0.201314 | - | - |
| v106 parent | 25.790945 | 0.799382 | 0.174480 | +0.761734 / +0.019347 / -0.026834 | - |
| v110b gain-margin gate | 25.430321 | 0.783703 | 0.186970 | +0.401110 / +0.003668 / -0.014345 | -0.360624 / -0.015679 / +0.012489 |

Conclusion: v110b is still not promotable. It fixes the flowers false accept by falling back to parent, but garden shows that train/odd calibration can still over-trust a candidate that does not generalize to held-out test. The next method step needs a stronger out-of-trajectory risk certificate rather than another scalar margin tweak.

Durable summary copied into the repo:

```text
docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md
docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.json
```

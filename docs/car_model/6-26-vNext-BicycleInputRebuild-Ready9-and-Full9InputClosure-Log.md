# 6-26 vNext Bicycle Input Rebuild, Ready9, and Full9 Input Closure

Date: 2026-06-26

This log records the `bicycle` rebuild that closes the missing-input side of the vNext structure-aware shrink full9 plan. It is an input/protocol milestone, not a full quality-closure claim.

## What Changed

The final missing vNext scene, `bicycle`, now has the full input chain required by the manifest runner:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/fit_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/carrier.json
```

The updated full9 preflight after this rebuild reports:

```text
ready_scene_count: 9
missing_input_scene_count: 0
failed_scene_count: 0
```

Committed lightweight artifacts:

```text
docs/car_model/vnext_artifacts/bicycle_structure_shrink_rebuild_tau002_20260626_1055/
```

## Input Evidence

The train visible-bary base used 46 `images_4` train views. The fit evidence was built from the Phase-G alpha-0.875 teacher renders with train-only parent/teacher residual selection:

| field | value |
|---|---:|
| processed train views | `46` |
| skipped train views | `0` |
| mean active fraction | `0.1958138896` |
| mean target L1 | `0.0075708739` |
| mean raw parent delta L1 | `0.0165493823` |
| mean positive teacher gain L1 | `0.0071865642` |
| top support rows | `8192` |
| nonzero support faces | `1457527` |

The render-visible carrier was then pruned with policy-val evidence only:

| field | value |
|---|---:|
| input carriers | `64` |
| output carriers | `60` |
| candidate faces | `3080` |
| retained faces | `1256` |
| removed faces | `1799` |
| prune alpha | `0.015625` |
| greedy removals | `0` |

The target evidence uses the `images_4` held-out test split with 25 views and is stripped of GT/residual keys before apply.

## Strict Run

Run root:

```text
/dev/shm/peilincai_spcarnet_vnext_structure_shrink_bicycle_strict_20260626_1055/bicycle
```

W&B offline run:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_bicycle_strict_20260626_1055/wandb/offline-run-20260626_110151-ff3szoau
```

Fixed policy:

```text
method_name=ours_26000_vnext_structure_aware_shrink
strict_no_target_gt_apply=true
texture_size=16
texture_size_candidates=16
support_expansion_mode=none
atlas_empty_bin_fill_mode=face_mean
surface_multiscale_prior_mode=local_patch
surface_multiscale_prior_blend_candidates=0.5
max_abs_delta_rgb_candidates=0.12
enable_policy_val_structure_aware_shrink=true
structure_shrink_l1_weight=1.0
structure_shrink_gradient_weight=1.0
structure_shrink_edge_weight=0.0
structure_shrink_risk_tau=0.002
structure_shrink_max_penalty=1.0
```

Protocol audit:

| field | value |
|---|---:|
| protocol audit passed | `true` |
| selection uses test GT | `false` |
| target GT visible to apply | `false` |
| target GT visible to selection | `false` |
| target forbidden keys stripped | `true` |
| target apply leak | `false` |

Adapter result:

| field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.015625` |
| written target views | `25` |
| changed pixels | `4421` |
| total pixels | `25420350` |
| changed fraction | `0.0001739158` |

## Same-Evidence Parent Comparison

The same-evidence parent baseline was exported from the exact rebuilt target evidence and evaluated in the same model directory.

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| same-evidence parent | `23.293506622` | `0.659650564` | `0.332269400` |
| vNext structure-aware shrink | `23.293516159` | `0.659650743` | `0.332269371` |
| delta, better direction | `+0.000009537` | `+0.000000179` | `-0.000000030` |

Per-view strict wins over the same-evidence parent are small:

| metric | wins / views |
|---|---:|
| PSNR | `4 / 25` |
| SSIM | `5 / 25` |
| LPIPS | `2 / 25` |

Interpretation: the strict certificate accepts a real nonzero edit, and the aggregate metrics move in the correct direction, but the effect is near the evaluation-noise floor. This should be presented as full9 input/protocol closure plus a tiny positive bicycle row, not as a visual breakthrough.

## Commands

The final same-evidence export and evaluation commands were:

```bash
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
RUN=/dev/shm/peilincai_spcarnet_vnext_structure_shrink_bicycle_strict_20260626_1055/bicycle/model
EVID=/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/bicycle/target_evidence
PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/car_model/ecsr_export_evidence_rgb_baseline.py \
  --evidence_dir "$EVID" \
  --output_model "$RUN" \
  --split test \
  --method_name bicycle_same_evidence_parent \
  --force

CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/car_model/evaluate_render_split_metrics.py \
  --model_path "$RUN" \
  --split test \
  --methods bicycle_same_evidence_parent ours_26000_vnext_structure_aware_shrink \
  --output /dev/shm/peilincai_spcarnet_vnext_structure_shrink_bicycle_strict_20260626_1055/bicycle/reports/bicycle_same_evidence_parent_vs_vnext_test_results.json \
  --per_view_output /dev/shm/peilincai_spcarnet_vnext_structure_shrink_bicycle_strict_20260626_1055/bicycle/reports/bicycle_same_evidence_parent_vs_vnext_test_per_view.json \
  --merge_model_results
```

## Claim Boundary

Safe claim:

```text
vNext full9 input readiness is now closed at 9/9, and the last missing scene has a strict no-target-GT apply run with protocol pass, accepted nonzero output, and same-evidence aggregate micro-gains.
```

Unsafe claim:

```text
Do not claim that vNext full9 quality evaluation is complete.
Do not claim that bicycle provides visible qualitative improvement.
Do not claim that vNext beats v106, Phase-J, or clean MeshSplatting on the current full9 table.
```

## Next Required Work

The next step is a fixed-policy full9 manifest run using the now-ready 9-scene input manifest. Because `/data` is effectively full and `/dev/shm` was at 99% during this rebuild, the full9 run should be launched only after freeing space or with a runner mode that deletes each scene's bulky intermediate outputs after compact reports are copied.

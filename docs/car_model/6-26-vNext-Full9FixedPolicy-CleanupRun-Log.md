# vNext Full9 Fixed-Policy Cleanup Run Log

Date: 2026-06-26

This log records the first strict full9 fixed-policy cleanup run for `vNext_certified_residual_surface_texture`. It should be read as a protocol/evidence milestone and bottleneck diagnosis, not as a promoted paper-quality result.

## Verdict

The vNext route in `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md` is reasonable and worth keeping: it tries to convert the strong Phase-J render-time residual repair into an auditable, MeshSplatting-compatible residual surface texture. The current implementation is real and end-to-end: it uses train/policy-val certificates, writes parent-preserving accepted or fallback outputs, hides target GT during selection/apply, runs all 9 scenes, and saves machine-readable audits.

The quality result is not yet good enough. The full9 fixed-policy run is below both local clean MeshSplatting and v106, so it must not be promoted as the final paper method.

## Run Command

```bash
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
RUN=/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_run_20260626_1200
COMPACT=/dev/shm/peilincai_spcarnet_vnext_full9_cleanup_compact_20260626_1200
WANDB=/dev/shm/peilincai_wandb_vnext_full9_cleanup_20260626_1200
rm -rf "$RUN" "$COMPACT" "$WANDB"
mkdir -p "$WANDB"
WANDB_DIR="$WANDB" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/car_model/run_vnext_certified_residual_texture_manifest.py \
  --scene_config_json docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json \
  --output_root "$RUN" \
  --method_name ours_26000_vnext_structure_aware_shrink \
  --max_parallel 1 \
  --continue_on_error \
  --skip_existing_complete \
  --wandb \
  --wandb_mode offline \
  --wandb_dir "$WANDB" \
  --wandb_group vnext_structure_shrink_full9_cleanup_20260626 \
  --wandb_name_prefix vnext-full9-cleanup- \
  --compact_artifact_root "$COMPACT" \
  --cleanup_scene_outputs \
  --skip_teacher_cache \
  --strict_no_target_gt_apply \
  --texture_size 16 \
  --texture_size_candidates 16 \
  --support_expansion_mode none \
  --atlas_empty_bin_fill_mode face_mean \
  --surface_multiscale_prior_mode local_patch \
  --surface_multiscale_prior_blend_candidates 0.5 \
  --max_abs_delta_rgb_candidates 0.12 \
  --no_policy_val_bin_uncertainty_guard \
  --enable_policy_val_structure_aware_shrink \
  --structure_shrink_l1_weight 1.0 \
  --structure_shrink_gradient_weight 1.0 \
  --structure_shrink_edge_weight 0.0 \
  --structure_shrink_risk_tau 0.002 \
  --structure_shrink_max_penalty 1.0
```

W&B was enabled in offline mode under:

```text
/dev/shm/peilincai_wandb_vnext_full9_cleanup_20260626_1200/wandb
```

The run used `max_parallel=1` and `--cleanup_scene_outputs` to avoid keeping bulky per-scene render/model trees. It ran on available lower/mid-pressure GPUs during execution.

## Artifact Package

Promoted lightweight package:

```text
docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/
```

Main files:

```text
docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md
docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.json
docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/vnext_full9_cleanup_promotion_manifest.md
docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/vnext_full9_cleanup_promotion_manifest.json
```

The promotion manifest records `205` copied lightweight files, `0` skipped files, 9 scene artifact folders, per-scene reports, model audits, selector summaries, eval logs, and root manifest logs. Large checkpoint/render/model trees are intentionally not committed.

The package is not a qualitative gallery. Because the run used `--cleanup_scene_outputs`, the promoted roots contain no PNG/JPG/MP4 render payloads; qualitative comparison still needs a separate render-panel export.

## Summary Metrics

| field | value |
|---|---:|
| scenes found | 9 |
| completed metric scenes | 9 |
| failed scenes | 0 |
| missing-input scenes | 0 |
| protocol audit passed | 9 / 9 |
| accepted nonzero scenes | 6 / 9 |
| fallback/no-op scenes | 3 / 9 |
| mean changed fraction | 0.002756271 |
| mean PSNR | 25.067699 |
| mean SSIM | 0.741260 |
| mean LPIPS | 0.306689 |

Comparison to current repo baselines:

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| clean MeshSplatting | 25.151682 | 0.749018 | 0.287621 | local clean baseline |
| v106 POD-MoE base-preserve | 25.831280 | 0.760830 | 0.268435 | current verified representation-quality line |
| vNext full9 fixed-policy cleanup | 25.067699 | 0.741260 | 0.306689 | protocol closure, not promoted |

The vNext full9 run is lower than clean by about `-0.083983` PSNR, `-0.007758` SSIM, and `+0.019068` LPIPS. It is lower than v106 by about `-0.763581` PSNR, `-0.019570` SSIM, and `+0.038254` LPIPS.

## Per-Scene Table

| scene | accepted | policy | alpha | changed fraction | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|
| bicycle | true | accepted_atlas | 0.015625 | 0.000173916 | 23.293516 | 0.659651 | 0.332269 |
| bonsai | true | accepted_atlas | 0.250000 | 0.001489739 | 28.865479 | 0.896003 | 0.259323 |
| counter | true | accepted_atlas | 0.125000 | 0.012343567 | 26.751171 | 0.862042 | 0.251955 |
| flowers | false | fallback_noop | 0.000000 | 0.000000000 | 19.519194 | 0.490780 | 0.424170 |
| garden | true | accepted_atlas | 0.125000 | 0.002050379 | 24.741142 | 0.754052 | 0.248015 |
| kitchen | true | accepted_atlas | 0.125000 | 0.003549714 | 27.817173 | 0.876445 | 0.199172 |
| room | true | accepted_atlas | 0.062500 | 0.005199120 | 28.739571 | 0.884797 | 0.249909 |
| stump | false | fallback_noop | 0.000000 | 0.000000000 | 25.043329 | 0.689480 | 0.349850 |
| treehill | false | fallback_noop | 0.000000 | 0.000000000 | 20.838715 | 0.558089 | 0.445541 |

Fallback/rejection reasons:

- `flowers`: failed lower-tail, min-view, SSIM, and image-L1 gates; `cvar20_view_relative_gain=-0.224441`, `min_view_relative_gain=-0.278408`, `ssim_gain=-0.000082279`.
- `stump`: failed lower-tail and min-view gates; `cvar20_view_relative_gain=-0.172454`, `min_view_relative_gain=-0.344907`.
- `treehill`: failed lower-tail, min-view, SSIM, and L1 gates; `cvar20_view_relative_gain=-0.053640`, `min_view_relative_gain=-0.077837`, `ssim_gain=-0.000009413`.

## What Is Satisfied

- Full9 fixed-policy execution is complete.
- `9 / 9` scenes pass protocol audit.
- Target/test GT is not used for branch, alpha, texture capacity, fallback, or threshold selection.
- Accepted outputs are parent-preserving residual surface atlas edits.
- Rejected outputs are explicit no-op/fallback with machine-readable reasons.
- Commands, configs, logs, per-scene metrics, per-view metrics, selector outputs, and target no-GT audits are saved in the artifact package.

## What Is Not Satisfied

- It does not beat clean MeshSplatting or v106.
- It does not meet the prompt success target of `6 / 9` strict scene RGB wins vs clean with at least `+0.5` mean PSNR.
- It does not capture a meaningful fraction of the Phase-J gain.
- It does not yet include the full required ablation table: no-certificate, no-structure, adaptive capacity variants, fresh teacher cache, Phase-J teacher, v104c, v106, and clean in one same-protocol table.
- It does not yet include a strong qualitative gallery where vNext improvements are clearly visible to a human observer; the cleanup artifact package has no PNG/JPG/MP4 render payloads.
- Texture capacity remains conservative and mostly fixed (`texture_size=16`, `support_expansion_mode=none`), so the run is closer to a certified surface-atlas proof-of-life than a high-capacity representation upgrade.

## Paper Story Boundary

Safe claim:

> vNext demonstrates that certified residual surface texturing can be run end-to-end on full9 with strict train-only policy selection, no target-GT leakage during apply, explicit fallback, and machine-readable audits.

Unsafe claim:

> vNext is the new promoted endpoint or a paper-grade improvement over MeshSplatting.

The current best paper-facing positioning remains:

- Phase-J: strongest broad RGB endpoint and teacher/upper bound.
- v106: current verified representation-quality line over local clean MeshSplatting.
- vNext full9 cleanup: strict representation-level protocol scaffold plus bottleneck evidence.

## Next Required Work

The next improvement should not be another scalar shrink scan. It should target representation capacity and teacher distillation:

1. Build a fresh hashed residual teacher cache that includes Phase-J/ELA teacher residuals where available.
2. Replace fixed `texture_size=16` with adaptive per-region capacity selected only by train/policy-val evidence.
3. Add no-certificate, no-structure, no-adaptive-capacity, and Phase-J-teacher ablations in the same manifest runner.
4. Generate changed-region qualitative panels: GT, clean MeshSplatting, Phase-J teacher, v106, vNext, amplified residual, and error maps.
5. Report triangle count, residual texture storage, parameter count, render overhead, fallback rate, and changed-pixel fraction.
6. Promote only if the same fixed policy beats clean/v106 under the same evaluator.

Final status: `NOT COMPLETE`.

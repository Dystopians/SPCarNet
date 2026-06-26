# SPCarNet vNext Implementation Log

Date: 2026-06-26

This log records the first implementation milestone for `vNext_certified_residual_surface_texture`, based on `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md`.

## Decision

The direction is reasonable and worth pursuing, but it is not a near-term guaranteed paper endpoint. The correct route is to reuse the existing surface evidence, teacher evidence, region texture, and policy-val certificate infrastructure, then make the protocol stricter and more explicit.

The route is realistic as a staged representation project:

```text
Phase-J render-time teacher
  -> train-only teacher surface evidence cache
  -> face/UV residual texture atlas
  -> train-policy-val capacity/alpha/certificate
  -> exact parent fallback on rejection
  -> test-only final evaluation
```

It is not realistic to promise immediate capture of `>=70%` of the Phase-J gain without pilot evidence. Prior surface-atlas results were safe but small, and v110/v113 showed that train-only gates can still miss held-out failures.

## Implemented Milestone

New files:

- `scripts/car_model/ecsr_vnext_protocol.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/run_vnext_certified_residual_texture_full9.py`
- `scripts/car_model/assemble_vnext_certified_residual_texture_report.py`
- `scripts/car_model/smoke_test_vnext_no_test_gt_certificate_schema.py`
- `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md`

The implementation does not rewrite the low-level renderer. It adds a vNext orchestration and provenance layer around existing components:

- `ecsr_build_teacher_surface_evidence_cache.py` for train-only teacher residual cache augmentation;
- `ecsr_apply_surface_residual_region_texture_adapter.py` for face/UV texture fitting, adaptive capacity candidates, policy-val guards, target-footprint checks, and no-op fallback;
- `evaluate_render_split_metrics.py` for final target split evaluation;
- `assemble_vnext_certified_residual_texture_report.py` for machine-readable multi-scene summaries.

## Protocol Properties

The vNext scene runner records:

- source paths and existence/hash records;
- generated commands;
- train-only fit/policy-val split provenance;
- explicit `selection_uses_test_gt=false`;
- capacity selected by train-policy-val plus GT-free target footprint;
- thresholds selected by train-policy-val only;
- exact no-op fallback support through the underlying region texture adapter.

The full9 runner uses `{scene}` path templates so it can run either existing historical caches or newly rebuilt clean vNext caches without hard-coding one prior experiment root.

## Verified Checks

Static and smoke checks completed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_vnext_protocol.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  scripts/car_model/run_vnext_certified_residual_texture_full9.py \
  scripts/car_model/assemble_vnext_certified_residual_texture_report.py \
  scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_vnext_no_test_gt_certificate_schema.py
```

Dry-run protocol check completed with synthetic placeholder paths:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene synthetic \
  --source_model /tmp/nonexistent_model \
  --fit_evidence_dir /tmp/nonexistent_fit_evidence \
  --target_evidence_dir /tmp/nonexistent_target_evidence \
  --region_carrier_json /tmp/nonexistent_region.json \
  --teacher_render_dir /tmp/nonexistent_teacher \
  --output_root /tmp/spcarnet_vnext_dryrun \
  --dry_run \
  --skip_eval
```

The dry-run wrote a manifest/report with protocol audit passing and `selection_uses_test_gt=false`.

W&B offline dry-run also completed:

```bash
WANDB_DIR=/tmp/spcarnet_vnext_wandb_dryrun WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene synthetic_wandb \
  --source_model /tmp/nonexistent_model \
  --fit_evidence_dir /tmp/nonexistent_fit_evidence \
  --target_evidence_dir /tmp/nonexistent_target_evidence \
  --region_carrier_json /tmp/nonexistent_region.json \
  --teacher_render_dir /tmp/nonexistent_teacher \
  --output_root /tmp/spcarnet_vnext_wandb_dryrun \
  --dry_run \
  --skip_eval \
  --wandb \
  --wandb_mode offline \
  --wandb_group vnext_dryrun
```

This verified that the runner records W&B status scalars and artifact paths without requiring online sync during resource-constrained runs.

Two-scene full9-wrapper dry-run completed and the assembler summary reported both rows as `DRY_RUN`:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_full9.py \
  --scenes flowers,garden \
  --source_model_template '/tmp/{scene}/model' \
  --fit_evidence_template '/tmp/{scene}/fit_evidence' \
  --target_evidence_template '/tmp/{scene}/target_evidence' \
  --region_carrier_template '/tmp/{scene}/carriers.json' \
  --teacher_render_template '/tmp/{scene}/teacher_renders' \
  --output_root /tmp/spcarnet_vnext_full9_dryrun \
  --dry_run \
  --skip_eval \
  --continue_on_error
```

## Current Resource Blocker

Full real experiments should not be launched from the repo/output tree right now:

- `/data` is effectively full, with only about `420M` free at the latest check;
- `/dev/shm` has about `36G` free while several long jobs are still running;
- active heavy jobs occupy GPUs `2`, `3`, and `5`.

The safe next real run is a one-scene pilot on GPU `1` or `4`, with outputs and W&B cache under `/dev/shm`, after ensuring enough temporary space.

## Recommended Next Command

Once the exact scene cache paths are chosen and space is available:

```bash
TAG=$(date +%Y%m%d_%H%M%S)
OUT=/dev/shm/peilincai_spcarnet_vnext_certified_residual_texture_${TAG}
export WANDB_MODE=offline
export WANDB_DIR=/dev/shm/peilincai_wandb_vnext_${TAG}

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene garden \
  --gpu 4 \
  --source_model <scene_model_path> \
  --fit_evidence_dir <train_surface_evidence_dir> \
  --target_evidence_dir <test_surface_evidence_dir> \
  --region_carrier_json <region_carrier_json> \
  --teacher_render_dir <phasej_train_teacher_render_dir> \
  --output_root "$OUT" \
  --wandb \
  --wandb_mode "$WANDB_MODE"
```

## Status

`NOT COMPLETE`.

The first protocol/interface milestone is implemented and verified. Real single-scene and full9 metrics remain blocked by resource pressure and by the need to choose or rebuild clean vNext evidence caches.

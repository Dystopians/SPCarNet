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
- explicit target GT visibility fields for selection/apply/eval;
- capacity selected by train-policy-val plus GT-free target footprint;
- thresholds selected by train-policy-val only;
- exact no-op fallback support through the underlying region texture adapter.

### Strict No-Target-GT Apply Upgrade

Subagent review identified that the first garden artifacts proved `selection_uses_test_gt=false`, but target evidence still contained `rgb_gt` during the adapter apply/fallback step. That is weaker than the final vNext prompt requirement, because a reviewer could argue that target GT was visible to non-evaluation code even if it was not used for branch selection.

The runner now supports a stricter path:

```bash
--strict_no_target_gt_apply
```

When enabled, the scene runner inserts two explicit stages around the adapter:

1. `ecsr_strip_target_evidence_for_vnext.py` writes `target_evidence_no_gt/views/*.npz` containing only geometry, parent render, camera, depth/normal, alpha, barycentric, face id, and texture keys. It strips `rgb_gt`, `residual_rgb`, `residual_l1`, and all `teacher_*` keys before the adapter sees target evidence.
2. `ecsr_populate_eval_gt_from_target_evidence.py` copies `rgb_gt` images only after target apply, immediately before `evaluate_render_split_metrics.py`.

The protocol audit now records:

```text
target_gt_visible_to_selection
target_gt_visible_to_apply
target_gt_visible_to_eval
target_forbidden_keys_stripped
target_apply_leak
```

For strict runs on `target_split=test`, `target_gt_visible_to_apply=false`, `target_gt_visible_to_eval=true`, and `target_apply_leak=false` are required for protocol audit pass. Non-strict runs remain executable for legacy comparison, but their audit marks target apply GT visibility explicitly.

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

Strict no-target-GT smoke checks completed:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_vnext_protocol.py \
  scripts/car_model/ecsr_strip_target_evidence_for_vnext.py \
  scripts/car_model/ecsr_populate_eval_gt_from_target_evidence.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

PYTHONDONTWRITEBYTECODE=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_strip_target_evidence_for_vnext.py \
  --target_evidence_dir /dev/shm/peilincai_vnext_strip_smoke_counter/in \
  --out_dir /dev/shm/peilincai_vnext_strip_smoke_counter/out \
  --force

PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene counter \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/counter_teacher_surface_evidence_phasej_trainval_alpha1 \
  --target_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/counter \
  --region_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_teacher_render_visible_region_carriers_phasej_trainval_alpha1_policyval_pruned.json \
  --output_root /dev/shm/peilincai_vnext_strict_dryrun_counter \
  --skip_teacher_cache \
  --texture_size_candidates 16 \
  --support_expansion_mode none \
  --atlas_empty_bin_fill_mode face_mean \
  --surface_multiscale_prior_blend_candidates 0.5 \
  --max_abs_delta_rgb_candidates 0.12 \
  --no_policy_val_bin_uncertainty_guard \
  --strict_no_target_gt_apply \
  --dry_run
```

The one-view strip smoke kept only:

```text
alpha, barycentric, barycentric_valid, camera_center, depth, face_id, normal, rgb_render, texture
```

and confirmed no forbidden target keys remained. The strict dry-run command chain was:

```text
strip_target_evidence_no_gt -> apply_certified_residual_texture -> populate_eval_gt_from_target_evidence -> evaluate_vnext_target
```

with protocol audit:

```text
passed=true
target_gt_visible_to_selection=false
target_gt_visible_to_apply=false
target_gt_visible_to_eval=true
target_forbidden_keys_stripped=true
target_apply_leak=false
```

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

## Real Garden Pilot Results

After the initial protocol milestone, three real `garden` pilots were run under W&B offline with outputs in `/dev/shm` and lightweight artifacts copied into the repo.

### Initial Full Candidate Pilot

Artifact root:

```text
docs/car_model/vnext_artifacts/garden_20260626_004134/
```

Result:

- status: `COMPLETE`
- protocol audit passed: `True`
- selection uses test GT: `False`
- accepted: `False`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed fraction: `0.0`
- test PSNR / SSIM / LPIPS: `24.741003 / 0.754049 / 0.248023`

This was a safety/fallback proof, not a quality improvement.

### Hard-Bin Soft-Shrink Diagnostic

Artifact root:

```text
docs/car_model/vnext_artifacts/garden_hardbin_softshrink_20260626_035631/
```

Result:

- status: `COMPLETE`
- protocol audit passed: `True`
- soft bin uncertainty shrink enabled: `keep_with_downweight`
- hard bin uncertainty guard enabled: `True`
- accepted: `False`
- effective policy: `fallback_noop`
- test PSNR / SSIM / LPIPS: `24.741003 / 0.754049 / 0.248023`

Diagnosis: alpha refinement and soft shrink made the image-level SSIM/L1 direction positive, but the hard bin guard still rejected the candidate because lower-tail relative gains stayed negative after the hard allowlist.

### Face-SoftShrink Accepted Milestone

Artifact root:

```text
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/
```

Result:

- status: `COMPLETE`
- protocol audit passed: `True`
- selection uses test GT: `False`
- hard bin uncertainty guard: disabled
- soft bin uncertainty shrink: `keep_with_downweight`
- face guard decision: `keep_face_gain_guard`
- accepted: `True`
- effective policy: `accepted_atlas`
- selected alpha: `0.0625`
- target changed pixels: `82767`
- target changed fraction: `0.002080`
- test PSNR / SSIM / LPIPS: `24.741079 / 0.754051 / 0.248020`
- delta vs no-op/fallback parent: `+0.000076` PSNR / `+0.00000197` SSIM / `-0.00000323` LPIPS
- per-view wins vs no-op/fallback: PSNR `22/24`, SSIM `24/24`, LPIPS `22/24`

This is the first real nonzero vNext residual surface texture milestone. The gain is positive but extremely small, so it should not be promoted as paper-level closure.

W&B offline roots:

```text
/dev/shm/peilincai_wandb_vnext_softshrink_garden_20260626_035631/wandb/offline-run-20260626_040400-2ha0iu2v
/dev/shm/peilincai_wandb_vnext_face_softshrink_garden_20260626_040558/wandb/offline-run-20260626_041227-nilps441
```

Qualitative panel:

```text
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png
```

### Counter Strict No-Target-GT Apply Milestone

Artifact root:

```text
docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/
```

This run repeated the frozen face-softshrink policy on `counter` with the strict no-target-GT apply path enabled:

```text
strip_target_evidence_no_gt -> apply_certified_residual_texture -> populate_eval_gt_from_target_evidence -> evaluate_vnext_target
```

Key protocol fields:

- status: `COMPLETE`
- protocol audit passed: `True`
- selection uses test GT: `False`
- target GT visible to apply: `False`
- target GT visible to eval: `True`
- target forbidden keys stripped: `True`
- target apply leak: `False`

Result:

- accepted: `True`
- effective policy: `accepted_atlas`
- selected alpha: `0.25`
- target changed pixels: `571207`
- target changed fraction: `0.01177355`
- policy-val relative gain: `0.04431575`
- policy-val SSIM gain: `0.00010365`
- test PSNR / SSIM / LPIPS: `26.752003 / 0.862004 / 0.251912`

Delta versus the Phase-F compact parent:

```text
+0.002131 PSNR / -0.000047 SSIM / -0.000085 LPIPS
```

This is the first strict no-target-GT single-scene nonzero evidence because the adapter sees stripped target evidence. It is still not a paper-level quality result: SSIM is slightly worse than the Phase-F compact parent, and no full9 strict table exists yet.

### Strict Frozen-Policy Multiscene Follow-up

Artifact roots:

```text
docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/
docs/car_model/vnext_artifacts/bonsai_strict_face_softshrink_20260626_052500/
docs/car_model/vnext_artifacts/room_strict_face_softshrink_20260626_052500/
docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/
```

The same frozen face-softshrink policy was then run on `counter,bonsai,room` under strict no-target-GT apply:

```text
texture_size_candidates=16
support_expansion_mode=none
atlas_empty_bin_fill_mode=face_mean
surface_multiscale_prior_blend_candidates=0.5
max_abs_delta_rgb_candidates=0.12
policy_val_bin_uncertainty_guard=disabled
strict_no_target_gt_apply=true
```

Aggregate result:

| scene | protocol pass | target GT visible to apply | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `True` | `False` | `True` | `0.25` | `0.01177355` | `+0.002131` | `-0.000047` | `-0.000085` |
| bonsai | `True` | `False` | `True` | `0.25` | `0.00151333` | `+0.001225` | `-0.000010` | `-0.000018` |
| room | `True` | `False` | `False` | `0.0` | `0.00000000` | `-0.000097` | `-0.000003` | `-0.000007` |
| mean | 3/3 | 0/3 visible | 2/3 nonzero | - | - | `+0.001086` | `-0.000020` | `-0.000037` |

This follow-up is the current strongest vNext protocol package. It proves that the no-target-GT apply path and frozen policy can run across multiple scenes, but it also shows the current method bottleneck: SSIM/structure quality regresses on all three strict scenes, and `room` requires fallback/no-op with `changed_fraction=0`; its tiny metric deltas are parent-level eval noise, not residual gain.

Execution lesson:

- do not rely on same-command temporary shell variables in arguments like `OUT=... command --output_root "$OUT"`; zsh expands `$OUT` before that temporary assignment is visible to the command;
- use explicit output roots or export variables first;
- keep real outputs in `/dev/shm` and copy only compact reports/artifacts into the repo.

## Current Resource Blocker

Full real experiments should still not be launched from the repo/output tree:

- `/data` is effectively full, with only about `420M` free at the latest check;
- `/dev/shm` is usable for focused pilots but not safe for unbounded full9 artifact trees;

Strict runs are more storage-heavy than legacy runs because they materialize stripped target evidence in `/dev/shm`. Use them for the final fair protocol and delete transient `/dev/shm` run trees after copying only manifests, audits, compact metrics, and qualitative panels into the repo.
- full9 should either clean temporary outputs first or use a larger external artifact location.

The safe next method step is not another identical frozen-policy pilot. It is an SSIM/structure-aware residual shrink or local-alpha mechanism, followed by the same strict no-target-GT multiscene validation.

## Recommended Next Command

Use this only as a template after implementing the next SSIM/structure-aware variant and verifying paths:

```bash
TAG=$(date +%Y%m%d_%H%M%S)
OUT=/dev/shm/peilincai_spcarnet_vnext_face_softshrink_<scene>_${TAG}
export WANDB_MODE=offline
export WANDB_DIR=/dev/shm/peilincai_wandb_vnext_face_softshrink_<scene>_${TAG}

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene <scene> \
  --gpu <low_or_mid_load_gpu> \
  --source_model <scene_model_path> \
  --fit_evidence_dir <train_teacher_surface_evidence_dir> \
  --target_evidence_dir <test_surface_evidence_dir> \
  --region_carrier_json <region_carrier_json> \
  --output_root "$OUT" \
  --skip_teacher_cache \
  --texture_size_candidates 16 \
  --support_expansion_mode none \
  --atlas_empty_bin_fill_mode face_mean \
  --surface_multiscale_prior_blend_candidates 0.5 \
  --max_abs_delta_rgb_candidates 0.12 \
  --no_policy_val_bin_uncertainty_guard \
  --strict_no_target_gt_apply \
  --wandb \
  --wandb_mode "$WANDB_MODE"
```

## Status

`NOT COMPLETE`.

The protocol/interface milestone is implemented and verified. Nonzero residual surface textures have been accepted on garden/counter/bonsai, and strict no-target-GT apply has been verified on counter/bonsai/room with fallback/no-op on room. Full9, v106/clean-baseline comparison, ablations, SSIM/structure repair, and materially visible/large gains remain unfinished.

# v168 Phase-J Distillation Profile Protocol Log

Date: 2026-06-28

## Status

`v168` is an engineering/protocol milestone, not a metric win yet.

It adds an explicit runner-level profile for the next intended research direction:

```text
--distillation_profile teacher_to_reparented_parent
```

The purpose is to make Phase-J-to-baked-representation experiments harder to misconfigure. The profile treats Phase-J train renders as the teacher and a split-matched parent render as the residual parent, then keeps target/test RGB GT out of apply.

## Why This Was Needed

The previous v165-v167 route showed that footprint expansion, multisample residual fill, and simple face-local affine/patch residual fields do not convert target-visible support into SSIM/LPIPS gains. The next credible direction is therefore Phase-J-distilled baked representation.

However, the existing runner allowed a dangerous ambiguity:

- if `teacher_render_dir` and parent render point to the same Phase-J output, teacher residual becomes near-zero;
- if train parent renders are reused for target/test evidence, test-frame reparenting is wrong or fails;
- if strict no-target-GT apply is omitted, target-footprint experiments can become unfair.

v168 makes these constraints explicit in the runner.

## Code Change

Edited:

- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Added:

- `_resolved_same_path(...)`
- `_apply_distillation_profile(...)`
- parser flag `--distillation_profile {none,teacher_to_reparented_parent}`

The profile:

- requires `--teacher_render_dir`;
- requires `--parent_render_dir` or `--reparent_fit_parent_render_dir`;
- rejects `teacher_render_dir == parent_render_dir`;
- requires explicit `--reparent_target_parent_render_dir` for non-train target splits;
- auto-fills missing fit reparent args from the parent render;
- forces `strict_no_target_gt_apply=True`;
- writes `_distillation_profile_audit` into the manifest settings.

## Verified Dry Run

Static checks:

```bash
PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_distill_profile \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py

git diff --check -- scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Dry-run root:

```text
/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers
```

Dry-run manifest:

```text
/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json
```

Dry-run status:

```text
DRY_RUN
```

The generated command chain correctly separates:

- fit/train parent:
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/renders`
- target/test parent:
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/test/ours_26000_phasef_extra_compact_base/renders`
- Phase-J train teacher:
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasej_trainval_gate/renders`

The manifest settings include:

```json
{
  "distillation_profile": "teacher_to_reparented_parent",
  "_distillation_profile_audit": {
    "enabled": true,
    "target_parent_defaulted_from_fit_parent": false,
    "target_or_test_gt_visible_to_apply": false,
    "zero_residual_guard": "teacher_render_dir must differ from parent_render_dir"
  }
}
```

Negative parser check:

```text
rc=1
--distillation_profile teacher_to_reparented_parent requires --reparent_target_parent_render_dir for non-train target splits; the fit parent render directory usually contains train frames only
```

## Claim Readiness Auto Report

The current conservative claim-readiness report was generated with:

```bash
PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_claim \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_spcarnet_claim_readiness_report.py \
  --output docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md
```

Output:

```text
docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md
```

Current machine-readable verdict from that report:

- Phase-J local RGB endpoint: `PASS_LOCAL`
- v106 baked representation over selected clean: `PARTIAL_PASS`
- v166 flowers all-axis vs Phase-J: `FAIL`
- v167 flowers all-axis vs Phase-J: `FAIL`
- v168 exact metric win: `NOT_RUN`
- vNext/new prompt paper-main method: `FAIL`

## Current Blocker

The first exact validation attempt failed before metrics because storage/quota was unsafe:

- `/data`: about `9.7M` free
- `/dev/shm`: about `6.4G` free
- `/tmp`: filesystem space existed, but the user quota on `/dev/nvme0n1p4` was exceeded

Failure mode:

```text
reparent_fit_evidence returncode=1
shutil.copytree(...)
OSError / shutil.Error: [Errno 122] Disk quota exceeded
```

The failed partial root was:

```text
/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers
```

It was documented in `feedback.md` and then removed to recover quota because it was not a valid completed experiment.

## Low-Copy Direct-Teacher Unblock

Implemented after the storage failure:

- `scripts/car_model/ecsr_reparent_surface_evidence_cache.py`
  - added `--copy_mode {copy,hardlink,symlink,auto_link}`;
  - linked unchanged cache files and atomically rewrote changed NPZ/text outputs.
- `scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py`
  - added `--copy_mode`;
  - added `--rewrite_rgb_render_to_parent`;
  - can now fuse fit-evidence reparenting into teacher-cache construction by writing parent `rgb_render` and parent residual fields directly into the teacher cache.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
  - added `--reparent_copy_mode`;
  - added `--teacher_cache_copy_mode`;
  - added `--teacher_cache_rewrite_rgb_render_to_parent`;
  - added `--skip_reparent_fit_evidence_for_teacher_cache`.

Validation completed:

```bash
PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_lowcopy2 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_reparent_surface_evidence_cache.py \
  scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py

git diff --check -- \
  scripts/car_model/ecsr_reparent_surface_evidence_cache.py \
  scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Additional checks:

- reparent low-copy smoke with `--copy_mode auto_link --max_views 1 --allow_resize`: passed.
- teacher-cache low-copy smoke with `--copy_mode auto_link --max_views 1 --allow_resize`: passed.
- parser guard for `--skip_reparent_fit_evidence_for_teacher_cache` without `--teacher_cache_rewrite_rgb_render_to_parent`: failed as expected.
- direct-teacher low-copy dry-run: passed; command chain no longer includes `reparent_fit_evidence`, and manifest records `_fit_reparent_execution.skipped_for_teacher_cache=true`.

## Current Exact Command In Progress

This flowers exact command was launched on GPU 1 with W&B offline logging. It is not a completed result yet.

```bash
WANDB_DIR=/dev/shm/peilincai_wandb_v168_direct_teacher_lowcopy_exact \
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 \
PYTHONPYCACHEPREFIX=/dev/shm/peilincai_pycache_v168_direct_teacher_lowcopy_exact \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene flowers \
  --source_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/fit_evidence \
  --target_evidence_dir /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/target_evidence \
  --region_carrier_json /dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json \
  --teacher_render_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasej_trainval_gate/renders \
  --parent_render_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/train/ours_26000_phasef_extra_compact_base/renders \
  --reparent_target_parent_render_dir outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model/test/ours_26000_phasef_extra_compact_base/renders \
  --distillation_profile teacher_to_reparented_parent \
  --output_root /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact \
  --method_name ours_26000_v168_direct_teacher_lowcopy_flowers \
  --enable_policy_val_sparse_residual_materialization \
  --enable_train_only_target_impact_residual_basis \
  --target_impact_max_extra_bins 1024 \
  --reparent_allow_resize \
  --reparent_copy_mode auto_link \
  --teacher_cache_copy_mode auto_link \
  --teacher_cache_rewrite_rgb_render_to_parent \
  --skip_reparent_fit_evidence_for_teacher_cache \
  --wandb --wandb_mode offline \
  --wandb_group v168_direct_teacher_lowcopy_flowers \
  --wandb_name v168-direct-teacher-lowcopy-flowers
```

Promotion gate remains unchanged: do not run full9 unless flowers beats Phase-J all-axis:

- PSNR > `20.304358`
- SSIM > `0.557770`
- LPIPS < `0.329222`

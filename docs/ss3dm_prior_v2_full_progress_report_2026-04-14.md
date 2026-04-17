# SS3DM Prior V2 Full Progress Report

- Updated at UTC: `2026-04-14T01:15:30+00:00`
- Status: `Step 1` through `Step 6` completed
- Scope: protocol audit, cache v2, hybrid model/losses, trainer/logging, eval/report, and ablation orchestration

## Executive Summary

The original v1 local patch denoiser pipeline has now been upgraded into a configurable v2 local geometry-prior stack without replacing the legacy baseline path. The repo now supports strict split auditing, visibility-aware patch caches, hybrid v2 reconstruction with prototype memory and occupancy supervision, trainer-side curriculum and checkpoint selection, richer eval/report outputs, and a paper-style ablation suite that organizes multiple variants into one manifest plus aggregate CSV/Markdown tables.

All major additions remain config-driven. Legacy v1 training and eval still exist as explicit baselines, while hybrid v2 features activate only when their model/loss/config switches are enabled.

## Step-by-Step Progress

### Step 1

- Added strict protocol auditing via `ss3dm_prior.tools.audit_run_protocol`.
- Repaired retrieval naming and added filtered metrics:
  - `retrieval_top1_self_aligned`
  - `retrieval_top5_self_aligned`
  - `retrieval_top1_nonself`
  - `retrieval_top5_nonself`
  - `retrieval_top1_cross_sequence`
- Added strict baseline config and launcher:
  - `configs/ss3dm_prior/train_v8_strict.yaml`
  - `scripts/ss3dm_prior/train_v8_strict.sh`

### Step 2

- Added patch cache v2 fields for:
  - surface queries
  - free-space queries
  - unknown queries
  - support counts
  - visibility fractions
  - intrinsic patch difficulty
- Added new cache tools:
  - `ss3dm_prior.tools.build_teacher_patch_cache_v2`
  - `ss3dm_prior.tools.check_teacher_patch_cache_v2`

### Step 3

- Added `HybridPatchPriorV2`.
- Added vector quantizer module and prototype-memory outputs.
- Added occupancy and intrinsic-difficulty heads.
- Extended losses and metrics for v2 supervision.
- Kept `LocalPatchDenoiser` as the stable entrypoint with `model_type` dispatch.

### Step 4

- Upgraded trainer collation, curriculum, checkpoint selection, weighted hard-example sampling, and v2 metric logging.
- Added new checkpoints:
  - `best_composite.pt`
  - `best_visibility.pt`
- Added qualitative outputs:
  - `visibility_panel`
  - `hybrid_reconstruction_panel`
  - `sequence_visibility_map`
  - `prototype_usage_gallery`

### Step 5

- Upgraded eval/report from legacy reconstruction-only emphasis to v2 metrics plus protocol context.
- Added global and grouped outputs for:
  - reconstruction
  - denoise gain
  - intrinsic difficulty
  - occupancy/free-space
  - non-self retrieval
  - prototype usage
  - protocol validity
- Added qualitative eval outputs:
  - `best_hidden_completion`
  - `worst_free_space_violation`
  - `largest_intrinsic_score_error`
  - `prototype_gallery`
  - `sequence_visibility_map`

### Step 6

- Added `ss3dm_prior.tools.run_ablation_suite`.
- Added `ss3dm_prior.tools.aggregate_ablation_results`.
- Added `scripts/ss3dm_prior/run_ablation_suite.sh`.
- Added a real `use_vector_quantization` model flag so `v2_visibility_no_vq` is a true no-VQ ablation.
- Added aggregate paper-style outputs:
  - `suite_manifest.json`
  - `ablation_summary.csv`
  - `ablation_summary.md`

## Current Standard Variant Set

- `legacy_v1_strict`
- `v2_no_visibility`
- `v2_visibility_no_vq`
- `v2_visibility_plus_vq`
- `v2_full`
- Optional:
  - `v2_no_camera_visibility`

## Latest Validation Snapshot

### Step 5 Eval Smoke

- Real debug eval output:
  - `outputs/ss3dm_prior_eval/step5_debug_eval`
- Confirmed:
  - new `metrics_summary.json`
  - new `report.md`
  - new qualitative PNG categories

### Step 6 Debug Ablation Suite

- Output root:
  - `outputs/ss3dm_prior_ablations/step6_debug_suite`
- Aggregate files:
  - `ablation_summary.csv`
  - `ablation_summary.md`
  - `suite_manifest.json`
- All five required variants completed on a synthetic strict-valid debug split.

Debug suite metric snapshot:

- `legacy_v1_strict`: `recon_chamfer_l1=1.3048`, `denoise_gain_chamfer=0.0438`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
- `v2_no_visibility`: `recon_chamfer_l1=1.3398`, `denoise_gain_chamfer=0.0088`, `intrinsic_difficulty_spearman=0.2571`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.3333`, `protocol_valid=True`
- `v2_visibility_no_vq`: `recon_chamfer_l1=1.3396`, `denoise_gain_chamfer=0.0090`, `intrinsic_difficulty_spearman=0.0286`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
- `v2_visibility_plus_vq`: `recon_chamfer_l1=1.3208`, `denoise_gain_chamfer=0.0278`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`
- `v2_full`: `recon_chamfer_l1=1.3208`, `denoise_gain_chamfer=0.0278`, `occupancy_iou_visible=0.5`, `free_space_violation_rate=1.0`, `retrieval_top1_nonself=0.1667`, `protocol_valid=True`

## Key Artifacts

- Strict baseline:
  - `configs/ss3dm_prior/train_v8_strict.yaml`
- Hybrid model config:
  - `configs/ss3dm_prior/model_v8_hybrid.yaml`
- Hybrid train config:
  - `configs/ss3dm_prior/train_v8_hybrid.yaml`
- Hybrid eval launcher:
  - `scripts/ss3dm_prior/eval_v8_hybrid.sh`
- Ablation launcher:
  - `scripts/ss3dm_prior/run_ablation_suite.sh`

## Known Limits

- Tiny debug suites can still produce `NaN` for ranking-based metrics such as `intrinsic_difficulty_spearman` or filtered retrieval under weak signal.
- The current `v2_no_camera_visibility` row depends on supplying a separate lidar-only patch cache; the suite supports it, but it is not part of the default mandatory set.
- The synthetic debug ablation suite validates orchestration and aggregation, not final paper conclusions.
- The hidden-completion qualitative selector is still heuristic and may need a more explicit geometry-grounded definition later.

## Recommended Next Real-Data Actions

- Run the ablation suite on a strict-valid multi-town patch cache with enough held-out test sequences for stable non-self retrieval.
- Decide on the final paper checkpoint-selection policy for cross-row fairness.
- If needed, build a lidar-only visibility cache so `v2_no_camera_visibility` becomes a fully populated paper row rather than an optional extension.

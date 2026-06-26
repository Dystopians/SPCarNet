# SPCarNet vNext Strict Frozen-Policy Multiscene Log

Date: 2026-06-26

This log records the first strict no-target-GT apply multiscene validation for the frozen face-softshrink vNext policy.

## Protocol

All three strict runs use the same policy:

```text
texture_size_candidates=16
support_expansion_mode=none
atlas_empty_bin_fill_mode=face_mean
surface_multiscale_prior_blend_candidates=0.5
max_abs_delta_rgb_candidates=0.12
hard bin uncertainty guard disabled
soft bin uncertainty shrink enabled
strict_no_target_gt_apply enabled
```

No per-scene retuning was applied in this batch.

Strict apply chain:

```text
strip_target_evidence_no_gt
  -> apply_certified_residual_texture
  -> populate_eval_gt_from_target_evidence
  -> evaluate_vnext_target
```

The target evidence seen by the adapter keeps only geometry/parent-render keys and strips `rgb_gt`, direct residuals, and `teacher_*` keys. Target GT is populated only after rendering for final evaluation.

## Results

Artifact summary:

```text
docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/strict_frozen_policy_multiscene_summary.md
```

| scene | protocol | accepted | alpha | changed fraction | PSNR delta vs Phase-F | SSIM delta vs Phase-F | LPIPS delta vs Phase-F | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| counter | pass / applyGT=False | `True` | `0.25` | `0.011774` | `+0.002131` | `-0.000047` | `-0.000085` | Nonzero accepted; PSNR/LPIPS micro-gain; SSIM micro-regression |
| bonsai | pass / applyGT=False | `True` | `0.25` | `0.001513` | `+0.001225` | `-0.000010` | `-0.000018` | Nonzero accepted; PSNR/LPIPS micro-gain; SSIM micro-regression |
| room | pass / applyGT=False | `False` | `0.0` | `0.000000` | `-0.000097` | `-0.000003` | `-0.000007` | Certificate rejected; fallback/no-op; metric deltas are parent-level eval noise |

Aggregate:

```json
{
  "scene_count": 3,
  "complete": 3,
  "protocol_passed": 3,
  "target_gt_hidden_from_apply": 3,
  "accepted_nonzero": 2,
  "fallback_noop": 1,
  "psnr_better_vs_parent": 2,
  "ssim_better_vs_parent": 0,
  "lpips_better_vs_parent": 3
}
```

Mean delta vs Phase-F compact parent:

```text
+0.001086 PSNR / -0.000020 SSIM / -0.000037 LPIPS
```

## Interpretation

This is a real engineering milestone:

- strict no-target-GT apply is executable on multiple scenes;
- all three strict runs pass protocol audit;
- two scenes accept nonzero residual surface textures without seeing target GT;
- W&B offline logging was enabled for the medium runs.

It is not a paper-level quality milestone:

- SSIM regresses on all three strict scenes;
- room falls back with `changed_fraction=0` because lower-tail and min-view evidence are unsafe;
- PSNR/LPIPS gains on counter/bonsai are micro-scale;
- the aggregate LPIPS count includes the room fallback row, but that row is not nonzero residual quality gain;
- the table compares against Phase-F compact parent, not clean MeshSplatting or v106.

The strongest current diagnosis is:

> vNext can safely apply small residual surface textures, but the current residual atlas and certificate are still optimizing MSE/LPIPS-like effects more reliably than image-structure preservation. A paper-grade next step must make the residual representation SSIM-aware or structure-preserving, not merely more aggressive.

## Run Roots

```text
/dev/shm/peilincai_spcarnet_vnext_face_softshrink_counter_strict_20260626_045300_counter_strict
/dev/shm/peilincai_spcarnet_vnext_face_softshrink_bonsai_strict_20260626_052500_bonsai_strict
/dev/shm/peilincai_spcarnet_vnext_face_softshrink_room_strict_20260626_052500_room_strict
```

W&B offline roots:

```text
/dev/shm/peilincai_wandb_vnext_face_softshrink_counter_strict_20260626_045300_counter_strict
/dev/shm/peilincai_wandb_vnext_face_softshrink_bonsai_strict_20260626_052500_bonsai_strict
/dev/shm/peilincai_wandb_vnext_face_softshrink_room_strict_20260626_052500_room_strict
```

## Execution Lesson

An initial relaunch mistakenly passed `--output_root "$OUT"` in the same shell command that assigned `OUT=...`; shell expansion happened before the temporary assignment, so output fell back to the repo working directory. The partial generated `room/` and `bonsai/` directories were removed, and the successful relaunch used literal `/dev/shm/...` paths.

Future long commands should either export variables before invoking Python or pass literal output roots.

## Next Method Step

The next method change should target SSIM/structure preservation directly:

- add a train-policy-val structural guard or shrink rule that uses image-level SSIM failure modes more directly;
- reduce or locally downweight residuals in high-gradient and texture-boundary regions that are not multi-view consistent;
- report a frozen-policy ablation with and without the new structure-aware shrink;
- promote only if strict multiscene SSIM no longer regresses while PSNR/LPIPS retain gains.

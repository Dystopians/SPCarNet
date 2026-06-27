# 2026-06-26 vNext Prompt Progress and Confidence Log

## Bottom Line

The new vNext prompt has produced meaningful progress in protocol, safety, and evidence accounting, but it has not yet reached the expected paper-level outcome. The current branch should be treated as an active method-construction stage, not a finished claim that SPCarNet comprehensively beats the strongest MeshSplatting baseline.

## What Has Clearly Improved

- Added a stricter vNext pipeline around train-only certification, target-GT-free application, policy-val selection, W&B logging, and artifact manifests.
- Added reference comparison tooling so each run can be compared against clean MeshSplatting and the retained v106 line instead of being judged in isolation.
- Added capacity/accounting tooling for method, acceptance, atlas size, changed fraction, triangle reduction, and command timing.
- Added method-side mechanisms beyond parameter scanning:
  - adaptive texture-size ladder from train-fit residual density;
  - SSIM/L1/LPIPS-capable policy validation;
  - effective-margin gates;
  - structure-aware shrink;
  - sparse policy-val residual materialization for cross-view-stable bins.
- Fixed a runner bug where negative scientific-notation thresholds were forwarded as `--flag -1e-7`, which caused `argparse` failures. The runner now emits `--flag=-1e-7` for negative values.

## Current Evidence

Retained broad best line before the new vNext prompt:

- v106 full9 mean: PSNR `25.831280`, SSIM `0.760830`, LPIPS `0.268435`.
- clean MeshSplatting full9 mean: PSNR `25.151682`, SSIM `0.749018`, LPIPS `0.287621`.

Current vNext full9 evidence:

- structure-aware shrink cleanup: PSNR `25.067699`, SSIM `0.741260`, LPIPS `0.306689`.
- effective-margin gate: PSNR `25.067410`, SSIM `0.741259`, LPIPS `0.306695`.

These vNext full9 results pass protocol checks but are below both clean and v106 in aggregate. They are useful as a certified scaffold, not as the final quality method.

Scene-level active probes:

- `v127 flowers`: completed as fallback/no-op. It remains non-regressive because the parent is strong, but it does not prove the new residual module improved flowers.
- `v127/v128b/v129 counter`: still running in adapter/candidate evaluation at log time. Early policy-val candidates show positive train-policy-val gains, but final test metrics are not yet available.
- `v130b flowers sparse materialization`: running after the negative-threshold runner fix. This is the current real test of whether stable-bin sparse residual materialization can repair the flowers no-op weakness.

## Honest Assessment

There is significant progress in making the method reliable, auditable, and fair. There is not yet significant enough progress in final visual or quantitative dominance. In particular, flowers exposes the central weakness: surface residuals can appear useful on train-fit evidence but become unstable across views, so a safe policy often rejects them and falls back to no-op.

My confidence is high that the new prompt can produce a much cleaner research-grade pipeline and a defensible failure analysis. My confidence is only medium that the current residual-texture family alone will yield a large visual/metric jump without an additional representation-level improvement. The sparse stable-bin materialization probe is the immediate test of that hypothesis.

## Next Required Steps

- Finish v127/v128b/v129 counter and v130b flowers; summarize against clean and v106 using `summarize_vnext_run_artifacts.py`.
- If v130b is accepted, run the same fixed policy on additional scenes without scene-specific hand tuning.
- If v130b is still no-op or regressive, stop treating residual texture as sufficient and escalate to a representation-level change that can improve outdoor/vegetation scenes rather than only certify safety.
- Update README and mentor report only after the new method has real test-set artifacts, qualitative panels, and accounting tables.

## Latest Run Closure, 2026-06-26 22:05 PDT

The v127-v130b probe batch is now complete.

### Verified Results

| scene | method | PSNR | SSIM | LPIPS | vs clean | vs v106 | accepted | changed fraction | result path |
|---|---|---:|---:|---:|---|---|---|---:|---|
| counter | `ours_26000_v127_adaptive_capacity_knee_counter` | 27.499430 | 0.867480 | 0.238782 | strict win | not non-regressive | yes | 0.012341135 | `/dev/shm/peilincai_spcarnet_v127_counter_adaptive_capacity_knee_20260626_205940/counter/reports/counter_ours_26000_v127_adaptive_capacity_knee_counter_test_results.json` |
| counter | `ours_26000_v128b_patched_adaptive_capacity_counter` | 27.500118 | 0.867516 | 0.238784 | strict win | PSNR/LPIPS win, SSIM -0.000006 | yes | 0.020829242 | `/dev/shm/peilincai_spcarnet_v128b_counter_patched_adaptive_capacity_20260626_2126/counter/reports/counter_ours_26000_v128b_patched_adaptive_capacity_counter_test_results.json` |
| counter | `ours_26000_v129_lpips_gate_counter` | 27.500118 | 0.867516 | 0.238784 | strict win | PSNR/LPIPS win, SSIM -0.000006 | yes | 0.020829242 | `/dev/shm/peilincai_spcarnet_v129_counter_lpips_gate_20260626_2135/counter/reports/counter_ours_26000_v129_lpips_gate_counter_test_results.json` |
| flowers | `ours_26000_v127_adaptive_capacity_knee_flowers` | 20.452776 | 0.549059 | 0.355544 | strict win | strict win | no, fallback | 0.000000000 | `/dev/shm/peilincai_spcarnet_v127_flowers_adaptive_capacity_knee_20260626_2113/flowers/reports/flowers_ours_26000_v127_adaptive_capacity_knee_flowers_test_results.json` |
| flowers | `ours_26000_v130b_sparsebin_flowers` | 20.452776 | 0.549059 | 0.355544 | strict win | strict win | no, fallback | 0.000000000 | `/dev/shm/peilincai_spcarnet_v130b_flowers_sparsebin_materialization_argfix_20260626_2155/flowers/reports/flowers_ours_26000_v130b_sparsebin_flowers_test_results.json` |

Interpretation:

- `v128b/v129 counter` are useful local results: they beat clean MeshSplatting on all three metrics and nearly tie v106, but they still miss strict v106 non-regression because SSIM is lower by about `0.000006`.
- `v130b flowers` is a negative result for the current sparse residual materialization idea. The sparse profile found `199` allowed bins, but post-materialization risk checks still rejected it: CVaR relative gain was `-0.047315`, min-view relative gain was `-0.130224`, and SSIM positive-view fraction was only `0.416667`.
- The flowers numbers are good only because fallback/no-op preserves a strong parent. They are not evidence that the residual module improved flowers.

### Code/Protocol Fixes From This Batch

- Runner now normalizes negative numeric CLI values before parsing and before forwarding adapter commands, so values such as `--min_policy_val_effective_ssim_gain -1e-7` are converted to `--min_policy_val_effective_ssim_gain=-1e-7`.
- Runner now errors if LPIPS effective thresholds are set without `--enable_policy_val_image_lpips_gate`.
- Sparse materialization now seeds from the risk-gated policy best row when available.
- Sparse materialization and normal bin uncertainty guard now use an intersection profile instead of one overwriting the other.
- Sparse replacement now synchronizes the selected alpha used by downstream guards.
- LPIPS metrics are included in view-basis and teacher-basis non-regressive guards when LPIPS gating is enabled.
- Sparse/target-support accepted candidates now require nonzero target-visible output; otherwise they are rejected and written as fallback.

### Current Judgment

This batch improves correctness and evidence quality, but it does not reach the expected vNext outcome. The new prompt produced a stronger and cleaner experimental scaffold, plus a local counter result that is almost v106-equivalent, but it did not solve the cross-view residual instability exposed by flowers. The next real method step should move beyond residual texture gating alone and add a representation-level mechanism for view-stable residual prediction or target-visible capacity allocation.

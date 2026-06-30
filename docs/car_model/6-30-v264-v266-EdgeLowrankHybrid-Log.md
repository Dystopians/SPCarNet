# v264-v266 Edge / Low-Rank Hybrid Deferred Source Renderer Log

Date: 2026-06-30

Prompt followed: `docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Status: **NOT COMPLETE** for paper-level all-axis success. Full9 remains blocked because flowers exact still does not beat Phase-J all-axis.

## Why This Round Was Run

v260-v263 showed that the deferred source renderer is a cleaner no-target-GT direction than earlier scalar atlas/alpha loops, but the best flowers exact result was still far below the Phase-J PSNR gate. The next prompt explicitly asked for a representation change rather than more alpha/footprint tuning.

This round implemented and tested:

- `edge_local_linear`: local ridge residual decoding with parent-edge features.
- `lowrank_source_basis`: per face/UV bin source-slot low-rank teacher residual basis, with `source_view_id` saved into the bank to audit source-view diversity.
- `hybrid_edge_lowrank`: edge-local-linear as the stable base plus disagreement-aware low-rank residual injection.

The target/test apply path still uses stripped no-GT target evidence. Target/test RGB GT is loaded only after apply for exact evaluation.

## Code Changes

Main file:

- `scripts/car_model/train_surface_deferred_source_residual_renderer.py`

New CLI/API:

- `--source_edge_score_weight`
- `--residual_decoder_mode edge_local_linear`
- `--residual_decoder_mode lowrank_source_basis`
- `--residual_decoder_mode hybrid_edge_lowrank`
- `--lowrank_basis_rank`
- `--lowrank_basis_min_sources`
- `--lowrank_basis_min_unique_views`
- `--lowrank_basis_l2`
- `--lowrank_basis_blend`
- `--lowrank_basis_residual_clip`
- `--lowrank_basis_disagreement_beta`
- `--target_edge_gain`
- `--target_edge_gain_clip`

Checkpoint schema additions:

- `parent_edge`
- `source_view_id`

## Method Summary

`edge_local_linear` extends the previous local-linear decoder with target/source parent-edge features. The goal is to preserve high-frequency residuals without blindly increasing alpha.

`lowrank_source_basis` fits a compact residual basis from the train-fit source slots in each face/UV bin. It predicts low-rank coefficients from source view direction, parent RGB, and parent-edge features, then evaluates them at the target view/parent state.

`hybrid_edge_lowrank` was added after the pure low-rank runs failed to dominate. It keeps edge-local-linear as a stable base and injects low-rank detail only through a disagreement-aware blend:

- if low-rank prediction agrees with the stable base, more low-rank detail survives;
- if the low-rank prediction diverges, it is automatically damped;
- this uses no target/test RGB GT.

This is a real train/eval pipeline method change, but it is still not enough to beat Phase-J.

## Experiment Paths

Machine-readable summary:

- `docs/car_model/results/v264_v266_edge_lowrank_hybrid_summary.json`

v264:

- `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v264a_edge_local_linear_targetvisible_32k_targetexact/v253_deferred_source_renderer_audit.json`
- `/tmp/peilincai_spcarnet_v253_deferred_flowers_20260629/v264b_edge_local_linear_edgegain025_targetexact/v253_deferred_source_renderer_audit.json`

v265:

- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265a_lowrank_source_basis_targetvisible_32k/v253_deferred_source_renderer_audit.json`
- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v265_lowrank_full_flowers_20260630/v265b_lowrank_rank2_blend05_loadedbank/v253_deferred_source_renderer_audit.json`

v266:

- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v266_hybrid_flowers_20260630/v266a_hybrid_edge_lowrank_loadedbank/v253_deferred_source_renderer_audit.json`
- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v266_hybrid_flowers_20260630/v266b_hybrid_edgegain010_loadedbank/v253_deferred_source_renderer_audit.json`
- `/data/peilincai/mesh-splatting/outputs/carnet/spcarnet_v266_hybrid_flowers_20260630/v266c_hybrid_conservative_loadedbank/v253_deferred_source_renderer_audit.json`

Each audit JSON contains the full command line, W&B offline run path, selected alpha, no-target-GT audit, policy-val table, target exact summary, and render directories.

## Flowers Exact Results

Phase-J flowers reference: `20.304358 / 0.557770 / 0.329222`.

| run | method | target PSNR | SSIM | LPIPS | gains | changed | PSNR tail | Phase-J PSNR gap |
|---|---|---:|---:|---:|---|---:|---:|---:|
| v263a | local-linear | 19.844512 | 0.620224 | 0.179968 | +0.012458 / +0.000314 / +0.000367 | 0.040890 | -0.002673 | -0.459846 |
| v264a | edge-local-linear | 19.844520 | 0.620226 | 0.179971 | +0.012467 / +0.000315 / +0.000364 | 0.040927 | -0.002670 | -0.459838 |
| v264b | edge gain 0.25 | 19.845366 | 0.620176 | 0.179872 | +0.013312 / +0.000266 / +0.000463 | 0.057264 | -0.003920 | -0.458992 |
| v265a | low-rank rank 3 | 19.844019 | 0.620207 | 0.179931 | +0.011965 / +0.000296 / +0.000403 | 0.040368 | -0.003271 | -0.460339 |
| v265b | low-rank blended | 19.844584 | 0.620177 | 0.179939 | +0.012530 / +0.000266 / +0.000396 | 0.052283 | -0.002389 | -0.459774 |
| v266a | hybrid | 19.845654 | 0.620199 | 0.179918 | +0.013600 / +0.000288 / +0.000417 | 0.054203 | -0.002161 | -0.458704 |
| v266b | hybrid + edge gain 0.10 | 19.845553 | 0.620196 | 0.179897 | +0.013499 / +0.000286 / +0.000438 | 0.055207 | -0.003085 | -0.458805 |
| v266c | conservative hybrid | 19.845698 | 0.620201 | 0.179915 | +0.013644 / +0.000290 / +0.000419 | 0.054285 | -0.002039 | -0.458660 |

Best by axis:

- Best target PSNR: `v266c`, `19.845698`.
- Best target SSIM: `v264a`, `0.620226`.
- Best target LPIPS: `v264b`, `0.179872`.

## Interpretation

The useful positive signal is narrow but real:

- v266c gives the best deferred-source target PSNR so far.
- v266c also improves PSNR tail CVaR over v263/v264.
- The hybrid mechanism is better than pure low-rank replacement.

The failure is also clear:

- No run beats Phase-J PSNR. The best v266c PSNR is still `0.458660` below Phase-J flowers.
- No single v264-v266 run dominates all axes against earlier deferred-source variants.
- The target changed fraction is still only about `0.054`, so most target-visible surface support remains visually unchanged.
- SSIM/LPIPS tails remain negative, meaning the method is not yet robust enough for full9 promotion.

## No-GT / Fairness Notes

- Target evidence is stripped before apply.
- Forbidden target/test GT keys are checked by `_verify_target_no_gt`.
- Target exact metrics are computed only after no-GT apply.
- Policy-val uses GT for certification, as allowed by the v169 prompt.
- All medium/full runs used W&B offline logs.
- Full9 was not launched because the flowers Phase-J gate failed.

## Next Bottleneck

The bottleneck is no longer interface completeness. It is representation strength and target-view generalization.

Recommended next mechanism:

1. Move from source-slot low-rank RGB residuals to coherent face/patch texture features across UV bins.
2. Add explicit local patch/gradient supervision in the residual representation rather than only RGB residual fitting.
3. Learn a target-free uncertainty/visibility model that predicts where low-rank/detail injection is safe.
4. Keep the v169 rule: no full9 until flowers exact beats Phase-J all-axis.

Current final status for this round: **NOT COMPLETE**.

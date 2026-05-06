# Final Stage SCE8 Multiscene SCE Policy Report

Date: 2026-05-06

Decision: `SCE_POLICY_V1_RENDER_PASS_GEOMETRY_MIXED`

## Status

The SCE8 collector is implemented in `scripts/car_model/final_collect_stageSCE8_multiscene_policy.py`. It builds a fair table from existing `results.json` and `geometry_eval_colmap` artifacts for any scene/model/iteration pair.

The current real SCE7 candidate is validated on courtyard and a first bonsai fixed-policy probe.

Courtyard is a strong partial:

- RGB, LPIPS, AbsRel, and normal all beat F82.
- Depth MAE remains `+0.001787` above F82.

Bonsai seed0 probe:

`outputs/carnet/meshsplatopt/final_stageSCE8_multiscene_sce_policy/bonsai/sce_probe_v1_26000to26200_seed0/recovery_model`

- W&B: `s6yztj51`
- fixed knobs: sparse lambda `0.003`, render-normal anchor `0.01`, LR `0.005`, no scene-specific retuning
- topology unchanged: `true`
- RGB worsened vs F82: PSNR `-0.176259`, SSIM `-0.044556`, LPIPS `+0.030025`
- sparse depth improved: AbsRel `-0.022422`, Depth MAE `-0.265169`
- normal slightly worsened: `+0.043792`

This is useful negative evidence. SCE v1 does not supersede F82 as a universal multiscene recovery policy; it behaves as a targeted geometry repair module unless appearance protection is improved.

Follow-up SCE19 implements the required policy guard. With strict opt-in render protection, the bonsai SCE8 candidate is rejected as `accept_parent_noop` because PSNR/SSIM decrease and LPIPS increases, despite sparse depth improvement. This turns the failure into controlled negative-transfer prevention rather than a manual scene exception.

Follow-up SCE21 implements CTR-SCE tail-risk sentinel rollback. On courtyard seed0 it is the first candidate to beat F82 on all six tracked independent metrics with unchanged topology:

- PSNR `+0.417478`
- SSIM `+0.030249`
- LPIPS `-0.006806`
- AbsRel `-0.003668`
- Depth MAE `-0.003262`
- Normal `-0.876624`

This updates the courtyard status from partial to all-metric pass, but does not change the multiscene decision label yet because bonsai/room/counter fixed-policy validation remains incomplete.

## Next Validation Requirement

Before claiming SCE-Repair fully beats F82, run CTR-SCE guarded fixed policy on bonsai, room, and counter with no per-scene retuning, then use the collector to build the final SCE8 table. Current multiscene label remains `SCE_POLICY_V1_RENDER_PASS_GEOMETRY_MIXED`; courtyard alone is now `SCE21_COURTYARD_ALL_METRIC_PASS_VS_F82`.

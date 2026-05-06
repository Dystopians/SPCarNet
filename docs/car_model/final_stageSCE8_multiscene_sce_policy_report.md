# Final Stage SCE8 Multiscene SCE Policy Report

Date: 2026-05-06

Decision: `SCE8_COLLECTOR_IMPLEMENTED_COURTYARD_PARTIAL`

## Status

The SCE8 collector is implemented in `scripts/car_model/final_collect_stageSCE8_multiscene_policy.py`. It builds a fair table from existing `results.json` and `geometry_eval_colmap` artifacts for any scene/model/iteration pair.

The current real SCE7 candidate is only validated on courtyard. It is a strong partial:

- RGB, LPIPS, AbsRel, and normal all beat F82.
- Depth MAE remains `+0.001787` above F82.
- No SCE8 all-scene fixed-policy claim is made yet.

## Next Validation Requirement

Before claiming SCE-Repair fully beats F82, run the fixed SCE7 policy on bonsai, room, and counter with scene-specific train/calibration sentinel caches but identical policy thresholds. Then use the collector to build the final SCE8 table.


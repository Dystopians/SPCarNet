# Car Model V4 Focus

## Scope Correction

This document corrects an important scope mistake:

- the `car model` work is **not** part of the SS3DM research scope
- it should **not** be described as an SS3DM sub-track
- the fact that some code lives under `ss3dm_prior/` is an engineering/repository detail, not a scope definition

The current priority is the standalone `car model` pipeline.

We may reuse infrastructure from this repository, including components currently stored under `ss3dm_prior/`, but the optimization target itself is the MeshFleet car task, not SS3DM.

## Current Priority

All near-term preprocessing, model, training, evaluation, and reporting changes should be justified by whether they help the `car model`.

## Scope Definition

The active task is the MeshFleet whole-car reconstruction / repair route as an independent task:

- dataset source: `MeshFleet_TRELLIS`
- reconstructed mesh root: `MeshFleet_TRELLIS_RECONSTRUCTED_v4`
- cache builder: `ss3dm_prior.tools.build_car_mesh_patch_cache`
- data config: `configs/ss3dm_prior/data_meshfleet_car.yaml`
- preprocessing entrypoint: `scripts/car_model/pre_process_car.sh`
- multi-run training entrypoint: `scripts/car_model/train_meshfleet_car_v4_trio.sh`
- evaluation entrypoint: `scripts/car_model/eval_meshfleet_car_v4_trio.sh`

The path names above should not be interpreted as a taxonomy claim that this work belongs to SS3DM.

## Optimization Policy

Until further notice, the following rules apply:

1. Any new optimization should be evaluated first by its benefit to `car model` quality.
2. If a change helps some other branch in this repository but does not help the car route, it is not a priority.
3. If a change increases complexity, it must improve one or more car-facing outcomes:
   - lower reconstruction Chamfer
   - better denoise gain
   - better shape repair on hard corrupted cars
   - better qualitative whole-car completion
   - more stable car training and evaluation
4. Car-specific preprocessing, cache quality, objective design, and model selection take priority over broad generalization work.

## Current Working Assumption

Based on the current evidence in `outputs/ss3dm_prior_car/v3_5_experiment_report.md`:

- the main bottleneck is not just raw parameter count
- occupancy supervision is not yet meaningful on the current whole-car cache
- the current VQ setup is too unstable for the present car objective mix
- the most important near-term direction is to improve car-oriented supervision and shape-repair signal

Therefore, all V4 work should be interpreted as standalone `car model` optimization work, not as SS3DM roadmap work.

## Documentation Rule

When updating related docs, scripts, or experiment notes:

- explicitly label runs with `_v4` when they belong to this new car-focused phase
- state whether the change is in service of the `car model`
- avoid presenting repository-level or SS3DM-level improvements as the main story unless they directly support the car route

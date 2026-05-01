# MeshPrior Stage 13 Evaluation Protocol Implementation Report

Date: 2026-05-01

## Scope

Stage 13 turns the accumulated MeshPrior work into a repeatable evaluation protocol rather than another model change. It defines the registry, runner, report generator, and table schema needed to compare object-prior quality, synthetic mesh repair, scene-level optimization, and safety ablations.

## Files Added

- `docs/car_model/meshprior_stage13_eval_protocol_design.md`
- `configs/ss3dm_prior/meshprior/meshprior_experiment_matrix.yaml`
- `scripts/car_model/meshprior_run_experiment_matrix.py`
- `scripts/car_model/meshprior_make_neurips_report.py`
- `scripts/car_model/smoke_test_meshprior_stage13_eval_protocol.py`
- `docs/car_model/reports/meshprior_neurips_main_report.md`

## Registry

The registry includes rows for:

- v0.7 residual baseline.
- v0.8.2 point-flow baseline.
- Stage 3 posterior encoder.
- Stage 4 MAP refinement.
- Stage 5 oracle K=8 analysis.
- protect/prune proposals.
- protect/prune + snap.
- protect/prune + snap + fill.
- `surface_support_v1` calibration.
- scene baseline no-cleanup wandb smoke.
- scene baseline + MeshPrior gated proposals.

Missing artifacts are retained as `MISSING` with an explicit `missing_reason`; they are not silently filtered from the report.

## Runner Behavior

`scripts/car_model/meshprior_run_experiment_matrix.py` supports:

- `--dry_run`
- `--smoke`
- `--only`
- `--group object|synthetic|scene|all`
- `--seeds 0,1,2`
- `--max_objects`
- `--no_train`
- `--eval_only`

The M13 implementation is intentionally evaluation-first. It reads existing metrics and does not launch training unless a future stage explicitly connects matrix rows to train commands.

## Report Outputs

The report generator writes:

- `docs/car_model/reports/meshprior_neurips_main_report.md`
- `outputs/carnet/meshprior/reports/object_table.csv`
- `outputs/carnet/meshprior/reports/synthetic_damage_table.csv`
- `outputs/carnet/meshprior/reports/scene_table.csv`
- `outputs/carnet/meshprior/reports/ablation_table.csv`
- `outputs/carnet/meshprior/reports/failure_cases.md`

## Full Dry-Run Result

Command:

```bash
micromamba run -n mesh_splatting python scripts/car_model/meshprior_run_experiment_matrix.py --dry_run --group all --no_train --eval_only --output_dir outputs/carnet/meshprior/experiment_matrix
```

Result:

- `total=11`
- `available=7`
- `missing=4`

Generated matrix:

- `outputs/carnet/meshprior/experiment_matrix/matrix_results.json`

Report command:

```bash
micromamba run -n mesh_splatting python scripts/car_model/meshprior_make_neurips_report.py --matrix_results outputs/carnet/meshprior/experiment_matrix/matrix_results.json --output_dir outputs/carnet/meshprior/reports --report_dir docs/car_model/reports
```

## Key Available Evidence

Object prior:

- v0.8.2 point-flow baseline: recon Chamfer L1 `0.1231232387131279`, hidden Chamfer L1 `0.1548924239225758`.
- Stage 3 posterior encoder: recon Chamfer L1 `0.0663909994752951`, hidden Chamfer L1 `0.0990753869336207`, mesh extraction success `1.0`.

Synthetic repair:

- `surface_support_v1` calibration keeps valid-surface protect recall at `0.9166666666666666`.
- Uncalibrated snap recall is `0.8333333333333334`.
- M11 fill dry-run closes `4.0` boundary edges with zero reported free-space delta.

Scene:

- 200-iteration no-cleanup wandb smoke: PSNR `6.933581471443176`, SSIM `0.16371289547532797`, LPIPS `0.694071426987648`, triangle count `5706`, controlled FPS `334.7374487692397`.
- COLMAP sparse geometry on that checkpoint: AbsRel `0.10470779720655764`, depth MAE `0.024122862845250084`, normal mean angle `37.51919533010328`.
- Synthetic MeshPrior proposal gate: `accepted_proposals=1`, `rejected_proposals=0`.

## Missing Rows

The following rows are expected to stay visible as `MISSING` until the corresponding artifacts are produced:

- `v0_7_residual_baseline`
- `spcarnet_stage4_map_refinement`
- `spcarnet_stage5_oracle_k8`
- `protect_prune_proposals`

## Safety Notes

- Oracle-only rows are labeled in the matrix and are not used as headline inference-time results.
- Missing baselines remain in the tables.
- Scene-level MeshPrior evidence is still dry-run proposal evidence; real render-gated proposal application remains future work.
- The report separates object, synthetic, scene, and ablation metrics so object-prior gains do not masquerade as scene optimization wins.

## Gate

M13 gate: `PASS`.

The next allowed stage is M14 only after this report, smoke report, research log entry, commit, and push are complete.

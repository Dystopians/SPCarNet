# v100 Checkpoint-Attached ELA Counter Validation

- status: `PASS_COUNTER_GATE`
- scene: `counter`
- method: `ours_26000_v100_checkpoint_attached_ela_endpoint`
- run root: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_20260625/counter_v100_checkpoint_attached_ela`
- model path: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_20260625/counter_v100_checkpoint_attached_ela/recovery_model`
- no test GT for policy: `True`

## Main Result

| Method | PSNR | SSIM | LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean |
|---|---:|---:|---:|---:|---:|---:|
| v100 endpoint | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 |
| clean MeshSplatting | 26.751774 | 0.862055 | 0.252003 | 0 | 0 | 0 |
| strict gate floor | 26.756138 | 0.862126 | 0.251691 | | | |
| source ELA | 27.240423 | 0.864144 | 0.249701 | +1.208748 | +0.029586 | -0.063229 |
| v98b checkpoint-baked negative | 26.728172 | 0.860831 | 0.257008 | +1.720999 | +0.032900 | -0.070535 |
| Phase-J ceiling | 28.449171 | 0.893731 | 0.186472 | +0.000000 | +0.000000 | +0.000000 |

## Non-Noop Evidence

- non-noop pass: `True`
- changed pixel fraction mean: `0.972526`
- mean abs RGB delta: `0.011307`
- max abs RGB delta: `0.250000`
- per-view strict RGB wins vs clean: `30/30`

## Geometry And Topology

- topology unchanged: `True`
- triangles: `9644247` endpoint delta `0`
- vertices: `2478825` endpoint delta `0`
- geometry inherited: `True`
- depth AbsRel: `0.007637892`
- depth MAE: `0.058701707`
- normal mean angle: `27.085450`
- geometry safe: `True`

## Artifacts

- comparison JSON: `outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_comparison.json`
- comparison CSV: `outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_comparison.csv`
- per-view CSV: `outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_per_view_deltas.csv`
- contact sheet: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_20260625/counter_v100_checkpoint_attached_ela/recovery_model/qualitative/ours_26000_v100_checkpoint_attached_ela_endpoint_contact_sheet.png`
- W&B offline dirs: `['/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_20260625/counter_v100_checkpoint_attached_ela/wandb/wandb/offline-run-20260625_031016-fwfl7gmr']`

## Interpretation

This v100 artifact converts the strongest Phase-J/ELA render-time repair into a checkpoint-attached endpoint sidecar. It does not mutate MeshSplatting geometry or select any policy from held-out test GT. On counter it reaches the Phase-J ceiling while preserving the 2.0% compact topology reduction and inherited COLMAP geometry.

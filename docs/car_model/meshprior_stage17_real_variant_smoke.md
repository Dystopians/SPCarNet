# MeshPrior Stage 17 Real Variant Smoke

Date: 2026-05-01

## Scope

This smoke verifies that the Stage 17 MeshPrior-edited initialization can be represented as a normal model directory and loaded from `iteration_200`.

## Required Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_prepare_parking_recovery_model.py --source_model outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --copied_checkpoint outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt --output_model outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model --iteration 200
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage17_real_variant.py
```

## Expected Result

- source model edited: `false`
- recovery model written: `true`
- checkpoint exists at `point_cloud/iteration_200/point_cloud_state_dict.pt`
- the copied checkpoint has nonzero triangles and vertices

## Gate

Stage 17 initialization smoke: `PASS`.

Executed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_prepare_parking_recovery_model.py --source_model outputs/carnet/meshprior/parking_phone_tiny/baseline_200iter/model --copied_checkpoint outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt --output_model outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model --iteration 200
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage17_real_variant.py
```

Result:

- source model edited: `false`
- recovery model written: `true`
- initialization checkpoint: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model/point_cloud/iteration_200/point_cloud_state_dict.pt`
- triangles: `63965`
- vertices: `191895`

Training smoke also passed:

- command resumed the MeshPrior-edited checkpoint from iteration `200` to `300`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/y4432er1`
- iteration 300 test PSNR / SSIM / LPIPS: `11.5936053771` / `0.3349873807` / `0.6415096864`
- final cleanup: `enabled=false`, `pruned=0`

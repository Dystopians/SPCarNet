# PRISM Re-Run Checklist (2026-03-30)

## Purpose

This standalone note lists only practical rerun cautions and defaults for the current 3-case geogate round.

## Fixed data/split

- `SCENE_PATH=/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix`
- `SPLIT_FILE=/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse/0/split_outoftrain_v1.json`

## WandB naming

- `WANDB_PROJECT` means WandB project name, not repository folder name.
- Recommended default for this round:
  - `WANDB_PROJECT=mesh-splatting-prune`
- Suggested group:
  - `WANDB_GROUP=parking_phone_tiny_geogate_round2`

## Known failure seen in rerun

- Symptom: runs looked "stuck" around iter `1000`.
- Actual cause: first eval at `test_iterations=1000` triggered LPIPS CUDA OOM in eval path.
- Fix already applied in `train.py`:
  - eval loops wrapped by `torch.no_grad()`
  - LPIPS frozen via `eval()` + `requires_grad_(False)`

## Run command template

```bash
SCENE_PATH=/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix \
SPLIT_FILE=/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse/0/split_outoftrain_v1.json \
WANDB_ENABLE=1 \
WANDB_PROJECT=mesh-splatting-prune \
WANDB_GROUP=parking_phone_tiny_geogate_round2 \
bash scripts/parking_ground/run_geogate_round2_parallel.sh
```

## Quick preflight

1. `nvidia-smi` is available and at least 3 GPUs are visible.
2. `SCENE_PATH` exists and contains COLMAP data.
3. `SPLIT_FILE` exists and matches this scene.
4. WandB login is valid in current environment.

## OOM prevention knobs (recommended defaults)

- `MIN_FREE_GPU_MEM_MB=42000`  
  launcher only picks GPUs with >= 42 GB free memory.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`  
  reduces fragmentation risk for long runs.
- LPIPS eval in training report is downsampled to max side 1024 and guarded against OOM.

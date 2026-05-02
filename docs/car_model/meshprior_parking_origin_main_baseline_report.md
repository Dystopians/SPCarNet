# MeshPrior Parking Origin/Main Baseline Report

Date: 2026-05-01

## Scope

This report records the first clean/original Mesh Splatting baseline candidate for `parking_phone_tiny`.

The run was executed from a separate worktree:

- worktree: `/tmp/mesh-splatting-origin-main`
- commit: `origin/main@1a714f3`

The current `clean-submit` branch was not switched.

## WandB Note

The clean `origin/main` training script does not expose the current branch's `--enable_wandb`, `--wandb_project`, or related logging arguments. The 2000-iteration run was therefore logged to W&B after training with an external summary logger.

Added:

- `scripts/car_model/meshprior_log_parking_run_to_wandb.py`

W&B run:

- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/142memiw`

This is not as good as training-time W&B logging. Future current-branch medium and long runs must use training-time W&B.

## Commands

200-iteration smoke:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s /data/peilincai/mesh-splatting/outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m /data/peilincai/mesh-splatting/outputs/carnet/meshprior/parking_phone_tiny/origin_main_200iter/model --images images --eval --iterations 200 --test_iterations 200 --save_iterations 200 --checkpoint_iterations 200 --resolution 4 --scene_name parking_phone_tiny_origin_main_200iter --wandb_name parking_phone_tiny_origin_main_200iter
```

2000-iteration medium run:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_meshprior PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/peilincai/micromamba/envs/mesh_splatting/bin/python train.py -s /data/peilincai/mesh-splatting/outputs/carnet/meshprior/parking_phone_tiny/dataset_view -m /data/peilincai/mesh-splatting/outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model --images images --eval --iterations 2000 --test_iterations 1000 2000 --save_iterations 2000 --checkpoint_iterations 2000 --resolution 4 --scene_name parking_phone_tiny_origin_main_2000iter --wandb_name parking_phone_tiny_origin_main_2000iter
```

External W&B logging:

```bash
WANDB_PROJECT=spcarnet_meshprior WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_log_parking_run_to_wandb.py --model_path outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model --iteration 2000 --project spcarnet_meshprior --group parking_origin_main_baseline --name parking_origin_main_2000iter_external_log --artifact_name parking_origin_main_2000iter_summary --source origin/main@1a714f3 --note 'clean-origin training script lacks current --enable_wandb integration' --paper_baseline_candidate
```

## Results

### Origin/Main 200 Iter

`render.py + metrics.py`:

- SSIM: `0.0092272`
- PSNR: `5.8725734`
- LPIPS: `0.7112017`

Decision: too short and unreliable; use only as a smoke result.

### Origin/Main 2000 Iter

Training script internal evaluation at iteration `2000`:

- test L1: `0.11052933887199119`
- test PSNR: `16.46195650100708`
- test SSIM: `0.4846517714085402`
- test LPIPS: `0.5333475658187159`
- test FPS: `271.31298105829023`

`render.py + metrics.py`:

- SSIM: `0.21993064880371094`
- PSNR: `11.047659873962402`
- LPIPS: `0.6417058110237122`
- triangles: `39079`
- vertices: `58458`

The training-time and post-render metric scripts use different output/evaluation pathways and should not be mixed in one table without labeling.

## Interpretation

The user's concern was correct: 200 iterations is too short for method judgment. It is useful for link validation, but it can strongly misrepresent clean Mesh Splatting quality.

The origin/main 2000-iteration run is the first usable paper-baseline candidate, but it is still only medium-length. The next fair comparison should run current-branch engineering baseline and any MeshPrior variant at the same 2000-iteration budget with training-time W&B enabled.

## Gate

Stage gate: SOFT PASS.

The paper-baseline path is now concrete and logged to W&B, but current-branch medium baselines and MeshPrior variants are still needed for a fair comparison.

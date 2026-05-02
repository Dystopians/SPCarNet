# MeshPrior Stage25 Multidataset Validation Report

Date: 2026-05-02

Status: `SOFT PASS`

## Goal

Move beyond the single `parking_phone_tiny` scene by preparing public multiview datasets and running short, W&B-tracked trainability checks on representative scenes from Mip-NeRF 360, Tanks and Temples, and ETH3D.

## Data Sources And Space

- disk before/after setup: `/data` has `4.4T` available, `84%` used; downloaded/prepared M25 data occupies about `30G`.
- Mip-NeRF 360 source: `http://storage.googleapis.com/gresearch/refraw360/360_v2.zip` as referenced by public Mip-NeRF 360 dataset instructions.
- Tanks and Temples source attempted: `https://www.tanksandtemples.org/download`; the official downloader returned login/HTML payloads in this environment, so the usable data came from the NSVF mirror `https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip`.
- ETH3D source: `https://www.eth3d.net/datasets`; representative `courtyard` DSLR undistorted, scan, and eval archives were downloaded. The all-scene high-resolution archive was started but interrupted because the single full archive was slow and unnecessary for the first M25 trainability milestone.

## Local Dataset Layout

Dataset audit command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_stage25_dataset_audit.py
```

Audit output:

- JSON: `outputs/carnet/meshprior/stage25_multidataset/dataset_audit.json`
- entries: `12`
- trainable with current COLMAP loader: `10`

Trainable public scenes now available:

| dataset | scene(s) | path | note |
|---|---|---|---|
| Mip-NeRF 360 | `bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump` | `/data/peilincai/mesh_datasets/mipnerf360/<scene>` | Native current-loader-compatible COLMAP scenes. |
| ETH3D | `courtyard` | `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard` | Created loader view by symlinking images and copying ETH3D COLMAP text calibration into `sparse/0`. |
| Tanks and Temples | `truck`, `barn` | `/data/peilincai/mesh_datasets/tanks_and_temples_colmap/<scene>` | Converted from NSVF posed-image format with synthetic bootstrap points; trainable, but lacks true COLMAP 2D-3D tracks. |

New utility scripts:

- `scripts/car_model/meshprior_stage25_dataset_audit.py`
- `scripts/car_model/meshprior_convert_nsvf_to_colmap.py`

## Training Runs

All representative runs used `WANDB_MODE=online`, `--enable_wandb`, group `m25_multidataset`, GPU `1`, and `700` iterations. Commands are preserved under each run's `logs/train_command.txt`.

| scene | output | W&B | result |
|---|---|---|---|
| Mip-NeRF 360 `bonsai` | `outputs/carnet/meshprior/stage25_multidataset/mipnerf360_bonsai_700iter/model` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/x75zddff` | Completed. Test PSNR improves `17.2853 -> 20.1716`, SSIM `0.5920 -> 0.7247`, LPIPS `0.4395 -> 0.3105`; PRISM validation observable at iter `350`. |
| Tanks and Temples `truck` | `outputs/carnet/meshprior/stage25_multidataset/tanks_truck_700iter_fix/model` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5pre7o19` | Completed after fixing validation summary on non-observable geometry. W&B final `loss_image=0.12022`, topology `100000` triangles. Geometry validation is `no_sparse_matches`. |
| ETH3D `courtyard` | `outputs/carnet/meshprior/stage25_multidataset/eth3d_courtyard_700iter/model` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/78iu6goq` | Completed. Test PSNR improves `16.5933 -> 17.9631`, SSIM `0.5596 -> 0.6043`, LPIPS `0.5460 -> 0.5050`; PRISM validation observable at iter `350`. |

Failed/intermediate run kept for diagnosis:

- `outputs/carnet/meshprior/stage25_multidataset/tanks_truck_700iter/`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r9k94xlu`
- failure: validation markdown attempted to iterate `stage_best_metrics=None` when no sparse matches existed.

## Code Fix

`utils/prism_validation.py` now accepts `stage_best_metrics=None` in `save_validation_summary()` and writes:

```text
- status: not_initialized
```

This is required for posed-image datasets where geometry validation is not observable yet. It prevents a reporting crash without converting a failed geometry gate into a pass.

## Interpretation

M25 proves that the current method code is no longer tied to the tiny parking scene. It can load and train on two proper public COLMAP-style datasets immediately:

- Mip-NeRF 360 validates indoor object-centric novel-view scenes with official COLMAP layout.
- ETH3D validates high-resolution real photogrammetry data with real sparse geometry.

Tanks and Temples is only a partial validation in the current environment. The NSVF mirror has calibrated poses and images, so rendering training runs, but the synthetic `points3D.txt` has no image tracks. That means PRISM sparse geometry validation correctly reports `no_sparse_matches`. For paper-grade Tanks and Temples evidence we need either official scene data with usable reconstruction tracks, or we must run COLMAP/pycolmap locally to build real sparse observations.

## Gate

`SOFT PASS`.

Reasons:

- PASS: disk capacity is sufficient; all three requested dataset families have local data.
- PASS: Mip-NeRF 360 and ETH3D are trainable and geometry-observable.
- PASS: W&B was active for all representative training runs.
- PASS: non-observable geometry reporting no longer crashes.
- SOFT: Tanks and Temples is trainable but not yet geometry-validatable because the usable mirror is posed-image-only for our purposes.
- SOFT: ETH3D full all-scene archive was not fully downloaded; one representative scene is prepared and validated.

## Next Step

M26 should turn M25 from trainability evidence into method evidence:

1. Run the M24.2 topology-retention schedule and a plain current-branch baseline on at least `mipnerf360_bonsai` and `eth3d_courtyard` for a medium budget.
2. Add a dataset flag or validator mode that refuses to use sparse-geometry claims when COLMAP tracks are absent.
3. For Tanks and Temples, either acquire official reconstruction assets or install/run COLMAP to generate true `images.txt` observations and `points3D` tracks.
4. Generate a cross-scene table with render metrics, topology, PRISM commit counts, and geometry observability status.

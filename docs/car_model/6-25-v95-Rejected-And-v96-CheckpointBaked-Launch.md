# v95 Rejection and v96 Checkpoint-Baked Launch

Date: 2026-06-25

## Summary

v95 completed as a valid counter run, but it is **not promoted**. It accepted an atlas and changed target pixels, yet it failed the pre-declared v84/v86 counter anchor on all three held-out RGB metrics and failed the stronger risk-gain floors.

The next branch is v96 checkpoint-baked certified ELA recovery. This is a method-form change: instead of another region-texture atlas parameter scan, it trains a real checkpoint using train-only ELA teacher renders plus parent rollback and geometry/sparse-depth certificates.

## v95 Evidence

Command class:

```text
scripts/car_model/run_l1risk_fairnoop_scene.py --scene counter --tag candidate_counter_region_texture_adapter_v95 ...
```

Primary artifact paths:

```text
/dev/shm/peilincai_spcarnet_candidate_20260625_v95/counter_candidate_counter_region_texture_adapter_v95
outputs/carnet/spcarnet/v95_counter_region_texture_adapter_20260625
/dev/shm/wandb_spcarnet_candidate_v95/wandb/offline-run-20260625_011713-kye8wgi1
```

Held-out counter result:

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 anchor floor | `> 26.7561378479` | `> 0.8621263504` | `< 0.2516906559` |
| v95 candidate | `26.7500514984` | `0.8620513678` | `0.2519962788` |

Audit state:

| Field | Value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.03125` |
| target changed fraction | `0.0184769104` |
| target-tail certificate enabled | `true` |
| allowed keep bins | `768` |

Promotion gate verdict:

```text
REJECT PSNR_not_above_anchor, SSIM_not_above_anchor, LPIPS_not_below_anchor,
selected_alpha_not_0.5, selected_ssim_gain_below_v84_v86_anchor,
selected_ssim_min_view_gain_below_v84_v86_anchor,
selected_image_l1_gain_below_v84_v86_anchor,
selected_image_l1_cvar20_view_gain_below_v84_v86_anchor
```

Interpretation: v95 is a useful negative result. The target coverage gate worked, but the learned edit magnitude and policy-val gains are too small. It should not be expanded to `counter,kitchen,bonsai` or full9.

## v96 Launch

New runner:

```text
scripts/car_model/run_v96_checkpoint_baked_certified_repair_scene.py
```

The runner stages the compact checkpoint in `/dev/shm`, builds a train-only sparse-depth sentinel cache, then calls the strict topology-frozen recovery runner with:

- teacher render loss from train-only Phase-J/ELA renders;
- parent render rollback in `l1_dssim_edge` space;
- checkpoint render depth/normal anchors;
- sparse-depth parent rollback from train sentinels;
- standard render, metrics, and geometry evaluation after training.

Dry-run validation succeeded and wrote:

```text
/dev/shm/spcarnet_v96_dryrun_20260625/counter_v96_checkpoint_baked_certified_repair/v96_manifest.json
/dev/shm/spcarnet_v96_dryrun_20260625/counter_v96_checkpoint_baked_certified_repair/build_sparse_depth_sentinel_command.txt
/dev/shm/spcarnet_v96_dryrun_20260625/counter_v96_checkpoint_baked_certified_repair/strict_recovery_command.txt
```

Live counter probe command:

```text
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline WANDB_DIR=/dev/shm/wandb_spcarnet_v96 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_v96_checkpoint_baked_certified_repair_scene.py \
  --scene counter \
  --output_root /dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625 \
  --tag v96_checkpoint_baked_certified_repair \
  --wandb_project spcarnet_meshprior \
  --wandb_group v96_checkpoint_baked_certified_repair \
  --wandb_name counter_v96_checkpoint_baked_certified_repair_30000 \
  --wandb_mode offline \
  --wandb_dir /dev/shm/wandb_spcarnet_v96 \
  --force_staging \
  --execute
```

Initial v96 status:

```text
sentinel cache built: 24 train views, 12000 sentinels, no test leakage
training target: counter compact checkpoint 26000 -> 30000
```

## Claim Boundary

v95 should be reported only as a rejected representation-atlas diagnostic. v96 is the current active attempt to close the representation-level gap. Until v96 finishes and beats the anchor with normal checkpoint metrics plus geometry evaluation, the paper-safe endpoint remains Phase-J guarded adaptive ELA plus compactness.

# Stage35 Retained Relaxed Refresh Report

Date: 2026-05-02

## Status

`PASS`.

Stage35 adds conservative retained-edit control on top of Stage34 post-commit candidate refresh. The new controls are default-off and only activate when explicitly requested.

## Code Changes

- `arguments/__init__.py`
  - added `--prism_post_commit_relaxed_max_commits`
  - added `--prism_post_commit_relaxed_strict_gate`
  - added strict relaxed-gate thresholds for PSNR, sparse-depth MAE, AbsRel, normal angle, and changed-pixel ratio
- `train.py`
  - tracks retained relaxed candidate commits separately from total candidate rounds
  - blocks additional relaxed fallback commits when the retained relaxed cap is reached
  - records relaxed commit metadata and validation rollback metadata
  - writes `prism_debug/relaxed_retained_topology_audit.json`
  - logs retained/rolled-back relaxed topology audit scalars to W&B

Defaults preserve previous behavior: relaxed commit cap is `0`, strict relaxed gate is disabled, and Stage34 refresh remains opt-in.

## Experiments

### Mip-NeRF 360 `bonsai`

- dataset: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`
- output: `outputs/carnet/meshprior/stage35_retained_refresh/mipnerf360_bonsai_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter_retry1/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rszvl7gn`
- GPU: `CUDA_VISIBLE_DEVICES=4`

The first run found and fixed a runtime bug in relaxed commit recording (`NameError: t is not defined`). The retry completed.

PRISM decisions:

- normal candidate commit at iter `1501`: `634299 -> 633787`
- relaxed commits attempted at `1592`, `1683`, `1774`, `1865`, `1956`
- validation rolled back the first four relaxed commits
- final active relaxed commit: iter `1956`, `633787 -> 633275`

Retained topology audit:

- final triangles: `633275`
- active relaxed commits: `1`
- validation-rolled-back relaxed commits: `4`
- relaxed topology retained: `true`
- relaxed topology erased: `false`

Independent `render.py + metrics.py`:

| row | triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Stage33 reference | 633787 | 12.1999207 | 0.2765326 | 0.6125830 |
| Stage35 retained relaxed | 633275 | 12.2673674 | 0.2776170 | 0.6119390 |

Gate result: `PASS`. Stage35 lowers final topology and improves all independent metrics versus the Stage33 reference.

### ETH3D `courtyard`

- dataset: `/data/peilincai/mesh_datasets/eth3d_colmap/courtyard`
- output: `outputs/carnet/meshprior/stage35_retained_refresh/eth3d_courtyard_retained1_strict_relaxed_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/u2s15ok0`
- GPU: `CUDA_VISIBLE_DEVICES=4`

PRISM decisions:

- normal candidate commit at iter `1501`: `102937 -> 102425`
- retained relaxed commit at iter `1592`: `102425 -> 101913`
- later relaxed fallback attempts were blocked by `relaxed_commit_cap_reached`

Retained topology audit:

- final triangles: `101913`
- active relaxed commits: `1`
- validation-rolled-back relaxed commits: `0`
- relaxed topology retained: `true`
- relaxed topology erased: `false`

Independent `render.py + metrics.py`:

| row | triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Stage35 retained relaxed | 101913 | 15.3831606 | 0.5080911 | 0.5846940 |

This is a cross-scene mechanism check, not a direct quality win claim against every prior courtyard row. It confirms retained relaxed cap behavior transfers cleanly to a second geometry-observable public scene.

## Verification

Commands completed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile train.py arguments/__init__.py
git diff --check
```

Independent render/metric commands completed for both scenes. W&B online runs exist for both training runs.

## Decision

Stage35 is the first post-commit refresh variant that passes the strict retained-edit gate on `bonsai`: it keeps one additional relaxed topology edit in the final checkpoint while improving independent PSNR, SSIM, and LPIPS versus Stage33 at lower topology.

The main remaining risk is metric-path reconciliation. Training-time metrics are higher because they use the training evaluation path, while the paper-facing values should use independent `render.py + metrics.py`. Future tables must keep those paths separate.


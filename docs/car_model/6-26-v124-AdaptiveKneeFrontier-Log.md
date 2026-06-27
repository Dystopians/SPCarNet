# v124 Adaptive Knee Frontier Log

Date: 2026-06-26

## Motivation

v123 made the alpha policy adaptive, but it exposed an over-aggressive selection failure on `counter`.
The `smallest_effective` frontier required 75% of the best train policy-val relative/SSIM/L1 gains.
For `counter`, only `alpha=0.375` satisfied all three thresholded gains, so the method selected a larger
target edit than the held-out image metrics could safely absorb.

v124 keeps the midpoint alpha grid, but changes the adaptive policy from threshold-only frontier selection
to a train-policy-val knee detector:

- compute normalized multi-axis gain across positive policy-val axes;
- scan safe alphas in increasing order;
- stop at the first alpha whose score is useful and whose next marginal score slope drops sharply;
- record the full knee profile in the adapter audit.

This is still target/test-GT-free: the decision uses train policy-val rows and the stripped target footprint
only for candidate support/ranking, matching the vNext protocol.

## Code Change

Changed files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

New CLI:

- `--policy_val_alpha_frontier_mode knee`
- `--policy_val_alpha_frontier_knee_min_score_fraction`
- `--policy_val_alpha_frontier_knee_slope_drop_fraction`

Function-level replay on the v123 `counter` audit selected `alpha=0.1875` instead of v123's `0.375`:

```text
accepted True alpha 0.1875
mode risk_gate_alpha_midpoint_frontier
reason selected_knee_before_diminishing_returns
```

Compilation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

## Fair Replay Runs

Both replays used W&B offline logging and the same v123 inputs, except for
`--policy_val_alpha_frontier_mode knee`.

### counter

Run root:

```text
/dev/shm/peilincai_spcarnet_v124_counter_knee_frontier_20260626_200516/counter
```

W&B offline:

```text
/dev/shm/peilincai_wandb_v124_counter_20260626_200516/wandb/offline-run-20260626_202307-tq202lw6
```

Key evidence:

- manifest: `reports/counter_vnext_certified_residual_texture_manifest.json`
- results: `reports/counter_ours_26000_v124_knee_frontier_counter_test_results.json`
- audit: `model/surface_residual_region_texture_adapter_audit.json`
- protocol audit: passed
- selected alpha: `0.1875`
- frontier reason: `selected_knee_before_diminishing_returns`
- target changed fraction: `0.020591547716511543`

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v115/v106 anchor | 27.499700546 | 0.867478549 | 0.238779992 |
| v123 adaptive frontier | 27.499490738 | 0.867385030 | 0.238734275 |
| v122 fixed alpha1875 | 27.500209808 | 0.867498755 | 0.238754511 |
| v124 adaptive knee frontier | 27.500129700 | 0.867495179 | 0.238751680 |

Interpretation:

- v124 fixes the v123 PSNR/SSIM regression while preserving the LPIPS gain direction.
- v124 is better than the v115/v106 anchor on all three RGB metrics.
- v124 is still slightly below v122 on PSNR and SSIM, so it is not yet a strict replacement for the best
  fixed-policy run.

### flowers

Run root:

```text
/dev/shm/peilincai_spcarnet_v124_flowers_knee_frontier_20260626_200546/flowers
```

W&B offline:

```text
/dev/shm/peilincai_wandb_v124_flowers_20260626_200546/wandb/offline-run-20260626_201448-vbdpufks
```

Key evidence:

- manifest: `reports/flowers_vnext_certified_residual_texture_manifest.json`
- results: `reports/flowers_ours_26000_v124_knee_frontier_flowers_test_results.json`
- audit: `model/surface_residual_region_texture_adapter_audit.json`
- protocol audit: passed
- selected alpha: `0.0`
- frontier reason: `no_safe_rows`
- target changed fraction: `0.0`

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v123 adaptive frontier / anchor fallback | 20.452775955 | 0.549059212 | 0.355544209 |
| v124 adaptive knee frontier | 20.452775955 | 0.549059212 | 0.355544209 |

Interpretation:

- flowers remains a safe no-op.
- This is desirable for reliability but not sufficient for a paper-level visual improvement story.

## Current Assessment

v124 is a real method improvement over v123 because it converts the failed threshold-only adaptive alpha
selection into a conservative policy-val knee. It restores `counter` from the v123 PSNR/SSIM regression and
keeps the method GT-free and auditable.

However, v124 is not the final method:

- it is not strictly better than the best fixed-policy v122 row on `counter`;
- it does not improve `flowers`;
- it has not yet been replayed over full9;
- qualitative visibility is still not strong enough for the final paper story.

## Next Fix

The next candidate should add a tail-aware knee rule. The v123 audit showed that higher alpha can improve
aggregate train policy-val gains while worsening robust tail axes such as min-view/CVaR relative gain,
SSIM tail gain, and image-L1 tail gain. A better policy should stop before such tail regressions when the
lower alpha already preserves enough aggregate score.

This should remain train-only and should not use target/test GT.

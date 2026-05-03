# Stage34 PRISM Post-Commit Candidate Refresh Report

Date: 2026-05-02

## Summary

Stage34 adds an opt-in post-commit candidate refresh path for PRISM. The root cause of post-commit no-candidate rounds on `bonsai` is now measured: after a candidate commit, topology sync marks all surviving triangles as recent, and `recent_t` both protects all triangles and drives the normal prune score to zero through `risk_t`.

This stage is a `SOFT PASS / diagnostic PASS`. The mechanism works: the refreshed run finds and keeps additional low-risk edits on `bonsai`. It is not a hard M34 `PASS` because the best retained-topology run improves independent PSNR but slightly regresses independent SSIM/LPIPS against Stage33.

## Code Changes

- `arguments/__init__.py`
  - Added default-off flags:
    - `--prism_post_commit_candidate_refresh`
    - `--prism_post_commit_refresh_min_prune_score`
- `train.py`
  - Added post-commit candidate-refresh state and W&B/TensorBoard counters.
  - Added candidate-pool diagnostics for no-candidate rounds, including blockers from recent, render keep, geometry keep, protected masks, and relaxed-pool size.
  - Added a post-commit relaxed score that removes only the `recent_t` risk term while preserving uncertainty, boundary, nonmanifold, ground/ROI, geometry/orientation keep, and render keep risk.
  - Kept the counterfactual gate mandatory for refreshed candidates.

All new behavior is opt-in; defaults are unchanged.

## Runs

### Parking Smoke

- `parking_refresh_smoke_180iter`
  - W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rt3cxxhh`
  - Result: diagnostic no-candidate path; short smoke was blocked by recent protection.
- `parking_refresh_recent0_smoke_180iter`
  - W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kke60qhc`
  - Result: normal candidate path commits multiple edits, final topology `62449`; relaxed refresh is not triggered because ordinary candidates remain abundant.

### Mip-NeRF 360 `bonsai`

- Dataset: `/data/peilincai/mesh_datasets/mipnerf360/bonsai`
- Stage33 reference: final `633787` triangles, PSNR `12.1999207`, SSIM `0.2765326`, LPIPS `0.6125830`.

#### v1/v2 Root-Cause Runs

- v1 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/szkqpowq`
- v2 W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/npagb743`
- Finding: after the first commit, every later no-candidate round had `block_recent=633787`, `candidate_pool_count=0`, and `relaxed_pool_count=0` when using the original prune score. The missing piece was that `recent_t` also zeroed `prune_score_t` through `risk_t`.

#### v3 Relaxed-Score Run

- Output: `outputs/carnet/meshprior/stage34_post_commit_refresh/mipnerf360_bonsai_refresh_v3_relaxed_score_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter/model`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/lt1v4652`
- Calibration manifest: `24` views (`12` diverse test, `4` diverse train, `8` hard train).
- PRISM decisions:
  - iter `1501`: normal candidate commit, `634299 -> 633787`
  - iter `1592`: relaxed refresh commit, `633787 -> 633275`
  - iter `1683`: relaxed refresh commit, `633275 -> 632763`
  - iter `1774`: relaxed refresh commit, `632763 -> 632251`
  - iter `1956`: relaxed refresh commit, `632251 -> 631739`
- Final checkpoint topology: `631739` triangles.
- Independent `render.py + metrics.py`: PSNR `12.2019978`, SSIM `0.2757282`, LPIPS `0.6129612`.

Comparison:

| row | final triangles | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Stage33 diverse calibration | `633787` | `12.1999207` | `0.2765326` | `0.6125830` |
| Stage34 v3 relaxed refresh | `631739` | `12.2019978` | `0.2757282` | `0.6129612` |

v3 proves the second-stage search works and keeps additional topology reduction, but it trades tiny SSIM/LPIPS regressions for a tiny PSNR gain.

#### v4 Second-Edit-Only Diagnostic

- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/zhy368pr`
- Result: iter `1592` logs a relaxed commit `633787 -> 633275`, but final checkpoint returns to `633787`. This run is diagnostic only; it does not retain the second edit.

## Decision

`SOFT PASS / diagnostic PASS`.

Stage34 should stay opt-in. It fixes the main no-candidate blind spot and shows that post-commit relaxed discovery can find additional low-risk edits under the same counterfactual gate. It is not promoted as a default schedule because the retained extra edits on `bonsai` are not strictly non-regressing across PSNR, SSIM, and LPIPS.

## Next Step

M35 should add a conservative retained-edit controller:

1. cap the number of post-commit relaxed edits separately from normal candidate edits;
2. add an independent-metric or held-out-view rollback signal after relaxed commits, not only local calibration acceptance;
3. preserve the relaxed-score diagnostic fields;
4. rerun `bonsai` with one retained relaxed edit and then `courtyard` only if all independent `bonsai` metrics match or improve Stage33.

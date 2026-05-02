# Stage24.1 Late-PRISM Pareto Sweep Report

Date: 2026-05-02

## Gate

`PASS`.

M24.1 found a stronger integrated training-time topology-control row than M24-v3 on `parking_phone_tiny`. The best topology/quality point is `pareto_ratio0p005_rounds8_retryfix_7000iter`: it preserves near-baseline independent render quality, improves the COLMAP normal proxy, and ends with `723438` triangles, materially below the M24-v3 `823651` triangles and current-branch `833775` triangles.

This is still not a final paper headline because it does not match the M21.5 posthoc `prune_50` topology budget (`416888` triangles). It is, however, the best integrated optimization-time evidence row so far.

## Code Changes

- `train.py`: PRISM candidate pruning now returns `no_candidates: 1` when all candidates are protected or blocked.
- `utils/prism_pipeline.py`: no-candidate attempts no longer consume a candidate round; the controller waits `prism_no_candidate_retry_iters` before retrying.
- `arguments/__init__.py`: added `--prism_no_candidate_retry_iters`, default `10`.
- `scripts/car_model/meshprior_collect_stage23_5_integrated_topology.py`: collector now excludes no-candidate retry events from effective PRISM round counts and reports retry counts separately.

## Runs

All runs used online W&B project `spcarnet_meshprior`, group `parking_stage24_1_late_prism_pareto`, GPU `1`, final cleanup disabled, counterfactual gate enabled, and independent post-evaluation.

| run | W&B | ratio | effective rounds | retry events | commits | triangles | vertices | PSNR | SSIM | LPIPS | depth AbsRel | normal mean deg | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pareto_ratio0p005_rounds8_7000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bqc4w18e` | `0.005` | `1` | legacy | `1` | `827438` | `1064193` | `17.107386` | `0.530902` | `0.456165` | `0.084803` | `43.258999` | diagnostic |
| `pareto_ratio0p005_rounds8_retryfix_7000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jnn9yauw` | `0.005` | `5` | `445` | `5` | `723438` | `904493` | `16.967005` | `0.530894` | `0.465932` | `0.082264` | `42.667905` | `PASS` |
| `pareto_ratio0p01_rounds8_retryfix_7000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/0n7kzim5` | `0.01` | `8` | `27` | `2` | `820117` | `1052590` | `17.064436` | `0.530386` | `0.455843` | `0.082148` | `42.522107` | `PASS` |

## Best Row

`pareto_ratio0p005_rounds8_retryfix_7000iter`.

It commits five accepted PRISM candidate edits and ends at `723438` triangles. Relative to current branch 7000, it removes about `13.2%` of final triangles while preserving very similar SSIM and improving normal proxy geometry:

- current branch 7000: PSNR `17.204679`, SSIM `0.535045`, LPIPS `0.450750`, depth AbsRel `0.076126`, normal `45.561976`, triangles `833775`
- M24-v3: PSNR `17.042757`, SSIM `0.529476`, LPIPS `0.454884`, depth AbsRel `0.082815`, normal `43.394721`, triangles `823651`
- M24.1 best: PSNR `16.967005`, SSIM `0.530894`, LPIPS `0.465932`, depth AbsRel `0.082264`, normal `42.667905`, triangles `723438`
- M21.5 posthoc `prune_50`: PSNR `17.051889`, SSIM `0.523914`, LPIPS `0.465400`, depth AbsRel `0.083265`, normal `45.825681`, triangles `416888`

The best M24.1 row is therefore stronger than M24-v3 as an integrated topology-control method row, but it is not yet as topology-efficient as the posthoc diagnostic.

## Diagnostics

The first `0.005` run exposed a controller issue: no-candidate attempts were consuming candidate rounds. This happened after a successful commit because `recent_age_iters=100` temporarily protected all available triangles. The retryfix changes corrected the semantics so no-candidate events are logged and retried without spending effective PRISM rounds.

The `0.005` retryfix row then exposed the opposite practical issue: without throttling, the controller can retry every iteration while recent-protection is active. `prism_no_candidate_retry_iters=10` was added to prevent log spam and wasted gate attempts. The `0.01` retryfix row used this throttle and had only `27` retry events.

The remaining method issue is topology retention. In both retryfix rows, late training densification can restore part of the topology after accepted PRISM edits. That explains why W&B summary topology and final checkpoint topology can differ from individual post-prune counts. The next prompt should test a post-PRISM topology-retention phase or a compaction-source checkpoint selection rule.

## Artifacts

- design: `docs/car_model/meshprior_stage24_1_late_prism_pareto_design.md`
- report: `docs/car_model/meshprior_stage24_1_late_prism_pareto_report.md`
- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage24_1_late_prism_pareto/`
- best summary: `outputs/carnet/meshprior/parking_phone_tiny/stage24_1_late_prism_pareto/pareto_ratio0p005_rounds8_retryfix_7000iter/summary/stage23_5_integrated_topology_summary.json`
- best model: `outputs/carnet/meshprior/parking_phone_tiny/stage24_1_late_prism_pareto/pareto_ratio0p005_rounds8_retryfix_7000iter/model`

## Verification

- `python -m compileall train.py utils/prism_pipeline.py arguments/__init__.py scripts/car_model ss3dm_prior -q`
- PRISM controller no-candidate retry smoke assertion
- `conda run -n VGGT python scripts/car_model/smoke_test_meshprior_stage23_5_integrated_topology_collector.py`
- `git diff --check`

## Next

M24.2 has now validated the first option: freeze densification after the first accepted PRISM candidate commit. See `docs/car_model/meshprior_stage24_2_topology_retention_report.md`.

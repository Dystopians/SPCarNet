# MeshPrior Stage 21.5 Topology-Controlled Current-Branch Ablation Report

Date: 2026-05-02

## Verdict

Gate: `PASS`.

Area-based post-training checkpoint-copy pruning gives a strong single-scene topology-control diagnostic. The best tradeoff is `prune_50`: it removes half the current-branch triangles, keeps render metrics above the clean 7000 baseline, and keeps COLMAP proxy depth AbsRel close to clean.

This is not a final paper claim because it is still one scene and a post-hoc checkpoint-copy ablation, but it fixes the immediate M21 concern that current-branch quality only exists at 83万 triangles.

## W&B

| run | W&B |
|---|---|
| prune_25 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/evid1gbt |
| prune_50 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w1ix6e9a |
| prune_66 | https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xzfqwpgi |

## Results

Independent render metrics, all at iteration 7000:

| row | PSNR | SSIM | LPIPS | triangles | depth AbsRel |
|---|---:|---:|---:|---:|---:|
| clean `origin/main` | 16.134155 | 0.452130 | 0.499124 | 285187 | 0.084499 |
| current branch | 17.204679 | 0.535045 | 0.450750 | 833775 | 0.076126 |
| prune_25 | 17.243082 | 0.535973 | 0.451929 | 625331 | 0.076709 |
| prune_50 | 17.051889 | 0.523914 | 0.465400 | 416888 | 0.083265 |
| prune_66 | 16.429369 | 0.492480 | 0.489681 | 283484 | 0.099246 |

`prune_50` is the current recommended row for M22: it uses `0.50x` current-branch triangles and `1.46x` clean triangles while keeping all render metrics better than clean and depth AbsRel slightly better than clean.

`prune_66` nearly matches clean triangle count (`0.99x`) and still beats clean render metrics, but its depth AbsRel regresses to `0.099246`; keep it as a useful Pareto endpoint, not the default.

## Files

- `scripts/car_model/meshprior_apply_topology_control_ablation.py`
- `scripts/car_model/smoke_test_meshprior_topology_control_ablation.py`
- `scripts/car_model/meshprior_collect_topology_control_ablation.py`
- `scripts/car_model/smoke_test_meshprior_topology_control_comparison.py`
- `docs/car_model/meshprior_stage21_5_topology_control_design.md`
- `outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/`

## Smoke And Verification

- `scripts/car_model/smoke_test_meshprior_topology_control_ablation.py`: PASS
- `scripts/car_model/smoke_test_meshprior_topology_control_comparison.py`: PASS
- `python -m compileall scripts/car_model ss3dm_prior -q`: PASS

## Decision

Proceed to M22 with `prune_50` as the topology-controlled current-branch row, `prune_66` as a high-compression Pareto endpoint, and Stage17 MeshPrior resume as a long-budget failure case.


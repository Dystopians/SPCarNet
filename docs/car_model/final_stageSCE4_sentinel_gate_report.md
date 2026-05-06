# Final Stage SCE4 - Sentinel Parent-Pareto Gate Report

Date: 2026-05-06

Decision: `PASS`

## Implementation

Files added:

- `utils/sentinel_parent_pareto_gate.py`
- `scripts/car_model/meshsplatopt_sentinel_parent_pareto_gate.py`
- `scripts/car_model/smoke_test_stageSCE4_sentinel_gate.py`
- `docs/car_model/final_stageSCE4_sentinel_gate_design.md`

The gate renders parent and candidate checkpoints on the views represented in a SCE2 sentinel cache, samples both models at the cached sparse correspondence pixels, and reports sentinel-level parent-Pareto checks before full recovery/evaluation runs.

## Smoke Test

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_stageSCE4_sentinel_gate.py
```

Result: `SCE4 sentinel parent-pareto gate smoke test PASS`.

The smoke verifies synthetic pass/fail behavior for mean sentinel AbsRel / Depth MAE non-regression.

## Real Gate: Courtyard F82 vs F95 on SCE2 Train Sentinels

Command:

```bash
CUDA_VISIBLE_DEVICES=7 /home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_sentinel_parent_pareto_gate.py \
  --source_path /data/peilincai/mesh_datasets/eth3d_colmap/courtyard \
  --images images \
  --resolution 4 \
  --eval \
  --parent_model_path outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/courtyard/adaptive_global_policy_v5_seed0/recovery_model \
  --parent_iteration 26000 \
  --candidate_model_path outputs/carnet/meshsplatopt/final_stageF95_render_geometry_anchor_repair/courtyard/adaptive_global_policy_v5_teacher0p001_sparse0p001_rendergeom0p01_27000_seed0/recovery_model \
  --candidate_iteration 27000 \
  --sentinel_cache outputs/carnet/meshsplatopt/final_stageSCE2_sentinel_cache/courtyard/sentinel_cache.npz \
  --output_dir outputs/carnet/meshsplatopt/final_stageSCE4_sentinel_gate/courtyard_f82_vs_f95 \
  --allow_fail_exit_zero
```

Outputs:

- `sentinel_parent_pareto_gate.json`
- `sentinel_per_view_summary.csv`
- `sentinel_cluster_summary.csv`
- `sentinel_gate_report.md`

Result:

| metric | F82 parent | F95 candidate | candidate - parent |
|---|---:|---:|---:|
| Sentinel AbsRel | 0.385219713 | 0.386472855 | +0.001253142 |
| Sentinel Depth MAE | 4.806230962 | 4.833562309 | +0.027331347 |

The gate correctly fails F95 on the train sentinel cache:

- decision: `FAIL_SENTINEL_PARENT_PARETO`
- both-valid sentinels: `13564`
- candidate-invalid sentinels: `66`
- gate-critical sentinels: `636`
- regressed AbsRel/MAE sentinels: `4985`
- worst-view regression count: `208`
- cluster failures: `493`

## Gate

SCE4 passes as an implementation stage because synthetic pass/fail behavior is correct and the real train-cache gate runs on local courtyard artifacts. The F95 candidate itself fails the gate, which is expected and useful: it proves SCE4 can catch the sparse-depth regression before another expensive full recovery is accepted.

The next stage is SCE5/F97 diagnostic packaging, followed by SCE6 targeted rollback recovery runs using the SCE2 cache and SCE3 loss.


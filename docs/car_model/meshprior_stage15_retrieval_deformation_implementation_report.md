# MeshPrior Stage 15 Retrieval-Deformation Implementation Report

Date: 2026-05-01

## Scope

M15 implements a retrieval-deformation fallback for MeshPrior proposal generation. It is a measured alternative to learned implicit-field proposals, not a replacement adopted by default.

## Files Added

- `ss3dm_prior/meshprior/retrieval_deformation.py`
- `scripts/car_model/meshprior_build_anchor_bank.py`
- `scripts/car_model/meshprior_eval_retrieval_deformation.py`
- `scripts/car_model/smoke_test_meshprior_stage15_retrieval_deformation.py`
- `docs/car_model/meshprior_stage15_retrieval_deformation_design.md`

## Implementation

### Anchor Bank

The anchor bank stores fixed-size canonical point anchors:

- `object_id`;
- `split`;
- sampled clean points;
- metadata.

Leakage controls:

- only `split=train` records are accepted by the builder;
- the loader refuses any bank containing non-train splits;
- retrieval excludes a matching `query_object_id` when provided.

### Retrieval-Only Baseline

Retrieval-only selects the nearest train anchor by symmetric Chamfer L1 between normalized observed points and normalized anchor points. It records:

- best anchor;
- best score;
- second-best score;
- retrieval margin;
- uncertainty;
- mean nearest distance.

### Deformation

The deformation path is deliberately conservative. It moves anchor points a clipped fraction toward nearest observed points and reports displacement diagnostics. It is not a neural deformation model.

### Proposal Export

The retrieval fallback exports MeshPrior-compatible proposal types:

- `protect`;
- `prune`;
- `snap`;
- `fill_candidate`;
- `uncertainty`.

Empty protect/prune proposals are still emitted with zero confidence so downstream proposal-type contracts remain stable.

## Evaluation

Commands:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_build_anchor_bank.py --object_index outputs/carnet/spcarnet/object_index_v1.json --output outputs/carnet/meshprior/retrieval_deformation/stage15_anchor_bank.npz --max_anchors 32 --points_per_anchor 512 --seed 0
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshprior_eval_retrieval_deformation.py --anchor_bank outputs/carnet/meshprior/retrieval_deformation/stage15_anchor_bank.npz --output_dir outputs/carnet/meshprior/retrieval_deformation/stage15_eval --damage_types local_hole floater vertex_noise density_imbalance
```

Outputs:

- `outputs/carnet/meshprior/retrieval_deformation/stage15_anchor_bank.npz`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_anchor_bank.summary.json`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/metrics.json`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/metrics.csv`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/summary.md`

Summary:

- anchor count: `32`
- points per anchor: `512`
- train-only: `true`
- evaluation rows: `12`
- recommendation: `KEEP_AS_BASELINE`

## Result Interpretation

Retrieval-only did not beat the Stage 3 posterior proxy on the synthetic proposal metrics.

Key observations:

- Stage3 posterior proxy kept valid-surface protect recall at `1.0` for local hole, floater, and density imbalance, and `0.9166666666666666` for vertex noise.
- Retrieval-only produced floater recall `1.0` on the floater case, but low precision (`0.07692307692307693`) and poor valid-surface protect recall in this benchmark.
- Conservative deformation did not improve the recommendation and should not be promoted to a neural deformation path yet.

## Decision

M15 gate: `PASS`.

Recommendation: `KEEP_AS_BASELINE`.

Do not pivot to retrieval-deformation based on current evidence. Keep it as a baseline/fallback row and continue prioritizing real scene proposal application and scene evidence.

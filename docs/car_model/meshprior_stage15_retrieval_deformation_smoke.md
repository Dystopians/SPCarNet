# MeshPrior Stage 15 Retrieval-Deformation Smoke

Date: 2026-05-01

## Commands

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_stage15_retrieval_deformation.py
```

## Result

`smoke_test_meshprior_stage15_retrieval_deformation.py`: `PASS`

Smoke output:

- synthetic train-only anchor bank built;
- `rows=9`;
- recommendation: `KEEP_AS_BASELINE`;
- proposal types include `protect`, `prune`, `snap`, `fill_candidate`, and `uncertainty`.

## Full M15 Evaluation

Train-only anchor bank:

- source: `outputs/carnet/spcarnet/object_index_v1.json`;
- anchors: `32`;
- points per anchor: `512`;
- train-only: `true`.

Evaluation:

- rows: `12`;
- recommendation: `KEEP_AS_BASELINE`.

## Gate

M15 smoke gate: `PASS`.

Retrieval-only was measured before deformation. It did not beat the Stage 3 posterior proxy, so no pivot is recommended.

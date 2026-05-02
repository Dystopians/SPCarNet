# MeshPrior Scene Application Loop Smoke

Date: 2026-05-01

## Commands

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshprior_scene_application.py
```

## Result

`smoke_test_meshprior_scene_application.py`: `PASS`

Smoke output:

- M10 synthetic pipeline status: `PASS`
- accepted proposals: `1`
- applied proposals: `1`
- initial mesh: `8` vertices, `10` faces
- final mesh: `9` vertices, `14` faces
- rollback written
- recovery command plan written

## Gate

Scene application smoke gate: `PASS`.

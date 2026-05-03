# MeshSplatOpt Stage R4 Defect Mining Implementation Report

Date: 2026-05-02

## Gate

`PASS`.

R4 converts CSEF region diagnostics into auditable defect records and distinguishes certified giant ground voids from unknown/unobserved voids.

## Files Added

- `ss3dm_prior/meshsplatopt/defect_types.py`
- `ss3dm_prior/meshsplatopt/defect_mining.py`
- `scripts/car_model/meshsplatopt_mine_defects.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR4_defect_mining.py`

Updated:

- `ss3dm_prior/meshsplatopt/__init__.py`

## Behavior

The miner reads `csef_regions.json` or an in-memory `CSEFBuildResult` and emits `DefectRecord` entries with:

- defect id and type;
- severity and confidence;
- affected faces;
- boundary loops;
- candidate edit types;
- evidence summary;
- uncertainty summary;
- no-repair reason when applicable.

Implemented R4 mining paths:

- small component to `FLOATER_COMPONENT`;
- boundary-supported large area to `GIANT_GROUND_VOID`;
- boundary-supported smaller area to `SMALL_BOUNDARY_HOLE`;
- weak/no-boundary coverage hints to `UNKNOWN_UNOBSERVED_VOID`.

Other defect types are present in the enum contract and will be activated by later snap, object-prior, and appearance evidence stages.

## Artifacts

Smoke artifacts:

- `outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/csef/`
- `outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/defects/defects.json`
- `outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/defects/defects_summary.csv`
- `outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/defects/defect_mining_report.md`
- `outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/stageR4_defect_mining_smoke_report.json`

CLI check:

- `outputs/carnet/meshsplatopt/stageR4_defect_mining_cli_check/`

## Verification

Commands:

```bash
python -m compileall scripts/car_model ss3dm_prior utils -q
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR4_defect_mining.py
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_mine_defects.py \
  --csef_regions_json outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/csef/csef_regions.json \
  --output_dir outputs/carnet/meshsplatopt/stageR4_defect_mining_cli_check \
  --giant_area_threshold 12.0
```

Smoke result:

```json
{
  "status": "PASS",
  "defect_types": ["GIANT_GROUND_VOID", "UNKNOWN_UNOBSERVED_VOID"],
  "defect_count": 2
}
```

## Decision

`PASS`. R5 can build reversible edit records against the defect contract.

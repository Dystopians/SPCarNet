# vNext Manifest Runner Summary

- status: `MISSING_INPUT`
- config: `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json`
- output root: `/dev/shm/peilincai_spcarnet_vnext_full9_gap_preflight_after_flowers_20260626`
- preflight-only: `True`
- max parallel: `1`
- ready scenes: `7`
- missing-input scenes: `2`
- failed scenes: `0`

| scene | status | ready | returncode | elapsed sec | gpu | log |
|---|---|---:|---:|---:|---:|---|
| bicycle | MISSING_INPUT | False |  | 0.000 | 2 | `` |
| bonsai | READY | True | 0 | 0.000 | 3 | `` |
| counter | READY | True | 0 | 0.000 | 2 | `` |
| flowers | READY | True | 0 | 0.000 | 3 | `` |
| garden | READY | True | 0 | 0.000 | 3 | `` |
| kitchen | MISSING_INPUT | False |  | 0.000 | 2 | `` |
| room | READY | True | 0 | 0.000 | 2 | `` |
| stump | READY | True | 0 | 0.000 | 3 | `` |
| treehill | READY | True | 0 | 0.000 | 2 | `` |

## Missing Inputs

- `bicycle`: fit_evidence_dir, target_evidence_dir, region_carrier_json
- `kitchen`: fit_evidence_dir, target_evidence_dir, region_carrier_json

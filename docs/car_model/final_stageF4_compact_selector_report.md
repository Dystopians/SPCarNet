# Final Stage F4 Compact Selector Report

Date: 2026-05-04

## Decision

`PASS`.

The selector now has non-area modes, emits the required candidate/score/report artifacts, and differs from area-only on synthetic data while protecting high-debt repair regions.

## Smoke Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_final_stageF4_compact_selector.py
```

Output:

```text
F4 selector smoke PASS: area=[2, 3, 6] csef=[2, 3, 7] random=[2, 3, 7]
```

## Synthetic Scene

The smoke mesh contains:

- supported surface faces;
- redundant small triangles;
- a protected hole rim;
- a high-debt repair region;
- a floater;
- a large protected ground patch.

## Verification

| requirement | status |
| --- | --- |
| non-area mode exists | pass: `csef_low_evidence`, `csef_low_evidence_boundary_protected`, `pareto_area_csef` |
| candidate choices differ from area-only | pass: area selects `[2, 3, 6]`, CSEF boundary-protected selects `[2, 3, 7]` |
| redundant small triangles selected | pass: faces `2,3` |
| floater selected by CSEF | pass: face `7` |
| high-debt repair region protected | pass: face `6` is not selected by boundary-protected CSEF |
| hole rim and large ground patch protected | pass: protected labels are absent from selected CSEF faces |
| random control selects same count | pass |

## Artifacts

```text
outputs/carnet/meshsplatopt/final_stageF4_selector_smoke/area_smallest/compaction_candidates.json
outputs/carnet/meshsplatopt/final_stageF4_selector_smoke/csef_low_evidence_boundary_protected/compaction_candidates.json
outputs/carnet/meshsplatopt/final_stageF4_selector_smoke/pareto_area_csef/compaction_candidates.json
outputs/carnet/meshsplatopt/final_stageF4_selector_smoke/random_same_count/compaction_candidates.json
```

## Next Step

Proceed to F5: apply selector candidates to real Mesh Splatting checkpoints and validate that the compact checkpoint remains loadable and renderable.

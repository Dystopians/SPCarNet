# Full9 Paper-Loop Evidence Status

This report is generated mechanically from existing clean MeshSplatting, Phase-J, and Phase-S artifacts. It is a status collector, not a new experiment, and it treats missing rows as explicit evidence gaps.

## Summary

- Clean-best rows: `9 / 9`
- Phase-J full9 rows: `9 / 9`
- Phase-J strict RGB wins vs recorded clean baseline: `9 / 9`
- Phase-J strict RGB wins vs clean-best row selected here: `9 / 9`
- Phase-S single-gate decisions: `9 / 9`; accepted `6 / 9`
- Phase-S strict four-offset gates: `7 / 9`; accepted `6 / 9`; rejected `1 / 9`
- Phase-S strict all-axis train-val wins: `3 / 7`
- Missing evidence entries: `2`
- Full9 clean/Phase-J/Phase-S closure: `False`

## Scene Status

| scene | clean-best | Phase-J vs clean-best | Phase-J strict | Phase-S single | Phase-S strict | Phase-S strict mean delta | missing evidence |
|---|---|---:|---|---|---|---:|---|
| bicycle | ours_26000 (23.302/0.660/0.332) | 0.719931 / 0.042489 / -0.065989 | yes | reject | reject | 0.000143 / 0.000001 / 0.000023 | none |
| flowers | ours_26000 (19.682/0.512/0.395) | 0.622101 / 0.045948 / -0.065341 | yes | accept | accept | 0.000030 / 0.000000 / 0.000000 | none |
| garden | ours_26000 (25.029/0.780/0.201) | 1.281900 / 0.047808 / -0.065472 | yes | accept | accept | 0.000520 / 0.000017 / -0.000082 | none |
| stump | ours_26000 (25.205/0.705/0.294) | 0.390062 / 0.018909 / -0.030095 | yes | accept | accept | 0.000001 / -0.000000 / -0.000000 | none |
| treehill | ours_26000 (20.934/0.565/0.406) | 0.362045 / 0.031083 / -0.069725 | yes | reject | missing | n/a / n/a / n/a | phase_s_strict_four_offset |
| room | ours_26000 (28.747/0.885/0.250) | 1.558363 / 0.020887 / -0.053913 | yes | accept | accept | 0.000051 / 0.000000 / -0.000000 | none |
| counter | ours_26000 (26.752/0.862/0.252) | 1.697397 / 0.031675 / -0.065531 | yes | reject | missing | n/a / n/a / n/a | phase_s_strict_four_offset |
| kitchen | ours_26000 (27.819/0.876/0.199) | 2.381180 / 0.039635 / -0.067231 | yes | accept | accept | 0.000072 / 0.000000 / -0.000001 | none |
| bonsai | ours_26000 (28.895/0.896/0.259) | 2.966772 / 0.033879 / -0.086937 | yes | accept | accept | 0.000156 / -0.000001 / 0.000020 | none |

## Missing Rows

| scene | evidence | path | reason |
|---|---|---|---|
| treehill | phase_s_strict_four_offset | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/treehill/multifold_trainval_gate.json` | strict four-offset train-val gate JSON missing |
| counter | phase_s_strict_four_offset | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/counter/multifold_trainval_gate.json` | strict four-offset train-val gate JSON missing |

## Reading

- `clean-best` is selected from the configured clean methods by `PSNR + 20 * SSIM - 20 * LPIPS`; this exposes when Phase-J used a different recorded clean row.
- Phase-J is checked against both its recorded clean baseline and the clean-best row selected by this collector.
- Phase-S closure is intentionally stricter than single-gate acceptance: a paper-loop row is not closed unless the strict four-offset train-val gate exists and accepts.
- Held-out Phase-S test deltas are report-only and are not used to decide acceptance.

## Artifacts

- summary JSON: `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_paper_loop_status.json`
- scene CSV: `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_paper_loop_status.csv`
- clean candidates CSV: `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_clean_candidate_rows.csv`
- missing rows CSV: `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_missing_rows.csv`

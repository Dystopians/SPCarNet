# MeshPrior Stage 4 Protect/Prune — Implementation Report

| Field | Value |
|---|---|
| Stage | M4 / protect-prune proposals |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage4_protect_prune_design.md` |

## 1. Files Added

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/proposals.py` | Proposal dataclasses: `MeshPriorProposal`, `TriangleScoreTable`, `ProposalBatch`. |
| `ss3dm_prior/meshprior/protect_prune.py` | Triangle sampling, shape-field support, score computation, proposal assembly. |
| `scripts/car_model/meshprior_make_protect_prune_proposals.py` | CLI for creating proposal artifacts from M2/M3 outputs. |
| `scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py` | Synthetic protect/prune smoke test. |
| `docs/car_model/meshprior_stage4_protect_prune_design.md` | Stage design. |

## 2. Implementation Summary

M4 computes triangle-level scores:

```text
surface_support = mean(sigmoid(f(samples; z)))
prior_violation = 1 - surface_support
protect_score = surface_support * observed_support * (1 - uncertainty_penalty)
prune_score = clamp(prior_violation + free_space_violation + low_observed_support - protect_score, 0, 1)
```

Current behavior:

- Supports a real `SPCarShapeFieldDecoder` plus region posterior `z`.
- Supports analytic callable fields for smoke tests.
- Uses deterministic barycentric triangle samples.
- Emits protect/prune proposals only.
- Does not move vertices.
- Does not add or fill geometry.

CLI outputs:

```text
triangle_scores.npz
proposals.json
summary.csv
```

`triangle_scores.npz` stores rows with:

```text
region_id, face_index, protect, prune, support, violation
```

## 3. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py
```

Smoke output:

```text
cube_protect: 0.9999899864196777
floater_protect: 0.00000999999883788405
cube_prune: 0.0
floater_prune: 0.9999799728393555
proposal_types: protect, prune
```

Interpretation:

- Synthetic cube surface receives high protect score.
- Synthetic far-away floater receives high prune score.
- Both protect and prune proposal types are generated.

## 4. Inference-Time / Oracle Separation

M4 uses only:

- scene triangle samples,
- SP-CarNet field support,
- posterior uncertainty,
- optional free-space / observed support hooks.

It does not use clean object ground truth to choose proposals.

## 5. Known Limitations

- Free-space violation is a hook but not yet active unless later stages provide per-triangle free-space evidence.
- Observed support currently defaults to `1.0`; later scene gates must add real evidence.
- Visual debug PLY export is not implemented in this first pass; the neutral NPZ/JSON/CSV contract is the canonical output.
- CLI path is implemented and compile-checked; the smoke currently verifies scoring directly with an analytic field rather than a full M2/M3/M4 filesystem pipeline.

## 6. Stage Gate

| Gate | Result |
|---|---|
| Smoke test passes | PASS |
| Valid synthetic surface gets higher protect score | PASS |
| Synthetic floater gets higher prune score | PASS |
| No vertex movement | PASS |
| No hole filling | PASS |
| Score contract exists for downstream optimizer | PASS |

Decision: `PASS`. The next allowed stage is M5 optimizer adapter.

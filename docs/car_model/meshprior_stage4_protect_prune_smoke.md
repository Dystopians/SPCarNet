# MeshPrior Stage 4 Protect/Prune — Smoke Report

| Field | Value |
|---|---|
| Stage | M4 / protect-prune smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Command

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py
```

## Synthetic Setup

The smoke test builds:

- a cube mesh centered in the canonical frame,
- one far-away floater triangle,
- an analytic box-support field.

No SP-CarNet checkpoint or clean target is used.

## Output Summary

```text
cube_protect: 0.9999899864196777
floater_protect: 0.00000999999883788405
cube_prune: 0.0
floater_prune: 0.9999799728393555
proposal_types: protect, prune
```

## Gate Verdict

`PASS`

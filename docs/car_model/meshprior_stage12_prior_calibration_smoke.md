# MeshPrior Stage 12 Prior Calibration — Smoke Report

| Field | Value |
|---|---|
| Stage | M12 / prior calibration smoke |
| Date | 2026-05-01 |
| Result | PASS |

## Command

```bash
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage12_prior_calibration.py
```

## Output Summary

Synthetic damage:

```text
vertex_noise
```

| Profile | Max disp | Baseline protect recall | Snapped protect recall | Surface-distance delta | Free-space delta |
|---|---:|---:|---:|---:|---:|
| `none` | `0.02` | `0.9166666666666666` | `0.8333333333333334` | `0.021305546164512634` | `0.0` |
| `surface_support_v1` | `0.005` | `0.9166666666666666` | `0.9166666666666666` | `0.01073157787322998` | `0.0` |

The calibrated profile:

- improved valid-surface protect recall versus the uncalibrated snap result;
- preserved baseline protect recall;
- still improved surface distance;
- did not increase free-space violation.

## Gate Verdict

`PASS`

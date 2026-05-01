# MeshPrior Stage 12 Prior Calibration — Implementation Report

| Field | Value |
|---|---|
| Stage | M12 / prior calibration |
| Date | 2026-05-01 |
| Status | PASS |
| Design | `docs/car_model/meshprior_stage12_prior_calibration_design.md` |

## 1. Files Added or Updated

| File | Role |
|---|---|
| `ss3dm_prior/meshprior/calibration.py` | Post-hoc surface-support calibration profiles and snap-risk comparison utilities. |
| `scripts/car_model/meshprior_calibrate_prior.py` | Targeted calibration experiment runner. |
| `scripts/car_model/smoke_test_meshprior_stage12_prior_calibration.py` | Stage-12 smoke test. |
| `scripts/car_model/meshprior_run_pipeline.py` | Adds `--calibration_profile`, used by snap proposals. |
| `docs/car_model/meshprior_stage12_prior_calibration_design.md` | Stage design. |

## 2. Chosen Upgrade

Implemented:

```text
surface_support_v1
```

This is a post-hoc proposal calibration profile, not a new Chamfer-oriented object model. It changes snap proposal behavior from:

```text
uncalibrated max_disp = 0.02
```

to:

```text
calibrated max_disp = 0.005
```

The calibration is motivated by M7 evidence that larger snap steps can improve surface distance while harming valid-surface protect recall.

## 3. Pipeline Integration

`scripts/car_model/meshprior_run_pipeline.py` now supports:

```text
--calibration_profile none|surface_support_v1
```

The default is `surface_support_v1`. It affects snap proposals by selecting the calibrated max displacement.

## 4. Targeted Experiment

Command:

```bash
micromamba run -n mesh_splatting python scripts/car_model/meshprior_calibrate_prior.py --output_dir outputs/carnet/meshprior/prior_calibration/stage12_surface_support_v1
```

Output:

```text
outputs/carnet/meshprior/prior_calibration/stage12_surface_support_v1/calibration_metrics.json
```

Synthetic case:

```text
vertex_noise box
```

Comparison:

| Profile | Max disp | Protect recall after snap | Surface-distance delta | Free-space delta | Accepted |
|---|---:|---:|---:|---:|---:|
| `none` | `0.02` | `0.8333333333333334` | `0.021305546164512634` | `0.0` | `false` |
| `surface_support_v1` | `0.005` | `0.9166666666666666` | `0.01073157787322998` | `0.0` | `true` |

Baseline valid-surface protect recall was `0.9166666666666666`.

## 5. Verification

Commands run:

```bash
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage12_prior_calibration.py
micromamba run -n mesh_splatting python scripts/car_model/smoke_test_meshprior_stage10_pipeline.py
```

Results:

- Stage-12 smoke: PASS.
- Stage-10 pipeline regression: PASS.
- Targeted calibration experiment: PASS.

## 6. Stage Gate

| Gate | Result |
|---|---|
| Improves a proposal-relevant metric vs uncalibrated snap | PASS |
| Valid-surface protect recall is not harmed | PASS |
| Surface-distance still improves | PASS |
| Free-space violation does not increase | PASS |
| Pipeline accepts calibration profile | PASS |

Decision: `PASS`. The next allowed stage is M13 evaluation protocol and matrix.

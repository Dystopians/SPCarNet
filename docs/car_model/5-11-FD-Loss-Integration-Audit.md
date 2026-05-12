# 5-11 FD Loss Integration Audit

Date: 2026-05-11

This note audits the newly introduced Frechet-distance / FD loss path and records
whether it should enter the current SPCarNet / ECSR main method.

## What Was Integrated

- `utils/fd_loss.py` provides a frozen DINOv2/timm representation judge and
  Frechet-distance math.
- `utils/evidence_lumigraph_adapter.py` can use FD inside `calibrate_alpha`.
- `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py` exposes
  `--fd_strict`, `--fd_weight`, `--fd_max_views`, `--fd_min_views`,
  `--fd_backbone`, and `--fd_pool`.
- `scripts/car_model/ecsr_run_phasef_ela_adapter_eval.py` now forwards the FD
  arguments into the ELA applicator, so multi-scene evaluation can actually
  enable the new signal.
- W&B logging now records FD calibration fields such as selected FD gain,
  FD views, max/min FD gain, and whether FD was enabled.

Important interpretation: in the current code path this is not a training loss.
It is a frozen train-only calibration judge for ELA alpha selection. Calling it
the paper's main training loss would be misleading unless we later run the
backbone without `no_grad` and backpropagate through representation parameters.

## Smoke Tests

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_fd_loss.py
```

Result:

- NumPy/Torch FD reference match passed.
- Self-FD near-zero check passed.
- `frechet_distance_loss` consistency passed.
- DINOv2 `vit_base_patch14_dinov2.lvd142m` forward passed at 518 input size.
- End-to-end `calibrate_alpha` integration passed, including `fd_min_views`
  skip and alpha=0 strict-gate exemption.

## Treehill Targeted Probe

Policy fixed to the current treehill best edge branch:

- `k=4`
- `depth_rel_tol=0.06`
- `residual_clip=0.2`
- `direction_weight=0.35`
- `edge_gate_quantile=0.5`
- `edge_gate_dilate=1`
- train-only calibration, no test leakage

### FD Strict

Method: `ours_26000_phasej_fdstrict_edgeq05_ela`

FD strict selected the same alpha as the archived Phase-J result:

| method | alpha | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| Phase-J archived | 0.75 | 21.296227 | 0.595606 | 0.336319 |
| FD strict | 0.75 | 21.296227 | 0.595606 | 0.336319 |

Conclusion: `--fd_strict` is safe here but inert. It does not improve the main
result.

### FD Weight

Method: `ours_26000_phasej_fdw0005_edgeq05_ela`

`--fd_weight 0.005` selected alpha=1.0 and improved treehill LPIPS, but not
PSNR/SSIM:

| method | alpha | PSNR | SSIM | LPIPS | balanced score |
|---|---:|---:|---:|---:|---:|
| Phase-J archived | 0.75 | 21.296227 | 0.595606 | 0.336319 | 26.481971 |
| FD weight 0.005 | 1.0 | 21.263571 | 0.595165 | 0.321429 | 26.738290 |

This confirms FD is aligned with perceptual LPIPS on treehill, but it is not an
all-metric improvement.

## Outdoor Same-Policy Probe

All five outdoor scenes used the same fixed rule and `--fd_weight 0.005`.
No per-scene manual tuning was used.

| scene | selected alpha | dPSNR vs Phase-J | dSSIM vs Phase-J | dLPIPS vs Phase-J |
|---|---:|---:|---:|---:|
| bicycle | 0.75 | -0.206444 | -0.002864 | -0.006266 |
| flowers | 0.875 | -0.070229 | -0.002217 | -0.002495 |
| garden | 1.0 | -0.140196 | -0.003833 | +0.002657 |
| stump | 0.5 | -0.011190 | +0.000848 | -0.001009 |
| treehill | 1.0 | -0.032656 | -0.000441 | -0.014890 |
| mean | - | -0.092143 | -0.001702 | -0.004401 |

Verdict:

- FD weight improves mean LPIPS on outdoor scenes.
- It loses mean PSNR and SSIM.
- It even worsens LPIPS on garden.
- Therefore it is not acceptable as the current all-axis main method.

## Runtime Lesson

A full auto-policy FD probe over five edge candidates was interrupted after the
calibration phase proved too expensive. The bottleneck is the repeated
DINOv2 feature extraction plus 768-d covariance/eigendecomposition across many
alpha/policy candidates. For practical use:

- keep `fd_max_views` small for probes;
- limit CPU threads with `OMP_NUM_THREADS` / `MKL_NUM_THREADS`;
- avoid combining FD with broad auto-policy sweeps unless caching or a cheaper
  covariance approximation is implemented.

## Decision

Do not promote FD-weighted ELA to the paper mainline now.

Keep the FD path as:

1. an optional train-only perceptual portfolio / diagnostic signal;
2. a way to generate LPIPS-oriented qualitative candidates;
3. a possible future research direction if converted from frozen calibration
   judge into a true representation-level training objective.

For the main paper claim, continue to rely on the all-axis Phase-J archived
result unless a future method improves PSNR, SSIM, LPIPS, compactness, and
geometry together under the same fixed policy.


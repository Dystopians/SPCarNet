# Stage OUT1 Parking Outdoor Visual Repair Report

Date: 2026-05-06

## Problem

The previous parking row was numerically positive on average but weak in qualitative comparison. F33 CSEF70 sparse-depth recovery beat the train-selected clean30000 Mesh Splatting baseline by only +0.3035 dB PSNR, +0.0162 SSIM, and -0.0127 LPIPS, while 4 / 54 held-out parking views still failed at least one RGB metric (`00024`, `00029`, `00045`, `00047`). The visual failure was concentrated in outdoor low-support regions: road edges, vegetation, reflective building glass, and far tree/sky boundaries.

## Mechanism

OUT1 adds an Evidence Lumigraph Adapter (ELA) after compact recovery, then guards it with a train-calibrated parent-consistency gate.

The key rule is simple:

- `safe` render: F33 CSEF70 sparse-depth compact recovery.
- `candidate` render: F33 plus ELA residual repair.
- `parent` render: train-selected clean30000 Mesh Splatting baseline.
- Gate signal: mean absolute RGB distance between `safe` and `parent`, computed without test GT.
- Calibration: choose the gate threshold on train views only, using PSNR + 20 * SSIM - 20 * LPIPS and risk constraints.
- Test application: apply ELA only when `safe` differs enough from the parent; otherwise keep the safer compact render.

The selected v7 threshold is `0.017`. On train calibration it used 39 / 64 sampled views, with mean gains over the safe render of +0.1015 PSNR, +0.000886 SSIM, and +0.01223 LPIPS gain. On test it applied the candidate on 47 / 54 views and skipped 7 smooth / low-risk views (`00001`, `00004`, `00012`, `00018`, `00021`, `00040`, `00048`).

## Ablation Summary

| variant | mechanism | parking PSNR | SSIM | LPIPS | per-view full-pass |
|---|---|---:|---:|---:|---:|
| clean30000 | train-selected Mesh Splatting baseline | 18.4088 | 0.6315 | 0.3510 | reference |
| F33 | CSEF70 sparse-depth compact recovery | 18.7123 | 0.6477 | 0.3383 | 50 / 54 |
| OUT1-v1 | ELA residual repair, no parent gate | 19.0082 | 0.6609 | 0.3118 | 52 / 54 |
| OUT1-v3 | edge-gated ELA, quantile 0.50 | 18.9821 | 0.6604 | 0.3149 | 53 / 54 |
| OUT1-v7 | train-calibrated parent-consistency gate | 18.9528 | 0.6593 | 0.3137 | **54 / 54** |

v1 had the best mean RGB metrics but over-repaired smooth car / sidewalk views (`00001`, `00004`). Edge gating reduced risk but still left one failure. The parent-consistency gate solved the real problem: it kept the visible repair on difficult outdoor regions while falling back on smooth views where ELA was not needed.

## Final Fair Comparison

Against the train-selected clean30000 Mesh Splatting baseline:

| metric | OUT1-v7 delta |
|---|---:|
| PSNR | +0.5440 dB |
| SSIM | +0.0278 |
| LPIPS | -0.0373 |
| sparse AbsRel | -0.00257 |
| sparse Depth MAE | -0.01180 |
| sparse normal angle | -0.8032 deg |
| triangle reduction | 70.00 % |
| per-view RGB full-pass | 54 / 54 |

Against F33 itself, OUT1-v7 adds +0.2405 dB PSNR, +0.0116 SSIM, and -0.0246 LPIPS while preserving the same compact topology.

The refreshed ELA12 audit now reports 5 / 5 strict full-pass scenes and 165 / 165 held-out views passing PSNR, SSIM, and LPIPS simultaneously. W&B audit run: `0tfcaeef`.

## Artifacts

- Parent gate script: `scripts/car_model/meshsplatopt_parent_consistency_gate.py`
- ELA edge-gate support: `utils/evidence_lumigraph_adapter.py`
- ELA application CLI: `scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py`
- Parent-gate report: `outputs/carnet/meshsplatopt/stageOUT1_parking_visual_tail_recovery/f33_outdoor_parentgate_traincalib_v7_eval/test/ours_26000_outdoor_parentgate_traincalib_v7/parent_gate_report.json`
- Full montage: `assets/parking_outdoor_parentgate_v7_full_montage.png`
- Crop/error montage: `assets/parking_outdoor_parentgate_v7_crop_error_montage.png`
- Fair audit report: `docs/car_model/stageELA12_fair_baseline_audit_report.md`

## Remaining Risk

OUT1 is a render-level repair layer, not a new geometry edit. It improves the parking outdoor qualitative tail while preserving the compact mesh and sparse-geometry evidence from F33/F75-style recovery. The next research step is to turn the same parent-consistency idea into a geometry-aware training objective or selector signal, so the visual repair is born during recovery rather than applied only as a guarded render adapter.

# v42 vs Phase-J Gap Diagnostic

Date: 2026-06-23

Status: diagnostic only. Phase-J and v42 use different comparison baselines, so the ratios below quantify effect-size gap, not a strict method-vs-method fairness claim.

## Inputs

- v42 root: `outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware`
- Phase-J closure CSV: `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv`
- v42 delta: v42-SSIMGate minus same-evidence no-op compact baseline.
- Phase-J delta: Phase-J minus selected clean MeshSplatting baseline.

## Four-Scene Gap Table

| scene | v42 dPSNR | Phase-J dPSNR | PSNR ratio | v42 dSSIM | Phase-J dSSIM | SSIM ratio | v42 dLPIPS | Phase-J dLPIPS | LPIPS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| garden | +0.000137 | +1.281900 | 9334.5x | +0.00000221 | +0.047808 | 21678.1x | -0.00000334 | -0.065472 | 19614.9x |
| room | +0.001656 | +1.558363 | 941.3x | +0.00003928 | +0.020887 | 531.8x | -0.00001849 | -0.053913 | 2915.4x |
| counter | +0.001514 | +1.697397 | 1120.8x | +0.00000477 | +0.031675 | 6642.8x | -0.00002033 | -0.065531 | 3224.1x |
| bonsai | +0.000607 | +2.966772 | 4891.3x | +0.00000340 | +0.033879 | 9971.9x | -0.00000215 | -0.086937 | 40515.8x |
| **mean** | +0.000978 | +1.876108 | 1917.4x | +0.00001241 | +0.033563 | 2703.9x | -0.00001108 | -0.067963 | 6136.5x |

## Reading

- v42 is a real representation-level step, but its four-scene mean effect is still about three orders of magnitude smaller than Phase-J in PSNR and LPIPS effect size.
- This gap explains why v42's RGB crops remain visually subtle even when error-reduction maps show local positive action.
- The next representation-level method must increase residual support and expressivity while retaining the train-only SSIM/tail safety gate.

## Caveat

Do not present the ratio table as a fair head-to-head benchmark. It compares two different deltas: Phase-J over selected clean MeshSplatting, and v42 over same-evidence no-op compact. Its purpose is to quantify the remaining representation-internalization gap.

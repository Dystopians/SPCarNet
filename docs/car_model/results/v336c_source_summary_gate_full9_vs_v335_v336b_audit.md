# v336c Source-Summary Gate Full9 Audit

Date: 2026-07-01

## Aggregate Apply Metrics

| method | scenes | selected PSNR gain | selected SSIM gain | all-axis safe |
|---|---:|---:|---:|---:|
| v335 | 9 | 0.274017908934 | 0.003741526179 | 9/9 |
| v336b | 9 | 0.274583943273 | 0.003745085387 | 9/9 |
| v336c | 9 | 0.274617423486 | 0.003744976625 | 9/9 |

- v336c vs v335: dPSNR `+0.000599514552`, dSSIM `+0.000003450447`, nonnegative PSNR scenes `9/9`, nonnegative SSIM scenes `9/9`.
- v336c vs v336b: dPSNR `+0.000033480213`, dSSIM `-0.000000108762`, nonnegative PSNR scenes `7/9`, nonnegative SSIM scenes `7/9`.

## Per-Scene v336c vs v335

| scene | dPSNR | dSSIM | generated active | suppressed reason |
|---|---:|---:|---|---|
| bicycle | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'source_summary_psnr_delta:-0.0152193543'}` |
| bonsai | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'source_summary_psnr_delta:-0.0454876448'}` |
| counter | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'source_summary_psnr_delta:-0.0722975952'}` |
| flowers | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'source_summary_psnr_delta:-0.0108105151'}` |
| garden | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'source_summary_psnr_delta:-0.00355610577'}` |
| kitchen | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'source_summary_psnr_delta:-0.0761276079'}` |
| room | +0.005395630969 | +0.000031054020 | True | `{}` |
| stump | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'scene_selected_fixed'}` |
| treehill | +0.000000000000 | +0.000000000000 | False | `{'adaptive': 'scene_selected_fixed'}` |

## Frontier Metrics

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v335 | 27.590394 | 0.028168 | 0.087742 | 0.057670 |
| v336b | 27.590928 | 0.028167 | 0.087737 | 0.057667 |
| v336c | 27.590966 | 0.028167 | 0.087738 | 0.057667 |

## Evidence Paths

- Full9 reports: `outputs/carnet/spcarnet_v336c_source_summary_gate_full9_20260701/<scene>/support_transport_apply_report.json`
- Frontier summary: `outputs/carnet/spcarnet_v336c_frontier_full9_20260701/frontier_lpips_qualitative_summary.{json,md}`
- Archived frontier panels: `docs/car_model/results/v336c_frontier_panels/`

## Honest Verdict

9/9 PSNR and SSIM non-decreasing; macro PSNR/SSIM improved; full9 frontier PSNR/MAE/DISTS improved and LPIPS remains better than clean26000 and near v336b.

Gains are still small and visually subtle; this is a reliability/candidate-gating milestone, not final paper closure.

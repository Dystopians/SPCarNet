# Phase-S Vertex-Delta Closed-Loop Audit

This collector summarizes existing Phase-S vertex-delta artifacts. It is
read-only and uses only train-val gate/search outputs plus operator audits.

## Summary

| metric | value |
|---|---|
| multifold gate rows | 6 |
| accepted multifold rows | 2 |
| render-calibrated searches | 2 |
| accepted render-calibrated searches | 0 |
| qualitative manifest | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v24_qualitative_20260513/qualitative_manifest.md` |

## Multi-Offset Gates

| scene | label | accepted | faces | mode | topology changed | attr changed | max attr delta | dPSNR | dSSIM | dLPIPS | reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | vertexdelta_v24_effective_structure_20260513 | true | 3 | vertex_delta | false | true | 0.0199697 | +0.000011921 | +0.000003606 | -0.000006706 | pass |
| treehill | vertexdelta_v24_effective_structure_20260513 | true | 2 | vertex_delta | false | true | 0.0232427 | +0.000016212 | +0.000000000 | +0.000000618 | pass |
| bicycle | vertexdelta_v25_visualreach_20260513 | false | 64 | vertex_delta | false | true | 0.0322942 | +0.000101566 | +0.000001967 | +0.000093929 | offset0:lpips_regression_exceeds_0.00015, offset3:lpips_regression_exceeds_0.00015 |
| treehill | vertexdelta_v25_visualreach_20260513 | false | 35 | vertex_delta | false | true | 0.0321063 | +0.000097275 | +0.000005350 | +0.000059128 | offset2:psnr_gain_below_0, offset2:ssim_regression_exceeds_5e-05, offset2:lpips_regression_exceeds_0.00015 |
| bicycle | vertexdelta_v27_shrink08_20260513 | false | 64 | vertex_delta | false | true | 0.0180767 | +0.000752926 | +0.000036806 | +0.000199035 | offset0:lpips_regression_exceeds_0.00015, offset3:lpips_regression_exceeds_0.00015 |
| treehill | vertexdelta_v27_shrink08_20260513 | false | 37 | vertex_delta | false | true | 0.0214042 | +0.000113487 | +0.000005186 | +0.000115633 | offset1:ssim_regression_exceeds_5e-05, offset1:lpips_regression_exceeds_0.00015 |

## Render-Calibrated Searches

| scene | label | accepted | faces | events | best objective | best trial | strict | trial objective | dPSNR | dSSIM | dLPIPS | last action reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| counter | vertexdelta_v24_rendercalib_counter_20260513 | false | 0 | 4 | +0.000000000 | s000_add0-1_1961193-2308565 | true | +0.000004873 | +0.000005245 | -0.000000030 | -0.000000011 | n/a |
| treehill | vertexdelta_v26_rendercalib_treehill_20260513 | false | 0 | 6 | +0.000000000 | s003_add0-4_488244-271838 | true | +0.000004381 | +0.000020027 | -0.000000179 | +0.000000603 | n/a |

## Interpretation

- v24-style vertex-delta gates show that topology-preserving feature edits can be made strict-gate safe.
- The effect size is still too small for a paper-level visual claim when the qualitative manifest reports near-zero image deltas.
- v25/v27-style stronger edits increase PSNR but fail LPIPS/offset stability, so they are useful negative ablations, not promoted methods.
- The render-calibrated searches were stopped because completed strict passes stayed below the fixed objective thresholds and did not change this conclusion.

## Artifact Index

| scene | label | primary artifact | audit/log |
|---|---|---|---|
| bicycle | vertexdelta_v24_effective_structure_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/vertexdelta_v24_effective_structure_20260513_bicycle/bicycle/multifold_trainval_gate.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v24_effective_structure_20260513_bicycle/bicycle/model/surface_residual_subdivision_delta_audit.json` |
| treehill | vertexdelta_v24_effective_structure_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/vertexdelta_v24_effective_structure_20260513_treehill/treehill/multifold_trainval_gate.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v24_effective_structure_20260513_treehill/treehill/model/surface_residual_subdivision_delta_audit.json` |
| bicycle | vertexdelta_v25_visualreach_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/vertexdelta_v25_visualreach_20260513_bicycle/bicycle/multifold_trainval_gate.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v25_visualreach_20260513_bicycle/bicycle/model/surface_residual_subdivision_delta_audit.json` |
| treehill | vertexdelta_v25_visualreach_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/vertexdelta_v25_visualreach_20260513_treehill/treehill/multifold_trainval_gate.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v25_visualreach_20260513_treehill/treehill/model/surface_residual_subdivision_delta_audit.json` |
| bicycle | vertexdelta_v27_shrink08_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/vertexdelta_v27_shrink08_20260513_bicycle/bicycle/multifold_trainval_gate.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v27_shrink08_20260513_bicycle/bicycle/model/surface_residual_subdivision_delta_audit.json` |
| treehill | vertexdelta_v27_shrink08_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/vertexdelta_v27_shrink08_20260513_treehill/treehill/multifold_trainval_gate.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v27_shrink08_20260513_treehill/treehill/model/surface_residual_subdivision_delta_audit.json` |
| counter | vertexdelta_v24_rendercalib_counter_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/rendercalib_vertexdelta_v24_counter_top8pairs_gpu7_20260513/counter/render_calibrated_search.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/rendercalib_vertexdelta_v24_counter_top8pairs_gpu7_20260513/counter/render_calibrated_search.log` |
| treehill | vertexdelta_v26_rendercalib_treehill_20260513 | `outputs/carnet/meshsplatopt/ecsr_phase_s/rendercalib_vertexdelta_v26_treehill_v25top12_gpu1_20260513/treehill/render_calibrated_search.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/rendercalib_vertexdelta_v26_treehill_v25top12_gpu1_20260513/treehill/render_calibrated_search.log` |


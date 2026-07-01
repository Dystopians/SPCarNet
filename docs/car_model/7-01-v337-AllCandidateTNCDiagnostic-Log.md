# v337 All-Candidate Target-Neighbor Diagnostic

Date: 2026-07-01

## Purpose

v336c showed that generated-candidate admission can be made safe, but it did
not materially reduce the remaining per-view oracle gap. The next question is
whether target-neighbor render/depth/camera consistency can rank all candidate
variants well enough to recover that headroom without target/test GT.

This milestone adds diagnostic infrastructure only. It does not change output
selection unless future work explicitly uses the diagnostic to design a frozen
policy.

## Implementation

Changed file:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New CLI:

```text
--enable_target_neighbor_all_candidate_diagnostic
--target_neighbor_all_candidate_diagnostic_eps 1e-12
```

When enabled, the apply loop:

1. computes target-neighbor render/depth/camera MAE for every candidate variant
   before target GT is read;
2. ranks candidates by target-neighbor MAE;
3. after selected output is saved and GT metrics are computed, attaches a
   post-hoc strict oracle alignment analysis;
4. writes both per-view diagnostics and a top-level summary.

Key report fields:

```text
per_view[*].target_neighbor_all_candidate_rank_diagnostics
target_neighbor_all_candidate_rank_diagnostic
selection_protocol.target_neighbor_all_candidate_diagnostic_affects_selection = false
selection_protocol.target_neighbor_all_candidate_diagnostic_scores_use_target_gt = false
selection_protocol.target_neighbor_all_candidate_diagnostic_oracle_uses_target_gt_posthoc = true
```

This means target-neighbor scores are no-GT diagnostics. The strict oracle part
uses GT only after output save for analysis, not for selection.

## Smoke Validation

Compile:

```bash
python -m py_compile \
  scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  scripts/car_model/analyze_support_transport_oracle_gap.py
```

Single-view smoke:

```text
outputs/carnet/spcarnet_v337_all_candidate_tnc_diag_smoke_room_20260701
wandb offline: offline-run-20260701_060816-vkuviems
```

This validated report schema but had no target neighbors because
`--max_target_views 1`.

Three-view smoke:

```text
outputs/carnet/spcarnet_v337_all_candidate_tnc_diag_smoke3_room_20260701
wandb offline: offline-run-20260701_060942-uo6fk6s9
```

Top-level diagnostic:

| metric | value |
|---|---:|
| enabled views | 3 |
| target-neighbor available views | 3 |
| TNC best matches strict oracle | 0/3 |
| mean strict-oracle minus output PSNR | +0.005372130291 |
| mean strict-oracle minus output SSIM | +0.000024656455 |
| mean TNC-best minus output PSNR | -0.042101944759 |
| mean TNC-best minus output SSIM | -0.000392635663 |

Per-view result:

| view | output | strict oracle | TNC best | TNC rank order |
|---|---|---|---|---|
| 00000 | adaptive | adaptive | fixed | fixed, mix0250, adaptive, hybrid, mix0750, learned |
| 00001 | hybrid | adaptive | hybrid | hybrid, mix0750, mix0250, adaptive, learned, fixed |
| 00002 | adaptive | adaptive | learned | learned, mix0750, hybrid, mix0250, adaptive, fixed |

## Interpretation

The smoke result confirms a key lesson from v335: target-neighbor consistency is
not a reliable standalone ranker. It can be useful as a certificate or
contradiction signal, but pure TNC ranking can prefer candidates that are worse
under target GT. The next policy should not simply choose the lowest
target-neighbor MAE candidate.

The productive next use of this diagnostic is feature analysis: run it on the
largest oracle-gap focus scenes, then test whether TNC score combined with
source-heldout evidence, support-dropout stability, and candidate proxy features
can identify strict-oracle candidates without causing the known learned-to-fixed
failures in `bonsai` and `kitchen`.

## Status

```text
Final status: NOT COMPLETE.
```

This is an engineering and diagnostic milestone, not a new promoted method.
It closes an observability gap needed for the next method design.

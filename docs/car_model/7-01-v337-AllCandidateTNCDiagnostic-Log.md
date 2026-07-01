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

## Focus6 Diagnostic Replay

Root:

```text
outputs/carnet/spcarnet_v337_all_candidate_tnc_diag_focus6_20260701
```

Scenes:

```text
stump, treehill, room, bicycle, bonsai, kitchen
```

W&B offline runs:

```text
stump:   offline-run-20260701_061342-vs7c2ej9
treehill:offline-run-20260701_061423-4qk59bfu
room:    offline-run-20260701_061518-673uu8tw
bicycle: offline-run-20260701_061604-mx3ym7zm
bonsai:  offline-run-20260701_061516-80vkjezo
kitchen: offline-run-20260701_061644-0gfih90m
```

Saved summaries:

```text
docs/car_model/results/v337_all_candidate_tnc_diag_focus6_oracle_gap.json
docs/car_model/results/v337_all_candidate_tnc_diag_focus6_oracle_gap.md
docs/car_model/results/v337_all_candidate_tnc_diag_focus6_tnc_rank_summary.json
docs/car_model/results/v337_all_candidate_tnc_diag_focus6_tnc_rank_summary.md
```

Macro result:

| scenes | views | available | TNC matches strict oracle | match frac | oracle-output PSNR | TNC-best-output PSNR | oracle rank | output rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 170 | 170 | 37 | 0.217647 | +0.009893758799 | -0.051566646277 | 3.894118 | 3.682353 |

Per-scene result:

| scene | views | match frac | oracle-output PSNR | TNC-best-output PSNR | note |
|---|---:|---:|---:|---:|---|
| stump | 16 | 0.500000 | +0.021858285120 | +0.001558000550 | partial positive signal |
| treehill | 18 | 0.611111 | +0.015234013794 | -0.005465829862 | partial signal, not enough alone |
| room | 39 | 0.025641 | +0.011576829779 | -0.045652079578 | strong false-positive risk |
| bicycle | 25 | 0.280000 | +0.010343502985 | -0.017025614542 | mixed |
| bonsai | 37 | 0.162162 | +0.007299128751 | -0.092994969647 | hard control failure |
| kitchen | 35 | 0.114286 | +0.002224071022 | -0.087028216981 | hard control failure |

The PSNR-primary oracle gap over the same focus6 reports is still substantial:

| method | scenes | views | selected mean | oracle mean | mean headroom | positive views |
|---|---:|---:|---:|---:|---:|---:|
| v337diag_focus6 | 6 | 170 | 0.301231404 | 0.313737956 | +0.012506552 | 63 |

## Interpretation

The smoke and focus6 results confirm a key lesson from v335:
target-neighbor consistency is not a reliable standalone ranker. It can be
useful as a certificate, contradiction signal, or feature in a combined
train-evidence policy, but pure TNC ranking often prefers candidates that are
worse under target GT. The next policy should not simply choose the lowest
target-neighbor MAE candidate.

The productive next use of this diagnostic is a combined no-target-GT
candidate-ranker: TNC should be used only when it agrees with source-heldout
evidence, support-dropout stability, and candidate proxy features. `bonsai` and
`kitchen` should remain hard controls because pure TNC ranking strongly fails
there.

## Status

```text
Final status: NOT COMPLETE.
```

This is an engineering and diagnostic milestone, not a new promoted method.
It closes an observability gap needed for the next method design.

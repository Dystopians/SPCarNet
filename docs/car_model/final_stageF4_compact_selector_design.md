# Final Stage F4 Compact Selector Design

Date: 2026-05-04

## Purpose

R53 proves that area-only clean-to-compact can work on parking, but R57/R60 show that smallest-area pruning is not universal. F4 adds a CSEF-compatible selector so the final method can compare area-only against evidence-aware candidate selection.

## Implementation

Core module:

```text
ss3dm_prior/meshsplatopt/compact_selector.py
```

CLI:

```text
scripts/car_model/meshsplatopt_select_compaction_candidates.py
```

Smoke:

```text
scripts/car_model/smoke_test_final_stageF4_compact_selector.py
```

## Signals

The selector can score each face from:

- triangle area;
- render contribution or visibility if present in the checkpoint;
- sparse geometry support;
- normal/orientation support;
- local redundancy and coplanarity;
- boundary/edge risk;
- CSEF positive surface evidence;
- CSEF negative free-space evidence;
- CSEF explanation debt;
- topology cost;
- uncertainty;
- explicit protected-face mask.

If render/sparse/normal evidence is missing, the implementation still runs with conservative defaults and records area, boundary, redundancy, topology cost, and CSEF scores.

## Modes

```text
area_smallest
csef_low_evidence
csef_low_evidence_boundary_protected
pareto_area_csef
random_same_count
```

`area_smallest` is the existing control. `csef_low_evidence` favors high topology cost, redundancy, negative free-space, and uncertainty while penalizing positive evidence and explanation debt. `csef_low_evidence_boundary_protected` additionally blocks explicitly protected faces and high-debt repair regions. `pareto_area_csef` combines area rank and CSEF rank. `random_same_count` is the count-matched control.

## Outputs

For each selector run:

```text
compaction_candidates.json
compaction_score_table.npz
compaction_summary.csv
compaction_report.md
```

These outputs are sufficient for F5 checkpoint compaction: `compaction_candidates.json` stores selected face ids; `compaction_score_table.npz` preserves all score components for audit; the CSV and Markdown files summarize the decision.

## Design Boundary

F4 only selects faces. It does not edit checkpoints. Checkpoint application and render validation belong to F5 and later stages.

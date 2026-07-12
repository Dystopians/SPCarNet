# TOPCONF READINESS REPORT — living status

Opened 2026-07-11. Final verdict is written when the stop conditions of the mission hold.
Initial overall score: **58/100**. Current: **~66/100** (in progress).

## Score trajectory

| Axis | Initial | Current | Delta driver |
|---|---|---|---|
| A. Novelty/distinction | 55 | 57 | terminology hardened ("audited", defined threat model); head-to-head still pending EXP-IBR |
| B. External baselines | 40 | 45 | IBRNet checkout + weights staged; adapter in build; numbers pending |
| C. Benchmark breadth | 55 | 58 | T&T+DB staged & registered (PROTOCOL 1.3.0); training chains running; rows pending |
| D. Statistical rigor | 85 | 95 | EXP-HBOOT banked: 8/8 headline CIs survive scene-cluster resampling |
| E. Temporal stability | 20 | 55 | garden banked: ECR path SMOOTHER than base (0.988/0.983); bonsai+town01 running; videos rendering |
| F. Storage/runtime fairness | 80 | 82 | (plots pending; accounting was already complete) |
| G. Threat model | 70 | 95 | PROTOCOL §4E.1 written; "certified" retired; CR4 v0.6 |
| H. Simplicity/ablations | 75 | 78 | E-08 reframed as the simplicity result in claims |
| I. Repro/paper readiness | 65 | 67 | (README/tables/plots pending) |

## Blocker status

| ID | Blocker | Status |
|---|---|---|
| P0-1 | second standard benchmark | RUNNING — t2b_chainA/B (truck+drjohnson GPU3, train+playroom GPU4): training → PJ → final rows + audits |
| P0-2 | external IBR baseline | IN BUILD — IBRNet weights local; Codex building torch-2.x job-file adapter; GPU runs + mirror = mine |
| P1-3 | temporal stability | 1/3 scenes banked (garden PASS, ratio < 1 — smoother); bonsai+town01 running |
| P1-4 | scene-cluster bootstrap | **CLOSED** — PASS 8/8 (`hierarchical_cis.md`; CLAIMS v0.6 robustness addendum) |
| P1-5 | threat model / "certified" | **CLOSED** — PROTOCOL §4E.1 + terminology sweep (CLAIMS v0.6) |
| P1-6 | unified R-D + ladder plots | QUEUED (Codex, after P0 cells so plots include new points) |
| P1-7 | repro README + paper tables | QUEUED (Codex) |

## Residual risks (tracked for the final report)

- T&T outdoor coverage may yield mid-range covered_fraction → smaller transport gains than full9;
  will be reported as-is (external-validity honesty is the point of the cell).
- IBRNet camera-convention correctness is the fabrication-risk hotspot → my verification plan:
  self-reconstruction test (render a train view from its own pose given neighboring sources; must be
  far above chance) BEFORE any cross-method number is computed.
- Difix/3DGS unfavorable evidence stays first-class (no burying).

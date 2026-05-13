# ECSR Phase-B View-Support Redundancy Graph

This report is generated from the Phase-A train-only surface evidence
cache. It upgrades the unit of reasoning from isolated face deletion to
auditable local support groups. The output is still candidate-only:
no checkpoint is modified and no held-out test view participates.

## Fixed Policy

- top-K residual supports per scene: `512`
- projected adjacency sampling stride: `8`
- edge score threshold: `0.5`
- cluster score threshold: `0.55`
- min shared train views: `1`
- max cluster size: `16`
- test usage: `none`

## Aggregate

| metric | value |
|---|---|
| scenes | 2 |
| cluster candidates | 51 |
| certificate-contraction candidates | 8 |
| surface-attribute recovery candidates | 43 |
| scenes with certification-ready candidates | 2 / 2 |
| mean triangle reduction upper bound from Phase-B clusters | 0.0049% |

## Per-Scene Result

| scene | type | nodes | edge cand. | cluster cand. | contraction cand. | attribute cand. | tri-red upper | Phase-A multiview | next action |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | outdoor | 512 | 21782 | 30 | 6 | 24 | 0.0061% | 25.00% | candidate_certification_ready |
| flowers | outdoor | 512 | 12244 | 21 | 2 | 19 | 0.0036% | 26.12% | candidate_certification_ready |

## Interpretation

Phase B confirms that the correct next unit is a local support group, not
a hand-picked per-scene parameter. The graph finds auditable candidate
groups from train evidence only, but the expected direct triangle
reduction of the top residual supports is still tiny. Therefore, Phase C
must not overclaim compression from these clusters alone. It should use
the graph as a safe candidate front-end for certificate contraction and
Phase D surface-attached recovery.

The strong research direction is now fixed:

1. use Phase-B groups as policy-defined local masks;
2. run certificate checks on train/policy-val only;
3. start with attribute-only recovery where contraction evidence is weak;
4. reserve held-out test for final full9 validation.

## Artifacts

| scene | graph JSON | candidates CSV | report |
|---|---|---|---|
| bicycle | `outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/bicycle/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/bicycle/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/bicycle/view_support_graph_report.md` |
| flowers | `outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/flowers/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/flowers/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/flowers/view_support_graph_report.md` |


# ECSR Phase-B View-Support Redundancy Graph

This report is generated from the Phase-A train-only surface evidence
cache. It upgrades the unit of reasoning from isolated face deletion to
auditable local support groups. The output is still candidate-only:
no checkpoint is modified and no held-out test view participates.

## Fixed Policy

- top-K residual supports per scene: `256`
- projected adjacency sampling stride: `8`
- edge score threshold: `0.52`
- cluster score threshold: `0.58`
- min shared train views: `1`
- max cluster size: `12`
- test usage: `none`

## Aggregate

| metric | value |
|---|---|
| scenes | 9 |
| cluster candidates | 123 |
| certificate-contraction candidates | 23 |
| surface-attribute recovery candidates | 99 |
| scenes with certification-ready candidates | 8 / 9 |
| mean triangle reduction upper bound from Phase-B clusters | 0.0028% |

## Per-Scene Result

| scene | type | nodes | edge cand. | cluster cand. | contraction cand. | attribute cand. | tri-red upper | Phase-A multiview | next action |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | outdoor | 256 | 3687 | 20 | 3 | 17 | 0.0031% | 23.05% | candidate_certification_ready |
| flowers | outdoor | 256 | 2100 | 11 | 1 | 10 | 0.0016% | 21.48% | candidate_certification_ready |
| garden | outdoor | 256 | 1294 | 16 | 3 | 13 | 0.0023% | 20.70% | candidate_certification_ready |
| stump | outdoor | 256 | 3011 | 14 | 2 | 12 | 0.0014% | 4.30% | candidate_certification_ready |
| treehill | outdoor | 256 | 1986 | 7 | 0 | 7 | 0.0006% | 23.44% | attribute_recovery_first |
| room | indoor | 256 | 947 | 15 | 1 | 14 | 0.0034% | 36.33% | candidate_certification_ready |
| counter | indoor | 256 | 968 | 15 | 7 | 8 | 0.0043% | 38.67% | candidate_certification_ready |
| kitchen | indoor | 256 | 980 | 12 | 2 | 10 | 0.0037% | 34.38% | candidate_certification_ready |
| bonsai | indoor | 256 | 2065 | 13 | 4 | 8 | 0.0050% | 53.12% | candidate_certification_ready |

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
| bicycle | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/bicycle/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/bicycle/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/bicycle/view_support_graph_report.md` |
| flowers | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/flowers/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/flowers/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/flowers/view_support_graph_report.md` |
| garden | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/garden/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/garden/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/garden/view_support_graph_report.md` |
| stump | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/stump/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/stump/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/stump/view_support_graph_report.md` |
| treehill | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/treehill/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/treehill/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/treehill/view_support_graph_report.md` |
| room | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/room/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/room/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/room/view_support_graph_report.md` |
| counter | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/counter/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/counter/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/counter/view_support_graph_report.md` |
| kitchen | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/kitchen/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/kitchen/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/kitchen/view_support_graph_report.md` |
| bonsai | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/bonsai/view_support_graph.json` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/bonsai/candidate_clusters.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph/bonsai/view_support_graph_report.md` |


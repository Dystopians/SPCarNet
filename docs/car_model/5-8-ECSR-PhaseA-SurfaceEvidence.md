# ECSR Phase-A Surface Evidence Diagnostics

This report is generated from train-view renders only. It is the Phase-A
acceptance artifact for the FinalDecision ECSR plan: it checks whether
current residual signals are surface-addressable before any new
contraction or surface-attached recovery module is promoted.

## Protocol

- split: `train` only
- scenes: `bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai`
- selected views: `8` per scene, stride `6`, offset `0`
- held-out test usage: `none`
- per-scene cache root: `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence`
- generated from compact checkpoint root: `compact_ela_sor_adaptive_geo_26k/*/sor_adaptive_geo/compact_model`

## Aggregate Result

| metric | value |
|---|---|
| scenes collected | 9 / 9 |
| surface addressability pass | 9 / 9 |
| residual multiview consistency pass | 4 / 9 |
| appearance-relocation promising | 5 / 9 |
| mean valid face-id fraction | 99.93% |
| mean top-error addressable fraction | 99.69% |
| mean top-support multiview fraction | 28.39% |
| mean top-support consistency | 0.9680 |
| outdoor mean multiview fraction | 18.59% |
| indoor mean multiview fraction | 40.62% |

## Per-Scene Diagnostic

| scene | type | views | valid face-id | top-error addressable | top support multiview | top consistency | A-address | A-consistency | B-relocation | next action |
|---|---|---|---|---|---|---|---|---|---|---|
| bicycle | outdoor | 8 | 99.95% | 99.88% | 23.05% | 0.9937 | pass | weak | appearance-relocation-promising | needs cluster-level view-support graph |
| flowers | outdoor | 8 | 99.97% | 99.82% | 21.48% | 0.9735 | pass | weak | appearance-relocation-promising | needs cluster-level view-support graph |
| garden | outdoor | 8 | 99.97% | 99.88% | 20.70% | 0.9569 | pass | weak | appearance-relocation-promising | needs cluster-level view-support graph |
| stump | outdoor | 8 | 99.67% | 98.07% | 4.30% | 0.9989 | pass | weak | appearance-relocation-promising | needs cluster-level view-support graph |
| treehill | outdoor | 8 | 99.84% | 99.61% | 23.44% | 0.9906 | pass | weak | appearance-relocation-promising | needs cluster-level view-support graph |
| room | indoor | 8 | 100.00% | 100.00% | 36.33% | 0.9428 | pass | pass | weak-relocation-signal | cluster/attribute recovery candidate |
| counter | indoor | 8 | 100.00% | 100.00% | 38.67% | 0.9305 | pass | pass | weak-relocation-signal | cluster/attribute recovery candidate |
| kitchen | indoor | 8 | 99.99% | 99.95% | 34.38% | 0.9585 | pass | pass | weak-relocation-signal | cluster/attribute recovery candidate |
| bonsai | indoor | 8 | 100.00% | 100.00% | 53.12% | 0.9664 | pass | pass | weak-relocation-signal | cluster/attribute recovery candidate |

## Surface Evidence Artifacts

| scene | report | top support list | contact sheet |
|---|---|---|---|
| bicycle | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/bicycle/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/bicycle/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/bicycle/surface_residual_contact_sheet.png` |
| flowers | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/flowers/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/flowers/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/flowers/surface_residual_contact_sheet.png` |
| garden | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/garden/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/garden/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/garden/surface_residual_contact_sheet.png` |
| stump | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/stump/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/stump/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/stump/surface_residual_contact_sheet.png` |
| treehill | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/treehill/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/treehill/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/treehill/surface_residual_contact_sheet.png` |
| room | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/room/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/room/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/room/surface_residual_contact_sheet.png` |
| counter | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/counter/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/counter/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/counter/surface_residual_contact_sheet.png` |
| kitchen | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/kitchen/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/kitchen/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/kitchen/surface_residual_contact_sheet.png` |
| bonsai | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/bonsai/surface_evidence_report.md` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/bonsai/top_residual_supports.csv` | `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/bonsai/surface_residual_contact_sheet.png` |

A combined visual index is written to `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/phase_a_surface_evidence_contact_sheet.png`.

## Interpretation

1. The residual signal is strongly surface-addressable: every collected scene
   passes the addressability diagnostic, with almost all high-error pixels
   carrying valid rendered face ids.
2. Direct per-face residual relocation is not yet a safe universal policy.
   Outdoor scenes have high addressability but sparse top-support multiview
   redundancy, so single-face residual deltas would risk view-specific
   artifacts. The next representation-level step must aggregate supports
   into local clusters or a view-support redundancy graph.
3. Indoor scenes have stronger multiview support but weak aggregate ELA
   relocation signal. They are better suited for certificate contraction
   plus attribute-only recovery than for high-capacity residual deltas.
4. Existing README qualitative crops remain presentation evidence only.
   Phase-A top support masks are the replacement source for train-defined
   local evaluation and must drive future qualitative crop selection.

## Phase-A Acceptance

Phase A is accepted as a diagnostic foundation, not as the final method.
It proves that surface addressing is technically available and that the
failure mode is not random image-space noise. It also rejects a naive
single-face SH-delta implementation as the next main method because the
top supports are not sufficiently multiview-stable on several outdoor
scenes.

## Concrete Next Step

Proceed to Phase B with a fixed View-Support Redundancy Graph:

- nodes: local face clusters, not isolated faces
- edges: adjacency plus train-view co-visibility, depth/normal compatibility, residual compatibility, and occlusion risk
- candidate outputs: attribute-only merge, conservative cluster contraction, and no-topology surface residual relocation
- certificate: train/policy-val only, with test reserved for final report

This is the cleanest path to turn the current ELA-dominated version into
a representation-level ECSR method without per-scene parameter games.


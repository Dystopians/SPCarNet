# ECSR FinalDecision Execution Log

Date: 2026-05-08

Guiding spec: `docs/car_model/5-7-FinalDecision.md`

This log records the concrete implementation status after executing the first
strict pass of the FinalDecision plan. It is deliberately conservative: an item
is marked complete only when there is executable code, generated evidence, and
a reproducible artifact.

## Completed Artifacts

| artifact | path | status |
|---|---|---|
| Current-state audit script | `scripts/car_model/ecsr_current_state_audit.py` | complete |
| Current-state audit doc | `docs/car_model/5-8-ECSR-CurrentStateAudit.md` | complete |
| Phase-A surface evidence cache | `scripts/car_model/ecsr_build_surface_evidence_cache.py` | complete MVP |
| Phase-A runner | `scripts/car_model/ecsr_run_phase_a_surface_diagnostics.sh` | complete |
| Phase-A full9 collector | `scripts/car_model/ecsr_collect_phase_a_surface_diagnostics.py` | complete |
| Phase-A report | `docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md` | complete |
| Phase-B view-support graph | `scripts/car_model/ecsr_build_phase_b_view_support_graph.py` | complete candidate generator |
| Phase-B report | `docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md` | complete |
| Phase-A/B cached-view policy split | `scripts/car_model/ecsr_make_phase_ab_policy_splits.py` | complete smoke split |
| Phase-A/B policy split doc | `docs/car_model/5-8-ECSR-PolicySplit.md` | complete |

## Current-State Audit Result

The archived Compact-ELA/SOR full9 version remains a valid same-protocol
baseline improvement, but it is not a sufficient final paper contribution.

- RGB + compact + geometry-safe pass: `9 / 9`
- strict all-axis pass: `5 / 9`
- mean RGB deltas vs selected clean MeshSplatting: `+0.4979 PSNR`,
  `+0.0158 SSIM`, `-0.0234 LPIPS`
- mean triangle reduction: `5.7632%`
- major limitation: the strongest RGB gain still comes from image-space ELA,
  while indoor/garden compactness is conservative.

## Phase-A Evidence Cache Result

Protocol: train split only, 8 views per scene, stride 6, no held-out test
usage.

| metric | result |
|---|---:|
| scenes collected | `9 / 9` |
| surface addressability pass | `9 / 9` |
| residual multiview consistency pass | `4 / 9` |
| appearance-relocation promising | `5 / 9` |
| mean valid face-id fraction | `99.93%` |
| mean top-error addressable fraction | `99.69%` |
| mean top-support multiview fraction | `28.39%` |

Interpretation: residuals are not random image-space noise; they can be
addressed back to rendered surface ids. However, outdoor residual hot supports
are often not sufficiently multiview-stable as isolated faces. A direct
single-face residual SH/RGB delta would be a risky next method.

## Phase-B View-Support Graph Result

Protocol: fixed graph thresholds, train-cache only, no checkpoint edits, no
held-out test usage.

| metric | result |
|---|---:|
| local support clusters | `123` |
| certificate-contraction candidates | `23` |
| surface-attribute recovery candidates | `99` |
| scenes with certification-ready candidates | `8 / 9` |
| mean triangle-reduction upper bound from residual-hot clusters | `0.0028%` |

Interpretation: the graph successfully upgrades the decision unit from a single
face to a local support group. It also exposes a key bottleneck: residual-hot
clusters are appearance-recovery targets, not meaningful compression targets.
Therefore the next ECSR design must explicitly maintain two fronts:

1. a compression front from low-risk redundant geometry supports;
2. an appearance front from train-defined residual supports.

Treating residual hotspots as the compression target would not produce the
required compactness improvement.

## FinalDecision Status Matrix

| requirement | status | evidence |
|---|---|---|
| Current Protocol / Result / Bottleneck / Leakage audit | complete | `5-8-ECSR-CurrentStateAudit.md` |
| Contribution reframed away from better ELA | started | README + Phase-A/B docs |
| Surface Evidence Cache | complete MVP | full9 train-only cache, summaries, contact sheets |
| Diagnostic A: residual surface addressability | complete | `9 / 9` pass |
| Diagnostic B: relocation necessity | complete MVP | `5 / 9` appearance-relocation promising |
| View-Support Redundancy Graph | complete candidate generator | 123 clusters, 23 contraction candidates |
| Certificate-Carrying Surface Contraction | not complete | candidates exist, certificates not run |
| Surface-Attached Appearance Recovery | not complete | target type identified, no representation delta yet |
| Train/policy-val split file | partial | cached-view split complete; full-train split still required before long Phase C/D |
| Full9 same-protocol ECSR validation | not complete | no final ECSR checkpoint yet |
| Reviewer Objection Audit | partial | leakage/post-processing risks documented; final rebuttal needs Phase C/D |

## Design Decision Locked By This Pass

The next method should not be another ELA variant and should not be a per-scene
parameter search. The fixed design should be:

1. build a train/policy-val split with a recorded seed;
2. generate compression candidates from low-risk redundant supports, not from
   residual hotspots;
3. use the Phase-B graph to define local support masks and support groups;
4. run certificate contraction only on candidates whose geometry/topology
   smoke tests pass;
5. implement Version 1 surface-attached recovery first: attribute-only
   re-fitting on retained primitives, selected by policy-val;
6. only then attempt Version 2 residual SH/RGB delta;
7. keep image-space ELA as teacher/upper bound, not the main method.

## Stop / Continue Decision

Continue. The current pass made real progress but does not yet satisfy the
FinalDecision endpoint. The key remaining blocker is not GPU time; it is method
separation:

- compression must come from geometry-redundancy candidates;
- visual improvement must come from surface-attached recovery;
- both must be accepted by train/policy-val certificates before held-out test.

The next concrete milestone is Phase C/D MVP: train/policy split, static
certificate schema, attribute-only recovery after graph-defined local support
groups, and full9 validation against the archived Compact-ELA/SOR and selected
clean MeshSplatting baselines.

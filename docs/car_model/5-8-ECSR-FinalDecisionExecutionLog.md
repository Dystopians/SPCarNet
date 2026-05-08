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
| Full-train fitting/policy-val split | `scripts/car_model/ecsr_make_full_train_policy_splits.py` | complete full9 split |
| Full-train split report | `docs/car_model/5-8-ECSR-FullTrainPolicySplit.md` | complete |
| Phase-C candidate preflight | `scripts/car_model/ecsr_phase_c_candidate_preflight.py` | complete pre-contraction filter |
| Phase-C preflight doc | `docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md` | complete |
| Phase-C static topology certificate | `scripts/car_model/ecsr_phase_c_static_topology_certificate.py` | complete static filter |
| Phase-C materialization | `scripts/car_model/ecsr_phase_c_materialize_static_pass.py` | complete checkpoint rewrite smoke |
| Phase-C renderer smoke collector | `scripts/car_model/ecsr_collect_phase_c_renderer_smoke.py` | complete |
| Phase-D attribute-only recovery smoke | `scripts/car_model/ecsr_collect_phase_d_attronly_smoke.py` | complete negative smoke |
| Phase-D surface residual delta MVP | `scripts/car_model/ecsr_apply_surface_residual_delta.py` | complete negative smoke |
| Phase-D residual delta collector | `scripts/car_model/ecsr_collect_phase_d_surface_residual_delta.py` | complete |
| Phase-D constrained attribute collector | `scripts/car_model/ecsr_collect_phase_d_constrained_attr_recovery.py` | complete negative validation |
| Policy-val COLMAP split exporter | `scripts/car_model/ecsr_export_policy_val_colmap_splits.py` | complete |
| Phase-E policy-validated ELA probes | `docs/car_model/5-8-ECSR-PhaseE-PolicyValidatedELAProbes.md` | complete negative validation |

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
| Certificate-Carrying Surface Contraction | partial MVP | 7/21 static pass; 3 contraction candidates materialized; 3/3 renderer smoke pass |
| Surface-Attached Appearance Recovery | partial negative MVP | V1 attribute-only and V2 DC residual delta are executable but rejected by smoke evidence |
| Train/policy-val split file | complete | full-train deterministic split for all 9 scenes |
| Full9 same-protocol ECSR validation | not complete | no final ECSR checkpoint yet |
| Reviewer Objection Audit | partial | leakage/post-processing risks documented; final rebuttal needs Phase C/D |

## Phase-C/D Execution Result

Full-train split is now available for all 9 scenes with seed `20260508`.
Phase-C static topology certification checked `21` preflight candidates and
passed `7`; among them `3` contraction candidates were materialized as real
checkpoint copies (`bicycle_C0001`, `bicycle_C0074`, `kitchen_C0019`). All
three materialized candidates pass one-train-view renderer smoke.

Phase-D produced two executable but currently rejected recovery routes:

- Version 1 attribute-only recovery: topology/vertices frozen, W&B enabled,
  200-step smoke on `bicycle_C0001` and `kitchen_C0019`; `0 / 2` accepted
  because RGB metrics regress vs compact-only.
- Version 2 bounded surface residual DC delta: residual is attached to
  checkpoint `features_dc` and rendered normally; policy-val L1 accepts
  `3 / 4`, but held-out diagnostics regress `4 / 4`, proving that mean
  policy-val L1 is too weak as the sole gate.

This is a useful negative result rather than a final method: the checkpoint
interface for representation-attached recovery is now real, but the accepted
ECSR method must use local-mask policy metrics and a least-squares or learned
residual solve instead of direct top-support DC offsets.

Extended Phase-D validation added three stricter topology-frozen recovery
experiments:

- Constrained attribute recovery V1 on `bicycle`, `flowers`, `treehill`, and
  `garden`: `0 / 4` accepted; all four regress RGB metrics vs compact-only.
- Cache-fixed constrained attribute recovery V2 on the same four scenes:
  teacher/parent render caches load correctly, but still `0 / 4` accepted.
- Policy-val teacher distillation V1/V2/V3 on `flowers` and `garden`: all
  variants keep topology unchanged but regress policy-val RGB metrics. V3 micro
  reduces the regression to near no-op but still fails strict acceptance.

This closes a concrete failure mode: direct topology-frozen feature/SH
continuation from ELA-style teacher renders does not become a reliable
representation-level recovery method under the current objective.

## Phase-E Policy Probe Result

Phase-E tested whether the visual bottleneck could be solved by a stronger
train-only ELA decision policy instead of representation-side recovery.

- Texture-aware benefit gate (`confidence_magnitude_edge`) on four outdoor
  scenes regressed against current Compact-ELA/SOR on every scene.
- Train-fit / train-policy-val policy selection is now implemented, but the
  first `flowers`/`garden` probe is mixed or negative and is not promoted.
- Expanding the ELA alpha grid to `1.5` did not help; calibration still selects
  `alpha=1.0` and reproduces the current method.

Artifacts:

- `docs/car_model/5-8-ECSR-PhaseE-PolicyValidatedELAProbes.md`
- W&B groups: `ecsr_tebg_edge_outdoor_v1`,
  `ecsr_ela_holdout_auto_v1`, `ecsr_ela_alpha150_probe_v1`

Interpretation: the current residual adapter is already close to the available
image-space repair ceiling under the fixed train-only protocol. The next
credible progress path is not another ELA policy tweak; it is cached
certificate-carrying contraction plus a better representation-attached recovery
objective.

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
clean MeshSplatting baselines. The Phase-C preflight has already narrowed the
first expensive candidate set to `21 / 123` eligible clusters.

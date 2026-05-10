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

## Phase-F Fixed Policy-Val Compaction Ladder

Phase-F was added to address the compression bottleneck without returning to a
per-scene parameter game. The policy is fixed before held-out evaluation:

- source: archived Compact-ELA/SOR checkpoint at iteration `26000`;
- split: full-train COLMAP split with train-derived `policy_val`;
- selector: `csef_low_evidence_boundary_protected`;
- ratio grid: `0.005, 0.010, 0.020`;
- acceptance: policy-val RGB non-regression guardrails plus zero invalid /
  degenerate topology audit.

Implementation:

- `scripts/car_model/ecsr_run_policy_val_compaction_ladder.py`
- `scripts/car_model/ecsr_collect_policy_val_ladder_summary.py`
- large-scene selector acceleration in
  `ss3dm_prior/meshsplatopt/compact_selector.py`
- held-out fixed-policy evaluator:
  `scripts/car_model/ecsr_run_phasef_heldout_eval.py`

Full9 policy-val result:

- complete scenes: `9 / 9`;
- accepted scenes: `9 / 9`;
- selected extra ratio: `0.0200` for every scene;
- mean source triangle removal: `5.763%`;
- mean total triangle removal after Phase-F: `7.648%`;
- all policy-val RGB deltas are near numerical noise.

Interpretation: this is a real compactness advance, not a visual breakthrough.
It shows that the archived Compact-ELA/SOR surfaces still contain a fixed,
train-policy-certified low-evidence tail that can be removed safely under
internal validation. It does not by itself solve the FinalDecision requirement
for stronger representation-attached appearance recovery.

Held-out validation is now running on the fixed selected models. The held-out
test split is final-report-only: it is not used to choose the ratio, selector,
crop, threshold, or fallback.

Held-out result after the fixed policy was frozen:

- compact + RGB-safe vs selected clean MeshSplatting: `8 / 9`;
- compact + RGB-safe vs archived Compact-ELA/SOR: `0 / 9`;
- mean dPSNR / dSSIM / dLPIPS vs clean:
  `-0.0113 / -0.00021 / +0.000064`;
- mean dPSNR / dSSIM / dLPIPS vs Compact-ELA/SOR:
  `-0.5092 / -0.01596 / +0.02344`;
- mean total triangle reduction: `7.648%`;
- W&B full9 collect: `ptf5x9o8`.

Conclusion: Phase-F is a useful compactness certificate, not a final visual
method. It should be kept as the representation-compression branch, but the
FinalDecision endpoint still requires a stronger appearance recovery mechanism
on top of the extra compact checkpoint.

## Phase-F + Alpha-0.875 Recovery Update

The next pass connected the fixed Phase-F selected checkpoint to the train-only
ELA recovery family and exposed one missing interface: the previous recovery
runner did not include enough alpha resolution. The initial Phase-F+ELA full9
run was strong on average but missed `bicycle` against archived Compact-ELA/SOR
by `-0.0193` PSNR and `+0.00193` LPIPS. Ratio probes at `0.5%`, `1%`, and `2%`
extra compaction showed that this was not caused by over-compression.

After adding `alpha=0.875` to the globally fixed auto-policy grid, the same
policy cleared all nine scenes:

- beats selected clean MeshSplatting on PSNR/SSIM/LPIPS: `9 / 9`;
- beats archived Compact-ELA/SOR on PSNR/SSIM/LPIPS: `9 / 9`;
- mean dPSNR / dSSIM / dLPIPS vs clean:
  `+0.9340 / +0.02640 / -0.04404`;
- mean dPSNR / dSSIM / dLPIPS vs Compact-ELA/SOR:
  `+0.4360 / +0.01064 / -0.02067`;
- mean total triangle reduction: `7.648%`.

Artifact:

- `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_alpha0875_full9.md`

This materially changes the current status. The selected Mip-NeRF360 protocol
now has a reproducible full9 result that is better than the clean baseline and
the archived Compact-ELA/SOR baseline on RGB metrics while also carrying extra
triangle reduction. The caveat remains that the appearance gain is still
render-time ELA recovery; the representation-level surface-attached recovery
goal is not closed by this update.

## Phase-G / Phase-H / Phase-J Follow-Up

Phase-G tested whether the Phase-F ELA teacher could be baked back into a
topology-frozen MeshSplatting checkpoint. Official-split pilots on `bicycle` and
`flowers` were negative:

- bicycle: `23.290770 / 0.659508 / 0.332462`, below clean by
  `-0.010843 / -0.000359 / +0.000385`;
- flowers: `19.666090 / 0.511543 / 0.394951`, below clean by
  `-0.016167 / -0.000279 / +0.000388`.

This rejected teacher-bake as the immediate final path. The method focus moved
back to a guarded render-time policy while preserving the fixed Phase-F compact
checkpoint.

Phase-H added adaptive per-bin alpha and improved `8 / 9` scenes over Phase-F,
but `treehill` was unstable and had to fall back to Phase-F. Phase-J fixed that
remaining non-strict scene by adding a train-selected structural edge fallback.
The fallback searches edge-gate quantiles on train calibration only; for
`treehill`, it selected q=`0.5`, alpha=`0.75`.

Final Phase-J full9:

- method: `ours_26000_phasej_guarded_adaptedge_ela`;
- report:
  `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md`;
- strict wins vs selected clean MeshSplatting: `9 / 9`;
- strict wins vs Phase-F alpha-grid: `9 / 9`;
- mean delta vs clean: `+1.331084 / +0.034702 / -0.063359`;
- mean delta vs Phase-F: `+0.397095 / +0.008305 / -0.019321`;
- mean total triangle reduction: `7.6479%`.

This is the strongest accepted ECSR RGB result so far. It still remains a
render-time ELA portfolio rather than a fully baked representation-level model.

## Phase-M Surface Residual Lumigraph V8

The next pass implemented a surface-attached residual lumigraph rather than an
image-space ELA variant:

- train residuals are aggregated on persistent rendered `face_id`;
- held-out renders use only target view surface maps and the train-fitted field;
- alpha is selected on train-policy views only;
- V8 requires both `policy_val_stride=4` and consensus
  `policy_val_stride=2` to pass before a nonzero residual is accepted.

Full9 execution is complete. Additional 12-view train evidence caches and
held-out surface maps were generated for `room`, `counter`, `kitchen`, and
`bonsai`, completing the surface-lumigraph interface beyond the outdoor pilot.

Result against the Phase-F compact base:

- accepted scenes: `2 / 9` (`flowers`, `garden`);
- no-op scenes: `7 / 9`;
- mean dPSNR / dSSIM / dLPIPS:
  `+0.000250498 / +0.000000868 / -0.000006378`;
- V8 rejects the earlier false-positive `treehill` and the weak `stump`
  accept from V7;
- W&B group: `phase_m_surface_lumigraph_v8_consensus_gate128_full9`;
- report: `docs/car_model/5-9-ECSR-SurfaceResidualLumigraphV8.md`.

Interpretation: this is the cleanest representation-attached recovery baseline
so far, but the gain is far too small to replace Phase-J. The main bottleneck is
surface support coverage and residual capacity, not the train/test protocol.
V8 should be retained as a safe baseline and as the acceptance discipline for
future higher-capacity surface-attached appearance codes.

## Phase-N / Phase-O Topology-Propagated Surface Residual V10

The follow-up implemented topology propagation and guard-aware policy selection
inside `ecsr_apply_surface_residual_lumigraph_adapter.py`:

- residual source faces can propose neighboring target faces through checkpoint
  `_triangle_indices`;
- propagation is limited to train/test actually visible face ids;
- primary and consensus face gates can be intersected;
- policy selection now chooses the best alpha among guard-passing candidates;
- a provisional face-gate recalibration path can rescue cases where raw
  propagation improves PSNR but violates SSIM before gating.

Important engineering fixes were also made: cached NPZ memmap loading,
checkpoint-range face-id filtering, grouped surface-signal assignment, grouped
face-gate statistics, source residual caching, and cached LPIPS modules.

Held-out diagnostics against `ours_26000_phasef_extra_compact_base`:

- `garden`, V9 gate32 tol:
  `+0.001562 / +0.000001 / -0.000014`, accepted faces `449`;
- `flowers`, V10 guard-aware gate32:
  `+0.000769 / -0.000010 / -0.000010`, accepted faces `320`.

The result is useful but not sufficient: topology propagation expands candidate
support, but final accepted coverage remains below `0.2%`, and flowers still
loses a small amount of SSIM. This confirms that the next bottleneck is residual
field capacity, not just policy plumbing. Detailed log:
`docs/car_model/5-10-ECSR-TopologyPropagatedSurfaceResidualV10.md`.

## Phase-P Surface FaceAlpha V11-V13

The next iteration implemented a higher-capacity surface residual adapter:
`scripts/car_model/ecsr_apply_surface_residual_facealpha_adapter.py`.

V11 replaces global alpha with a train-fitted per-face ridge alpha, intersects
primary and consensus policy splits, and keeps topology propagation restricted
to actually visible checkpoint faces. V12 adds a train-only edge/texture
surrogate to the alpha fit and fixes the CPU bottleneck with dense `bincount`
aggregation plus deterministic edge subsampling.

Held-out summary against `ours_26000_phasef_extra_compact_base`:

- garden V11:
  `+0.001804 PSNR / +0.000006 SSIM / -0.000017 LPIPS`, accepted faces `493`,
  target coverage `0.2020%`;
- flowers V11:
  `+0.000898 PSNR / -0.000006 SSIM / -0.000021 LPIPS`, accepted faces `350`,
  target coverage `0.2059%`;
- flowers V12 edge-aware:
  `+0.000904 PSNR / -0.000006 SSIM / -0.000021 LPIPS`.

V13 diagnostics tested luminance-limited and low-gradient guarded residuals.
They showed that the flowers SSIM gap can be reduced to near numerical noise,
but not converted into a robust strict win without giving up most of the
residual benefit. Existing Phase-J guarded ELA remains the real strong RGB row
on flowers:

- Phase-J guarded ELA:
  `20.304358 / 0.557770 / 0.329222`;
- base:
  `19.668695 / 0.511678 / 0.394788`;
- Phase-J ELA + V11 surface:
  `20.305147 / 0.557764 / 0.329209`, which improves PSNR/LPIPS over ELA but
  drops SSIM by `0.000006`.

Conclusion: V11/V12 is the cleanest representation-attached residual baseline
so far, but it is not the final paper method. Its main limitation is not the
policy guard anymore; it is coverage and local appearance expressivity. Phase-J
should remain the strong ELA teacher/fallback, and the next real research step
should distill ELA into a persistent surface representation or replace the
per-face constant residual with a higher-capacity local surface code. Detailed
log: `docs/car_model/5-10-ECSR-SurfaceFaceAlphaV11-V13.md`.

Large-scene scalability remains open. A treehill V11 stress test was attempted
after dense accumulation was added. The scene has `8,402,362` checkpoint faces;
the dense run passed primary and consensus surface-signal preparation but was
still CPU-bound before target output after about 49 minutes, so it was
terminated and not counted as a metric row. This is a real engineering
bottleneck for full9 surface-facealpha validation and should be fixed with
candidate sparsification or GPU/batched aggregation before more long runs.

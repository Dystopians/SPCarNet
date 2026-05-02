# MeshPrior NeurIPS Paper Roadmap and Claim-Risk Analysis

Date: 2026-05-01

## 1. Title Options

1. Prior Proposes, Evidence Disposes: Object Priors for Safe Scene Mesh Optimization
2. MeshPrior: Object-Centric Shape Posteriors for Gated Scene Mesh Repair
3. Safe Scene Mesh Optimization with Learned Object Shape Priors
4. From Object Completion to Scene Mesh Repair via Proposal-Gated Shape Priors
5. Object Priors as Proposal Generators for Geometry-Aware Scene Mesh Optimization
6. Gated Object Shape Priors for Repairing Real Scene Meshes
7. Learning to Propose, Measuring to Accept: Object-Guided Scene Mesh Repair
8. SP-CarNet MeshPrior: Safe Object-Aware Mesh Optimization for Parking-Lot Scenes
9. Evidence-Gated Shape Priors for Robust Scene Mesh Cleanup
10. Scene Mesh Repair by Object-Posterior Proposals and Rollback Gates

## 2. Central Claim

Learned object-centric shape posteriors can safely guide scene mesh optimization when converted into proposals and filtered by scene evidence gates.

This is the paper's strongest defensible direction. The claim is not that SP-CarNet simply improves object completion Chamfer. The intended contribution is a safety-controlled scene optimization loop:

```text
scene region -> object posterior -> bounded proposal -> scene gate -> accept / reject / rollback
```

The slogan is:

```text
Prior proposes; evidence disposes.
```

## 3. Negative Results That Motivate the Method

The method should be framed as a response to object-only and prior-only failure modes.

| Negative result | Evidence / interpretation | Paper role |
|---|---|---|
| v0.7 residual smoothing collapse | Residual point completion did not produce a robust scene-usable prior. M13 keeps this row as `MISSING` if the old metrics are absent, but M0/M1 document it as the original failure line. | Historical baseline and motivation. |
| v0.8.x point-flow plateau | Available M13 row reports recon Chamfer L1 `0.1231232387131279`, hidden Chamfer L1 `0.1548924239225758`, worse than Stage 3 posterior. | Shows point-space flow is not enough. |
| Stage 2 decoder ceiling | M1 records Stage 2 around `0.066` Chamfer and larger variants not lifting the ceiling. | Explains why object Chamfer alone is not the route. |
| Stage 5 inference reranker failure | M1 records K=8 oracle `0.065528` but inference top-1 reranking `0.073501`. | Shows oracle headroom is not deployable. |
| Chamfer-only limitations | M12 improves proposal safety without optimizing Chamfer; M13 separates object, synthetic, and scene metrics. | Justifies scene metrics and proposal gates. |

The paper should present these as narrowing the research path: the useful prior is not a direct object completion output, but a source of conservative scene proposals.

## 4. Method Summary

### Region Mining

M2 builds dry-run scene/object region mining. It finds connected mesh components, records geometry statistics, and emits `regions.json` without modifying geometry. Known limitation: it is geometry-only unless segmentation artifacts exist.

### Posterior Inference

M3 canonicalizes eligible regions and runs the Stage 3 posterior encoder. It writes latent and diagnostic artifacts per region, handles missing checkpoints clearly, and keeps outputs inference-time only.

### Protect / Prune Proposals

M4 converts posterior field support into triangle-level proposal scores:

- `protect`: valid object-like surface should survive pruning or compaction.
- `prune`: unsupported floaters or prior-inconsistent triangles are removal candidates.

Clean object ground truth is not used to select proposals.

### Snap Proposals

M7 adds bounded vertex movement toward high-confidence prior surfaces. M12 calibrates this with `surface_support_v1`, reducing the snap max displacement from the uncalibrated risky setting and preserving valid-surface protect recall at `0.9166666666666666`.

### Fill Proposals

M8 adds guarded local hole filling. Fill is proposal-only and must pass local topology/free-space checks. The M11 dry-run closed `4.0` boundary edges with no reported free-space delta.

### Scene Gate

M9 implements the first scene-side gate and rollback. Object evidence can tighten or explain a decision but cannot accept a proposal without scene support. The smoke accepts a topology-improving fill and rejects a disconnected floater.

### Rollback

Rollback snapshots preserve before-state mesh arrays and metadata. This is critical for the paper's safety framing: proposal acceptance is reversible and auditable.

### Evaluation Protocol

M13 defines the table structure and matrix runner. Missing experiments remain visible as `MISSING`; oracle-only rows are labeled; dry-run rows are separated from scene training rows.

## 5. Main Figures

| Figure | Content | Required evidence |
|---|---|---|
| Prior proposes / evidence disposes diagram | End-to-end flow from scene region to posterior to proposal to gate to rollback/accept. | Can be drawn now from M2-M10. |
| Object prior posterior visualization | Region points, posterior mean mesh, uncertainty/samples. | Use M3 Stage 3 posterior artifacts. |
| Proposal types figure | Protect, prune, snap, fill on the same synthetic or scene mesh. | M4/M7/M8 artifacts. |
| Accepted/rejected proposal examples | Show accepted local-hole fill and rejected floater with gate reasons. | M9/M11 dry-run gate report. |
| Scene-level before/after | Real parking or COLMAP scene before/after accepted proposals. | Not ready. Needs real render-gated proposal application. |

The first four figures can be prepared from current artifacts. The fifth is mandatory for a strong submission and remains the largest evidence gap.

## 6. Main Tables

Use the M13 report tables as the paper table skeleton:

1. Object prior quality: v0.7, v0.8.2, Stage 3 posterior, Stage 4 MAP, Stage 5 oracle K=8.
2. Synthetic mesh repair: protect/prune, snap calibration, snap+fill.
3. Scene mesh optimization: baseline scene checkpoint, scene + MeshPrior rows.
4. Safety ablation: no gate, prior-only, free-space gate, geometry gate, render gate, full gated method.

Current M13 status:

- `total=11`
- `available=7`
- `missing=4`

Rows currently missing and intentionally visible:

- `v0_7_residual_baseline`
- `spcarnet_stage4_map_refinement`
- `spcarnet_stage5_oracle_k8`
- `protect_prune_proposals`

The current scene table is diagnostic, not headline-ready. The 200-iteration no-cleanup scene smoke has PSNR `6.933581471443176`, SSIM `0.16371289547532797`, LPIPS `0.694071426987648`, COLMAP AbsRel `0.10470779720655764`, normal mean angle `37.51919533010328`, and `5706` triangles. It is useful for plumbing and cleanup repair validation, not for claiming scene improvement.

## 7. Ablations Required Before Submission

| Ablation | Purpose | Current status |
|---|---|---|
| no prior | Establish scene optimizer baseline. | Diagnostic 200-iteration smoke exists; full baseline needed. |
| prior without gate | Show direct insertion is unsafe or worse. | Not run; should be a safety ablation, not a recommended method. |
| free-space gate removed | Quantify hallucination/free-space risk. | M13 table row planned; real evidence missing. |
| geometry gate removed | Show topology/sparse-geometry gate matters. | M9 dry-run indicates topology gate value; real scene evidence missing. |
| render gate removed | Show photometric/render validation matters. | Not implemented. |
| protect/prune only | Isolate low-risk score proposals. | Protect/prune smoke exists; matrix artifact missing. |
| +snap | Measure calibrated bounded movement. | M12 calibration available. |
| +fill | Measure local hole closure and risk. | Synthetic dry-run available. |
| posterior uncertainty removed | Show posterior confidence matters for proposal risk. | Not implemented as a matrix row. |
| retrieval/symmetry calibration if implemented | Test fallback priors for brittle posterior regions. | Future M15/M16 only. |

## 8. Submission Risks

| Risk | Severity | Current evidence | Required mitigation |
|---|---:|---|---|
| Scene scale too small | high | One short COLMAP smoke plus synthetic dry-run. | Run multiple real parking/vehicle scenes with fixed split protocol. |
| Weak scene-level improvement | high | No real accepted MeshPrior scene update yet. | Show AbsRel/normal/car ROI improvement over baseline without render regression. |
| Object prior hallucination | high | Gates reject obvious floater in dry-run. | Add direct-prior/no-gate ablation and accepted/rejected examples. |
| Object-region miner brittle | medium | M2 is geometry-only unless segmentation exists. | Add parking-scene region audit and false-positive counts. |
| Lack of real parking scene metrics | high | Current real scene is a diagnostic video smoke. | Build at least one full real-scene benchmark row. |
| PRISM integration incomplete | high | M5 exports passive artifacts; final cleanup bug was repaired. | Connect accepted proposals to optimizer or present method as optimizer-agnostic with evidence. |
| Novelty looks like engineering | medium | Proposal+gate framing is coherent but must be crisp. | Emphasize safety theorem/contract: prior cannot accept without scene evidence. |
| Generated artifacts not in git | low | M13 handles missing rows. | Keep matrix runner reproducible and reports explicit about `MISSING`. |

## 9. What Result Is Strong Enough for NeurIPS

A credible NeurIPS submission needs at least the following:

1. Scene geometry improves over baseline on COLMAP sparse AbsRel or sparse normal proxy.
   - Target: at least `5%` relative improvement in AbsRel or normal mean angle on two or more scenes, or one larger scene with strong ablations.
2. Rendering does not meaningfully regress.
   - Target: PSNR drop less than `0.2 dB`, SSIM drop less than `0.005`, LPIPS increase less than `0.01`, unless geometry/FPS gains are explicitly traded off.
3. FPS or triangle budget stays controlled.
   - Target: triangle count does not grow by more than `5%`, or FPS does not drop by more than `5%`.
4. Car ROI holes or floaters decrease.
   - Target: boundary-edge/hole metric decreases and disconnected floater count does not increase in car-like regions.
5. Direct prior insertion fails or is risky, but gated proposal succeeds.
   - This is the cleanest evidence for the paper's core idea.
6. Safety ablation shows gates matter.
   - Removing free-space, geometry, or render gates should either degrade metrics or accept proposals that the full method rejects.
7. Accepted/rejected examples are auditable.
   - Each proposal should have a gate report, rollback snapshot, and before/after visualization.

If only one of these categories is strong, the work is not yet a full NeurIPS scene paper.

## 10. What Result Is Not Enough

The following are insufficient for the main claim:

- Object Chamfer improves only.
- Oracle K=8 improves only.
- Qualitative car completions only.
- Synthetic-only repair without scene metrics.
- Proposal scores with no accepted scene-level benefit.
- A 200-iteration training smoke with no MeshPrior application.
- Dry-run topology acceptance without real render or sparse-geometry validation.
- A prior that improves geometry while clearly degrading rendering or speed.

These results can still be valuable as motivation, ablations, or negative results.

## 11. Final Recommendation

Update after M21-M23 on 2026-05-02: recommendation is now `CLAIM_CONSERVATIVE_FRAMEWORK_NOT_FULL_METHOD`.

The current evidence is stronger than the original M14 state because it now includes:

- clean/current/Stage17 7000-iteration single-scene runs with W&B and independent render/geometry metrics;
- M21.5 topology-control ablation showing `prune_50` beats the clean 7000 baseline on render metrics with `416888` triangles;
- M22 paper-evidence package that keeps missing rows and failure cases explicit.

However, the full method claim remains under-evidenced. Stage17 MeshPrior resume is refuted as a long-budget method candidate, and M21.5 is post-hoc checkpoint-copy pruning rather than integrated optimization-time topology control. The paper story should therefore be a conservative proposal/gate/evidence framework with topology-aware diagnostics and honest negative results, unless a second scene or integrated topology controller is added.

Legacy M14 recommendation retained below for provenance.

Recommendation: `MORE_SCENE_EVIDENCE_REQUIRED`.

The direction is coherent and currently stable:

- M2-M13 smoke regression passes.
- The proposal/gate/rollback safety contract is implemented in dry-run form.
- M13 gives a reproducible evaluation matrix and prevents missing or oracle-only rows from contaminating headline claims.
- The non-PRISM final cleanup bug was repaired and verified with a wandb scene smoke.

However, the current evidence is not yet submission-ready:

- real render-gated MeshPrior insertion is not implemented;
- scene-level MeshPrior evidence is still dry-run/gated proposal evidence;
- the real scene run is a diagnostic 200-iteration smoke, not a full training/evaluation benchmark;
- several baseline rows remain `MISSING`;
- no full ablation proves that gates are necessary on real scenes.

The next work should not be another object-prior improvement unless scene integration fails. The highest-value next milestone is a real-scene proposal application loop with render/sparse-geometry gates, rollback, and at least one full baseline-vs-gated-MeshPrior table row.

## 12. Immediate Next Milestones

1. Produce a real scene baseline with fixed split, full checkpoint rule, and wandb logging.
2. Apply accepted MeshPrior proposals to a copy of the scene mesh, with rollback snapshots.
3. Run recovery optimization after accepted proposals.
4. Evaluate COLMAP sparse AbsRel, DepthMAE, normal mean angle, PSNR/SSIM/LPIPS, FPS, and triangle count.
5. Add direct-prior/no-gate and gate-removed ablations.
6. Regenerate M13 tables with real scene rows.
7. Only then revisit the final recommendation.

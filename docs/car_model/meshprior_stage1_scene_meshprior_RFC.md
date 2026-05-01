# MeshPrior Stage 1 RFC — Object-Prior Guided Scene Mesh Optimization

| Field | Value |
|---|---|
| Stage | M1 / scene MeshPrior RFC |
| Date | 2026-05-01 |
| Status | COMPLETE |
| Predecessor | `docs/car_model/meshprior_stage0_repository_audit.md` |
| Scope | Research design only; no model-code changes |

## 0. Executive Summary

SP-CarNet should now pivot from an object-only completion benchmark to a scene-level mesh optimization system:

> **SP-CarNet MeshPrior: learned object-centric shape posteriors as safe proposal generators for scene mesh optimization.**

The method slogan is:

> **Prior proposes; evidence disposes.**

The object prior should not directly overwrite scene geometry. It should propose local mesh operations, and scene evidence should accept, reject, or roll back those operations. The downstream target is the project-level task: optimize real parking-lot scene meshes by protecting valid car geometry, removing floaters, repairing holes, improving triangle distribution, and improving geometry without degrading rendering or speed.

## 1. Current SP-CarNet Status

The M0 audit confirms that Stage 1-5 code, smoke tests, checkpoints, and JSON artifacts are present and usable.

Current object-prior facts:

- Stage 1 object cache and canonicalization audit is operational: 2,433 objects, with object-level points, visible/hidden splits, occupancy queries, surface queries, and free-space queries.
- Stage 2 shape-field auto-decoder is implemented but did not cleanly pass its original chamfer gate. v1 train chamfer is about `0.066`; bigger v3 did not lift the ceiling and reports `recon_chamfer_l1_mean=0.069174`.
- Stage 3 posterior encoder is the strongest deployable object-level result: `recon_chamfer_l1_mean=0.066391`, `free_space_violation_rate_mean=0.033535`, `mesh_extraction_success_rate=1.0`, and zero-corruption chamfer is essentially unchanged from normal chamfer.
- Stage 4 MAP refinement is a safety tool: on 50 val objects it improves free-space violation from `0.035820` to `0.014688`, while chamfer improves modestly from `0.071490` to `0.069032`.
- Stage 5 multi-hypothesis sampling shows oracle headroom but practical reranking failure: K=8 oracle reaches `0.065528`, while inference-time top1 reranking is worse at `0.073501`. It is useful as uncertainty analysis, not a headline method.

Interpretation:

- The project has escaped the v0.7 smoothing-collapse pathology.
- Further object-only Chamfer gains are likely limited by the Stage-2 field representation, not by the posterior encoder.
- The next useful research step is to convert the object posterior into safe scene-level mesh proposals.

## 2. New Central Claim

Learned object-centric shape posteriors can safely guide scene mesh optimization when converted into bounded local proposals and filtered by scene-level evidence gates.

This is deliberately different from the weaker claim:

> SP-CarNet improves car point-cloud completion Chamfer.

The target claim is about scene mesh quality:

- object-region geometry improves,
- valid car-like triangles are protected,
- unsupported floaters are pruned,
- holes are repaired only when evidence supports repair,
- triangle budget is better allocated,
- rendering and sparse geometry metrics do not regress.

## 3. Final Method

The final system has seven layers.

| Layer | Role |
|---|---|
| A. Repository and object-prior integrity | Verify SP-CarNet code, checkpoints, metrics, and smoke tests before scene work. |
| B. Scene/object region mining | Identify car-like and repair-worthy regions in scene meshes without applying priors to ground, walls, vegetation, or ambiguous clutter. |
| C. Object posterior inference | Canonicalize each region and run Stage-3 SP-CarNet posterior inference. |
| D. Mesh repair proposal generation | Generate protect, prune, snap, fill, split, and collapse proposals from the posterior field and uncertainty. |
| E. Scene evidence gates and rollback | Accept proposals only if rendering, sparse geometry, free-space, topology, and budget gates agree. |
| F. Alternating scene optimization | Interleave scene optimizer recovery with proposal generation and validation. |
| G. Evaluation and reporting | Report object-level, synthetic damage, and real scene metrics with strict inference-time/oracle separation. |

The safe operation order is:

```text
protect/prune first -> snap second -> guarded fill third -> split/collapse refinement last
```

## 4. Proposal Types

| Proposal | Definition | Risk level | First allowed stage |
|---|---|---:|---|
| `protect` | Mark valid object-like triangles as high keep-risk during pruning or compaction. | low | M4 |
| `prune` | Mark unsupported floaters or prior-inconsistent triangles as removable candidates. | low to medium | M4 |
| `snap` | Softly move noisy vertices toward a high-confidence object surface. | medium | M7 |
| `fill` | Add local patch geometry for holes only under strict evidence gates. | high | M8 |
| `split` | Allocate more triangles to high-curvature or boundary object regions. | medium | post-M8/M12 |
| `collapse` | Reduce redundant triangles on smooth, well-supported object surfaces. | medium | post-M8/M12 |

No proposal may be selected using clean object ground truth at inference time. Clean shape labels are evaluation-only.

## 5. Why This Serves the Downstream Task

Parking-lot scenes have a different failure profile from object-completion benchmarks:

- car bodies may be under-triangulated or partially missing,
- valid parked-car geometry may be pruned as redundant clutter,
- unsupported floaters can survive because they improve local photometric fit,
- ground and car geometry interact at tires, shadows, and occlusion boundaries,
- speed/triangle budget matters as much as local object Chamfer.

MeshPrior targets these failures directly:

- `protect` keeps valid car surface triangles through PRISM or other compaction.
- `prune` removes object-inconsistent floaters that scene-only metrics may under-penalize.
- `snap` reduces noisy object surfaces without inserting new hallucinated parts.
- `fill` handles holes only when posterior confidence, free-space evidence, and scene gates agree.
- `split/collapse` improves triangle distribution around object boundaries and smooth panels.

The scene-level gates prevent the prior from hallucinating attractive but unsupported cars.

## 6. Relation to Current Code

### Reuse

| Current module / artifact | MeshPrior role |
|---|---|
| `ss3dm_prior/data/spcarnet_object_dataset.py` | Object-level data contract and canonicalization reference. |
| `ss3dm_prior/models/spcarnet_shape_field.py` | Shape-field decoder for support / occupancy queries. |
| `ss3dm_prior/models/spcarnet_posterior.py` | Stage-3 posterior encoder for region inference. |
| `ss3dm_prior/training/spcarnet_posterior.py` | Checkpoint schema and posterior training reference. |
| `ss3dm_prior/losses_spcarnet_observation.py` | Observation/free-space scoring for proposal safety. |
| `scripts/car_model/refine_spcarnet_latent_map.py` | MAP refinement pattern for proposal validation. |
| `scripts/car_model/eval_spcarnet_multihypothesis.py` | Posterior uncertainty and K-sampling analysis, not headline selection. |
| `train.py` PRISM / ground hooks | Scene-level optimization, rollback, and validation targets. |
| `evaluate_geometry_colmap.py` | Sparse scene geometry gate. |
| `create_colmap_outoftrain_split.py` | Strict scene split protocol. |
| `mesh.py`, `create_ply.py`, `render.py`, `metrics.py` | Scene artifact export and evaluation. |

### Demote to Baselines

| Existing line | MeshPrior status |
|---|---|
| v0.7 residual decoder | Baseline and negative result: smoothing-collapse floor. |
| v0.8.x point-flow | Baseline and negative result: point-space flow plateau. |
| Stage 2 auto-decoder | Decoder ceiling / object prior component, not final scene method. |
| Stage 5 score-based reranker | Negative ablation; oracle analysis only. |

### New Modules Needed

```text
ss3dm_prior/meshprior/
scripts/car_model/meshprior_*.py
configs/ss3dm_prior/meshprior/
docs/car_model/meshprior_*.md
outputs/carnet/meshprior/
```

Initial modules should be:

- region mining,
- scene-region posterior inference,
- protect/prune proposal scoring,
- optimizer adapter,
- synthetic damage benchmark,
- snap/fill proposal modules,
- scene gate and rollback,
- alternating runner,
- evaluation/report matrix.

## 7. Research Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Object prior hallucination | Adds plausible but false geometry. | Prior only proposes; scene gates decide. |
| Bad object-region mining | Applies car prior to ground, walls, vegetation, or non-car objects. | Conservative car-likeness thresholds, dry-run mode, reject ambiguous regions. |
| Canonical-frame mismatch | Posterior field is rotated/scaled incorrectly. | Persist transforms, report orientation confidence, avoid snap/fill under low confidence. |
| Weak shape-field surface accuracy | Support scores are too blurry for snap/fill. | Start with protect/prune; add calibration/SDF only if proposal metrics demand it. |
| Proposal acceptance too conservative | No scene-level changes accepted. | Treat rejected proposals as safety evidence; tune thresholds using synthetic benchmark first. |
| Proposal acceptance too aggressive | Rendering or sparse geometry regresses. | Rollback on PSNR/MAE/AbsRel/normal/topology regression. |
| Mesh-splatting render quality degradation | Object geometry improves but image metrics fall. | Rendering gate and recovery fine-tune before accepting headline claims. |
| Scene gate cost too high | Method becomes impractical. | Use dry-run topology/free-space gates first, then sparse validation, then full render gates only for candidates. |

## 8. Kill Criteria for Future Stages

| Stage | Hard gate |
|---|---|
| M2 region mining | Synthetic smoke passes; dry-run without segmentation exits cleanly; no car prior is emitted for clearly invalid regions. |
| M3 region posterior | Imports pass; missing checkpoint fails clearly; with checkpoint, at least one region posterior artifact is produced. |
| M4 protect/prune | Synthetic valid surface gets higher protect score and floater gets higher prune score; no geometry is moved. |
| M5 optimizer adapter | Exports reloadable JSON/NPZ; bounded combination cannot dominate base optimizer scores. |
| M6 synthetic benchmark | Damage generation works; protect/prune identifies floaters and preserves valid surfaces; inference/oracle metrics are separated. |
| M7 snap | Snap reduces synthetic surface distance without moving protected/boundary vertices beyond max displacement or increasing free-space violation. |
| M8 fill | Fill closes a controlled synthetic hole without disconnected floaters or free-space regression; otherwise fill is killed. |
| M9 scene gate | Dry-run gate accepts an obviously improving proposal, rejects an obvious floater, and rollback restore works. |
| M10 pipeline | Synthetic dry-run completes end-to-end and does not modify geometry unless explicitly enabled. |
| M11 scene experiment | Real or dry-run scene experiment produces metrics; regressions are reported, not hidden. |
| M12 prior calibration | Proposal-relevant metrics improve without harming free-space safety; object Chamfer alone is insufficient. |
| M13 eval protocol | Missing experiments are marked `MISSING`; tables generate without invalid oracle/headline comparisons. |
| M14 paper roadmap | Final recommendation is explicit and tied to scene-level evidence. |

If a hard gate fails, write the failure report and stop that branch.

## 9. Evaluation Policy

All reports must separate:

```text
inference_time_metrics
oracle_analysis_metrics
gt_dependent_eval_metrics
```

Headlines may use only inference-time or scene-evaluation metrics. Oracle best-of-K and clean-shape selection are allowed only as analysis.

Primary scene metrics:

- COLMAP sparse AbsRel,
- sparse DepthMAE,
- sparse normal mean angle,
- PSNR / SSIM / LPIPS / MAE when render eval exists,
- controlled render time or FPS,
- triangle count,
- car-region hole/floater metrics,
- free-space violation,
- accepted/rejected proposal counts.

## 10. Documentation Policy

Every MeshPrior stage must leave:

```text
docs/car_model/meshprior_stageX_<topic>_design.md
docs/car_model/meshprior_stageX_<topic>_implementation_report.md
docs/car_model/meshprior_stageX_<topic>_smoke.md
```

If a stage fails or a key hypothesis is falsified, also write:

```text
docs/car_model/meshprior_stageX_<topic>_failure.md
```

Every stage must append a dated entry to:

```text
docs/car_model/SPCarNet_research_log.md
```

## 11. Decision

M1 is complete and internally consistent.

Recommendation:

```text
PROCEED_TO_M2_REGION_MINING
```

Do not implement snap, fill, scene gates, or optimization runners before the region mining layer passes.

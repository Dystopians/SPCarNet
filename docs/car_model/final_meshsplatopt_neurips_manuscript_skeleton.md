# MeshSplatOpt NeurIPS Manuscript Skeleton

Decision: `FINAL_F14_MANUSCRIPT_SKELETON_PASS_HONEST_COMPACT_REPAIR_STORY`.

## Title

MeshSplatOpt: Counterfactually Certified Compact-Repair Optimization for Mesh Splatting

## Abstract

Mesh Splatting can produce strong view synthesis quality, but high-quality scene checkpoints often contain far more topology than needed and naive local mesh repair can damage rendering. MeshSplatOpt treats scene optimization as a counterfactually certified compact-repair problem. Starting from a clean long-run Mesh Splatting checkpoint, it estimates surface evidence, applies evidence-compatible topology compaction, freezes topology updates, and recovers appearance and sparse geometry. Across five validated scenes, MeshSplatOpt reduces triangle counts by 40-70 percent while matching or improving clean-long render quality under independent evaluation. Negative ablations show that snap/fill edits and aggressive continuation are not yet load-bearing headline methods; the current robust contribution is compact-recovery with auditable rollback.

## 1. Introduction

- Overcomplete Mesh Splatting checkpoints are high quality but topology-heavy.
- Naive pruning, snap, fill, and teacher/distillation attempts produced several failures.
- The final successful path is clean-long training followed by evidence-compatible compaction and strict topology-frozen recovery.
- Main empirical claim: compact-recovery finds better quality/topology Pareto points than the clean long baseline on parking, bonsai, courtyard, room, and counter.
- Scope limitation: local snap/fill are implemented and safety-gated, but not promoted as headline quality wins.

## 2. Related Work

- 3D Gaussian and mesh-based splatting for novel-view synthesis.
- Mesh simplification, QEM, and posthoc decimation.
- Geometry-aware neural rendering regularization and sparse COLMAP depth supervision.
- View-consistency and counterfactual edit validation.
- Classical hole filling and mesh repair, contrasted with triangle-soup checkpoints where edge-loop assumptions are invalid.

## 3. Method

### 3.1 Mesh Splatting Background

Define the scene checkpoint, triangles, vertices, radiance parameters, optimization loop, and independent render evaluation.

### 3.2 Counterfactual Surface Evidence Field

CSEF records positive surface support, free-space risk, explanation debt, prior support, topology cost, and uncertainty. In the current validated branch, CSEF is used to select compactable low-evidence regions while preserving boundary/evidence-protected regions.

### 3.3 Reversible Edit Calculus

Supported operations include delete/prune, collapse, snap, split, fill, and appearance recovery. Every edit must have rollback snapshots and exact metadata. Current headline uses delete/compaction; snap/fill remain auxiliary safety branches.

### 3.4 Evidence-Compatible Compaction

Apply topology reduction to clean long checkpoints using conservative evidence and topology constraints. Parking uses the accepted area70 Pareto row; public scenes use CSEF boundary-protected compaction, with counter selecting CSEF40 because CSEF50 is a boundary case.

### 3.5 Counterfactual Certification

Render/geometry gates reject edits that damage changed pixels, sparse geometry, free-space consistency, or topology accounting. Existing evidence supports the mechanism; full no-gate ablations remain a reviewer-risk item.

### 3.6 Strict Topology-Frozen Recovery

After compaction, training resumes with topology updates frozen and sparse COLMAP geometry guidance enabled. This prevents topology re-inflation and isolates recovery from hidden densification.

### 3.7 Optional Repair Branches

Snap and fill are described as implemented but not headline-validated. The paper should include them as future-capable edit types and report negative findings.

## 4. Experiments

### 4.1 Datasets

- `parking_phone_tiny`
- `mipnerf360/bonsai`
- `eth3d_colmap/courtyard`
- `mipnerf360/room`
- `mipnerf360/counter`

### 4.2 Baselines

- scene-matched clean long checkpoint;
- Stage35/PRISM where available;
- sparse-depth-only recovery where available;
- compaction-only and aggressive compaction diagnostics;
- failed snap/fill controls as negative evidence.

### 4.3 Metrics

Use independent `render.py` outputs and independent image/geometry evaluation. Main tables must not use training-time metrics.

### 4.4 Main Results

Use `docs/car_model/final_stageF12_multiscene_package_report.md`. Five scenes have compact-recovery pass decisions. Triangle reduction ranges from 40 percent on counter to 70 percent on parking.

### 4.5 Pareto Curves

Show quality versus triangle count. Include parking R48/R53/R55, bonsai CSEF50/CSEF70, and counter CSEF40/CSEF50/CSEF50-30k.

### 4.6 Ablations

Use `docs/car_model/final_stageF11_ablation_suite_report.md`. The current strict ablation gap is area-only versus CSEF across all public scenes, random same-count compaction, no-sparse-depth, no-freeze, and no-gate controls.

### 4.7 Qualitative Results

Use assets from `outputs/carnet/meshsplatopt/final_paper_assets/`.

### 4.8 Limitations

- CSEF is strong as a compact-recovery selector but not yet fully separated from area/topology heuristics on every scene.
- Snap/fill are not final quality winners.
- Geometry metrics are COLMAP sparse proxies, not dense ground truth.
- Counter requires a gentler 40 percent operating point.

## 5. Conclusion

MeshSplatOpt should be presented as an evidence-certified compact-repair optimizer. The strongest contribution is not arbitrary local repair; it is auditable movement along the quality/topology Pareto frontier using clean-long initialization, certified compaction, and topology-frozen recovery.


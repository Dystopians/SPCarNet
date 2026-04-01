# PRISM GeoGate Round2 Experiment Report (2026-04-01)

## 1) Goal And Expectation

This round targets three high-information runs on `parking_phone_tiny_anonymized`:

- `PRISM-GeoGateFix`
- `PRISM-LatePrune`
- `PRISM-GeoGateFixKeep`

Expected outcome:

1. Keep or improve image quality vs previous baseline.
2. Preserve or improve official geometry (COLMAP sparse proxy).
3. Improve speed after pruning (or at least avoid severe degradation at final checkpoint).

---

## 2) Experiment Setup

- Dataset: `/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix`
- Split: `/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse/0/split_outoftrain_v1.json`
- Iterations: `30000`
- Saved checkpoints: `15000, 16000, 18000, 20000, 21000, 24000, 30000`
- WandB project: `mesh-splatting-prune`
- Run group: `parking_phone_tiny_geogate_round2_retry`

Case-level differences (from `scripts/parking_ground/run_case.sh`):

- `GeoGateFix`: sparse COLMAP depth loss on; keep/dilation effectively off.
- `LatePrune`: pruning starts later (`geometry_acq_until_iter=18000`), no sparse depth loss.
- `GeoGateFixKeep`: sparse depth loss on + keep thresholds + dilation enabled.

---

## 3) Core Results (Final 30000)

Reference baseline (previous stable run, same dataset family):

- PSNR `15.1331`, SSIM `0.5315`, LPIPS `0.4936`
- official geometry: AbsRel `0.03484`, MeanAngle `36.7059`, DepthMAE `0.4290`
- test FPS (training-time report) `33.41`

Round2 final results:

| Run | PSNR | SSIM | LPIPS | FPS (training-time) | Triangles | AbsRel | MeanAngle | DepthMAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PRISM-GeoGateFix | 15.3126 | 0.5300 | 0.4900 | 33.29 | 10,556,456 | 0.03190 | 35.4911 | 0.4527 |
| PRISM-LatePrune | 15.4416 | 0.5365 | 0.4858 | 15.97 | 9,661,586 | 0.03795 | 37.8007 | 0.5327 |
| PRISM-GeoGateFixKeep | 15.2713 | 0.5304 | 0.4909 | 15.33 | 10,661,568 | 0.03479 | 35.2191 | 0.4903 |

Interpretation:

- Image quality: all three are at least competitive; `LatePrune` is strongest on PSNR/SSIM/LPIPS.
- Geometry:
  - `GeoGateFix` is best overall this round on official geometry (best AbsRel among all three; good angle).
  - `LatePrune` geometry is clearly worse (AbsRel/MeanAngle/DepthMAE all degraded).
  - `GeoGateFixKeep` improves angle but not depth MAE enough.
- Speed:
  - training-time FPS for `LatePrune` and `GeoGateFixKeep` collapses at 30000.
  - this large gap is **not fully reliable as a fair cross-run speed metric** (see section 6).

---

## 4) Mid-Stage (20000) Check

Official geometry at 20000:

| Run | AbsRel@20000 | MeanAngle@20000 | DepthMAE@20000 |
|---|---:|---:|---:|
| PRISM-GeoGateFix | 0.03164 | 37.0806 | 0.4589 |
| PRISM-LatePrune | 0.03757 | 38.1963 | 0.5258 |
| PRISM-GeoGateFixKeep | 0.03391 | 37.0904 | 0.5041 |

Key observation:

- `GeoGateFix` is consistently better on depth-related geometry from 20000 to 30000.
- `LatePrune` remains the weakest on official geometry at both 20000 and 30000.
- `GeoGateFixKeep` shows some angle benefit but still weaker than `GeoGateFix` on depth MAE.

---

## 5) Incremental Impact Analysis (What Each Delta Caused)

### 5.1 GeoGateFix (official geometry gate + sparse depth loss)

Positive:

- Reaches strong geometry balance for this round.
- Outperforms baseline on image metrics while keeping geometry competitive or improved (AbsRel/MeanAngle).

Negative:

- Triangle count remains high at final (`~10.56M`), limiting final speed upside.

Conclusion:

- This is the most aligned with the round objective ("quality + geometry safety").

### 5.2 LatePrune (later start)

Positive:

- Best image metrics in this round.

Negative:

- Geometry regresses significantly (AbsRel/MeanAngle/DepthMAE all worse than `GeoGateFix`).
- Final speed report is poor.

Likely mechanism:

- Delaying prune lets high-complexity topology persist longer; image fit improves, but geometry/efficiency tradeoff worsens by final stage.

### 5.3 GeoGateFixKeep (keep + dilation)

Expected:

- Should protect geometry-critical triangles better than `GeoGateFix`.

Observed:

- No clear net gain over `GeoGateFix`; depth MAE is worse.

Critical clue from WandB summaries:

- `prism/geometry_keep_mean = 0`
- `prism/orientation_keep_mean = 0`

This means keep-related signals were effectively inactive in this round (or always clamped out), so the intended keep increment did not materially contribute.

---

## 6) Why The FPS Looked "Too Bad"

The training-time `test/fps` values are not a strict apples-to-apples throughput benchmark across independent long runs:

- They are collected during training, influenced by current GPU load and concurrent activity.
- They can be distorted by phase transitions, memory state, and run-specific contention.

A same-machine sequential re-render check at iteration 30000 gives:

- `GeoGateFix`: `0.571` views/s
- `LatePrune`: `0.633` views/s
- `GeoGateFixKeep`: `0.591` views/s

Absolute numbers include full render pipeline overhead (camera loading + IO), so they are not deployment FPS, but relative order does **not** match the extreme training-time FPS gap.  
Therefore, the 30000 training-time FPS collapse should not be treated as final speed truth by itself.

---

## 7) Current Problems Exposed By This Round

1. **Keep branch effectiveness is near zero**
   - keep statistics indicate inactive signals (`geometry_keep_mean`, `orientation_keep_mean` near zero).
   - `GeoGateFixKeep` did not beat `GeoGateFix` as designed.

2. **LatePrune over-optimizes image, underperforms geometry**
   - clear regression in official depth/normal proxy metrics.

3. **Final topology is still too heavy**
   - all three runs remain near 9.6M-10.7M triangles at 30000.
   - speed upside from pruning is smaller than expected.

4. **Evaluation pipeline robustness gap**
   - the all-checkpoint benchmark did not complete in one shot under the current environment invocation, causing missing consolidated artifacts unless manually re-run.

---

## 8) What To Fix Next (Concrete)

### A) Restore keep signal effectiveness (highest priority)

- Audit `compute_prism_scores` path for keep terms:
  - verify non-zero inputs for sparse support/structure/render keep.
  - verify thresholds are reachable after normalization.
  - verify keep affects candidate eligibility before sorting.
- Add hard debug assertions/logs:
  - fraction of triangles with keep > 0
  - top-K keep triangles and why they are/aren't filtered.

Target: `GeoGateFixKeep` should show non-zero keep means and measurable geometry protection.

### B) Retune LatePrune for geometry safety

- Shorten delay (`geometry_acq_until_iter` back toward 16000).
- Keep candidate rounds low but raise geometry rejection strictness.
- Prefer `GeoGateFix` schedule as base and only minimally delay prune.

Target: keep image gains without sacrificing official geometry.

### C) Make speed comparison fair and reproducible

- Use a dedicated, post-training benchmark script for FPS under controlled GPU conditions.
- Do not compare only training-time `test/fps` across separate long runs.

Target: avoid false speed conclusions from runtime noise.

### D) Reduce final topology pressure

- Slightly increase effective prune (or add one guarded extra candidate round) only if geometry gate passes.
- Keep rollback safety enabled.

Target: bring final triangle count down while staying within geometry tolerance.

---

## 9) Recommended Next Run Plan

1. **Fix keep-path activation first** (code-level verification + debug stats).
2. Re-run only two focused cases:
   - `GeoGateFix` (control)
   - `GeoGateFixKeep-v2` (keep fixed)
3. Compare at checkpoints `20000` and `30000` with official geometry + controlled FPS benchmark.
4. If keep-v2 works, then revisit `LatePrune` as a third-stage ablation.

---

## 10) Final Judgment For This Round

- **Partially meets expectation**:
  - Image quality improvements were achieved.
  - One run (`GeoGateFix`) gives reasonable geometry and stable behavior.
- **Does not fully meet expectation**:
  - Speed improvement is not convincingly demonstrated at final checkpoint.
  - `LatePrune` geometry is below target.
  - `GeoGateFixKeep` increment is not effective yet because keep signals appear inactive.

Overall, this round is a valid diagnostic success: it identified exactly where the current bottlenecks are and what must be fixed before the next claim of "quality + geometry + speed" can be considered achieved.


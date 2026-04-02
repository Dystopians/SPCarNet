# PRISM Compaction Round Experiment Report (2026-04-01)

## 1) Goal And Expected Outcome

This round extends the previous `GeoGateFix` line toward a more complete story:

- retain the positive image + official-geometry behavior of `GeoGateFix`
- restore keep/protect effectiveness at scale
- add a geometry-safe compaction stage so final triangle count can go down further
- compare against a fresh plain baseline on the same split

Planned core runs:

- `Baseline`
- `PRISM-GeoGateFix`
- `PRISM-GeoGateFixKeep`
- `PRISM-GeoGateCompact`

Expected outcome:

1. `GeoGateFix` should remain the stable geometry-safe PRISM control.
2. `GeoGateFixKeep` should outperform or at least match `GeoGateFix` on geometry once keep/protect is truly active.
3. `GeoGateCompact` should reduce final triangle count and improve runtime proxy while staying close to `GeoGateFix` on official geometry.
4. All PRISM runs should complete stably through the `16000+` transition into PRISM scoring / stats-heavy stages.

---

## 2) Experiment Setup

- Dataset: `/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix`
- Split: `/data2/peilincai/parking_phone_tiny_anonymized/colmap_undistorted_fix/sparse/0/split_outoftrain_v1.json`
- Iterations: `30000`
- Saved checkpoints: `15000, 16000, 18000, 20000, 21000, 24000, 30000`
- WandB project: `mesh-splatting-prune`
- Run group: `parking_phone_tiny_compaction_round`
- Run tag: `parking_phone_tiny_compaction_round_20260401_153515`

Case definitions:

- `Baseline`
  - no PRISM
  - no sparse COLMAP depth supervision
  - no keep/protect
  - no compaction

- `PRISM-GeoGateFix`
  - PRISM enabled
  - official geometry gate enabled
  - sparse COLMAP depth supervision enabled
  - keep thresholds effectively off
  - compaction off

- `PRISM-GeoGateFixKeep`
  - same base as `GeoGateFix`
  - keep/dilation path enabled
  - intended to protect geometry-critical / ROI / neighborhood triangles

- `PRISM-GeoGateCompact`
  - same base as `GeoGateFix`
  - geometry-safe compaction stage enabled after the main training process

Important implementation state during this run:

- This run used the first version of the scalable heavy-eval path.
- That version still built a global structure cache before local heavy evaluation.
- This became the main failure source of the round.

---

## 3) Run Status Summary

### 3.1 Completion Status

| Run | Status | Last observed stage |
|---|---|---|
| Baseline | completed | full `30000` |
| PRISM-GeoGateFix | failed | crashed at `16010` |
| PRISM-GeoGateFixKeep | failed | crashed at `16010` |
| PRISM-GeoGateCompact | failed | crashed at `16010` |

Observed failure signature:

- `GeoGateFix`: `Segmentation fault (core dumped)` right after the `16000 -> 16010` transition
- `GeoGateFixKeep`: same failure pattern
- `GeoGateCompact`: same failure pattern, with even stronger slowdown before crash

This is not a generic training failure:

- the plain baseline completed normally
- the PRISM runs all failed at the same stage boundary
- therefore the failure is strongly tied to the PRISM heavy-eval / score-refresh path

---

## 4) Baseline Results (This Round)

### 4.1 Test-Curve Trajectory

Fresh baseline run:

| Iter | PSNR | SSIM | LPIPS | L1 | FPS |
|---|---:|---:|---:|---:|---:|
| 15000 | 15.5632 | 0.5539 | 0.4780 | 0.11878 | 26.68 |
| 16000 | 15.6140 | 0.5575 | 0.4742 | 0.11773 | 26.70 |
| 17000 | 15.6602 | 0.5616 | 0.4699 | 0.11674 | 26.86 |
| 18000 | 15.6385 | 0.5570 | 0.4688 | 0.11738 | 26.94 |
| 19000 | 15.6546 | 0.5566 | 0.4669 | 0.11718 | 27.00 |
| 20000 | 15.6923 | 0.5629 | 0.4666 | 0.11646 | 21.98 |
| 21000 | 15.6933 | 0.5611 | 0.4660 | 0.11639 | 22.11 |
| 22000 | 15.6922 | 0.5595 | 0.4668 | 0.11634 | 22.30 |
| 23000 | 15.6805 | 0.5570 | 0.4692 | 0.11641 | 22.55 |
| 24000 | 15.6747 | 0.5559 | 0.4716 | 0.11635 | 22.83 |
| 25000 | 15.6514 | 0.5530 | 0.4734 | 0.11660 | 14.28 |
| 26000 | 15.6189 | 0.5499 | 0.4749 | 0.11727 | 14.43 |
| 27000 | 15.6058 | 0.5490 | 0.4771 | 0.11729 | 14.65 |
| 28000 | 15.5727 | 0.5454 | 0.4798 | 0.11767 | 14.89 |
| 29000 | 15.5151 | 0.5390 | 0.4826 | 0.11863 | 15.14 |
| 30000 | 15.4701 | 0.5346 | 0.4844 | 0.11913 | 15.85 |

### 4.2 Interpretation

This baseline does **not** support the statement that "near 30000 the result becomes much better."

Instead, it shows:

1. strong steady improvement from `15000` to roughly `20000-21000`
2. a broad peak around `20000-22000`
3. clear regression from `24000` onward

This is consistent with the earlier observation that the natural best test checkpoint on this dataset often appears around:

- `15000`
- `16000`
- `20000`
- `21000`

rather than at the final `30000`.

### 4.3 What This Means For Acceptance

The fresh baseline reinforces two points:

1. "final checkpoint" is not the same as "best checkpoint"
2. any PRISM acceptance story must be judged against the baseline not only at `30000`, but also around the best natural mid-late checkpoints

---

## 5) PRISM Results Before Failure

The three PRISM runs did not reach final benchmark, so only pre-crash signals are available.

### 5.1 Official Validation Proxy At `16000`

These values come from `prism_validation/validation_iter_016000.json`.

| Run | PSNR | MAE | AbsRel | Delta<1.25 | DepthMAE | MeanAngle | AbsCos | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GeoGateFix | 19.7535 | 0.07070 | 0.03398 | 0.96673 | 0.76792 | 42.4487 | 0.6708 | pass |
| GeoGateFixKeep | 19.7403 | 0.07066 | 0.03480 | 0.96630 | 0.80322 | 42.4404 | 0.6714 | pass |
| GeoGateCompact | 19.7685 | 0.07060 | 0.03584 | 0.96462 | 0.82118 | 42.3320 | 0.6736 | pass |

### 5.2 Relative Interpretation At `16000`

- `GeoGateFix`
  - best depth-side geometry among the three at this point
  - best `AbsRel`
  - best `DepthMAE`
  - solid control behavior

- `GeoGateFixKeep`
  - very close to `GeoGateFix`
  - slightly worse on depth metrics
  - slightly better on angle than its own earlier checkpoints
  - no clear evidence yet that keep/protect had produced a decisive gain

- `GeoGateCompact`
  - best image-side metrics at this point (`PSNR`, `MAE`)
  - best `MeanAngle` and `AbsCos`
  - but clearly worse on `AbsRel`, `Delta<1.25`, and `DepthMAE`
  - already shows the classic early sign of "image / angle improvement with depth-side geometry tradeoff"

### 5.3 Important Positive Finding

All three PRISM runs at `16000` still had:

- observable geometry
- enough depth matches
- enough normal matches
- validation-gate pass
- stage-best update accepted under the new geometry-first rule

So the new official-geometry gate itself was **not** the failure point.

The crash happened **after** geometry was already being evaluated successfully.

---

## 6) Main Failure Analysis

### 6.1 Direct Symptom

All three PRISM variants show the same pattern:

1. training is stable until about `16000`
2. right after the first heavy PRISM scoring transition, step time spikes hard
3. the process then dies with `Segmentation fault (core dumped)`

Examples:

- `GeoGateFix`: normal speed up to `16000`, then `16010` slows to about `3.81 it/s`, then segfault
- `GeoGateFixKeep`: similar, drops to about `4.03 it/s`, then segfault
- `GeoGateCompact`: slows even more drastically to about `1.60 it/s`, then segfault

### 6.2 Most Likely Root Cause

The first version of the scalable heavy-eval path still did the following:

1. enter large-topology PRISM score recomputation after `geometry_acq_until_iter=16000`
2. build a **global** structure cache for the full triangle set
3. then only use a small local subset for true heavy metrics

That design is internally inconsistent:

- it claims to be scalable
- but still pays a huge global topology-construction cost at the first heavy-eval transition

At the current scale, and with three PRISM jobs entering that path concurrently, the result was:

- extreme slowdown
- then native crash in the structure/adjacency-related path

### 6.3 Why Baseline Survived

Baseline does not enter any of these paths:

- no PRISM score refresh
- no scalable heavy-eval
- no structure cache building
- no counterfactual pruning
- no post-16000 PRISM phase transition

So it continued normally while the PRISM runs all failed in the same place.

### 6.4 Why This Is "Repeating The Same Mistake"

The earlier problem pattern was:

- large-topology PRISM paths look logically safe
- but hidden heavy structure / sparse-support computations still re-enter a global expensive path
- that path only becomes active once the PRISM phase changes
- crashes happen not at startup, but right at the first real heavy-eval boundary

This round repeated that pattern almost exactly at `16000+`.

---

## 7) Comparison Against Previous Stable Standards

The previous stable reference report used:

- `baseline`: `pfull_geom_first_no_prism`
- `prism`: `pfull_geom_first_full_prism`
- `prism_ground`: `pfull_geom_first_full_prism_ground_protect`

Their final official benchmark at `30000` was:

| Run | PSNR | SSIM | LPIPS | DepthMAE | AbsRel | Delta<1.25 | MeanAngle | AbsCos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 15.1331 | 0.5315 | 0.4936 | 0.4290 | 0.0348 | 0.9679 | 36.7059 | 0.7372 |
| prism | 15.2789 | 0.5311 | 0.4943 | 0.4575 | 0.0349 | 0.9691 | 37.2540 | 0.7331 |
| prism_ground | 15.2694 | 0.5295 | 0.4936 | 0.4976 | 0.0364 | 0.9655 | 36.7715 | 0.7336 |

### 7.1 What Those Older Results Mean For Current Judgment

- The older PRISM variants had already demonstrated that:
  - stable full-run completion was possible
  - image/fps gains were plausible
  - geometry was not yet uniformly better than baseline

- The current compaction round was expected to improve the "speed / triangle-count" story further.

- Instead, the current round regressed on the most basic engineering acceptance requirement:
  - the new PRISM variants did not even finish

So compared with the earlier stable standard, this round is currently **behind** on engineering robustness.

---

## 8) Gap Between Actual Results And Expectation

### Expected

1. `GeoGateFix` remains stable and geometry-safe.
2. `GeoGateFixKeep` activates real keep signals at scale and improves geometry safety.
3. `GeoGateCompact` adds a safe post-training compaction phase and lowers final triangle count.
4. The whole suite completes so final benchmark can compare:
   - quality
   - official geometry
   - speed

### Actual

1. `Baseline` completed and again showed a strong late-mid checkpoint around `20000-21000`.
2. All three PRISM variants crashed at the first heavy-eval transition.
3. No final benchmark exists for the new PRISM variants.
4. Compaction-stage effectiveness cannot yet be judged because the runs never reached compaction.

### Practical Gap

The gap is therefore not "small parameter mismatch."

It is:

- first an engineering failure of the new scalable scoring implementation
- then, only after that is fixed, an algorithmic question about whether keep/compact really improve the quality-geometry-speed tradeoff

Right now, the algorithmic story is still blocked by engineering instability.

---

## 9) Current Technical Judgment

### 9.1 On GeoGateFix

`GeoGateFix` still looks like the most trustworthy control branch.

Evidence:

- best depth-side geometry among the three at `16000`
- validation gate passes
- stage-best updates proceed correctly under geometry-first logic

So the intended role of `GeoGateFix` as the stable PRISM control still appears correct.

### 9.2 On GeoGateFixKeep

This run did not finish, so there is still no complete evidence that the keep branch now beats `GeoGateFix`.

At `16000`, it is close to `GeoGateFix`, but not clearly better.

Current judgment:

- promising, but not yet demonstrated
- still needs a completed run before any claim can be made

### 9.3 On GeoGateCompact

The compaction branch showed the strongest slowdown before crashing.

At `16000`, it had:

- best image metrics among the three
- best angle-side geometry
- weakest depth-side geometry

That pattern suggests the compaction base schedule may already be more aggressive, even before actual compaction begins.

Current judgment:

- too early to claim success
- likely to need stronger geometry-side conservatism after the crash fix is validated

---

## 10) Fixes Already Applied After This Failure

After diagnosing the crash, the current branch has already been patched:

- large-topology two-stage heavy-eval no longer builds a global structure cache first
- large-topology mode now stays local-only for structure evaluation
- local submesh structure metrics are computed only on the selected heavy subset
- neighborhood mapping for protection is built from the local subset, not a global cache

This is the correct direction, but it is still a **post-mortem code fix**.

It has not yet been validated by a fresh rerun of the full compaction suite.

Therefore:

- the report should treat the current round as failed
- the new patch should be treated as a hotfix candidate, not yet as a confirmed solution

---

## 11) Recommended Next Experiment Plan

### 11.1 Immediate Priority

Re-run only the compact-round core suite after the heavy-eval hotfix:

- `Baseline`
- `PRISM-GeoGateFix`
- `PRISM-GeoGateFixKeep`
- `PRISM-GeoGateCompact`

### 11.2 Acceptance Criteria For The Rerun

1. All PRISM runs must pass the `16000+` transition without segfault.
2. `GeoGateFix` must finish completely and serve as the new stable PRISM control.
3. `GeoGateFixKeep` must complete and show whether keep/protect actually beats or matches `GeoGateFix`.
4. `GeoGateCompact` must complete and demonstrate whether triangle count can be reduced without obvious official-geometry regression.

### 11.3 Evaluation Standard For The Rerun

Use the same standard as before:

- training-time `test` trends
- official geometry proxy
- final benchmark
- compare not only at `30000`, but also around:
  - `15000`
  - `16000`
  - `20000`
  - `21000`

because the fresh baseline again confirms that these checkpoints are often more informative than final `30000`.

---

## 12) Final Conclusion

This round should be judged as:

- a **diagnostically valuable failure**
- not a successful algorithmic result

What was learned:

1. The fresh baseline again confirms that the dataset's natural best test window is around `20000-21000`, not necessarily `30000`.
2. The new geometry-first validation logic is functioning: all three PRISM runs still passed official geometry validation at `16000`.
3. The new scalable heavy-eval implementation was still not truly scalable in its first version, because it retained a hidden global structure-cache build.
4. This hidden global step is the most plausible reason all PRISM runs crashed at the same stage boundary.

Therefore the current status is:

- **Baseline:** valid and strong
- **GeoGateFix:** still the best conceptual control branch, but this run failed before completion
- **GeoGateFixKeep:** unresolved
- **GeoGateCompact:** unresolved, and likely more aggressive / fragile than intended

The next step is not a new redesign first.  
The next step is to validate the heavy-eval hotfix by rerunning the same suite and only then judge the actual quality-geometry-speed tradeoff.

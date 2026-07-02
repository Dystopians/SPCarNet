# SPCarNet / MeshSplatting Feedback for Next-Stage AI Model

Date: 2026-06-28

# 2026-07-02 v346-v348 Feedback Addendum: Learned Residual Direction Is the Bottleneck

New files:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
docs/car_model/7-02-v346-v348-RepresentationTransport-Log.md
docs/car_model/results/v348_directional_smooth_transport_flowers_summary.json
docs/car_model/results/v348_directional_smooth_transport_flowers_summary.md
outputs/carnet/spcarnet_v346_source_heldout_image_proxy_20260702
outputs/carnet/spcarnet_v347_texture_anchor_flowers_20260702
outputs/carnet/spcarnet_v347_128face_texture_anchor_flowers_20260702
outputs/carnet/spcarnet_v347_smooth_transport_probe_flowers_20260702
outputs/carnet/spcarnet_v348_directional_smooth_transport_flowers_20260702
```

Implemented changes:

v346-v348 moved back into the real train/eval pipeline instead of adding another
selector-only layer. The new interfaces add source-heldout image/patch proxy
supervision, a source-baked texture residual anchor, support-normalized smooth
transport, and explicit residual magnitude diagnostics.

Key result:

| method | steps | faces | all-axis pass | Phase-J gate | PSNR gain | SSIM gain | LPIPS gain |
|---|---:|---:|---|---|---:|---:|---:|
| v348a directional smooth no anchor | 600 | 128 | false | false | +0.000052119297 | -0.000000819564 | +0.000000728294 |
| v348b directional smooth anchor | 600 | 128 | false | false | +0.000057815692 | -0.000000899037 | +0.000001224379 |

This is a negative result. Longer training and stronger direction-oriented
supervision increased raw residual amplitude, but they still did not produce an
all-axis policy-val pass. Do not describe v346-v348 as beating Phase-J.

Phase-J boundary:

```text
Phase-J reference: ours_26000_phasej_guarded_adaptedge_ela
strict RGB scene wins vs clean: 9/9
strict RGB per-view wins vs clean: 244/246
mean gain vs clean: +1.331084 PSNR / +0.034702 SSIM / -0.063359 LPIPS
mean triangle reduction: 7.6479%
```

Hard lesson:

The support/apply path is not the only blocker. A teacher-residual oracle on
the same 128 candidate faces gives positive PSNR/SSIM and mostly positive LPIPS
headroom. The learned residual decoder is the bottleneck: the current feature
set and loss produce conservative, low-bandwidth, structurally weak residuals.
Smoothing spreads the signal but also spreads direction errors. Texture anchors
increase amplitude but do not make the residual target-view compatible.

Recommended next prompt:

```text
Replace the per-face/UV mean residual decoder with a high-bandwidth
support-view residual transport model. Keep source-view residual patches or
local descriptors, condition prediction on target/source geometry and neighbor
evidence, train with explicit gradient/SSIM-compatible direction objectives,
and use Phase-J/v345e gates only as final certification. Do not run more
alpha-grid or selector-only variants until the generator itself can reproduce
the teacher-residual oracle direction on heldout views.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v335 Feedback Addendum: Guarded Unlock Works, Pure TNC Ranking Fails

New files:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
scripts/car_model/probe_target_neighbor_candidate_rerank.py
docs/car_model/7-01-v335-TargetNeighborCandidateUnlock.md
docs/car_model/results/v335_target_neighbor_candidate_unlock_full9_vs_v334_v333_v329b_audit.json
docs/car_model/results/v335_target_neighbor_candidate_unlock_full9_vs_v334_v333_v329b_audit.md
docs/car_model/results/v335_target_neighbor_candidate_rerank_probe.json
docs/car_model/results/v335_target_neighbor_candidate_rerank_probe.md
docs/car_model/results/v335_target_neighbor_candidate_unlock_treehill_fair_report.json
docs/car_model/results/v335_frontier_lpips_qualitative_summary.json
docs/car_model/results/v335_frontier_lpips_qualitative_summary.md
docs/car_model/results/v335_frontier_panels/
outputs/carnet/spcarnet_v335_target_neighbor_candidate_unlock_full9_20260701
outputs/carnet/spcarnet_v335_frontier_comparison_full9_20260701
```

Implemented change:

v335 adds a guarded target-neighbor candidate unlock after the full v334
rollback stack. It only promotes `fixed -> learned` when the learned candidate
has lower target-neighbor render/depth/camera self-consistency error by at
least `0.0002` MAE. It remains opt-in, target/test-GT-free at decision time,
and the source-heldout selector remains frozen before the target loop. The new
online step is a target-neighbor proxy refinement, so it must be described as a
test-time target-split/transductive certificate rather than single-view
independent inference.

Negative probe:

| metric | current/v334 | fixed | learned | pure TNC | oracle |
|---|---:|---:|---:|---:|---:|
| PSNR gain | 0.272793021725 | 0.230035428440 | 0.274551449972 | 0.235473066023 | 0.283612355038 |
| SSIM gain | 0.003738933009 | 0.003414926490 | 0.003670204304 | 0.003419653533 | 0.003790476986 |

Pure target-neighbor candidate ranking is a failure, not the method. It loses
`-0.037319955702` PSNR gain versus v334 and damages several indoor scenes. The
lesson is that target-neighbor consistency is a useful certificate, but not a
standalone quality oracle.

Full9 result:

| metric | v329b | v333 | v334 | v335 | v335-v334 | v335-v329b |
|---|---:|---:|---:|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | 0.272793021725 | 0.274017908934 | +0.001224887209 | +0.001495256455 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | 0.003738933009 | 0.003741526179 | +0.000002593170 | +0.000004865505 |
| rollback count | 0 | 2 | 3 | 3 | +0 | +3 |
| candidate unlock count | 0 | 0 | 0 | 2 | +2 | +2 |
| all-axis safe scenes | 9/9 | 9/9 | 9/9 | 9/9 |  |  |

Perceptual/frontier result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v329b | 27.588444 | 0.028173 | 0.087733 | 0.057664 |
| v334 | 27.588834 | 0.028170 | 0.087735 | 0.057664 |
| v335 | 27.590394 | 0.028168 | 0.087742 | 0.057670 |

Hard lesson:

The reflection is now producing falsifiable mechanisms: v335 first falsified the
over-broad idea, then retained only the narrow case that full9 evidence supports.
This is the right research discipline. But the gain is still not enough for a
paper-level endpoint: only treehill changes, the visible difference is subtle,
and LPIPS/DISTS are slightly worse than v334/v329b. The next model should create
a stronger candidate generator or representation, then use v335 as the
arbitration/safety layer.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v334 Feedback Addendum: Reflection Finally Became a Mechanism, But the Gain Is Still Narrow

New files:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
docs/car_model/7-01-v334-SourceTargetContradiction-Certificate.md
docs/car_model/results/v334_source_target_contradiction_treehill_report.json
docs/car_model/results/v334_source_target_contradiction_full9_vs_v333_v329b_audit.json
docs/car_model/results/v334_source_target_contradiction_full9_vs_v333_v329b_audit.md
docs/car_model/results/v334_frontier_lpips_qualitative_summary.json
docs/car_model/results/v334_frontier_lpips_qualitative_summary.md
docs/car_model/results/v334_frontier_panels/treehill_00009_v333_v334_gt_panel.png
outputs/carnet/spcarnet_v334_source_target_contradiction_treehill_20260701
outputs/carnet/spcarnet_v334_source_target_contradiction_full9_20260701
outputs/carnet/spcarnet_v334_frontier_comparison_full9_20260701
```

Implemented change:

v334 extends the v333 target-neighbor consistency certificate with a
source-target contradiction rule. The new rule handles the case where
source-local pairwise support is very confident, but target-neighbor
self-consistency mildly prefers the incumbent. It is opt-in and target/test-GT
free at decision time.

Full9 result:

| metric | v329b | v333 | v334 | v334-v333 | v334-v329b |
|---|---:|---:|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | 0.272793021725 | +0.000076448372 | +0.000270369246 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | 0.003738933009 | +0.000000024651 | +0.000002272335 |
| rollback count | 0 | 2 | 3 | +1 | +3 |

Perceptual/frontier result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v329b | 27.588444 | 0.028173 | 0.087733 | 0.057664 |
| v333 | 27.588734 | 0.028171 | 0.087735 | 0.057664 |
| v334 | 27.588834 | 0.028170 | 0.087735 | 0.057664 |

Hard lesson:

The reflection started to work only after it was converted into a falsifiable
failure-mode certificate. The missed treehill `00009` case was not caught by
support dropout or ordinary source LCB because those signals were stable and
over-confident. It was caught by looking for a contradiction: high source-local
confidence plus mild target-neighbor disagreement.

Remaining bottleneck:

This still does not solve the deeper problem. v334 is a narrow tail-risk repair
over an existing candidate generator. It improves full9 metrics without
regressing LPIPS/DISTS, but the visible difference is subtle and the gain is
concentrated in treehill. A next-stage model should not keep stacking vetoes as
the main contribution; it needs a stronger raw representation/candidate module
or a richer target-blind evidence field that produces visibly larger
improvements before arbitration.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v333 Feedback Addendum: Target-Neighbor Consistency Is the First Positive Post-v332 Fix, But Not Final Closure

New files:

```text
scripts/car_model/probe_target_neighbor_self_consistency.py
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
docs/car_model/7-01-v333-TargetNeighborConsistency-Certificate.md
docs/car_model/results/v333_target_neighbor_consistency_probe_treehill_base_reference.json
docs/car_model/results/v333_target_neighbor_consistency_probe_treehill_same_variant.json
docs/car_model/results/v333_target_neighbor_consistency_shadow_treehill_report.json
docs/car_model/results/v333_target_neighbor_consistency_enforce_treehill_report.json
docs/car_model/results/v333_target_neighbor_consistency_enforce_stump_report.json
docs/car_model/results/v333_target_neighbor_consistency_full9_vs_v329b_audit.json
docs/car_model/results/v333_target_neighbor_consistency_full9_vs_v329b_audit.md
docs/car_model/results/v333_frontier_lpips_qualitative_summary.json
docs/car_model/results/v333_frontier_lpips_qualitative_summary.md
docs/car_model/results/v333_frontier_panels/treehill_00007_00008_v329b_v333_gt_panel.png
```

Implemented change:

v333 adds a target-neighbor render self-consistency certificate to the apply
pipeline. For a pairwise promotion, it warps the promoted candidate and its
incumbent into nearby target cameras using target render/depth/camera only, then
compares each warped image against neighboring base renders. If the candidate is
more inconsistent than the incumbent by more than the frozen margin, the
certificate can shadow-log or enforce rollback. Target/test GT is not used for
the decision.

Focused results:

| run | scene | mode | selected PSNR gain | selected SSIM gain | rollback |
|---|---|---|---:|---:|---:|
| v331 reference | treehill | none | 0.104664074413 | 0.001673645443 | 0 |
| v333 shadow | treehill | shadow | 0.104664074413 | 0.001673645443 | 2 would rollback |
| v333 enforce | treehill | enforce | 0.106409362285 | 0.001693874598 | 2 applied |
| v333 enforce | stump | enforce | 0.057029761393 | 0.001208242029 | 0 applied |

Full9 result versus v329b:

| metric | v329b | v333 | delta |
|---|---:|---:|---:|
| selected PSNR gain | 0.272522652479 | 0.272716573354 | +0.000193920875 |
| selected SSIM gain | 0.003736660673 | 0.003738908357 | +0.000002247684 |
| target-neighbor rollback count | 0 | 2 | +2 |

Perceptual/frontier result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v329b | 27.588444 | 0.028173 | 0.087733 | 0.057664 |
| v333 | 27.588734 | 0.028171 | 0.087735 | 0.057664 |

Hard lesson:

The base-reference target-neighbor signal is useful as a conservative tail-risk
veto, not as a complete selector. It correctly rolls back treehill `00007` and
`00008`, giving a verified full9 macro gain over v329b, but it still keeps
`00009`, which is target-negative. The same-variant neighbor check is mostly
self-coherence and is not discriminative. The visual/perceptual evidence also
remains weak: PSNR/MAE/DISTS are slightly better than v329b, but LPIPS is not a
strict win and the rollback panel shows subtle differences.

Next-stage implication:

Freeze v333 as the current post-v329b candidate policy, but do not claim paper
closure yet. The full9 gain is real but narrow and entirely comes from treehill.
The next improvement should either catch the remaining `00009`-style
stable-but-wrong case or improve the raw candidate generator so that policy
vetoes are not doing most of the work.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v332 Feedback Addendum: Support-Dropout Stability Also Fails to Separate Treehill Bad Views

New files:

```text
scripts/car_model/probe_support_dropout_consistency.py
docs/car_model/7-01-v332-SupportDropoutConsistency-NegativeProbe.md
docs/car_model/results/v332_support_dropout_treehill_consistency.json
outputs/carnet/spcarnet_v332_support_dropout_treehill_20260701/support_dropout_consistency.json
```

Implemented diagnostic:

The v332 probe recomputes candidate residuals after dropping individual support
frames, then measures stability of `delta[output_variant] -
delta[incumbent_variant]`. It is target-blind; target/test deltas are copied
only for post-hoc correlation.

Key result:

Treehill bad promoted views `00007`, `00008`, and `00009` are not cleanly
separable from positive controls by pair relative std, cosine, or sign flip.
`00009` is the clearest failure: it is target-negative, but has high dropout
cosine (`0.940110` pair, `0.965856` output). This means the failure is not just
"unstable support"; it can be stable but wrong under the current evidence
representation.

Next-stage implication:

The next model should not rely on another source/support-side veto alone. It
needs a new evidence family, such as camera-neighborhood render self-consistency
or uncertainty from independently generated candidates, or a stronger raw
candidate generator with visibly larger gains before arbitration.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v331 Feedback Addendum: Promotion Rollback LCB Is Implemented, But Source Evidence Is Still Over-Confident

New files:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
docs/car_model/7-01-v331-PromotionRollbackCertificate-Probe.md
docs/car_model/results/v331_promotion_rollback_shadow_treehill_report.json
docs/car_model/results/v331_promotion_rollback_shadow_stump_report.json
docs/car_model/results/v331c_fineladder_treehill_report.json
outputs/carnet/spcarnet_v331_promotion_rollback_shadow_treehill_20260701
outputs/carnet/spcarnet_v331_promotion_rollback_shadow_stump_20260701
outputs/carnet/spcarnet_v331c_fineladder_treehill_20260701
```

Implemented change:

The apply pipeline now has an opt-in post-decision promotion rollback
certificate. It fits source-heldout pairwise leave-one-out over-prediction
bounds, then audits pairwise target promotions before the selected image is
saved. It supports `shadow` and `enforce` modes and remains disabled by default.

Focused results:

| probe | scene | PSNR gain | SSIM gain | result |
|---|---|---:|---:|---|
| v331 shadow | treehill | 0.104664074413 | 0.001673645443 | no rollback; unchanged vs v329b |
| v331 shadow | stump | 0.057029761393 | 0.001208242029 | no rollback; unchanged vs v329b |
| v331c fine ladder | treehill | 0.103565986827 | 0.001683145761 | PSNR down, SSIM up |

Hard lesson:

The current source-local pairwise evidence is not enough to distinguish the
bad treehill target views. `00007`, `00008`, and `00009` are target-negative
after evaluation, but their target-blind source-local/LCB diagnostics still
look safe. Requiring source-reliability/pairwise agreement is too blunt: it
would reject some bad views, but also rejects strong positives such as
`00011` and `00015`. Fine laddering only trades PSNR for SSIM.

Next-stage implication:

Do not spend the next attempt on more scalar relaxations of source-local gates.
The next model needs either:

- new target-blind evidence not captured by current residual statistics, for
  example temporal/camera-neighborhood consistency, target-render self-risk, or
  uncertainty from multiple independent support subsets; or
- a stronger representation/candidate generator with a visibly larger raw
  improvement before policy arbitration.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v329b Feedback Addendum: Fixed Rollback Certificate Is Useful, But Not Enough

New files:

```text
docs/car_model/7-01-v329b-FixedRollbackCertificate-Full9-Log.md
docs/car_model/results/v329b_fixed_rollback_strict_full9_vs_v322c_audit.json
docs/car_model/results/v329b_fixed_rollback_strict_full9_vs_v327b_audit.json
docs/car_model/results/v329b_fixed_rollback_strict_focused3_vs_v322c_audit.json
docs/car_model/results/v329b_fixed_rollback_strict_garden_vs_v322c_audit.json
docs/car_model/results/v329b_fixed_rollback_panels/v329b_key_changed_views_panel.png
docs/car_model/results/v329b_fixed_rollback_panels/v329b_key_changed_views_panel_manifest.json
docs/car_model/7-01-v329b-PerceptualGeometry-v330LocalSupport-Update.md
docs/car_model/results/v329b_frontier_lpips_qualitative_summary.json
docs/car_model/results/v329b_frontier_lpips_qualitative_summary.md
docs/car_model/results/v329b_frontier_panels/
docs/car_model/results/v329b_frontier_geometry_accounting_summary.json
docs/car_model/7-01-v329b-Frontier-Geometry-Accounting.md
outputs/carnet/spcarnet_v329b_fixed_rollback_strict_full9_20260701
```

Implemented change:

`scripts/car_model/apply_source_heldout_support_transport_calibrator.py` now
has an opt-in source-reliability fixed rollback certificate. It specifically
addresses the case where the source-heldout model predicts that `fixed` is
better than the scene-selected `learned` or `hybrid` output, but the previous
incumbent-preservation rule rejects it as `fixed_when_scene_nonfixed`.

The certificate is target-blind. It uses source-heldout predicted objective,
PSNR/SSIM margins, best fixed-source margins, and scene-consistency
opposition/alignment checks. It does not use held-out target/test metrics for
selection. Target/test metrics are only used after apply for reporting.

Full9 result:

| comparison | PSNR gain delta | SSIM gain delta |
|---|---:|---:|
| v329b vs v322C | +0.001188315360 | +0.000009419319 |
| v329b vs v327b | +0.001097159569 | +0.000008436946 |

Per-scene result versus v322C:

| scene | delta PSNR gain | delta SSIM gain | changed views |
|---|---:|---:|---:|
| bonsai | +0.007963041411 | +0.000063155148 | 3 |
| room | +0.001369726094 | +0.000009183700 | 1 |
| garden | +0.000541668618 | +0.000003593663 | 1 |
| treehill | +0.000820402117 | +0.000008841356 | 7 |
| bicycle | +0.000000000000 | +0.000000000000 | 0 |
| flowers | +0.000000000000 | +0.000000000000 | 0 |
| stump | +0.000000000000 | +0.000000000000 | 0 |
| counter | +0.000000000000 | +0.000000000000 | 0 |
| kitchen | +0.000000000000 | +0.000000000000 | 0 |

Key changed views:

| view | change | delta PSNR gain | delta SSIM gain |
|---|---|---:|---:|
| bonsai/00017 | learned -> fixed | +0.102373814711 | +0.000340580940 |
| bonsai/00019 | learned -> fixed | +0.107126042486 | +0.001131176949 |
| bonsai/00026 | learned -> fixed | +0.085132675013 | +0.000864982605 |
| room/00002 | hybrid -> fixed | +0.053419317668 | +0.000358164310 |
| garden/00016 | hybrid -> fixed | +0.013000046828 | +0.000086247921 |

Fresh frontier/perceptual and geometry evidence:

| method | scenes | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|
| clean26000 | 9 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v322c | 9 | 27.587073 | 0.028173 | 0.087735 | 0.057659 |
| v327b | 9 | 27.587183 | 0.028174 | 0.087733 | 0.057660 |
| v329b | 9 | 27.588444 | 0.028173 | 0.087733 | 0.057664 |

| scenes | clean triangles | support-transport triangles | total triangle reduction | clean vertices | support-transport vertices | total vertex reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 91019714 | 84219015 | 7.471677% | 28914623 | 27795247 | 3.871315% |

Important interpretation: v329b is above local clean MeshSplatting on the
frontier table and has the best PSNR among clean26000/v322c/v327b/v329b, but it
does not all-axis dominate v327b because LPIPS and DISTS are mixed.

Important ablation lesson:

v329a used a looser rollback certificate. It improved `bonsai` and `room`, but
also accepted `garden/00008`, producing a garden regression of
`-0.000699067991` PSNR gain and `-0.000007919967` SSIM gain. v329b tightened
`min_best_psnr_delta` to `0.005`, rejected the bad `garden/00008` rollback, and
kept the good `garden/00016` rollback. This is useful evidence that the
certificate is not a cosmetic flag; it prevents a real target regression found
by the ablation.

Hard lessons for the next model:

- Incumbent preservation alone is too conservative. It blocks high-confidence
  fixed rollbacks on `bonsai`, `room`, and `garden`.
- Loosening rollback without a certificate is unsafe. It creates exactly the
  garden failure above.
- Source-heldout reliability can identify some good fixed rollbacks, but the
  effect size is still very small at full9 macro scale.
- Qualitative panels remain subtle. Even views with clear numeric PSNR gains do
  not yet produce the obvious before/after improvement expected from a strong
  top-conference result.
- The missing LPIPS/DISTS/frontier and geometry/triangle evidence is now
  filled, but the result is mixed versus v327b. Evidence completeness improved;
  the effect-size and perceptual-dominance bottlenecks remain.
- v330a/v330b local-support probes on treehill/stump did not change selected
  outputs. This suggests that scalar relaxation of the current local-support
  gate is not the right path; the next model needs stronger per-view candidate
  arbitration or a better representation candidate.
- The next improvement should increase representation capacity or residual
  transport quality, then use this certificate style to guard it. More threshold
  scans around the current carrier are unlikely to create a large win.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v322C Feedback Addendum: Candidate Ladder Helped Only After Incumbent Preservation

New files:

```text
docs/car_model/7-01-v322C-CandidateLadder-IncumbentPreserve-Log.md
docs/car_model/results/v322c_baseknn_ladder_full9_vs_v321g_summary.json
docs/car_model/results/v322c_frontier_lpips_qualitative_summary.json
docs/car_model/results/v322c_frontier_lpips_qualitative_summary.md
docs/car_model/results/v322c_frontier_panels/
outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701
outputs/carnet/spcarnet_v322c_frontier_comparison_full9_20260701
```

Implemented change:

The apply pipeline now supports dynamic residual blend candidates
`mix0250/mix0750` and records candidate-level counterfactual target metrics.
The important lesson is that candidate expansion alone was unsafe: v322B let
KNN and source-reliability auto-margin freely use ladder candidates and
regressed `bicycle` and `bonsai`. v322C fixes this by keeping KNN base-only and
using a fixed low source-reliability objective margin, so ladder candidates can
supplement the incumbent without destabilizing it.

Full9 apply result:

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |
| v321G | +0.271248 | +0.003727 | +0.014301 | +0.039726 | 8 | 9/9 |
| v322C | +0.271334 | +0.003727 | +0.014301 | +0.039726 | 8 | 9/9 |

Frontier result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v319c | 27.583642 | 0.028181 | 0.087746 | 0.057678 |
| v321G | 27.586900 | 0.028173 | 0.087736 | 0.057660 |
| v322C | 27.587073 | 0.028173 | 0.087735 | 0.057659 |

Hard lessons for the next model:

- Expanding the candidate space is useful only when protected by an incumbent.
  v322B selected `mix0750/mix0250` in several views but lost more than it gained.
- KNN should not be trusted to extrapolate to ladder candidates from sparse
  source-heldout evidence; it mis-ranked `bicycle/00002`.
- Auto-margin search can become too conservative after the candidate set
  changes; this caused the `bonsai/00005` hybrid view to be rejected in v322B.
- v322C is the best current verified incumbent, but its gain over v321G is very
  small: `+0.000086` mean PSNR gain and preserved tails/safety.
- A stronger next method should unlock more of the visible learned/mix oracle
  gap on `stump`, `treehill`, `room`, and `bicycle`, ideally through a better
  representation or calibrated per-view model rather than more threshold scans.

Current verdict:

```text
Final status: NOT COMPLETE.
```

## 2026-07-01 Feedback: v327b Reflection Helped, But Only Narrowly

New detailed log:

```text
docs/car_model/7-01-v327b-BlendStep-Pairwise-Full9-Log.md
```

New audit:

```text
docs/car_model/results/v327b_pairwise_blendstep_full9_vs_v322c_audit.json
```

What changed:

- `scripts/car_model/apply_source_heldout_support_transport_calibrator.py`
  now supports `--pairwise_dominance_max_blend_step`.
- This rejects pairwise candidates that jump too far from the current incumbent
  in the candidate-ladder blend space.
- The intended fix is to prevent the v327a failure mode where relaxed source
  evidence allowed target overreach, especially `fixed -> mix0750` jumps.

Full9 result versus v322C:

| metric | delta |
|---|---:|
| selected PSNR gain mean | +0.000091155791 |
| selected SSIM gain mean | +0.000000982373 |

Per-scene result:

- treehill improves by `+0.000820402117` PSNR gain and `+0.000008841356` SSIM
  gain versus v322C.
- bicycle, bonsai, counter, flowers, garden, kitchen, room, and stump exactly
  match v322C under replay audit.

Lessons:

- The recent reflection was not useless: it found that pairwise local/source
  evidence needs an overreach bound.
- The zero-accept guard and blend-step guard convert a dangerous pairwise module
  into a conservative no-regression module.
- The benefit is still too small and too localized. This remains engineering
  closure evidence, not a paper-level method breakthrough.
- Future prompts should not ask for broad parameter sweeps first. They should
  demand a stronger target-blind residual reliability model or representation
  update that can activate safely on more than one scene.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v325b/v326 Replay Closure Feedback Addendum

This addendum records the latest post-v322C lesson for the next model/prompt.

Main new implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
scripts/car_model/audit_v322c_replay_consistency.py
```

Detailed log and machine-readable summaries:

```text
docs/car_model/7-01-v325-v326-ReplayClosure-And-PairwiseGuard-Log.md
docs/car_model/results/v325b_full9_v322c_profile_replay_audit.json
docs/car_model/results/v325b_knnfix_delta3_v322c_profile_replay_audit.json
docs/car_model/results/v326_pairwise_strict_treehill_vs_v322c_audit.json
docs/car_model/results/v326b_zeroaccept_guard_treehill_vs_v322c_audit.json
```

What changed:

- Added `--policy_profile v322c_incumbent` so archived v322C is no longer a
  hand-reconstructed command.
- The profile includes the hidden fairness-critical settings that were missing
  from earlier representative commands: source-reliability predictive gates,
  KNN fixed-downgrade guard, KNN scene-margin, calibrated LCB/OOD settings,
  `evidence_max_side=256`, and `ssim_max_side=256`.
- Added an audit script that compares archived v322C and replay/current report
  roots scene-by-scene, including per-view output variant mismatches and
  macro gain deltas.
- Added a pairwise safety guard: target-time pairwise overrides are disabled if
  source leave-one-out accepted zero pairwise candidates.

Key result:

```text
v325b full9 replay vs archived v322C:
scenes = 9/9
macro_delta_psnr_gain = 0.0
macro_delta_ssim_gain = 0.0
macro_delta_candidate_psnr = 0.0
macro_delta_candidate_ssim = 0.0
```

Important negative evidence:

- v326 strict pairwise looked principled but regressed treehill because source
  LOO accepted `0/11` pairwise candidates while the target full model still
  overrode two target views to `mix0250`.
- This confirms that the current bottleneck is target-free selection, not raw
  candidate capacity. The strict target-GT oracle remains positive, but the
  learned/source-only selector cannot yet capture it safely.

Next recommended prompt for a stronger model:

```text
Start from the frozen v322C replay profile, not from handwritten flags.
Use --policy_profile v322c_incumbent as the incumbent baseline and preserve the
v325b full9 zero-delta replay audit. Build v327 as a source-validated selector
that may override incumbent only when source LOO proves nonzero accepted views
and positive PSNR/SSIM/tail deltas against the exact incumbent. Do not enable a
target-time full model when source LOO accepts zero candidates. First beat v322C
on focused oracle-gap scenes (treehill, stump, room, bicycle) without any SSIM
or tail regression, then run full9 and frontier LPIPS/DISTS. Document commands,
W&B offline paths, report roots, audit JSONs, and failure modes.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v321G Feedback Addendum: Reflection Finally Produced a Clean Incumbent Upgrade

New files:

```text
docs/car_model/7-01-v321G-RawMarginAccept10-Log.md
docs/car_model/results/v321g_full9_apply_metrics_vs_prior_summary.json
docs/car_model/results/v321g_frontier_lpips_qualitative_summary.json
docs/car_model/results/v321g_frontier_lpips_qualitative_summary.md
docs/car_model/results/v321g_frontier_panels/
outputs/carnet/spcarnet_v321g_rawmargin_accept10_full9_20260701
outputs/carnet/spcarnet_v321g_frontier_comparison_full9_20260701
```

Implemented change:

The current source reliability policy now treats v319c as a true incumbent.
Raw source predictions choose the auto-margin. Calibrated LCB can only provide
diagnostics/limited fallback behavior; it cannot silently change the raw
incumbent threshold. The final v321G policy also uses a `0.10` source accept
support floor, which rejects the low-support margin that caused the v321E/F
bonsai regression.

Full9 apply result:

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |
| v321E | +0.270871 | +0.003725 | +0.014301 | +0.039726 | 8 | 9/9 |
| v321G | +0.271248 | +0.003727 | +0.014301 | +0.039726 | 8 | 9/9 |

Clean-frontier result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v319c | 27.583642 | 0.028181 | 0.087746 | 0.057678 |
| v321G | 27.586900 | 0.028173 | 0.087736 | 0.057660 |

Hard lessons for the next model:

- Reflection was not enough as rhetoric. It only helped when translated into
  no-regression checks against v319c, v321E, and the failing v320/v321F cases.
- The key failure mode was not lack of a fancier model. It was letting a
  low-support source-heldout margin (`9.09%` accept fraction on bonsai) override
  a stronger incumbent.
- v321G is the cleanest current engineering incumbent: no scene regresses
  against v319c in apply metrics, and room improves.
- The scientific bottleneck remains: most scenes tie v319c, so the paper story
  still cannot claim a broad visual breakthrough.
- The next model should target tail/visual improvements that affect multiple
  scenes, not another broad parameter scan.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v319 Feedback Addendum: Incumbent Fallback Worked, Perceptual Hard Gate Failed

New files:

```text
docs/car_model/7-01-v319-IncumbentReliability-Log.md
docs/car_model/results/v319c_full9_apply_metrics_vs_prior_summary.json
docs/car_model/results/v319c_frontier_lpips_qualitative_summary.json
docs/car_model/results/v319c_frontier_lpips_qualitative_summary.md
docs/car_model/results/v319c_frontier_panels/
docs/car_model/results/v319d_full9_apply_metrics_vs_prior_summary.json
docs/car_model/results/v319d_frontier_lpips_qualitative_summary.json
docs/car_model/results/v319d_frontier_lpips_qualitative_summary.md
docs/car_model/results/v319d_frontier_panels/
outputs/carnet/spcarnet_v319c_incumbent_reliability_full9_20260701
outputs/carnet/spcarnet_v319d_perceptual_reliability_full9_20260701
```

Implemented change:

`scripts/car_model/apply_source_heldout_support_transport_calibrator.py` now
contains a source-only relative reliability policy. It predicts whether a
candidate should override the scene-selected incumbent using source-heldout
proxy evidence and OOD distance. The important protocol correction is
abstention: rejection falls back to the v315d incumbent path instead of
replacing it.

Full9 apply result:

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v315d | +0.269175 | +0.003718 | +0.014301 | +0.039726 | 8 | 9/9 |
| v319c | +0.269725 | +0.003720 | +0.014301 | +0.039726 | 8 | 9/9 |
| v319d | +0.267239 | +0.003702 | +0.014301 | +0.039696 | 8 | 8/9 |

Full9 clean-baseline frontier:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v315d | 27.582989 | 0.028182 | 0.087739 | 0.057679 |
| v319c | 27.583642 | 0.028181 | 0.087746 | 0.057678 |
| v319d | 27.580252 | 0.028191 | 0.087746 | 0.057673 |

Hard lessons for the next model:

- The reflection did help once it became a concrete protocol audit. The winning
  correction was not "more parameters"; it was "do not replace a strong
  incumbent unless the source-only reliability model has a reason to override."
- v319c finally beats v315d on full9 mean PSNR/SSIM and preserves its tail
  metrics, but the gain is small.
- v319c does not dominate v315d: LPIPS is still slightly worse.
- v319d proves that current source-heldout LPIPS/DISTS prediction is too noisy
  for hard target-time gating. It lowers PSNR/MAE and breaks stump fixed safety.
- The next useful method must either learn a better calibrated abstention model
  with source-heldout perceptual/tail objectives, or stop claiming universal
  quality dominance and frame the result as a quality-complexity Pareto method.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v318e Feedback Addendum: Reflection Helped, But Did Not Beat v315d

New files:

```text
docs/car_model/7-01-v318-SourcePerceptualAutoRisk-Log.md
docs/car_model/results/v318e_apply_metrics_vs_prior_summary.json
docs/car_model/results/v318e_source_perceptual_autorisk_frontier_summary.json
docs/car_model/results/v318e_source_perceptual_autorisk_frontier_summary.md
docs/car_model/results/v318e_frontier_panels/
outputs/carnet/spcarnet_v318e_source_perceptual_autorisk_multiscene_20260701
outputs/carnet/spcarnet_v318e_source_perceptual_autorisk_frontier_comparison_20260701
```

Implemented change:

`scripts/car_model/apply_source_heldout_support_transport_calibrator.py` now
supports source-heldout LPIPS/DISTS selector evidence, source-objective weights,
and source-only automatic risk-objective margin search. It also records a
selection protocol proving that target/test GT is first read only after the
selected render is saved and evaluated.

Full9 apply result:

| method | PSNR gain | SSIM gain | mean min PSNR | mean CVaR10 PSNR | negative views | safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| v305 | +0.266578 | +0.003701 | +0.013917 | +0.039504 | 8 | 9/9 |
| v315d | +0.269175 | +0.003718 | +0.014301 | +0.039726 | 8 | 9/9 |
| v316c | +0.268444 | +0.003710 | +0.013917 | +0.039504 | 8 | 9/9 |
| v318e | +0.268629 | +0.003715 | +0.013917 | +0.039504 | 8 | 9/9 |

Full9 clean-baseline perceptual result:

| method | PSNR | MAE | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| clean26000 | 27.193643 | 0.029112 | 0.090207 | 0.059902 |
| v305 | 27.578504 | 0.028198 | 0.087748 | 0.057662 |
| v315d | 27.582989 | 0.028182 | 0.087739 | 0.057679 |
| v316c | 27.580930 | 0.028183 | 0.087745 | 0.057673 |
| v318e | 27.581262 | 0.028185 | 0.087743 | 0.057674 |

Hard lesson for the next model:

> The latest reflection was useful because it fixed the evaluation/policy
> protocol and exposed a treehill configuration failure. It did not solve the
> scientific bottleneck. Source-heldout LPIPS/DISTS signals are too weak or too
> noisy to select a globally better policy than v315d. Any next-stage method
> should treat v315d as the mean-quality frontier and must beat it directly,
> rather than declaring success from being better than clean MeshSplatting.

Next useful direction:

Build a reliability model for support-transport residuals, not another selector
weight scan. The model should predict whether a residual remains valid under
target viewpoints using source agreement, OOD distance, view/normal consistency,
local texture stability, support count, and perceptual train-heldout evidence.
The acceptance rule must be frozen before target/test evaluation and compared
against v305, v315d, v316c, v318e, and clean26000.

Current verdict:

```text
Final status: NOT COMPLETE.
```

Latest evidence closure on 2026-07-01, v317:

```text
docs/car_model/7-01-v317-Perceptual-DISTS-Qualitative-Geometry-Closure.md
docs/car_model/7-01-v317-Frontier-Geometry-Accounting.md
docs/car_model/results/v317_frontier_lpips_qualitative_summary.json
docs/car_model/results/v317_frontier_geometry_accounting_summary.json
docs/car_model/results/v317_frontier_panels/
scripts/car_model/build_support_transport_frontier_comparison.py
scripts/car_model/build_support_transport_geometry_accounting.py
```

Full9 local clean-MeshSplatting baseline comparison:

- clean26000: PSNR `27.193643`, MAE `0.029112`, LPIPS `0.090207`, DISTS
  `0.059902`;
- v305: PSNR `27.578504`, MAE `0.028198`, LPIPS `0.087748`, DISTS
  `0.057662`;
- v315d: PSNR `27.582989`, MAE `0.028182`, LPIPS `0.087739`, DISTS
  `0.057679`;
- v316c: PSNR `27.580930`, MAE `0.028183`, LPIPS `0.087745`, DISTS
  `0.057673`.

Geometry result:

- clean MeshSplatting triangles `91,019,714`;
- current compact-parent triangles `84,219,015`;
- total triangle reduction `7.471677%`;
- total vertex reduction `3.871315%`;
- topology errors `0` across all 9 scenes.

Key reflection:

> The reflection finally became operationally useful. It forced the work away
> from parameter-scanning claims and into a fixed evidence package with fair
> clean-baseline comparison, perceptual metrics, DISTS, qualitative panels,
> geometry accounting, topology validity, exact commands, and W&B logs.

But do not tell the next model that this is a finished top-conference result.
The true state is:

- v315d is the mean-quality frontier: best PSNR, MAE, and LPIPS;
- v316c is the stricter source-tail acceptance frontier, but does not dominate
  v315d on mean quality;
- v305 is still fractionally best on DISTS, although all three current methods
  beat clean on DISTS;
- full-frame qualitative gains remain subtle, so the paper story should use
  crops/error maps/complexity evidence instead of pretending there is a large
  visible transformation.

Next-stage instruction:

> Stop asking for another broad parameter scan as the main route. Either build
> a unified source-only selector that dominates v305/v315d/v316c, or make the
> paper story explicitly about a quality-complexity Pareto improvement with
> conservative tail-risk policy selection. Any new method must be judged against
> the v317 evidence package, not against an older weak baseline.

Latest method update on 2026-07-01, v316c:

```text
docs/car_model/7-01-v316c-SourceTailAcceptanceFixed-Log.md
docs/car_model/results/v316c_source_tail_acceptance_fixed_multiscene_summary.json
outputs/carnet/spcarnet_v316c_source_tail_acceptance_fixed_multiscene_20260701
```

Main lesson:

v315d exposed a subtle but important protocol bug: fixed-threshold KNN was not
enforcing the configured source-heldout CVaR/min/positive-view gates in final
policy acceptance. v316c fixes this and uses a global source-tail eligibility
rule. It is not per-scene tuning and it does not read target/test GT before
selection.

Full9 headline:

- v316c: PSNR `+0.268444`, SSIM `+0.003710`, mean min PSNR `+0.013917`,
  mean CVaR PSNR `+0.082235`, negative views `8`;
- relative to v305: PSNR `+0.001866`, SSIM `+0.00000961`, mean CVaR
  `+0.0000621`, mean min PSNR tie, negative views tie;
- relative to v315d: mean CVaR improves by `+0.000235`, but PSNR drops by
  `-0.000732` and SSIM drops by `-0.00000789`.

Do not tell the next model that v316c is universally better than v315d. The
correct state is a two-frontier result: v315d is the mean-quality frontier,
v316c is the source-tail-safe frontier. The next model should either unify them
with a better source-only selector or build a paper story around the
mean-vs-tail Pareto frontier, plus perceptual/geometry/qualitative evidence.

Latest method update on 2026-07-01, v315d:

```text
docs/car_model/7-01-v315-CompositeTailGuard-Log.md
docs/car_model/results/v315d_no_fixed_downgrade_multiscene_summary.json
outputs/carnet/spcarnet_v315d_no_fixed_downgrade_multiscene_20260701
```

Main lesson:

v315d finally turns the repeated reflection into a stronger target-blind policy
instead of another scene-specific parameter game. KNN needs a scene-margin and
must not downgrade non-fixed scene choices to `fixed`; learned risk should only
repair fixed fallback scenes and must pass source-heldout OOD support.

Full9 headline:

- v315d: PSNR `+0.269175`, SSIM `+0.003718`, mean min PSNR `+0.014301`,
  mean CVaR PSNR `+0.082000`, negative views `8`;
- v315d beats v309/v310c/v314 on the tracked mean and tail metrics;
- v315d beats v305 on PSNR, SSIM, and mean-min tail, and matches v305 negative
  views, but still trails v305 mean CVaR by `0.000173`.

Do not claim 100% paper closure yet. The next model should focus on the tiny
remaining CVaR gap, perceptual metrics, qualitative visibility, and geometry /
triangle-count evidence.

Latest method update on 2026-06-30, v302:

New files:

- `scripts/car_model/train_source_heldout_support_transport_calibrator.py`
- `docs/car_model/6-30-v302-ConstrainedHybridSupportTransport-Log.md`
- `docs/car_model/results/v302_constrained_hybrid_support_transport_summary.json`

Key result:

- full flowers source-heldout protocol uses 113 source train views, 25
  calibrator-train heldout views, and 13 calibrator-validation heldout views;
- raw fixed ELA alpha `0.25`: PSNR `+0.072807`, SSIM `+0.001316`;
- learned-only scale `0.5`: PSNR `+0.100625`, SSIM `+0.001164`;
- selected v302 hybrid alpha `0.25`, scale `0.5`, blend `0.5`: PSNR
  `+0.088643`, SSIM `+0.001350`;
- v302 hybrid vs fixed raw ELA: PSNR `+0.015836`, SSIM `+0.0000339`;
- positive-view fraction `1.0`, changed fraction `0.693829`;
- all-axis source-heldout pass `true`;
- W&B offline:
  `outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/wandb/offline-run-20260630_175614-cakafhs3`.

Important lesson:

> The reflection finally produced a real positive method step, but the
> mechanism must be described honestly.  Learned support-transport calibration
> raises PSNR/tail but can lose SSIM if selected greedily.  The constrained
> hybrid anchor is the currently best version because it preserves raw ELA's
> structure-safe behavior while accepting learned transport only when it also
> beats the fixed anchor on SSIM.

Next-stage instruction:

> Freeze the constrained hybrid no-GT policy and test it on the real flowers
> target/test protocol.  If target/test passes, run multi-scene validation and
> ablations.  Do not claim paper closure from source-heldout validation alone.

Latest positive diagnostic on 2026-06-30, v298:

New files:

- `scripts/car_model/diagnose_source_heldout_ela_transport.py`
- `docs/car_model/6-30-v298-HighBandwidthELA-SourceHeldoutDiagnostic.md`
- `docs/car_model/results/v298_source_heldout_ela_transport_summary.json`

Key result:

- full flowers source-heldout diagnostic uses 113 source train views and 38
  heldout train views;
- best alpha `0.25`;
- PSNR gain `+0.075520`;
- SSIM gain `+0.001146`;
- changed fraction `0.683378`;
- all-axis pass `true`;
- W&B offline:
  `outputs/carnet/spcarnet_v298_source_heldout_ela_transport_20260630/wandb/offline-run-20260630_170937-urlbgvz1`.

Important lesson:

> Phase-J/ELA's high-bandwidth support-view warp path still has train-only
> source-heldout headroom. The current failure is therefore not that residual
> transport is impossible; it is that the baked face/UV/bin/latent carrier loses
> too much target-conditioned support-view information.

Next-stage instruction:

> Build a target-conditioned support-warp feature decoder. Do not continue
> alpha/rank/gate scans as the main route.

Latest root-cause reflection on 2026-06-30:

New file:

- `docs/car_model/6-30-PhaseJ-Stall-RootCause-Reflection.md`

Hard answer:

> The current representation-level line keeps stalling below Phase-J because
> Phase-J is a high-bandwidth render-time residual transport endpoint, while the
> newer baked routes compress that transport into a weaker face/UV/bin/latent
> carrier. The learned residual direction is not stable enough across views, so
> the safety gates correctly shrink the method toward near no-op.

Evidence that this is not just a parameter problem:

- v293a captures only about `4.76%` of the MSE reduction needed to move from the
  parent to Phase-J on flowers.
- v294 carrier upper-bound gives only `+0.000164 dB` policy-val PSNR under a
  favorable projection setup.
- v285/v286 source-heldout direction cosine is only about `0.214671`, with
  heldout error ratio about `2.078181`.
- v296 heldout features selected alpha `0.0` in reduced same-budget tests.
- v297 source-heldout transport loss runs, but the pilot remains numerical-noise
  scale; alpha expansion raises changed fraction while making PSNR more
  negative.

Next-stage instruction:

> Stop treating alpha/rank/gate scans as the main route. The next model must
> preserve more of Phase-J's target-conditioned support-view information path and
> train source-A to heldout-source-B residual transport with RGB, direction,
> magnitude, patch/perceptual, and uncertainty supervision.

Latest update on 2026-06-30, v297:

New files:

- `docs/car_model/6-30-v297-SourceHeldoutTransportLoss-Log.md`
- `docs/car_model/results/v297_source_heldout_transport_summary.json`

Important change:

- v297 implements the missing source-heldout residual transport objective in
  `scripts/car_model/train_perceptual_surface_residual_decoder.py`.
- It builds a source-only surface texture from train-fit source views and uses
  heldout-source views as an auxiliary residual prediction target.
- It adds `--policy_val_min_changed_fraction` with default `1e-5`, because the
  first pilot exposed a no-op false positive in the old policy-val gate.

Pilot evidence:

- Smoke passed: source-heldout loss was computed and logged with 28 source views
  and 14 heldout views.
- 24-step no-transport pilot: PSNR gain `-3.88e-7`, SSIM gain `+7.15e-8`,
  changed fraction `1.57e-5`, policy-val pass `false`.
- 24-step transport pilot initially showed an old-gate pass, but selected
  changed fraction was `0.0`; after fixed changed-fraction gate, transport
  policy-val pass is `false`.
- Fixed-gate transport pilot: PSNR gain `-5.12e-8`, SSIM gain `+8.34e-8`,
  changed fraction `2.05e-5`.
- Alpha expansion to `1.0` increased changed fraction to `8.89e-5` but made
  PSNR more negative (`-6.48e-6`), so the immediate issue is wrong residual
  direction, not only too-small alpha.

Direct verdict:

> v297 is the correct kind of method change, but not yet a quality
> breakthrough. It should be improved, not promoted. Next work should focus on
> residual-energy scaling and a cached/cheaper source-heldout sampler before
> longer runs.

Latest update on 2026-06-30, Phase-J stall investigation + v296:

New files:

- `docs/car_model/6-30-PhaseJ-Stall-Thorough-Investigation.md`
- `docs/car_model/results/v296_reduced_v2_v3_comparison_summary.json`

Hard lesson:

> The current method family is not missing one more parameter scan. It is
> missing a source-heldout residual transport objective.

Evidence:

- v293a is the best recent PSNR route, but it captures only about `4.76%` of the
  parent-to-Phase-J MSE reduction on flowers.
- v294 showed the current carrier's favorable projection upper bound is only
  `+0.000164 dB` policy-val PSNR and fails robust SSIM/LPIPS tails.
- v285/v286 showed weak cross-view residual direction stability: heldout cosine
  about `0.214671`, heldout error ratio about `2.078181`.
- v296 `lowrank_view_holdout_v3` adds target-blind heldout direction statistics,
  but same-budget reduced testing selected alpha `0.0` for both v2 and v3.
  Nonzero alpha rows changed only `1e-5` scale pixels and already hurt
  SSIM/LPIPS tails.

Direct verdict:

> v296 is a useful diagnostic scaffold but not a quality breakthrough. Do not
> promote this route to flowers exact or full9. The next model must train on
> source-A to heldout-source-B residual prediction with explicit RGB residual,
> direction, magnitude, and confidence losses. Policy-val should certify the
> learned transport model, not act as the main mechanism.

Latest update on 2026-06-30, v295:

The newest committed change is an **engineering interface closure**, not a
quality breakthrough. It completes texture-bin reliability calibration in
`scripts/car_model/train_perceptual_surface_residual_decoder.py`.

New files:

- `docs/car_model/6-30-v295-TextureReliability-Interface-Log.md`
- `docs/car_model/results/v295_texture_reliability_interface_smoke_summary.json`

Important facts:

- Added CLI flags for texture-bin calibration and threshold grids.
- Threaded selected texture reliability threshold through policy-val, best
  render writing, `_predict_delta_image`, and target no-GT exact apply.
- Added payload / W&B / Markdown / stdout reporting for texture calibration,
  selected texture threshold, and mean texture keep fraction.
- Fixed interpretation wording so a run with `target_eval_mode=never` cannot be
  presented as a flowers exact or Phase-J comparison.

Smoke validation:

- output root:
  `/tmp/peilincai_spcarnet_v295_texture_reliability_smoke_20260630`
- `steps=1`, 16 candidate faces, no LPIPS, target exact disabled.
- texture calibration enabled: `true`
- valid-bin fraction: `0.281250`
- positive-bin fraction: `0.555556`
- no-target-GT audit pass: `true`
- target exact ran: `false`

Direct verdict:

> v295 closes a real protocol gap but does not solve the Phase-J bottleneck.
> It should be used only as a cleaner substrate for the next real method:
> source-heldout cross-view residual direction prediction or a multi-source
> residual basis. Full9 remains blocked.

Latest update on 2026-06-30, v293:

The newest branch is **TextureBinLatent PatchViewMoE** in
`scripts/car_model/train_perceptual_surface_residual_decoder.py`. New files:

- `docs/car_model/6-30-v293-TextureLatent-v169-Gate-Log.md`
- `docs/car_model/results/v293_texture_latent_summary.json`

Important new facts:

- v293 adds a learned neural texture latent per face/UV bin.
- The latent is used in train, image proxy loss, policy-val, best-render
  writing, and target no-GT exact apply.
- v293 also adds `--allow_partial_init_checkpoint`, which can warm-start older
  PatchViewMoE checkpoints by expanding `net.0.weight` and zero-initializing the
  new latent input columns.
- v293a trains from scratch with latent dim 8.
- v293b warm-starts from v290 PatchViewMoE with latent dim 4.

Effective results:

- v292d prior balanced frontier: `19.851452 / 0.620343 / 0.180212`, gains
  `+0.019398 / +0.000432 / +0.000123`.
- v293a target exact: `19.853420 / 0.620328 / 0.180312`, gains
  `+0.021366 / +0.000418 / +0.000022`, Phase-J PSNR gap `-0.450938`.
- v293b target exact: `19.852988 / 0.620345 / 0.180246`, gains
  `+0.020934 / +0.000435 / +0.000088`, Phase-J PSNR gap `-0.451370`.

Current direct verdict:

> v293 is a real representation-capacity upgrade and gives the best target PSNR
> so far, but it does not close the paper blocker. The learned texture latent
> improves PSNR while increasing target perceptual tail risk. v293a is the best
> PSNR frontier; v292d remains the better balanced frontier because v293 LPIPS is
> worse. Full9 remains blocked by the v169 flowers Phase-J gate. The next model
> should not scan latent dimension as the main route; it should keep the latent
> carrier and add target-blind perceptual/tail reliability so the extra capacity
> is suppressed on views that cause negative SSIM/LPIPS tails.

Previous update on 2026-06-30, v290-v292:

The newest branch is **PatchViewMoE surface decoder with target-blind
view-support gating** in `scripts/car_model/train_perceptual_surface_residual_decoder.py`.
New files:

- `docs/car_model/6-30-v290-v292-PatchViewMoE-ViewSupport-v169-Gate-Log.md`
- `docs/car_model/results/v290_v292_patch_view_moe_view_support_summary.json`
- `docs/car_model/assets/v292d_view_support_flowers_exact_panel.png`

Important new facts:

- v290 adds `--surface_texture_mode lowrank_view_v2`, storing low-rank teacher
  residual bases plus source camera mean/concentration and target-source camera
  cosine.
- v290 adds `--decoder_output_mode patch_view_moe`, a low-rank plus
  patch/view-conditioned direct expert residual decoder.
- v291 adds strict policy-val tail-safety fields and makes automatic target
  exact depend on tail safety.
- v292 adds `--view_support_gate_mode lowrank_view_cos`, a target-blind gate
  that suppresses residuals where source-view support is weak.
- All v292 target exact runs used stripped target no-GT evidence first; target
  GT was loaded only after apply for evaluation.

Effective results:

- v290a raw PatchViewMoE target exact: `19.850131 / 0.619529 / 0.180489`,
  gains `+0.018077 / -0.000381 / -0.000155`.
- v292b floor 0.25 target exact: `19.850320 / 0.620385 / 0.180131`,
  gains `+0.018266 / +0.000474 / +0.000204`.
- v292c floor 0.15 target exact: `19.848830 / 0.620398 / 0.180057`,
  gains `+0.016777 / +0.000487 / +0.000278`.
- v292d floor 0.35 target exact: `19.851452 / 0.620343 / 0.180212`,
  gains `+0.019398 / +0.000432 / +0.000123`.
- v292e floor 0.00 target exact: `19.845929 / 0.620358 / 0.180018`,
  gains `+0.013875 / +0.000447 / +0.000317`.

Current direct verdict:

> v292 proves that target source-view support is a real missing safety signal:
> the raw PatchViewMoE carrier had strong PSNR but damaged target SSIM/LPIPS,
> while view-support gating turned it into an all-axis parent win. v292d is the
> current frontier and beats v290a on all target axes. It still does not pass the
> v169 Phase-J flowers gate because it is `-0.452906 dB` below the Phase-J PSNR
> threshold. Do not run full9. The next model should keep the view-support gate
> but replace the small PatchViewMoE carrier with a higher-capacity
> surface-attached deferred feature field or neural texture that can carry more
> teacher residual energy without losing the v292 SSIM/LPIPS repair.

Previous update on 2026-06-30, v289:

The newest branch is **target-compatible source aggregation for deferred source residual rendering** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v289-TargetCompatibility-v169-Gate-Log.md`
- `docs/car_model/results/v289_target_compatibility_summary.json`

Important new facts:

- v289 adds `target_compatibility_*` controls to the train/eval pipeline.
- The method reweights train-fit Phase-J teacher residual source slots by target-view compatibility before deferred source aggregation.
- Optional confidence shrink estimates target row risk from view gap, parent mismatch, edge mismatch, residual disagreement, effective source count, and unique source-view count.
- Three full flowers target exact runs were completed with W&B offline and strict stripped target no-GT apply.
- All v289 runs passed policy-val all-axis, passed no-GT audit, and improved over the parent on flowers exact.

Effective results:

- v286b reference target exact: `19.840910 / 0.620183 / 0.180100`, gains `+0.008856 / +0.000272 / +0.000235`, Phase-J PSNR gap `-0.463448`.
- v289a soft weighting + mild shrink target exact: `19.841450 / 0.620205 / 0.180094`, gains `+0.009396 / +0.000294 / +0.000241`, Phase-J PSNR gap `-0.462908`.
- v289b sharper weighting + stronger shrink target exact: `19.841702 / 0.620217 / 0.180109`, gains `+0.009648 / +0.000306 / +0.000226`, Phase-J PSNR gap `-0.462656`.
- v289c source-weighting-only target exact: `19.841839 / 0.620214 / 0.180080`, gains `+0.009785 / +0.000303 / +0.000255`, Phase-J PSNR gap `-0.462519`.

Current direct verdict:

> v289 proves that target-compatible source weighting is a small positive component, but it is not the missing paper-level mechanism. The best version is v289c, where confidence shrink is disabled; shrink reduces useful residual energy. The exact gain over v286b is only about `+0.000929` PSNR, so the remaining blocker is not source-selection calibration. The baked/deferred carrier still cannot inject enough correct high-frequency Phase-J residual energy into target views. Do not run full9. Do not continue target-compatibility beta/floor scans as the main route. The next model should keep source weighting as a minor component and move to a higher-capacity patch-aware learned view-dependent surface decoder or stronger patch/perceptual residual supervision.

Latest update on 2026-06-30, v287-v288:

The newest branch is **patch-aware teacher proxy and lowrank-plus-direct hybrid decoder** in `scripts/car_model/train_perceptual_surface_residual_decoder.py`. New files:

- `docs/car_model/6-30-v287-v288-PatchHybridDecoder-v169-Gate-Log.md`
- `docs/car_model/results/v287_v288_patch_hybrid_decoder_summary.json`

Important new facts:

- v287 adds `--image_loss_mode patch_edge_v1`, a support-local teacher proxy using local luma, luma-gradient map, high-pass patch residual, and residual-gradient matching.
- v287 also keeps `global_proxy` as the same-budget ablation.
- v288 adds `--decoder_output_mode lowrank_plus_direct`, which keeps the v282 low-rank surface texture basis and adds a bounded direct RGB residual head controlled by `--lowrank_direct_scale`.
- v287/v288 exact runs used stripped target no-GT evidence for apply and loaded target/test GT only after apply for metrics. W&B was offline on GPU1/GPU2.

Effective results:

- v287a patch policy-val: `20.635616 / 0.718321 / 0.152239`, gains `+0.029179 / +0.000795 / +0.001078`.
- v287a patch target exact: `19.848754 / 0.619176 / 0.180822`, gains `+0.016700 / -0.000735 / -0.000487`, Phase-J PSNR gap `-0.455604`.
- v287b global-proxy target exact: `19.845959 / 0.618907 / 0.180593`, gains `+0.013905 / -0.001004 / -0.000258`, Phase-J PSNR gap `-0.458399`.
- v288a lowrank-plus-direct scale 0.10 target exact: `19.845385 / 0.618843 / 0.180607`, gains `+0.013331 / -0.001067 / -0.000272`, Phase-J PSNR gap `-0.458973`.
- v288b scale 0.20 policy only: gains `+0.029731 / +0.000831 / +0.000805`; larger direct capacity did not justify exact.

Current direct verdict:

> v287-v288 proves two additional bottleneck facts. First, patch/gradient teacher proxy is implementable and policy-val positive, but it does not beat the same-budget global proxy and does not improve flowers exact beyond the prior v282b alpha 0.50 frontier. Second, adding a bounded direct RGB head on top of low-rank texture does not fix target-view transfer; it weakens perceptual stability before closing the Phase-J PSNR gap. Do not run full9. Do not continue patch-loss weight scans, direct-head scale scans, or alpha scans as the main route. The next model should change source-view evidence aggregation and visibility/OOD mismatch modeling.

Latest update on 2026-06-30, v285-v286:

The newest branch is **source-heldout calibration for view-feature ridge texture** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v285-v286-HeldoutCalibration-v169-Gate-Log.md`
- `docs/car_model/results/v285_v286_holdout_calibration_summary.json`

Important new facts:

- v285 adds target-free source-heldout residual-direction calibration:
  - `--view_feature_ridge_holdout_beta`
  - `--view_feature_ridge_holdout_floor`
  - `--view_feature_ridge_holdout_min_sources`
- The method splits source slots by source-view/slot parity, fits ridge on one split, predicts heldout source residuals, and shrinks target row blend using heldout error ratio plus residual-direction cosine.
- v286 removes inherited checkpoint policy fields with `--drop_checkpoint_policy_fields` and recalibrates `patch_perceptual_v1` reliability/gain for the current decoder.
- v285/v286 exact runs used stripped target no-GT evidence for apply and loaded target/test GT only after apply for evaluation. W&B was offline on GPU1.

Effective results:

- v285a policy-val: `20.664685 / 0.719837 / 0.152346`, gains `+0.058248 / +0.002310 / +0.000970`.
- v285b target exact: `19.842752 / 0.620126 / 0.180018`, gains `+0.010698 / +0.000215 / +0.000317`, Phase-J PSNR gap `-0.461606`.
- v286a recalibrated policy-val: `20.650730 / 0.719454 / 0.152572`, gains `+0.044293 / +0.001928 / +0.000745`.
- v286b recalibrated target exact: `19.840910 / 0.620183 / 0.180100`, gains `+0.008856 / +0.000272 / +0.000235`, Phase-J PSNR gap `-0.463448`.
- v286b improves target PSNR tail CVaR versus v285b from `-0.002830` to `-0.000542`, but loses mean PSNR.

Current direct verdict:

> v285-v286 proves that target-free source-heldout calibration is useful as a diagnostic and can make target tails safer after current-decoder policy recalibration. It does not solve the paper blocker. The missing problem is not only uncertainty calibration; the local ridge carrier still cannot inject enough correct RGB energy to approach Phase-J. Do not run full9. The next model should implement a global or patch-aware learned view-dependent surface decoder with stronger patch/perceptual teacher supervision, not another alpha/threshold/holdout-strength scan.

Latest update on 2026-06-30, v283-v284:

The newest branch is **view-feature ridge texture deferred surface rendering** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v283-v284-ViewFeatureRidgeTexture-v169-Gate-Log.md`
- `docs/car_model/results/v283_v284_view_feature_ridge_texture_summary.json`

Important new facts:

- v283 adds `--residual_decoder_mode view_feature_ridge_texture`, a MeshSplatting-compatible surface-attached view-dependent residual decoder.
- It gathers same-face/UV-neighborhood train-fit Phase-J teacher residual source slots and directly fits weighted ridge RGB residuals from view, normal, parent RGB, edge, teacher support, source gain, support count, source-target similarity, and relative-UV features.
- This deliberately removes the v282 PCA low-rank coefficient bottleneck.
- v284 adds target-free self-error shrink with `--view_feature_ridge_self_error_beta` and `--view_feature_ridge_self_error_floor`.
- A new audit switch `--drop_checkpoint_policy_fields` removes inherited `policy_reliability`, `policy_gain`, and `policy_tail_risk` from loaded source-bank checkpoints.
- v283/v284 target exact no-GT audits passed. All medium/exact runs used W&B offline on GPU1.

Effective results:

- v283 policy-val: `20.664684 / 0.719834 / 0.152349`, gains `+0.058247 / +0.002308 / +0.000967`, all-axis tails positive.
- v283b target exact: `19.842806 / 0.620127 / 0.180020`, gains `+0.010752 / +0.000217 / +0.000315`, Phase-J PSNR gap `-0.461552`.
- v284 policy-val: `20.664695 / 0.719835 / 0.152348`, gains `+0.058258 / +0.002309 / +0.000968`; self-confidence mean `0.912442`.
- v284b target exact: `19.842785 / 0.620127 / 0.180019`, gains `+0.010731 / +0.000216 / +0.000316`, Phase-J PSNR gap `-0.461573`.
- v284c drop-policy-fields policy-val ablation: `20.615103 / 0.713699 / 0.155407`, gains `+0.008666 / -0.003828 / -0.002090`; all-axis fails.

Current direct verdict:

> v283-v284 is a real representation-level attempt and answers a useful bottleneck question: the v282 failure was not only the PCA low-rank bottleneck. Direct view-feature ridge decoding improves policy-val and shifts target quality toward better SSIM/LPIPS, but it loses PSNR relative to v282b fixed alpha 0.50 and remains about `0.462 dB` below the Phase-J flowers PSNR gate. The v284c ablation shows stable gains still depend on inherited policy reliability/gain fields from the loaded bank. Do not promote to full9. The next model should not continue alpha/threshold scans; it should build a stronger learned view-dependent surface decoder with source-heldout calibration or patch/perceptual teacher objectives that can carry substantially more correct RGB energy to target views.

Latest update on 2026-06-30, v281-v282:

The newest branch is **Phase-J low-rank teacher residual texture + coefficient decoder** in `scripts/car_model/train_perceptual_surface_residual_decoder.py`. New files:

- `docs/car_model/6-30-v281-v282-LowRankTexture-v169-Gate-Log.md`
- `docs/car_model/results/v281_v282_lowrank_texture_summary.json`

Important new facts:

- A runtime interface bug was fixed: fitted `surface_feature_texture` now preserves `mode`, so v2/lowrank reliability no longer risks falling back to v1 reliability logic.
- v282 adds `--surface_texture_mode lowrank_v1`: each train-fit face/UV bin stores mean teacher residual plus three PCA residual bases.
- v282 adds `--decoder_output_mode lowrank_texture`: the decoder predicts mixture coefficients over baked surface bases instead of unconstrained RGB residual.
- PCA/covariance fitting is vectorized for the 65536-face flowers scale.
- v282 texture coverage is about `0.998` covered faces and `0.407` covered UV bins.
- v282a/b used W&B offline, stripped target apply, and target/test GT only after apply for evaluation.

Effective results:

- v281a target: `19.832653 / 0.619412 / 0.180652`, gains `+0.000599 / -0.000498 / -0.000317`.
- v282a lowrank + confidence target: `19.842574 / 0.618900 / 0.180701`, gains `+0.010520 / -0.001010 / -0.000366`.
- v282b lowrank no-confidence target: `19.847127 / 0.619016 / 0.180564`, gains `+0.015073 / -0.000895 / -0.000229`.
- v282b fixed alpha 0.25 target: `19.845635 / 0.620099 / 0.180523`, gains `+0.013581 / +0.000188 / -0.000188`.
- v282b fixed alpha 0.50 target: `19.850666 / 0.619745 / 0.180620`, gains `+0.018612 / -0.000165 / -0.000286`.

Current direct verdict:

> v282 is a real v169-style representation upgrade and substantially improves policy-val plus target PSNR over v281. It still fails the Phase-J flowers PSNR gate by about `0.453692` at best. Fixed-alpha diagnostics show the issue is not just alpha: lower alpha recovers SSIM but still hurts LPIPS and remains far below Phase-J PSNR. Do not promote to full9. Do not continue lowrank/alpha variants as the main route. The next model needs a stronger coherent view-dependent deferred surface renderer or patch/gradient teacher supervision with target-free uncertainty.

Latest update on 2026-06-30, v279-v280:

The newest branch is **train-fit baked surface feature texture + neural residual decoder** in `scripts/car_model/train_perceptual_surface_residual_decoder.py`. New files:

- `docs/car_model/6-30-v279-v280-SurfaceFeatureTexture-v169-Gate-Log.md`
- `docs/car_model/results/v279_v280_surface_feature_texture_summary.json`

Important new facts:

- v279 added a calibration face reliability gate and confirmed that reliability gating alone only makes the method conservative; it does not solve target perceptual transfer.
- v280 added `--surface_texture_mode v1`, a real representation-level upgrade. It bakes train-fit teacher residual statistics into a `face x UV-bin` surface feature texture and appends those features to the neural residual decoder during train, policy-val, and stripped no-GT target apply.
- v280a texture coverage: `65536` candidate faces, `1048576` bins, `0.998383` covered faces, `0.406538` covered bins, `3788244` train-fit samples.
- target/test apply no longer opens eval GT before adapted output is generated; eval GT is loaded only after no-GT apply for metrics.
- v280a greatly improves policy-val: gains `+0.031057 PSNR / +0.000530 SSIM / +0.000980 LPIPS`.
- v280a still fails target exact: `19.840939 / 0.618396 / 0.181078`, gains `+0.008885 / -0.001514 / -0.000743`.
- v280a is still `0.463419` PSNR below the Phase-J flowers gate. Do not run full9.
- Offline alpha-rescale diagnostic from v280a target renders shows that alphas `0.025` through `0.300` never achieve all-axis target gains; LPIPS remains negative even when SSIM becomes slightly positive.

Current direct verdict:

> v280 is the right kind of representation-level attempt, and it proves that a train-fit surface feature texture can carry more policy-val teacher signal. It still fails held-out target exact because the residual direction is perceptually unsafe. The next model should not continue alpha/face-gate scans. It should change the supervision target, for example patch/perceptual teacher residual distillation, or add a target-free residual-direction uncertainty model learned from source-view disagreement.

Latest update on 2026-06-30, v274:

The newest branch is **structure-safe texture low-rank residual carrier** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v274-StructureSafeTexture-Gate-Log.md`
- `docs/car_model/results/v274_structure_safe_texture_summary.json`

Important new facts:

- v274 adds a real decoder mode: `--residual_decoder_mode structure_safe_texture_lowrank`.
- The source bank now saves `residual_edge`, `residual_luma_abs`, and `teacher_better_fraction`.
- The texture low-rank branch now gates residual injection by source/target edge agreement, residual-edge support, teacher-better support, and unique-view support.
- A structure prefilter was required before covariance/eigendecomposition; otherwise full-resolution texture-lowrank evaluation became too slow.
- All effective v274 runs used W&B offline and completed flowers policy-val + target exact with target/test GT stripped before apply.

Effective results:

- v266c reference target: `19.845698 / 0.620201 / 0.179915`, gains `+0.013644 / +0.000290 / +0.000419`.
- v270d reference target: `19.844320 / 0.620226 / 0.179934`, gains `+0.012266 / +0.000315 / +0.000401`.
- v274d loaded-v266 bank target: `19.845704 / 0.620200 / 0.179917`, gains `+0.013650 / +0.000290 / +0.000418`.
- v274e fresh-fit structure stats target: `19.844540 / 0.620225 / 0.180015`, gains `+0.012486 / +0.000314 / +0.000320`.
- v274f loaded-v270 bank target: `19.844289 / 0.620224 / 0.179933`, gains `+0.012235 / +0.000314 / +0.000402`.

Direct verdict:

> v274 is a valid representation-level implementation and exact flowers validation, but not a paper-level win. It barely changes the v266c/v270d target frontier and remains about `0.459` PSNR below the Phase-J flowers gate. The fresh structure statistics improve policy-val but do not transfer to target exact. Do not run full9 from v274. The next model should move beyond local source-slot texture blending toward a learned view-dependent surface feature decoder or patch-level teacher residual carrier.

Latest update on 2026-06-30, v273:

The newest branch is **source-consensus residual denoise** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v273-ConsensusDenoise-Gate-Log.md`
- `docs/car_model/results/v273_consensus_denoise_summary.json`

Important new facts:

- v273 is not another scalar confidence head. It modifies the residual carrier by rewriting train-fit source residual slots toward leave-one-out source-view consensus residuals.
- New CLI:
  - `--source_consistency_mode denoise`
  - `--source_consistency_denoise_blend`
- v273a blend 0.50 denoised `626926` source slots, `69.81%` of valid slots, and reduced residual energy to `89.73%`.
- v273b blend 0.15 preserved more residual energy, `96.70%`, but still did not beat v266c target exact.
- v273 target exact:
  - v266c reference: `19.845698 / 0.620201 / 0.179915`
  - v270d reference: `19.844320 / 0.620226 / 0.179934`
  - v273a: `19.844213 / 0.620207 / 0.179945`
  - v273b: `19.844259 / 0.620205 / 0.179934`

Current direct verdict:

> v273 is a valid residual-bank/carrier modification but a quality failure. Source-consensus denoise improves policy-val but does not transfer to target exact. The current bottleneck is probably missing coherent view-dependent/high-frequency capacity, not source-slot residual noise. Do not continue denoise strength scans.

Latest update on 2026-06-30, v272:

The newest branch is **learned source-consistency feature head** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v272-LearnedConsistencyHead-Gate-Log.md`
- `docs/car_model/results/v272_learned_consistency_head_summary.json`

Important new facts:

- v272 added `--source_consistency_mode feature_only`, so source-view consistency can be used as a feature rather than another hard residual/weight multiplier.
- Checkpoints now save/load `source_consistency_apply_weight`, `source_consistency_apply_amplitude`, and `learned_ood_head_ceiling`.
- The learned OOD/gain head now sees source consistency reliability/amplitude/gap, base confidence, and raw residual magnitude.
- W&B offline logs and audit Markdown now record learned-head floor/ceiling.
- The implementation passed `py_compile`, `git diff --check`, CLI help, one smoke run, and four full flowers target exact runs.
- All v272 full runs improved policy-val, but none improved target exact over the v266/v270 frontier:
  - v266c reference target: `19.845698 / 0.620201 / 0.179915`.
  - v270d reference target: `19.844320 / 0.620226 / 0.179934`.
  - v272b target: `19.843843 / 0.620191 / 0.179945`.
  - v272c target: `19.844036 / 0.620193 / 0.179934`.
  - v272d target: `19.843998 / 0.620177 / 0.179918`.
  - v272e target: `19.844132 / 0.620207 / 0.179923`.

Current direct verdict:

> v272 is a valid engineering/method interface upgrade but a quality failure. Learned scalar confidence, even with target-free source-consistency features and boost-only variants, overfits policy-val and does not transfer to flowers target exact. Do not promote v272 to full9. The next model should change the residual carrier or supervision target instead of stacking another scalar policy head.

Latest update on 2026-06-30, v264-v266:

The newest branch is **edge-aware / low-rank hybrid deferred source residual rendering** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-30-v264-v266-EdgeLowrankHybrid-Log.md`
- `docs/car_model/results/v264_v266_edge_lowrank_hybrid_summary.json`

Important new facts:

- v264 added `edge_local_linear`: parent-edge features are included in local face/UV ridge residual decoding. This is a real decoder change, not an alpha scan.
- v265 added `lowrank_source_basis`: source-slot low-rank teacher residual bases are fit from train-fit evidence only. Checkpoints now save `source_view_id` so source-view diversity can be audited.
- v266 added `hybrid_edge_lowrank`: edge-local-linear is used as the stable base, and low-rank detail is injected through a disagreement-aware blend. This directly responds to the v265 negative result where pure low-rank replacement hurt PSNR/SSIM.
- v266c is the best deferred-source target PSNR so far: `19.845698 / 0.620201 / 0.179915`, gains `+0.013644 / +0.000290 / +0.000419`, changed fraction `0.054285`, PSNR tail CVaR `-0.002039`.
- However, v266c is not all-axis best. v264a still has the best target SSIM `0.620226`; v264b still has the best target LPIPS `0.179872`.
- Most importantly, v266c still fails the Phase-J flowers PSNR gate by `0.458660` PSNR. Full9 remains blocked.

Current direct verdict:

> v266 is a meaningful mechanism improvement over v263-v264 on PSNR and PSNR-tail, but it is still **NOT COMPLETE** for paper-level success. The source-slot low-rank representation is too local and not coherent enough across UV bins. The next model should not repeat low-rank slot blending, edge-gain nudges, or alpha scans. It should move to coherent face/patch texture features across UV bins, explicit patch/gradient residual supervision, and a target-free uncertainty/visibility model.

Latest update on 2026-06-29, v260-v263:

The newest branch is **local-linear deferred source residual decoding with no-GT target-visible face expansion** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`. New files:

- `docs/car_model/6-29-v260-v263-LocalLinearTargetVisible-Log.md`
- `docs/car_model/results/v260_v263_local_linear_target_visible_summary.json`

Important new facts:

- v260 added `--ood_gain_mode learned_linear`, a policy-val-supervised OOD/gain head over target-free support features. It learned a nontrivial signal, but it is only an auxiliary guard and did not improve target exact over v259/v258.
- v261 added the real representation change: `--residual_decoder_mode local_linear`, replacing convex source residual averaging with a per-face/UV local ridge decoder from source camera/parent RGB to residual, evaluated at target camera/parent RGB.
- v262 rebuilt a 32k-face source bank; v263 added `--target_visible_face_quota`, using only stripped target geometry/alpha/face visibility to expand the carrier. No target/test RGB GT or target residual keys are read during apply.
- v263a is the best result in this line: `19.844512 / 0.620224 / 0.179968`, gains `+0.012458 / +0.000314 / +0.000367`, changed fraction `0.040890`, target active fraction `0.199257`.
- v263a still fails the Phase-J flowers gate because PSNR is `0.459846` below Phase-J flowers `20.304358`. Full9 remains blocked.
- v263b extended alpha to `3.0`; policy-val selected `alpha=1.5`, but target exact degraded to `19.839942 / 0.619739 / 0.179855`, with SSIM gain `-0.000172` and PSNR tail CVaR `-0.013618`. Therefore simple alpha amplification is not the solution.

Current direct verdict:

> v263a is meaningful representation-level progress and the best v260-v263 flowers target exact result, but it is still **NOT COMPLETE**. The bottleneck is now target-visible useful changed fraction and cross-view/OOD generalization, not just source residual energy or scalar confidence. The next model should not repeat learned OOD-head-only, alpha amplification, or fixed beta scans; it should build a stronger patch/edge-aware or learned source-feature surface decoder that can safely affect more target-visible pixels.

Latest update on 2026-06-29, v259:

The newest branch is **target-support / OOD-aware gain** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.  New files:

- `docs/car_model/6-29-v259-TargetSupportOODGain-Log.md`
- `docs/car_model/results/v259_ood_gain_summary.json`

Important new facts:

- v259 adds `policy_tail_risk`, learned from policy-val positive fraction, negative gain magnitude, and gain variance per face/UV bin.
- v259 adds `--ood_gain_mode boosted_soft`, which automatically shrinks only boosted residuals using target-free OOD/source-support features: source camera view gap, residual variance ratio, parent RGB mismatch, effective source count concentration, and policy-val tail risk.
- v259a OOD beta 1 gives the best target SSIM mean in the deferred-source line: `19.838006 / 0.620050 / 0.180238`, gains `+0.005952 / +0.000139 / +0.000097`.
- v259b OOD beta 2 makes target PSNR tail CVaR positive for the first time in this v253-v259 line: tail `+0.000040 / -0.000116 / -0.000148`, but mean drops to `19.837280 / 0.620046 / 0.180256`.
- v258a remains best mean PSNR/LPIPS in this local line, but its tails are much riskier: target tail `-0.002007 / -0.000258 / -0.000380`.
- No v259 run passes the Phase-J flowers gate because PSNR is still about `0.466` below Phase-J flowers `20.304358`.  Full9 remains blocked.

Current direct verdict:

> v259 is meaningful method progress for OOD/tail safety, but it is still **NOT COMPLETE**.  It confirms that hand-crafted OOD shrink can trade residual energy for safer tails, but a fixed beta is not enough.  The next stage should train a policy-val-supervised OOD/gain head or use a stronger residual carrier, not continue manual beta/gain scanning.

Previous update on 2026-06-29, v257-v258:

The current newest branch is **policy-calibrated deferred residual gain** in `scripts/car_model/train_surface_deferred_source_residual_renderer.py`.  New files:

- `docs/car_model/6-29-v257-v258-PolicyCalibratedGain-Log.md`
- `docs/car_model/results/v257_v258_policy_calibrated_gain_summary.json`

Important new facts:

- v257 added `patch_perceptual_v1` reliability: policy-val reliability now uses local RGB L1, luma patch, and luma-gradient gains, not only scalar L1.
- v258 added `positive_soft` policy gain: the bank now learns a per-face/UV-bin `policy_gain` in addition to `policy_reliability`, so trusted bins can retain more teacher residual energy.
- This is a real train/eval pipeline change.  Checkpoints save/load `policy_gain`; target no-GT apply uses the same prediction path; target/test GT is loaded only after apply for exact evaluation.
- v258a increased active teacher residual energy retention from v257a `0.035923` to `0.467043` and improved target exact mean gains to `+0.006250 PSNR / +0.000108 SSIM / +0.000139 LPIPS`.
- v258b with lower max gain improved target SSIM slightly more: target exact `19.838286 / 0.620047 / 0.180217`, gains `+0.006232 / +0.000137 / +0.000118`.
- v258c added source-agreement confidence; it reduced target tail damage relative to v258a/b, but also reduced mean gains.
- No v257-v258 run passes the v169 Phase-J flowers gate.  Best PSNR remains about `0.466` below Phase-J flowers `20.304358`, so full9 remains blocked.
- The new bottleneck is sharper: stronger residual energy helps means but creates target-tail/OOD risk.  The next model should build a train/policy-val target-support/OOD-aware gain predictor rather than manual gain caps, alpha scans, or full9 promotion.

Current direct verdict:

> v258 is meaningful method progress and the best deferred-source flowers target mean result in this line, but it is still **NOT COMPLETE** for paper-level all-axis success because Phase-J PSNR is not beaten and target tails are not safe enough.

Previous update on 2026-06-29:

The v169 prompt has now been executed through v249-v252 representation-gate experiments.  Treat the current status as **NOT COMPLETE for paper-level all-axis win**, but with a much clearer bottleneck diagnosis.  New files:

- `docs/car_model/6-29-v249-v252-v169-RepresentationGate-Log.md`
- `docs/car_model/results/v249_v252_v169_representation_gate_summary.json`

Important new facts:

- Phase-J teacher signal is strong on flowers policy-val: about `+0.913279 PSNR / +0.065512 SSIM / +0.017600 LPIPS` teacher headroom in v251/v252 reports.
- v249a LPIPS no-harm GT-assisted U-Net has positive mean gains but fails tails: `+0.027357 PSNR / +0.000589 SSIM / +0.000250 LPIPS`, with min SSIM `-0.000152` and min LPIPS `-0.001432`; projection energy retention is only `0.020147`, cosine `0.127558`.
- v250 memory textures improve active local projection but fail GT policy-val SSIM/LPIPS: v250a `+0.007847 PSNR / -0.000152 SSIM / -0.000019 LPIPS`, v250b `+0.007915 PSNR / -0.000107 SSIM / -0.000004 LPIPS`.
- v251 low-rank/surface-feature carriers select `alpha=0` under strict tail guard, meaning the safest policy is no-op.
- v252 added a real train-fit-only `teacher_benefit_mask_mode` method and excluded `alpha=0` from policy best by default.  It reduced damage but collapsed useful residual magnitude:
  - v252a: `+0.000094 PSNR / +0.000002 SSIM / +0.000002 LPIPS`, changed fraction `0.000369`, projection energy `0.000019`, cosine `0.021462`.
  - v252b: `+0.000382 PSNR / +0.000011 SSIM / +0.000004 LPIPS`, changed fraction `0.003078`, projection energy `0.000158`, cosine `0.026398`.
- No full9 was launched because the v169 flowers gate was not passed.  Target/test apply was skipped when policy-val all-axis failed, so no target/test RGB GT leakage occurred in v252.

Lesson for the next model:

> Do not continue alpha scans, face gates, support thresholds, footprint expansion, or simple baked RGB residual carriers.  The measured blocker is residual representation capacity/alignment: current carriers retain almost none of the teacher residual once no-harm/tail constraints are enforced.  The next viable direction should be a stronger view-dependent source-feature/deferred surface renderer or another genuinely new representation class, then flowers policy-val and exact must be re-certified before any full9.

Latest update: the first v168 exact flowers attempt failed before evaluation because the old pipeline copied a full reparented evidence cache. A low-copy/direct-teacher unblock patch is now implemented and validated by smoke tests plus dry-run. A new v168 direct-teacher low-copy exact flowers run is currently running, so treat v168 as **protocol-ready, exact-metric-in-progress, not yet a metric win**.

This file is a handoff report for a stronger AI model. It records current facts, experiment data, failures, and lessons. The goal is to prevent repeating the same loops: small parameter tuning, unfair comparisons, and footprint expansion without real visual/metric gains.

## 0. Direct Status

The project is **not paper-final yet**.

The strongest local RGB endpoint is currently **Phase-J guarded adaptive edge policy**, not the latest vNext certified residual surface texture route.

The newest complete vNext idea tested so far, **v167 target-impact affine/patch residual fill**, is:

> **engineering-progress / quality-fail / NOT COMPLETE**

It completed the strict no-target-GT pipeline and produced valid manifests, W&B logs, target-evidence stripping, verifier outputs, metrics, and renders. However, it did **not** beat Phase-J on all metrics and did **not** improve meaningfully over v165/v166. In fact, the new affine candidate was rejected by policy-val and fell back to no-op.

Direct answer for future agents:

> Compared with Phase-J, the current vNext/new-prompt route is weaker. The latest improvement idea succeeded as an engineering mechanism but failed as a paper-level quality result.

Newest post-v167 progress:

> `v168` adds a runner-level Phase-J distillation profile, `--distillation_profile teacher_to_reparented_parent`, to make the next Phase-J-to-baked-representation experiment explicit and harder to misconfigure. It has passed py_compile, dry-run, negative parser guard, low-copy smoke tests, and a direct-teacher low-copy dry-run. The first exact flowers attempt failed during the fit-evidence reparent copy with `OSError: [Errno 122] Disk quota exceeded`; after the low-copy/direct-teacher patch, a new exact flowers run was launched and is still in progress. Therefore v168 has **not** produced completed exact metrics and is **not** a metric win yet.

Newest reporting/tooling progress:

> `scripts/car_model/build_spcarnet_claim_readiness_report.py` now generates a conservative claim-readiness report from current local artifacts. The current generated report is `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`; it marks Phase-J local endpoint as `PASS_LOCAL`, v106 baked representation as `PARTIAL_PASS`, v166/v167 flowers gates as `FAIL`, v168 exact metric win as `NOT_RUN`, and vNext paper-main readiness as `FAIL`. This auto report predates the failed v168 exact attempt classification; the more precise current v168 status in this handoff is `BLOCKED_PARTIAL_NO_METRICS`.

## 0.1 Claim Readiness Matrix

| possible claim | current status | evidence that supports it | missing blocker |
|---|---|---|---|
| Phase-J is the strongest local RGB endpoint | supported locally | full9 Phase-J mean `26.482766 / 0.783720 / 0.224261`, `9 / 9` scene wins vs selected clean | must state it is render-time endpoint, not baked representation |
| v106 is the strongest verified baked representation over clean MeshSplatting | partially supported | full9 v106 mean `25.831280 / 0.760830 / 0.268435`, better than selected clean | visually subtle; still weaker than Phase-J |
| vNext certified residual texture is paper-main quality method | not supported | vNext has no-GT verifier, manifests, audits, fallback, v165-v167 negative evidence | does not beat Phase-J/v106/clean; needs Phase-J-distilled exact win |
| v168 Phase-J distillation profile is a quality improvement | not supported yet | py_compile, dry-run, negative parser guard; low-copy smoke tests; direct-teacher low-copy exact currently running | no completed exact metrics yet; `/dev/shm` remains critically tight |
| Current project is paper-final | no | engineering/reporting progress is significant | no all-axis vNext win, no fixed full9 promotion, weak qualitative evidence |

## 0.2 Latest Hard Blocker: v168 Exact Run Failed Before Metrics

This is a critical handoff fact. The most recent exact validation attempt is not a negative-quality result and not a success. It is a **storage/quota-blocked partial run**.

Attempted run root:

- `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers`

Partial artifacts:

- manifest: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- report: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md`
- first log: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/logs/00_reparent_fit_evidence.log`
- partial copied evidence: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/fit_evidence_reparented`

Observed failure:

```text
reparent_fit_evidence returncode=1
shutil.copytree(...)
OSError / shutil.Error: [Errno 122] Disk quota exceeded
```

Current storage snapshot at the time of this update:

```text
/data:     28T total, 27T used, 9.6M available, 100% used
/dev/shm: 252G total, 246G used, 6.5G available, 98% used
/tmp (/): 14T total, 7.1T used, 6.1T available, but user quota exceeded
quota:    /dev/nvme0n1p4 user space 100G*, limit 100G
```

The partial v168 exact output itself uses about:

```text
391M  /tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact
5.3M  /tmp/peilincai_pycache_v168_exact
```

Recommendation for the next model:

1. Do not report this partial v168 run as completed evidence.
2. Either free durable/user-quota space first or implement a no-copy / symlink / overlay reparent mode before rerunning exact validation.
3. If cleaning space, it is reasonable to remove the failed partial v168 `/tmp` output after documenting it, because it is not a valid completed experiment.
4. Re-run v168 exact only after ensuring the output location can hold copied/reparented evidence, target evidence, rendered outputs, reports, and W&B offline logs.

Implemented follow-up after this blocker:

- Added `--copy_mode {copy,hardlink,symlink,auto_link}` to `scripts/car_model/ecsr_reparent_surface_evidence_cache.py`.
- Added `--copy_mode` and `--rewrite_rgb_render_to_parent` to `scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py`.
- Added runner flags `--reparent_copy_mode`, `--teacher_cache_copy_mode`, `--teacher_cache_rewrite_rgb_render_to_parent`, and `--skip_reparent_fit_evidence_for_teacher_cache`.
- The direct-teacher low-copy path skips the separate `fit_evidence_reparented` cache and lets teacher-cache construction rewrite output `rgb_render`/parent residual fields against the parent render.
- Static checks, smoke tests with `--max_views 1`, negative parser guard, and direct-teacher dry-run passed.
- A new exact flowers run was started at `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers` with W&B offline logging under `/dev/shm/peilincai_wandb_v168_direct_teacher_lowcopy_exact`. At the time of this handoff update, it had reached `02_certified_texture.log` policy-candidate evaluation and had not yet completed final metrics.

## 1. Best Known Metrics

### 1.1 Full9 Summary

All numbers below are from local selected-clean MeshSplatting full9 evaluation unless otherwise noted.

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean MeshSplatting | role |
|---|---:|---:|---:|---:|---|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | local fair baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 | stable representation anchor |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 | strongest verified baked representation |
| Phase-J guarded adaptive edge policy | 9 | 26.482766 | 0.783720 | 0.224261 | +1.331084 / +0.034702 / -0.063360 | strongest RGB endpoint/reference |
| vNext structure-aware shrink cleanup | 9 | 25.067699 | 0.741260 | 0.306689 | -0.083983 / -0.007758 / +0.019068 | protocol complete but weaker |
| vNext effective-margin gate | 9 | 25.067410 | 0.741259 | 0.306695 | -0.084272 / -0.007759 / +0.019074 | safer but mostly no-op |

Interpretation:

- v106 is a real baked-representation improvement over clean MeshSplatting on full9.
- Phase-J is much stronger than v106 in RGB metrics, but Phase-J is a render-time endpoint rather than the same kind of baked representation.
- vNext certified residual surface texture currently has strong engineering/audit value but is not yet a quality winner.

### 1.2 Phase-J Per-Scene Facts

Phase-J closure audit:

- strict RGB scene wins vs selected clean MeshSplatting: `9 / 9`
- per-view strict RGB wins: `244 / 246`
- mean delta vs clean: `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS
- mean total triangle reduction: `7.6479%`
- sparse geometry strict wins: `6 / 9`
- geometry-safe scenes: `9 / 9`

| scene | Phase-J PSNR | SSIM | LPIPS | dPSNR vs clean | dSSIM | dLPIPS | tri red. | per-view strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | 11.81% | 25 / 25 |
| flowers | 20.304358 | 0.557770 | 0.329222 | +0.622101 | +0.045948 | -0.065341 | 11.82% | 22 / 22 |
| garden | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | 3.47% | 24 / 24 |
| stump | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | 11.82% | 16 / 16 |
| treehill | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | 11.81% | 17 / 18 |
| room | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | 2.10% | 38 / 39 |
| counter | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | 2.10% | 30 / 30 |
| kitchen | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | 2.10% | 35 / 35 |
| bonsai | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | 11.80% | 37 / 37 |

Evidence:

- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv`
- `/dev/shm/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md`
- `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`

## 2. vNext Flowers Diagnostic History

Flowers is the clearest vNext diagnostic scene.

Core discovery: **footprint expansion alone does not produce quality gain**. v165 expanded target changed pixels by about `9.68x` over v164, but metrics barely moved. v166 then added train-only multisample residual fill, but still failed to improve SSIM/LPIPS and did not beat Phase-J.

| version | status | mechanism | accepted | alpha | changed pixels | allowed bins / faces | PSNR | SSIM | LPIPS | diagnosis |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v162 | complete | sparse-selective bridge semantic repair | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | real repair but footprint too small |
| v163 | complete | target-footprint residual-debt support expansion | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | only 1 eligible face, no improvement |
| v164 | complete | target-visible connected region growth | true | 0.375 | 860 | 121 / 13 | 20.452797 | 0.549059 | 0.355544 | no eligible connected bins |
| v165 | complete | train-only target-impact residual basis | true | 0.1875 | 8324 | 1145 / 26 | 20.452848 | 0.549059 | 0.355544 | footprint expanded, metrics unchanged |
| v166 | complete | train-only target-impact multisample residual fill | true | 0.1875 | 3859 | 457 / 4; filled 105 / 130 bins | 20.452814 | 0.549059 | 0.355544 | no-GT fill works, quality still fails |
| v167 | complete | train-only target-impact affine/patch residual field | false | 0.0 | 0 | 1182 final bins; affine filled 313 / 393 bins | 20.452776 | 0.549059 | 0.355544 | stronger capacity ran but was rejected; fallback no-op |

### 2.1 v166 vs Phase-J Flowers

| method | PSNR | SSIM | LPIPS | verdict |
|---|---:|---:|---:|---|
| Phase-J flowers | 20.304358 | 0.557770 | 0.329222 | reference to beat |
| v165 flowers exact | 20.452848 | 0.549059 | 0.355544 | PSNR higher, SSIM/LPIPS worse |
| v166 flowers exact | 20.452814 | 0.549059 | 0.355544 | PSNR higher, SSIM/LPIPS worse |
| v167 flowers exact | 20.452776 | 0.549059 | 0.355544 | fallback no-op after policy-val rejection |

v166 delta vs Phase-J flowers:

- PSNR: `+0.148457`
- SSIM: `-0.008711`
- LPIPS: `+0.026322`

This is not an all-axis win. It is a failure under the current project standard.

v167 confirms the same conclusion with a stronger representation attempt. It filled `313 / 393` target-impact affine bins using train-fit evidence only, but policy-val rejected both candidates because SSIM/L1/tail-risk were negative. Final target apply had `changed_pixels=0`.

### 2.2 v166 Artifacts

Root:

- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers`

Key outputs:

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- audit: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- metrics: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json`
- no-GT verifier: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_target_apply_no_gt_verify.json`
- renders: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/test/ours_26000_v166_target_impact_multisample_flowers/renders`
- W&B offline run: `/dev/shm/peilincai_wandb_v166_target_impact_multisample_exact/wandb/offline-run-20260628_165449-r68qgrb6`

v166 verified facts:

- manifest status: `COMPLETE`
- manifest errors: `[]`
- no-GT verifier passed: `true`
- `target_gt_visible_to_apply=false`
- `target_residual_visible_to_apply=false`
- adapter accepted: `true`
- effective policy: `accepted_atlas`
- selected alpha: `0.1875`
- target changed pixels: `3859`
- PNG-quantized changed pixels: `3807`
- changed fraction: `0.0001040139`
- target-impact final allowed bins/faces: `457 / 4`
- target-impact added bins: `456`
- target-impact added policy-row bins: `326`
- target-impact added no-policy-row bins: `130`
- multisample eligible bins: `130`
- multisample filled bins: `105`
- train-fit views used: `34`
- sample events: `3127`
- uses policy-val GT: `false`
- uses train-fit GT/residual evidence: `true`
- uses target/test GT: `false`

### 2.3 v167 Artifacts

Root:

- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers`

Key outputs:

- manifest: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- audit: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- metrics: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_results.json`
- no-GT verifier: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_target_apply_no_gt_verify.json`
- renders: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/test/ours_26000_v167_affine_flowers/renders`
- W&B offline run: `/dev/shm/peilincai_wandb_v167_affine_exact/wandb/offline-run-20260628_173303-a59lvtxg`

v167 verified facts:

- manifest status: `COMPLETE`
- manifest errors: `[]`
- no-GT verifier passed: `true`
- `target_gt_visible_to_apply=false`
- `target_residual_visible_to_apply=false`
- adapter accepted: `false`
- effective policy: `fallback_noop`
- selected alpha: `0.0`
- target changed pixels: `0`
- affine eligible bins: `393`
- affine filled bins: `313`
- train-fit views used by affine: `34`
- affine sample events: `7774`
- affine fit faces: `24`
- uses policy-val GT: `false`
- uses train-fit GT/residual evidence: `true`
- uses target/test GT: `false`
- rejection included `cvar20_view_relative_gain=-0.134897`, `min_view_relative_gain=-0.341250`, `ssim_gain=-0.000002156`, and `image_l1_gain=-0.000000127`

Interpretation: v167 moved beyond local neighbor averaging and implemented a stronger face-local ridge/patch field, but the learned correction direction was still not safe on held-out policy-val views. Future work should not simply add another small per-face regression layer; it should distill a stronger teacher or change the representation target.

## 3. Engineering State

### 3.1 vNext Pipeline Strengths

The vNext certified residual surface texture pipeline has strong engineering infrastructure:

- scene runner and manifest runner exist;
- W&B offline logging is used in medium/long runs;
- target evidence is stripped before apply;
- strict no-target-GT verifier exists;
- audit JSON records settings, selection, sparse profile, target apply stats, and fallback/no-op behavior;
- policy-val gates include SSIM/L1/effective-margin checks;
- fallback/no-op is available on rejected candidates;
- result paths, commands, and errors are recorded in manifests.

Important code paths:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/run_vnext_certified_residual_texture_manifest.py`
- `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`
- `scripts/car_model/summarize_vnext_accounting.py`

### 3.2 Current Worktree Warning

The repository is dirty. Do not blindly revert files.

Known current local additions/edits relevant to this handoff:

- `docs/Latest.md` exists and contains the latest honest status report.
- `feedback.md` is this handoff file.
- `scripts/car_model/build_spcarnet_claim_readiness_report.py` exists and builds `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`.
- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` contains the completed v167 target-impact affine/patch residual fill implementation, including `_teacher_distilled_basis_features_from_uv_camera_normal(...)`, `apply_target_impact_affine_residual_fill(...)`, CLI parsing, validation, audit fields, and candidate-loop integration.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py` forwards the v167 affine-fill flags through the certified scene runner.
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py` now also contains the v168 distillation profile `teacher_to_reparented_parent`, which forces strict no-target-GT apply and requires split-matched parent renders for Phase-J-to-baked residual experiments.
- `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md` records the v168 dry-run and negative parser guard.
- v167 was exact-tested on flowers and is a **completed negative result**, not an unfinished draft. It filled many bins without target/test GT leakage, but policy-val rejected the candidate and the final result fell back to no-op.
- v168 exact flowers was attempted after the dry-run, but it is a **partial storage-failed run**: `reparent_fit_evidence` returned code `1` due to `/tmp` user quota, before teacher cache, target stripping, adapter apply, eval, or W&B metric logging could complete.
- v168 direct-teacher low-copy was then implemented to remove the separate fit reparent copy from the exact path. This is an engineering unblock, not a quality claim.

There are many other modified/untracked files from previous work. A next model should inspect `git status --short` and avoid reverting unrelated changes.

## 4. Lessons Learned

### 4.1 Fair Comparison Lessons

Do not compare weak or short baselines against long/improved methods. Earlier confusion came from comparing mismatched training lengths or choosing checkpoints by train metrics. The fair baseline must be the best local clean MeshSplatting checkpoint/eval under the same scene, split, and evaluation protocol.

Do not select checkpoints using train metrics. That mostly rewards longer training and overfit behavior. Use held-out/test protocol, or a fixed official validation split if available.

Do not tune a separate hand-picked parameter set per scene and call it a general method. A paper-level method needs a fixed adaptive policy that reads allowed scene statistics and makes decisions automatically.

### 4.2 Method Lessons

Footprint expansion is not quality improvement. v165 expanded flowers target changed pixels from `860` to `8324`, but PSNR moved only about `+0.000051` and SSIM/LPIPS were unchanged. This is the most important negative result.

Local neighbor residual fill is too weak. v166 filled `105 / 130` eligible no-policy target-impact bins from train-fit multisample residuals, but metrics did not improve. The residual representation itself is not expressive enough.

Simple face-local affine/patch residual fields are also insufficient. v167 filled `313 / 393` eligible bins, but policy-val rejected it for negative tail risk and image metrics. The issue is not only interface capacity; the residual target/prediction direction is misaligned with robust held-out quality.

Alpha tuning is not the bottleneck anymore. v162-v166 show that changing alpha or allowing more target bins does not automatically improve SSIM/LPIPS.

Policy-val gates are necessary but not sufficient. They prevent catastrophic regressions but can select near-no-op candidates or certify changes that do not matter visually.

Target/test GT leakage must remain forbidden. Any new method can use target visibility, geometry, camera, face IDs, barycentric footprints, and rendered target evidence with GT stripped; it must not read target/test RGB GT or residual GT during apply/selection.

### 4.3 Paper-Story Lessons

Phase-J is currently the strongest empirical RGB endpoint, but it is not a baked representation. The paper story must clearly distinguish:

- clean MeshSplatting baseline;
- Phase-J render-time endpoint/reference;
- v106 baked representation;
- vNext certified representation route.

v106 is the most honest current positive representation result, but its gain is modest and may be visually subtle. It is not enough for a strong top-conference claim without a deeper representation-level advance.

vNext has a stronger research story around certified/no-GT/safe repair, but current metrics are not good enough. It needs a real representation upgrade, not more interface additions.

### 4.4 Qualitative Lessons

Full-image qualitative panels often fail to reveal small improvements. Future qualitative evidence should include:

- targeted crops at high residual/error regions;
- difference/error maps;
- before/after zoomed comparison;
- outdoor scenes where current weak spots are visible;
- geometry/triangle reduction overlays if claiming compression/geometry gains.

But visual storytelling cannot compensate for losing SSIM/LPIPS or failing the all-axis win requirement.

### 4.5 Runtime and Storage Lessons

The current vNext exact runs are expensive:

- v162 flowers adapter: about `5771.652s`
- v163 flowers adapter: about `8684.925s`
- v164 exact apply: about `23702.957s`
- v165 exact apply: about `5415.726s`
- v166 exact apply: about `3473.020s`

Storage is fragile:

- `/data` is currently full: about `9.6M` available at the latest check.
- `/dev/shm` is also near full: about `6.5G` available at the latest check.
- `/tmp` has filesystem space but the user quota on `/dev/nvme0n1p4` is exceeded: `100G* / 100G`.
- v168 exact failed before metrics because `ecsr_reparent_surface_evidence_cache.py` uses `shutil.copytree(...)`, so it tries to materialize another evidence copy under the output root.
- The low-copy/direct-teacher patch avoids the extra fit reparent cache, but target reparenting, GT-stripping, teacher cache, model output, W&B, and reports still require several GB. `/dev/shm` remains the main live risk.
- W&B offline logs and long-run manifests can fail if storage is not checked.
- Many latest artifacts are under `/dev/shm`, which is temporary storage. If the machine reboots or `/dev/shm` is cleaned, these paths may disappear. Persist the important JSON reports and qualitative renders to a durable location once `/data` has space.

Next experiments should use a staged gate:

1. syntax/static check;
2. dry-run manifest;
3. storage/quota preflight;
4. no-copy or low-copy evidence reparent if storage remains constrained;
5. flowers exact;
6. only if flowers beats Phase-J all-axis, fixed full9 promotion.

## 5. What Not To Do Next

Do not continue with only:

- more alpha-grid tuning;
- more per-scene handpicked parameters;
- more target footprint expansion without stronger residual content;
- train-metric checkpoint selection;
- clean short-run vs our long-run comparisons;
- qualitative-only claims without quantitative all-axis support;
- full9 expensive promotion before flowers passes the Phase-J gate.

These have already consumed many iterations and did not solve the core bottleneck.

## 6. Suggested Next Research Direction

The next model should focus on a stronger **train-only residual representation** that converts target-visible footprint into real RGB/SSIM/LPIPS improvement while preserving no-target-GT apply.

Promising directions:

1. **Phase-J-distilled baked representation**
   - Treat Phase-J as the strong teacher endpoint on train/policy-val views.
   - Distill its correction into a surface/texture representation.
   - Apply only the distilled representation at test time.
   - This directly targets the observed gap: Phase-J is strong, vNext baked representation is weak.
   - Use the v168 runner profile `--distillation_profile teacher_to_reparented_parent` so teacher renders, fit parent renders, target parent renders, and no-target-GT apply are recorded and checked in one manifest.

2. **Face-local residual field upgrade, but only with a stronger target**
   - Learn a per-face or per-region residual field over `(u, v, normal, train-view camera, support confidence)`.
   - Use train-fit residual evidence only, or preferably Phase-J teacher corrections on train/policy-val views.
   - Predict target-visible UV bins without target/test RGB GT.
   - Use policy-val to gate the learned field before test apply.

3. **Adaptive fixed policy, not scanned parameters**
   - Build a scene-adaptive policy from scene statistics: residual distribution, coverage, face support, view consistency, bin uncertainty, camera spread, and geometry risk.
   - Freeze the policy before full9 promotion.
   - Do not choose custom parameters per scene after seeing test results.

4. **Multi-objective gate**
   - Require RGB improvement and geometric/compression safety together.
   - Metrics should include PSNR, SSIM, LPIPS, changed fraction, triangle reduction, geometry safety, fallback rate, and per-view strict wins.

5. **Target-aware but GT-free**
   - It is acceptable to use target/test camera, face visibility, barycentric footprint, alpha/coverage, and GT-stripped target evidence.
   - It is not acceptable to use target/test RGB GT, target residual GT, or any metric computed against target GT during selection/apply.

## 7. Minimum Success Criteria for the Next Prompt

A future prompt/model should not stop until it has evidence for all of the following:

1. A real method change is implemented in the train/eval pipeline.
2. The method uses a fixed or genuinely adaptive policy, not per-scene hand tuning.
3. Flowers exact run beats Phase-J all-axis:
   - PSNR higher than `20.304358`;
   - SSIM higher than `0.557770`;
   - LPIPS lower than `0.329222`.
4. If flowers passes, full9 is run under the same fixed policy.
5. Full9 reports clean MeshSplatting, Phase-J, v106, vNext previous best, improved method, and ablations.
6. Metrics and qualitative outputs are saved.
7. Commands, configs, result paths, and errors are documented.
8. The final report honestly marks weaknesses and failed experiments.

## 8. Suggested Prompt for the Next Stronger Model

```text
You are working in /data/peilincai/mesh-splatting.

Read feedback.md and docs/Latest.md first. Treat current evidence as authoritative.

The current bottleneck is not footprint expansion; v165/v166 proved that larger target-impact footprints and local multisample residual fills do not improve SSIM/LPIPS. Build a stronger train-only residual representation that can be baked into the surface/texture pipeline without target/test RGB GT leakage.

Requirements:
- Preserve strict no-target-GT apply.
- Do not tune per-scene parameters manually.
- Do not use train metrics for checkpoint/model selection.
- First validate on flowers exact against Phase-J flowers: 20.304358 PSNR, 0.557770 SSIM, 0.329222 LPIPS.
- Only promote to full9 if flowers is an all-axis win.
- Compare against clean MeshSplatting, Phase-J, v106, vNext previous best, and ablations.
- Save metrics, qualitative renders/crops, commands, configs, errors, and W&B offline logs.

Suggested route:
1. Implement a train-only face-local residual field or Phase-J-distilled baked representation.
2. Gate it with policy-val/nonregression checks.
3. Run dry-run, then flowers exact.
4. If failed, diagnose whether the failure is representation capacity, target coverage, gate selection, or render/apply mismatch.
5. Iterate until the evidence supports a paper-level claim or clearly proves the direction is blocked.
```

## 9. Important Index

Latest status:

- `docs/Latest.md`
- `feedback.md`
- `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`
- `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`

Core code:

- `scripts/car_model/build_spcarnet_claim_readiness_report.py`
- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/run_vnext_certified_residual_texture_manifest.py`
- `scripts/car_model/ecsr_verify_target_evidence_no_gt.py`
- `scripts/car_model/summarize_vnext_accounting.py`

Phase-J evidence:

- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv`
- `/dev/shm/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md`
- `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`

vNext full9 evidence:

- `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`

v166 evidence:

- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_ours_26000_v166_target_impact_multisample_flowers_test_target_apply_no_gt_verify.json`
- `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/model/test/ours_26000_v166_target_impact_multisample_flowers/renders`

v167 evidence:

- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/surface_residual_region_texture_adapter_audit.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_results.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_ours_26000_v167_affine_flowers_test_target_apply_no_gt_verify.json`
- `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/model/test/ours_26000_v167_affine_flowers/renders`
- `/dev/shm/peilincai_wandb_v167_affine_exact/wandb/offline-run-20260628_173303-a59lvtxg`

v168 protocol evidence:

- dry-run root: `/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers`
- dry-run manifest: `/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`
- durable log: `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`
- failed exact attempt root: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers`
- failed exact attempt first log: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/logs/00_reparent_fit_evidence.log`
- failed exact attempt manifest/report: `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json`, `/tmp/peilincai_spcarnet_20260629_v168_phasej_distill_flowers_exact/flowers/reports/flowers_vnext_certified_residual_texture_report.md`
- direct-teacher low-copy exact in-progress root: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers`
- direct-teacher low-copy W&B offline root: `/dev/shm/peilincai_wandb_v168_direct_teacher_lowcopy_exact`
- direct-teacher low-copy command markers: `--reparent_copy_mode auto_link --teacher_cache_copy_mode auto_link --teacher_cache_rewrite_rgb_render_to_parent --skip_reparent_fit_evidence_for_teacher_cache --reparent_allow_resize`

Exact replay/inspection commands are recorded in the `commands` arrays inside the v166, v167, and v168 manifest JSON files. The v168 exact manifest is a partial failed manifest, so use it only for command/error reconstruction. The current environment does not have `jq`; use Python JSON parsing or another JSON viewer if needed.

## 10. Concrete v167 Implementation and Failure Notes

These notes came from the completed v167 implementation and flowers exact run. They should help a future model avoid wasting time repeating the same patch.

Current implementation points:

- Bin-selection contract: `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`, function area around `target_impact_residual_basis`. Consume `added_bins_by_face`, `added_policy_bins_by_face`, and `added_no_policy_bins_by_face`.
- Existing fill functions: `apply_target_impact_carrier_fill(...)`, `apply_target_impact_multisample_residual_fill(...)`, and the implemented `apply_target_impact_affine_residual_fill(...)`.
- Candidate-loop insertion: affine fill is called after carrier/multisample atlas mutation and before the post-materialization `evaluate_policy_val(...)`. This preserves policy-val recertification before target apply.
- Do not patch target rendering directly. `apply_to_target(...)` should only render the already-certified atlas.
- Runner forwarding: flags are added in `scripts/car_model/run_vnext_certified_residual_texture_scene.py` beside existing target-impact fill flags, inside the `enable_train_only_target_impact_residual_basis` block.

Actual v167 CLI shape:

- `--target_impact_affine_fill_mode {off,no_policy_rows,all_added}`
- `--target_impact_affine_fill_feature_mode {face_uv_normal_camera_ridge,face_uv_patch_mixture_ridge}`
- `--target_impact_affine_fill_min_samples`
- `--target_impact_affine_fill_max_samples_per_face`
- `--target_impact_affine_fill_max_views`
- `--target_impact_affine_fill_blend`
- `--target_impact_affine_fill_ridge`
- `--target_impact_affine_fill_max_condition`
- `--target_impact_affine_fill_min_norm`
- `--target_impact_affine_fill_synthetic_count`

Validation requirements:

- nonnegative max counts/views/min norm/synthetic count;
- positive ridge and max condition;
- blend in `[0, 1]`;
- enough min samples for affine/ridge fit;
- non-`off` mode requires `--enable_train_only_target_impact_residual_basis`;
- preferably also require sparse materialization to be enabled.

Audit requirements:

- Write summary into `sparse_materialization_profile["target_impact_affine_fill"]`.
- Also write summary into `cand_fit_summary["target_impact_affine_fill"]`.
- Include `uses_train_fit_gt=true`, `uses_policy_val_gt=false`, `uses_target_or_test_gt=false`.
- Include eligible/filled/skipped counts, train views used, sample events, fit condition/ridge stats, filled bins by face, top filled bins, old/new residual norms.

No-target-GT invariants:

- Target footprint can read only target/test geometry/visibility style keys: `face_id`, `barycentric`, optional `barycentric_valid`, and `alpha`.
- New residual fitting must read residual values only from train-fit evidence / `cand_fit_views`, not target/test evidence.
- Keep target evidence stripping and verification before apply.
- Eval GT should only be populated after texture apply.
- Preserve clipping through `clip_delta_rgb(...)`.
- Update `FaceAtlas.texture`, `counts`, `variance`, and `sign_consistency` consistently.
- Keep post-fill policy-val gate between atlas mutation and target apply.

Likely v167 failure modes:

- Per-face affine/ridge fit can be underdetermined or ill-conditioned from collinear/same-bin samples.
- Synthetic filled bins may be masked out if `counts` stays below atlas/bin thresholds.
- Uncapped target-impact bins can make the run too slow.
- Carrier, multisample, and affine fills can overwrite the same bins; define order or make modes mutually exclusive.
- Direct adapter invocation can bypass runner-level target evidence stripping; certified experiments should go through the runner/manifest path.
- Most importantly, v167 already proved that simple face-local affine/patch residual fields can fill target-impact bins but still point in the wrong perceptual direction. The next model should not merely retune these v167 flags; it should change the residual target or representation, preferably by distilling a stronger teacher such as Phase-J into a baked representation.

## 11. Final Honest Evaluation

Current progress is significant but not enough:

- Engineering closure: high, roughly `80%+`.
- Evidence/reporting closure: moderate-high, roughly `75%`.
- Paper-quality method closure: not enough, roughly `45-55%`.
- Main quality blocker: no current vNext representation beats Phase-J all-axis; v106 beats clean but is not visually or conceptually strong enough yet.
- Main execution blocker: the latest v168 exact run cannot complete under current storage/quota because evidence reparenting performs a full copy.

The next breakthrough likely requires distilling or replacing Phase-J-like render-time correction with a truly baked, train-only, policy-val-certified representation, rather than continuing to adjust sparse texture footprints.

## 12. Recommended Execution Priority for the Next Model

The next model should not start by launching another full exact run into the same quota problem. Recommended order:

1. Read this file, `docs/Latest.md`, `docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md`, and `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`.
2. Inspect `scripts/car_model/ecsr_reparent_surface_evidence_cache.py` and decide whether to add a no-copy / symlink / overlay reparent mode. This is the fastest route to unblock v168 under constrained storage.
3. If storage is freed externally, rerun v168 exact without changing method code. If not, implement the low-copy reparent path and validate it with a tiny dry-run.
4. Run v168 exact flowers with W&B offline logging and strict no-target-GT verifier.
5. Compare only against the Phase-J flowers gate: `20.304358 / 0.557770 / 0.329222`. Passing means all three: higher PSNR, higher SSIM, lower LPIPS.
6. If v168 fails quality after exact completion, diagnose whether the teacher residual is being diluted, clipped, masked by sparse materialization thresholds, or rejected by policy-val. Do not retune by test metrics.
7. If flowers passes, freeze the policy and promote to fixed full9. Include clean MeshSplatting, v106, Phase-J, previous vNext, improved vNext, and ablations.
8. Save metrics, per-view results, qualitative crops, error maps, commands, logs, W&B offline paths, manifests, and failure reasons.

Minimum first engineering patch likely needed:

```text
Add a low-copy mode to evidence reparenting so v168 exact does not materialize a full duplicate evidence cache under /tmp or /dev/shm before any metric can be computed.
```

Minimum first experiment after that patch:

```text
Run v168 Phase-J-distilled flowers exact, not full9. Full9 is justified only after the flowers all-axis gate beats Phase-J.
```
# 2026-06-29 v195-v199 Surface-Texture / Low-Rank Feedback Addendum

This addendum records the latest attempt based on
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Main files:

- Implementation:
  `scripts/car_model/train_surface_conditioned_residual_unet.py`
- Standalone apply:
  `scripts/car_model/apply_surface_conditioned_residual_unet_checkpoint.py`
- Detailed log:
  `docs/car_model/6-29-v195-v199-SurfaceTexture-LowRank-Diagnostics.md`
- Machine-readable summary:
  `docs/car_model/results/v195_v199_surface_texture_lowrank_summary.json`

New method pieces implemented:

- `surface_texture_mlp`: trainable per-face/per-UV-bin surface feature texture
  with a compact decoder.
- `lowrank_surface_texture`: support-aware rank-K residual basis with a hard
  inactive-support no-op gate.
- `--surface_target_visible_evidence_dir`: no-GT target-visible face priority,
  used only for capacity allocation.
- `--artifact_prefix`: future checkpoint/report artifacts can avoid stale
  `v184_*` filenames.

Hard gate:

| Reference | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| Phase-J flowers | 20.304358 | 0.557770 | 0.329222 |

Official flowers exact results:

| Run | Method | Train GT | PSNR | SSIM | LPIPS | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v195 | surface texture MLP | no | 19.878033 | 0.509020 | 0.402998 | fail all axes |
| v196 | surface texture MLP | yes | 20.084991 | 0.523929 | 0.385202 | fail all axes; diagnostic only |
| v197 | support-aware low-rank | no | 19.834993 | 0.505835 | 0.405083 | fail all axes |
| v198 | support-aware low-rank | yes | 19.833418 | 0.505749 | 0.404551 | fail all axes; diagnostic only |
| v199 | low-rank + target-visible capacity | no | 19.835337 | 0.505801 | 0.404194 | fail all axes |

Key lessons for the next model:

1. The surface texture MLP can pass policy-val, but official target transfer
   collapses. This is likely policy-val/source-view overfitting.
2. The support-aware low-rank gate prevents unsafe writes: inactive-support
   changed fraction is exactly `0.0` in v197-v199.
3. The same gate is too conservative unless target-visible capacity is added.
   v199 raises known target face fraction to `0.167715` and active support to
   `0.105916`, but official metrics still stay near v197/v198.
4. Therefore the current blocker is not just face capacity or memory size. The
   blocker is cross-view residual generalization under the current surface
   representation.
5. The next serious route should keep the no-GT target-visible allocator and
   inactive-support no-op guarantee, but replace static per-row residual storage
   with a stronger view-conditioned residual field validated on held-out source
   views before any target apply.
6. Before building another carrier, run a teacher-residual projection audit:
   compare raw `Phase-J - parent` residual, projected carrier residual, and final
   applied residual per view/per region. If projection loses energy or structure,
   the carrier is the bottleneck. If projection is healthy but target apply
   fails, debug masking/confidence/clipping/target-transfer dilution.
7. Important fairness nuance: `--surface_target_visible_evidence_dir` uses
   target-view geometry/visibility for capacity allocation. It does not use
   target RGB GT or residuals, but it is transductive and must be disclosed.
8. Policy-val all-axis pass only means improvement over the parent render on
   held-out fit/policy-val views. It is not the Phase-J gate and did not predict
   official target success in v195-v199.
9. The script defaults still include nonzero train-fit GT loss. Teacher-only
   claims require explicitly setting all GT weights to zero.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v314 Feedback Addendum

New implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New policy option:

```text
--per_view_risk_model_only_when_scene_fixed
```

Artifact index:

```text
docs/car_model/7-01-v314-SceneFixedRiskKNN-Log.md
docs/car_model/results/v314_scene_fixed_risk_knn_focused_summary.json
docs/car_model/results/v314_scene_fixed_risk_knn_multiscene_summary.json
outputs/carnet/spcarnet_v314_scene_fixed_risk_knn_multiscene_20260701
```

What was learned:

- Learned risk is not safe as a universal per-view override.
- It becomes useful only as a fixed-fallback repair branch, and only with
  source-heldout tail guards.
- KNN remains better for scenes where the scene-level source-heldout selector
  already chooses a non-fixed variant.

Full9 result:

| method | macro PSNR gain | macro SSIM gain | safe scene rate | positive-view fraction | mean min PSNR | mean CVaR PSNR | negative views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v305 | +0.266578 | +0.003701 | 1.00 | 0.954228 | +0.013917 | +0.082173 | 8 |
| v309 | +0.267843 | +0.003711 | 1.00 | 0.949784 | +0.013817 | +0.081414 | 9 |
| v310c | +0.267134 | +0.003704 | 1.00 | 0.954228 | +0.014003 | +0.081866 | 8 |
| v314 | +0.268348 | +0.003715 | 1.00 | 0.949784 | +0.001562 | +0.078339 | 9 |

Important failure:

`treehill` is the bottleneck. v314 risk repair increases mean PSNR from
`+0.090757` to `+0.095295`, but worsens worst-view PSNR from `-0.049846` to
`-0.160136` and CVaR from `+0.002068` to `-0.025606`.

Lesson for the next model:

Do not optimize only macro mean. The next policy must explicitly preserve
target-blind tail safety. A credible v315 should block candidate switches whose
source-heldout evidence suggests worst-view collapse, while keeping v314's
mean-quality gains on `flowers`, `garden`, and `treehill`.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v313 Consistency-Feature Risk Model Feedback Addendum

New implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New reports:

```text
docs/car_model/7-01-v313-ConsistencyFeatureRiskModel-Log.md
docs/car_model/results/v313_consistency_feature_risk_model_focused_summary.json
docs/car_model/results/v313_consistency_tailguard_risk_model_focused_summary.json
```

What changed:

- Added residual-consistency proxy features:
  `delta_signal_cosine`, `opposition_fraction`, `aligned_fraction`,
  `delta_to_signal_ratio`, `std_to_signal_ratio`, and `support_confidence`.
- Tested v313a with these features in the learned risk model.
- Tested v313b with the same features plus a source-heldout min-tail guard.

Key focused result:

| method | macro PSNR gain | macro SSIM gain | safe scene rate | positive-view fraction | negative views |
| --- | ---: | ---: | ---: | ---: | ---: |
| v309 selective KNN | +0.173055 | +0.003173 | 1.00 | 0.887014 | 9 |
| v310c tail-risk scene fallback | +0.172930 | +0.003176 | 1.00 | 0.897014 | 8 |
| v313a consistency features | +0.167239 | +0.003093 | 0.75 | 0.905347 | 7 |
| v313b consistency + source min guard | +0.170377 | +0.003166 | 1.00 | 0.905347 | 7 |

Important lesson:

Residual-consistency features are more useful than plain source-feature OOD
distance. They fixed the `treehill` learned-risk failure, and the source min-tail
guard correctly disabled the bad `stump` branch. However, v313b still loses mean
PSNR/SSIM versus v309/v310c, mostly because `counter` and `bicycle` give up too
much mean quality.

Current bottleneck:

```text
We can now make the learned-risk branch safer, but not yet better. Reliability
guarding recovers safety at the cost of suppressing or misranking high-quality
scene-level choices.
```

Next recommended prompt:

```text
Continue from v313b. Keep residual-consistency features and source min-tail
safety, but stop letting the learned risk branch override strong scene-level
choices unless it has a clear source-heldout mean-quality margin. Build a
two-level policy: v309/v310c remains the default frontier; learned-risk
refinement is allowed only for views/scenes where source-heldout evidence proves
both tail safety and mean utility. Optimize this as a single fixed policy on
focused scenes, then full9 only if macro PSNR/SSIM match or exceed v309/v310c
while preserving v313b's lower negative-view count.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v312 OOD-Guarded Risk Model Feedback Addendum

This addendum records the follow-up attempt after v311.

New implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New report:

```text
docs/car_model/7-01-v312-OODGuardRiskModel-Log.md
docs/car_model/results/v312_ood_guard_risk_model_focused_summary.json
```

What changed:

- The learned risk-model policy now stores source candidate feature entries.
- It estimates a leave-one-view nearest-neighbor distance distribution from
  source-heldout candidate features.
- Target-time per-view switches are rejected to the scene-level choice if the
  chosen candidate exceeds the source distance quantile.
- Per-view diagnostics now include raw risk output, final decision, selected
  proxy variant, OOD distance, OOD threshold, and reject reason.

Focused result:

| method | macro PSNR gain | macro SSIM gain | safe scene rate | mean min PSNR gain | OOD rejects |
| --- | ---: | ---: | ---: | ---: | ---: |
| v309 selective KNN | +0.173055 | +0.003173 | 1.00 | -0.031668 | 0 |
| v310c tail-risk scene fallback | +0.172930 | +0.003176 | 1.00 | -0.031668 | 0 |
| v311c dual-guard risk model | +0.165518 | +0.003099 | 0.50 | -0.061860 | 0 |
| v312a OOD-guarded risk model | +0.165518 | +0.003099 | 0.50 | -0.061860 | 2 |

Important lesson:

The v312 OOD guard did not solve the failure. It rejected 2 `counter` views, but
it rejected 0 `stump` and 0 `treehill` views, exactly where the risk-model
switches were harmful. The damaging target candidates were inside the source
feature-distance support, so ordinary OOD distance is not the right reliability
signal.

Updated bottleneck:

```text
The problem is not just feature OOD. It is label/utility mismatch:
source-heldout features that look in-distribution can still imply the wrong
target risk ordering.
```

Next recommended prompt:

```text
Continue from v309/v310c/v311/v312. Stop trying scalar gates over learned risk
predictions. Build a residual-consistency reliability model: for each target
candidate, estimate whether the transported residual is supported by multiple
source views with consistent sign, color direction, depth ordering, normal/view
alignment, and low residual variance. This reliability score must be target-GT
free and must be learned/frozen on source-heldout policy-val. Only allow a
per-view switch when reliability and predicted utility agree; otherwise fall
back to v309/v310c scene/KNN policy. First validate on bicycle/counter/stump/
treehill, then full9 only if all-axis safety and mean quality exceed v309/v310c.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v311 Learned Risk Model Feedback Addendum

This addendum records a focused learned-policy attempt after the v309/v310c
frontier.

Code touched:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New logs and summaries:

```text
docs/car_model/7-01-v311-LearnedRiskModel-Audit.md
docs/car_model/results/v311_risk_model_focused_comparison_summary.json
```

What was tried:

- A source-heldout ridge risk model was added to predict per-view candidate
  objective, PSNR gain, and SSIM gain for `fixed`, `learned`, and `hybrid`.
- v311a used strict source-side gates and mostly disabled the model.
- v311b relaxed the source gates so the model actually selected per-view
  variants.
- v311c added predicted PSNR/SSIM dominance constraints versus the scene-level
  source-heldout selected variant.

Focused evidence on `bicycle/counter/stump/treehill`:

| method | macro PSNR gain | macro SSIM gain | safe scene rate | mean min PSNR gain | negative views |
| --- | ---: | ---: | ---: | ---: | ---: |
| v309 selective KNN | +0.173055 | +0.003173 | 1.00 | -0.031668 | 9 |
| v310c tail-risk KNN scene fallback | +0.172930 | +0.003176 | 1.00 | -0.031668 | 8 |
| v311a strict risk model | +0.171559 | +0.003166 | 1.00 | -0.031668 | 8 |
| v311b relaxed risk model | +0.172679 | +0.003045 | 0.25 | -0.070348 | 8 |
| v311c dual-guard risk model | +0.165518 | +0.003099 | 0.50 | -0.061860 | 7 |

Important lesson:

v311 proved that a naive learned per-view selector is not enough. Relaxing
source-side gates activates the model, but it tends to exchange SSIM/tail safety
for PSNR and becomes unsafe on `bicycle`, `stump`, and `treehill`. Adding
predicted dual-axis constraints is still insufficient because the source-heldout
proxy ranking does not reliably transfer to target views.

Current best interpretation:

- v309 remains the best mean-quality policy on full9 and focused comparisons.
- v310c remains the useful tail-balanced frontier.
- v311 should be kept as a negative ablation and diagnostic.
- The next path should be a target-blind reliability/representation upgrade, not
  a looser risk-model gate or parameter scan.

Next recommended prompt for a stronger model:

```text
Continue from v309/v310c/v311. Treat v311 as a failed learned per-view risk
selector. Build a target-blind reliability model for source-to-target proxy
validity before allowing any per-view switch. Use only train/source-heldout
evidence: multi-source residual agreement, source-view diversity, support depth
consistency, normal/view consistency, residual variance, confidence coverage,
and candidate delta stability. Selection must run without target GT; evaluation
loads target GT only after outputs are written. Compare against v309 and v310c
on focused scenes before full9, and require both mean PSNR/SSIM and tail safety
to be no worse before expanding.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-07-01 v310 Tail-Risk KNN Feedback Addendum

v310 is the follow-up to v309's main weakness: v309 improves macro mean metrics
but slightly worsens per-view tails.

Implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
docs/car_model/7-01-v310-TailRiskKNN-SceneFallback-Log.md
docs/car_model/results/v310_tailrisk_knn_scenefallback_multiscene_summary.json
```

The new mechanism searches source-heldout KNN acceptance thresholds and can
constrain source-heldout PSNR, CVaR20, minimum gain, and positive-view fraction.
The important new option is:

```text
--per_view_knn_reject_variant scene
```

This means low-confidence KNN choices fall back to the source-heldout
scene-level branch instead of no-oping to the base render.

Main result:

| method | PSNR gain | SSIM gain | positive-view fraction | mean min PSNR gain | mean CVaR20 PSNR gain | total negative views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v305 | +0.266578 | +0.003701 | 0.954228 | +0.013917 | +0.082173 | 8 |
| v309 | +0.267843 | +0.003711 | 0.949784 | +0.013817 | +0.081414 | 9 |
| v310c | +0.267134 | +0.003704 | 0.954228 | +0.014003 | +0.081866 | 8 |

Lesson:

- v309 remains the best mean-quality policy.
- v310c is a valid tail-balanced frontier, not a new main result.
- v310b proved that target no-op fallback is unsafe: stump collapsed to all
  no-op and treehill became unsafe versus fixed.
- scene fallback is the correct safety baseline for target-blind KNN rejection.
- source-heldout threshold search is still too weak to repair fixed fallback
  scenes; stump and treehill remain unchanged under safe v310c.

Next recommended direction:

```text
Keep v309 as the mean-quality main policy and v310c as the tail-balanced
ablation. Do not continue threshold-only scans. Build a learned tail-risk
predictor trained on source-heldout per-view outcomes. It should predict
per-view probability of negative PSNR/SSIM tail under fixed/learned/hybrid,
calibrate uncertainty, and fall back to scene-selected output unless the
predicted tail risk is low. It must be evaluated against both v309 and v310c,
with LPIPS/DISTS and geometry accounting added before any paper-level claim.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v305-v309 Support-Transport Policy Feedback

This addendum supersedes the earlier v253-v270 branch as the current strongest
working direction.

## What finally worked

The useful shift was to stop treating the problem as a parameter sweep or a
weak surface-carrier fitting problem. The current best method keeps the strong
online support-transport residual signal and learns/selects how to use it from
source-heldout evidence:

```text
scripts/car_model/train_source_heldout_support_transport_calibrator.py
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

The pipeline is:

1. build support evidence from source/train views;
2. train a bounded calibrator on source-heldout views;
3. evaluate fixed, learned, and hybrid branches on source-heldout validation;
4. choose the scene-level branch without target/test GT;
5. in v309, use source-heldout KNN to score fixed/learned/hybrid per-view
   choices with `PSNR gain + 20 * SSIM gain`;
6. allow that KNN policy only when leave-one-out source-heldout evidence has a
   non-negative PSNR delta over the scene-level branch;
7. apply to target/test and read GT only afterward for evaluation.

## Current best result

Latest machine-readable summary:

```text
docs/car_model/results/v309_selective_knn_policy_multiscene_summary.json
docs/car_model/6-30-v309-SelectiveKNNPolicy-Log.md
```

v309 over 9 scenes / 246 target-test views:

```text
selected PSNR gain:              +0.267843
selected SSIM gain:              +0.003711
selected minus fixed PSNR gain:  +0.037808
selected minus fixed SSIM gain:  +0.000296
safe vs fixed scene rate:        9/9
positive vs base scene rate:     9/9
mean positive-view fraction:     0.949784
support source mode:             source_split on 9/9 scenes
```

KNN enable audit:

```text
bicycle +0.003363 PSNR source delta -> KNN enabled
flowers +0.006032 PSNR source delta -> KNN enabled
garden  +0.004292 PSNR source delta -> KNN enabled
bonsai  -0.019883 PSNR source delta -> KNN disabled
counter -0.013092 PSNR source delta -> KNN disabled
kitchen -0.011621 PSNR source delta -> KNN disabled
room    -0.014647 PSNR source delta -> KNN disabled
stump/treehill fixed fallback -> KNN disabled
```

Compared with v305:

```text
v305 macro: +0.266578 PSNR / +0.003701 SSIM
v309 macro: +0.267843 PSNR / +0.003711 SSIM
delta:      +0.001265 PSNR / +0.000010 SSIM
```

Compared with v308:

```text
v308 macro: +0.265521 PSNR / +0.003699 SSIM
v309 macro: +0.267843 PSNR / +0.003711 SSIM
delta:      +0.002322 PSNR / +0.000011 SSIM
```

## Lessons from failures

v306 threshold gate failed. A hand-designed target-blind scalar threshold can
look reasonable on source-heldout evidence but over-noop hard target views:
stump and treehill became unsafe versus fixed.

v307 unconditional KNN failed as a general policy. It improved bicycle, but
overrode fixed fallback scenes and hurt stump/treehill safety.

v308 hierarchical KNN partially fixed the problem by keeping fixed scenes fixed,
but still enabled KNN on learned indoor scenes where source-heldout evidence
predicted a loss. It remained safe but fell below v305 macro PSNR.

v309 is the lesson encoded as policy: per-view intelligence must be selectively
enabled by source-heldout evidence, not globally enabled. It is still only a
small refinement because its macro gain over v305 is tiny and its mean
positive-view fraction is lower.

## Current bottleneck

The current bottleneck is not basic scene-level safety. v309 already has 9/9
positive-vs-base and 9/9 safe-vs-fixed scene rates on the current compact-model
full9 protocol.

The bottleneck is that the additional improvement over v305 is marginal:

- `+0.001265` macro PSNR over v305 is real but tiny;
- mean positive-view fraction drops from `0.954228` to `0.949784`;
- the KNN enable gate checks source PSNR delta versus the scene-level branch,
  not full source fixed-safety;
- negative individual PSNR views remain in bicycle, counter, stump, and
  treehill;
- LPIPS/DISTS and geometry/triangle-accounting metrics are not yet included;
- qualitative changes remain subtle because the method is conservative.

## Next prompt for a stronger model

Continue from v309. Do not replace it with another parameter sweep. The next
method should attack the remaining negative per-view tails while preserving the
9/9 scene-level safety guarantee:

```text
Build a target-GT-free tail-risk controller on top of v309. Use source-heldout
leave-one-out evidence to predict not only mean branch gain, but per-view tail
risk. The controller must optimize a multi-objective source-heldout score:
mean PSNR/SSIM gain, CVaR20 PSNR gain, minimum gain, positive-view fraction,
and optional perceptual proxies. It should be allowed to choose fixed/learned/
hybrid/no-op per target view only when the source-heldout policy improves mean
gain and does not degrade CVaR/min-gain beyond a pre-registered tolerance.
Evaluate against v305 and v309 on the same 9-scene compact-model protocol, then
add LPIPS/DISTS and geometry/triangle accounting. Test GT must remain evaluation
only.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v302-v305 Source-Heldout Support-Transport Feedback Addendum

This addendum records the first successful reflection-driven method revision
after the v253-v270 representation attempts.

Main implementation:

```text
scripts/car_model/train_source_heldout_support_transport_calibrator.py
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

Detailed log and summaries:

```text
docs/car_model/6-30-v305-SourceHeldoutAutoPolicy-Multiscene-Log.md
docs/car_model/results/v304_frozen_hybrid_policy_multiscene_summary.json
docs/car_model/results/v305_sourceheldout_auto_policy_multiscene_summary.json
```

What changed:

- v302 trains a bounded support-transport calibrator from train source-heldout
  views.
- v303 confirms the v302 policy on flowers test.
- v304 freezes the hybrid branch and tests all 9 scenes. It is positive versus
  base on every scene but fails all-axis versus fixed on stump.
- v305 adds a train-only source-heldout output guard. The guard selects fixed,
  learned, or hybrid without reading target/test GT before output renders are
  saved.

Key v305 result:

```text
9 scenes / 246 test views
selected PSNR gain:              +0.266578
selected SSIM gain:              +0.003701
selected minus fixed PSNR gain:  +0.036542
selected minus fixed SSIM gain:  +0.000286
selected safe vs fixed rate:     9/9
selected positive vs base rate:  9/9
```

Important lesson:

The previous bottleneck was not only weak representation capacity. A large part
of the failure came from forcing one residual branch everywhere. The support
transport signal is useful, but it needs a train-heldout risk guard because some
scenes prefer learned, some prefer hybrid, and some should fall back to fixed.
This turns the method from scene-manual tuning into a fixed adaptive policy.

Remaining bottleneck:

v305 is not yet a paper-complete result. It still has negative PSNR tail views
on bicycle, stump, and treehill, does not include LPIPS/DISTS in the latest pass,
and has not been rerun from a fresh clean long MeshSplatting baseline. The next
model should add a target-GT-free per-view risk gate trained on source-heldout
views, then rerun v305 against fresh clean long baselines with PSNR/SSIM/LPIPS
and qualitative panels.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v294 Cross-View Direction Feedback Addendum

The latest integrated diagnosis is:

```text
docs/car_model/6-30-v294-CrossViewResidualDirection-Synthesis.md
docs/car_model/results/v294_cross_view_direction_synthesis.json
```

Important lesson for the next model:

The bottleneck is not merely capacity. v294 shows the current carrier upper
bound is almost no-op at robust policy-val scale. v285/v286 show why: source
heldout residual direction is weak, with cosine around `0.214671` and error
ratio around `2.078181`. A reliability gate can suppress bad residuals, but then
the method loses the energy needed to approach Phase-J.

Next prompt should ask for a representation that learns residual direction
transport itself, supervised by source-heldout loss, instead of only fitting
teacher residual RGB and gating after the fact.

# 2026-06-30 v294 Teacher Projection Upper-Bound Feedback Addendum

This addendum records the direct answer to the v169 diagnostic question:

> Can the current face/UV/low-rank carrier represent a meaningful fraction of
> Phase-J teacher residual on train-policy-val?

Short answer: **only a tiny, non-robust fraction**.

Artifacts:

```text
scripts/car_model/analyze_v169_policy_val_upper_bound.py
docs/car_model/6-30-v294-TeacherProjectionUpperBound-Diagnostic.md
docs/car_model/results/v294_teacher_projection_upper_bound_summary.json
outputs/carnet/spcarnet_v294_projection_diagnostics_20260630/flowers_v169_projection_upper_bound.json
```

Best candidate:

```text
texture_size=8
low_rank_rank=4
alpha=0.03125
PSNR gain=+0.0001638198 dB
SSIM gain=+0.0000003924
LPIPS gain=+0.0000009562
SSIM positive-view fraction=0.5
LPIPS positive-view fraction=0.666667
robust all-axis pass=false
```

Lesson:

The carrier is not completely broken, but its upper-bound image-level effect is
essentially negligible compared with the Phase-J PSNR gap. Raising alpha or rank
increases residual magnitude but quickly hurts SSIM/LPIPS. Therefore the next
route should not be an alpha/rank scan. It needs a representation that explicitly
models cross-view residual direction reliability, source-view uncertainty, and
target support before applying Phase-J residuals.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v290-v292 PatchViewMoE + View-Support Feedback Addendum

This addendum records the latest attempt based on:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

Main implementation:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

Detailed log and machine-readable summary:

```text
docs/car_model/6-30-v290-v292-PatchViewMoE-ViewSupport-v169-Gate-Log.md
docs/car_model/results/v290_v292_patch_view_moe_view_support_summary.json
```

New qualitative panel:

```text
docs/car_model/assets/v292d_view_support_flowers_exact_panel.png
```

What changed:

- v290 added a stronger surface feature carrier:
  `lowrank_view_v2 + patch_view_moe`.
- `lowrank_view_v2` keeps a low-rank teacher residual texture and source camera
  support statistics on the surface.
- `patch_view_moe` decodes residuals through low-rank coefficients plus a small
  patch/view-conditioned expert mixture.
- v291 added strict policy-val tail certificates.
- v292 added a target-blind view-support gate based on source-target camera
  cosine and source camera concentration.

Important experimental result:

| run | target PSNR | target SSIM | target LPIPS | PSNR gain | SSIM gain | LPIPS gain | target all-axis |
|---|---:|---:|---:|---:|---:|---:|---|
| v290a raw PatchViewMoE | 19.850131 | 0.619529 | 0.180489 | +0.018077 | -0.000381 | -0.000155 | fail |
| v291b structure-gain tail-safe | 19.845900 | 0.619911 | 0.180715 | +0.013846 | +0.000000 | -0.000380 | fail |
| v292b view-support floor 0.25 | 19.850320 | 0.620385 | 0.180131 | +0.018266 | +0.000474 | +0.000204 | pass |
| v292c view-support floor 0.15 | 19.848830 | 0.620398 | 0.180057 | +0.016777 | +0.000487 | +0.000278 | pass |
| **v292d view-support floor 0.35** | **19.851452** | **0.620343** | **0.180212** | **+0.019398** | **+0.000432** | **+0.000123** | **pass** |
| v292e view-support floor 0.00 | 19.845929 | 0.620358 | 0.180018 | +0.013875 | +0.000447 | +0.000317 | pass |

Main lesson:

v290 proved that the new PatchViewMoE carrier can produce a meaningful residual,
but target SSIM/LPIPS failed. v292 proved that target view support is a real
failure mode: once the residual is attenuated by source-view support, v290a's
SSIM/LPIPS regression turns into an all-axis target win versus the parent.

Remaining hard bottleneck:

The method still does not pass the v169 Phase-J flowers gate:

```text
Phase-J flowers reference: 20.304358 PSNR / 0.557770 SSIM / 0.329222 LPIPS
v292d flowers exact:       19.851452 PSNR / 0.620343 SSIM / 0.180212 LPIPS
```

Under the prompt's reported metric scale, v292d clears SSIM and LPIPS but is
still `-0.452906 dB` below the Phase-J PSNR threshold. Therefore full9 is still
blocked.

What the next model should not do:

- Do not launch full9 before the flowers Phase-J PSNR gate passes.
- Do not continue alpha/floor scans as the main research contribution.
- Do not rely on mean policy-val only; tail safety and target support must be
  checked because v291 showed mean/tail policy-val can fail to transfer.

Most useful next direction:

Build a carrier that closes the remaining `~0.45 dB` PSNR gap without sacrificing
the v292 SSIM/LPIPS repair. The most direct next research step is a stronger
surface-attached deferred feature field: keep the view-support gate, but replace
the current small PatchViewMoE with a representation that can preserve more
teacher residual energy in source-supported regions, such as a per-face-group
neural texture with local UV features and explicit edge/high-frequency teacher
targets.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v278 Structure/Perceptual Target Feedback Addendum

Detailed log and machine-readable summary:

```text
docs/car_model/6-30-v278-StructurePerceptualTarget-Negative-Log.md
docs/car_model/results/v278_structure_perceptual_target_summary.json
```

Main implementation:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
```

What changed:

- Added train-time residual target transforms:
  `raw`, `gain_soft`, `structure_safe`, and `structure_gain`.
- v278a trained against a `structure_gain` target using train-fit
  `teacher_gain_l1`, parent/residual luma-gradient support, and chroma shrink.
- This changed the actual supervised residual target used by pixel and image
  proxy losses; it was not an apply-only threshold.

Result:

| run | policy PSNR/SSIM/LPIPS gain | target PSNR/SSIM/LPIPS gain | lesson |
| --- | --- | --- | --- |
| v278a | +0.016578 / +0.000043 / +0.000299 | +0.008341 / -0.001218 / -0.000904 | stronger policy-val, worse target exact |

Lesson:

The simple scalar structure/gain target transform is not the missing ingredient.
It makes policy-val look more convincing but worsens target SSIM and LPIPS. The
next route should learn target-safety from multi-view agreement or a calibration
split with held-out-view structure/perceptual gains, then certify on a separate
policy-val split before target exact.

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v275-v277 Learned Surface Decoder Feedback Addendum

This addendum records the latest work based on:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

Detailed log and machine-readable summary:

```text
docs/car_model/6-30-v275-v277-LearnedSurfaceDecoder-v169-Gate-Log.md
docs/car_model/results/v275_v277_learned_surface_decoder_summary.json
```

Main implementation:

```text
scripts/car_model/train_perceptual_surface_residual_decoder.py
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

What changed:

- Added a learned surface-attached residual decoder path that trains on
  Phase-J teacher-parent residuals and evaluates with target no-GT evidence.
- Added strict target exact evaluation and Phase-J flowers gate reporting.
- Added a parent-luma-gradient structure gate that uses only target-blind parent
  render and predicted residual structure.
- Added gain-soft confidence labels from train-fit `teacher_gain_l1`, replacing
  the ineffective all-one confidence target observed in v275b.
- Added deploy-time confidence thresholding selected only on policy-val.

Important results:

| run | target PSNR gain | target SSIM gain | target LPIPS gain | changed fraction | lesson |
| --- | ---: | ---: | ---: | ---: | --- |
| v275b | +0.009091 | -0.000808 | -0.000724 | 0.139362 | learned decoder improves PSNR only |
| v276a | +0.009069 | -0.001037 | -0.000304 | 0.139342 | structure gate reduces LPIPS damage but worsens SSIM |
| v277a | +0.010690 | -0.001008 | -0.000456 | 0.139102 | gain-soft confidence improves PSNR but not structure |
| v277c | +0.009657 | -0.000896 | -0.000488 | 0.132016 | confidence threshold modestly reduces changed area |
| v277d | +0.000945 | -0.000138 | -0.000284 | 0.004060 | conservative threshold almost no-ops and still fails |

Key lesson:

Policy-val all-axis success is not enough. The learned decoder can pass
policy-val, but target exact still has negative SSIM and LPIPS. Confidence
thresholding can make the failure smaller, but it does not turn the residual
direction into a reliable positive correction. This is evidence that the current
surface carrier and raw RGB teacher residual target remain underpowered for
Phase-J distillation.

Next recommended prompt for a stronger model:

```text
Continue from v275-v277. Do not tune alpha or thresholds first. Replace the raw
RGB teacher-parent residual target with a structure/perceptual teacher target:
for example, train a view-dependent surface representation that predicts a
low-rank residual basis plus a learned reliability score supervised by
held-out-view SSIM/LPIPS gains. The model must use train-fit evidence only,
certify on policy-val, apply to stripped target no-GT evidence, and require
flowers exact PSNR/SSIM/LPIPS all-axis vs parent and Phase-J before full9.
Measure whether the new target improves the target SSIM/LPIPS sign, not only
whether it preserves PSNR.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v271 Source-View Consistency Feedback Addendum

New implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

Detailed log and summary:

```text
docs/car_model/6-30-v271-SourceViewConsistency-Gate-Log.md
docs/car_model/results/v271_source_view_consistency_summary.json
```

What changed:

- Added source-view leave-one-out residual consistency calibration.
- Each source residual slot is predicted from other source-view slots in the
  same face/UV bin.
- LOO cosine and relative error are converted into source-slot reliability.
- The reliability map is frozen before policy-val and target no-GT apply.

Key result:

| run | exact PSNR | exact SSIM | exact LPIPS | lesson |
|---|---:|---:|---:|---|
| v266c | 19.845698 | 0.620201 | 0.179915 | previous best |
| v271c | 19.845337 | 0.620191 | 0.179887 | LPIPS improves, PSNR/SSIM drop |
| v271d | 19.845648 | 0.620200 | 0.179919 | almost recovers PSNR/SSIM, loses LPIPS |

Lesson:

LOO source consistency is a meaningful uncertainty signal, but directly
multiplying source weights by it is too blunt. It removes some teacher residuals
that are inconsistent across source views but still useful on the target
trajectory. The result is an LPIPS/PSNR tradeoff, not an all-axis win.

Next recommended direction:

```text
Continue from v271. Keep source_consistency_reliability, LOO cosine/error,
policy reliability, tail risk, parent mismatch, view gap, source count, and
residual variance as features. Train a compact policy-val confidence/amplitude
head that predicts whether to apply, shrink, preserve, or slightly boost each
surface residual. Do not use source consistency as a hard source-weight
multiplier. The head must be frozen on policy-val and evaluated on stripped
target no-GT evidence. Flowers exact must beat Phase-J PSNR 20.304358, SSIM
0.557770, and LPIPS 0.329222 before any full9.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-30 v269-v270 Face-Texture Low-Rank Feedback Addendum

This addendum records the direct follow-up to:

```text
docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md
```

New implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

Detailed log and JSON summary:

```text
docs/car_model/6-30-v269-v270-FaceTextureLowrank-v169-Gate-Log.md
docs/car_model/results/v269_v270_face_texture_lowrank_summary.json
```

What was implemented:

- `patch_coherent_hybrid`: same-face neighboring UV/bin residual carrier.
- `face_texture_lowrank`: coherent same-face UV low-rank Phase-J teacher
  residual texture. It predicts target coefficients from view, parent RGB,
  edge, and relative UV offset features.
- `hybrid_edge_texture_lowrank`: stable edge-local-linear base plus the new
  coherent face-texture low-rank carrier.

Key result:

| run | mode | alpha | flowers exact PSNR | SSIM | LPIPS | Phase-J PSNR gap | result |
|---|---|---:|---:|---:|---:|---:|---|
| v266c | hybrid_edge_lowrank | 1.000 | 19.845698 | 0.620201 | 0.179915 | -0.458660 | previous best |
| v269c | face_texture_lowrank | 0.125 | 19.834773 | 0.620011 | 0.180294 | -0.469585 | too diluted |
| v270d | hybrid_edge_texture_lowrank | 1.000 | 19.844320 | 0.620226 | 0.179934 | -0.460038 | not better overall |

Important lesson:

The v169 prompt was directionally correct: a surface-attached teacher residual
texture carrier is more principled than another alpha/footprint tweak. However,
this implementation proves that same-face UV low-rank capacity alone is not
enough. It gives strong policy-val all-axis gains, but target exact PSNR remains
below both Phase-J and the previous v266c best.

Concrete bottleneck:

- v270d policy-val is strong: `+0.066941 PSNR / +0.002718 SSIM /
  +0.001205 LPIPS`.
- v270d target exact is only `+0.012266 PSNR / +0.000315 SSIM /
  +0.000401 LPIPS` vs parent.
- v270d target exact is slightly better than v266c in SSIM, but worse in PSNR
  and LPIPS.
- v270d remains `-0.460038` PSNR below the Phase-J flowers reference.

No-GT protocol:

The completed exact runs used stripped target no-GT evidence and loaded target
GT only after apply for evaluation. The no-GT verifier passed.

Next recommendation for a stronger model:

```text
Continue from v270d, but do not tune alpha or UV radius first. The same-face
low-rank texture carrier is not enough. Replace the residual-only projection
with a teacher-student objective that jointly predicts residual amplitude and
confidence under held-out source-view validation. The method should explicitly
penalize teacher residual directions that pass train-policy-val but fail target
trajectory transfer. Candidate directions: view-held-out residual sign
consistency, residual covariance/uncertainty calibration, compact learned
surface decoder with confidence head, or pseudo-target/source-view leave-one-out
distillation. Keep the v169 gate: flowers exact must exceed Phase-J PSNR
20.304358, SSIM 0.557770, and LPIPS 0.329222 before full9.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

# 2026-06-29 v255 Source-Agreement Confidence Addendum

v255 tested whether the v253 LPIPS failure can be fixed by a simple target-blind
source agreement confidence:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
--source_agreement_mode soft
--source_agreement_beta 0.25
```

Summary artifact:

```text
docs/car_model/results/v255_source_agreement_confidence_summary.json
docs/car_model/6-29-v255-SourceAgreementConfidence-Log.md
```

Result:

| stage | alpha | PSNR gain | SSIM gain | LPIPS gain | mean confidence | all-axis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| policy-val | 0.046875 | +0.001655 | +0.000018 | +0.000001 | 0.655315 | pass |
| target exact | 0.046875 | +0.001395 | +0.000036 | -0.000008 | 0.651719 | fail |

Lesson:

The source residual variance gate attenuates residuals, but it does not solve
perceptual transfer. It even makes target LPIPS more negative than v253b/v253d
while preserving PSNR/SSIM gains. Therefore the next model should not repeat a
hand-designed agreement scalar. It should learn or calibrate perceptual
reliability from held-out policy-val evidence, using richer features:
multi-source agreement, source view diversity, residual variance, edge support,
teacher-gain stability, normal/view consistency, and parent-color consistency.

# 2026-06-29 v256 Policy-Val L1 Reliability Addendum

v256 implements a learned/calibrated target-blind reliability map:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
--policy_reliability_mode local_l1
```

The policy uses policy-val GT only to learn whether each face/UV bin locally
reduces L1 error. The learned reliability map is frozen before target apply.
Target evidence remains stripped no-GT; target GT is loaded only after apply for
evaluation.

Artifacts:

```text
docs/car_model/6-29-v256-PolicyL1Reliability-Log.md
docs/car_model/results/v256_policy_l1_reliability_summary.json
```

Results:

| run | min positive fraction | alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v256a | 0.52 | 0.125 | +0.002737 | +0.000087 | +0.000035 | +0.000830 | +0.000026 | +0.000013 | pass |
| v256b | 0.50 | 0.250 | +0.005508 | +0.000175 | +0.000070 | +0.001659 | +0.000050 | +0.000026 | pass |
| v256c | 0.48 | 0.500 | +0.010844 | +0.000343 | +0.000144 | +0.003185 | +0.000091 | +0.000050 | pass |

Current best:

```text
v256c target exact = 19.835239 / 0.620001 / 0.180285
v256c target gains vs parent = +0.003185 / +0.000091 / +0.000050
```

This is the first v253-family result that fixes the target mean LPIPS failure.
It should replace v253/v255 as the current best method state.

Remaining limitations:

- It still does not pass the Phase-J flowers PSNR gate (`20.304358`).
- Target SSIM and LPIPS tails are still slightly negative.
- The visual changed fraction remains small (`0.007788` in v256c), so
  qualitative improvements may be subtle.
- Full9 is still blocked by the v169 rule.

Next recommended prompt:

```text
Continue from v256c. Preserve the target-blind policy-val reliability principle,
but replace local L1 reliability with a richer patch/perceptual reliability
model. Use policy-val only to learn reliability from patch L1, luma-gradient
error, SSIM proxy, LPIPS-sensitive edge statistics, source view diversity,
teacher-gain stability, and residual variance. Require policy-val and target
exact mean metrics and tails to improve before any full9. Do not use target/test
GT for policy selection.
```

# 2026-06-29 Residual Projection Audit Addendum

New tool:

```text
scripts/car_model/audit_surface_checkpoint_residual_projection.py
```

Compact artifacts:

```text
docs/car_model/6-29-v191-v199-ResidualProjectionAudit-Summary.md
docs/car_model/results/v191_v199_residual_projection_summary.json
```

Key result:

| Run | Policy retention | Policy cosine | Target retention | Target cosine |
| --- | ---: | ---: | ---: | ---: |
| v191 image-space U-Net calibration | 9.916031 | 0.279888 | 0.253365 | 0.393485 |
| v195 surface texture MLP | 0.068206 | 0.112638 | 0.002863 | 0.133734 |
| v196 GT-assisted surface MLP diagnostic | 1.427611 | 0.138419 | 0.029127 | 0.199612 |
| v199 support-aware low-rank | 0.015229 | 0.039391 | 0.000847 | 0.028702 |

Main lesson for the next model: the surface carrier is not simply losing at
official target evaluation; it fails to project/aligned the teacher residual
already on held-out policy-val evidence. A future method should require a
source-view projection gate before full target runs:

```text
policy residual cosine >= 0.25
target-free policy residual energy retention in [0.25, 4.0]
policy PSNR/SSIM vs teacher does not degrade materially
```

# 2026-06-29 v253-v254 Deferred Source Renderer Feedback Addendum

This addendum records the latest attempt based on
`docs/6-28-SPCarNet-v169-Lessons-Learned-ImprovedPrompt.md`.

Main new implementation:

```text
scripts/car_model/train_surface_deferred_source_residual_renderer.py
```

Detailed log and machine-readable summary:

```text
docs/car_model/6-29-v253-v254-DeferredSourceRenderer-Log.md
docs/car_model/results/v253_v254_deferred_source_renderer_summary.json
```

What changed:

- v253 is a real representation change, not another alpha scan.
- It builds a train-fit Phase-J teacher residual source bank over face/UV bins.
- Each target pixel gathers source residuals by view direction, normal
  agreement, parent-RGB similarity, support count, and teacher gain.
- Target apply uses stripped no-GT evidence; target GT is loaded only after
  apply for evaluation.
- `--bank_checkpoint` allows fixed-bank policy/eval ablations without rebuilding
  the representation.
- v254 tested residual channel shaping (`luma_only`, `chroma_shrink`) as a
  perceptual-transfer diagnostic.

Key result:

| run | selected alpha | policy PSNR gain | policy SSIM gain | policy LPIPS gain | target PSNR gain | target SSIM gain | target LPIPS gain | target all-axis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v253b raw RGB | 0.031250 | +0.001240 | +0.000015 | +0.000004 | +0.001063 | +0.000028 | -0.000002 | fail |
| v253c fine alpha | 0.046875 | +0.001837 | +0.000020 | +0.000001 | +0.001579 | +0.000040 | -0.000007 | fail |
| v253d conservative alpha | 0.015625 | +0.000628 | +0.000008 | +0.000006 | +0.000537 | +0.000014 | -0.000001 | fail |
| v254a luma only | 0.031250 | +0.001141 | +0.000012 | +0.000002 | +0.000985 | +0.000025 | -0.000005 | fail |
| v254b chroma shrink | 0.031250 | +0.001166 | +0.000013 | +0.000003 | +0.001005 | +0.000025 | -0.000004 | fail |

Important lesson:

v253 is the strongest representation-level step after v249-v252 because it
produces consistent PSNR/SSIM target gains and a policy-val all-axis pass.
However, it is still not enough. The fixed-policy target exact LPIPS gain is
slightly negative in every variant. Conservative alpha reduces the damage but
nearly collapses the visual change. Luma/chroma shaping does not fix it.

Current bottleneck:

The source bank can transfer a small MSE/SSIM-improving correction, but it cannot
yet certify that the residual direction is perceptually correct out of source
trajectory. Active projection cosine is around `0.279`, but selected-alpha
energy retention is only about `0.00119`, so the method is still too weak to make
visible or paper-level improvements.

Next recommended prompt for a stronger model:

```text
Continue from v253/v254. Do not tune alpha first. Add a target-blind perceptual
confidence/reliability predictor for source-bank residuals. It should use only
train-fit and policy-val evidence to estimate whether a face/bin/source residual
is safe: multi-source agreement, residual variance, source view diversity,
normal/view consistency, edge support, teacher-gain stability, and parent-color
consistency. Freeze the policy on policy-val, apply to stripped target no-GT
evidence, and require target exact PSNR/SSIM/LPIPS all-axis vs parent before any
full9. Compare against v253b/v253d and report no-GT audit, W&B offline path,
commands, and qualitative target render triplets.
```

Current verdict:

```text
Final status: NOT COMPLETE.
```

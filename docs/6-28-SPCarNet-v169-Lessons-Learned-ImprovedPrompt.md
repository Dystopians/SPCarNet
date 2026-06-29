# SPCarNet v169 Lessons-Learned Improved Prompt

Date: 2026-06-28

This document is a revised server-side Codex handoff prompt after the v162-v168 evidence. It supersedes the broad vNext prompt in spirit. The old direction, "distill Phase-J into a surface representation", is still the right high-level target, but the execution constraints were too soft and produced too many safe-but-weak engineering iterations.

## 1. What Went Wrong In The Previous Prompt

The previous prompt did not fail because the research direction was silly. It failed because it did not encode the real bottleneck sharply enough.

### Problem 1: The prompt optimized for infrastructure before proof of effect

It asked for manifests, full9 runs, ablations, storage accounting, fallback, and galleries. Those are useful, but they made it easy for the agent to spend its budget building a strong protocol shell while the residual representation remained weak.

Lesson: require an early single-scene decisive quality gate before full9, galleries, or large artifact promotion.

### Problem 2: The primary gate was too broad and too late

The old "minimum useful milestone" allowed full9 progress and clean-baseline wins as success signals. The latest evidence shows that this is not enough. vNext full9 can be protocol-complete while still below clean, v106, and Phase-J.

Lesson: the first hard gate must be flowers exact all-axis vs Phase-J:

- PSNR > 20.304358
- SSIM > 0.557770
- LPIPS < 0.329222

No full9 run should be launched before this gate passes.

### Problem 3: It did not explicitly forbid already-failed loops

The prompt said "not another small gate", but it did not name the loops that would later fail:

- footprint expansion without stronger residual content;
- alpha frontier and selected-alpha variants;
- face reliability and face-gain pruning;
- local multisample residual fill;
- simple face-local affine/patch residual fields;
- full9 promotion before flowers all-axis success.

Lesson: the prompt must forbid these as main contributions unless the method changes the residual target or representation.

### Problem 4: It did not require a teacher-signal projection sanity check

v167 filled many bins but the correction direction was unsafe. This should have been caught before a long exact test by asking:

- Does Phase-J teacher residual improve the parent on train-policy-val?
- Can the proposed face/UV carrier represent a meaningful fraction of that teacher residual on policy-val?
- Does projected teacher residual improve SSIM/LPIPS, not only MSE?
- Is the residual being clipped, masked, diluted, or applied to the wrong parent?

Lesson: before implementing a larger method, first measure whether the representation can carry the teacher signal under policy-val.

### Problem 5: It under-specified storage/runtime preflight

v168 exact failed before metrics because evidence reparenting materialized a full copy under a quota-constrained filesystem.

Lesson: the prompt must require storage/quota preflight, low-copy/auto-link modes, and "do not relaunch duplicate exact runs if one is already running".

## 2. More Relevant Related Work And Actionable Lessons

The useful lesson from related work is not "copy their method"; it is "do not use a scalar RGB residual atlas when the missing signal is view-dependent, perceptual, and high frequency".

| family | papers / repos | actionable lesson for SPCarNet |
|---|---|---|
| mesh/surface splatting | [MeshSplatting](https://meshsplatting.github.io/), [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [SuGaR](https://github.com/Anttwo/SuGaR), [GOF](https://niujinshuchong.github.io/gaussian-opacity-fields/), MiLo, Triangle Splatting, 3D Convex Splatting | The field is moving toward surface-aware splatting. Our method should stay MeshSplatting/surface-native, not pure image postprocessing. |
| texture/appearance disentanglement | [Texture-GS](https://github.com/slothfulxtx/Texture-GS), [Textured Gaussians](https://textured-gaussians.github.io/), [GStex](https://github.com/victor-rong/GStex), Content-Aware Texturing for Gaussian Splatting, TextureSplat, [Neural Shell Texture Splatting](https://zhangxin-cg.github.io/nest-splatting/), Neural Texture Splatting, LGTM | Geometry and appearance capacity should be decoupled. A face/primitive needs a texture or feature field, not only one RGB delta. |
| neural texture / deferred rendering | [Deferred Neural Rendering / Neural Textures](https://niessnerlab.org/projects/thies2019neural.html), Deep Blending, [IBRNet](https://ibrnet.github.io/), [Unstructured Lumigraph Rendering](https://cs.harvard.edu/~sjg/papers/ulr.pdf) | Render-time view repair works because it uses view-dependent evidence. To bake Phase-J, use teacher residuals to train a surface feature/decoder, not just average residual colors. |
| compression/distillation | [LightGaussian](https://lightgaussian.github.io/), Mini-Splatting, [PUP 3D-GS](https://pup3dgs.github.io/), [RDO-Gaussian](https://rdogaussian.github.io/), ContextGS, [HAC/HAC++](https://yihangchen-ee.github.io/project_hac/), EAGLES, Self-Organizing Gaussians | Do not sell 7.6479% triangle reduction as compression novelty. Use compaction only as budget accounting; the novelty must be certified appearance transfer/distillation. |
| sparse/generalizable GS | FSGS, DNGaussian, SparseGS, [pixelSplat](https://davidcharatan.com/pixelsplat/), MVSplat, Splatter Image, NoPoSplat, FreeSplatter, InstantSplat | Generalization needs strong priors and multi-view feature aggregation. For this project, target "per-scene train-only teacher distillation" first; do not suddenly claim zero-shot. |
| perceptual optimization | [Drop-In Perceptual Optimization for 3DGS](https://machinelearning.apple.com/research/drop-in), [Perceptual Wrapper / common-randomness style methods](https://arxiv.org/abs/2606.11782), LPIPS/DISTS/SSIM-oriented training | PSNR-improving residuals can hurt SSIM/LPIPS. The fit objective and policy gate must explicitly optimize perceptual/structure axes, not only MSE/L1. |
| teacher/student baking | LightGaussian distillation/pseudo-view augmentation, PlenOctree-style baking, neural texture distillation | Use the strong endpoint as a teacher, but verify the teacher-parent residual survives projection into the student representation. |

Practical conclusion:

The next useful attempt should be a **Phase-J teacher residual projection and neural/low-rank surface texture distillation**, not another sparse footprint or alpha variant.

## 3. Revised Research Direction

Name the next route:

**v169 Phase-J Residual Projection and Baked Surface Feature Distillation**

Core idea:

1. Treat Phase-J train renders as the teacher.
2. Treat Phase-F/v106-compatible parent renders as the parent.
3. Fit a surface-attached residual representation to `teacher - parent`, not merely `GT - parent`.
4. Certify on train-policy-val against real GT.
5. Apply to target/test with GT stripped.
6. Stop immediately unless flowers exact beats Phase-J on PSNR, SSIM, and LPIPS.

### 3.1 Required diagnostics before new large experiments

Run these diagnostics on flowers first:

1. **Teacher gain audit**
   - On policy-val views, compare parent, v106, Phase-J teacher, and candidate.
   - Confirm Phase-J teacher actually improves all axes over the chosen parent.

2. **Teacher residual magnitude audit**
   - Measure `teacher - parent` residual norm, clipped fraction, sign consistency, edge/gradient energy, and face/bin coverage.
   - If residual is near zero, the teacher/parent paths are wrong.

3. **Carrier projection upper bound**
   - Fit the current face/UV/bin carrier on train-fit teacher residual.
   - Evaluate projected residual on policy-val.
   - If this upper bound cannot improve SSIM/LPIPS, do not run exact target; the carrier is too weak.

4. **Mask/dilution audit**
   - Report how much teacher residual is lost by alpha, confidence, selected faces, bin thresholds, target-impact filters, clipping, and fallback.
   - If more than half the teacher residual energy is removed before policy-val, diagnose before adding new methods.

5. **Storage preflight**
   - Check `/data`, `/dev/shm`, `/tmp`, and user quota.
   - Use `auto_link` / low-copy paths.
   - Do not duplicate full evidence caches unless preflight proves enough space.

### 3.2 Representation change that is actually worth trying

Do not implement another scalar RGB atlas or simple per-face affine field as the main method.

Implement one of these, in this priority order:

1. **Low-rank teacher residual texture**
   - Per face or face-group, fit K residual bases, e.g. K=4.
   - Predict mixture weights from view direction, normal, UV/barycentric position, parent color, support count, residual variance, and edge/boundary features.
   - Use ridge/robust fitting and policy-val recertification.

2. **Surface feature texture + tiny decoder**
   - Store compact per-face/UV feature vectors.
   - Decode residual RGB and confidence through a small MLP or linear basis.
   - Train on train-fit teacher residual; certify on policy-val GT.
   - Keep it small enough to run flowers exact first.

3. **Patch/gradient-aware teacher residual target**
   - Fit not only RGB residual but also a structure-preserving component:
     luma residual, gradient residual, or patch-normalized residual.
   - Prefer residuals that improve policy-val SSIM/LPIPS tails, even if MSE gain is smaller.

The important change is the residual target and representation. The target should be Phase-J teacher correction, and the representation should have view-dependent texture capacity.

## 4. Improved Server-Side Codex Prompt

Copy the block below to the server-side Codex.

```text
You are working in the SPCarNet / MeshSplatting repository on the server.

Read these files first and treat them as authoritative:

- feedback.md
- docs/Latest.md
- docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md
- docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md
- docs/car_model/6-27-v149-v151-FaceReliability-SelectedAlpha-Log.md
- docs/car_model/6-26-v118-v119-FaceGraphResidualTransfer-Milestone-And-Bottleneck-Log.md
- scripts/car_model/run_vnext_certified_residual_texture_scene.py
- scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
- scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py
- scripts/car_model/ecsr_reparent_surface_evidence_cache.py

Objective:

Build evidence for or against v169 Phase-J residual projection and baked surface feature distillation. The goal is not another safety gate, alpha scan, footprint expansion, or full9 protocol package. The goal is to prove whether Phase-J teacher residual can be baked into a MeshSplatting-compatible surface representation that beats Phase-J on flowers all-axis, without target/test RGB GT leakage.

Hard current facts:

- Phase-J is the strongest local RGB endpoint: full9 26.482766 / 0.783720 / 0.224261.
- v106 is the strongest verified baked representation: full9 25.831280 / 0.760830 / 0.268435.
- v166 and v167 flowers are failures vs Phase-J: they win PSNR only or fallback no-op, but lose SSIM/LPIPS.
- v168 is protocol-ready but not a metric win unless the exact flowers run has completed after this handoff.
- Do not run full9 until flowers exact beats Phase-J all-axis.

Absolute flowers gate:

- Phase-J flowers reference: PSNR 20.304358, SSIM 0.557770, LPIPS 0.329222.
- A candidate passes only if PSNR is higher, SSIM is higher, and LPIPS is lower.
- PSNR-only wins are failures.

Step 0: status and storage preflight

1. Check git status and do not revert user/local changes.
2. Check whether the v168 direct-teacher low-copy exact run already completed:
   /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers
3. If it completed, parse manifest, no-GT verifier, adapter audit, and test metrics.
4. If it is still running, do not launch a duplicate.
5. If it failed for storage/quota, fix only low-copy/no-copy mechanics first.
6. Before any exact run, report free space for /data, /dev/shm, /tmp, and user quota. Use auto_link or symlink/hardlink modes whenever possible.

Step 1: teacher signal sanity checks

Before implementing a new method, create a small diagnostic script or report for flowers:

1. Compare parent, v106, Phase-J teacher, and existing vNext outputs on train-policy-val views.
2. Verify teacher_render_dir and parent_render_dir differ and produce nonzero teacher-parent residual.
3. Measure teacher-parent residual norm, clipping rate, edge/gradient energy, face/bin coverage, sign consistency, support count, and residual variance.
4. Estimate how much residual energy is removed by current masks, alpha, selected faces, bin thresholds, clipping, and target-impact filters.
5. Save the diagnostic as a Markdown + JSON report.

Step 2: carrier projection upper bound

Do not go straight to test. First measure whether the current carrier can represent the teacher:

1. Fit the current face/UV/bin residual carrier on train-fit Phase-J teacher residual, i.e. teacher render minus parent render.
2. Evaluate on policy-val against GT.
3. Report parent vs projected-teacher carrier on PSNR/SSIM/LPIPS, positive-view fraction, min-view gain, and CVaR/tail gains.
4. If projected-teacher carrier cannot improve SSIM and LPIPS on policy-val, stop and report "current carrier too weak"; do not launch flowers exact.
5. If projection helps policy-val all-axis, proceed to Step 3.

Step 3: one real representation change

Implement exactly one representation upgrade, not a bundle of tuning:

Preferred v169 candidate:

- low-rank Phase-J teacher residual texture;
- per face or face group, fit K residual bases, start with K=4;
- predict mixture weights from UV/barycentric coordinates, view direction, normal/depth/boundary features, parent RGB, support count, residual variance, and teacher confidence;
- train/fill only from train-fit teacher residual;
- certify on policy-val GT;
- apply to target/test with GT stripped;
- output = parent + confidence * residual.

Alternative if MLP/low-rank is too heavy:

- patch/gradient-aware teacher residual target;
- fit luma + gradient/edge residual components instead of raw RGB only;
- select the component that improves policy-val SSIM/LPIPS tails.

Do not make the main change any of the following:

- alpha-grid scan;
- face reliability gate only;
- target footprint expansion only;
- local multisample fill only;
- simple face-local affine/ridge patch field only;
- full9 manifest packaging;
- qualitative-only panel generation.

Step 4: policy-val certificate

Keep strict no-target-GT apply:

- target/test RGB GT and target residual GT must be stripped before apply;
- policy-val may use GT for certification;
- target/test GT may appear only after apply for final evaluation;
- write no-GT verifier output;
- write audit fields for uses_train_fit_gt, uses_policy_val_gt, uses_target_or_test_gt.

The gate must include:

- PSNR, SSIM, LPIPS;
- positive-view fraction;
- min-view and CVaR/tail gains;
- image L1 / gradient or SSIM proxy if implemented;
- target OOT/camera support if available;
- fallback/no-op with explicit reason.

Step 5: flowers exact only

Run only flowers exact first. Compare against Phase-J flowers:

- PSNR > 20.304358
- SSIM > 0.557770
- LPIPS < 0.329222

If not all three pass:

- mark result FAIL, even if PSNR improves;
- do not run full9;
- diagnose whether failure was teacher residual zeroing, carrier under-capacity, mask/alpha dilution, clipping, target support mismatch, or policy-val rejection;
- write a concise report with the next bottleneck.

If all three pass:

- freeze the policy and command line;
- run fixed-policy full9;
- compare clean MeshSplatting, v106, Phase-J, old vNext, and v169;
- include ablations: parent only, teacher projection carrier, v169 without view-dependent weights, v169 full.

Deliverables:

1. Markdown report under docs/car_model/ with:
   - exact command lines;
   - storage preflight;
   - teacher signal diagnostics;
   - carrier projection upper bound;
   - flowers exact metrics;
   - no-target-GT audit;
   - pass/fail verdict.
2. JSON summaries for:
   - metrics;
   - policy decisions;
   - no-GT verifier;
   - teacher residual projection diagnostics.
3. If flowers passes, then and only then produce full9 report and qualitative panels.

Completion standard:

The task is complete only when either:

A. flowers exact beats Phase-J all-axis and the fixed policy is ready for full9; or
B. a diagnostic report proves why the current Phase-J-distilled carrier cannot improve SSIM/LPIPS, with no target/test GT leakage.

Do not claim paper readiness until fixed-policy full9 also passes.
```

## 5. Short Recommendation

The next agent should not be asked to "make vNext better" in general. It should be asked to answer one sharper question:

> Can Phase-J's teacher-parent residual be projected into a baked surface representation that improves flowers all-axis against Phase-J without target/test GT leakage?

If the answer is no under the current carrier, the correct output is a negative diagnostic and a new representation proposal, not another full9 protocol run.

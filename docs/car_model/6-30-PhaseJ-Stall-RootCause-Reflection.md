# 2026-06-30 Phase-J Stall Root-Cause Reflection

This note answers why the current SPCarNet/v169 representation route keeps
stalling below the Phase-J reference, despite many implementation and evaluation
iterations.

## Executive Verdict

The current route is not failing because of one missing hyperparameter, one
unlucky GPU run, or one missing full9 sweep. It is failing because the accepted
v169 representation line can only transfer a very small amount of correct
Phase-J-like RGB residual energy to target views.

The strongest recent flowers exact results are:

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| parent | 19.832054 | 0.619910 | 0.180335 | reference parent |
| v292d balanced frontier | 19.851452 | 0.620343 | 0.180212 | beats parent all-axis, fails Phase-J PSNR |
| v293a PSNR frontier | 19.853420 | 0.620328 | 0.180312 | best recent PSNR, worse perceptual tail |
| Phase-J flowers gate | 20.304358 | 0.557770 | 0.329222 | required reference |

Under this metric scale, current v292/v293 already clear the reported Phase-J
SSIM and LPIPS thresholds. The blocker is PSNR: the best recent route is still
about `0.451 dB` below Phase-J.

## Why The Gap Is Not Small

PSNR makes the gap look moderate, but MSE exposes the scale:

| item | PSNR | MSE |
|---|---:|---:|
| parent | 19.832054 | 0.010394285 |
| v293a | 19.853420 | 0.010343274 |
| Phase-J | 20.304358 | 0.009323183 |

v293a reduces MSE by about `5.10e-05` over the parent. Phase-J requires about
`1.071e-03` MSE reduction over the parent. Current v293a therefore captures only
about `1 / 21` of the needed MSE reduction.

This is the central reason the work feels stuck: many variants are real, but
they are moving the needle in the wrong order of magnitude.

## What The Experiments Actually Proved

### 1. The teacher signal exists

Phase-J is not a weak or empty teacher. Earlier audits showed strong policy-val
teacher headroom, and the v169 prompt correctly asked for teacher residual
projection before launching more expensive promotions.

This rules out the explanation that "Phase-J has no useful residual to distill."
The useful signal exists.

### 2. The current carrier cannot project enough of that signal

The v294 teacher projection upper-bound diagnostic directly tested the current
face/UV/low-rank carrier on train-fit Phase-J residuals and certified on
train-policy-val only.

Best candidate:

| item | value |
|---|---:|
| texture size | 8 |
| low-rank rank | 4 |
| alpha | 0.03125 |
| full-image PSNR gain | +0.000163820 dB |
| SSIM gain | +0.000000392 |
| LPIPS gain | +0.000000956 |
| SSIM positive-view fraction | 0.500000 |
| LPIPS positive-view fraction | 0.666667 |
| robust all-axis pass | false |

This is the strongest evidence so far. Even when the carrier is given the
teacher residual fitting problem directly, the useful full-image effect is
near noise level. Increasing alpha/rank raises MSE slightly but quickly damages
SSIM/LPIPS tails. Therefore the main limitation is not alpha selection.

### 3. Cross-view residual direction is unreliable

Source-heldout diagnostics from v285/v286 show:

| statistic | value |
|---|---:|
| heldout residual direction cosine | 0.214671 |
| heldout error ratio | 2.078181 |
| v285b target PSNR gain | +0.010698 |
| v286b target PSNR gain | +0.008856 |
| Phase-J PSNR gap after these runs | about -0.462 dB |

The cosine is too low for confident target-view residual transport. The error
ratio says a residual predictor fitted from part of the source evidence is still
substantially wrong on heldout source views. Reliability calibration can suppress
bad tails, but it cannot create missing residual energy.

### 4. Policy-val success has not meant target success

Recent models often look stronger on policy-val than on exact target:

| run | policy-val PSNR gain | target PSNR gain | target issue |
|---|---:|---:|---|
| v293a | +0.028777 | +0.021366 | perceptual tails worsen |
| v292d | +0.024338 | +0.019398 | safe but still weak |
| v289c | policy-val positive | +0.009785 | source weighting too small |

This means the policy-val split is useful, but it is not enough to certify the
cross-view direction of high-frequency residuals. We need a stronger heldout
transport objective, not just a stronger post-hoc gate.

### 5. More coverage did not translate into quality

Earlier v165/v166 results showed that expanding target-impact footprint or
filling more bins can be mechanically successful while visually and perceptually
ineffective. The added pixels/bins did not carry the right residual direction.

This killed the "just affect more pixels" hypothesis.

### 6. More capacity helped only slightly

PatchViewMoE, low-rank view support, texture latent embeddings, local ridge
features, and direct RGB heads were real representation changes. They were not
just logging or parameter scans. But their measured effect remained small:

- v292d became a better balanced parent win.
- v293a became the best recent PSNR frontier.
- Neither moved close to Phase-J.
- Added capacity often increased perceptual tail risk.

The issue is not that the code never changed. The issue is that the changes
mostly improved safe local residual fitting, not reliable target-view residual
transport.

## Why Some Historical Results Look Confusing

There are three different result families that must not be mixed:

1. Phase-J endpoint/reference:
   strong RGB endpoint, not the desired baked representation.

2. Older or scene-limited wins:
   some versions passed one scene or one axis. For example v165/v166 won PSNR on
   flowers but lost SSIM/LPIPS; v191 passed flowers all-axis but did not close
   counter and teacher-only/representation validity.

3. Current v169/vNext accepted representation line:
   stricter no-target-GT, baked/surface-compatible, policy-val certified route.
   This is the route currently stuck below Phase-J PSNR.

The apparent contradiction comes from mixing these families. A fair paper claim
cannot use a single-scene or GT-dependent endpoint to declare the representation
route solved.

## Root Cause

The short version:

> We are trying to bake a view-dependent, high-frequency, perceptual Phase-J
> correction into a sparse face/UV/bin residual carrier whose cross-view
> residual direction is weak and whose reliable coverage is too small.

More concretely:

1. The Phase-J residual is view-dependent.
2. The current carrier stores or predicts residuals in a mostly local surface
   coordinate system.
3. Target views often observe the same surface under different visibility,
   support, parent color, edge, and normal/view configurations.
4. Current gates can detect some unsafe regions, but then they shrink useful
   residual energy.
5. If gates are relaxed, PSNR rises a little while SSIM/LPIPS tails degrade.
6. If gates are tightened, perceptual safety improves but the method becomes too
   close to no-op.

This is why the loop keeps returning to small `+0.01` to `+0.02 dB` target PSNR
gains instead of Phase-J-scale improvement.

## Engineering Reflection

The repository now has a much stronger evaluation shell than before:

- strict no-target-GT apply checks;
- policy-val and target exact separation;
- W&B offline logging for medium/exact runs;
- baseline, Phase-J, v106, vNext comparison records;
- projection upper-bound diagnostics;
- source-heldout residual direction diagnostics;
- documented negative results.

That is real progress, but it is not the same as a final paper method. We spent
too much effort making weak residual mechanisms safe and auditable. Safety made
the failures trustworthy, but it did not make the method strong.

There is also a current implementation hygiene issue: texture-bin reliability
calibration has been partially introduced in
`scripts/car_model/train_perceptual_surface_residual_decoder.py`, but its CLI,
target exact path, and prediction path still need to be fully wired before it
can be treated as a clean experimental feature. This is not the root cause of
the Phase-J stall, but it must be closed before using that branch for new claims.

## What Should Stop

The following should not remain the main route:

- more alpha/rank scans on the same carrier;
- footprint expansion without a stronger residual target;
- scalar face/bin reliability gates as the claimed innovation;
- target-compatible source weighting as the main novelty;
- local ridge/affine/patch fills without source-heldout direction supervision;
- full9 promotion before flowers exact beats Phase-J all-axis.

These are useful diagnostics or components, not the missing breakthrough.

## What Should Happen Next

The next credible method should directly model residual transport:

1. Build a cross-view residual direction predictor.
   - Train with source-heldout loss.
   - Predict RGB residual and confidence together.
   - Use source-view diversity, heldout cosine/error, normal-view agreement,
     parent color, edge/structure, UV position, and support statistics.

2. Replace single-bin texture latent with multi-source residual evidence.
   - Store source residual sets or a source-conditioned basis.
   - Decode target residual from target/source compatibility rather than only
     from a static UV bin code.

3. Keep Phase-J teacher projection as a required upper-bound check.
   - If the new carrier cannot project teacher residual on policy-val with
     robust SSIM/LPIPS tails, do not launch exact target or full9.

4. Separate scientific claims:
   - Phase-J is the endpoint to beat.
   - v106 is the strongest current baked baseline.
   - v169/vNext is still an active research route, not a solved paper result.

## Current Honest Status

Current status is `NOT COMPLETE`.

The current v169/vNext route has produced useful engineering infrastructure and
clear negative evidence, but it has not yet produced a representation-level
method that beats Phase-J all-axis under the required gate. The strongest
explanation is a representation and residual-transport bottleneck, not lack of
training length or missing full9 runs.

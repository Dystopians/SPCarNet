# 2026-06-30 Phase-J Stall Root-Cause Reflection

This note records the direct answer to a hard question:

```text
Why do recent SPCarNet representation-level attempts keep stalling below
Phase-J?
```

The short answer is:

```text
Phase-J is a high-bandwidth render-time residual transport endpoint.
The later baked/representation routes compress that transport into a much
weaker face/UV/bin/latent carrier, and the learned residual direction is not
stable enough across views.
```

Therefore the current blocker is not a missing alpha, rank, threshold, GPU run,
or W&B log.  It is a method-family bottleneck.

## Current Hard Numbers

The relevant flowers Phase-J reference is:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J flowers gate | 20.304358 | 0.557770 | 0.329222 |

Recent representation-level flowers results remain below that PSNR gate:

| method | PSNR | SSIM | LPIPS | PSNR gap vs Phase-J | reading |
|---|---:|---:|---:|---:|---|
| parent | 19.832054 | 0.619910 | 0.180335 | -0.472304 | direct parent |
| v292d | 19.851452 | 0.620343 | 0.180212 | -0.452906 | best balanced recent route |
| v293a | 19.853420 | 0.620328 | 0.180312 | -0.450938 | best recent PSNR route |

The v293a improvement captures only about `4.76%` of the parent-to-Phase-J MSE
reduction on flowers.  This is not a near miss.

## What Phase-J Has That The Baked Carrier Lacks

Phase-J / ELA is not a static texture update.  At render time it:

1. selects nearby support training views for the current target view;
2. projects target pixels into support views using target depth and camera
   geometry;
3. samples support residual images;
4. fuses residuals per target pixel using confidence, support, edge, and policy
   gates;
5. applies train/policy-val selected alpha or an edge/fallback branch.

In code this path is centered around:

- `utils/evidence_lumigraph_adapter.py::select_support_frames`
- `utils/evidence_lumigraph_adapter.py::warp_support_residual`
- `utils/evidence_lumigraph_adapter.py::compute_evidence_signal`
- `utils/evidence_lumigraph_adapter.py::adapt_frame`

The baked/representation route instead tries to compress the same information
into surface features such as face IDs, UV bins, low-rank texture coefficients,
texture latents, source-view summary statistics, and a decoder MLP.  That route
does not keep the full target-conditioned support-image residual path.

This is the central information-path mismatch.

## Evidence That This Is Not Just A Parameter Problem

### v294 Projection Upper Bound

The v294 teacher projection diagnostic gave the current face/UV/low-rank carrier
a favorable policy-val projection setup.

Best row:

| field | value |
|---|---:|
| texture size | 8 |
| rank | 4 |
| alpha | 0.03125 |
| PSNR gain | +0.000163820 |
| SSIM gain | +0.000000392 |
| LPIPS gain | +0.000000956 |
| robust all-axis pass | false |

This says the carrier can produce a tiny all-axis numerical gain, but not enough
to justify flowers exact or full9 promotion.

### v285/v286 Source-Heldout Direction Test

The source-heldout diagnostic estimated whether residual directions learned from
one source split transfer to heldout source views.

| statistic | value |
|---|---:|
| heldout residual direction cosine | 0.214671 |
| heldout error ratio | 2.078181 |

A direction cosine around `0.21` is too weak for confident cross-view residual
transport.  Calibration can detect the risk, but it does not create the missing
directional signal.

### v296 Heldout Feature Scaffold

v296 added target-blind heldout direction features through:

```text
--surface_texture_mode lowrank_view_holdout_v3
```

Reduced same-budget testing selected `alpha=0.0` for both v2 and v3.  Nonzero
alpha rows had only microscopic changed fractions and did not produce a robust
all-axis signal.

This proves that heldout diagnostics as features are not enough.  They tell the
model where it is uncertain; they do not force the model to learn transport.

### v297 Source-Heldout Transport Loss

v297 made the first correct objective-level move:

```text
--enable_source_heldout_transport_loss
```

It splits train-fit views into source and heldout-source subsets, builds a
source-only surface texture, and adds heldout-source residual prediction loss.

The interface works, but the first pilot is still not a quality breakthrough:

| run | selected alpha | PSNR gain | SSIM gain | changed fraction | pass |
|---|---:|---:|---:|---:|---|
| no transport pilot | 0.12500 | -0.000000388 | +0.0000000715 | 0.000015655 | false |
| transport fixed-gate pilot | 0.12500 | -0.0000000512 | +0.0000000834 | 0.000020517 | false |

The wider alpha diagnostic is decisive:

| alpha | PSNR gain | SSIM gain | changed fraction |
|---:|---:|---:|---:|
| 0.125 | -0.000000051 | +0.000000083 | 0.000020517 |
| 0.250 | -0.000000329 | +0.000000167 | 0.000050166 |
| 0.500 | -0.000001632 | +0.000000310 | 0.000072225 |
| 1.000 | -0.000006476 | +0.000000513 | 0.000088947 |

Increasing alpha raises changed fraction but makes PSNR worse.  The immediate
problem is wrong residual direction, not merely under-application.

## What Went Wrong In The Research Loop

The loop made real engineering progress:

- no-target-GT apply and target exact separation;
- W&B offline logging for medium/exact runs;
- fair clean MeshSplatting baseline records;
- Phase-J, v106, vNext, v169/v297 artifact records;
- policy-val, tail-risk, no-op fallback, and changed-fraction gates;
- projection upper-bound and source-heldout diagnostics.

Those pieces are valuable.  They made the failure visible.

The mistake was over-investing in making a weak residual carrier safe.  Safety
can prevent regressions, but it cannot turn an underpowered representation into
a high-bandwidth residual transport model.

The failed assumption was:

```text
If the surface residual carrier is safer, more adaptive, and more feature-rich,
it will eventually carry Phase-J-like corrections.
```

The evidence now says:

```text
The carrier/objective is not learning stable cross-view residual direction.
More gates mostly decide when not to apply it.
```

## What Should Stop

These should not remain the main route:

- alpha/rank/floor scans on the same carrier;
- adding scalar reliability features without a transport objective;
- footprint expansion without stronger residual direction learning;
- global or per-bin alpha tuning as the headline novelty;
- full9 promotion before flowers exact closes the Phase-J PSNR gap;
- treating near-no-op policy-val numerical positives as success.

## What The Next Real Method Must Do

The next credible paper-level attempt must preserve more of Phase-J's
target-conditioned information path while staying train-only at decision time.

Minimum viable direction:

1. keep a support-view residual feature bank instead of only static face/UV bins;
2. train source-A to heldout-source-B residual transport explicitly;
3. condition the decoder on target view geometry, normal/view angle, depth
   consistency, source-view agreement, and occlusion uncertainty;
4. supervise RGB residual, direction cosine, magnitude, and patch/perceptual
   structure on heldout-source views;
5. use policy-val only as certification and fixed-policy selection after the
   model has learned nontrivial transport;
6. require nontrivial changed fraction, positive view fractions, and tail-safe
   PSNR/SSIM/LPIPS before target exact or full9.

## Updated Success Criteria

Before another full9 promotion, the improved method should show:

| gate | minimum expectation |
|---|---|
| policy-val | nonzero selected alpha with nontrivial changed fraction |
| flowers exact | at least `+0.10 dB` over v293a before claiming a real breakthrough |
| Phase-J gap | materially reduced, not `0.001 dB` scale |
| tails | no SSIM/LPIPS tail damage under fixed policy |
| protocol | target/test GT used only for final reporting |
| evidence | commands, configs, W&B paths, renders, metrics, and errors documented |

## Final Verdict

```text
Final status: NOT COMPLETE.
```

The current work has a strong Phase-J endpoint and a strong audit shell, but the
representation-level internalization of Phase-J is still unresolved.  The next
step is not another parameter sweep.  It must be a higher-bandwidth,
target-conditioned, source-heldout-supervised residual transport model.

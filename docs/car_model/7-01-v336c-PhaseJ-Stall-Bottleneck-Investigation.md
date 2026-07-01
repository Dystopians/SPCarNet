# v336c Phase-J Stall and Oracle-Gap Bottleneck Investigation

Date: 2026-07-01

## Executive Summary

v336c is a verified reliability milestone, but it does not close the paper-level
gap. It improves the v33x support-transport frontier over v335 while preserving
9/9 non-regression, yet the remaining headroom is still dominated by per-view
candidate selection and weak pre-arbitration candidate generation.

The strongest current conclusion is:

```text
The v33x line has built a useful audit and safety shell, but it has not replaced
Phase-J's much stronger render-time ELA endpoint. More arbitration alone is
saturated; the next real method must improve the raw residual/candidate field or
learn a stronger target-blind per-view ranker from richer no-GT evidence.
```

## Comparable Metric Mouths

These result mouths are not interchangeable.

| mouth | comparable rows | result | interpretation |
|---|---|---:|---|
| v33x full9 apply PSNR/SSIM gain | v329b/v334/v335/v336c | v336c `0.274617 / 0.003745`; v335 `0.274018 / 0.003742` | v336c is only `+0.0005995` PSNR gain over v335; the gain is tiny and mostly room-local. |
| v33x frontier 512 LPIPS/DISTS | clean26000/v335/v336c | clean26000 `27.193643 / 0.029112 / 0.090207 / 0.059902`; v336c `27.590966 / 0.028167 / 0.087738 / 0.057667` | v336c is above clean26000 in this support-transport frontier mouth. |
| original selected-clean full9 RGB | clean MeshSplatting / Phase-J | clean `25.151682 / 0.749018 / 0.287621`; Phase-J `26.482766 / 0.783720 / 0.224261` | Phase-J remains the stronger endpoint by `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS. |
| PSNR-only bridge | v336c selected PSNR vs selected clean PSNR | v336c selected PSNR mean `25.415012`; clean `25.151682` | v336c is above original clean on PSNR only by `+0.263330`; no comparable full9 LPIPS/SSIM mouth should be claimed from this bridge. |

Risk: v33x `selected PSNR/SSIM gain` is a gain over that run's fixed/base
support-transport row, not the original MeshSplatting baseline. Frontier
metrics are internally valid but use a different LPIPS/DISTS mouth. Do not claim
that v336c dominates Phase-J or fully replaces the original MeshSplatting
evaluation protocol.

## What v336c Actually Fixes

v336c adds:

- an `adaptive` generated residual candidate;
- source-heldout admission for generated candidates;
- fixed-scene suppression and generated-candidate filtering so weak generated
  candidates cannot perturb the learned policy pool.

It fixes the v336b garden regression while keeping the room gain:

| comparison | dPSNR | dSSIM | nonnegative PSNR scenes | nonnegative SSIM scenes |
|---|---:|---:|---:|---:|
| v336c - v335 | +0.000599514552 | +0.000003450447 | 9/9 | 9/9 |
| v336c - v336b | +0.000033480213 | -0.000000108762 | 7/9 | 7/9 |

But the admission pattern is narrow: adaptive is accepted only on `room`, and it
is suppressed as source-summary unsafe on six scenes or disabled when the scene
selects `fixed`.

## Remaining Oracle Headroom

Two oracle diagnostics were archived:

```text
docs/car_model/results/v336c_strict_oracle_gap_vs_v335.json
docs/car_model/results/v336c_strict_oracle_gap_vs_v335.md
docs/car_model/results/v336c_psnr_primary_oracle_gap_vs_v335.json
docs/car_model/results/v336c_psnr_primary_oracle_gap_vs_v335.md
scripts/car_model/analyze_support_transport_oracle_gap.py
```

Strict oracle means a candidate must be non-worse than the selected output on
both `psnr_gain` and `ssim_gain` before it can be chosen.

| oracle mouth | v335 headroom | v336c headroom | interpretation |
|---|---:|---:|---|
| strict PSNR+SSIM non-regressive oracle | +0.008669481579 | +0.008649866410 | v336c barely reduces safe per-view headroom. |
| PSNR-primary oracle | +0.009594446104 | +0.009561210143 | v336c barely reduces maximum PSNR headroom. |

Largest v336c strict misses:

| scene/view | selected | strict oracle | dPSNR | dSSIM |
|---|---|---|---:|---:|
| bonsai/00035 | learned | fixed | +0.256317117624 | +0.002426087856 |
| room/00011 | hybrid | learned | +0.121721897073 | +0.000903427601 |
| treehill/00011 | fixed | learned | +0.097477838368 | +0.000548839569 |
| treehill/00016 | fixed | learned | +0.084025668437 | +0.000266134739 |
| room/00023 | hybrid | learned | +0.081403643404 | +0.000925242901 |
| stump/00014 | fixed | learned | +0.079235783778 | +0.000358104706 |
| kitchen/00018 | learned | fixed | +0.077108477521 | +0.000544965267 |

The pattern is not a simple learned-vs-fixed bias. Many misses are
`fixed/hybrid -> learned`, but the largest single miss is `learned -> fixed`,
and kitchen has another learned-to-fixed failure. A naive learned-biased rule
would overfit the oracle table and break safety.

## Why Repeated Arbitration Stalls

The post-v322C sequence is mostly arbitration and safety:

| step | role | incremental character |
|---|---|---|
| v333/v334 | target-neighbor rollback and contradiction certificates | one/few-view tail repairs |
| v335 | narrow fixed-to-learned unlock | treehill-local unlock |
| v336c | generated-candidate admission safety | room-local adaptive admission |

This line is valuable because it made decisions auditable and no-target-GT at
apply time. It is not enough for a paper endpoint because the raw candidate
field is weak and the learned policy cannot reliably identify the strict oracle
candidate on the largest-miss views.

Prior stall evidence points the same way:

- favorable projection upper bounds were too small to bridge Phase-J;
- heldout residual direction alignment was weak;
- representation-level residual carriers captured only a small fraction of the
  parent-to-Phase-J MSE reduction;
- pure target-neighbor ranking hurt macro metrics, so target-neighbor evidence
  is useful as a certificate but not as a standalone selector.

## Next Method Requirement

A credible next method must satisfy all of the following before promotion:

1. Improve pre-arbitration candidate quality or no-target-GT per-view ranking on
   the large-miss focus set: `stump`, `treehill`, `room`, `bicycle`, with
   `bonsai/kitchen` as learned-to-fixed safety controls.
2. Recover a material portion of the `+0.00865` strict oracle headroom without
   using target/test GT for decisions.
3. Preserve v336c-style source-summary admission and rollback certificates.
4. Show nontrivial changed fraction and visible crops, not only tiny metric
   movement.
5. Re-run full9 and frontier LPIPS/DISTS only after the focused diagnostic is
   positive.

The immediate engineering tool gap is an all-candidate target-neighbor/proxy
rank diagnostic inside the apply pipeline. Current reports already contain
candidate metrics, but they do not store target-neighbor scores for every
candidate. That diagnostic should be optional and read-only with respect to
selection, so it can expose whether a target-blind ranker has enough signal
before we build another policy.

## Claim Boundary

Defensible now:

- v336c is a target/test-GT-free generated-candidate admission certificate that
  fixes v336b's garden regression while preserving room gain.
- v336c is stronger than v335 on the v33x support-transport full9 replay and
  remains above local clean26000 in the support-transport frontier mouth.
- The current bottleneck has been diagnosed: remaining headroom is per-view
  candidate selection and raw residual capacity, not documentation or README
  framing.

Not defensible now:

- v336c is a final paper method.
- v336c broadly surpasses Phase-J.
- arbitration-only progress will plausibly close the remaining gap.

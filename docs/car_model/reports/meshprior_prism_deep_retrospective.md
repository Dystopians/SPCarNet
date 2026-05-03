# PRISM MeshPrior Deep Retrospective

Date: 2026-05-02

## Executive Summary

This project did not reach the original ambition. The codebase transformation is substantial and the PRISM infrastructure is real, but the empirical payoff is far too small for a NeurIPS-level method claim. The strongest current result is not a breakthrough reconstruction method; it is an auditable topology-control mechanism with modest medium-budget evidence. That is useful engineering, but it is not enough for a top-tier paper.

The central failure is imbalance: we introduced many mechanisms, gates, schedules, diagnostics, W&B runs, reports, and paper assets, but the final public-scene improvement is tiny. On `bonsai`, Stage35 improves Stage33 by only `+0.067 dB` PSNR, `+0.001084` SSIM, `-0.000644` LPIPS, and `-512` triangles. On `courtyard`, topology/PSNR/SSIM improve, but LPIPS regresses. This is not the scale of evidence that justifies a complex new method.

If the target is NeurIPS-level publication, the current path should not be treated as nearly complete. It should be treated as a negative research result plus an engineering platform that may support a future pivot.

## What We Set Out To Do

The broad goal was to build a method around Mesh Splatting / MeshPrior that could optimize scene meshes under camera and COLMAP evidence, potentially connecting to car/parking-scene priors and eventually a stronger paper narrative.

The implicit target was stronger than merely "make a system run." A top-conference method would need at least one of the following:

- clear quality improvement over strong baselines;
- strong topology or memory reduction at matched quality;
- a genuinely new optimization principle with convincing ablations;
- robust cross-scene full-budget evidence;
- a compelling task formulation that existing methods do not address.

The final state does not satisfy those standards.

## What Actually Happened

The work drifted into a long sequence of incremental controller mechanisms:

- proposal/gate/rollback plumbing;
- candidate caps;
- adaptive retry schedules;
- microbatch gates;
- candidate-quality ranking;
- measured impact ranking;
- calibration-view diversity;
- post-commit refresh;
- retained relaxed refresh;
- metric reconciliation and paper packaging.

Each step was rational locally. Most addressed a real bug, failure mode, or ambiguity. But the cumulative result is a large and complex system whose measured gains remain marginal.

The strongest final selected rows are:

| scene | comparison | result |
|---|---|---|
| `bonsai` | Stage35 vs Stage33 | topology `633275` vs `633787`; PSNR `12.267367` vs `12.199921`; SSIM `0.277617` vs `0.276533`; LPIPS `0.611939` vs `0.612583` |
| `courtyard` | Stage35 vs selected M32 row | topology `101913` vs `102404`; PSNR `15.383161` vs `15.138977`; SSIM `0.508091` vs `0.484960`; LPIPS `0.584694` vs `0.579188` |
| parking | M24.2 long-budget row | useful single-scene topology evidence, but not public-scene generalization |

These are not enough. The `bonsai` improvement is too small. The `courtyard` result is mixed. The parking row is local and cannot carry a general claim.

## Why The Result Is Not Paper-Strong

### 1. The effect size is too small

The public-scene improvements are around the noise level for a complex training system. A reviewer would reasonably ask whether the gain survives seeds, full-budget training, stronger baselines, or more scenes. We do not have enough evidence to answer yes.

For a topology-control paper, reducing `512` triangles from a `~633k` mesh is not a meaningful compression result. It is about `0.08%`. That is not a topology-efficiency story.

For an image-quality paper, `+0.067 dB` PSNR on one scene is not a compelling improvement. It is especially weak when the second public scene has an LPIPS regression.

### 2. The method complexity is disproportionate

PRISM accumulated many control layers. Each layer adds implementation and explanation burden:

- scoring;
- gating;
- rollback;
- calibration selection;
- measured ranking;
- post-commit candidate discovery;
- relaxed scoring;
- retained commit caps;
- validation rollback accounting.

This complexity would be acceptable if the effect were large. With tiny gains, the method looks over-engineered. A reviewer is likely to see a pile of heuristics rather than a clean algorithmic contribution.

### 3. The innovation is not sharp enough

"Auditable topology-control with rollback" is a reasonable engineering contribution, but it is not obviously a new scientific principle. The current implementation is mostly a safety controller around pruning decisions. It does not yet introduce a powerful new objective, representation, or optimization scheme.

The strongest conceptual idea, "prior proposes, scene evidence disposes," is good framing. But the actual experiments do not show that this principle unlocks a capability that baseline methods lack.

### 4. The baseline story is weak and unstable

There are several possible baselines:

- clean Mesh Splatting;
- sparse-depth Mesh Splatting;
- prior PRISM stages like M29/M33;
- post-hoc simplification rows;
- mesh-aligned Gaussian methods such as SuGaR/MeshGS;
- simple geometry-aware pruning.

The final table compares selected internal rows more than it defeats strong external baselines. This is a major weakness. The method cannot claim broad superiority if the baseline definition keeps shifting.

### 5. Full-budget public-scene validation is missing

We have long-budget parking evidence, but public scenes are medium-budget. If the claim concerns general scene optimization, public full-budget evidence matters. We deliberately did not run it because the current table did not justify more GPU time. That was the right pragmatic decision, but it leaves the paper claim weak.

### 6. The original car/SP-CarNet narrative was not preserved

Earlier work involved object priors and vehicle-centered repair. The current strongest method has mostly shed that narrative. That is not automatically bad, but it means the final story is not the one initially imagined. The project became a mesh topology controller, not a car-prior scene optimizer.

## What Went Well

This was not useless. Several pieces are genuinely valuable:

- The codebase now has robust W&B logging, final-cleanup accounting, retained topology audits, and reproducible report generation.
- The project identified real failure modes:
  - topology sync can eliminate candidate pools;
  - local counterfactual gates can pass edits that recovery validation later rejects;
  - training-time metrics and independent render metrics differ substantially;
  - public dataset geometry observability matters.
- M35's audit mechanism is sound: it distinguishes total relaxed attempts from active retained commits.
- M36-M43 created a clean evidence package, manuscript draft, reproducibility appendix, and reviewer-risk checklist.

These are good foundations for future work. They are not sufficient results for a top conference.

## What Went Wrong

### Local progress was mistaken for global progress

Each prompt solved the next visible failure. That created a sense of progress. But the global objective, a strong publishable method, was not improving at the same rate. The project needed an earlier hard stop: "if this does not reduce topology by at least X% at matched quality, pivot."

### The acceptance gates became too conservative for meaningful topology change

The controller was designed to avoid degradation. It succeeded too well. It allowed only tiny edits, and when relaxed edits were attempted, many were rolled back. The final result is safe but timid.

### We optimized around symptoms rather than reframing the objective

Candidate caps, microbatches, measured ranking, and refresh logic all address symptoms of the same issue: the method lacks a strong topology objective. It tries to find safe triangles to remove, but it does not solve a constrained optimization problem like:

> minimize triangle count subject to bounded render degradation.

Without that explicit objective, the system remains a chain of heuristics.

### We over-invested in paper packaging before the result deserved it

M36-M43 are useful, but they are also premature if the goal is a NeurIPS paper. The current manuscript package makes the evidence legible; it does not make the evidence strong.

## Current Honest Status

The right status is:

- Code/infrastructure: strong.
- Reproducibility: strong.
- Diagnostics: strong.
- Empirical method result: weak.
- Innovation: underdeveloped.
- NeurIPS readiness: not close enough.

The earlier estimate of `97%` paper readiness was too optimistic if interpreted as "ready for top-tier submission." A more honest estimate:

- engineering package readiness: `90-95%`;
- paper-result strength: `35-45%`;
- NeurIPS-level method readiness: `25-35%`.

## What Would Be Needed To Rescue This Direction

The path forward cannot be "add another small PRISM gate." It needs a stronger objective and stronger evidence.

### Option A: Make it a topology-compression Pareto paper

This is the most plausible rescue.

Target claim:

> At matched independent render quality, PRISM reduces explicit mesh topology by 30-70% across public scenes.

Required changes:

- Optimize against explicit topology budgets.
- Report Pareto curves, not single rows.
- Compare against post-hoc simplification and simple pruning baselines.
- Use at least 3-5 public scenes.
- Define acceptable degradation thresholds in advance.

This could turn the work into a real paper if the curves are strong.

### Option B: Pivot to a clean constrained optimizer

Replace heuristic gates with a clearer formulation:

> minimize topology cost + render loss + sparse-geometry penalty under validation constraints.

The current PRISM pieces could become implementation support for this, but the method would need a cleaner mathematical story.

### Option C: Rebuild the dataset/task around car-scene priors

If the user's original final goal is vehicle/parking-scene mesh optimization, then the current public `bonsai/courtyard` scenes are not ideal. A better task might use:

- multiple parking-lot COLMAP scenes;
- vehicle masks/instances;
- car mesh priors;
- explicit car-region topology and geometry metrics.

This would revive the SP-CarNet/vehicle-prior narrative, but it requires dataset work, not more PRISM micro-adjustments.

### Option D: Stop aiming for NeurIPS with this result

This is also reasonable. The current system could become:

- an internal technical report;
- an arXiv systems note;
- workshop paper;
- codebase foundation for a future stronger method.

That may be more honest than stretching weak improvements into a main-conference claim.

## Recommended Next Step

Do not continue incremental PRISM modifications immediately.

The next step should be a short, decisive Pareto feasibility experiment:

1. Pick `bonsai`, `courtyard`, and one more public COLMAP-compatible scene.
2. Implement or script explicit topology-budget sweeps:
   - 90%, 75%, 50%, 25% final triangle budget;
   - compare PRISM-controlled pruning against simple post-hoc simplification/pruning.
3. Use independent `render.py + metrics.py` only.
4. Decide a hard gate:
   - continue only if PRISM gives a clearly better Pareto frontier;
   - pivot if it does not.

This is the fastest way to know whether PRISM can become a real paper or should be demoted to infrastructure.

## Final Reflection

The disappointing part is not that the method failed completely. It did not. The infrastructure works. The audits are careful. The failure modes are understood. The work is technically nontrivial.

The disappointing part is that the technical effort did not convert into a meaningful scientific result. The method became more complex faster than it became more effective. That is exactly the pattern that top-tier reviewers punish.

The honest conclusion is:

> PRISM is a useful engineering scaffold and a possible starting point for topology-aware mesh-splatting research, but the current result is not a NeurIPS-level contribution. To become one, it needs a much stronger topology-compression or constrained-optimization result, not more packaging of marginal metric gains.


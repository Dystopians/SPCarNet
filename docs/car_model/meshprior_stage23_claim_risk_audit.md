# MeshPrior Stage 23 Claim-Risk Audit

Date: 2026-05-02

## Gate

Gate: `PASS`.

Decision: the strongest defensible current story is not "full MeshPrior scene optimization improves real scenes." The defensible story is a conservative evidence package:

> Object-prior scene mesh proposals can be made auditable and safe through scene gates, rollback, and topology-aware evaluation; current single-scene long-budget evidence supports a topology-controlled current-branch diagnostic, while integrated MeshPrior optimization remains under-evidenced.

## Claim Classification

| claim | status | evidence | decision |
|---|---|---|---|
| Stage 3 object posterior is a strong object prior versus older point-space baselines | supported | M13/M22 object row: recon Chamfer L1 `0.066391`, hidden Chamfer L1 `0.099075`, extraction success `1.0` | Can be claimed as object-prior evidence. |
| Proposal gates and rollback prevent obvious unsafe edits | supported | Parking patch gate rejects `8` no-op and `8` floater proposals, accepts `8` cleanup proposals on copies, source model unedited | Can be claimed as safety/plumbing evidence. |
| Current branch improves long-budget single-scene render/geometry metrics versus clean MeshSplatting | supported with caveat | M21 current branch beats clean at 7000 on PSNR/SSIM/LPIPS and depth AbsRel, but uses `833775` vs `285187` triangles | Claim only with topology caveat. |
| M21.5 `prune_50` resolves the immediate topology-inflation objection for this scene | plausible but under-evidenced | `prune_50` keeps PSNR/SSIM/LPIPS above clean and uses `416888` triangles | Use as a diagnostic row, not final algorithm. |
| Stage17 MeshPrior resume is a viable long-budget method | refuted | M21 Stage17 resume collapses at 7000: PSNR `10.839708`, depth AbsRel `0.744099` | Do not continue this as default. |
| Full MeshPrior scene optimization improves real scenes | unsafe to claim | No second scene, no integrated optimization-time topology control, no render-gated full insertion | Must remain a future-work / missing-row item. |
| Multi-scene generalization | unsafe to claim | M20 found no second suitable parking/COLMAP scene locally | Requires new data. |

## Strongest Paper Story Now

Recommended story: **conservative proposal/gate framework with topology-aware single-scene diagnostics and honest negative results**.

Do not frame the current state as a completed full scene-optimization method. A stronger but still accurate framing is:

- object prior: SP-CarNet posterior supplies object-centric shape evidence;
- safety contract: object evidence cannot accept edits without scene evidence;
- engineering contribution: proposal, gate, rollback, checkpoint-copy validation, W&B/logging, and evidence collectors;
- diagnostic result: topology-controlled current-branch row beats clean MeshSplatting render metrics on one scene;
- negative result: the first real MeshPrior resume variant fails at long budget.

## Required Next Experiments

Only two experiment paths are worth doing next:

1. **Data path**: add a second real vehicle/parking COLMAP scene, then rerun clean/current/`prune_50`-style evidence with W&B.
2. **Algorithm path**: move M21.5 topology control from post-hoc checkpoint-copy pruning into the optimization loop with scheduled cleanup, render/geometry gates, rollback, and W&B.

Longer Stage17 resume sweeps should not be launched without a new hypothesis. More short smoke runs will not change the paper decision.

## Final Recommendation

Recommendation: `CLAIM_CONSERVATIVE_FRAMEWORK_NOT_FULL_METHOD`.

Proceed to M24 hardening only if the immediate goal is repository stability and reproducibility. For research progress, prioritize either a new second scene or integrated optimization-time topology control.

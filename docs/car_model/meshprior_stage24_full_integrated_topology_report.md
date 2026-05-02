# Stage24 Full Integrated Topology-Control Report

Date: 2026-05-02

## Status

Gate: `PASS`

Stage24 reaches the first full-budget integrated topology-control milestone on `parking_phone_tiny`: online W&B, full 7000-iteration training, training-time PRISM commit metadata, final-cleanup accounting, independent render metrics, COLMAP proxy geometry, and topology-visible baseline comparison.

The best current method row is `full_v3_late_fine_prune_7000iter`.

## Runs

| run | W&B | status | purpose |
|---|---|---|---|
| `full_v1_7000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/7i6n8jfj` | PASS mechanism, poor quality | Early/repeated PRISM schedule. It over-froze normal densification and compressed to 57k triangles, hurting render/geometry. |
| `full_v2_late_quality_guard_7000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ytex9896` | SOFT PASS | Late 5% prune attempts. Counterfactual gate rejected all four rounds and preserved quality. |
| `full_v3_late_fine_prune_7000iter` | `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/e92jwttk` | PASS | Late 1% prune attempts. Two rounds committed, two were rejected, no rollback crash. |

## Key Diagnosis

M24-v1 exposed a real scheduling bug-risk: using many early PRISM candidate rounds with `recovery_iters=250` and `post_commit_recollect_iters=120` keeps the controller in freeze/recovery/stat phases for most of training. Since standard Mesh Splatting densification only runs when topology mutation is allowed, v1 effectively suppressed normal densification and produced an over-compressed low-quality model.

M24-v2 fixed that by delaying PRISM until after normal densification. A 5% candidate prune was still too aggressive for the counterfactual gate, so all four rounds were rejected. This is useful safety evidence, not a topology-control win.

M24-v3 uses late 1% candidate pruning. This is the current milestone configuration.

## M24-v3 Configuration

- output root: `outputs/carnet/meshprior/parking_phone_tiny/stage24_full_integrated_topology/full_v3_late_fine_prune_7000iter/`
- iterations: `7000`
- GPU: `1`
- W&B group: `parking_stage24_full_integrated_topology`
- W&B name: `full_v3_late_fine_prune_7000iter`
- geometry acquisition until iteration `6000`
- stats collection `150`
- dead rounds `0`
- candidate rounds `4`
- candidate prune ratio per round `0.01`
- recovery after commit `100`
- post-commit recollect `20`
- counterfactual gate enabled
- final cleanup disabled
- sparse COLMAP depth loss enabled

## PRISM Decisions

| iteration | pre triangles | post triangles | committed | rollback | counterfactual accept |
|---:|---:|---:|---|---:|---:|
| 6151 | 612458 | 606334 | true | 0 | 1 |
| 6272 | 606334 | 600271 | true | 0 | 1 |
| 6393 | 606334 | 606334 | false | 0 | 0 |
| 6394 | 606334 | 606334 | false | 0 | 0 |

Collector output:

- gate: `PASS`
- rounds: `4`
- committed rounds: `2`

## Full 7000-Iteration Comparison

Independent `render.py + metrics.py` and `evaluate_geometry_colmap.py`:

| row | PSNR | SSIM | LPIPS | depth AbsRel | normal deg | triangles | vertices |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean origin/main | 16.134155 | 0.452130 | 0.499124 | 0.084499 | 45.300650 | 285187 | 517863 |
| current branch | 17.204679 | 0.535045 | 0.450750 | 0.076126 | 45.561976 | 833775 | 1071408 |
| Stage17 MeshPrior resume | 10.839708 | 0.285366 | 0.662528 | 0.744099 | 52.580674 | 838883 | 1087793 |
| M21.5 posthoc prune_50 | 17.051889 | 0.523914 | 0.465400 | 0.083265 | 45.825681 | 416888 | 662039 |
| M21.5 posthoc prune_66 | 16.429369 | 0.492480 | 0.489681 | 0.099246 | 46.555197 | 283484 | 490594 |
| M24-v1 early PRISM | 13.030146 | 0.403779 | 0.576543 | 0.285859 | 49.100193 | 57156 | 193491 |
| M24-v2 late 5% rejected | 17.150599 | 0.531539 | 0.452803 | 0.079628 | 44.052758 | 829324 | 1067737 |
| M24-v3 late 1% committed | 17.042757 | 0.529476 | 0.454884 | 0.082815 | 43.394721 | 823651 | 1058219 |

Training internal test@7000:

| row | PSNR | SSIM | LPIPS | note |
|---|---:|---:|---:|---|
| M24-v1 | 17.648366 | 0.565388 | 0.465162 | Internal metrics overstate v1 relative to independent render. |
| M24-v2 | 18.059103 | 0.578560 | 0.424071 | No PRISM commit. |
| M24-v3 | 17.904139 | 0.575078 | 0.427551 | Two PRISM commits. |

## Interpretation

M24-v3 is a real integrated optimization-time topology-control row, but it is not yet a decisive paper headline.

Positive evidence:

- It commits PRISM topology edits inside training with counterfactual acceptance and no rollback crash.
- It preserves render quality close to the current-branch 7000 row.
- It improves the COLMAP proxy normal angle over current branch: `43.39` degrees vs `45.56`.
- It slightly reduces topology versus current branch: `823651` vs `833775` triangles.
- It demonstrates safety behavior: v2 rejected overly aggressive 5% edits; v3 accepted smaller 1% edits.

Limitations:

- Render metrics are slightly below the current branch: PSNR `17.043` vs `17.205`, SSIM `0.5295` vs `0.5350`, LPIPS `0.4549` vs `0.4508`.
- Topology reduction is small compared with M21.5 posthoc `prune_50`, which reaches `416888` triangles with similar render metrics.
- The current PRISM counterfactual gate is conservative and needs a tuned quality/topology Pareto schedule rather than larger blind candidate rounds.
- This is still a single-scene result.

## Goal Progress Estimate

For the stated goal, "a MeshSplatting-based model that optimizes mesh scenes with camera/COLMAP evidence and can support a NeurIPS-level paper", the current codebase is about `90%` complete as a research prototype and about `65%` complete as a NeurIPS-strength empirical paper.

The implementation now has:

- clean/current baselines,
- topology-aware diagnostics,
- posthoc topology Pareto rows,
- integrated training-time PRISM commits,
- rollback/final-cleanup accounting,
- W&B records,
- separated render and geometry proxy metrics.

The missing paper-critical pieces are:

1. A stronger integrated Pareto row that approaches M21.5 `prune_50` topology while preserving current-branch quality.
2. A second real scene, preferably larger than `parking_phone_tiny`.
3. A full ablation table showing why gates matter: no gate, render gate off, geometry gate off, aggressive prune rejected, accepted small prune.
4. Visual failure/success cases tied to PRISM decisions.

## Next Step

Do not return to Stage17. The best next technical move is `M24.1`: late PRISM Pareto sweep with small candidate ratios and tuned gate tolerance, probably around:

- geometry acquisition until `6000`;
- candidate ratios `{0.005, 0.01, 0.02}`;
- candidate rounds `{4, 8}`;
- explicit validation of per-round accepted/rejected edits;
- optional lower-cost resume/checkpoint support so full 7000-iteration runs are not repeated from scratch.

The highest-value data move remains adding a second camera/COLMAP scene so the method can be evaluated beyond one tiny parking sequence.


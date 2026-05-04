# Final Stage F0 Current-State Audit

Date: 2026-05-04

## Decision

`PROCEED_TO_F1`.

The repository compiles under the project environment, and the final-paper claim is reset. MeshSplatOpt should be framed as **evidence-certified compact-repair optimization for Mesh Splatting**, not as a local snap/fill repair method unless later stages produce new real-scene edit-driven evidence. The strongest validated rows are R53/R48/R55, which are substantially stronger than R44 under the corrected long-horizon clean-baseline comparison. Snap/fill remain important rollback-compatible edit interfaces and diagnostics, but they are not the headline method.

## Repository State

- Current branch: `neurips-meshsplatopt-repair`
- Current commit: `97b9d6d`
- Worktree root: `/data/peilincai/mesh-splatting`
- System Python: `Python 3.13.2`
- Project Python: `Python 3.11.14`
- Compile command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior utils -q`
- Compile status: pass

Dirty/untracked files at audit time:

```text
 ? submodules/effrdel
 ? submodules/simple-knn
?? assets/meshsplatopt_baseline_progression_montage.png
?? docs/FinalNeurIPSPrompts_MeshSplatOpt.md
```

The prompt file is user-provided planning context and is not staged by this audit. The submodule marks and montage asset are existing untracked worktree state and are not part of the F0 commit.

## Strongest Validated Result

The strongest validated result is the corrected parking clean-to-compact path, not R44.

R53.01 starts from the clean current-branch 22k checkpoint, removes the smallest-area 70 percent of triangles, freezes topology, and recovers from 22k to 26k with W&B run `q15qg2b8`. Against the strongest clean long baselines, it is an all-metric independent win:

| comparison | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal deg | triangles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean 22k | 18.479990 | 0.634623 | 0.346913 | 0.082177 | 1.868398 | 45.108437 | 8,548,242 |
| clean 30k | 18.408827 | 0.631504 | 0.350967 | 0.081639 | 1.865811 | 44.838918 | 8,548,242 |
| R53.01 prune70 recovery | 18.705738 | 0.647807 | 0.338492 | 0.079555 | 1.853751 | 44.261391 | 2,564,473 |

R53.01 beats clean 22k by PSNR `+0.225748`, SSIM `+0.013184`, LPIPS `-0.008421`, AbsRel `-0.002622`, Depth MAE `-0.014647`, and normal angle `-0.847046` while removing 69.999999 percent of clean-long triangles.

R48.01 remains the more compact 80-percent-pruned Pareto row: it keeps only 20 percent of clean-long triangles and beats clean 22k on PSNR, SSIM, AbsRel, and Depth MAE, but not LPIPS. R55.01 is the LPIPS/normal Pareto row: it improves LPIPS and normal versus R53.01, but gives back PSNR and uses more triangles.

Cross-scene matched evidence now includes one public-scene all-metric positive:

| scene | row | matched decision | key delta compact-clean |
| --- | --- | --- | --- |
| `bonsai` | R58 prune70 7k->9k vs clean 7k->9k | pass | PSNR `+0.280336`, SSIM `+0.017475`, LPIPS `-0.007539`, AbsRel `-0.006582`, Depth MAE `-0.062115`, normal `-0.515667`, triangles `-70%` |
| `room` | R59 prune70 7k->9k vs clean 7k->9k | render-positive, geometry-negative | PSNR `+0.438885`, SSIM `+0.005325`, LPIPS `-0.000389`, AbsRel `+0.002058`, Depth MAE `+0.006847`, normal `+0.610254`, triangles `-70%` |
| `courtyard` | R57 prune70 7k->9k vs clean 7k->9k | fail | PSNR `-0.001726`, SSIM `-0.000522`, LPIPS `+0.027805`, AbsRel `+0.035424`, Depth MAE `+0.209014`, normal `-1.032962`, triangles `-70%` |
| `counter` | R60 prune70 7k->9k vs clean 7k->9k | fail/mixed | PSNR `+0.134289`, SSIM `-0.009183`, LPIPS `+0.021316`, AbsRel `+0.009285`, Depth MAE `+0.027983`, normal `+1.565465`, triangles `-70%` |

This supports a compact-repair Pareto story and scene-dependent feasibility, but it does not yet support universal cross-scene clean-to-compact dominance.

## Strongest Negative Results

- R44.01 is not a render-quality win over the best clean long baseline. Clean 22k beats R44.01 on PSNR (`18.479990` vs `17.169540`), SSIM (`0.634623` vs `0.548714`), LPIPS (`0.346913` vs `0.441888`), AbsRel (`0.082177` vs `0.187067`), and Depth MAE (`1.868398` vs `2.919396`). R44.01 only wins on normal proxy and topology. R44 must be described as a low-topology/normal Pareto point, not as the final method.
- R45/R46 teacher distillation from R44 are rejected. Full-image teacher and counterfactual teacher-mask recovery do not recover clean-long quality from the 0.78M-triangle R44 checkpoint.
- R51/R52 direct LPIPS training loss from R48 are rejected. LPIPS optimization worsens PSNR/SSIM and does not solve the clean-long comparison.
- R56 true fixed-topology continuation from R53 26k to 28k is rejected. It sharply worsens render metrics, so the accepted budget is 26k.
- R17-R21 snap/checkpoint-edit gates show rollback-safe real checkpoint editing, but not an equal-budget performance gain.
- R22/R26/R28 fill/grid-fill variants do not beat the matched sparse-depth control. The sparse COLMAP recovery mechanism is useful; current fill edits are not the headline.
- R25 unfrozen continuation exposes topology-growth risk. Strict topology control requires `--freeze_topology_updates --skip_restricted_delaunay`.
- R57 courtyard and R60 counter show that area-only prune70 is not universal. A selector must predict when compaction is safe before claiming a general method.

## Partially Achieved Goals

- Topology reduction: achieved on parking R48/R53/R55 and public-scene matched rows. R53 removes 70 percent of clean-long parking triangles while improving all tracked independent metrics.
- Render improvement: achieved strongly on parking R53/R55 and bonsai R58; partially on room R59; not achieved on courtyard R57 or counter R60.
- Sparse geometry improvement: achieved on parking R53/R55 and bonsai R58; room and counter matched screens show geometry regressions under area-only pruning.
- Real checkpoint edit application: implemented for delete, snap, and fill-style proposals with checkpoint materialization and rollback-compatible audit records.
- Rollback/counterfactual gate: implemented and used to keep unsafe edits out of headline claims.
- Giant-hole repair: synthetic and diagnostic scaffolds exist, but real-scene giant-hole repair is not validated as a paper claim.

## Not Achieved Goals

- Cross-scene clean-to-compact dominance is not yet universal. Current evidence is parking positive, bonsai positive, room render-positive/geometry-negative, courtyard negative, and counter mixed/negative.
- Real giant-hole repair is not validated. The project must not claim real parking-lot hole completion as a final method result.
- Snap/fill edit-driven full-budget gain is not validated. These remain interfaces and diagnostics.
- Universal trusted sparse sampling is not established. Trusted/random sampling is a tunable confidence knob with scene-dependent tradeoffs.
- NeurIPS-ready multi-scene evidence is incomplete. The current package is much stronger than the R44 state, but it still needs a selector, fair baseline registry, and multi-scene table discipline.

## Final Claim Reset

The final paper story should be:

> MeshSplatOpt performs evidence-certified compact repair: it starts from a strong mesh-splatting checkpoint, proposes topology reduction or repair under an evidence contract, freezes accepted topology, recovers appearance, and certifies the result by independent rendering and sparse-geometry evaluation against the strongest matched clean baseline.

The story should not be:

> Local snap/fill edits repair real scenes and beat clean baselines.

The latter is not supported by the current evidence. Snap/fill should be presented as reversible edit interfaces and future-facing repair mechanisms unless later stages prove real edit-driven gains.

## F0 Gate

`PASS`.

F0 satisfies the hard gate because the repository compiles and this audit explicitly states that:

- R53/R48/R55 are stronger than R44 under the corrected clean-long comparison.
- Snap/fill are not the headline method.
- The next stage should proceed to a claim-safe method specification.

Decision: `PROCEED_TO_F1`.

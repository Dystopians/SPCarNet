# Final Stage F15 - NeurIPS Go/No-Go

Decision: `NEURIPS_BORDERLINE_NEEDS_STRICT_ABLATIONS`.

## Evidence Strength

- Scenes beating fair clean-long baselines under the accepted compact-recovery rule: `5/5`.
- Significant topology reduction: `40%` to `70%` fewer triangles.
- Independent metrics: yes for the main rows in F8/F9/F10 and parking compact-recovery evidence.
- Fair baselines: improved substantially after the clean 7k versus method 22k issue was fixed; current public-scene comparisons are clean-long 22k versus compact-recovery 26k.

## Novelty Strength

- Strongest novelty: evidence-compatible compact-recovery with strict topology freeze and auditable rollback.
- CSEF is used in public-scene selector naming/logic, but area-only versus CSEF separation is still incomplete across all final rows.
- Sparse-depth recovery is an earlier useful branch, but the final compact-recovery main rows should be described as topology-frozen appearance recovery with independent COLMAP sparse-geometry evaluation unless new sparse-depth-loss rows are launched.
- Repair operations are auxiliary. Snap/fill are not currently load-bearing headline improvements.

## Reviewer Risks

| risk | current answer | mitigation |
| --- | --- | --- |
| This is just area pruning plus finetuning. | Counter area40 beats CSEF40; courtyard CSEF50 and area50 are near-tied, while random controls fail badly on both scenes. | Run one more public-scene selector triplet and include the mixed selector result honestly. |
| Sparse depth does all the work. | The final main rows did not enable sparse-depth loss; they use COLMAP sparse geometry for independent evaluation. | Keep sparse-depth as a separate branch, or launch explicit sparse-depth-loss compact-recovery variants before claiming it as part of the final method. |
| Only one scene works. | No longer true: F12 has five scene-matched long-baseline rows. | Keep F12 main table prominent. |
| Triangle soup means mesh repair is not real mesh repair. | Correct; edge-loop repair was rejected. | Frame as compact-repair optimization, not classical watertight mesh repair. |
| Hole fill is synthetic or weak. | Correct for current real-scene evidence. | Do not headline fill; keep it in limitations/negative results. |
| Baselines are weak. | Clean-long baseline is now fair, but posthoc decimation/QEM remains missing. | Add posthoc simplification baseline with equal recovery budget. |

## Go/No-Go

This is no longer a weak single-scene result. The current repository has a credible compact-recovery paper core: five scene-matched long-baseline comparisons, large topology reductions, and several honest negative results. However, it is not yet a clean `NEURIPS_MAIN_READY` package because the strict ablation matrix is incomplete.

Recommended label: `NEURIPS_BORDERLINE_NEEDS_STRICT_ABLATIONS`.

Minimum remaining work before submission:

- final area-only versus CSEF selector ablation;
- selector ablation beyond completed counter and courtyard controls;
- replicated no-freeze recovery controls beyond counter;
- explicit sparse-depth-loss compact-recovery variants if sparse-depth is claimed as final-method training;
- posthoc QEM/decimation baseline;
- final paper figures from `outputs/carnet/meshsplatopt/final_paper_assets/`.

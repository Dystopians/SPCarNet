# GEMS Submission Handoff — Figure Notes

Generated 2026-07-04 for Stage3 closure.

## Main Figure Recommendation

Use one figure that makes the paper's real contribution obvious:

1. Clean MeshSplatting checkpoint enters a train-evidence analysis block.
2. Evidence prune selects a smaller triangle set.
3. Features-only recovery repairs appearance without moving geometry.
4. Evaluation splits into rendering/efficiency, geometry diagnostics, and
   downstream consumers.
5. Downstream branch ends in a red "impossibility x4" box, not a success box.

The figure should visually distinguish method outputs from analysis-only
artifacts. Do not imply the occupancy consumers are part of test-time rendering.

## Figure List

| figure | source | purpose | notes |
|---|---|---|---|
| F1 method pipeline | new art + `F3_pipeline_diagram` as reference | show compactness method and analysis branches | Promote R3-FINAL as closed negative, not pending. |
| F2 Pareto curves | `RESULTS/figures/F2_*` | main compactness-quality result | Use B50 as stable regime; show B25/B12.5 honestly. |
| F3 qualitative grids | `RESULTS/figures/qual/*_qual_grid.png` | visual appearance preservation | Choose scenes where B50 is visually close to clean; do not oversell tiny differences. |
| F4 evidence-vs-error curve | `analysis/e2geo_evidence_vs_error/plots/` | core analysis novelty | Delta-Memo-002 recommends promoting this to main text. |
| F5 downstream closure | `RESULTS/figures/F6_*`, `T5b_r3_trilogy.md` | explain why planning fails | Include four route-family icons or panels: raw, TSDF, certified, three-state. |
| F6 failure taxonomy | `RESULTS/figures/F8_failure_board.*` | honest limitations | Keep as appendix or main limitation figure depending on venue. |
| F7 robustness | `T7_robustness.md` | seed/view-drop closure | A small table is enough; no decorative figure needed. |

## Art Direction

- Make the method branch green/blue and the analysis/failure branch gray/red.
- Use consistent labels: `B0 clean`, `B5@B50 GEMS-core`, `B6R diagnostic`,
  `R3-FINAL`.
- For occupancy maps, show FREE, OCCUPIED, UNKNOWN with three fixed colors.
- Avoid "safe planning" language. Use "consumer failed at frozen bar" instead.
- All captions must include scope and anchor: clean@30k or clean-fixed@30k.

## R3-FINAL Figure Spec

If time allows, render a compact 2x4 panel:

- columns: raw voxel, TSDF, certified sub-mesh, three-state V1;
- rows: toy_parking, courtyard;
- annotate found/100 and coll/100 directly on each cell;
- three-state V1 annotation: "false-free controlled, free space blocked".

Evidence path for numbers:
`RESULTS/CONSUMPTION_IMPOSSIBILITY.md` and `T5b_r3_trilogy.md`.

---

## STAGE-4 REFRESH (2026-07-10) — ECR figure set

Revised main-figure recommendation: the pipeline figure now ends in the ECR
render loop, not the analysis branches — {mesh checkpoint + evidence cache}
→ base render → K-source depth-consistent warp → learned α/β routing →
final frame, with the audit boundary (pose-primitives only, GT sentinel)
drawn as a hard wall. The Stage-3 branches (geometry suite, impossibility)
become a compact side box labeled "scope".

| figure | source | purpose | notes |
|---|---|---|---|
| F-E1 ECR pipeline + audit wall | new art | the method core + the no-test-GT boundary | show the cache as a shipped artifact; β·valid gate explicit |
| F-E2 qual grids (best/median/failure) | `RESULTS/figures/ecr_qual/<scene>_ecr_qual_grid.png` (5 scenes) | GT/base/PJ-2026/final + β + confidence rows under the frozen crop rule | bicycle failure column IS the coverage-gap story — keep it |
| F-E3 β/confidence close-ups | `analysis/quals/<scene>_final/<view>/{beta,conf}.png` | routing concentrates on high-frequency supported regions; vanishes at occlusion seams | garden DSC07956 β map is the poster child |
| F-E4 ladder bar chart | `analysis/e0_pj2026/l{1,2,3,4}_gate.json` | per-rung CI'd deltas incl. the L1 negative | plot CI bars, mark the promotion floor |
| F-E5 storage Pareto | `analysis/final_stack/l5_pareto.{md,json}` (E-06, in flight) | quality vs TOTAL MB incl. matched-3DGS point | overlay e07 3DGS points + Difix point for the full trade picture |
| F-E6 failure cases | `analysis/final_stack/ecr_failure_cases.md` + quals planes | E9-style: 1 negative view in 139; graceful degradation | pair each case with its conf/β plane |

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

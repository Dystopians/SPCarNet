# GEMS — MATRIX.md (Stage Two experiment matrix)

Status ∈ {TODO, RUNNING, DONE-PASS, DONE-FAIL, INFEASIBLE(+reason)}. Tier 1 = required. Budgets: {50, 25, 12.5}% (100%-realloc not implemented → dropped with note: reallocation stretch never built in Stage One).
Methods: B0 clean · B1 no-op · B2 random+FT(safe) · B3 QEM+FT · B4 evidence prune no-FT · B5 GEMS-core (evidence prune + features-only FT) · B6 = B5+geometry (DEMOTED — runs as DIAGNOSTIC row only) · B7 = B5+teacher (DEMOTED — DIAGNOSTIC row only) · H1 v106 historical · R1 3DGS+compression reference.

## Datasets
| id | content | status |
|---|---|---|
| S-REND | mipnerf360 full9 (holdout unlocked) | READY (data+B0 ckpts on disk) |
| S-GEO/S-DOWN | SS3DM ≥4 seqs + toy_parking + 1–2 variants | SS3DM ACQUIRED (D-1 DONE-PASS); ingestion converter D-1b TODO; toy READY |
| S-GEN | 1–2 unseen-type (T&T barn/truck on disk) | READY (data); needs B0 training |

## Cells
| cell | tier | scope | status | notes |
|---|---|---|---|---|
| D-1 SS3DM acquisition | T1-blocking | 4 seqs (Town01/02/03/06, 150f×6cam×5lidar) + GT meshes | **DONE-PASS** | ~15 GB via Zip64 central-directory range extraction (of 137 GB); CRC-verified; `mesh_datasets/SS3DM/ACQUISITION_LOG.md` |
| D-1b SS3DM ingestion | T1 | scenario.pt(pickle)→COLMAP-format per sequence + scene registry + split policy; verify with readColmapSceneInfo + 1 render | RUNNING (builder agent) | anchor-independent (R0.2 allows) |
| D-2 toy variants ×2 | T1 | builder re-runs, altered layout/occlusion | TODO | tools/gems/build_toy_parking.py parameterization |
| E1-PARETO S-REND | T1 | {50,25,12.5} × {B2,B3,B4,B5} × 9 scenes (+B6,B7 diagnostic on 2 scenes) | B50 slice DONE-PASS (B5 within −0.10: 8/9 scenes incl. garden, ABOVE clean 5/9, LPIPS better 9/9; flowers −0.147 → E9-FAIL; B4 within −0.20: 8/9). B25 slice DONE (floor 3/9 at 75% reduction; FT helps 9/9); B12.5 TODO | garden rows exist from Stage One |
| E1-PARETO S-GEO | T1 | same on SS3DM+toy | TODO (blocked D-1 for SS3DM; toy rows partially exist) | |
| E2-GEO tables | T1 | g1–g4 all scenes + evidence-vs-error analysis | TODO (metrics auto-computed in every eval; analysis script needed) | |
| E3-REND tables | T1 | per-scene S-REND tables + teacher-headroom analysis | TODO (headroom analysis partially done in Stage One) | |
| E4-EFF | T1 | tris/disk/VRAM/FPS@2res + pipeline overhead | TODO (1-res numbers exist everywhere; add 2nd res + overhead table) | |
| E5-DOWN | T1 | occupancy confusion, ESDF/costmap error, ≥500 maneuvers | TODO (d1/d2@200 exist; extend to 500 + ESDF; SS3DM pending) | |
| E6-ABL | T1 | importance families; schedule; geometry losses; teacher variants — @B50, 2 S-REND + 1 S-GEO scene | TODO (several cells already exist from Stage One variants — map them in) | |
| E7-SENS | T2 | 3 seeds; dev-vs-full res; 3-point loss weights | TODO | |
| E8-ROBUST | T2 | 50% view drop; pose noise ×2; S-GEN | TODO | |
| E9-FAIL | T1 | ≥10 curated failure panels + diagnoses | TODO (6+ candidates already identified in STAGE1_REPORT §5) | |
| E10-STATS | T1 | CIs everywhere + multiple-comparisons caveat | ONGOING (CI discipline already universal) | |
| E11-QUAL | T1 | qualitative grids + before/after maps; (T2) 2 videos | TODO | |
| H1 row | T1 | v106 historical context row | TODO (locate v106 artifacts) | |
| R1 row | T1 | 3DGS + public compression at matched storage | TODO (needs 3DGS training env check) | |
| R2 row | T3 | 2DGS-style on S-GEO | TODO | |

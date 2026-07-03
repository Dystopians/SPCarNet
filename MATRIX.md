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
| D-1b SS3DM ingestion | T1 | 4 towns → COLMAP + registry + split.json (393/57 whole-frame holdout) | **DONE** | left-handed CARLA frame mirrored (det=−1, projection-invariant); depth-PNG scale 65535/1000 decoded; 3 front cams × 150 fr @-r2 (VRAM-frozen policy); GT OBJs are cm (×0.01); Town06 g4 RAM risk flagged; B0 trainings pending |
| D-2 toy variants ×2 | T1 | builder re-runs, altered layout/occlusion | TODO | tools/gems/build_toy_parking.py parameterization |
| E1-PARETO S-REND | T1 | {50,25,12.5} × {B2,B3,B4,B5} × 9 scenes (+B6,B7 diagnostic on 2 scenes) | B50 slice DONE-PASS (B5 within −0.10: 8/9 scenes incl. garden, ABOVE clean 5/9, LPIPS better 9/9; flowers −0.147 → E9-FAIL; B4 within −0.20: 8/9). B50+B25+B12.5 slices DONE; B12.5 dominance: importance beats random +5.23 dB mean, 9/9 CI (GOAL #011); remaining: B2/B3 columns at B50/B25, B6.25 T2 | garden rows exist from Stage One |
| E1-PARETO S-GEO | T1 | same on SS3DM+toy | TODO (blocked D-1 for SS3DM; toy rows partially exist) | |
| E2-GEO tables | T1 | g1–g4 all scenes + evidence-vs-error analysis | TODO (metrics auto-computed in every eval; analysis script needed) | |
| E3-REND tables | T1 | per-scene S-REND tables + teacher-headroom analysis | TODO (headroom analysis partially done in Stage One) | |
| E4-EFF | T1 | tris/disk/VRAM/FPS@2res + pipeline overhead | TODO (1-res numbers exist everywhere; add 2nd res + overhead table) | |
| E5-DOWN | T1 | occupancy confusion, ESDF/costmap error, ≥500 maneuvers | TODO (d1/d2@200 exist; extend to 500 + ESDF; SS3DM pending) | |
| E6-ABL | T1 | importance families; schedule; geometry losses; teacher variants — @B50, 2 S-REND + 1 S-GEO scene | TODO (several cells already exist from Stage One variants — map them in) | |
| E7-SENS | T2 | 3 seeds; dev-vs-full res; 3-point loss weights | TODO | |
| E8-ROBUST | T2 | 50% view drop; pose noise ×2; S-GEN | TODO | |
| E9-FAIL | T1 | ≥10 curated failure panels + diagnoses | **DONE** (13 cases, 5 mechanism families; `gems_stage1/analysis/e9_failure_taxonomy/TAXONOMY.md`; LEDGER #R-09) | 0 candidates rejected; 2 kept with qualifications |
| E10-STATS | T1 | CIs everywhere + multiple-comparisons caveat | ONGOING (CI discipline already universal) | |
| E11-QUAL | T1 | qualitative grids + before/after maps; (T2) 2 videos | TODO | |
| H1 row | T1 | v106 historical context row | TODO (locate v106 artifacts) | |
| R1 row | T1 | 3DGS + public compression at matched storage | TODO (needs 3DGS training env check) | |
| R2 row | T3 | 2DGS-style on S-GEO | TODO | |
| R3.a occupancy routes | T1 | voxelization vs TSDF, toy-calibrated frozen | **DONE-FAIL (hypothesis falsified; citable)** | TSDF halves false-occ but worsens false-free 26–41% — voxelization is the safe consumer; B50 surf_depth bit-identical to clean |
| R3.c planner loop v0 | T1 | Hybrid-A*-lite, 100 paired problems × 12 cells | **DONE** | B50 preservation exact (CI[0,0]); raw grids unconsumable (route-i 93–100% spurious infeasible; route-ii 10.7 coll/100 courtyard) → elevates R3.b |
| R3.b certified sub-mesh | T2→elevated | one-time global labeling → collision-grade sub-mesh → route-i | TODO (next after GPUs free) | R3.c quantifies the need |
| R3.b certified sub-mesh | T2-elevated | k×m calibration + frozen application, 5 cells | **DONE-FAIL (falsified 0/4; citable)** | sheds load-bearing surface (courtyard 16.7 coll/100); kept-sets EXACTLY identical clean↔B50; consumption trilogy closed: blocker is checkpoint geometry, target ≥30/100 @ ≤3 coll/100 |

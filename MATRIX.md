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
| E1-PARETO S-GEO | T1 | same on SS3DM+toy | B50/B25 × {B5,B4} × 4 towns DONE (GOAL #R-05/#R-08). **B6R-on-SS3DM (T2 generalization of the E2R courtyard positive) DONE-FAIL as pre-registered** (GOAL #014: g3 FRACTION arm 0/3 towns at −14/−23/−19% vs the ≥30% bar; PSNR guard held 3/3, CI-lo > −0.10 everywhere; g3 COMPONENTS −31.6/−31.6/−33.5%; LPIPS + g1 better CI-excl-0 3/3; d1-ff worse +0.6..+1.1% CI-excl-0 3/3 = the R-02/R-08 coverage bound; rows `eval/b6r_ss3dm_town0{1,2,3}_B50_v1`, verdicts `analysis/b6r_ss3dm/`) — B6R stays a bounded courtyard-scoped positive, NOT claim-grade | town06 excluded from B6R (B50 floor fail #R-05 + g4/d1/d2 infeasible #R-08) |
| E2-GEO tables | T1 | g1–g4 all scenes + evidence-vs-error analysis | TODO (metrics auto-computed in every eval; analysis script needed) | |
| E3-REND tables | T1 | per-scene tables + teacher-headroom (T2/T6/F7) | **DONE** (evidence pack v1, commit 3dce47c) | |
| E4-EFF | T1 | T4 + half-res FPS bench + pipeline-overhead column | **DONE** (contention caveat in T4 header; laptop bench waived) | |
| E5-DOWN | T1 | occupancy confusion, ESDF/costmap error, ≥500 maneuvers | TODO (d1/d2@200 exist; extend to 500 + ESDF; SS3DM pending) | |
| E6-ABL | T1 | importance families; schedule; geometry losses; teacher variants — @B50, 2 S-REND + 1 S-GEO scene | importance-families sub-cell **DONE** (GOAL #012: pixels_total vs max_blending_max vs ckpt_importance_score @B50 garden/kitchen/town01 — revision trigger NOT tripped, pixels_total stands; family axis nearly flat, all pairwise |dPSNR| ≤ 0.052 dB; ckpt-stat slightly worst 2/3 CI; `analysis/e6_abl/e6_table.md`); schedule/geometry-losses/teacher sub-cells map to Stage One variants (E1 v3 iterative, M3 tombstone, E3 tombstone) — mapping TODO | town01 max_blending keep-set degenerate-equal to pixels_total at B50 (pre-registered) → measured pipeline noise floor 1.6e-5 dB |
| E7-SENS | T2 | 3 seeds; dev-vs-full res; 3-point loss weights | TODO | |
| E8-ROBUST | T2 | 50% view drop; pose noise ×2; S-GEN | TODO | |
| E9-FAIL | T1 | ≥10 curated failure panels + diagnoses | **DONE** (13 cases, 5 mechanism families; `gems_stage1/analysis/e9_failure_taxonomy/TAXONOMY.md`; LEDGER #R-09) | 0 candidates rejected; 2 kept with qualifications |
| E10-STATS | T1 | CIs everywhere + multiple-comparisons caveat | **DONE** (universal CI discipline; caveat EXPERIMENT_REPORT §5; per-scene win/loss in T1) | |
| E11-QUAL | T1 | qualitative grids + before/after maps; (T2) 2 videos | **grids DONE** (GOAL #014: 7 scenes × rows {clean,B50 B5,B25 B5,+B6R where exists} × cols {RGB, median-depth, floater-overlay} under the SS5 crop rule — best/median/failure views script-chosen from banked per-view arrays, every invocation logged; `RESULTS/figures/qual/*_qual_grid.png` + `manifest.json`; tool `tools/gems/report/qual_grids.py`) — T2 flythrough videos still TODO | failure views from E9 taxonomy where a case names one (6/7 scenes); town01 argmin fallback |
| H1 row | T1 | v106 historical context row | RUNNING (GOAL #013: locating v106 artifacts) | |
| R1 row | T1 | 3DGS + public compression at matched storage | RUNNING (GOAL #013: feasibility check only) | |
| B3 QEM column @B50 | T1 (E1 slice) | garden+toy+courtyard, QEM decimation + safe-FT, tag b3 | RUNNING (GOAL #013, pre-registered: B3 < B5) | tools/gems_train/qem_prune.py |
| R2 row | T3 | 2DGS-style on S-GEO | TODO | |
| R3.a occupancy routes | T1 | voxelization vs TSDF, toy-calibrated frozen | **DONE-FAIL (hypothesis falsified; citable)** | TSDF halves false-occ but worsens false-free 26–41% — voxelization is the safe consumer; B50 surf_depth bit-identical to clean |
| R3.c planner loop v0 | T1 | Hybrid-A*-lite, 100 paired problems × 12 cells | **DONE** | B50 preservation exact (CI[0,0]); raw grids unconsumable (route-i 93–100% spurious infeasible; route-ii 10.7 coll/100 courtyard) → elevates R3.b |
| R3.b certified sub-mesh | T2-elevated | k×m calibration + frozen application, 5 cells | **DONE-FAIL (falsified 0/4; citable)** | sheds load-bearing surface (courtyard 16.7 coll/100); kept-sets EXACTLY identical clean↔B50; consumption trilogy closed: blocker is checkpoint geometry, target ≥30/100 @ ≤3 coll/100 |

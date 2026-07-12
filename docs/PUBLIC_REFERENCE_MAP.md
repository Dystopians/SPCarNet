# PUBLIC REFERENCE MAP — internal IDs → paper/artifact locations

Pre-submission mapping (2026-07-12). Internal governance IDs never appear in the paper; this table is
the translation layer for writing and for reviewers examining the artifact pack.

| Internal reference | Public form (paper) | Artifact path (pack) |
|---|---|---|
| CR1 (quality) | main quality table (T1) + §Experiments | `RESULTS/STAGE4_ECR/tables/final_stack_tables.md`, `tables/t2b_tandt_db.md` |
| CR2 (honest cost) | cost table (T4) + R-D figure | `tables/l5_pareto.md`, `tables/e07_matched_total_3dgs.md`, `RESULTS/figures/ecr_paper/rd_master.pdf` |
| CR3 (compact/L6) | compact table (T3) | `tables/final_stack_summary.json` (l6 block) |
| CR4 (audited transport) | §Method threat-model + audits appendix | PROTOCOL §4E/§4E.1 excerpt in supp; audit example report |
| CR5 (edit-consistent evidence) | editing section (§6.5) + edit table (T6) | `edit_aware/routeA_master_table.md`, `edit_aware/oracle_eval.md`, figure grids `RESULTS/figures/edit_aware/` |
| LEDGER #E-00..#E-16 | supp: "experiment log" appendix (pre-registrations quoted as method prose) | `LEDGER.md` STAGE-4/4+ sections (ship as supp text) |
| ROUTE_A_TOPCONF_{GAP_AUDIT,EXECUTION_PLAN,READINESS_REPORT} | supp: red-team appendix | `docs/ROUTE_A_TOPCONF_*.md` |
| EDIT_AWARE_ECR_{PROTOCOL,VALUE_REPORT,...} | supp: editing protocol appendix | `docs/EDIT_AWARE_ECR_*.md` |
| l1–l4 gates / vs_floor | ladder table (T2) + CI bar figure | `gates/*.json`, `RESULTS/figures/ecr_paper/ladder_ci.pdf` |
| E-08 conf-off ablation | ablation rows in T2 | `RESULTS/tables_tex/T2_ladder.tex` rows |
| E-09 Difix / E-12 IBRNet / E-07 3DGS | external-baselines table (T4b) | `difix/difix_table.md`, `ibr/ibr_table.md`, `tables/e07_*` |
| E-13 temporal / edited temporal | temporal table (T5) + supp videos | `tables/temporal_summary.md`, `analysis/temporal/*/mp4`, `analysis/edit_aware/garden_temporal/` |
| EXP-HBOOT hierarchical CIs | stats paragraph + T1 second aggregate row | `tables/hierarchical_cis.md` |
| DS-1 / CONSUMPTION_IMPOSSIBILITY | scope/limitations citation | `RESULTS/CONSUMPTION_IMPOSSIBILITY.md` |
| boundary (translation) figure | limitations figure | `edit_aware/boundary_translate.md` + `analysis/edit_aware/boundary_translate/translated_base.png` |

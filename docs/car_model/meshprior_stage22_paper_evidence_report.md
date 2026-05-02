# MeshPrior Stage 22 Paper Evidence Report

Date: 2026-05-02

Gate: `SOFT PASS`.

Decision: paper evidence is reproducible and separated, but multi-scene and integrated topology-control rows remain MISSING.

## Scene Evidence

| row_id | method | render_psnr | render_ssim | render_lpips | triangles | geometry_depth_absrel | claim_role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean_origin_main_7000 | clean_mesh_splatting | 16.1341552734375 | 0.45213010907173157 | 0.49912410974502563 | 285187 | 0.0844994634276995 | clean long-budget baseline |
| current_branch_7000 | current_branch_unpruned | 17.204679489135742 | 0.5350445508956909 | 0.4507499039173126 | 833775 | 0.07612569271786734 | quality-positive but topology-inflated diagnostic |
| current_branch_prune_50_7000 | current_branch_area_prune_50 | 17.051889419555664 | 0.5239143371582031 | 0.46540042757987976 | 416888 | 0.08326522687327094 | default topology-controlled row for M22 |
| current_branch_prune_66_7000 | current_branch_area_prune_66 | 16.42936897277832 | 0.49247995018959045 | 0.48968085646629333 | 283484 | 0.09924590675193089 | high-compression Pareto endpoint |
| stage17_meshprior_resume_7000 | stage17_meshprior_resume | 10.83970832824707 | 0.28536638617515564 | 0.6625283360481262 | 838883 | 0.7440986329928528 | long-budget failure case |

## Object Prior Evidence

| row_id | recon_chamfer_l1_mean | hidden_chamfer_l1_mean | free_space_violation_rate_mean | mesh_extraction_success_rate |
| --- | --- | --- | --- | --- |
| stage3_posterior_encoder | 0.0663909994752951 | 0.0990753869336207 | 0.033534966626213594 | 1.0 |

## Proposal Gate And Rollback Evidence

| row_id | accepted | rejected | cleanup_accepted | floater_rejected | source_model_edited |
| --- | --- | --- | --- | --- | --- |
| parking_patch_proposal_gate | 8 | 16 | 8 | 8 | False |
| m11_synthetic_scene_gate | 1 | 0 |  |  |  |

## Failure Cases

| case_id | type | evidence | action |
| --- | --- | --- | --- |
| stage17_long_budget_collapse | method_failure | Stage17 MeshPrior resume reaches PSNR 10.839708 and depth AbsRel 0.744099 at 7000. | Do not continue longer Stage17 resume sweeps by default. |
| current_branch_topology_inflation | topology_inflation | Current branch 7000 uses 833775 triangles versus clean 285187. | Use prune_50 as the topology-controlled M22 row. |
| prune_66_proxy_regression | proxy_metric_disagreement | prune_66 beats clean render metrics but has worse depth AbsRel than clean. | Keep prune_66 as a Pareto endpoint, not the default row. |
| rejected_noop_and_floater_patch_proposals | scene_gate_rejection | 8 no-op and 8 floater proposals rejected. | Use as safety evidence for proposal gates and rollback. |

## Missing Rows

| row_id | metric_class | status | reason |
| --- | --- | --- | --- |
| second_real_scene | scene_generalization | MISSING | M20 found no second suitable parking-lot COLMAP/image scene under /data/peilincai. |
| integrated_optimization_time_topology_control | method_algorithm | MISSING | M21.5 is post-hoc checkpoint-copy pruning, not training-loop topology control. |
| render_gated_full_meshprior_insertion | scene_method | MISSING | Real-scene MeshPrior edits are validated as copied-patch and checkpoint-copy diagnostics, not full render-gated insertion. |

## Output Files

- `outputs/carnet/meshprior/paper_evidence/failure_case_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/missing_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/object_prior_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/paper_evidence.json`
- `outputs/carnet/meshprior/paper_evidence/proposal_gate_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/scene_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/synthetic_damage_rows.csv`

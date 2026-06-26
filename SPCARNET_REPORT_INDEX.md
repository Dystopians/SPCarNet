# SPCarNet Current Report Entry Point

Date: 2026-06-26

This file is the root-level entry point for the current SPCarNet technical-report package.

Latest vNext qualitative addendum:

- `docs/car_model/6-26-vNext-CounterQualitativePanel-RunLog.md`
- `docs/car_model/vnext_artifacts/counter_qualitative_panel_20260626_125352/counter_cleanbest_base_vnext_panel.png`
- `docs/car_model/6-26-vNext-FrameContract-and-EffectiveMarginGate-Log.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/garden_same_resolution_diagnostic/garden_target_parent_vs_vnext_same_resolution.png`
- `docs/car_model/vnext_artifacts/garden_effective_margin_gate_20260626/reports/garden_ours_26000_vnext_effective_margin_gate_test_results.json`

Read in this order:

1. `docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md`
2. `docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md`
3. `docs/car_model/6-26-SPCarNet-Mentor-PPT-Status-And-vNext-Strict-Report.zh.md`
4. `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md`
5. `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md`
6. `docs/car_model/6-26-SPCarNet-vNext-Strict-FrozenPolicy-Multiscene-Log.md`
7. `docs/car_model/6-26-vNext-StructureAwareShrink-Strict-Multiscene-Log.md`
8. `docs/car_model/6-26-vNext-ManifestRunner-and-Full9Gap-Log.md`
9. `docs/car_model/6-26-vNext-Full9FixedPolicy-CleanupRun-Log.md`
10. `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
11. `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/vnext_full9_cleanup_promotion_manifest.md`
12. `docs/car_model/6-26-vNext-FrameContract-and-EffectiveMarginGate-Log.md`
13. `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`
14. `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary.md`
15. `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_runner_summary.md`
16. `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/cleanbest_qualitative_batch_summary.md`
17. `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/garden_same_resolution_diagnostic/garden_target_parent_vs_vnext_same_resolution_summary.md`
18. `docs/car_model/vnext_artifacts/garden_effective_margin_gate_20260626/reports/garden_vnext_certified_residual_texture_report.md`
19. `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md`
20. `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.json`
21. `docs/car_model/6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md`
22. `docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md`
23. `docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/stump_vnext_certified_residual_texture_report.md`
24. `docs/car_model/6-26-vNext-TreehillInputRebuild-Ready6-and-Rejection-Log.md`
25. `docs/car_model/vnext_artifacts/full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.md`
26. `docs/car_model/vnext_artifacts/treehill_structure_shrink_rebuild_tau002_20260626_0832/treehill_vnext_certified_residual_texture_report.md`
27. `docs/car_model/6-26-vNext-FlowersInputRebuild-Ready7-and-SameEvidenceFallback-Log.md`
28. `docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/flowers_vnext_certified_residual_texture_report.md`
29. `docs/car_model/vnext_artifacts/flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.md`
30. `docs/car_model/6-26-vNext-KitchenInputRebuild-Ready8-and-AcceptedMilestone-Log.md`
31. `docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/kitchen_vnext_certified_residual_texture_report.md`
32. `docs/car_model/vnext_artifacts/kitchen_structure_shrink_rebuild_tau002_20260626_1023/preflight/vnext_manifest_runner_summary.md`
33. `docs/car_model/6-26-vNext-BicycleInputRebuild-Ready9-and-Full9InputClosure-Log.md`
34. `docs/car_model/vnext_artifacts/bicycle_structure_shrink_rebuild_tau002_20260626_1055/reports/bicycle_vnext_certified_residual_texture_report.md`
35. `docs/car_model/vnext_artifacts/bicycle_structure_shrink_rebuild_tau002_20260626_1055/preflight/vnext_manifest_runner_summary.md`
36. `docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.md`
37. `docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.json`
38. `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md`
39. `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.json`
40. `docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_scene_config_20260626.json`
41. `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json`
42. `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_multiscene_20260626_0718/strict_structure_aware_shrink_multiscene_summary.md`
43. `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md`
44. `docs/car_model/6-26-SPCarNet-vNext-SoftShrink-Garden-Milestone-Log.md`
45. `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md`
46. `docs/car_model/vnext_artifacts/README.md`
47. `docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md`
48. `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md`
49. `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md`
50. `docs/car_model/6-25-v113b-OOT-Tail-Safe-Gate-Log.md`
51. `docs/car_model/6-25-v113c-FrameFallback-and-v114-OOFRefit-Log.md`
52. `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
53. `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md`
54. `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md`
55. `docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md`
56. `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md`
57. `docs/car_model/README.md`

Current short status:

- `v106 POD-MoE base-preserve` is the current verified quality line.
- On the assembled selected full9 table, v106 improves over the local clean MeshSplatting baseline by `+0.679598 PSNR`, `+0.011812 SSIM`, and `-0.019185 LPIPS`.
- `v113b OOT tail-safe gate` is the current strict-gate safety repair.
- v113b preserves v106 on flowers and repairs the prior garden v110b regression back to v106 without using target GT for gate decisions.
- `run_v113b_oot_tail_gate_replay_scene.py` is the current replay entry point for applying the v113b certificate to prebuilt strict candidates without rebuilding fields.
- `v113c frame_fallback` improves garden v110b but remains below v106, so it is not promoted as the final method.
- `v114_oof_refit_pod_moe` is the current candidate-side attempt; garden field build is running and not yet a completed result.
- `v111` end-to-end strict validation is still running and is not yet a completed result.
- Latest addendum: v110 counter strict candidate failed with return code `-9` during field build after about `15695.6s`; no field artifact was written. This is recorded as a resource/memory execution blocker, not a promoted or rejected metric result.
- `vNext_certified_residual_surface_texture` now has two strict full9 runs. The first fixed-policy cleanup run is `9/9` completed with `6/9` nonzero accepted and mean `25.067699 / 0.741260 / 0.306689`. The stricter effective-margin gate rerun is also `9/9` completed with `1/9` nonzero accepted, `8/9` fallback/no-op, and mean `25.067410 / 0.741259 / 0.306695`. Both are below local clean MeshSplatting (`25.151682 / 0.749018 / 0.287621`) and v106 (`25.831280 / 0.760830 / 0.268435`). This is full9 protocol/evidence closure and an honest bottleneck diagnosis, not quality superiority.
- Latest vNext safety update: strict qualitative panel tooling now rejects native-size frame-contract mismatches, and the full9 effective-margin gate completed `9 / 9` scenes with `1 / 9` accepted nonzero and `8 / 9` fallback/no-op. Mean metrics are `25.067410 / 0.741259 / 0.306695`, still below clean MeshSplatting and v106. Same-resolution target-evidence parent vs vNext on garden is only `+0.000139 PSNR / +0.000003 SSIM / -0.00000791 LPIPS`, confirming that current visual gains are extremely small.
- `run_vnext_certified_residual_texture_manifest.py` now supports heterogeneous per-scene paths. The original full9 preflight was `4 / 9` input-ready (`bonsai,counter,garden,room`) with five missing evidence/carrier chains.
- Latest local rebuilds: `stump`, `treehill`, `flowers`, `kitchen`, and `bicycle` fit/target evidence plus policy-val pruned carriers were rebuilt under `/dev/shm`, moving local preflight to `9 / 9` input-ready with `0 / 9` missing inputs. `stump/treehill/flowers` strict runs completed with W&B offline and protocol pass, but the certificate rejected them to exact fallback/no-op: `stump` failed tail risk (`cvar20_view_relative_gain=-0.172454`, `min_view_relative_gain=-0.344907`), `treehill` failed lower-tail/SSIM/L1 gates (`cvar20_view_relative_gain=-0.053640`, `min_view_relative_gain=-0.077837`, `ssim_gain=-0.000009413`), and `flowers` failed lower-tail/SSIM/L1 gates (`cvar20_view_relative_gain=-0.224441`, `min_view_relative_gain=-0.278408`, `ssim_gain=-0.000082279`). The flowers same-evidence parent export matches fallback exactly at `19.519194 / 0.490780 / 0.424170`, confirming the rebuilt `images_2` evidence comparison is fair. `kitchen` is accepted nonzero under the same strict no-target-GT protocol (`alpha=0.125`, `changed_fraction=0.003549714`) with same-evidence delta `+0.000786 PSNR / +0.00000256 SSIM / -0.00002818 LPIPS`. `bicycle` closes the final input gap, is accepted nonzero (`alpha=0.015625`, `changed_fraction=0.000173916`), and gives same-evidence micro-delta `+0.00000954 PSNR / +0.000000179 SSIM / -0.000000030 LPIPS`; this is input/protocol closure, not full9 quality closure.
- `spcarnet_v110_v111_v114_package` is the current mechanical strict-branch collector output; it is intentionally incomplete until the running long jobs finish.
- For a fresh clone and mentor/PPT preparation, use `docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md` as the compact executive report, then follow the artifact indices above for tables, images, commands, and protocol audits.

The large render/model trees remain local under `/dev/shm`; the committed repo contains the lightweight Markdown/JSON summaries and qualitative contact sheets needed for PPT analysis.

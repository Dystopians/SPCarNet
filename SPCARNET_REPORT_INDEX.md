# SPCarNet Current Report Entry Point

Date: 2026-06-26

This file is the root-level entry point for the current SPCarNet technical-report package.

Read in this order:

1. `docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md`
2. `docs/car_model/6-26-SPCarNet-Mentor-PPT-Status-And-vNext-Strict-Report.zh.md`
3. `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md`
4. `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md`
5. `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md`
6. `docs/car_model/6-26-SPCarNet-vNext-SoftShrink-Garden-Milestone-Log.md`
7. `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md`
8. `docs/car_model/vnext_artifacts/README.md`
9. `docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md`
10. `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md`
11. `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md`
12. `docs/car_model/6-25-v113b-OOT-Tail-Safe-Gate-Log.md`
13. `docs/car_model/6-25-v113c-FrameFallback-and-v114-OOFRefit-Log.md`
14. `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
15. `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md`
16. `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md`
17. `docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md`
18. `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md`
19. `docs/car_model/README.md`

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
- `vNext_certified_residual_surface_texture` now has two real nonzero single-scene milestones. Garden face-softshrink accepts a residual atlas, changes `0.208%` target pixels, and gives a tiny held-out improvement over the no-op/fallback parent. Counter strict face-softshrink has `target_gt_visible_to_apply=false`, `accepted=true`, `selected_alpha=0.25`, `changed_fraction=1.177%`, and test delta vs Phase-F compact parent of `+0.002131 PSNR / -0.000047 SSIM / -0.000085 LPIPS`. This strengthens the fairness/protocol evidence but is still not full9 closure or proof of superiority over v106 or clean MeshSplatting.
- `spcarnet_v110_v111_v114_package` is the current mechanical strict-branch collector output; it is intentionally incomplete until the running long jobs finish.

The large render/model trees remain local under `/dev/shm`; the committed repo contains the lightweight Markdown/JSON summaries and qualitative contact sheets needed for PPT analysis.

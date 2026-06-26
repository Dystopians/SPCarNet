# SPCarNet Current Report Entry Point

Date: 2026-06-25

This file is the root-level entry point for the current SPCarNet technical-report package.

Read in this order:

1. `docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md`
2. `docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md`
3. `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md`
4. `docs/car_model/6-25-v113b-OOT-Tail-Safe-Gate-Log.md`
5. `docs/car_model/6-25-v113c-FrameFallback-and-v114-OOFRefit-Log.md`
6. `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
7. `docs/car_model/results/v113_oot_tail_20260625/summary/v113b_oot_tail_safe_summary.md`
8. `docs/car_model/results/v113c_frame_fallback_v114_oof_20260625/summary/v113c_v114_summary.md`
9. `docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md`
10. `docs/car_model/results/v110_v111_v114_strict_branch_20260625/summary/spcarnet_v110_v111_v114_package.md`
11. `docs/car_model/README.md`

Current short status:

- `v106 POD-MoE base-preserve` is the current verified quality line.
- On the assembled selected full9 table, v106 improves over the local clean MeshSplatting baseline by `+0.679598 PSNR`, `+0.011812 SSIM`, and `-0.019185 LPIPS`.
- `v113b OOT tail-safe gate` is the current strict-gate safety repair.
- v113b preserves v106 on flowers and repairs the prior garden v110b regression back to v106 without using target GT for gate decisions.
- `run_v113b_oot_tail_gate_replay_scene.py` is the current replay entry point for applying the v113b certificate to prebuilt strict candidates without rebuilding fields.
- `v113c frame_fallback` improves garden v110b but remains below v106, so it is not promoted as the final method.
- `v114_oof_refit_pod_moe` is the current candidate-side attempt; garden field build is running and not yet a completed result.
- `v111` end-to-end strict validation is still running and is not yet a completed result.
- `spcarnet_v110_v111_v114_package` is the current mechanical strict-branch collector output; it is intentionally incomplete until the running long jobs finish.

The large render/model trees remain local under `/dev/shm`; the committed repo contains the lightweight Markdown/JSON summaries and qualitative contact sheets needed for PPT analysis.

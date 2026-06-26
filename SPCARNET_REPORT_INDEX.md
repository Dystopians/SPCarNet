# SPCarNet Current Report Entry Point

Date: 2026-06-25

This file is the root-level entry point for the current SPCarNet technical-report package.

Read in this order:

1. `docs/car_model/6-25-SPCarNet-Cloneable-Report-Index.md`
2. `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md`
3. `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
4. `docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md`
5. `docs/car_model/README.md`

Current short status:

- `v106 POD-MoE base-preserve` is the current verified quality line.
- On the assembled selected full9 table, v106 improves over the local clean MeshSplatting baseline by `+0.679598 PSNR`, `+0.011812 SSIM`, and `-0.019185 LPIPS`.
- `v110/v110b/v111` are strict-split fairness/safety diagnostics, not the final promoted method.
- v110b preserves v106 on flowers but still regresses relative to v106 on garden, exposing a real gate-generalization weakness.

The large render/model trees remain local under `/dev/shm`; the committed repo contains the lightweight Markdown/JSON summaries and qualitative contact sheets needed for PPT analysis.

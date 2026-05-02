# MeshPrior Stage 18 Topology Budget Design

Date: 2026-05-01

## Goal

Prevent false method claims caused by comparing render quality without topology and speed context.

## Inputs

Stage 18 reads the three currently available 2000-iteration parking runs:

- clean candidate: `outputs/carnet/meshprior/parking_phone_tiny/origin_main_2000iter/model`
- current branch engineering baseline: `outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model`
- Stage17 MeshPrior real variant: `outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter/model`

## Metrics

The collector must include:

- checkpoint triangle and vertex count;
- `render.py + metrics.py` PSNR, SSIM, LPIPS;
- COLMAP sparse geometry proxy depth and normal metrics;
- training-internal FPS with source label;
- PSNR and SSIM per 100k triangles;
- triangle ratio versus clean baseline and current baseline;
- final-cleanup state where available.

## Interpretation Rule

If Stage17 improves quality but uses more than `5x` the clean candidate topology, the decision is:

`QUALITY_GAIN_NOT_TOPOLOGY_NORMALIZED`

This does not invalidate Stage17. It means M18 must block paper-level claims until either budget-matched cleanup, topology control, or efficiency-normalized reporting is complete.

## Gate

Stage gate is `PASS` if the collector produces JSON, CSV, and Markdown tables and no row hides topology or speed.

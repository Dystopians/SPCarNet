# MeshSplatOpt Stage R13 Synthetic Repair Benchmark Design

Date: 2026-05-02

## Goal

Build a controlled synthetic benchmark to prove repair capability before public-scene GPU runs.

## Damage Types

R13 tracks:

- floater triangles;
- local dent;
- noisy rough patch;
- vehicle side discontinuity;
- ground/wall misalignment;
- small hole;
- giant ground void;
- prior-only unobserved void;
- appearance corruption placeholder.

## Methods

Implemented in the first benchmark:

- `no_repair`;
- `delete_only_prism_style`;
- `full_meshsplatopt_repair`.

The result schema reserves rows for QEM/decimation, classical hole fill, snap only, fill only, and no-counterfactual ablations.

## Metrics

Synthetic metrics are evaluation-only:

- triangle count;
- surface error proxy;
- hole boundary reduction;
- giant void area repaired;
- free-space violation;
- normal error proxy;
- topology validity;
- accepted/rejected edits;
- prior-only false fill rate.

Clean synthetic state is used only for metrics, never proposal selection.

## Gate

`PASS` requires full MeshSplatOpt to improve at least four of seven synthetic damage categories over delete-only and reject the prior-only unknown void in normal mode.

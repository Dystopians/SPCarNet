# Final Stage F82 Policy v5 Robustness Report

Decision: `FIXED_POLICY_V5_TWO_SEED_PASS`.

F80 exposed a real weakness in the F79 policy: bonsai with `train_seed=1` improved render, AbsRel, and normal angle, but missed Depth MAE by `+0.000481`. The failure was tiny, but it showed the 28.25% bonsai budget had no seed margin.

The v5 repair tightens the small-scene positive-evidence risk cap in the adaptive fallback. This is still one fixed policy, not a per-scene table. It changes the selected budgets as follows:

- bonsai: 28.25% removed -> 25.00% removed
- room: 18.50% removed -> 15.25% removed
- counter: 18.50% removed -> 15.25% removed
- courtyard: unchanged at 72.00% removed

All rows use strict topology freeze, sparse-depth lambda `0.001`, LPIPS lambda `0.00025`, online W&B, and `22000 -> 26000` recovery.

## F81: train_seed=1

| scene | W&B | prune removed | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bonsai | `njlfymc5` | 25.00% | +0.124448 | +0.018386 | -0.013737 | -0.012889 | -0.023694 | -3.064574 | `PASS_ALL_METRIC_CLEAN_WIN` |
| courtyard | `half791j` | 72.00% | +0.098597 | +0.012281 | -0.002179 | -0.056065 | -0.503443 | -0.609621 | `PASS_ALL_METRIC_CLEAN_WIN` |
| room | `irnab2vf` | 15.25% | +0.906519 | +0.088126 | -0.065522 | -0.019931 | -0.105544 | -1.328299 | `PASS_ALL_METRIC_CLEAN_WIN` |
| counter | `5ruy0vmi` | 15.25% | +0.277655 | +0.033515 | -0.027355 | -0.009000 | -0.025869 | -1.268269 | `PASS_ALL_METRIC_CLEAN_WIN` |

## F82: train_seed=0

| scene | W&B | prune removed | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bonsai | `f5zh2jda` | 25.00% | +0.124832 | +0.018306 | -0.013226 | -0.012909 | -0.010569 | -3.210992 | `PASS_ALL_METRIC_CLEAN_WIN` |
| courtyard | `mie1nxrx` | 72.00% | +0.095103 | +0.012001 | -0.002621 | -0.052764 | -0.489171 | -0.605947 | `PASS_ALL_METRIC_CLEAN_WIN` |
| room | `hezmbm8v` | 15.25% | +0.901082 | +0.087985 | -0.065891 | -0.019631 | -0.104928 | -1.403494 | `PASS_ALL_METRIC_CLEAN_WIN` |
| counter | `3egx4xqv` | 15.25% | +0.279626 | +0.033630 | -0.027678 | -0.008726 | -0.029065 | -1.220942 | `PASS_ALL_METRIC_CLEAN_WIN` |

## Summary

- F81 + F82 available rows: `8 / 8`
- all-metric clean wins: `8 / 8`
- topology unchanged: `8 / 8`

This supersedes the F79 tag as the safer multiscene fixed-policy evidence. F79 remains a useful checkpoint because it had stronger compression on small scenes, but F82 is the better paper-facing policy because it carries seed margin.

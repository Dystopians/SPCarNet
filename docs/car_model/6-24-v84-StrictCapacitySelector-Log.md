# v84 Strict Capacity Selector Log

Date: `2026-06-24`

Status: `REPORT_ONLY_MATERIALIZED_SELECTOR`

## Motivation

The raw `v82 capacity-prerank + face-alpha` candidate produced a tiny strict win on
`counter`, but hard-triad validation showed that the same raw fixed policy was not
strictly non-regressive against the v64 anchor on `kitchen` and `bonsai`.

The next useful engineering step is therefore not to promote raw v82. Instead, v84
materializes a conservative fixed selector:

```text
use v82 capacity-prerank only when train/policy-val evidence is strict;
otherwise fallback to the already materialized v64 selected policy.
```

This keeps selection auditable and avoids manual per-scene parameter picking.

## Implementation

New script:

```text
scripts/car_model/summarize_v84_strict_v82_capacity_selector.py
```

Outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_selected_full9/
```

The selected tree contains small copied JSON/MD/log artifacts and render/GT symlinks.
It was intentionally kept small because `/data` had less than 500 MB free. Some
symlinks point into `/dev/shm`, so this materialized tree is valid for the current
machine state but is not a durable archival artifact unless those sources are
preserved or copied later.

## Fixed Guard

The selector does not read held-out PSNR/SSIM/LPIPS values for branch selection.
Its guard uses train/policy-val audit fields plus a target-side changed-fraction
effect-size check from `target_apply`; that target check does not use target GT
metrics.

| guard | value |
|---|---:|
| selected alpha range | `[0.5, 0.5]` |
| min target changed fraction | `0.001` |
| min policy-val SSIM gain | `0.0002` |
| min policy-val SSIM positive fraction | `1.0` |
| min policy-val SSIM min-view gain | `0.00005` |
| min policy-val image-L1 gain | `0.00002` |
| min policy-val image-L1 positive fraction | `0.9` |
| min policy-val image-L1 min-view gain | `-0.000001` |
| min policy-val image-L1 CVaR20 gain | `0.000001` |

## Result

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v84 vs v64 | `9` | `1` | `9` | `+0.000000848` | `+0.000000013` | `-0.000000079` |
| v84 vs v56 | `9` | `2` | `9` | `+0.000410928` | `+0.000000291` | `-0.000019030` |
| v84 vs no-op | `9` | `7` | `8` | `+0.002256817` | `+0.000038094` | `-0.000093525` |

Per-scene decision:

| scene | selected | reason |
|---|---|---|
| counter | `v82_capacity_prerank` | passes strict evidence guard |
| kitchen | `v64_fallback` | rejected because selected alpha is `1.0 > 0.5` |
| bonsai | `v64_fallback` | rejected because alpha and policy-val gains are too weak |
| other six scenes | `v64_fallback` | no v82 candidate materialized yet |

## Interpretation

v84 is useful for ablation hygiene and engineering closure: it makes the best current
representation-level tweak non-regressive against v64 on full9 and captures the tiny
`counter` win without hand-selecting a scene at reporting time.

It is not a paper-clean breakthrough. The rule was formed after the v82 hard-triad
diagnosis, and the v82 candidate set is incomplete for full9. Treat v84 as a fixed
candidate that still needs fresh blind validation.

## Validation

Command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/summarize_v84_strict_v82_capacity_selector.py
```

Static check:

```text
git diff --check -- scripts/car_model/summarize_v84_strict_v82_capacity_selector.py outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v84_strict_v82_capacity_selector_full9_summary.md
```

Result: passed.

# v86 Anchor-Preserving Tail-Risk Selector Log

Date: `2026-06-24`

Status: `COMPLETED_GUARDRAIL_NOT_PROMOTED`

## Motivation

v85 target-footprint tail-risk proved that the certificate can accept a
non-degenerate edit without collapsing the held-out view. However, the final
`counter` metrics were only an anchor-level micro-tie:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v85 target-footprint tail-risk | `26.756134033` | `0.862126231` | `0.251691371` |
| v56/v64/v79 anchor | `26.756130219` | `0.862126231` | `0.251691371` |
| v84 counter row | `26.756137848` | `0.862126350` | `0.251690656` |

This exposed a selector problem: a safety certificate is not enough. A future
tail-risk candidate should not replace an already stronger selected anchor unless
its train/policy-val evidence also dominates the anchor.

## Method Change

Implemented:

```text
scripts/car_model/summarize_v86_anchor_preserving_tailrisk_selector.py
```

The script materializes a fixed selector:

```text
anchor = v84 selected full9 endpoint
candidate = v85 target-footprint tail-risk candidate, if available

use candidate only if:
  - v85 audit accepted an atlas;
  - policy-val risk gate passed;
  - selected candidate is a prior-bin-gain hybrid;
  - target-footprint tail-risk certificate was enabled and retained bins;
  - fixed SSIM/L1/changed-fraction gates pass;
  - if the v84 anchor has comparable train-policy audit, v85 dominates it on
    policy-val SSIM/L1 mean and tail fields;
otherwise preserve v84/v64 fallback.
```

Selection uses only train/policy-val audit fields. Held-out metrics are used only
after selection to validate the materialized endpoint.

## Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v86_anchor_preserving_tailrisk_selector.py
```

## Output

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v86_anchor_preserving_tailrisk_selector_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v86_anchor_preserving_tailrisk_selector_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v86_anchor_preserving_tailrisk_selector_selected_full9/
```

The selected tree contains all `9 / 9` scenes and has render/GT links for all
`9 / 9` scenes.

## Result

Aggregate summary:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v86 vs v84 anchor | `9` | `0` | `9` | `+0.000000000` | `+0.000000000` | `+0.000000000` |
| v86 vs v64 | `9` | `1` | `9` | `+0.000000848` | `+0.000000013` | `-0.000000079` |
| v86 vs v56 | `9` | `2` | `9` | `+0.000410928` | `+0.000000291` | `-0.000019030` |
| v86 vs no-op | `9` | `7` | `8` | `+0.002256817` | `+0.000038094` | `-0.000093525` |

Per-scene decision that matters:

| scene | v85 found | selected | reason |
|---|---:|---|---|
| counter | `1` | `v84_anchor_fallback` | v85 tail-risk does not dominate v84/v82b train-policy audit on SSIM/L1 fields |
| other eight scenes | `0` | `v84_anchor_fallback` | no v85 candidate |

The concrete counter rejection reasons are:

```text
selected_ssim_gain_below_anchor:
  0.0002939055363337199 < 0.0002947101990381877
selected_image_l1_gain_below_anchor:
  0.000026771643509467442 < 0.000026923759529987972
selected_image_l1_min_view_gain_below_anchor:
  -0.0000008121132850646973 < -0.0000008009374141693115
```

## Interpretation

v86 is a useful guardrail and a real pipeline artifact: it prevents target-side
tail-risk certificates from accidentally replacing a stronger representation-level
anchor. It is not a new promoted paper endpoint, because the selected metrics are
identical to v84 and the v85 candidate set is still only `counter`.

This changes the next research direction. The right next experiment is no longer
"promote v85 as-is"; it is to generate genuinely stronger v85-style candidates
that can pass this anchor-dominance selector before spending GPU time on
hard-triad or full9 expansion.

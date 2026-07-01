# v336c Strict Per-View Oracle Gap vs v335

Date: 2026-07-01

Command:

```bash
python scripts/car_model/summarize_v323_dominance_gap.py \
  --baseline_root outputs/carnet/spcarnet_v335_target_neighbor_candidate_unlock_full9_20260701 \
  --method_root outputs/carnet/spcarnet_v336c_source_summary_gate_full9_20260701 \
  --output_json docs/car_model/results/v336c_strict_oracle_gap_vs_v335.json
```

This is an offline diagnostic over already saved
`support_transport_apply_report.json` files. It does not rerun rendering and
does not introduce a new decision rule. The strict oracle can only pick a
candidate whose `psnr_gain` and `ssim_gain` are both at least the selected
output for that view, then maximizes PSNR with SSIM as tie-break.

## Macro Result

| method root | scenes | views | improved oracle views | selected PSNR | strict oracle PSNR | oracle gap | selected SSIM | strict oracle SSIM | oracle gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v335 | 9 | 246 | 66 | 0.274017908934 | 0.282687390512 | +0.008669481579 | 0.003741526179 | 0.003815885708 | +0.000074359529 |
| v336c | 9 | 246 | 62 | 0.274617423486 | 0.283267289896 | +0.008649866410 | 0.003744976625 | 0.003817232841 | +0.000072256216 |

v336c improves the selected policy over v335 by `+0.000599514552` PSNR and
`+0.000003450447` SSIM, but it does not materially reduce the remaining strict
per-view oracle gap. The method still leaves about `+0.00865 dB` scene-macro
strict headroom in the already computed candidate pool.

## Largest v336c Scene Gaps

| scene | views | improved oracle views | mean PSNR gap | mean SSIM gap | selected variants | improved oracle variants |
|---|---:|---:|---:|---:|---|---|
| stump | 16 | 7 | +0.021858285120 | +0.000097338110 | `fixed:16` | `learned:5, mix0750:2` |
| treehill | 18 | 6 | +0.015234013794 | +0.000111914343 | `fixed:18` | `learned:5, hybrid:1` |
| room | 39 | 13 | +0.011576829779 | +0.000119154270 | `hybrid:39` | `learned:8, adaptive:3, mix0750:2` |
| bicycle | 25 | 13 | +0.010343502985 | +0.000128235817 | `hybrid:25` | `fixed:7, learned:4, mix0250:1, mix0750:1` |
| bonsai | 37 | 3 | +0.007299128751 | +0.000070952080 | `learned:37` | `fixed:2, mix0750:1` |

## Largest v336c Per-View Misses

| scene/view | selected | strict oracle | dPSNR | dSSIM |
|---|---|---|---:|---:|
| bonsai/00035 | learned | fixed | +0.256317117624 | +0.002426087856 |
| room/00011 | hybrid | learned | +0.121721897073 | +0.000903427601 |
| treehill/00011 | fixed | learned | +0.097477838368 | +0.000548839569 |
| treehill/00016 | fixed | learned | +0.084025668437 | +0.000266134739 |
| room/00023 | hybrid | learned | +0.081403643404 | +0.000925242901 |
| stump/00014 | fixed | learned | +0.079235783778 | +0.000358104706 |
| kitchen/00018 | learned | fixed | +0.077108477521 | +0.000544965267 |
| stump/00002 | fixed | learned | +0.075188994536 | +0.000432372093 |
| stump/00000 | fixed | learned | +0.066775879615 | +0.000259459019 |
| bicycle/00003 | hybrid | learned | +0.061318412745 | +0.000898897648 |
| room/00012 | hybrid | adaptive | +0.061090445730 | +0.000020802021 |
| stump/00012 | fixed | learned | +0.052504093273 | +0.000145792961 |

## Interpretation

The key bottleneck is still target-blind per-view arbitration. v336c is safer
than v336b and slightly stronger than v335, but the oracle profile remains
nearly unchanged:

- many large misses are `fixed -> learned` on outdoor `stump/treehill` and
  indoor `room`;
- some misses are `learned -> fixed`, especially `bonsai/00035` and
  `kitchen/00018`, so a simple learned-biased rule would be unsafe;
- `adaptive` helps only a few room views in the strict oracle table, confirming
  that its coverage is narrow;
- the next useful method must provide a stronger no-target-GT reliability signal
  for per-view candidate selection, or a candidate generator whose source-heldout
  safety evidence is positive on more scenes.

The immediate research direction should therefore be a fixed-policy diagnostic
over all existing candidates: compute target-neighbor/support-dropout/source
proxy features for every candidate on the largest oracle-gap views, then test
whether a target-blind ranker can recover the strict oracle headroom without
regressing the known learned-to-fixed failure cases.

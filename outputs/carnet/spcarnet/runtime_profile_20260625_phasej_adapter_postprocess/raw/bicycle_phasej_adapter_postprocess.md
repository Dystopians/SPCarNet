# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bicycle/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `25`
- support frame count: `169`
- repeats: `2`
- alpha: `0.0`
- k: `8`
- mode: `residual`
- depth rel tol: `0.12`
- residual clip: `0.2`
- direction weight: `0.2`
- CPU wall mean sec: `31.024441`
- mean ms/frame: `1240.977650`
- CUDA peak allocated MiB max: `2865.45654296875`
- CUDA peak reserved MiB max: `2952.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.264411 | 1410.576439 | 0.708930 | 2865.093 | 2938.000 |
| 2 | 26.784472 | 1071.378862 | 0.933377 | 2865.457 | 2952.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

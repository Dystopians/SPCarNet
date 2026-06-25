# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `37`
- support frame count: `255`
- repeats: `2`
- alpha: `0.0`
- k: `4`
- mode: `residual`
- depth rel tol: `0.06`
- residual clip: `0.25`
- direction weight: `0.35`
- CPU wall mean sec: `33.183776`
- mean ms/frame: `896.858816`
- CUDA peak allocated MiB max: `4437.76611328125`
- CUDA peak reserved MiB max: `4802.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.531397 | 960.308019 | 1.041333 | 4437.766 | 4780.000 |
| 2 | 30.836156 | 833.409614 | 1.199890 | 4437.684 | 4802.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/stump/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `16`
- support frame count: `109`
- repeats: `2`
- alpha: `0.0`
- k: `4`
- mode: `residual`
- depth rel tol: `0.12`
- residual clip: `0.2`
- direction weight: `0.2`
- CPU wall mean sec: `14.345431`
- mean ms/frame: `896.589428`
- CUDA peak allocated MiB max: `2865.306640625`
- CUDA peak reserved MiB max: `2980.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.091849 | 943.240541 | 1.060175 | 2030.614 | 2064.000 |
| 2 | 13.599013 | 849.938315 | 1.176556 | 2865.307 | 2980.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

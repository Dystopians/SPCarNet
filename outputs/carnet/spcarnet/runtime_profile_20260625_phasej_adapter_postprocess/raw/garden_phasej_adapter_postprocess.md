# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `24`
- support frame count: `161`
- repeats: `2`
- alpha: `0.0`
- k: `4`
- mode: `residual`
- depth rel tol: `0.06`
- residual clip: `0.2`
- direction weight: `0.35`
- CPU wall mean sec: `23.584908`
- mean ms/frame: `982.704509`
- CUDA peak allocated MiB max: `2978.78076171875`
- CUDA peak reserved MiB max: `3410.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.033238 | 1043.051594 | 0.958725 | 2754.352 | 3108.000 |
| 2 | 22.136578 | 922.357424 | 1.084178 | 2978.781 | 3410.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `22`
- support frame count: `151`
- repeats: `2`
- alpha: `0.0`
- k: `4`
- mode: `residual`
- depth rel tol: `0.12`
- residual clip: `0.2`
- direction weight: `0.2`
- CPU wall mean sec: `21.171509`
- mean ms/frame: `962.341330`
- CUDA peak allocated MiB max: `2866.12060546875`
- CUDA peak reserved MiB max: `2988.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.176910 | 1008.041366 | 0.992023 | 2490.742 | 2540.000 |
| 2 | 20.166108 | 916.641293 | 1.090939 | 2866.121 | 2988.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `39`
- support frame count: `272`
- repeats: `2`
- alpha: `0.0`
- k: `4`
- mode: `residual`
- depth rel tol: `0.06`
- residual clip: `0.25`
- direction weight: `0.35`
- CPU wall mean sec: `44.050467`
- mean ms/frame: `1129.499160`
- CUDA peak allocated MiB max: `4415.2958984375`
- CUDA peak reserved MiB max: `4846.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.109648 | 1156.657650 | 0.864560 | 4415.294 | 4846.000 |
| 2 | 42.991286 | 1102.340671 | 0.907161 | 4415.296 | 4816.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

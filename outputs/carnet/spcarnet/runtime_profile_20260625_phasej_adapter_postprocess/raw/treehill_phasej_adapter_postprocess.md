# ELA Postprocess Runtime Profile

This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.

## Summary

- base model path: `/data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model`
- base method name: `ours_26000_phasef_extra_compact_base`
- target split: `test`
- device: `cuda:0`
- target frame count: `18`
- support frame count: `123`
- repeats: `2`
- alpha: `0.75`
- k: `4`
- mode: `residual`
- depth rel tol: `0.06`
- residual clip: `0.2`
- direction weight: `0.35`
- CPU wall mean sec: `15.513958`
- mean ms/frame: `861.886577`
- CUDA peak allocated MiB max: `2885.111328125`
- CUDA peak reserved MiB max: `3470.0`

## Repeats

| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.041869 | 946.770501 | 1.056222 | 2217.569 | 2638.000 |
| 2 | 13.986048 | 777.002652 | 1.286997 | 2885.111 | 3470.000 |

## Scope

- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.
- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.
- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.

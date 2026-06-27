# 6-26 v116/v117 New Prompt Progress and Bottleneck Log

## Short Answer

The new prompt produced meaningful engineering and protocol progress, but it has not yet reached the expected paper-level effect.

Current confidence:

- Direction confidence: medium-high. The train-only residual surface texture pipeline is now more auditable and less like manual parameter search.
- Result confidence: low-medium. The current v116/v117 counter results are still tiny deltas vs v106 and do not yet support a strong paper claim.
- Completion status: NOT COMPLETE.

## What Was Implemented

### v116 Target-visible residual energy and quantized audit

Implemented in `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` and exposed through `scripts/car_model/run_vnext_certified_residual_texture_scene.py`.

Key changes:

- Target-support candidate ranking can use target-visible residual energy instead of raw float changed area.
- Target apply/audit now records PNG-quantized changed pixels/fraction, not only float-level changed pixels.
- Fixed single-candidate target support profiling so fixed-policy runs also get target-support audit fields.
- Runner passes the new energy-ranking flag by default unless explicitly disabled.
- Runner now forwards `--allow_resize` to teacher-cache construction when requested.

### v117 face-level target-footprint residual-debt transfer

Implemented a new explicit switch:

```bash
--target_footprint_residual_debt_match_level face
```

This keeps the default old behavior as `bin`, but allows train-certified residual debt on a face to be matched to GT-free target footprint at face level instead of requiring the exact same face/UV bin.

The goal was to test whether the previous support expansion was too sparse because bin matching was too strict.

## Counter Pilot Results

Fair anchor: local v106 counter from `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json`.

| Method | Status | PSNR | SSIM | LPIPS | Notes |
|---|---:|---:|---:|---:|---|
| v106 POD-MoE counter | complete | 27.499645 | 0.867521 | 0.238847 | current representation-style anchor |
| v116 quant-single | complete | 27.499702 | 0.867478 | 0.238779 | tiny PSNR/LPIPS gain, SSIM down |
| v116 full-basis multi-candidate | complete | 27.499722 | 0.867478 | 0.238779 | selected target-footprint candidate, still only +1 face |
| v116 no-basis ablation | complete | 27.499630 | 0.867521 | 0.238846 | rejected/no-op; basis is needed |
| v117 face-transfer | complete | 27.499702 | 0.867478 | 0.238779 | same as v116 quant-single; selected base carrier |

Important audit values:

- v116/v117 changed fraction: `0.0132671780`
- v116/v117 PNG-quantized changed fraction: `0.0070996403`
- v116 full-basis changed fraction: `0.0137059394`; this run was launched before the PNG-quantized audit patch, so it has no PNG-quantized changed-fraction field.
- v116 no-basis changed fraction: `0.0`, fallback no-op
- v117 target-footprint expansion still only added `1` face, so face-level transfer did not solve the coverage bottleneck.

Result paths:

- v116 quant-single: `/dev/shm/peilincai_spcarnet_v116_counter_quant_single_20260626_1643/counter`
- v116 no-basis: `/dev/shm/peilincai_spcarnet_v116_counter_energy_nobasis_20260626_1702/counter`
- v117 face-transfer: `/dev/shm/peilincai_spcarnet_v117_counter_face_transfer_20260626_1645/counter`
- v116 full-basis multi-candidate run: `/dev/shm/peilincai_spcarnet_v116_counter_energy_small_20260626_1645/counter`

## Interpretation

The new prompt has produced real progress in protocol quality:

- target/test GT is not used for candidate selection;
- fallback behavior is explicit;
- ablations now distinguish full representation capacity from no-basis behavior;
- changed-region visibility is audited in the actual saved PNG space.

However, it has not produced the expected visual or metric breakthrough:

- v116 quant-single improves PSNR and LPIPS vs v106 by only about `+0.000057 PSNR` and `-0.000068 LPIPS`, while SSIM drops by about `-0.000043`.
- v116 full-basis multi-candidate improves PSNR by only about `+0.000076` vs v106, while SSIM remains lower. The scan therefore does not change the main conclusion.
- v117 does not improve over v116 because the extra support footprint remains almost empty.
- Current target-visible changed area is still too small to support a strong qualitative claim.

## Main Bottleneck

The bottleneck is not just alpha tuning or candidate ranking.

Current support expansion requires target-visible regions to overlap train-certified residual-debt regions. Even after relaxing bin-level matching to face-level matching, only one extra face was eligible on counter. This means:

- target-visible residual capacity is starved;
- adding extra empty faces does not help unless there is a reliable residual source for them;
- future work needs a real residual transfer mechanism, not another threshold scan.

The `input.ply` in the compact model contains only vertices, not face elements, so direct triangle adjacency is unavailable from the model PLY. A more realistic next method is to build an empirical face graph from evidence:

- image-space neighboring face ids,
- multi-view co-visible face ids,
- train residual-debt faces as source nodes,
- target-visible faces as destination nodes,
- policy-val/tail-risk certificate as the safety gate.

## Next Required Method Direction

The next implementation should not be another small v106/v116 parameter variant.

Recommended v118 direction:

1. Build a train-only co-visible face graph from evidence face-id maps.
2. Propagate residual priors only from train-certified, multi-view-consistent source faces.
3. Allow target-visible destination faces to receive a low-confidence residual prior through this graph.
4. Keep parent-preserving output and policy-val certificate.
5. Reject or no-op if the propagated prior fails PSNR/SSIM/LPIPS/tail-risk gates.

This is the first next step that plausibly addresses the actual observed bottleneck: no residual source for target-visible regions.

## Current Assessment

Significant progress? Yes, in engineering closure, auditability, and diagnosis.

Reached expected effect? No. The expected effect was a clear representation-level improvement with stronger visual evidence; current gains are tiny and not enough.

Confidence in eventual target? Conditional. I have confidence that the current evidence identifies the right bottleneck. I do not have confidence that the current v116/v117 method itself can meet the final target without a stronger residual transfer mechanism.

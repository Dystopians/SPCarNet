# vNext Strict Frozen-Policy Multiscene Summary

Date: 2026-06-26

Policy: frozen face-softshrink strict no-target-GT apply. This table uses one fixed policy across counter, bonsai, and room. No scene-specific retuning is applied in these runs.

| scene | protocol | accepted | alpha | changed fraction | PSNR delta vs Phase-F | SSIM delta vs Phase-F | LPIPS delta vs Phase-F | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| counter | pass / applyGT=False | `True` | `0.25` | `0.011774` | `+0.002131` | `-0.000047` | `-0.000085` | Strict no-target-GT apply protocol passes and produces nonzero accepted residual surface texture; PSNR and LPIPS improve slightly versus Phase-F compact parent while SSIM slightly regresses, so this is a protocol/multiscene milestone, not comprehensive MeshSplatting/v106 superiority. |
| bonsai | pass / applyGT=False | `True` | `0.25` | `0.001513` | `+0.001225` | `-0.000010` | `-0.000018` | Strict no-target-GT apply passes and accepts a nonzero residual surface texture; PSNR and LPIPS improve slightly while SSIM slightly regresses versus the Phase-F compact parent. |
| room | pass / applyGT=False | `False` | `0.0` | `0.000000` | `-0.000097` | `-0.000003` | `-0.000007` | Strict no-target-GT apply passes but the certificate rejects the nonzero candidate and writes exact fallback/no-op; final metrics are effectively parent-level with tiny evaluation noise. |

## Counts

```json
{
  "scene_count": 3,
  "complete": 3,
  "protocol_passed": 3,
  "target_gt_hidden_from_apply": 3,
  "accepted_nonzero": 2,
  "fallback_noop": 1,
  "psnr_better_vs_parent": 2,
  "ssim_better_vs_parent": 0,
  "lpips_better_vs_parent": 3
}
```

Mean delta vs Phase-F compact parent: `+0.001086` PSNR / `-0.000020` SSIM / `-0.000037` LPIPS.

## Claim Boundary

This strict three-scene frozen policy proves leak-safe multi-scene execution and nonzero accepted residual textures on counter/bonsai, but it is not a paper-final win because SSIM regresses on all three scenes and room falls back.

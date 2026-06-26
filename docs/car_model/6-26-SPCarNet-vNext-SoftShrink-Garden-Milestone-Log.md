# SPCarNet vNext Soft-Shrink Garden Milestone Log

Date: 2026-06-26

This log records the first real nonzero `vNext_certified_residual_surface_texture` milestone after the initial fallback-only pilot.

## Summary

`vNext` is no longer only a protocol/fallback proof on `garden`: the `face-softshrink` variant accepted a nonzero residual surface texture under train-policy-val gates and improved the held-out test aggregate by a tiny amount versus the exact no-op/fallback parent.

This is a useful method milestone, but not a paper-level closure. The effect is visually subtle and numerically very small. Full9, v106 comparison, clean MeshSplatting comparison, and ablations remain unfinished.

## Implemented Method Change

The scene runner now exposes the existing adapter's soft bin uncertainty shrink policy and allows the hard bin uncertainty guard to be disabled:

- `--bin_uncertainty_shrink_policy_mode keep_with_downweight`
- `--bin_uncertainty_shrink_min_bin_samples 16`
- `--bin_uncertainty_shrink_min_positive_view_fraction 0.5`
- `--bin_uncertainty_shrink_fallback_shrink 1.0`
- `--no_policy_val_bin_uncertainty_guard`

The motivation is empirical: the hard bin guard was over-restrictive on `garden`, retaining only `5.03%` of policy-val samples and rejecting a candidate that had already passed the face-level train-policy-val certificate. Soft shrink keeps the evidence-aware local downweighting without turning sparse bins into a brittle all-or-nothing allowlist.

## Runs

### Initial Full Candidate Pilot

Artifact root:

```text
docs/car_model/vnext_artifacts/garden_20260626_004134/
```

Outcome:

| field | value |
|---|---:|
| accepted | `False` |
| effective policy | `fallback_noop` |
| selected alpha | `0.0` |
| target changed fraction | `0.0` |
| PSNR / SSIM / LPIPS | `24.741003 / 0.754049 / 0.248023` |

Diagnosis: the best mean-MSE candidate had strong average residual gain, but lower-tail and image-SSIM gates rejected it.

### Hard-Bin Soft-Shrink Diagnostic

Artifact root:

```text
docs/car_model/vnext_artifacts/garden_hardbin_softshrink_20260626_035631/
```

Outcome:

| field | value |
|---|---:|
| accepted | `False` |
| effective policy | `fallback_noop` |
| selected alpha | `0.0` |
| target changed fraction | `0.0` |
| PSNR / SSIM / LPIPS | `24.741003 / 0.754049 / 0.248023` |

Key diagnostic: alpha refinement and soft shrink fixed the SSIM direction, but the hard bin guard still rejected the candidate:

```text
cvar20_view_relative_gain -0.000741 < 0
min_view_relative_gain -0.002424 < -0.000001
```

The face guard sub-stage was already accepted:

| field | value |
|---|---:|
| face guard accepted | `True` |
| face guard selected alpha | `0.0625` |
| face guard relative gain | `0.006009` |
| face guard CVaR20 gain | `0.002396` |
| face guard min-view gain | `0.000258` |
| face guard SSIM gain | `0.000001659` |

### Face-SoftShrink Accepted Milestone

Artifact root:

```text
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/
```

Outcome:

| field | value |
|---|---:|
| accepted | `True` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.0625` |
| target changed pixels | `82767` |
| target changed fraction | `0.002080` |
| policy-val relative gain | `0.006009` |
| policy-val positive-view fraction | `1.000000` |
| policy-val CVaR20 relative gain | `0.002396` |
| policy-val min-view relative gain | `0.000258` |
| policy-val SSIM gain | `0.000001659` |

Held-out garden test aggregate:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| no-op/fallback parent | `24.741003` | `0.754049` | `0.248023` |
| vNext face-softshrink | `24.741079` | `0.754051` | `0.248020` |
| delta, better direction | `+0.000076` | `+0.00000197` | `-0.00000323` |

Per-view better/tie/worse versus the no-op/fallback parent:

| metric | better | tie | worse |
|---|---:|---:|---:|
| PSNR | `22` | `0` | `2` |
| SSIM | `24` | `0` | `0` |
| LPIPS | `22` | `0` | `2` |

Qualitative panel:

```text
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png
```

The panel includes GT, parent, vNext, amplified parent error, amplified vNext error, and amplified `vNext-parent` difference. The visual difference is real but weak, so this should be presented as a first nonzero representation milestone rather than a visually decisive result.

## W&B Offline Runs

```text
/dev/shm/peilincai_wandb_vnext_softshrink_garden_20260626_035631/wandb/offline-run-20260626_040400-2ha0iu2v
/dev/shm/peilincai_wandb_vnext_face_softshrink_garden_20260626_040558/wandb/offline-run-20260626_041227-nilps441
```

## Claim Boundary

What can be claimed:

- real train-policy-val method change in the vNext train/eval path;
- first nonzero accepted residual surface texture on `garden`;
- tiny but three-metric positive aggregate delta versus exact no-op/fallback parent;
- complete command, audit, manifest, W&B offline, per-view, and qualitative artifacts saved.

What cannot be claimed:

- no full9 validation yet;
- no proof of superiority over v106;
- no proof of superiority over the strongest clean MeshSplatting checkpoint;
- no visible qualitative breakthrough;
- no paper-level closed loop.

## Next Required Work

1. Freeze `face-softshrink` as the current vNext pilot policy.
2. Run the same frozen policy on `flowers` and at least one harder indoor scene before full9.
3. Compare against clean MeshSplatting, v104c, v106, and Phase-J under one explicit table.
4. Add fixed/no-shrink/no-face-guard ablations.
5. Report model size, residual texture bytes, runtime overhead, triangle count, fallback rate, and per-view tail deltas.
6. Only promote vNext if multiple scenes accept nonzero textures and the mean gains become materially larger than the current garden micro-gain.

## Final Status

`NOT COMPLETE`.

This is a real vNext milestone, but still far from the top-conference closed loop.

# v327b Blend-Step Pairwise Guard Full9 Log

Date: 2026-07-01

## Question

This pass answers a narrow reflection question: did the recent failure analysis
actually change the method in a useful way, or was it still blind parameter
search?

Short answer: the reflection helped, but only in a limited way. It produced a
safer policy variant that avoids the v326/v327a overreach failure and gives a
small positive full9 delta over the v322C incumbent. It is not yet a paper-level
breakthrough.

## Implemented Change

`scripts/car_model/apply_source_heldout_support_transport_calibrator.py` now
adds `--pairwise_dominance_max_blend_step`.

The pairwise dominance policy compares a candidate variant against the current
incumbent. Earlier relaxed pairwise probes could jump too far in blend space,
for example from `fixed` to `mix0750`, based on weak source-heldout evidence.
That caused target overreach on treehill. v327b rejects any pairwise candidate
whose blend distance from the incumbent exceeds the configured maximum.

v327b uses:

```text
--policy_profile v322c_incumbent
--enable_pairwise_dominance_policy
--pairwise_dominance_enable_ood_guard
--pairwise_dominance_min_local_ssim_delta -0.001
--pairwise_dominance_min_local_min_delta -0.005
--pairwise_dominance_min_source_ssim_delta -0.0002
--pairwise_dominance_min_source_min_delta -0.005
--pairwise_dominance_max_blend_step 0.25
--enable_wandb
```

This is deliberately not a per-scene parameter set. The same fixed policy was
used for all scenes.

## Full9 Result Versus v322C

Audit file:

```text
docs/car_model/results/v327b_pairwise_blendstep_full9_vs_v322c_audit.json
```

Replay root:

```text
outputs/carnet/spcarnet_v327b_pairwise_blendstep_full9_20260701
```

Incumbent archive:

```text
outputs/carnet/spcarnet_v322c_baseknn_ladder_fixedmargin_full9_20260701
```

Full9 macro result:

| metric | v322C | v327b | delta |
|---|---:|---:|---:|
| selected PSNR gain mean | 0.271334337119 | 0.271425492910 | +0.000091155791 |
| selected SSIM gain mean | 0.003727241355 | 0.003728223728 | +0.000000982373 |
| selected PSNR mean | 25.411728563384 | 25.411819719175 | +0.000091155791 |
| selected SSIM mean | 0.840481245798 | 0.840482228171 | +0.000000982373 |

Per-scene delta versus v322C:

| scene | PSNR gain delta | SSIM gain delta | output mismatches |
|---|---:|---:|---:|
| bicycle | +0.000000000000 | +0.000000000000 | 0 |
| bonsai | +0.000000000000 | +0.000000000000 | 0 |
| counter | +0.000000000000 | +0.000000000000 | 0 |
| flowers | +0.000000000000 | +0.000000000000 | 0 |
| garden | +0.000000000000 | +0.000000000000 | 0 |
| kitchen | +0.000000000000 | +0.000000000000 | 0 |
| room | +0.000000000000 | +0.000000000000 | 0 |
| stump | +0.000000000000 | +0.000000000000 | 0 |
| treehill | +0.000820402117 | +0.000008841356 | 7 |

The improvement is real under the v322C replay audit, but it is almost entirely
from treehill. The other eight scenes preserve v322C exactly.

## Policy Behavior

Source-heldout pairwise decisions:

| scene | pairwise verdict | source non-incumbent choices |
|---|---|---:|
| bicycle | selected | 1 |
| bonsai | accepted no source views | 0 |
| counter | accepted no source views | 0 |
| flowers | accepted no source views | 0 |
| garden | accepted no source views | 0 |
| kitchen | accepted no source views | 0 |
| room | did not clear source PSNR delta | 1 |
| stump | did not clear source PSNR delta | 2 |
| treehill | selected | 2 |

The guard is conservative. It only changes target outputs on treehill in the
full9 audit, and it leaves the other scenes identical to v322C.

## W&B Offline Runs

The focused and full9 runs were logged offline.

Focused4:

```text
outputs/carnet/spcarnet_v327b_pairwise_blendstep_focused4_20260701/bicycle/wandb/offline-run-20260701_020132-2mhfbfoq
outputs/carnet/spcarnet_v327b_pairwise_blendstep_focused4_20260701/room/wandb/offline-run-20260701_020230-uonvdx49
outputs/carnet/spcarnet_v327b_pairwise_blendstep_focused4_20260701/stump/wandb/offline-run-20260701_020308-wz0dkmnw
outputs/carnet/spcarnet_v327b_pairwise_blendstep_treehill_20260701/wandb/offline-run-20260701_015952-tofeq09v
```

Full9补齐场景:

```text
outputs/carnet/spcarnet_v327b_pairwise_blendstep_full9_20260701/bonsai/wandb/offline-run-20260701_020748-guv7oao7
outputs/carnet/spcarnet_v327b_pairwise_blendstep_full9_20260701/counter/wandb/offline-run-20260701_020911-oaupa6k5
outputs/carnet/spcarnet_v327b_pairwise_blendstep_full9_20260701/flowers/wandb/offline-run-20260701_020653-tb3b3v8c
outputs/carnet/spcarnet_v327b_pairwise_blendstep_full9_20260701/garden/wandb/offline-run-20260701_020748-93di6ci7
outputs/carnet/spcarnet_v327b_pairwise_blendstep_full9_20260701/kitchen/wandb/offline-run-20260701_021046-c750ophv
```

## Execution Note

An initial zsh command failed because `SCENES="bonsai counter kitchen"` was
treated as one path by zsh in this context:

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/bonsai counter kitchen/...
```

The run was restarted with explicit loop items:

```text
for scene in bonsai counter kitchen; do
  ...
done
```

No valid scene result was produced by the failed command.

## Interpretation

The reflection did change the method in a meaningful engineering sense:

- v326 showed that pairwise dominance can overreach when source evidence accepts
  no useful alternatives.
- v326b added a zero-accept safety guard and exactly replayed v322C.
- v327a showed that simply relaxing source thresholds reintroduces overreach.
- v327b added a representation-aware blend-step bound and recovered a small
  positive treehill gain while preserving the other scenes.

However, the current method is still far from the desired paper endpoint:

- the full9 macro gain over v322C is tiny;
- only one scene changes;
- this does not address the deeper representation bottleneck;
- visual differences are expected to remain subtle;
- it cannot be claimed as a broad or decisive win over MeshSplatting.

## Verdict

Final status: NOT COMPLETE.

The next high-value step is not another threshold scan. The method needs a
stronger target-blind residual reliability model or representation update that
can safely unlock more of the oracle gap on multiple scenes, while preserving
the v322C/v327b replay audit discipline.

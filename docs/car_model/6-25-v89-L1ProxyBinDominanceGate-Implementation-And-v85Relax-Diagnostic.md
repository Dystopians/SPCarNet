# v89 L1-Proxy Bin-Dominance Gate and v85 Relaxed Tail-Risk Diagnostic

Date: 2026-06-25

## Purpose

This log records two linked updates:

1. `v85_tailrisk_relax075_counter_rerun_20260625` finished and is archived as a diagnostic, but it does **not** beat the v84/v86 representation anchor.
2. A new v89 mechanism was implemented to address the observed bottleneck: prior-bin hybrid selection used residual-MSE gain, while promotion is decided by image-level SSIM/L1/tail risk.

## v85 Relaxed Tail-Risk Result

Archived evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_tailrisk_relax075_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_tailrisk_relax075_counter_20260625/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_tailrisk_relax075_counter_20260625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v85_tailrisk_relax075_counter_20260625/logs/apply_metrics_counter.log
```

Counter result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 counter anchor | `26.7561378479` | `0.8621263504` | `0.2516906559` |
| v85 relaxed tail-risk rerun | `26.7561340332` | `0.8621262312` | `0.2516913712` |
| delta vs v84/v86 | `-0.0000038147` | `-0.0000001192` | `+0.0000007153` |

Policy audit:

| field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| target changed fraction | `0.0639013177` |
| selected hybrid | `true` |
| allowed hybrid bins | `1548` |
| policy-val SSIM gain | `0.0002939055` |
| policy-val image-L1 gain | `0.0000267716` |
| policy-val image-L1 min-view gain | `-0.0000008121` |
| policy-val image-L1 CVaR20 gain | `0.0000026797` |

Verdict:

```text
Do not promote v85 relaxed tail-risk to hard-triad or full9.
```

The run is useful because it confirms the bottleneck: the candidate can pass risk gates and make a non-trivial target edit, but its SSIM/L1 fields are still just below the v84/v86 anchor.

## v89 Mechanism

Implemented files:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New mechanism:

```text
anchor-aware L1-proxy bin-dominance hybrid gate
```

Before v89, `build_policy_val_prior_bin_gain_hybrid_atlas` copied a prior atlas bin into the hybrid atlas when that bin reduced policy-val residual MSE and passed support/tail-risk filters. That is not fully aligned with the final promotion gate, which judges whole-image SSIM/L1/tail behavior.

v89 adds an optional local RGB-L1 proxy gate at the bin level. For every changed policy-val sample, the hybrid builder now also computes:

```text
baseline_l1_after = mean(abs(residual - baseline_delta))
prior_l1_after    = mean(abs(residual - prior_delta))
l1_gain           = baseline_l1_after - prior_l1_after
```

A bin can be copied only if it satisfies both the existing residual-MSE gate and, when enabled, the new L1-proxy gate:

```text
l1_abs_gain >= threshold
l1_relative_gain >= threshold
l1_positive_view_fraction >= threshold
l1_min_view_gain >= threshold
l1_cvar20_view_gain >= threshold
prior_l1_after < baseline_l1_after
```

New adapter flags:

```text
--enable_prior_bin_gain_hybrid_l1_proxy_gate
--prior_bin_gain_hybrid_min_l1_abs_gain
--prior_bin_gain_hybrid_min_l1_relative_gain
--prior_bin_gain_hybrid_min_l1_positive_view_fraction
--prior_bin_gain_hybrid_min_l1_min_view_gain
--prior_bin_gain_hybrid_min_l1_cvar20_view_gain
```

The wrapper forwards the same flags through `run_l1risk_fairnoop_scene.py` and records them in the run config.

## Recommended v89 First Test

Do not grid-search per scene. Start with a single fixed counter probe:

```text
teacher_distilled_basis_mode=none
texture_size_candidates=32
support_expansion_max_extra_faces_candidates=4096,8192
target_support_prerank_top_k=1
surface_multiscale_prior_blend_candidates=0,0.5,1.0
enable_prior_bin_gain_hybrid_l1_proxy_gate=true
prior_bin_gain_hybrid_min_l1_positive_view_fraction=0.9
prior_bin_gain_hybrid_min_l1_min_view_gain=-0.0000008
prior_bin_gain_hybrid_min_l1_cvar20_view_gain=0.0
```

Promotion gate for counter:

```text
PSNR > 26.7561378479
SSIM > 0.8621263504
LPIPS < 0.2516906559
accepted_atlas
target changed fraction >= 0.001
policy-val image-L1 and SSIM not weaker than v84/v86 audit
```

Only if this fixed counter probe passes should we run hard-triad (`counter,kitchen,bonsai`) and then full9.

## Run Status

The first launch root below is a command-interface diagnostic, not a method result:

```text
/dev/shm/peilincai_spcarnet_v89_l1proxy_counter_20260625/
```

It failed before training because negative scientific notation in a new wrapper-forwarded threshold was parsed by `argparse` as a missing option value. The wrapper now forwards the new L1-proxy gate floats through fixed decimal formatting.

The fixed counter smoke was:

```text
/dev/shm/peilincai_spcarnet_v89b_l1proxy_counter_20260625/
W&B: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/xugf4qc4
```

Launch scope:

```text
scene=counter
gpu=5
teacher_distilled_basis_mode=none
texture_size_candidates=32
support_expansion_max_extra_faces_candidates=4096,8192
target_support_prerank_top_k=1
enable_prior_bin_gain_hybrid_l1_proxy_gate=true
prior_bin_gain_hybrid_min_l1_positive_view_fraction=0.9
prior_bin_gain_hybrid_min_l1_min_view_gain=-0.0000008
prior_bin_gain_hybrid_min_l1_cvar20_view_gain=0.0
```

This probe is now finished; the final held-out result is summarized below. The promotion gate remains strict and is not satisfied.

## v89b Counter Result

Archived evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v89b_l1proxy_counter_20260625/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v89b_l1proxy_counter_20260625/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v89b_l1proxy_counter_20260625/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v89b_l1proxy_counter_20260625/apply_metrics_counter.log
```

Counter result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v84/v86 counter anchor | `26.7561378479` | `0.8621263504` | `0.2516906559` |
| v89b L1-proxy bin-dominance | `26.7561397552` | `0.8621263504` | `0.2516907156` |
| delta vs v84/v86 | `+0.0000019073` | `+0.0000000000` | `+0.0000000597` |

Policy audit:

| field | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| target changed fraction | `0.0641776589` |
| selected hybrid | `true` |
| allowed hybrid bins | `1505` |
| L1-proxy candidate bins | `428460` |
| L1-proxy rejected bins | `258249` |
| policy-val SSIM gain | `0.0002948691` |
| policy-val image-L1 gain | `0.0000269436` |
| policy-val image-L1 min-view gain | `-0.0000008009` |
| policy-val image-L1 CVaR20 gain | `0.0000026617` |

Verdict:

```text
Do not promote v89b to hard-triad or full9.
```

The L1-proxy bin-dominance gate produces a non-empty accepted edit and a tiny PSNR improvement over the v84/v86 counter anchor. However, it fails the strict promotion gate because LPIPS is slightly worse (`+5.97e-8`, lower is better) and SSIM is only tied at printed precision. This is useful as a positive diagnostic for the gate but not a paper-facing endpoint.

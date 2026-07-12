# v82b Capacity Pre-Rank + Face-Alpha Counter Probe Log

Date: 2026-06-24

## Purpose

v82b tested whether target-support pre-ranking plus policy-val face-alpha calibration can recover a stricter fixed representation-level counter result from the v56/v64/v79 anchor.

This is a real adapter/runner train-eval pipeline probe, not a README-only analysis. W&B online logging was enabled.

## Command And Run

W&B run: `0a5ueh2u`

Output root:

```text
/dev/shm/peilincai_spcarnet_v82_capacity_prerank_facealpha_20260624
```

Persistent small artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v82_capacity_prerank_facealpha_20260624/counter/apply_metrics_counter.log
```

## Result

Current counter anchor:

```text
v56/v64/v79: 26.756130219 / 0.862126231 / 0.251691371
```

v82b result:

```text
PSNR  26.756137848
SSIM   0.862126350
LPIPS  0.251690656
```

Delta vs anchor:

```text
dPSNR  +0.000007629
dSSIM  +0.000000119
dLPIPS -0.000000715
```

The selected candidate was:

```text
1/2 support=fit_residual_topk_8192 added=4106 faces=5680 texture=32 fill=nearest_observed prior_blend=1 cap=0.12
selected_alpha=0.5
```

## Verdict

Counter-level strict micro-win, not yet a promoted full method.

This is the first post-v80 probe in this batch that strictly beats the v56/v64/v79 anchor on all three counter RGB metrics. The margin is extremely small, so it should be treated as a next-stage validation seed rather than a paper headline. The required next step is a fair full9 expansion or at least hard-triad expansion with the same fixed policy, followed by qualitative and per-view audit.

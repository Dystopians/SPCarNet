# v106 POD-MoE Base-Preserve Hard-Triad Log

Date: 2026-06-25

This log records the current v106 POD-MoE milestone and the follow-up hard-triad validation. It is intentionally conservative: counter now passes the local v104c anchor, but the method is not promoted to a paper-final headline until multi-scene evidence is complete.

## Fixed Policy

- Field variant: `pod_moe`
- Basis: `affine_barycentric_viewdir_pod_mixture`
- Builder: `v106_perceptual_occlusion_detail_moe`
- Base: v104c-like shrink view-affine residual field
- Experts: `detail`, `occlusion_boundary`
- Certificate: `weighted_normal_equation_lambda_star`
- Boundary mode: `base_preserving_boundary`
- Residual dtype: `float16`
- Renderer scaling: `4`
- Ridge: `0.001`
- Residual clip: `0.08`
- View std floor: `1e-4`
- Rank rtol: `1e-7`
- Condition max: `1e8`
- Gate source: `normal_equation`
- Chunk pixels: `262144`

## Counter Result

Run root:

- Field: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_field`
- Report: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports`
- Log: `/tmp/spcarnet_logs/v106_podmoe_basepreserve_counter.log`

Summary:

- JSON: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_counter_summary.json`
- Markdown: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_counter_summary.md`
- Delta-MSE JSON: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_vs_v104c_delta_mse.json`
- Delta-MSE Markdown: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_vs_v104c_delta_mse.md`

Counter metrics:

| method | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |
|---|---:|---:|---:|---:|---:|---:|
| clean MeshSplatting | 26.751774 | 0.862055 | 0.252003 | - | - | - |
| v104c shrink view-affine | 27.498068 | 0.867420 | 0.238986 | 0.000000 | 0.000000 | 0.000000 |
| v106 POD-MoE old | 27.480730 | 0.867727 | 0.238923 | -0.017338 | +0.000308 | -0.000064 |
| v106 POD-MoE debtguard | 27.486620 | 0.867725 | 0.238849 | -0.011448 | +0.000305 | -0.000137 |
| v106 POD-MoE cert | 27.486565 | 0.867725 | 0.238849 | -0.011503 | +0.000305 | -0.000137 |
| v106 POD-MoE base-preserve | 27.499645 | 0.867521 | 0.238847 | +0.001577 | +0.000102 | -0.000139 |

Counter MSE-direction diagnostic versus v104c:

| candidate | views | MSE-improved | MSE-worse | mean delta-MSE | mean 2ed | mean d2 | mean abs delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| old POD-MoE | 30 | 4 | 26 | +0.00000711 | +0.00000342 | 0.00000369 | 0.00047899 |
| debtguard POD-MoE | 30 | 4 | 26 | +0.00000479 | +0.00000283 | 0.00000196 | 0.00031411 |
| cert POD-MoE | 30 | 4 | 26 | +0.00000481 | +0.00000284 | 0.00000197 | 0.00031485 |
| base-preserve POD-MoE | 30 | 23 | 7 | -0.00000026 | -0.00000068 | 0.00000042 | 0.00006754 |

Interpretation:

- The base-preserving boundary mode is the first v106 POD-MoE variant that strictly beats v104c on counter across PSNR, SSIM, and LPIPS.
- The improvement is small in PSNR, so it is not enough by itself for a paper-final claim.
- The MSE decomposition is materially healthier than old/debtguard/cert: most views now improve MSE, and the cross term turns negative on average.
- The remaining worst MSE view is `00009.png`, which still has a positive delta-MSE of about `+0.00001759`; this remains a qualitative/error-map inspection target.

## Hard-Triad Runs

These runs use the same fixed policy as counter. They are launched to test whether base-preserving POD-MoE generalizes beyond the counter scene.

Kitchen command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  --scene kitchen \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --field_root /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_field \
  --report_root /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports \
  --v102_report_root /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v102_reports \
  --output_method ours_26000_v106_podmoe_basepreserve_kitchen \
  --field_variant pod_moe \
  --gate_source normal_equation \
  --renderer_scaling 4 --residual_dtype float16 \
  --ridge 0.001 --residual_clip 0.08 --view_std_floor 1e-4 \
  --rank_rtol 1e-7 --condition_max 1e8 --gate_boost 0.5 \
  --view_gate_temperature 0.0 --chunk_pixels 262144 --gpu 2 \
  --force_field --force_render --force_eval
```

Bonsai command:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  --scene bonsai \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --field_root /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_field \
  --report_root /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports \
  --v102_report_root /dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v102_reports \
  --output_method ours_26000_v106_podmoe_basepreserve_bonsai \
  --field_variant pod_moe \
  --gate_source normal_equation \
  --renderer_scaling 4 --residual_dtype float16 \
  --ridge 0.001 --residual_clip 0.08 --view_std_floor 1e-4 \
  --rank_rtol 1e-7 --condition_max 1e8 --gate_boost 0.5 \
  --view_gate_temperature 0.0 --chunk_pixels 262144 --gpu 3 \
  --force_field --force_render --force_eval
```

Status:

- Kitchen session completed on GPU 2, log `/tmp/spcarnet_logs/v106_podmoe_basepreserve_kitchen.log`.
- Bonsai session completed on GPU 3, log `/tmp/spcarnet_logs/v106_podmoe_basepreserve_bonsai.log`.
- Summary: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v106_podmoe_basepreserve_hardtriad_kitchen_bonsai_summary.json`
- Comparison table: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_compare_20260625.md`
- Kitchen delta-MSE: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v106_podmoe_basepreserve_kitchen_vs_v104c_delta_mse.md`
- Bonsai delta-MSE: `/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v106_podmoe_basepreserve_bonsai_vs_v104c_delta_mse.md`

Hard-triad metrics:

| scene | v104c PSNR | v106 PSNR | dPSNR | v104c SSIM | v106 SSIM | dSSIM | v104c LPIPS | v106 LPIPS | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | 27.498068 | 27.499645 | +0.001577 | 0.867420 | 0.867521 | +0.000102 | 0.238986 | 0.238847 | -0.000139 |
| kitchen | 28.770449 | 28.772043 | +0.001595 | 0.881590 | 0.881652 | +0.000062 | 0.188021 | 0.187815 | -0.000206 |
| bonsai | 30.310877 | 30.316090 | +0.005213 | 0.907367 | 0.907520 | +0.000154 | 0.230186 | 0.230050 | -0.000136 |
| mean | - | - | +0.002795 | - | - | +0.000106 | - | - | -0.000160 |

Hard-triad MSE-direction diagnostics:

| scene | views | MSE-improved | MSE-worse | mean delta-MSE | mean 2ed | mean d2 | mean abs delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| counter | 30 | 23 | 7 | -0.00000026 | -0.00000068 | 0.00000042 | 0.00006754 |
| kitchen | 35 | 30 | 5 | -0.00000050 | -0.00000100 | 0.00000050 | 0.00005965 |
| bonsai | 37 | 36 | 1 | -0.00000017 | -0.00000075 | 0.00000058 | 0.00009093 |

Interpretation:

- Base-preserve passes the hard-triad fixed-policy gate against v104c on all three reported image metrics.
- The mean metric gains are small, but all three scenes have consistent signs.
- The MSE-direction diagnostic supports the claim that this is not just metric noise: `89 / 102` hard-triad views improve MSE over v104c.
- Remaining qualitative risks are concentrated in the worst views: counter `00009.png`, kitchen `00015.png`, and bonsai `00028.png`.

## Next Promotion Gate

v106 base-preserve can only replace v104c as the active method if full9 fixed-policy validation remains favorable.

Minimum hard-triad gate:

- Counter remains strictly better than v104c on PSNR, SSIM, LPIPS: passed.
- Kitchen and bonsai must not reveal systematic PSNR regression: passed.
- Delta-MSE diagnostics should show improved or at least non-worse MSE direction on the majority of views: passed.

Remaining full9 gate:

- Run fixed-policy v106 base-preserve on `bicycle`, `flowers`, `garden`, `room`, `stump`, and `treehill`.
- Summarize all nine scenes with the same comparison helper.
- If any scene regresses, diagnose whether the failure is a view-specific outlier or a scene-class failure.

If full9 fails, the next mechanism is `v107_mse_descent_locked_podmoe`: solve a per-triangle two-expert box QP on the normal-equation proxy so the expert ray is MSE-descent aligned relative to the v104c-like base. This is a mechanism-level repair, not another parameter scan.

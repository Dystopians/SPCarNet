# v107 Cross-Fitted POD-MoE Reliability Probe Log

Date: 2026-06-25

This log tracks the v107 cross-fitted reliability ablation for POD-MoE. The current run is a probe, not a promoted method. v107 is not yet validated, has not produced a complete metric set in this document, and must not be described as better than v106 until the result and evidence sections below are filled from completed runs.

## Known Background

The current validated reference is v106 full9 strict mean. LPIPS is lower-is-better.

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| clean MeshSplatting | 25.151682 | 0.749018 | 0.287621 | reference baseline |
| v104c | 25.829099 | 0.760727 | 0.268548 | prior strong anchor |
| v106 | 25.831280 | 0.760830 | 0.268435 | current validated POD-MoE anchor |

Known v106 delta versus v104c:

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v106 - v104c | +0.002181 | +0.000103 | -0.000112 |

Interpretation boundary:

- v106 is the current reference point for any v107 comparison.
- v107 is a reliability ablation, not a confirmed upgrade.
- Any v107 claim must be made only after per-scene metrics, identity checks, and report artifacts are collected.

## v107 Probe Definition

Target:

- Experiment version: `v107`
- Purpose: cross-fitted reliability ablation for POD-MoE expert reliability
- Initial scenes: `counter`, `flowers`, `garden`, `bonsai`
- Field variant: `pod_moe`
- Gate source: `crossfit_risk`
- Method version expected from launcher: `v107_crossfit_pod_moe_expert_reliability`
- POD reliability expected from field stats: `v107_crossfit_heldout_weighted_risk`
- Cross-fit split expected from field stats: `target_view_even_odd`

Run roots:

```text
REPORT_ROOT=/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports
FIELD_ROOT=/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_field
```

Fixed policy inherited from v106 unless explicitly changed by `--gate_source crossfit_risk`:

| setting | value |
|---|---|
| package root | `/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625` |
| v102 bank root | `/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625` |
| clean root | `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k` |
| endpoint method | `ours_26000_v100_checkpoint_attached_ela_endpoint` |
| iteration | `26000` |
| renderer scaling | `4` |
| residual dtype | `float16` |
| ridge | `0.001` |
| residual clip | `0.08` |
| view std floor | `1e-4` |
| rank rtol | `1e-7` |
| condition max | `1e8` |
| gate boost | `0.5` |
| view gate temperature | `0.0` |
| POD view gate mode | `temperature_controlled` for v107 cross-fit POD-MoE; legacy v106 field defaults stay `implicit_unit_temperature` for compatibility |
| chunk pixels | `262144` |

## Mainline Patch Notes Pending Validation

These patches have landed on the mainline v107 builder path, but they are implementation fixes awaiting fresh probe evidence. They do not establish any metric improvement over v106.

| patch | expected behavior | validation status |
|---|---|---|
| A. POD-crossfit statistic accounting | The v107 POD-crossfit builder should avoid repeatedly accumulating the full detail and boundary statistics while assembling cross-fit reliability. Full detail and boundary counts/means should now reflect one accounting pass rather than duplicated accumulation. | pending; verify from fresh field stats, per-scene report JSON, and summary CSV after the probe reruns. |
| B. POD view gate mode | `pod_view_gate_mode=temperature_controlled` should make `view_gate_temperature=0.0` truly skip the POD view gate in v107 cross-fit fields. Old v106 field default behavior remains `implicit_unit_temperature` for compatibility. | pending; verify `field_stats.pod_view_gate_mode`, manifest, or solve stats report `temperature_controlled`, and confirm metrics come from fields rebuilt after this patch. |
| C. Render report audit fields | `render.py` now records `method_version`, reliability variant, MSE certificate, expert combine mode, and cross-fit split inside `surface_residual_field`. Future v107 runner reports check render-side `method_version` strictly for crossfit artifacts. | smoke/compile passed; full v107b scene evidence still pending because active parent processes were launched before this runner-side strict-check patch. |
| D. POD view-gate CPU smoke | Added `scripts/car_model/smoke_test_pod_view_gate_modes.py` to assert that `temperature_controlled + view_gate_temperature=0.0` keeps POD delta, while missing legacy mode behaves like `implicit_unit_temperature`. | passed: `temperature_controlled_vgt0_gain=0.05000000`, `implicit_gain=0.00000000`, `legacy_missing_gain=0.00000000`. |

Validation boundary:

- v107 currently has no complete validated metric table in this log.
- Do not claim v107 is better than v106 until all probe scenes have completed metrics, identity checks, patch-specific diagnostics, and explicit v107-v106 deltas.
- Because the artifact filename records `view_gate_temperature` but not the semantic gate mode, the manifest/report field stats are the source of truth for `pod_view_gate_mode`.

## Running Command Template

Use this template for each currently probed scene. Set `SCENE` to one of `counter`, `flowers`, `garden`, or `bonsai`, and set `GPU` to the device used by the actual session.

```bash
SCENE=counter
GPU=0
REPORT_ROOT=/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports
FIELD_ROOT=/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_field
LOG=${REPORT_ROOT}/logs/${SCENE}_gpu${GPU}_run.log

CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  --scene ${SCENE} \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --field_root ${FIELD_ROOT} \
  --report_root ${REPORT_ROOT} \
  --v102_report_root ${REPORT_ROOT}/v102_reports \
  --output_method ours_26000_v107_podmoe_crossfit_tc_${SCENE} \
  --field_variant pod_moe \
  --gate_source crossfit_risk \
  --renderer_scaling 4 --residual_dtype float16 \
  --ridge 0.001 --residual_clip 0.08 --view_std_floor 1e-4 \
  --rank_rtol 1e-7 --condition_max 1e8 --gate_boost 0.5 \
  --view_gate_temperature 0.0 --chunk_pixels 262144 --gpu ${GPU} \
  --force_field --force_render --force_eval \
  > ${LOG} 2>&1
```

Expected default output method if `--output_method` is not supplied:

```text
ours_26000_v107_crossfit_pod_moe_expert_reliability_surface_field_${SCENE}
```

Actual v107b probe output method:

```text
ours_26000_v107_podmoe_crossfit_tc_${SCENE}
```

Expected per-scene report files:

```text
${REPORT_ROOT}/${SCENE}/${SCENE}_v107_crossfit_pod_moe_report.json
${REPORT_ROOT}/${SCENE}/${SCENE}_v107_crossfit_pod_moe_report.md
${REPORT_ROOT}/${SCENE}/${SCENE}_field.log
${REPORT_ROOT}/${SCENE}/${SCENE}_render.log
${REPORT_ROOT}/${SCENE}/${SCENE}_eval.log
```

Expected field artifact pattern:

```text
${FIELD_ROOT}/${SCENE}/v107_podmoe_crossfit_mc1_mv1_r1em03_clip8em02_vs1em04_rr1em07_cm1ep08_gb5em01_crossfit_risk_vgt0ep00_float16_s4_field.pt
${FIELD_ROOT}/${SCENE}/v107_podmoe_crossfit_mc1_mv1_r1em03_clip8em02_vs1em04_rr1em07_cm1ep08_gb5em01_crossfit_risk_vgt0ep00_float16_s4_field.manifest.json
```

## Summary Command After Probe Completion

Run this only after the four per-scene reports are present.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/summarize_v105_evidence_gated_mixture.py \
  --root /dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports \
  --scenes counter flowers garden bonsai \
  --out_dir /dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports \
  --prefix v107b_podmoe_crossfit_tc_probe_counter_flowers_garden_bonsai_summary
```

Expected summary outputs:

```text
/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_counter_flowers_garden_bonsai_summary.json
/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_counter_flowers_garden_bonsai_summary.csv
/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_counter_flowers_garden_bonsai_summary.md
```

## v107-v106 Comparison Command After Probe Completion

Run this after the four v107 per-scene reports are present. It compares v107 directly against the validated v106 full9 assembly, not only against v104c.

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/compare_v107_probe_to_v106.py \
  --v107_root /dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports \
  --v106_assembled docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.json \
  --scenes counter flowers garden bonsai \
  --out_dir /dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports \
  --prefix v107b_podmoe_crossfit_tc_probe_vs_v106
```

Expected comparison outputs:

```text
/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_vs_v106.json
/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_vs_v106.csv
/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_vs_v106.md
```

## Result Table Placeholder

Fill this table after the four runs complete. Do not infer missing cells from partial output.

| scene | status | clean PSNR | clean SSIM | clean LPIPS | v104c PSNR | v104c SSIM | v104c LPIPS | v106 PSNR | v106 SSIM | v106 LPIPS | v107 PSNR | v107 SSIM | v107 LPIPS | dPSNR v107-v104c | dSSIM v107-v104c | dLPIPS v107-v104c | dPSNR v107-v106 | dSSIM v107-v106 | dLPIPS v107-v106 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | complete / not promoted | 26.751774 | 0.862055 | 0.252003 | 27.498068 | 0.867420 | 0.238986 | 27.499645 | 0.867521 | 0.238847 | 27.496471 | 0.867417 | 0.238981 | -0.001596 | -0.000002 | -0.000005 | -0.003174 | -0.000104 | +0.000134 |
| flowers | complete / not promoted | 19.682257 | 0.511822 | 0.394563 | 20.075844 | 0.531076 | 0.374473 | 20.077723 | 0.531240 | 0.374393 | 20.075916 | 0.531084 | 0.374471 | +0.000072 | +0.000007 | -0.000001 | -0.001806 | -0.000156 | +0.000079 |
| garden | complete / not promoted | 25.029211 | 0.780035 | 0.201315 | 25.788094 | 0.799263 | 0.174584 | 25.790945 | 0.799382 | 0.174480 | 25.788160 | 0.799266 | 0.174580 | +0.000067 | +0.000003 | -0.000004 | -0.002785 | -0.000116 | +0.000099 |
| bonsai | complete / not promoted | 28.895233 | 0.896400 | 0.259493 | 30.310877 | 0.907367 | 0.230186 | 30.316090 | 0.907520 | 0.230050 | 30.309280 | 0.907362 | 0.230155 | -0.001596 | -0.000005 | -0.000031 | -0.006809 | -0.000158 | +0.000105 |
| mean over completed probe scenes | complete / not promoted | 25.089619 | 0.762578 | 0.276843 | 25.918221 | 0.776281 | 0.254557 | 25.921101 | 0.776416 | 0.254443 | 25.917454 | 0.776282 | 0.254547 | -0.000766 | +0.000001 | -0.000010 | -0.003646 | -0.000134 | +0.000104 |

Field diagnostics to fill from the per-scene report JSON or summary CSV:

| scene | passed | method_version | gate_source | POD view gate mode | expert reliability | cross-fit split | full stat accounting check | valid triangles | detail triangles | boundary triangles | detail reliability mean | boundary reliability mean | detail crossfit gain mean | boundary crossfit gain mean | mean abs delta |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | true | `v107_crossfit_pod_moe_expert_reliability` | `crossfit_risk` | `temperature_controlled` | `v107_crossfit_heldout_weighted_risk` | `target_view_even_odd` | fresh stats present | 2716422 | 18683 | 7408 | 0.080899 | 0.038173 | 0.021146 | 0.000070 | pending strict rerun |
| flowers | true | `v107_crossfit_pod_moe_expert_reliability` | `crossfit_risk` | `temperature_controlled` | `v107_crossfit_heldout_weighted_risk` | `target_view_even_odd` | fresh stats present | 1853843 | 21963 | 10554 | 0.099256 | 0.032760 | 0.046721 | 0.000102 | 0.014429 |
| garden | true | `v107_crossfit_pod_moe_expert_reliability` | `crossfit_risk` | `temperature_controlled` | `v107_crossfit_heldout_weighted_risk` | `target_view_even_odd` | fresh stats present | 3311206 | 10538 | 3668 | 0.114583 | 0.069121 | 0.051936 | 0.000084 | 0.011178 |
| bonsai | true | `v107_crossfit_pod_moe_expert_reliability` | `crossfit_risk` | `temperature_controlled` | `v107_crossfit_heldout_weighted_risk` | `target_view_even_odd` | fresh stats present | 3405888 | 30621 | 10088 | 0.095192 | 0.040912 | 0.025121 | 0.000078 | pending strict rerun |
| mean over completed probe scenes | true |  | `crossfit_risk` | `temperature_controlled` | `v107_crossfit_heldout_weighted_risk` | `target_view_even_odd` | fresh stats present | 2778340 | 26736 | 7929 | 0.097482 | 0.045242 | 0.036231 | 0.000083 |  |

## Evidence To Fill After Completion

For each of `counter`, `flowers`, `garden`, and `bonsai`, collect:

- Process log path from `${REPORT_ROOT}/logs/${SCENE}_gpu${GPU}_run.log`.
- Per-scene report JSON and Markdown under `${REPORT_ROOT}/${SCENE}/`.
- Field `.pt` and `.manifest.json` under `${FIELD_ROOT}/${SCENE}/`.
- `return_codes` from the report JSON: `v102`, `field`, `render`, and `eval`.
- `passed` value from the report JSON.
- `field_identity.checks` and `render_stats.identity_checks`; all required checks should be true before considering the scene valid.
- `metrics`, `clean_metrics`, `v104c_metrics`, and `deltas.vs_v104c`.
- v106 metrics for the same scenes from the current v106 full9 report or source table, so v107-v106 deltas can be filled explicitly.
- `field_stats.method_version`, `pod_view_gate_mode`, `pod_expert_reliability_variant`, `expert_reliability_variant`, `expert_reliability_combine`, `expert_mse_certificate`, and `pod_crossfit_split`.
- `detail_crossfit_supported_triangles`, `boundary_crossfit_supported_triangles`, `detail_crossfit_gain_mean`, `boundary_crossfit_gain_mean`, `detail_crossfit_mse_scale_mean`, and `boundary_crossfit_mse_scale_mean`.
- Patch-specific checks: confirm full detail/boundary statistics are not duplicated, and confirm `view_gate_temperature=0.0` with `pod_view_gate_mode=temperature_controlled` skips the POD view gate for v107 artifacts.
- Summary JSON/CSV/Markdown paths from the summary command above.
- Any scene-specific failure reason, missing artifact, or identity-check mismatch.

Minimum interpretation rules:

- If any scene is missing, failed, or has false identity checks, report v107 as incomplete.
- If v107 improves one metric but regresses another, report the mixed result directly.
- If v107 is worse than v106 on the probe mean or on material individual scenes, do not promote it.
- If v107 appears better than v106 on all four probe scenes, still label it as a probe result until the remaining full9 scenes are evaluated.
- If the patch-specific diagnostics are missing or were produced from stale pre-patch artifacts, report v107 as unverified even if metric files exist.

## Current Status

Status update after the first completed probe scenes:

- `flowers` completed with `passed=true`, all subprocess return codes `0`, and fresh render identity fields for `method_version`, `pod_view_gate_mode`, `pod_crossfit_split`, reliability variant, MSE certificate, and field hash.
- `flowers` is a negative result relative to v106: `-0.001806 PSNR`, `-0.000156 SSIM`, and `+0.000079 LPIPS`. It is only a tiny positive relative to v104c. Therefore v107b must not be promoted over v106 from the current evidence.
- `garden` completed, was rerun render/eval-only after the stricter render identity patch, and is also negative relative to v106: `-0.002785 PSNR`, `-0.000116 SSIM`, and `+0.000099 LPIPS`. It is only a tiny positive relative to v104c. The first two valid probe scenes therefore both argue against promoting v107b.
- `counter` completed and is also negative relative to v106: `-0.003174 PSNR`, `-0.000104 SSIM`, and `+0.000134 LPIPS`. It is also slightly below v104c on PSNR/SSIM, while LPIPS is marginally lower than v104c. A render/eval-only rerun is in progress to refresh the stricter render identity fields, but the metric direction is already negative versus v106.
- `bonsai` completed, was rerun render/eval-only after the stricter render identity patch, and is also negative relative to v106: `-0.006809 PSNR`, `-0.000158 SSIM`, and `+0.000105 LPIPS`. It is still clearly above clean, but below both v104c and v106 on PSNR/SSIM.
- The four-scene mean is `-0.003646 PSNR`, `-0.000134 SSIM`, and `+0.000104 LPIPS` versus v106. This makes v107b a completed negative reliability probe, not a promoted method.
- The partial comparison summary is `/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_podmoe_crossfit_tc_probe_vs_v106_partial.md`.
- The read-only field diagnostic summary is `/dev/shm/peilincai_spcarnet_v107b_podmoe_crossfit_tc_probe_20260625_reports/v107b_field_diagnostics_partial.md`.
- The counter/garden/bonsai render/eval-only strict refreshes have completed; all four scenes now have strict render identity fields.
- v107 has produced a complete four-scene metric table here and must not be claimed to outperform v106.
- Mainline patch A, POD-crossfit full detail/boundary statistic de-duplication, is documented but still pending evidence.
- Mainline patch B, `pod_view_gate_mode=temperature_controlled` with `view_gate_temperature=0.0` truly skipping the POD view gate for v107 while preserving v106 `implicit_unit_temperature` compatibility, has CPU smoke evidence and fresh `flowers` artifact evidence.
- Mainline patch C/D, render report audit fields plus POD view-gate smoke, has compile/smoke evidence and fresh `flowers` render evidence, but still needs fresh per-scene render reports for full probe validation.
- The first probe set is `counter`, `flowers`, `garden`, and `bonsai`.
- Active v107b probe commands were launched on GPU2/3/5/4 for `counter`/`flowers`/`garden`/`bonsai` with `--output_method ours_26000_v107_podmoe_crossfit_tc_${SCENE}` and `WANDB_MODE=offline`.
- Report root and field root are recorded above.
- Result and evidence tables remain partial until all runs complete.

Current diagnostic interpretation:

- Completed scenes are still better than the local clean baseline, so v107b is not a broken renderer.
- Completed scenes are worse than v106, so the cross-fit reliability change is a conservative regression.
- The likely mechanism is over-suppression of the boundary expert: `boundary_crossfit_gain_mean` is `0.000102` on `flowers`, `0.000084` on `garden`, `0.000070` on `counter`, and `0.000078` on `bonsai`, far smaller than detail crossfit gain. This suggests the `even_to_odd_and_odd_to_even_min_requires_both_splits` combine rule is too strict for sparse boundary evidence.
- The next method attempt is now implemented as v108 MSE-descent-locked POD-MoE. It keeps the POD-MoE representation but adds a joint two-expert weighted normal-equation box-QP descent certificate. See `docs/car_model/6-25-v108-MSE-Descent-Locked-PODMoE-Log.md`.

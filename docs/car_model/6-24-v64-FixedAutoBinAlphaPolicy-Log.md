# v64 Fixed Auto Bin-Alpha Policy Log

Date: 2026-06-24

Purpose: turn the v63b bin-level residual magnitude calibration probe into a fixed, train/policy-val driven auto policy over full9.

---

## Motivation

v63b produced a useful but incomplete result:

- `kitchen` strictly improved over v52/v60 on PSNR, SSIM, and LPIPS;
- `counter` remained worse than v52/v56/v60;
- several other scenes either rejected the residual atlas internally or had too little reliable bin support.

The next required step was to stop manual scene reasoning and build a fixed policy:

```text
if v63b has strong train/policy-val bin-alpha evidence:
    use v63b bin-alpha residual atlas
else:
    fallback to v56 selected policy
```

This makes the selection deployable and auditable. It does not use held-out metrics for scene selection.

---

## Implementation

New scripts:

```text
scripts/car_model/summarize_v64_bin_alpha_auto_policy.py
scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py
```

The summarizer reads:

- v56 selected full9 summary;
- v56 selected materialized tree;
- all v63b candidate audits/results;
- train/policy-val audit fields from each v63b candidate.

It writes:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/
```

The pipeline also validates the selected tree and builds:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/qualitative_gallery.html
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_selected_full9/v64_bin_alpha_auto_policy_pipeline_report.md
```

---

## Fixed Guard

v64 promotes v63b only if all conditions pass:

| Condition | Threshold |
|---|---:|
| accepted atlas | true |
| local alpha mode | `policy_val_bin_alpha` |
| bin-alpha count | `[32, 256]` |
| selected alpha | `[0.5, 1.0]` |
| policy-val relative gain | `>= 0.05` |
| policy-val SSIM gain | `>= 0.0003` |
| policy-val SSIM positive fraction | `>= 1.0` |
| policy-val image-L1 gain | `>= 0.00004` |
| policy-val image-L1 positive fraction | `>= 1.0` |
| policy-val image-L1 min-view gain | `>= 0.00001` |

Otherwise it falls back to v56.

This rejects the known `counter` failure without writing a scene name into the policy, and accepts `kitchen` because it has strong train/policy-val evidence.

---

## Full9 Candidate Execution

New v63b full9 candidates were run with W&B online logging:

| scene | W&B run | status |
|---|---|---|
| bicycle | `jvfx3s6s` | completed, v63b rejected by policy-val |
| flowers | `8e692zyt` | completed, v63b rejected by policy-val |
| garden | `xtsz2bry` | completed, v63b rejected by policy-val |
| stump | `lifyawln` | completed, v63b rejected by policy-val |
| treehill | `byyuaduj` | completed, v63b rejected by policy-val |
| room | `xaho7gyq` | completed, v63b accepted but rejected by v64 fixed guard |
| bonsai | `olt8riwt` | completed, v63b accepted but rejected by v64 fixed guard |
| counter | `rlctknlk` | completed earlier, rejected by v64 fixed guard |
| kitchen | `tyqm9u38` | completed earlier, selected by v64 |

All candidate results are under:

```text
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_full9_20260624/
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_counter_20260624/
/dev/shm/peilincai_spcarnet_v63b_bin_alpha_kitchen_20260624/
```

---

## v64 Full9 Result

Status: `FULL9_CANDIDATE_POLICY_EVALUATED_REPORT_ONLY`

Validation:

| Item | Value |
|---|---:|
| v63b candidate complete scenes | `9 / 9` |
| selected tree scene count | `9 / 9` |
| render/GT linked scenes | `9 / 9` |
| selection uses held-out metrics | `false` |

Aggregate:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |
| v64 vs v50 | 9 | 6 | 6 | +0.000991609 | +0.000016345 | -0.000059394 |

Per-scene selected sources:

| scene | selected source | guard |
|---|---|---:|
| bicycle | v56 fallback | 0 |
| flowers | v56 fallback | 0 |
| garden | v56 fallback | 0 |
| stump | v56 fallback | 0 |
| treehill | v56 fallback | 0 |
| room | v56 fallback | 0 |
| counter | v56 fallback | 0 |
| kitchen | v63b bin-alpha | 1 |
| bonsai | v56 fallback | 0 |

---

## Interpretation

v64 is a real engineering and method-policy milestone:

- it completes all full9 v63b candidate probes with W&B online logging;
- it fixes the v63b failure mode by making promotion automatic and conservative;
- it improves over v56 without regressing any scene by the selected summary metrics;
- it converts a scene-specific observation into a fixed auditable policy.

But the effect size is still very small:

- only `kitchen` is newly selected beyond v56;
- mean improvements are in the `1e-4` to `1e-6` range;
- this is not enough to claim the paper-level persistent residual representation is solved.

The correct claim is:

> v64 closes the fixed-policy engineering loop for bin-level residual magnitude calibration, but remains a report-only candidate. Phase-J is still the presentation-safe headline; v64 is the latest evidence that residual magnitude calibration is useful but not yet strong enough as a standalone paper endpoint.

---

## Commands

Full policy pipeline:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py
```

Static validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/summarize_v64_bin_alpha_auto_policy.py \
  scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py
```

---

## Next Step

The next meaningful research step is not another threshold tweak. It should be one of:

1. fresh blind/long-run validation of v64 to verify the fixed policy was not overfit to counter/kitchen probes;
2. teacher distillation from Phase-J render-time ELA into a stronger persistent surface field;
3. uncertainty-aware residual field learning that predicts magnitude and confidence jointly instead of estimating alpha from sparse bins only.

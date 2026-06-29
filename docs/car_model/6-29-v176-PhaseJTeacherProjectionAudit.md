# v176 Phase-J Teacher Residual Projection Audit

Date: 2026-06-29

## Verdict

- flower Phase-J all-axis gate target: `20.304358 / 0.557770 / 0.329222`
- teacher improves parent on policy-val: `True`
- any carrier projection improves MSE + SSIM + LPIPS on policy-val: `False`
- full9 allowed by this audit: `False`
- v175 strict ELA deferred branch: strict no-target-GT apply audit passed, but final PNG output is exact no-op and metrics equal the parent baseline.

## Teacher Signal

- parent policy-val: `20.516130 / 0.729221 / 0.145680`
- Phase-J teacher policy-val: `21.017230 / 0.745371 / 0.134851`
- teacher minus parent gain: `0.501100` PSNR, `0.016150` SSIM, `0.010829` LPIPS-improvement
- teacher-parent residual energy: `0.00252719`
- teacher/GT residual cosine: `0.320144`; sign agreement: `0.626625`

## Evidence Residual Audit

- raw teacher residual energy: `0.00201047`
- used teacher residual energy: `0.00123791`
- used/raw energy ratio: `0.615731`
- selected pixel fraction: `0.338489`
- teacher_better_mask fraction: `0.338489`

## Carrier Projection

| mode | pass | best alpha | MSE rel gain | SSIM gain | LPIPS gain | pos views | LPIPS pos views |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | False | 0.250000 | 0.037612 | -0.000021 | -0.000038 | 0.583 | 0.333 |
| low_rank_view_texture_rich | False | 0.250000 | 0.059065 | -0.000013 | -0.000020 | 0.667 | 0.333 |
| low_rank_view_texture_rich + edge_luma_mix target | False | 0.250000 | 0.056798 | -0.000014 | -0.000019 | 0.667 | 0.333 |

Both carriers reduce teacher-residual MSE on the train-policy-val samples, but the rendered-image metrics move in the wrong direction: SSIM and LPIPS gains are negative at the best MSE alpha. This is the concrete failure that blocks flowers exact and full9 under the v169 improved prompt.

The low-rank run uses a train-fit-only top residual face prefilter:

- eligible train-fit faces: `116659`
- selected candidate faces: `512`
- low-rank supported faces: `132 / 509`
- low-rank supported UV-bin fraction: `0.129084`
- mean retained low-rank energy: `0.884535`

This means the representation can model some teacher residual energy, but its support is too sparse and its image-space direction is not perceptually safe.

The `edge_luma_mix` residual target was tested as the prompt's patch/gradient-aware alternative. It did not reverse the perceptual failure: best-MSE alpha still improves residual MSE while lowering SSIM and worsening LPIPS. Therefore the current negative result is not only a raw-RGB target issue.

## v175 Strict ELA Negative

v175 tested the deferred/evidence-lumigraph route with `--strict_no_target_gt_apply`.

- strict no-target-GT apply: `True`
- target GT visible during apply: `False`
- target GT copied before apply: `False`
- target GT copied after apply: `True`, for eval only
- mean covered fraction: `0.642966`
- mean alpha: `0.0`
- PNG changed pixels vs parent: `0 / 22879296`
- test metrics: `19.668695 / 0.511678 / 0.394788`

Conclusion: v175 is useful as an I/O audit because it proves the strict apply path can avoid target GT leakage, but it is not a quality method. The adaptive alpha guard zeroed the effective output, so it cannot be promoted.

## Commands

v176 audited command:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/analyze_phasej_teacher_residual_projection.py \
  --compute_lpips \
  --projection_modes none,low_rank_view_texture_rich \
  --max_candidate_faces 512 \
  --max_candidate_face_samples_per_view 4096 \
  --max_samples_per_view 2048 \
  --max_policy_val_samples_per_view 2048 \
  --alpha_grid 0,0.0625,0.125,0.25,0.5 \
  --output_json /tmp/peilincai_spcarnet_v176_phasej_teacher_projection_audit.json \
  --output_md docs/car_model/6-29-v176-PhaseJTeacherProjectionAudit.md
```

v175 evaluated command:

```bash
CUDA_VISIBLE_DEVICES=1 /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/evaluate_render_split_metrics.py \
  --model_path /tmp/peilincai_spcarnet_20260629_v175_strict_ela_flowers/compact_model \
  --split test \
  --methods ours_26000_v175_strict_ela_flowers \
  --output /tmp/peilincai_spcarnet_20260629_v175_strict_ela_flowers/flowers_v175_strict_ela_test_results.json \
  --per_view_output /tmp/peilincai_spcarnet_20260629_v175_strict_ela_flowers/flowers_v175_strict_ela_test_per_view.json
```

## Current Blocking Status

The long-running v168 low-copy flowers exact remains in `RUNNING` state at `apply_certified_residual_texture`. It has completed reparenting, teacher-cache build, target no-GT stripping, and no-GT verification, but has not produced exact test metrics yet.

## Interpretation

The Phase-J teacher signal is measurable, but the tested baked surface carrier did not produce a policy-val all-axis projection win. Under the v169 improved prompt this blocks new full9 runs and points to carrier under-capacity or mask/energy dilution rather than missing experiment packaging.

## Artifact Paths

- JSON: `/tmp/peilincai_spcarnet_v176_phasej_teacher_projection_audit.json`
- repo JSON: `docs/car_model/vnext_artifacts/v176_phasej_teacher_projection_audit.json`
- edge/luma target JSON: `docs/car_model/vnext_artifacts/v176_edge_luma_projection_audit.json`
- v175 ELA report: `docs/car_model/vnext_artifacts/v175_strict_ela_flowers_report.json`
- v175 test metrics: `docs/car_model/vnext_artifacts/v175_strict_ela_flowers_test_results.json`
- Markdown: `docs/car_model/6-29-v176-PhaseJTeacherProjectionAudit.md`
- evidence dir: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- model path: `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/flowers/ratio_0200/compact_model`

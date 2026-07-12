# v133 Adaptive Low-Support Teacher-Basis Status

Date: 2026-06-27

## Purpose

This run tests whether the new prompt direction has moved beyond engineering
plumbing into a real weak-scene improvement. The target failure case is
`flowers`, where v132b still fell back to no-op because the residual policy-val
signal was too weak.

## Implemented Change

v133 adds an adaptive low-support teacher-basis policy to the residual texture
adapter:

- Keep `--teacher_distilled_basis_min_face_samples` as the requested ceiling.
- Compute an effective per-candidate threshold from fit-evidence face support
  only.
- Lower the effective threshold when the support distribution is sparse.
- Increase ridge on newly enabled low-support faces.
- Record requested/effective thresholds, support statistics, solved face counts,
  and ridge multipliers in the adapter audit.

No target/test GT is used by the adaptive threshold. Target GT remains stripped
during apply and is restored only for final evaluation.

Main files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Validation:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
git diff --check scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_vnext_certified_residual_texture_scene.py
```

Both checks passed.

## Experiment

Scene: `flowers`

Run root:

```text
/dev/shm/peilincai_spcarnet_v133_adaptive_teacher_flowers_20260627_0000
```

W&B offline run:

```text
/dev/shm/peilincai_wandb_v133_adaptive_teacher_flowers_20260627_0000/wandb/offline-run-20260627_002738-kl5uycej
```

Audit:

```text
/dev/shm/peilincai_spcarnet_v133_adaptive_teacher_flowers_20260627_0000/flowers/model/surface_residual_region_texture_adapter_audit.json
```

Metrics:

```text
PSNR  20.452776
SSIM   0.549059
LPIPS  0.355544
```

These metrics are fallback/no-op metrics, not accepted-method gains.

## Key Findings

Positive engineering result:

```text
teacher requested min-face samples: 1024
teacher effective min-face samples: 139
teacher supported faces: 15 -> 256
new low-support solved faces: 241
supported-face fraction: 0.748538
max ridge multiplier: 4.183453
```

Negative method result:

```text
accepted: false
effective_policy: fallback_noop
selected_alpha: 0.0
target changed fraction: 0.0
```

Risk-gate rejection:

```text
cvar20_view_relative_gain -0.000348 < 0.000000
min_view_relative_gain   -0.001043 < -0.000001
effective_relative_gain   0.000418 < 0.001000
```

View-confidence diagnosis:

```text
positive policy-val views: 1 / 12
negative policy-val views: 11 / 12
post-confidence accepted: false
```

## Verdict

The new prompt has produced a concrete implementation and a measurable
representation-capacity expansion, but it has not reached the expected effect.
For `flowers`, wider teacher-basis support still fails robust policy-val gates
and does not produce accepted target changes.

Current confidence:

- Engineering correctness: moderate.
- Paper-level performance gain: low.
- Confidence that this exact v133 policy solves the bottleneck: low.

The next fix should not merely lower thresholds further. The failure mode is now
clearer: more faces can be fit, but the fitted residual field is not robust
across policy-val views. The next method step should target view-consistent
residual learning or a fast risk-aware residual cache that can distinguish
beneficial local geometry/appearance regions from globally unstable residuals.

Final status for this v133 check: NOT COMPLETE.

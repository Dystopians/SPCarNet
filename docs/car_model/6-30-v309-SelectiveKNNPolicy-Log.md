# v309 Selective Source-Heldout KNN Policy Log

Date: 2026-06-30

## Reflection Status

The useful reflection was not "try more parameters". It was the recognition
that the previous per-scene policy was still too coarse, while a naive per-view
gate could damage stable scenes. v309 therefore uses a two-level target-GT-free
decision rule:

1. choose the scene branch with the existing source-heldout auto selector;
2. keep fixed scenes fixed, because v307 showed that overriding fixed scenes is
   unsafe on stump and treehill;
3. score per-view fixed/learned/hybrid candidates with a source-heldout KNN
   objective `PSNR gain + 20 * SSIM gain`;
4. enable the KNN policy only when leave-one-out source-heldout evidence has a
   non-negative PSNR delta over the scene-selected branch;
5. apply the final policy to target/test views, then read GT only for metrics.

This is a small but real method change: the policy is data-adaptive, uses
source-heldout evidence, and rejects the unsafe per-view mechanisms discovered
in v306-v308.

## Implementation

Main implementation:

```text
scripts/car_model/apply_source_heldout_support_transport_calibrator.py
```

New interfaces:

```text
--enable_per_view_risk_gate
--per_view_gate_score_grid
--per_view_gate_min_accept_fraction
--per_view_gate_max_accept_fraction
--per_view_gate_min_mean_psnr_delta
--per_view_gate_min_mean_ssim_delta
--per_view_gate_min_cvar_psnr_delta
--per_view_gate_cvar_weight
--per_view_gate_min_weight
--enable_per_view_knn_policy
--per_view_knn_feature_grid
--per_view_knn_k
--per_view_knn_min_predicted_score
--per_view_knn_min_source_psnr_delta
--per_view_knn_require_source_safe
--per_view_knn_allow_when_scene_fixed
```

The final v309 policy uses:

```text
--output_variant source_heldout_auto
--enable_per_view_knn_policy
--per_view_knn_k 3
--per_view_knn_min_source_psnr_delta 0.0
```

No `--per_view_knn_allow_when_scene_fixed` is used, so fixed fallback scenes
remain fixed.

## Negative Attempts That Shaped v309

v306 threshold risk gate, focused on bicycle/counter/stump/treehill, failed as
a main method. It over-nooped difficult outdoor views and became unsafe versus
fixed on stump and treehill.

| scene | mode | selected PSNR gain | selected SSIM gain | safe vs fixed | no-op fraction |
|---|---|---:|---:|:---:|---:|
| bicycle | threshold | +0.108638 | +0.002849 | yes | 0.040000 |
| counter | threshold | +0.415465 | +0.006646 | yes | 0.100000 |
| stump | threshold | +0.025380 | +0.000615 | no | 0.750000 |
| treehill | threshold | +0.059700 | +0.001110 | no | 0.388889 |

v307 unconditional KNN proved that per-view KNN can help bicycle, but it should
not override fixed scenes.

| scene | mode | selected PSNR gain | selected SSIM gain | safe vs fixed | no-op fraction |
|---|---|---:|---:|:---:|---:|
| bicycle | KNN | +0.118074 | +0.002981 | yes | 0.000000 |
| counter | KNN | +0.424732 | +0.006910 | yes | 0.000000 |
| stump | KNN | +0.060241 | +0.001171 | no | 0.250000 |
| treehill | KNN | +0.068576 | +0.001251 | no | 0.277778 |

v308 fixed the safety issue by disabling KNN when the scene selector chose
fixed, but it still enabled KNN on learned indoor scenes where source-heldout
evidence predicted a loss. It stayed safe but was slightly worse than v305:

```text
v308 macro PSNR gain: +0.265521
v308 macro SSIM gain: +0.003699
v308 minus v305 PSNR: -0.001057
v308 minus v305 SSIM: -0.000001
```

## Source-Heldout KNN Enable Audit

The aggregate summary now records the source-heldout evidence used to enable or
disable KNN. All v309 runs used `support_source_mode=source_split`.

| scene | scene branch | source KNN PSNR delta vs branch | source KNN SSIM delta vs branch | KNN enabled | reason |
|---|---|---:|---:|:---:|---|
| bicycle | hybrid | +0.003363 | -0.000134 | yes | positive source PSNR delta |
| bonsai | learned | -0.019883 | -0.000169 | no | source PSNR delta negative |
| counter | learned | -0.013092 | -0.000085 | no | source PSNR delta negative |
| flowers | hybrid | +0.006032 | -0.000043 | yes | positive source PSNR delta |
| garden | hybrid | +0.004292 | -0.000023 | yes | positive source PSNR delta |
| kitchen | learned | -0.011621 | +0.000021 | no | source PSNR delta negative |
| room | hybrid | -0.014647 | +0.000019 | no | source PSNR delta negative |
| stump | fixed | n/a | n/a | no | scene branch is fixed |
| treehill | fixed | n/a | n/a | no | scene branch is fixed |

This means v309 is not a manually chosen per-scene parameter set. It is a
single policy: use source-heldout evidence to decide whether per-view KNN is
allowed, then apply the frozen rule to target/test.

Important boundary: the v309 enable gate requires non-negative source-heldout
PSNR delta versus the scene-selected branch. It does not require the KNN policy
to be source-safe versus fixed unless `--per_view_knn_require_source_safe` is
enabled. Therefore the 9/9 safe-vs-fixed statement below is a target/test
evaluation result, not a pre-application source certificate.

## v309 Full9 Result

Machine-readable summary:

```text
docs/car_model/results/v309_selective_knn_policy_multiscene_summary.json
```

Output root:

```text
outputs/carnet/spcarnet_v309_selective_knn_policy_multiscene_20260630
```

Macro over 9 scenes / 246 target-test views:

```text
selected PSNR gain:              +0.267843
selected SSIM gain:              +0.003711
selected minus fixed PSNR gain:  +0.037808
selected minus fixed SSIM gain:  +0.000296
safe vs fixed scene rate:        9/9
positive vs base scene rate:     9/9
mean positive-view fraction:     0.949784
KNN enabled scene rate:          3/9
negative PSNR scene count:       4/9
```

Direct comparison:

```text
v309 minus v305 PSNR gain: +0.001265
v309 minus v305 SSIM gain: +0.000010
v309 minus v308 PSNR gain: +0.002322
v309 minus v308 SSIM gain: +0.000011
```

Scene deltas versus v305:

| scene | KNN enabled | PSNR delta vs v305 | SSIM delta vs v305 |
|---|:---:|---:|---:|
| bicycle | yes | +0.005986 | +0.000028 |
| bonsai | no | +0.000000 | +0.000000 |
| counter | no | +0.000000 | +0.000000 |
| flowers | yes | +0.002564 | +0.000059 |
| garden | yes | +0.002839 | +0.000004 |
| kitchen | no | +0.000000 | +0.000000 |
| room | no | +0.000000 | +0.000000 |
| stump | no | +0.000000 | +0.000000 |
| treehill | no | +0.000000 | +0.000000 |

Representative command:

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/apply_source_heldout_support_transport_calibrator.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/room/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --output_dir outputs/carnet/spcarnet_v309_selective_knn_policy_multiscene_20260630/room \
  --target_split test \
  --support_source_mode source_split \
  --heldout_stride 4 \
  --heldout_offset 0 \
  --device cuda \
  --k 4 \
  --anchor_alpha 0.25 \
  --learned_scale 0.5 \
  --blend 0.5 \
  --output_variant source_heldout_auto \
  --selector_val_stride 3 \
  --selector_val_offset 0 \
  --enable_per_view_knn_policy \
  --per_view_knn_k 3 \
  --per_view_knn_min_source_psnr_delta 0.0 \
  --evidence_max_side 256 \
  --compute_ssim \
  --ssim_max_side 256 \
  --save_example_views 1 \
  --copy_gt \
  --enable_wandb \
  --wandb_project spcarnet-transport-diagnostics \
  --wandb_run_name v309-selective-knn-room
```

W&B offline examples:

```text
outputs/carnet/spcarnet_v309_selective_knn_policy_multiscene_20260630/bicycle/wandb/offline-run-20260630_184355-uvm9hi71
outputs/carnet/spcarnet_v309_selective_knn_policy_multiscene_20260630/flowers/wandb/offline-run-20260630_184601-xw184q7d
outputs/carnet/spcarnet_v309_selective_knn_policy_multiscene_20260630/garden/wandb/offline-run-20260630_184604-0nsvmjlb
outputs/carnet/spcarnet_v309_selective_knn_policy_multiscene_20260630/room/wandb/offline-run-20260630_185030-r42w25y0
```

## Honest Verdict

v309 is the current best support-transport policy because it is slightly better
than v305 while preserving the 9/9 safe-vs-fixed scene rate. The improvement is
real but marginal, and mean positive-view fraction is lower than v305, so it
should not be oversold as a paper-level breakthrough.

Remaining weaknesses:

- macro gain over v305 is only `+0.001265` PSNR and `+0.000010` SSIM;
- mean positive-view fraction drops from `0.954228` to `0.949784`;
- KNN is not yet guarded by a source fixed-safety certificate;
- bicycle, counter, stump, and treehill still have individual negative PSNR
  views;
- this pass still lacks LPIPS/DISTS, fresh clean long MeshSplatting reruns, and
  full geometry/triangle-accounting tables;
- qualitative changes are expected to be subtle because the method is a
  conservative support-transport correction.

Current verdict:

```text
Final status: NOT COMPLETE.
```

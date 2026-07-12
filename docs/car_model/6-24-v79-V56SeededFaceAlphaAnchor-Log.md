# v79 V56-Seeded Face-Alpha Anchor Log

日期：2026-06-24  
目的：把 representation-level 诊断线重新锚定到此前最强的 v56/v64 counter reference，确认后续改进应从 face-alpha + tex32 + support4096 的强配置继续，而不是从 v75-v78 的弱配置继续。

---

## 命令

```bash
env WANDB_DIR=/dev/shm/wandb_spcarnet_v79 WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter --gpu 5 \
  --output_root /dev/shm/peilincai_spcarnet_v79_v56_seeded_anchor_20260624 \
  --tag v79_v56_seeded_anchor_facealpha_support4096_tex32_nearest_counter_region_texture_adapter \
  --support_expansion_max_extra_faces_candidates 4096 \
  --texture_size_candidates 32 \
  --atlas_empty_bin_fill_mode nearest_observed \
  --enable_policy_val_face_alpha_calibration \
  --face_alpha_calibration_max_alpha 0.5 \
  --face_alpha_calibration_min_alpha 0.0 \
  --face_alpha_calibration_multipliers 0.5,0.75,1.0,1.25 \
  --face_alpha_calibration_min_face_samples 256 \
  --min_policy_val_l1_positive_view_fraction 0.9 \
  --min_target_changed_fraction 0.001 \
  --wandb_project SPCarNet \
  --wandb_group v79_v56_seeded_anchor \
  --wandb_run_name v79_v56_seeded_anchor_counter_20260624 \
  --wandb_mode online --force
```

W&B run：`tyvbnmrp`

---

## 结果

| method | scene | PSNR | SSIM | LPIPS | status |
|---|---|---:|---:|---:|---|
| v56/v64 reference | counter | `26.756130219` | `0.862126231` | `0.251691371` | previous best fixed representation-level reference |
| v79 v56-seeded anchor | counter | `26.756130219` | `0.862126231` | `0.251691371` | reproduced reference |
| v75 local patch prior | counter | `26.753995895` | `0.862119257` | `0.251853049` | below anchor |
| v78 target-footprint certificate | counter | `26.753528595` | `0.862111032` | `0.251881272` | below anchor |

Policy/audit highlights：

| item | value |
|---|---:|
| accepted | `true` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.5` |
| local face-alpha enabled | `true` |
| face-alpha face count | `394` |
| fallback alpha | `0.5` |
| changed fraction on target render | `0.063901318` |

---

## 结论

v79 不是新提升，但它非常关键：它证明此前 v75-v78 的下滑不是 representation-level 路线必然失败，而是后续 probe 离开了 v56/v64 的强锚点配置。下一步应该从以下固定锚点继续：

```text
texture_size = 32
support_expansion_max_extra_faces = 4096
atlas_empty_bin_fill_mode = nearest_observed
enable_policy_val_face_alpha_calibration = true
face_alpha_calibration_max_alpha = 0.5
```

后续任何 prior、hybrid、target-support 或 patch certificate 都必须先包含这个 zero/new-module-off anchor，并证明不回退到 v75-v78 的弱线。

---

## 证据路径

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v79_v56_seeded_anchor_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v79_v56_seeded_anchor_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v79_v56_seeded_anchor_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v79_v56_seeded_anchor_20260624/counter/apply_metrics_counter.log
```


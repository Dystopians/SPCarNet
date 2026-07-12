# Stage 2 v4 Normal-Band Autodecoder Log

Date: 2026-06-24  
Status: `TRAINING_COMPLETE_WITH_FULL_VAL_EVAL_SOFT_FAIL`  
Goal: improve the Stage-2 shape-field autodecoder quality gate after v3 held-out MAP-fit reached `206/206` extraction but missed chamfer and filled-IoU thresholds.

---

## 1. Motivation

The v3 autodecoder is no longer blocked by evaluation plumbing:

```text
outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json
W&B: svtbc8sn
val extraction: 206 / 206
recon chamfer: 0.0698447353
filled IoU: 0.5531548112
shell IoU: 0.9112784961
```

The remaining problem is quality. The decoder can produce meshes, but the occupancy boundary is not sharp/accurate enough for the Stage-2 gate:

| metric | gate | v3 MAP-fit |
|---|---:|---:|
| `recon_chamfer_l1_mean` | <= `0.05` | `0.0698447353` |
| `mesh_iou_at_0.5_mean` | >= `0.92` | `0.5531548112` |
| `mesh_extraction_success_rate` | >= `0.95` | `1.0` |

v4 therefore does not scale the decoder again. It keeps v3 capacity fixed and changes the objective.

---

## 2. Method Change

v4 adds surface-normal band supervision:

```text
x_inner = x_surface - epsilon * normal
x_outer = x_surface + epsilon * normal
```

For occupancy training:

```text
BCE(f(x_inner, z), 1) + BCE(f(x_outer, z), 0)
```

This is meant to give Marching Cubes a clearer `0.5` crossing around the actual surface, addressing the boundary ambiguity that pure positive-surface / free-space BCE leaves behind.

Implementation:

```text
ss3dm_prior/training/spcarnet_autodecoder.py
scripts/car_model/eval_spcarnet_shape_field_autodecoder.py
scripts/car_model/smoke_test_spcarnet_stage2.py
configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml
configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml
scripts/car_model/train_spcarnet_shape_field_autodecoder_v4_band.sh
```

Key settings:

```text
latent_dim = 512
hidden_dim = 768
depth = 8
w_band = 0.5
queries_band = 256
band_epsilon = 0.025
```

---

## 3. Validation Before Full Training

Static checks:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  ss3dm_prior/training/spcarnet_autodecoder.py \
  scripts/car_model/eval_spcarnet_shape_field_autodecoder.py \
  scripts/car_model/smoke_test_spcarnet_stage2.py

git diff --check -- \
  ss3dm_prior/training/spcarnet_autodecoder.py \
  scripts/car_model/eval_spcarnet_shape_field_autodecoder.py \
  scripts/car_model/smoke_test_spcarnet_stage2.py \
  configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml \
  configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml \
  scripts/car_model/train_spcarnet_shape_field_autodecoder_v4_band.sh
```

Smoke test:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_spcarnet_stage2.py
```

Smoke result:

```text
loss_band present: yes
loss_total: 2.252728 -> 2.249160
MC mesh_present: true
PASS
```

100-step sanity train:

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  -m ss3dm_prior.training.spcarnet_autodecoder_cli \
  --model_config configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml \
  --train_config configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --output_dir outputs/carnet/spcarnet/autodecoder_v4_band_sanity_100step_20260624 \
  --run_name spcarnet_autodecoder_v4_band_sanity_100step_20260624 \
  --device cuda \
  --max_steps 100
```

Sanity eval:

```text
outputs/carnet/spcarnet/autodecoder_v4_band_sanity_100step_20260624/eval_val_mapfit_limit2.json
n_extracted = 2 / 2
strict JSON = yes
```

---

## 4. Full Training Command

```bash
GPU=2 WANDB_MODE=online WANDB_PROJECT=spcarnet \
RUN_NAME=spcarnet_autodecoder_v4_band_20260624 \
OUTPUT_DIR=/data/peilincai/mesh-splatting/outputs/carnet/spcarnet/autodecoder_v4_band_20260624 \
bash scripts/car_model/train_spcarnet_shape_field_autodecoder_v4_band.sh \
  2>&1 | tee outputs/carnet/spcarnet/autodecoder_v4_band/logs/train_full_20260624.log
```

W&B:

```text
https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/dysg8508
```

---

## 5. Full Training Result

Training completed.

Online run and artifacts:

```text
W&B run id: dysg8508
W&B URL: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/dysg8508
steps: 69300 / 69300
epochs: 300
elapsed: 3957.1347 sec
checkpoint: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
fit summary: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/fit_summary.json
resolved config: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/resolved_config.json
```

Early checkpoint probes:

| checkpoint | eval subset | recon chamfer | filled IoU | shell IoU | decision |
|---|---:|---:|---:|---:|---|
| v3 full checkpoint, same probe | 20 val objects | `0.0931878027` | `0.5577932618` | `0.8629142571` | reference |
| v4 epoch 10 | 20 val objects | `0.1026277483` | `0.4818364483` | `0.8530468456` | too early |
| v4 epoch 50 | 20 val objects | `0.0882688718` | `0.5752584533` | `0.8284169075` | promising |
| v4 epoch 100 | 20 val objects | `0.0897679076` | `0.5576571527` | `0.8594210818` | worse than epoch 50 on chamfer/filled IoU |

Interpretation:

> v4 is not a final quality pass, but the epoch-50 probe beats the v3 full checkpoint on subset chamfer and filled IoU under the same MAP-fit/eval settings. The epoch-100 probe suggests later training can over-smooth or overfit the occupancy boundary, so epoch50 is treated as the current best Stage-2 quality checkpoint unless the final checkpoint full eval proves otherwise.

---

## 6. Full Held-Out MAP-Fit Eval

Epoch-50 checkpoint full held-out eval:

```bash
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/eval_spcarnet_shape_field_autodecoder.py \
  --checkpoint outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_epoch50_probe.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --splits val \
  --output outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json \
  --mc_resolution 32 \
  --sample_count 4096 \
  --device cuda \
  --fit_missing_latents \
  --fit_steps 100 \
  --fit_lr 0.01 \
  --fit_queries_surface 384 \
  --fit_queries_free 384 \
  --fit_queries_hard 128 \
  --fit_queries_mixed 128 \
  --fit_queries_band 128 \
  --fit_band_epsilon 0.025 \
  --resolved_config outputs/carnet/spcarnet/autodecoder_v4_band_20260624/resolved_config.json \
  --wandb_project spcarnet \
  --wandb_run_name stage2_autodecoder_v4_band_epoch50_val_mapfit_full206_20260624 \
  --wandb_mode online
```

Result:

```text
W&B run: 4wu9w305
Output: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
n_extracted: 206 / 206
recon_chamfer_l1_mean: 0.0607328202
hidden_chamfer_l1_mean: 0.0933915632
mesh_iou_at_0.5_mean: 0.5683319216
mesh_iou_at_0.5_shell_mean: 0.8783071888
surface_normal_consistency_mean: 0.7195177524
```

Comparison to v3 full-val MAP-fit:

| metric | v3 MAP-fit | v4 epoch50 MAP-fit | delta |
|---|---:|---:|---:|
| `recon_chamfer_l1_mean` | `0.0698447353` | `0.0607328202` | `-0.0091119151` |
| `hidden_chamfer_l1_mean` | `0.1023846301` | `0.0933915632` | `-0.0089930669` |
| `mesh_iou_at_0.5_mean` | `0.5531548112` | `0.5683319216` | `+0.0151771104` |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` | `0.8783071888` | `-0.0329713073` |

Decision so far:

> v4 normal-band supervision is a real improvement over v3 on the primary chamfer metrics and filled IoU, but it still misses the original Stage-2 gate (`chamfer <= 0.05`, `filled IoU >= 0.92`). The full 300-epoch run completed; final-checkpoint full-val MAP-fit was run as a secondary check and did not beat epoch50. Epoch50 remains the current best documented Stage-2 quality candidate.

Final checkpoint full held-out eval:

```bash
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/eval_spcarnet_shape_field_autodecoder.py \
  --checkpoint outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --splits val \
  --output outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json \
  --mc_resolution 32 \
  --sample_count 4096 \
  --device cuda \
  --fit_missing_latents \
  --fit_steps 100 \
  --fit_lr 0.01 \
  --fit_queries_surface 384 \
  --fit_queries_free 384 \
  --fit_queries_hard 128 \
  --fit_queries_mixed 128 \
  --fit_queries_band 128 \
  --fit_band_epsilon 0.025 \
  --resolved_config outputs/carnet/spcarnet/autodecoder_v4_band_20260624/resolved_config.json \
  --wandb_project spcarnet \
  --wandb_run_name stage2_autodecoder_v4_band_final_val_mapfit_full206_20260624 \
  --wandb_mode online
```

Final checkpoint result:

```text
W&B run: q1jjwvdm
Output: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
n_extracted: 206 / 206
recon_chamfer_l1_mean: 0.0655826944
hidden_chamfer_l1_mean: 0.0963624408
mesh_iou_at_0.5_mean: 0.5314717742
mesh_iou_at_0.5_shell_mean: 0.8563237802
surface_normal_consistency_mean: 0.6890807638
strict JSON: yes
```

Comparison among v3, v4 epoch50, and v4 final:

| metric | v3 MAP-fit | v4 epoch50 | v4 final | best |
|---|---:|---:|---:|---|
| `recon_chamfer_l1_mean` | `0.0698447353` | `0.0607328202` | `0.0655826944` | v4 epoch50 |
| `hidden_chamfer_l1_mean` | `0.1023846301` | `0.0933915632` | `0.0963624408` | v4 epoch50 |
| `mesh_iou_at_0.5_mean` | `0.5531548112` | `0.5683319216` | `0.5314717742` | v4 epoch50 |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` | `0.8783071888` | `0.8563237802` | v3 |

Interpretation:

> The final checkpoint confirms that longer training is not monotonically better for this objective. Normal-band supervision helps early-to-mid training, but late training degrades filled IoU, shell IoU, and normal consistency. Future Stage-2 work should add validation-based checkpoint selection and stronger regularization rather than simply extending epochs.

Validation-driven checkpoint selector:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/select_spcarnet_stage2_checkpoint.py \
  --candidate v3_full:outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt:outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json \
  --candidate v4_epoch50:outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_epoch50_probe.pt:outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json \
  --candidate v4_final:outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt:outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json \
  --output_json outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json \
  --output_md outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
```

Selector result:

```text
Status: BEST_AVAILABLE_GATE_FAIL_WITH_LATE_DEGRADATION
Best candidate: v4_epoch50
Best score: 0.039685866
Gate pass: false
Markdown: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
JSON: outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json
```

---

## 7. Decision

Current decision: **do not promote Stage 2 v4 as headline**.

Use v4 normal-band as a documented method improvement and next-step evidence:

- It is a real train/eval pipeline change, not a parameter-only report.
- It improves v3 on full-val recon chamfer, hidden chamfer, and filled-volume IoU.
- The epoch50 checkpoint is the current best v4 checkpoint; the final checkpoint is worse and should be treated as a late-training degradation diagnostic.
- It worsens shell IoU and still misses the original quality gate by a large margin.
- For the mentor/PPT report, it should be described as evidence that boundary-aware object-prior training is moving in the right direction, not as a solved shape prior.

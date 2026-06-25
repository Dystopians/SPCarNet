# v101 Claim-to-Artifact Manifest

Date: 2026-06-25

Scope: artifact map for the current v101 paper-loop checkpoint. This file records which claims are currently supported by local commands and outputs, and which claims remain unsafe.

## Supported claims

| claim | status | primary artifacts |
|---|---|---|
| v101 can be executed through `render.py` as a checkpoint-attached endpoint. | supported | `render.py`; `scripts/car_model/run_v101_renderpy_endpoint_full9.py`; `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_20260625/v101_renderpy_endpoint_full9_summary.json` |
| v101 can force train-derived evidence-bank consumption across the local full9 set. | supported | `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.json`; banks under `/dev/shm/peilincai_spcarnet_v101_bankfp16_full9_fixed_20260625/<scene>/v101_evidence_bank.pt` |
| The strict bankfp16 v101 endpoint beats the selected local clean MeshSplatting baseline on full9 RGB metrics. | supported | `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.json`; mean `+1.329627 PSNR / +0.034657 SSIM / -0.063316 LPIPS` |
| Detached v101 packages reproduce the bankfp16 render.py endpoint exactly on all local full9 scenes. | supported | `outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_full9_20260625/v101_detached_package_full9_summary.json`; `all_present=true`, `all_passed=true`, `all_used_required_bank=true`, `all_hash_exact=true` |
| The detached package path does not need the original compact train render folder at render time. | supported for full9 packages | per-scene detached reports in `outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_full9_20260625/*_detached_package_report.json`; each run passes an explicit package bank path and `--checkpoint_endpoint_base_model /__spcarnet_detached_package_must_not_read_train_evidence__` |
| The endpoint path does not use held-out target GT as support evidence. | supported by counter smoke | `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_20260625/target_gt_nonuse_smoke_counter.json`; `max_abs_output_diff=0.0` |
| The current qualitative panel is traceable to exact source images and selected crops. | supported | `assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png`; `assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json` |
| v101 has measured deployment overhead. | supported, negative for speed | `outputs/carnet/meshsplatopt/ecsr_phase_v101_runtime_audit_20260625/counter_runtime_audit.json`; standard `2.238598 sec/view`, v101 `4.220285 sec/view`, slowdown `1.885235x` |

## Key commands

Strict bankfp16 full9 render/eval:

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_v101_renderpy_endpoint_full9.py \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625 \
  --gpus 1,2,3,5 --max_parallel 4 \
  --method_name ours_26000_v101_bankfp16_renderpy_endpoint_full9_fixed \
  --build_banks --require_bank \
  --bank_root /dev/shm/peilincai_spcarnet_v101_bankfp16_full9_fixed_20260625 \
  --bank_residual_dtype float16 --bank_depth_dtype float16 \
  --wandb --wandb_dir /dev/shm/peilincai_spcarnet_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/wandb \
  --wandb_group v101_bankfp16_renderpy_endpoint_full9_fixed \
  --wandb_name v101_bankfp16_renderpy_endpoint_full9_fixed
```

Detached package validation template:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/validate_v101_detached_package.py \
  --scene <scene> \
  --gpu <gpu> \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_full9_20260625 \
  --method_name ours_26000_v101_detached_package_full9_<scene> \
  --force
```

Detached summary:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/summarize_v101_detached_package_full9.py \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_full9_20260625
```

Runtime audit:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/benchmark_v101_detached_runtime.py --gpu 2
```

## Unsafe or incomplete claims

| claim | status | why |
|---|---|---|
| v101 is a vanilla checkpoint-baked MeshSplatting representation. | not supported | The repaired output still requires the `render.py` endpoint hook and evidence bank. |
| v101 is faster than MeshSplatting. | not supported | The detached counter runtime audit shows a `1.885235x` wall slowdown versus standard `render.py`. |
| v101 independently improves over Phase-J quality. | not supported | v101 is a deployment closure over Phase-J/v100; strict detached packages intentionally reproduce the bankfp16 endpoint reference exactly. |
| fp16 banks are exactly equal to Phase-J across full9. | not supported | The strict bankfp16 run has tiny mean drift versus Phase-J. Use float32 targeted checks if exact Phase-J parity is required. |
| v101 adds new geometry/triangle-count gains beyond the compact parent. | not supported | Geometry/triangle reductions come from the compact parent and Phase-J pipeline; v101 packages the endpoint path. |

## Completion checklist status

| item | status |
|---|---|
| real method change implemented in train/eval/render pipeline | done for `render.py` endpoint and evidence-bank path |
| baseline/current/improved/ablation run or blocked | local clean baseline, Phase-J/v100, v101 auto endpoint, v101 strict bank, detached full9, and runtime audit are documented; vanilla checkpoint-baked representation remains not supported by current evidence |
| metrics and qualitative outputs saved | done |
| commands/configs/result paths/errors documented | done in this manifest and `docs/car_model/6-25-v101-RenderPyEndpoint-EvidenceBank-Log.md` |
| final method story clear enough for slides/paper | done in `docs/car_model/6-25-v101-Subagent-Review-And-PaperStory.md` |
| final review honestly marks weaknesses | done; main weaknesses are endpoint dependency, runtime overhead, and lack of baked representation |

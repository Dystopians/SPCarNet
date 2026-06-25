# v101 Render.py Endpoint / Evidence Bank 技术日志

Date: 2026-06-25

Status: milestone evidence update. This note records the v101 mechanism, counter reproduction, full9 render.py endpoint validation, full9 bank-backed validation, qualitative panel, and remaining claim boundary. It does not claim a paper-level final representation-baked method.

## 0. 2026-06-25 milestone update

v101 now has two validated full9 render paths:

- Auto endpoint full9: `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_20260625/v101_renderpy_endpoint_full9_summary.json`
- Require-bank fp16 full9: `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.json`

Both runs completed all 9 scenes with per-scene render/eval return codes equal to zero. The require-bank fp16 run additionally completed `bank_rc=0` for every scene and forced `render.py` to consume a scene-specific `v101_evidence_bank.pt`.

| run | build bank | require bank | mean PSNR | mean SSIM | mean LPIPS | dPSNR vs clean | dSSIM vs clean | dLPIPS vs clean | dPSNR vs Phase-J | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| auto render.py endpoint | no | no | 26.482349 | 0.783685 | 0.224293 | +1.330667 | +0.034667 | -0.063328 | -0.000417 | Counter used the existing bank automatically; the other scenes used ELA report support names. |
| require-bank fp16 endpoint | yes | yes | 26.481309 | 0.783675 | 0.224305 | +1.329627 | +0.034657 | -0.063316 | -0.001457 | All scenes used external fp16 train-derived evidence banks from `/dev/shm/peilincai_spcarnet_v101_bankfp16_full9_fixed_20260625`. |

Additional evidence:

- Target-GT non-use smoke passed: `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_20260625/target_gt_nonuse_smoke_counter.json` reports `max_abs_output_diff=0.0` after replacing the target GT path with a dummy nonexistent path.
- Qualitative comparison panel: `assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png`; manifest: `assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json`.
- W&B offline runs:
  - auto endpoint: `/dev/shm/peilincai_spcarnet_v101_renderpy_endpoint_full9_20260625/wandb/wandb/offline-run-20260625_042308-44qa0b99`
  - require-bank fp16: `/dev/shm/peilincai_spcarnet_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/wandb/wandb/offline-run-20260625_043746-3pivbb1n`

The bank-backed run is the cleaner v101 evidence-bank deployment claim. The auto run is still useful as a render.py entrypoint reproduction check, but it should not be described as a uniform bank-backed run because `counter` already had a float32 bank while the other scenes fell back to report support frames.

## 1. Motivation: from v100 materialized sidecar to render.py endpoint / evidence bank

v100 made the Phase-J / ELA repair mechanically auditable by packaging it as a checkpoint-attached sidecar endpoint. That was useful because it fixed provenance, frame-set denominator checks, W&B logging, non-noop evidence, and full9 accounting around a concrete endpoint artifact. Its limitation was also explicit: the endpoint was still a sidecar materialization of Phase-J behavior, not a standard MeshSplatting checkpoint path that vanilla `render.py` naturally consumes.

v101 moves the next step into the rendering entrypoint. The target is not to invent a new paper claim in this draft, but to reduce the artifact gap:

- `render.py` can load a checkpoint-attached endpoint report, recompute target base renders online, apply the guarded residual transfer, and write the endpoint render output as a normal render method.
- A train-derived evidence bank can sit beside the endpoint report and provide residual, depth, and camera evidence without relying on pre-materialized target endpoint images.
- The counter target is to reproduce the v100 / Phase-J endpoint result through this render.py path, then reserve full9 validation for a separate summary artifact.

## 2. Module notes

### render.py endpoint hook

File: `render.py`

Key behavior:

- CLI flags: `--checkpoint_endpoint_method`, `--checkpoint_endpoint_output_method`, `--checkpoint_endpoint_base_model`, `--checkpoint_endpoint_base_method`, `--checkpoint_endpoint_evidence_max_side`, `--checkpoint_endpoint_bank_path`.
- `_load_endpoint_runtime(...)` resolves the endpoint directory under `point_cloud/iteration_<iter>/render_residual_endpoint/<endpoint_method>/`, reads `ela_report.json`, extracts policy / alpha / calibrators, and loads `v101_evidence_bank.pt` when present.
- If no evidence bank is present, the hook falls back to training split support frames from the base model and ELA report.
- `--checkpoint_endpoint_require_bank` now fails closed when the requested bank is absent; this prevents a bank validation run from silently falling back to train folders.
- `_render_endpoint_set(...)` calls the normal triangle renderer for each target view, saves a base render / GT / depth record, runs `adapt_frame(...)`, writes endpoint renders, and emits `render_py_endpoint_report.json`.
- The emitted report states the boundary directly: this is an online checkpoint-attached endpoint that recomputes target renders and does not rely on pre-materialized target endpoint images.

### Evidence bank builder

File: `scripts/car_model/build_v101_endpoint_evidence_bank.py`

Key behavior:

- Reads the endpoint ELA report and selects support frames from the base model train split.
- Stores train-derived residual tensors (`gt - render`), depth tensors, camera metadata, per-frame SHA-256 hashes, and a manifest.
- Default endpoint location: `point_cloud/iteration_26000/render_residual_endpoint/ours_26000_v100_checkpoint_attached_ela_endpoint/v101_evidence_bank.pt`.
- Default residual clipping: `0.25`.
- The manifest note is important for claim hygiene: the bank contains train-derived residual/depth/camera evidence and no held-out target GT.
- A bug found during the first fp16 bank full9 launch was fixed: external `--output_bank` paths now create their parent directory before `torch.save(...)`.

Counter manifest observed:

- path: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/recovery_model/point_cloud/iteration_26000/render_residual_endpoint/ours_26000_v100_checkpoint_attached_ela_endpoint/v101_evidence_bank_manifest.json`
- support frames: `210`
- missing report support names: `[]`
- residual dtype: `float32`
- depth dtype: `float32`
- bank SHA-256: `934fd5f145485bc48aee3f521d0d2b758560edea06660ef774b5c4afd743ab6c`

### full9 runner

File: `scripts/car_model/run_v101_renderpy_endpoint_full9.py`

Key behavior:

- Fixed scene list: `bicycle`, `flowers`, `garden`, `stump`, `treehill`, `room`, `counter`, `kitchen`, `bonsai`.
- Default v100 root: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625`.
- Default v100 summary JSON: `outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.json`.
- Per scene, calls:
  - `render.py -m <scene>/recovery_model --iteration 26000 --skip_train --checkpoint_endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint --checkpoint_endpoint_output_method <method>`
  - `scripts/car_model/evaluate_render_split_metrics.py -m <model> --split test --methods <method> --merge_model_results`
- Writes `v101_renderpy_endpoint_full9_summary.{json,csv,md}` under the requested `--report_root`.
- Default method name: `ours_26000_v101_renderpy_endpoint_full9`.
- Supports `--build_banks`, `--require_bank`, and `--bank_root` for full9 bank-backed validation without overwriting the counter float32 bank already stored under the endpoint sidecar.

## 3. Counter known evidence

Evidence source:

- metrics: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/recovery_model/results.json`
- endpoint gate reference: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/recovery_model/endpoint_gate_report.json`
- bank-fix render.py report: `/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/recovery_model/test/ours_26000_v101_bank_renderpy_endpoint_fix/render_py_endpoint_report.json`

| row | support path | PSNR | SSIM | LPIPS | image hash check vs v100 | interpretation |
|---|---|---:|---:|---:|---|---|
| v100 endpoint reference | checkpoint-attached ELA sidecar | 28.44917106628418 | 0.8937307000160217 | 0.18647237122058868 | reference | Phase-J endpoint materialization baseline |
| v101 render.py endpoint, non-bank | train split support frames loaded from base model | 28.44917106628418 | 0.8937307000160217 | 0.18647237122058868 | `30/30` PNG SHA-256 match | render.py path can reproduce v100 on counter without bank |
| v101 evidence-bank bug negative | `v101_evidence_bank.pt`, pre-fix behavior | 26.827314376831055 | 0.8608036637306213 | 0.24935589730739594 | not promoted | known bank-path negative; useful regression evidence |
| v101 evidence-bank fix | `v101_evidence_bank.pt`, fixed path | 28.44917106628418 | 0.8937307000160217 | 0.18647237122058868 | `30/30` PNG SHA-256 match | bank path reproduces v100 on counter |

Bank-fix render.py report details:

- target frames: `30`
- support frames: `210`
- support source: `v101_evidence_bank:/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/recovery_model/point_cloud/iteration_26000/render_residual_endpoint/ours_26000_v100_checkpoint_attached_ela_endpoint/v101_evidence_bank.pt`
- mean abs RGB delta: `0.011413626279681921`
- mean changed fraction: `0.9990177710851034`

## 4. full9 validation

### Auto endpoint full9

Command:

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_v101_renderpy_endpoint_full9.py \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_20260625 \
  --gpus 1,2,3,5 --max_parallel 4 \
  --wandb --wandb_dir /dev/shm/peilincai_spcarnet_v101_renderpy_endpoint_full9_20260625/wandb \
  --wandb_group v101_renderpy_endpoint_full9 --wandb_name v101_renderpy_endpoint_full9
```

Result: all 9 scenes completed with `render_rc=0` and `eval_rc=0`. This run proves the `render.py` endpoint can reproduce the v100 / Phase-J endpoint path over the full9 set. Because counter already had a float32 bank under its sidecar, this run is best described as an auto endpoint run, not a strict no-bank run.

### Require-bank fp16 full9

Command:

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

Result: all 9 scenes completed with `bank_rc=0`, `render_rc=0`, and `eval_rc=0`. The shell pipeline returned nonzero because the optional `tee` target directory was not pre-created, but the runner completed and wrote the summary, W&B offline run, per-scene logs, and all render/eval outputs.

| scene | bank rc | render rc | eval rc | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR Phase-J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 0 | 0 | 0 | 24.021442 | 0.702352 | 0.266102 | +0.719830 | +0.042484 | -0.065975 | -0.000101 |
| flowers | 0 | 0 | 0 | 20.300581 | 0.557456 | 0.329513 | +0.618324 | +0.045634 | -0.065051 | -0.003777 |
| garden | 0 | 0 | 0 | 26.310476 | 0.827830 | 0.135863 | +1.281265 | +0.047795 | -0.065451 | -0.000635 |
| stump | 0 | 0 | 0 | 25.595201 | 0.724082 | 0.263924 | +0.390160 | +0.018917 | -0.030080 | +0.000097 |
| treehill | 0 | 0 | 0 | 21.296227 | 0.595606 | 0.336322 | +0.362045 | +0.031084 | -0.069722 | +0.000000 |
| room | 0 | 0 | 0 | 30.305668 | 0.905688 | 0.195890 | +1.558392 | +0.020845 | -0.054013 | +0.000029 |
| counter | 0 | 0 | 0 | 28.442907 | 0.893696 | 0.186557 | +1.691133 | +0.031640 | -0.065447 | -0.006264 |
| kitchen | 0 | 0 | 0 | 30.197395 | 0.916093 | 0.132004 | +2.378843 | +0.039641 | -0.067182 | -0.002337 |
| bonsai | 0 | 0 | 0 | 31.861883 | 0.930276 | 0.172566 | +2.966650 | +0.033875 | -0.086926 | -0.000122 |

### Qualitative panel

Panel: `assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png`

Manifest: `assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json`

The panel compares local clean MeshSplatting `official_clean30k/<scene>/test/ours_26000` against `ours_26000_v101_bankfp16_renderpy_endpoint_full9_fixed` and GT. Rows are selected by held-out LPIPS improvement, and the crop is selected by local absolute-error reduction. It is designed for PPT use because full-frame visual differences are often subtle.

## 5. Claim boundary

- v101 currently supports a render.py-consuming checkpoint-attached endpoint path plus an optional train-derived evidence bank.
- The strongest verified v101 evidence is now the full9 require-bank fp16 run: every scene builds a train-derived evidence bank, `render.py` is forced to consume it, and all 9 scenes stay strongly above the local clean MeshSplatting baseline.
- Counter has exact float32 bank evidence: non-bank render.py endpoint and bank-fix endpoint both match v100 metrics, and both have `30/30` render PNG SHA-256 matches against the v100 endpoint.
- This is not evidence that a static MeshSplatting checkpoint has absorbed Phase-J / ELA. The endpoint still evaluates a sidecar policy at render time.
- This is not an independent improvement over Phase-J. The intended near-term standard is faithful render.py consumption of the v100 / Phase-J endpoint, not a new metric claim.
- The fp16 bank run shows a tiny mean Phase-J drift (`-0.001457 dB` PSNR, `-0.000044` SSIM, `+0.000043` LPIPS) from bank quantization / render-path differences; this should be described as deployment packaging drift, not a quality regression relative to clean.
- The evidence bank claim is limited to train-derived residual/depth/camera evidence. It should not be described as using held-out target GT for policy selection.
- The target-GT non-use smoke passed on counter by replacing the target GT path with a nonexistent dummy path and obtaining exactly identical adapted output.

## 6. Next steps

1. Add a detached-package test: copy only checkpoint links / endpoint report / evidence bank into a fresh directory, temporarily hide the original compact train render folder, and verify `render.py --checkpoint_endpoint_require_bank` still reproduces the outputs.
2. Add a float32-bank full9 or targeted float32-bank checks for the scenes with the largest fp16 drift (`counter`, `flowers`, `kitchen`) if exact Phase-J parity is needed.
3. Add a runtime table for auto endpoint versus fp16 bank endpoint. Current evidence supports quality and packaging, not speed.
4. Keep the paper story explicit: v101 is a render-entrypoint and train-evidence-bank closure over Phase-J/v100, not a checkpoint-baked representation-level final method.
5. Continue representation-level work separately if the target claim is a vanilla MeshSplatting checkpoint with absorbed repair behavior.

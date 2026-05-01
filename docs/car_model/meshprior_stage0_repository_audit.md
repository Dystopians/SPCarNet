# MeshPrior Stage 0 Repository Audit

| Field | Value |
|---|---|
| Stage | M0 / repository integrity, code audit, current-state verification |
| Date | 2026-05-01 |
| Decision | `PROCEED` |

## 1. Git State

- Current branch: `main`
- `git status --short` at audit time:

```text
 M scripts/car_model/eval_spcarnet_multihypothesis.py
 ? submodules/effrdel
 ? submodules/simple-knn
?? docs/prompts.md
```

Interpretation:

- The worktree is dirty before MeshPrior work begins.
- `scripts/car_model/eval_spcarnet_multihypothesis.py` is already modified and was not changed by this audit.
- `docs/prompts.md` is untracked and is the stage instruction source for this audit.
- Submodule state is not clean for `submodules/effrdel` and `submodules/simple-knn`; this is noted but did not block Python imports or smoke tests.

## 2. Python / Torch / CUDA

Default shell Python:

```text
Python 3.13.2
ModuleNotFoundError: No module named 'torch'
```

Project environment:

```text
micromamba run -n mesh_splatting python --version
Python 3.11.14
```

Torch / CUDA in the project environment:

```text
torch 2.7.1+cu126 cuda True
cuda_device_count 8
```

Conclusion:

- The default shell Python is not the correct project environment.
- All SP-CarNet / MeshPrior work should use `micromamba run -n mesh_splatting ...` or an activated `mesh_splatting` environment.
- The project environment matches the README expectation of Python 3.11 and CUDA-capable torch.

## 3. Integrity Commands

Commands run:

```bash
git status --short
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
micromamba run -n mesh_splatting python --version
micromamba run -n mesh_splatting python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q
```

Result:

- `compileall` passed in the project environment.
- `ss3dm_prior.models.spcarnet_shape_field` and `ss3dm_prior.models.spcarnet_posterior` import cleanly in the project environment.

## 4. Expected Code Files

| File | Status |
|---|---|
| `ss3dm_prior/data/spcarnet_object_dataset.py` | present |
| `ss3dm_prior/models/spcarnet_shape_field.py` | present, imports |
| `ss3dm_prior/models/spcarnet_posterior.py` | present, imports |
| `ss3dm_prior/training/spcarnet_autodecoder.py` | present |
| `ss3dm_prior/training/spcarnet_posterior.py` | present |
| `ss3dm_prior/losses_spcarnet_observation.py` | present |
| `scripts/car_model/build_spcarnet_object_index.py` | present |
| `scripts/car_model/eval_spcarnet_posterior_encoder.py` | present |
| `scripts/car_model/refine_spcarnet_latent_map.py` | present |
| `scripts/car_model/eval_spcarnet_multihypothesis.py` | present |
| `scripts/car_model/smoke_test_spcarnet_stage1.py` | present |
| `scripts/car_model/smoke_test_spcarnet_stage2.py` | present |
| `scripts/car_model/smoke_test_spcarnet_stage3.py` | present |
| `scripts/car_model/smoke_test_spcarnet_stage4.py` | present |
| `scripts/car_model/smoke_test_spcarnet_stage5.py` | present |

During the audit, one early file-existence check returned missing paths for several untracked SP-CarNet files, while subsequent direct checks and `find` showed the files present. The final state used for this audit is the repeated direct check above plus successful `compileall` and smoke tests.

## 5. Smoke Tests

All smoke tests were run with:

```bash
micromamba run -n mesh_splatting python <script>
```

| Smoke test | Result | Notes |
|---|---|---|
| `scripts/car_model/smoke_test_spcarnet_stage1.py` | PASS | Built temporary 8-object index; dataset, transforms, collate passed. |
| `scripts/car_model/smoke_test_spcarnet_stage2.py` | PASS | Loss decreased from `2.0794415` to `2.0752068`; MC untrained-mesh fallback expected. |
| `scripts/car_model/smoke_test_spcarnet_stage3.py` | PASS | Encoder forward, sampling, frozen-decoder decode, and backward checks passed. |
| `scripts/car_model/smoke_test_spcarnet_stage4.py` | PASS | Two-object MAP refinement smoke passed; free-space violation `0.0` in smoke. |
| `scripts/car_model/smoke_test_spcarnet_stage5.py` | PASS | K=4 sampling/reranking smoke passed; latent diversity finite. |

No smoke test was skipped for missing data.

## 6. Output Artifacts and Reported Metrics

Key artifacts present:

| Artifact | Status |
|---|---|
| `outputs/carnet/spcarnet/object_index_v1.json` | present |
| `outputs/carnet/spcarnet/autodecoder_v1/checkpoint_last.pt` | present |
| `outputs/carnet/spcarnet/autodecoder_v1/eval_train_64.json` | present |
| `outputs/carnet/spcarnet/autodecoder_v1/eval_train_16_iou_fix.json` | present |
| `outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt` | present |
| `outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json` | present |
| `outputs/carnet/spcarnet/map_refinement/val_50_default/refinement.json` | present |
| `outputs/carnet/spcarnet/multihypothesis/val_50_K1/K1.json` | present |
| `outputs/carnet/spcarnet/multihypothesis/val_50_K4/K4.json` | present |
| `outputs/carnet/spcarnet/multihypothesis/val_50_K8/K8.json` | present |
| `outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt` | present |
| `outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json` | present |

Metrics supported by actual JSONs:

| Stage | Artifact | Supported numbers |
|---|---|---|
| Stage 2 v1 | `autodecoder_v1/eval_train_64.json` | `mesh_extraction_success_rate=1.0`, `recon_chamfer_l1_mean=0.0661817`, `hidden_chamfer_l1_mean=0.0971487`, old sparse `mesh_iou_at_0.5_mean=0.488416` |
| Stage 2 v3 | `autodecoder_v3/eval/train_eval.json` | `mesh_extraction_success_rate=1.0`, `recon_chamfer_l1_mean=0.0691737`, `hidden_chamfer_l1_mean=0.101070`, `mesh_iou_at_0.5_shell_mean=0.913854` |
| Stage 3 | `posterior_encoder_v1/eval_val.json` | `mesh_extraction_success_rate=1.0`, `recon_chamfer_l1_mean=0.0663910`, `hidden_chamfer_l1_mean=0.0990754`, `visible_preservation_error_mean=0.0626813`, `free_space_violation_rate_mean=0.0335350`, `zero_corruption_recon_chamfer_l1_mean=0.0666457` |
| Stage 4 | `map_refinement/val_50_default/refinement.json` | `before_recon_chamfer_l1_mean=0.0714902`, `after_recon_chamfer_l1_mean=0.0690319`, `before_free_space_violation_rate_mean=0.0358203`, `after_free_space_violation_rate_mean=0.0146875` |
| Stage 5 K=1 | `multihypothesis/val_50_K1/K1.json` | `top1_score_recon_chamfer_l1_mean=0.0714693`, `oracle_best_of_k_recon_chamfer_l1_mean=0.0714693`, `top1_score_free_space_violation_rate_mean=0.0366406` |
| Stage 5 K=4 | `multihypothesis/val_50_K4/K4.json` | `top1_score_recon_chamfer_l1_mean=0.0734116`, `oracle_best_of_k_recon_chamfer_l1_mean=0.0669106`, `diversity_latent_l2_mean=3.90602` |
| Stage 5 K=8 | `multihypothesis/val_50_K8/K8.json` | `top1_score_recon_chamfer_l1_mean=0.0735006`, `oracle_best_of_k_recon_chamfer_l1_mean=0.0655285`, `diversity_latent_l2_mean=3.88172` |

Conclusion:

- The latest SP-CarNet documents are supported by local checkpoint / JSON artifacts.
- Stage 3 remains the strongest current deployable object-prior result.
- Stage 4 and Stage 5 artifacts preserve the documented safety / oracle-vs-reranker conclusions.

## 7. Blocking Issues

| Issue | Severity | Status | Action |
|---|---|---|---|
| Default shell Python lacks torch | medium | not blocking when using `mesh_splatting` env | Use `micromamba run -n mesh_splatting ...` for all future stages. |
| Dirty worktree before MeshPrior work | medium | not blocking for M1, but must be preserved | Do not overwrite existing modified `scripts/car_model/eval_spcarnet_multihypothesis.py`; record dirty state in future reports. |
| Submodule status dirty/unknown | low to medium | not blocking current Python smokes | Recheck before any CUDA extension rebuild or original MeshSplatting training. |
| Early file visibility mismatch during audit | low | resolved by repeated checks and passing smokes | No action unless it recurs. |

No hard blocker remains for moving to Prompt M1.

## 8. Recommendation

`PROCEED`

Stage M0 is complete:

1. Required docs were read or verified present.
2. Required code files exist.
3. Required imports and `compileall` pass in the project environment.
4. Stage 1-5 smoke tests pass.
5. Reported Stage 2/3/4/5 metrics are backed by local output artifacts.

Do not proceed to implementation stages yet. The next allowed step under `docs/prompts.md` is Prompt M1: write `docs/car_model/meshprior_stage1_scene_meshprior_RFC.md` without changing model code.

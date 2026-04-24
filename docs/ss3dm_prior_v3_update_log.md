# SS3DM Prior V3 Update Log

## Step 1 - Protocol and Code Hygiene Fixes + Strict Control Baseline

### Modified Files

- `ss3dm_prior/train.py`
- `ss3dm_prior/losses.py`
- `ss3dm_prior/reporting.py`
- `ss3dm_prior/eval.py`
- `ss3dm_prior/tools/audit_run_protocol.py`
- `configs/ss3dm_prior/train_v9_strict_control.yaml`
- `scripts/ss3dm_prior/train_v9_strict_control.sh`
- `tests/ss3dm_prior/test_train_entrypoint_wandb.py`
- `tests/ss3dm_prior/test_loss_weighting.py`
- `docs/ss3dm_prior_v3_plan.md`
- `docs/ss3dm_prior_v3_update_log.md`
- `docs/ss3dm_prior_v3_experiments.md`

### Design Rationale

- Make `wandb` override behavior explicit and testable so CLI invocation cannot silently change config-driven logging policy.
- Align hard-example loss weighting with the intrinsic-difficulty-aware training story by adding configurable `corruption` / `intrinsic` / `blend` sources.
- Promote protocol audit fields to first-class eval/report outputs so strict validity can be checked from saved artifacts instead of inferred manually.
- Separate synthetic corruption difficulty from intrinsic geometry difficulty in reporting to avoid paper-facing conclusions that conflate two different notions of hardness.
- Add a new strict v9 control config for publication-grade baseline runs while keeping model choice unchanged.

### Actual Commands

- `python -m ss3dm_prior.train --help`
- `python -m ss3dm_prior.eval --help`
- `python -m ss3dm_prior.tools.audit_run_protocol --help`
- `pytest tests/ss3dm_prior/test_train_entrypoint_wandb.py -q`
- `pytest tests/ss3dm_prior/test_loss_weighting.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/train.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/audit_run_protocol.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_train_entrypoint_wandb.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_loss_weighting.py"`

### Test Results

- `python -m ss3dm_prior.train --help`: passed
- `python -m ss3dm_prior.eval --help`: passed
- `python -m ss3dm_prior.tools.audit_run_protocol --help`: passed
- `pytest tests/ss3dm_prior/test_train_entrypoint_wandb.py -q`: passed (`2 passed`), with one environment-level CUDA driver warning during `torch.cuda.is_available()` probing
- `pytest tests/ss3dm_prior/test_loss_weighting.py -q`: passed (`2 passed`)
- `python -m py_compile ...`: passed

### Risk / TODO

- `grad_accum_steps: 2` is recorded in the new v9 control config, but trainer-side accumulation is not implemented in this step.
- Existing older reports or downstream readers may still expect `mean_corruption_severity`; Step 1 keeps it as an alias in grouped CSV output for compatibility.
- The current strict control launch script still uses the existing `model_v7_gain.yaml` control model; this is intentional for Step 1 because no new model work is allowed yet.

## Step 2 - Data and Supervision Upgrade + V3 Patch Semantics

### Modified Files

- `ss3dm_prior/data/patch_types.py`
- `ss3dm_prior/data/train_dataset.py`
- `ss3dm_prior/data/corruptions.py`
- `ss3dm_prior/data/teacher_patch_builder.py`
- `ss3dm_prior/data/teacher_patch_builder_v3.py`
- `ss3dm_prior/tools/build_teacher_patch_cache_v3.py`
- `ss3dm_prior/tools/check_teacher_patch_cache_v3.py`
- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/eval.py`
- `ss3dm_prior/metrics.py`
- `ss3dm_prior/reporting.py`
- `configs/ss3dm_prior/teacher_patch_v3.yaml`
- `tests/ss3dm_prior/test_corruption_schedule.py`
- `tests/ss3dm_prior/test_patch_cache_v3.py`
- `docs/ss3dm_prior_v3_plan.md`
- `docs/ss3dm_prior_v3_update_log.md`
- `docs/ss3dm_prior_v3_experiments.md`

### Design Rationale

- Add a fully parallel v3 cache path so multi-scale semantics and visible/hidden decomposition do not destabilize the existing v2 cache tooling.
- Keep patch ids deterministic across rebuilds by including tile id, scale id, and radius token in the v3 patch identifier.
- Store visible/hidden geometry splits directly in cache files so later model steps can consume them without recomputing geometry support online.
- Add epoch-driven corruption scheduling only at dataset level, which preserves fixed validation/eval corruption while enabling future curriculum training.
- Extend eval/reporting with visible reconstruction and hidden completion metrics now, even before any new v3 model is introduced.

### Actual Commands

- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v3 --help`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v3 --help`
- `pytest tests/ss3dm_prior/test_corruption_schedule.py -q`
- `pytest tests/ss3dm_prior/test_patch_cache_v3.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/data/patch_types.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/corruptions.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/train_dataset.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/teacher_patch_builder.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/data/teacher_patch_builder_v3.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/build_teacher_patch_cache_v3.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/check_teacher_patch_cache_v3.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/metrics.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_corruption_schedule.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_patch_cache_v3.py"`
- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v3 --manifest "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/manifests/ss3dm_raw_manifest.json" --split_config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/splits/default_town_split.yaml" --config "/data2/peilincai/mesh-splatting/configs/ss3dm_prior/teacher_patch_v3.yaml" --observed_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/observed_cache" --town_mesh_cache_root "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/town_mesh_cache" --out_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v3_debug" --subsets train --debug_max_sequences 1 --debug_max_tiles_per_sequence 1 --num_workers 1 --seed 0`
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v3 --patch_cache_dir "/data2/peilincai/mesh-splatting/outputs/ss3dm_prior/teacher_patch_cache_v3_debug" --num_visualizations 2 --seed 0`

### Test Results

- `python -m ss3dm_prior.tools.build_teacher_patch_cache_v3 --help`: passed
- `python -m ss3dm_prior.tools.check_teacher_patch_cache_v3 --help`: passed
- `pytest tests/ss3dm_prior/test_corruption_schedule.py -q`: passed (`2 passed`)
- `pytest tests/ss3dm_prior/test_patch_cache_v3.py -q`: passed (`3 passed`)
- `python -m py_compile ...`: passed
- Real v3 debug build: passed, `written_patches: 2`
- Real v3 checker: passed, produced at least 3 local PNGs including two semantic views and one multi-scale comparison view

### Risk / TODO

- The real debug build produced only `scale_1` and `scale_2`; the smallest `scale_0` patches did not pass current validity thresholds on that sample sequence.
- Visible/hidden clean point sets are variable-length and therefore intentionally collated as lists instead of dense tensors; later model steps will need to preserve that convention or add explicit padding.
- Eval/report now expose visible and hidden geometry metrics, but no new v3 model has been added yet to optimize for them directly.

## Step 3 - V9 Wide Stronger Capacity Baseline

### Modified Files

- `ss3dm_prior/models/pointnet.py`
- `ss3dm_prior/models/hybrid_patch_prior_v2.py`
- `ss3dm_prior/models/patch_denoiser.py`
- `ss3dm_prior/engine/trainer.py`
- `configs/ss3dm_prior/model_v9_wide.yaml`
- `configs/ss3dm_prior/train_v9_wide.yaml`
- `scripts/ss3dm_prior/train_v9_wide.sh`
- `tests/ss3dm_prior/test_model_v9_wide_forward.py`
- `docs/ss3dm_prior_v3_plan.md`
- `docs/ss3dm_prior_v3_update_log.md`
- `docs/ss3dm_prior_v3_experiments.md`

### Design Rationale

- Add a stronger baseline by widening the existing hybrid path rather than introducing a new architecture family.
- Keep the stable public entrypoint as `LocalPatchDenoiser` and extend dispatch with `hybrid_v2_wide`, which preserves trainer compatibility.
- Reuse the current forward output schema so losses, metrics, and trainer logic do not require special-case handling.
- Restrict the change to wider MLPs, optional residual MLP blocks, optional layer norm, and optional dropout so the result remains a clean “capacity only” answer to whether the model is too small.

### Actual Commands

- `pytest tests/ss3dm_prior/test_model_v9_wide_forward.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/pointnet.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/hybrid_patch_prior_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/patch_denoiser.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/metrics.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_model_v9_wide_forward.py"`
- `python - <<'PY' ... real v3 patch wide forward ... PY`

### Test Results

- `pytest tests/ss3dm_prior/test_model_v9_wide_forward.py -q`: passed (`2 passed`)
- `python -m py_compile ...`: passed
- Real v3 patch batch forward: passed with:
  - `patch_id: Town01__1000_streetsurf__tile_000000__scale_01__r4p00m`
  - `recon_points: (1, 3072, 3)`
  - `query_occupancy_logits: (1, 1280)`
  - `retrieval_embedding: (1, 256)`
  - `intrinsic_difficulty_pred: (1,)`

### Risk / TODO

- `hybrid_v2_wide` increases decoder and codebook size substantially, so actual training memory may become the next bottleneck even if forward tests pass on CPU.
- The current trainer visualization path treats `hybrid_v2` and `hybrid_v2_wide` as the same family, which is intentional, but later wide-only diagnostics may still be useful.
- This step only answers the “is the model too small?” question structurally; it does not yet provide full training/eval ablation results against the smaller baseline.

## Step 4 - V10 Cross-Attention Hybrid

### Modified Files

- `ss3dm_prior/models/attention_blocks.py`
- `ss3dm_prior/models/cross_attention_patch_prior_v10.py`
- `ss3dm_prior/models/patch_denoiser.py`
- `ss3dm_prior/models/hybrid_patch_prior_v2.py`
- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/eval.py`
- `configs/ss3dm_prior/model_v10_crossattn.yaml`
- `configs/ss3dm_prior/train_v10_crossattn.yaml`
- `scripts/ss3dm_prior/train_v10_crossattn.sh`
- `tests/ss3dm_prior/test_model_v10_crossattn_forward.py`
- `docs/ss3dm_prior_v3_plan.md`
- `docs/ss3dm_prior_v3_update_log.md`
- `docs/ss3dm_prior_v3_experiments.md`

### Design Rationale

- Keep `LocalPatchDenoiser` as the stable public entrypoint and extend its dispatch with a new `v10_cross_attention_hybrid` implementation.
- Use latent cross-attention over compact token sets instead of full point-wise self-attention, so the model can strengthen conditional geometry reasoning without paying the coupling cost of a full point transformer rewrite.
- Let visible, hidden, observed, corrupted, and query token sets flow into the same latent query bank, which matches the local patch setting better than global all-pairs attention.
- Preserve the existing output contract and loss interface so trainer, eval, and report code can remain largely unchanged.
- Add explicit trainer/eval support for moving collated variable-length visible/hidden tensors onto device so the new model can consume Step 2 semantics directly.

### Actual Commands

- `pytest tests/ss3dm_prior/test_model_v10_crossattn_forward.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/attention_blocks.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/cross_attention_patch_prior_v10.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/pointnet.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/hybrid_patch_prior_v2.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/patch_denoiser.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_model_v10_crossattn_forward.py"`
- `python - <<'PY' ... print v9/v10 parameter counts ... PY`
- `python - <<'PY' ... real v3 patch cross-attention forward ... PY`

### Test Results

- `pytest tests/ss3dm_prior/test_model_v10_crossattn_forward.py -q`: passed (`2 passed`)
- `python -m py_compile ...`: passed
- Parameter count comparison:
  - `hybrid_v2_wide_params: 40052874`
  - `v10_cross_attention_hybrid_params: 52579338`
- Real v3 patch batch forward: passed with:
  - `patch_id: Town01__1000_streetsurf__tile_000000__scale_01__r4p00m`
  - `recon_points: (1, 3072, 3)`
  - `query_occupancy_logits: (1, 1280)`
  - `retrieval_embedding: (1, 256)`
  - `intrinsic_difficulty_pred: (1,)`

### Risk / TODO

- This step validates forward compatibility only; it does not yet provide full train/eval ablations against `hybrid_v2_wide`.
- `v10_cross_attention_hybrid` is materially larger than `hybrid_v2_wide`, so training-time memory and throughput may become a practical bottleneck before architecture quality is fully measured.
- Visible/hidden token conditioning is now wired through trainer/eval, but only the new cross-attention model actually consumes those optional tensors; later steps may decide whether broader family support is worth adding.

## Step 5 - Trainer Stabilization + Eval/Report/Ablation Upgrade

### Modified Files

- `ss3dm_prior/engine/checkpoint.py`
- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/eval.py`
- `ss3dm_prior/reporting.py`
- `ss3dm_prior/metrics.py`
- `ss3dm_prior/viz/render_patch_panels.py`
- `ss3dm_prior/tools/run_ablation_suite.py`
- `ss3dm_prior/tools/aggregate_ablation_results.py`
- `configs/ss3dm_prior/train_v10_crossattn.yaml`
- `scripts/ss3dm_prior/run_v3_ablation_suite.sh`
- `tests/ss3dm_prior/smoke_v3_utils.py`
- `tests/ss3dm_prior/test_trainer_v3_smoke.py`
- `tests/ss3dm_prior/test_eval_v3_smoke.py`
- `docs/ss3dm_prior_v3_plan.md`
- `docs/ss3dm_prior_v3_update_log.md`
- `docs/ss3dm_prior_v3_experiments.md`

### Design Rationale

- Upgrade the existing trainer rather than branching to a new pipeline, so `v9_wide` and `v10_cross_attention_hybrid` stay compatible with the same train/eval entrypoints.
- Make stability features fully config-driven: grad accumulation, EMA, staged curriculum, and corruption schedule should be orthogonal rather than hard-wired to one model family.
- Add a paper-facing `best_paper.pt` composite checkpoint so publication selection criteria are explicit and reproducible.
- Push Step 2 visible/hidden/free-space semantics all the way through validation and reporting, so the evaluation story goes beyond single Chamfer and generic denoise gain.
- Replace the older v2 ablation suite layout with a Step 5 v3 runner focused on the five requested variants.

### Actual Commands

- Attempted: `pytest tests/ss3dm_prior/test_trainer_v3_smoke.py -q`
- Attempted: `pytest tests/ss3dm_prior/test_eval_v3_smoke.py -q`
- Attempted: `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/checkpoint.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/metrics.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_patch_panels.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/run_ablation_suite.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/tools/aggregate_ablation_results.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/smoke_v3_utils.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_trainer_v3_smoke.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_eval_v3_smoke.py"`
- `ReadLints(paths=[Step 5 modified files])`

### Test Results

- `ReadLints(...)`: passed, no linter diagnostics on the modified Step 5 files.
- Shell-based validation could not be trusted in this agent session:
  - direct shell invocations returned immediately without producing side effects or stdout
  - shell subagent startup also timed out
- Because of that environment issue, `pytest` and `py_compile` were prepared and attempted but are not marked as passed in this log.
- New smoke tests and synthetic v3 fixtures were added for:
  - tiny debug train with EMA / grad accumulation / best_paper checkpoint coverage
  - tiny debug eval with new metrics and new PNG categories

### Risk / TODO

- Step 5 code paths are lint-clean, but the current agent session could not execute shell-based validation, so `pytest` and `py_compile` should be rerun in a healthy terminal session before treating this step as fully verified.
- `best_paper.pt` currently prefers EMA weights at eval time when EMA is enabled; this is intentional, but future comparisons should keep that selection rule explicit in experiment notes.
- The new ablation runner assumes access to both baseline and multiscale patch caches for the full non-debug suite; if only one cache exists, the multiscale variants will need that path provided explicitly.

## Step 6 - Optional Latent Flow Exploratory Branch

### Modified Files

- `ss3dm_prior/models/latent_flow_patch_prior_v11.py`
- `ss3dm_prior/models/patch_denoiser.py`
- `ss3dm_prior/losses.py`
- `ss3dm_prior/engine/trainer.py`
- `ss3dm_prior/eval.py`
- `ss3dm_prior/reporting.py`
- `ss3dm_prior/viz/render_patch_panels.py`
- `configs/ss3dm_prior/model_v11_latent_flow.yaml`
- `configs/ss3dm_prior/train_v11_latent_flow.yaml`
- `scripts/ss3dm_prior/train_v11_latent_flow.sh`
- `tests/ss3dm_prior/test_model_v11_latent_flow_forward.py`
- `docs/ss3dm_prior_v3_plan.md`
- `docs/ss3dm_prior_v3_update_log.md`
- `docs/ss3dm_prior_v3_experiments.md`

### Design Rationale

- Keep the deterministic `v10` path unchanged and add `v11_latent_flow_hybrid` as a separate exploratory branch so the paper baseline remains stable.
- Use latent flow matching in conditional latent space instead of raw point diffusion, which keeps coupling low and reuses the current cross-attention encoder, VQ/prototype conditioning, decoder, and occupancy head.
- Train the generative branch on hidden-region residual completion rather than full-patch generation, matching the hidden-completion uncertainty modeling goal more directly.
- Push candidate sampling and reranking into eval so the report can compare deterministic vs stochastic completion at `K=1/4/8` without rewriting the trainer contract.

### Actual Commands

- `pytest tests/ss3dm_prior/test_model_v11_latent_flow_forward.py -q`
- `python -m py_compile "/data2/peilincai/mesh-splatting/ss3dm_prior/models/latent_flow_patch_prior_v11.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/models/patch_denoiser.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/losses.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/engine/trainer.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/eval.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/reporting.py" "/data2/peilincai/mesh-splatting/ss3dm_prior/viz/render_patch_panels.py" "/data2/peilincai/mesh-splatting/tests/ss3dm_prior/test_model_v11_latent_flow_forward.py"`

### Test Results

- `pytest tests/ss3dm_prior/test_model_v11_latent_flow_forward.py -q`: passed (`3 passed`)
- `python -m py_compile ...`: passed
- IDE lint diagnostics on the Step 6 modified files were clean.

### Risk / TODO

- `v11_latent_flow_hybrid` is exploratory and currently smoke-tested only; it still needs real train/eval comparison against deterministic `v10`.
- The current reranker is heuristic and intentionally paper-facing rather than learned; later work can ablate its weights once real stochastic runs exist.
- The new stochastic metrics can be sparse when no free-space-safe candidate exists, so downstream analysis should treat `free_space_safe_best_of_k = NaN` as “no safe candidate found,” not as zero quality.

# SPCarNet Paper-Loop Continuation Log: v83 + Subagent Audit

Date: 2026-06-24

This log records the continuation after the current complete method/report pass.
It is intentionally separated from the main report so the README can keep a
clean headline while unfinished probes remain auditable.

## Current Proven Headline

The promoted presentation-safe endpoint is still **SPCarNet Phase-J**:

- local Mip-NeRF360 full9 selected-clean MeshSplatting comparison: `9 / 9`
  scene-level PSNR/SSIM/LPIPS strict wins;
- held-out view strict wins: `244 / 246`;
- mean delta: `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS;
- mean triangle reduction: `7.6479%`;
- geometry-safe scenes: `9 / 9`.

The main report with quantitative and qualitative evidence is:

```text
docs/car_model/6-24-SPCarNet-Current-Complete-Method-Experiment-Report-With-Render-Comparisons.zh.md
```

Recommended visual evidence:

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
```

## Current Representation-Level Anchor

The fair fixed representation-level anchor remains:

```text
v56/v64/v79 counter anchor:
PSNR  26.756130219
SSIM   0.862126231
LPIPS  0.251691371
```

v82b is the only recent counter probe with a strict all-metric micro-win:

```text
v82b counter:
PSNR  26.756137848
SSIM   0.862126350
LPIPS  0.251690656
```

Delta versus anchor:

```text
dPSNR  +0.000007629
dSSIM  +0.000000119
dLPIPS -0.000000715
```

Verdict: **not promoted**. The margin is too small and only `counter` has been
validated. It is a seed for hard-triad/full9 validation, not a paper endpoint.

## v83 Probe Status

Completed probe:

```text
v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter
root: /dev/shm/peilincai_spcarnet_v83_patchmix_20260624
W&B run: mdjxzdyw
GPU: CUDA_VISIBLE_DEVICES=2
```

Final result:

```text
PSNR  26.756147385
SSIM   0.862125337
LPIPS  0.251688808
```

Delta versus v56/v64/v79 anchor
`26.756130219 / 0.862126231 / 0.251691371`:

```text
dPSNR  +0.000017166
dSSIM  -0.000000894
dLPIPS -0.000002563
```

Delta versus v82b counter
`26.756137848 / 0.862126350 / 0.251690656`:

```text
dPSNR  +0.000009537
dSSIM  -0.000001013
dLPIPS -0.000001848
```

Verdict:

- accepted by policy-val with selected alpha `0.5`;
- patch-mixture teacher basis guard fell back to legacy teacher basis;
- PSNR and LPIPS improve, but SSIM regresses;
- therefore v83 is **not promoted** and should be recorded as a mixed
  representation-level diagnostic.

Persistent small artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/counter_v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/counter_v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/counter_v83_teacher_patchmix_facealpha_localpatch_hybrid_tex32_support4096_counter_region_texture_adapter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v83_patchmix_facealpha_localpatch_counter_20260624/logs/apply_metrics_counter.log
```

## Subagent Coordination

Parallel subagents were launched for:

1. repo/codebase mapping;
2. method-gap analysis;
3. implementation;
4. experiment execution;
5. review + paper-story synthesis.

They are expected to return concrete code paths, fairness gaps, implementation
patches, experiment status, and reviewer-style paper-story risks. Any returned
implementation must be reviewed before promotion because the worktree is dirty
and multiple scripts have accumulated prior experimental changes.

Returned subagent findings were integrated as follows:

- paper-story review: mentor/PPT reporting is allowed, but only as "strong
  progress plus honest gap"; do not claim a completed paper loop;
- method-gap analysis: the current headline is still a render-time
  self-auditing adapter, so representation-level claims require new hard-triad
  and full9 evidence;
- repo mapping: the safest next implementation scope is
  `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` plus
  `scripts/car_model/run_l1risk_fairnoop_scene.py`;
- experiment execution: v83 completed as mixed counter evidence and is not
  promoted because SSIM regressed against the anchor.

## Integrated Engineering Change

A real train/eval pipeline change has been added to reduce future candidate
search cost and improve auditability without changing default results:

```text
scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py
scripts/car_model/run_l1risk_fairnoop_scene.py
```

New flags:

```text
--enable_policy_candidate_dominance_pruning
--policy_candidate_early_stop_mode {none,first_accepted}
```

Default behavior remains unchanged:

- dominance pruning is off unless explicitly enabled;
- early-stop is `none` unless explicitly enabled;
- `first_accepted` is automatically disabled when target-support candidate
  selection or prior-bin hybrid requires a full candidate pool;
- audit metadata records planned/pruned/executed/skipped candidates and the
  selected candidate.

Verification:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py

git diff --check -- \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py \
  scripts/car_model/run_l1risk_fairnoop_scene.py \
  README.md README.zh.md \
  docs/car_model/6-24-SPCarNet-Claim-Boundary-And-Paper-Gap.zh.md \
  docs/car_model/6-24-SPCarNet-PaperLoop-Continuation-v83-And-Subagents.md

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_l1risk_fairnoop_scene.py --help

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py --help
```

All four checks passed. This is an engineering-loop improvement, not a promoted
method result until a flagged run is validated.

## Immediate Engineering Bottlenecks

- `/data` is full (`456M` free at the last check); large runs must write to
  `/dev/shm`, and only small artifacts should be persisted.
- Recent probes are CPU-policy-val bound even when GPU memory is mostly free.
  Candidate pruning or early-stop audit metadata is therefore an engineering
  priority.
- README currently contains too much historical material. It is usable for
  provenance, but mentor/PPT-facing claims should point to the clean report and
  keep negative probes out of the headline.

## Next Fair Validation Commands

Use these only after v83 or a later fixed policy passes the counter gate.

Hard-triad first:

```bash
WANDB_DIR=/dev/shm/wandb_spcarnet_hardtriad \
WANDB_MODE=online \
CUDA_VISIBLE_DEVICES=<low_or_mid_gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_l1risk_fairnoop_scene.py \
  --scene counter \
  --gpu <low_or_mid_gpu> \
  --output_root /dev/shm/peilincai_spcarnet_<tag>_hardtriad \
  --tag <fixed_policy_tag> \
  --wandb_project SPCarNet \
  --wandb_group <fixed_policy_tag> \
  --wandb_mode online \
  --force
```

Then repeat the same fixed policy on:

```text
counter, kitchen, bonsai
```

Full9 only after hard-triad is non-regressive:

```text
bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai
```

## Current Closure Verdict

Engineering loop: **NOT COMPLETE**.

Paper loop: **NOT COMPLETE**.

The current Phase-J story is strong enough for a mentor/PPT report, but the
paper endpoint is still limited by the gap between a strong render-time
self-auditing adapter and a broadly validated representation-level method.

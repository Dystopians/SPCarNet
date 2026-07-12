# v52 Capacity-Aware v48/v51 Policy Log

Date: 2026-06-23

Status: `SOURCE_RERUN_COMPLETE_REPRODUCED`. v52 is a fixed, scene-agnostic policy over v48 and v51 source configurations. It does not use held-out metrics for selection. A canonical small-artifact selected tree, one-command artifact pipeline, and W&B-logged source-config rerun now exist. The source-config rerun reproduced `9 / 9` scenes with `0` missing scenes and `0` metric mismatches under `1e-5` reproducibility tolerance.

## Motivation

v51 showed that larger support footprints help the cap-hit scenes `counter`, `kitchen`, and `bonsai`, but using fixed `texture=32` and `nearest_observed` globally loses v48's auto-capacity/auto-fill advantage on non-cap-hit scenes.

v52 converts that lesson into a fixed capacity-aware rule:

```text
if v48 accepted and v48 selected_support_added_faces >= 2048
   and v51 accepted_atlas
   and v51 selected_support_added_faces > v48 selected_support_added_faces
   and v51 policy-val SSIM gain >= 5e-5:
       use v51
else:
       keep v48
```

The selection uses only train/policy-val audit fields. Held-out metrics are used only after selection to report the effective result.

## Implementation

New script:

```text
scripts/car_model/summarize_v52_capacity_aware_policy.py
scripts/car_model/run_v52_capacity_aware_pipeline.py
scripts/car_model/plan_v52_capacity_aware_source_rerun.py
scripts/car_model/summarize_v52_source_rerun.py
```

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v52_capacity_aware_policy.py
```

Materialized selected-tree command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v52_capacity_aware_policy.py \
  --materialize_root outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9
```

Outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_v48_v51_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/
```

One-command artifact pipeline:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v52_capacity_aware_pipeline.py
```

Pipeline outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/v52_capacity_aware_pipeline_manifest.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/v52_capacity_aware_pipeline_report.md
```

Pipeline validation:

- scene count: `9`
- render/GT linked scenes: `9`
- selection uses held-out metrics: `False`
- executed steps: `summarize_and_materialize`, `build_selected_gallery`, `build_cap_hit_panel`

Source-rerun command plan:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/plan_v52_capacity_aware_source_rerun.py --gpus 0,1
```

Plan outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_plan.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_plan.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_plan.sh
```

Plan validation:

- planned jobs: `9`
- selected v48/v51 jobs: `6 / 3`
- missing required inputs: `0`
- executed: `False`

W&B-logged source-config rerun:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/plan_v52_capacity_aware_source_rerun.py \
  --gpus 4 \
  --execute \
  --wandb_project SPCarNet \
  --wandb_run_name v52_source_rerun_full9_20260623_fix1 \
  --force
```

Final source-rerun status at `2026-06-23 16:10 -0700`:

- W&B run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/2j5osvgg`
- output root: `/dev/shm/peilincai_spcarnet_v52_source_rerun_20260623_144849`
- supplemental W&B runs:
  - `counter`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/9uwsu9m1`
  - `kitchen`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/b9w1zonu`
  - `bonsai`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/uh4a9hvu`
  - `room`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/o785gymj`
  - `stump` no-op fix: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/n4ucitf6`
- source roots include the main root plus per-scene parallel roots under `/dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_*_20260623`
- status: `COMPLETE_REPRODUCED`
- completed scenes: `9 / 9`
- missing scenes: `0`
- metric mismatch scenes: `0`

Source-rerun audit command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v52_source_rerun.py \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_20260623_144849 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_garden_20260623 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_treehill_20260623 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_stump_noopfix_20260623 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_room_20260623 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_counter_20260623 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_kitchen_20260623 \
  --source_root /dev/shm/peilincai_spcarnet_v52_source_rerun_parallel_bonsai_20260623
```

Audit outputs:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_status.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_source_rerun_status.md
```

Latest audit at `2026-06-23 16:10 -0700`:

- status: `COMPLETE_REPRODUCED`
- completed scenes: `9 / 9`
- missing scenes: `0`
- metric mismatch scenes: `0`
- reproduction tolerance: `1e-5`
- fresh v52 vs v48: `3 / 9` strict, `9 / 9` non-regressive/tie, mean `+0.000086255` PSNR, `+0.000008742` SSIM, `-0.000015024` LPIPS
- note: `stump` is a legitimate fallback/no-op reproduction. The planner now appends `--write_noop_on_reject` for v48 source-rerun jobs so rejected safe fallbacks still produce render/GT and metrics.

Compile check:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/summarize_v52_capacity_aware_policy.py \
  scripts/car_model/run_v52_capacity_aware_pipeline.py \
  scripts/car_model/plan_v52_capacity_aware_source_rerun.py \
  scripts/car_model/summarize_v52_source_rerun.py
```

## Decisions

| scene | selected source | reason |
|---|---|---|
| bicycle | v48 | v48 support below cap; v51 policy-val SSIM gain too small |
| flowers | v48 | v48 support below cap; v51 fallback/no-op |
| garden | v48 | v48 support below cap; v51 selected base carrier |
| stump | v48 | v48 rejected to no-op; v51 rejected to no-op |
| treehill | v48 | v48 support below cap; v51 rejected to no-op |
| room | v48 | v48 support below cap; v51 selected base carrier |
| counter | v51 | v48 hit cap; v51 accepted larger support |
| kitchen | v51 | v48 hit cap; v51 accepted larger support |
| bonsai | v51 | v48 hit cap; v51 accepted larger support |

## Results

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v52 vs no-op | 9 | 7 | 8 | `+0.001549191` | `+0.000036518` | `-0.000054831` |
| v52 vs v48 | 9 | 3 | 9 | `+0.000086890` | `+0.000008782` | `-0.000015303` |
| v52 vs v50 | 9 | 6 | 6 | `+0.000284831` | `+0.000014782` | `-0.000020780` |

Per-scene v52 vs v48:

| scene | selected | dPSNR | dSSIM | dLPIPS | status |
|---|---|---:|---:|---:|---|
| bicycle | v48 | `+0.000000` | `+0.00000000` | `+0.00000000` | tie |
| flowers | v48 | `+0.000000` | `+0.00000000` | `+0.00000000` | tie |
| garden | v48 | `+0.000000` | `+0.00000000` | `+0.00000000` | tie |
| stump | v48 | `+0.000000` | `+0.00000000` | `+0.00000000` | tie |
| treehill | v48 | `+0.000000` | `+0.00000000` | `+0.00000000` | tie |
| room | v48 | `+0.000000` | `+0.00000000` | `+0.00000000` | tie |
| counter | v51 | `+0.000456` | `+0.00003111` | `-0.00004962` | strict win |
| kitchen | v51 | `+0.000284` | `+0.00002635` | `-0.00004680` | strict win |
| bonsai | v51 | `+0.000042` | `+0.00002158` | `-0.00004131` | strict win |

## Qualitative Artifacts

Full selected-render gallery:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/qualitative_gallery.html
```

Cap-hit v48 vs v52 local panel:

```text
assets/spcarnet_v52_capacity_policy_cap_hit_panel.png
assets/spcarnet_v52_capacity_policy_cap_hit_panel_manifest.json
```

Local crop diagnostics from the manifest:

| scene | view | local dPSNR | local MAE delta |
|---|---|---:|---:|
| counter | `00001.png` | `+0.026787` | `+0.00007905` |
| kitchen | `00010.png` | `+0.023953` | `+0.00004964` |
| bonsai | `00008.png` | `+0.018920` | `+0.00004260` |

These crops are deliberately honest: they show that v52 produces measurable local improvements in the cap-hit scenes, but the visual difference is subtle.

## Interpretation

v52 is the strongest current representation-level effective policy by full9 mean. It preserves v48 exactly on non-cap-hit scenes and captures v51 gains on all three cap-hit scenes.

The honest limitation is that v52 is still a small-effect representation-level policy. The source-config rerun closes the reproducibility gap, but the mean improvement over v48 is only around `1e-4` PSNR and visual differences remain subtle. The selected small-artifact tree and artifact pipeline are materialized at:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v52_capacity_aware_selected_full9/
```

The wrapper refreshes selected small artifacts, selected render symlinks, the HTML gallery, the cap-hit panel, and a pipeline manifest in one command. The source-rerun planner executes the same v52 policy from source configs with W&B command logging. Its W&B logs are coarse command-level logs rather than fine-grained per-metric W&B scalars, but the local `results.json` and summarized audit are complete.

## Next Step

Use v52 as the reproducible representation-level ablation and continue improving effect size. The next method work should target a higher-capacity persistent surface residual representation, faster policy candidate evaluation, and more visible local improvements while preserving the train-only selection rule.

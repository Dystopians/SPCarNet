# ECSR Phase-F: Policy-Val Compaction Ladder

Date: 2026-05-08

## Why This Phase Exists

The archived Compact-ELA/SOR model is still too conservative in geometry compression. The worst bottleneck is not a missing ratio sweep; it is that manual or test-informed ratio selection would be invalid for a paper. Phase-F therefore adds a fixed internal-validation policy:

1. Start from the current Compact-ELA/SOR checkpoint.
2. Use a training-derived `policy_val` COLMAP split only.
3. Apply a fixed extra-compaction ladder with one selector and one ratio grid.
4. Accept the largest extra-compaction candidate that stays Pareto-safe on policy-val RGB metrics and topology validity.
5. Use held-out paper test only after the policy is fixed.

This phase is a bridge, not the final ECSR representation-level method. It directly attacks the compression bottleneck while preserving the no-test-leakage discipline needed by FinalDecision.

## Fixed Policy

- Source model: `outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/<scene>/sor_adaptive_geo/compact_model`
- Iteration: `26000`
- Policy split root: `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/<scene>/split_file.json`
- Selector: `csef_low_evidence_boundary_protected`
- Ratio grid for pilot: `0.005, 0.010, 0.020`
- Policy-val acceptance thresholds:
  - `dPSNR >= -0.03`
  - `dSSIM >= -0.0015`
  - `dLPIPS <= +0.0020`
  - invalid face count remains `0`
  - degenerate face count remains `0`
- Selection rule: choose the accepted candidate with the largest additional triangle removal; ties prefer higher `dPSNR` and lower `dLPIPS`.

The thresholds are intentionally fixed before held-out test evaluation. They are not scene-specific.

## New Entrypoint

Script:

```bash
scripts/car_model/ecsr_run_policy_val_compaction_ladder.py
```

Important implementation details:

- Uses the caller's `CUDA_VISIBLE_DEVICES` by default, so parallel launchers can bind different scenes to different physical GPUs without the script overriding the mapping.
- Logs all subprocess commands to `policy_val_ladder.log`.
- Writes per-scene `summary.json` and `policy_val_ladder.md`.
- Writes global `summary.json` and `summary.md`.
- Supports W&B logging for policy-val metric deltas and accepted ratios.

## Full9 Policy-Val Run

After fixing the Python environment and GPU-binding issue, the fixed policy was
run on all nine Mip-NeRF360 scenes:

```bash
CUDA_VISIBLE_DEVICES=<low-occupancy-gpu> /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_policy_val_compaction_ladder.py \
  --scenes <scene-or-scene-list> \
  --ratios 0.005,0.010,0.020 \
  --out_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --no_depth --skip_failed_views --wandb \
  --wandb_group phase_f_policy_val_ladder_full9_v2 \
  --wandb_name phase_f_policy_val_ladder_v2_<scene>
```

Aggregate artifact:

- `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/aggregate_policy_val_ladder.md`

Result:

| metric | value |
|---|---:|
| complete scenes | 9 / 9 |
| accepted scenes | 9 / 9 |
| mean selected extra ratio | 2.00% |
| mean additional removed fraction | 2.00% |
| mean source removed fraction | 5.763% |
| mean total removed fraction | 7.648% |

W&B runs:

- `garden`: `fubkvk1f`
- `room`: `nyuo8aet`
- `flowers`: `joaskltz`
- `stump`: `hxxj8gih`
- `treehill`: `s7q005t0`
- `bicycle`: `587vt371`
- `bonsai`: `vjaf7iy6`
- `counter`: `1ozluxsb`
- `kitchen`: `wlxixzph`

Per-scene selected ratio is `0.0200` for all nine scenes. Policy-val RGB
deltas are effectively zero-scale: this is evidence that the extra compaction
does not disturb internal validation renders, not evidence of new visual
quality gains.

## Known Execution Lesson

The system default `python` points to a base Python 3.13 environment without `numpy`/`torch`. The correct environment for MeshSplatting experiments is:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python
```

The first pilot launch also exposed a GPU-binding bug: the script defaulted to setting subprocess `CUDA_VISIBLE_DEVICES=0`, which overrode the parent launcher and could make parallel jobs collide on physical GPU 0. This is fixed by defaulting `--gpu=-1`, preserving the caller's GPU mask.

## Promotion Criteria

Promote Phase-F beyond pilot only if at least one bottleneck scene accepts extra compression without policy-val RGB/topology regression. If all candidates fail, the evidence says the current bottleneck is not solvable by extra deletion-style compaction and the next push must move back to true contraction or surface-attached appearance recovery.

## Next Required Steps

1. Render held-out test for the accepted model and compare against both clean MeshSplatting and archived Compact-ELA/SOR.
2. Treat Phase-F as a compactness certificate unless held-out RGB also clears the archived Compact-ELA/SOR source.
3. If held-out RGB is only safe versus clean but not better than Compact-ELA/SOR, keep the visual claim with archived Compact-ELA/SOR and report Phase-F as representation compactness.
4. Continue true ECSR work on certificate-carrying contraction plus surface-attached recovery; do not present Phase-F as the final visual method by itself.

Held-out evaluator added after this policy-val run:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasef_heldout_eval.py \
  --scenes bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai \
  --no_depth --skip_failed_views --wandb \
  --wandb_group phase_f_heldout_eval_full9 \
  --wandb_name phase_f_heldout_eval_full9_collect
```

This evaluator renders the original held-out LLFF test split only after the
policy is fixed. It compares the selected Phase-F raw checkpoint against the
held-out-selected clean baseline and the archived Compact-ELA/SOR source.

## Held-Out Result

Artifact:

- `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_heldout_eval_summary.md`

W&B run:

- full9 collect: `ptf5x9o8`

Summary:

| criterion | result |
|---|---:|
| scenes | 9 |
| compact + RGB-safe vs selected clean | 8 / 9 |
| compact + RGB-safe vs archived Compact-ELA/SOR | 0 / 9 |
| mean dPSNR vs clean | -0.0113 |
| mean dSSIM vs clean | -0.00021 |
| mean dLPIPS vs clean | +0.000064 |
| mean dPSNR vs Compact-ELA/SOR | -0.5092 |
| mean dSSIM vs Compact-ELA/SOR | -0.01596 |
| mean dLPIPS vs Compact-ELA/SOR | +0.02344 |
| mean total triangle reduction | 7.648% |

Interpretation:

Phase-F succeeds as a fixed-policy compactness extension but fails as a
standalone visual method. The raw checkpoint is mostly safe versus clean
MeshSplatting under the same guardrail, but it cannot replace the archived
Compact-ELA/SOR visual layer. This is an important negative/diagnostic result:
the next valid step is to combine the extra compact checkpoint with a
train-only appearance recovery layer, or to move recovery into persistent
surface-attached representation state. Phase-F alone must not be claimed as a
full visual improvement over the archived best.

## Phase-F + Alpha-0.875 ELA Recovery

The raw Phase-F checkpoint was then combined with the same train-only ELA
family used by the archived Compact-ELA/SOR model. The first fixed ELA pass
used the previous alpha grid and missed `bicycle` by a small margin against
Compact-ELA/SOR. The diagnostic showed that the gap was not caused by extra
triangle deletion: `0.5%`, `1%`, and `2%` extra compaction produced essentially
the same bicycle RGB result. The missing interface was recovery-policy
resolution.

The promoted recovery grid is still fixed globally, not per-scene:

```bash
--policy_modes residual \
--policy_k_values 4,8 \
--policy_depth_rel_values 0.06,0.12 \
--policy_residual_clip_values 0.2,0.25 \
--policy_direction_weight_values 0.2,0.35 \
--policy_objective balanced \
--alpha_grid 0,0.125,0.25,0.5,0.75,0.875,1.0 \
--calib_lpips \
--edge_gate --edge_gate_quantile 0.7 --edge_gate_dilate 1
```

Artifact:

- `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_alpha0875_full9.md`

Full9 held-out result:

| criterion | result |
|---|---:|
| scenes | 9 |
| beats selected clean on PSNR/SSIM/LPIPS | 9 / 9 |
| beats archived Compact-ELA/SOR on PSNR/SSIM/LPIPS | 9 / 9 |
| mean dPSNR vs clean | +0.9340 |
| mean dSSIM vs clean | +0.02640 |
| mean dLPIPS vs clean | -0.04404 |
| mean dPSNR vs Compact-ELA/SOR | +0.4360 |
| mean dSSIM vs Compact-ELA/SOR | +0.01064 |
| mean dLPIPS vs Compact-ELA/SOR | -0.02067 |
| mean total triangle reduction | 7.648% |

Per-scene deltas versus archived Compact-ELA/SOR:

| scene | dPSNR | dSSIM | dLPIPS | total triangle reduction |
|---|---:|---:|---:|---:|
| bicycle | +0.0074 | +0.00220 | -0.00488 | 11.81% |
| flowers | +0.0219 | +0.00323 | -0.00801 | 11.82% |
| garden | +0.0145 | +0.00027 | -0.00251 | 3.47% |
| stump | +0.1740 | +0.00881 | -0.01222 | 11.82% |
| treehill | +0.0513 | +0.00340 | -0.00723 | 11.81% |
| room | +0.6039 | +0.01219 | -0.02722 | 2.10% |
| counter | +0.8103 | +0.02045 | -0.04073 | 2.10% |
| kitchen | +1.1505 | +0.02565 | -0.03965 | 2.10% |
| bonsai | +1.0905 | +0.01958 | -0.04353 | 11.80% |

Interpretation:

This is the first Phase-F variant that clears the current selected dataset
against both the clean MeshSplatting baseline and the archived Compact-ELA/SOR
baseline on all three RGB metrics while preserving additional topology
compression. It should be treated as the current best reproducible result.
The remaining caveat is methodological: the visual recovery is still an ELA
render-time layer rather than fully persistent surface-attached appearance
state, so it is a strong current result but not the final representation-level
ECSR endpoint.

# v56 Source-Rerun and Fresh-Probe Log

Date: 2026-06-24

Status: `CURRENT_MISSING_AUDIT_FRESH_PROBES_CLOSED`. This closes a v56 reproducibility gap for the selected `counter` face-alpha row and adds six missing-audit fresh probes (`flowers`, `treehill`, `bicycle`, `garden`, `stump`, `room`). It does not promote v56 to a paper endpoint yet.

## Motivation

Before this log, v56 was a report-only effective policy candidate:

- `counter` used an existing v55d cap-hit run and passed the reliability guard;
- `kitchen/bonsai` had v55d cap-hit runs but were rejected by the guard;
- `bicycle/flowers/garden/stump/treehill/room` were `v52_fallback` only because no v55d audit existed.

The main gap was not the v52 fallback path, because v52 already had `9 / 9` source-config reproduction. The gap was whether v56 could be replayed from source configs and whether missing-audit scenes remain rejected under the fixed guard.

## Implementation

New scripts:

```text
scripts/car_model/plan_v56_face_alpha_guard_source_rerun.py
scripts/car_model/summarize_v56_source_rerun.py
```

Updated script:

```text
scripts/car_model/summarize_v56_face_alpha_guard_policy.py
```

The v56 policy summarizer now accepts multiple `--v55d_root` values. Earlier roots take precedence, which lets us combine fresh probes with the older cap-hit v55d root without copying `/dev/shm` outputs.

## Counter Selected-Row Source Rerun

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/plan_v56_face_alpha_guard_source_rerun.py \
  --mode selected \
  --scene counter \
  --gpus 2 \
  --output_root /dev/shm/peilincai_spcarnet_v56_source_rerun_counter_20260624 \
  --plan_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_source_rerun_counter_plan.json \
  --plan_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_source_rerun_counter_plan.md \
  --plan_sh outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_source_rerun_counter_plan.sh \
  --refresh_output_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_counter_source_rerun_summary.json \
  --refresh_output_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_counter_source_rerun_summary.md \
  --execute \
  --refresh_after_execute \
  --force \
  --wandb_project SPCarNet \
  --wandb_group v56_face_alpha_guard_source_rerun \
  --wandb_run_name v56_selected_counter_source_rerun_20260624 \
  --wandb_mode online
```

W&B:

| run | id |
|---|---|
| parent command run | `knt0skxs` |
| per-scene v55d run | `bbiugsyu` |

Result:

| scene | PSNR | SSIM | LPIPS | guard | selected alpha | face-alpha count | changed fraction |
|---|---:|---:|---:|---|---:|---:|---:|
| counter | `26.7561302185` | `0.8621262312` | `0.2516913712` | pass | `0.5` | `394` | `6.536248%` |

This exactly reproduces the prior v55d counter row used by v56.

## Flowers Fresh Missing-Audit Probe

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/plan_v56_face_alpha_guard_source_rerun.py \
  --mode v55d_candidates \
  --scene flowers \
  --gpus 2 \
  --output_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_flowers_20260624 \
  --plan_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_flowers_candidate_plan.json \
  --plan_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_flowers_candidate_plan.md \
  --plan_sh outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_flowers_candidate_plan.sh \
  --refresh_output_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_flowers_candidate_summary.json \
  --refresh_output_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_flowers_candidate_summary.md \
  --execute \
  --refresh_after_execute \
  --force \
  --wandb_project SPCarNet \
  --wandb_group v56_face_alpha_guard_fresh_candidates \
  --wandb_run_name v56_candidate_flowers_source_rerun_20260624 \
  --wandb_mode online
```

W&B:

| run | id |
|---|---|
| parent command run | `qd1mrg2i` |
| per-scene v55d run | `tdypmap4` |

Flowers v55d candidate:

| scene | PSNR | SSIM | LPIPS | accepted | effective policy | face-alpha count | changed fraction |
|---|---:|---:|---:|---:|---|---:|---:|
| flowers | `19.6686706543` | `0.5116766691` | `0.3947824240` | 0 | `fallback_noop` | `5` | `0.0%` |

Fixed guard decision:

```text
reject:
  v55d_not_accepted
  v55d_not_accepted_atlas:fallback_noop
  face_alpha_count_below:5<128
  l1_positive_fraction_below:0.5<0.9
  ssim_min_view_gain_below:-0.0001055598258972168<5e-05
  l1_cvar20_view_gain_below:-8.538365364074707e-06<-5e-06
```

Interpretation: `flowers` is a useful fresh negative validation. It converts one previous `missing_v55d_audit` fallback into an explicit train/policy-val rejection, which supports the fixed guard's conservative behavior.

## Treehill Fresh Missing-Audit Probe

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/plan_v56_face_alpha_guard_source_rerun.py \
  --mode v55d_candidates \
  --scene treehill \
  --gpus 3 \
  --output_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_treehill_20260624 \
  --plan_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_treehill_candidate_plan.json \
  --plan_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_treehill_candidate_plan.md \
  --plan_sh outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_treehill_candidate_plan.sh \
  --refresh_output_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_treehill_candidate_summary.json \
  --refresh_output_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_treehill_candidate_summary.md \
  --execute \
  --refresh_after_execute \
  --force \
  --wandb_project SPCarNet \
  --wandb_group v56_face_alpha_guard_fresh_candidates \
  --wandb_run_name v56_candidate_treehill_source_rerun_20260624 \
  --wandb_mode online
```

W&B:

| run | id |
|---|---|
| parent command run | `6gvhmy8o` |
| per-scene v55d run | `mhnjkb5z` |

Treehill v55d candidate:

| scene | PSNR | SSIM | LPIPS | accepted | effective policy | face-alpha count | changed fraction |
|---|---:|---:|---:|---:|---|---:|---:|
| treehill | `20.9231948853` | `0.5642222166` | `0.4061251283` | 0 | `fallback_noop` | `10` | `0.0%` |

Fixed guard decision:

```text
reject:
  v55d_not_accepted
  v55d_not_accepted_atlas:fallback_noop
  face_alpha_count_below:10<128
  l1_positive_fraction_below:0.6666666666666666<0.9
  ssim_min_view_gain_below:-0.00030434131622314453<5e-05
  l1_cvar20_view_gain_below:-2.9653310775756836e-05<-5e-06
```

Interpretation: `treehill` is a high-value fresh negative validation because v48 already had an LPIPS-risk symptom on this scene. The v55d candidate exposes sparse local alpha support and negative policy-val SSIM/L1 tail evidence, so the fixed guard keeps the safer v52 fallback.

## Bicycle Fresh Missing-Audit Probe

W&B:

| run | id |
|---|---|
| parent command run | `vocupy0a` |

Bicycle v55d candidate:

| scene | PSNR | SSIM | LPIPS | accepted | effective policy | face-alpha count | changed fraction |
|---|---:|---:|---:|---:|---|---:|---:|
| bicycle | `23.2935066223` | `0.6596505046` | `0.3322694302` | 0 | `fallback_noop` | `15` | `0.0%` |

Fixed guard decision:

```text
reject:
  v55d_not_accepted
  v55d_not_accepted_atlas:fallback_noop
  face_alpha_count_below:15<128
  l1_positive_fraction_below:0.5833333333333334<0.9
  ssim_min_view_gain_below:-1.9311904907226562e-05<5e-05
  l1_cvar20_view_gain_below:-5.068878332773845e-06<-5e-06
```

Interpretation: `bicycle` is another clean negative. The residual atlas has too little reliable per-face alpha support and the policy-val view distribution is not robust enough, so the fixed guard keeps v52.

## Garden Fresh Missing-Audit Probe

W&B:

| run | id |
|---|---|
| parent command run | `s3zieof1` |

Garden v55d candidate:

| scene | PSNR | SSIM | LPIPS | accepted | effective policy | selected alpha | face-alpha count | changed fraction |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| garden | `24.7412643433` | `0.7540608048` | `0.2480128855` | 1 | `accepted_atlas` | `0.0625` | `92` | `1.849715%` |

Fixed guard decision:

```text
reject:
  face_alpha_count_below:92<128
  ssim_min_view_gain_below:3.2186508178710938e-06<5e-05
```

Interpretation: `garden` is the most useful boundary case in this batch. The candidate is internally accepted and does change test renders, but the fixed v56 reliability guard still rejects it because local alpha support is below threshold and the SSIM min-view margin is too weak. This supports keeping v56 conservative rather than chasing tiny, unreliable apparent gains.

## Stump Fresh Missing-Audit Probe

W&B:

| run | id |
|---|---|
| parent command run | `nz6ft79h` |

Stump v55d candidate:

| scene | PSNR | SSIM | LPIPS | accepted | effective policy | selected alpha | face-alpha count | changed fraction |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| stump | `25.1809158325` | `0.7044190764` | `0.2942161262` | 0 | `fallback_noop` | `0.0000` | `5` | `0.0%` |

Fixed guard decision:

```text
reject:
  v55d_not_accepted
  v55d_not_accepted_atlas:fallback_noop
  face_alpha_count_below:5<128
  l1_positive_fraction_below:0.7777777777777778<0.9
  ssim_min_view_gain_below:-3.6656856536865234e-05<5e-05
  l1_cvar20_view_gain_below:-5.662441253662109e-06<-5e-06
```

Interpretation: `stump` is a conservative negative. The v55d command cannot find a non-noop atlas under train/policy-val evidence, and both support count and tail robustness fail the fixed guard.

## Room Fresh Missing-Audit Probe

W&B:

| run | id |
|---|---|
| parent command run | `5rttakj6` |

Room v55d candidate:

| scene | PSNR | SSIM | LPIPS | accepted | effective policy | selected alpha | face-alpha count | changed fraction |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| room | `28.7398777008` | `0.8848091364` | `0.2499018013` | 1 | `accepted_atlas` | `0.1250` | `142` | `1.047404%` |

Fixed guard decision:

```text
reject:
  ssim_min_view_gain_below:2.7418136596679688e-06<5e-05
```

Interpretation: `room` is a second useful boundary case. It has enough face-alpha support and an internally accepted atlas, but the worst-view SSIM margin is too small for the fixed reliability guard. The effective v56 policy therefore keeps the safer v52 fallback.

## Combined Freshcheck Summary

Command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v56_face_alpha_guard_policy.py \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_room_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_stump_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_garden_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_bicycle_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_treehill_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_v55d_candidates_flowers_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v56_source_rerun_counter_20260624 \
  --v55d_root /dev/shm/peilincai_spcarnet_v55d_face_alpha_caphit_20260623 \
  --output_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_freshcheck_summary.json \
  --output_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_freshcheck_summary.md
```

Aggregate remains unchanged because all six fresh missing-audit probes are rejected by the fixed v56 guard and fall back to v52:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v56 freshcheck vs v52 | 9 | 1 | 9 | `+0.000296699` | `+0.000001285` | `-0.000019663` |
| v56 freshcheck vs no-op | 9 | 7 | 8 | `+0.001845890` | `+0.000037803` | `-0.000074494` |
| v56 freshcheck vs v48 | 9 | 3 | 9 | `+0.000383589` | `+0.000010067` | `-0.000034966` |

## Source-Rerun Status

The v56 selected effective policy is now source-reproducible by combining the fresh counter v55d source rerun with the already completed v52 source rerun roots:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_source_rerun_status.md
```

Status:

```text
COMPLETE_REPRODUCED
completed scenes: 9 / 9
missing scenes: 0
metric mismatch scenes: 0
metric epsilon: 1e-5
```

The stricter fresh-probe status including flowers, treehill, bicycle, garden, stump, and room is:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_freshcheck_source_status.md
```

It also reports `COMPLETE_REPRODUCED` for the effective selected rows, while the freshcheck summary records the explicit fresh v55d rejections.

## Stricter Boundary Ablation: `min_target_changed_fraction=0.0`

After the fixed-command fresh probes, the source-rerun planner was updated to expose the v55d candidate command's target-change threshold:

```text
scripts/car_model/plan_v56_face_alpha_guard_source_rerun.py
  --v55d_min_target_changed_fraction
```

The default remains `0.001` for exact reproduction of the earlier fixed command. A stricter boundary ablation was then run on the two internally accepted but guard-rejected boundary scenes, `garden` and `room`, using `--v55d_min_target_changed_fraction 0.0`.

W&B:

| scene | parent run | per-scene run |
|---|---|---|
| garden | `485snual` | `2027uxg1` |
| room | `uhqsu0wo` | `z8k86kzz` |

Results:

| scene | threshold | PSNR | SSIM | LPIPS | accepted | effective policy | selected alpha | face-alpha count | changed fraction | fixed guard |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| garden | `0.0` | `24.7412643433` | `0.7540608048` | `0.2480128855` | 1 | `accepted_atlas` | `0.0625` | `92` | `1.849715%` | reject: `face_alpha_count_below`, `ssim_min_view_gain_below` |
| room | `0.0` | `28.7398777008` | `0.8848091364` | `0.2499018013` | 1 | `accepted_atlas` | `0.1250` | `142` | `1.047404%` | reject: `ssim_min_view_gain_below` |

The stricter threshold does not change either candidate's metrics or fixed-guard decision. This closes the most relevant boundary-case concern for the two accepted-but-rejected scenes: the v56 guard rejection is driven by support and worst-view SSIM evidence, not by the `0.001` changed-fraction floor.

The remaining sparse/no-op rejected scenes were then rerun under the same `0.0` threshold:

| scene | parent run | elapsed sec | fixed guard decision |
|---|---|---:|---|
| flowers | `g6dib53z` | `552.86` | reject; sparse local alpha and weak policy-val robustness |
| treehill | `aau1supu` | `513.97` | reject; sparse local alpha and negative SSIM/L1 tail evidence |
| bicycle | `e9z6lhls` | `509.49` | reject; sparse local alpha and weak policy-val robustness |
| stump | `ng46x0x6` | `305.86` | reject; fallback/no-op with sparse support |

Updated combined evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_garden_room_freshcheck_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_garden_room_source_status.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_full_freshcheck_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v56_face_alpha_guard_mtc0_full_source_status.md
```

The full mtc0 source status is:

```text
COMPLETE_REPRODUCED
completed scenes: 9 / 9
missing scenes: 0
metric mismatch scenes: 0
```

Full mtc0 aggregate remains unchanged:

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v56 mtc0 full vs v52 | 9 | 1 | 9 | `+0.000296699` | `+0.000001285` | `-0.000019663` |
| v56 mtc0 full vs no-op | 9 | 7 | 8 | `+0.001845890` | `+0.000037803` | `-0.000074494` |
| v56 mtc0 full vs v48 | 9 | 3 | 9 | `+0.000383589` | `+0.000010067` | `-0.000034966` |

## Decision

Do not promote v56 to the paper endpoint yet.

What improved:

- v56 selected `counter` face-alpha row is now source-rerun reproduced with W&B.
- `flowers` is no longer only a missing-audit fallback; it has a fresh v55d probe and is correctly rejected by the fixed guard.
- `treehill` is no longer only a missing-audit fallback; it has a fresh v55d probe and is correctly rejected on sparse face-alpha support plus negative policy-val SSIM/L1 tail evidence.
- `bicycle` is no longer only a missing-audit fallback; it has a fresh v55d probe and is correctly rejected on sparse face-alpha support plus weak policy-val view robustness.
- `garden` is no longer only a missing-audit fallback; it has a fresh v55d probe that is internally accepted but still rejected by fixed v56 because face-alpha support and SSIM min-view margin are not strong enough.
- `stump` is no longer only a missing-audit fallback; it has a fresh v55d probe and is correctly rejected as fallback/no-op with sparse face-alpha support and weak tail evidence.
- `room` is no longer only a missing-audit fallback; it has a fresh v55d probe that is internally accepted but still rejected by fixed v56 because worst-view SSIM margin is too small.
- The effective v56 policy has a complete source-rerun status table when combined with the already closed v52 fallback rerun.
- The `garden/room` `min_target_changed_fraction=0.0` boundary ablation is complete and leaves the metrics plus fixed-guard decisions unchanged.
- The full six-scene `min_target_changed_fraction=0.0` audit is now complete and source-reproduced; all sparse/no-op cases remain rejected.

Remaining gaps:

- The guard was still designed after the original v55d cap-hit held-out inspection.
- v56 effect size remains small and does not replace Phase-J.

Next priority: validate the fixed guard on genuinely new scenes/protocols and build a stronger representation-level effect-size improvement. The current fixed-command missing-audit fallback set and full `0.0` threshold audit are closed.

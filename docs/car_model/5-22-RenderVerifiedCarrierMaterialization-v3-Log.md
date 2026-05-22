# 2026-05-22 Render-Verified Carrier Materialization v3

Status: `NOT_COMPLETE_SCIENTIFICALLY`, `COMPLETE_AS_A_FAIR_V3_VALIDATION`.

This log records the first full four-scene replay of the render-verified carrier
materialization policy. The engineering goal was completed: a real method change
was wired into the train/eval pipeline, evaluated against the Phase-J guarded
baseline, logged to W&B, and visualized. The scientific goal is not yet met:
the method is safer than the previous local-objective variants, but the accepted
gain remains at numerical-noise scale and does not yet support a paper-level
claim.

## Method Delta

The v2 render-region objective failed because locally positive cropped regions
did not reliably predict global train-val/test behavior. The v3 policy moves the
decision before materialization and filters whole carrier groups rather than
single rows:

1. Generate candidate face-local carrier plans from the existing Phase-K
   PatchCert/face-local pipeline.
2. Map each plan carrier to train render-visible region carriers.
3. Keep a carrier only when its matched train render regions are actually
   changed, have nonnegative mean core-balanced evidence, have nonnegative tail
   evidence, and do not create context MSE regression.
4. Materialize only the kept rows.
5. Run the full Phase-K train-val gate and compact topology gate. Test metrics
   remain report-only and are not used for promotion.

The main added/updated interfaces are:

- `scripts/car_model/ecsr_filter_facelocal_plan_by_render_region.py`
- `scripts/car_model/ecsr_eval_train_render_region_objective.py`
- `scripts/car_model/ecsr_decide_phasek_trainval_gate.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`

This is a real pipeline change because it changes the materialized candidate
mesh before render/eval, not only the final selection threshold.

## Evidence Roots

```text
Experiment root:
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522

Final eval:
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/final_eval

Filtered plans:
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/filtered_plans/{scene}/filtered_plan.json

Qualitative panels:
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522_qualitative

W&B:
project=mesh-splatting-ecsr
group=phasek_render_region_filtered_v3_20260522
```

Full replay command:

```text
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/final_eval/replay_command.sh
```

## Filtering Results

Filter thresholds:

```text
min_regions=1
min_changed_regions=1
min_changed_fraction=0.05
min_mean_core_balanced_delta=0.0
min_tail_core_balanced_delta=-0.00000001
max_context_mse_regression=0.000001
require_positive_plan_proxy=true
```

| scene | input rows | input carriers | kept rows | kept carriers | interpretation |
|---|---:|---:|---:|---:|---|
| flowers | 16 | 2 | 0 | 0 | all carriers lacked usable render-region overlap/change |
| counter | 18 | 3 | 0 | 0 | local positives existed but failed tail or mean safety |
| bonsai | 75 | 10 | 56 | 7 | only scene with broad carrier support |
| room | 16 | 2 | 8 | 1 | one carrier passed local evidence but later failed compact PSNR gate |

Detailed filter audits:

```text
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/filtered_plans/{scene}/filter_summary.md
```

## Full Gate Results

Evaluation used `iteration=26000`, Phase-J guarded ELA as baseline, W&B online
logging, and the existing Phase-K train-val and compact topology gates.

| scene | selected | accepted | faces | verts | no-op | test dPSNR | test dSSIM | test dLPIPS | train-val dPSNR | train-val dSSIM | train-val dLPIPS | render-region changed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flowers | phasej_guarded_adaptedge | false | 0 | 0 | true | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | 0/26 |
| counter | phasej_guarded_adaptedge | false | 0 | 0 | true | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | +0.000000000 | 0/48 |
| bonsai | phase_s_render_region_filtered_v3_20260522 | true | 56 | 168 | false | +0.000059128 | +0.000000477 | -0.000000119 | +0.000022888 | -0.000000060 | +0.000000402 | 7/48 |
| room | phasej_guarded_adaptedge | false | 8 | 24 | false | +0.000005722 | +0.000000000 | +0.000000119 | +0.000007629 | +0.000000000 | -0.000000045 | 5/48 |

Summary files:

```text
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/final_eval/phasek_barycentric_gate_summary.json
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/final_eval/phasek_barycentric_gate_summary.md
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/final_eval/{scene}/phasek_scene_summary.json
```

Raw metric values:

| scene | baseline test PSNR | candidate test PSNR | baseline test SSIM | candidate test SSIM | baseline test LPIPS | candidate test LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| flowers | 20.300607681 | 20.300607681 | 0.557457805 | 0.557457805 | 0.329505473 | 0.329505473 |
| counter | 28.449171066 | 28.449171066 | 0.893730700 | 0.893730700 | 0.186472371 | 0.186472371 |
| bonsai | 31.862005234 | 31.862064362 | 0.930279613 | 0.930280089 | 0.172555298 | 0.172555178 |
| room | 30.305639267 | 30.305644989 | 0.905730247 | 0.905730247 | 0.195989445 | 0.195989564 |

## Qualitative Outputs

Command:

```text
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_build_phase_s_patchcert_qualitative.py \
  --scenes counter,bonsai,room,flowers \
  --root_template /data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522/final_eval \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --out_dir /data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522_qualitative \
  --views_per_scene 4 \
  --image_width 300 \
  --diff_boost 80
```

Outputs:

```text
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522_qualitative/patchcert_qualitative_contact_sheet.png
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522_qualitative/qualitative_summary.md
/data/peilincai/spcarnet_runs/phasek_render_region_filtered_v3_20260522_qualitative/qualitative_manifest.json
```

The qualitative panels confirm the same finding as the metrics: the visible
delta is extremely small. Bonsai has a few localized positive panels, but the
effect is not visually compelling enough for a top-conference story.

## Honest Review

What improved:

- The policy no longer accepts the v2 counter/bonsai style local false positives.
- It preserves no-op scenes by rejecting unchanged filtered plans before final
  selection.
- It records carrier-level reasons, render-region change counts, materialized
  face counts, final train-val/test deltas, and qualitative panels.

What remains weak:

- Only 1/4 scenes was accepted after full gate.
- Accepted bonsai gains are around `1e-5` to `1e-4`, which is still too close to
  numerical/rendering noise.
- Flowers and counter become no-op under this policy, so they provide safety but
  no positive evidence.
- Room passes the local carrier filter but fails the compact PSNR threshold,
  which means render-region evidence is still not strong enough as a promotion
  signal.
- Candidate plans were replayed with `delta_facelocal_materialize_allow_uncertified_plan`;
  this is acceptable as an ablation but not yet the strict final method.

## Next Hard Gate

Do not claim paper-level improvement from this version. The next method upgrade
must pass all of the following before promotion:

1. At least 3/4 scenes accepted by train-val and compact topology gates.
2. Mean test PSNR gain above `+0.002 dB` or an equivalent geometry/triangle-count
   Pareto win that is visibly meaningful.
3. No scene with positive LPIPS regression larger than `5e-5`.
4. Qualitative panels that show visible localized improvement, not only boosted
   difference noise.
5. Strict certified carrier plans without the uncertified replay escape hatch.


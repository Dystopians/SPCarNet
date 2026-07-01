# v332 Support-Dropout Consistency Negative Probe

Date: 2026-07-01

## Purpose

v331 showed that source-heldout pairwise LCB evidence is over-confident on the
treehill bad target views. v332 tests a more independent target-blind signal:
whether the candidate residual field is stable when individual support frames
are dropped and evidence is recomputed.

This is implemented as a separate diagnostic first, not as a promoted apply
policy.

## Implemented Diagnostic

Script:

```text
scripts/car_model/probe_support_dropout_consistency.py
```

The script reuses the apply pipeline's calibrator, evidence construction, and
candidate delta generation. For each selected promoted target view it:

1. recomputes the normal candidate residuals;
2. drops each used support frame one at a time;
3. recomputes candidate residuals under each support subset;
4. measures stability for `delta[output_variant] - delta[incumbent_variant]`.

Target/test GT is not used to compute stability. The script can copy true
post-eval deltas from a reference report only for diagnostic correlation.

## Command

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/probe_support_dropout_consistency.py \
  --base_model_path outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/treehill/ratio_0200/compact_model \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --checkpoint outputs/carnet/spcarnet_v302_constrained_hybrid_anchor_flowers_20260630/support_transport_calibrator.pt \
  --reference_report docs/car_model/results/v331_promotion_rollback_shadow_treehill_report.json \
  --output_json outputs/carnet/spcarnet_v332_support_dropout_treehill_20260701/support_dropout_consistency.json \
  --enable_candidate_ladder \
  --candidate_ladder_blends 0.25,0.75 \
  --promotions_only \
  --views 00002,00004,00007,00008,00009,00011,00015 \
  --max_dropout_samples 4 \
  --evidence_max_side 256
```

Committed diagnostic JSON:

```text
docs/car_model/results/v332_support_dropout_treehill_consistency.json
```

## Result

| view | true PSNR delta vs fixed | true SSIM delta vs fixed | pair relative std | pair cosine | pair sign flip | output relative std | output cosine | output sign flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 00002 | +0.007259 | +0.000269 | 0.472745 | 0.892474 | 0.127811 | 0.476917 | 0.902368 | 0.173336 |
| 00004 | +0.002511 | -0.000220 | 0.521711 | 0.863009 | 0.168755 | 0.536132 | 0.865961 | 0.207842 |
| 00007 | -0.026469 | -0.000043 | 0.503789 | 0.820512 | 0.158815 | 0.453758 | 0.802321 | 0.243400 |
| 00008 | -0.004946 | -0.000321 | 0.516541 | 0.867959 | 0.151190 | 0.514406 | 0.878067 | 0.202654 |
| 00009 | -0.012385 | -0.000004 | 0.547497 | 0.940110 | 0.125608 | 0.520551 | 0.965856 | 0.169087 |
| 00011 | +0.033720 | +0.000269 | 0.480325 | 0.908454 | 0.175988 | 0.469942 | 0.919011 | 0.224807 |
| 00015 | +0.015077 | +0.000209 | 0.579338 | 0.897376 | 0.159033 | 0.603453 | 0.907258 | 0.232368 |

## Interpretation

The support-dropout stability signal is also not discriminative enough for the
current treehill failure. Bad views `00007`, `00008`, and `00009` overlap with
positive controls on relative std, cosine, and sign-flip fraction; `00009` is
especially problematic because it is target-negative but has high dropout
cosine.

This explains why the current source/support-side certificate family is stuck:
the bad treehill views are not obviously unstable under the available
target-blind support evidence. They appear to be **stable but wrong**.

## Verdict

Final status: NOT COMPLETE.

v332 adds useful diagnostic tooling, but it does not solve the metric/visual
bottleneck. The next route should move beyond residual-support stability gates:
either add genuinely new target-blind evidence, such as camera-neighborhood
render self-consistency or multi-render uncertainty from independent candidate
generators, or improve the raw representation/candidate generator so that
policy arbitration no longer has to rescue tiny, ambiguous residual edits.

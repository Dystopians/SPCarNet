# Phase-K Rank-K PatchCert Carrier Basis Validation

Date: 2026-05-22

Status: `NOT_COMPLETE_REPRESENTATION_CAPACITY_TESTED_BUT_NOT_PAPER_LEVEL`

This log records the first fixed-policy validation of the rank-K PatchCert
carrier basis operator. The goal was to test a real representation-level
capacity change, not another edge/plain or ELA parameter sweep.

## Method Change

Commit: `542b0b6 Add rank-K PatchCert carrier basis`

The prior PatchCert cluster-basis path had a hard-coded rank-2 mixture inside
`fit_patch_cluster_shared_basis()`. This update exposes the basis rank through:

- `--patch_cert_cluster_basis_rank` in
  `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `--delta_patch_cert_cluster_basis_rank` in
  `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

The default remains `2`, so existing rank-2 behavior is preserved. For this
validation the rank was fixed to `4`. The operator still materializes ordinary
face-local SH coefficients, and the same Phase-K train-val gate, compact gate,
fresh Phase-J replay, and held-out-test report-only protocol are used.

In plain language: the method now allows each certified patch to use a richer
set of coherent residual carriers instead of forcing every patch through a
rank-2 correction bottleneck. This is still a baked representation edit, not a
test-time crop selector.

## Aborted Misconfigured Run

An earlier run at:

```text
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_20260522
```

used `--delta_patch_cert_rings 0`. That reduced each carrier to essentially
single-face support, so the rank-4 cluster basis did not actually test the
intended coherent multi-face patch operator. Audit showed no valid cluster rows
for the intended patch-basis path. The run was killed, deleted, and is not used
as evidence.

## Correct Validation Run

Output root:

```text
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_rings1_20260522
```

W&B:

```text
project: mesh-splatting-ecsr
group: phasek_rank4_patchbasis_rings1_20260522
name: phasek_rank4_patchbasis_rings1
mode: online
gpu: 7
```

Key fixed arguments:

```bash
--scenes flowers,counter,bonsai,room
--iteration 26000
--delta_patch_cert_rings 1
--delta_patch_cert_neighbor_mode both
--delta_patch_cert_cluster_basis
--delta_patch_cert_cluster_basis_mode rank2
--delta_patch_cert_cluster_basis_rank 4
--delta_patch_cert_cluster_basis_steps 260
--delta_patch_cert_cluster_basis_lr 0.02
--delta_patch_cert_cluster_basis_max_fit_mse_regression 0.02
--delta_patch_cert_cluster_basis_view_hinge_weight 0.02
--delta_patch_cert_cluster_basis_geometry_smooth_weight 0.02
--cleanup_train_artifacts_after_scene
--ela_policy_source per_model_auto
--ela_policy_objective balanced
--ela_calib_lpips
--ela_alpha_grid 0,0.0625,0.125,0.25,0.5
```

Summary files:

```text
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_rings1_20260522/rank4_rings1_phasek_summary.md
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_rings1_20260522/rank4_rings1_phasek_summary.json
```

Qualitative contact sheet:

```text
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_rings1_20260522_qualitative/patchcert_qualitative_contact_sheet.png
```

Train-defined local support metrics and panels:

```text
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_rings1_20260522_local_support/{scene}/surface_support_local_metrics.md
/home/peilincai/spcarnet_runs/phasek_rank4_patchbasis_rings1_20260522_local_support/{scene}/panels/*.png
```

## Operator Audit

The corrected run exercised the rank-4 path in all four scenes.

| scene | accepted faces | rank arg | cluster rows | applied cluster rows |
|---|---:|---:|---:|---:|
| flowers | 22 | 4 | 40 | 25 |
| counter | 16 | 4 | 36 | 17 |
| bonsai | 113 | 4 | 40 | 40 |
| room | 22 | 4 | 40 | 27 |

This confirms the validation is not another no-op or miswired interface test.

## Four-Scene Phase-K Result

Selection uses train-policy-val decisions only. Held-out test deltas below are
report-only. Effective deltas set rejected scenes to fallback.

| scene | selected | accepted | faces | train-val balanced | report-only balanced | report dPSNR | report dSSIM | report dLPIPS | reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| flowers | rank4 patch basis | true | 22 | +0.000038743 | +0.000005484 | +0.000011444 | +0.000000179 | +0.000000477 | tiny positive, LPIPS slightly worse |
| counter | Phase-J fallback | false | 16 | +0.000002980 | +0.000003755 | +0.000007629 | +0.000000000 | +0.000000194 | rejected by compact PSNR floor |
| bonsai | rank4 patch basis | true | 113 | +0.000108004 | +0.000186026 | +0.000158310 | +0.000000179 | -0.000001207 | clearest positive row |
| room | rank4 patch basis | true | 22 | +0.000055134 | -0.000008345 | +0.000000000 | -0.000000417 | +0.000000000 | train-val accepted, held-out balanced slightly negative |

Aggregate effective held-out deltas:

```text
dPSNR: +0.0000424385
dSSIM: -0.0000000149
dLPIPS: -0.0000001825
accepted scenes: 3 / 4
```

Compared with the previous candidate-aware plain portfolio on the same four
scenes:

```text
previous dPSNR/dSSIM/dLPIPS: +0.000053883 / -0.000000268 / +0.000000469
rank4 dPSNR/dSSIM/dLPIPS:    +0.000042439 / -0.000000015 / -0.000000183
```

The rank-4 operator improves the LPIPS direction and makes `bonsai` more
credible, but it does not increase mean PSNR and it still leaves `counter`
unresolved. It is a real method improvement over the representation interface,
but it is not a breakthrough.

## Local Support Evidence

The local-support protocol fixes masks from train evidence, projects them to
held-out views using surface maps, then computes metrics. It does not mine test
crops by improvement.

| scene | dMaskPSNR | dMaskMAE | dCropPSNR | dCropSSIM | dCropLPIPS | crop LPIPS wins |
|---|---:|---:|---:|---:|---:|---:|
| flowers | +0.000349 | -0.00000013 | +0.000061 | +0.00000131 | +0.00000274 | 2 / 12 |
| counter | +0.000098 | -0.00000108 | +0.000029 | +0.00000009 | +0.00000060 | 3 / 12 |
| bonsai | +0.002012 | -0.00002260 | +0.000361 | +0.00000125 | -0.00000596 | 3 / 12 |
| room | +0.000087 | +0.00000083 | +0.000016 | -0.00000027 | +0.00000070 | 2 / 12 |

The same conclusion appears locally: `bonsai` has the only clearly readable
support-region gain; the other scenes are near-zero or mixed.

## Verdict

This milestone is useful but not final.

What it proves:

- the rank-K interface is real, wired through train/eval, and exercised by the
  corrected run;
- W&B online logging, fresh Phase-J replay, symmetric per-model-auto ELA, and
  report-only held-out test evaluation all ran successfully;
- `bonsai` improves both globally and in train-defined local support regions;
- the richer coherent patch basis improves LPIPS direction versus the previous
  four-scene candidate-aware portfolio.

What it does not solve:

- the mean gains are still noise-scale;
- `counter` remains below the compact PSNR acceptance floor;
- `room` exposes a train-val to held-out generalization gap;
- the qualitative contact sheet remains subtle rather than visually decisive.

Therefore the next step should not be another rank/edge/plain parameter sweep.
The next method should directly optimize a train-only render-region objective:
fixed train residual masks, coherent multi-face support, differentiable
render-space crop/mask loss on train/policy-val views, context/tail penalties,
and the same Phase-K held-out report-only protocol. If that also fails on
`counter` and only preserves `flowers/bonsai`, the residual-coefficient family
should be treated as saturated and the project should pivot to a topology or
geometry-level mechanism.


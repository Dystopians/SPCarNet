# v238 Fixed-Policy Full9 Progress Log

Date: 2026-06-29

## Status

v238 has passed the v169 flowers exact all-axis gate, so the current task is no
longer to search for another flowers-only alpha or local tweak. The next
required step is fixed-policy multi-scene validation under the same no-target-GT
contract.

Current fixed policy:

- method: `surface_texture_unet`
- carrier: face/UV surface texture features plus local U-Net decoder
- support: `--enable_surface_support_gate`
- surface texture size: `8`
- surface feature dim: `8`
- max selected faces: `8192`
- alpha grid: `0,0.25,0.5,0.75,1`
- policy select mode: `tail_guard`
- W&B: offline

Important boundary: v238 is a weakly surface-native representation. It has a
real surface texture carrier and hard support gate, but the residual is decoded
by a U-Net that still sees image-space parent render and geometry buffers. It
should be described as a surface-gated neural texture residual adapter, not as a
pure baked per-surface residual field.

## Automation Added

New runner:

```bash
scripts/car_model/run_v238_surface_texture_full9_scene.py
```

Responsibilities:

- locate fixed Phase-J references:
  - `train/ours_26000_phasej_trainval_gate`
  - `test/ours_26000_phasej_guarded_adaptedge_ela`
- locate usable train and target evidence with nonzero `.npz` view counts;
- rebase train evidence to Phase-J train native render resolution;
- rebase target evidence to Phase-J test native render resolution while
  stripping target/test GT and residual keys;
- verify target no-GT evidence before apply;
- run the frozen v238 training/apply configuration;
- link test GT only after apply for final metric evaluation;
- evaluate v238 and Phase-J under the same local evaluator;
- write a per-scene summary JSON.

Output root:

```text
/tmp/peilincai_spcarnet_v238_full9/<scene>/
```

## Evidence Mapping

All full9 scenes have the fixed Phase-J train/test reference methods available.

The old vNext input root has complete train-fit evidence for:

- `bicycle`
- `flowers`
- `kitchen`
- `stump`
- `treehill`

The target evidence in that root is nested as:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/<scene>/target_evidence
/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/<scene>/target_visible_bary_base/<scene>/views/*.npz
```

The v39 multi-scene root has train/target evidence for:

- `counter`
- `garden`
- `room`

Relevant root:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/
```

Fallback support was added to the runner so it does not accept an empty evidence
directory just because the directory exists.

## Running Experiments

Two non-flowers fixed-policy runs were launched:

```bash
CUDA_VISIBLE_DEVICES=2 TMPDIR=/tmp WANDB_MODE=offline \
PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v238full_kitchen \
PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v238_surface_texture_full9_scene.py \
  --scene kitchen --gpu 2 --steps 3200 --seed 238 --force_rebase
```

```bash
CUDA_VISIBLE_DEVICES=3 TMPDIR=/tmp WANDB_MODE=offline \
PYTHONPYCACHEPREFIX=/tmp/peilincai_pycache_v238full_stump \
PYTHONDONTWRITEBYTECODE=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v238_surface_texture_full9_scene.py \
  --scene stump --gpu 3 --steps 3200 --seed 239 --force_rebase
```

Expected summary files:

- `/tmp/peilincai_spcarnet_v238_full9/kitchen/v238_kitchen_fixed_policy_summary.json`
- `/tmp/peilincai_spcarnet_v238_full9/stump/v238_stump_fixed_policy_summary.json`

## Required Verdict Fields

For each completed scene, record:

- no-GT verifier status;
- policy-val all-axis status and selected alpha;
- target known-face fraction;
- target active-support fraction;
- target changed fraction;
- exact PSNR / SSIM / LPIPS for v238 and Phase-J;
- per-view deltas;
- whether v238 beats Phase-J all-axis;
- whether failure is a no-op, support coverage, policy-val rejection, or exact
  target generalization failure.

## Known Weaknesses To Watch

- v238 flowers passed by a small margin.
- LPIPS positive-view fraction on flowers policy-val was only `0.6667`.
- Target writable area was small on flowers:
  - known face fraction `0.112010`;
  - active support fraction `0.062922`;
  - changed fraction `0.052635`.
- Strict low-rank surface texture failed on flowers, so the current win still
  depends on image-context decoding.

If fixed-policy full9 fails, the next method change should target surface-bin
consistency and LPIPS-tail stabilization, not another alpha scan.

## Completed Results So Far

### stump: v238 fixed policy passed exact Phase-J comparison

Summary:

- report: `/tmp/peilincai_spcarnet_v238_full9/stump/v238_surface_texture_unet_native1256/v238_surface_texture_unet_native1256_stump_report.json`
- exact metrics: `/tmp/peilincai_spcarnet_v238_full9/stump/v238_stump_native1256_exact_results.json`
- fixed-policy summary: `/tmp/peilincai_spcarnet_v238_full9/stump/v238_stump_fixed_policy_summary.json`
- no-GT verifier: passed before apply.

Policy-val selected `alpha=0.25` and passed all-axis over the Phase-J parent on
held-out fit views:

| metric | parent | v238 | delta |
|---|---:|---:|---:|
| PSNR | 23.380641 | 23.389486 | +0.008844 |
| SSIM | 0.774258 | 0.775018 | +0.000759 |
| LPIPS | 0.165912 | 0.163814 | -0.002099 |

Exact target/test comparison against the same Phase-J native-resolution renders:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-J reference | 25.595104 | 0.724074 | 0.263909 |
| v238 | 25.601849 | 0.724176 | 0.262521 |
| delta | +0.006744 | +0.000102 | -0.001388 |

Verdict: v238 beats Phase-J all-axis on stump under this local exact evaluator,
but the margin is small.

### kitchen: v238 fixed policy correctly rejected the adapter

Summary:

- report: `/tmp/peilincai_spcarnet_v238_full9/kitchen/v238_surface_texture_unet_native1256/v238_surface_texture_unet_native1256_kitchen_report.json`
- summary: `/tmp/peilincai_spcarnet_v238_full9/kitchen/v238_kitchen_fixed_policy_summary.json`
- train rebase and target no-GT verifier both passed.
- exact target eval was skipped because policy-val did not produce target renders.

Policy-val mean gains by alpha:

| alpha | PSNR gain | SSIM gain | LPIPS gain | changed fraction | LPIPS positive views |
|---:|---:|---:|---:|---:|---:|
| 0.25 | -0.004412 | -0.000153 | +0.000001 | 0.044690 | 0.666667 |
| 0.50 | -0.022994 | -0.000551 | -0.000107 | 0.061356 | 0.416667 |
| 0.75 | -0.057972 | -0.001338 | -0.000396 | 0.073625 | 0.083333 |
| 1.00 | -0.087000 | -0.002043 | -0.000696 | 0.079328 | 0.000000 |

Verdict: kitchen is not an evaluation or no-GT failure. The adapter introduces a
small but systematic PSNR/SSIM regression on held-out fit views, so the strict
all-axis gate falls back to no-op. This is useful negative evidence: on already
strong indoor Phase-J renders, the current residual representation is not
selective enough.

## v239 Residual-Debt Stabilization

Implemented in:

- `scripts/car_model/train_surface_conditioned_residual_unet.py`
- `scripts/car_model/run_v238_surface_texture_full9_scene.py`

Method change:

- add a train-fit-only residual-debt mask from parent-vs-GT/teacher error;
- inside the mask, train the surface texture U-Net to repair residual debt;
- outside the mask, blend the supervision target back to parent and add a
  no-op residual penalty;
- keep target/test apply no-GT: the mask is never built from target/test GT.

Runner variant:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v238_surface_texture_full9_scene.py \
  --scene kitchen \
  --variant v239_residual_debt \
  --steps 3200 \
  --skip_rebase
```

Current run:

```text
/tmp/peilincai_spcarnet_v238_full9/kitchen/v239_residual_debt_surface_texture_unet_native1256
```

W&B offline:

```text
/tmp/peilincai_spcarnet_v238_full9/kitchen/v239_residual_debt_surface_texture_unet_native1256/wandb/
```

Expected verdict:

- if v239 passes kitchen all-axis and exact eval, it becomes an adapter
  stabilization candidate;
- if v239 still fails, record it as evidence that Phase-J-plus local residual
  repair is not enough for kitchen.

## v240 True v169 Distillation Entry

Subagent and local path checks found a stricter and more correct v169 route:
distill Phase-J teacher renders into a Phase-F/baked parent surface
representation, rather than adding residuals on top of Phase-J itself.

New runner variant:

```bash
scripts/car_model/run_v238_surface_texture_full9_scene.py \
  --variant v240_phasej_to_phasef_distill
```

Contract:

- train parent: `train/ours_26000_phasef_extra_compact_base/renders`
- train teacher: `train/ours_26000_phasej_trainval_gate/renders`
- target/test parent: `test/ours_26000_phasef_extra_compact_base/renders`
- target/test reference: `test/ours_26000_phasej_guarded_adaptedge_ela/renders`
- train residual target: `Phase-J teacher - Phase-F parent`
- target/test apply input: Phase-F parent plus no-GT geometry only.

The live filesystem has complete parent/teacher train/test renders and GT for
`flowers`, `kitchen`, and `stump`. v106 is not available as a live train/test
render parent, so Phase-F is the practical parent for the next v169 proof.

Default v240 is teacher-only:

- teacher losses are enabled;
- train-fit GT losses are zero;
- policy-val GT is still used only for certification;
- target/test GT is linked only after apply for final eval.

Optional `--v240_gt_assist` creates a separate GT-assisted ablation output
directory and must not be mixed with teacher-only claims.

## Current Claim Boundary

The current best honest statement is:

- v238 has passed flowers and stump exact all-axis against Phase-J, but kitchen
  is a strict policy-val rejection.
- v238/v239 are Phase-J-plus adapters, not the final baked-representation answer
  requested by v169.
- v240 is now the correct implementation entry for the real v169 question:
  whether Phase-J's render-time improvement can be baked into a Phase-F parent
  surface representation without target/test RGB GT leakage.

Therefore the project is still `NOT COMPLETE` for paper-level closure until
v240 or a successor passes flowers exact all-axis and then fixed-policy
multi-scene validation.

## 2026-06-29 Native1256 v169 Follow-Up

The v169 prompt was executed more strictly after discovering an important
protocol issue: the old v168/v240 evidence path used raw `1600x1054` surface
evidence while the Phase-F/Phase-J gate references are `1256x828`. A new
native1256 flowers evidence chain was built under:

```text
/tmp/peilincai_spcarnet_v169_native1256_flowers_20260629
```

The fixed path used:

- Phase-F native1256 train/test parent renders;
- Phase-J native1256 train teacher renders;
- real train GT only for policy-val certification;
- target/test no-GT evidence with forbidden GT/residual keys stripped.

Policy-val passed all-axis, but only with very small gains:

| metric | gain over parent |
|---|---:|
| PSNR | +0.002417 |
| SSIM | +0.000117 |
| LPIPS | +0.000101 |

Exact flowers native1256 result:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-F parent | 19.668695 | 0.511678 | 0.394788 |
| v169 teacher-only | 19.670961 | 0.511814 | 0.394431 |
| Phase-J reference | 20.300608 | 0.557458 | 0.329505 |

Verdict: **v169 native1256 teacher-only fails the Phase-J hard gate**. It only
recovers about `0.7%` of the Phase-J-minus-Phase-F image delta on exact test.
The current bottleneck is not teacher-signal absence or target-GT leakage; it
is surface-carrier coverage/capacity. Full9 remains blocked.

Detailed report:

```text
docs/car_model/6-29-v169-Native1256-TeacherResidual-Diagnostic.md
```

## 2026-06-29 v241-v244 Dense Target-Visible Follow-Up

After the native1256 v169 failure, the method was moved closer to the actual
v169 contract: parent is Phase-F baked compact render, teacher is Phase-J, and
the residual target is Phase-J minus Phase-F parent rather than a Phase-J-plus
adapter.

Code-level additions:

- `SurfaceTextureConditionedUNet` now supports soft known/unknown support
  floors via `--surface_support_gate_floor` and
  `--surface_support_unknown_gate_floor`.
- `--residual_l1_key` allows raw teacher-parent residual magnitude to drive
  face/bin selection.
- residual-debt masking and target no-GT precheck fields remain recorded in the
  report JSON.
- `run_v238_surface_texture_full9_scene.py` contains a dense v241
  Phase-J-to-Phase-F distillation variant, although v242-v244 were launched
  directly as flowers-only proof attempts.

Exact flowers result progression:

| method | PSNR | SSIM | LPIPS | vs Phase-F parent | vs Phase-J |
|---|---:|---:|---:|---|---|
| Phase-F parent | 19.668695 | 0.511678 | 0.394788 | n/a | `-0.631912 / -0.045780 / +0.065282` |
| Phase-J reference | 20.300608 | 0.557458 | 0.329505 | `+0.631912 / +0.045780 / -0.065282` | n/a |
| v241 dense teacher-only | 19.694145 | 0.513428 | 0.391717 | `+0.025450 / +0.001750 / -0.003071` | `-0.606462 / -0.044030 / +0.062212` |
| v242 raw-dense teacher-only | 19.692448 | 0.512409 | 0.391072 | `+0.023752 / +0.000731 / -0.003716` | `-0.608160 / -0.045049 / +0.061566` |
| v243 raw-dense GT-assisted | 19.724134 | 0.518789 | 0.377212 | `+0.055439 / +0.007111 / -0.017576` | `-0.576473 / -0.038669 / +0.047706` |
| v244 dense-high GT-assisted | 19.803230 | 0.526760 | 0.361711 | `+0.134535 / +0.015082 / -0.033077` | `-0.497377 / -0.030697 / +0.032205` |

v244 is the current best true Phase-J-to-Phase-F distillation attempt. It
recovers about `21.29%` of the Phase-J PSNR gap, `32.94%` of the SSIM gap, and
`50.66%` of the LPIPS gap over the Phase-F parent. This is a substantial
improvement over the near-no-op v169 result, but it still fails the v169 hard
flowers gate. Therefore fixed-policy full9 remains blocked.

Detailed report:

```text
docs/car_model/6-29-v169-to-v244-DenseTargetVisible-Upgrade-Report.md
```

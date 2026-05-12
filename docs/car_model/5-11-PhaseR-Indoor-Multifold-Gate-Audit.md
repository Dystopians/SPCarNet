# Phase-R Indoor Multi-Fold Gate Audit

Date: 2026-05-11

## Why This Audit Exists

The earlier Phase-R representation candidates sometimes looked positive on the
held-out test split while failing, or barely passing, the single train-heldout
gate. That is not strong enough for a paper policy because it can hide
split-specific luck. I therefore added a deterministic multi-offset
train-heldout validation layer and require an indoor representation edit to
pass every offset before it can enter the fixed policy.

## Interface Changes

- `meshsplatopt_apply_evidence_lumigraph_adapter.py`
  - added `--policy_holdout_offset`;
  - train-policy-val frames can now be selected by several complementary
    deterministic offsets instead of a single fixed slice.
- `ecsr_run_phasek_barycentric_gate_scene.py`
  - forwards `--policy_holdout_offset` into train-val ELA.
- `ecsr_run_phasek_multifold_trainval_gate.py`
  - new robust audit script for an existing Phase-K candidate;
  - reruns train-heldout ELA/eval over multiple offsets;
  - accepts a candidate only if all offsets pass PSNR/SSIM/LPIPS thresholds;
  - marks held-out test metrics as report-only.
- `ecsr_select_phase_r_policy.py`
  - now recognizes both legacy `decisions/SCENE_decision.json` and stricter
    `SCENE/multifold_trainval_gate.json` decisions;
  - normalizes multi-fold mean train-val deltas for unified fixed-ladder
    summaries.

## Completed Multi-Fold Results

Thresholds: min PSNR gain `0.0`, max SSIM regression `5e-5`, max LPIPS
regression `1.5e-4`.

| scene | candidate | decision | mean train-val dPSNR | mean dSSIM | mean dLPIPS | worst reason | report-only test dPSNR | dSSIM | dLPIPS |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| room | dense16 SH1 | reject | -0.002081 | -0.000053 | +0.000034 | PSNR/SSIM/LPIPS unstable across offsets | +0.002779 | +0.000239 | -0.000290 |
| room | micro1024 SH1 | reject | +0.000068 | -0.000007 | +0.000013 | offset2 PSNR below 0 | +0.000097 | +0.000001 | -0.000001 |
| room | micro1024 SH1 + gamma trust 0.75 | accept | +0.000089 | -0.000002 | +0.000000 | pass | +0.000084 | +0.000001 | -0.000000 |
| counter | micro1024 SH1 | reject | +0.000238 | +0.000078 | -0.000013 | offset1/2 PSNR below 0 | -0.005699 | -0.000253 | +0.000318 |
| kitchen | sparse4096 SH1 | accept | +0.000537 | +0.000014 | -0.000011 | pass | +0.022673 | +0.000719 | -0.001068 |
| bonsai | dense16 SH1 | reject | +0.000216 | +0.000015 | +0.000073 | offset3 LPIPS +0.000161 > 0.000150 | +0.000814 | +0.000017 | -0.000046 |
| bonsai | dense16 SH1 + gamma trust 0.75 | reject | +0.000219 | +0.000019 | +0.000071 | offset3 LPIPS +0.000162 > 0.000150 | -0.006533 | +0.000678 | +0.000721 |

Interpretation:

- `kitchen` is the first indoor representation edit that survives multi-fold
  train-only validation and also wins all report-only test metrics.
- `room` demonstrates why test-positive candidates still cannot be promoted
  without a trust region: the unscaled micro residual fails offset 2, while a
  fixed train-only gamma blend of the same residual passes all offsets.
- `bonsai` would pass the older single gate and has positive report-only test
  metrics, but the fourth fold violates the LPIPS regression bound. It is a
  weak candidate, not a paper-clean main result. A fixed gamma 0.75
  trust-region rerun did not fix this: it still fails the same LPIPS fold and
  also loses report-only test PSNR/LPIPS versus Phase-J.

## Current Full9 Fixed Policy Snapshot

Artifact:

`outputs/carnet/meshsplatopt/ecsr_phase_r/fixed_candidate_ladder_v10_gamma_trust_full9/phase_r_fixed_candidate_ladder.md`

Policy:

1. Train-only gamma trust-region residual gate.
2. Indoor robust dense multi-fold gate.
3. Indoor robust sparse multi-fold gate.
4. Indoor robust micro multi-fold gate.
5. Outdoor dense fixed ladder.
6. Outdoor sparse fixed ladder.
7. Predeclared no-op for structural-edge fallback scenes such as `treehill`.

Result:

- scenes: `9`
- accepted selections: `6 / 9`
- strict report-only RGB wins: `6 / 9`
- mean report-only delta: PSNR `+0.002993`, SSIM `+0.000136`,
  LPIPS `-0.000217`

The mean is still helped substantially by the robust `kitchen` gain, but the
new `room` result is important because it converts a previously rejected indoor
scene into a strict multi-fold train-heldout accept without using held-out test
for selection. It is a small-margin stability improvement, not a large-margin
visual breakthrough.

## Counter Micro Negative Result

The current bottleneck is not logging or evaluation coverage. It is the
representational edit itself: the SH1 face residual can be too blunt for
indoor scenes with many view-dependent specular/occlusion changes. The next
experiment tried a more conservative `counter` micro residual candidate:
fewer faces, lower strength, smaller RGB/SH bounds, and stronger regularization.

Single-split result:

| scene | candidate | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report-only test dPSNR | dSSIM | dLPIPS | decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| counter | micro1024 SH1 | +0.001822 | +0.000024 | -0.000058 | -0.005699 | -0.000253 | +0.000318 | do not promote |

This is a useful negative result: making the residual smaller did not fix the
counter generalization gap. The candidate overfits the train-heldout slice and
hurts all held-out test metrics.

Multi-fold result:

| offset | dPSNR | dSSIM | dLPIPS | pass |
|---:|---:|---:|---:|---:|
| 0 | +0.001822 | +0.000024 | -0.000058 | yes |
| 1 | -0.002550 | -0.000030 | +0.000099 | no |
| 2 | -0.000011 | -0.000001 | -0.000025 | no |
| 3 | +0.001694 | +0.000321 | -0.000066 | yes |

The stricter multi-fold policy catches this false positive, so counter remains
a fallback scene in the fixed policy.

## Room Micro Negative Result

The same conservative micro residual was tested on `room` because its dense16
candidate had positive report-only test metrics but unstable train-heldout
behavior. The micro candidate looked clean under the single gate and had tiny
positive report-only test deltas:

- single train-val: `+0.000202` PSNR, `+0.000002` SSIM, `-0.000000` LPIPS;
- report-only test: `+0.000097` PSNR, `+0.000001` SSIM, `-0.000001` LPIPS.

Multi-fold result:

| offset | dPSNR | dSSIM | dLPIPS | pass |
|---:|---:|---:|---:|---:|
| 0 | +0.000202 | +0.000002 | -0.000000 | yes |
| 1 | +0.000134 | +0.000001 | +0.000001 | yes |
| 2 | -0.000084 | -0.000019 | +0.000022 | no |
| 3 | +0.000019 | -0.000013 | +0.000028 | yes |

This confirms that the micro residual improves stability relative to dense16
but still does not meet the paper-clean all-offset gate. Room remains fallback.

## Room Gamma Trust-Region Positive Result

The rejected room micro candidate failed only one fold. I added
`scripts/car_model/ecsr_blend_checkpoint_delta.py`, which materializes a
same-topology checkpoint blend:

`source + gamma * (candidate - source)`

Only appearance SH tensors are blended by default; geometry and topology remain
fixed. This makes the residual policy a train-only trust-region decision rather
than a fixed-strength write. The held-out test split is not used to select
gamma.

Artifacts:

- gamma 0.75 model:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/gamma_trust_region_v1/room_micro1024_sh1_v9_gamma075/room/model`
- multi-fold decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/multifold_trainval_gate/room_micro1024_sh1_v9_gamma075_trust_v1/room/multifold_trainval_gate.json`
- fixed full9 snapshot:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/fixed_candidate_ladder_v10_gamma_trust_full9/phase_r_fixed_candidate_ladder.md`

Multi-fold result:

| offset | dPSNR | dSSIM | dLPIPS | pass |
|---:|---:|---:|---:|---:|
| 0 | +0.000168 | +0.000001 | +0.000000 | yes |
| 1 | +0.000095 | -0.000009 | -0.000000 | yes |
| 2 | +0.000092 | -0.000000 | +0.000000 | yes |
| 3 | +0.000000 | -0.000000 | +0.000002 | yes |

Report-only held-out test delta vs Phase-J:
`+0.000084` PSNR, `+0.000001` SSIM, `-0.000000` LPIPS.

Negative controls:

- `gamma=0.5` fixed the original offset 2 failure but missed offset 3 by a
  numerical-scale PSNR delta of `-0.000004`, so it is not promoted.
- `facelocal_sh1_v3_micro512` reduced local proxy error but failed render-level
  train-heldout gates on both `room` and `counter`; local face proxy alone is
  not sufficient.

## Bonsai Gamma Trust-Region Negative Result

Because the original `bonsai` dense16 SH1 candidate missed the multi-fold gate
only by a small offset-3 LPIPS violation, I tested the same fixed gamma `0.75`
trust-region used for `room`. This is not a scene-tuned gamma sweep; it applies
the same policy value to a second indoor scene.

Artifacts:

- gamma 0.75 model:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/gamma_trust_region_v1/bonsai_dense_sh1_v6_gamma075/bonsai/model`
- multi-fold decision:
  `outputs/carnet/meshsplatopt/ecsr_phase_r/multifold_trainval_gate/bonsai_dense_sh1_v6_gamma075_trust_v1/bonsai/multifold_trainval_gate.json`
- W&B test ELA run:
  `bonsai_dense_gamma075_test_ela`
- W&B train-fold runs:
  `bonsai_dense_gamma075_multifold_*_o0_train` through
  `bonsai_dense_gamma075_multifold_*_o3_train`

Multi-fold result:

| offset | dPSNR | dSSIM | dLPIPS | pass |
|---:|---:|---:|---:|---:|
| 0 | +0.000416 | -0.000015 | +0.000010 | yes |
| 1 | +0.000359 | +0.000016 | +0.000106 | yes |
| 2 | +0.000031 | -0.000004 | +0.000005 | yes |
| 3 | +0.000072 | +0.000080 | +0.000162 | no |

Report-only held-out test delta vs Phase-J:
`-0.006533` PSNR, `+0.000678` SSIM, `+0.000721` LPIPS.

Decision: do not promote. The train-only gate still rejects the candidate for
the same LPIPS reason, and the report-only test split confirms that reducing
residual strength can worsen perceptual quality on `bonsai` even when SSIM
increases. This is a useful boundary case for the method: `room` benefits from
gamma trust-region residual blending, but `bonsai` needs a different operator
or a perceptual non-regression mechanism rather than a global strength shrink.

Status: `GAMMA_TRUST_REGION_ADDED_ROOM_ACCEPTED_BONSAI_GAMMA_REJECTED_COUNTER_AND_BONSAI_STILL_FALLBACK`.

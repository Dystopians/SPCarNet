# vNext Frame-Contract Audit and Effective-Margin Gate Log

Date: 2026-06-26

This log records the follow-up to `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md`.

## Verdict on the New vNext Prompt

The prompt direction is reasonable and research-relevant: distill a train-only Phase-J/ELA residual teacher into a persistent, parent-preserving residual surface texture with explicit safety certificates. The repository already has most of the infrastructure needed for this direction: teacher evidence caches, face/UV residual atlas fitting, strict target-GT stripping, policy-val gates, fallback/no-op writing, W&B logging, and full9 manifest runners.

However, the current vNext artifact is not yet a quality-success endpoint. The full9 fixed-policy cleanup run is `9/9` complete and `9/9` protocol-pass, but its mean metrics are `25.067699 PSNR / 0.741260 SSIM / 0.306689 LPIPS`, below the local clean MeshSplatting baseline `25.151682 / 0.749018 / 0.287621` and below v106 `25.831280 / 0.760830 / 0.268435`. It should be treated as protocol closure and bottleneck diagnosis, not promoted as the paper method.

## Subagent Audit Summary

Five subagents were used for parallel review:

- repo/codebase mapping: confirmed the vNext entry points, protocol audit files, train/eval pipeline, and artifacts.
- method-gap analysis: confirmed the prompt is implementable but current results miss the success criteria.
- experiment audit: confirmed `0/9` strict RGB wins vs clean in the retained full9 vNext summary and highlighted fallback/frame-contract ambiguity.
- panel tooling review: found that old qualitative panels silently resized mismatched images, lacked selected-frame hashes, and did not encode a clean-best selection contract.
- paper-story synthesis: recommended keeping Phase-J as teacher/upper bound and v106 as the current stronger representation-quality line.

## Implemented Changes

### 1. Strict Qualitative Panel Provenance

Updated:

- `scripts/car_model/build_vnext_qualitative_panels.py`
- `scripts/car_model/build_vnext_cleanbest_batch_panels.py`

New behavior:

- image differences now fail on native-size mismatch instead of silently resizing;
- requested frame names must exist in all inputs;
- `candidate_label` and `reference_label` must match method labels;
- panel manifests now include `schema_version`, `argv`, selected-frame SHA1 hashes, selected-frame native sizes, and alignment policy;
- clean-best selection now records the explicit policy, defaulting to `PSNR + 20*SSIM - 20*LPIPS`;
- clean-best batch generation reports `FRAME_CONTRACT_MISMATCH` instead of writing a misleading panel when frame sizes differ.

### 2. Effective-Margin Policy-Val Gate

Updated:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

New adapter flags:

```text
--enable_policy_val_effective_margin_gate
--min_policy_val_effective_relative_gain
--min_policy_val_effective_ssim_gain
--min_policy_val_effective_l1_gain
--min_policy_val_effective_ssim_cvar20_gain
--min_policy_val_effective_l1_cvar20_gain
```

Motivation:

The previous policy could accept candidates whose train policy-val gains were near numerical noise or whose SSIM/tail risk was already weak. This is unsafe for a paper claim because a tiny train gain does not provide meaningful evidence of target/test generalization. The new gate lets a fixed policy require effect-size margins before any target render is modified.

## Garden Evidence

### Clean-Best Frame Contract

The previous clean-best/base/vNext garden panel is now marked as invalid for fair visual comparison:

```text
status: FRAME_CONTRACT_MISMATCH
error: Cannot diff images with different native sizes: (1600, 1036) vs (1297, 840)
```

Artifact:

- `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/cleanbest_qualitative_batch_summary.md`

Interpretation:

The local clean/base renders and the vNext target-evidence renders are not on the same native image contract. A fair paper figure must not silently resize one into the other for diff maps.

### Same-Resolution Parent vs vNext Diagnostic

To isolate the actual vNext target-evidence effect, a same-resolution parent baseline was exported from the target evidence and compared directly against vNext.

Artifacts:

- `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/garden_same_resolution_diagnostic/garden_target_parent_vs_vnext_same_resolution.png`
- `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/garden_same_resolution_diagnostic/garden_target_parent_vs_vnext_same_resolution_manifest.json`
- `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/garden_same_resolution_diagnostic/reports/garden_target_parent_vs_vnext_same_resolution_results.json`

Same-resolution metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| target-evidence parent | `24.741003036` | `0.754049003` | `0.248023212` |
| vNext structure-aware shrink | `24.741142273` | `0.754052162` | `0.248015299` |
| delta | `+0.000139236` | `+0.000003159` | `-0.000007913` |

Interpretation:

The garden vNext accepted output is nearly identical to the target-evidence parent. This explains why qualitative improvement is hard to see and why this is not a paper-quality visual result.

### Effective-Margin Gate Validation

A new garden run was launched with W&B offline on GPU 2:

```text
output root: /dev/shm/peilincai_spcarnet_vnext_garden_margin_gate_20260626_1350
wandb root: /dev/shm/peilincai_wandb_vnext_garden_margin_gate_20260626_1350
method: ours_26000_vnext_effective_margin_gate
```

Key flags:

```text
--enable_policy_val_effective_margin_gate
--min_policy_val_effective_relative_gain 0.02
--min_policy_val_effective_ssim_gain 0.00001
--min_policy_val_effective_l1_gain 0.000001
--min_policy_val_effective_ssim_cvar20_gain 0.000001
--min_policy_val_effective_l1_cvar20_gain 0.0
```

Result:

| field | value |
|---|---|
| run status | `COMPLETE` |
| protocol audit | `passed=true` |
| accepted | `false` |
| selected alpha | `0.0` |
| effective policy | `fallback_noop` |
| fallback source | `target_evidence` |
| changed fraction | `0.0` |
| W&B mode | `offline` |

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| effective-margin fallback | `24.741003036` | `0.754049003` | `0.248023212` |

Main rejection reasons include negative lower-tail and SSIM evidence:

```text
cvar20_view_relative_gain -0.088840 < 0
min_view_relative_gain -0.170154 < -0.000001
ssim_gain -0.000009457 < -0.000000100
ssim_positive_view_fraction 0.416667 < 0.55
effective_ssim_gain -0.000009457 < 0.000010000
effective_ssim_cvar20_view_gain -0.000056684 < 0.000001000
```

Artifacts:

- `docs/car_model/vnext_artifacts/garden_effective_margin_gate_20260626/reports/garden_vnext_certified_residual_texture_manifest.json`
- `docs/car_model/vnext_artifacts/garden_effective_margin_gate_20260626/reports/garden_ours_26000_vnext_effective_margin_gate_test_results.json`
- `docs/car_model/vnext_artifacts/garden_effective_margin_gate_20260626/model_audits/surface_residual_region_texture_adapter_audit.json`
- `docs/car_model/vnext_artifacts/garden_effective_margin_gate_20260626/logs/02_certified_texture.log`

## Full9 Effective-Margin Gate Validation

The same effective-margin gate was then run on all nine selected scenes with
W&B offline logging and the fixed full9 manifest config.

Artifact root:

```text
docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500
```

Run root:

```text
/dev/shm/peilincai_spcarnet_vnext_full9_margin_gate_20260626_1500
```

Summary:

| item | value |
|---|---:|
| runner status | `COMPLETE` |
| completed scenes | `9 / 9` |
| failed scenes | `0 / 9` |
| missing inputs | `0 / 9` |
| protocol pass | `9 / 9` |
| accepted nonzero | `1 / 9` |
| fallback/no-op | `8 / 9` |
| mean changed fraction | `0.001371507` |
| mean PSNR | `25.067410` |
| mean SSIM | `0.741259` |
| mean LPIPS | `0.306695` |

Scene decisions:

| scene | decision | policy | alpha | changed fraction |
|---|---|---|---:|---:|
| bicycle | rejected | `fallback_noop` | `0.0` | `0.0` |
| bonsai | rejected | `fallback_noop` | `0.0` | `0.0` |
| counter | accepted | `accepted_atlas` | `0.125` | `0.012343567` |
| flowers | rejected | `fallback_noop` | `0.0` | `0.0` |
| garden | rejected | `fallback_noop` | `0.0` | `0.0` |
| kitchen | rejected | `fallback_noop` | `0.0` | `0.0` |
| room | rejected | `fallback_noop` | `0.0` | `0.0` |
| stump | rejected | `fallback_noop` | `0.0` | `0.0` |
| treehill | rejected | `fallback_noop` | `0.0` | `0.0` |

Comparison:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | `25.151682` | `0.749018` | `0.287621` |
| v106 POD-MoE base-preserve | `25.831280` | `0.760830` | `0.268435` |
| vNext fixed-policy cleanup | `25.067699` | `0.741260` | `0.306689` |
| vNext effective-margin gate | `25.067410` | `0.741259` | `0.306695` |

Interpretation:

This closes the full9 effective-margin safety audit. The stricter gate suppresses
almost all low-effect residual texture candidates, but it does not create a
quality-superior method. The run confirms that the current vNext representation
has insufficient target impact: only `counter` survives the effect-size and
lower-tail checks.

Key artifacts:

- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_runner_summary.md`
- `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/*/model_audits/surface_residual_region_texture_adapter_audit.json`

## Current Interpretation

This milestone improves the reliability of the vNext evidence pipeline but does not solve the quality gap.

What improved:

- the qualitative comparison pipeline now rejects unfair frame-size contracts;
- clean-best checkpoint selection is recorded instead of implicit;
- selected panel frames are hash-auditable;
- the adapter can reject low-effect policy-val candidates instead of accepting numerically tiny gains;
- garden margin-gate validation correctly falls back to parent/no-op under strict no-target-GT apply.
- full9 effective-margin validation is complete and correctly rejects `8 / 9` scenes to fallback/no-op.

What remains weak:

- vNext still does not beat clean MeshSplatting or v106 on the retained full9 table;
- the effect size of accepted residual surface texture is extremely small;
- same-resolution garden vNext improves target-evidence parent only by micro-deltas;
- the stricter full9 gate leaves only `counter` as a nonzero accepted scene;
- the current residual surface texture is therefore a weak representation edit, not a paper-final visual endpoint.

## Next Required Work

1. For paper visuals, use only panels with verified matching native size and selected-frame hashes.
2. Build a same-frame-contract clean/parent/v106/vNext comparison table; do not mix official clean renders and target-evidence renders unless resize/camera contracts are explicitly proven equivalent.
3. Treat Phase-J as teacher/upper bound and v106 as current stronger representation endpoint until vNext produces meaningful gains.
4. Investigate why the residual surface field has such small target impact; likely next directions are adaptive capacity, stronger teacher residual field distillation, or region-local neural residual decoding rather than more global gates.
5. Any next vNext promotion must beat clean MeshSplatting and v106 under this full9 protocol, not merely pass the no-op fallback safety audit.

## Final Status for This Milestone

`COMPLETE` for the frame-contract audit, panel provenance hardening, effective-margin gate implementation, garden rejection validation, and full9 effective-margin safety validation.

`NOT COMPLETE` for the overall paper-level vNext objective.

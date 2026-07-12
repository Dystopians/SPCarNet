# 6-28 v162 Sparse-Selective Bridge Log

Date: 2026-06-28

## Purpose

v161 fixed the hard failure from v160, but it exposed a second certification-semantics bug: after sparse residual materialization was bridged through an empty bin-uncertainty intersection, the re-evaluated policy-val payload no longer carried sparse-selective non-regression annotation. That caused no-op views outside the sparse allowlist to be judged as strict positive-view failures, reducing the final selected alpha.

v162 is a targeted method/engineering fix for that issue. It is not a parameter scan.

## Code Change

Touched files:

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

Changes:

- Preserve sparse-selective non-regression semantics after bin-guard intersection or bridge by re-annotating the guarded policy-val payload with `annotate_sparse_materialization_selective_policy_val(...)`.
- Record a `sparse_selective_annotation` audit block inside `bin_uncertainty_guard_profile`.
- Change the vNext scene runner default for `--bin_uncertainty_guard_empty_intersection_policy` from `reject` to `sparse_if_post_accepted`. The standalone adapter default remains strict unless explicitly configured.

## Mechanistic Check

A direct audit simulation on the previous v161 JSON showed the expected effect:

| condition | accepted | selected alpha | relative gain | SSIM nonnegative views | L1 nonnegative views |
|---|---:|---:|---:|---:|---:|
| without restored sparse annotation | true | 0.0625 | 0.009115777 | n/a | n/a |
| with restored sparse annotation | true | 0.3750 | 0.031881346 | 0.916666667 | 1.000000000 |

This confirmed that the v161 low-alpha result was partly a policy-val annotation problem rather than a residual-generation failure.

## Dry Run

Dry-run output:

```text
/dev/shm/peilincai_spcarnet_20260628_0330_v162_sparse_selective_dryrun
```

Dry-run status:

- manifest status: `DRY_RUN`
- command count: `3`
- errors: `0`
- protocol audit: passed
- W&B offline run: `/data/peilincai/mesh-splatting/wandb/offline-run-20260628_023015-2lptb7y3`

## Real Flowers Run

Output root:

```text
/dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective
```

Primary artifacts:

```text
manifest: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json
adapter audit: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/model/surface_residual_region_texture_adapter_audit.json
results: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/reports/flowers_ours_26000_v162_sparse_selective_flowers_test_results.json
per-view: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/reports/flowers_ours_26000_v162_sparse_selective_flowers_test_per_view.json
texture log: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/logs/02_certified_texture.log
eval log: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/logs/03_eval.log
W&B offline run: /data/peilincai/mesh-splatting/wandb/offline-run-20260628_040818-dqbhw1cy
```

Runner settings:

- scene: `flowers`
- method: `ours_26000_v162_sparse_selective_flowers`
- GPU: `5`
- W&B: enabled, offline mode
- strict no-target-GT apply: enabled
- apply evidence: `/dev/shm/peilincai_spcarnet_v131b_viewconf_flowers_20260626_223006/flowers/target_evidence_no_gt`
- eval GT evidence: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/target_evidence`

Manifest result:

- status: `COMPLETE`
- errors: `[]`
- protocol audit: passed
- target GT visible to apply: `false`
- target GT visible to eval: `true`
- selection uses test GT: `false`
- adapter elapsed: `5771.652s`
- eval GT population elapsed: `25.619s`
- final metric evaluation elapsed: `45.216s`

## Quantitative Result

| version | accepted | alpha | allowed bins | allowed faces | changed pixels | png-changed pixels | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v159 sparse face-guard skip | true | 0.3750 | 40 | 5 | 466 | 465 | 20.452793 | 0.549059 | 0.355544 |
| v161 bridge | true | 0.0625 | 121 | 13 | 860 | 689 | 20.452782 | 0.549059 | 0.355544 |
| v162 sparse-selective bridge | true | 0.3750 | 121 | 13 | 860 | 849 | 20.452797 | 0.549059 | 0.355544 |

v162 deltas:

- vs v161: selected alpha improves from `0.0625` to `0.3750`; PSNR improves by about `+0.000015259`; SSIM decreases by about `-0.000000119`; LPIPS worsens by about `+0.000000060`.
- vs v159: target changed pixels increase from `466` to `860`, and PNG-quantized changed pixels increase from `465` to `849`.
- The changed-pixel fraction is still only `2.318009315e-05`, so the final full-image metrics remain almost unchanged.

## Audit Interpretation

v162 proves a real mechanism-level repair:

- sparse materialization post-gate accepts at `alpha=0.375`;
- target-visible expansion grows the allowed bins from `40` to `121`;
- added target-visible bins: `81`;
- added target-visible pixels: `479`;
- final allowlist: `121` bins / `13` faces;
- bin uncertainty guard still has an empty hard intersection, but `sparse_if_post_accepted` bridge activates safely;
- sparse-selective annotation is now preserved through the post-guard policy-val pass.

The key remaining bottleneck is footprint, not alpha. v162 certifies a stronger alpha over the same tiny footprint. It therefore improves the engineering correctness of the method, but not enough visual area to create a large full-image metric or qualitative gain.

## Engineering Assessment

Strong:

- Real train/eval pipeline method change implemented.
- Strict no-target-GT apply protocol preserved.
- W&B offline logging active.
- Full command manifest, protocol audit, result JSON, per-view JSON, adapter audit, and logs are saved.
- The v161 annotation bug is fixed and directly verified by a successful real run.

Weak:

- Runtime is high: the adapter command took about `5771.652s` on a single flowers scene. Most time is repeated CPU-heavy policy-val passes, not GPU training.
- The target changed area is still extremely small.
- The run is a single-scene diagnostic, not a full9 promotion run.
- The metrics are not visibly or statistically large enough to claim paper-level quality superiority.

## Paper-Level Assessment

v162 should be recorded as an engineering/method milestone, not as a paper-final endpoint.

Credible claim:

```text
Sparse-selective certification must preserve its non-regression semantics after uncertainty-guard bridging; otherwise safe sparse edits are artificially down-weighted during post-guard policy validation. v162 fixes this and restores the intended alpha while preserving strict no-target-GT apply.
```

Not yet credible:

```text
v162 fully solves the vNext quality gap or beats MeshSplatting/v106 on the selected benchmark.
```

## Next Technical Direction

The next meaningful step is v163 support expansion, not another alpha/threshold adjustment. The current method can now certify the sparse bins it trusts; it must expand the trusted support footprint before a large visual or metric gain is plausible.

Candidate next experiment:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_vnext_certified_residual_texture_scene.py \
  --scene flowers \
  --output_root /dev/shm/peilincai_spcarnet_20260628_v163_support_expansion \
  --method_name ours_26000_v163_support_expansion_flowers \
  --support_expansion_mode target_footprint_residual_debt \
  --support_expansion_max_extra_faces_candidates 2048,4096 \
  --strict_no_target_gt_apply \
  --bin_uncertainty_guard_empty_intersection_policy sparse_if_post_accepted \
  --wandb --wandb_mode offline --wandb_group v163_support_expansion
```

This should keep the v162 sparse-selective fix while testing whether target-footprint support expansion can increase edited area beyond the current `860 / 37,100,800` pixels.

## Final Status

Final status: **NOT COMPLETE**.

v162 is a real method fix and closes a correctness gap in the vNext certification path. It does not yet close the paper-level benchmark gap because the edited footprint remains too small and the full-image metric/visual effect is still negligible.

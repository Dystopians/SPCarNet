# SPCarNet Current Metrics / Engineering / Paper-Level Evaluation

Date: 2026-06-28

This report is a current-state audit for the SPCarNet / MeshSplatting repair line. It separates three different claims that must not be mixed:

1. the strongest historical RGB endpoint;
2. the strongest verified MeshSplatting-compatible representation endpoint;
3. the newer vNext certified residual surface texture route.

## Executive Conclusion

Current status: **NOT COMPLETE for a paper-final closed loop**.

Metrics status:

- The strongest verified MeshSplatting-compatible representation line is still **v106 POD-MoE base-preserve**.
- v106 is positive against the local clean MeshSplatting full9 baseline: `+0.679598` PSNR, `+0.011812` SSIM, `-0.019185` LPIPS.
- The newer vNext certified residual surface texture route has a real engineering/protocol loop, but the latest completed full9 metrics are below clean MeshSplatting and below v106.
- The latest v162 flowers sparse-selective bridge experiment completed successfully and fixed a v161 post-guard annotation weakness. It restores the intended sparse-materialization alpha, but the edited footprint is still tiny, so the full-image metric change remains negligible.

Engineering status:

- The project has strong audit infrastructure: full9 runners, no-target-GT apply protocol, manifest logging, eval GT population audit, W&B offline runs, topology audits, model audits, and archived report artifacts.
- The current vNext runner and adapter have been extended to fix a real protocol/interface bug: apply uses target evidence without GT, while final eval can now use a separate `--eval_gt_evidence_dir`.
- The v161/v162 adapter line adds a risk-bounded bridge for the case where sparse residual materialization passes post-gate but the bin-uncertainty hard intersection is empty, and v162 preserves sparse-selective non-regression semantics after that bridge.

Paper-level status:

- The work is credible as an evidence-certified MeshSplatting repair program, but it is **not yet a convincing final top-conference endpoint**.
- The strongest broad RGB gains still come from the older Phase-J / v101-v102 style endpoint, which is not a clean baked representation claim.
- The strongest baked/representation result, v106, is stable and positive, but its incremental gain over v104c is small.
- vNext is currently best described as a protocol and representation scaffold plus bottleneck diagnosis, not yet a quality-superior final method.

## Full9 Quantitative Summary

Local clean MeshSplatting is the fair local baseline under the selected full9 evaluator.

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean | current role |
|---|---:|---:|---:|---:|---|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline | local fair baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 | stable representation anchor |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 | current verified representation line |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 | +1.329628 / +0.034657 / -0.063316 | strong RGB endpoint/reference, but not the cleanest baked representation claim |
| vNext structure-aware shrink cleanup | 9 | 25.067699 | 0.741260 | 0.306689 | -0.083983 / -0.007758 / +0.019068 | protocol-complete, not promoted |
| vNext effective-margin gate | 9 | 25.067410 | 0.741259 | 0.306695 | -0.084272 / -0.007759 / +0.019074 | safer gate, still not promoted |

Interpretation:

- v106 clears the local clean MeshSplatting baseline on all aggregate metrics.
- v106's additional gain over v104c is small: `+0.002181` PSNR, `+0.000103` SSIM, `-0.000112` LPIPS.
- vNext full9 does not meet the expected metric bar. It is worse than clean MeshSplatting on PSNR/SSIM/LPIPS and therefore cannot be used as the headline endpoint.

## v106 Per-Scene Evidence

Source: `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`.

| scene | PSNR | SSIM | LPIPS | dPSNR vs v104c | dSSIM vs v104c | dLPIPS vs v104c |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.719175 | 0.675086 | 0.313405 | +0.001526 | +0.000115 | -0.000098 |
| flowers | 20.077723 | 0.531240 | 0.374393 | +0.001879 | +0.000163 | -0.000080 |
| garden | 25.790945 | 0.799382 | 0.174480 | +0.002851 | +0.000119 | -0.000104 |
| stump | 25.460457 | 0.714661 | 0.282135 | +0.001146 | +0.000061 | -0.000078 |
| treehill | 21.245092 | 0.578518 | 0.384177 | +0.001329 | +0.000099 | -0.000121 |
| room | 29.600351 | 0.891889 | 0.230616 | +0.002516 | +0.000051 | -0.000048 |
| counter | 27.499645 | 0.867521 | 0.238847 | +0.001577 | +0.000102 | -0.000139 |
| kitchen | 28.772043 | 0.881652 | 0.187815 | +0.001595 | +0.000062 | -0.000206 |
| bonsai | 30.316090 | 0.907520 | 0.230050 | +0.005213 | +0.000154 | -0.000136 |

v106 is consistent and stable. The weakness is not direction; the weakness is effect size.

## vNext Full9 Completed Evidence

Source: `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`.

| scene | protocol pass | accepted | alpha | changed fraction | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | true | true | 0.015625 | 0.000173916 | 23.293516 | 0.659651 | 0.332269 |
| bonsai | true | true | 0.250000 | 0.001489739 | 28.865479 | 0.896003 | 0.259323 |
| counter | true | true | 0.125000 | 0.012343567 | 26.751171 | 0.862042 | 0.251955 |
| flowers | true | false | 0.000000 | 0.000000000 | 19.519194 | 0.490780 | 0.424170 |
| garden | true | true | 0.125000 | 0.002050379 | 24.741142 | 0.754052 | 0.248015 |
| kitchen | true | true | 0.125000 | 0.003549714 | 27.817173 | 0.876445 | 0.199172 |
| room | true | true | 0.062500 | 0.005199120 | 28.739571 | 0.884797 | 0.249909 |
| stump | true | false | 0.000000 | 0.000000000 | 25.043329 | 0.689480 | 0.349850 |
| treehill | true | false | 0.000000 | 0.000000000 | 20.838715 | 0.558089 | 0.445541 |

Completed vNext full9 facts:

- `9 / 9` scenes completed.
- `9 / 9` protocol audits passed.
- `6 / 9` scenes accepted nonzero residual output.
- `3 / 9` scenes fell back/no-op.
- Mean changed fraction is only `0.002756271`.
- Mean metrics are `25.067699 / 0.741260 / 0.306689`.

The protocol is strong, but the metric result is not strong enough.

## Latest v159 / v160 / v161 / v162 Flowers Diagnostic

This is the current active vNext improvement thread after the full9 bottleneck.

| version | status | key mechanism | accepted | alpha | changed pixels | metrics / result | diagnosis |
|---|---|---|---:|---:|---:|---|---|
| v159 | complete | sparse residual materialization with face-guard skip | true | 0.3750 | 466 / 37,100,800 | `20.452793 / 0.549059 / 0.355544` | positive but extremely sparse; useful proof-of-life, not a full9 claim |
| v160 | failed/rejected | target-visible sparse growth | false | 0.0000 | 0 | no final eval; manifest failed at eval GT population | target-visible expansion found more bins, but bin-uncertainty hard intersection became empty; runner also tried to populate eval GT from no-GT target evidence |
| v161 | complete | bridge sparse profile when post-gate accepted and bin guard intersection is empty; separate eval GT evidence path | true | 0.0625 | 860 / 37,100,800 | `20.452782 / 0.549059 / 0.355544` | protocol complete and bridge activated, but post-guard sparse-selective semantics were lost and alpha collapsed |
| v162 | complete | preserve sparse-selective annotation after bin-guard bridge; runner default uses sparse-if-post-accepted bridge | true | 0.3750 | 860 / 37,100,800 | `20.452797 / 0.549059 / 0.355544` | real correctness fix: alpha restored and PNG-changed pixels increase, but footprint is still too small for a visible quality breakthrough |

v160 diagnostic details:

- sparse allowed bins grew from `40` to `121`;
- added target-visible bins: `81`;
- added target-visible pixels: `479`;
- post-materialization gate accepted;
- bin uncertainty guard allowed `0` bins;
- hard intersection produced `0` bins;
- adapter wrote fallback/no-op and then eval GT population failed because it used no-GT target evidence.

v161 completed diagnostic:

- `run_vnext_certified_residual_texture_scene.py` now supports `--eval_gt_evidence_dir`, so final evaluation can use GT while apply remains strict no-target-GT.
- `ecsr_apply_surface_residual_region_texture_adapter.py` now supports `--bin_uncertainty_guard_empty_intersection_policy sparse_if_post_accepted`.
- If sparse materialization already passes its post-gate and the bin guard is empty, v161 can bridge to the sparse profile instead of forcing a no-op.
- v161 manifest status is `COMPLETE`, with `errors=[]` and protocol audit passed.
- adapter audit: `accepted=true`, `fallback_written=false`, `selected_alpha=0.0625`.
- target apply: `changed_pixels=860`, `changed_fraction=0.000023180093`.
- target-visible expansion again grew sparse bins from `40` to `121`, adding `81` bins and `479` target-visible pixels.
- bin uncertainty guard still produced an empty hard intersection, but bridge policy activated: `bridge_activated=true`, `bridge_allowed_bin_count=121`.
- final flowers result: `20.452781677 PSNR / 0.549059272 SSIM / 0.355543971 LPIPS`.
- versus v159, v161 increases edited target pixels from `466` to `860`, improves LPIPS by about `0.000000119`, ties SSIM to displayed precision, but PSNR is about `0.000011444` lower. This is not a meaningful quality gain.
- Static checks passed:
  - `python -m py_compile scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`
  - `git diff --check -- scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- v161 dry-run passed with no manifest errors.

v161 run artifacts:

```text
output root: /dev/shm/peilincai_spcarnet_20260628_0113_v161_bridge
manifest: /dev/shm/peilincai_spcarnet_20260628_0113_v161_bridge/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json
texture log: /dev/shm/peilincai_spcarnet_20260628_0113_v161_bridge/flowers/logs/02_certified_texture.log
adapter audit: /dev/shm/peilincai_spcarnet_20260628_0113_v161_bridge/flowers/model/surface_residual_region_texture_adapter_audit.json
results: /dev/shm/peilincai_spcarnet_20260628_0113_v161_bridge/flowers/reports/flowers_ours_26000_v161_bridge_flowers_test_results.json
wandb offline run: /dev/shm/peilincai_wandb_20260628_0113_v161_bridge/wandb/offline-run-20260628_022105-ki393dhg
```

v162 completed diagnostic:

- `ecsr_apply_surface_residual_region_texture_adapter.py` now preserves sparse-selective non-regression semantics after bin-guard intersection or bridge by re-annotating the guarded policy-val payload.
- `run_vnext_certified_residual_texture_scene.py` now defaults `--bin_uncertainty_guard_empty_intersection_policy` to `sparse_if_post_accepted`, while the standalone adapter remains strict unless configured.
- v162 dry-run passed with manifest status `DRY_RUN`, command count `3`, errors `0`, protocol audit passed, and W&B offline logging enabled.
- v162 real run manifest status is `COMPLETE`, with `errors=[]`, protocol audit passed, and W&B offline logging enabled.
- adapter elapsed time: `5771.652s`; eval GT population: `25.619s`; final metric evaluation: `45.216s`.
- adapter audit: `accepted=true`, `fallback_written=false`, `selected_alpha=0.375`.
- bin guard: `allowed_bin_count=121`, `allowed_face_count=13`, `decision=keep_bin_uncertainty_guard`, `reason=sparse_materialization_post_gate_accepted_bin_guard_empty`.
- sparse-selective annotation preserved: source sparse allowed bins `121`, guard allowed bins `121`.
- target apply: `changed_pixels=860`, `png_quantized_changed_pixels=849`, `changed_fraction=0.000023180093`.
- final flowers result: `20.452796936 PSNR / 0.549059153 SSIM / 0.355544031 LPIPS`.
- versus v161, v162 restores alpha from `0.0625` to `0.3750`, improves PSNR by about `+0.000015259`, slightly lowers SSIM by about `-0.000000119`, and changes LPIPS by about `+0.000000060`.
- versus v159, v162 increases changed pixels from `466` to `860` and PNG-quantized changed pixels from `465` to `849`, but the full-image metric delta remains negligible.

v162 run artifacts:

```text
output root: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective
manifest: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json
texture log: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/logs/02_certified_texture.log
adapter audit: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/model/surface_residual_region_texture_adapter_audit.json
results: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/reports/flowers_ours_26000_v162_sparse_selective_flowers_test_results.json
per-view: /dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/flowers/reports/flowers_ours_26000_v162_sparse_selective_flowers_test_per_view.json
wandb offline run: /data/peilincai/mesh-splatting/wandb/offline-run-20260628_040818-dqbhw1cy
```

The correct v162 conclusion is narrow: it fixes a real certification semantics bug and restores the intended sparse alpha, but the support footprint remains the main bottleneck. It is a method-correctness milestone, not a paper-quality endpoint.

## Engineering Evaluation

What is strong:

- The codebase now has real method-level components, not just parameter scans: residual fields, POD-MoE experts, shrink gates, policy validation, sparse materialization, target-visible expansion, uncertainty guards, effective-margin gates, and exact fallback behavior.
- The vNext runner has a better fairness interface after the `eval_gt_evidence_dir` split.
- The apply stage can be audited to hide target/test GT.
- Completed full9 vNext artifacts were copied into the repo under `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/`.
- The v106 full9 result package is already repo-local under `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/`.

What is weak:

- vNext full9 quality is below clean MeshSplatting and v106.
- vNext accepted edits are often tiny; qualitative improvements are therefore hard to see.
- v162 completed and fixed the sparse-selective post-guard semantics, but its metric effect is still tiny and mixed; it does not justify full9 promotion by itself.
- `/data` is nearly full, so new large artifacts must go to `/dev/shm` or cleaned storage.
- Policy-val runtime is slow; v161 flowers took about `4933.77s` and v162 flowers took about `5771.652s` for the adapter command before final eval.

## Qualitative Evidence Status

Useful existing visual artifacts:

- v106 garden qualitative panel: `docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png`
- vNext garden face-softshrink panel: `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png`
- Same-resolution garden diagnostic panel: `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626/garden_same_resolution_diagnostic/garden_target_parent_vs_vnext_same_resolution.png`

Current qualitative conclusion:

- Existing visuals are useful for audit and demonstration, but they do not yet show a visually obvious vNext advantage.
- For a mentor/PPT, the honest visual story should emphasize v106/Phase-J as stronger visual endpoints and present vNext as a representation-level distillation attempt whose current changes are still too subtle.

## Paper-Level Evaluation

Does the current project meet the paper-level target?

No, not yet.

A defensible paper-facing story exists:

```text
MeshSplatting is a strong mesh renderer.
SPCarNet adds evidence-certified residual repair and conservative representation-level correction.
The system can audit where repairs are supported, preserve parent behavior where evidence is weak, and improve local full9 baselines under strict held-out evaluation.
```

But the final top-conference bar is not fully met because:

- the strongest broad metric gain is still tied to older render-time/post-training repair;
- the cleanest baked representation line, v106, is stable but only a small incremental improvement over v104c;
- the newest vNext representation route is not yet metric-superior to clean MeshSplatting;
- visual differences in vNext are not obvious enough for a compelling qualitative story;
- v162 has completed only as a single-scene diagnostic and has not shown a meaningful single-scene metric gain or full9 validation.

## Immediate Next Actions

1. Treat v162 as an engineering/method-correctness milestone and diagnostic, not as a promoted quality endpoint.
2. Do not spend the next iteration only on scalar thresholds; the bottleneck is support footprint and representation capacity, not merely alpha or bridge policy.
3. If continuing vNext, test support expansion (`target_footprint_residual_debt` or a similar target-footprint-aware support growth policy) while preserving the v162 sparse-selective guard semantics.
4. Do not promote vNext until a fixed policy beats clean MeshSplatting and v106 on the same full9 protocol.
5. For a mentor update, lead with v106 and Phase-J evidence, then frame vNext as the active representation-level research frontier.

## Evidence Index

Primary repo-local evidence:

- `README.md`
- `docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md`
- `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`
- `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md`
- `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md`
- `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/vnext_full9_cleanup_promotion_manifest.md`

Current live evidence outside repo:

- `/dev/shm/peilincai_spcarnet_20260627_2145_v159_sparse_faceguard_skip/`
- `/dev/shm/peilincai_spcarnet_20260628_0018_v160_target_visible_growth/`
- `/dev/shm/peilincai_spcarnet_20260628_0110_v161_bridge_dryrun/`
- `/dev/shm/peilincai_spcarnet_20260628_0113_v161_bridge/`
- `/dev/shm/peilincai_spcarnet_20260628_0330_v162_sparse_selective_dryrun/`
- `/dev/shm/peilincai_spcarnet_20260628_0335_v162_sparse_selective/`

New v162 log:

- `docs/car_model/6-28-v162-SparseSelectiveBridge-Log.md`

## Final Status

Final status: **NOT COMPLETE**.

Unfinished checklist item: the currently completed vNext full9 metrics do not beat clean MeshSplatting or v106, and v162's completed flowers sparse-selective bridge is only a tiny single-scene diagnostic despite fixing a real certification bug. The project has strong engineering scaffolding and a verified positive v106 representation line, but it has not yet reached a paper-final, visually compelling, fully superior vNext endpoint.

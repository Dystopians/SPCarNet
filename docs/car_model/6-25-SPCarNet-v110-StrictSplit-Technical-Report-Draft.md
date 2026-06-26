# SPCarNet v110 Strict-Split Technical Report Draft

Date: 2026-06-25

Audience: mentor discussion and PPT conversion.

Scope: this draft uses the existing v106/v109/v110 local logs plus the current `/dev/shm` result snapshot. It separates quality evidence from fairness/safety evidence. The current quality headline is v106 POD-MoE base-preserve; v109 and v110 are parent-gate / strict-split validation steps and must not be presented as quality breakthroughs unless later reports show completed positive test results.

## 0. Slide-Ready Takeaway

Safe one-slide message:

```text
MeshSplatting gives the base scene render.
SPCarNet attaches a guarded, surface-addressed residual layer to the visible mesh.
v106 is the currently verified quality line.
v109/v110 are fairness and safety gates that protect the v106 parent from harmful candidate edits.
```

Current status:

| line | current role | verified result | claim boundary |
|---|---|---|---|
| clean MeshSplatting | unmodified local render baseline | mean PSNR 25.151682, SSIM 0.749018, LPIPS 0.287621 on selected full9 baseline table | baseline only |
| v104c shrink view-affine | stable field anchor | 9/9 scenes improve over clean; mean +0.677417 PSNR, +0.011709 SSIM, -0.019073 LPIPS | still below endpoint/reference ceiling |
| v106 POD-MoE base-preserve | current verified quality candidate | current assembled 9/9 table improves over v104c by +0.002181 PSNR, +0.000103 SSIM, -0.000112 LPIPS; every scene has the favorable sign vs v104c | gain is small; evidence source is mixed full9/counter/hardtriad and should be promoted with that provenance |
| v109 render-realized parent gate | safety/fairness gate over a v108 candidate | flowers gate rejects harmful v108 normal-equation candidate and exactly preserves v106 without target/test GT policy selection | no quality improvement beyond v106 |
| v110/v110b strict-split parent gate | stricter protocol: fit train/even, calibrate train/odd, evaluate test | flowers default v110 is a false accept; flowers v110b preserves v106; garden v110b still regresses vs v106 while beating clean | not promotable; useful safety/fairness diagnostic |

PPT-safe headline:

> SPCarNet now has a verified v106 residual-field improvement over the v104c field anchor. v109/v110/v110b are protocol milestones that exposed the fairness gap: parent-preserving gates can prevent some regressions, but strict train/odd calibration is not yet enough to beat v106 on held-out test.

## 1. 方法简介

SPCarNet treats MeshSplatting's trained scene as the parent representation and adds a surface-addressed residual correction layer. The mesh triangles are used as an address space: each rendered pixel knows which triangle it comes from, so residual evidence can be stored and replayed on the surface rather than as a free target image delta.

The current method ladder is:

```text
clean MeshSplatting checkpoint
  -> v104c shrink view-affine residual field
  -> v106 POD-MoE base-preserve residual field
  -> v109 render-realized parent gate
  -> v110 strict-split parent gate
```

The important distinction is that these are not all quality lines:

- v104c and v106 are residual-field representation lines with measured render-quality improvements.
- v109 is a render-space parent preservation gate. It can reject a harmful candidate and fall back to the parent.
- v110 is the stricter fairness protocol around the parent gate. It separates candidate fitting, gate calibration, and test evaluation.

For mentor discussion, the story should be framed as:

1. v106 shows a small but consistent representation improvement over v104c.
2. v108 showed proxy certificates can pass while render metrics regress, so render-realized parent protection is needed.
3. v109 proves the gate can safely reject a harmful candidate on flowers.
4. v110 is the strict-split version needed before claiming paper-grade fairness.

## 2. 模块作用

| module | role | what it consumes | what it outputs | current evidence status |
|---|---|---|---|---|
| clean MeshSplatting parent | base render and geometry/address space | trained checkpoint, cameras | RGB render, triangle visibility | selected local clean baseline |
| v101/v102 endpoint evidence | high-quality reference/teacher evidence | train-derived render residual/depth/camera evidence | endpoint/preprojected delta banks | useful ceiling/reference, not a vanilla field result |
| v104c shrink view-affine field | stable compact residual field | triangle id, barycentric coordinate, view direction | one shrink-stabilized affine residual per triangle | full9 anchor, 9/9 better than clean |
| v106 POD-MoE base-preserve | current quality candidate | v104c-like base plus detail/boundary residual cues | base residual plus two additive expert corrections | current assembled 9/9 improves over v104c by small margins |
| detail expert | recover local texture/detail residuals | luminance/detail cues from base and teacher residual | additive RGB residual delta | included in v106 field artifacts |
| occlusion-boundary expert | handle residuals near visibility/depth/triangle boundaries | rendered boundary cues and reliability stats | additive boundary residual delta | included in v106 field artifacts |
| base-preserve renderer | prevents expert from deleting stable base | base residual plus weighted expert deltas | adapted render | fixes earlier POD-MoE counter direction problem |
| v109 parent gate | render-realized candidate filter | parent render, candidate render, calibration GT only on calibration split | masked blend or parent fallback | flowers fallback-to-parent validated; no test GT used for policy |
| v110 strict-split runner | fairness orchestration | train/even candidate fit, train/odd gate calibration, test eval | per-scene strict report | implemented; flowers/garden diagnostics completed, not promoted |

v106 render identity:

```text
rendered residual =
  v104c-like base residual
  + weighted detail expert delta
  + weighted occlusion-boundary expert delta
```

v109/v110 parent-gate identity:

```text
output = parent + mask(candidate, parent) * (candidate - parent)
```

If the calibration split cannot prove the candidate is safe, the gate selects `mask = 0`, so output equals the parent. This is valuable safety evidence, but it is not a quality improvement.

## 3. 与 MeshSplatting Baseline 的区别

Clean MeshSplatting renders the trained checkpoint directly:

```text
checkpoint -> held-out test render -> PSNR / SSIM / LPIPS
```

SPCarNet adds three extra ideas:

1. Surface addressing: residuals are attached to visible mesh triangles rather than stored as unrelated image-space edits.
2. Evidence-gated residual replay: corrections are only applied through certified field artifacts or render-realized gates.
3. Parent preservation: v106 keeps the stable v104c-like base residual, and v109/v110 can fall back exactly to the v106 parent when a candidate is unsafe.

Important baseline boundary:

- SPCarNet is not a vanilla MeshSplatting checkpoint. It needs the SPCarNet field/gate path in rendering.
- v106 does not claim new topology compression. It inherits the parent geometry/topology and changes rendered appearance through residual fields.
- v109/v110 do not replace v106's quality evidence. They test whether future candidate edits can be selected under stricter fairness and safety rules.

## 4. 已验证定量证据路径

### 4.1 v104c and v106 Mean Metrics

Current assembled evidence:

```text
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/v106_podmoe_basepreserve_full9_strict_assembled_report.md
```

| method | scenes | PSNR | SSIM | LPIPS | relation |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | selected local baseline |
| v104c shrink view-affine | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 PSNR, +0.011709 SSIM, -0.019073 LPIPS vs clean |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 PSNR, +0.011812 SSIM, -0.019185 LPIPS vs clean |

v106 vs v104c:

| comparison | dPSNR | dSSIM | dLPIPS | interpretation |
|---|---:|---:|---:|---|
| v106 - v104c | +0.002181 | +0.000103 | -0.000112 | small but consistent full9 assembled gain |

Scene-level note: the assembled v106 table has 9/9 scenes available and every listed scene has favorable signs vs v104c on PSNR, SSIM, and LPIPS. The selected sources are mixed:

- `bicycle`, `flowers`, `garden`, `room`, `stump`, `treehill`: full9 report root;
- `counter`: counter report root;
- `kitchen`, `bonsai`: hard-triad report root.

This is acceptable for an internal mentor/PPT progress slide if the provenance is shown. For paper tables, re-export from one durable result package.

### 4.2 v106 Hard-Triad Direction Check

Evidence:

```text
docs/car_model/6-25-v106-PODMoE-BasePreserve-HardTriad-Log.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_counter_20260625_reports/v106_podmoe_basepreserve_vs_v104c_delta_mse.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v106_podmoe_basepreserve_kitchen_vs_v104c_delta_mse.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_hardtriad_20260625_reports/v106_podmoe_basepreserve_bonsai_vs_v104c_delta_mse.md
```

| scene | views | MSE-improved | MSE-worse | mean delta-MSE |
|---|---:|---:|---:|---:|
| counter | 30 | 23 | 7 | -0.00000026 |
| kitchen | 35 | 30 | 5 | -0.00000050 |
| bonsai | 37 | 36 | 1 | -0.00000017 |
| total | 102 | 89 | 13 | negative mean |

Interpretation: v106 base-preserve fixes the earlier POD-MoE direction issue on the hard triad. Earlier POD-MoE variants on counter had only `4 / 30` MSE-improved views; base-preserve changes counter to `23 / 30` and hard triad to `89 / 102`.

### 4.3 v109 Parent Gate Safety Result

Evidence:

```text
docs/car_model/6-25-v109-RenderRealizedParentGate-Log.md
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v109_render_realized_parent_gate_ne_flowers/v109_render_realized_parent_gate_report.md
/dev/shm/peilincai_spcarnet_v109_render_realized_parent_gate_20260625_reports/v109_vs_v106_flowers_render_delta_mse.md
```

Flowers final test metrics:

| method | PSNR | SSIM | LPIPS | relation |
|---|---:|---:|---:|---|
| v106 parent | 20.077723 | 0.531240 | 0.374393 | parent baseline |
| v108 normal-equation candidate | 20.076418 | 0.531125 | 0.374427 | worse than v106 |
| v109 parent gate | 20.077723 | 0.531240 | 0.374393 | identical to v106 |

Gate report facts:

- `no_target_gt_used_for_policy=True`;
- `fallback_to_parent=True`;
- `target_views=22`;
- `target_mean_mask=0.00000000`;
- render delta vs v106 records `mse_improved_views=22`, `mse_worse_views=0`, and zero mean absolute delta; this should be read as non-worse parent preservation, because the gate selected exact parent output.

Interpretation: v109 is a safety success. It turns a harmful v108 flowers candidate into a parent-preserving output. It does not improve beyond v106.

### 4.4 v110/v110b Strict-Split Current Status

Evidence:

```text
docs/car_model/6-25-v110-StrictSplitParentGate-Log.md
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/
/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/
```

Implemented protocol:

```text
fit candidate field on train/even
calibrate parent gate on train/odd
evaluate on test
keep v106 as immutable parent
```

Current snapshot:

| scene | strict-split status | evidence path |
|---|---|---|
| flowers | default v110 accepted a harmful nonzero mask; v110b falls back to v106 and exactly preserves parent | `docs/car_model/results/v110_strict_split_20260625/flowers/` |
| garden | v110b accepts a nonzero mask and beats clean, but regresses vs v106 parent on test | `docs/car_model/results/v110_strict_split_20260625/garden/` |
| counter | candidate field build still running in local `/dev/shm` at the latest audit | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/counter/logs/01_build_candidate_field.log` |
| bonsai | candidate field build relaunched after stale preflight issue; no final strict result yet | `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/bonsai/logs/01_build_candidate_field.log` |

Two-scene v110b diagnostic:

| scene | clean | v106 parent | v110b | v110b vs clean | v110b vs v106 |
|---|---:|---:|---:|---:|---:|
| flowers | 19.682257 / 0.511822 / 0.394563 | 20.077723 / 0.531240 / 0.374393 | 20.077723 / 0.531240 / 0.374393 | +0.395466 / +0.019418 / -0.020170 | +0.000000 / +0.000000 / +0.000000 |
| garden | 25.029211 / 0.780035 / 0.201314 | 25.790945 / 0.799382 / 0.174480 | 25.430321 / 0.783703 / 0.186970 | +0.401110 / +0.003668 / -0.014345 | -0.360624 / -0.015679 / +0.012489 |

Durable repo summary:

```text
docs/car_model/results/v110_strict_split_20260625/summary/v110b_manual_flowers_garden_summary.md
```

Train-bank manifests currently show:

| scene | target frames | note |
|---|---:|---|
| flowers | 151 | manifest says generated without held-out target GT |
| garden | 161 | manifest says generated without held-out target GT |
| counter | 210 | manifest says generated without held-out target GT |
| bonsai | 255 | manifest says generated without held-out target GT |

A v110/v110b diagnostic table can now be shown, but only as a safety/fairness diagnostic. The correct slide wording is: "strict-split gates are implemented and have already exposed a gate-generalization weakness; v110/v110b are not promoted over v106."

## 5. 已验证定性证据路径

Existing v106 qualitative contact sheets:

```text
docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png
docs/car_model/assets/v106_qualitative/garden_frame00000_crop_contact_sheet.png
docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png
docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.png
docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.png
```

Their JSON manifests record v104c as `base_method`, v106 POD-MoE base-preserve as `candidate_method`, and the corresponding GT/render paths under:

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/{scene}/detached_model/test/
```

Use these as PPT evidence paths, not as a universal visual claim. The visual slide should say:

- v106 crops are available for flowers, garden, room, and treehill;
- the crop panels compare v104c vs v106 with GT/error context;
- final PPT should include both best-looking examples and at least one hard/worst MSE view.

Recommended extra qualitative targets from the MSE diagnostics:

| scene | view | reason |
|---|---|---|
| counter | `00009.png` | known remaining worst MSE view in hard-triad log |
| kitchen | `00015.png` | known remaining worst MSE view in hard-triad log |
| bonsai | `00028.png` | known remaining worst MSE view in hard-triad log |

## 6. 当前短板

1. v106 gain is real but small: mean improvement over v104c is about `+0.002181 PSNR`, `+0.000103 SSIM`, `-0.000112 LPIPS`.
2. v106 still has a gap to the endpoint/reference ceiling. Using the existing endpoint/reference mean, v106 remains roughly `-0.650030 PSNR`, `-0.022845 SSIM`, `+0.044130 LPIPS` behind the endpoint/reference row.
3. v106 evidence provenance is mixed across full9, counter, and hard-triad roots. This is fine for an internal draft, but the paper/PPT appendix should show exact source roots.
4. v106 is still a residual-field renderer path, not a vanilla MeshSplatting checkpoint and not a topology-compression result.
5. v109 only proves safe rejection on flowers. It selected parent fallback and does not provide positive improvement.
6. v110 has no completed final report in the current result root. It should not be described as outperforming v106.
7. Qualitative evidence is not yet systematic. Existing contact sheets cover selected crops, not full-scene failure/success taxonomy.
8. `/dev/shm` artifacts are ephemeral. Durable copies under `docs/car_model/results/` or a stable `outputs/` package are needed before formal handoff.

## 7. 下一步实验清单

Priority 1: finish v110 strict-split runs.

- Wait for flowers and garden v110 candidate field builds to finish.
- Verify expected reports:
  - `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/reports/flowers_v110_strict_split_parent_gate_report.md`
  - `/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/garden/reports/garden_v110_strict_split_parent_gate_report.md`
- Record whether the gate selected nonzero mask or parent fallback.
- Compare v110 against v106 parent, v108 candidate, v104c, and clean.

Priority 2: expand strict-split to hard scenes.

- Run the same v110 strict protocol on counter and bonsai now that train banks exist.
- Add kitchen train bank and strict run if the same fairness package is needed for the hard triad.
- For every scene, export render-delta-MSE diagnostics vs v106 parent and vs v104c.

Priority 3: decide promotion criteria.

- If v110 falls back to parent: mark it as safety-pass only and keep v106 as the quality line.
- If v110 accepts a nonzero mask and improves test metrics: inspect whether the same policy passes on more scenes before claiming a new method line.
- If v110 accepts a candidate and hurts test: tighten calibration risk constraints and treat it as a reliability failure case.

Priority 4: build durable mentor/PPT assets.

- Copy final metric tables and report JSON/MD out of `/dev/shm`.
- Regenerate v106 qualitative panels with consistent columns: GT, clean, v104c, v106, endpoint/reference, absolute error, delta error.
- Add best/worst view panels for counter/kitchen/bonsai.
- Add a claim-boundary slide that explicitly separates "quality improvement" from "safety/fairness validation".

## 8. Claim Boundary for Mentor

Safe to say now:

- v106 POD-MoE base-preserve is the current verified quality line over v104c.
- The current assembled v106 table has 9/9 scenes and favorable signs vs v104c on PSNR, SSIM, and LPIPS.
- v106 hard-triad MSE diagnostics are healthier than earlier POD-MoE variants.
- v109 proves render-realized parent fallback can protect v106 from a harmful v108 flowers candidate without target/test GT policy selection.
- v110 is the strict-split protocol intended to close the fairness gap.

Do not say yet:

- v110 beats v106.
- v109 improves image quality beyond v106.
- v106 closes the endpoint/reference gap.
- SPCarNet is a vanilla MeshSplatting checkpoint.
- v106/v109/v110 provide topology compression.
- The current `/dev/shm` v110 result is complete.

## 9. Source Paths Used

Primary docs:

```text
docs/car_model/6-25-v106-PODMoE-Technical-Report-Draft.md
docs/car_model/6-25-v106-PODMoE-BasePreserve-HardTriad-Log.md
docs/car_model/6-25-v109-RenderRealizedParentGate-Log.md
docs/car_model/6-25-v110-StrictSplitParentGate-Log.md
```

Primary result summaries:

```text
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md
/dev/shm/peilincai_spcarnet_v106_podmoe_basepreserve_full9_20260625_reports/v106_podmoe_basepreserve_full9_strict_assembled_report.md
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v109_render_realized_parent_gate_ne_flowers/v109_render_realized_parent_gate_report.md
/dev/shm/peilincai_spcarnet_v109_render_realized_parent_gate_20260625_reports/v109_vs_v106_flowers_render_delta_mse.md
```

Current v110 result roots:

```text
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/
/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_train_20260625/
```

## 10. 2026-06-25 20:20 PDT Addendum

This addendum records the post-subagent coordination update.

### 10.1 Correct Clean Baseline Path

For strict clean MeshSplatting comparison, use:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/<scene>/results.json
```

with key `ours_26000`. Do not use v101 detached package results as clean baseline; v101 is an endpoint package method line.

Four representative scenes currently show v106 above the selected local clean MeshSplatting baseline:

| scene | clean PSNR / SSIM / LPIPS | v106 PSNR / SSIM / LPIPS | v106 minus clean |
|---|---:|---:|---:|
| flowers | 19.682257 / 0.511822 / 0.394563 | 20.077723 / 0.531240 / 0.374393 | +0.395466 / +0.019418 / -0.020170 |
| garden | 25.029211 / 0.780035 / 0.201314 | 25.790945 / 0.799382 / 0.174480 | +0.761734 / +0.019347 / -0.026834 |
| counter | 26.751774 / 0.862055 / 0.252003 | 27.499645 / 0.867521 / 0.238847 | +0.747871 / +0.005466 / -0.013156 |
| bonsai | 28.895233 / 0.896400 / 0.259493 | 30.316090 / 0.907520 / 0.230050 | +1.420856 / +0.011120 / -0.029443 |

The live collector is:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/collect_v110_strict_split_report.py \
  --scenes flowers garden counter bonsai
```

It writes:

```text
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/v110_strict_split_parent_gate_summary.md
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/v110_strict_split_parent_gate_summary.json
```

### 10.2 v110 Live/Completed Runs

All four representative v110 strict jobs were launched; two scenes now have useful diagnostics and two remain unfinished in the local workspace:

| scene | current stage | note |
|---|---|---|
| flowers | completed default v110 and v110b manual follow-up | default v110 false-accepts; v110b falls back and preserves v106 |
| garden | completed v110b manual follow-up | v110b accepts a nonzero mask and regresses relative to v106 |
| counter | train/even field build running | very slow, but train bank and parent train renders exist |
| bonsai | train/even field build running | initial preflight failure fixed after generating v106 parent train renders |

The report must still say `v110/v110b not promoted`, not `v110 improves`.

### 10.3 v111 End-to-End Strict Interface

The main fairness objection raised by the method-gap review is that v110 uses a v106 parent whose field was not itself train-only strict. To prepare the actual end-to-end strict closure, a new runner was added:

```text
scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py
scripts/car_model/smoke_test_v111_runner_args.py
```

v111 protocol:

```text
train/all  -> build parent field
train+test -> render parent
train/even -> build candidate field
train+test -> render candidate
train/odd  -> calibrate render-realized parent gate
test       -> final evaluation only
```

Smoke/static verification passed:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m py_compile \
  scripts/car_model/run_v111_end_to_end_strict_parent_gate_scene.py \
  scripts/car_model/smoke_test_v111_runner_args.py

/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_v111_runner_args.py
```

No real v111 GPU run has completed yet. A flowers v111 parent-field build has been launched, but it has not reached candidate/gate/eval completion in the cloneable report package.

### 10.4 Updated Mentor Wording

PPT-safe wording:

> v106 is the current verified quality line against the local clean MeshSplatting baseline. v110/v110b are strict candidate/gate diagnostics: flowers can be made parent-safe, but garden still shows a held-out regression relative to v106. A v111 end-to-end strict runner has been implemented to remove the remaining parent-source fairness objection, but it still needs a completed GPU run before it can be claimed as evidence.

Do not present v111 as a result yet; present it as the next protocol closure.

## 11. Flowers Strict Result: v110 False Accept, v110b Safety Repair

The first completed strict-split evidence is flowers.

### 11.1 Default v110

After fixing a CLI parsing issue for negative thresholds, the already-built flowers candidate was gated and evaluated:

```text
gate report:
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/flowers/detached_model/test/ours_26000_v110_strict_train_even_odd_parent_gate_flowers/v109_render_realized_parent_gate_report.json

test eval:
/dev/shm/peilincai_spcarnet_v110_strict_split_parent_gate_20260625/flowers/reports/flowers_ours_26000_v110_strict_train_even_odd_parent_gate_flowers_test_results.json
```

The train-odd gate accepted a nonzero policy:

| signal | value |
|---|---:|
| calibration dPSNR | +0.382280 |
| calibration dSSIM | +0.025216 |
| target mean mask | 0.493082 |
| no target GT used for policy | true |

Held-out test result:

| method | PSNR | SSIM | LPIPS | interpretation |
|---|---:|---:|---:|---|
| clean MeshSplatting | 19.682257 | 0.511822 | 0.394563 | local clean baseline |
| v106 parent | 20.077723 | 0.531240 | 0.374393 | current parent |
| default v110 strict gate | 19.966076 | 0.522843 | 0.380387 | better than clean, worse than v106 |

Default v110 delta:

| reference | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| vs clean | +0.283819 | +0.011021 | -0.014176 |
| vs v106 | -0.111647 | -0.008397 | +0.005994 |

Conclusion: default v110 is a strict-split false accept on flowers. It should be treated as a negative diagnostic, not a result improvement.

### 11.2 v110b Gain-Margin Safety Repair

A fixed, train-calibration-only safety margin was tested:

```text
method: ours_26000_v110b_strict_gainmargin_parent_gate_flowers
change: --min_mean_psnr_gain 0.5
```

The candidate's train-odd dPSNR `+0.382280` is below the margin, so the gate falls back to the parent:

| signal | value |
|---|---:|
| fallback to parent | 1 |
| target mean mask | 0.000000 |
| no target GT used for policy | true |

v110b test metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| v106 parent | 20.077723 | 0.531240 | 0.374393 |
| v110b gain-margin gate | 20.077723 | 0.531240 | 0.374393 |

The v110/v111 runner defaults now use `--min_mean_psnr_gain 0.5`.

Interpretation for PPT:

- v110 showed why strict validation is necessary: mean train-odd improvement can still false-accept a harmful candidate.
- v110b restores safety by requiring a stronger calibration margin.
- This improves reliability, but it is not a quality breakthrough over v106.

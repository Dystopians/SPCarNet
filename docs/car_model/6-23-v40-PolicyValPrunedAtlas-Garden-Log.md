# v40 Policy-Val Pruned Atlas Log

Date: 2026-06-23

Status: `NOT PROMOTED`. This is a real representation-level method change that fixes the v39 garden robust-gate failure, but the held-out effect size is still far too small to replace Phase-J.

## Motivation

The v39 SSIM-aware atlas was the first representation-level atlas pilot that could weakly beat the compact parent on Bonsai, but the same policy failed on garden:

- raw garden v39 atlas had positive mean policy-val MSE gain;
- however, it failed robust train-only gates because two held-out train views regressed;
- lower alpha and stricter bin certification reduced the negative tail but did not remove it.

The failure mode is important: this is not a simple alpha problem. Some residual-region faces have the wrong transfer direction for certain policy-val views. v40 therefore adds a train-only face/carrier certification step before applying the atlas.

## Method Change

New tools:

```text
scripts/car_model/ecsr_prune_region_carriers_by_policy_val.py
scripts/car_model/ecsr_export_evidence_rgb_baseline.py
```

`ecsr_prune_region_carriers_by_policy_val.py` performs train-only pruning:

1. fit the same surface residual atlas on train evidence;
2. use the policy-val split induced by `--policy_val_stride`;
3. compute each face's MSE contribution on policy-val residuals;
4. retain only faces with positive aggregate contribution and no robust held-out tail failure after pruning;
5. write a pruned carrier JSON consumed by the normal atlas apply script.

This is a method-level guard, not a test-set adjustment. Held-out test GT is not used for pruning.

## Evidence Inputs

Garden target evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/garden
```

Garden train + teacher evidence:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1
```

Original carriers:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1.json
```

Original carrier build summary:

```text
raw regions: 552
merged carriers: 64
evidence faces: 838
```

## Failed v39 Attempts

Raw v39 lowpass/bin-count run:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_region_texture_adapter_v39_lowpass1_bin1_face32_a0015625_phasej_resize
```

Audit:

```text
accepted: false
selected alpha: 0.015625
policy-val relative gain: 0.003431
positive-view fraction: 0.833333
CVaR20 view relative gain: -0.001646
min-view relative gain: -0.005322
reject reason: positive_view_fraction / CVaR20 / min-view robust gate failure
```

Lower alpha did not fix direction:

| alpha | rel gain | positive-view fraction | CVaR20 | min view |
|---:|---:|---:|---:|---:|
| 0.001953 | 0.000436 | 0.833333 | -0.000196 | -0.000660 |
| 0.003906 | 0.000871 | 0.833333 | -0.000394 | -0.001321 |
| 0.007812 | 0.001733 | 0.833333 | -0.000800 | -0.002649 |
| 0.015625 | 0.003431 | 0.833333 | -0.001646 | -0.005322 |

Certification variants with count/sign/variance gates still failed with `0.75-0.833333` positive-view fraction.

## v40 Pruning Command

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_prune_region_carriers_by_policy_val.py \
  --input_carrier_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1.json \
  --fit_evidence_dir outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/garden_teacher_surface_evidence_phasej_trainval_resize_alpha1 \
  --out_json outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.json \
  --out_md outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_teacher_render_visible_region_carriers_phasej_trainval_resize_alpha1_policyval_pruned.md \
  --texture_size 16 \
  --max_carriers 64 \
  --max_faces_per_carrier 128 \
  --max_faces 4096 \
  --policy_val_stride 4 \
  --alpha 0.015625 \
  --min_l1 0.001 \
  --min_alpha 0.03 \
  --min_atlas_bin_count 2 \
  --min_atlas_face_samples 32 \
  --max_atlas_bin_rgb_variance 0.004 \
  --min_atlas_bin_sign_consistency 0.5 \
  --atlas_lowpass_passes 1 \
  --atlas_lowpass_neighbor_min_count 1 \
  --max_samples_per_view 240000 \
  --min_face_total_gain 0.0 \
  --min_view_relative_gain 0.0 \
  --greedy_repair \
  --force
```

Pruning output:

```text
input carriers: 64
output carriers: 50
candidate faces: 833
atlas faces: 817
retained faces: 319
removed faces: 498
greedy removals: 0
```

Policy-val retained relative gains:

| view | rel gain |
|---|---:|
| 00080 | 0.00069243 |
| 00072 | 0.00074907 |
| 00040 | 0.00212503 |
| 00032 | 0.00291120 |
| 00008 | 0.00309000 |
| 00088 | 0.00363763 |
| 00048 | 0.00449203 |
| 00000 | 0.00498659 |
| 00024 | 0.00547419 |
| 00016 | 0.00771393 |
| 00064 | 0.00775561 |
| 00056 | 0.00780350 |

## v40 Apply Result

Output model:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v40_policyval_pruned_region_texture_adapter
```

Method:

```text
ours_26000_garden_v40_policyval_pruned_region_texture_adapter
```

Audit:

```text
accepted: true
selected alpha: 0.015625
atlas faces: 319
fit samples: 95747
policy-val samples: 32994
policy-val relative gain: 0.005177
positive-view fraction: 1.000000
CVaR20 view relative gain: 0.002537
min-view relative gain: 0.001035
target written views: 24
target changed fraction: 0.0013748
```

The target output renders are saved here:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_v40_policyval_pruned_region_texture_adapter/test/ours_26000_garden_v40_policyval_pruned_region_texture_adapter/renders
```

## Same-Evidence Metrics

Because these atlas experiments use the `images_2` evidence cache and write images from cached `rgb_render/rgb_gt`, they must first be compared against a no-op baseline exported from the same target evidence. The older compact-model `results.json` is not the same file naming/resolution path.

No-op export:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/garden_evidence_noop_compact_baseline
```

Metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| same-evidence no-op compact baseline | 24.741003 | 0.75404900 | 0.24802321 |
| v40 policy-val pruned atlas | 24.741049 | 0.75404984 | 0.24802260 |
| delta | +0.0000458 | +0.00000083 | -0.00000061 |

Strict same-evidence win: `yes`, but only by a tiny margin.

## Carrier-Unit Pruning Ablation

After the face-unit run, the pruning script was extended with:

```text
--prune_unit carrier
```

This keeps coherent carrier units rather than independent faces, then greedily removes carrier units only when needed to recover the policy-val robust gate.

Carrier-unit pruning output:

```text
output carriers: 45
input units: 64
retained units: 44
retained faces: 468
removed faces: 349
greedy removals: 1
```

Carrier-unit apply audit:

```text
accepted: true
atlas faces: 468
policy-val relative gain: 0.005005
positive-view fraction: 1.000000
CVaR20 view relative gain: 0.002720
min-view relative gain: 0.000931
target changed fraction: 0.001214
```

Same-evidence metrics:

| method | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| face-unit v40 | 24.741049 | 0.75404984 | 0.24802260 | +0.0000458 | +0.00000083 | -0.00000061 |
| carrier-unit v40 | 24.741037 | 0.75404960 | 0.24802257 | +0.0000343 | +0.00000060 | -0.00000064 |

Carrier-unit pruning retained more faces but did not increase target coverage or RGB effect size on garden. The current best garden v40 row remains the face-unit pruned atlas.

For reference only, the older full model result for garden compact base is:

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/garden/ratio_0200/compact_model/results.json
ours_26000_phasef_extra_compact_base: 25.027536 / 0.7800307 / 0.2013215
Phase-J: 26.311111 / 0.8278434 / 0.1358426
```

That row is still the presentation-safe result. v40 does not replace Phase-J.

## Interpretation

v40 proves a useful mechanism:

- the v39 garden robust-gate failure was caused by face-level residual directions that do not transfer across held-out train views;
- train-only policy-val face pruning can remove those faces;
- the pruned atlas passes strict policy-val tail gates and gives a held-out same-evidence strict win.

However, the visual/metric gain is almost invisible because target coverage is only `0.137%` of pixels and the retained atlas is very conservative. This is not yet a paper-level representation result.

## Room Cross-Scene Validation

After garden, the same face-unit v40 pipeline was run on `room`.

Evidence:

```text
target evidence: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/room
train evidence: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/room
teacher evidence: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/room_teacher_surface_evidence_phasej_trainval_alpha1
```

Coverage summaries:

```text
room test views: 39
room test barycentric valid pixel fraction: 0.92671694
room train views: 46
room train barycentric valid pixel fraction: 0.92023991
teacher mean active fraction: 0.189005
```

Carrier and pruning summary:

```text
raw regions: 552
merged carriers: 64
candidate faces: 2191
atlas faces: 2189
retained faces: 1160
removed faces: 1029
prune unit: face
greedy removals: 0
```

Policy-val retained relative gains were all positive:

```text
min policy-val relative gain: 0.001002
mean policy-val relative gain: 0.002913
```

Room v40 apply audit:

```text
accepted: true
atlas faces: 1160
fit samples: 462164
policy-val samples: 161316
selected alpha: 0.015625
policy-val relative gain: 0.002807
positive-view fraction: 1.000000
CVaR20 view relative gain: 0.001389
min-view relative gain: 0.001260
target written views: 39
target changed fraction: 0.0036668
```

Same-evidence room metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| same-evidence no-op compact baseline | 28.739004 | 0.88479000 | 0.24991596 |
| room v40 policy-val pruned atlas | 28.739227 | 0.88479781 | 0.24991086 |
| delta | +0.0002232 | +0.00000781 | -0.00000510 |

Strict same-evidence win: `yes`.

Reference full-protocol rows:

```text
compact base: 28.739101 / 0.8847932 / 0.2499234
Phase-J: 30.305639 / 0.9057302 / 0.1959894
```

Interpretation: v40 now has two same-evidence strict wins (`garden`, `room`) and a real train-only robustness mechanism, but the effect size is still orders of magnitude below Phase-J.

## Counter Cross-Scene Validation

The same face-unit v40 pipeline was then run on `counter`.

Evidence:

```text
target evidence: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/target_visible_bary_images2/counter
train evidence: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/counter
teacher evidence: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/train_visible_bary_images2/counter_teacher_surface_evidence_phasej_trainval_alpha1
```

Coverage summaries:

```text
counter test views: 30
counter test barycentric valid pixel fraction: 0.93400406
counter train views: 46
counter train barycentric valid pixel fraction: 0.93732042
teacher mean active fraction: 0.201384
```

Carrier and pruning summary:

```text
raw regions: 552
merged carriers: 64
candidate faces: 2817
atlas faces: 2811
retained faces: 1574
removed faces: 1237
prune unit: face
greedy removals: 0
```

Policy-val retained relative gains were all positive:

```text
min policy-val relative gain: 0.000816
mean policy-val relative gain: 0.002097
```

Counter v40 apply audit:

```text
accepted: true
atlas faces: 1574
fit samples: 679257
policy-val samples: 255007
selected alpha: 0.015625
policy-val relative gain: 0.002829
positive-view fraction: 1.000000
CVaR20 view relative gain: 0.001046
min-view relative gain: 0.000930
target written views: 30
target changed fraction: 0.006411
```

Same-evidence counter metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| same-evidence no-op compact baseline | 26.749836 | 0.86204934 | 0.25199798 |
| counter v40 policy-val pruned atlas | 26.750059 | 0.86205125 | 0.25199622 |
| delta | +0.0002232 | +0.00000191 | -0.00000176 |

Strict same-evidence win: `yes`.

Reference full-protocol rows:

```text
compact base: 26.749872 / 0.8620513 / 0.2519977
Phase-J: 28.449171 / 0.8937307 / 0.1864724
```

## Three-Scene v40 Summary

| scene | target changed fraction | dPSNR | dSSIM | dLPIPS | strict same-evidence win |
|---|---:|---:|---:|---:|---|
| garden | 0.001375 | +0.0000458 | +0.00000083 | -0.00000061 | yes |
| room | 0.003667 | +0.0002232 | +0.00000781 | -0.00000510 | yes |
| counter | 0.006411 | +0.0002232 | +0.00000191 | -0.00000176 | yes |

Conclusion: the v40 policy-val pruned atlas is now a repeatable train-only representation-level mechanism across three scenes. The weakness is not stability; it is effect size and visual salience. The method is still not strong enough to replace Phase-J or claim a paper-level representation breakthrough.

## v41 Face-Mean Coverage Expansion

v41 keeps the same v40 train-only policy-val pruned carrier sets, but changes target support from certified UV bins only to retained-face mean residual fallback:

```text
--min_atlas_bin_count 0
--min_atlas_face_samples 32
```

Because the atlas fitting already fills empty bins with the face mean by default, this allows unseen target bins on retained policy-val-safe faces to receive a small residual. The same robust policy-val gates remain active:

```text
min_policy_val_relative_gain = 0.0002
min_policy_val_positive_view_fraction = 1.0
min_policy_val_cvar20_relative_gain = 0.0
min_policy_val_min_view_relative_gain = 0.0
```

This is a coverage-aware expansion, not a held-out test selection. Held-out test GT is still only used for final metrics.

### v41 Audit Summary

| scene | v40 changed frac | v41 changed frac | v41 policy-val gain | v41 min-view gain | accepted |
|---|---:|---:|---:|---:|---|
| garden | 0.001375 | 0.003852 | 0.010628 | 0.000269 | yes |
| room | 0.003667 | 0.010996 | 0.006623 | 0.001024 | yes |
| counter | 0.006411 | 0.019906 | 0.006271 | 0.000154 | yes |
| bonsai | n/a | 0.007670 | 0.014207 | 0.004015 | yes |

### v41 Same-Evidence Metrics

| scene | dPSNR vs no-op | dSSIM vs no-op | dLPIPS vs no-op | strict win | dPSNR vs v40 | dSSIM vs v40 | dLPIPS vs v40 |
|---|---:|---:|---:|---|---:|---:|---:|
| garden | +0.0000858 | +0.00000107 | -0.00000133 | yes | +0.0000401 | +0.00000024 | -0.00000072 |
| room | +0.0005856 | +0.00001425 | -0.00000687 | yes | +0.0003624 | +0.00000644 | -0.00000177 |
| counter | +0.0005417 | +0.00000280 | -0.00000313 | yes | +0.0003185 | +0.00000089 | -0.00000137 |
| bonsai | +0.0009670 | +0.00000274 | -0.00000277 | yes | n/a | n/a | n/a |

v41 is a strictly better representation-level row than v40 on the three direct v40/v41 same-evidence evaluations. It increases target coverage by roughly `2.8x-3.1x` and improves all three metrics versus both no-op and v40. On `bonsai`, v41 was run with the already available v37 visible-barycentric evidence and is therefore logged as a direct no-op comparison rather than a v40-vs-v41 delta.

However, the gains are still tiny compared with Phase-J render-time ELA. v41 should be treated as a stronger representation-level diagnostic and method step, not as the final paper headline.

## Bonsai v41 Cross-Scene Validation

The Bonsai v41 run uses the existing v37 visible-barycentric evidence cache:

```text
target evidence: outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_target_images2/bonsai
fit evidence: outputs/carnet/meshsplatopt/ecsr_phase_v37_visible_bary_train_images2/bonsai_teacher_surface_evidence_visible_alpha1
pruned carriers: outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/bonsai_teacher_render_visible_region_carriers_v37_visible_alpha1_policyval_pruned.json
```

This is a valid train-only pruning and same-evidence held-out comparison, but it is not the exact same evidence-generation path as the garden/room/counter v39 multiscene cache. It should therefore be reported with this caveat.

Carrier and pruning summary:

```text
input carriers: 64
output carriers: 58
candidate faces: 2247
atlas faces: 2208
retained faces: 1110
removed faces: 1098
prune unit: face
greedy removals: 0
```

Bonsai v41 apply audit:

```text
accepted: true
atlas faces: 1110
fit samples: 159447
policy-val samples: 61159
selected alpha: 0.015625
policy-val relative gain: 0.014207
positive-view fraction: 1.000000
CVaR20 view relative gain: 0.005240
min-view relative gain: 0.004015
target written views: 37
target changed fraction: 0.0076696
```

Same-evidence Bonsai metrics:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| same-evidence no-op compact baseline | 28.864380 | 0.89601004 | 0.25933361 |
| bonsai v41 face-mean expanded atlas | 28.865347 | 0.89601278 | 0.25933084 |
| delta | +0.0009670 | +0.00000274 | -0.00000277 |

Strict same-evidence win: `yes`.

Reference full-protocol rows:

```text
compact base: 28.864340 / 0.8960123 / 0.2593397
Phase-J: 31.862005 / 0.9302796 / 0.1725553
```

## Next Step

The next method step should not be more manual alpha scanning. It should improve effect size while preserving the v40 safety mechanism:

1. make policy-val pruning carrier-aware rather than only face-aware, so coherent regions survive together;
2. add a coverage-aware objective that rewards safe target coverage, not just no-regression;
3. increase support using multi-resolution / neighborhood expansion after face pruning;
4. rerun the same pipeline on `room`, `counter`, and `bonsai`;
5. promote only if same-evidence gains become nontrivial and scene-matched full-protocol metrics are also positive.

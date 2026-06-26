# vNext Artifacts Index

日期：2026-06-26
用途：索引当前 repo 内保存的 `vNext_certified_residual_surface_texture` artifacts，并说明哪些可以作为 PPT 证据。

---

## Summary

当前 artifacts 有三个真实 `garden` 单场景 run、三个 strict frozen-policy 场景 run、三场景 structure-aware shrink 表、四场景 ready4 structure-aware shrink 表、`stump/treehill/flowers` 输入链重建/拒绝结果，以及 full9 preflight 缺口记录：

```text
garden_20260626_004134
garden_hardbin_softshrink_20260626_035631
garden_face_softshrink_20260626_040558
counter_strict_face_softshrink_20260626_045300
bonsai_strict_face_softshrink_20260626_052500
room_strict_face_softshrink_20260626_052500
strict_frozen_policy_multiscene_20260626_052500
counter_structure_shrink_tau002_20260626_0558
bonsai_structure_shrink_tau002_20260626_0718
room_structure_shrink_tau002_20260626_0718
strict_structure_aware_shrink_multiscene_20260626_0718
garden_structure_shrink_tau002_20260626_071413
strict_structure_aware_shrink_ready4_20260626_071413
stump_structure_shrink_rebuild_tau002_20260626_080257
full9_gap_after_stump_preflight_20260626
treehill_structure_shrink_rebuild_tau002_20260626_0832
full9_gap_after_treehill_preflight_20260626
flowers_structure_shrink_rebuild_tau002_20260626_0935
vnext_structure_shrink_ready4_scene_config_20260626.json
vnext_structure_shrink_full9_gap_scene_config_20260626.json
vnext_structure_shrink_ready4_preflight_20260626.md
vnext_structure_shrink_ready4_preflight_20260626.json
vnext_structure_shrink_full9_gap_preflight_20260626.md
vnext_structure_shrink_full9_gap_preflight_20260626.json
```

结论必须诚实表述：

```text
initial full candidate: protocol passed, accepted=false, fallback_noop
hard-bin soft-shrink: protocol passed, accepted=false, fallback_noop
face-softshrink: protocol passed, accepted=true, selected_alpha=0.0625, changed_fraction=0.002080
counter strict face-softshrink: protocol passed, target_gt_visible_to_apply=false, accepted=true, selected_alpha=0.25, changed_fraction=0.01177355
bonsai strict face-softshrink: protocol passed, target_gt_visible_to_apply=false, accepted=true, selected_alpha=0.25, changed_fraction=0.00151333
room strict face-softshrink: protocol passed, target_gt_visible_to_apply=false, accepted=false, fallback_noop, changed_fraction=0.000000
```

因此，这个目录证明的是两件事：

1. **vNext 安全证书和 no-op fallback 机制有效，并且同一套 frozen face-softshrink policy 已经在 counter/bonsai 上产生真实非零 accepted residual surface texture，在 room 上安全回退且 `changed_fraction=0`**。
2. **新一轮 structure-aware shrink policy 在严格 no-target-GT apply 下让 counter/bonsai/room/garden 四场景全部 accepted，其中 room 从旧策略 fallback 变为真实非零 residual output，garden 则相对旧 face-softshrink 也三指标小幅提升**。

四场景 ready4 平均仍只有 `+0.00076151 PSNR / -0.00000302 SSIM / -0.00002038 LPIPS`，后续 `stump/treehill/flowers` 是安全拒绝的 fallback/no-op 负结果，因此还不是 full9、v106、clean MeshSplatting 或 Phase-J teacher 超越证据。完整解释见 `../6-26-vNext-StructureAwareShrink-Strict-Multiscene-Log.md`、`../6-26-vNext-ManifestRunner-and-Full9Gap-Log.md`、`../6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md`、`../6-26-vNext-TreehillInputRebuild-Ready6-and-Rejection-Log.md`、`../6-26-vNext-FlowersInputRebuild-Ready7-and-SameEvidenceFallback-Log.md`。

---

## Directory Index

| path | 内容 | PPT 用法 |
|---|---|---|
| `garden_20260626_004134/garden_vnext_certified_residual_texture_report.md` | run 顶层报告：method、scene、protocol audit、settings、commands、errors | 用于展示 run 是真实完成而非 dry-run |
| `garden_20260626_004134/garden_vnext_certified_residual_texture_manifest.json` | 机器可读 provenance：source paths、commands、settings、protocol audit | 用于证明 `selection_uses_test_gt=false`、阈值和 capacity 来自 train-policy-val |
| `garden_20260626_004134/surface_residual_region_texture_adapter_audit.md` | 人类可读 certificate/audit 摘要 | 最适合 PPT：直接显示 `accepted=false`、`fallback_noop`、拒绝原因 |
| `garden_20260626_004134/surface_residual_region_texture_adapter_audit.json` | 完整机器可读 audit，包含 candidate profiles、policy gates、target apply | 用于深挖失败原因，不建议直接贴 PPT |
| `garden_20260626_004134/garden_ours_26000_vnext_certified_residual_surface_texture_test_results.json` | final target split aggregate metrics | 只能作为 fallback 输出指标，不可当作 vNext improvement |
| `garden_20260626_004134/garden_ours_26000_vnext_certified_residual_surface_texture_test_per_view.json` | per-view PSNR/SSIM/LPIPS | 用于检查 fallback 输出 per-view 分布 |
| `garden_hardbin_softshrink_20260626_035631/` | hard-bin soft-shrink 诊断：soft shrink 激活但 hard bin guard 仍拒绝 | 用于解释为什么最终策略改为 soft shrink 替代 hard bin allowlist |
| `garden_face_softshrink_20260626_040558/` | face-softshrink accepted run：manifest、report、audit、metrics、per-view、logs、summary JSON、qualitative panel | 第一个非零 vNext 里程碑证据 |
| `garden_face_softshrink_20260626_040558/garden_face_softshrink_summary.json` | 三轮结果、delta、per-view win counts、guard 诊断 | PPT/脚本可读摘要 |
| `garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png` | GT / parent / vNext / error maps / vNext-parent amplified diff | 展示非零改动存在，但视觉收益非常弱 |
| `counter_strict_face_softshrink_20260626_045300/` | counter strict face-softshrink accepted run：manifest、report、audit、metrics、per-view、summary JSON、logs | strict no-target-GT apply 的非零证据 |
| `counter_strict_face_softshrink_20260626_045300/counter_strict_face_softshrink_summary.json` | parent/vNext metrics、delta、protocol audit、target apply 摘要 | counter 单场景行数据源；face-softshrink 三场景汇总见 `strict_frozen_policy_multiscene_summary.md` |
| `counter_strict_face_softshrink_20260626_045300/target_evidence_no_gt_audit.json` | target evidence stripped-key 审计 | 证明 adapter apply 阶段没有 target GT/residual keys |
| `bonsai_strict_face_softshrink_20260626_052500/` | bonsai strict face-softshrink accepted run：manifest、report、audit、metrics、per-view、summary JSON、logs | frozen policy 第二个非零 accepted 场景 |
| `room_strict_face_softshrink_20260626_052500/` | room strict face-softshrink fallback run：manifest、report、audit、metrics、per-view、summary JSON、logs | frozen policy 安全拒绝/回退场景 |
| `strict_frozen_policy_multiscene_20260626_052500/strict_frozen_policy_multiscene_summary.md` | counter/bonsai/room 三场景聚合表 | ready4 之前的 face-softshrink predecessor diagnostic |
| `counter_structure_shrink_tau002_20260626_0558/` | counter structure-aware shrink strict run：manifest、audit、metrics、per-view | 新结构风险 shrink 的 counter 证据；SSIM 回退显著小于旧 face-softshrink，但 PSNR/LPIPS 收益也更小 |
| `bonsai_structure_shrink_tau002_20260626_0718/` | bonsai structure-aware shrink strict run：manifest、audit、metrics、per-view、logs | 新结构风险 shrink 的 bonsai 证据；结果基本接近旧 face-softshrink |
| `room_structure_shrink_tau002_20260626_0718/` | room structure-aware shrink strict run：manifest、audit、metrics、per-view、logs | 新结构风险 shrink 的最重要新证据：旧策略 fallback 的 room 变为 accepted nonzero，并相对 Phase-F parent 三指标全正向 |
| `strict_structure_aware_shrink_multiscene_20260626_0718/strict_structure_aware_shrink_multiscene_summary.md` | counter/bonsai/room 新结构风险 shrink 聚合表 | ready4 之前的三场景前序证据 |
| `garden_structure_shrink_tau002_20260626_071413/` | garden structure-aware shrink strict run：manifest、audit、metrics、per-view、logs | 第四个 ready scene；相对 Phase-F parent 和旧 garden face-softshrink 都三指标正向 |
| `strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md` | counter/bonsai/room/garden ready4 聚合表 | 当前 vNext structure-aware shrink 首选聚合表；仍非 full9 |
| `strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.json` | ready4 聚合表机器可读版本 | 脚本/表格复用入口 |
| `vnext_structure_shrink_ready4_scene_config_20260626.json` | manifest runner 的 ready4 场景配置 | `bonsai/counter/garden/room` 四个输入齐全场景的可执行配置 |
| `vnext_structure_shrink_full9_gap_scene_config_20260626.json` | manifest runner 的 full9 gap 场景配置 | 记录 full9 目标配置；5 个缺失场景指向待重建 normalized input tree |
| `vnext_structure_shrink_ready4_preflight_20260626.md` | ready4 preflight summary | 证明当前 4/4 ready |
| `vnext_structure_shrink_full9_gap_preflight_20260626.md` | full9 gap preflight summary | 证明当前 full9 是 4/9 ready、5/9 missing input |
| `stump_structure_shrink_rebuild_tau002_20260626_080257/` | rebuilt stump input-chain strict scene run：manifest、audit、metrics、per-view、policy-val carrier summary | 第五个 input-ready scene；strict run 完成但证书拒绝为 fallback/no-op |
| `full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md` | stump rebuild 后的 full9 preflight summary | 证明 local snapshot 从 4/9 ready 推进到 5/9 ready；剩余缺 `bicycle/flowers/kitchen/treehill` |
| `treehill_structure_shrink_rebuild_tau002_20260626_0832/` | rebuilt treehill input-chain strict scene run：manifest、audit、metrics、per-view、policy-val carrier summary、teacher evidence summary | 第六个 input-ready scene；strict run 完成但证书因 lower-tail/SSIM/L1 风险拒绝为 fallback/no-op |
| `full9_gap_after_treehill_preflight_20260626/vnext_manifest_runner_summary.md` | treehill rebuild 后的 full9 preflight summary | 证明 local snapshot 从 5/9 ready 推进到 6/9 ready；剩余缺 `bicycle/flowers/kitchen` |
| `flowers_structure_shrink_rebuild_tau002_20260626_0935/` | rebuilt flowers input-chain strict scene run：manifest、audit、metrics、per-view、same-evidence parent comparison、policy-val carrier summary、teacher evidence summary | 第七个 input-ready scene；strict run 完成但证书因 lower-tail/SSIM/L1 风险拒绝为 fallback/no-op；same-evidence parent 与 fallback 指标完全一致 |
| `flowers_structure_shrink_rebuild_tau002_20260626_0935/preflight/vnext_manifest_runner_summary.md` | flowers rebuild 后的 full9 preflight summary | 证明 local snapshot 从 6/9 ready 推进到 7/9 ready；剩余缺 `bicycle/kitchen` |

---

## Pre-Ready4 Structure-Aware Three-Scene Key Facts

Artifact roots:

```text
counter_structure_shrink_tau002_20260626_0558
bonsai_structure_shrink_tau002_20260626_0718
room_structure_shrink_tau002_20260626_0718
```

Fixed policy:

```text
enable_policy_val_structure_aware_shrink=true
structure_shrink_l1_weight=1.0
structure_shrink_gradient_weight=1.0
structure_shrink_edge_weight=0.0
structure_shrink_risk_tau=0.002
strict_no_target_gt_apply=true
```

Versus Phase-F compact parent:

| scene | protocol pass | target GT visible to apply | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `True` | `False` | `True` | `0.125` | `0.01234357` | `+0.00129890` | `-0.00000906` | `-0.00004268` |
| bonsai | `True` | `False` | `True` | `0.25` | `0.00148974` | `+0.00113869` | `-0.00000954` | `-0.00001693` |
| room | `True` | `False` | `True` | `0.0625` | `0.00519912` | `+0.00046921` | `+0.00000334` | `-0.00001399` |
| mean | 3/3 | 0/3 visible | 3/3 nonzero | - | - | `+0.00096893` | `-0.00000509` | `-0.00002453` |

Interpretation:

- This is a real strict method milestone because room changes from old fallback/no-op to accepted nonzero residual output.
- It reduces structure risk and SSIM regression but does not solve the small effect-size bottleneck.
- Do not use `counter_structure_edge_confidence_20260626_0623` as parent-edge positive evidence; it was produced before a later interface fix that made final target apply receive the same `parent_edge_apply_profile` as policy-val.
- The current preferred aggregate is the later ready4 table, which adds `garden` and is indexed in the next section.

## Structure-Aware Shrink Ready4 Key Facts

Artifact root:

```text
strict_structure_aware_shrink_ready4_20260626_071413
```

Versus Phase-F compact parent:

| scene | protocol pass | target GT visible to apply | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `True` | `False` | `True` | `0.125` | `0.01234357` | `+0.00129890` | `-0.00000906` | `-0.00004268` |
| bonsai | `True` | `False` | `True` | `0.25` | `0.00148974` | `+0.00113869` | `-0.00000954` | `-0.00001693` |
| room | `True` | `False` | `True` | `0.0625` | `0.00519912` | `+0.00046921` | `+0.00000334` | `-0.00001399` |
| garden | `True` | `False` | `True` | `0.125` | `0.00205038` | `+0.00013924` | `+0.00000316` | `-0.00000791` |
| mean | 4/4 | 0/4 visible | 4/4 nonzero | - | - | `+0.00076151` | `-0.00000302` | `-0.00002038` |

Garden also improves over the previous garden face-softshrink pilot by `+0.00006294 PSNR / +0.00000119 SSIM / -0.00000468 LPIPS`.

Ready4 full9 preflight was `4 / 9` ready. Missing input scenes were `bicycle,flowers,kitchen,stump,treehill`.

After the local stump, treehill, and flowers input rebuilds, the current local preflight is `7 / 9` ready. None of these outdoor/tail-risk rebuilds is an accepted quality row: strict no-target-GT vNext completed for all three, but stump fell back/no-op because the policy-val tail-risk certificate rejected it, treehill fell back/no-op because lower-tail, SSIM, and L1 gates rejected it, and flowers fell back/no-op for the same lower-tail/SSIM/L1 reason. Flowers additionally records a same-evidence parent export proving that fallback and parent are identical under the rebuilt `images_2` target evidence resolution. The remaining missing-input scenes are `bicycle,kitchen`. See `../6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md`, `../6-26-vNext-TreehillInputRebuild-Ready6-and-Rejection-Log.md`, and `../6-26-vNext-FlowersInputRebuild-Ready7-and-SameEvidenceFallback-Log.md`.

---

## Initial Garden Run Key Facts

| 字段 | 值 |
|---|---:|
| scene | `garden` |
| status | `COMPLETE` |
| protocol audit passed | `True` |
| selection uses test GT | `False` |
| capacity selected on | `train_policy_val_and_gt_free_target_footprint` |
| thresholds selected on | `train_policy_val` |
| source parent | `ours_26000_phasef_extra_compact_base` |
| policy candidates executed | `48` |
| atlas base faces | `319` |
| fit samples | `201676` |
| policy-val samples | `71583` |
| accepted | `False` |
| selected alpha | `0.0` |
| target written views | `24` |
| target changed fraction | `0.000000` |

Final aggregate metrics in `test_results.json`:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| `ours_26000_vnext_certified_residual_surface_texture` | `24.741003` | `0.754049` | `0.248023` |

Interpretation:

- These metrics are from fallback/no-op output.
- `target_changed_fraction=0.0`, so this is not evidence of nonzero residual texture benefit.
- The fallback parent is Phase-F compact parent, so do not compare it casually against selected clean MeshSplatting or v106 without labeling the parent.

---

## Face-SoftShrink Key Facts

Artifact root:

```text
garden_face_softshrink_20260626_040558
```

| 字段 | 值 |
|---|---:|
| scene | `garden` |
| status | `COMPLETE` |
| protocol audit passed | `True` |
| selection uses test GT | `False` |
| hard bin uncertainty guard | `disabled` |
| soft bin uncertainty shrink | `keep_with_downweight` |
| accepted | `True` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.0625` |
| target changed pixels | `82767` |
| target changed fraction | `0.002080` |
| policy-val relative gain | `0.006009` |
| policy-val CVaR20 gain | `0.002396` |
| policy-val min-view gain | `0.000258` |
| policy-val SSIM gain | `0.000001659` |

Held-out garden test:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| no-op/fallback parent | `24.741003` | `0.754049` | `0.248023` |
| vNext face-softshrink | `24.741079` | `0.754051` | `0.248020` |
| delta, better direction | `+0.000076` | `+0.00000197` | `-0.00000323` |

Per-view better/tie/worse versus no-op/fallback:

| metric | better | tie | worse |
|---|---:|---:|---:|
| PSNR | `22` | `0` | `2` |
| SSIM | `24` | `0` | `0` |
| LPIPS | `22` | `0` | `2` |

Interpretation:

- This is the first real nonzero vNext residual surface texture milestone.
- The improvement is positive but extremely small.
- It does not establish superiority over clean MeshSplatting, v106, or Phase-J.
- The qualitative panel should be used to show traceability and changed regions, not to claim obvious visual superiority.

---

## Counter Strict Face-SoftShrink Key Facts

Artifact root:

```text
counter_strict_face_softshrink_20260626_045300
```

| 字段 | 值 |
|---|---:|
| scene | `counter` |
| status | `COMPLETE` |
| protocol audit passed | `True` |
| selection uses test GT | `False` |
| target GT visible to apply | `False` |
| target GT visible to eval | `True` |
| target forbidden keys stripped | `True` |
| target apply leak | `False` |
| accepted | `True` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.25` |
| target changed pixels | `571207` |
| target changed fraction | `0.01177355` |
| policy-val relative gain | `0.04431575` |
| policy-val SSIM gain | `0.00010365` |

Held-out counter test:

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Phase-F compact parent | `26.749872` | `0.862051` | `0.251998` |
| vNext strict face-softshrink | `26.752003` | `0.862004` | `0.251912` |
| delta, better direction | `+0.002131` | `-0.000047` | `-0.000085` |

Interpretation:

- This is the first strict no-target-GT single-scene evidence and one row of the frozen multiscene table; the current strongest protocol package is `strict_frozen_policy_multiscene_20260626_052500`.
- It is a nonzero accepted residual surface texture with about `1.18%` changed target pixels.
- It improves PSNR and LPIPS slightly relative to the Phase-F compact parent, but SSIM slightly regresses.
- It should be described as strict protocol proof-of-life, not as a three-metric or paper-final win.

---

## Strict Frozen-Policy Multiscene Key Facts

Artifact root:

```text
strict_frozen_policy_multiscene_20260626_052500
```

The same frozen policy was applied to `counter,bonsai,room`:

```text
texture_size_candidates=16
support_expansion_mode=none
atlas_empty_bin_fill_mode=face_mean
surface_multiscale_prior_blend_candidates=0.5
max_abs_delta_rgb_candidates=0.12
policy_val_bin_uncertainty_guard=disabled
strict_no_target_gt_apply=true
```

| scene | protocol pass | target GT visible to apply | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| counter | `True` | `False` | `True` | `0.25` | `0.01177355` | `+0.002131` | `-0.000047` | `-0.000085` |
| bonsai | `True` | `False` | `True` | `0.25` | `0.00151333` | `+0.001225` | `-0.000010` | `-0.000018` |
| room | `True` | `False` | `False` | `0.0` | `0.00000000` | `-0.000097` | `-0.000003` | `-0.000007` |
| mean | 3/3 | 0/3 visible | 2/3 nonzero | - | - | `+0.001086` | `-0.000020` | `-0.000037` |

Interpretation:

- This is the older frozen-policy strict vNext protocol package: no target GT is visible to adapter apply on all three scenes.
- It gives weak positive PSNR and LPIPS movement versus the Phase-F compact parent mainly through the two nonzero accepted scenes; the `room` row is fallback/no-op with `changed_fraction=0`, so its tiny metric deltas should be treated as parent-level evaluation noise rather than residual gain.
- It does not solve structure quality: SSIM regresses on all three scenes, and `room` correctly falls back to parent output.
- It is useful for a mentor update and method diagnosis, not for a paper-final superiority claim.

---

## Main Reject Reason

The first full-candidate adapter rejected the selected nonzero candidate because risk gates failed:

```text
cvar20_view_relative_gain -0.155822 < 0
min_view_relative_gain -0.248611 < -0.000001
ssim_gain -0.000018626 < -0.000000100
ssim_positive_view_fraction 0.250000 < 0.550000
ssim_min_view_gain -0.000094116 < -0.000010000
image_l1_min_view_gain -0.000016468 < -0.000001000
```

Short explanation for slides:

> The candidate improved mean MSE on policy-val, but lower-tail views and image-level SSIM were unsafe. The certificate rejected it and wrote exact no-op fallback.

Hard-bin soft-shrink follow-up:

```text
cvar20_view_relative_gain -0.000741 < 0
min_view_relative_gain -0.002424 < -0.000001
```

This shows that hard bin allowlisting was still too brittle after soft shrink. The accepted face-softshrink run keeps the soft local downweighting and face-level train-policy-val guard, but disables hard bin allowlisting.

---

## Recommended Citation In Main Report

Use this wording:

> The garden vNext pilot first validated the safety path through exact fallback/no-op rejection. The follow-up face-softshrink run then accepted a nonzero residual surface texture (`accepted=true`, `selected_alpha=0.0625`, `target_changed_fraction=0.002080`) and gave tiny held-out gains versus the no-op/fallback parent. This is a real nonzero milestone, not full9 or v106/clean-baseline superiority.

For the strict frozen-policy multiscene run, use this wording:

> The strict frozen-policy vNext pilot applied the same face-softshrink policy to `counter,bonsai,room` with `target_gt_visible_to_apply=false` in all scenes. It accepted nonzero residual surface textures on counter and bonsai and safely fell back/no-op on room with `changed_fraction=0`. Mean PSNR/LPIPS move slightly in the right direction versus the Phase-F compact parent, but the fallback row is parent-level evaluation noise and SSIM regresses on all three scenes, so this is a protocol and diagnosis milestone rather than a comprehensive quality win.

For the current structure-aware ready4 run, use this wording:

> The fixed structure-aware vNext policy applies train-policy-val L1/gradient risk shrink and strict no-target-GT apply. On the four currently input-ready scenes (`counter,bonsai,room,garden`), it is `4/4` protocol pass, `4/4` target-GT-hidden, and `4/4` nonzero accepted, with mean delta vs Phase-F compact parent of `+0.00076151 PSNR / -0.00000302 SSIM / -0.00002038 LPIPS`. This is the current vNext protocol/method milestone, but it is still not a full9 or v106/clean-baseline superiority result because five scenes still lack rebuilt evidence/carrier inputs.

Avoid this wording:

> vNext fully improves garden / vNext beats MeshSplatting.

Avoid this table label:

```text
vNext vs MeshSplatting baseline
```

Use this label instead:

```text
ready4 vNext structure-aware shrink vs Phase-F compact parent; not full9 or v106/clean superiority
```

---

## Related Reports And Machine-Readable Inputs

The human-readable technical report and PPT index is:

```text
docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md
docs/car_model/6-26-vNext-ManifestRunner-and-Full9Gap-Log.md
docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md
docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_scene_config_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.md
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.json
```

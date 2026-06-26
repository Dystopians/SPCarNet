# vNext Artifacts Index

日期：2026-06-26
用途：索引当前 repo 内保存的 `vNext_certified_residual_surface_texture` artifacts，并说明哪些可以作为 PPT 证据。

---

## Summary

当前 artifacts 有三个真实 `garden` 单场景 run 和一个 `counter` strict run：

```text
garden_20260626_004134
garden_hardbin_softshrink_20260626_035631
garden_face_softshrink_20260626_040558
counter_strict_face_softshrink_20260626_045300
```

结论必须诚实表述：

```text
initial full candidate: protocol passed, accepted=false, fallback_noop
hard-bin soft-shrink: protocol passed, accepted=false, fallback_noop
face-softshrink: protocol passed, accepted=true, selected_alpha=0.0625, changed_fraction=0.002080
counter strict face-softshrink: protocol passed, target_gt_visible_to_apply=false, accepted=true, selected_alpha=0.25, changed_fraction=0.01177355
```

因此，这个目录证明的是：**vNext 安全证书和 no-op fallback 机制有效，并且 face-softshrink 已经在 garden/counter 上产生真实非零 accepted residual surface texture**。但这些非零收益仍很小或指标混合，还不是 full9、v106 或 clean MeshSplatting 超越证据。

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
| `garden_face_softshrink_20260626_040558/` | face-softshrink accepted run：manifest、report、audit、metrics、per-view、logs、summary JSON、qualitative panel | 当前 vNext 最重要的非零里程碑证据 |
| `garden_face_softshrink_20260626_040558/garden_face_softshrink_summary.json` | 三轮结果、delta、per-view win counts、guard 诊断 | PPT/脚本可读摘要 |
| `garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png` | GT / parent / vNext / error maps / vNext-parent amplified diff | 展示非零改动存在，但视觉收益非常弱 |
| `counter_strict_face_softshrink_20260626_045300/` | counter strict face-softshrink accepted run：manifest、report、audit、metrics、per-view、summary JSON、logs | 当前最重要的 strict no-target-GT apply vNext 证据 |
| `counter_strict_face_softshrink_20260626_045300/counter_strict_face_softshrink_summary.json` | parent/vNext metrics、delta、protocol audit、target apply 摘要 | PPT 表格首选数据源 |
| `counter_strict_face_softshrink_20260626_045300/target_evidence_no_gt_audit.json` | target evidence stripped-key 审计 | 证明 adapter apply 阶段没有 target GT/residual keys |

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

- This is the strongest current vNext protocol evidence because target evidence is stripped before adapter apply.
- It is a nonzero accepted residual surface texture with about `1.18%` changed target pixels.
- It improves PSNR and LPIPS slightly relative to the Phase-F compact parent, but SSIM slightly regresses.
- It should be described as strict protocol proof-of-life, not as a three-metric or paper-final win.

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

For the strict counter run, use this wording:

> The counter vNext strict pilot validated the no-target-GT apply path: `target_gt_visible_to_apply=false`, `accepted=true`, `selected_alpha=0.25`, `target_changed_fraction=0.01177355`. It gives a tiny PSNR/LPIPS improvement versus the Phase-F compact parent while SSIM slightly regresses, so it is a protocol milestone rather than a comprehensive quality win.

Avoid this wording:

> vNext fully improves garden / vNext beats MeshSplatting.

Avoid this table label:

```text
vNext vs MeshSplatting baseline
```

Use this label instead:

```text
garden vNext face-softshrink nonzero micro-gain vs no-op/fallback parent
```

---

## Related Report

The human-readable technical report and PPT index is:

```text
docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md
```

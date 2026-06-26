# SPCarNet 当前技术报告与 vNext 严格协议增补

日期：2026-06-26
用途：给 mentor/PPT 汇报使用的当前状态总览、证据索引和 claim 边界。

---

## 1. 克隆后入口

```bash
git clone https://github.com/Dystopians/SPCarNet.git
cd SPCarNet
```

建议阅读顺序：

1. `SPCARNET_REPORT_INDEX.md`
2. `docs/car_model/6-26-SPCarNet-Mentor-PPT-Status-And-vNext-Strict-Report.zh.md`
3. `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md`
4. `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md`
5. `docs/car_model/6-26-SPCarNet-vNext-Strict-FrozenPolicy-Multiscene-Log.md`
6. `docs/car_model/vnext_artifacts/README.md`
7. `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md`
8. `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md`
9. `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md`

---

## 2. 一页版结论

当前最稳妥的主线仍是 **v106 POD-MoE base-preserve**：它保留 MeshSplatting parent，在 mesh surface 上挂载 train-evidence residual experts，并用可靠性 gate 控制何处允许 residual 生效。它在本地 assembled selected full9 表上超过 clean MeshSplatting baseline：

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 |

2026-06-26 的新增进展是 **vNext Evidence-Certified Residual Surface Texturing**：把 Phase-J render-time teacher 的 residual repair 尝试蒸馏成 persistent face/UV residual surface texture，并补上严格 no-target-GT apply 协议。

这轮新增证据可以诚实表述为：

- vNext strict runner 已实现并通过 smoke/dry-run。
- target split 为 test 时，adapter apply 阶段不再能看到 `rgb_gt` 或 teacher/GT residual keys。
- garden 有第一个非零 accepted residual texture，但收益极小。
- frozen face-softshrink policy 已在 `counter,bonsai,room` 三个 strict 场景上完成：3/3 protocol pass，3/3 `target_gt_visible_to_apply=false`，2/3 accepted nonzero，1/3 fallback/no-op 且 `changed_fraction=0`。
- 三场景均值相对 Phase-F compact parent 为 `+0.001086` PSNR、`-0.000020` SSIM、`-0.000037` LPIPS；PSNR/LPIPS 弱正信号主要来自 counter/bonsai，room 是 parent-level fallback 微小评估差异，SSIM 三场景全退是当前最明确瓶颈。
- vNext 仍不能宣称全面超越 clean MeshSplatting、v106 或 Phase-J；它目前是协议和表示路线的里程碑，不是 paper-final 质量闭环。

---

## 3. vNext 方法概述

vNext 的核心思想：

```text
Phase-J render-time teacher
  -> train-only surface residual evidence
  -> face / UV / barycentric residual texture atlas
  -> train-policy-val alpha/capacity/certificate
  -> no-target-GT target apply
  -> final evaluation only sees target GT
```

和基础 MeshSplatting 的区别：

- MeshSplatting 只渲染原始训练得到的 mesh/splat parent。
- SPCarNet/v106 在 parent 上增加 surface-addressed residual experts。
- vNext 进一步尝试把 render-time teacher 的 residual 修复压缩进持久 surface texture，并用证书避免 out-of-trajectory view 崩塌。

vNext 的严格公平性约束：

- candidate fitting 只用 train evidence；
- alpha/capacity/certificate 只用 train-policy-val；
- target apply 阶段不能读取 target `rgb_gt` 或 target residual；
- target `rgb_gt` 只在渲染完成后由 eval-only population stage 写入 `gt/`，供最终评估使用。

---

## 4. 新增 vNext 结果

### 4.1 Garden Face-SoftShrink

Artifact root：

```text
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/
```

| 字段 | 值 |
|---|---:|
| protocol audit passed | `True` |
| accepted | `True` |
| effective policy | `accepted_atlas` |
| selected alpha | `0.0625` |
| changed fraction | `0.002080` |
| PSNR / SSIM / LPIPS | `24.741079 / 0.754051 / 0.248020` |
| delta vs no-op/fallback parent | `+0.000076 / +0.00000197 / -0.00000323` |

结论：这是第一个非零 accepted residual surface texture，但提升非常小，适合证明路线可跑，不适合证明视觉优势明显。

### 4.2 Counter Strict Face-SoftShrink

Artifact root：

```text
docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/
```

| 字段 | 值 |
|---|---:|
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
| changed pixels | `571207` |
| changed fraction | `0.01177355` |
| policy-val relative gain | `0.04431575` |
| policy-val SSIM gain | `0.00010365` |
| test PSNR / SSIM / LPIPS | `26.752003 / 0.862004 / 0.251912` |

相对 Phase-F compact parent：

| metric | Phase-F parent | vNext strict | delta |
|---|---:|---:|---:|
| PSNR | 26.749872 | 26.752003 | +0.002131 |
| SSIM | 0.862051 | 0.862004 | -0.000047 |
| LPIPS | 0.251998 | 0.251912 | -0.000085 |

结论：counter strict 是第一条 strict no-target-GT 单场景非零证据；当前更完整的协议证据是 4.3 的 strict frozen-policy multiscene，因为同一套 policy 已跨 `counter,bonsai,room` 跑通。质量上 counter 是 PSNR/LPIPS 微增、SSIM 微降，仍不能叫全面胜出。

### 4.3 Strict Frozen-Policy Multiscene

Artifact root：

```text
docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/
```

同一套 frozen policy 被复制到 `counter,bonsai,room`，没有按场景调参：

| scene | protocol / apply GT | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|
| counter | pass / `False` | `True` | `0.25` | `0.011774` | `+0.002131` | `-0.000047` | `-0.000085` |
| bonsai | pass / `False` | `True` | `0.25` | `0.001513` | `+0.001225` | `-0.000010` | `-0.000018` |
| room | pass / `False` | `False` | `0.0` | `0.000000` | `-0.000097` | `-0.000003` | `-0.000007` |
| mean | 3/3 pass | 2/3 nonzero | - | - | `+0.001086` | `-0.000020` | `-0.000037` |

结论：这组实验比单场景 counter 更重要，因为它证明了 strict no-target-GT apply 协议和 frozen policy 可以跨场景跑通；但它也暴露了当前方法的核心质量短板：SSIM/结构一致性不足，`room` 会被证书正确拒绝并回退为 no-op，表中微小 delta 只能视为 parent-level 评估差异。

---

## 5. 证据索引

### v106 / 当前主线

| 路径 | 内容 |
|---|---|
| `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` | clean/v104c/v106 full9 对比 |
| `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md` | full9 assembled table |
| `docs/car_model/assets/v106_qualitative/` | v106 qualitative contact sheets |
| `docs/car_model/6-25-SPCarNet-PPT-Technical-Report-Current.md` | PPT-ready 当前主线报告 |
| `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md` | mentor 技术报告 |

### vNext / 严格协议

| 路径 | 内容 |
|---|---|
| `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md` | vNext runner、strict apply、smoke/dry-run、真实 pilot 日志 |
| `docs/car_model/6-26-SPCarNet-vNext-Strict-FrozenPolicy-Multiscene-Log.md` | strict frozen-policy 三场景日志和结论边界 |
| `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md` | vNext 中文技术报告和 claim 边界 |
| `docs/car_model/vnext_artifacts/README.md` | vNext artifact 根索引 |
| `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_summary.json` | garden 非零 accepted 摘要 |
| `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png` | garden 定性面板 |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/counter_strict_face_softshrink_summary.json` | counter strict 摘要 |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/counter_vnext_certified_residual_texture_manifest.json` | counter strict provenance / commands / protocol audit |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/surface_residual_region_texture_adapter_audit.md` | counter adapter 人类可读证书 |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/target_evidence_no_gt_audit.json` | target evidence stripping 审计 |
| `docs/car_model/vnext_artifacts/bonsai_strict_face_softshrink_20260626_052500/bonsai_strict_face_softshrink_summary.json` | bonsai strict 摘要 |
| `docs/car_model/vnext_artifacts/room_strict_face_softshrink_20260626_052500/room_strict_face_softshrink_summary.json` | room strict 摘要 |
| `docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/strict_frozen_policy_multiscene_summary.md` | 三场景 frozen-policy 聚合表 |

---

## 6. 推荐 PPT 叙事

1. MeshSplatting 是强 parent。
2. SPCarNet 不从零替换它，而是在 mesh surface 上增加有证据支持的 residual capacity。
3. v106 已经在本地 selected full9 表上超过 clean MeshSplatting baseline，是当前最稳的结果主线。
4. 更严格的 split 实验暴露出 naive gate 会失败，因此 paper-level 方法不能只看平均指标。
5. vNext 的贡献是把 Phase-J 的 render-time teacher 转成 persistent residual surface texture，并补上 no-target-GT apply 协议。
6. garden/counter/bonsai/room 证明非零 residual texture 与 fallback/no-op 都可以在 strict 协议下工作，但目前提升还很小，没有达到最终论文闭环。
7. 下一步应集中解决“SSIM/结构一致性”和“surface-addressed residual 的视觉可见性”，而不是继续扩大脆弱的参数搜索。

---

## 7. 当前短板

- vNext 非零收益仍是微小级别，视觉优势不明显。
- strict frozen-policy 三场景没有三指标全胜，SSIM 在 `counter,bonsai,room` 全部略低于 Phase-F compact parent。
- full9 strict vNext 尚未完成。
- vNext 尚未证明超过 v106 或 Phase-J。
- `/data` 已接近满载，长程实验需要先清理或迁移 artifact root。

---

## 8. 结论状态

`NOT COMPLETE`。

当前 repo 已经包含可克隆的 v106 汇报包、vNext 严格协议实现记录、garden/counter/bonsai/room 真实 vNext artifacts 和索引文件。它足够支持一次诚实的 mentor/PPT 技术汇报，但还不能作为“全面超越 MeshSplatting 的 paper-final 方法”提交。

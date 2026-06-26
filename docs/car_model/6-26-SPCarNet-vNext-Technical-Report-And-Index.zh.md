# SPCarNet vNext 技术报告与汇报索引

日期：2026-06-26
用途：给 2026-06-27 PPT 准备的中文技术报告、证据索引和结论边界。
范围：基于现有 vNext prompt、feasibility plan、implementation log、`vnext_artifacts` 已有结果整理；2026-06-26 已补入 garden soft-shrink/face-softshrink、counter/bonsai/room strict frozen-policy face-softshrink、ready4 structure-aware shrink，以及 full9 fixed-policy cleanup 结果。

---

## 0. 一页版结论

vNext 的目标不是再做一个小 gate 或 alpha tweak，而是把当前最强的 Phase-J render-time ELA residual repair 蒸馏成 MeshSplatting-compatible 的持久 surface representation：

```text
Phase-J render-time teacher
  -> train-only residual / surface evidence cache
  -> face / UV / barycentric residual texture atlas
  -> train-policy-val capacity, alpha, certificate
  -> certified nonzero texture or exact parent fallback
  -> test-only final evaluation
```

当前可以诚实汇报的是：

- **方法方向合理**：Phase-J 已证明 residual repair 有大收益，v106 证明 residual representation 可与 MeshSplatting parent 兼容。
- **协议与编排层已跑通**：vNext scene/full9 runner、manifest、no-test-GT audit、dry-run、W&B offline dry-run、assembler dry-run 已完成。
- **已有真实 pilot 从 garden 扩展到 strict 三场景 frozen-policy**：garden face-softshrink 是第一个非零 accepted proof-of-life；同一套 frozen face-softshrink policy 又在 `counter,bonsai,room` 上完成 strict no-target-GT apply 验证，3/3 protocol pass、3/3 `target_gt_visible_to_apply=false`、2/3 nonzero accepted、1/3 fallback/no-op 且 `changed_fraction=0`。
- **最新 structure-aware shrink 是当前首选 vNext 里程碑**：它新增 train-policy-val 局部 L1/gradient structure-risk shrink，在 `counter,bonsai,room,garden` strict no-target-GT apply 下达到 4/4 protocol pass、4/4 `target_gt_visible_to_apply=false`、4/4 nonzero accepted，并把 `room` 从旧策略 fallback/no-op 变成 accepted nonzero。相对 Phase-F compact parent 均值为 `+0.00076151` PSNR、`-0.00000302` SSIM、`-0.00002038` LPIPS。
- **最新 full9 固定策略结果已完成但不 promotion**：`stump/treehill/flowers/kitchen/bicycle` 五个缺失输入链重建后，固定 structure-aware shrink policy 完成 full9 cleanup run：`9 / 9` scenes complete、`9 / 9` protocol pass、`6 / 9` accepted nonzero、`3 / 9` fallback/no-op，mean changed fraction `0.002756271`，mean metrics `25.067699 PSNR / 0.741260 SSIM / 0.306689 LPIPS`。
- **不能宣称 vNext 已全面超越 MeshSplatting baseline**：full9 fixed-policy cleanup 低于本地 clean MeshSplatting（`25.151682 / 0.749018 / 0.287621`）和 v106（`25.831280 / 0.760830 / 0.268435`）。它是真实 full9 协议闭环、审计闭环和瓶颈诊断，不是质量超越证据。
- **当前 verified representation 质量线仍是 v106 POD-MoE base-preserve**；当前 verified broad RGB endpoint 仍是 Phase-J，但 Phase-J 是 render-time guarded ELA portfolio，不是 baked representation。

PPT 推荐讲法：

> vNext 是把 Phase-J 的强 render-time residual teacher 转成可部署 surface texture representation 的下一阶段。我们已经完成 strict no-target-GT full9 固定策略执行：9/9 场景完成、6/9 accepted nonzero、3/9 fallback/no-op，并保存了完整机器可读审计。当前收益仍不足，full9 均值低于 clean MeshSplatting 和 v106，因此它是协议闭环与下一阶段研究 scaffold，不是 promoted endpoint。

---

## 1. 证据等级与 claim 边界

| 层级 | 当前证据 | 可以说 | 不能说 |
|---|---|---|---|
| Phase-J endpoint | full9：9/9 scene strict RGB wins vs selected clean MeshSplatting；mean `+1.3311` PSNR / `+0.0347` SSIM / `-0.0634` LPIPS；平均删去 `7.6479%` triangles | 当前最强 broad RGB endpoint，可作为 teacher / upper bound | 不能说它已经是持久 baked representation；不要把 7.65% triangle reduction 讲成 compression-paper 级结果 |
| v106 POD-MoE base-preserve | full9：mean `+0.6796` PSNR / `+0.0118` SSIM / `-0.0192` LPIPS vs clean；但仅比 v104c 多约 `+0.0022` PSNR | 当前 verified MeshSplatting-compatible representation 质量线 | 不能说 v106 已关闭 paper-final branch；不能把微小增量讲成大突破 |
| v110/v113 系列 | strict split / OOT / fallback 诊断 | 证明 naive train-only gate 仍可能 miss test failures，fallback 很重要 | 不能把 v113b/v113c 讲成质量突破，它们主要是安全修复 |
| vNext protocol | py_compile、schema smoke、scene dry-run、W&B offline dry-run、two-scene wrapper dry-run、full9 fixed-policy cleanup | 协议、manifest、no-test-GT audit、runner interface 和 full9 执行已可用 | 不能说 full9 协议完成等价于质量完成 |
| vNext garden fallback pilot | 单场景真实 run，协议审计通过，但 fallback/no-op | 安全证书拒绝不可靠候选，未使用 test GT 做选择 | 不能说 fallback 指标是 vNext 非零方法收益 |
| vNext garden face-softshrink pilot | 单场景真实 run，协议审计通过，非零 accepted atlas，`0.208%` target pixels changed | 第一个可汇报的非零 vNext residual surface texture 里程碑；相对 no-op parent 有极小三指标正向变化 | 不能说已 full9 闭环；不能说已超过 v106 或 clean MeshSplatting；不能说视觉效果明显 |
| vNext counter strict face-softshrink pilot | 单场景真实 run，strict no-target-GT apply 协议审计通过，非零 accepted atlas，`1.177%` target pixels changed | strict no-target-GT apply 的第一个非零证据；adapter apply 阶段看不到 target GT | 不能说已三指标全胜；SSIM 相对 Phase-F compact parent 微退 |
| vNext strict frozen-policy multiscene | `counter,bonsai,room` 三场景真实 run；3/3 protocol pass；3/3 target GT hidden from apply；2/3 nonzero accepted；1/3 fallback/no-op with `changed_fraction=0` | 当前最强的 vNext 公平性/protocol package；证明 frozen policy 可跨场景执行 | 不能说质量闭环；均值只是 `+0.001086` PSNR / `-0.000020` SSIM / `-0.000037` LPIPS，room delta 是 parent-level 评估差异，SSIM 0/3 胜 |
| vNext structure-aware shrink ready4 | `counter,bonsai,room,garden` 四场景真实 run；4/4 protocol pass；4/4 target GT hidden from apply；4/4 nonzero accepted；room 从 old fallback/no-op 变成 accepted nonzero | 当前首选 vNext 严格里程碑；证明 policy-val structure-risk shrink 能缓解 room fallback 和部分 SSIM 风险，并把 garden 纳入 strict row | 不能说 paper-final；均值只是 `+0.00076151` PSNR / `-0.00000302` SSIM / `-0.00002038` LPIPS，仍未 full9 quality table，仍未超过 v106/clean |
| vNext full9 fixed-policy cleanup | 9/9 complete；9/9 protocol pass；6/9 accepted nonzero；3/9 fallback/no-op；mean `25.067699 / 0.741260 / 0.306689` | full9 协议/审计/证据闭环已完成，当前方法瓶颈已被清楚定位 | 不能说质量 promotion；结果低于 clean MeshSplatting 和 v106，且没有 full9 定性图片 payload |

---

## 2. 方法模块

### 2.1 Residual Teacher Cache

目标：只用 train / policy-val 视角建立 residual teacher evidence。

缓存内容应包括：

- parent render；
- `GT - parent` RGB residual；
- 可选 Phase-J / ELA teacher residual；
- depth、normal、face id、UV / barycentric bin；
- support count、residual variance、sign consistency；
- split、source path、hash / mtime、`selection_uses_test_gt=false` 审计字段。

vNext 的关键公平性约束：

```text
candidate generation: fit train evidence only
policy / alpha / capacity selection: train-policy-val only
test GT: final evaluation only
```

### 2.2 Adaptive Residual Surface Texture

核心表示是贴在 mesh surface 上的 residual texture：

```text
output_rgb = parent_rgb + confidence * residual_rgb
```

surface address 可来自：

- face id；
- UV 或 barycentric bin；
- normal / view direction / depth / boundary cues；
- parent color；
- per-bin support、variance、tail risk。

它和普通图像后处理的差异：

- residual 必须绑定到 surface address；
- 不允许用 held-out test GT 选择 alpha、capacity 或 fallback；
- 证据不足时必须 exact parent fallback。

### 2.3 Capacity Reallocation

vNext 不应包装成 generic compression。合理说法是：

- geometry-safe compaction 只用于释放低价值 surface budget；
- residual capacity 分配给 residual-hot、multi-view-consistent、surface-addressable 区域；
- 必须报告 texture size、face/bin count、storage、runtime overhead、triangle count 和 fallback rate。

已有 garden pilot 中，adapter 规划并执行了 `48` 个 policy candidates，atlas 使用 `319` 个 base faces，fit samples 为 `201676`，policy-val samples 为 `71583`，但最后未接受任何非零 target edit。

### 2.4 Train-Only Safety Certificate

证书的作用不是提高分数，而是阻止不可靠 candidate 进入 target/test render。

主要条件包括：

- policy-val mean gain；
- positive-view fraction；
- CVaR / lower-tail gain；
- min-view gain；
- image-level SSIM gate；
- image-level L1 gate；
- face/bin support 与 uncertainty guard；
- target footprint / OOT support；
- rejected candidate 写 no-op fallback 和 machine-readable reject reason。

garden pilot 的价值在这里：候选在 mean MSE 上有正向迹象，但 lower-tail、SSIM 和 min-view 不稳定，因此证书拒绝并写 no-op。这是安全证据，不是质量提升证据。

---

## 3. 当前进展

### 3.1 已完成的文档和协议

| 文件 | 作用 |
|---|---|
| `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md` | vNext 原始任务 prompt，定义 Evidence-Certified Residual Surface Texturing |
| `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md` | feasibility verdict、风险、阶段计划、go/no-go 标准 |
| `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md` | 第一阶段 implementation log，记录脚本、dry-run、资源阻塞 |
| `docs/car_model/vnext_artifacts/` | 当前已复制入 repo 的 vNext run artifacts |

### 3.2 已完成的实现里程碑

Implementation log 记录的新接口层包括：

- `scripts/car_model/ecsr_vnext_protocol.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`
- `scripts/car_model/run_vnext_certified_residual_texture_full9.py`
- `scripts/car_model/assemble_vnext_certified_residual_texture_report.py`
- `scripts/car_model/smoke_test_vnext_no_test_gt_certificate_schema.py`

这些是 vNext 的 orchestration / provenance / report layer，不是低层 renderer 重写。

已验证：

- py_compile；
- no-test-GT certificate schema smoke；
- synthetic scene dry-run；
- W&B offline dry-run；
- two-scene full9-wrapper dry-run，assembler 汇总为 `DRY_RUN`。

### 3.3 资源状态

Implementation log 记录：真实 full9 不应直接从 repo/output tree 启动，原因是 `/data` 接近满、`/dev/shm` 与 GPU 有长任务压力。建议先做 one-scene pilot，并把输出和 W&B cache 放到 `/dev/shm`。

---

## 4. 已有 vNext garden pilot

artifact root：

```text
docs/car_model/vnext_artifacts/garden_20260626_004134/
```

### 4.1 Run 摘要

| 字段 | 值 |
|---|---|
| scene | `garden` |
| method | `vNext_certified_residual_surface_texture` |
| status | `COMPLETE` |
| protocol audit passed | `True` |
| target split | `test` |
| selection uses test GT | `False` |
| capacity selected on | `train_policy_val_and_gt_free_target_footprint` |
| thresholds selected on | `train_policy_val` |
| source parent | Phase-F compact parent：`ours_26000_phasef_extra_compact_base` |
| skip_teacher_cache | `True`，使用既有 train evidence/cache |
| texture command elapsed | `8775.8s` |
| eval elapsed | `46.3s` |

### 4.2 Certificate 结果

| 字段 | 值 |
|---|---:|
| accepted | `False` |
| selected alpha | `0.0` |
| effective policy | `fallback_noop` |
| target written views | `24` |
| target changed fraction | `0.000000` |
| policy-val relative gain, selected candidate | `0.138933` |
| policy-val positive-view fraction | `0.750000` |
| policy-val CVaR20 view relative gain | `-0.155822` |
| policy-val min-view relative gain | `-0.248611` |
| policy-val image SSIM gain | `-0.000018626` |
| policy-val image SSIM positive-view fraction | `0.250000` |
| policy-val image L1 min-view gain | `-0.000016468` |

主要拒绝原因：

```text
cvar20_view_relative_gain < 0
min_view_relative_gain < threshold
ssim_gain < threshold
ssim_positive_view_fraction < threshold
ssim_min_view_gain < threshold
image_l1_min_view_gain < threshold
```

解释：

- candidate 的 mean MSE 方向有局部正信号；
- 但 tail view 和 SSIM 风险没有过证书；
- 因此系统写入 exact no-op/fallback；
- 这是“安全闸门有效”的证据，不是“vNext texture 有效”的证据。

### 4.3 Test 指标

final eval JSON 中的 garden vNext fallback 输出：

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| `ours_26000_vnext_certified_residual_surface_texture` | `24.741003` | `0.754049` | `0.248023` |

必须同时说明：

- 这组数来自 fallback/no-op 输出，`changed_fraction=0.0`；
- source parent 是 Phase-F compact parent，不是 selected clean MeshSplatting baseline；
- 因此不能用这组数宣称 vNext 非零 residual texture 超过 clean MeshSplatting 或 v106。

### 4.4 Soft-Shrink 后续里程碑

后续两轮聚焦实验补齐了关键诊断：

| run | artifact root | hard bin guard | accepted | alpha | changed fraction | PSNR | SSIM | LPIPS | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| initial full candidate | `docs/car_model/vnext_artifacts/garden_20260626_004134/` | 未进入 | `False` | `0.0` | `0.000000` | `24.741003` | `0.754049` | `0.248023` | fallback-only 安全证据 |
| hard-bin soft-shrink | `docs/car_model/vnext_artifacts/garden_hardbin_softshrink_20260626_035631/` | enabled | `False` | `0.0` | `0.000000` | `24.741003` | `0.754049` | `0.248023` | soft shrink 修复 SSIM 方向，但 hard bin guard 仍拒绝 |
| face-softshrink | `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/` | disabled | `True` | `0.0625` | `0.002080` | `24.741079` | `0.754051` | `0.248020` | 第一个非零 accepted residual surface texture |
| counter strict face-softshrink | `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/` | disabled | `True` | `0.25` | `0.011774` | `26.752003` | `0.862004` | `0.251912` | strict no-target-GT apply；vs Phase-F compact parent PSNR/LPIPS 微增、SSIM 微退 |
| bonsai strict face-softshrink | `docs/car_model/vnext_artifacts/bonsai_strict_face_softshrink_20260626_052500/` | disabled | `True` | `0.25` | `0.001513` | `28.865564` | `0.896002` | `0.259322` | strict no-target-GT apply；vs Phase-F compact parent PSNR/LPIPS 微增、SSIM 微退 |
| room strict face-softshrink | `docs/car_model/vnext_artifacts/room_strict_face_softshrink_20260626_052500/` | disabled | `False` | `0.0` | `0.000000` | `28.739004` | `0.884790` | `0.249916` | strict no-target-GT apply；证书拒绝并 fallback/no-op，微小 delta 视为 parent-level 评估差异 |

face-softshrink 的 train-policy-val 证书字段：

| 字段 | 值 |
|---|---:|
| policy-val relative gain | `0.006009` |
| positive-view fraction | `1.000000` |
| CVaR20 view relative gain | `0.002396` |
| min-view relative gain | `0.000258` |
| image SSIM gain | `0.000001659` |
| image SSIM positive-view fraction | `0.916667` |
| image L1 gain | `0.000000179` |
| face guard allowed faces | `58` |
| face guard allowed sample fraction | `0.557348` |

face-softshrink 相对 no-op/fallback parent 的 held-out garden test delta：

| 指标 | aggregate delta | per-view better | tie | worse |
|---|---:|---:|---:|---:|
| PSNR | `+0.000076` | `22` | `0` | `2` |
| SSIM | `+0.00000197` | `24` | `0` | `0` |
| LPIPS | `-0.00000323` | `22` | `0` | `2` |

定性面板：

```text
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png
```

这张图包含 GT、parent、vNext、放大误差图和放大 `vNext-parent` 差分。需要诚实说明：图上可见改动很弱，定性收益不明显；它适合作为“非零修复已发生”的证据，不适合作为“视觉效果大幅提升”的证据。

完整里程碑日志：

```text
docs/car_model/6-26-SPCarNet-vNext-SoftShrink-Garden-Milestone-Log.md
```

---

## 5. 相对 MeshSplatting baseline 的诚实结论

### 5.1 已验证的强结果属于 Phase-J

在本地 Mip-NeRF360 full9、相同 split、相同 evaluator、selected clean MeshSplatting baseline 下，Phase-J 的既有结果是：

| 指标 | Phase-J vs selected clean MeshSplatting |
|---|---:|
| scene-level strict RGB wins | `9 / 9` |
| held-out view strict RGB wins | `244 / 246` |
| mean PSNR | `25.1517 -> 26.4828` |
| mean SSIM | `0.7490 -> 0.7837` |
| mean LPIPS | `0.2876 -> 0.2243` |
| mean delta | `+1.3311` PSNR / `+0.0347` SSIM / `-0.0634` LPIPS |
| mean triangle removed | `7.6479%` |

但 Phase-J 是 render-time guarded ELA portfolio。它适合作为 vNext teacher / upper bound，不是 vNext 已完成的持久 representation。

### 5.2 已验证的 representation 线属于 v106

v106 POD-MoE base-preserve 的 assembled full9 结果：

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | `25.151682` | `0.749018` | `0.287621` | baseline |
| v104c shrink view-affine field | 9 | `25.829099` | `0.760727` | `0.268548` | `+0.677417 / +0.011709 / -0.019073` |
| v106 POD-MoE base-preserve | 9 | `25.831280` | `0.760830` | `0.268435` | `+0.679598 / +0.011812 / -0.019185` |

诚实边界：

- v106 是当前 verified representation 质量线；
- v106 比 clean 有稳定正收益；
- 但 v106 相比 v104c 的增量很小，不能包装成大幅突破；
- vNext 必须证明非零 texture/capacity/certificate 能在 v106 之上产生实质收益。

### 5.3 vNext 当前仍没有 baseline / v106 超越证据

当前 vNext 的最新结论是：

```text
protocol passed
selection_uses_test_gt = false
garden face-softshrink candidate accepted
counter strict face-softshrink candidate accepted
bonsai strict face-softshrink candidate accepted
room strict face-softshrink candidate rejected to fallback
counter target_gt_visible_to_apply = false
bonsai target_gt_visible_to_apply = false
room target_gt_visible_to_apply = false
counter changed_fraction = 0.01177355
counter held-out delta vs Phase-F compact parent = +0.00213 PSNR / -0.000047 SSIM / -0.000085 LPIPS
strict frozen-policy 3-scene mean delta vs Phase-F compact parent = +0.001086 PSNR / -0.000020 SSIM / -0.000037 LPIPS
```

所以对外结论应写成：

> vNext 当前完成了 strict no-target-GT apply protocol、certificate/fallback proof-of-life，并在 garden/counter/bonsai 上得到非零 accepted residual surface texture，在 room 上安全回退且 `changed_fraction=0`。strict 三场景均值显示 counter/bonsai 带来 PSNR/LPIPS 弱正信号，但 SSIM 三场景全退，room 的微小 delta 只是 parent-level 评估差异。因此它还没有提供超过 clean MeshSplatting baseline、v106 或 Phase-J teacher 的证据。

---

## 6. 关键实验路径与索引

### 6.1 必读文档

| 目的 | 路径 |
|---|---|
| vNext 原始 prompt | `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md` |
| vNext feasibility plan | `docs/car_model/6-26-SPCarNet-vNext-Feasibility-And-Execution-Plan.md` |
| vNext implementation log | `docs/car_model/6-26-SPCarNet-vNext-Implementation-Log.md` |
| vNext strict multiscene log | `docs/car_model/6-26-SPCarNet-vNext-Strict-FrozenPolicy-Multiscene-Log.md` |
| 当前状态 addendum | `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md` |
| Phase-J vs clean MeshSplatting 完整报告 | `docs/car_model/6-25-SPCarNet-Current-Method-vs-MeshSplatting-Complete-Report.zh.md` |
| v106 final technical report | `docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md` |
| v106 full9 assembled table | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md` |
| v106 vs v104c compare | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` |

### 6.2 vNext artifacts

| artifact | 说明 |
|---|---|
| `docs/car_model/vnext_artifacts/README.md` | artifact 根目录索引和解读说明 |
| `docs/car_model/vnext_artifacts/garden_20260626_004134/garden_vnext_certified_residual_texture_report.md` | garden run 顶层报告 |
| `docs/car_model/vnext_artifacts/garden_20260626_004134/garden_vnext_certified_residual_texture_manifest.json` | provenance、commands、protocol audit、settings |
| `docs/car_model/vnext_artifacts/garden_20260626_004134/surface_residual_region_texture_adapter_audit.md` | 最重要的 certificate/fallback 人类可读报告 |
| `docs/car_model/vnext_artifacts/garden_20260626_004134/surface_residual_region_texture_adapter_audit.json` | 完整机器可读 audit，体积较大 |
| `docs/car_model/vnext_artifacts/garden_20260626_004134/garden_ours_26000_vnext_certified_residual_surface_texture_test_results.json` | final target split aggregate metrics |
| `docs/car_model/vnext_artifacts/garden_20260626_004134/garden_ours_26000_vnext_certified_residual_surface_texture_test_per_view.json` | final target split per-view metrics |
| `docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_summary.json` | garden 非零 accepted micro-gain 摘要 |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/counter_strict_face_softshrink_summary.json` | counter strict 单场景摘要 |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/counter_vnext_certified_residual_texture_manifest.json` | counter strict command/provenance/protocol audit |
| `docs/car_model/vnext_artifacts/counter_strict_face_softshrink_20260626_045300/target_evidence_no_gt_audit.json` | target evidence 去除 GT/residual keys 的审计 |
| `docs/car_model/vnext_artifacts/bonsai_strict_face_softshrink_20260626_052500/bonsai_strict_face_softshrink_summary.json` | bonsai strict 摘要 |
| `docs/car_model/vnext_artifacts/room_strict_face_softshrink_20260626_052500/room_strict_face_softshrink_summary.json` | room strict 摘要 |
| `docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/strict_frozen_policy_multiscene_summary.md` | strict frozen-policy 三场景聚合表 |
| `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md` | structure-aware shrink ready4 聚合表 |
| `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md` | full9 fixed-policy cleanup 聚合表 |
| `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/vnext_full9_cleanup_promotion_manifest.md` | full9 cleanup 轻量 artifact promotion manifest |

### 6.3 PPT 建议页序

1. **Motivation**：Phase-J 很强，但是 render-time teacher，不是 baked representation。
2. **vNext Goal**：Evidence-Certified Residual Surface Texturing。
3. **Method Diagram**：teacher cache -> residual surface texture -> capacity -> certificate -> fallback。
4. **Current Baselines**：Phase-J upper bound；v106 verified representation line。
5. **Protocol Integrity**：fit/policy-val/test split；`selection_uses_test_gt=false`；machine-readable manifests。
6. **Garden/Ready4 Pilot**：从 fallback/no-op 到 face-softshrink，再到 ready4 structure-aware shrink 4/4 nonzero accepted。
7. **Full9 Fixed Policy**：9/9 complete，9/9 no-target-GT protocol pass，6/9 accepted nonzero，3/9 fallback/no-op。
8. **Honest Gap**：full9 均值低于 clean/v106，mean changed fraction 只有 0.2756%，缺 full9 定性图片、完整 ablation 与预算表。

---

## 7. 主要短板

1. **full9 协议已闭环，但质量低于 baseline**
   full9 cleanup 为 `9 / 9` complete、`6 / 9` accepted nonzero、`3 / 9` fallback/no-op，mean changed fraction 为 `0.2756%`。但均值 `25.067699 / 0.741260 / 0.306689` 低于 clean MeshSplatting 和 v106，因此不能 promotion。

2. **SSIM / 结构一致性是当前最硬瓶颈**
   ready4/accepted scenes 中局部 PSNR/LPIPS 有微小正向，但全局 SSIM 与 LPIPS aggregate 仍不够。`flowers/stump/treehill` 被 lower-tail、min-view、SSIM/L1 风险拒绝为 fallback/no-op，说明当前 surface texture capacity 还不能稳定覆盖困难 outdoor/low-support view tail。

3. **surface-addressability 未证明足够强**
   Phase-J 收益可能部分来自 view-support / ELA portfolio 效应，不一定能直接 bake 到 face/UV texture。需要 residual consistency 和 face/bin transfer audit。

4. **parent choice 需要冻结**
   garden pilot 使用 Phase-F compact parent fallback，不能直接和 selected clean 或 v106 指标混讲。下一轮要明确 parent 是 clean、v106，还是 compact parent，并在表格中分开。

5. **teacher cache 不是完全 fresh vNext cache**
   garden run `skip_teacher_cache=True`，复用了既有 train evidence/cache。后续要跑 clean vNext teacher cache 或明确复用边界。

6. **full9 已完成，但 ablations 和同表比较未完成**
   prompt 要求的 clean、Phase-F、Phase-J teacher、v104c、v106、fixed texture、no-certificate、adaptive capacity 和 full vNext 同表 ablation 仍未完成。

7. **size / runtime / storage accounting 不完整**
   当前有 command elapsed time，但还缺系统化的 texture bytes、model storage、render overhead、triangle count、fallback rate 表。

8. **full9 cleanup 没有保留定性 payload**
   为了节省 `/dev/shm` 空间，本次 full9 run 使用了 scene cleanup。repo 内有定量、per-view、audit、selector 和 log 证据，但没有 full9 PNG/JPG/MP4 定性输出；已有 vNext 定性图只来自早期 garden face-softshrink pilot。

---

## 8. 下一步建议

### 8.1 以 full9 cleanup 作为协议 baseline，改做强表示升级

推荐顺序：

1. 固定当前 full9 cleanup 作为 protocol baseline，不再把 9/9 execution 当成待验证项；
2. 明确 parent row：至少包含 selected clean、v106 parent、Phase-F compact parent 三类；
3. 重建或明确标记 train-only Phase-J/ELA teacher cache；
4. 从 fixed-capacity texture 升级到 adaptive per-region capacity；
5. 加 no-certificate、no-structure、no-adaptive-capacity、fresh-teacher-cache ablation；
6. 只有同一固定协议超过 clean/v106，才 promotion。

### 8.2 针对 full9 失败模式的诊断

当前候选不是完全无信号，6/9 场景会产生非零 accepted residual output；但 aggregate 低于 clean/v106，说明收益过小且 view-tail 风险仍强。下一轮应优先诊断：

- 哪些 face/bin 贡献了 accepted gain，哪些 view/region 造成 tail loss；
- SSIM/LPIPS 失败是否来自局部高频错位、边界错改或 view-conditioned basis OOD；
- 是否需要 per-region/local alpha 和 adaptive texture capacity，而不是固定 texture size 16；
- `flowers/stump/treehill` 的 target footprint 是否本质低支持或 out-of-trajectory；
- Phase-J teacher residual 是否在这些区域真正 surface-consistent。

### 8.3 Promotion / reject 标准

最小可推广 milestone：

- full9 在一个冻结 train-only protocol 下完成；
- no test GT 用于 branch、alpha、capacity、fallback、threshold；
- 9/9 对 parent non-regressive 或 exact tie；
- 至少 6/9 strict scene RGB wins vs clean MeshSplatting；
- mean gain vs clean 至少 `+0.5` PSNR，并改善 LPIPS；
- 多个场景有 material nonzero accepted improvements，而不是仅有微小改动或 fallback-only；
- 报告 storage、runtime、texture budget、triangle count、fallback rate。

当前 full9 已经显示“只能得到不可见的微小改动或 fallback/no-op”这一瓶颈。后续若强 teacher/adaptive capacity 仍不能放大收益，应停止盲目重跑，改为给出 no-go diagnosis：

```text
teacher residual is not surface-addressable
or capacity overfits sparse evidence
or OOT / tail support is insufficient
or current parent/candidate mismatch is too large
```

---

## 9. 最终推荐

当前推荐状态：

```text
Keep Phase-J as teacher / upper bound.
Keep v106 as current verified representation baseline.
Continue vNext as staged representation project.
Do not promote vNext as completed or superior yet.
```

PPT 最后一页建议写：

> vNext 已经把“方法应该如何公平验证”搭起来了，并完成 full9 固定策略执行：9/9 场景、6/9 非零 accepted、3/9 fallback/no-op、无 target-GT selection/apply 泄漏。下一步的真正目标不是再证明 proof-of-life，而是用更强 teacher distillation 和 adaptive surface capacity 解决 SSIM/结构一致性瓶颈，把非零收益放大到可见、可复现、可超过 v106/clean baseline 的程度。

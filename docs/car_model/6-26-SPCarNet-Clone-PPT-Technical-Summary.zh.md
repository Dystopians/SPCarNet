# SPCarNet 克隆汇报总览与技术索引

日期：2026-06-26

用途：本文件是从远程仓库重新 clone 后给 mentor/PPT 汇报使用的最高层入口。它只总结当前已经落盘到仓库中的证据、报告和索引，不把本地 `/dev/shm` 中未完成的大模型产物当成已完成结果。

## 一句话现状

当前项目已经从最初的 MeshSplatting 后处理/调参尝试推进到一套可复现的 SPCarNet 方法梯队：有本地 clean MeshSplatting baseline、有 v106 representation-level 主线、有严格 split/gate 诊断、有 vNext no-test-GT residual surface texture 里程碑，也有可克隆的定量/定性报告包。

但它还不是 100% paper-final closed loop。最诚实的结论是：

```text
v106 POD-MoE base-preserve 是当前 verified representation-quality line；
它在 assembled selected full9 表上全面优于本地 clean MeshSplatting baseline。
vNext structure-aware shrink 是最新 strict no-target-GT surface-texture 里程碑，
目前完成 counter/bonsai/room/garden 四个 ready 场景，收益很小，还不能宣称超过 v106 或完成顶会终局。
stump 输入链已本地重建并把 preflight 推到 5/9 ready，但 strict stump run 被证书拒绝为 fallback/no-op。
```

## 当前最重要结果

### 1. 相对本地 clean MeshSplatting baseline

本地 full9 口径下，clean baseline 按 held-out test 指标在 clean `26000/30000` checkpoint 中选择，选择分数为：

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

训练集指标不用于选择 baseline 或最终 test 结果。

| method | scenes | PSNR | SSIM | LPIPS | delta vs clean |
|---|---:|---:|---:|---:|---|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 | baseline |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 | +0.677417 / +0.011709 / -0.019073 |
| v106 POD-MoE base-preserve | 9 | 25.831280 | 0.760830 | 0.268435 | +0.679598 / +0.011812 / -0.019185 |

这说明：在当前本地 assembled selected full9 表上，v106 相对 clean MeshSplatting 三个均值指标都更好。

同时必须说明：v106 相对 v104c 的增量很小，约为 `+0.002181 PSNR / +0.000103 SSIM / -0.000112 LPIPS`。因此它是“已验证的 representation-level 正向线”，不是“大幅突破”。

### 2. Phase-J broad RGB endpoint

Phase-J 是当前最强 broad RGB endpoint：

| 指标 | 结果 |
|---|---|
| full9 scene-level strict wins vs selected clean MeshSplatting | 9 / 9 |
| per-view RGB strict wins | 244 / 246 |
| mean delta vs selected clean | +1.3311 PSNR / +0.0347 SSIM / -0.0634 LPIPS |
| mean triangle reduction | 7.6479% |

但 Phase-J 是 render-time guarded ELA portfolio，不是完全 baked 到 MeshSplatting 表示里的最终模型。PPT 里可以把它讲成 teacher / upper bound / 当前最强工程端点，不能把它和 v106/vNext 的 representation-level claim 混为一谈。

### 3. 最新 vNext strict 里程碑

最新结构感知 shrink 已经把 vNext 从旧 face-softshrink 的 `2 / 3` nonzero accepted 推到 ready4 的 `4 / 4` nonzero accepted，并修复了 room 旧策略 fallback/no-op 的短板，同时补齐 garden 的 strict structure-aware shrink 结果。

固定策略：

```text
enable_policy_val_structure_aware_shrink=true
structure_shrink_l1_weight=1.0
structure_shrink_gradient_weight=1.0
structure_shrink_edge_weight=0.0
structure_shrink_risk_tau=0.002
strict_no_target_gt_apply=true
```

相对 Phase-F compact parent：

| scene | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| counter | true | 0.125 | 0.01234357 | +0.00129890 | -0.00000906 | -0.00004268 |
| bonsai | true | 0.25 | 0.00148974 | +0.00113869 | -0.00000954 | -0.00001693 |
| room | true | 0.0625 | 0.00519912 | +0.00046921 | +0.00000334 | -0.00001399 |
| garden | true | 0.125 | 0.00205038 | +0.00013924 | +0.00000316 | -0.00000791 |
| mean | 4 / 4 | - | - | +0.00076151 | -0.00000302 | -0.00002038 |

这是真实方法里程碑，因为：

- target apply 阶段通过 strict no-target-GT audit；
- `selection_uses_test_gt=false`；
- `target_gt_visible_to_apply=false`；
- `target_forbidden_keys_stripped=true`；
- `target_apply_leak=false`；
- room 从旧策略 fallback/no-op 变成 accepted nonzero，并且 room 自身三指标相对 Phase-F parent 都正向；
- garden 相对 Phase-F parent 和旧 garden face-softshrink pilot 都三指标小幅正向；
- manifest runner 和 preflight artifact 已记录 ready4 的 `4 / 9` 状态；本地 stump 重建后 preflight 已推进到 `5 / 9` ready。

但它还不能当作论文终局，因为：

- 只覆盖 `counter,bonsai,room,garden` 四个 ready 场景，不是 full9；
- 平均收益仍是 `1e-3 / 1e-5` 量级；
- counter/bonsai 仍有极小 SSIM 回退；
- 尚未同表完整比较 clean MeshSplatting、v104c、v106、Phase-J teacher 和 ablation；
- `bicycle,flowers,kitchen,treehill` 仍需重建 fit/target evidence 与 policy-val pruned carrier；
- stump 不是质量正例：strict run 完成但被 tail-risk certificate 拒绝为 fallback/no-op。

## 已经取得的主要进展

### 方法与代码层

- 建立了 v104c/v106 representation-level residual field 主线。
- v106 在 MeshSplatting parent 上加入 base-preserve POD-MoE detail / occlusion-boundary residual experts。
- 实现了 strict split 所需的 train/test delta-bank、field-builder view subset、render-realized parent gate、lower-tail/OOT/frame fallback。
- 实现了 vNext scene/full9 runner、manifest、no-test-GT audit、report assembler、W&B offline run 支持。
- 实现了 policy-val structure-aware shrink：用 policy-val 局部 RGB L1 worsening 和 luminance-gradient worsening 给 face/bin residual 做结构风险 downweight。
- 修复了 vNext parent-edge apply/profile 接口转发不一致问题，避免证书和最终 target apply 不一致。

### 实验与证据层

- clean MeshSplatting baseline 和 v106 selected full9 表已经落盘为 Markdown/JSON/CSV。
- v106 定性 contact sheets 已放入仓库，可直接用于 PPT。
- v110/v110b 暴露了 strict train-even -> train-odd -> test 泛化失败，是重要负结果。
- v113b/v113c 修复安全 fallback，但没有产生超过 v106 的质量突破。
- vNext 已完成 garden proof-of-life、counter/bonsai/room strict face-softshrink、garden structure-aware shrink、counter/bonsai/room/garden ready4 structure-aware shrink 聚合表、manifest runner、ready4 preflight、full9 gap preflight，以及 stump 输入链重建/strict fallback 负结果。
- 最新 vNext artifact 说明已经把“哪些可以汇报、哪些不能过度声称”写清楚。

### 文档与可克隆性

- 根索引、README/README.zh、报告 manifest、vNext artifact index 已维护。
- 关键定量结果、定性图、JSON/CSV、run manifest 和 protocol audit 均已复制为轻量仓库 artifact。
- 大模型 checkpoint、render 大树、W&B cache 仍留在本地 `/dev/shm` 或输出目录，不强行塞进 git。

## 汇报推荐结构

1. 问题：MeshSplatting 很强，但压缩/修复后容易出现局部纹理和视角泛化风险。
2. 方法：SPCarNet 把 mesh surface 当作地址空间，在三角形/face/bin 上挂 residual experts，并用 train evidence 证书控制何时应用。
3. 当前 verified 主线：v106 POD-MoE base-preserve，在本地 selected full9 表上优于 clean MeshSplatting baseline。
4. 严格性教训：v110/v113 证明 naive gate 不够，必须做 split、lower-tail、OOT 和 fallback。
5. 最新推进：vNext 把 Phase-J render-time residual teacher 变成 strict no-target-GT residual surface texture；structure-aware shrink 已在四个 ready 场景全部 nonzero accepted。
6. 诚实边界：vNext 还不是 full9/顶会终局；下一步要扩大到 full9、补预算/速度/三角形/定性 changed-region panel。

## 克隆后阅读顺序

| order | 用途 | 文件 |
|---:|---|---|
| 1 | 根入口 | `SPCARNET_REPORT_INDEX.md` |
| 2 | 本文件：PPT 总览 | `docs/car_model/6-26-SPCarNet-Clone-PPT-Technical-Summary.zh.md` |
| 3 | 最新状态附录 | `docs/car_model/6-26-SPCarNet-Current-Status-Upload-Report.md` |
| 4 | vNext manifest runner / full9 gap | `docs/car_model/6-26-vNext-ManifestRunner-and-Full9Gap-Log.md` |
| 5 | vNext ready4 聚合表 | `docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md` |
| 6 | vNext stump 重建/拒绝日志 | `docs/car_model/6-26-vNext-StumpInputRebuild-Ready5-and-Rejection-Log.md` |
| 7 | stump 后 full9 gap preflight | `docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md` |
| 8 | vNext ready4 preflight | `docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.md` |
| 9 | vNext full9 gap preflight | `docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md` |
| 10 | vNext 最新结构感知 shrink 日志 | `docs/car_model/6-26-vNext-StructureAwareShrink-Strict-Multiscene-Log.md` |
| 11 | vNext artifact 索引 | `docs/car_model/vnext_artifacts/README.md` |
| 12 | vNext 技术报告与旧 pilot 解释 | `docs/car_model/6-26-SPCarNet-vNext-Technical-Report-And-Index.zh.md` |
| 13 | 报告包 manifest | `docs/car_model/6-25-SPCarNet-Report-Package-Manifest.md` |
| 14 | v106 full9 对比表 | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md` |
| 15 | v106 assembled 表 | `docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_assembled.md` |
| 16 | v106 mentor 技术报告 | `docs/car_model/6-25-v106-PODMoE-Mentor-Technical-Report-Final.md` |
| 17 | 当前长版中文技术报告 | `docs/car_model/6-25-SPCarNet-Mentor-Technical-Report.md` |
| 18 | car-model 文档目录 | `docs/car_model/README.md` |

## PPT 可直接引用的 artifact

### 定量表

```text
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.md
docs/car_model/results/v106_podmoe_basepreserve_full9_20260625/full9_compare.json
docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.md
docs/car_model/vnext_artifacts/strict_structure_aware_shrink_ready4_20260626_071413/strict_structure_aware_shrink_ready4_summary.json
docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.md
docs/car_model/vnext_artifacts/full9_gap_after_stump_preflight_20260626/vnext_manifest_runner_summary.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/stump_ours_26000_vnext_structure_aware_shrink_test_results.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.json
docs/car_model/vnext_artifacts/strict_frozen_policy_multiscene_20260626_052500/strict_frozen_policy_multiscene_summary.md
```

### 定性图

```text
docs/car_model/assets/v106_qualitative/flowers_frame00001_bestcrop_contact_sheet.png
docs/car_model/assets/v106_qualitative/garden_frame00000_crop_contact_sheet.png
docs/car_model/assets/v106_qualitative/garden_frame00004_bestcrop_contact_sheet.png
docs/car_model/assets/v106_qualitative/room_frame00029_bestcrop_contact_sheet.png
docs/car_model/assets/v106_qualitative/treehill_frame00010_bestcrop_contact_sheet.png
docs/car_model/vnext_artifacts/garden_face_softshrink_20260626_040558/garden_face_softshrink_qualitative_panel.png
```

### 协议审计

```text
docs/car_model/vnext_artifacts/counter_structure_shrink_tau002_20260626_0558/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/bonsai_structure_shrink_tau002_20260626_0718/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/room_structure_shrink_tau002_20260626_0718/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/garden_structure_shrink_tau002_20260626_071413/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/target_evidence_no_gt_audit.json
docs/car_model/vnext_artifacts/counter_structure_shrink_tau002_20260626_0558/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/bonsai_structure_shrink_tau002_20260626_0718/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/room_structure_shrink_tau002_20260626_0718/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/garden_structure_shrink_tau002_20260626_071413/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/stump_structure_shrink_rebuild_tau002_20260626_080257/surface_residual_region_texture_adapter_audit.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_scene_config_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_scene_config_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.md
docs/car_model/vnext_artifacts/vnext_structure_shrink_ready4_preflight_20260626.json
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.md
docs/car_model/vnext_artifacts/vnext_structure_shrink_full9_gap_preflight_20260626.json
```

## 当前仍未完成的事项

- vNext structure-aware shrink 还没有 full9 固定策略验证；当前本地输入覆盖是 `5 / 9`，其中 stump 为 fallback/no-op 负结果。
- vNext 还没有完整同表比较 clean MeshSplatting、Phase-F parent、v104c、v106、Phase-J teacher、old face-softshrink 和 ablation。
- vNext 定性图目前更适合证明“有非零变化且可审计”，不适合宣称人眼强可见优势。
- v110/v111/v114 长程 strict jobs 仍有失败或未完成项，不能当成 final branch。
- `/data` 与 `/dev/shm` 压力会影响大实验稳定性；继续做 full9 前需要低内存 field-builder 或清理空间。

## 最终汇报口径

可以说：

```text
我们已经搭建出一个可复现的 SPCarNet 方法闭环雏形：
在本地 full9 上，v106 representation-level line 优于 clean MeshSplatting baseline；
同时，vNext 已经把 residual surface texture 的 no-test-GT 协议、证书、fallback、
结构感知 shrink 和四个 ready 场景非零 accepted 结果跑通，并已完成 stump 输入链重建与安全拒绝负结果。
```

不要说：

```text
不要说 vNext 已经全面超越 MeshSplatting/v106/Phase-J。
不要说所有 strict branch 长程实验已经结束。
不要说当前 ready4 tiny gain 已足够支撑顶会最终 claim。
不要把 stump fallback/no-op 计入 accepted quality win。
```

## Final Status

`NOT COMPLETE` for paper-final closed loop.

`READY FOR MENTOR/PPT ANALYSIS` for current cloneable status review.

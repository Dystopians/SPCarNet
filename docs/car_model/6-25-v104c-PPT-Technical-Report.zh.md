# SPCarNet 当前方法技术报告（用于 Mentor/PPT 汇报）

日期：2026-06-25

本文档面向组会/PPT 汇报，目标是用清晰、诚实、可防守的方式说明当前 SPCarNet 相比最基础 MeshSplatting baseline 做了什么、已经验证了什么、还差什么。

## 1. 一句话结论

最基础的 MeshSplatting 是“训练一个表面高斯/三角 mesh 表示，然后直接渲染测试视角”。SPCarNet 的核心变化是：不只相信 checkpoint 本身，而是在表面上附加一层由训练/支持视角导出的证据与残差修复机制，用它修正 MeshSplatting 在局部表面、遮挡边界、纹理细节上的系统性错误。

当前最强质量线仍然是 v101/v102a evidence endpoint：它已经在我们本地 full9 口径下相对 clean MeshSplatting 取得明显提升。v104c 是最新的 representation-field 方向：它尝试把 endpoint 的残差行为压缩成一个固定策略的表面残差场。最新 full9 固定策略验证已经完成，9/9 场景在 PSNR、SSIM、LPIPS 三项上都超过本地 clean MeshSplatting `ours_26000` baseline；但 v104c 还没有完全追上 v101/v102a endpoint ceiling。

适合 PPT 的主标题是：

> SPCarNet uses surface evidence to repair MeshSplatting, and v104c is our latest step toward baking the repair behavior into a compact view-conditioned surface field.

不建议把当前结论说成“v104c 已全面替代 endpoint”或“已经终局完成”。更准确的说法是：v104c 已经完成固定策略 full9 验证并稳定超过 clean baseline，但仍是把强 endpoint 烘焙进 surface-field representation 的中间阶段。

## 2. Clean MeshSplatting 与 SPCarNet 的区别

Clean MeshSplatting 的流程很直接：

```text
训练 MeshSplatting checkpoint
-> 用 checkpoint 直接渲染测试视角
-> 计算 PSNR / SSIM / LPIPS
```

它的问题是，checkpoint 本身没有显式记录“哪些表面可靠、哪些局部误差在多视角里反复出现、哪些区域应该修复、哪些区域应该保守不动”。如果某个三角面在训练中有局部颜色偏差、几何边缘误差或支持视角不稳定，clean 方法只能直接把这些误差渲染出来。

SPCarNet 的思路是把 mesh/surface 当成一个可寻址的证据载体：

```text
训练视角/支持视角
-> 投影到表面三角形
-> 收集残差、深度一致性、可见性、局部可靠性
-> 在测试视角渲染时只应用被证据支持的修复
```

通俗讲，clean MeshSplatting 是“只看模型自己怎么画”；SPCarNet 是“先看模型在哪些表面经常画错，再把这些可重复的错误修正烘焙/传递到测试渲染里”。

## 3. 当前方法模块

### 3.1 v101 Evidence Bank

v101 是当前最强质量 endpoint 的基础模块。

它做的事情是：

- 从训练/支持视角收集 residual、depth、camera、hash 等证据；
- 把证据打包成 `v101_evidence_bank.pt`；
- 在 `render.py` 中通过 endpoint hook 使用这些证据；
- 支持 `--checkpoint_endpoint_require_bank`，保证缺失证据时失败，而不是静默退化成不公平或不可复现的路径；
- 通过 detached package 验证，证明离线包可以复现 bank-backed endpoint 输出。

它的意义是：证明“表面证据 + 受控残差修复”确实能显著提升 MeshSplatting，而不是只靠调参。

局限是：它仍是一个特殊 render endpoint，不是普通 MeshSplatting checkpoint；它也不比 clean render 更快。

### 3.2 v102 Preprojected Delta

v102 是对 v101 endpoint 的加速和固化。

它做的事情是：

```text
离线阶段：运行一次 v101 endpoint，保存 adapted_render - base_render 的 target-camera delta
在线阶段：渲染 clean base，再直接加上预投影 delta
```

这可以把复杂的在线证据投影和 gating 变成一个 target-camera delta bank。它在 hard triad 上与 v101 endpoint 数值等价，是当前高质量结果的 ceiling/reference。

局限是：v102 不是 unseen-camera 泛化方法。它更像“已验证 target-camera set 上的 endpoint 加速缓存”。

### 3.3 v103 Surface Affine Field

v103 开始把 endpoint 修复行为压缩进表面残差场：

```text
[1, barycentric_u, barycentric_v] -> RGB residual
```

它说明“每个三角形上的局部残差不是完全随机的”，可以用低阶表面函数拟合并超过 clean baseline。

局限是：它不看视角方向，因此无法表达 view-dependent residual。

### 3.4 v104a View-Affine Field

v104a 在 v103 的基础上加入 view direction：

```text
[1, u, v, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

它说明视角方向确实对 residual 修复有帮助。hard triad 上 v104a 明显强于 v103。

局限是：直接拟合 view-affine 容易受少视角、病态三角形、support 不稳定影响。

### 3.5 v104c Shrink View-Affine Field

v104c 是当前 representation-field 线的最好版本。它不是手动逐场景调参，而是固定策略：

- 对 view direction 特征做中心化和尺度归一化；
- 训练/构建时用 ridge 稳定求解；
- 根据 rank、view support、condition number 等诊断得到一个代数置信度；
- 不把不稳定三角形硬丢弃，而是把 view-affine 系数向更保守的 v103 affine fallback 收缩；
- 渲染时仍然使用同一个 surface field 接口。

它解决的是 v104a 的“表达能力更强但可能过拟合/病态”的问题。hard triad 上 v104c 相比 v104a 虽然提升不大，但三项指标一致变好，说明 shrink 策略是有效的稳定化改动。

## 4. 当前定量结果

### 4.1 Hard Triad 平均结果

Hard triad 包含：`counter`、`kitchen`、`bonsai`。

| 方法 | PSNR | SSIM | LPIPS | 解释 |
|---|---:|---:|---:|---|
| clean MeshSplatting | 27.821853 | 0.878303 | 0.236894 | 最基础 baseline |
| v103 affine field | 28.384418 | 0.879855 | 0.226611 | 表面残差场首次稳定超过 clean |
| v104a view-affine field | 28.823045 | 0.884927 | 0.219492 | 视角方向有效 |
| v104c shrink view-affine field | 28.859798 | 0.885459 | 0.219064 | 当前最强 surface-field 版本 |
| v101/v102a endpoint ceiling | 30.167397 | 0.913355 | 0.163709 | 当前最强质量参考 |

### 4.2 v104c 的平均提升

| 对比 | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c - clean | +1.037945 | +0.007156 | -0.017830 |
| v104c - v103 | +0.475380 | +0.005604 | -0.007547 |
| v104c - v104a | +0.036753 | +0.000532 | -0.000427 |
| v104c - v101/v102a ceiling | -1.307599 | -0.027896 | +0.055355 |

结论：

- v104c 确实比 clean MeshSplatting 强；
- v104c 也确实比早期 surface-field 方法更强；
- 但 v104c 还没有追上 v101/v102a endpoint ceiling，因此不能把 v104c 说成终局方法。

### 4.3 Full9 固定策略结果

Full9 自动聚合脚本：

```text
scripts/car_model/summarize_v104c_shrink_view_affine_full9.py
```

当前输出路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.csv
```

当前自动 summary 显示：`present_scenes=9`，`ok_scenes=9`，`all_present=True`，`all_ok=True`。9 个场景都用同一个固定 v104c policy，不是逐场景调参。

| 方法 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting (`ours_26000`) | 25.151682 | 0.749018 | 0.287621 |
| v104c shrink view-affine field | 25.829099 | 0.760727 | 0.268548 |
| v101/v102a endpoint/reference | 26.481310 | 0.783675 | 0.224305 |

| 对比 | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c - clean | +0.677417 | +0.011709 | -0.019073 |
| v104c - endpoint/reference | -0.652211 | -0.022949 | +0.044243 |

逐场景结果：

| scene | clean PSNR | v104c PSNR | endpoint PSNR | dPSNR clean | dSSIM clean | dLPIPS clean |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 23.301613 | 23.717649 | 24.021442 | +0.416037 | +0.015104 | -0.018574 |
| bonsai | 28.895233 | 30.310877 | 31.861889 | +1.415644 | +0.010966 | -0.029307 |
| counter | 26.751774 | 27.498068 | 28.442907 | +0.746294 | +0.005364 | -0.013017 |
| flowers | 19.682257 | 20.075844 | 20.300581 | +0.393587 | +0.019255 | -0.020090 |
| garden | 25.029211 | 25.788094 | 26.310476 | +0.758883 | +0.019228 | -0.026730 |
| kitchen | 27.818552 | 28.770449 | 30.197395 | +0.951897 | +0.005138 | -0.011165 |
| room | 28.747276 | 29.597836 | 30.305668 | +0.850559 | +0.006994 | -0.019239 |
| stump | 25.205042 | 25.459311 | 25.595201 | +0.254269 | +0.009434 | -0.011791 |
| treehill | 20.934181 | 21.243763 | 21.296227 | +0.309582 | +0.013896 | -0.021746 |

口径说明：上述 clean baseline 是本地 `official_clean30k/*/results.json` 中的 selected clean `ours_26000`，不是直接引用论文表格，也不是 train 指标。v104c 和 clean 使用同场景、同 held-out test split。v104c field 构建基于 v102 target-camera delta/reference 做 distillation，因此它不是 train-only unseen-camera 泛化结论，也不是 vanilla MeshSplatting checkpoint。

## 5. 定性展示建议

当前方法的视觉优势不应该只用整张 RGB 对比图展示，因为很多提升在 full-frame 上对人眼比较细微。更合适的展示方式是：

1. clean / SPCarNet / GT 三列并排；
2. 同时展示 crop-level 放大区域；
3. 再加 absolute error heatmap；
4. 标注每个 crop 对应的 scene、view、metric delta；
5. 区分 v101 endpoint 的视觉证据和 v104c surface-field 的视觉证据。

已有较强的 v101 可视化资产：

```text
assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png
assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json
```

v104c 的 full9 定性展示现在可以从 9/9 per-scene renders 中挑选真实提升最明显的 view/crop。渲染路径模板：

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/renders/*.png
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/gt/*.png
```

展示时应优先选 crop/error-map，而不是只放整图 RGB；否则很多提升对人眼并不明显。

## 6. 为什么这是研究工作，而不只是工程 patch

工程 patch 通常是“加一个后处理、调几个阈值、换一个 checkpoint”。SPCarNet 当前线索的研究点在于：

- 把 mesh/surface 从单纯渲染几何变成证据寻址空间；
- 将多视角 residual、visibility、depth consistency 与 surface triangle 绑定；
- 通过 evidence bank 和 fail-closed package 机制保证修复来源可追踪；
- 进一步尝试把 endpoint 行为蒸馏为 surface residual field；
- 通过 v103 -> v104a -> v104c 的 ablation 证明 barycentric、view direction、shrink reliability 各自的作用。

这条线的科学问题是：

> 如何把强但复杂的 per-view evidence endpoint，压缩成一个可部署、可泛化、可解释的表面表示？

当前 v104c 给出了一个正向但不完整的答案：低阶 view-affine field 可以稳定超过 clean，但单模态低阶 field 仍然损失了 endpoint 的多条件决策能力。

## 7. 目前最重要的短板

最大短板不是“还没调好参数”，而是表示能力不足。

v101/v102a endpoint 可以基于每个 target view 的支持证据、可见性、局部 trust 和 fallback 决策来修复。v104c 把这些压缩成“每个三角形一个低阶 view-affine 函数”。这种压缩会丢掉：

- 多模态 residual；
- 复杂遮挡/边界上下文；
- per-view policy fallback；
- support agreement 与风险估计；
- 局部纹理细节。

因此后续真正有研究价值的改进应该是：

- per-triangle residual mixture，而不是一个单函数；
- evidence-gated field，在渲染时输出 residual 和 trust；
- 用 v102a endpoint 与 v104c field 的差距构造 calibrated teacher blend，再蒸馏回 surface field；
- 从 same-target-camera delta distillation 逐步迁移到 train/policy-val-only 证据构建。

## 8. 可安全汇报与不可夸大

可以安全汇报：

- SPCarNet 的 evidence endpoint 在我们本地 full9 口径下相对 clean MeshSplatting 有明显提升；
- v101/v102a 是当前最强质量参考；
- v104c 是当前最好的 surface-field/representation-baking 版本；
- v104c full9 固定策略已经 9/9 完成，并且相对 clean `ours_26000` 在 PSNR、SSIM、LPIPS 上逐场景全部同向改善；
- hard triad 上 v104c 全指标超过 clean、v103、v104a；
- v104c 的 fixed shrink policy 是真实方法改动，不是逐场景调参。

不应夸大：

- 不能说 v104c 已全面替代 v101/v102a；
- 不能说 v104c 已经追平 endpoint/reference；
- 不能说当前 field 是 unseen-camera train-only 泛化；
- 不能说所有 full-frame RGB 图都能肉眼明显看出提升；
- 不能说 endpoint/field 是普通 vanilla MeshSplatting checkpoint，无需特殊 render 逻辑。

## 9. 建议 PPT 结构

1. Motivation：MeshSplatting 的局部错误与 surface evidence 缺失。
2. Baseline：clean MeshSplatting 直接渲染。
3. SPCarNet overview：surface-addressed evidence repair。
4. v101/v102 endpoint：强质量结果和 full9 证据。
5. v103/v104/v104c representation field：从 endpoint 到可烘焙表示。
6. Quantitative results：hard triad + full9 自动表格。
7. Qualitative results：crop + error map，避免只放整图。
8. Ablation：clean, v103, v104a, v104b, v104c, endpoint ceiling。
9. Limitations：endpoint gap、same-target-camera distillation、field 表达力不足。
10. Next step：evidence-gated mixture field。

## 10. 关键路径

代码与结果：

```text
scripts/car_model/build_v104b_centered_view_affine_residual_field.py
scripts/car_model/run_v104c_shrink_view_affine_scene.py
scripts/car_model/summarize_v104c_shrink_view_affine_full9.py

outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_hardtriad_20260625/v104c_hardtriad_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.md
```

单场景复现命令模板：

```bash
SCENE=<scene>
GPU=<gpu>

CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 WANDB_MODE=offline \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/run_v104c_shrink_view_affine_scene.py \
  --scene ${SCENE} \
  --gpu ${GPU} \
  --package_root /dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625 \
  --v102_bank_root /dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625 \
  --field_root /dev/shm/peilincai_spcarnet_v104c_shrink_view_affine_field_20260625 \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625 \
  --v102_report_root outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625 \
  --clean_root outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
  --endpoint_method ours_26000_v100_checkpoint_attached_ela_endpoint \
  --iteration 26000 \
  --renderer_scaling 4 \
  --residual_dtype float16 \
  --ridge 1e-3 \
  --residual_clip 0.08 \
  --view_std_floor 1e-4 \
  --rank_rtol 1e-7 \
  --condition_max 1e8 \
  --chunk_pixels 262144 \
  --build_v102_if_missing
```

Full9 汇总命令：

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/summarize_v104c_shrink_view_affine_full9.py \
  --root outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625 \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625
```

英文详细报告：

```text
docs/car_model/6-25-v104c-Full9-PaperLoop-Technical-Report.md
```

本文档：

```text
docs/car_model/6-25-v104c-PPT-Technical-Report.zh.md
docs/car_model/6-25-v104c-Subagent-Synthesis.zh.md
```

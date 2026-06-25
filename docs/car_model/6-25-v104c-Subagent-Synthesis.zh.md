# SPCarNet v104c Subagent Synthesis

日期：2026-06-25

用途：给 mentor / PPT 的补充报告。本文只综合当前已经落盘的 v101-v104c 证据，不把 v104c 夸大成最终方法。

## 1. 一句话结论

SPCarNet 相对 clean MeshSplatting 的核心变化是：不只直接渲染 checkpoint，而是把 MeshSplatting 的表面当成证据寻址空间，从训练/支持视角收集 residual、depth、visibility、局部可靠性，再把可信残差传递到测试视角。当前有两条线：

- v101/v102a endpoint/reference：质量最强，full9 上相对 clean 明显提升，但仍依赖特殊 endpoint / delta bank。
- v104c shrink view-affine field：最新 representation-field 方向，尝试把 endpoint 残差行为压缩成固定策略的表面残差场。它在 full9 9/9 场景上超过 clean，但仍低于 endpoint/reference。

适合 PPT 的保守标题：

> SPCarNet uses surface evidence to repair MeshSplatting; v104c is the current fixed-policy step toward baking that repair into a compact view-conditioned surface field.

## 2. 方法模块解释

### 2.1 Clean MeshSplatting baseline

Clean MeshSplatting 的流程是：

```text
训练 MeshSplatting checkpoint
-> 直接渲染 held-out/test views
-> 计算 PSNR / SSIM / LPIPS
```

它没有显式记录哪些三角面可靠、哪些残差跨视角重复出现、哪些区域应该回退不修。因此局部纹理偏差、遮挡边界错误、少视角区域误差会直接进入测试渲染。

### 2.2 v101 Evidence Bank endpoint

v101 是当前质量最强线的部署化 endpoint。它把训练/支持视角中的 residual、depth、camera、hash 等证据打包成 evidence bank，并通过 `render.py` endpoint hook 使用；`--checkpoint_endpoint_require_bank` 让缺 bank 时 fail-closed，避免静默读取不该读的证据。

它证明了“surface evidence + guarded residual repair”确实能提升 MeshSplatting，而不是只靠随机调参。它的边界也很清楚：不是 vanilla MeshSplatting checkpoint，运行时仍需要 endpoint 逻辑，也没有速度优势。

### 2.3 v102 Preprojected Delta

v102 把 v101 endpoint 输出预投影成 target-camera delta：

```text
offline: adapted_render - base_render -> delta bank
online: clean base render + stored delta
```

它是 v101 endpoint 在已验证 target-camera set 上的加速/缓存形式，也是当前 surface-field 线需要追赶的 endpoint/reference ceiling。它不是 unseen-camera 泛化方法。

### 2.4 v103/v104a/v104c Surface Residual Field

v103 开始把 endpoint 残差压缩到表面上：

```text
[1, barycentric_u, barycentric_v] -> RGB residual
```

v104a 加入 view direction：

```text
[1, u, v, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

v104c 在 v104a 上加稳定化：构建时对 view direction 做中心化/尺度归一化，用 ridge 求解，并根据 rank、view support、condition number 得到代数置信度，把不稳定的 view-affine 系数向 v103 affine fallback 收缩。渲染时仍使用同一个 surface-field endpoint payload。

直观解释：v101/v102a 是“强但复杂的 per-view evidence endpoint”；v104c 是“把这个修复行为压到每个三角面上的低阶 view-conditioned 函数”。这个压缩带来可解释性和 representation 方向的价值，也带来明显质量损失。

## 3. Full9 定量结果

结果来源：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.csv
```

当前聚合状态：`present_scenes=9`，`ok_scenes=9`，`all_present=True`，`all_ok=True`。`room` 当前已经完成，不是 pending。若后续复跑导致 `room/room_v104c_shrink_view_affine_report.json` 缺失，PPT 表格应把 `room` 单独标成 `pending / missing_report_json`。

### 3.1 Full9 mean

| 方法 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 25.151682 | 0.749018 | 0.287621 |
| v104c shrink view-affine field | 25.829099 | 0.760727 | 0.268548 |
| endpoint/reference v101/v102a | 26.481310 | 0.783675 | 0.224305 |

| 对比 | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c - clean | +0.677417 | +0.011709 | -0.019073 |
| v104c - endpoint/reference | -0.652211 | -0.022949 | +0.044243 |

### 3.2 Full9 per-scene compact table

LPIPS 越低越好；`dLPIPS` 为负表示优于 clean，为正表示差于 endpoint/reference。

| scene | status | clean PSNR | v104c PSNR | v104c SSIM | v104c LPIPS | endpoint PSNR | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR endpoint | dSSIM endpoint | dLPIPS endpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | ok | 23.301613 | 23.717649 | 0.674972 | 0.313503 | 24.021442 | +0.416037 | +0.015104 | -0.018574 | -0.303793 | -0.027380 | +0.047400 |
| bonsai | ok | 28.895233 | 30.310877 | 0.907367 | 0.230186 | 31.861889 | +1.415644 | +0.010966 | -0.029307 | -1.551012 | -0.022909 | +0.057620 |
| counter | ok | 26.751774 | 27.498068 | 0.867420 | 0.238986 | 28.442907 | +0.746294 | +0.005364 | -0.013017 | -0.944839 | -0.026276 | +0.052429 |
| flowers | ok | 19.682257 | 20.075844 | 0.531076 | 0.374473 | 20.300581 | +0.393587 | +0.019255 | -0.020090 | -0.224737 | -0.026380 | +0.044960 |
| garden | ok | 25.029211 | 25.788094 | 0.799263 | 0.174584 | 26.310476 | +0.758883 | +0.019228 | -0.026730 | -0.522383 | -0.028567 | +0.038721 |
| kitchen | ok | 27.818552 | 28.770449 | 0.881590 | 0.188021 | 30.197395 | +0.951897 | +0.005138 | -0.011165 | -1.426947 | -0.034503 | +0.056017 |
| room | ok | 28.747276 | 29.597836 | 0.891837 | 0.230664 | 30.305668 | +0.850559 | +0.006994 | -0.019239 | -0.707832 | -0.013850 | +0.034774 |
| stump | ok | 25.205042 | 25.459311 | 0.714599 | 0.282213 | 25.595201 | +0.254269 | +0.009434 | -0.011791 | -0.135891 | -0.009483 | +0.018289 |
| treehill | ok | 20.934181 | 21.243763 | 0.578418 | 0.384298 | 21.296227 | +0.309582 | +0.013896 | -0.021746 | -0.052464 | -0.017188 | +0.047976 |

## 4. 定性输出路径

当前最稳的 endpoint qualitative panel：

```text
assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png
assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json
```

v104c full9 渲染输出路径模板：

```text
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/renders/*.png
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/gt/*.png
/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625/<scene>/detached_model/test/ours_26000_v104c_shrink_view_affine_min1_minviews1_<scene>/render_py_endpoint_report.json
```

v104c per-scene 报告路径模板：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/<scene>/<scene>_v104c_shrink_view_affine_report.md
outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/<scene>/<scene>_v104c_shrink_view_affine_report.json
```

PPT 建议：不要只放整图 RGB。优先用 clean / v104c / GT 三列，加 crop 和 absolute-error heatmap；caption 写 scene、view id、crop 来源、metric delta。v101 panel 可以用来说明 endpoint 质量上限，v104c 图要明确是 surface-field 线，不要混成同一结论。

## 5. 相对 clean MeshSplatting 的结论

v104c full9 9/9 场景全部 ok，平均相对 clean 提升：

- PSNR: `+0.677417`
- SSIM: `+0.011709`
- LPIPS: `-0.019073`

逐场景看，v104c 对 clean 的 PSNR、SSIM、LPIPS 三项都是同向改善；最明显的 PSNR 提升出现在 `bonsai`、`kitchen`、`room`，较小但仍正向的提升出现在 `stump`、`treehill`。因此可以安全说：v104c 是一个真实的、固定策略的 representation-field 改进，已经不只是 hard-triad 局部现象。

## 6. 相对 endpoint/reference 的诚实短板

v104c 仍没有追上 v101/v102a endpoint/reference。Full9 平均差距：

- PSNR: `-0.652211`
- SSIM: `-0.022949`
- LPIPS: `+0.044243`

短板的本质不是“参数没扫够”，而是表示压缩损失：

- endpoint/reference 可以按 target view 使用支持证据、可见性、局部 trust、fallback；
- v104c 把这些决策压成每个三角面一个低阶 view-affine 函数；
- 这种函数难以表达多模态 residual、遮挡边界上下文、per-view 置信度、复杂纹理细节；
- v102a 还是 target-camera delta/reference，不应把 v104c 的当前结果说成已经达到 endpoint ceiling；
- v104c 仍通过 `render.py` 的 surface-field endpoint 使用，不是无需特殊逻辑的 vanilla MeshSplatting checkpoint。

PPT 上应该直接展示 endpoint/reference 行。这样结论更可信：SPCarNet 的 evidence endpoint 已证明强收益，v104c 是把强 endpoint 烘焙成 representation 的中间步骤，而不是最终替代品。

## 7. 下一步任务清单

1. 生成 v104c full9 qualitative panel：从当前 9/9 per-scene renders 中挑选有真实 crop/error-map 改善的 view，保留 manifest。
2. 做 v104c vs endpoint gap audit：按 scene/view 找出 LPIPS 和 error-map 差距最大的区域，判断是遮挡、少支持、纹理高频还是 view-dependent residual。
3. 尝试 per-triangle residual mixture：从单一 view-affine 函数升级到 2-component mixture 或 trust-gated residual。
4. 引入 evidence-gated field：field 同时输出 residual 和 trust，让 render 时可以保守回退，而不是只靠构建阶段 shrink。
5. 用 v102a endpoint 与 v104c 的差距构造 calibrated teacher blend，再蒸馏回 surface field。
6. 明确论文 claim boundary：full9 可以报 v104c > clean；endpoint/reference gap 必须保留；不要声称 vanilla checkpoint、unseen-camera 泛化或 endpoint 已被完全替代。
7. 整理可复现实验索引：summary CSV/JSON、per-scene report、field manifest、render report、qualitative manifest 放进 PPT 备注或 appendix。

## 8. 可直接放进 PPT 的 takeaways

- Clean MeshSplatting 只直接渲染 checkpoint；SPCarNet 用 surface evidence 修复局部可重复错误。
- v101/v102a endpoint 是当前质量上限；v104c 是把 endpoint 行为压缩进 surface-field representation 的最新版本。
- v104c full9 9/9 完成，平均相对 clean: `+0.677 PSNR / +0.0117 SSIM / -0.0191 LPIPS`。
- v104c 仍落后 endpoint/reference: `-0.652 PSNR / -0.0229 SSIM / +0.0442 LPIPS`。
- 最诚实的研究问题是：如何把强但复杂的 per-view evidence endpoint，压缩成可部署、可泛化、可解释的表面表示。

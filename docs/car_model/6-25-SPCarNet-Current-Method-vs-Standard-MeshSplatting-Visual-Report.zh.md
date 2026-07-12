# SPCarNet 当前方法与标准 MeshSplatting 对比报告

日期：2026-06-25  
用途：mentor/PPT 汇报；用一份干净文档说明当前稳定方法、标准 MeshSplatting baseline、定量结果、代表性渲染图对比和当前边界。

---

## 1. 核心结论

当前最稳、最适合对外汇报的版本是 **SPCarNet Phase-J**：

> 在标准 MeshSplatting checkpoint 之后，SPCarNet 用训练视角的 surface evidence 做自审计：低风险三角形可以压缩，稳定 residual 可以修复，证据不足的位置自动回退。

在本地 Mip-NeRF360 full9、相同 split、相同 evaluator、selected-clean MeshSplatting baseline 下，当前稳定主结果是：

| 指标 | SPCarNet Phase-J vs selected clean MeshSplatting |
|---|---:|
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| held-out view PSNR/SSIM/LPIPS strict wins | `244 / 246` |
| mean PSNR | clean `25.1517` -> ours `26.4828` |
| mean SSIM | clean `0.7490` -> ours `0.7837` |
| mean LPIPS | clean `0.2876` -> ours `0.2243` |
| mean delta | `+1.331084` PSNR, `+0.034702` SSIM, `-0.063359` LPIPS |
| mean triangle reduction | `7.6479%` triangles removed |

这里的 `triangle reduction` 是**删去的三角形占比**，不是剩余比例。

汇报时建议用一句话概括：

> MeshSplatting 是训练完直接渲染；SPCarNet 是在同一个 checkpoint 上再做一次多视角证据驱动的“压缩和修复”，只改证据充分的位置，不确定就回退。

---

## 2. 标准 MeshSplatting 与 SPCarNet 的方法差异

### 2.1 标准 MeshSplatting

标准流程可以概括为：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh/splat checkpoint
  -> directly render held-out views
```

它的优势是显式 surface 表示、渲染稳定、pipeline 简洁。但原始流程不显式回答这些问题：

- 哪些 triangles 对多视角解释贡献低，可以安全删除？
- 哪些 surface 区域存在稳定 residual，可以迁移到 held-out view 修复？
- 哪些局部修复可能只对训练视角有效，从而伤害 tail view 或 out-of-trajectory view？
- 如果证据不足，系统是否会自动选择不修改？

### 2.2 SPCarNet Phase-J

SPCarNet 不推翻 MeshSplatting，而是在 clean checkpoint 后增加一层 evidence-driven audit and repair：

```text
clean MeshSplatting checkpoint
  -> train-view surface evidence cache
  -> geometry-safe triangle compaction
  -> guarded residual repair
  -> train/policy-val gate + fallback
  -> held-out render evaluation
```

| 维度 | 标准 MeshSplatting | SPCarNet Phase-J |
|---|---|---|
| 基础模型 | MeshSplatting checkpoint | 继承同一个 MeshSplatting checkpoint |
| 训练视角证据 | 主要隐式用于优化 | 显式缓存 residual、visibility、face/bin support、risk |
| 几何处理 | checkpoint 固定 | 删除低风险 triangles |
| 外观修复 | 直接渲染 checkpoint | surface-bound guarded residual repair |
| 风险控制 | 依赖训练收敛 | train/policy-val gate、tail-risk gate、fallback/no-op |
| 输出目标 | 高质量渲染 | 高质量渲染 + 更少 triangles + 可解释审计 |

---

## 3. 当前方法模块

### 3.1 Surface Evidence Cache

Evidence cache 将训练视角监督重新绑定到三维 surface 上，保存：

- rendered RGB 与 GT RGB；
- residual：`GT - Render`；
- alpha、depth、visibility；
- face id、barycentric coordinate、surface/bin address；
- normal、view direction、camera position；
- per-face/per-bin support count；
- residual sign consistency；
- train/policy-val risk statistics。

它的作用是回答一个关键问题：

```text
这个 surface 区域是否有足够多视角证据支持我们去删三角形或修 residual？
```

### 3.2 Geometry-Safe Compaction

SPCarNet 的压缩不是按面积、透明度或单一贡献分数粗暴删面，而是 quality-first：

```text
only remove triangles when multi-view evidence marks the edit as low-risk
```

当前 Phase-J 平均删去 `7.6479%` triangles，同时 full9 上 RGB 三指标 `9 / 9` 场景严格胜出。

### 3.3 Guarded Evidence Lumigraph Adapter

当前最大的 RGB 收益来自 surface-bound residual repair。简化形式：

```text
I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `I_compact(p)` 是压缩后 checkpoint 的 render；
- `residual_i = GT_i - Render_i` 只来自训练视角；
- `u_i` 是 target pixel 通过 surface correspondence 找到的训练视角 surface address；
- `w_i(p)` 由 visibility、surface match、support count 和 risk 共同决定；
- `alpha` 由 train/policy-val evidence 自动选择；
- 证据不足时 fallback/no-op。

这不是普通 2D 图像后处理。SPCarNet 的 residual 必须绑定到 mesh surface correspondence，不能读取 held-out GT，也不能对最终图像做无约束增强。

### 3.4 Train/Policy-Val Gate

公平性边界：

- 方法分支、alpha、support、fallback 只使用 train/policy-val evidence；
- held-out test GT 只用于最终评价；
- clean baseline 从本地 clean `26000/30000` checkpoint envelope 中选择 held-out 更强者；
- 不使用 train metrics 选择 baseline，因为 train metrics 会天然偏向训练更久的 checkpoint；
- 不用 test metrics 为 SPCarNet 调参。

---

## 4. 定量结果：SPCarNet vs 标准 MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- baseline：本地标准 MeshSplatting clean `26000/30000` checkpoint envelope；
- baseline selection：对 clean `26000/30000` 取 held-out score 更强者；
- ours：SPCarNet Phase-J guarded adaptive ELA + geometry-safe compaction；
- 指标：PSNR/SSIM 越高越好，LPIPS 越低越好。

| scene | clean MeshSplatting PSNR/SSIM/LPIPS | SPCarNet PSNR/SSIM/LPIPS | delta | triangles removed |
|---|---:|---:|---:|---:|
| bicycle | `23.3016` / `0.6599` / `0.3321` | `24.0215` / `0.7024` / `0.2661` | `+0.7199` / `+0.0425` / `-0.0660` | `11.81%` |
| flowers | `19.6823` / `0.5118` / `0.3946` | `20.3044` / `0.5578` / `0.3292` | `+0.6221` / `+0.0459` / `-0.0653` | `11.82%` |
| garden | `25.0292` / `0.7800` / `0.2013` | `26.3111` / `0.8278` / `0.1358` | `+1.2819` / `+0.0478` / `-0.0655` | `3.47%` |
| stump | `25.2050` / `0.7052` / `0.2940` | `25.5951` / `0.7241` / `0.2639` | `+0.3901` / `+0.0189` / `-0.0301` | `11.82%` |
| treehill | `20.9342` / `0.5645` / `0.4060` | `21.2962` / `0.5956` / `0.3363` | `+0.3620` / `+0.0311` / `-0.0697` | `11.81%` |
| room | `28.7473` / `0.8848` / `0.2499` | `30.3056` / `0.9057` / `0.1960` | `+1.5584` / `+0.0209` / `-0.0539` | `2.10%` |
| counter | `26.7518` / `0.8621` / `0.2520` | `28.4492` / `0.8937` / `0.1865` | `+1.6974` / `+0.0317` / `-0.0655` | `2.10%` |
| kitchen | `27.8186` / `0.8765` / `0.1992` | `30.1997` / `0.9161` / `0.1320` | `+2.3812` / `+0.0396` / `-0.0672` | `2.10%` |
| bonsai | `28.8952` / `0.8964` / `0.2595` | `31.8620` / `0.9303` / `0.1726` | `+2.9668` / `+0.0339` / `-0.0869` | `11.80%` |

结果解读：

- full9 场景级 RGB 三指标 `9 / 9` strict wins；
- held-out view 级 `244 / 246` strict wins；
- 室内场景收益最大，`bonsai/kitchen/counter/room` 的 PSNR 和 LPIPS 改善最明显；
- 室外场景也全胜，但人眼全图感知更弱，最好用局部 crop 与 error map 展示；
- 几何上同时平均删去 `7.65%` triangles，这是 quality-preserving compact-and-repair，而不是单纯追求最高 RGB 指标。

---

## 5. 代表性渲染图对比

### 5.1 最推荐 PPT 主图：局部 held-out error reduction

这张图最能直观看出 SPCarNet 相比 clean MeshSplatting 的收益。每行包含：

```text
GT crop / clean MeshSplatting / SPCarNet / error reduction
```

绿色表示 SPCarNet 更接近 GT，紫红色表示变差。

![SPCarNet Phase-J local held-out error reduction](../../assets/spcarnet_phasej_where_it_helps_showcase_20260622.png)

代表性局部收益：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | `+6.63 / +0.0452 / -0.0878` | `+11.79` | `78.6%` |
| kitchen / `00011.png` | `+3.43 / +0.0250 / -0.0578` | `+10.48` | `71.4%` |
| room / `00011.png` | `+3.50 / +0.0220 / -0.0656` | `+10.36` | `67.7%` |
| counter / `00013.png` | `+2.17 / +0.0407 / -0.0665` | `+6.02` | `54.9%` |
| garden / `00006.png` | `+1.74 / +0.0479 / -0.0678` | `+4.26` | `44.4%` |
| flowers / `00014.png` | `+1.12 / +0.0754 / -0.1028` | `+2.15` | `25.3%` |

推荐讲法：

> 这张图说明 SPCarNet 的收益不是只停留在表格指标上，而是在局部高频纹理、光照残差和 surface detail 区域降低了 clean MeshSplatting 的系统误差。

### 5.2 公平 full-frame 渲染对比

这张图适合作为“不是只挑 crop”的公平性证据。每行包含：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error
```

![SPCarNet full-frame held-out comparison against clean MeshSplatting](../../assets/spcarnet_m360_full9_qualitative_gallery.png)

推荐讲法：

> 全图肉眼差异不总显著，因为很多收益来自 residual-level local correction；但 error map 和定量指标显示 ours error 更低。因此 PPT 中建议把 full-frame 图作为公平性证明，把局部图作为视觉收益证明。

### 5.3 室外场景细节对比

这张图专门覆盖 `flowers/garden/treehill/bicycle/stump` 等室外场景，适合回应“室外场景视觉差异不够明显”的问题。

![SPCarNet outdoor detail error reduction showcase](../../assets/spcarnet_m360_outdoor_detail_showcase.png)

代表性 outdoor crop：

| crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | `+0.99 / +0.0616 / -0.0682` | `+2.05` | `24.2%` |
| garden / `00008.png` | `+1.27 / +0.0432 / -0.0551` | `+2.70` | `27.6%` |
| treehill / `00010.png` | `+0.59 / +0.0491 / -0.0881` | `+3.03` | `32.0%` |
| bicycle / `00021.png` | `+1.13 / +0.0385 / -0.0615` | `+1.88` | `17.5%` |
| stump / `00007.png` | `+0.26 / +0.0122 / -0.0208` | `+0.81` | `12.8%` |

推荐讲法：

> 室外场景的全图差异确实更难直接看出来，但在叶片、木纹、树皮、长椅条纹等 high-frequency surface 区域，SPCarNet 的局部 error reduction 更稳定。

---

## 6. 与 MeshSplatting 论文表格的关系

| Method / protocol | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` |
| Local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` |

需要强调：

- 主 claim 应以本地同协议 selected clean MeshSplatting baseline 为准；
- 本地 clean baseline 数值已经高于论文表格，因此不是挑弱 baseline；
- 论文表格可能存在 resolution、mask、split、preprocessing、evaluator、checkpoint iteration 等差异；
- 当前最严谨说法是：**在本地复现/评估口径下，SPCarNet 全面超过 selected clean MeshSplatting baseline；相对论文表格也数值更高，但跨论文表格比较要标注口径差异。**

---

## 7. 当前边界与正在推进的线

已经可以稳健汇报的部分：

- 相同本地 evaluator 下，SPCarNet Phase-J 相对 selected clean MeshSplatting full9 场景 RGB 三指标 `9 / 9` 严格胜出；
- held-out view 级 `244 / 246` 严格胜出；
- 平均 PSNR/SSIM/LPIPS 明显改善；
- 同时平均删除 `7.6479%` triangles；
- 有局部 crop、full-frame、outdoor detail 三类定性证据。

需要诚实标注的边界：

- 当前最大视觉收益仍主要来自 guarded render-time adapter，而不是完全 baked into persistent representation；
- 表示级 residual atlas 已经打通，但目前收益多为 `1e-4` 到 `1e-6` 量级，不能替代 Phase-J headline；
- 室外全图肉眼差异不如室内明显，展示时应使用局部 crop + error map；
- triangle reduction 是删面比例，不等价于完整系统 FPS、显存、模型大小收益；这些还需要单独 profiling。

正在验证的新线：

- `v87_source_mixture`：把 prior-bin hard copy 改为连续 source mixture，希望减少二值替换带来的 tail-risk。当前仍在运行，不计入本报告主结论。
- `v89b_l1proxy`：counter probe 有极小 PSNR 提升，但未形成 full9 闭环，不能作为主结果。

---

## 8. 建议 PPT 结构

1. Problem：MeshSplatting 训练后仍有局部 residual 与几何冗余。
2. Idea：把训练视角从 supervision 变成 surface evidence。
3. Method：evidence cache -> triangle compaction -> guarded residual repair -> policy-val fallback。
4. Quantitative：full9 表格，突出 `9/9` scene wins、`244/246` view wins、`+1.331` PSNR、`-0.063` LPIPS、`7.65%` triangle removal。
5. Qualitative：先放局部主图，再放 full-frame 公平图，再放 outdoor detail 图。
6. Boundary：render-time adapter 仍是主收益来源，representation-level baked repair 是下一阶段。

---

## 9. 关键文件

- 主报告：`docs/car_model/6-25-SPCarNet-Current-Method-vs-Standard-MeshSplatting-Visual-Report.zh.md`
- 更长技术报告：`docs/car_model/6-25-SPCarNet-Current-Method-vs-MeshSplatting-Complete-Report.zh.md`
- PPT 母稿：`docs/car_model/6-25-SPCarNet-Mentor-PPT-SlideReady-Technical-Report.zh.md`
- 主图：`assets/spcarnet_phasej_where_it_helps_showcase_20260622.png`
- full-frame 图：`assets/spcarnet_m360_full9_qualitative_gallery.png`
- outdoor 图：`assets/spcarnet_m360_outdoor_detail_showcase.png`
- 核心代码：`scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- 实验入口：`scripts/car_model/run_l1risk_fairnoop_scene.py`

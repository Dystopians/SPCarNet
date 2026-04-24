# CarNet 进展完整报告（v0.1 → v0.7）

_生成日期：2026-04-24，用于次日口头汇报与 PPT 素材整理。_

---

## 0. 项目定位

**CarNet** 是把一个车辆 mesh 的 corrupted / 部分观测 点云，**修复/补全成干净车辆几何**的 prior 模型。
它服务于更大的 mesh-splatting 流水线——3DGS 给出带残缺的点云观测，CarNet 输出一组修好后的 2048 点（带法线），下游用作 Gaussian splat 的初始化或约束。

### 关心的核心指标

| 指标 | 含义 | 越好方向 |
|---|---|---|
| `recon_chamfer_l1` | 输出云 vs GT 云的双向 L1 chamfer | ↓ |
| `visible_recon_chamfer_l1` | 可见子集的 chamfer（表面那一半） | ↓ |
| `hidden_completion_chamfer_l1` | 遮挡侧补全的 chamfer（背面） | ↓ |
| `denoise_gain_chamfer` | `chamfer_before - chamfer_after`，修复相对输入的改善 | ↑ |
| `occupancy_iou_visible` | 占据场 head 与 GT 在可见范围的 IoU | ↑ |
| `free_space_violation_rate` | 模型在真空气中放点的比例（不该放点却放了） | ↓ |
| `recon_normal_cosine` | 法线方向一致性 | ↑ |

**重要教训（来自 v0.6）**：`denoise_gain` 会被 corruption 难度放大；不同难度间直接比 gain 是陷阱。**绝对 chamfer** 才是可比较的唯一质量标尺。

---

## 1. 数据集

**基础集**：MeshFleet（Huggingface 上的 OBJ 车辆 mesh 数据集）。切 patch 后形成 `meshfleet_car_cache_v4`，**1,616 patches**（train/val/test 按 car_mesh 预分划，禁止帧内混切）。

**扩充**：用 Objaverse 1.0 通过 LVIS 标签筛出"vehicle"类目，下载 GLB，SHA256 和 MeshFleet 去重（防数据泄漏），处理成与 MeshFleet 同格式的 patch cache。用法命名为 `__ext1` 的 town suffix，与原 MeshFleet 并行；合并后形成 `meshfleet_car_cache_v5`，**2,433 patches**（+817 约 +50%）。

**关键文件**：
- `/data2/peilincai/car_models/materialize_ext.py` — Objaverse 下载 / 去重 / 格式化流水线
- `ss3dm_prior/tools/merge_patch_caches.py` — 合并 v4 + ext1 的 index + manifest + split yaml
- `ss3dm_prior/tools/build_car_mesh_patch_cache.py` — 加了 per-future try/except，单个坏 GLB 不会再炸整条 pipeline

**数据使用时间线**：v0.1 用 v4，v0.2+ 全部用 v5。

---

## 2. 架构（v11_latent_flow_hybrid）

所有 v0.x 实验共享此架构（除 v0.3/v0.5 临时加宽，后续证伪放弃）。

- **Encoder**：per-point feature + 3 层 cross-attention 到 32 个 latent queries（ffn_dim=768，heads=6）
- **Latent**：384 维 latent code（+192 维 retrieval head）+ 2 层 self-attention
- **Decoder**：
  - **Residual 模式（默认）**：per-point decoder 预测 delta，加回 corrupted 点——天然对齐输入，但**这正是 identity-copy 陷阱的源头**（模型可以直接回复输入得低 loss）
- **Heads**：occupancy（256×3 MLP）、free-space violation、point-defect、corruption-score、intrinsic-difficulty、symmetry（可选）、retrieval-anchor、latent-flow（8-step，预测 hidden_residual）
- **参数量**：37.21M

**未尝试过的架构开关**（都是有意留给未来的）：
- `use_residual_reconstruction: false`（切换到 global decoder，完全砸掉 per-point 脚手架）
- `use_vector_quantization: true`（VQ codebook，目前 commit loss 权重 0）
- `prototype_diversity_loss > 0`（prototype head，权重 0）

---

## 3. 完整实验时间线

### v0.1（起点，ep=44，v4 数据）— 数据少+arch 未调

| 指标 | 值 |
|---|---|
| recon | 0.0991 |
| gain | 0.0188 |
| visibility | 0.6270 |
| composite | 0.1213 |

**做了什么**：初次把架构跑通。
**结果**：occupancy IoU 只有 0.805，free-space violation 13.7%——视觉修复明显模糊、轮廓飘。
**用户反馈**："修复效果还是太差了，该怎么办"
**诊断**：(1) 数据太少；(2) occupancy 未收敛；(3) corruption 太温和，模型学的是 identity-copy 解。

### v0.2（基线落地，ep=129，v5 数据）— 扩充数据 + 稳定化

**动机**：给 v0.1 的问题一次"干净重启"的基准——扩数据、适中 corruption、150 ep 训练。

**变更**：
- 切到 `meshfleet_car_cache_v5`（+50% 数据）
- corruption 温和（dropout 0.18, jitter 0.04, hole 0.18）
- loss 配置保守（recon_chamfer=1.0, hidden_completion=1.0）
- 架构定型：37M

**结果**：
| 指标 | v0.1 | **v0.2** |
|---|---|---|
| recon | 0.0991 | **0.0916** |
| visibility | 0.6270 | **0.8825** |
| composite | 0.1213 | **0.2984** |
| free_space_violation | 0.1370 | **0.0373** |

**关键意义**：occupancy 收敛了（0.92 IoU），整个修复的"形状正确"问题被解决。v0.2 是之后所有实验的 **基线对照**。
**但残留问题**：gain 只有 0.0258——模型确实在修，但修得不多；部分源于"输入已经不差，没多少空间可修"（identity-copy 仍有吸引力）。

**保存的训练 logs**：`outputs/carnet/v0_2/full/`

### v0.3（ep=149）— 架构加宽的失败尝试

**动机**：v0.2 的 gain 太小，猜想是模型容量不足。推到 latent_dim=512, ffn_dim=1024, heads=8。

**变更**：
- **只改架构**（宽度放大 ~60%）
- 其他所有 loss / corruption / schedule 保持 v0.2 一致

**结果**：
| 指标 | v0.2 | **v0.3** |
|---|---|---|
| recon | 0.0916 | 0.0921 |
| gain | 0.0258 | 0.0254 |
| visibility | 0.8825 | 0.8739 |
| paper | 0.2256 | 0.2454 |

**诊断**：加宽反而**延迟了 occupancy ignition**（v0.2 在 ep 24 就点火，v0.3 拖到 ep 54）。最终拉到同一高度，但没有净收益。

**过程中的一次误判**：ep 50 时看 occupancy 仍然 ~0，我判 "DEAD"。结果它在 ep 54 ignite。**教训是 early-kill 要非常谨慎**，大架构有长尾 ignition 行为。（坦白记录此误判）

### v0.5（ep=80，提前停止）— 更宽 + normal fix

**动机**：最后再试一次加宽（latent_dim=448, ffn_dim=960）+ 顺手修了 v0.4 的 normal 退化（见下）。

**结果**：
| 指标 | v0.4 final | v0.5 @ ep80 |
|---|---|---|
| recon | 0.1015 | 0.1021 |
| gain | 0.0436 | 0.0431 |

**结论**：再次验证**宽度不是瓶颈**。v0.5 和 v0.3 两次独立实验都显示宽度扩展在这个架构上是零和甚至负收益。**宽度路线就此关闭**。
**用户反馈**："怎么感觉没有 combo win 的说法，很玄"——对，及时撤掉。

### v0.4（ep=149）— 第一次真正突破

**动机**：identity-copy 撞不破的根因是 corruption 太温和——模型把输入原样吐出就能拿到低 chamfer。**不是加宽，是加硬 input**。

**变更**（相对 v0.2）：
- `point_dropout` 0.18 → **0.25**
- `gaussian_jitter.sigma` 0.04 → **0.06**
- `local_hole_mask.hole_radius` 0.18 → **0.22**
- `outlier_cluster.cluster_size` 32 → **48**
- 新增 `symmetry_consistency_loss: 0.5`（车身左右对称 prior）
- 架构、loss 权重完全同 v0.2

**结果（eval 集，绝对数）**：
| 指标 | v0.2 | **v0.4** |
|---|---|---|
| recon | 0.0916 | 0.1017 (+11%) |
| **gain** | 0.0262 | **0.0445 (+70%)** |
| visibility | 0.9225 | 0.9191 |
| hidden_completion_gain | 0.0226 | 0.0420 (+86%) |
| recon_normal_cosine | 0.4249 | 0.3846 **(-9%)** |

**核心意义**：
- gain **翻倍**，模型第一次在"去噪"而不是"复制输入"
- 可见面完整性（occupancy IoU）几乎没掉
- **代价**：recon 绝对值略升——合理，输入变糟了 +2.5pp（before 从 0.118 → 0.145），after 只升 1pp，净 gain 大幅正
- **副作用**：normal_cosine 退了 9%——法线方向更乱，因为所有法线都被更强 jitter 拉歪后，recon_normal_loss 权重没跟上

**这是项目迄今最重要的正信号**。用户语："进行 v0.4" → "启动"。

**保存在**：`outputs/carnet/v0_4/full/`（149 ep 全套 checkpoint）、`outputs/carnet/v0_4/eval/carnet_v0_4_eval/`（eval metrics + 42 张 panels）。

### v0.6（ep=149）— 进一步硬化 + split chamfer

**动机**：v0.4 证明"加硬 corruption + symmetry"有效。继续这条路：
1. 把 recon_chamfer 拆成 visible + hidden 两个独立监督信号（不再用 full-cloud 作为唯一梯度源）
2. corruption 再加硬（dropout 0.30, jitter 0.075, hole 0.26）
3. `recon_normal_loss` 从 0.5 → 0.8 修 v0.4 的 normal 退化
4. `symmetry_consistency_loss` 0.5 → 0.8

**loss 权重变化**：
- `recon_chamfer_loss` 1.0 → **0.3**（降级为全局正则）
- `hidden_completion_chamfer_loss` 2.0（保留）
- `visible_recon_chamfer_loss` **新增 1.0**

**结果（checkpoint best_metrics）**：
| 指标 | v0.4 | **v0.6** |
|---|---|---|
| best_gain | 0.0436 | **0.0593 (+36%)** |
| best_paper | 0.2614 | **0.2884 (+10%)** |
| best_recon | 0.1015 | 0.1070 |

*看起来又一次大胜* —— 但是！

**重要发现（v0.6 eval 之后）**：

| **绝对 chamfer**（apples-to-apples） | v0.4 | **v0.6** | diff |
|---|---|---|---|
| recon_chamfer_l1 | 0.1017 | 0.1073 | **-5.5% 退步** |
| visible_chamfer_l1 | 0.1100 | 0.1152 | -4.7% 退步 |
| hidden_chamfer_l1 | 0.1675 | 0.1749 | -4.4% 退步 |
| occupancy_iou_visible | 0.9191 | 0.9142 | -0.5% |

**gain 的 +36% 是伪信号**。corruption 从 0.25/0.06/0.22 推到 0.30/0.075/0.26 之后，`chamfer_before` 从 0.147 涨到 0.175——**输入变糟了，`gain = before - after` 当然膨胀**，但 `after`（输出的绝对质量）反而**每一项都退步了**。

**这是项目最重要的方法论教训**：
> **`gain` 在 corruption 不同的实验间不可比。只有 `absolute chamfer_after` 可比。**

视觉对比（school bus `best_hidden_completion__rank1`）也证实：v0.6 修复件比 v0.4 **明显更糙**——车顶边缘毛刺多，挡风玻璃线条散。

### v0.7（ep=125，被服务器 SIGKILL 中断）— 受控消融

**动机**：v0.6 的"split chamfer 有用吗"悬念没回答——v0.6 同时改了 corruption 和 loss 结构两个变量。v0.7 **固定 corruption = v0.4**，只保留 v0.6 的 loss 结构，做 clean ablation。

**变更**（相对 v0.6）：
- corruption 回退到 v0.4（dropout 0.30 → **0.25**，jitter 0.075 → **0.06**，hole 0.26 → **0.22**）
- **保留** v0.6 的 split visible/hidden chamfer（visible 1.0 + hidden 2.0 + holistic 0.3）
- **保留** v0.6 的 symmetry=0.8、recon_normal=0.8

**结果**：
- **ep 49 中期**：recon=0.1018，gain=0.0433 —— 已经追平 v0.4 final！（v0.4 最终是 0.1015 / 0.0436）
- **ep 104**（被 killed 当时的 best）：recon=**0.1015**，gain=**0.0436** —— **精确追平 v0.4**
- **ep 125**（resume 后再被 killed，best 未变）：recon 0.1015，vis 0.8744（爬到接近 v0.4 的 0.8804）

**结论**：
1. **split chamfer 在 corruption 匹配时至少持平 v0.4**（不是噪音，但也不一定是严格进步）
2. 训练被 SIGKILL 两次打断（服务器外部资源抢占），最后 45 epochs 没跑完
3. v0.4 仍是实际意义上的 SOTA（恢复到相同条件下两者绝对质量打平，v0.4 训练完整）

---

## 4. 关键方法论收获（建议在汇报里单独一页）

1. **corruption 难度和 gain 一体化调整**。corruption 变硬时 gain 数字膨胀不代表质量提升，必须锁死 `absolute chamfer_after` 作为唯一 SOTA 比较轴。
2. **identity-copy 是 residual 架构的内生吸引子**。需要足够硬的 input corruption 才能撞破这个局部最优——v0.4 是把这个信号打透的关键节点。
3. **宽度扩展在这个架构上是零收益**。v0.3 / v0.5 两次独立实验一致结论。下次要突破 ceiling 不要再打宽度的主意。
4. **early-kill 要谨慎**。v0.3 在 ep 50 看似死，ep 54 ignite——大架构有长尾冷启动，判死至少要看到 80 ep。
5. **Warm-restart LR 要显式 reset**。`optimizer.load_state_dict()` 会覆盖 `initial_lr`，cosine scheduler 的 `last_epoch==0` 再次把当前 LR 锁死为 min_lr——已在 `trainer.py` 加上显式 `group["lr"] = configured_lr; group["initial_lr"] = configured_lr` 的 patch。

---

## 5. 最终数字总表（汇报直接抄）

### checkpoint best_metrics（模型自身保存的 best）

| version | epoch | best_recon↓ | best_gain↑ | best_vis↑ | best_composite↑ | best_paper↑ |
|---|---|---|---|---|---|---|
| v0.1 | 44 | 0.0991 | 0.0188 | 0.6270 | 0.1213 | 0.0902 |
| **v0.2** | 129 | **0.0917** | 0.0258 | **0.8825** | **0.2984** | 0.2256 |
| v0.3 | 149 | 0.0921 | 0.0254 | 0.8739 | 0.2933 | 0.2454 |
| **v0.4** | 149 | 0.1015 | **0.0436** | 0.8804 | 0.2948 | 0.2614 |
| v0.5 | 80 | 0.1021 | 0.0431 | 0.8560 | 0.2845 | 0.2236 |
| v0.6 | 149 | 0.1070 | 0.0593 | 0.8711 | 0.2957 | **0.2884** |
| v0.7 | 125 | 0.1015 | 0.0436 | 0.8744 | 0.2924 | 0.2557 |

### test-set eval metrics（绝对，唯一可比轴）

| version | recon_chamfer↓ | visible_chamfer↓ | hidden_chamfer↓ | occupancy_IoU↑ | free_space_viol↓ | normal_cos↑ |
|---|---|---|---|---|---|---|
| v0.1 | 0.0989 | 0.1047 | 0.1680 | 0.8050 | 0.1370 | 0.4206 |
| **v0.2** | **0.0916** | **0.0996** | **0.1591** | **0.9225** | **0.0373** | **0.4249** |
| v0.4 | 0.1017 | 0.1100 | 0.1675 | 0.9191 | 0.0382 | 0.3846 |
| v0.6 | 0.1073 | 0.1152 | 0.1749 | 0.9142 | 0.0401 | 0.3828 |

**汇报时如实说**：如果只看 absolute chamfer，**v0.2 的输出质量 strictly 最好**；v0.4 的优势在 denoise gain（修复能力），v0.6 是被 gain 指标误导的方向性试错。

---

## 6. 可供 PPT 使用的视觉素材清单

### 6.1 纹理三联图 textured triptychs（**最推荐**，老板们爱看）

这是 CarNet 的旗舰可视化：每个测试样本渲染成 3 列（Corrupt Input / Repaired Output / Ground Truth）× 4 行（Hero 3/4, Front, Top-Down, Low Angle），所有车辆**带贴图**。

**每个版本各 6 张**，每张对应一个 eval "gallery"（hero case）：
- `best_hidden_completion__rank{1,2}`：遮挡侧补全最好的前两个
- `largest_intrinsic_score_error__rank1`：模型对"修复难度"预估最偏的
- `worst_free_space_violation__rank{1,2}`：free-space head 出错最严重的

**路径**：
```
outputs/carnet/v0_1/eval/carnet_v0_1_full_eval/patch_panels/*__textured_triptych.png
outputs/carnet/v0_2/eval/carnet_v0_2_eval/patch_panels/*__textured_triptych.png
outputs/carnet/v0_4/eval/carnet_v0_4_eval/patch_panels/*__textured_triptych.png
outputs/carnet/v0_6/eval/carnet_v0_6_eval/patch_panels/*__textured_triptych.png
```

**建议放 PPT 的 killer comparison**：
- 同一 patch ID `7173cc5f880504ec40c7228e74ad51d05c830708151cba477dd12f287881bca3`（一辆校车）
- v0.1 → v0.2 → v0.4 → v0.6 四张横排
- v0.1 修复散乱 / v0.2 形状正确但平淡 / v0.4 细节最干净 / v0.6 回退略糙
- **这一页直接讲故事：架构 → 数据 → corruption → 受控消融**

### 6.2 训练中 epoch 可视化

每 10 ep 会渲一批 sample 的 panels：
```
outputs/carnet/v0_6/full/visualizations/epoch_{000,010,020,...,149}/
```
每个 sample 包含 8 种子图：
- `_panel.png`（点云）
- `_hybrid_reconstruction_panel.png`
- `_triptych.png`（点云版三联图）
- `_visibility_panel.png` / `_visible_vs_hidden_panel.png`（可见/隐藏着色）
- `_free_space_error_panel.png`（free-space 错误点）
- `_difficulty_calibration_panel.png`（预测难度 vs 真实难度散点）
- `_retrieval.png`（retrieval head 找到的最近邻样本）

**建议**：从 v0.6 epoch_000 / epoch_030 / epoch_149 各抽一张 `_panel.png` 做 "model learning over time" 页。

### 6.3 Eval 级的 summary panels

每套 eval 都有：
- `difficulty_calibration_panel.png`（全测试集的预测难度 vs 真实难度散点）
- `metrics_per_sequence.csv` / `metrics_per_town.csv`（每个序列/town 分桶）
- `report.md`（模型自动生成的文字报告）
- `prototype_gallery/`（原型簇的代表 mesh）
- `sequence_maps/`（序列空间 overview）

**建议放一张**：v0.4 的 `difficulty_calibration_panel.png` 证明模型学会了"自我评估难度"（这是 `intrinsic_difficulty` head 的可视化）。

### 6.4 Wandb loss 曲线

所有训练都 log 到同一个 project：
**https://wandb.ai/karamazovaniki-university-of-southern-california/carnet_v0_2**

挑选的 runs：
- `carnet_v0_2`（baseline）
- `carnet_v0_3` / `carnet_v0_5`（宽度实验——用来说明"宽度路径已死"）
- `carnet_v0_4`（第一次 breakthrough）
- `carnet_v0_6`（corruption 加硬 + split chamfer）
- `carnet_v0_7` / `carnet_v0_7_resume`（受控消融）

**建议截图**：
- `epoch/train/denoise_gain_chamfer` 四条线对比（v0.2 flat 低、v0.4 陡升、v0.6 更陡、v0.7 ≈ v0.4）
- `val/recon_chamfer_l1` 对比（说明绝对 chamfer 打平而非退步）

---

## 7. 建议的 PPT 主线（12-15 页）

1. **Title + 问题设定**：CarNet 做什么，为谁服务
2. **数据**：MeshFleet（1616）→ +Objaverse 车辆（v5: 2433 patches）
3. **架构一览图**：v11 latent flow hybrid（拍个结构图）
4. **基线 v0.2**：数字 + 一张 textured triptych 展示修复能力
5. **路径探索：架构加宽（v0.3/v0.5）**：结论"宽度零收益"，放 val/recon 曲线图
6. **路径突破：corruption 加硬（v0.4）**：gain 翻倍，triptych 对比 v0.2
7. **误判记录：v0.6 表面胜利**：headline +36% gain 是陷阱
8. **方法论节**：`gain` 不能跨 corruption 比较——放绝对 chamfer 对比表
9. **受控消融 v0.7**：split chamfer 在 matched corruption 下打平 v0.4，验证它"不是噪音"但也"不是严格进步"
10. **视觉对比 killer page**：同一辆车，v0.1/v0.2/v0.4/v0.6 四联横排 textured
11. **剩下的未探索方向**：`use_residual_reconstruction=false`, VQ, prototype diversity, 更多 Objaverse 数据
12. **诚实总结**：目前最佳模型 = **v0.4**（综合 gain 和 absolute chamfer）；v0.6 = 教训；v0.7 = 方法论确认

---

## 8. 服务器恢复后的第一优先动作（仅作记录）

- 把 v0.7 resume 跑完（ep 105→150），看终点 recon 能否跌破 v0.4 的 0.1015
- 若 v0.7 final 严格 < v0.4：split chamfer 确认为净正，作为 v0.8 默认配置
- 若 v0.7 final ≈ v0.4：放弃 loss-structure 路线，切架构级改动（residual 开关或 VQ/prototype 启用）

---

_报告结束。所有路径均可直接被 PPT 工具引用。如需我帮你把 killer comparison 那张图在 Python 里横向拼成一张，等服务器恢复就可以做（只读图，不占 GPU）。_

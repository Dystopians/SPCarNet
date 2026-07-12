# SPCarNet 当前方法完整技术报告（Mentor/PPT 汇报版）

日期：2026-06-24  
汇报用途：给 mentor 说明当前方法、实验现状、相对 MeshSplatting baseline 的收益、仍未解决的短板，以及下一步论文级路线。  
建议 PPT 主线：**SPCarNet = Self-auditing MeshSplatting：用训练视角 surface evidence 做安全压缩与残差修复**。  

---

## 0. 一页结论

当前最适合汇报的主结果不是 v75/v76/v77 这些最新诊断分支，而是已经闭环验证过的：

> **Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction**

它的核心 claim 是：

> 在已经训练好的 MeshSplatting checkpoint 上，SPCarNet 利用训练视角的 surface evidence 自动判断哪些三角形可以安全删、哪些 surface 区域存在稳定 residual 可以修、哪些区域证据不足必须回退。当前 Phase-J endpoint 在本地 same-protocol Mip-NeRF360 full9 selected-clean MeshSplatting baseline 上实现 9/9 场景三指标严格胜出，并平均删去 7.6479% triangles。

主结果数字：

| 指标 | Phase-J vs 本地 selected clean MeshSplatting |
|---|---:|
| 场景数 | `9 / 9` |
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` |
| per-view PSNR/SSIM/LPIPS strict wins | `244 / 246` |
| mean dPSNR | `+1.331084` |
| mean dSSIM | `+0.034702` |
| mean dLPIPS | `-0.063359` |
| mean triangle reduction | `7.6479%` |
| geometry-safe scenes | `9 / 9` |

当前必须诚实表达的边界：

- **可以强讲**：Phase-J 是一个强的、可审计的 MeshSplatting 后处理/修复 endpoint；在本地 selected-clean full9 口径上，RGB 三指标与 triangle count 同时优于 clean MeshSplatting。
- **不能过度讲**：当前最强广泛收益主要来自 render-time evidence-lumigraph adapter，不是已经完全 baked into checkpoint 的终局 representation。
- **表示级路线现状**：v64 fixed auto bin-alpha policy 是当前最稳固定 representation-level 候选，但只带来非常小的均值增益；v65-v77 诊断证明瓶颈不在继续扫 alpha/blend/cap，而在 sparse surface support、residual capacity 和 target-view generalization certificate。
- **对象级 shape-prior 路线现状**：Stage 2 v4 normal-band autodecoder 已经完成真实训练/评估链路；相比 v3 明确改善 chamfer 与 filled IoU，但仍未达到原 gate，因此只能作为下一阶段几何先验储备，不能作为本次 headline。

---

## 1. 给 PPT 开场的 30 秒版本

中文：

> MeshSplatting 已经能训练出高质量显式 mesh，但 clean checkpoint 仍有局部纹理残差和几何冗余。SPCarNet 不重新发明一个完全不同的表示，而是让训练视角中的 surface evidence 反过来审计这个 mesh：多视角证据稳定的位置可以修复，低风险面片可以压缩，证据不足的位置自动回退。当前 Phase-J 在本地 full9 Mip-NeRF360 selected-clean baseline 上 9/9 场景三指标全胜，同时平均删除 7.65% triangles。

英文：

> SPCarNet turns a trained MeshSplatting model into a self-auditing surface system: it repairs only where training-view surface evidence certifies stable residuals, compacts only low-risk triangles, and falls back when the certificate is weak.

最建议用作 slide title 的一句话：

> **From MeshSplatting to Self-Auditing Surface Repair and Compaction**

---

## 2. 研究问题

MeshSplatting 的优势是把高质量 novel-view synthesis 放到显式 mesh/surface 表示上，便于部署、压缩、编辑和几何审计。但 clean MeshSplatting checkpoint 仍存在三类问题：

| 问题 | 具体表现 | SPCarNet 对应目标 |
|---|---|---|
| 局部外观 residual | 桌面、叶片、遮挡边界、细纹理区域仍有稳定 RGB 残差 | 从训练视角 residual 中提取可迁移修复信号 |
| 几何冗余 | 一部分 triangles 对多视角解释贡献低 | 在质量不退化前提下删去低风险 faces |
| 泛化风险 | 盲目修复容易伤害 tail views 或 out-of-trajectory 视角 | 用 train/policy-val evidence gate 选择修复、回退或 no-op |

研究问题可以写成：

```text
Given a trained MeshSplatting checkpoint,
can training-view surface evidence certify where the mesh can be compacted
and where appearance residuals can be safely repaired?
```

这不是普通图像增强。SPCarNet 的修复必须被真实 surface、face id、visibility、support、risk 和 train-only policy gate 共同约束。

---

## 3. 与原始 MeshSplatting 的区别

基础 MeshSplatting pipeline：

```text
training images + cameras
  -> train MeshSplatting
  -> mesh checkpoint
  -> render held-out views
```

SPCarNet pipeline：

```text
training images + cameras
  -> train/load MeshSplatting checkpoint
  -> build train/policy-val surface evidence
  -> estimate support, residual, visibility, and risk
  -> geometry-safe compaction
  -> evidence-certified residual repair
  -> train-only policy selection and fallback
  -> held-out evaluation
```

| 维度 | MeshSplatting | SPCarNet |
|---|---|---|
| 表示基础 | 显式 mesh/splat representation | 继承 MeshSplatting checkpoint |
| 是否利用训练残差做 surface 审计 | 不显式做 | 构建 face/bin/residual/support/risk evidence |
| 几何压缩 | checkpoint 固定 | train-only quality-first compaction |
| 外观修复 | 由原 checkpoint 直接渲染 | stable residual transfer + policy gate |
| 安全机制 | 依赖训练收敛和 checkpoint 选择 | train/policy-val gate、tail risk、fallback |
| 当前收益来源 | 原始 MeshSplatting 表示 | compact checkpoint + guarded residual repair |

---

## 4. 方法总览

SPCarNet 当前由五个核心模块组成。

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Evidence Cache | train renders、GT、camera、surface maps | residual、face id、normal、UV/bin、support、risk | 把训练视角变成可审计证据 |
| Geometry-Safe Compaction | mesh checkpoint + evidence | compact checkpoint | 删除低风险冗余 triangles |
| Guarded Evidence Lumigraph Adapter | compact render + train residual evidence | repaired render | 把稳定 surface residual 转移到 held-out view |
| Train-Only Policy Gate | policy-val metrics、view-tail risk | accept / fallback / no-op | 避免 test leakage 和不安全修复 |
| Surface Residual Atlas | face/UV/bin evidence | persistent residual field | 当前 representation-level 内化路线 |

代码入口与关键脚本：

| 部分 | 路径 |
|---|---|
| Surface residual region texture adapter | `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py` |
| L1-risk fair no-op scene runner | `scripts/car_model/run_l1risk_fairnoop_scene.py` |
| v64 full9 fixed policy runner | `scripts/car_model/run_v64_bin_alpha_auto_policy_pipeline.py` |
| v64 summary | `scripts/car_model/summarize_v64_bin_alpha_auto_policy.py` |
| Phase-J主结果报告 | `outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md` |

---

## 5. 技术模块细节

### 5.1 Evidence Cache

Evidence cache 是方法的数据底座。它记录 train/policy-val 视角中的：

- rendered RGB；
- ground-truth RGB；
- residual map：`GT - Render`；
- alpha、visibility、depth；
- face id、barycentric coordinate 或 UV/bin；
- normal、view direction、camera center；
- per-face / per-bin support count；
- residual sign consistency；
- per-view PSNR、SSIM、LPIPS、image L1；
- min-view 和 CVaR tail risk。

这一步的意义是把训练数据从“只用于优化 checkpoint 参数”升级为“用于审计 surface 是否可靠”的证据。

### 5.2 Geometry-Safe Compaction

压缩原则是 quality-first，而不是最大化删面：

```text
remove triangles only when multi-view evidence says the edit is low-risk
```

主要机制：

- 低可见性、低贡献、低风险 faces 优先删除；
- 遮挡边界、thin structure、sparse geometry 区域受保护；
- compact checkpoint 必须能被 renderer 正常加载；
- RGB、sparse geometry、topology 分开审计；
- 一旦 policy-val 发现风险，自动回退更保守版本。

报告中的 `triangle reduction` 指**删去的 triangles 占比**，不是剩余比例。当前 Phase-J 平均删面为 `7.6479%`。

### 5.3 Guarded Evidence Lumigraph Adapter

Phase-J 的主要 RGB 提升来自 guarded Evidence Lumigraph Adapter。直观理解是：

```text
target render = compact render + train-evidence residual correction
```

简化公式：

```text
residual_i = GT_i - Render_i

I_ours(p) = I_compact(p) + alpha * sum_i w_i(p) * residual_i(u_i)
```

其中：

- `p` 是 target held-out view 像素；
- `residual_i(u_i)` 来自训练视角中相同或相近 surface 区域；
- `w_i(p)` 由 visibility、surface correspondence、support 和风险门控决定；
- `alpha` 由 train/policy-val evidence 选择；
- 证据不足时自动 fallback 或 no-op。

通俗讲法：

> 我们不是凭空增强图片，而是看训练视角里同一个 surface 区域是否反复出现稳定错误。如果错误稳定且多视角支持充足，就把这部分 residual 迁移到目标视角；否则不动。

### 5.4 Train-Only Policy Gate

公平性规则：

- branch selection、alpha、support、fallback 都来自 train 或 policy-val evidence；
- held-out test GT 只用于最终报告；
- 不用 test metric 选择我们方法参数；
- baseline envelope 用 held-out test 选择最强 clean checkpoint，是为了不低估 baseline；
- 不用 train metric 选择 baseline，因为那会偏向训练更久的 checkpoint；
- 风险高时回退稳定版本。

当前 Phase-J 中：

- `8 / 9` 场景走 adaptive-alpha ELA；
- `treehill` 走 train-selected edge fallback；
- fallback 的存在是方法安全性的一部分，不是失败后手动挑图。

### 5.5 Surface Residual Atlas

为了把 render-time adapter 的收益内化到表示里，我们实现了 face/UV-addressed residual atlas：

```text
fit-view residual evidence
  -> reliable face/bin selection
  -> residual atlas fitting
  -> policy-val risk gate
  -> target view surface lookup
  -> persistent surface-addressed repair
```

它回答的问题是：

> 能否把训练视角 residual 变成绑定在 mesh 表面的、可审计、可回退、可复用的修复表示？

当前结论：接口、安全策略和 full9 自动 policy 已经成型，但 RGB 效果量级仍小，还不能替代 Phase-J 作为 headline endpoint。

---

## 6. 主结果：Phase-J vs selected clean MeshSplatting

评估口径：

- 数据集：Mip-NeRF360 full9；
- clean baseline candidates：local clean `26000` 和 `30000`；
- baseline selection score：`PSNR + 20 * SSIM - 20 * LPIPS`；
- 选择 baseline 时使用 held-out test metrics，是为了构造 strongest clean baseline envelope；
- 我们方法的 branch/policy 不使用 held-out test metrics。

主表：

| scene | branch | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | triangle reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | adaptive alpha | 24.021544 | 0.702357 | 0.266088 | +0.719931 | +0.042489 | -0.065989 | 11.81% |
| flowers | adaptive alpha | 20.304358 | 0.557770 | 0.329222 | +0.622101 | +0.045948 | -0.065341 | 11.82% |
| garden | adaptive alpha | 26.311111 | 0.827843 | 0.135843 | +1.281900 | +0.047808 | -0.065472 | 3.47% |
| stump | adaptive alpha | 25.595104 | 0.724074 | 0.263909 | +0.390062 | +0.018909 | -0.030095 | 11.82% |
| treehill | auto edge fallback | 21.296227 | 0.595606 | 0.336319 | +0.362045 | +0.031083 | -0.069725 | 11.81% |
| room | adaptive alpha | 30.305639 | 0.905730 | 0.195989 | +1.558363 | +0.020887 | -0.053913 | 2.10% |
| counter | adaptive alpha | 28.449171 | 0.893731 | 0.186472 | +1.697397 | +0.031675 | -0.065531 | 2.10% |
| kitchen | adaptive alpha | 30.199732 | 0.916087 | 0.131955 | +2.381180 | +0.039635 | -0.067231 | 2.10% |
| bonsai | adaptive alpha | 31.862005 | 0.930280 | 0.172555 | +2.966772 | +0.033879 | -0.086937 | 11.80% |

Aggregate：

| 指标 | 值 |
|---|---:|
| mean dPSNR vs selected clean | `+1.331084` |
| mean dSSIM vs selected clean | `+0.034702` |
| mean dLPIPS vs selected clean | `-0.063359` |
| mean dPSNR vs source ELA | `+0.833143` |
| mean dSSIM vs source ELA | `+0.018946` |
| mean dLPIPS vs source ELA | `-0.039986` |
| mean triangle reduction | `7.6479%` |

主结果证据路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

---

## 7. 与 MeshSplatting paper table 的关系

当前最严谨的说法需要分两层：

1. **本地主结果**：Phase-J vs local selected-clean MeshSplatting，这是最可信的主 claim，因为 split、evaluator、baseline checkpoint 和结果路径都在本地可追溯。
2. **paper-table 辅助对比**：仓库里有一个早期 Compact-ELA/SOR 版本与 MeshSplatting paper table 的对齐 audit，它相对 paper table 为正，但不是最新 Phase-J endpoint。

早期 paper-table audit 摘要：

| 评估口径 | 结果 |
|---|---:|
| RGB wins vs selected clean MeshSplatting | `9 / 9` |
| mean dPSNR vs selected clean | `+0.497941` |
| mean dSSIM vs selected clean | `+0.015755` |
| mean dLPIPS vs selected clean | `-0.023373` |
| mean dPSNR vs MeshSplatting paper table | `+0.868512` |
| mean dSSIM vs MeshSplatting paper table | `+0.036551` |
| mean dLPIPS vs MeshSplatting paper table | `-0.046530` |
| mean triangle reduction | `5.7632%` |

建议向 mentor 这样解释：

> 我们主表应该以本地 same-protocol selected-clean full9 为准，因为这是完整可复现的公平 baseline。paper table 可以作为辅助说明数值量级，但如果最终写论文，需要严格确认与原文在 split、mask、metric implementation、iteration 和 checkpoint selection 上完全一致。

---

## 8. 表示级路线：v64 到 v77 的现状

当前 best representation-level 固定策略是 **v64 fixed auto bin-alpha policy**。

v64 的规则：

```text
if v63b has strong train/policy-val bin-alpha evidence:
    use v63b bin-alpha residual atlas
else:
    fallback to v56 selected policy
```

v64 full9 结果：

| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |
|---|---:|---:|---:|---:|---:|---:|
| v64 vs v56 | 9 | 1 | 9 | +0.000410080 | +0.000000278 | -0.000018951 |
| v64 vs v52 | 9 | 2 | 9 | +0.000706779 | +0.000001563 | -0.000038614 |
| v64 vs no-op | 9 | 7 | 8 | +0.002255970 | +0.000038081 | -0.000093445 |
| v64 vs v48 | 9 | 3 | 9 | +0.000793669 | +0.000010345 | -0.000053917 |
| v64 vs v50 | 9 | 6 | 6 | +0.000991609 | +0.000016345 | -0.000059394 |

v64 selected source：

| scene | selected source |
|---|---|
| bicycle | v56 fallback |
| flowers | v56 fallback |
| garden | v56 fallback |
| stump | v56 fallback |
| treehill | v56 fallback |
| room | v56 fallback |
| counter | v56 fallback |
| kitchen | v63b bin-alpha |
| bonsai | v56 fallback |

Interpretation：

- v64 是一个真实的固定 policy 里程碑，完成了 full9 W&B 复跑和自动选择；
- 它不读场景名，不用 held-out test metric 选策略；
- 它能保证 full9 non-regressive/tie；
- 但效果量级只有 `1e-4` 到 `1e-6`，还不是论文级最终 endpoint。

### 8.1 v65-v77 诊断链

v65-v77 的价值在于定位 representation-level bottleneck，而不是都要作为 promoted method。

| 版本 | 改动 | 结果定位 |
|---|---|---|
| v65 | teacher-distilled shared basis | 未形成稳定强收益 |
| v66 | bin-RGB alpha calibration | 证明 channel-wise alpha 不足以解决核心问题 |
| v67 | uncertainty shrink | 能减少不确定残差，但会过度保守 |
| v68 | keep-with-downweight uncertainty | 保留覆盖但降低风险，仍未超过 v64/v56 reference |
| v69 | count-pyramid multi-scale surface prior | 尝试补 low-support bin coverage |
| v70 | policy-val blend ladder prior | 自动选择 prior strength，counter/kitchen 仍偏向 `blend=0.0` |
| v71a | evidence-consistent prior gate | prior gate 更公平，但未带来晋级收益 |
| v72 | local prior allowlist | 证明局部 allowlist 不足以泛化到 target |
| v73 | target-support candidate selection | 把 target support 纳入候选选择 |
| v73b | target-support pre-rank | 把 support candidates 从 `2` 剪到 `1`，但仍选择 zero-blend |
| v74 | residual delta-cap ladder | 修复 policy-val/final apply cap 一致性，仍选择原始 cap |
| v75 | local patch surface prior | same-face local patch prior 覆盖更多 low-support bins，最终仍选择 `blend=0.0` |
| v76 | policy-val bin-gain hybrid prior | 真实选中 hybrid atlas，但 held-out 指标低于 v75 和 v64/v56 reference |
| v77 | stricter multi-view bin-gain hybrid policy | 已完成 W&B counter 验证；严格 multi-view/abs-gain gate 阻断弱 hybrid，最终选择 `blend=0.0`，未晋级 |

v76/v77 counter 结果：

| method | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| v77 strict bin-gain hybrid | `26.753528595` | `0.862111032` | `0.251881331` | not promoted |
| v76 policy-val bin-gain hybrid | `26.753532410` | `0.862111092` | `0.251881331` | not promoted |
| v75 local patch prior / zero-blend | `26.753995895` | `0.862119257` | `0.251853049` | stronger |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | stronger |

v76/v77 诊断结论：

> v76 证明当前 bin certificate 太弱：局部 bin 在 policy-val 上正收益，不等价于 target-view 一定泛化。v77 加强 samples、policy-val views、absolute gain 和 positive-view fraction 后，弱 hybrid 被阻断，最终 `selected_policy_val_prior_bin_gain_hybrid=false`、`selected_surface_multiscale_prior_blend=0.0`。这说明安全 gate 是必要的，但也确认当前方向没有带来指标晋级。下一步必须提高 residual representation capacity 和 target-view generalization certificate，而不是继续调 blend 数值。

---

## 9. SPCarNet 形状先验与对象级主线（Stage 1-5）

除了 Mip-NeRF360 场景级 MeshSplatting 修复/压缩，我们还推进了一条对象级 SPCarNet shape-prior 主线。它回答的是另一个互补问题：

```text
Can object-level priors recover plausible complete car geometry from partial observations,
and then provide a stronger geometric prior for downstream scene repair?
```

当前这条线已经有完整工程接口，并且 2026-06-24 又补上了 v4 normal-band autodecoder 训练与 full-val MAP-fit 评估。但它还不能作为本次 mentor 汇报的主结果。最合适的定位是：

> 对象级 SPCarNet 已经从 cache、autodecoder、posterior encoder、observation MAP 到 multi-hypothesis reranking 跑通。Stage 2 held-out eval 接口已经在 2026-06-24 修复为 clean-val z-only MAP-fit，`206/206` val objects 可抽取；最新 v4 normal-band objective 相比 v3 降低了 full-val chamfer 并提高了 filled IoU，但仍未过原 gate。因此它是下一阶段“更强几何/表示先验”的技术储备，而不是当前 headline。

对象级 pipeline：

```text
MeshFleet / patch cache
  -> object index and visible/hidden/free-space query cache
  -> canonical shape-field autodecoder
  -> posterior encoder from partial observations
  -> observation-aware latent MAP refinement
  -> multi-hypothesis sampling and reranking
```

### 9.1 Stage 状态总表

| Stage | 模块 | 当前状态 | 关键结果 | 汇报时的口径 |
|---|---|---|---|---|
| Stage 1 | Object index/cache | 已实现并 smoke PASS | `2433` objects；train/val/test = `1854/206/373` | 可讲：数据索引和查询接口已经稳定 |
| Stage 2 | Shape-field autodecoder | 已实现、已训练；v4 normal-band 完成 full training 和 full-val MAP-fit，gate 仍 soft FAIL | v3 val MAP-fit `206/206`：chamfer `0.0698447`，filled IoU `0.553155`；v4 epoch50 val MAP-fit `206/206`：chamfer `0.0607328`，filled IoU `0.568332`，shell IoU `0.878307` | 可讲真实方法改动和质量改善；不能强 claim 质量过关 |
| Stage 3 | Posterior encoder | 已实现、已训练、val 可抽取 | val `206/206` extracted；recon chamfer `0.066391`，hidden chamfer `0.099075`，visible preservation `0.062681`，free-space violation `0.033535`，IoU `0.470897` | 可讲：从 partial observation 到 latent posterior 已跑通 |
| Stage 4 | Observation MAP | 已实现，50-val 子集有效 | free-space violation `0.035820 -> 0.014688`；visible error `0.064352 -> 0.060971`；recon chamfer `0.071490 -> 0.069032` | 可讲：observation loss 确实改善可见/自由空间一致性 |
| Stage 5 | Multi-hypothesis | 已实现，reranking 未完全闭合 | full-val K8 visible rescoring recon chamfer `0.062587`，hidden `0.094253`，free violation `0.03217`；oracle `0.061317` | 可讲：存在 oracle headroom，但当前 inference-only reranker 不够强 |

### 9.2 每个 Stage 具体做了什么

**Stage 1：对象索引与查询 cache**

Stage 1 把 patch-level/object-level 数据统一成一个可复用索引。它维护：

- `object_id`、split、源 mesh 路径；
- visible clean points、hidden clean points；
- occupancy/free-space query；
- 后续 autodecoder、posterior encoder、MAP refinement 共享的数据入口。

这一步的意义是把原先零散的 patch cache 变成论文实验可追溯的数据层。证据路径：

```text
outputs/carnet/spcarnet/object_index_v1.json
docs/car_model/spcarnet_stage1_object_cache_report.md
```

**Stage 2：canonical shape-field autodecoder**

Stage 2 训练一个对象级 implicit shape field。每个 object 有一个 latent code，decoder 学习从 latent 和 query point 预测 occupancy/SDF-like field。它是想提供一个“完整形状先验”的核心表示。

当前事实需要诚实区分：

- train-split eval 能抽取 mesh：`100/100`；
- train recon chamfer 为 `0.0691737`，IoU 为 `0.51573`；
- 旧 held-out val eval 没有 z-only MAP fitting，因此跳过所有 val objects：`0/206`，结果含 `NaN`；
- 2026-06-24 已补上 clean-val z-only MAP-fit eval：`206/206` extracted，strict JSON，无 `NaN` / `Infinity`；
- v3 full-val MAP-fit 指标为：recon chamfer `0.0698447353`，hidden chamfer `0.1023846301`，filled IoU `0.5531548112`，shell IoU `0.9112784961`；
- v4 normal-band objective 新增 surface-normal band supervision：`x_inner = x_surface - epsilon * normal` 作为 occupied，`x_outer = x_surface + epsilon * normal` 作为 free，用 BCE 强化 Marching Cubes 的 `0.5` surface crossing；
- v4 完整训练已完成：`69300` steps / `300` epochs，W&B run `dysg8508`，`checkpoint_last.pt` 已落盘；
- 当前最好的 v4 full-val 证据来自 epoch50 checkpoint：`206/206` extracted，recon chamfer `0.0607328202`，hidden chamfer `0.0933915632`，filled IoU `0.5683319216`，shell IoU `0.8783071888`，normal consistency `0.7195177524`，W&B run `4wu9w305`；
- final checkpoint full-val eval 也已完成：`206/206` extracted，recon chamfer `0.0655826944`，hidden chamfer `0.0963624408`，filled IoU `0.5314717742`，shell IoU `0.8563237802`，normal consistency `0.6890807638`，W&B run `q1jjwvdm`；它比 epoch50 差，因此作为后期训练退化诊断保留；
- 2026-06-24 已新增 validation-driven checkpoint selector，正式选择 `v4_epoch50` 为 best available checkpoint；selector 状态为 `BEST_AVAILABLE_GATE_FAIL_WITH_LATE_DEGRADATION`，说明“选 best”已经工程化，但 Stage 2 仍没有过质量 gate；
- 相比 v3，v4 epoch50 改善了 recon chamfer `-0.0091119151`、hidden chamfer `-0.0089930669`、filled IoU `+0.0151771104`，但 shell IoU 下降 `-0.0329713073`；
- 结论：Stage 2 eval 接口已经修复，v4 是真实质量改进，但 gate 仍是 soft FAIL，不能作为 headline shape-prior result。

证据路径：

```text
ss3dm_prior/training/spcarnet_autodecoder.py
ss3dm_prior/models/spcarnet_shape_field.py
outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v3/fit_summary.json
outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json
outputs/carnet/spcarnet/autodecoder_v3/eval/val_eval.json
outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json
W&B run: svtbc8sn
configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder_v4_band.yaml
configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder_v4_band.yaml
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
W&B runs: dysg8508, 4wu9w305, q1jjwvdm
```

**Stage 3：posterior encoder**

Stage 3 用 partial observation 编码出 latent posterior，让系统从可见点推断完整车体形状。它解决的是“只看局部观测，如何进入 shape latent space”的问题。

当前 val 结果：

```text
n_extracted = 206 / 206
recon_chamfer = 0.066391
hidden_chamfer = 0.099075
visible_preservation_error = 0.062681
free_space_violation = 0.033535
IoU = 0.470897
```

证据路径：

```text
ss3dm_prior/models/spcarnet_posterior.py
ss3dm_prior/training/spcarnet_posterior.py
outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt
outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json
```

**Stage 4：observation-aware latent MAP refinement**

Stage 4 不直接改 decoder，而是在观测约束下优化 latent。它利用 visible/free-space observation loss，让 posterior latent 更贴近当前对象证据。

50-object val 结果：

| metric | before | after | trend |
|---|---:|---:|---|
| free-space violation | `0.035820` | `0.014688` | 明显改善 |
| visible error | `0.064352` | `0.060971` | 小幅改善 |
| recon chamfer | `0.071490` | `0.069032` | 小幅改善 |

这说明 observation MAP 是有效模块，但几何主指标收益仍有限。

证据路径：

```text
ss3dm_prior/losses_spcarnet_observation.py
scripts/car_model/refine_spcarnet_latent_map.py
outputs/carnet/spcarnet/map_refinement/val_50_default/refinement.json
docs/car_model/spcarnet_stage4_observation_map_implementation_report.md
```

**Stage 5：multi-hypothesis sampling and reranking**

Stage 5 让 posterior 采样多个 latent hypothesis，再用可见观测、free-space、prior score 等信号 rerank。它的目标是避免单一 posterior mean 被遮挡歧义困住。

当前结论：

- K=8 full-val visible rescoring 可抽取 `206/206`；
- recon chamfer `0.062587`，hidden chamfer `0.094253`，free violation `0.03217`；
- oracle best-of-K 为 `0.061317`，说明多假设存在上界收益；
- 但 inference-only reranking 还没有稳定吃到 oracle headroom。

证据路径：

```text
scripts/car_model/eval_spcarnet_multihypothesis.py
scripts/car_model/rescore_spcarnet_multihypothesis.py
outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_visible_rescored.json
docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md
```

### 9.3 与当前 MeshSplatting 修复主线的关系

两条线的关系可以这样讲：

| 主线 | 当前成熟度 | 作用 |
|---|---|---|
| Phase-J scene-level guarded ELA + compaction | 最高，full9 已闭环 | 当前 mentor 汇报和论文主结果候选 |
| Surface residual atlas / v64-v77 | 接口完整但收益小 | 把 Phase-J 收益内化到 persistent representation 的过渡路线 |
| SPCarNet object-level shape prior | Stage 1/3/4/5 有证据，Stage 2 v4 normal-band 相比 v3 有真实改善但质量仍 soft FAIL | 下一阶段加强几何先验、out-of-trajectory 稳定性和对象级补全的技术储备 |

PPT 中建议只用 1 页作为“下一阶段 shape prior backbone”，不要把它包装成已经全面胜出的主结果。最稳表述：

> 当前场景级结果已经证明 surface evidence 能修复和压缩 MeshSplatting。对象级 SPCarNet 进一步提供一个完整形状先验方向；Stage 2 held-out eval 接口已修复，v4 normal-band 也带来清晰改善，但质量 gate 仍未过，所以这条线目前是 future/core upgrade，而不是 headline result。

---

## 10. 定性结果与 PPT 图选择

全图对比经常视觉差异不明显，这是方法本身当前的展示短板：SPCarNet 的大部分收益来自 residual-level 局部修复和 LPIPS/SSIM 细节改善，直接 full-frame 肉眼看不一定显著。

因此 PPT 建议使用三类图：

| 图类型 | 用途 | 推荐路径 |
|---|---|---|
| Full-frame fair comparison | 证明同 test view、同 baseline、不是 cherry-pick 局部裁剪 | `assets/spcarnet_m360_full9_qualitative_gallery.png` |
| Phase-J where-it-helps local crops | 展示最直观的局部误差下降，是首选视觉证据 | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| Outdoor detail crops | 回应 outdoor 场景质疑 | `assets/spcarnet_m360_outdoor_detail_showcase.png` |
| Mixed local crops | 展示 indoor/outdoor 都有局部收益 | `assets/spcarnet_m360_where_it_helps_showcase.png` |
| v56/v64 representation diagnostic | 展示 representation-level 改进非常细微，作为 honest ablation | `assets/spcarnet_v56_counter_face_alpha_guard_panel.png` |

PPT 中要避免只放全图，因为观众会觉得“看不出来”。建议放：

```text
GT / clean MeshSplatting / SPCarNet / clean error / ours error / delta heatmap
```

并在每个 crop 下方写 local MAE drop 或 local dPSNR，而不是只写 full-view PSNR。

现有 qualitative crop 统计示例：

| qualitative crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| bonsai / `00001.png` | +6.63 / +0.0452 / -0.0878 | +11.79 | 78.6% |
| kitchen / `00011.png` | +3.43 / +0.0250 / -0.0578 | +10.48 | 71.4% |
| room / `00011.png` | +3.50 / +0.0220 / -0.0656 | +10.36 | 67.7% |
| counter / `00013.png` | +2.17 / +0.0407 / -0.0665 | +6.02 | 54.9% |
| garden / `00006.png` | +1.74 / +0.0479 / -0.0678 | +4.26 | 44.4% |
| flowers / `00014.png` | +1.12 / +0.0754 / -0.1028 | +2.15 | 25.3% |

---

## 11. 消融与负结果如何讲

这部分对 mentor 很重要，因为它证明我们不是只在调参。

| 变体 | 想验证什么 | 结论 |
|---|---|---|
| Clean MeshSplatting `26000/30000` | baseline 是否被公平选择 | clean `26000` 在 full9 selected-clean envelope 中更强 |
| Compact-only checkpoint | 删面本身是否足够 | 几何安全，但 RGB 提升不够 headline |
| Compact + ELA without SSIM-peak guard | scalar score 是否足够 | 会在 room 等场景出现 SSIM 风险 |
| Compact + guarded ELA | 当前 Phase-J policy | full9 selected-clean 三指标全胜 |
| v48/v52/v56/v64 surface atlas policies | representation-level 内化是否可行 | 有真实接口和小正收益，但效果量级不足 |
| v69-v76 prior/certificate chain | low-support bin 是否可以靠 prior 修复 | 支持当前 bottleneck 诊断：prior 容易过拟合 policy-val local bins |
| Optional FD gate | perceptual Frechet gate 是否有帮助 | 有 LPIPS 导向收益但 PSNR/SSIM tradeoff，不作为默认主方法 |

建议讲法：

> Phase-J 证明“surface evidence 可以显著改善 MeshSplatting 输出”；v64-v77 证明“要把这个收益完全内化成 persistent surface representation，不能只靠 alpha/blend/cap，需要更强的 residual basis 和 target-view 泛化证书”。

---

## 12. 当前是否全面超越 MeshSplatting baseline？

需要分口径回答：

| 问题 | 回答 |
|---|---|
| 相对本地 selected-clean MeshSplatting，Phase-J 是否 RGB 全面超越？ | 是。full9 9/9 场景 PSNR/SSIM/LPIPS 三指标严格胜出。 |
| 是否同时有 geometry/triangle 收益？ | 有。平均删去 7.6479% triangles，9/9 geometry-safe；但 strict all-axis geometry pass 不是 9/9。 |
| 是否已经完全是 baked representation method？ | 不是。最强 endpoint 仍是 render-time guarded ELA portfolio。 |
| v64/v76/v77 是否是新的 headline？ | 不是。v64 是最稳表示级候选但量级小；v76 未晋级；v77 完成后也未晋级。 |
| 是否可直接 claim 超过 MeshSplatting paper table？ | 可以作为辅助 audit 说已有早期版本相对 paper table 为正，但论文最终表必须重新严格对齐 paper protocol。 |

---

## 13. 当前短板

1. **Full-frame 视觉提升不总是直观**  
   指标提升很强，但很多全图差异是 residual-level 的。PPT 需要用局部 crop、error map 和 local MAE drop。

2. **表示级内化还没有形成大收益**  
   v64-v77 已经补了大量接口和安全证书，但 persistent residual atlas 的收益仍远小于 Phase-J render-time ELA。

3. **Triangle reduction 仍偏保守**  
   7.65% 是安全、诚实的平均压缩，但对于“压缩论文”的视觉冲击可能不够强。室内场景尤其受 micro-prune guard 限制。

4. **Strict geometry all-axis 还不是 9/9**  
   当前可以讲 geometry-safe 9/9，但不能讲所有几何指标在所有场景 strict win。

5. **Paper-table 对齐还需最终复核**  
   当前本地主表最稳。与 MeshSplatting 论文表的同口径比较，需要在论文前确认 split、mask、metric implementation、iteration、checkpoint envelope 全部一致。

6. **对象级 shape prior 还没有闭合成主结果**  
   Stage 1/3/4/5 已经有实证结果；Stage 2 autodecoder held-out eval 接口已经修复到 `206/206` extraction。v4 normal-band 的 epoch50 checkpoint 把 full-val chamfer 从 `0.0698447` 降到 `0.0607328`，filled IoU 从 `0.553155` 提到 `0.568332`；final checkpoint 又回退到 chamfer `0.0655827`、filled IoU `0.531472`，说明后期训练有退化。现在已有 selector artifact 固定选择 `v4_epoch50`，但 Stage 2 仍未过原 gate，不能作为强 shape-field 先验来讲。

---

## 14. 下一步技术路线

建议下一阶段不要继续做纯参数扫描，而是把工作推进到 representation-level 的强策略：

1. **更强 residual representation**  
   从 face/bin constant residual 升级到 low-rank/view-conditioned/local-basis residual field，但必须带 target-view support certificate。

2. **target-footprint generalization certificate**  
   当前 policy-val bin positive 不足以保证 target positive；需要把 target view footprint、multi-view support、view-direction diversity 和 local residual stability 一起做证书。

3. **stronger visual evidence mining**  
   为 PPT 和论文主动生成 error-reduction crops、delta heatmaps、local MAE tables，让视觉收益更可见。

4. **paper-protocol replication**  
   最终必须重新跑同口径 MeshSplatting baseline 和 SPCarNet endpoint，固定 checkpoint selection，保证能直接写进论文主表。

5. **representation-level ablation table**  
   把 v48/v52/v56/v64/v69-v77 组织成一张逻辑清晰的消融表：从 support/capacity/alpha/prior/certificate 一路证明为什么需要下一代 basis。

6. **improve Stage 2 held-out quality**  
   val extraction 和 JSON `NaN` 问题已经修复；下一步是提升 shape-field objective / representation，使 MAP-fit chamfer 和 filled IoU 过 gate。只有 Stage 2 质量闭合后，对象级 shape prior 才能进入主论文故事。

---

## 15. 建议 PPT 结构

建议 13 页：

1. Title：SPCarNet: Self-Auditing MeshSplatting for Surface Repair and Compaction
2. Problem：MeshSplatting still has residual errors and redundant geometry
3. Key Idea：training-view surface evidence certifies safe repair and compaction
4. Pipeline：MeshSplatting -> evidence cache -> compaction -> guarded repair -> policy gate
5. Evidence Cache：face/bin/residual/support/risk 可视化
6. Geometry-Safe Compaction：删面原则和 triangle reduction
7. Guarded ELA：residual transfer 公式和 fallback
8. Main Quantitative Result：full9 9/9 wins table
9. Qualitative Result：Phase-J local crop/error-map showcase
10. Representation-Level Track：v64 positive but small; v76 negative diagnostic
11. Object-Level Shape Prior：Stage 1-5 status and why it is future backbone
12. Limitations：render-time endpoint、视觉差异、geometry all-axis、paper-protocol、Stage 2 quality gate
13. Next Step：strong target-footprint certificate + stronger surface residual basis

---

## 16. Mentor 可能会问的问题

### Q1：这是不是在 test set 上调参？

不是。SPCarNet 的 branch、alpha、support、fallback 来自 train/policy-val evidence。held-out test 只用于最终报告。clean baseline 用 held-out test 选择 strongest envelope，是为了不低估 baseline，不参与我们方法调参。

### Q2：每个场景是不是有一套手动参数？

Phase-J portfolio 会按 train-only evidence 自动选择分支，不是人工按 test 结果调场景参数。v64 以后也在推动固定 policy：用 evidence 条件自动接受或回退，不写场景名。

### Q3：第三步是在训练一个模型吗？

Phase-J 主 endpoint 更准确地说是 train-evidence based render-time residual repair，不是重新训练一个神经网络。Surface residual atlas/v64-v77 是 representation-level 内化路线，已经进入 train/eval pipeline，但目前效果量级仍小。

### Q4：out-of-trajectory 会不会崩？

这是主要风险之一，所以方法有 support、visibility、tail-risk 和 fallback。证据不足时不强行修复。当前结论是本地 held-out split 安全，但 out-of-trajectory 仍需要更强 target-footprint certificate。

### Q5：为什么定性图看起来不总明显？

因为很多改进是局部 residual-level 的；full-frame 下很容易被缩放隐藏。应展示 error map、local crop 和 local MAE drop，而不是只放整图。

### Q6：现在离顶会论文还差什么？

Phase-J 的工程闭环和指标很强，但顶会论文还需要更强 representation-level 核心贡献、paper-protocol 复现、更多跨数据集验证，以及更直观的视觉展示。

### Q7：对象级 SPCarNet shape prior 是不是已经解决了几何问题？

还没有。Stage 1 数据层、Stage 3 posterior、Stage 4 MAP 和 Stage 5 multi-hypothesis 已经有有效结果；Stage 2 的 held-out eval 接口已经修复为 `206/206` extraction，但 chamfer/filled IoU 没过原 gate，所以现在只能作为下一阶段几何先验方向，不能作为已经闭合的主结果。

---

## 17. 证据索引

主结果：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
```

当前完整报告与 README：

```text
README.md
README.zh.md
docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CurrentMethod-Full.zh.md
docs/car_model/6-24-SPCarNet-PaperLoop-Closure-Audit-and-SlidePlan.zh.md
outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
outputs/carnet/spcarnet/current_evidence_manifest_20260624.json
```

v64/v75/v76 诊断：

```text
docs/car_model/6-24-v64-FixedAutoBinAlphaPolicy-Log.md
docs/car_model/6-24-v75-LocalPatchPrior-Log.md
docs/car_model/6-24-v76-PolicyValBinGainHybrid-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v64_bin_alpha_auto_policy_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v76_policyval_bin_gain_hybrid_20260624/summary.md
```

推荐 PPT 图片：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
```

最新完成诊断：

```text
v77 stricter multi-view bin-gain hybrid policy
W&B run: 3ho2y4s1
W&B URL: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/3ho2y4s1
Result: 26.753528595 / 0.862111032 / 0.251881331
Decision: completed negative diagnostic, not promoted
Persistent artifacts:
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/summary.md
docs/car_model/6-24-v77-StrictBinGainHybrid-Log.md
```

SPCarNet Stage 1-5 object-level shape-prior evidence：

```text
docs/car_model/spcarnet_stage1_object_cache_report.md
docs/car_model/spcarnet_stage2_shape_field_implementation_report.md
docs/car_model/spcarnet_stage4_observation_map_implementation_report.md
docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md
outputs/carnet/spcarnet/object_index_v1.json
outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json
outputs/carnet/spcarnet/autodecoder_v3/eval/val_eval.json
outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json
outputs/carnet/spcarnet/map_refinement/val_50_default/refinement.json
outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_visible_rescored.json
```

---

## 18. 最终汇报口径

最稳妥、最不容易被 mentor 追问击穿的总结是：

> 目前 SPCarNet 已经证明，在 MeshSplatting 之上引入 train-evidence surface audit 可以同时带来 RGB 提升和安全删面。Phase-J 在本地 Mip-NeRF360 full9 selected-clean baseline 上 9/9 场景三指标全胜，且平均减少 7.65% triangles。我们已经把下一阶段 representation-level 路线推进到 v64-v77：接口、policy、W&B 验证和负结果诊断都很完整，但 persistent surface residual field 的收益仍小。对象级 shape-prior 线也已经跑通 Stage 1/3/4/5；Stage 2 held-out eval 接口已修复为 `206/206` extraction，v4 normal-band 进一步把 chamfer/filled-IoU 往正确方向推进，但质量 gate 仍 soft FAIL。下一步的真正论文级突破点不是继续调参，而是更强的 surface residual basis、target-footprint generalization certificate 和更高质量的 object-level shape prior。

# v106 POD-MoE Paper Story Draft

Date: 2026-06-25

Scope: review and paper-story synthesis for the current v106 POD-MoE base-preserve candidate. This draft is meant to be pasted into the main report or PPT. It is conservative: v106 base-preserve has passed the current hard-triad comparison against v104c, but it should not become the paper-final headline until full9 validation and qualitative panels are complete.

## 1. 通俗方法描述：和原始 MeshSplatting 的区别

Clean MeshSplatting 的故事很简单：

```text
trained MeshSplatting checkpoint
-> direct test rendering
-> PSNR / SSIM / LPIPS
```

它主要相信训练后的 mesh/splat 表示本身。它不会显式回答这些问题：

- 哪些可见三角形区域有稳定的多视角残差信号？
- 哪些残差是可重复的 surface-local error，哪些只是某个视角的偶然修复？
- 在纹理细节和遮挡边界附近，应该加 residual，还是回退到更保守的 base render？

SPCarNet 的 v106 POD-MoE 线把 MeshSplatting 的 surface 当作地址空间：

```text
base MeshSplatting render
+ surface-addressed residual field
+ certified detail / occlusion-boundary expert corrections
-> adapted render
```

和 v104c 相比，v106 不是继续把所有 endpoint delta 压进一个低阶 per-triangle 函数。v104c 是一个 v104c-like shrink view-affine base：

```text
[1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

v106 在这个 base 上增加两个 residual experts：

- detail expert: 针对纹理、亮度梯度、高频细节区域；
- occlusion-boundary expert: 针对 triangle id / depth 边界附近的残差模式。

直觉上，v104c 是“每个三角形一个压缩 residual 函数”；v106 是“先保留 v104c 的稳态 base，再让有证据的专家只在细节和边界处做小幅补偿”。这正好对应当前 endpoint-to-field gap：endpoint 能做 per-view/per-pixel support/fallback，v104c 把这些都压扁成单一低阶函数，v106 尝试把 residual mode 分开。

## 2. 技术模块

### 2.1 Policy

当前 v106 fixed policy:

```text
field_variant: pod_moe
basis_type: affine_barycentric_viewdir_pod_mixture
builder_variant: v106_perceptual_occlusion_detail_moe
base: v104c_like_shrink_view_affine
experts: detail, occlusion_boundary
gate_source: normal_equation
expert_mse_certificate: weighted_normal_equation_lambda_star
pod_base_keep_mode: base_preserving_boundary
renderer_scaling: 4
residual_dtype: float16
ridge: 0.001
residual_clip: 0.08
view_std_floor: 1e-4
rank_rtol: 1e-7
condition_max: 1e8
view_gate_temperature: 0.0
```

Policy 层的作用不是挑某个场景的最好参数，而是锁定同一套表示、证书和渲染规则，再做 counter, kitchen, bonsai 以及后续 full9 的固定策略验证。

### 2.2 POD-MoE field

Field 里存的是 surface-attached tensors，而不是临时后处理：

```text
triangle_base_coefficients [T, 6, 3]
triangle_expert_delta_coefficients [T, 2, 6, 3]
triangle_expert_reliability [T, 2]
triangle_expert_mse_scale [T, 2]
triangle_occlusion_base_keep [T]
triangle_view_means / triangle_view_scales
triangle_counts / triangle_view_counts
```

构建时先拟合 v104c-like base，再分别在 detail cue 和 boundary cue 加权的 normal equations 上拟合两个 expert。expert 存的是相对 base 的 delta，而不是替代 base 的完整残差。

### 2.3 Base-preserve rendering

旧 POD-MoE 的问题是 boundary expert 可能在遮挡边界处过度替换或压制 base residual，导致 PSNR/MSE 方向变坏。base-preserve 的关键改动是：

```text
base_keep = 1.0
rendered_residual = base_residual + weighted_detail_delta + weighted_boundary_delta
```

也就是说，边界附近不再让 boundary cue 把 v104c-like base 拿掉。专家只作为 additive residual correction 参与。这个改变很小，但机制上很重要：它把 v104c 的稳定性当作下界，把 MoE 当作细节/边界补偿层。

### 2.4 MSE certificate / evidence gate

v106 的 expert 权重不是裸 expert 输出。构建时每个 expert 都有几类约束：

- support: 由对应 cue 加权后的 triangle counts / view counts 提供；
- gain: normal-equation proxy 上的 weighted risk gain；
- debt guard: expert coefficient 和 base coefficient 偏离过大时降权；
- MSE scale: `weighted_normal_equation_lambda_star` 证书给出的 expert MSE scale；
- runtime cue: 渲染时用 image detail score 和 boundary score 决定哪些像素激活哪个 expert；
- view gate: 用 view direction 和训练统计约束 OOD view direction，当前温度为 0。

渲染时权重近似为：

```text
weight_expert = runtime_cue * expert_reliability * expert_mse_scale * view_gate
mixture = weight / (1 + sum(weight))
```

这使得 expert 不是“看到边界就修”，而是“该 triangle 的该 expert 在证据、风险和 MSE proxy 上都可接受时才小幅修”。

### 2.5 Delta-MSE diagnosis

delta-MSE 是当前最重要的 reviewer-facing diagnostic。它比较 candidate render 和 base render 对 GT 的 MSE：

```text
delta_mse = MSE(candidate, GT) - MSE(base, GT)
          = 2 * (base - GT) * (candidate - base)
            + (candidate - base)^2
```

这里 base 是 v104c，candidate 是 v106。这个诊断能回答一个关键问题：v106 的微小 SSIM/LPIPS 提升是不是用 MSE-positive drift 换来的。

旧 POD-MoE/debtguard/cert 在 counter 上只有 `4 / 30` views 的 MSE 优于 v104c，`26 / 30` 变坏。base-preserve 后变为 `23 / 30` views 改善，mean delta-MSE 变为 `-0.00000026`。hard triad 汇总是 `89 / 102` views 改善。

## 3. 为什么是研究贡献而不是纯调参

可以把贡献写成一个逐层闭环，而不是“调了一组 gate 超参数”：

1. Representation change: 从 clean MeshSplatting 的直接渲染，升级到 surface-addressed residual field；从 v104c 的单一低阶 residual，升级到 base plus detail/boundary expert residual mixture。
2. Mechanism-level safety: base-preserve 明确保护 v104c-like base，不让边界专家替代稳定 base，这是一个 failure-mode repair。
3. Certificate-carrying experts: expert reliability 由 support、normal-equation gain、debt guard、MSE scale 共同决定，不是单一阈值。
4. Fail-closed artifact identity: runner 检查 field type、basis type、builder variant、sha256、renderer scaling、gate source、residual dtype 等，避免 stale artifact 或 silent fallback。
5. Diagnosis beyond aggregate metrics: delta-MSE 把 “PSNR 为什么掉” 分解成 cross term 和 delta energy，能定位 bad views，而不是只报均值。
6. Research hypothesis is falsifiable: 如果 full9 fail，下一步不是继续扫参数，而是 v107 MSE-descent-locked POD-MoE，用 per-triangle two-expert box QP 约束 expert ray 和 v104c base 的 MSE-descent alignment。

这套贡献的论文表述应是：

> SPCarNet turns MeshSplatting's explicit surface into an auditable residual carrier. v106 POD-MoE preserves a stable v104c-like base while attaching certified detail and occlusion-boundary residual experts, then validates whether those experts are MSE-aligned rather than merely metric-favorable.

## 4. 当前证据与不足

### 4.1 定量证据

Stable anchor: v104c full9 已完成，9/9 scenes 对 clean MeshSplatting 三指标均提升。

| method | scenes | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| clean MeshSplatting | 9 | 25.151682 | 0.749018 | 0.287621 |
| v104c shrink view-affine field | 9 | 25.829099 | 0.760727 | 0.268548 |
| v101/v102 endpoint/reference | 9 | 26.481310 | 0.783675 | 0.224305 |

v104c vs clean:

```text
+0.677417 PSNR / +0.011709 SSIM / -0.019073 LPIPS
```

v104c still trails endpoint/reference:

```text
-0.652211 PSNR / -0.022949 SSIM / +0.044243 LPIPS
```

v106 base-preserve hard triad against v104c:

| scene | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| counter | +0.001577 | +0.000102 | -0.000139 |
| kitchen | +0.001595 | +0.000062 | -0.000206 |
| bonsai | +0.005213 | +0.000154 | -0.000136 |
| mean | +0.002795 | +0.000106 | -0.000160 |

Delta-MSE:

| scene | views | MSE improved | MSE worse | mean delta-MSE |
|---|---:|---:|---:|---:|
| counter | 30 | 23 | 7 | -0.00000026 |
| kitchen | 35 | 30 | 5 | -0.00000050 |
| bonsai | 37 | 36 | 1 | -0.00000017 |
| hard triad total | 102 | 89 | 13 | negative mean |

Interpretation:

- v106 base-preserve is the first POD-MoE variant that strictly beats v104c on counter, kitchen, and bonsai across PSNR, SSIM, LPIPS.
- The gains are real-sign but small. This is a candidate-method win, not yet a paper-final headline.
- MSE-direction evidence is stronger than the aggregate metric deltas: old/debtguard/cert were MSE-worse on most counter views; base-preserve flips the direction.

### 4.2 定性证据

Current weakness: v106-specific qualitative panels are not yet sufficient for paper. The report should say this explicitly.

Required qualitative panels:

```text
GT | clean MeshSplatting | v104c | v106 base-preserve | endpoint/reference | error maps
```

Use three buckets:

- best MSE-improved views: counter `00008.png`, `00010.png`, `00002.png`; kitchen and bonsai equivalents from their delta-MSE reports;
- worst MSE views: counter `00009.png`, kitchen `00015.png`, bonsai `00028.png`;
- detail/boundary crops where expert activation should be visually interpretable.

Panel rule:

- show absolute-error and delta-error maps, not only RGB crops;
- caption every crop with scene, view id, crop coordinates, and local metric deltas;
- keep endpoint/reference separate as ceiling, not as v106 result.

### 4.3 几何压缩与 triangle story

v106 POD-MoE currently does not mutate topology. Its geometry story is “inherits the base MeshSplatting/compact-parent geometry while improving appearance residuals,” not “adds new triangle-count reduction.”

What is supported:

- v100/v101 endpoint packaging preserves inherited topology and geometry. For counter, v100 records topology unchanged, `9,644,247` triangles, `2,478,825` vertices, endpoint delta `0`, geometry inherited, depth AbsRel `0.007637892`, normal mean angle `27.085450`.
- v101 manifest explicitly marks “v101 adds new geometry/triangle-count gains beyond the compact parent” as not supported. Triangle/geometry gains come from compact parent and Phase-J pipeline, not the endpoint packaging itself.
- PRISM/MeshPrior is a separate auditable topology-control line. It supports topology reduction under scene-evidence gates on selected scenes, but it is not automatically merged with v106 POD-MoE.

How to preserve the geometry/triangle story:

- treat v106 as the appearance residual field module;
- report topology and geometry inheritance for the exact parent checkpoint used by v106;
- if using compact parent rows, include clean compact parent, v104c, v106, endpoint/reference in the same table;
- do not claim v106 itself compresses geometry unless a same-checkpoint v106 plus topology-control run is actually validated.

### 4.4 跨场景证据

Current cross-scene status:

- v104c: full9 complete, 9/9 scenes over selected clean MeshSplatting.
- v106 base-preserve: hard triad complete, 3/3 scenes over v104c with consistent signs.
- v106 full9: not complete yet. Remaining scenes are `bicycle`, `flowers`, `garden`, `room`, `stump`, `treehill`.

Minimum promotion gate:

- full9 present and ok;
- mean metrics exceed v104c;
- no scene has severe PSNR/MSE regression;
- if any scene regresses, provide delta-MSE and qualitative diagnosis showing whether it is a view-specific outlier or scene-class failure;
- durable artifacts copied or regenerated under `outputs/`, not only `/dev/shm`.

### 4.5 Claim boundaries

Safe current claim:

> v106 base-preserve is a fixed-policy hard-triad POD-MoE candidate that improves over the v104c surface-field anchor on PSNR, SSIM, LPIPS, and MSE direction, while retaining the v104c base residual.

Unsafe current claims:

- v106 is the final full9 headline.
- v106 closes the endpoint gap.
- v106 is train-only unseen-camera generalization.
- v106 provides new triangle-count reduction.
- visual improvement is obvious in every full-frame render.

Important fairness boundary:

The field line uses v102 target-camera endpoint deltas as teacher. It uses no held-out target GT for policy, but it is not a fully train-only unseen-camera representation claim.

## 5. 下一步 full9 完成后，README / 技术报告应该怎么组织

### 5.1 README update structure

Do not bury v106 in chronological logs. Organize the README around claim layers:

1. Current headline status
   - Best quality endpoint: v101/v102 endpoint/reference.
   - Best validated field before full9: v104c full9.
   - Current candidate: v106 POD-MoE base-preserve hard triad.

2. Method ladder
   - clean MeshSplatting;
   - v101/v102 endpoint/reference;
   - v103 affine field;
   - v104a view-affine field;
   - v104c shrink view-affine field;
   - v105 residual-mixture negative/diagnostic;
   - v106 POD-MoE base-preserve.

3. Main evidence table
   - full9 clean vs v104c vs v106 vs endpoint/reference;
   - per-scene deltas vs clean and vs v104c;
   - strict scene win counts;
   - endpoint gap.

4. Diagnostics table
   - delta-MSE per scene;
   - MSE-improved/worse view counts;
   - worst view ids;
   - expert support fractions and MSE scale means.

5. Geometry/topology table
   - parent checkpoint triangle/vertex counts;
   - whether v106 changed topology;
   - geometry metrics if available;
   - separate PRISM/topology-control evidence if included.

6. Qualitative assets
   - path to v106 crop/error-map panels;
   - manifest with scene/view/crop coordinates;
   - statement that endpoint/reference is a ceiling row.

7. Claim boundary
   - no target-GT policy use;
   - target-camera endpoint distillation caveat;
   - not a vanilla checkpoint without render.py surface-field support;
   - no speed claim unless benchmarked.

### 5.2 Technical report structure

Recommended report sections:

1. Executive summary
   - one paragraph on v104c full9 anchor;
   - one paragraph on v106 hard-triad or full9 outcome;
   - one paragraph on remaining endpoint gap.

2. Problem formulation
   - clean MeshSplatting direct render;
   - endpoint repair quality ceiling;
   - field distillation bottleneck.

3. Method
   - surface address space;
   - v104c base;
   - POD-MoE experts;
   - base-preserve rendering equation;
   - MSE certificate and evidence gates;
   - artifact identity checks.

4. Experiments
   - datasets and scenes;
   - clean baseline definition;
   - v104c, v106, endpoint/reference rows;
   - full9 protocol;
   - no held-out GT policy statement.

5. Results
   - full9 aggregate;
   - per-scene table;
   - hard-triad historical ladder;
   - endpoint gap.

6. Diagnostics and ablations
   - old POD-MoE vs debtguard vs cert vs base-preserve;
   - delta-MSE;
   - expert support/reliability;
   - worst-view failure analysis.

7. Geometry and topology
   - inherited geometry;
   - compact-parent / PRISM relation;
   - what v106 does and does not claim.

8. Qualitative evidence
   - crop/error-map panels;
   - best and worst MSE views;
   - detail and occlusion-boundary examples.

9. Limitations and next step
   - target-camera teacher caveat;
   - full9 status if pending;
   - small effect size;
   - need durable artifacts;
   - v107 MSE-descent-locked POD-MoE if full9 exposes regressions.

### 5.3 Slide-ready one-line narrative

If full9 passes:

> v106 POD-MoE upgrades v104c from one shrink view-affine residual into a base-preserving mixture of certified detail and occlusion-boundary experts, improving the fixed-policy surface-field line while retaining MeshSplatting's geometry story.

If full9 is mixed:

> v106 POD-MoE validates the residual-mode separation hypothesis on the hard triad, but full9 reveals where expert MSE alignment or scene-class generalization still fails; v104c remains the stable headline.

If full9 fails:

> POD-MoE remains a useful diagnostic: naive expert capacity is not enough unless expert rays are explicitly MSE-descent aligned with the v104c base.


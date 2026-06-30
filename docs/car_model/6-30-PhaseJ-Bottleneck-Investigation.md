# 2026-06-30 Phase-J Bottleneck Investigation

本文回答一个核心问题：为什么 v169 后的一系列 SPCarNet 表示升级一直卡在
Phase-J flowers gate 之下，以及这说明当前路线到底出了什么问题。

## 结论先行

当前不是所有指标都输给 Phase-J。在 flowers exact 当前口径下，v292/v293 的
SSIM 和 LPIPS 都已经明显优于 Phase-J gate 数值，真正卡住的是 PSNR：

| run | PSNR | SSIM | LPIPS | vs Phase-J PSNR |
|---|---:|---:|---:|---:|
| parent | 19.832054 | 0.619910 | 0.180335 | -0.472304 |
| v292d balanced frontier | 19.851452 | 0.620343 | 0.180212 | -0.452906 |
| v293a best PSNR frontier | 19.853420 | 0.620328 | 0.180312 | -0.450938 |
| Phase-J flowers reference | 20.304358 | 0.557770 | 0.329222 | 0.000000 |

所以目前的瓶颈更精确地说是：

> 我们能做出 target-blind、no-GT、MeshSplatting-compatible 的小幅
> all-axis 修复，但没有把 Phase-J 的 PSNR 级别残差能量可靠地蒸馏到
> target views。

## MSE 量级诊断

PSNR 的 0.45 dB 差距不是小数点误差。按当前 flowers target 平均值换算：

| item | PSNR | MSE |
|---|---:|---:|
| parent | 19.832054 | 0.010394285 |
| v293a | 19.853420 | 0.010343274 |
| Phase-J | 20.304358 | 0.009323183 |

v293a 相比 parent 的 MSE reduction 约为 `5.10e-05`；Phase-J 相比 parent 需要
`1.071e-03`。也就是说，当前 v293a 只拿到了大约 `1 / 21` 的所需 MSE 改善。

这解释了为什么很多实验看起来“有进展”，但始终不接近 Phase-J：我们每次
只在非常小的 residual 能量上做微调，缺的不是 0.001 的超参数，而是一个数量级
以上的有效、正确、可泛化残差注入。

## Per-view 尾部诊断

v293a 的平均 PSNR 比 v292d 稍高，但 target 视角尾部更差：

| run | PSNR gain mean | PSNR min gain | SSIM positive frac | LPIPS positive frac | LPIPS tail CVaR |
|---|---:|---:|---:|---:|---:|
| v292d | +0.019398 | -0.000429 | 0.818182 | 0.727273 | -0.001627 |
| v293a | +0.021366 | -0.002855 | 0.681818 | 0.727273 | -0.002217 |
| v293b | +0.020934 | -0.001524 | 0.681818 | 0.636364 | -0.002110 |

最坏视角也高度一致，例如 v293a 的最坏 LPIPS 主要集中在 `00007`、`00003`、
`00008`、`00001`、`00018`。这说明问题不是随机噪声，而是 target-view
generalization 出现稳定失败模式。

## 表示与数据证据

v293 的 texture latent 是真实表示升级，但 surface texture 证据本身很稀疏：

| field | value |
|---|---:|
| candidate faces | 65,536 |
| total texture bins | 1,048,576 |
| covered bins | 425,867 |
| covered bin fraction | 0.406138 |
| mean bin count on covered bins | 8.878641 |
| mean luma sign consistency | 0.649574 |
| mean low-rank reliability | 0.503196 |

这组数值说明：每个 surface bin 的教师残差观测很少，方向一致性和低秩可靠性都不高。
在这种输入上简单增加 MLP、MoE 或 latent 容量，会优先提高 policy-val/局部 PSNR，
但也更容易把错误高频残差带到 target views。

## 根因判断

### 1. 当前方法学到的是弱残差，而不是 Phase-J 级别的强残差

v292d/v293a 的 mean changed fraction 约 `0.11`，active fraction 约 `0.12`。
这对安全修复是合理的，但若目标是追上 Phase-J 的 PSNR，当前改动覆盖和残差幅度明显不够。

更关键的是，我们为了保持 no-target-GT 和 perceptual safety，反复引入 gating、
confidence、view-support floor。这些机制能避免崩坏，却也把 residual energy 限制在
很小范围内。

### 2. policy-val 成功不能直接迁移到 target

v293a policy-val 是全轴正收益：

- PSNR gain `+0.028777`
- SSIM gain `+0.000981`
- LPIPS gain `+0.000665`

但 target exact 上变成：

- PSNR gain `+0.021366`
- SSIM gain `+0.000418`
- LPIPS gain `+0.000022`

这说明 policy-val 与 target views 的 view/visibility/support 分布仍不一致。
view-support gate 修好了 v290a 的明显感知退化，但还没有解决 residual direction
在 target 视角上的可靠预测。

### 3. 新容量增加了 PSNR，也增加了 perceptual tail risk

v293a 相比 v292d 的 PSNR 只提升约 `+0.001967 dB`，但 SSIM/LPIPS 尾部更差。
这说明 texture latent 的作用不是完全无效，而是没有被正确约束：它能携带更多细节，
但不知道哪些细节在 target view 上会变成错误纹理。

### 4. 当前 prompt 的硬 gate 很正确，但我们实际执行仍偏向工程补丁

v169 prompt 明确要求不要再做 alpha scan、footprint expansion、简单 gate。
后续确实做了 PatchViewMoE、lowrank view support、texture latent 等表示升级，
但它们仍然没有真正解决最核心的问题：

> teacher residual 到 target-view residual 的方向投影是否可信。

目前大部分机制是在 residual 产生后做筛选或缩放，而不是从训练阶段就建模
cross-view residual transport 的不确定性和多视图一致性。

## 反思

我们之前的很多进展是“系统可信度”的进展，而不是“质量闭环”的进展：

- no-target-GT apply、audit、W&B、policy-val、target exact 都更严谨了；
- clean baseline / Phase-J / v106 的口径更清楚了；
- 但质量层面仍停留在 parent 上 `+0.02 dB` 左右的修复，而 Phase-J 要求约
  `+0.47 dB` 的 PSNR 改善。

换句话说，当前框架已经能证明一个方法是否可靠，却还没有产生足够强的方法本身。
继续沿着“加一个 gate、换一个 threshold、加一点 latent dim”的路线，预期收益仍会很小。

## 下一步应当做什么

我认为下一步不能再以 v293 的小修为主线，而应该直接验证一个更强的假设：

1. 做 teacher-residual projection upper bound：
   - 不训练复杂模型，先测 Phase-J residual 在当前 surface carrier 上的理论可投影上限；
   - 若上限仍只有 `+0.02 dB`，说明 carrier 天花板过低；
   - 若上限接近 Phase-J，说明训练/regularization/policy 有问题。

2. 做 source-heldout residual direction test：
   - 对每个 face/UV/bin，从 train source views 中留出一部分视角；
   - 预测 heldout residual direction，而不是只拟合 residual magnitude；
   - 用这个误差训练 target-blind reliability，而不是 target 后验 gate。

3. 用多源一致性替代单点 texture latent：
   - latent 不能只按 bin 存参数；
   - 应当显式存 source-view residual set / basis / uncertainty；
   - target 视角由 source-view compatibility、visibility、normal/view direction 和 heldout
     consistency 一起决定 residual 权重。

4. 如果 upper bound 证明 surface carrier 不够，应果断升级 representation：
   - 从 face/UV-bin residual field 升级到 patch-level / image-space constrained
     neural surface renderer；
   - 或把 Phase-J 作为 differentiable/teacher endpoint，训练一个能直接重建
     Phase-J residual image patches 的 view-conditioned decoder。

## 当前版本状态

- 当前最好 PSNR frontier：v293a，`19.853420 / 0.620328 / 0.180312`。
- 当前最好 balanced frontier：v292d，`19.851452 / 0.620343 / 0.180212`。
- 两者都通过 no-target-GT audit。
- 两者都没有通过 Phase-J flowers gate，原因是 PSNR 仍差约 `0.451-0.453 dB`。
- full9 仍不应启动，除非 flowers exact 先超过 Phase-J PSNR gate。

Final status: NOT COMPLETE.

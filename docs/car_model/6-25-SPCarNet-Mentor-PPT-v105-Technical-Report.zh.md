# SPCarNet Mentor PPT Technical Report: v104c Baseline Closure and v105/v105b Update

Date: 2026-06-25

## 0. Executive Takeaway

当前最适合作为 PPT 主线的结论不是 v105，而是两层结果：

1. **v101/v102 endpoint line**: 这是当前质量上限。它在本地 Mip-NeRF360 full9 上相对 clean MeshSplatting 有显著提升，但它是 checkpoint-attached / preprojected endpoint sidecar，不是 vanilla checkpoint。
2. **v104c shrink view-affine surface field**: 这是当前最稳定的 representation-attached field。它在 full9 固定策略下 9/9 scenes 全部超过 clean MeshSplatting，但仍明显低于 v101/v102 endpoint 上限。

本轮新增 v105/v105b 的价值主要在工程和研究诊断：

- v105 新增了真实的 **evidence-gated residual-mixture surface field**，不是单纯扫参数。
- v105b 进一步把 gate 从 in-sample normal-equation gain 改成 **even/odd cross-fitted teacher-risk gate**，更接近公平科研机制。
- 但 v105/v105b 在 counter 上没有明确超过 v104c，因此当前不能作为 headline 方法扩展到 full9。

## 1. Method Story for Slides

MeshSplatting clean baseline 是：

```text
trained MeshSplatting checkpoint -> direct held-out render -> PSNR / SSIM / LPIPS
```

SPCarNet 的核心思想是：

```text
use MeshSplatting mesh/surface as address space
-> attach evidence-certified residual repair to visible triangles
-> bake reliable endpoint corrections back into a compact surface field
```

通俗说，MeshSplatting 直接渲染训练好的三角形/高斯表示；SPCarNet 会额外问三个问题：

1. 哪些三角形区域在多视角里有可靠证据？
2. 哪些残差可以稳定地压到 surface-local 函数里？
3. 哪些残差应该回退，避免把 endpoint 的 view-specific 修复硬塞进三角形表示？

## 2. Current Best Evidence

### Full9: v104c vs clean vs endpoint/reference

Source:

- `outputs/carnet/meshsplatopt/ecsr_phase_v104c_shrink_view_affine_field_full9_20260625/v104c_shrink_view_affine_full9_summary.json`

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean MeshSplatting | 25.151682 | 0.749018 | 0.287621 |
| v104c shrink view-affine field | 25.829099 | 0.760727 | 0.268548 |
| v101/v102 endpoint/reference | 26.481310 | 0.783675 | 0.224305 |

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v104c - clean | +0.677417 | +0.011709 | -0.019073 |
| v104c - endpoint/reference | -0.652211 | -0.022949 | +0.044243 |

Interpretation:

- v104c 已经在本地 full9 上形成稳定固定策略证据：9/9 scenes OK，且均值三指标优于 clean。
- v104c 仍没有追上 endpoint/reference，说明低阶 per-triangle surface field 表达力不足，特别是 LPIPS / detail gap 仍明显。

## 3. Module-Level Method Details

### v101/v102 Endpoint

Role:

- v101 在线使用 train-derived evidence bank 做 checkpoint-attached repair。
- v102 把 v101 endpoint 的 target-camera residual delta 预投影成 sidecar bank，用于加速和作为 field distillation teacher。

Boundary:

- 不使用 held-out target GT 做 policy。
- 但 v102 是 target-camera endpoint delta，不应描述为 train-only unseen-camera generalization。

### v104c Shrink View-Affine Surface Field

Role:

- 把 endpoint teacher delta 压缩到 triangle-local field。
- 每个 triangle 存一个 view-conditioned affine residual：

```text
[1, barycentric_0, barycentric_1, viewdir_x, viewdir_y, viewdir_z] -> RGB residual
```

Key design:

- 用 centered/scaled view basis 提高数值稳定性。
- 用 shrink/fallback 把病态 view-affine solution 收缩到 conservative v103 barycentric affine fallback。
- 固定策略 full9，不按场景手调。

Status:

- 当前最佳 representation-field headline。
- 已 full9 验证超过 clean，但没有追平 endpoint。

### v105 Evidence-Gated Residual-Mixture Field

Implemented files:

- `render.py`
- `scripts/car_model/build_v105_evidence_gated_mixture_field.py`
- `scripts/car_model/run_v105_evidence_gated_mixture_scene.py`
- `scripts/car_model/summarize_v105_evidence_gated_mixture.py`

Representation change:

```text
base expert: conservative fallback affine residual
delta expert: raw view-affine residual debt
gate: evidence-derived triangle weight
rendered residual = base + gate * delta
```

Field schema:

```text
field_type = v102_surface_residual_field
basis_type = affine_barycentric_viewdir_mixture
builder_variant = v105_evidence_gated_residual_mixture
triangle_base_coefficients [T, 6, 3]
triangle_delta_coefficients [T, 6, 3]
triangle_gate [T]
```

Render path:

- `render.py` now has an explicit branch for `affine_barycentric_viewdir_mixture`.
- Unknown surface-field basis now fails closed instead of silently falling back to constant residual.
- Render report now writes field identity: `field_type`, `basis_type`, `builder_variant`, `gate_source`, `field_sha256`, and key parameters.

Runner/report improvements:

- Field filename includes parameter signature to avoid stale artifact reuse.
- Runner validates manifest and render identity before marking `passed=true`.
- `renderer_scaling` is locked to 4 because `render.py` uses `TriangleModel.scaling=4`.
- v104c metrics must be present for v105 report to pass.

## 4. v105/v105b Experiment Results

### Hard-triad v105 no-viewgate probe

Source:

- `outputs/carnet/meshsplatopt/ecsr_phase_v105_evidence_gated_mixture_probe_noviewgate_20260625/v105_hardtriad_noviewgate_summary.json`

Scenes: counter, kitchen, bonsai.

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean | 27.821853 | 0.878303 | 0.236894 |
| v104c | 28.859798 | 0.885459 | 0.219064 |
| v105 no-viewgate | 28.864001 | 0.885513 | 0.219103 |
| endpoint/reference | 30.167397 | 0.913355 | 0.163709 |

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v105 - v104c | +0.004203 | +0.000055 | +0.000039 |
| v105 - endpoint/reference | -1.303396 | -0.027842 | +0.055394 |

Interpretation:

- v105 no-viewgate gives tiny PSNR/SSIM gain but LPIPS slightly worse.
- Improvement is too small and not perceptually stable, so not enough to replace v104c headline.

### Counter v105 debt-guard probe

Source:

- `outputs/carnet/meshsplatopt/ecsr_phase_v105_evidence_gated_mixture_debtguard_probe_20260625/counter/counter_v105_evidence_gated_mixture_report.json`

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean | 26.751774 | 0.862055 | 0.252003 |
| v104c | 27.498068 | 0.867420 | 0.238986 |
| v105 debt-guard | 27.497105 | 0.867424 | 0.238974 |
| endpoint/reference | 28.442907 | 0.893696 | 0.186557 |

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v105 debt-guard - clean | +0.745331 | +0.005369 | -0.013029 |
| v105 debt-guard - v104c | -0.000963 | +0.000005 | -0.000012 |
| v105 debt-guard - endpoint/reference | -0.945803 | -0.026271 | +0.052417 |

Interpretation:

- debt-guard is essentially tied with v104c, not a meaningful breakthrough.
- It slightly improves SSIM/LPIPS at counter but loses PSNR.

### Counter v105b crossfit-risk probe

Source:

- `outputs/carnet/meshsplatopt/ecsr_phase_v105b_crossfit_risk_counter_probe_20260625/counter/counter_v105_evidence_gated_mixture_report.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_v105b_crossfit_risk_counter_probe_20260625/v105b_crossfit_counter_summary.md`

| method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| clean | 26.751774 | 0.862055 | 0.252003 |
| v104c | 27.498068 | 0.867420 | 0.238986 |
| v105b crossfit-risk | 27.496115 | 0.867405 | 0.239000 |
| endpoint/reference | 28.442907 | 0.893696 | 0.186557 |

| comparison | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| v105b - clean | +0.744341 | +0.005350 | -0.013004 |
| v105b - v104c | -0.001953 | -0.000014 | +0.000013 |
| v105b - endpoint/reference | -0.946793 | -0.026290 | +0.052443 |

Diagnostics:

| item | value |
|---|---:|
| gate_source | crossfit_risk |
| gate_mean | 0.515348 |
| gain_score_mean | 0.023501 |
| crossfit_gain_mean | 0.049375 |
| crossfit_gain_supported_triangles | 1292949 |
| valid_triangles | 2716470 |
| field build sec | 1244.756 |
| render sec | 58.266 |

Interpretation:

- v105b is more methodologically defensible than the in-sample gate, but it is too conservative and still does not close the endpoint gap.
- It validates a negative lesson: cross-fitted teacher-risk alone is not enough; the remaining gap is not only overfitting from gate selection.

## 5. What This Means Scientifically

The bottleneck is now clearer:

1. **Surface-local low-order field capacity is insufficient.** Endpoint can make per-view, per-pixel support/fallback decisions; v104c/v105 compress those decisions into one low-order triangle-local function.
2. **Gate evidence and image metrics are not perfectly aligned.** Normal-equation gain can over-trust residual debt; crossfit-risk gate avoids some overfit but becomes too conservative and still does not improve final PSNR/SSIM/LPIPS.
3. **LPIPS/detail gap is the hardest part.** Small PSNR/SSIM deltas are easy to create; stable perceptual improvement requires better local detail modeling, occlusion handling, and multi-modal residual representation.

Therefore, v105/v105b should be presented as a diagnostic research step, not as the final method.

## 6. Recommended PPT Positioning

Use this hierarchy:

1. Main result: **v101/v102 endpoint shows large quality ceiling over MeshSplatting clean.**
2. Main representation result: **v104c proves that a fixed surface-field policy can bake part of endpoint gains and beat clean full9.**
3. Honest limitation: **v104c still leaves a large endpoint gap, especially LPIPS/detail.**
4. New research direction: **v105/v105b mixture and crossfit-risk gate show that simply adding a scalar residual-debt gate is not enough; future work needs higher-capacity, locally adaptive, perceptual/occlusion-aware field.**

Do not claim:

- v105/v105b全面超过 v104c.
- 当前 field 已经达到 endpoint quality.
- 当前 sidecar field 是 vanilla MeshSplatting checkpoint.
- v102 target-camera distillation is train-only unseen-camera generalization.

## 7. Next Work Items

Minimum next sequence:

1. Keep v104c as current stable headline.
2. Preserve v105/v105b code as ablation/negative diagnostic.
3. Build a stronger representation-level model, not another scalar gate:
   - multi-expert residual field with spatial/detail expert;
   - occlusion-aware boundary expert;
   - LPIPS/edge-aware local validation objective;
   - train/policy-val-only teacher for publishable generalization claim.
4. Only expand to hard-triad/full9 after counter beats v104c on PSNR, SSIM, and LPIPS with nontrivial margin.
5. Add qualitative panels focused on endpoint-gap regions: clean / v104c / v105 candidate / endpoint / GT / error heatmap.

## 8. Reproduction Commands

v105b counter probe:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 WANDB_MODE=offline TMPDIR=/tmp \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  --scene counter \
  --gpu 3 \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v105b_crossfit_risk_counter_probe_20260625 \
  --field_root /dev/shm/peilincai_spcarnet_v105b_crossfit_risk_counter_probe_20260625 \
  --build_v102_if_missing \
  --force_field \
  --force_render \
  --force_eval
```

v105b counter summary:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/summarize_v105_evidence_gated_mixture.py \
  --root outputs/carnet/meshsplatopt/ecsr_phase_v105b_crossfit_risk_counter_probe_20260625 \
  --scenes counter \
  --out_dir outputs/carnet/meshsplatopt/ecsr_phase_v105b_crossfit_risk_counter_probe_20260625 \
  --prefix v105b_crossfit_counter_summary
```

Static compile check:

```bash
python -m py_compile \
  render.py \
  scripts/car_model/build_v105_evidence_gated_mixture_field.py \
  scripts/car_model/run_v105_evidence_gated_mixture_scene.py \
  scripts/car_model/summarize_v105_evidence_gated_mixture.py
```

## 9. Final Review

What is strong:

- full9 v104c beats clean MeshSplatting with fixed policy.
- v101/v102 endpoint establishes a clear quality ceiling above clean.
- v105/v105b now have real code-level representation changes and strict identity checks.

What is still weak:

- v105/v105b did not beat v104c on counter, so they cannot be promoted.
- The current best field still relies on target-camera delta distillation.
- Visual improvements may remain subtle in full-frame views; crop/error-map panels are necessary.
- We still need a stronger representation, not another scalar gate, to close endpoint gap.

Current honest status:

```text
Engineering closure: improved for v105/v105b pipeline and verification.
Paper-method closure: not complete.
Best current field method: v104c.
Best quality method: v101/v102 endpoint.
Next required innovation: higher-capacity, perceptual and occlusion-aware residual field with train/policy-val-safe teacher.
```

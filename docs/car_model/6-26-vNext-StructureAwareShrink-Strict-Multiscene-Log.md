# 6-26 vNext Structure-Aware Shrink Strict Multiscene Log

日期：2026-06-26

本日志记录对 `docs/6-26-SPCarNet-vNext-ServerCodexPrompt.md` 的判断、工程落地、subagent 审查、bug 修复，以及 `counter/bonsai/room` strict no-target-GT 多场景验证结果。

## 判断

`Evidence-Certified Residual Surface Texturing for MeshSplatting` 方向是合理的、现实的，但它是一个分阶段的 representation-level 路线，而不是一次小 gate/alpha 改动即可完成的论文终局。

合理之处：

- 目标从 Phase-J 的 render-time ELA repair 转成持久的 face/UV residual surface texture，具备更明确的 representation-level 身份。
- 继续保留 parent-preserving 输出：`parent + confidence * residual`，且 unsafe candidate 必须 no-op fallback。
- train split 内部拆 fit/policy-val，selection、alpha、capacity、threshold 都不读 target/test GT，协议上可做严密审计。

当前边界：

- 这轮只证明了结构感知 shrink 能缓解部分 SSIM/room fallback 问题，还不是 full9，也不是对 clean MeshSplatting/v106/Phase-J 的最终胜利。
- 平均 PSNR/LPIPS 收益仍是 `1e-3`/`1e-5` 量级，视觉效果很难作为强定性证据。
- Paper-grade 仍需要 full9、clean/v104c/v106/Phase-J 对比、预算核算、定性 changed-region panel 和更大 fraction of Phase-J gain。

## 本轮实现

核心文件：

- `scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py`
- `scripts/car_model/run_vnext_certified_residual_texture_scene.py`

新增机制：

1. `policy_val_structure_aware_shrink`
   - 在 policy-val 视角上计算 residual apply 前后的局部 RGB L1 worsening 和 luminance-gradient worsening。
   - 按 face/bin 聚合成 `structure_risk_confidence`，注入原有 bin uncertainty shrink。
   - 只使用 train-policy-val `rgb_render/rgb_gt`，audit 显式记录 `uses_target_or_test_gt=false`。

2. `parent_edge_apply_shrink`
   - 基于 target/policy-val 的 parent render luminance edge strength 做 GT-free apply-time residual downweight。
   - 用于未来边界安全实验；本轮不把早期 edge artifact 作为正向证据，因为 subagent 审查发现修复前 final `apply_to_target` 漏传 profile。

3. 接口修复
   - `evaluate_target_support_profile(...)` 新增 `parent_edge_apply_profile` 参数。
   - target-support profiling 与最终 `apply_to_target(...)` 均传入相同 `parent_edge_apply_profile`。
   - 修复后通过：
     - `python -m py_compile scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`
     - `git diff --check -- scripts/car_model/ecsr_apply_surface_residual_region_texture_adapter.py scripts/car_model/run_vnext_certified_residual_texture_scene.py`
     - parent-edge/structure strict dry-run command parse。

## Subagent 审查结论

代码审查：

- 发现并促成本轮修复两个 P0 问题：
  - `evaluate_target_support_profile()` 使用 `parent_edge_apply_profile` 但签名缺参数。
  - policy-val 评估传入 parent-edge profile，但最终 target apply 未传入，可能造成证书和输出不一致。
- 还指出 `--parent_render_dir` 只在 teacher-cache 构建中使用；若 evidence `rgb_render` 与外部 parent render 不一致，必须拒绝或显式对齐。当前本轮实验未使用外部 parent render。

实验审查：

- 旧 `counter` face-softshrink 仍是 PSNR/LPIPS 较强，但 SSIM 回退明显。
- 新 structure-aware `tau=0.002` 在 `counter` 缓解 SSIM 回退，但牺牲一部分 PSNR/LPIPS。
- 修复前 `counter_structure_edge_confidence_20260626_0623` 不应作为 parent-edge 正向证据，只能作为旧接口问题的 negative/no-op 记录。

论文叙事审查：

- 目前足够称为 strict certified residual texture proof-of-life。
- 还不能称为 paper-grade 性能闭环。

## Strict Multiscene Results

固定策略：

```text
texture_size=16
support_expansion_mode=none
atlas_empty_bin_fill_mode=face_mean
surface_multiscale_prior_blend=0.5
view_conditioned_basis_mode=normal_camera_linear
teacher_distilled_basis_mode=face_uv_patch_mixture_ridge
bin_uncertainty_shrink_policy_mode=keep_with_downweight
enable_policy_val_structure_aware_shrink=true
structure_shrink_l1_weight=1.0
structure_shrink_gradient_weight=1.0
structure_shrink_edge_weight=0.0
structure_shrink_risk_tau=0.002
strict_no_target_gt_apply=true
```

### Versus Phase-F Compact Parent

| scene | accepted | alpha | changed fraction | delta PSNR | delta SSIM | delta LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| counter | true | 0.125 | 0.01234357 | +0.00129890 | -0.00000906 | -0.00004268 |
| bonsai | true | 0.25 | 0.00148974 | +0.00113869 | -0.00000954 | -0.00001693 |
| room | true | 0.0625 | 0.00519912 | +0.00046921 | +0.00000334 | -0.00001399 |
| mean | 3/3 | - | - | +0.00096893 | -0.00000509 | -0.00002453 |

Interpretation:

- `room` 是本轮最重要修复点：旧 strict face-softshrink 在 room fallback/no-op；structure-aware shrink 使 room 变为 accepted nonzero，并且 PSNR/SSIM/LPIPS 三项相对 Phase-F parent 全部正向。
- `counter/bonsai` 仍有极小 SSIM 回退，但明显小于旧 face-softshrink 的 counter SSIM 回退。
- 三场景都通过 strict no-target-GT apply：`selection_uses_test_gt=false`、`target_gt_visible_to_apply=false`、`target_forbidden_keys_stripped=true`、`target_apply_leak=false`。

### Versus Old Strict Face-SoftShrink

| scene | old accepted | new accepted | delta PSNR, new-old | delta SSIM, new-old | delta LPIPS, new-old |
|---|---:|---:|---:|---:|---:|
| counter | true | true | -0.00083160 | +0.00003827 | +0.00004265 |
| bonsai | true | true | -0.00008583 | +0.00000036 | +0.00000060 |
| room | false | true | +0.00056648 | +0.00000656 | -0.00000659 |

Interpretation:

- New structure-aware shrink trades some counter PSNR/LPIPS for much safer SSIM behavior.
- It converts room from fallback to accepted nonzero output and improves all three metrics versus old fallback evaluation.
- It is a real method change, not only parameter scanning, because the per-bin shrink now depends on policy-val local structure-risk evidence.

## Artifact Index

New artifacts:

```text
docs/car_model/vnext_artifacts/counter_structure_shrink_tau002_20260626_0558/
docs/car_model/vnext_artifacts/bonsai_structure_shrink_tau002_20260626_0718/
docs/car_model/vnext_artifacts/room_structure_shrink_tau002_20260626_0718/
```

Each new `bonsai/room` directory includes:

- `*_test_results.json`
- `*_test_per_view.json`
- `*_test_eval_gt_population_audit.json`
- `*_vnext_certified_residual_texture_manifest.json`
- `*_vnext_certified_residual_texture_report.md`
- `surface_residual_region_texture_adapter_audit.json`
- `target_evidence_no_gt_audit.json`
- `02_certified_texture.log`
- `03_eval.log`

W&B offline runs:

```text
/dev/shm/peilincai_wandb_vnext_structure_shrink_bonsai_strict_20260626_0718_bonsai_structure_strict_fix/wandb/offline-run-20260626_065533-d6oke3nt
/dev/shm/peilincai_wandb_vnext_structure_shrink_room_strict_20260626_0718_room_structure_strict_fix/wandb/offline-run-20260626_065735-6u9v3q7h
```

## Current Recommendation

Promote structure-aware shrink as the next strict vNext milestone, but do not call it paper-final.

The honest claim is:

> A train-policy-val structure-risk certificate can turn residual surface texturing from a brittle small-gain atlas into a safer accepted representation on the previous room failure case, while preserving strict no-target-GT apply and maintaining positive PSNR/LPIPS mean movement.

The remaining bottleneck is effect size. The method still captures only a tiny fraction of Phase-J's render-time teacher gain. The next paper-level push should focus on adaptive capacity and teacher residual distillation quality, not further scalar shrink tuning.

## Next Required Work

1. Run full9 with this exact frozen structure-aware policy.
2. Add full comparison rows: clean MeshSplatting parent, Phase-F compact parent, Phase-J teacher, v104c, v106, old face-softshrink, new structure-aware shrink.
3. Build changed-region qualitative panels for room/counter, not random whole-frame panels.
4. Add budget accounting: triangle count, residual texture storage, parameter count, render overhead, fallback rate.
5. Fix or explicitly reject external `--parent_render_dir` mismatch when evidence `rgb_render` is not the same parent used to define residuals.
6. Re-run parent-edge apply shrink after the interface fix before using it as evidence.

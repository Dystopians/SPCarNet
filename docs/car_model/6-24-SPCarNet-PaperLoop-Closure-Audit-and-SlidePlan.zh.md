# SPCarNet Paper-Loop 闭环审计与 PPT 拆页计划

日期：2026-06-24  
状态：`NOT COMPLETE`  
用途：把当前 SPCarNet 的工程证据、论文证据、PPT 可讲内容和未闭合缺口压缩成一份执行清单。  

---

## 1. 当前结论

当前可安全汇报的 headline 仍是：

```text
Phase-J guarded adaptive Evidence Lumigraph Adapter + geometry-safe compaction
```

它在本地 same-protocol Mip-NeRF360 full9 selected-clean MeshSplatting baseline 上已经有强证据：

| 维度 | 当前证据 | 状态 |
|---|---:|---|
| scene-level PSNR/SSIM/LPIPS strict wins | `9 / 9` | 可讲 |
| per-view PSNR/SSIM/LPIPS strict wins | `244 / 246` | 可讲 |
| mean dPSNR vs selected clean | `+1.331084` | 可讲 |
| mean dSSIM vs selected clean | `+0.034702` | 可讲 |
| mean dLPIPS vs selected clean | `-0.063359` | 可讲 |
| mean triangle reduction | `7.6479%` | 可讲 |
| geometry-safe scenes | `9 / 9` | 可讲 |
| strict all-axis geometry wins | not `9 / 9` | 不能夸大 |
| fully baked representation endpoint | not achieved | 不能夸大 |

主证据路径：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md
docs/car_model/6-24-SPCarNet-Mentor-PPT-Technical-Report-CurrentMethod-Full.zh.md
README.md
README.zh.md
outputs/carnet/spcarnet/current_evidence_manifest_20260624.md
```

---

## 2. 工程闭环审计

| 要求 | 当前证据 | 判断 |
|---|---|---|
| train/eval pipeline 有真实方法改动 | `ecsr_apply_surface_residual_region_texture_adapter.py`、`run_l1risk_fairnoop_scene.py` 已接入 atlas、policy-val gate、v77 strict hybrid flags | 已部分闭合 |
| W&B 记录中程/长程验证 | v64-v77 多个 W&B run；最新 v77 为 `3ho2y4s1` | 已部分闭合 |
| baseline/current/improved/ablation 可追溯 | README 和 v48-v77 summaries 指向 full9 / probe artifacts | 已部分闭合 |
| metrics 和 qualitative outputs 保存 | 主表、v64-v77 summaries、assets qualitative panels 已保存 | 已部分闭合 |
| 当前证据 manifest | `outputs/carnet/spcarnet/current_evidence_manifest_20260624.md/json` 覆盖 `18 / 18` 个关键证据文件，必需项缺失 `0`；状态语义为 existence/hash manifest，不代表论文闭环通过 | 已闭合 |
| 最新 v77 结果持久化 | `outputs/.../v77_strict_bin_gain_hybrid_20260624/summary.md` | 已闭合 |
| 脚本静态检查 | v77 改动已 `py_compile` 和 `git diff --check` | 已闭合 |
| Stage 2 held-out autodecoder eval | 已补 `--fit_missing_latents` clean-val z-only MAP-fit；v3 和 v4 normal-band 都达到 `206 / 206` val objects extracted；strict JSON；v4 full training W&B run `dysg8508`，epoch50 full-val eval W&B run `4wu9w305`，final checkpoint full-val eval W&B run `q1jjwvdm`；新增 selector artifact 固定选择 `v4_epoch50` | 工程闭合，checkpoint 选择闭合，v4 epoch50 有质量改善但质量未闭合 |
| 表示级最终方法 | v64 最稳但收益微小；v77 未晋级 | 未闭合 |
| fair paper-protocol final rerun | 本地主表可信，但 paper-table 完全同口径最终复核仍缺 | 未闭合 |
| 完整跨数据集强验证 | 有 ETH3D courtyard 外部验证，但不是完整跨数据集 paper suite | 未闭合 |
| 旧 Phase-S strict four-offset collector | `full9_paper_loop_status` 仍缺 `treehill/counter` gate JSON | 未闭合 |

结论：

> 工程接口和审计链条已经非常完整，但“论文终局方法”还没闭合。当前强 endpoint 是 render-time guarded ELA portfolio；persistent representation-level residual field 仍未达到能替代 Phase-J 的效果。

### 2.1 Stage 2 held-out eval closure update

2026-06-24 修复了 `scripts/car_model/eval_spcarnet_shape_field_autodecoder.py` 的 Stage-2 held-out eval 合同：

- 原问题：Stage-2 auto-decoder 只有 train-object latent table；旧 val eval 对 val objects 没有 z，导致 `0 / 206` extraction 和裸 `NaN` JSON。
- 当前修复：新增 `--fit_missing_latents`，对 missing val/test objects 做 decoder-frozen、z-only clean-shape MAP fitting，并把 per-object `latent_source` 标成 `heldout_map_fit`。
- 结果路径：`outputs/carnet/spcarnet/autodecoder_v3/eval/val_mapfit_full206_20260624.json`。
- W&B：`https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/svtbc8sn`。

Full-val result:

| metric | value |
|---|---:|
| `n_objects_evaluated` | `206` |
| `n_extracted` | `206` |
| `mesh_extraction_success_rate` | `1.0` |
| `recon_chamfer_l1_mean` | `0.0698447353` |
| `hidden_chamfer_l1_mean` | `0.1023846301` |
| `mesh_iou_at_0.5_mean` | `0.5531548112` |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` |
| `surface_normal_consistency_mean` | `0.7182239138` |

Decision:

> Stage 2 is engineering-closed enough to evaluate held-out decoder capacity, but it remains a quality soft FAIL: extraction passes, while chamfer and filled IoU miss the original gate. This is a real closure improvement, not a new headline result.

### 2.2 Stage 2 v4 normal-band objective update

v4 在 v3 容量不变的基础上加入 surface-normal band supervision：

```text
x_inner = x_surface - epsilon * normal -> occupied
x_outer = x_surface + epsilon * normal -> free
```

它的目的是让 occupancy field 在真实 surface 附近形成更清晰的 `0.5` crossing，从而改善 Marching-Cubes mesh 的边界质量。

证据：

```text
docs/car_model/6-24-Stage2-v4-NormalBand-Autodecoder-Log.md
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_last.pt
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_epoch50_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/eval/val_mapfit_final_full206_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.json
outputs/carnet/spcarnet/autodecoder_v4_band_20260624/checkpoint_selection/stage2_v4_checkpoint_selection_20260624.md
W&B train: dysg8508
W&B eval: 4wu9w305
W&B final eval: q1jjwvdm
```

Full-val comparison:

| metric | v3 MAP-fit | v4 epoch50 MAP-fit | v4 final MAP-fit | best |
|---|---:|---:|---:|---|
| `recon_chamfer_l1_mean` | `0.0698447353` | `0.0607328202` | `0.0655826944` | v4 epoch50 |
| `hidden_chamfer_l1_mean` | `0.1023846301` | `0.0933915632` | `0.0963624408` | v4 epoch50 |
| `mesh_iou_at_0.5_mean` | `0.5531548112` | `0.5683319216` | `0.5314717742` | v4 epoch50 |
| `mesh_iou_at_0.5_shell_mean` | `0.9112784961` | `0.8783071888` | `0.8563237802` | v3 |

Decision:

> v4 是真实方法改动；epoch50 带来 chamfer/filled-IoU 改善，但仍未过 `chamfer <= 0.05` 与 `filled IoU >= 0.92` 的原 gate，且 shell IoU 回退。final checkpoint 比 epoch50 差，说明这条线需要验证驱动的 checkpoint selection 和更强正则，而不是继续拉长训练。因此它是 next-step evidence，不是 headline。

Selector status:

```text
BEST_AVAILABLE_GATE_FAIL_WITH_LATE_DEGRADATION
best candidate: v4_epoch50
best score: 0.039685866
```

---

## 3. 最新 v77 诊断

v77 是严格 multi-view bin-gain hybrid policy：

```text
min_bin_samples = 16
min_views = 2
min_abs_gain = 1e-5
min_relative_gain = 0.005
min_positive_view_fraction = 0.75
```

结果：

| method | PSNR | SSIM | LPIPS | 判断 |
|---|---:|---:|---:|---|
| v77 strict bin-gain hybrid | `26.753528595` | `0.862111032` | `0.251881331` | 不晋级 |
| v76 policy-val bin-gain hybrid | `26.753532410` | `0.862111092` | `0.251881331` | 不晋级 |
| v75 zero-blend line | `26.753995895` | `0.862119257` | `0.251853049` | 更强 |
| v64/v56 counter reference | `26.756130219` | `0.862126231` | `0.251691371` | 更强 |

关键解释：

> v77 不是失败的无效改动，而是一个安全诊断。它把 v76 中容易过拟合的 weak hybrid bins 阻断掉，最终回到 `blend=0.0`。这说明继续靠更复杂的 local bin prior 不能自然产生突破，下一步必须换成更强的 residual representation 和 target-view generalization certificate。

证据：

```text
docs/car_model/6-24-v77-StrictBinGainHybrid-Log.md
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v77_strict_bin_gain_hybrid_20260624/summary.md
```

---

## 4. 论文 claim 分级

### 可以强 claim

1. **Self-auditing MeshSplatting**  
   SPCarNet 把训练视角 evidence 投回 surface，用来审计哪里可删、哪里可修、哪里必须回退。

2. **Full9 selected-clean RGB 全胜**  
   Phase-J 在本地 full9 selected-clean MeshSplatting baseline 上 `9 / 9` scene-level 三指标全胜。

3. **RGB 与 triangle reduction 同时改善**  
   mean dPSNR `+1.331084`，mean dSSIM `+0.034702`，mean dLPIPS `-0.063359`，mean triangle reduction `7.6479%`。

4. **Train-only policy gate**  
   方法选择 branch/alpha/fallback 不用 held-out test GT。

5. **负结果可审计**  
   v65-v77 证明团队已经系统性排除 weak certificates，而不是只挑正结果。

### 必须谨慎 claim

1. **不是 fully baked representation endpoint**  
   当前最强 endpoint 仍依赖 render-time Evidence Lumigraph Adapter。

2. **不是所有几何指标全胜**  
   可以说 geometry-safe `9 / 9`，不能说 strict all-axis geometry `9 / 9`。

3. **paper-table 对比是辅助，不是最终同口径主表**  
   早期 Compact-ELA/SOR 相对 MeshSplatting paper table 为正，但最终论文主表必须重跑严格同协议。

4. **定性 full-frame 视觉差异不一定显著**  
   必须用 local crop、error map、local MAE drop 来展示改进。

---

## 5. PPT 拆页计划

建议 12 页，每页只讲一个核心点。

| 页码 | 标题 | 主要内容 | 推荐证据 |
|---:|---|---|---|
| 1 | SPCarNet | subtitle: Self-Auditing MeshSplatting for Repair and Compaction | 一句话贡献 |
| 2 | Problem | MeshSplatting 有局部 residual 和冗余 triangles | clean vs GT residual crop |
| 3 | Key Idea | 用 train-view surface evidence 审计 mesh | pipeline diagram |
| 4 | Evidence Cache | residual / face id / support / risk | method module table |
| 5 | Geometry-Safe Compaction | 低风险删面，风险区回退 | triangle reduction table |
| 6 | Guarded ELA | residual transfer 公式 | `I_ours = I_compact + alpha * residual` |
| 7 | Policy Gate | train-only branch/alpha/fallback | no-test-GT policy text |
| 8 | Main Result | full9 selected-clean 9/9 wins | Phase-J result table |
| 9 | Qualitative | local crop + error map | `assets/spcarnet_phasej_where_it_helps_showcase_20260622.png` |
| 10 | Representation Track | v64 positive but tiny; v77 negative diagnostic | v64/v77 mini table |
| 11 | Limitations | render-time endpoint, geometry all-axis, paper-protocol | honest weakness list |
| 12 | Next Step | stronger residual basis + target footprint certificate | future method diagram |

推荐图片：

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_outdoor_detail_showcase.png
assets/spcarnet_m360_where_it_helps_showcase.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_v56_counter_face_alpha_guard_panel.png
```

---

## 6. 还差什么才算真正 100%

| 缺口 | 为什么重要 | 下一步证据 |
|---|---|---|
| representation-level 方法效果太小 | 顶会主方法不能只依赖 render-time adapter | 新 residual basis 在 full9 上超越 v64/v56 且接近 Phase-J |
| Stage 2 shape prior 质量不过 gate | 对象级先验不能支撑强几何主线 | 改进 shape-field objective/representation，使 val MAP-fit chamfer ≤ 0.05 且 filled IoU ≥ 0.92 |
| strict paper-protocol rerun | 避免与 MeshSplatting paper table 口径不一致 | baseline 和 ours 全量重跑脚本、W&B、结果表 |
| qualitative 差异不够直观 | mentor/审稿人会先看图 | 每个关键场景 error-map/local-crop panel |
| 几何 all-axis 不全胜 | “全面超越”必须包括几何 | sparse depth/normal/topology 统一表 |
| 外部数据集覆盖有限 | 当前 ETH3D courtyard 只是辅助验证 | 至少再选 1-2 个外部场景或 dataset subset |
| 旧 Phase-S collector 有显式缺口 | full9 paper-loop status 中 `treehill/counter` 的 strict four-offset gate 缺失 | 恢复或重建旧 candidate artifacts 后补齐 gate，或在论文中明确废弃旧 Phase-S collector |

---

## 7. 下一步推荐任务

优先级从高到低：

1. **构建 target-footprint certified residual basis**  
   不再只在 policy-val bin 上看 gain，而是把 target footprint、view direction diversity、multi-view support 和 residual stability 绑定成一个证书。

2. **重跑 strict paper-protocol baseline/ours table**  
   固定 same evaluator、same split、same checkpoint selection，生成可以直接放论文的主表。

3. **生成 slide-ready qualitative panels**  
   每个 panel 必须包含 clean/ours/error/delta/local metrics；不要只放 full-frame。

4. **统一 geometry table**  
   triangle count、vertices、sparse depth、normal/topology 分开列，避免只讲 triangles。

5. **把 v65-v77 整理成 ablation story**  
   证明不是调参失败，而是逐步定位出 representation-level 的真实 bottleneck。

---

## 8. Exact Next Commands

当前最直接的下一步不是再跑 v77，而是先生成更强的 PPT 定性证据并启动 paper-protocol 复核。

建议命令模板：

```bash
# 1. 复核当前 headline 主表
sed -n '1,120p' \
  outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.md

# 2. 检查当前可用 qualitative assets
find assets -maxdepth 1 -type f \
  \( -name 'spcarnet_phasej*.png' -o -name 'spcarnet_m360*.png' -o -name 'spcarnet_v56*.png' \) \
  -printf '%f %s\n' | sort

# 3. 复核 v77 负诊断
sed -n '1,180p' docs/car_model/6-24-v77-StrictBinGainHybrid-Log.md
```

如果继续做方法改进，下一轮实验应避免单场景单参数扫描，至少满足：

```text
real train/eval pipeline change
counter + kitchen + one outdoor scene
baseline/v64-v56 reference comparison
W&B online logging
persisted results + audit + log
explicit promote / reject decision
```

## 9. 已证实的旧 collector 缺口

实验审计发现 `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_missing_rows.csv` 仍有两条缺失：

```text
treehill,phase_s_strict_four_offset,outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/treehill/multifold_trainval_gate.json
counter,phase_s_strict_four_offset,outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/counter/multifold_trainval_gate.json
```

当前状态验证：

```bash
ls outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/treehill/multifold_trainval_gate.json
ls outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/counter/multifold_trainval_gate.json
```

两条路径当前都不存在。`multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512` 下已有 7 个场景的 gate 结果，缺口确实只集中在 `treehill/counter`。不过旧报告中引用的 candidate source root：

```text
outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512
```

当前工作区没有完整保留，因此补跑不是简单执行两个 gate 命令。必须先恢复或重建这两个场景对应的 candidate model 和 audit JSON，然后才能按旧脚本补齐：

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
  --scene <treehill_or_counter> \
  --phasej_model outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model \
  --candidate_model outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/<scene>/model \
  --candidate_audit_json outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/<scene>/model/surface_residual_facelocal_sh1_delta_audit.json \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_label facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela \
  --phasej_trainval_method_prefix ours_26000_phasej_gaincert_v1_20260512 \
  --candidate_trainval_method_prefix ours_26000_facelocal_gaincert_v1_20260512_multifold \
  --offsets 0,1,2,3 \
  --iteration 26000 \
  --gpu <low_or_mid_util_gpu> \
  --policy_holdout_fraction 0.25 \
  --calib_sampler uniform \
  --calib_max_views 32 \
  --calib_stride 1
```

论文汇报建议：

> 不要把旧 Phase-S collector 描述成 full9 完整闭环。当前可以讲 Phase-J full9 完整闭环、v64 representation-level report-only fixed policy、v77 strict certificate negative diagnostic。旧 Phase-S collector 只能作为历史路线和缺口审计。

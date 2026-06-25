# SPCarNet v100 技术报告（Mentor / PPT 母稿）

日期：2026-06-25
范围：本文只基于当前仓库内可读的 v100 fixed full9 与 counter 证据。
建议 PPT 标题：**SPCarNet v100: Checkpoint-Attached Evidence Lumigraph Endpoint**

---

## 0. 一页结论

v100 的最准确表述：

```text
compact MeshSplatting checkpoint
  + inherited topology
  + Phase-J / ELA train-derived repair report
  + checkpoint-attached render residual sidecar
  + provenance / frame-set / non-noop / RGB gate
```

一句话讲法：

> v100 把已有 Phase-J / Evidence Lumigraph Adapter 渲染时修复，封装成一个挂在 checkpoint 旁边的 endpoint sidecar；它让结果可复现、可审计、可随 checkpoint 交付，但不是相对 Phase-J 的独立算法提升。

当前 fixed full9 结果：

| 项 | 结果 |
|---|---:|
| full9 scenes | `9 / 9` |
| return code zero | `true` |
| endpoint gate pass | `9 / 9` |
| mean dPSNR vs selected clean | `+1.330667` |
| mean dSSIM vs selected clean | `+0.034667` |
| mean dLPIPS vs selected clean | `-0.063328` |
| mean topology removed | `2.0000007%` |
| Phase-J exact ceiling hit | `8 / 9` scenes |

最重要的 claim boundary：

- **可以说**：v100 是 Phase-J endpoint 的 checkpoint-attached package；full9 fixed replay 全部通过 clean/gate/source baseline；counter 有完整 non-noop、per-view、topology、geometry、W&B 证据。
- **不能说**：v100 是独立超过 Phase-J 的新方法。fixed full9 中 `8 / 9` 场景与 Phase-J 完全相同，`flowers` 还略低于 Phase-J。
- **不能说**：这是 vanilla MeshSplatting checkpoint。标准 `render.py` 不理解 sidecar，需要 v100 endpoint loader / runner。

---

## 1. 动机：为什么需要 v100

Phase-J 已经证明：

```text
train / policy-val surface evidence
  -> guarded residual transfer
  -> held-out RGB improvement
```

但 Phase-J 的问题是工程形态不够清晰：

| 问题 | Phase-J 状态 | v100 解决方式 |
|---|---|---|
| 交付形态 | repair report 和模型目录相对分离 | sidecar 写入 `point_cloud/iteration_26000/render_residual_endpoint/...` |
| 复现风险 | 依赖外部 report 是否匹配当前 checkpoint | source-report provenance 强制检查 |
| 公平性 | 需要人工确认 train/test denominator | exact render/GT frame-set equality 与 per-view count 检查 |
| no-op 风险 | 需要证明不是空包 | non-noop render delta gate |
| baseline 口径 | clean/source/Phase-J 容易混淆 | endpoint gate report 固定 comparison rows |

v100 因此不是“又一个效果更好的 policy”，而是一个**证据封装和 endpoint 合约**：

```text
The repair is attached to the checkpoint and audited as part of the endpoint.
```

---

## 2. 方法模块

### 2.1 Compact parent checkpoint

v100 不重新训练基础 MeshSplatting。它读取已经选择好的 compact parent：

```text
outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/<scene>/ratio_0200/compact_model
```

该 parent 已经做了 topology-safe compaction。v100 endpoint 继承 parent 的 geometry 与 checkpoint state。

### 2.2 Phase-J / ELA source report

每个 scene 的 source report 来自 compact parent 内的 Phase-J ELA 输出：

```text
<compact_model>/test/ours_26000_phasej_guarded_adaptedge_ela/ela_report.json
```

v100 从 report 中读取 policy、alpha、support frames、calibrator 和 residual transfer 参数。

### 2.3 Checkpoint-attached sidecar

v100 materialization 会把基础 checkpoint 链接或复制到新 model path，然后把 endpoint manifest 和 ELA report 写入：

```text
point_cloud/iteration_26000/render_residual_endpoint/ours_26000_v100_checkpoint_attached_ela_endpoint/
```

关键点：

- checkpoint state 不被 v100 修改；
- topology 不被 endpoint 修改；
- render-time residual endpoint 作为 sidecar 附着；
- `checkpoint_attached_model_links.json` 记录链接或复制的模型资产。

### 2.4 Provenance / fairness gate

v100 endpoint 在运行前检查：

| 检查 | 目的 |
|---|---|
| `base_model_matches` | source report 必须来自当前 compact parent |
| `base_method_matches` | source base method 必须匹配 |
| `policy_fit_views_subset_of_train` | policy 拟合视角必须来自 train |
| `policy_val_views_subset_of_train` | policy-val 视角必须来自 train/policy-val 证据域 |
| `adapt_support_view_names_subset_of_train` | support frames 不能偷看 held-out test |
| `no_declared_test_gt_policy_flag` | report 不声明 test GT policy 使用 |
| render/GT frame-set equality | metric denominator 公平 |
| per-view metric count equality | 每个方法评测同一批 held-out view |

### 2.5 Non-noop 和 RGB gate

v100 不是只把目录复制一份。counter fixed evidence：

| 字段 | 值 |
|---|---:|
| adapt support frames | `210` |
| mean changed pixel fraction | `0.972526` |
| mean abs RGB delta | `0.011307` |
| max abs RGB delta | `0.250000` |
| per-view strict RGB wins vs clean | `30 / 30` |

gate 要求：

```text
PSNR > anchor_PSNR
SSIM > anchor_SSIM
LPIPS < anchor_LPIPS
non-noop render delta is active
```

---

## 3. 训练 / 评估 pipeline

v100 更准确地说是 **materialization + evaluation pipeline**，不是新训练 pipeline：

```text
1. select compact parent from Phase-F policy-val compaction ladder
2. load Phase-J ELA report from parent/test/...
3. verify report provenance against train split
4. create recovery_model by linking checkpoint assets
5. attach endpoint sidecar under point_cloud/iteration_26000
6. render held-out test frames through v100 endpoint
7. copy GT only for metric evaluation
8. run evaluate_render_split_metrics.py
9. write endpoint_gate_report.json, results.json, per_view.json
10. write contact sheet and W&B offline evidence
11. summarize counter and fixed full9 tables
```

重要公平性说明：

- held-out test GT 只在第 7-8 步用于 metric；
- policy、support、alpha、gate selection 都来自 source report 和 train/policy-val evidence；
- counter report 中 `no_test_gt_used_for_policy = True`；
- v100 的 Phase-J comparison row 是 ceiling，不是另一个可被超过的 baseline。

---

## 4. Exact Commands

### 4.1 Fixed full9 runner

当前 fixed full9 汇总对应的 runner 入口如下。per-scene 子命令被写入每个 scene 的 `v100_endpoint_command.log`。

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_v100_checkpoint_attached_ela_full9.py \
  --output_root /dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625 \
  --report_root outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625 \
  --closure_csv outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv \
  --scenes bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai \
  --gpus 1,2,3,5 \
  --max_parallel 4 \
  --iteration 26000 \
  --contact_sheet_views 4 \
  --wandb_project spcarnet_meshprior \
  --wandb_group v100_checkpoint_attached_ela_full9 \
  --wandb_name v100_checkpoint_attached_ela_full9 \
  --wandb_mode offline \
  --force
```

### 4.2 Full9 counter child command from log

这个命令来自：

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/v100_endpoint_command.log
```

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/run_checkpoint_attached_ela_endpoint_scene.py \
  --scene counter \
  --base_model_path /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model \
  --output_model_path /dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/counter/recovery_model \
  --source_ela_report /data/peilincai/mesh-splatting/outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/counter/ratio_0200/compact_model/test/ours_26000_phasej_guarded_adaptedge_ela/ela_report.json \
  --iteration 26000 \
  --base_method_name ours_26000_phasef_extra_compact_base \
  --method_name ours_26000_v100_checkpoint_attached_ela_endpoint \
  --target_split test \
  --device cuda \
  --evaluate \
  --make_contact_sheet \
  --contact_sheet_views 4 \
  --anchor_psnr 26.7561378479 \
  --anchor_ssim 0.8621263504 \
  --anchor_lpips 0.2516906559 \
  --clean_psnr 26.751773834228516 \
  --clean_ssim 0.862055242061615 \
  --clean_lpips 0.2520033121109009 \
  --source_ela_psnr 27.24042320251465 \
  --source_ela_ssim 0.8641442060470581 \
  --source_ela_lpips 0.24970106780529022 \
  --wandb \
  --wandb_project spcarnet_meshprior \
  --wandb_group v100_checkpoint_attached_ela_full9 \
  --wandb_name v100_checkpoint_attached_ela_full9_counter \
  --wandb_mode offline \
  --wandb_dir /dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/wandb \
  --force
```

### 4.3 Standalone counter metric evaluation command

这个命令来自：

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/endpoint_commands.log
```

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/evaluate_render_split_metrics.py \
  -m /dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model \
  --split test \
  --methods ours_26000_v100_checkpoint_attached_ela_endpoint \
  --merge_model_results
```

---

## 5. Fair Baseline Comparison

### 5.1 Baseline 口径

| Baseline | 用途 | 备注 |
|---|---|---|
| selected clean MeshSplatting | 主公平对比 | same split / same evaluator / selected clean row |
| strict anchor floor | counter gate 下限 | counter 用 v84/v86 anchor floor |
| compact parent noop | 证明不是只靠 compact parent | 同 checkpoint parent，未加 endpoint repair |
| same-evidence noop | 证明不是 evidence plumbing 本身 | counter 专用 no-op control |
| legacy source ELA baseline | 旧 source-ELA row | v100 sidecar 报告中明确标为 legacy |
| v98b checkpoint-baked negative | 表示级失败对照 | checkpoint-baked 尝试没有通过 |
| Phase-J ceiling | v100 真实来源上限 | v100 是 replay/materialization，不是独立超越 |

### 5.2 Counter baseline table

| Method | PSNR | SSIM | LPIPS | 与 v100 的关系 |
|---|---:|---:|---:|---|
| v100 endpoint | `28.449171` | `0.893731` | `0.186472` | endpoint candidate |
| Phase-J ceiling | `28.449171` | `0.893731` | `0.186472` | v100 delta = `0 / 0 / 0` |
| selected clean MeshSplatting | `26.751774` | `0.862055` | `0.252003` | v100: `+1.697397 / +0.031675 / -0.065531` |
| strict anchor floor | `26.756138` | `0.862126` | `0.251691` | v100: `+1.693033 / +0.031604 / -0.065218` |
| compact parent noop | `26.749872` | `0.862051` | `0.251998` | v100: `+1.699299 / +0.031679 / -0.065525` |
| same-evidence noop | `26.749836` | `0.862049` | `0.251998` | v100: `+1.699335 / +0.031681 / -0.065526` |
| legacy source ELA baseline | `27.240423` | `0.864144` | `0.249701` | v100: `+1.208748 / +0.029586 / -0.063229` |
| v98b checkpoint-baked negative | `26.728172` | `0.860831` | `0.257008` | v100: `+1.720999 / +0.032900 / -0.070535` |

解读：

- 相对 clean/noop/v98b，v100 counter 是明确强结果。
- 相对 Phase-J，v100 没有增益；它只是把 Phase-J endpoint 固定到 checkpoint sidecar。

---

## 6. Quantitative Full9 Table

full9 fixed summary path：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.md
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.csv
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.json
```

| scene | status | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR legacy source | dSSIM legacy source | dLPIPS legacy source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | PASS_COUNTER_GATE | `24.021544` | `0.702357` | `0.266088` | `+0.719931` | `+0.042489` | `-0.065989` | `+0.108862` | `+0.008642` | `-0.014180` |
| flowers | PASS_COUNTER_GATE | `20.300608` | `0.557458` | `0.329505` | `+0.618351` | `+0.045636` | `-0.065058` | `+0.117828` | `+0.010157` | `-0.021487` |
| garden | PASS_COUNTER_GATE | `26.311111` | `0.827843` | `0.135843` | `+1.281900` | `+0.047808` | `-0.065472` | `+0.276281` | `+0.010731` | `-0.016469` |
| stump | PASS_COUNTER_GATE | `25.595104` | `0.724074` | `0.263909` | `+0.390062` | `+0.018909` | `-0.030095` | `+0.232574` | `+0.011545` | `-0.017840` |
| treehill | PASS_COUNTER_GATE | `21.296227` | `0.595606` | `0.336319` | `+0.362045` | `+0.031083` | `-0.069725` | `+0.097832` | `+0.007414` | `-0.021805` |
| room | PASS_COUNTER_GATE | `30.305639` | `0.905730` | `0.195989` | `+1.558363` | `+0.020887` | `-0.053913` | `+1.174671` | `+0.020848` | `-0.052740` |
| counter | PASS_COUNTER_GATE | `28.449171` | `0.893731` | `0.186472` | `+1.697397` | `+0.031675` | `-0.065531` | `+1.208748` | `+0.029586` | `-0.063229` |
| kitchen | PASS_COUNTER_GATE | `30.199732` | `0.916087` | `0.131955` | `+2.381180` | `+0.039635` | `-0.067231` | `+2.200171` | `+0.039169` | `-0.066995` |
| bonsai | PASS_COUNTER_GATE | `31.862005` | `0.930280` | `0.172555` | `+2.966772` | `+0.033879` | `-0.086937` | `+2.077568` | `+0.032111` | `-0.084846` |

汇总：

| 汇总项 | 数值 |
|---|---:|
| mean dPSNR vs clean | `+1.330667` |
| mean dSSIM vs clean | `+0.034667` |
| mean dLPIPS vs clean | `-0.063328` |
| mean dPSNR vs legacy source | `+0.832726` |
| mean dSSIM vs legacy source | `+0.018912` |
| mean dLPIPS vs legacy source | `-0.039955` |
| mean dPSNR vs Phase-J | `-0.000417` |
| mean dSSIM vs Phase-J | `-0.000035` |
| mean dLPIPS vs Phase-J | `+0.000031` |

Phase-J 边界：

| 项 | 结果 |
|---|---|
| `8 / 9` scenes | v100 与 Phase-J delta exact zero |
| `flowers` | v100 比 Phase-J `-0.003750 PSNR / -0.000312 SSIM / +0.000283 LPIPS` |

---

## 7. Qualitative Evidence Paths

### 7.1 Counter standalone package

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/qualitative/ours_26000_v100_checkpoint_attached_ela_endpoint_contact_sheet.png
```

Counter per-view CSV 包含每个 held-out frame 的 clean render、v100 render、GT path 与 per-view delta：

```text
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_per_view_deltas.csv
```

其中每行包括：

```text
base_render_path
method_render_path
gt_path
clean_PSNR / method_PSNR / dPSNR_vs_clean
clean_SSIM / method_SSIM / dSSIM_vs_clean
clean_LPIPS / method_LPIPS / dLPIPS_vs_clean
changed_pixel_fraction / mean_abs_rgb_delta / max_abs_rgb_delta
```

### 7.2 Fixed full9 contact sheets

每个 full9 scene 都有 contact sheet：

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/<scene>/recovery_model/qualitative/ours_26000_v100_checkpoint_attached_ela_endpoint_contact_sheet.png
```

scene 列表：

```text
bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai
```

每个 scene 的核心审计文件：

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/<scene>/recovery_model/endpoint_gate_report.json
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/<scene>/recovery_model/results.json
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/<scene>/recovery_model/per_view.json
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/<scene>/v100_endpoint_command.log
```

---

## 8. Geometry / Topology / Efficiency Notes

### 8.1 Full9 topology inheritance

v100 endpoint 不改变 topology。它继承 compact parent 的 `ratio_0200` topology。

| scene | pre triangles | post triangles | removed fraction |
|---|---:|---:|---:|
| bicycle | `8,479,336` | `8,309,749` | `2.000003%` |
| flowers | `8,683,018` | `8,509,358` | `1.999996%` |
| garden | `11,394,477` | `11,166,587` | `2.000004%` |
| stump | `8,347,076` | `8,180,134` | `2.000006%` |
| treehill | `8,573,839` | `8,402,362` | `2.000003%` |
| room | `11,161,890` | `10,938,652` | `2.000002%` |
| counter | `9,841,068` | `9,644,247` | `1.999996%` |
| kitchen | `9,706,523` | `9,512,393` | `1.999995%` |
| bonsai | `9,750,544` | `9,555,533` | `2.000001%` |

平均 removed fraction：`2.0000007%`。

### 8.2 Counter geometry evidence

counter standalone validation：

| 字段 | 值 |
|---|---:|
| topology unchanged | `True` |
| endpoint triangle delta | `0` |
| endpoint vertex delta | `0` |
| post triangles | `9,644,247` |
| post vertices | `2,478,825` |
| depth AbsRel | `0.007637892` |
| depth MAE | `0.058701707` |
| normal mean angle | `27.085450` |
| geometry inherited | `True` |
| geometry safe | `True` |

解释：

- v100 的 RGB 改善来自 render residual endpoint；
- geometry 指标是 inherited geometry 的安全检查，不是 v100 新几何优化；
- 如果要做 geometry improvement claim，需要另一个真正修改 geometry 且独立评估的实验。

### 8.3 Efficiency notes

fixed full9 per-scene endpoint materialization/evaluation elapsed：

| scene | elapsed sec | support frames | mean changed fraction |
|---|---:|---:|---:|
| bicycle | `102.890` | `169` | `0.810071` |
| flowers | `86.403` | `151` | `0.839378` |
| garden | `92.528` | `161` | `0.976197` |
| stump | `70.923` | `109` | `0.789049` |
| treehill | `67.326` | `123` | `0.704694` |
| room | `161.853` | `272` | `0.945317` |
| counter | `132.216` | `210` | `0.972526` |
| kitchen | `145.881` | `244` | `0.986145` |
| bonsai | `152.528` | `255` | `0.976357` |

平均 elapsed：`112.506 sec`。这个时间包含 endpoint materialization、render/eval、contact sheet/W&B 等流程，不是严格部署 FPS。

效率结论要保守：

- v100 是 endpoint packaging / audit milestone，不是 speedup milestone；
- endpoint sidecar 仍需要专用 loader，不是 vanilla render path；
- compact parent 有 `2%` topology reduction，但 v100 本身没有进一步减 triangles；
- 真正的 runtime claim 需要单独 end-to-end profiling、render-only comparison、VRAM/FPS 表。

---

## 9. Ablation / Claim Boundary

### 9.1 支持的 claim

可以在 PPT 里这样说：

```text
v100 checkpoint-attached endpoint makes Phase-J / ELA repair auditable and packageable:
it preserves compact topology, verifies source provenance, avoids test-GT policy leakage,
passes full9 clean/source gates, and provides per-view qualitative and metric evidence.
```

更短版本：

> v100 把 Phase-J 从“外部 replay 结果”变成“checkpoint-attached audited endpoint”。

### 9.2 不支持的 claim

不要这样说：

- “v100 beats Phase-J.”
- “v100 is a new representation-level improvement.”
- “v100 is a standard MeshSplatting checkpoint.”
- “v100 improves geometry.”
- “v100 proves runtime/VRAM efficiency.”

原因：

- fixed full9 中 `8 / 9` 场景与 Phase-J exact same；
- `flowers` 相对 Phase-J 还略微低；
- endpoint gate 明确标记 topology inherited 与 checkpoint_state_mutated `false`；
- v100 仍依赖 render-time residual endpoint；
- standalone counter 的 v98b checkpoint-baked 对照是 negative。

### 9.3 Ablation 结论

counter evidence 的 ablation 读法：

| 对照 | 结论 |
|---|---|
| compact parent noop | 只继承 compact checkpoint 不够，endpoint residual 是 load-bearing |
| same-evidence noop | evidence plumbing 本身不是收益来源，真正收益来自 residual application |
| strict anchor floor | v100 超过 counter gate floor |
| v98b checkpoint-baked negative | 直接 checkpoint-baked internalization 当前失败 |
| Phase-J ceiling | v100 只是 materialize Phase-J，不是超过 Phase-J |

---

## 10. Weaknesses

1. **不是独立方法提升**
   v100 的价值在 packaging、audit、deployment contract；不是 Phase-J 之后的新质量提升。

2. **仍是 sidecar endpoint**
   需要 `run_checkpoint_attached_ela_endpoint_scene.py` 或等价 loader；vanilla MeshSplatting `render.py` 不会自动消费 sidecar。

3. **geometry claim 边界很窄**
   topology reduction 来自 compact parent；v100 endpoint 自身 triangle/vertex delta 为 `0`。

4. **runtime 未闭合**
   当前 elapsed 不是正式 deployment benchmark；adapter-side inference 成本仍需要专门 profile。

5. **持久化风险**
   主要 qualitative/contact-sheet artifacts 在 `/dev/shm`。PPT 前应确认这些路径仍在，或把图片复制到 durable artifact area。

6. **representation-level 内化仍失败**
   v98b negative 说明把 Phase-J 直接 bake 进 checkpoint 还没有成功。

---

## 11. Next Steps

### 11.1 论文 / 汇报短期

- 把 v100 作为 **Phase-J packaging and audit endpoint** 讲清楚；
- PPT headline 不写“beats Phase-J”，写“checkpoint-attached audited replay of Phase-J”；
- 使用 full9 table 展示相对 clean/source baseline 的强结果；
- 使用 counter package 展示 per-view、non-noop、geometry、topology、W&B 证据完整性。

### 11.2 工程短期

- 将 `/dev/shm` contact sheets 和关键 endpoint_gate_report 持久化到 `outputs/` 或 `assets/`；
- 增加一个轻量 endpoint loader，减少对实验 runner 的耦合；
- 写一个 `render.py` compatibility note：普通 renderer 不读 sidecar，endpoint renderer 才读。

### 11.3 研究中期

- 做真正 representation-level internalization：让 checkpoint 本身吸收 residual，而不是依赖 file-backed report replay；
- 对 v98b negative 做失败归因：teacher loss、rollback、normal/depth anchor、residual capacity 是否不足；
- 加 official runtime / memory / rate-distortion 表，分离 compact parent speed 与 endpoint adapter cost；
- 如果要 claim geometry improvement，必须引入会真正改变 geometry 且保持 RGB/geometry metric 的独立实验。

---

## 12. Result Path Index

### v100 docs

```text
docs/car_model/6-25-v100-FixedFull9-CheckpointAttachedELA-Sidecar.md
docs/car_model/6-25-v100-CheckpointAttachedELA-Counter-Validation.md
```

### fixed full9 summary

```text
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.json
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.csv
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_full9_fixed_20260625/v100_checkpoint_attached_ela_full9_summary.md
```

### standalone counter package

```text
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_comparison.json
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_comparison.csv
outputs/carnet/meshsplatopt/ecsr_phase_v100_checkpoint_attached_ela_counter_20260625/v100_counter_checkpoint_attached_ela_per_view_deltas.csv
```

### run roots

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625
```

### counter visual and gate artifacts

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/qualitative/ours_26000_v100_checkpoint_attached_ela_endpoint_contact_sheet.png
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/endpoint_gate_report.json
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/results.json
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/per_view.json
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/recovery_model/test/ours_26000_v100_checkpoint_attached_ela_endpoint/ela_report.json
```

### W&B offline evidence

fixed full9 has `9 / 9` W&B offline scene dirs under:

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_full9_fixed_20260625/wandb/wandb/offline-run-*
```

standalone counter W&B offline dir:

```text
/dev/shm/peilincai_spcarnet_v100_checkpoint_attached_ela_fixed_counter_20260625/wandb/wandb/offline-run-20260625_033242-2sa2vhiw
```

---

## 13. Slide Deck Suggested Order

1. **Problem**：MeshSplatting checkpoint 训练完后缺少自审计和安全修复。
2. **Main Idea**：train/policy-val surface evidence 可以 certify compaction 和 residual repair。
3. **Phase-J Recap**：guarded ELA 是当前质量来源。
4. **v100 Contribution**：把 Phase-J package 成 checkpoint-attached audited endpoint。
5. **Pipeline**：compact parent -> source report -> sidecar -> render/eval/gate。
6. **Fairness Gate**：provenance、no test-GT policy、frame-set equality、non-noop。
7. **Full9 Table**：9/9 pass，相对 clean/source baseline 强。
8. **Counter Deep Dive**：clean/noop/v98b/Phase-J 对比。
9. **Qualitative Evidence**：contact sheet 和 per-view CSV path。
10. **Geometry / Efficiency**：2% inherited topology reduction，endpoint delta 0，runtime claim 保守。
11. **Claim Boundary**：不是 beats Phase-J，不是 vanilla checkpoint，不是 geometry improvement。
12. **Next Step**：representation-level internalization 与 endpoint loader。

你是 MeshSplatOpt / SPCarNet 项目的研究总指挥、方法设计者、工程执行规划者和严苛顶会审稿人。你的任务不是继续做参数搜索，也不是改写已有 2D ELA 后处理，而是把 SPCarNet 从 image-space residual repair 升级为 representation-level mesh/surface compression and appearance recovery 方法。

项目背景：
SPCarNet 当前 archived full9 Compact-ELA/SOR 版本已经在 selected clean MeshSplatting baseline 之上取得 9/9 RGB + compact + geometry-safe pass，但仍有三个瓶颈：

1. 视觉优势不够强：
   full-frame qualitative difference 往往很细，主要需要 crop / error-reduction map 才能看出收益，说明当前方法偏 residual-level correction，而非 representation-level improvement。

2. 压缩率保守：
   当前 mean triangle reduction 约 5.7632%。room / counter / kitchen 被限制到 0.1%，garden 约 1.5%，说明直接 face deletion 对几何敏感场景过于粗暴。

3. 方法形态容易被质疑为 post-processing：
   当前 ELA 使用 train evidence，并非 test leakage；但若方法长期停留在 renderer-side / image-space residual repair，会被审稿人认为不是对 MeshSplatting representation 的根本推进。

核心目标：
将 SPCarNet 升级为一个可发表的 representation-level 方法，暂命名为：

ECSR: Evidence-Certified Surface Relocation
副标题：Representation-Level Compression and Appearance Recovery for MeshSplatting

核心研究假设：
MeshSplatting 中存在 view-support redundant surface primitives。它们在多个训练视角中具有相近的投影覆盖、深度、法向、appearance residual 和 occlusion support。直接删除这些 primitives 会破坏局部外观或几何；但若通过 evidence-certified contraction / merge 将冗余 surface support 压缩到邻近可靠 support，并将被压缩区域的 appearance residual / residual state 迁移到保留 surface primitive 上，则可以同时提升：

- triangle / vertex / attribute compactness；
- sparse depth / normal geometry stability；
- full-frame RGB / LPIPS；
- local perceptual quality；
- downstream mesh usability；
- 论文贡献的 representation-level 可信度。

你必须围绕以下原则工作：

A. 不允许 test leakage。
   held-out test views 只能用于最终报告，不能用于策略选择、超参选择、crop 选择、alpha 选择、candidate 接受或失败回退。

B. 不允许 per-scene manual tuning。
   允许 scene evidence 自适应，但策略必须由固定算法产生。任何 per-scene 阈值扫描、人工挑选 scene-specific alpha、人工选择 crop，都必须标记为无效。

C. 不允许把最终收益主要归因于 image-space ELA。
   可以保留 ELA 作为 teacher / diagnostic / upper bound / optional adapter，但主方法必须报告 representation-attached residual / surface-attached appearance recovery 的独立贡献。

D. 不允许只报告平均数。
   必须报告 per-scene、per-view、geometry、compactness、local quality、storage、FPS / renderer cost，以及失败 case。

E. 不允许只做概念设计。
   输出必须包含可执行的数据结构、算法步骤、工程任务、smoke tests、acceptance certificates、ablation plan、stop/continue criteria 和 paper framing。

F. 所有 claim 必须可被实验证明。
   若某个 claim 当前无法证明，必须降级为 hypothesis，并设计最小实验验证。

------------------------------------------------------------
第一部分：先完成 Current-State Audit
------------------------------------------------------------

请先审计当前 SPCarNet 状态，形成一张表，不允许直接跳到新方法。

必须确认：

1. 当前 baseline：
   - selected clean MeshSplatting checkpoint；
   - 26000 / 30000 held-out scoring protocol；
   - 9 scenes list；
   - per-scene PSNR / SSIM / LPIPS；
   - sparse depth / normal metrics；
   - triangle / vertex count；
   - checkpoint version；
   - W&B run id；
   - qualitative output path。

2. 当前 SPCarNet archived best：
   - Compact-only；
   - Compact-ELA；
   - Compact-ELA/SOR；
   - prune ratio / ELA alpha / SOR setting；
   - per-scene RGB win/loss；
   - per-scene compactness；
   - per-scene geometry safety；
   - strict all-axis pass；
   - full9 mean triangle reduction；
   - known failure / near-failure cases。

3. 当前瓶颈归因：
   将每个场景归为：
   - appearance-sensitive；
   - geometry-sensitive；
   - occlusion-sensitive；
   - texture-detail-sensitive；
   - compression-friendly；
   - compression-hostile。

4. 当前方法中哪些步骤使用 train evidence，哪些步骤可能被误解为 test-driven。
   如果有任何 test-selected crop、test-selected alpha、test-view fallback，必须指出并提出替代方案。

输出格式：
- Current Protocol Table
- Current Result Table
- Bottleneck Diagnosis Table
- Leakage Risk Table
- One-paragraph conclusion: “为什么当前版本不足以作为顶会主贡献”

------------------------------------------------------------
第二部分：重新定义论文级 Contribution
------------------------------------------------------------

不要把新方法写成 “better ELA”。你必须将论文主张重构为：

We identify train-view-certified redundant surface support in MeshSplatting, compact it through geometry-safe surface contraction, and relocate the lost appearance state onto retained surface primitives. This creates a representation-level compact mesh-splat method that improves or preserves visual quality, geometry reliability, and downstream mesh usability under a fair held-out protocol.

请将贡献拆成三条，不多不少：

Contribution 1:
View-Support Redundancy Graph
- 从单 face prune score 升级为基于多训练视角证据的 surface redundancy graph；
- 节点是 face / vertex / local face cluster；
- 边表示两个 surface supports 是否在 visibility、projection coverage、depth、normal、photometric residual、occlusion role 上冗余；
- 输出 candidate contraction groups，而不是直接 deletion list。

Contribution 2:
Certificate-Carrying Surface Contraction
- 从 face deletion 升级为 edge / vertex / cluster contraction；
- 每个操作必须携带 train-only / policy-validation certificate；
- certificate 同时检查 RGB、SSIM、LPIPS、sparse depth、normal、local topology、degenerate faces、triangle reduction、renderer validity；
- 只接受 Pareto-safe candidate。

Contribution 3:
Surface-Attached Appearance Recovery / Residual Relocation
- 将 ELA residual 从 image plane 回投到保留 surface support；
- residual 不作为 final post-render correction，而作为 per-face / per-vertex / local surface feature / residual SH delta / low-dimensional appearance code；
- 渲染时由 retained primitive 解释 appearance recovery；
- image-space adapter 最多作为 upper bound 或 optional post-hoc ablation，不能作为主收益来源。

请同时写清楚：
- 这不是 QEM：因为 contraction 由 multi-view train evidence 和 photometric / geometry certificate 认证；
- 这不是 neural texture 的简单复刻：因为目标不是只增强外观，而是在 compression / contraction 后恢复 lost appearance capacity；
- 这不是 3DGS pruning 的简单迁移：因为对象是 connected opaque mesh support，且必须维护 downstream mesh validity；
- 这不是 post-processing：因为 residual state 存在 representation 中，test rendering 不读取 GT residual，也不做 per-test image optimization。

------------------------------------------------------------
第三部分：设计 Surface Evidence Cache
------------------------------------------------------------

实现或规划一个 Surface Evidence Cache，作为 ECSR 的基础数据结构。

对每个 train view 渲染并缓存：

Per-pixel fields:
- RGB_render；
- RGB_gt；
- residual r = RGB_gt - RGB_render；
- absolute error；
- LPIPS / perceptual proxy map，如果可行；
- face_id 或 nearest surface id；
- barycentric coordinate；
- depth_render；
- sparse depth target / confidence，如果有；
- normal_render；
- normal target / confidence，如果有；
- opacity / visibility / contribution weight；
- foreground / background / occlusion mask；
- local image gradient / texture strength。

Per-face / per-vertex aggregated fields:
- visibility count；
- visible view set；
- projected area statistics；
- residual mean / variance；
- residual direction consistency；
- depth residual mean / variance；
- normal consistency；
- photometric contribution；
- silhouette / boundary participation；
- occlusion participation；
- local texture complexity；
- deletion sensitivity if already available；
- prior prune score if already available；
- topology fields: adjacency, boundary flag, component id, valence, face area, aspect ratio。

必须加入两个 diagnostic：

Diagnostic A: Surface addressability of residual
回答：ELA residual 是否能被稳定回投到 surface support？
指标：
- top residual pixels 中有多少能找到 stable face_id；
- 同一 surface support 在多视角 residual 是否方向一致；
- residual 是否集中在 texture / geometry support 上；
- residual 是否只是 view-specific artifact。

Diagnostic B: Relocation necessity
回答：当前 compact-only 失败是否真的来自 appearance capacity loss，而不是纯 geometry break？
指标：
- 删除 / 收缩前后 residual 增量是否集中在被操作 surface 的投影区域；
- residual 是否可被邻近 retained support 解释；
- depth / normal 是否未明显变坏但 RGB / LPIPS 变坏；
- 若是，则说明 appearance relocation 有意义；
- 若 depth / normal 同时坏，则应优先修 contraction / topology，而不是加 residual。

输出：
- Surface residual heatmap；
- residual-on-mesh heatmap；
- per-scene residual concentration score；
- top-K residual support list；
- failure crop alignment report；
- conclusion: “哪些场景适合 residual relocation，哪些场景必须先解决 geometry contraction”。

------------------------------------------------------------
第四部分：构建 View-Support Redundancy Graph
------------------------------------------------------------

构建 graph G = (N, E)。

节点 N：
- 初始可以是 face；
- 若 face 太细碎，则先做 local cluster：
  - connected faces；
  - similar normal；
  - similar depth range；
  - similar material / color residual；
  - shared visibility views；
  - bounded geodesic radius。

边 E：
仅在满足基本 topology / proximity 条件时建立：
- adjacent face / edge-neighbor；
- geodesic distance 小；
- projected overlap 高；
- depth ordering 一致；
- normal direction 接近；
- 不跨明显 silhouette / occlusion boundary；
- 不跨 high-depth-discontinuity；
- 不跨 strong normal discontinuity，除非 policy 证明安全。

为每条候选边或 cluster pair 计算以下证据：

1. Visibility overlap:
   V_ij = |Views(i) ∩ Views(j)| / |Views(i) ∪ Views(j)|

2. Projection support overlap:
   P_ij = average IoU of projected masks over shared visible train views

3. Depth consistency:
   D_ij = robust similarity between rendered depths over overlapping pixels

4. Normal consistency:
   N_ij = robust average of max(0, dot(n_i, n_j))

5. Photometric residual compatibility:
   R_ij = similarity between residual statistics, residual direction, and texture class

6. Appearance transferability:
   A_ij = can residual from support i be explained by retained support j under barycentric / local projection?

7. Occlusion risk:
   O_ij = penalty if i or j participates in sparse occluder, silhouette, thin structure, or depth discontinuity

8. Topology risk:
   T_ij = penalty if contraction creates degenerate faces, inverted normals, non-manifold edge, disconnected important component, or severe valence spike

Redundancy score:
S_ij = weighted combination of V, P, D, N, R, A minus O and T

但不要把 score 当成最终决策。score 只生成 candidates；最终是否接受必须由 certificate 决定。

Candidate 类型：
- edge collapse；
- vertex pair contraction；
- face cluster contraction；
- local retriangulation；
- attribute-only merge；
- no-topology appearance relocation；
- fallback deletion only if contraction unavailable and certificate passes。

输出：
- candidate list；
- per-candidate evidence record；
- risk flags；
- expected triangle / vertex reduction；
- expected affected support；
- affected train-view masks；
- affected policy-val masks。

------------------------------------------------------------
第五部分：Certificate-Carrying Surface Contraction
------------------------------------------------------------

将当前 face deletion 替换为更温和、更可解释的 contraction / merge operator。

每个 candidate contraction 必须经历四层检查：

Layer 1: Static topology smoke test
- face index range valid；
- vertex index range valid；
- no NaN / Inf；
- no zero-area face；
- no inverted face beyond threshold；
- no catastrophic valence spike；
- no illegal non-manifold if renderer cannot support；
- component count not unexpectedly changed；
- boundary / silhouette not destroyed；
- tensor lengths match MeshSplatting checkpoint requirements；
- renderer can load checkpoint；
- one train view smoke render no crash。

Layer 2: Local rendering certificate
只渲染 affected train-view local regions：
- local RGB Δ not worse beyond epsilon；
- local SSIM Δ not worse；
- local LPIPS Δ not worse；
- sparse depth local error not worse；
- normal angular error not worse；
- local silhouette / edge mismatch not worse；
- residual heat not newly concentrated。

Layer 3: Policy-validation certificate
在 policy-validation train subset 上验证：
- full-frame PSNR / SSIM / LPIPS Pareto-safe；
- depth / normal Pareto-safe；
- compactness improved；
- no newly invalid geometry；
- no reliance on test view。

Layer 4: Global acceptance policy
候选按 risk-adjusted compactness gain 排序，逐个或 batch 接受。
接受规则必须固定，不得 per-scene 手调。
若 candidate 失败：
- rollback；
- log failure reason；
- mark candidate class as risky；
- 不允许凭人工观察强行保留。

Certificate 输出必须保存 JSON：
{
  scene,
  candidate_id,
  operator_type,
  affected_faces,
  affected_vertices,
  evidence_score,
  topology_before,
  topology_after,
  train_local_metrics_before_after,
  policy_val_metrics_before_after,
  compactness_gain,
  accepted,
  rejection_reason,
  random_seed,
  code_version
}

------------------------------------------------------------
第六部分：Surface-Attached Appearance Recovery / Residual Relocation
------------------------------------------------------------

这是方法从 post-processing 升级为 representation-level 的核心。请明确区分四种版本：

Version 0: ELA diagnostic teacher
- 使用当前 ELA 产生 train residual；
- 仅用于分析 residual signal；
- 不作为最终主方法。

Version 1: Attribute-only recovery
- 不改 renderer；
- 在 contraction / pruning 后重新优化 retained vertex / face SH coefficients；
- 使用 train-fitting subset；
- policy-validation subset 选择 early stop / strength；
- 这是最低风险 baseline。

Version 2: Surface residual SH delta
- 为 retained vertex / face 增加 residual SH delta 或 residual RGB delta；
- delta 从 removed / contracted support 的 residual 聚合得到；
- delta 有 magnitude bound、smoothness regularization、visibility weighting；
- 渲染时 delta 与原 appearance attribute 合成；
- 不读取 test residual。

Version 3: Low-dimensional surface appearance code
- 为 face / vertex / local cluster 存低维 residual code；
- 小型 decoder 根据 surface code、normal、view direction、barycentric coordinate 输出 appearance correction；
- decoder 必须很轻，参数量和存储量计入 compactness；
- 不能变成黑盒 neural renderer；
- 必须证明即使移除 final image-space ELA，仍保留主要收益。

Residual 聚合方式：
对每个 contraction group C，被移除或收缩的 source supports 为 S_removed，保留 supports 为 S_keep。
在 fitting train views 中，将 residual r(p) 通过 face_id / barycentric / nearest projection 回投到 source supports。
然后解一个 regularized relocation problem：

min_delta Σ_v Σ_p∈affected(v) w(p) || Render(M_contracted, A + delta; p) - GT(p) || 
          + λ_mag ||delta||²
          + λ_smooth Σ_(i,j)∈adj ||delta_i - delta_j||²
          + λ_geo * geometry_penalty
          + λ_leak * policy_overfit_penalty

约束：
- delta 只能附着在 retained surface primitive；
- delta 不能按 test view 存储；
- delta 不能按 image coordinate 存储；
- delta 的参数量必须计入 storage；
- delta 不能破坏 depth / normal；
- delta 不能导致 train 好、policy-val 坏。

必须输出：
- residual relocation map；
- retained support heatmap；
- delta magnitude distribution；
- per-scene percentage of ELA gain recovered by surface-attached residual；
- surface residual vs image-space ELA ablation；
- storage overhead；
- renderer cost overhead。

主指标：
surface-attached residual should recover most of ELA’s benefit without using final image-space correction.

若 Version 2 / 3 无法稳定，必须诚实报告，并回退到 Version 1 as minimal representation-level appearance recovery。

------------------------------------------------------------
第七部分：Train-Only Pareto Policy
------------------------------------------------------------

划分训练视角：

1. fitting_train:
   - build evidence；
   - estimate residual；
   - generate candidates；
   - optimize appearance relocation。

2. policy_val:
   - accept / reject candidates；
   - choose residual strength；
   - choose early stopping；
   - choose candidate budget；
   - choose scene-adaptive but rule-based policy。

3. held_out_test:
   - final report only；
   - never used for policy selection。

必须固定 split seed，并记录 split file。
若 scene train views 很少，使用 k-fold policy validation，但 held-out test 仍不可参与。

Pareto rule：
一个 candidate 或 model 被接受，当且仅当：
- compactness improves；
- policy-val RGB not worse beyond fixed epsilon；
- policy-val LPIPS not worse beyond fixed epsilon；
- policy-val sparse depth not worse beyond fixed epsilon；
- policy-val normal not worse beyond fixed epsilon；
- invalid mesh count does not increase；
- storage / runtime overhead is bounded。

epsilon 必须在方法设计阶段固定，不能看 test 后改。

------------------------------------------------------------
第八部分：实验矩阵
------------------------------------------------------------

必须包含以下 variants：

1. Clean MeshSplatting baseline 26000 / 30000
   目的：same-protocol baseline envelope。

2. Current archived Compact-ELA/SOR
   目的：当前最强版本，证明新方法不是退步。

3. Compact-only deletion
   目的：暴露当前 direct face deletion 的瓶颈。

4. View-support graph + deletion
   目的：隔离 graph 是否比单 face score 更好。

5. Certificate contraction only
   目的：隔离 topology operator 是否改善 compactness / geometry。

6. Attribute-only recovery after contraction
   目的：最低风险 representation-level appearance recovery。

7. Surface residual SH delta
   目的：测试 residual 是否能真正附着到 surface representation。

8. Low-dimensional surface appearance code
   目的：测试更强 appearance capacity 是否值得。

9. ELA only
   目的：image-space upper bound / teacher。

10. Surface-attached residual + no image ELA
    目的：证明主收益不来自 post-processing。

11. Full ECSR
    目的：最终方法。

12. No policy-validation split
    目的：暴露 overfitting / leakage 风险。

13. No geometry certificate
    目的：证明 depth / normal guard 必要。

14. QEM-style contraction baseline if feasible
    目的：证明 ECSR 不是简单 mesh simplification。

15. Neural-texture-like appearance-only baseline if feasible
    目的：证明 ECSR 不只是加 appearance capacity。

每个 variant 必须报告：
- PSNR / SSIM / LPIPS；
- per-view metrics；
- per-scene metrics；
- mean and median；
- triangle count；
- vertex count；
- attribute storage；
- total checkpoint size；
- renderer FPS / memory if feasible；
- sparse depth；
- normal；
- invalid face count；
- degenerate face count；
- component count；
- local metrics；
- qualitative full-frame；
- train-defined local crops；
- error-reduction maps。

------------------------------------------------------------
第九部分：Local Metrics 与 Qualitative Protocol
------------------------------------------------------------

不能再只靠人工挑 crop。必须定义 train-evidence-driven local evaluation。

Crop / mask selection protocol:
- 使用 fitting_train residual heatmap 定义 top-K surface supports；
- 将这些 support 投影到 policy-val / held-out-test views；
- 自动生成 corresponding local masks / crops；
- test 上不能人工重新挑选；
- crop 选择只依赖 train evidence 和固定规则。

Local metrics:
- local PSNR；
- local SSIM；
- local LPIPS；
- local residual reduction；
- edge / texture region perceptual score；
- depth / normal local consistency；
- failure support recovery rate。

Qualitative panels:
- same camera；
- same crop；
- GT；
- Clean MeshSplatting；
- Current Compact-ELA/SOR；
- ECSR compact-only；
- ECSR surface residual；
- Full ECSR；
- error maps；
- residual-on-surface visualization；
- contraction certificate visualization。

必须说明：
如果 full-frame gain 小，但 local metric 明显提升，则论文 claim 应谨慎改为 local-detail-preserving compact representation，而不是夸大全局 SOTA。

------------------------------------------------------------
第十部分：工程实现路线
------------------------------------------------------------

Phase A: Evidence Cache + Diagnostic
目标：证明 residual 可被 surface-addressed。
任务：
- 实现 train-view render cache；
- 导出 face_id / barycentric / depth / normal / residual；
- 聚合 per-face residual；
- 生成 residual-on-surface heatmap；
- 生成 top residual supports；
- 在 flowers / garden / treehill / bicycle / room 至少五个场景验证。

晋升标准：
- top residual regions 与当前 failure crops 对齐；
- residual 在 surface 上有多视角一致性；
- 能区分 appearance loss 与 geometry break；
- 无 test view 参与。

Phase B: View-Support Redundancy Graph
目标：替代单 face prune score。
任务：
- 构建 face adjacency；
- 构建 local clusters；
- 计算 visibility / projection / depth / normal / residual compatibility；
- 输出 candidate contraction groups；
- 生成 risk flags。

晋升标准：
- indoor / garden 不再只给 deletion；
- candidates 有可解释证据；
- compact-only 不复现严重 geometry 失败；
- triangle reduction 至少不低于当前版本。

Phase C: Certificate-Contraction
目标：将 deletion 换成安全 contraction。
任务：
- 实现 edge collapse / vertex pair contraction / cluster contraction prototype；
- 保持 MeshSplatting checkpoint tensor consistency；
- degenerate cleanup；
- renderer smoke test；
- train local certificate；
- policy-val certificate；
- rollback / logging。

晋升标准：
- RGB 不低于当前 compact-only；
- depth / normal 不低于当前；
- garden / indoor triangle reduction 超过 micro-prune；
- 无 renderer crash；
- no test leakage。

Phase D: Surface-Attached Appearance Recovery
目标：把 ELA 主要收益迁回 representation。
任务：
- Version 1: attribute-only recovery；
- Version 2: residual SH / RGB delta；
- Version 3: low-dimensional surface code；
- surface residual strength 由 policy-val 选择；
- report percentage of ELA gain recovered；
- final image-space ELA 只作为 optional ablation。

晋升标准：
- surface-attached residual recovers meaningful portion of ELA benefit；
- outdoor scenes LPIPS / local metrics 提升；
- indoor / garden geometry 不退；
- storage overhead 可控；
- qualitative crop 更直观；
- 不靠 post-render alpha 赢。

Phase E: Full9 Same-Protocol Validation
目标：形成新主结果。
必须重新跑：
- selected clean MeshSplatting baseline；
- current archived best；
- full9 ECSR；
- W&B logging；
- geometry JSON；
- per-view metrics；
- qualitative full-frame + local panels；
- ablations；
- failure case report。

晋升标准：
- 保留 9/9 RGB + compact + geometry-safe，或明确说明 trade-off；
- mean triangle reduction 明显高于 5.7632%；
- strict all-axis pass 高于当前 5/9，或至少 garden / indoor geometry 明确改善；
- surface-attached residual 独立 ablation 成立；
- 方法解释能被审稿人接受为 representation-level。

------------------------------------------------------------
第十一部分：风险处理
------------------------------------------------------------

Risk 1: Renderer integration too invasive
优先级回退：
- Version 1 attribute-only recovery；
- per-vertex SH delta；
- per-face residual color delta；
- lightweight CPU-side checkpoint rewrite；
- 最后才改 CUDA renderer。

Risk 2: Topology surgery breaks checkpoint
必须加入：
- tensor length audit；
- face index audit；
- degenerate cleanup；
- renderer smoke test；
- rollback mechanism；
- per-candidate JSON log。

Risk 3: residual relocation 仍像 post-processing
解决：
- final main method 禁用 image-space ELA；
- surface residual ablation 单独报告；
- residual state 存在 mesh / vertex / face / local code；
- storage overhead 计入模型；
- 渲染不读取 GT residual。

Risk 4: local gain 不变成 full-frame gain
解决：
- 引入 train-defined local metric；
- 不夸大全局指标；
- 论文 claim 改为 detail-preserving compactness；
- 若 full-frame 无提升但 compactness / geometry / local detail 强，仍可形成合理贡献。

Risk 5: contraction 不如 deletion
解决：
- 记录 operator failure；
- 使用 graph + attribute-only recovery；
- 将 contraction 降级为 optional；
- 论文主 claim 改为 evidence-certified compactness policy + surface appearance recovery。

Risk 6: appearance residual overfits train
解决：
- policy-validation split；
- smoothness regularization；
- magnitude bound；
- view-direction capacity limit；
- report no-policy-val ablation；
- report train vs policy-val gap。

------------------------------------------------------------
第十二部分：顶会审稿人反驳表
------------------------------------------------------------

你必须输出 Reviewer Objection Audit：

Objection 1:
“This is just post-processing.”
Response:
主方法不使用 final image-space correction；residual is attached to retained surface primitives；image ELA only appears as teacher / upper bound / ablation。

Objection 2:
“This is just mesh simplification / QEM.”
Response:
ECSR uses train-view photometric, depth, normal, visibility, occlusion certificates, and relocates appearance state after contraction。

Objection 3:
“This is just neural texture.”
Response:
ECSR is not appearance-only; it starts from compression / contraction of redundant mesh support and measures compactness, geometry validity, and downstream mesh usability。

Objection 4:
“Hyperparameters are scene-tuned.”
Response:
Policy is fixed; scene adaptation only comes from train evidence; policy-val split selects candidates; held-out test is final only。

Objection 5:
“Local crops are cherry-picked.”
Response:
Crops are derived from train evidence support masks and projected to test by fixed rules.

Objection 6:
“Compression is too small.”
Response:
Report triangle, vertex, attribute storage, FPS, and rate-distortion curves; if triangle reduction remains small, do not overclaim compression SOTA.

Objection 7:
“Geometry is unsafe.”
Response:
Every contraction carries sparse depth, normal, topology, and renderer validity certificates; no geometry certificate ablation shows why guards matter。

------------------------------------------------------------
第十三部分：最终输出要求
------------------------------------------------------------

请最终输出以下内容：

1. One-page paper story
   - problem；
   - insight；
   - method；
   - why not post-processing；
   - why useful for downstream mesh；
   - key expected experiments。

2. Method specification
   - data structures；
   - equations；
   - candidate generation；
   - certificate logic；
   - residual relocation logic；
   - policy split；
   - renderer integration options。

3. Engineering plan
   - files likely to modify；
   - new scripts；
   - cache format；
   - JSON schema；
   - smoke tests；
   - run commands；
   - W&B logging fields。

4. Experiment plan
   - full9 scenes；
   - variants；
   - metrics；
   - qualitative protocol；
   - local metric protocol；
   - storage / speed protocol。

5. Stop / continue decision tree
   Continue if:
   - mean triangle reduction improves clearly over 5.7632% without losing current RGB wins；or
   - strict all-axis pass improves；or
   - surface-attached residual recovers most ELA benefit；or
   - local detail gains are strong and train-defined。
   
   Stop or redesign if:
   - improvements require per-scene manual tuning；
   - full9 RGB pass collapses；
   - gains only appear in test-selected examples；
   - surface residual cannot be distinguished from post-processing；
   - contraction repeatedly breaks geometry / renderer consistency。

6. Honest uncertainty
   对每个 claim 标注：
   - proven by current data；
   - likely but needs diagnostic；
   - risky；
   - not yet supported。

语气要求：
- 严肃、研究型、工程可执行；
- 不要营销化；
- 不要声称已有未跑出的结果；
- 不要为了显得强而隐藏失败风险；
- 遇到不确定处，提出最小验证实验；
- 所有结论都必须能落到 metric、ablation、certificate 或 qualitative protocol。
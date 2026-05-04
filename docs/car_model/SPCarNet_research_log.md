# SP-CarNet Research Log

Single source of truth for "what was tried under the SP-CarNet research line and how it went". Date-stamped, append-only. Each entry links to the relevant design / implementation / smoke / failure documents per the policy in `SPCarNet_radical_RFC.md` §8.

---

## 2026-05-04 — MeshSplatOpt F34 parking sparse-depth long continuation — FAIL_KEEP_F33_26K

**Outcome**: Ran a W&B-logged long-continuation control for the current strongest
`parking_phone_tiny` row. The run starts from F33 CSEF70 + sparse-depth at iteration
`26000`, keeps strict topology freeze, continues to `30000`, then renders and evaluates
with independent image metrics and sparse COLMAP geometry.

**W&B**: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/d3nyktd4`

**Verification**:
- topology at 30000: `2,564,473` triangles, `1,661,616` vertices
- independent render metrics: PSNR `18.51044464111328`, SSIM `0.632079541683197`, LPIPS `0.3543379008769989`
- sparse geometry: AbsRel `0.07902251998756822`, Depth MAE `1.8474553216191132`, normal mean angle `44.42787469632362`
- comparison to F33: PSNR `-0.201885`, SSIM `-0.015650`, LPIPS `+0.016079`, AbsRel `-0.000048`, Depth MAE `-0.006560`, normal `+0.392167`

**Decision**: `FAIL_KEEP_F33_26K`. F34 answers the long-budget fairness concern:
continuing the best sparse-depth parking row from `26k` to `30k` slightly improves
sparse depth proxies but visibly and quantitatively hurts render quality. F33 remains
the validated parking headline row; F34 is recorded as a negative long-continuation
control.

**Linked artefacts**:
- `docs/car_model/final_stageF34_parking_long_continuation_report.md`
- `outputs/carnet/meshsplatopt/final_stageF34_parking_sparse_depth_long_continuation/`

---

## 2026-05-02 — MeshSplatOpt R14.19-R14.20 bonsai medium continuation — MEDIUM_CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN

**Outcome**: Ran W&B-logged medium continuations from iteration `2000` to `4000` on `bonsai` for both accepted non-delete snap and unedited baseline continuation.

**W&B**:
- snap: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fjzy6lun`
- baseline: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gxeskhta`

**Metrics at 4000**:
- snap: PSNR `15.81759262084961`, SSIM `0.33459141850471497`, LPIPS `0.5731096863746643`, AbsRel `0.40904864176963485`, normal `47.83674765098326`, triangles `5090526`
- baseline continuation: PSNR `15.834700584411621`, SSIM `0.33469849824905396`, LPIPS `0.5714929699897766`, AbsRel `0.40514114339865287`, normal `48.11943889631045`, triangles `5090601`

**Decision**: `MEDIUM_CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN`. The current snap selector is not a full-budget candidate. It remains useful as safety/stability infrastructure, but R15 needs topology retention, render-residual proposal selection, or a stronger equal-budget gate before 7000iter sweeps.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_19_20_bonsai_medium_continuation_report.md`
- `outputs/carnet/meshsplatopt/stageR14_19_bonsai_snap_medium_continuation_2000step/`
- `outputs/carnet/meshsplatopt/stageR14_20_bonsai_baseline_medium_continuation_2000step/`

---

## 2026-05-02 — MeshSplatOpt R14.18 bonsai equal-step control — CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN

**Outcome**: Ran a W&B-logged 200-step unedited baseline continuation on `bonsai` from iteration `2000` to `2200`, matching the R14.17 snap-recovery budget.

**W&B**: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kic0euiq`

**Verification**:
- train/render/metrics exit codes: `0/0/0`
- topology after continuation: `2487474` triangles, `2478890` vertices
- render metrics at 2200: PSNR `13.274771690368652`, SSIM `0.2403060346841812`, LPIPS `0.6113919019699097`
- sparse geometry at 2200: AbsRel `0.47338970412280024`, Depth MAE `4.765895956720541`, normal mean angle `49.19677426124215`

**Comparison to R14.17 snap recovery**:
- snap is lower on PSNR by `0.0007829666137695312`
- snap is higher on SSIM by `0.00008484721183776855`
- snap is worse on LPIPS by `0.00024008750915527344`
- snap is worse on AbsRel by `0.0010631128424631347`
- snap is worse on normal mean angle by `0.11891194155121613`

**Decision**: `CONTROL_PASS_NEGATIVE_FOR_SNAP_GAIN`. R14 snap remains a safe real-edit and recovery-stability mechanism, but this selector should not be claimed as an equal-step quality improvement.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_18_bonsai_equal_step_control_report.md`
- `outputs/carnet/meshsplatopt/stageR14_18_bonsai_baseline_continuation_200step/`

---

## 2026-05-02 — MeshSplatOpt R14.17 bonsai snap recovery — PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET

**Outcome**: Ran a W&B-logged 200-step recovery diagnostic on the accepted `bonsai` non-delete `SNAP_VERTICES` checkpoint. Training resumed from iteration `2000` to `2200`, then rendered, evaluated with image metrics, and passed sparse COLMAP geometry evaluation.

**W&B**: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/8qdzfu6h`

**Verification**:
- train/render/metrics exit codes: `0/0/0`
- topology after recovery: `2487474` triangles, `2478890` vertices
- render metrics at 2200: PSNR `13.273988723754883`, SSIM `0.24039088189601898`, LPIPS `0.6116319894790649`
- sparse geometry at 2200: AbsRel `0.47445281696526337`, Depth MAE `4.772623802825101`, normal mean angle `49.315686202793366`

**Decision**: `PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET`. Non-delete recovery is stable and improves over the 2000iter baseline, but it uses 200 extra steps and is slightly weaker than the R14.13 delete-recovery diagnostic.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_17_bonsai_snap_postedit_recovery_diagnostic_report.md`
- `outputs/carnet/meshsplatopt/stageR14_17_bonsai_snap_postedit_recovery_diagnostic/`

---

## 2026-05-02 — MeshSplatOpt R14.14-R14.16 non-delete snap gates — PASS

**Outcome**: Implemented a real checkpoint area-outlier `SNAP_VERTICES` selector and validated it with render-backed gates on `parking_phone_tiny`, `bonsai`, and `courtyard`. This is the first R14 real-checkpoint non-delete edit pass across multiple scenes.

**Selection**:
- parking: selected face `727102`, area `247.026230 -> 15.439142`, max displacement `12.383613`
- bonsai: selected face `2462659`, area `164.058243 -> 10.253642`, max displacement `10.094805`
- courtyard: selected face `404443`, area `873.247437 -> 54.577950`, max displacement `23.436289`

**Gate deltas**:
- parking: PSNR `+0.000002861`, SSIM `-0.000001252`, LPIPS `-0.000002086`, AbsRel `0.0`
- bonsai: PSNR `-0.000190735`, SSIM `-0.000013679`, LPIPS `-0.000055611`, AbsRel `0.0`
- courtyard: PSNR `-0.005673409`, SSIM `+0.000041097`, LPIPS `+0.000064254`, AbsRel `0.0`

**Decision**: `PASS_DIAGNOSTIC_CROSS_SCENE`. This unblocks public-scene W&B recovery for non-delete edits, but it is still not an equal-budget training win.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_14_16_snap_nondelete_cross_scene_report.md`
- `docs/car_model/meshsplatopt_stageR14_aggregate_decision_report.md`

---

## 2026-05-02 — MeshSplatOpt Stage R14.13 bonsai post-edit recovery diagnostic — PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET

**Outcome**: Ran a W&B-logged 200-step recovery diagnostic on `bonsai` after the R14.11 accepted area-outlier edit. The edited checkpoint resumed from iteration 2000 and trained to iteration 2200, then rendered and evaluated independently.

**W&B**: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z498br53`

**Verification**:
- train/render/metrics exit codes: `0/0/0`
- topology after recovery: `2487473` triangles, `2478890` vertices
- render metrics at 2200: PSNR `13.276382446289062`, SSIM `0.24055197834968567`, LPIPS `0.6113873720169067`
- sparse geometry at 2200: AbsRel `0.4733479577347401`, Depth MAE `4.762276469029142`, normal mean angle `49.21947049923495`

**Decision**: `PASS_DIAGNOSTIC_NOT_EQUAL_BUDGET`. Recovery is stable and improves metrics versus the 2000iter baseline, but it uses 200 extra training steps and must not be reported as an equal-budget R14 win.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_13_bonsai_postedit_recovery_diagnostic_report.md`
- `outputs/carnet/meshsplatopt/stageR14_13_bonsai_postedit_recovery_diagnostic/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.12 courtyard area-outlier diagnostic — PASS_DIAGNOSTIC

**Outcome**: Ran the automatic checkpoint area-outlier selector and render-backed gate on ETH3D `courtyard`, the third scene tested by this conservative real checkpoint edit path.

**Verification**:
- selected face: `404443`
- selected area: `873.2474365234375`
- median triangle area: `0.007861965335905552`
- triangles: `410254 -> 410253`
- render deltas: PSNR `-0.0005950927734375`, SSIM `0.000011831521987915039`, LPIPS `0.00007200241088867188`
- geometry deltas: AbsRel `0.0`, Depth MAE `0.0`, normal mean angle `0.0`

**Decision**: `PASS_DIAGNOSTIC`. The conservative area-outlier selector and render-backed gate are stable on a third scene. This supports safety/infrastructure, not the final repair-quality claim.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_12_courtyard_area_outlier_diagnostic_report.md`
- `outputs/carnet/meshsplatopt/stageR14_12_courtyard_area_outlier_diagnostic/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.11 bonsai area-outlier diagnostic — PASS_DIAGNOSTIC

**Outcome**: Ran the automatic checkpoint area-outlier selector and render-backed gate on a second public scene, Mip-NeRF 360 `bonsai`. The selector was also optimized to compute triangle areas with torch chunking for large checkpoints.

**Verification**:
- selected face: `2462659`
- selected area: `164.05824279785156`
- median triangle area: `0.0002083771105390042`
- triangles: `2487474 -> 2487473`
- render deltas: PSNR `-0.0003681182861328125`, SSIM `-0.000012442469596862793`, LPIPS `-0.0000036954879760742188`
- geometry deltas: AbsRel `0.0`, Depth MAE `0.0`, normal mean angle `0.000000008903604964416445`

**Decision**: `PASS_DIAGNOSTIC`. This validates second-scene stability for conservative checkpoint-statistics selection and render-backed gating. It is not a second W&B medium recovery run, so it does not by itself upgrade R14 to full `PASS`.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_11_bonsai_area_outlier_diagnostic_report.md`
- `outputs/carnet/meshsplatopt/stageR14_11_bonsai_area_outlier_diagnostic/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.10 medium area-outlier pilot — SOFT PASS_SINGLE_SCENE

**Outcome**: Ran the first W&B-logged medium-budget MeshSplatOpt candidate on `parking_phone_tiny`. The run starts from the R14.9 automatic area-outlier edit at iteration 200, resumes training to iteration 2000, renders independently, runs `metrics.py`, and evaluates sparse COLMAP geometry.

**W&B**: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/81kwhzr3`

**Verification**:
- train/render/metrics exit codes: `0/0/0`
- candidate topology: `783509` triangles, `822064` vertices
- current-branch 2000iter baseline render: PSNR `11.599437713623047`, SSIM `0.2702677547931671`, LPIPS `0.6347319483757019`
- MeshSplatOpt candidate render: PSNR `13.276764869689941`, SSIM `0.30384060740470886`, LPIPS `0.6081721186637878`
- baseline geometry: AbsRel `0.42787965657189714`, Depth MAE `4.414160625200222`, normal mean angle `52.565184963415106`
- candidate geometry: AbsRel `0.3640420630578014`, Depth MAE `3.806375643108584`, normal mean angle `52.672900862227785`

**Decision**: `SOFT PASS_SINGLE_SCENE`. The candidate improves all independent render metrics and sparse depth geometry on one scene, with small topology growth and a small normal-angle regression. Full R14 PASS still requires at least a second scene and stronger baseline comparison where compatible.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_10_medium_area_outlier_pilot_report.md`
- `outputs/carnet/meshsplatopt/stageR14_10_medium_area_outlier_recovery/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.9 area-outlier real edit selection — PASS

**Outcome**: Added and ran the first automatic real-checkpoint edit selector after the topology audit. The selector uses checkpoint triangle-area statistics, not shared-edge boundary loops, and emits an auditable `DELETE_TRIANGLES` edit JSON for the largest extreme area outlier.

**Verification**:
- selected face: `55379`
- selected area: `15501.270805580434`
- median triangle area: `0.005547030811843575`
- render-backed gate ran on GPU 4
- triangles: `64497 -> 64496`
- render deltas: PSNR `0.0`, SSIM `0.0`, LPIPS `0.0`
- geometry deltas: AbsRel `0.0`, Depth MAE `0.0`, normal mean angle `0.0`

**Decision**: `PASS`. The automatic real edit-selection chain now works end to end for a conservative checkpoint-statistics deletion. This is infrastructure evidence, not the final R14 full-repair claim.

**Linked artefacts**:
- `scripts/car_model/meshsplatopt_select_checkpoint_area_outlier_edit.py`
- `docs/car_model/meshsplatopt_stageR14_9_area_outlier_real_edit_selection_report.md`
- `outputs/carnet/meshsplatopt/stageR14_9_area_outlier_real_edit_selection/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.8 checkpoint topology evidence audit — PASS_WITH_EDGE_CSEF_FAIL

**Outcome**: Added and ran a real checkpoint topology-evidence audit before automatic edit selection. The audit found that the saved triangle-splat checkpoint is a triangle-soup representation, not an edge-connected mesh.

**Verification**:
- vertices: `193491`
- triangles: `64497`
- connected components: `64497`
- largest component faces: `1`
- shared edges: `0`
- boundary face fraction: `1.0`

**Decision**: `PASS_WITH_EDGE_CSEF_FAIL`. The audit itself is successful, but shared-edge boundary-loop CSEF is invalid for real checkpoint proposal selection. Real edit selection must use spatial adjacency, render residuals, sparse COLMAP evidence, or checkpoint/raster evidence rather than mesh edge connectivity.

**Linked artefacts**:
- `scripts/car_model/meshsplatopt_audit_checkpoint_topology_evidence.py`
- `docs/car_model/meshsplatopt_stageR14_8_checkpoint_topology_evidence_audit.md`
- `outputs/carnet/meshsplatopt/stageR14_8_checkpoint_topology_evidence_audit/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.7 teacher recovery tiny — PASS

**Outcome**: Upgraded the teacher recovery runner from cache-only contract to a real tiny recovery path. The edited R14.5 checkpoint was copied, resumed from iteration 200, trained for 20 recovery steps to iteration 220 with W&B online, rendered, evaluated with independent metrics, and checked with sparse COLMAP geometry.

**Verification**:
- W&B run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/n05mce4y`
- train/render/metrics exit codes: `0/0/0`
- topology after recovery: `64498` triangles, `193494` vertices
- render metrics after recovery: PSNR `10.995698928833008`, SSIM `0.29370972514152527`, LPIPS `0.6429890990257263`
- geometry after recovery: AbsRel `0.325047677579098`, Depth MAE `3.6494193758930376`, normal mean angle `51.93818681106907`

**Decision**: `PASS`. This validates real checkpoint resume/recovery with W&B and independent evaluation. It remains a tiny functionality test; R14 still needs real public-scene edit selection and medium-budget comparison before paper-facing claims.

**Linked artefacts**:
- `scripts/car_model/meshsplatopt_run_teacher_recovery.py`
- `docs/car_model/meshsplatopt_stageR14_7_teacher_recovery_tiny_report.md`
- `outputs/carnet/meshsplatopt/stageR14_7_teacher_recovery_tiny/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.6 render-backed checkpoint gate — PASS

**Outcome**: Added a reusable checkpoint-level counterfactual gate that compares baseline and edited candidate models using independent render metrics, sparse COLMAP geometry metrics, and checkpoint topology.

**Verification**:
- script compiles: `scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py`
- validation ran on GPU 4
- baseline: `64497` triangles, `193491` vertices
- candidate: `64498` triangles, `193494` vertices
- render deltas: PSNR `0.0`, SSIM `0.0`, LPIPS `0.0`
- geometry deltas: AbsRel `0.0`, Depth MAE `0.0`, normal mean angle `-0.00004203616886400141`

**Decision**: `PASS`. The gate now turns checkpoint-copy edits into auditable render/geometry accept-reject decisions. This validates the infrastructure path only; R14 still needs real edit selection, teacher recovery, and medium-budget W&B-logged public-scene comparison before paper-facing claims.

**Linked artefacts**:
- `scripts/car_model/meshsplatopt_render_backed_checkpoint_gate.py`
- `docs/car_model/meshsplatopt_stageR14_6_render_backed_checkpoint_gate_report.md`
- `outputs/carnet/meshsplatopt/stageR14_6_render_backed_checkpoint_gate/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.5 real checkpoint fill dry-run — PASS

**Outcome**: Materialized a tiny constructive `FILL_PATCH` on a real 200-iteration checkpoint copy, rendered it, ran independent metrics, and ran comparable COLMAP geometry evaluation.

**Verification**:
- fill checkpoint schema valid
- triangles: `64497 -> 64498`
- vertices: `193491 -> 193494`
- render completed on GPU 4
- metrics completed on GPU 4: PSNR `10.949986457824707`, SSIM `0.2898596525192261`, LPIPS `0.6441746354103088`
- comparable geometry completed on GPU 4: AbsRel `0.32417137460470213`, Depth MAE `3.6485552222775537`, normal mean angle `51.68793149935674`

**Decision**: `PASS`. Constructive checkpoint materialization is renderable. R14 still requires real edit selection, counterfactual acceptance, and teacher recovery before medium-scene claims.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_5_real_checkpoint_fill_dryrun_report.md`
- `outputs/carnet/meshsplatopt/stageR14_5_real_checkpoint_fill_dryrun/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.4 constructive checkpoint fill — PASS

**Outcome**: Added `FILL_PATCH` support to the checkpoint adapter. New vertices are initialized from nearest existing vertex radiance/weight attributes, and new per-face stats are initialized conservatively to zero.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py`
- smoke status: `PASS`
- fill appends vertices/faces and keeps checkpoint schema valid

**Decision**: `PASS`. R8 fill proposals can now be materialized in checkpoint copies. Teacher recovery and render-backed gates remain required before medium-scene claims.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_4_constructive_checkpoint_fill_design.md`
- `docs/car_model/meshsplatopt_stageR14_4_constructive_checkpoint_fill_report.md`
- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.3 render eval dry-run — PASS

**Outcome**: Ran `render.py`, `metrics.py`, and comparable `evaluate_geometry_colmap.py` on the R14.2 real checkpoint dry-run copy. The edited checkpoint loads and evaluates through the normal independent paths.

**Verification**:
- render command completed on GPU 4
- metrics command completed on GPU 4
- geometry command completed on GPU 4 with `--max_points_per_view 500`
- dry-run delete-one render metrics: PSNR `10.949986457824707`, SSIM `0.28985968232154846`, LPIPS `0.6441748142242432`
- comparable sparse geometry: AbsRel `0.32417137460470213`, Depth MAE `3.6485552222775537`, normal mean angle `51.68804758349445`

**Decision**: `PASS`. Adapter outputs are renderable and independently evaluable. This remains a path-validation result, not a medium public-scene MeshSplatOpt repair pilot.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_3_render_eval_dryrun_report.md`
- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/results.json`
- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/model/geometry_eval_colmap/iter_200_max500.json`

---

## 2026-05-02 — MeshSplatOpt Stage R14.2 real checkpoint dry-run — PASS

**Outcome**: Applied a low-risk `DELETE_TRIANGLES` dry-run edit to a real `parking_phone_tiny` 200-iteration checkpoint copy and created a normal model directory layout for future render/metrics evaluation.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/meshsplatopt_real_checkpoint_dryrun.py`
- input schema valid: `true`
- output schema valid: `true`
- triangles: `64497 -> 64496`
- planned eval commands written for `render.py`, `metrics.py`, and `evaluate_geometry_colmap.py`

**Decision**: `PASS`. Real checkpoint-copy path works for delete/snap style edits. It is still a path-validation result, not a MeshSplatOpt method-quality result.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_2_real_checkpoint_dryrun_design.md`
- `docs/car_model/meshsplatopt_stageR14_2_real_checkpoint_dryrun_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR14_2_real_checkpoint_dryrun_smoke.md`
- `outputs/carnet/meshsplatopt/stageR14_2_real_checkpoint_dryrun/`

---

## 2026-05-02 — MeshSplatOpt Stage R14.1 checkpoint adapter — PASS

**Outcome**: Implemented a conservative Mesh Splatting checkpoint adapter for MeshSplatOpt edits. It can apply `DELETE_TRIANGLES` and `SNAP_VERTICES` to checkpoint copies while preserving schema consistency, and it rejects fill/split/collapse/merge edits that require radiance/optimizer attribute initialization.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR14_1_checkpoint_adapter.py`
- smoke status: `PASS`
- delete synchronizes per-face fields; snap updates vertex positions; fill is explicitly deferred

**Decision**: `PASS`. R14 is partially unblocked for real checkpoint-copy delete/snap experiments. Certified public-scene fill still requires radiance initialization and render/recovery integration.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_1_checkpoint_adapter_design.md`
- `docs/car_model/meshsplatopt_stageR14_1_checkpoint_adapter_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR14_1_checkpoint_adapter_smoke.md`
- `outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R14 medium scene pilot — STOP_BEFORE_GPU

**Outcome**: Wrote the R14 medium public-scene pilot design and stop report. No GPU training was launched because the current MeshSplatOpt implementation is synthetic/generic-mesh only and lacks real checkpoint edit application, render-backed counterfactual validation, and real teacher recovery.

**Verification**:
- GPU availability checked; GPU 4 was the relatively light option at the check
- Stage35 public-scene artifacts exist locally and remain baselines
- no R14 training command was run

**Decision**: `STOP_BEFORE_GPU`. Do not proceed to R15. Required next work is a real Mesh Splatting checkpoint adapter, render-backed edit gate, and at least one real tiny recovery run with W&B before medium public-scene claims.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR14_medium_scene_pilot_design.md`
- `docs/car_model/meshsplatopt_stageR14_medium_scene_pilot_report.md`

---

## 2026-05-02 — MeshSplatOpt Stage R13 synthetic repair benchmark — PASS

**Outcome**: Implemented a controlled synthetic repair benchmark and collector. The benchmark compares no repair, delete-only PRISM-style cleanup, and full MeshSplatOpt repair across synthetic damage categories.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- benchmark commands completed with project Python environment
- benchmark status: `PASS`
- full MeshSplatOpt improves five categories over delete-only: `giant_ground_void`, `ground_wall_misalignment`, `local_dent`, `noisy_rough_patch`, `small_hole`
- prior-only unknown void rejected

**Decision**: `PASS`. Synthetic gate is satisfied. R14 medium public-scene pilot is now the next stage before any full-budget GPU sweep.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR13_synthetic_repair_benchmark_design.md`
- `docs/car_model/meshsplatopt_stageR13_synthetic_repair_benchmark_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR13_synthetic_repair_benchmark_smoke.md`
- `outputs/carnet/meshsplatopt/stageR13_synthetic_repair_benchmark/`

---

## 2026-05-02 — MeshSplatOpt Stage R12 edit portfolio state machine — PASS

**Outcome**: Implemented portfolio scoring and a repair state machine with auditable trace, accepted/rejected edits, and final audit outputs. The synthetic smoke accepts cleanup, snap, fill, and appearance-reset classes while rejecting a prior-only fill in normal mode.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR12_portfolio.py`
- smoke status: `PASS`
- accepted edit classes: `DELETE_TRIANGLES`, `SNAP_VERTICES`, `FILL_PATCH`, `APPEARANCE_RESET`

**Decision**: `PASS`. The state machine executes at least three edit classes on synthetic data and produces an auditable trace.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR12_edit_portfolio_design.md`
- `docs/car_model/meshsplatopt_stageR12_edit_portfolio_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR12_portfolio_smoke.md`
- `outputs/carnet/meshsplatopt/stageR12_portfolio_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R11 teacher recovery contract — SOFT PASS

**Outcome**: Implemented teacher recovery cache and report contract. The smoke writes RGB/depth/normal/alpha/visibility/edit-region placeholder cache files and clearly marks real recovery metrics unavailable when no renderable model path exists.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR11_teacher_recovery.py`
- smoke status: `SOFT PASS`
- real recovery run: `false`

**Decision**: `SOFT PASS`. The contract works, but public-scene claims still require a real renderable recovery run.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR11_teacher_recovery_design.md`
- `docs/car_model/meshsplatopt_stageR11_teacher_recovery_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR11_teacher_recovery_smoke.md`
- `outputs/carnet/meshsplatopt/stageR11_teacher_recovery_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R10 generalized counterfactual edit gate — PASS

**Outcome**: Implemented generalized edit validation for arbitrary reversible edits. The gate snapshots state, applies an edit, checks topology and risk/certificate metadata, accepts or rejects, and rolls back rejected edits exactly. Render/sparse/changed-pixel fields are present but marked unavailable when no render path exists.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR10_counterfactual_edits.py`
- smoke status: `PASS`
- good fill accepted
- bad floater insertion, snap through free space, and delete-supported-surface edits rejected with rollback

**Decision**: `PASS`. At least one non-delete edit is accepted and harmful non-delete edits are rejected in smoke.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR10_generalized_counterfactual_design.md`
- `docs/car_model/meshsplatopt_stageR10_counterfactual_edits_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR10_counterfactual_edits_smoke.md`
- `outputs/carnet/meshsplatopt/stageR10_counterfactual_edits_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R9 object-prior repair proposals — PASS

**Outcome**: Implemented a bounded object-prior proposal generator for vehicle regions. Confident priors can emit protect, snap, and discontinuity-fill candidates; uncertain priors are restricted to protect metadata. Every proposal records that scene counterfactual validation is required.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR9_object_prior_repair.py`
- smoke status: `PASS`
- confident synthetic vehicle package includes protect and fill
- uncertain prior emits no fill
- all proposals record `prior_proposes_evidence_disposes=true`

**Decision**: `PASS`. Object-prior proposals are bounded and cannot bypass scene gates.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR9_object_prior_repair_design.md`
- `docs/car_model/meshsplatopt_stageR9_object_prior_repair_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR9_object_prior_repair_smoke.md`
- `outputs/carnet/meshsplatopt/stageR9_object_prior_repair_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R8 giant void fill proposals — PASS

**Outcome**: Implemented boundary-loop fill, ground-plane void fill, fill certificates, prior-only diagnostic fill labeling, normal-mode unknown-void rejection, and rollback-compatible `FILL_PATCH` proposals.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR8_giant_void_fill.py`
- smoke status: `PASS`
- small-hole boundary count reduced from `20` to `4`
- giant ground void patch valid
- unknown void rejected in normal mode
- diagnostic prior-only fill emitted with `prior_only_flag=true`

**Decision**: `PASS`. Giant ground void synthetic repair works and unknown voids are not silently filled in normal mode.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR8_giant_void_fill_design.md`
- `docs/car_model/meshsplatopt_stageR8_giant_void_fill_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR8_giant_void_fill_smoke.md`
- `outputs/carnet/meshsplatopt/stageR8_giant_void_fill_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R7 snap/deform proposals — PASS

**Outcome**: Implemented safe snap/deform proposal generation using plane-fit targets, capped displacements, step sizes `0.1/0.25/0.5`, unsupported-floater rejection, and R5-compatible `SNAP_VERTICES` edits.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py`
- smoke status: `PASS`
- dent error reduced from `0.03072` to `0.019831720797113993`
- misalignment error reduced from `0.019200000000000002` to `0.009984000000000002`
- unsupported floater rejected and rollback exact

**Decision**: `PASS`. R8 can add fill/patch proposals using the same reversible edit contract.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR7_snap_deform_design.md`
- `docs/car_model/meshsplatopt_stageR7_snap_deform_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR7_snap_smoke.md`
- `outputs/carnet/meshsplatopt/stageR7_snap_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R6 topology baselines — PASS

**Outcome**: Implemented topology-reduction baselines for delete, random delete, low-visibility delete, boundary-protected delete, greedy QEM-style edge collapse, planar face merge, and an explicit external-simplification JSON contract placeholder.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR6_topology_baselines.py`
- smoke status: `PASS`
- delete and boundary-protected delete hit target counts; collapse/merge-style baselines produce valid meshes

**Decision**: `PASS`. Future repair claims now have stronger topology baselines than random or weak deletion.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR6_topology_baselines_design.md`
- `docs/car_model/meshsplatopt_stageR6_topology_baselines_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR6_topology_baselines_smoke.md`
- `outputs/carnet/meshsplatopt/stageR6_topology_baselines_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R5 reversible edit abstraction — PASS

**Outcome**: Implemented generic numpy mesh state, edit records, snapshot/rollback, edit application, topology delta summary, and mesh integrity checks. All required edit types are reversible through snapshots; protect and appearance reset are metadata-only operations in R5.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR5_reversible_edits.py`
- smoke status: `PASS`
- integrity checker catches invalid indices and degenerate faces

**Decision**: `PASS`. Delete, snap, fill, collapse, split, protect, and appearance reset round-trip through exact rollback. R6 can build topology baselines on this edit contract.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR5_reversible_edits_design.md`
- `docs/car_model/meshsplatopt_stageR5_reversible_edits_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR5_reversible_edits_smoke.md`
- `outputs/carnet/meshsplatopt/stageR5_reversible_edits_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R4 defect mining — PASS

**Outcome**: Implemented defect records and CSEF-driven defect mining. The miner emits auditable JSON/CSV/Markdown artifacts and distinguishes boundary-supported giant ground voids from unknown/unobserved voids that cannot be repaired in normal mode.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR4_defect_mining.py`
- smoke status: `PASS`
- emitted defect types: `GIANT_GROUND_VOID`, `UNKNOWN_UNOBSERVED_VOID`

**Decision**: `PASS`. Huge ground holes are explicitly detected and distinguished from unknown/unobserved voids. R5 can implement reversible edit records on top of this defect contract.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR4_defect_mining_design.md`
- `docs/car_model/meshsplatopt_stageR4_defect_mining_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR4_defect_mining_smoke.md`
- `outputs/carnet/meshsplatopt/stageR4_defect_mining_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R3 CSEF data model and diagnostics — PASS

**Outcome**: Implemented the first CSEF data contract and diagnostic builder under `ss3dm_prior/meshsplatopt/`, plus a CLI and synthetic smoke. The builder samples faces, computes boundary/component/area diagnostics, writes CSEF NPZ/JSON/CSV/Markdown artifacts, and does not modify geometry.

**Verification**:
- compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- smoke command: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/smoke_test_meshsplatopt_stageR3_csef.py`
- smoke status: `PASS`
- CLI check wrote `outputs/carnet/meshsplatopt/stageR3_csef_cli_check/`

**Smoke metrics**:
- normal debt: `0.18554923879355764`
- hole boundary debt: `0.34568891727769624`
- floater uncertainty: `0.9`
- floater positive surface evidence: `0.10520833333333335`

**Decision**: `PASS`. Synthetic CSEF separates normal surface, floater, and hole/debt region. R4 can mine actionable defect regions from these diagnostics.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR3_csef_design.md`
- `docs/car_model/meshsplatopt_stageR3_csef_implementation_report.md`
- `docs/car_model/meshsplatopt_stageR3_csef_smoke.md`
- `outputs/carnet/meshsplatopt/stageR3_csef_smoke/`

---

## 2026-05-02 — MeshSplatOpt Stage R2 related-work and baseline matrix — PASS

**Outcome**: Wrote the related-work/novelty-threat matrix and baseline plan. The plan explicitly names threats from Mesh Splatting, mesh-aware splatting, 3DGS compression/pruning, QEM, classical hole filling, COLMAP/MVS, plane priors, object priors, and depth/normal priors.

**Decision**: `PASS`. The strongest novelty is constrained to unified CSEF, reversible bidirectional edit calculus, and certified giant-hole repair. Training-time pruning, mesh simplification, geometry priors, and counterfactual validation alone are not treated as novel.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR2_related_work_matrix.md`
- `docs/car_model/meshsplatopt_stageR2_baseline_plan.md`

---

## 2026-05-02 — MeshSplatOpt Stage R1 repair RFC — PASS

**Outcome**: Wrote the MeshSplatOpt repair RFC. The method is locked as `MeshSplatOpt: Evidence-Certified Bidirectional Mesh Surgery for Mesh Splatting`, centered on the Counterfactual Surface Evidence Field and reversible edit calculus across protect, delete, collapse, snap, split, fill, and appearance recovery.

**Decision**: `PASS`. The RFC explicitly separates pruning, constructive repair, and hallucination risk. Stage35 PRISM is a required retained-pruning baseline, while giant-hole repair and prior-supported fills require evidence certificates and uncertainty labels.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR1_repair_RFC.md`

---

## 2026-05-02 — MeshSplatOpt Stage R0 pivot audit — PASS

**Outcome**: Created branch `neurips-meshsplatopt-repair`, read the required PRISM retrospective, handoff, reviewer-risk, RFC, roadmap, topology-retention, retained-refresh, metric-reconciliation, and remaining-work documents, and locked the pivot from delete-centric PRISM to evidence-certified bidirectional mesh surgery.

**Verification**:
- repository compile gate: `python -m compileall scripts/car_model ss3dm_prior utils -q` passed
- branch: `neurips-meshsplatopt-repair`
- commit at audit time: `6344a0c`
- dirty files before R0 docs: untracked `docs/NeurIPSRepairPrompts.md` and untracked submodule directories only

**Decision**: `PROCEED_TO_R1`. Stage35 remains a named retained-PRISM baseline, not the final method. MeshSplatOpt must support reversible delete, collapse, snap, split, fill, protect, and appearance-recovery operations under CSEF and counterfactual gates.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR0_pivot_audit.md`

---

## 2026-05-02 — Deep retrospective on PRISM/MeshPrior — FAIL as NeurIPS-strength method result

**Outcome**: Added a frank retrospective after reviewing the final M35-M43 evidence. The conclusion is that the engineering infrastructure is strong, but the empirical method result and innovation level are not sufficient for a NeurIPS-level claim.

**Key judgment**:
- `bonsai` M35 vs Stage33 improves by only `+0.067 dB` PSNR, `+0.001084` SSIM, `-0.000644` LPIPS, and `-512` triangles.
- `courtyard` M35 improves topology/PSNR/SSIM but regresses LPIPS.
- Public full-budget Stage35 evidence is missing.
- The method accumulated too many modules relative to the measured payoff.

**Decision**: Do not continue incremental PRISM gate/schedule tuning by default. The next defensible step is a short, decisive Pareto feasibility experiment for topology compression, or a pivot back to a more clearly novel task/dataset formulation.

**Artifact**:
- `docs/car_model/reports/meshprior_prism_deep_retrospective.md`

---

## 2026-05-02 — Stage43 final handoff and experiment trigger — PASS

**Outcome**: Added a concise final handoff report describing the method, strongest evidence, main artifacts, forbidden claims, and exact trigger conditions for any future full-budget public Stage35 run.

**Artifact**:
- `docs/car_model/reports/meshprior_prism_final_handoff.md`

**Decision**: Stage43 `PASS`. Full-budget public Stage35 training remains `NO_GO_FOR_NOW`. The trigger for more GPU work is now explicit: a named missing row that would change the core claim or reviewer risk, on a geometry-observable scene, with W&B and GPU check.

---

## 2026-05-02 — Stage42 figure index, bibliography draft, reviewer-risk checklist — PASS

**Outcome**: Added handoff-oriented paper assets: final figure index, draft BibTeX file, and reviewer-risk checklist covering claims, metrics, datasets, and implementation risks.

**Artifacts**:
- `docs/car_model/reports/meshprior_prism_figure_index.md`
- `docs/car_model/reports/meshprior_prism_bibliography_draft.bib`
- `docs/car_model/reports/meshprior_prism_reviewer_risk_checklist.md`

**Decision**: Stage42 `PASS`. Paper assets now preserve source traceability for figures and citations. Full-budget public Stage35 training remains `NO_GO_FOR_NOW` until a concrete table gap appears.

---

## 2026-05-02 — Stage41 citation-backed related work and claim tightening — PASS

**Outcome**: Replaced the related-work placeholder in the PRISM manuscript draft with citation-backed draft text covering NeRF, Instant-NGP, 3D Gaussian Splatting, Gaussian-to-mesh / mesh-aligned splatting, COLMAP SfM/MVS, and the distinction between generic simplification and PRISM's rollback-audited topology control.

**Artifacts**:
- `docs/car_model/reports/meshprior_prism_manuscript_draft.md`
- `docs/car_model/reports/meshprior_prism_related_work_sources.md`

**Decision**: Stage41 `PASS`. Claims remain evidence-aligned: PRISM is framed as auditable topology control under scene-evidence gates, not universal quality dominance and not a radar-only reconstruction method. No training was run because no specific paper-table gap emerged.

---

## 2026-05-02 — Stage40 manuscript integration and evidence-gap review — PASS

**Outcome**: Expanded the PRISM skeleton into a fuller manuscript draft with introduction, related-work placeholders, method, experimental setup, results, diagnostics, limitations, conclusion, and final evidence-gap review.

**Artifact**:
- `docs/car_model/reports/meshprior_prism_manuscript_draft.md`

**Decision**: Stage40 `PASS`. The final evidence-gap decision remains `NO_GO_FOR_NOW` for full-budget public-scene training. The manuscript draft is coherent enough for human editing and citation work; the next work should be citation-backed related work, figure formatting, and reviewer-facing claim tightening.

---

## 2026-05-02 — Stage39 manuscript skeleton and reproducibility appendix — PASS

**Outcome**: Drafted a manuscript skeleton and reproducibility appendix from the M35-M38 evidence chain. The draft keeps claims aligned with the evidence: PRISM is framed as an auditable topology-control layer, not as a universal image-quality optimizer or radar-only reconstruction method.

**Artifacts**:
- `docs/car_model/reports/meshprior_prism_manuscript_skeleton.md`
- `docs/car_model/reports/meshprior_prism_reproducibility_appendix.md`

**Decision**: Stage39 `PASS`. No full-budget public-scene run is justified yet because the current missing work is manuscript integration, not a specific absent row. Any future full-budget run must identify the exact table gap first and use W&B plus a GPU availability check.

---

## 2026-05-02 — Stage38 final paper assets — PASS

**Outcome**: Added a paper-asset builder that turns the M36 metric table and M37 visual/failure package into selected paper rows, figure captions, limitations, and a full-budget training decision.

**Generated artifacts**:
- `outputs/carnet/meshprior/stage38_paper_assets/paper_assets_package.json`
- `outputs/carnet/meshprior/stage38_paper_assets/final_paper_table.md`
- `outputs/carnet/meshprior/stage38_paper_assets/figure_captions.md`
- `outputs/carnet/meshprior/stage38_paper_assets/limitations.md`

**Decision**: Stage38 `PASS`. Full-budget public-scene Stage35 training is `NO_GO_FOR_NOW`: the next blocker is paper table/figure clarity, not the absence of a short-run signal. If a full-budget public run is revisited, W&B and GPU availability checks remain mandatory.

**Linked artefacts**:
- `docs/car_model/meshprior_stage38_paper_assets_report.md`
- `scripts/car_model/meshprior_make_paper_assets.py`

---

## 2026-05-02 — Stage37 visual/failure package — PASS

**Outcome**: Added a packaging script for visual panels, failure cases, and paper-safe claim wording. It generated render-vs-GT panels for parking M24.2, `bonsai` M35, and `courtyard` M35, plus a six-row failure table tied to concrete local artifacts.

**Generated artifacts**:
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_failure_package.json`
- `outputs/carnet/meshprior/stage37_visual_failure_package/failure_case_table.md`
- `outputs/carnet/meshprior/stage37_visual_failure_package/paper_claim_wording.md`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/parking_m24_2_retention_7000.png`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/bonsai_m35_retained_relaxed.png`
- `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/courtyard_m35_retained_relaxed.png`

**Decision**: Stage37 `PASS`. Do not start full-budget public-scene training yet. The current highest-value work is polishing the final paper figures/tables and only then deciding whether one full-budget Stage35 public-scene run is worth the GPU time.

**Linked artefacts**:
- `docs/car_model/meshprior_stage37_visual_failure_package_report.md`
- `scripts/car_model/meshprior_package_visual_failures.py`

---

## 2026-05-02 — Stage36 metric reconciliation evidence table — PASS

**Outcome**: Added a reproducible collector for paper-facing MeshPrior evidence rows. The collector reads local M24-M35 artifacts, exports JSON/CSV/Markdown tables, preserves W&B links, records topology/audit metadata, and keeps training-time metrics separate from independent `render.py + metrics.py` metrics.

**Generated artifacts**:
- `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_report.json`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.csv`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.md`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/visual_panels/bonsai_m35_retained_relaxed.png`
- `outputs/carnet/meshprior/stage36_metric_reconciliation/visual_panels/courtyard_m35_retained_relaxed.png`

**Key result**: Stage35 is the current best `bonsai` retained-edit row: `633275` triangles, PSNR `12.267367`, SSIM `0.277617`, LPIPS `0.611939`, with `1` active relaxed commit and `4` validation-rolled-back relaxed commits explicitly recorded. On `courtyard`, Stage35 has the best selected-row PSNR/SSIM and lowest topology, but LPIPS is worse than Stage32/33, so paper wording must report a scene-dependent perceptual tradeoff.

**Decision**: Stage36 `PASS`. The evidence table is reproducible from local artifacts and metric paths are no longer mixed. The next step should be visual/failure-case packaging and, if compute allows, full-budget public-scene validation of the Stage35 retained-refresh row.

**Linked artefacts**:
- `docs/car_model/meshprior_stage36_metric_reconciliation_report.md`
- `scripts/car_model/meshprior_collect_metric_reconciliation.py`

---

## 2026-05-02 — Stage35 retained relaxed refresh control — PASS

**Outcome**: Added conservative retained-edit control for Stage34 post-commit relaxed candidate refresh. The controller now caps active retained relaxed commits, records validation rollbacks explicitly, writes a final retained-topology audit, and can require a strict counterfactual proxy gate before relaxed commits. Defaults remain unchanged and all new behavior is opt-in.

**W&B**:
- `bonsai` retained relaxed retry: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rszvl7gn`
- ETH3D `courtyard` retained relaxed check: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/u2s15ok0`

**Key metrics**:
- `bonsai`: final `633275` triangles, `1` active retained relaxed commit, independent PSNR `12.2673674`, SSIM `0.2776170`, LPIPS `0.6119390`.
- Stage33 `bonsai` reference: `633787` triangles, PSNR `12.1999207`, SSIM `0.2765326`, LPIPS `0.6125830`.
- ETH3D `courtyard`: final `101913` triangles, `1` active retained relaxed commit, independent PSNR `15.3831606`, SSIM `0.5080911`, LPIPS `0.5846940`.

**Decision**: Stage35 is a real `PASS`: `bonsai` keeps the additional relaxed edit in the final checkpoint and improves all independent metrics versus Stage33 while reducing topology. `courtyard` confirms the retained relaxed cap and final audit transfer to a second public scene. The next step is to turn this into a paper-facing method row: metric-path reconciliation, unified tables, visuals, and full-budget validation.

**Linked artefacts**:
- `docs/car_model/meshprior_stage35_retained_refresh_report.md`

---

## 2026-05-02 — Stage34 post-commit candidate refresh — SOFT PASS / diagnostic PASS

**Outcome**: Added opt-in post-commit candidate refresh and measured the root cause of the post-commit no-candidate failure. After a candidate commit, topology sync marks all surviving triangles as recent; `recent_t` then protects every triangle and also zeroes the normal prune score through `risk_t`. The new relaxed score removes only recent risk while keeping other risk and keep signals, and keeps the counterfactual gate mandatory.

**W&B**:
- parking refresh smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rt3cxxhh`
- parking recent0 smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kke60qhc`
- bonsai root-cause v1: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/szkqpowq`
- bonsai root-cause v2: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/npagb743`
- bonsai relaxed-score v3: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/lt1v4652`
- bonsai second-edit-only diagnostic v4: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/zhy368pr`

**Best retained-topology result**:
- run: `mipnerf360_bonsai_refresh_v3_relaxed_score_diverse_calib_measured_rank_cap512_adaptive_ratio0p02_geom1400_2000iter`
- decisions: normal commit at `1501`, then relaxed commits at `1592`, `1683`, `1774`, and `1956`
- final topology: `631739` triangles versus Stage33 `633787`
- independent render: PSNR `12.2019978`, SSIM `0.2757282`, LPIPS `0.6129612`
- Stage33 reference: PSNR `12.1999207`, SSIM `0.2765326`, LPIPS `0.6125830`, topology `633787`

**Decision**: M34 is a mechanism and diagnosis success, not a default schedule. It lowers retained topology and slightly improves PSNR, but SSIM/LPIPS regress slightly. The next step is M35 conservative retained-edit control: allow one retained relaxed edit, log whether it survives recovery/final checkpoint, and gate it on stricter held-out or independent-metric proxy behavior before running `courtyard`.

**Linked artefacts**:
- `docs/car_model/meshprior_stage34_post_commit_refresh_report.md`

---

## 2026-05-02 — Stage33 PRISM calibration diversity diagnostics — SOFT PASS / diagnostic PASS

**Outcome**: Added opt-in view-diverse PRISM calibration diagnostics. The counterfactual gate can now seed calibration with evenly spaced held-out/test views and train views before adding hard train views, writes `prism_debug/calibration_views.json`, and records per-view counterfactual deltas. This improves `bonsai` over Stage29 cap512 at equal topology, but does not beat Stage32 on `courtyard`.

**W&B**:
- parking diverse-calibration smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ms95810g`
- `bonsai` diverse calibration: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/kg5htc8u`
- `courtyard` diverse calibration: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w9c0b65f`

**Key metrics**:
- parking smoke: diverse calibration rejected all candidate edits; final topology stayed `64497`, with per-view deltas exposing local regressions.
- `bonsai`: iter `1501` committed `634299 -> 633787`; independent PSNR `12.1999`, SSIM `0.2765`, LPIPS `0.6126`.
- `courtyard`: iter `1501` committed `102919 -> 102407`; independent PSNR `15.0737`, SSIM `0.4840`, LPIPS `0.5790`.

**Decision**: Stage33 is useful safety and calibration infrastructure, not the new universal default. It should be used for diagnostic view coverage and scenes where local hard-view calibration was misleading. Stage29 cap512 remains the conservative baseline, while Stage32 remains the better `courtyard` measured-rank row.

**Linked artefacts**:
- `docs/car_model/meshprior_stage33_calibration_diversity_report.md`
- `outputs/carnet/meshprior/stage33_calibration_diversity/`

---

## 2026-05-02 — Stage32 PRISM measured candidate-impact ranking — SOFT PASS / diagnostic PASS

**Outcome**: Added opt-in measured candidate-impact ranking. The controller can now draw a larger candidate pool, split it into deterministic groups, evaluate each group with the existing counterfactual calibration path, and select the final cap-limited candidate set from measured impact. The mechanism is stable and improves `courtyard`, but it does not beat Stage29 cap512 on `bonsai`.

**W&B**:
- parking smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xg4fsvd8`
- `bonsai` measured rank: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/56l3tz23`
- `courtyard` measured rank: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/fb7jfcaj`
- `bonsai` measured+quality diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xooe27um`

**Key metrics**:
- parking smoke: iter `91` committed `64497 -> 63985` after `3/3` measured groups accepted.
- `bonsai` measured: iter `1501` committed `634299 -> 633787`; independent PSNR `12.1742`, SSIM `0.2758`, LPIPS `0.6137`.
- `courtyard` measured: iter `1501` committed `102916 -> 102404`; independent PSNR `15.1390`, SSIM `0.4850`, LPIPS `0.5792`.
- `bonsai` measured+quality diagnostic: independent PSNR `12.1708`, SSIM `0.2760`, LPIPS `0.6133`.

**Decision**: Stage32 is useful infrastructure but not a default. It gives the best `courtyard` PSNR/SSIM so far, yet fails the M32 gate on `bonsai`. The next step should improve calibration-view representativeness and candidate diversity instead of further hand-tuning local score weights.

**Linked artefacts**:
- `docs/car_model/meshprior_stage32_measured_candidate_rank_report.md`
- `outputs/carnet/meshprior/stage32_measured_candidate_rank/`

---

## 2026-05-02 — Stage31 PRISM candidate-quality ranking — SOFT PASS / diagnostic PASS

**Outcome**: Added opt-in candidate-quality ranking for PRISM candidate pruning. The selector can now rank cap-limited candidates by a blended score that rewards raw prune pressure while penalizing render, geometry, orientation, utility, and uncertainty risk. The mechanism is stable and logged, but it is not promoted as the default because the public-scene result is mixed.

**W&B**:
- parking smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ucqyou26`
- `bonsai` quality-rank cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/22r3et7s`
- `courtyard` quality-rank cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xt4a2cn0`

**Key metrics**:
- parking smoke: iter `91` committed `64497 -> 63985`.
- `bonsai`: iter `1501` committed `634299 -> 633787`; independent PSNR `12.1891`, SSIM `0.2756`, LPIPS `0.6136`.
- `courtyard`: iter `1501` committed `102916 -> 102404`; independent PSNR `15.0732`, SSIM `0.4837`, LPIPS `0.5788`.

**Decision**: Stage31 is useful as diagnostic infrastructure but not a default schedule. It improves `courtyard` versus M29 cap512, but `bonsai` only gains tiny PSNR while losing SSIM/LPIPS. The next step should use measured calibration-view impact for ranking, not only hand-weighted proxy tensors.

**Linked artefacts**:
- `docs/car_model/meshprior_stage31_candidate_quality_report.md`
- `outputs/carnet/meshprior/stage31_candidate_quality/`

---

## 2026-05-02 — Stage30 PRISM microbatch candidate gate — SOFT PASS / diagnostic PASS

**Outcome**: Added opt-in microbatch counterfactual gating for candidate pruning. Large candidate sets can now be split into smaller cumulative batches, with only accepted batches committed. The mechanism works, but `1024 x 256` is not better than the Stage29 cap512 Pareto row on independent metrics.

**W&B**:
- parking smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dioe1cz1`
- `bonsai` microbatch1024x256: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mfvhexjb`
- `courtyard` microbatch1024x256: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ha9qi1ih`

**Key metrics**:
- parking smoke: iter `91` accepted `3/3` microbatches and committed `64497 -> 63853`.
- `bonsai`: iter `1501` accepted `3/4` microbatches, committed `634299 -> 633531`; independent PSNR `12.1423`, SSIM `0.2770`, LPIPS `0.6136`.
- `courtyard`: iter `1501` accepted `4/4` microbatches, committed `102919 -> 101895`; independent PSNR `15.0635`, SSIM `0.4828`, LPIPS `0.5802`.

**Decision**: Stage30 is a useful diagnostic mechanism, not the next default. Keep cap512 as the current conservative topology-quality row. M31 should improve candidate quality/ranking, because simply accepting more microbatches trades independent PSNR/LPIPS away on `bonsai`.

**Linked artefacts**:
- `docs/car_model/meshprior_stage30_microbatch_gate_report.md`
- `outputs/carnet/meshprior/stage30_microbatch_gate/`

---

## 2026-05-02 — Stage29 bonsai candidate-cap sweep — PASS diagnostic

**Outcome**: Completed a Mip-NeRF 360 `bonsai` cap sweep with online W&B and independent metrics. Cap `256` and `512` commit; cap `1024` rolls back all attempts. Cap `512` is the best current topology-quality Pareto row.

**W&B**:
- cap256: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mzglj2qw`
- cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ck157wtl`
- cap1024: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j5v0debo`

**Key metrics**:
- cap256: final `634043` triangles; independent PSNR `12.1430`, SSIM `0.2753`, LPIPS `0.6134`.
- cap512: final `633787` triangles; independent PSNR `12.1859`, SSIM `0.2764`, LPIPS `0.6129`.
- cap1024: final `1357128` triangles; independent PSNR `12.2882`, SSIM `0.2398`, LPIPS `0.6211`.

**Decision**: Stage29 cap sweep is a `PASS` diagnostic. The next useful algorithmic step is microbatch candidate gating: cap1024 likely contains useful removable triangles, but the whole batch is too risky as one counterfactual edit.

**Linked artefacts**:
- `docs/car_model/meshprior_stage29_candidate_cap_sweep_report.md`
- `outputs/carnet/meshprior/stage29_candidate_selection/`

---

## 2026-05-02 — Stage29 candidate cap medium ablation — SOFT PASS

**Outcome**: Ran the M29 cap512 public-scene medium ablation with online W&B and independent render metrics. Candidate capping makes `bonsai` accept a PRISM edit for the first time in the M27-M29 public-scene sequence, but quality/topology tradeoffs remain.

**W&B**:
- `bonsai` cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ck157wtl`
- `courtyard` cap512: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1ey4qzbd`

**Key metrics**:
- `bonsai`: final `633787` triangles, `1` commit, independent PSNR `12.1859`, SSIM `0.2764`, LPIPS `0.6129`.
- `courtyard`: final `102916` triangles, `1` commit, independent PSNR `15.0344`, SSIM `0.4812`, LPIPS `0.5804`.

**Decision**: Stage29 medium ablation is a `SOFT PASS`. Cap512 is a strong Pareto diagnostic, not a final default. Next work should sweep cap sizes and diagnose why `courtyard` immediate topology `102404` returns to final `102916`.

**Linked artefacts**:
- `docs/car_model/meshprior_stage29_candidate_cap_medium_report.md`
- `docs/car_model/meshprior_stage29_candidate_cap_report.md`
- `outputs/carnet/meshprior/stage29_candidate_selection/`

---

## 2026-05-02 — Stage29 PRISM candidate cap smoke — PASS

**Outcome**: Added an opt-in cap for PRISM candidate prune count per round. This directly targets the M28 `bonsai` failure where even a `0.005` ratio selected `3171` triangles. Defaults are unchanged because the cap defaults to disabled.

**W&B**:
- parking cap smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/rgvzhx6k`

**Verification**:
- output: `outputs/carnet/meshprior/stage29_candidate_selection/parking_candidate_cap_smoke_256_140iter/model`
- cap sequence: ratio targets `2579`, `1289`, `644`; cap target `256`; selected count `256` on all candidate attempts.
- first two attempts rolled back under a strict gate; third attempt committed `64497 -> 64241` triangles.

**Decision**: Stage29 implementation smoke `PASS`. The next step is the medium `bonsai` / `courtyard` public-scene ablation with candidate cap enabled.

**Linked artefacts**:
- `docs/car_model/meshprior_stage29_candidate_cap_report.md`
- `outputs/carnet/meshprior/stage29_candidate_selection/parking_candidate_cap_smoke_256_140iter/model`

---

## 2026-05-02 — Stage28 adaptive PRISM schedule medium ablation — SOFT PASS

**Outcome**: Completed the M28 medium public-scene ablation with online W&B on Mip-NeRF 360 `bonsai` and ETH3D `courtyard`. Adaptive rollback-driven candidate-ratio decay is working and auditable, but it does not solve the `bonsai` topology failure. It preserves the strong ETH3D result from M27.

**W&B**:
- `bonsai` adaptive `0.02 -> 0.01 -> 0.005`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/38p6bgw4`
- `courtyard` adaptive `0.02`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/piadupsm`

**Key metrics**:
- `bonsai`: final `1357119` triangles, `0` commits, `8` rejected candidate gates; independent PSNR `12.3054`, SSIM `0.2410`, LPIPS `0.6196`.
- `courtyard`: final `100858` triangles, `1` commit, `41` no-candidate retries; independent PSNR `15.0919`, SSIM `0.4844`, LPIPS `0.5778`.

**Decision**: Stage28 medium ablation is a `SOFT PASS`. The next technical bottleneck is candidate selection granularity: on `bonsai`, even the decayed `0.005` ratio still selects `3171` triangles and is rejected. M29 should cap or microbatch candidate sets and gate the smaller batches.

**Linked artefacts**:
- `docs/car_model/meshprior_stage28_adaptive_schedule_medium_report.md`
- `docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md`
- `outputs/carnet/meshprior/stage28_adaptive_schedule/`

---

## 2026-05-02 — Stage28 adaptive PRISM schedule smoke — PASS

**Outcome**: Added an opt-in adaptive candidate retry path for PRISM. When a candidate prune is rejected by the counterfactual gate, the active candidate ratio can decay and retry before the controller consumes the effective candidate round. This directly targets the M27 `bonsai` failure mode where 2% candidates rolled back while lower-pressure schedules sometimes committed.

**W&B**:
- adaptive rollback-ratio smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1kmwbu8g`

**Verification**:
- output: `outputs/carnet/meshprior/stage28_adaptive_schedule/parking_adaptive_retry_smoke_v5_140iter/model`
- candidate retry sequence: `0.04 -> 0.02 -> 0.01`
- candidate selected counts: `2579 -> 1289 -> 644`
- all candidate attempts intentionally rolled back under a strict gate; final checkpoint accounting remained consistent at `64497` triangles.

**Decision**: Stage28 implementation smoke `PASS`. The next step is a medium public-scene ablation comparing adaptive scheduling against M27 fixed `ratio0p02_geom1400` on `bonsai` and `courtyard`.

**Linked artefacts**:
- `docs/car_model/meshprior_stage28_adaptive_schedule_smoke_report.md`
- `outputs/carnet/meshprior/stage28_adaptive_schedule/parking_adaptive_retry_smoke_v5_140iter/model`

---

## 2026-05-02 — Stage27 schedule ablation — SOFT PASS

**Outcome**: Completed M27 schedule tuning after the topology-accounting fix. All valid current-branch runs used online W&B and were evaluated with independent `render.py + metrics.py`. The best schedule, `ratio0p02_geom1400`, gives a strong ETH3D `courtyard` result but does not reduce topology on Mip-NeRF 360 `bonsai`, so this is an interpretable `SOFT PASS`, not a final paper schedule.

**W&B**:
- `bonsai` ratio `0.01`, geometry acquisition until `1200`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mlftnbt5`
- `courtyard` ratio `0.01`, geometry acquisition until `1200`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qvrnsj2v`
- `bonsai` ratio `0.02`, geometry acquisition until `1400`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/27vl4jnt`
- `courtyard` ratio `0.02`, geometry acquisition until `1400`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ffp07dua`

**Key metrics**:
- `bonsai` `ratio0p02_geom1400`: final `1357128` triangles, `0` commits, `6` rollbacks, validation `3/3` observable and `2/3` pass; independent PSNR `12.3005`, SSIM `0.2408`, LPIPS `0.6194`.
- `courtyard` `ratio0p02_geom1400`: final `100858` triangles, `1` commit, `0` rollbacks, validation `4/4` observable and `3/4` pass; independent PSNR `15.0739`, SSIM `0.4857`, LPIPS `0.5794`.

**Decision**: M27 confirms accounting is fixed and shows stronger topology pressure can work on ETH3D, but the fixed schedule is not cross-scene robust. The next prompt should make PRISM scheduling adaptive instead of launching a large fixed-schedule full-budget sweep.

**Linked artefacts**:
- `docs/car_model/meshprior_stage27_schedule_ablation_report.md`
- `docs/car_model/meshprior_stage27_accounting_fix_report.md`
- `outputs/carnet/meshprior/stage27_schedule_ablation/`

---

## 2026-05-02 — Stage27.0 topology accounting fix — PASS

**Outcome**: Fixed the topology accounting mismatch found during M26. The training loop previously logged W&B `mesh/triangle_count` before the end-of-iteration standard prune/densify block, while final checkpoints and `final_cleanup_summary.json` reflected the post-mutation topology. Future runs now log post-topology counts and final-checkpoint counts explicitly.

**W&B**:
- accounting smoke: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/i6lfgt66`

**Verification**:
- smoke output: `outputs/carnet/meshprior/stage27_accounting/eth3d_courtyard_accounting_smoke_520iter/`
- local W&B summary: `mesh/triangle_count = 33487`, `mesh/final_checkpoint_triangle_count = 33487`
- final cleanup summary: `post_prune_triangle_count = 33487`
- vertex counts also agree: `100461`

**Decision**: M27.0 gate `PASS`. The next M27 work is schedule tuning for stronger direct cross-scene PRISM topology pressure.

**Linked artefacts**:
- `docs/car_model/meshprior_stage27_accounting_fix_report.md`

---

## 2026-05-02 — Stage26 cross-scene method evidence — SOFT PASS

**Outcome**: Ran aligned 2000-iteration sparse-depth baselines and M24.2 PRISM topology-retention rows on two public COLMAP-style scenes: Mip-NeRF 360 `bonsai` and ETH3D `courtyard`. All current-branch runs used online W&B. Independent `render.py + metrics.py` was completed for all four checkpoints, and a new collector writes JSON/CSV/Markdown summary tables.

**W&B**:
- `bonsai` sparse-depth baseline: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xdct9uys`
- `bonsai` M24.2 PRISM: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dmasxcej`
- `courtyard` sparse-depth baseline: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mdan8yc2`
- `courtyard` M24.2 PRISM: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r9zgtuyp`

**Metrics**:
- `bonsai`: training delta `+0.0960` PSNR, `+0.0027` SSIM, `-0.0036` LPIPS; independent delta `-0.0304` PSNR, `+0.0305` SSIM, `-0.0060` LPIPS; W&B triangle delta `-0.50%`; PRISM `1` commit, `3` rollbacks, `2` no-candidate retries; validation `4/4` observable and `2/4` pass.
- `courtyard`: training delta `+0.0103` PSNR, `+0.0011` SSIM, `-0.0011` LPIPS; independent delta `+0.1152` PSNR, `+0.0347` SSIM, `-0.0087` LPIPS; W&B triangle delta `-1.49%`; PRISM `3` commits, `0` rollbacks, `4` no-candidate retries; validation `5/5` observable and `3/5` pass.

**Decision**: M26 proves the method transfers mechanically to public geometry-observable scenes, but direct 2000-iteration W&B topology reduction is still too small for a strong final paper claim. Checkpoint-topology deltas are larger but must be treated as schedule/accounting effects until runtime W&B topology, checkpoint topology, and final-cleanup summaries are reconciled. Next step is M27 schedule/accounting tuning before full-budget public-scene sweeps.

**Linked artefacts**:
- `docs/car_model/meshprior_stage26_cross_scene_report.md`
- `scripts/car_model/meshprior_collect_stage26_cross_scene.py`
- `outputs/carnet/meshprior/stage26_cross_scene/summary/stage26_cross_scene_summary.md`

---

## 2026-05-02 — Stage25 public multidataset validation — SOFT PASS

**Outcome**: Prepared public datasets under `/data/peilincai/mesh_datasets`, audited current-loader compatibility, ran three representative 700-iteration training checks with online W&B, and fixed PRISM validation reporting for non-observable geometry.

**Data**:
- Mip-NeRF 360 full `360_v2.zip` extracted; seven COLMAP scenes are trainable.
- ETH3D `courtyard` downloaded and converted into the current `images + sparse/0` loader layout; the official all-scene high-resolution training undistorted archive is also complete at `/data/peilincai/mesh_datasets/eth3d/downloads/multi_view_training_dslr_undistorted.7z`.
- Tanks and Temples official downloader was blocked by login/HTML responses, so `truck` and `barn` were prepared from the NSVF mirror using `scripts/car_model/meshprior_convert_nsvf_to_colmap.py`.

**W&B**:
- Mip-NeRF 360 `bonsai`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/x75zddff`
- Tanks and Temples `truck` fixed run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5pre7o19`
- ETH3D `courtyard`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/78iu6goq`

**Metrics**:
- Mip-NeRF 360 `bonsai`: test PSNR `17.2853 -> 20.1716`, SSIM `0.5920 -> 0.7247`, LPIPS `0.4395 -> 0.3105`.
- ETH3D `courtyard`: test PSNR `16.5933 -> 17.9631`, SSIM `0.5596 -> 0.6043`, LPIPS `0.5460 -> 0.5050`.
- Tanks `truck`: training completed after the validation-summary fix, but sparse geometry validation reports `no_sparse_matches` because the available mirror lacks real COLMAP image tracks.

**Decision**: M25 is a multidataset trainability `SOFT PASS`. The code is ready for cross-scene method experiments on Mip-NeRF 360 and ETH3D. Tanks and Temples needs official reconstruction assets or a local COLMAP reconstruction before paper-grade geometry claims.

**Linked artefacts**:
- `docs/car_model/meshprior_stage25_multidataset_validation_report.md`
- `scripts/car_model/meshprior_stage25_dataset_audit.py`
- `scripts/car_model/meshprior_convert_nsvf_to_colmap.py`
- `outputs/carnet/meshprior/stage25_multidataset/dataset_audit.json`

---

## 2026-04-29 — Stage 1 (Object cache & canonicalization audit) — DONE

**Outcome**: cache audit passed. SP-CarNet can proceed on real object-level data without any cache rebuild.

**Headline numbers**:
- 2 433 objects (1 854 train / 206 val / 373 test); 1 patch per object across the board.
- 100 % of objects link to a source GLB via the manifest.
- Cache split: 1 616 records at format v2 (no symmetry persisted), 817 records at format v3 (symmetry persisted as `symmetry_plane_normal/offset/confidence/chamfer_residual`).
- `clean_points (2048, 3)`, `clean_normals (2048, 3)`, `visible_clean_points / hidden_clean_points`, `observed_points (768, 3)`, `query_points_all (1280, 3)` with `query_labels_all` and `query_ignore_mask`, `surface_query_points (512, 3)`, `free_query_points (512, 3)`, `free_space_query_hard_negatives (128, 3)` are all present.
- Every record is already centred (`patch_center_world == 0`) and unit-radius (`patch_radius_m == 1.0`) — canonical identity transform is the working default.
- Front-axis convention is **not** annotated. PCA orientation is provided as an opt-in fallback with a flagged eigenvector-sign caveat.
- Scanner pose is not persisted; runtime sampling via the existing LiDAR corruption module remains the route for `L_ray` evidence in Stage 3+.

**Files added**: `docs/car_model/spcarnet_stage1_object_cache_design.md`, `scripts/car_model/build_spcarnet_object_index.py`, `ss3dm_prior/data/spcarnet_object_dataset.py`, `scripts/car_model/smoke_test_spcarnet_stage1.py`, `outputs/carnet/spcarnet/object_index_v1.json`, `docs/car_model/spcarnet_stage1_object_cache_report.md`, this log.

**No file modified**: CarNet_v0 / v0.7 / v0.8.x configs, the patch-centric dataset, and the trainer remain untouched.

**Smoke test**: `[smoke] PASS` — index build, dataset open over all three splits, `clean_points_object (2048, 3) float32` non-NaN, `partial_observed_points (768, 3) float32` non-NaN, occupancy labels strictly ∈ {0, 1} after applying the ignore mask, identity round-trip error 0.0, PCA-style round-trip error 5.96 × 10⁻⁸, batch collate produces the expected fixed-shape stacks for `B = 2`.

**Decision**: Stage 1 gate **PASSED**. Proceeding to Stage 2 (shape-field auto-decoder).

**Linked artefacts**:
- Design — `docs/car_model/spcarnet_stage1_object_cache_design.md`
- Report — `docs/car_model/spcarnet_stage1_object_cache_report.md`
- Index artefact — `outputs/carnet/spcarnet/object_index_v1.json`

---

## 2026-04-29 — Stage 2 (Canonical object-level shape-field auto-decoder) — IMPLEMENTED, smoke PASS, full-training pending

**Outcome**: code complete, smoke green; the headline auto-decoder run has not yet been launched. A small pre-launch hardening pass (checkpoint emission inside `fit()`, periodic eval, wandb integration) is the only remaining work before the first headline run.

**Architecture locked in**:
- Decoder: 6-layer FiLM-modulated MLP, hidden_dim=384, latent_dim=256, Fourier features with 32 log-spaced frequencies, occupancy logit head (SDF head + eikonal regulariser are wired as ablation).
- Per-object latent table `LatentTable(num_objects, latent_dim)` initialised `N(0, 0.01)`.
- Query budget per object per step: 384 surface + 384 free + 128 hard-negative + 128 mixed (with `query_ignore_mask` honoured) = 1024. SDF mode adds 256 eikonal samples.
- Optim: Adam, decoder LR 5e-4, latent LR 1e-3, grad_clip 1.0. Latent prior `w_zL2 = 1e-4 · ||z||² / d_z`.
- Trains on `train` only; `val` reserved for the eval entrypoint and the Stage gate.

**Files added**: `docs/car_model/spcarnet_stage2_shape_field_design.md`, `ss3dm_prior/models/spcarnet_shape_field.py`, `ss3dm_prior/training/__init__.py`, `ss3dm_prior/training/spcarnet_autodecoder.py`, `ss3dm_prior/training/spcarnet_autodecoder_cli.py`, `configs/ss3dm_prior/spcarnet/model_spcarnet_shape_field_autodecoder.yaml`, `configs/ss3dm_prior/spcarnet/train_spcarnet_shape_field_autodecoder.yaml`, `scripts/car_model/train_spcarnet_shape_field_autodecoder.sh`, `scripts/car_model/eval_spcarnet_shape_field_autodecoder.py`, `scripts/car_model/smoke_test_spcarnet_stage2.py`, `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`.

**No file modified**: `ss3dm_prior/engine/trainer.py`, the v0.x configs, the patch-centric dataset, the v0.x launchers — the auto-decoder line is fully isolated under `configs/ss3dm_prior/spcarnet/` and `ss3dm_prior/training/`. RFC §6 "demote, don't delete" honoured.

**Smoke test** — `scripts/car_model/smoke_test_spcarnet_stage2.py`:
- `[stage2-smoke] PASS` after 2 iters on 2 objects with a tiny 32-d latent / 64-wide / depth-3 decoder.
- `loss_total` 2.0794 → 2.0742 (strict decrease).
- Each BCE term lands at 0.6931 ≈ ln 2 at iter 0, confirming a clean "uninformative" init.
- Decoder gradients non-zero on at least one parameter; latent table gradients non-zero. Both pathways live.
- Marching-Cubes call returns `mesh=None, vertex_count=0` at resolution=16 — expected fallback for an untrained sigmoid field; smoke validates the pipeline runs, not the mesh quality.

**Stage gate** (unchanged, conditional on the headline run):
- `mesh_iou_at_0.5_mean ≥ 0.92`
- `recon_chamfer_l1_mean ≤ 0.05` (canonical units)
- `mesh_extraction_success_rate ≥ 0.95`

All three must hold simultaneously on `val`.

**Decision**: Stage 2 implementation gate **PASSED**. Stage 2 *training* gate is conditional on the headline run; advancing to Stage 3 (per-object MAP refinement at val time) is blocked on §5 of the implementation report.

**Linked artefacts**:
- Design — `docs/car_model/spcarnet_stage2_shape_field_design.md`
- Implementation report — `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md`

---

## 2026-04-29 — Stage 2 (autodecoder_v1, headline) — TRAIN COMPLETE; gate **soft FAIL** on chamfer, IoU metric was broken

**Outcome**: 200-epoch run on the full train split (1854 objects) finished cleanly in **34 minutes** on GPU 5. Final wandb summary `loss_total=0.00468`, `loss_surf=0.00394`, `loss_free=0.00064`, `loss_hard=0.0`, `loss_mixed=0.0002`, `loss_zL2=0.03031`. Wandb run: `5ipij4y9`.

**Train-set eval (64 obj subsample, mc_resolution=32)**:
- `mesh_extraction_success_rate = 1.000` (✓, gate ≥ 0.95)
- `recon_chamfer_l1 = 0.066` (✗, gate ≤ 0.05 — over by 32 %)
- `mesh_iou_at_0.5 = 0.488` (✗, gate ≥ 0.92) — but this number was **a metric bug**, not a model failure (see below)
- `surface_normal_consistency = 0.735`
- `hidden_chamfer_l1 = 0.097`

**Val eval was not informative** — the auto-decoder paradigm has no per-object latent for val/test by construction (those splits had no Stage-2 z to optimise over). Val mesh extraction ran 0/206 because the eval script skipped objects without a Stage-2 latent. This is the Stage-2 → Stage-3 boundary, not a bug.

**IoU metric correction (sub-task)**: the `_voxelise_points` step in `eval_spcarnet_shape_field_autodecoder.py` voxelised only 2 048 `clean_points` at 32³, which biases IoU to ~0.5 even on perfect reconstruction (sparse shell vs filled volume). Fixed in-place by `_voxelise_gt_mesh` which loads the source GLB via the manifest, applies the Stage-1 canonical transform from `patch_metadata_json`'s `original_centroid_world / original_radius_world`, and uses `mesh.voxelized(2/res).fill().matrix` as filled GT. Falls back to a dilated-shell IoU (reported under `mesh_iou_at_0.5_shell`) when the GLB is missing.

**Re-eval on first 16 train objects (post-fix, mc_resolution=32)**:
- `mesh_iou_at_0.5_mean = 0.590` (filled GT, n=6 with local GLB)
- `mesh_iou_at_0.5_shell_mean = 0.922` (shell fallback, n=10 missing GLB)
- vs broken `0.488`

The shell-IoU at 0.922 is consistent with the geometry being substantially correct but the chamfer being slightly looser than the gate.

**Surprises documented for Stage 1 / cache layout**:
1. Manifest's nominal `dataset_root + ./raw/<id>.glb` does **not** exist on disk; actual GLBs live at `/data/peilincai/car_models/meshfleet_ext_v02/{train,test}/raw/`, with only ~6/16 of the first 16 train cars present locally — heavy fallback usage.
2. The cache's canonicalisation is **not** identity. NPZ headers report `patch_center_world=0, patch_radius_m=1`, but the actual world→cache transform is `(v - original_centroid_world) / original_radius_world` from `patch_metadata_json`. Stage 1's "identity is the working default" finding is misleading — the points are pre-canonicalised, but the canonical transform is non-trivial when re-projecting external mesh data into the cache frame. The Stage-1 design doc and Stage-2 eval script both depend on the post-fixed transform now.

**Decision**: Stage-2 v1 is "soft pass" — pipeline is healthy, geometry is recognisable, but the chamfer gate is missed by ~30 %. A v2 retrain with bigger query budget + 300 epochs is in flight (see next entry).

**Linked artefacts**:
- Implementation report — `docs/car_model/spcarnet_stage2_shape_field_implementation_report.md`
- v1 eval (broken IoU): `outputs/carnet/spcarnet/autodecoder_v1/eval_train_64.json`
- v1 eval (fixed IoU): `outputs/carnet/spcarnet/autodecoder_v1/eval_train_16_iou_fix.json`
- Wandb: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/5ipij4y9

---

## 2026-04-29 — Stage 2 (autodecoder_v2, retrain) — IN FLIGHT

**Goal**: push `recon_chamfer_l1` below 0.05 to clear the Stage-2 gate cleanly.

**Diff vs v1**: queries doubled (`surface=768, free=768, hard=256, mixed=256`, total 2048 / object / step), epochs `200 → 300`. All other hyperparameters unchanged. Output dir `autodecoder_v2/`; v1 preserved at `autodecoder_v1/`.

**Status**: PID 1070553 on GPU 5. Wandb run `mpdb1mm7`. Currently ~4350/69300 steps (epoch 18) at sub-agent handover; loss curve healthy and decreasing; no Traceback/OOM.

**Linked artefacts**:
- Wandb: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/mpdb1mm7
- Log: `outputs/carnet/spcarnet/autodecoder_v2/logs/train.log`

---

## 2026-04-29 — Stage 3 (posterior encoder `q(z | O)`) — IMPLEMENTED, smoke + integration smoke PASS, headline pending

**Outcome**: code complete; standalone smoke and trainer-integration smoke (3 steps against the real Stage-2 v1 checkpoint) both pass. The headline run is **not yet launched** — GPU 5 is occupied by the Stage-2 v2 retrain.

**Architecture locked in**:
- Encoder: PointNet tokeniser + 4 cross-attention / 2 self-attention blocks over 32 learnable queries, `feature_dim=256`, ffn 1024, heads 8, dropout 0.1.
- Posterior: variational `(μ, log σ²)` with reparameterisation; KL warmup `0 → 1e-3` over 10 ep; free-bits 0.1 nats/dim.
- Latent supervision: L2 regression of `μ` against the Stage-2 v1 latent table (frozen, train-only by construction); `w_z` warmup 2 → 10 over 10 ep.
- Reconstruction terms: BCE on partial-observed surface + free queries + hard negatives + mixed queries (with ignore mask), all decoded through the **frozen** Stage-2 v1 decoder.
- Optim: AdamW, encoder LR 3e-4, weight_decay 1e-4, grad_clip 1.0, batch 16, 150 epochs.
- Decoder finetune ablation wired (last 2 FiLM blocks + field head, LR 1e-5, off by default).

**Files added**: `docs/car_model/spcarnet_stage3_posterior_encoder_design.md`, `ss3dm_prior/models/spcarnet_posterior.py`, `ss3dm_prior/training/spcarnet_posterior.py`, `ss3dm_prior/training/spcarnet_posterior_cli.py`, `configs/ss3dm_prior/spcarnet/{model,train}_spcarnet_posterior_encoder.yaml`, `scripts/car_model/{train,smoke_test,eval}_spcarnet_posterior_encoder.{sh,py,py}`, `scripts/car_model/smoke_test_spcarnet_stage3.py`, `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md`.

**No file modified**: Stage-1 dataset, Stage-2 trainer/decoder, v0.x configs/launchers, the patch-centric trainer. Stage-2 v1 checkpoint is read-only input.

**Smoke test**: standalone CPU smoke `[stage3-smoke] PASS` — encoder forward shape `(2, 32)`, initial logvar `−9.21` matches `log(0.01²)`, two reparameterised samples differ by 0.011, decoded logits `(2, 64)` finite with sigmoid mean 0.500 (uniform field at init), encoder gradients flow, decoder gradients **don't** (frozen). Integration smoke (3 steps, real Stage-2 v1 ckpt) ran cleanly in 2.18 s on GPU 5; wandb run `9kehaimo` synced; checkpoint payload schema matches the eval script.

**Stage gate** (per RFC §7, conditional on the headline run):
- `recon_chamfer_l1_mean ≤ 0.10` on val (matches v0.7's floor).
- `free_space_violation_rate_mean` strictly better than v0.7's.
- Both within 150 epochs.

**Decision**: Stage-3 implementation gate **PASSED**. Headline run is queued behind the Stage-2 v2 retrain on GPU 5; can be parallelised on a free GPU at user discretion.

**Linked artefacts**:
- Design — `docs/car_model/spcarnet_stage3_posterior_encoder_design.md`
- Implementation report — `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md`
- RFC — `docs/car_model/SPCarNet_radical_RFC.md` (§3.4–§3.7, §6 EN-Q row, §7 Stage-3 gate)

---

## 2026-04-29 — Stage 3 (posterior encoder) — TRAIN COMPLETE; gate **PASS**

**Outcome**: 150-epoch run on the full train split (1854 objects) finished cleanly in **23 minutes** on GPU 1. Wandb run `eau9yg7m`. Final summary `loss_total=0.674`, `loss_z=0.012` (latent regression converged), `loss_surf=0.059`, `loss_free=0.111`, `loss_kl=346`, `posterior/logvar_mean=-3.65` (no collapse — would need to be < −8 to indicate collapse). KL stable at 346 vs free-bits floor of 25.6 (0.1 nats × 256 dims) — encoder is using meaningful capacity.

**Val eval (full 206 objects, mc_resolution=32)**:
- `mesh_extraction_success_rate = 1.000` (all 206 objects produced a mesh)
- `recon_chamfer_l1_mean = **0.0664**` — beats v0.7 (0.10) by 33 %, beats v0.8.2 (0.12) by 45 %
- `hidden_chamfer_l1_mean = 0.0991`
- `visible_preservation_error_mean = 0.0627`
- `free_space_violation_rate_mean = **0.0335**` (excellent; gate "strictly better than v0.7")
- `mesh_iou_at_0.5_mean = 0.471` (sparse-point fallback; not gated)
- `zero_corruption_recon_chamfer_l1_mean = 0.0666` ≈ `recon_chamfer_l1_mean = 0.0664` — **amortisation gap is essentially zero**
- `latent_retrieval_error_mean = NaN` (correctly masked on val — no leakage)

**Stage-3 gate PASS** on both the chamfer threshold (≤ 0.10) and the free-space violation requirement (strictly better than v0.7). The bottleneck is now **the Stage-2 decoder ceiling**, not the encoder.

**Linked artefacts**:
- Wandb: https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet/runs/eau9yg7m
- Eval JSON: `outputs/carnet/spcarnet/posterior_encoder_v1/eval_val.json`
- Implementation report: `docs/car_model/spcarnet_stage3_posterior_encoder_implementation_report.md`

---

## 2026-04-29 — Stage 4 (observation-consistency MAP refinement) — IMPLEMENTED, smoke + 50-obj refinement PASS, gate **soft pass**

**Outcome**: code complete; smoke and 50-object val refinement run both finished cleanly. Refinement helps on every metric in the right direction; chamfer improvement is half the design-side margin gate but inside the RFC §7 no-degradation triggers. Free-space violation **almost halves** (−59 %).

**Architecture / protocol locked in**:
- Loss module `ss3dm_prior/losses_spcarnet_observation.py`: Tier-1 `L_surf_obs + L_free + L_mixed`, Tier-2 `L_ray + L_incidence` (Tier-2 disabled on the current cache because scanner pose is not persisted — fallback documented in design §2).
- Huber wrap with `δ = 0.5` on every BCE term (robust to outlier observations).
- Refinement protocol: init `z = μ(O)` from the Stage-3 encoder, Adam on `[z]` only with frozen decoder, default 50 steps × LR 1e-2.
- Held-out validation partition (default 20 % of `query_points_all`) for keep-best tracking.
- Three early-stop triggers: plateau (10-step patience on held-out score), free-space violation increase (> 10 % over initial), z drift > 5×prior σ.
- Output JSON splits `inference_only_metrics` (real-deployment-safe) from `gt_dependent_metrics` (eval only).

**50-object val refinement results (default config)**:

| Metric | Before | After | Δ |
|---|---|---|---|
| `recon_chamfer_l1_mean` | 0.0715 | 0.0690 | −0.0025 |
| `hidden_chamfer_l1_mean` | 0.1078 | 0.1054 | −0.0024 |
| `visible_preservation_error_mean` | 0.0644 | 0.0610 | −0.0034 |
| `free_space_violation_rate_mean` | 0.0358 | **0.0147** | **−0.0211 (−59 %)** |

21/50 early stops (20 plateau, 1 free-space-increase — safeguard fired correctly). 0.92 s / object refinement time. `z_drift_final_mean = 1.86` (well within prior bound).

**Gate verdict**:
- RFC §7 hidden-chamfer ceiling (≤ 5 % degradation): ✓ — improved 2.2 %.
- RFC §7 free-space ceiling (≤ 10 % degradation): ✓ — improved 59 %.
- Design-side chamfer margin (≥ 0.005 improvement): **missed by ~2×** (got 0.0025).
- Decision: **soft pass**. Refinement is helpful but bounded by the Stage-2 decoder ceiling, exactly as predicted by the Stage-3 amortisation-gap diagnostic.

**No file modified**: Stage 1/2/3 modules and the v0.x line are untouched. Stage-3 checkpoint is read-only input.

**Files added**: `docs/car_model/spcarnet_stage4_observation_map_design.md`, `ss3dm_prior/losses_spcarnet_observation.py`, `scripts/car_model/refine_spcarnet_latent_map.py`, `scripts/car_model/smoke_test_spcarnet_stage4.py`, `docs/car_model/spcarnet_stage4_observation_map_implementation_report.md`.

**Linked artefacts**:
- Refinement JSON: `outputs/carnet/spcarnet/map_refinement/val_50_default/refinement.json`
- Implementation report: `docs/car_model/spcarnet_stage4_observation_map_implementation_report.md`

---

## 2026-04-30 — Stage 5 (multi-hypothesis sampling & reranking) + Stage 2 v3 sanity check

**Outcome**: Stage 5 implemented end-to-end. K∈{1, 4, 8} sweep on 50 val objects. **Mixed gate**: oracle best-of-K=8 beats K=1 by 0.0060 chamfer (passes RFC §7 ≥0.005 margin) — *the posterior is genuinely multi-modal*. But the inference-only reranker score (BCE losses + `log p(z)`) ranks the wrong candidate: top1-reranked is +0.002 chamfer *worse* than K=1. The headline-gate (top1 reranked beats K=1 by ≥0.005) **fails**.

**Stage-2 v3 sanity (run in parallel)**: bigger decoder (latent 512, hidden 768, depth 8, 300 ep) → train chamfer 0.0692 vs v1 ~0.066. **Did not lift the ceiling**; v1/v2/v3 are within 0.003 chamfer of each other. Stage 3 is **not** re-paired against v3.

**Architecture / protocol**:
- Sample K from variational posterior with `torch.manual_seed(seed_base + k)` per candidate (one encoder pass, K MC extractions).
- Score = `−L_obs(z_k) + log p(z_k)` where `L_obs = w_surf·BCE(P_obs,1) + w_free·BCE(Q_free,0) + w_mixed·BCE_with_ignore(Q_all)` (no Huber wrap; likelihood form).
- Diversity primary metric: pairwise top-3 chamfer; secondary: latent-L2.
- Mesh extraction (MC res 32) post-hoc per candidate; failed extractions excluded from rerank/oracle.

**50-object val sweep results**:

| Metric | K=1 | K=4 | K=8 |
|---|---|---|---|
| `top1_score_recon_chamfer_l1` | **0.0715** | 0.0734 | 0.0735 |
| `oracle_best_of_k_recon_chamfer_l1` | 0.0715 | **0.0669** | **0.0655** |
| `top1_score_visible_preservation_error` | 0.0632 | 0.0644 | 0.0650 |
| `top1_score_free_space_violation_rate` | 0.0366 | 0.0395 | 0.0364 |
| `diversity_chamfer_top3` | NaN | 0.0348 | 0.0342 |
| `diversity_latent_l2` | NaN | 3.91 | 3.88 |
| `mesh_extraction_success_rate` | 1.00 | 1.00 | 1.00 |
| seconds / object | 0.60 | 2.33 | 3.23 |

**Why the reranker fails — and why fixes don't help**: post-hoc, we tested three score variants on the existing K=8 JSON via `scripts/car_model/rescore_spcarnet_multihypothesis.py` (recomputes top1 without re-running):

| K=8 variant | top1 chamfer | vs K=1 (0.0715) |
|---|---|---|
| default (`-L_obs + log p(z)`) | 0.0735 | +0.0020 |
| no_prior (`-L_obs`) | 0.0737 | +0.0022 |
| norm_penalty (`-L_obs - 0.5·max(0,‖z‖-4)`) | 0.0738 | +0.0023 |
| **oracle (GT chamfer)** | **0.0655** | **−0.0060** |

K=4 same pattern (no_prior best non-oracle at 0.0725, still +0.0010 over K=1). **No inference-only variant beats K=1.** The real issue is not the prior term: `L_obs` (BCE on observation queries) is decorrelated from chamfer-to-GT in the local neighbourhood of the posterior, because BCE only sees 768 partial-obs points + a fixed query grid, not the unobserved surface that chamfer measures. This rules out a whole family of approaches (any score that uses only `(z, decoder, partial obs)`) — Stage 7-aux now has a strong empirical motivation to bring in evidence the reranker doesn't currently see (symmetry consistency, RAG against a shape bank, manifold quality scores).

**Why oracle wins**: posterior σ is calibrated such that ~1 in 8 samples lands inside the GT-closer side of the local mode. Latent-L2 spread (3.9) is comparable to prior σ × √D; mesh-space top-3 chamfer spread (0.034) is half the typical chamfer level — meaningful but not chaotic.

**v3 sanity numbers (train, 100 obj, MC 32)**: chamfer_l1 = 0.0692, mesh_iou_shell = 0.914, n_extracted = 100/100. Confirms decoder ceiling is family-level, not capacity-level.

**Gate verdict**:
- RFC §7 chamfer margin ≥ 0.005 (top1 reranked vs K=1): **✗** (wrong direction by 0.002).
- RFC §7 chamfer margin ≥ 0.005 (oracle vs K=1): ✓ (−0.006).
- RFC §7 free-space ceiling (≤ 10 % degradation): ✓ (K=8: 0.0364 vs K=1: 0.0366 — flat).
- RFC §7 mesh-extraction (no regression): ✓ (1.00 across all K).
- Diversity-doubling gate (K=8 top-3 ≥ 2× K=4): **✗** (0.0342 vs 0.0348 — flat). Gate-design issue: doubling assumes multi-modal; ours is unimodal-broad.
- **Decision: drop multi-hypothesis from headline, keep K=1; retain K=8 oracle as ablation row in the paper.**

**No file modified**: Stage 1/2/3/4 modules untouched. Stage-3 v1 checkpoint is read-only input.

**Files added**: `docs/car_model/spcarnet_stage5_multihypothesis_design.md`, `scripts/car_model/eval_spcarnet_multihypothesis.py`, `scripts/car_model/smoke_test_spcarnet_stage5.py`, `scripts/car_model/rescore_spcarnet_multihypothesis.py`, `docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md`. Stage-2 v3 launcher: `scripts/car_model/train_spcarnet_shape_field_autodecoder_v3.sh`.

**Linked artefacts**:
- Stage 5 K=1 / K=4 / K=8 JSONs: `outputs/carnet/spcarnet/multihypothesis/val_50_K{1,4,8}/K{1,4,8}.json`
- v3 checkpoint (preserved, not used downstream): `outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt`
- v3 train eval: `outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json`
- Implementation report: `docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md`

---

## 2026-05-01 — MeshPrior Stage 0 (repository audit) — PASS / PROCEED

**Outcome**: M0 repository integrity audit completed for the SP-CarNet → MeshPrior transition. No new method code was implemented.

**Environment**:
- Default shell Python is `3.13.2` and does not have `torch`; it is not the project environment.
- `micromamba run -n mesh_splatting` provides Python `3.11.14`, `torch 2.7.1+cu126`, CUDA available, `cuda_device_count=8`.
- `python -m compileall scripts/car_model ss3dm_prior -q` passes in the `mesh_splatting` environment.

**Code audit**:
- Required SP-CarNet source files are present, including `spcarnet_object_dataset.py`, `spcarnet_shape_field.py`, `spcarnet_posterior.py`, Stage-2/Stage-3 trainers, Stage-4 observation loss, and Stage-1/3/4/5 scripts.
- `ss3dm_prior.models.spcarnet_shape_field` and `ss3dm_prior.models.spcarnet_posterior` import cleanly.
- Worktree was already dirty before this audit: `scripts/car_model/eval_spcarnet_multihypothesis.py` modified, `docs/prompts.md` untracked, and two submodules reported as dirty/unknown.

**Smoke tests**:
- `smoke_test_spcarnet_stage1.py`: PASS.
- `smoke_test_spcarnet_stage2.py`: PASS.
- `smoke_test_spcarnet_stage3.py`: PASS.
- `smoke_test_spcarnet_stage4.py`: PASS.
- `smoke_test_spcarnet_stage5.py`: PASS.

**Artifact audit**:
- Stage-2/3/4/5 checkpoints and JSONs exist under `outputs/carnet/spcarnet/`.
- Key reported metrics are supported by local JSONs, including Stage-3 `recon_chamfer_l1_mean=0.066391`, `free_space_violation_rate_mean=0.033535`, Stage-4 refinement `0.071490 -> 0.069032` chamfer and `0.035820 -> 0.014688` free-space violation, and Stage-5 K=8 oracle `0.065528` vs top1 reranked `0.073501`.

**Decision**: M0 recommendation is `PROCEED`. The next allowed prompt is M1, the MeshPrior scene-optimization RFC, with no model-code changes.

**Linked artefact**:
- Audit report: `docs/car_model/meshprior_stage0_repository_audit.md`

---

## 2026-05-01 — MeshPrior Stage 1 (scene mesh-prior RFC) — COMPLETE / PROCEED_TO_M2

**Outcome**: Wrote the MeshPrior research RFC that pivots SP-CarNet from object-only completion to object-prior-guided scene mesh optimization. No model code was changed.

**Central claim**: learned object-centric shape posteriors can safely guide scene mesh optimization when converted into bounded local proposals and filtered by scene-level evidence gates.

**Method slogan**: `Prior proposes; evidence disposes.`

**Planned system layers**:
- repository/object-prior integrity,
- scene/object region mining,
- object posterior inference in canonical frame,
- mesh repair proposal generation,
- scene evidence gates and rollback,
- alternating scene optimization,
- NeurIPS-grade evaluation and reporting.

**Proposal order**: protect/prune first, snap second, guarded fill third, split/collapse refinement last.

**Decision**: M1 is complete. The next allowed stage is M2 region mining.

**Linked artefact**:
- RFC: `docs/car_model/meshprior_stage1_scene_meshprior_RFC.md`

---

## 2026-05-01 — MeshPrior Stage 2 (scene/object region mining) — PASS

**Outcome**: Implemented the first scene/object bridge for MeshPrior: a conservative region mining layer that can process PLY meshes when present and emits clean dry-run artifacts when no scene mesh or segmentation exists.

**Files added**:
- `ss3dm_prior/meshprior/__init__.py`
- `ss3dm_prior/meshprior/region_types.py`
- `scripts/car_model/meshprior_mine_regions.py`
- `scripts/car_model/smoke_test_meshprior_stage2_region_mining.py`
- `docs/car_model/meshprior_stage2_region_mining_design.md`
- `docs/car_model/meshprior_stage2_region_mining_implementation_report.md`
- `docs/car_model/meshprior_stage2_region_mining_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage2_region_mining.py`: PASS, synthetic two-component mesh produced `regions=2`, `eligible_for_posterior=1`.
- Missing-data dry-run: PASS, emitted empty region set and exited cleanly.

**Contract**:
- Outputs `regions.json`, `regions_summary.csv`, and `region_mining_report.md`.
- Very small components are retained as diagnostics but not marked eligible for posterior inference.
- No SP-CarNet posterior inference and no scene geometry modification happen in M2.

**Decision**: M2 gate `PASS`. The next allowed stage is M3 scene-region posterior inference.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage2_region_mining_design.md`
- Implementation report: `docs/car_model/meshprior_stage2_region_mining_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage2_region_mining_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 3 (scene-region posterior inference) — PASS

**Outcome**: Implemented the wrapper that takes mined scene regions, samples region point clouds, canonicalizes them with a conservative bbox/PCA transform, runs the Stage-3 SP-CarNet posterior encoder, and writes posterior diagnostics for later proposal generation.

**Files added**:
- `ss3dm_prior/meshprior/scene_region_posterior.py`
- `scripts/car_model/meshprior_infer_region_posterior.py`
- `scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py`
- `docs/car_model/meshprior_stage3_scene_region_posterior_design.md`
- `docs/car_model/meshprior_stage3_scene_region_posterior_implementation_report.md`
- `docs/car_model/meshprior_stage3_scene_region_posterior_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage3_region_posterior.py`: PASS.
- Missing-checkpoint path fails clearly with `posterior_checkpoint not found`.
- With local checkpoint `outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt`, one synthetic region produced `z_mean.npy`, `z_logvar.npy`, `canonical_transform.json`, `posterior_summary.json`, sampled points, and an occupancy grid.

**Diagnostics from smoke**:
- `field_occupancy_ratio=0.070068`.
- `posterior_mu_norm=2.835622`.
- `posterior_logvar_mean=-3.936054`.
- `uncertainty_score=0.146494`.
- MC extraction succeeded at smoke resolution with `vertex_count=461`, `face_count=926`, watertight.

**Decision**: M3 gate `PASS`. The next allowed stage is M4 protect/prune proposal generation.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage3_scene_region_posterior_design.md`
- Implementation report: `docs/car_model/meshprior_stage3_scene_region_posterior_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage3_scene_region_posterior_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 4 (protect/prune proposals) — PASS

**Outcome**: Implemented the first safe MeshPrior proposal types: protect and prune. This stage only emits triangle-level scores and proposal records; it does not move vertices and does not fill holes.

**Files added**:
- `ss3dm_prior/meshprior/proposals.py`
- `ss3dm_prior/meshprior/protect_prune.py`
- `scripts/car_model/meshprior_make_protect_prune_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage4_protect_prune.py`
- `docs/car_model/meshprior_stage4_protect_prune_design.md`
- `docs/car_model/meshprior_stage4_protect_prune_implementation_report.md`
- `docs/car_model/meshprior_stage4_protect_prune_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage4_protect_prune.py`: PASS.

**Synthetic scoring result**:
- cube surface protect score mean `0.999990`.
- floater protect score `0.000010`.
- cube prune score mean `0.0`.
- floater prune score `0.999980`.
- both `protect` and `prune` proposal types generated.

**Decision**: M4 gate `PASS`. The next allowed stage is M5 optimizer adapter.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage4_protect_prune_design.md`
- Implementation report: `docs/car_model/meshprior_stage4_protect_prune_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage4_protect_prune_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 5 (optimizer adapter) — PASS

**Outcome**: Implemented a neutral optimizer adapter that exports MeshPrior protect/prune scores for downstream consumption without patching PRISM or overriding scene evidence.

**Files added**:
- `ss3dm_prior/meshprior/optimizer_adapter.py`
- `scripts/car_model/meshprior_export_optimizer_scores.py`
- `scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py`
- `docs/car_model/meshprior_stage5_optimizer_adapter_design.md`
- `docs/car_model/meshprior_stage5_optimizer_adapter_implementation_report.md`
- `docs/car_model/meshprior_stage5_optimizer_adapter_smoke.md`

**PRISM status**: PRISM is present (`utils/prism_scoring.py`, `utils/prism_counterfactual.py`, `utils/prism_pipeline.py`). M5 exports passive artifacts only; no `train.py` or PRISM internals were changed.

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage5_optimizer_adapter.py`: PASS.
- Generic NPZ and PRISM JSON export/reload verified.
- Bounded-add rule verified: MeshPrior score delta cannot exceed configured weight (`0.25` in smoke).

**Decision**: M5 gate `PASS`. The next allowed stage is M6 synthetic mesh-damage benchmark.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage5_optimizer_adapter_design.md`
- Implementation report: `docs/car_model/meshprior_stage5_optimizer_adapter_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage5_optimizer_adapter_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 6 (synthetic mesh-damage benchmark) — PASS

**Outcome**: Implemented a controlled synthetic mesh-damage benchmark for proposal behavior before real scene integration.

**Files added**:
- `ss3dm_prior/meshprior/synthetic_damage.py`
- `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py`
- `scripts/car_model/meshprior_make_synthetic_damage_report.py`
- `scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py`
- `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md`
- `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_implementation_report.md`
- `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage6_synthetic_damage.py`: PASS.
- Synthetic benchmark produced 4 rows across local hole, floater, vertex noise, and density imbalance.
- Controlled floater case achieved `floater_prune_recall=1.0` and valid-surface protect recall >= 0.9.

**Outputs**:
- `metrics.json`, `metrics.csv`, `table_by_damage_type.csv`, `failure_cases.md`.
- Markdown report generation verified.

**Decision**: M6 gate `PASS`. The next allowed stage is M7 conservative snap.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_design.md`
- Implementation report: `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage6_synthetic_damage_benchmark_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 7 (conservative snap proposals) — PASS

**Outcome**: Implemented bounded vertex snap proposals with explicit risk evaluation and a downstream acceptance gate. This is the first MeshPrior stage that proposes geometry movement, so proposals remain passive unless a later scene gate accepts them.

**Files added/updated**:
- `ss3dm_prior/meshprior/snap.py`
- `scripts/car_model/meshprior_make_snap_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage7_snap.py`
- `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py`
- `docs/car_model/meshprior_stage7_conservative_snap_design.md`
- `docs/car_model/meshprior_stage7_conservative_snap_implementation_report.md`
- `docs/car_model/meshprior_stage7_conservative_snap_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage7_snap.py`: PASS.
- `smoke_test_meshprior_stage6_synthetic_damage.py`: PASS with 8 benchmark rows after adding `protect_prune_snap`.
- Small M7 benchmark over `vertex_noise` and `floater`: PASS.

**Benchmark gate detail**:
- A first `snap_max_disp=0.02` benchmark trial improved vertex-noise surface distance but reduced valid-surface protect recall from `0.9167` to `0.8333`; this exceeded the 5 percent preservation tolerance.
- The benchmark snap default was tightened to `0.005`.
- Final `protect_prune_snap` on `vertex_noise` improved surface distance by `0.01073157787322998` while preserving valid-surface protect recall at `0.9166666666666666`.
- Floater prune recall stayed `1.0`.

**Decision**: M7 gate `PASS`. The next allowed stage is M8 guarded patch/fill proposals.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage7_conservative_snap_design.md`
- Implementation report: `docs/car_model/meshprior_stage7_conservative_snap_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage7_conservative_snap_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 8 (guarded fill proposals) — PASS

**Outcome**: Implemented guarded local hole-fill proposals. Fill remains proposal-only and is not approved for scene-level hidden-side completion until M9 evidence gates and rollback exist.

**Files added/updated**:
- `ss3dm_prior/meshprior/fill.py`
- `scripts/car_model/meshprior_make_fill_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage8_fill.py`
- `scripts/car_model/meshprior_run_synthetic_damage_benchmark.py`
- `docs/car_model/meshprior_stage8_guarded_fill_design.md`
- `docs/car_model/meshprior_stage8_guarded_fill_implementation_report.md`
- `docs/car_model/meshprior_stage8_guarded_fill_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage8_fill.py`: PASS.
- `smoke_test_meshprior_stage6_synthetic_damage.py`: PASS.
- Small local-hole benchmark over `damaged_input`, `guarded_fill`, and `snap_fill`: PASS.

**Benchmark gate detail**:
- `damaged_input` local hole had `boundary_edge_count=4`.
- `guarded_fill` reduced boundary edges to `0`, added `4` faces, and kept component-count delta at `0`.
- `snap_fill` also reduced boundary edges to `0`; snap moved no vertices in this case because boundary vertices are fixed by default.
- Free-space violation stayed `0.0` in the controlled analytic benchmark.

**Decision**: M8 gate `PASS`. The next allowed stage is M9 scene evidence gates and rollback.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage8_guarded_fill_design.md`
- Implementation report: `docs/car_model/meshprior_stage8_guarded_fill_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage8_guarded_fill_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 9 (scene gates and rollback) — PASS

**Outcome**: Implemented dry-run scene evidence gates and rollback snapshots for MeshPrior proposals. Proposal acceptance now requires scene-side evidence; object-prior confidence alone is insufficient.

**Files added**:
- `ss3dm_prior/meshprior/scene_gate.py`
- `scripts/car_model/meshprior_evaluate_proposals.py`
- `scripts/car_model/smoke_test_meshprior_stage9_scene_gate.py`
- `docs/car_model/meshprior_stage9_scene_gate_rollback_design.md`
- `docs/car_model/meshprior_stage9_scene_gate_rollback_implementation_report.md`
- `docs/car_model/meshprior_stage9_scene_gate_rollback_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage9_scene_gate.py`: PASS.
- `smoke_test_meshprior_stage8_fill.py`: PASS.

**Gate behavior**:
- Topology-improving fill proposal accepted.
- Disconnected-floater proposal rejected because component count increased.
- Rollback snapshot and restore verified for vertices, faces, and metadata.
- CLI dry-run report generated `accepted_count=1` and `rejected_count=1`.

**Decision**: M9 gate `PASS`. The next allowed stage is M10 scene-level optimization integration.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage9_scene_gate_rollback_design.md`
- Implementation report: `docs/car_model/meshprior_stage9_scene_gate_rollback_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage9_scene_gate_rollback_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 10 (alternating runner) — PASS

**Outcome**: Implemented a dry-run orchestration runner that connects synthetic scene setup, region artifacts, posterior summary, proposal generation, scene gate evaluation, accepted proposal export, and report generation.

**Files added**:
- `scripts/car_model/meshprior_run_pipeline.py`
- `scripts/car_model/smoke_test_meshprior_stage10_pipeline.py`
- `docs/car_model/meshprior_stage10_alternating_runner_design.md`
- `docs/car_model/meshprior_stage10_alternating_runner_implementation_report.md`
- `docs/car_model/meshprior_stage10_alternating_runner_smoke.md`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage10_pipeline.py`: PASS.
- `smoke_test_meshprior_stage9_scene_gate.py`: PASS.

**Pipeline output**:
- Synthetic dry-run completed with `accepted_count=1` and `rejected_count=0`.
- Artifacts written: `regions.json`, `posterior/posterior_summary.json`, proposal files, `scene_gate/gate_report.json`, `accepted_proposals.json`, and `pipeline_report.md`.
- Geometry application remains disabled; `--apply` raises in M10.

**Decision**: M10 gate `PASS`. The next allowed stage is M11 actual scene training/evaluation and wandb.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage10_alternating_runner_design.md`
- Implementation report: `docs/car_model/meshprior_stage10_alternating_runner_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage10_alternating_runner_smoke.md`

---

## 2026-05-01 — MeshPrior Stage 11 (scene experiment) — PASS

**Outcome**: Ran one dry-run scene experiment on the synthetic local-hole scene produced by the M10 pipeline.

**Files added**:
- `docs/car_model/meshprior_stage11_scene_experiment_design.md`
- `docs/car_model/meshprior_stage11_scene_experiment_report.md`
- `scripts/car_model/meshprior_collect_scene_experiment.py`

**Required outputs generated**:
- `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/commands.sh`
- `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/metrics.json`
- `outputs/carnet/meshprior/scene_experiments/m11_synthetic_dryrun/summary.md`

**Smoke / verification**:
- `git status --short`: only existing dirty submodules and M11 files before commit.
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `nvidia-smi`: available with elevated permissions. No fully idle GPU was available because every GPU had active processes and memory allocations.
- `smoke_test_meshprior_stage10_pipeline.py`: PASS.
- Dry-run M11 experiment: PASS.

**Metrics**:
- `proposal_count=1`.
- `accepted_count=1`.
- `rejected_count=0`.
- `boundary_edge_delta_sum=4.0`.
- `component_count_delta_max=0.0`.
- `floater_count_delta_max=0.0`.
- `free_space_violation_delta_max=0.0`.

**Wandb / training**:
- Wandb is installed, but no online wandb run was started.
- Full training was not launched because no fully idle GPU was available.

**Decision**: M11 gate `PASS` for dry-run scene experiment. The next allowed stage is M12 prior calibration, with the caveat that render/COLMAP improvements remain unproven until a real scene checkpoint is evaluated.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage11_scene_experiment_design.md`
- Report: `docs/car_model/meshprior_stage11_scene_experiment_report.md`

---

## 2026-05-01 — MeshPrior Stage 12 (prior calibration) — PASS

**Outcome**: Implemented a post-hoc surface-support calibration profile for proposal reliability. The upgrade targets snap risk and valid-surface preservation, not object Chamfer.

**Files added/updated**:
- `ss3dm_prior/meshprior/calibration.py`
- `scripts/car_model/meshprior_calibrate_prior.py`
- `scripts/car_model/smoke_test_meshprior_stage12_prior_calibration.py`
- `scripts/car_model/meshprior_run_pipeline.py`
- `docs/car_model/meshprior_stage12_prior_calibration_design.md`
- `docs/car_model/meshprior_stage12_prior_calibration_implementation_report.md`
- `docs/car_model/meshprior_stage12_prior_calibration_smoke.md`

**Evidence and calibration**:
- Uncalibrated snap (`max_disp=0.02`) reduced valid-surface protect recall from `0.9167` to `0.8333`.
- `surface_support_v1` snap (`max_disp=0.005`) preserved valid-surface protect recall at `0.9167`.
- Calibrated snap still improved surface distance by `0.01073157787322998`.
- Free-space violation delta remained `0.0`.

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage12_prior_calibration.py`: PASS.
- `smoke_test_meshprior_stage10_pipeline.py`: PASS.
- Targeted experiment wrote `outputs/carnet/meshprior/prior_calibration/stage12_surface_support_v1/calibration_metrics.json`.

**Decision**: M12 gate `PASS`. The next allowed stage is M13 evaluation protocol and matrix.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage12_prior_calibration_design.md`
- Implementation report: `docs/car_model/meshprior_stage12_prior_calibration_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage12_prior_calibration_smoke.md`

---

## 2026-05-01 — Training cleanup blocker repair before M13 — PASS

**Outcome**: Repaired destructive final cleanup behavior found by the wandb training smoke.

**Problem**:
- A non-PRISM 200-iteration training run pruned `5706` triangles to `15` at final cleanup.
- Root cause: final cleanup executed by default when PRISM was disabled.

**Fix**:
- `train.py` now executes final cleanup only when PRISM pruning is enabled and `prism_disable_final_cleanup_prune` is false.
- Ordinary non-PRISM training skips the PRISM-specific destructive cleanup path.

**Verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- 200-iteration wandb repair run on GPU 1: PASS.
- Wandb run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3swt58x2`.
- Final cleanup summary: `final_cleanup_enabled=false`, `final_cleanup_pruned=0`.
- Triangle count preserved: `5706 -> 5706`.
- Vertex count preserved: `17118 -> 17118`.
- COLMAP sparse geometry eval passed at iteration 200 with depth AbsRel `0.10470779720655764`, depth MAE `0.024122862845250084`, normal mean angle `37.51919533010328`.

**Decision**: blocker `PASS`. M13 may proceed only after this repair is committed and pushed.

**Linked artefact**:
- `docs/car_model/meshprior_training_cleanup_repair_report.md`

---

## 2026-05-01 — MeshPrior Stage 13 (evaluation protocol and matrix) — PASS

**Outcome**: Implemented the evaluation protocol, experiment matrix registry, dry-run matrix runner, and NeurIPS-style report generator.

**Files added**:
- `docs/car_model/meshprior_stage13_eval_protocol_design.md`
- `configs/ss3dm_prior/meshprior/meshprior_experiment_matrix.yaml`
- `scripts/car_model/meshprior_run_experiment_matrix.py`
- `scripts/car_model/meshprior_make_neurips_report.py`
- `scripts/car_model/smoke_test_meshprior_stage13_eval_protocol.py`
- `docs/car_model/meshprior_stage13_eval_protocol_implementation_report.md`
- `docs/car_model/meshprior_stage13_eval_protocol_smoke.md`
- `docs/car_model/reports/meshprior_neurips_main_report.md`

**Generated outputs**:
- `outputs/carnet/meshprior/experiment_matrix/matrix_results.json`
- `outputs/carnet/meshprior/reports/object_table.csv`
- `outputs/carnet/meshprior/reports/synthetic_damage_table.csv`
- `outputs/carnet/meshprior/reports/scene_table.csv`
- `outputs/carnet/meshprior/reports/ablation_table.csv`
- `outputs/carnet/meshprior/reports/failure_cases.md`

**Full dry-run matrix**:
- `total=11`
- `available=7`
- `missing=4`

**Smoke / verification**:
- `micromamba run -n mesh_splatting python -m compileall scripts/car_model ss3dm_prior -q`: PASS.
- `smoke_test_meshprior_stage13_eval_protocol.py`: PASS.
- Report generation from `outputs/carnet/meshprior/experiment_matrix/matrix_results.json`: PASS.

**Key available evidence**:
- Stage 3 posterior encoder: recon Chamfer L1 `0.0663909994752951`, hidden Chamfer L1 `0.0990753869336207`, mesh extraction success `1.0`.
- `surface_support_v1` calibration preserves valid-surface protect recall at `0.9166666666666666`.
- 200-iteration no-cleanup scene smoke preserved `5706` triangles and reports COLMAP depth AbsRel `0.10470779720655764`.

**Missing rows retained**:
- `v0_7_residual_baseline`
- `spcarnet_stage4_map_refinement`
- `spcarnet_stage5_oracle_k8`
- `protect_prune_proposals`

**Decision**: M13 gate `PASS`. The next allowed stage is M14, with the caveat that scene MeshPrior application is still dry-run/gated proposal evidence rather than real render-gated insertion.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage13_eval_protocol_design.md`
- Implementation report: `docs/car_model/meshprior_stage13_eval_protocol_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage13_eval_protocol_smoke.md`
- Generated report: `docs/car_model/reports/meshprior_neurips_main_report.md`

---

## 2026-05-01 — Pre-M14 stability audit — PASS

**Outcome**: Ran a stability audit before starting M14 and fixed one reproducibility bug in the smoke tests.

**Risk found and fixed**:
- Some MeshPrior smoke tests used ambient `python` for subprocesses, which can resolve to the wrong interpreter and fail with missing dependencies.
- Updated those subprocess calls to use `sys.executable`.

**Files updated**:
- `scripts/car_model/smoke_test_meshprior_stage2_region_mining.py`
- `scripts/car_model/smoke_test_meshprior_stage3_region_posterior.py`
- `scripts/car_model/smoke_test_meshprior_stage5_optimizer_adapter.py`
- `scripts/car_model/smoke_test_meshprior_stage6_synthetic_damage.py`
- `docs/car_model/meshprior_pre_m14_stability_audit.md`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- MeshPrior smoke tests M2, M3, M4, M5, M6, M7, M8, M9, M10, M12, and M13: PASS.
- M13 matrix/report dry-run: PASS with `total=11`, `available=7`, `missing=4`.

**Remaining non-collapse risks**:
- Scene MeshPrior application is still dry-run/gated proposal evidence.
- The 200-iteration scene result is a smoke run, not a full headline training run.
- Historical missing rows remain intentionally visible as `MISSING`.

**Decision**: Pre-M14 stability gate `PASS`.

**Linked artefact**:
- `docs/car_model/meshprior_pre_m14_stability_audit.md`

---

## 2026-05-01 — MeshPrior Stage 14 (paper roadmap and claim-risk analysis) — PASS

**Outcome**: Wrote the paper-level roadmap and claim-risk analysis for the MeshPrior direction.

**File added**:
- `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`

**Recommendation**:
- `MORE_SCENE_EVIDENCE_REQUIRED`

**Reasoning**:
- The proposal/gate/rollback direction is coherent and stable after M13 plus the pre-M14 audit.
- Current evidence supports a research direction, not a submission-ready scene result.
- Real render-gated MeshPrior insertion is not implemented.
- The scene evidence remains a 200-iteration diagnostic smoke plus synthetic dry-run proposal evidence.

**Required next evidence before strong submission**:
- real scene baseline and gated MeshPrior rows under fixed split;
- scene geometry improvement on COLMAP sparse AbsRel or normal proxy;
- no meaningful render regression;
- controlled triangle/FPS budget;
- car ROI hole/floater reduction;
- safety ablations showing direct prior insertion or gate removal is worse.

**Decision**: M14 gate `PASS`. The next allowed stage is M15 only if we intentionally pursue retrieval-deformation fallback; otherwise the higher-priority engineering milestone is real scene proposal application and render-gated evaluation.

**Linked artefact**:
- `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`

---

## 2026-05-01 — MeshPrior Stage 15 (retrieval-deformation fallback) — PASS

**Outcome**: Implemented and measured a train-only retrieval-deformation fallback for MeshPrior proposals.

**Files added**:
- `ss3dm_prior/meshprior/retrieval_deformation.py`
- `scripts/car_model/meshprior_build_anchor_bank.py`
- `scripts/car_model/meshprior_eval_retrieval_deformation.py`
- `scripts/car_model/smoke_test_meshprior_stage15_retrieval_deformation.py`
- `docs/car_model/meshprior_stage15_retrieval_deformation_design.md`
- `docs/car_model/meshprior_stage15_retrieval_deformation_implementation_report.md`
- `docs/car_model/meshprior_stage15_retrieval_deformation_smoke.md`

**Evaluation outputs**:
- `outputs/carnet/meshprior/retrieval_deformation/stage15_anchor_bank.npz`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/metrics.json`
- `outputs/carnet/meshprior/retrieval_deformation/stage15_eval/summary.md`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- Stage 15 smoke: PASS.
- Train-only anchor bank built from `outputs/carnet/spcarnet/object_index_v1.json`: `32` anchors, `512` points each.
- Retrieval/deformation evaluation rows: `12`.

**Decision**:
- Stage gate: `PASS`.
- Recommendation: `KEEP_AS_BASELINE`.
- Retrieval-only did not beat the Stage 3 posterior proxy on synthetic proposal metrics, so no pivot to retrieval-deformation is recommended.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_stage15_retrieval_deformation_design.md`
- Implementation report: `docs/car_model/meshprior_stage15_retrieval_deformation_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_stage15_retrieval_deformation_smoke.md`

---

## 2026-05-01 — MeshPrior scene application bridge — PASS

**Outcome**: Implemented a safe accepted-proposal application bridge before attempting real scene recovery training.

**Files added**:
- `ss3dm_prior/meshprior/apply_proposals.py`
- `scripts/car_model/meshprior_apply_accepted_proposals.py`
- `scripts/car_model/smoke_test_meshprior_scene_application.py`
- `docs/car_model/meshprior_scene_application_loop_design.md`
- `docs/car_model/meshprior_scene_application_loop_implementation_report.md`
- `docs/car_model/meshprior_scene_application_loop_smoke.md`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- Scene application smoke: PASS.
- Applied existing M11 accepted synthetic fill proposal to a copy.

**Synthetic application result**:
- accepted proposals: `1`
- applied proposals: `1`
- initial mesh: `8` vertices, `10` faces
- final mesh: `9` vertices, `14` faces
- rollback written
- recovery command plan written

**Decision**: bridge gate `PASS`. The next step is real scene proposal application plus recovery optimization, but it requires user confirmation of target scene/model and GPU before launching.

**Linked artefacts**:
- Design: `docs/car_model/meshprior_scene_application_loop_design.md`
- Implementation report: `docs/car_model/meshprior_scene_application_loop_implementation_report.md`
- Smoke report: `docs/car_model/meshprior_scene_application_loop_smoke.md`

---

## 2026-05-01 — Parking phone tiny scene audit and short baseline — PASS

**Outcome**: Audited the parking scene dataset, created a repo-local symlink view, and ran a 200-iteration wandb baseline.

**Files added**:
- `scripts/car_model/meshprior_prepare_parking_scene.py`
- `docs/car_model/meshprior_parking_phone_tiny_scene_audit.md`
- `docs/car_model/meshprior_parking_phone_tiny_baseline_200iter_report.md`

**Dataset view**:
- `outputs/carnet/meshprior/parking_phone_tiny/dataset_view`
- images: `425`
- COLMAP images: `425`
- missing/extra image mismatch: `0`
- segmentation masks: `425`
- ground masks: `425`
- out-of-train split present.

**Training**:
- GPU: `1`
- iterations: `200`
- wandb run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/icjop1fq`
- test PSNR: `11.576681349012587`
- test SSIM: `0.3399546378188663`
- test LPIPS: `0.6316130017792737`
- test FPS: `374.0412913994465`
- triangles: `64497`
- vertices: `193491`
- final cleanup pruned: `0`

**Geometry eval**:
- evaluated test views: `54`
- depth AbsRel: `0.32417137460470213`
- depth MAE: `3.6485552222775537`
- normal mean angle: `51.68797353552561`

**Decision**: parking scene readiness gate `PASS`. This is a short baseline smoke, not a final baseline. Next high-value step is vehicle/ground-aware region mining and gated MeshPrior recovery smoke on this scene, or a longer baseline if a stronger reference is needed first.

**Linked artefacts**:
- Scene audit: `docs/car_model/meshprior_parking_phone_tiny_scene_audit.md`
- Baseline report: `docs/car_model/meshprior_parking_phone_tiny_baseline_200iter_report.md`

---

## 2026-05-01 — Parking phone tiny image/COLMAP region mining — PASS

**Outcome**: Implemented image/COLMAP ROI mining from segmentation masks, ground masks, and COLMAP sparse observations.

**Files added**:
- `scripts/car_model/meshprior_mine_parking_image_regions.py`
- `scripts/car_model/smoke_test_meshprior_parking_image_regions.py`
- `docs/car_model/meshprior_parking_image_region_mining_report.md`

**Full mining output**:
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_regions_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/image_region_mining/image_region_mining_report.md`

**Metrics**:
- images considered: `425`
- candidate regions: `340`
- eligible candidates: `273`
- median sparse point count: `4`
- median mask area fraction: `0.0030437403549382716`
- max eligible ground overlap: `0.25251004016064255`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_image_regions.py`: PASS.

**Decision**: region mining gate `PASS`. The next step is multi-view clustering / 3D consolidation before proposal scoring; these 2D ROI candidates must not directly edit scene geometry.

**Linked artefact**:
- `docs/car_model/meshprior_parking_image_region_mining_report.md`

---

## 2026-05-01 — Parking phone tiny region consolidation — PASS

**Outcome**: Consolidated parking image ROI candidates into coarse multi-view 3D vehicle-region candidates.

**Files added**:
- `scripts/car_model/meshprior_cluster_parking_regions.py`
- `scripts/car_model/smoke_test_meshprior_parking_region_consolidation.py`
- `docs/car_model/meshprior_parking_region_consolidation_report.md`

**Full consolidation output**:
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions.json`
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidated_regions_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/region_consolidation/consolidation_report.md`

**Metrics**:
- input ROI regions: `340`
- sparse-supported eligible inputs used: `140`
- consolidated clusters: `17`
- eligible clusters: `9`
- top cluster support: `32` views and `3851` sparse points.

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_region_consolidation.py`: PASS.

**Decision**: consolidation gate `PASS`. The next step is proposal scoring for the consolidated clusters; no scene geometry has been edited.

**Linked artefact**:
- `docs/car_model/meshprior_parking_region_consolidation_report.md`

---

## 2026-05-01 — Parking phone tiny cluster proposal scoring — PASS

**Outcome**: Converted consolidated parking scene clusters into MeshPrior proposal metadata.

**Files added**:
- `scripts/car_model/meshprior_score_parking_clusters.py`
- `scripts/car_model/smoke_test_meshprior_parking_cluster_scoring.py`
- `docs/car_model/meshprior_parking_cluster_proposal_scoring_report.md`

**Full scoring output**:
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposals.json`
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposal_scores.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/cluster_proposals/proposal_report.md`

**Metrics**:
- eligible clusters scored: `9`
- proposals emitted: `45`
- proposal types: `protect`, `prune`, `snap_candidate`, `fill_candidate`, `uncertainty`
- metadata-only proposals: `45`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_cluster_scoring.py`: PASS.

**Decision**: proposal scoring gate `PASS`. These proposals are not yet geometry edits; every proposal is marked `requires_mesh_extraction` and `requires_scene_gate`.

**Linked artefact**:
- `docs/car_model/meshprior_parking_cluster_proposal_scoring_report.md`

---

## 2026-05-01 — Parking phone tiny metadata proposal gate — PASS

**Outcome**: Gated metadata-only parking proposals into a local mesh-extraction action plan.

**Files added**:
- `scripts/car_model/meshprior_gate_parking_metadata_proposals.py`
- `scripts/car_model/smoke_test_meshprior_parking_metadata_gate.py`
- `docs/car_model/meshprior_parking_metadata_gate_report.md`

**Full gate output**:
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/action_plan.json`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/metadata_gate/metadata_gate_report.md`

**Metrics**:
- proposals evaluated: `45`
- candidate_extract: `24`
- deferred: `17`
- diagnostic: `1`
- rejected: `3`
- mesh extraction targets: `8`
- diagnostic targets: `1`
- geometry edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_metadata_gate.py`: PASS.

**Decision**: metadata gate `PASS`. The next missing bridge is local scene mesh patch extraction with stable face IDs; prune remains deferred until real scene mesh evidence exists.

**Linked artefact**:
- `docs/car_model/meshprior_parking_metadata_gate_report.md`

---

## 2026-05-01 — Parking phone tiny local mesh patch extraction — PASS

**Outcome**: Extracted local mesh patches for metadata-gated parking targets from the trained triangle checkpoint.

**Files added**:
- `scripts/car_model/meshprior_extract_parking_mesh_patches.py`
- `scripts/car_model/smoke_test_meshprior_parking_mesh_patch_extraction.py`
- `docs/car_model/meshprior_parking_mesh_patch_extraction_report.md`

**Full extraction output**:
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/patches/*.npz`

**Metrics**:
- checkpoint vertices: `193491`
- checkpoint triangles: `64497`
- patches extracted: `8`
- nonempty patches: `8`
- total patch faces: `10826`
- patch face range: `97` - `3902`
- geometry edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_mesh_patch_extraction.py`: PASS.

**Decision**: local patch extraction gate `PASS`. The parking pipeline now has real local mesh assets with original face/vertex indices for downstream before/after gates and rollback.

**Linked artefact**:
- `docs/car_model/meshprior_parking_mesh_patch_extraction_report.md`

---

## 2026-05-01 — Parking phone tiny patch no-op/protect gate — PASS

**Outcome**: Ran a no-op/protect readiness gate over extracted parking mesh patches and wrote rollback snapshots.

**Files added**:
- `scripts/car_model/meshprior_gate_parking_mesh_patches.py`
- `scripts/car_model/smoke_test_meshprior_parking_mesh_patch_gate.py`
- `docs/car_model/meshprior_parking_mesh_patch_gate_report.md`

**Full gate output**:
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/patch_gate_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_gate/rollback_snapshots/*.npz`

**Metrics**:
- patches evaluated: `8`
- protect_ready: `8`
- deferred: `0`
- failed: `0`
- rollback snapshots: `8`
- geometry edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_mesh_patch_gate.py`: PASS.

**Decision**: patch no-op/protect gate `PASS`. The parking real-scene bridge now has stable local mesh patches plus rollback snapshots; next step is copied-patch before/after proposal testing.

**Linked artefact**:
- `docs/car_model/meshprior_parking_mesh_patch_gate_report.md`

---

## 2026-05-01 — Parking phone tiny copied-patch proposal tests — SOFT PASS

**Outcome**: Ran copied-patch before/after tests over extracted parking mesh patches.

**Files added**:
- `scripts/car_model/meshprior_test_parking_patch_proposals.py`
- `scripts/car_model/smoke_test_meshprior_parking_patch_proposals.py`
- `docs/car_model/meshprior_parking_patch_proposal_test_report.md`

**Full test output**:
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_results.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/patch_proposal_test_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/proposal_meshes/*/*.npz`
- `outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests/rollback_snapshots/*.npz`

**Metrics**:
- patches tested: `8`
- proposal tests: `24`
- accepted: `8`
- rejected: `16`
- protect_noop_rejected: `8`
- cleanup_accepted: `8`
- floater_rejected: `8`
- source model edited: `false`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_patch_proposals.py`: PASS.

**Decision**: copied-patch proposal test gate `SOFT PASS`. The gate behaves correctly on copied local patches, but accepted cleanup candidates still need checkpoint-copy application and render/geometry validation before they can be treated as scene improvements.

**Linked artefact**:
- `docs/car_model/meshprior_parking_patch_proposal_test_report.md`

---

## 2026-05-01 — Parking phone tiny checkpoint-copy cleanup — SOFT PASS

**Outcome**: Applied accepted copied-patch cleanup candidates to a duplicated parking triangle checkpoint and verified state-array integrity.

**Files added**:
- `scripts/car_model/meshprior_apply_parking_patch_cleanup_to_checkpoint_copy.py`
- `scripts/car_model/smoke_test_meshprior_parking_checkpoint_copy_cleanup.py`
- `docs/car_model/meshprior_parking_checkpoint_copy_cleanup_report.md`

**Full application output**:
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/point_cloud_state_dict.pt`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_rows.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/checkpoint_copy_cleanup/checkpoint_copy_application_report.md`

**Metrics**:
- cleanup applications: `8`
- unique removed faces: `532`
- faces: `64497` -> `63965`
- vertices: `193491` -> `191895`
- source model edited: `false`
- checkpoint copy edited: `true`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_checkpoint_copy_cleanup.py`: PASS.

**Decision**: checkpoint-copy cleanup gate `SOFT PASS`. Writeback bookkeeping is valid, but render/geometry validation is still pending before this can be claimed as a scene improvement.

**Linked artefact**:
- `docs/car_model/meshprior_parking_checkpoint_copy_cleanup_report.md`

---

## 2026-05-01 — Parking phone tiny recovery model geometry eval — SOFT PASS

**Outcome**: Wrapped the cleaned checkpoint copy in a loadable recovery model directory and ran COLMAP sparse geometry evaluation.

**Files added**:
- `scripts/car_model/meshprior_prepare_parking_recovery_model.py`
- `scripts/car_model/smoke_test_meshprior_parking_recovery_model.py`
- `docs/car_model/meshprior_parking_recovery_model_eval_report.md`

**Recovery model output**:
- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/point_cloud/iteration_200/point_cloud_state_dict.pt`
- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/meshprior_recovery_model_report.json`
- `outputs/carnet/meshprior/parking_phone_tiny/recovery_model_cleanup/geometry_eval_colmap/iter_200.json`

**Metrics**:
- recovery triangles: `63965`
- recovery vertices: `191895`
- evaluated views: `54`
- depth count: `21910`
- depth AbsRel baseline -> recovery: `0.32417137460470213` -> `0.3241717166185642`
- normal mean angle baseline -> recovery: `51.68797353552561` -> `51.6880043093792`

**Verification**:
- Compileall over `scripts/car_model` and `ss3dm_prior`: PASS.
- `smoke_test_meshprior_parking_recovery_model.py`: PASS.
- GPU1 COLMAP geometry eval: PASS.

**Decision**: recovery model eval gate `SOFT PASS`. The cleanup checkpoint copy is loadable and geometry-proxy stable, but its metric deltas are neutral; do not claim improvement before render-metric validation or a short resumed training run.

**Linked artefact**:
- `docs/car_model/meshprior_parking_recovery_model_eval_report.md`

---

## 2026-05-01 — Parking phone tiny render metric comparison — SOFT PASS

**Outcome**: Rendered and evaluated the recovery cleanup model and the current engineering baseline with the same `render.py` + `metrics.py` pipeline.

**Important baseline clarification**:
- `parking_phone_tiny/baseline_200iter` is an engineering baseline: current repository, no MeshPrior proposal application, short 200-iteration run.
- The paper baseline should be original/clean Mesh Splatting on the same data, budget, and evaluation scripts.

**Metrics**:
- engineering baseline SSIM / PSNR / LPIPS: `0.2898596525` / `10.9499864578` / `0.6441746354`
- recovery cleanup SSIM / PSNR / LPIPS: `0.2898600996` / `10.9499950409` / `0.6441848874`
- deltas: SSIM `+0.0000004470`, PSNR `+0.0000085831`, LPIPS `+0.0000102520`

**Decision**: render comparison gate `SOFT PASS`. The cleanup checkpoint copy is render-stable but not meaningfully better. This supports stability, not a final improvement claim.

**Comparison collector**:
- Added `scripts/car_model/meshprior_collect_parking_comparison.py`
- Added `scripts/car_model/smoke_test_meshprior_parking_comparison.py`
- Output: `outputs/carnet/meshprior/parking_phone_tiny/comparison_summary/parking_comparison_summary.{json,csv,md}`
- Collector decision: `SOFT_PASS_STABILITY_ONLY`
- Paper baseline status: `MISSING`

**Linked artefact**:
- `docs/car_model/meshprior_parking_render_metric_comparison.md`

---

## 2026-05-01 — Parking phone tiny origin/main baseline — SOFT PASS

**Outcome**: Created a separate `/tmp/mesh-splatting-origin-main` worktree at `origin/main@1a714f3` and ran clean Mesh Splatting baseline candidates.

**User-corrected baseline framing**:
- 200-iteration results are smoke/stability evidence only.
- The paper baseline should be clean/original Mesh Splatting under the same dataset and budget.

**Runs**:
- origin/main 200 iter: completed; post-render PSNR `5.8725734`, SSIM `0.0092272`, LPIPS `0.7112017`.
- origin/main 2000 iter: completed; training internal test PSNR `16.46195650100708`, SSIM `0.4846517714085402`, LPIPS `0.5333475658187159`.
- origin/main 2000 post-render metrics: PSNR `11.047659873962402`, SSIM `0.21993064880371094`, LPIPS `0.6417058110237122`, triangles `39079`, vertices `58458`.

**W&B**:
- origin/main has no current-branch `--enable_wandb` integration.
- Added `scripts/car_model/meshprior_log_parking_run_to_wandb.py` for external summary logging.
- Logged run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/142memiw`

**Decision**: origin/main baseline gate `SOFT PASS`. The true baseline path is now concrete and W&B-recorded, but fair medium comparisons require current-branch 2000-iteration engineering and MeshPrior variants with training-time W&B enabled.

**Linked artefact**:
- `docs/car_model/meshprior_parking_origin_main_baseline_report.md`

---

## 2026-05-01 — Parking phone tiny medium 2000-iteration baseline comparison — SOFT PASS

**Outcome**: Completed a medium-budget comparison between the clean `origin/main@1a714f3` Mesh Splatting candidate and the current `clean-submit` engineering branch on `parking_phone_tiny`.

**W&B correction**:
- Current branch 2000 iter used training-time online W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nk2w04wn`
- Clean `origin/main` lacks current W&B flags and was externally logged: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/142memiw`
- Future current-branch training runs must use training-time W&B by default.

**Training internal test metrics at 2000**:
- origin/main PSNR / SSIM / LPIPS / FPS: `16.4619565010` / `0.4846517714` / `0.5333475658` / `271.3129810583`
- current branch PSNR / SSIM / LPIPS / FPS: `16.4415020589` / `0.4834401826` / `0.5322314313` / `257.5665033592`

**Post-render metrics at 2000**:
- origin/main PSNR / SSIM / LPIPS: `11.0476598740` / `0.2199306488` / `0.6417058110`
- current branch PSNR / SSIM / LPIPS: `11.5994377136` / `0.2702677548` / `0.6347319484`
- origin/main topology: `39079` triangles, `58458` vertices
- current branch topology: `782982` triangles, `820107` vertices

**COLMAP geometry proxy**:
- origin/main depth MAE / AbsRel: `13.7902993339` / `5.6119052058`
- current branch depth MAE / AbsRel: `4.4141606252` / `0.4278796566`
- origin/main normal mean angle: `52.1989385790`
- current branch normal mean angle: `52.5651849634`

**Decision**: medium baseline gate `SOFT PASS`. The current branch is better on post-render metrics and sparse depth proxy, but uses much more topology and is not yet a MeshPrior proposal-applied 2000-iteration variant. Do not make a paper-level improvement claim from this alone.

**Linked artefact**:
- `docs/car_model/meshprior_parking_medium_baseline_2000iter_report.md`

---

## 2026-05-01 — Stage17 real MeshPrior 2000-iteration variant — PASS / CLAIM SOFT

**Outcome**: Built and evaluated the first real MeshPrior scene-training variant on `parking_phone_tiny`. The run starts from a MeshPrior-cleaned copied checkpoint at iteration `200` and resumes current-branch training to iteration `2000`.

**W&B**:
- smoke resumed-training run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/y4432er1`
- Stage17 2000-iteration run: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vyrun0qo`

**Proposal / gate inputs**:
- accepted cleanup proposals: `8`
- rejected no-op proposals: `8`
- rejected floater proposals: `8`
- source model edited: `false`

**Training internal test metrics**:
- iteration 300 smoke PSNR / SSIM / LPIPS: `11.5936053771` / `0.3349873807` / `0.6415096864`
- iteration 1000 PSNR / SSIM / LPIPS: `13.1176037435` / `0.3794519540` / `0.6071134640`
- iteration 2000 PSNR / SSIM / LPIPS / FPS: `13.4438069308` / `0.3471139595` / `0.6021583963` / `272.8530837309`

**Post-render metrics at 2000**:
- Stage17 PSNR / SSIM / LPIPS: `13.2782726288` / `0.3039793670` / `0.6076099277`
- current branch PSNR / SSIM / LPIPS: `11.5994377136` / `0.2702677548` / `0.6347319484`
- origin/main PSNR / SSIM / LPIPS: `11.0476598740` / `0.2199306488` / `0.6417058110`

**COLMAP geometry proxy**:
- Stage17 depth MAE / AbsRel: `3.8259249166` / `0.3666914408`
- current branch depth MAE / AbsRel: `4.4141606252` / `0.4278796566`
- Stage17 normal mean angle: `52.1695839576`
- current branch normal mean angle: `52.5651849634`

**Topology and cleanup**:
- Stage17 triangles / vertices: `777251` / `816498`
- final cleanup enabled: `false`
- final cleanup pruned: `0`

**Decision**: Stage17 execution gate `PASS`; claim status `SOFT`. The first real MeshPrior training variant is implemented, W&B-logged, and metric-positive on this scene, but topology remains very large. M18 topology-budget comparison is mandatory before any paper-level improvement claim.

**Linked artefacts**:
- `docs/car_model/meshprior_stage17_real_variant_design.md`
- `docs/car_model/meshprior_stage17_real_variant_smoke.md`
- `docs/car_model/meshprior_stage17_real_variant_implementation_report.md`

---

## 2026-05-01 — Stage18 topology-budget comparison — PASS / CLAIM BLOCKED

**Outcome**: Added a reproducible topology-budget collector for the three 2000-iteration parking runs.

**Files**:
- `scripts/car_model/meshprior_collect_topology_budget_comparison.py`
- `scripts/car_model/smoke_test_meshprior_topology_budget_comparison.py`
- `docs/car_model/meshprior_stage18_topology_budget_design.md`
- `docs/car_model/meshprior_stage18_topology_budget_implementation_report.md`

**Output**:
- `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/topology_budget_comparison.json`
- `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/topology_budget_comparison.csv`
- `outputs/carnet/meshprior/parking_phone_tiny/topology_budget_comparison/topology_budget_comparison.md`

**Main table**:
- origin/main: PSNR `11.047660`, SSIM `0.219931`, LPIPS `0.641706`, triangles `39079`, PSNR/100k tri `28.270068`, AbsRel `5.611905`, FPS `271.313`
- current branch: PSNR `11.599438`, SSIM `0.270268`, LPIPS `0.634732`, triangles `782982`, PSNR/100k tri `1.481444`, AbsRel `0.427880`, FPS `257.567`
- Stage17 MeshPrior: PSNR `13.278273`, SSIM `0.303979`, LPIPS `0.607610`, triangles `777251`, PSNR/100k tri `1.708364`, AbsRel `0.366691`, FPS `272.853`

**Decision**: M18 gate `PASS`; collector decision `QUALITY_GAIN_NOT_TOPOLOGY_NORMALIZED`. Stage17 improves quality metrics versus current branch, but has `19.889x` the clean candidate triangle count. Stronger claims remain blocked until topology-control or budget-matched reporting is complete.

**Linked artefact**:
- `docs/car_model/meshprior_stage18_topology_budget_implementation_report.md`

---

## 2026-05-01 — Stage19 clean MeshSplatting baseline audit — PASS

**Outcome**: Confirmed that the clean baseline commit used for `origin_main_2000iter` matches the official MeshSplatting repository.

**Remote evidence**:
- official remote checked: `https://github.com/meshsplatting/mesh-splatting.git`
- official `HEAD` / `main`: `1a714f33dd758a42be8fa86e1041c3c67df0d0a8`
- local `origin/main`: `1a714f33dd758a42be8fa86e1041c3c67df0d0a8`

**Decision**: M19 gate `PASS`. `origin/main@1a714f3` is a valid clean MeshSplatting medium-budget baseline for the current parking experiments.

**Caveat**: This validates code lineage, not final experimental sufficiency. The baseline remains single-scene and 2000-iteration; long-budget and multi-scene evidence are still required for strong paper claims.

**Linked artefact**:
- `docs/car_model/meshprior_stage19_clean_baseline_audit.md`

---

## 2026-05-01 — Stage20 second scene audit — STOP

**Outcome**: Audited parent-directory data for a second real MeshPrior scene.

**Findings**:
- `/data/peilincai/parking_phone_tiny_anonymized`: valid current parking scene, already used.
- `/data/peilincai/car_models`: object mesh data, not a COLMAP scene.
- `/data/peilincai/vggt`: contains example image/sparse data, but not a supplied parking-lot / vehicle-rich target scene.
- No second suitable parking-lot COLMAP/image scene was found under `/data/peilincai` at this audit depth.

**Decision**: M20 gate `STOP`. This is a data availability stop, not a code failure. Multi-scene validation remains blocked until a second vehicle/parking COLMAP scene is added.

**Linked artefacts**:
- `docs/car_model/meshprior_stage20_second_scene_design.md`
- `docs/car_model/meshprior_stage20_second_scene_audit.md`
- `docs/car_model/meshprior_stage20_second_scene_implementation_report.md`

---

## 2026-05-02 — Stage21 7000-iteration long-budget single-scene diagnostic — PASS / NEGATIVE METHOD RESULT

**Outcome**: Completed the aligned 7000-iteration diagnostic requested after M20 stopped on second-scene availability. All three rows finished with checkpoints, W&B records, independent `render.py + metrics.py`, COLMAP proxy geometry evaluation, and topology counts.

**W&B**:
- clean `origin/main@1a714f3` external log: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/yiwb4d2n`
- current branch training-time W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/l5buxl3m`
- Stage17 MeshPrior resume training-time W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w3kczubb`

**Independent render metrics at 7000**:
- clean `origin/main`: PSNR `16.134155`, SSIM `0.452130`, LPIPS `0.499124`, triangles `285187`
- current branch: PSNR `17.204679`, SSIM `0.535045`, LPIPS `0.450750`, triangles `833775`
- Stage17 MeshPrior resume: PSNR `10.839708`, SSIM `0.285366`, LPIPS `0.662528`, triangles `838883`

**COLMAP proxy geometry at 7000**:
- clean `origin/main`: depth AbsRel `0.084499`, normal mean angle `45.300650`
- current branch: depth AbsRel `0.076126`, normal mean angle `45.561976`
- Stage17 MeshPrior resume: depth AbsRel `0.744099`, normal mean angle `52.580674`

**Decision**: M21 execution gate `PASS`, but the Stage17 MeshPrior resume variant is rejected as a long-budget method candidate. It improved the 2000-iteration diagnostic but collapses by 7000 iterations. Current branch is the best long-budget single-scene row, but its quality gain over clean MeshSplatting is not topology-normalized because it uses about `2.92x` more triangles.

**Next priority**: topology control or scheduled cleanup on the current branch before M22 paper-evidence packaging. Do not launch a longer Stage17 MeshPrior resume sweep.

**Linked artefacts**:
- `docs/car_model/meshprior_stage21_long_budget_design.md`
- `docs/car_model/meshprior_stage21_long_budget_report.md`

---

## 2026-05-02 — Stage21.5 topology-controlled current-branch ablation — PASS

**Outcome**: Added a post-training checkpoint-copy topology-control diagnostic for the current-branch 7000 checkpoint. The ablation prunes smallest-area triangles without editing the original checkpoint, then evaluates each copied model with independent render metrics, COLMAP proxy geometry, topology counts, and external W&B summary logs.

**W&B**:
- `prune_25`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/evid1gbt`
- `prune_50`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/w1ix6e9a`
- `prune_66`: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/xzfqwpgi`

**Independent render / geometry metrics at 7000**:
- clean `origin/main`: PSNR `16.134155`, SSIM `0.452130`, LPIPS `0.499124`, triangles `285187`, depth AbsRel `0.084499`
- current branch: PSNR `17.204679`, SSIM `0.535045`, LPIPS `0.450750`, triangles `833775`, depth AbsRel `0.076126`
- `prune_50`: PSNR `17.051889`, SSIM `0.523914`, LPIPS `0.465400`, triangles `416888`, depth AbsRel `0.083265`
- `prune_66`: PSNR `16.429369`, SSIM `0.492480`, LPIPS `0.489681`, triangles `283484`, depth AbsRel `0.099246`

**Decision**: M21.5 gate `PASS`. Use `prune_50` as the topology-controlled current-branch row in M22 because it keeps all render metrics above clean while reducing current topology by `50%` and keeping depth AbsRel close to clean. Keep `prune_66` as a high-compression Pareto endpoint. This is still a diagnostic post-hoc ablation, not integrated optimization-time topology control.

**Linked artefacts**:
- `docs/car_model/meshprior_stage21_5_topology_control_design.md`
- `docs/car_model/meshprior_stage21_5_topology_control_implementation_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/stage21_5_topology_control/comparison/topology_control_ablation.md`

---

## 2026-05-02 — Stage22 unified paper evidence package — SOFT PASS

**Outcome**: Added a reproducible collector and smoke test that consolidate local MeshPrior evidence into separated paper-style metric classes. Missing rows remain explicit instead of being filtered from headline tables.

**Files**:
- `scripts/car_model/meshprior_collect_paper_evidence.py`
- `scripts/car_model/smoke_test_meshprior_paper_evidence.py`
- `docs/car_model/meshprior_stage22_paper_evidence_design.md`
- `docs/car_model/meshprior_stage22_paper_evidence_report.md`

**Output**:
- `outputs/carnet/meshprior/paper_evidence/paper_evidence.json`
- `outputs/carnet/meshprior/paper_evidence/scene_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/object_prior_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/synthetic_damage_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/proposal_gate_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/failure_case_rows.csv`
- `outputs/carnet/meshprior/paper_evidence/missing_rows.csv`

**Main scene rows**:
- clean `origin/main` 7000: PSNR `16.134155`, SSIM `0.452130`, LPIPS `0.499124`, triangles `285187`, depth AbsRel `0.084499`
- current branch `prune_50` 7000: PSNR `17.051889`, SSIM `0.523914`, LPIPS `0.465400`, triangles `416888`, depth AbsRel `0.083265`
- Stage17 MeshPrior resume 7000: PSNR `10.839708`, SSIM `0.285366`, LPIPS `0.662528`, triangles `838883`, depth AbsRel `0.744099`

**Missing rows kept visible**:
- second real scene
- integrated optimization-time topology control
- render-gated full MeshPrior insertion

**Verification**:
- `scripts/car_model/smoke_test_meshprior_paper_evidence.py`: PASS
- `python -m compileall scripts/car_model ss3dm_prior -q`: PASS
- `git diff --check`: PASS

**Decision**: M22 gate `SOFT PASS`. The paper-evidence package is reproducible and metric-separated, but remains under-evidenced for a strong method claim because multi-scene validation and integrated topology control are still missing. The next prompt should be M23 claim-risk audit, not more Stage17 training.

---

## 2026-05-02 — Stage23 claim-risk audit and paper decision — PASS

**Outcome**: Completed the post-M22 claim-risk audit and updated the NeurIPS roadmap.

**Decision**: strongest defensible story is `CLAIM_CONSERVATIVE_FRAMEWORK_NOT_FULL_METHOD`.

**Supported claims**:
- Stage 3 posterior is a strong object prior for this codebase.
- Proposal gates and rollback reject obvious unsafe copied-patch edits.
- Current branch and M21.5 `prune_50` provide a topology-aware single-scene diagnostic that beats clean MeshSplatting render metrics.

**Refuted / unsafe claims**:
- Stage17 MeshPrior resume is not a viable long-budget method candidate.
- Full MeshPrior scene optimization improvement is unsafe to claim.
- Multi-scene generalization is unsafe to claim until a second valid scene exists.

**Next high-value paths**:
- add a second real vehicle/parking COLMAP scene and rerun the evidence package;
- or integrate M21.5 topology control into the training/optimization loop with render/geometry gates and rollback.

**Linked artefacts**:
- `docs/car_model/meshprior_stage23_claim_risk_audit.md`
- `docs/car_model/MeshPrior_NeurIPS_paper_roadmap.md`

---

## 2026-05-02 — Stage23.5 integrated topology-control smoke — PASS

**Outcome**: Moved topology-control validation from post-hoc checkpoint-copy pruning toward the training loop. The successful trigger run committed one PRISM candidate prune during optimization, wrote rollback/accounting metadata, kept final cleanup disabled, and passed independent render, COLMAP proxy geometry, and collector checks.

**Task clarification**: the current paper setting is posed multi-view images plus COLMAP/camera geometry plus Mesh Splatting scene mesh optimization. It is not a radar-only mesh reconstruction pipeline.

**W&B**:
- protected 800-iteration debug: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/5ekk5gjz`
- protected 350-iteration debug: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/esyvtvwn`
- successful 180-iteration trigger: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/an7l2ec0`

**Successful trigger metrics**:
- PRISM commit: iteration `141`, candidate prune, `64497 -> 63208` triangles, rollback `0`
- independent render metrics at iteration `180`: PSNR `10.790648`, SSIM `0.284250`, LPIPS `0.645548`
- COLMAP proxy geometry: depth AbsRel `0.327274`, normal mean angle `51.771524`
- final cleanup: disabled and not executed
- collector gate: `PASS`

**Decision**: M23.5 is a mechanism PASS, not a paper-quality row. The default PRISM protection rules are too conservative for short early smokes, while the fully relaxed trigger is useful for debugging but not final. Next priority is a tuned medium integrated-topology run with online W&B and topology-aware comparison.

**Linked artefacts**:
- `docs/car_model/meshprior_stage23_5_integrated_topology_design.md`
- `docs/car_model/meshprior_stage23_5_integrated_topology_implementation_report.md`
- `outputs/carnet/meshprior/parking_phone_tiny/stage23_5_integrated_topology/prism_unprotected_trigger_180iter/summary/stage23_5_integrated_topology_summary.md`

---

## 2026-05-02 — Stage23.6 tuned medium integrated topology control — PASS

**Outcome**: Ran tuned 2000-iteration integrated PRISM topology control. The first `tuned_medium_2000iter` attempt showed that `orientation_keep=1.0` protected all triangles under threshold `0.85`. The useful `tuned_medium_v2_2000iter` run set `--prism_keep_orientation_threshold 1.1`, committed two counterfactual-accepted PRISM candidate edits, and passed collector checks.

**W&B**:
- v1 diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3209wi9z`
- v2 useful row: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/j0c8zwkx`

**V2 metrics**:
- PRISM commits: `551` (`64497 -> 63853`) and `922` (`63853 -> 63215`)
- independent render: PSNR `12.046110`, SSIM `0.286099`, LPIPS `0.629034`
- COLMAP proxy: depth AbsRel `0.393866`, normal mean angle `51.945426`
- collector gate: `PASS`

**Decision**: Stage23.6 is a medium-budget mechanism `PASS`. It validates tuned training-time PRISM commits, but does not provide a final long-budget paper claim.

**Linked artefacts**:
- `docs/car_model/meshprior_stage23_6_tuned_integrated_topology_design.md`
- `docs/car_model/meshprior_stage23_6_tuned_integrated_topology_report.md`

---

## 2026-05-02 — Stage24 full integrated topology control — PASS

**Outcome**: Ran three 7000-iteration M24 variants with online W&B and full post-evaluation. M24-v1 proved that early/repeated PRISM rounds can over-freeze standard densification and hurt quality. M24-v2 delayed PRISM and rejected aggressive 5% edits through the counterfactual gate. M24-v3 used late 1% PRISM edits and became the first full-budget integrated row with committed training-time topology edits.

**W&B**:
- v1 early PRISM: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/7i6n8jfj`
- v2 late 5% reject: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ytex9896`
- v3 late 1% commit: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/e92jwttk`

**M24-v3 metrics**:
- PRISM decisions: commit at `6151` (`612458 -> 606334`), commit at `6272` (`606334 -> 600271`), reject at `6393` and `6394`
- independent render: PSNR `17.042757`, SSIM `0.529476`, LPIPS `0.454884`
- COLMAP proxy: depth AbsRel `0.082815`, normal mean angle `43.394721`
- final topology: `823651` triangles, `1058219` vertices
- collector gate: `PASS`

**Comparison**:
- current branch 7000: PSNR `17.204679`, SSIM `0.535045`, LPIPS `0.450750`, depth AbsRel `0.076126`, normal `45.561976`, triangles `833775`
- M21.5 `prune_50`: PSNR `17.051889`, SSIM `0.523914`, LPIPS `0.465400`, depth AbsRel `0.083265`, normal `45.825681`, triangles `416888`
- M24-v3 preserves near-current render quality and improves normal proxy geometry, but topology reduction is still small compared with posthoc M21.5.

**Decision**: Stage24 is a real integrated optimization-time topology-control `PASS`, not the final paper headline. The next technical prompt should be M24.1 late-PRISM Pareto sweep, plus second-scene data as soon as available.

**Linked artefacts**:
- `docs/car_model/meshprior_stage24_full_integrated_topology_design.md`
- `docs/car_model/meshprior_stage24_full_integrated_topology_report.md`

---

## 2026-05-02 — Stage24.1 late-PRISM Pareto sweep — PASS

**Outcome**: Ran three late-PRISM 7000-iteration Pareto rows with online W&B and full post-evaluation. M24.1 produced the strongest integrated topology-control row so far: `pareto_ratio0p005_rounds8_retryfix_7000iter` commits five late candidate edits and ends at `723438` triangles, below both current branch 7000 (`833775`) and M24-v3 (`823651`) while keeping similar independent render and better normal proxy geometry.

**W&B**:
- 0.5% legacy no-candidate diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bqc4w18e`
- 0.5% retryfix best topology row: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jnn9yauw`
- 1% retryfix throttle row: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/0n7kzim5`

**Best M24.1 metrics**:
- run: `pareto_ratio0p005_rounds8_retryfix_7000iter`
- PRISM decisions: `5` effective rounds, `445` no-candidate retry events, `5` commits
- independent render: PSNR `16.967005`, SSIM `0.530894`, LPIPS `0.465932`
- COLMAP proxy: depth AbsRel `0.082264`, normal mean angle `42.667905`
- final topology: `723438` triangles, `904493` vertices
- collector gate: `PASS`

**Comparison**:
- current branch 7000: PSNR `17.204679`, SSIM `0.535045`, LPIPS `0.450750`, depth AbsRel `0.076126`, normal `45.561976`, triangles `833775`
- M24-v3: PSNR `17.042757`, SSIM `0.529476`, LPIPS `0.454884`, depth AbsRel `0.082815`, normal `43.394721`, triangles `823651`
- M21.5 `prune_50`: PSNR `17.051889`, SSIM `0.523914`, LPIPS `0.465400`, depth AbsRel `0.083265`, normal `45.825681`, triangles `416888`

**Code finding**: no-candidate attempts were previously able to consume candidate rounds. The controller now records them as retry events, does not spend an effective candidate round, and throttles retry attempts with `prism_no_candidate_retry_iters`.

**Decision**: M24.1 is an integrated topology-control `PASS`, but still not a final paper headline. The next prompt is M24.2 topology retention, because late densification can partially undo accepted PRISM topology edits.

**Linked artefacts**:
- `docs/car_model/meshprior_stage24_1_late_prism_pareto_design.md`
- `docs/car_model/meshprior_stage24_1_late_prism_pareto_report.md`

---

## 2026-05-02 — Stage24.2 topology retention — PASS

**Outcome**: Added an opt-in schedule flag, `--prism_freeze_densification_after_first_commit`, and ran a 7000-iteration topology-retention row. This is the strongest result so far: final topology drops to `254491` triangles while independent render and COLMAP proxy metrics improve over current branch, M21.5 `prune_50`, M24-v3, and M24.1 best on the available single scene.

**W&B**:
- freeze after first commit: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/vsv2bs79`

**Metrics**:
- PRISM decisions: `8` effective rounds, `27` no-candidate retry events, `2` commits, `6` rollback-protected rejects
- independent render: PSNR `17.314823`, SSIM `0.559230`, LPIPS `0.442099`
- COLMAP proxy: depth AbsRel `0.078840`, normal mean angle `41.010093`
- final topology: `254491` triangles, `463687` vertices
- collector gate: `PASS`

**Comparison**:
- M21.5 `prune_50`: PSNR `17.051889`, SSIM `0.523914`, LPIPS `0.465400`, depth AbsRel `0.083265`, normal `45.825681`, triangles `416888`
- M24.1 best: PSNR `16.967005`, SSIM `0.530894`, LPIPS `0.465932`, depth AbsRel `0.082264`, normal `42.667905`, triangles `723438`
- M24.2 improves both topology and metrics on this scene.

**Decision**: M24.2 upgrades the project from a mechanism proof to a plausible single-scene method result. Remaining NeurIPS-level risk is now generality and evidence quality: the next stage should be M25 multi-scene validation plus paper-grade visual/failure analysis.

**Linked artefacts**:
- `docs/car_model/meshprior_stage24_2_topology_retention_design.md`
- `docs/car_model/meshprior_stage24_2_topology_retention_report.md`

---

## 2026-05-02 — MeshSplatOpt R14.21-R14.22 freeze-densify recovery — PASS

**Outcome**: Added recovery-time densification overrides and an opt-in `--skip_restricted_delaunay` train flag. The first freeze-only diagnostic run stalled at the delayed Delaunay refresh, which established that topology-retention recovery must disable that refresh when `densify_until_iter` is pinned to the loaded checkpoint. The successful W&B rows use `--densify_until_iter 2000 --skip_restricted_delaunay`.

**W&B**:
- aborted freeze-only diagnostic: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/gpqeybmc`
- baseline freeze: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/qdwbbpob`
- snap freeze: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/srdr58z6`

**Best medium result**:
- row: R14.22 snap freeze, `bonsai` 2000->4000
- independent render: PSNR `17.437725`, SSIM `0.433732`, LPIPS `0.506797`
- COLMAP proxy: depth AbsRel `0.272852`, depth MAE `2.893086`, normal mean angle `43.570729`
- final topology: `2487474` triangles, `2478890` vertices

**Comparison to R14.20 unfrozen medium baseline**:
- triangles: `5090601 -> 2487474` (`-51.135946%`)
- PSNR: `15.834701 -> 17.437725`
- SSIM: `0.334698 -> 0.433732`
- LPIPS: `0.571493 -> 0.506797`
- depth AbsRel: `0.405141 -> 0.272852`
- normal mean angle: `48.119439 -> 43.570729`

**Decision**: R14.21-R14.22 is a topology-retention `PASS`. It does not yet make the snap selector itself a strong standalone method claim, because snap-vs-freeze-baseline deltas are small and mixed. It does justify a full or multi-scene R15 schedule using freeze-densify plus skip-Delaunay as the default recovery policy.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR14_21_22_freeze_densify_recovery_control_report.md`

---

## 2026-05-03 — MeshSplatOpt R15.01-R15.04 multi-scene freeze medium — PASS

**Outcome**: Extended the freeze-densify/skip-Delaunay recovery schedule to `courtyard` and `parking_phone_tiny`, with online W&B and full render/geometry evaluation. Together with the previous `bonsai` rows, the schedule now has three-scene medium-budget support. The current `SNAP_VERTICES` area-outlier selector remains weak under equal schedule controls.

**W&B**:
- courtyard baseline freeze: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/cvf6t7do`
- courtyard snap freeze: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/d3h2ruj3`
- parking baseline freeze: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/evj36lvp`
- parking snap freeze: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/3r7inkj0`

**Key schedule gains**:
- `courtyard` baseline 2000 -> freeze 4000: PSNR `14.946162 -> 17.819637`, SSIM `0.438775 -> 0.578303`, LPIPS `0.592443 -> 0.460392`, AbsRel `0.354800 -> 0.243054`, topology unchanged at `410254` triangles.
- `parking_phone_tiny` baseline 2000 -> freeze 4000: PSNR `11.599438 -> 14.251087`, SSIM `0.270268 -> 0.383800`, LPIPS `0.634732 -> 0.569749`, AbsRel `0.427880 -> 0.324794`, topology unchanged at `782982` triangles.

**Selector finding**: snap-freeze is slightly negative versus baseline-freeze on `courtyard` and `parking_phone_tiny`; it remains only a safe edit materialization path, not a performance selector.

**Decision**: R15 is now a genuine multi-scene schedule `PASS`. The next high-value work is a full-budget freeze run and a stronger edit/proposal selector.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR15_01_04_multiscene_freeze_medium_report.md`

---

## 2026-05-03 — MeshSplatOpt R16.01 courtyard full freeze — PASS

**Outcome**: Ran the freeze-densify/skip-Delaunay schedule from `courtyard` iteration 2000 to 7000 with online W&B. The full-budget row preserves topology exactly and improves beyond the R15.01 medium row on render and depth proxy metrics.

**W&B**:
- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/z2i5ndyu`

**Metrics**:
- topology: `410254` triangles, `444301` vertices, unchanged from the loaded 2000 checkpoint
- independent render: PSNR `18.321131`, SSIM `0.594281`, LPIPS `0.440022`
- COLMAP proxy: depth AbsRel `0.171453`, depth MAE `2.067510`, normal mean angle `37.575696`

**Comparison**:
- baseline 2000: PSNR `14.946162`, SSIM `0.438775`, LPIPS `0.592443`, AbsRel `0.354800`, normal `35.324712`
- R15.01 medium 4000: PSNR `17.819637`, SSIM `0.578303`, LPIPS `0.460392`, AbsRel `0.243054`, normal `37.967884`
- R16.01 improves over medium without topology growth; normal remains worse than baseline and should be handled as an explicit limitation.

**Decision**: R16.01 is a full-budget schedule `PASS` on one public scene. The next full row should be `bonsai` or `parking_phone_tiny`, and the next method improvement should add a stronger selector or normal-aware recovery.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR16_01_courtyard_full_freeze_report.md`

---

## 2026-05-03 — MeshSplatOpt R16.02 bonsai full freeze — PASS

**Outcome**: Ran the same full-budget freeze-densify/skip-Delaunay schedule on `bonsai` from iteration 2000 to 7000. This gives two public full-budget rows: `courtyard` and `bonsai`.

**W&B**:
- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/nsj76h7d`

**Metrics**:
- topology: `2487474` triangles, `2478890` vertices, unchanged from the loaded 2000 checkpoint
- independent render: PSNR `18.303303`, SSIM `0.455556`, LPIPS `0.490660`
- COLMAP proxy: depth AbsRel `0.220888`, depth MAE `2.392198`, normal mean angle `41.233611`

**Comparison**:
- baseline 2000: PSNR `12.201612`, SSIM `0.207315`, LPIPS `0.624259`, AbsRel `0.495874`, normal `50.118301`
- freeze medium 4000: PSNR `17.429750`, SSIM `0.432352`, LPIPS `0.506490`, AbsRel `0.271062`, normal `43.347689`
- full freeze improves over both while preserving topology exactly.

**Decision**: R16 is now a two-scene full-budget schedule `PASS`. The next method gap is selector strength, not schedule validation.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR16_01_02_two_scene_full_freeze_report.md`

---

## 2026-05-03 — MeshSplatOpt R16.03 parking full freeze — PASS

**Outcome**: Completed the third full-budget freeze-densify/skip-Delaunay row on `parking_phone_tiny`, again with online W&B and exact topology preservation. R16 is now a three-scene full-budget validation set.

**W&B**:
- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/dq8urgr7`

**Metrics**:
- topology: `782982` triangles, `820107` vertices, unchanged from the loaded 2000 checkpoint
- independent render: PSNR `15.570565`, SSIM `0.448212`, LPIPS `0.528052`
- COLMAP proxy: depth AbsRel `0.257815`, depth MAE `3.085023`, normal mean angle `49.789749`

**Comparison**:
- baseline 2000: PSNR `11.599438`, SSIM `0.270268`, LPIPS `0.634732`, AbsRel `0.427880`, normal `52.565185`
- freeze medium 4000: PSNR `14.251087`, SSIM `0.383800`, LPIPS `0.569749`, AbsRel `0.324794`, normal `51.043451`
- full freeze improves over both on render, depth, and sparse-normal proxy metrics while preserving topology exactly.

**Decision**: R16 is now a three-scene full-budget schedule `PASS`. The method claim should center on topology-retained recovery/continuation; the next critical gap is a stronger selector or normal-aware recovery term.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR16_01_03_three_scene_full_freeze_report.md`

---

## 2026-05-03 — MeshSplatOpt R17.01 CSEF local snap selector — PASS

**Outcome**: Strengthened the weak `SNAP_VERTICES` selector by replacing global-plane snap targets with local neighbor plane targets and adding explicit CSEF-style evidence/risk metadata plus negative free-space rejection.

**Implementation**:
- `ss3dm_prior/meshsplatopt/snap_proposals.py`
- `scripts/car_model/smoke_test_meshsplatopt_stageR7_snap.py`

**Smoke**:
- compileall over `scripts/car_model`, `ss3dm_prior`, and `utils`: `PASS`
- dent plane error: `0.03072 -> 0.019831720797113993`
- misalignment plane error: `0.019200000000000002 -> 0.0096`
- unsupported floater rejected: `true`
- negative free-space snap rejected: `true`
- rollback exact: `true`

**Decision**: This is a selector-quality `PASS`, not yet a real-scene performance claim. The next gate should generate a real-checkpoint CSEF-local snap proposal, run the render-backed counterfactual gate, and only then launch W&B recovery.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR17_01_csef_local_snap_report.md`

---

## 2026-05-03 — MeshSplatOpt R17.02 real checkpoint local snap gate — PASS

**Outcome**: Added a real-checkpoint local snap selector and validated its selected non-delete edit through the existing render-backed counterfactual gate on `parking_phone_tiny`.

**Implementation**:
- `scripts/car_model/meshsplatopt_select_checkpoint_local_snap_edit.py`
- optimized `make_snap_proposals` to evaluate only explicit candidate vertices when provided

**Selection**:
- checkpoint: `outputs/carnet/meshprior/parking_phone_tiny/current_branch_2000iter/model/point_cloud/iteration_2000/point_cloud_state_dict.pt`
- candidate faces above threshold: `3915`
- candidate vertices: `45`
- proposals: `135`
- valid proposals: `113`
- selected vertex: `704480`
- expected local residual: `0.042196625106825536 -> 0.021098312553412768`

**Gate**:
- status: `PASS`
- topology: `782982` triangles and `820107` vertices before/after
- render deltas: PSNR `-9.5367431640625e-07`, SSIM `0.0`, LPIPS `+1.7881393432617188e-07`
- geometry deltas: AbsRel `+2.4770185902411868e-11`, Depth MAE `+4.0913414878218646e-09`, normal mean deg `+4.2470329475463586e-07`

**Decision**: Real-checkpoint local snap is now integrated and gate-safe. The result is a safety/integration pass, not a quality-gain claim; next is multi-candidate portfolio selection before W&B recovery.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR17_02_checkpoint_local_snap_gate_report.md`

---

## 2026-05-03 — MeshSplatOpt R17.03-R17.05 local snap portfolio recovery — MIXED/FAIL

**Outcome**: Extended the real-checkpoint local snap selector into a 16-vertex portfolio edit, validated it with the render-backed checkpoint gate, and ran equal-budget 200-step W&B recovery against a baseline continuation.

**Selection**:
- candidate faces above threshold: `7831`
- candidate vertices: `443`
- proposals: `1446`
- valid proposals: `1291`
- selected vertices: `16`
- total expected local residual reduction: `2.5543751879508467`

**Gate**:
- status: `PASS`
- topology unchanged at `782982` triangles and `820107` vertices
- render/geometry deltas at iteration 2000 are numerical-noise level

**W&B**:
- baseline continuation: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/2puomo88`
- portfolio snap: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/d6dc9qja`

**Equal-budget 2200 result**:
- baseline: PSNR `12.331465`, SSIM `0.298222`, LPIPS `0.622323`, AbsRel `0.409263`, Depth MAE `4.300273`, normal `52.595639`
- portfolio: PSNR `12.326042`, SSIM `0.297809`, LPIPS `0.621754`, AbsRel `0.410215`, Depth MAE `4.307691`, normal `52.827494`
- portfolio minus baseline: PSNR `-0.005423`, SSIM `-0.000413`, LPIPS `-0.000569`, AbsRel `+0.000952`, Depth MAE `+0.007418`, normal `+0.231855`

**Decision**: `PORTFOLIO_SNAP_GATE_PASS_RECOVERY_QUALITY_FAIL`. The portfolio edit is safe and auditable but not better than continuation. The next selector must use stronger evidence, likely render residuals, sparse-depth residuals, normal disagreement, or defect-mined CSEF regions instead of large-area seeding alone.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR17_03_05_portfolio_snap_recovery_report.md`

---

## 2026-05-03 — MeshSplatOpt R17.06 risk-filtered local snap gate — PASS

**Outcome**: Added proposal-risk controls to the checkpoint local snap selector and validated a non-boundary, uncertainty-filtered 16-vertex portfolio through the render-backed checkpoint gate.

**Implementation**:
- `--max_proposal_uncertainty`
- `--exclude_boundary_vertices`

**Selection**:
- candidate faces above threshold: `11746`
- selected vertices: `16`
- all selected proposals are non-boundary vertices
- max selected uncertainty: `0.35`
- total expected local residual reduction: `0.8844110663521292`

**Gate**:
- status: `PASS`
- topology unchanged at `782982` triangles and `820107` vertices
- deltas are numerical-noise level: PSNR `-0.000001`, SSIM `-0.00000018`, LPIPS `+0.00000054`, AbsRel `0.0`, Depth MAE `0.0`, normal `-0.00000125`

**Decision**: This is a selector safety improvement, not a quality claim. Because R17.03-R17.05 already showed area-seeded snap portfolios fail equal-budget recovery, the next selector should use render residuals, sparse-depth residuals, normal disagreement, or CSEF defect regions.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR17_06_risk_filtered_snap_gate_report.md`

---

## 2026-05-03 — MeshSplatOpt R18.01-R18.03 train-residual local snap recovery — MOSTLY POSITIVE

**Outcome**: Added a residual-aware checkpoint snap selector and validated a 16-vertex train-residual portfolio through held-out render-backed gate plus equal-budget W&B recovery.

**Implementation**:
- new selector: `scripts/car_model/meshsplatopt_select_checkpoint_residual_snap_edit.py`
- proposal evidence: input/train render residuals, large-area candidate prefilter, local plane CSEF snap residual reduction
- protocol guard: test residual selection is marked diagnostic; paper-valid selection used `render_set=train` and `camera_index_offset=54`

**Train-residual selection**:
- status: `PASS`
- candidate faces: `19575`
- candidate vertices: `4469`
- scored vertices: `3918`
- proposals: `3000`
- valid proposals: `438`
- selected vertices: `16`
- top selected vertices: `730295`, `500770`, `676458`

**Gate**:
- status: `PASS`
- topology unchanged at `782982` triangles and `820107` vertices
- held-out deltas at iteration 2000: PSNR `0.0`, SSIM `-1.4901161193847656e-07`, LPIPS `+1.7881393432617188e-07`, AbsRel `0.0`, Depth MAE `0.0`, normal `-2.47278670428841e-07`

**W&B**:
- train-residual snap recovery: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1oqymqmp`

**Equal-budget 2200 result**:
- baseline continuation: PSNR `12.331465`, SSIM `0.298222`, LPIPS `0.622323`, AbsRel `0.409263`, Depth MAE `4.300273`, normal `52.595639`
- area portfolio snap: PSNR `12.326042`, SSIM `0.297809`, LPIPS `0.621754`, AbsRel `0.410215`, Depth MAE `4.307691`, normal `52.827494`
- train-residual snap: PSNR `12.342549`, SSIM `0.298893`, LPIPS `0.622299`, AbsRel `0.408892`, Depth MAE `4.302941`, normal `52.354489`

**Decision**: `TRAIN_RESIDUAL_SNAP_GATE_PASS_RECOVERY_MOSTLY_POSITIVE`. This fixes the main R17 selector weakness: the portfolio is now tied to observed residual evidence and beats same-budget continuation on PSNR, SSIM, AbsRel, and normal angle. Effect size remains small and Depth MAE is slightly worse, so the next step is multi-scene validation and richer residual evidence.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR18_01_03_residual_snap_report.md`

---

## 2026-05-03 — MeshSplatOpt R19.01-R19.08 cross-scene residual snap — MIXED POSITIVE

**Outcome**: Generalized residual-aware local snap from parking to courtyard and bonsai, added automatic camera-offset inference, calibrated proposal uncertainty from `0.35` to `0.55`, ran held-out gates on both new scenes, and completed same-source W&B 200-step recovery baselines/candidates.

**Implementation**:
- automatic `camera_index_offset` inference in `scripts/car_model/meshsplatopt_select_checkpoint_residual_snap_edit.py`
- richer selector audit fields: render-view count, rejection reasons, pre/post risk-filter counts
- default `--max_proposal_uncertainty` changed to `0.55` after cross-scene gate calibration

**Selection/gate**:
- strict `0.35` returned `NO_CANDIDATE` on courtyard and bonsai
- calibrated `0.55` selected `16` vertices on both scenes
- courtyard gate: `PASS`, topology unchanged, PSNR delta `-0.000409`, LPIPS delta `+0.000014`, normal delta `-0.000597`
- bonsai gate: `PASS`, topology unchanged, PSNR delta `-0.000010`, LPIPS delta `+0.000003`, normal delta `+0.0000003`

**W&B**:
- courtyard baseline: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ajvqp7ou`
- courtyard residual snap: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/mhjbnm2t`
- bonsai baseline: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/b9miy649`
- bonsai residual snap: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/p33pm98r`

**Equal-budget recovery**:
- courtyard residual snap minus baseline: PSNR `-0.002344`, SSIM `-0.000018`, LPIPS `-0.000183`, AbsRel `-0.000306`, Depth MAE `-0.002072`, normal `+0.285845`
- bonsai residual snap minus baseline: PSNR `-0.000485`, SSIM `-0.000061`, LPIPS `-0.000112`, AbsRel `-0.000154`, Depth MAE `-0.001383`, normal `-0.035446`

**Decision**: `CROSS_SCENE_RESIDUAL_SNAP_GATE_PASS_RECOVERY_MIXED_POSITIVE`. This materially reduces the single-scene risk and shows consistent LPIPS/depth improvements on two new scenes, but the effect sizes remain small and PSNR/SSIM are slightly negative. Next required step is patch-level residual repair rather than isolated vertex snaps.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR19_01_08_cross_scene_residual_snap_report.md`

---

## 2026-05-03 — MeshSplatOpt R20.01 parking medium residual snap — DEPTH GAIN / RENDER FAIL

**Outcome**: Ran a medium-budget W&B recovery for the R18 train-residual parking snap candidate from `2000` to `4000` iterations on GPU 4.

**W&B**:
- `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/tu85uksa`

**Protocol**:
- source: R18 train-residual snap gate candidate
- load iteration: `2000`
- train until: `4000`
- densify until: `2000`
- restricted Delaunay: skipped
- train/render/metrics exit codes: `0/0/0`

**Medium result vs existing parking baseline**:
- baseline 2000->4000: PSNR `14.251087`, SSIM `0.383800`, LPIPS `0.569749`, AbsRel `0.324794`, Depth MAE `3.636891`, normal `51.043451`
- residual snap 2000->4000: PSNR `14.207231`, SSIM `0.383298`, LPIPS `0.570288`, AbsRel `0.323844`, Depth MAE `3.589209`, normal `51.225949`
- residual snap minus baseline: PSNR `-0.043857`, SSIM `-0.000501`, LPIPS `+0.000539`, AbsRel `-0.000951`, Depth MAE `-0.047682`, normal `+0.182499`

**Decision**: `MEDIUM_RESIDUAL_SNAP_DEPTH_GAIN_RENDER_QUALITY_FAIL`. The edit improves depth but does not survive medium-budget appearance-quality comparison. Isolated vertex snaps are not enough for a top-tier headline; the next required method step is clustered patch repair or fill/split proposals.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR20_01_parking_medium_residual_snap_report.md`

---

## 2026-05-03 — MeshSplatOpt R21.01-R21.03 residual patch snap — MIXED

**Outcome**: Added the first checkpoint-compatible patch-level residual repair primitive by expanding train-residual snap seed vertices to local mesh neighborhoods, then validated it with held-out gate and 200-step W&B recovery.

**Implementation**:
- new script: `scripts/car_model/meshsplatopt_expand_snap_edit_to_patch.py`
- edit type remains `SNAP_VERTICES`, so rollback/checkpoint gate support is preserved
- patch policy: k-hop adjacency, radius filter, distance-weighted seed displacement

**Patch candidate**:
- seed vertices: `16`
- patch vertices: `95`
- affected faces: `217`
- max displacement: `0.074180`
- mean displacement: `0.018138`

**Gate**:
- status: `PASS`
- topology unchanged at `782982` triangles and `820107` vertices
- gate deltas: PSNR `+0.00000095`, SSIM `+0.00000009`, LPIPS `-0.00000089`, AbsRel `0.0`, Depth MAE `0.0`, normal `-0.00000130`

**W&B**:
- patch recovery: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/76fgy4z5`

**Equal-budget 2200 result**:
- baseline continuation: PSNR `12.331465`, SSIM `0.298222`, LPIPS `0.622323`, AbsRel `0.409263`, Depth MAE `4.300273`, normal `52.595639`
- single residual snap: PSNR `12.342549`, SSIM `0.298893`, LPIPS `0.622299`, AbsRel `0.408892`, Depth MAE `4.302941`, normal `52.354489`
- patch residual snap: PSNR `12.329646`, SSIM `0.298382`, LPIPS `0.622157`, AbsRel `0.409988`, Depth MAE `4.303037`, normal `52.586082`

**Decision**: `PATCH_SNAP_GATE_PASS_RECOVERY_MIXED`. This fixes the missing patch-primitive architecture, but the naive displacement-diffusion policy is not yet a dominant method result. Next step: residual-cluster optimization or fill/split proposals with an explicit render/depth objective.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR21_01_03_patch_snap_report.md`

---

## 2026-05-03 — MeshSplatOpt R22.01-R22.04 boundary fill — GATE PASS / SHORT PROMISING / MEDIUM FAIL

**Outcome**: Added and validated the first real checkpoint boundary-loop `FILL_PATCH` selector. The edit passes held-out gate and trains successfully, but naive centroid-fan fill does not survive medium-budget comparison.

**Implementation**:
- new selector: `scripts/car_model/meshsplatopt_select_checkpoint_boundary_fill_edit.py`
- selects checkpoint boundary loops by loop length and XY area
- emits checkpoint-compatible `FILL_PATCH` with boundary certificate

**Selected fill**:
- parking boundary loops found: `48858`
- filtered candidates: `4545`
- selected loop vertices: `6`
- selected XY area: `24.723803`
- topology delta: `+1` vertex, `+6` triangles

**Gate**:
- status: `PASS`
- deltas: PSNR `+0.000097`, SSIM `-0.00000039`, LPIPS `+0.00000364`, AbsRel `0.0`, Depth MAE `0.0`, normal `+0.00000110`

**W&B**:
- short fill recovery: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/jzxzz4g2`
- medium fill recovery: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1tqd66ah`

**Short 2200 result**:
- baseline: PSNR `12.331465`, SSIM `0.298222`, LPIPS `0.622323`, AbsRel `0.409263`, Depth MAE `4.300273`, normal `52.595639`
- boundary fill: PSNR `12.354150`, SSIM `0.298658`, LPIPS `0.621934`, AbsRel `0.410232`, Depth MAE `4.302468`, normal `52.328850`
- decision: `FILL_SHORT_RECOVERY_APPEARANCE_NORMAL_PASS_DEPTH_FAIL`

**Medium 4000 result**:
- baseline: PSNR `14.251087`, SSIM `0.383800`, LPIPS `0.569749`, AbsRel `0.324794`, Depth MAE `3.636891`, normal `51.043451`
- boundary fill: PSNR `14.224104`, SSIM `0.381926`, LPIPS `0.570877`, AbsRel `0.329337`, Depth MAE `3.645573`, normal `51.527010`
- decision: `FILL_MEDIUM_RECOVERY_FAIL`

**Decision**: `BOUNDARY_FILL_GATE_PASS_SHORT_PROMISING_MEDIUM_FAIL`. The codebase now supports real topology-adding repair with gate and recovery evidence. The weakness is selector/geometry quality: centroid fan fill is too naive. Next step should be residual/depth-aware fill placement and local fairing.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR22_01_04_boundary_fill_report.md`

---

## 2026-05-03 — MeshSplatOpt R23.01 residual-aware boundary fill selector — PASS

**Outcome**: Upgraded the boundary-loop fill selector to support train-residual ranking, aligning fill proposals with CSEF explanation debt instead of area-only selection.

**Implementation**:
- `scripts/car_model/meshsplatopt_select_checkpoint_boundary_fill_edit.py` now supports `--rank residual`
- residual mode projects candidate loop vertices into high-residual train views and ranks by `mean_loop_residual * sqrt(area)`
- camera offset is auto-inferred using the same protocol as residual snap

**Parking selection**:
- loop count: `48858`
- candidates: `4545`
- selected loop index: `46134`
- loop vertices: `6`
- area: `24.723803`
- train residual score: `0.387146`
- rank score: `1.925007`
- camera offset: `54`

**Decision**: `RESIDUAL_BOUNDARY_FILL_SELECTOR_PASS_GEOMETRY_STILL_WEAK`. The selector is now evidence-aligned, but it chose the same loop as R22; therefore R22's medium failure is more likely due to crude centroid-fan geometry than proposal ranking. Next fix should target depth-aware/fair inserted geometry.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR23_01_residual_boundary_fill_selector_report.md`

---

## 2026-05-03 — MeshSplatOpt R24-R26 fill initialization and grid fill — ENGINEERING PASS / MEDIUM FAIL

**Outcome**: Tested three follow-ups to the R22 boundary-fill weakness: nearest-face checkpoint field initialization, unrestricted densification recovery, and a denser plane-grid Delaunay fill. The implementation and gates passed, but the medium-budget public-scene result still does not beat the strong baseline.

**Implementation**:
- `ss3dm_prior/meshsplatopt/checkpoint_adapter.py` now initializes appended `FILL_PATCH` face fields from nearest old faces instead of zeros.
- added `scripts/car_model/meshsplatopt_expand_boundary_fill_to_grid.py` for checkpoint-compatible plane-grid Delaunay fill expansion.

**R24 gate and short recovery**:
- gate: `PASS`, topology `+1` vertex / `+6` triangles, PSNR delta `+0.000097`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/1iam7x3c`
- 2200 result: PSNR `12.347798`, SSIM `0.297994`, LPIPS `0.621984`, AbsRel `0.409399`, Depth MAE `4.302556`, normal `52.568240`

**R25 densification-on diagnostic**:
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/hkzqqedj`
- result: PSNR `12.031141`, SSIM `0.310603`, LPIPS `0.641519`
- topology exploded to `5,889,468` triangles / `4,964,968` vertices
- decision: unrestricted post-edit densification is rejected.

**R26 grid fill**:
- generated grid fill from R22 loop: `+51` vertices / `+106` triangles
- gate: `PASS`, PSNR delta `+0.0000925`
- short W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/bg5cflp8`
- medium W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/phki0fj4`
- 4000 result: PSNR `14.212496`, SSIM `0.383164`, LPIPS `0.570729`, AbsRel `0.329141`, Depth MAE `3.667578`, normal `51.594204`

**Decision**: `FILL_INIT_GRID_ENGINEERING_PASS_MEDIUM_REPAIR_FAIL`. The system now has stronger topology-adding edit machinery, but real-scene gains remain insufficient. The next high-value fix is true external-edit teacher recovery: pre-edit teacher render/depth cache, unedited-region distillation, and edit-region metrics.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR24_R26_fill_init_and_grid_report.md`

---

## 2026-05-03 — MeshSplatOpt R27 sparse-depth recovery — MEDIUM PASS

**Outcome**: Found the first strong parking medium-budget repair gain. Low-weight sparse COLMAP depth recovery (`lambda=0.005`) makes the R26 grid fill edit outperform the strong frozen-topology baseline on render and geometry, and it also beats a matched baseline+sparse control.

**Implementation**:
- `scripts/car_model/meshsplatopt_run_teacher_recovery.py` now supports `--train_extra_args` for reproducible recovery diagnostics.

**Negative diagnostic**:
- high sparse-depth weight `0.05` failed.
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/hrug0itm`
- result: PSNR `12.315643`, AbsRel `0.411106`, normal `52.800506`

**Short pass**:
- R27.02 output: `outputs/carnet/meshsplatopt/stageR27_02_parking_boundary_grid_fill_sparse_depth_lam0p005_2000to2200`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/ogabx44c`
- result: PSNR `12.362178`, SSIM `0.299357`, LPIPS `0.621872`, AbsRel `0.407613`, Depth MAE `4.307866`, normal `52.595478`
- decision: best short-run render and AbsRel among parking repair variants.

**Medium pass**:
- R27.03 output: `outputs/carnet/meshsplatopt/stageR27_03_parking_boundary_grid_fill_sparse_depth_lam0p005_2000to4000`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/81hryi53`
- result: PSNR `14.325891`, SSIM `0.385450`, LPIPS `0.567749`, AbsRel `0.306381`, Depth MAE `3.605697`, normal `49.906129`

**Matched control**:
- R27.04 baseline+sparse output: `outputs/carnet/meshsplatopt/stageR27_04_parking_baseline_sparse_depth_lam0p005_2000to4000`
- W&B: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/b726rga8`
- baseline+sparse result: PSNR `14.301250`, SSIM `0.384772`, LPIPS `0.567846`, AbsRel `0.309894`, Depth MAE `3.666060`, normal `50.012948`
- edit+sparse delta versus matched sparse control: PSNR `+0.024641`, SSIM `+0.000678`, LPIPS `-0.000097`, AbsRel `-0.003513`, Depth MAE `-0.060363`, normal `-0.106820`

**Decision**: `SPARSE_DEPTH_REPAIR_MEDIUM_PASS`. This is not yet a full paper result because sparse recovery is the dominant contributor, but the repair edit adds measurable benefit under an identical recovery setting. Next step: cross-scene sparse recovery controls and edit-region metrics.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR27_sparse_depth_recovery_report.md`

---

## 2026-05-03 - R28-R30 full sparse recovery pivot

**Goal**: push the parking repair path beyond the small medium-run gains and determine whether the boundary grid-fill edit or sparse COLMAP depth recovery is the real full-budget contributor.

**Full-budget attribution**:
- R28.01 grid-fill+sparse, 2000->7000, W&B `94pkp05l`: PSNR `15.770156`, SSIM `0.459545`, LPIPS `0.519976`, AbsRel `0.240156`, normal `46.143910`
- R28.02 baseline+sparse, 2000->7000, W&B `zm1ztyf4`: PSNR `15.822877`, SSIM `0.458552`, LPIPS `0.519231`, AbsRel `0.231866`, normal `45.929940`
- R28.03 grid-fill+sparse lower weight, 2000->7000, W&B `7u0onsok`: PSNR `15.741236`, SSIM `0.455811`, LPIPS `0.520650`

**Decision**: `GRID_FILL_REJECTED_AT_FULL_BUDGET`. The current boundary fill edit does not beat the matched baseline+sparse control at full budget. The method narrative must pivot to sparse-geometry-guided recovery.

**Loss-space diagnostic**:
- Added optional sparse depth loss spaces: `depth`, `relative`, `log`, and `inverse`.
- R29.01 relative loss, W&B `zk7dfh9z`: PSNR `15.643266`, SSIM `0.454726`, LPIPS `0.522929`
- R29.02 log loss, W&B `j93ejnsk`: PSNR `15.608345`, SSIM `0.452642`, LPIPS `0.525190`

**Decision**: `METRIC_DEPTH_SMOOTH_L1_RETAINED`. Relative/log variants hurt parking full-budget rendering.

**Long-run breakthrough**:
- R30.01 baseline+sparse, 7000->12000, W&B `9oi1skys`: PSNR `16.872860`, SSIM `0.514039`, LPIPS `0.475757`, AbsRel `0.192306`, normal `42.638562`
- R30.02 baseline+sparse, 12000->16000, W&B `6gsab26p`: PSNR `17.081682`, SSIM `0.531858`, LPIPS `0.458050`, AbsRel `0.185581`, normal `41.859201`

**Delta vs R16 full baseline**: PSNR `+1.511117`, SSIM `+0.083646`, LPIPS `-0.070003`.

**Decision**: `LONG_HORIZON_SPARSE_RECOVERY_FULL_PASS`. This is now the strongest parking result and should be the new main experimental axis.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`

---

## 2026-05-03 - R31 early-stop and cross-scene sparse recovery

**Goal**: raise completion beyond the single parking-scene breakthrough by testing saturation and cross-scene generalization.

**Parking saturation**:
- R31.01 continued R30.02 from 16000 to 20000, W&B `ekcjc7qi`
- 20000 result: PSNR `17.027088`, SSIM `0.532724`, LPIPS `0.455719`, AbsRel `0.187616`, normal `41.740965`
- Compared with R30.02 at 16000: PSNR `-0.054594`, SSIM `+0.000866`, LPIPS `-0.002330`, AbsRel `+0.002035`, normal `-0.118235`

**Decision**: `EARLY_STOP_16000_FOR_RENDER`. Use R30.02/16000 as the main parking table entry; mention R31.01 as saturation evidence.

**Cross-scene pass**:
- R31.02 courtyard Stage35 sparse-depth continuation, W&B `s35bmzau`
  - 2000 baseline: PSNR `15.383161`, SSIM `0.508091`, LPIPS `0.584694`
  - 7000 recovery: PSNR `16.313482`, SSIM `0.547770`, LPIPS `0.520214`, AbsRel `0.127543`, normal `30.207450`
  - delta: PSNR `+0.930322`, SSIM `+0.039679`, LPIPS `-0.064480`
- R31.03 bonsai Stage35 sparse-depth continuation, W&B `3wygm9u4`
  - 2000 baseline: PSNR `12.267367`, SSIM `0.277617`, LPIPS `0.611939`
  - 7000 recovery: PSNR `20.299246`, SSIM `0.606873`, LPIPS `0.388372`, AbsRel `0.130567`, normal `34.987466`
  - delta: PSNR `+8.031878`, SSIM `+0.329256`, LPIPS `-0.223567`

**Decision**: `CROSS_SCENE_SPARSE_RECOVERY_PASS`. The method now has positive evidence on parking, courtyard, and bonsai.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`

---

## 2026-05-03 - R32 trusted sparse correspondence sampling

**Goal**: improve the strongest sparse-depth recovery path with a real algorithmic change, not just longer training. The new option replaces uniform visible-COLMAP-point subsampling with reprojection-error-aware sparse correspondence sampling.

**Implementation**:
- added `--sparse_colmap_depth_sample_mode` with `random`, `low_error`, and `mixed_low_error`;
- added `--sparse_colmap_depth_low_error_fraction` for trusted/random mixtures;
- reused the same sampler in sparse depth training and sparse geometry evaluation paths.

**Parking trusted-sampling validation**:
- R32.01b low-error-only, 12000->16000, W&B `m8fu6936`
  - result: PSNR `17.086828`, SSIM `0.532577`, LPIPS `0.457497`, AbsRel `0.185512`, Depth MAE `2.966934`, normal `41.771796`
- R32.02b mixed low-error/random, 12000->16000, W&B `j58gdh9q`
  - result: PSNR `17.105490`, SSIM `0.532643`, LPIPS `0.457859`, AbsRel `0.184374`, Depth MAE `2.957988`, normal `41.764144`

**Delta versus previous best R30.02**:
- PSNR `+0.023808`
- SSIM `+0.000785`
- LPIPS `-0.000191`
- AbsRel `-0.001207`
- Depth MAE `-0.003582`
- normal angle `-0.095057`

**Decision**: `TRUSTED_MIXED_SPARSE_SAMPLING_PASS`. R32.02b is the new strongest parking result and gives the paper a cleaner method contribution: confidence-aware COLMAP sparse correspondence sampling on top of long-horizon sparse-geometry recovery.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`

---

## 2026-05-03 - R33 cross-scene trusted sampling check

**Goal**: test whether R32 mixed trusted/random sampling generalizes as a render improvement or mainly acts as a geometry-confidence regularizer.

**Runs**:
- R33.01 courtyard Stage35 mixed trusted/random sparse-depth, 2000->7000, W&B `s1po8x07`
  - result: PSNR `16.304310`, SSIM `0.545805`, LPIPS `0.521787`, AbsRel `0.123796`, Depth MAE `1.536491`, normal `29.875990`
  - delta versus R31.02 random: PSNR `-0.009172`, SSIM `-0.001965`, LPIPS `+0.001573`, AbsRel `-0.003747`, Depth MAE `-0.034883`, normal `-0.331460`
- R33.02 bonsai Stage35 mixed trusted/random sparse-depth, 2000->7000, W&B `xj2ng1s1`
  - result: PSNR `20.279762`, SSIM `0.605154`, LPIPS `0.390035`, AbsRel `0.128458`, Depth MAE `1.417768`, normal `35.109088`
  - delta versus R31.03 random: PSNR `-0.019484`, SSIM `-0.001719`, LPIPS `+0.001663`, AbsRel `-0.002109`, Depth MAE `-0.034337`, normal `+0.121622`

**Decision**: `TRUSTED_SAMPLING_GEOMETRY_PASS_RENDER_MIXED`. R33 strengthens the geometry side of the trusted sampler claim but prevents overclaiming render generalization. The current paper-safe conclusion is: random sparse sampling remains cross-scene render-best at this budget; mixed trusted sampling is parking render-best and cross-scene sparse-depth-geometry-best for AbsRel/MAE.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`

---

## 2026-05-03 - R34-R35 parking trusted-fraction ablation

**Goal**: improve completion by replacing a single trusted/random mixture result with a measured ablation over trusted COLMAP-track fractions.

**Runs**:
- R34.01 fraction `0.25`, W&B `jfcn9ug0`: PSNR `17.098461`, SSIM `0.531578`, LPIPS `0.458490`, AbsRel `0.184467`, Depth MAE `2.964016`, normal `41.684424`
- R32.02b fraction `0.50`, W&B `j58gdh9q`: PSNR `17.105490`, SSIM `0.532643`, LPIPS `0.457859`, AbsRel `0.184374`, Depth MAE `2.957988`, normal `41.764144`
- R35.01 fraction `0.625`, W&B `t8y6ryn9`: PSNR `17.105064`, SSIM `0.532436`, LPIPS `0.457493`, AbsRel `0.183602`, Depth MAE `2.959589`, normal `41.472216`
- R34.02 fraction `0.75`, W&B `ympoevql`: PSNR `17.099464`, SSIM `0.532346`, LPIPS `0.457681`, AbsRel `0.183488`, Depth MAE `2.959905`, normal `41.606181`

**Decision**: `TRUSTED_FRACTION_PARETO_PASS`. Fraction `0.50` remains PSNR/SSIM-best. Fraction `0.625` is the geometry-balanced Pareto setting: it gives up only `0.000425` PSNR versus R32.02b while improving LPIPS by `0.000366`, AbsRel by `0.000772`, and normal angle by `0.291927` degrees.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`

---

## 2026-05-03 - R36-R38 trusted-sampling refinement and lambda sweep

**Goal**: turn the trusted/random sparse correspondence sampler from a single parking gain into a stronger, better-supported method contribution with cross-scene tuning evidence and a new parking best result.

**R36 cross-scene fraction `0.625` checks**:
- R36.01b courtyard, W&B `qguqasou`: PSNR `16.376713`, SSIM `0.548868`, LPIPS `0.520534`, AbsRel `0.126731`, Depth MAE `1.564874`, normal `29.581638`
- R36.02b bonsai, W&B `xq21lzsm`: PSNR `20.267965`, SSIM `0.605809`, LPIPS `0.391068`, AbsRel `0.130851`, Depth MAE `1.441631`, normal `35.098674`

**R36 decision**: `CROSS_SCENE_FRACTION_TUNING_PARTIAL_PASS`. The `0.625` mixture is a strong courtyard setting, improving PSNR over R31.02 by `+0.063231` and normal angle by `-0.625812`. It is not a bonsai render improvement, so the paper-safe claim is scene-dependent trusted-fraction selection rather than a universal fraction.

**R37 stratified-sampling probe**:
- R37.01 courtyard, W&B `tn0uxiwy`: PSNR `16.273159`, SSIM `0.546080`, LPIPS `0.521507`, AbsRel `0.128638`, Depth MAE `1.577220`, normal `30.181338`
- R37.02 bonsai, W&B `nrylaqan`: PSNR `20.252667`, SSIM `0.605428`, LPIPS `0.390547`, AbsRel `0.128677`, Depth MAE `1.423667`, normal `35.203336`

**R37 decision**: `STRATIFIED_SAMPLING_NOT_RETAINED`. Error-stratified sampling improved bonsai sparse-depth geometry relative to R36 but hurt courtyard and did not improve render. The implementation probe was therefore not kept in the main code path.

**R38 parking sparse-loss lambda refinement**:
- R38.01 fraction `0.50`, lambda `0.003`, W&B `yo6oxofn`: PSNR `17.124186`, SSIM `0.533355`, LPIPS `0.456678`, AbsRel `0.183460`, Depth MAE `2.945563`, normal `41.679295`
- R38.02 fraction `0.625`, lambda `0.003`, W&B `j8t2tyc9`: PSNR `17.107119`, SSIM `0.532528`, LPIPS `0.456906`, AbsRel `0.183256`, Depth MAE `2.933642`, normal `41.632026`

**R38 decision**: `NEW_PARKING_RENDER_AND_GEOMETRY_BEST`. R38.01 is the new strongest parking render result. Versus R32.02b it improves PSNR by `+0.018696`, SSIM by `+0.000712`, LPIPS by `-0.001181`, AbsRel by `-0.000915`, Depth MAE by `-0.012425`, and normal angle by `-0.084849`. R38.02 is the geometry-biased lambda-refined variant: it gives up PSNR versus R38.01 but further improves AbsRel, Depth MAE, and normal angle.

**Linked artefact**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`

---

## 2026-05-03 - R39 sparse-depth lambda fine sweep and table collector

**Goal**: raise completion by checking whether the R38 `lambda=0.003` point is actually optimal, and by adding a reproducible collector for sparse-recovery paper tables.

**Implementation**:
- added `scripts/car_model/meshsplatopt_collect_sparse_recovery_results.py`;
- collector reads independent `results.json` plus `geometry_eval_colmap/iter_*_max500.json`;
- collector writes JSON, CSV, and Markdown under `outputs/carnet/meshsplatopt/sparse_recovery_tables`.

**R39 parking lambda fine sweep**:
- R39.01 fraction `0.50`, lambda `0.002`, W&B `jqcn7cwc`: PSNR `17.142246`, SSIM `0.534422`, LPIPS `0.456627`, AbsRel `0.181240`, Depth MAE `2.825327`, normal `41.812617`
- R39.02 fraction `0.50`, lambda `0.004`, W&B `o9f9e03g`: PSNR `17.088505`, SSIM `0.532507`, LPIPS `0.457398`, AbsRel `0.184764`, Depth MAE `2.956959`, normal `41.736803`

**Decision**: `NEW_STRONGEST_PARKING_RESULT_AND_LAMBDA_CURVE_PASS`. R39.01 supersedes R38.01. Versus R38.01 it improves PSNR by `+0.018061`, SSIM by `+0.001067`, LPIPS by `-0.000051`, AbsRel by `-0.002219`, and Depth MAE by `-0.120237`. R39.02 confirms that increasing lambda back toward `0.005` loses both render and depth geometry. The current best sparse-depth lambda for parking is therefore `0.002`.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`
- `scripts/car_model/meshsplatopt_collect_sparse_recovery_results.py`

---

## 2026-05-03 - R40 low-lambda sparse-depth Pareto and cross-scene jump

**Goal**: respond to the weak-gain bottleneck by testing whether the R39 `lambda=0.002` optimum is a parking-only point or part of a broader lower-lambda regime.

**Runs**:
- R40.01 parking fraction `0.50`, lambda `0.001`, W&B `czebaxco`: PSNR `17.145630`, SSIM `0.534154`, LPIPS `0.456297`, AbsRel `0.181336`, Depth MAE `2.849124`, normal `42.151608`
- R40.02 courtyard fraction `0.625`, lambda `0.002`, W&B `coqls9rm`: PSNR `16.801973`, SSIM `0.559031`, LPIPS `0.508579`, AbsRel `0.106783`, Depth MAE `1.388936`, normal `29.394197`

**Decision**: `LOW_LAMBDA_CROSS_SCENE_STRONG_PASS`. R40.01 becomes the parking render/LPIPS Pareto row: relative to R39.01 it improves PSNR by `+0.003384` and LPIPS by `-0.000330`, while giving back `0.000267` SSIM and a small amount of sparse-depth geometry. R40.02 is the more important milestone: relative to the previous courtyard tuned row R36.01b it improves PSNR by `+0.425260`, SSIM by `+0.010163`, LPIPS by `-0.011955`, AbsRel by `-0.019948`, Depth MAE by `-0.175938`, and normal angle by `-0.187441`. This upgrades the claim from a parking-tuned result to a cross-scene low-lambda sparse-depth regime with a large courtyard gain.

**R41 bonsai follow-up**:
- R41.01 bonsai fraction `0.50`, lambda `0.002`, W&B `poh8k4be`: PSNR `21.601114`, SSIM `0.677450`, LPIPS `0.347170`, AbsRel `0.161510`, Depth MAE `1.824463`, normal `36.047671`
- relative to R31.03 random sparse-depth, R41.01 improves PSNR by `+1.301868`, SSIM by `+0.070577`, and LPIPS by `-0.041202`, but worsens AbsRel by `+0.030943`, Depth MAE by `+0.372358`, and normal angle by `+1.060204`

**R41 decision**: `BONSAI_RENDER_BREAKTHROUGH_GEOMETRY_TRADEOFF`. This closes the previous bonsai render weakness and makes the low-lambda regime cross-scene-render-positive, but it should be presented as a render/geometry Pareto branch rather than a universal geometry improvement.

**R42 fraction repair check**:
- R42.01 bonsai fraction `0.625`, lambda `0.002`, W&B `l2inxutg`: PSNR `21.543251`, SSIM `0.672968`, LPIPS `0.349113`, AbsRel `0.161678`, Depth MAE `1.824630`, normal `35.622191`
- relative to R41.01, R42.01 gives up PSNR `-0.057863`, SSIM `-0.004483`, and LPIPS `+0.001943`; it slightly improves normal angle by `-0.425480`, with effectively unchanged depth

**R42 decision**: `BONSAI_FRACTION_REPAIR_BOUNDARY`. Raising the trusted fraction does not recover bonsai depth geometry, so the best current bonsai claim remains R41.01 as a render Pareto breakthrough.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR28_R30_full_sparse_recovery_report.md`
- `scripts/car_model/meshsplatopt_collect_sparse_recovery_results.py`

---

## 2026-05-03 - R43 long-horizon validation

**Goal**: answer whether the R40-R42 medium/full rows are enough, or whether longer training changes the conclusion.

**Runs**:
- R43.01b parking fraction `0.50`, lambda `0.001`, `16000->30000`, W&B `mhz6t8ps`: PSNR `16.249155`, SSIM `0.511035`, LPIPS `0.477426`, AbsRel `0.193679`, Depth MAE `3.018124`, normal `43.714506`
- R43.02b courtyard fraction `0.625`, lambda `0.002`, `7000->20000`, W&B `cla3utia`: PSNR `17.793036`, SSIM `0.560976`, LPIPS `0.496724`, AbsRel `0.158907`, Depth MAE `1.991568`, normal `28.950016`

**Decision**: `LONG_HORIZON_VALIDATION_SPLIT`. Parking long-horizon continuation is a clear overtraining/failure boundary: relative to R40.01 it drops PSNR by `-0.896475`, SSIM by `-0.023119`, LPIPS worsens by `+0.021129`, AbsRel worsens by `+0.012343`, and Depth MAE worsens by `+0.168999`. Courtyard long-horizon continuation improves render strongly relative to R40.02, with PSNR `+0.991062`, SSIM `+0.001944`, and LPIPS `-0.011855`, while sacrificing sparse depth agreement by AbsRel `+0.052124` and Depth MAE `+0.602633`; normal angle improves by `-0.444181`.

**Paper implication**: the method needs a budget-aware Pareto claim. R40.02 is the all-metric courtyard row, R43.02b is the courtyard render-best long row, and R43.01b proves that parking should not be blindly extended beyond the validated 16000 budget.

---

## 2026-05-03 - R44 sparse-depth decay long-horizon repair and clean baseline comparison

**Goal**: answer the long-training weakness found by R43 and produce direct clean-baseline render evidence.

**Implementation**:
- added a sparse COLMAP depth loss decay schedule:
  - `--sparse_colmap_depth_decay_start_iter`
  - `--sparse_colmap_depth_decay_end_iter`
  - `--sparse_colmap_depth_decay_final_mult`
- default behavior is unchanged because decay is disabled unless the start/end window is set.

**Runs**:
- R44.01 parking fraction `0.50`, lambda `0.001`, decay `16000->20000` to zero, trained `16000->22000`, W&B `c1rxa6q6`: PSNR `17.169540`, SSIM `0.548714`, LPIPS `0.441888`, AbsRel `0.187067`, Depth MAE `2.919396`, normal `42.218251`
- R44.02 courtyard fraction `0.625`, lambda `0.002`, decay `7000->14000` to `0.25x`, trained `7000->20000`, W&B `5tleod3c`: PSNR `17.829701`, SSIM `0.561812`, LPIPS `0.493252`, AbsRel `0.147102`, Depth MAE `1.915970`, normal `26.520612`

**Decision**: `SPARSE_DECAY_LONG_HORIZON_REPAIR_PARTIAL_PASS_CLEAN_LONG_RENDER_FAIL`. R44.01 repairs the R43 parking overtraining failure: relative to R43.01b it improves PSNR by `+0.920386`, SSIM by `+0.037679`, LPIPS by `-0.035538`, AbsRel by `-0.006612`, and Depth MAE by `-0.098728`. It also improves the prior sparse-recovery parking rows R40.01/R39.01 on render, but that is not sufficient for a clean-baseline claim. After a corrected long-horizon clean comparison, the strongest parking render row is the clean current-branch 22000-iteration baseline, not R44.01. R44.02 improves the R43 courtyard long row on every tracked metric: PSNR `+0.036665`, SSIM `+0.000836`, LPIPS `-0.003472`, AbsRel `-0.011805`, Depth MAE `-0.075598`, and normal angle `-2.429404`.

**Clean baseline comparison artefacts**:
- `outputs/carnet/meshsplatopt/baseline_vs_method_qualitative/parking_clean_baseline_vs_ours_render_montage.png`
- `outputs/carnet/meshsplatopt/baseline_vs_method_qualitative/parking_clean_baseline_vs_ours_abs_error_montage.png`
- `outputs/carnet/meshsplatopt/baseline_vs_method_qualitative/parking_clean_baseline_vs_ours_summary.md`
- `outputs/carnet/meshsplatopt/best_clean_long_vs_method_long/best_clean_long_vs_method_long_render_montage.png`
- `outputs/carnet/meshsplatopt/best_clean_long_vs_method_long/best_clean_long_vs_method_long_summary.md`
- `docs/car_model/parking_best_clean_long_vs_method_long_report.md`

**Corrected clean-long comparison**: the earlier R16.03 clean 7000-iteration comparison is only a historical weak-clean reference and must not be used as the main claim. A proper same-scene long-horizon clean comparison was run with online W&B:
- clean current-branch `7000` baseline: PSNR `17.204679`, SSIM `0.535045`, LPIPS `0.450750`, AbsRel `0.0761`, Depth MAE `1.7522`, normal `45.5620`, triangles `833775`
- clean current-branch `7000->22000`, W&B `uus7fi39`: PSNR `18.479990`, SSIM `0.634623`, LPIPS `0.346913`, AbsRel `0.082177`, Depth MAE `1.868398`, normal `45.108437`, triangles `8548242`
- clean current-branch `22000->30000`, W&B `2q807xuf`: PSNR `18.408827`, SSIM `0.631504`, LPIPS `0.350967`, AbsRel `0.081639`, Depth MAE `1.865811`, normal `44.838918`, triangles `8548242`

Against the best clean long render baseline, R44.01 is worse on PSNR by `-1.310450`, SSIM by `-0.085909`, LPIPS by `+0.094975`, AbsRel by `+0.104890`, and Depth MAE by `+1.050998`. R44.01 only wins on the normal proxy by `-2.890186` degrees and on topology size (`782982` vs `8548242` triangles). The defensible parking claim is therefore topology/normal Pareto under much lower topology, not render-quality dominance over the strongest clean long baseline.

---

## 2026-05-03 - R45-R48 clean-to-compact repair

**Goal**: repair the R44 clean-baseline failure by finding a route that preserves clean-long render quality while removing most of the clean-long topology.

**Negative controls**:
- R45.01, R44.01 plus full-image clean-render teacher loss, lambda `0.5`, DSSIM `0.2`, W&B `1vmbmftd`: PSNR `16.975172`, SSIM `0.538638`, LPIPS `0.454413`
- R45.02, R44.01 plus full-image clean-render teacher loss, lambda `1.0`, DSSIM `0.4`, W&B `1lsrbnys`: PSNR `16.925661`, SSIM `0.532397`, LPIPS `0.461958`
- R46.01, R44.01 plus counterfactual teacher mask (`teacher_better`, margin `0.005`), W&B `awwaei5j`: PSNR `16.967775`, SSIM `0.535215`, LPIPS `0.455750`

**Negative-control decision**: `LOW_TOPOLOGY_TEACHER_DISTILLATION_REJECTED`. Starting from the 0.78M-triangle R44 checkpoint is too constrained; render-teacher supervision does not recover clean-level appearance or geometry.

**Clean-to-compact runs**:
- R47 prune80: prune the smallest-area 80% of clean 22k triangles, yielding `1709648` triangles and `1322214` vertices. Independent metrics: PSNR `17.9758396`, SSIM `0.5996068`, LPIPS `0.3873217`; geometry: AbsRel `0.0811635`, Depth MAE `1.8489281`, normal `45.0001905`.
- R47 prune90: prune the smallest-area 90% of clean 22k triangles, yielding `854824` triangles and `806482` vertices. Independent metrics: PSNR `16.0933704`, SSIM `0.5029448`, LPIPS `0.4616031`. This is rejected as too aggressive.
- R48.01: recovery from R47 prune80, `22000->26000`, W&B `1n6jv232`. Independent metrics: PSNR `18.6200047`, SSIM `0.6417572`, LPIPS `0.3493703`; geometry: AbsRel `0.0802411`, Depth MAE `1.8474095`, normal `44.7432287`; topology unchanged at `1709648` triangles.
- R49.01: continuation `26000->30000` with the legacy `--skip_restricted_delaunay` control, W&B `xdaixz33`. Independent metrics: PSNR `18.3612633`, SSIM `0.6288872`, LPIPS `0.3608204`; geometry: AbsRel `0.0820096`, Depth MAE `1.8361890`, normal `45.3555216`; topology dropped to `934205` triangles.
- R50.01: true fixed-topology continuation `26000->30000` after adding `--freeze_topology_updates`, W&B `zwafhpte`. Independent metrics: PSNR `18.4548378`, SSIM `0.6287037`, LPIPS `0.3614763`; geometry: AbsRel `0.0809017`, Depth MAE `1.8447213`, normal `45.3189719`; topology preserved at `1709648` triangles.

**Implementation repair**: added `--freeze_topology_updates`. The old `--skip_restricted_delaunay` flag skipped only the Delaunay refresh; the standard 500-step prune/densify branch could still run before `densify_until_iter + 1000`, which is exactly what R49 exposed. The new flag disables both the standard prune/densify branch and the Delaunay refresh for strict topology-frozen continuation.

**Decision**: `CLEAN_TO_COMPACT_RECOVERY_PASS_EARLY_STOP_AT_26K`. R48.01 is the first corrected parking result that beats the strongest clean 22k baseline on independent PSNR (`+0.140015`), SSIM (`+0.007134`), AbsRel (`-0.001936`), and Depth MAE (`-0.020989`) while using 20.0% of the clean long triangles. LPIPS is nearly tied but slightly worse (`+0.002457`). Relative to R44.01, it improves PSNR by `+1.450465`, SSIM by `+0.093043`, LPIPS by `-0.092518`, AbsRel by `-0.106826`, and Depth MAE by `-1.071986`, at the cost of `2.18x` more triangles and a weaker normal proxy. R49 and R50 reject 30k continuation; R48.01 remains the accepted checkpoint.

**Linked artefact**:
- `docs/car_model/parking_clean_to_compact_repair_report.md`

---

## 2026-05-03 - R51-R56 clean-to-compact dominance repair

**Goal**: close the remaining R48 weakness. R48.01 beat clean 22k on PSNR/SSIM/depth but still lost LPIPS by `+0.002457`, so it was not a true all-metric clean-long win.

**Implementation**:
- added optional direct LPIPS training supervision:
  - `--lambda_lpips_loss`
  - `--lpips_loss_start_iter`
  - `--lpips_loss_warmup_iters`
  - `--lpips_loss_max_side`
- default behavior is unchanged because the LPIPS training loss is disabled at lambda `0.0`.

**Negative LPIPS-loss screen**:
- R51.01, R48.01 plus direct LPIPS loss lambda `0.02`, `26000->27000`, W&B `fss9t32k`: training-eval PSNR `18.314338`, SSIM `0.621097`, LPIPS `0.361453`.
- R52.01, R48.01 plus direct LPIPS loss lambda `0.05`, `26000->27000`, W&B `dxzdhl2m`: training-eval PSNR `18.291863`, SSIM `0.619340`, LPIPS `0.355752`.

**Decision**: `DIRECT_LPIPS_LOSS_REJECTED`. Direct LPIPS optimization from the compact R48 checkpoint worsens render quality and does not solve the clean-long comparison. The failure points to topology budget, not a missing perceptual term.

**Less-aggressive clean-to-compact repair**:
- R53 prune70: prune the smallest-area 70% of clean 22k triangles, yielding `2564473` triangles and `1661616` vertices.
- R54 prune75: prune the smallest-area 75% of clean 22k triangles, yielding `2137060` triangles and `1510147` vertices.
- R55 prune65: prune the smallest-area 65% of clean 22k triangles, yielding `2991885` triangles and `1783669` vertices.
- R53.01, R53 prune70 fixed-topology recovery `22000->26000`, W&B `q15qg2b8`: training eval PSNR `18.739616`, SSIM `0.648180`, LPIPS `0.338372`; independent metrics PSNR `18.7057381`, SSIM `0.6478074`, LPIPS `0.3384919`; geometry AbsRel `0.0795553`, Depth MAE `1.8537511`, normal `44.2613910`.
- R54.01, R54 prune75 fixed-topology recovery `22000->26000`, W&B `4cmm2tdb`: training eval PSNR `18.721855`, SSIM `0.646616`, LPIPS `0.342506`. This is promising but not independently promoted because R53 is stronger in the screen.
- R55.01, R55 prune65 fixed-topology recovery `22000->26000`, W&B `ja7t57cx`: training eval PSNR `18.731598`, SSIM `0.647960`, LPIPS `0.336811`; independent metrics PSNR `18.6975975`, SSIM `0.6475888`, LPIPS `0.3369454`; geometry AbsRel `0.0799188`, Depth MAE `1.8624248`, normal `44.2353729`.
- R56.01, R53 true fixed-topology continuation `26000->28000`, W&B `bwf2up51`: training eval PSNR `18.356278`, SSIM `0.623526`, LPIPS `0.367352`. This rejects continuation past 26k for the R53 topology budget.

**Clean-long deltas for R53.01**:
- versus clean 22k: PSNR `+0.225748`, SSIM `+0.013184`, LPIPS `-0.008421`, AbsRel `-0.002622`, Depth MAE `-0.014647`, normal `-0.847046`, triangles `-5983769` (`-69.999999%`).
- versus clean 30k: PSNR `+0.296911`, SSIM `+0.016303`, LPIPS `-0.012475`, AbsRel `-0.002084`, Depth MAE `-0.012060`, normal `-0.577527`, triangles `-5983769` (`-69.999999%`).

**Decision**: `CLEAN_TO_COMPACT_DOMINATES_CLEAN_LONG_BASELINES`. R53.01 is the first corrected parking checkpoint that beats the strongest clean long baselines on independent PSNR, SSIM, LPIPS, sparse COLMAP depth, and normal proxy while retaining only 30% of clean-long triangles. R48.01 remains the more compact 20%-triangle Pareto point; R53.01 is now the headline quality-dominating result.

**Pareto update**: R55.01 becomes the LPIPS/normal Pareto row, with LPIPS `0.3369454` and normal `44.2353729`, but it is not the headline row because it gives back PSNR (`-0.008141`) and uses `427412` more triangles than R53.01. R56.01 confirms that the 26k early stop is not cosmetic; continuing the same fixed topology to 28k sharply worsens all render metrics.

**Linked artefacts**:
- `docs/car_model/parking_clean_to_compact_repair_report.md`
- `assets/meshsplatopt_clean_vs_r53_montage.png`

---

## 2026-05-03 - R15-R17 interface completion

**Goal**: turn the validated R53/R55 results into paper-grade interfaces for full-budget sweeps, ablations, and manuscript packaging.

**Implemented interfaces**:
- `ss3dm_prior/meshsplatopt/evaluation_contracts.py`: shared `MethodResult`, `MetricTargets`, and `PairwiseComparison` contracts for baseline dominance checks.
- `scripts/car_model/meshsplatopt_collect_clean_to_compact_results.py`: writes JSON/CSV/Markdown clean-to-compact tables from independent `results.json` and sparse-geometry JSON files.
- `scripts/car_model/meshsplatopt_run_full_budget_sweep.py`: writes reproducible R15 job manifests and optional shell runner with W&B-enabled train/render/metrics/geometry commands.
- `scripts/car_model/meshsplatopt_run_ablation_suite.py`: writes the R16 14-row ablation contract and evidence status summary.
- `scripts/car_model/meshsplatopt_make_neurips_package.py`: writes R17 paper-package scaffolds and a final go/no-go document.

**Generated artefacts**:
- `outputs/carnet/meshsplatopt/clean_to_compact_tables/clean_to_compact_results.md`
- `outputs/carnet/meshsplatopt/full_budget_sweep/full_budget_jobs.json`
- `outputs/carnet/meshsplatopt/ablation_suite/ablation_suite_contract.md`
- `outputs/carnet/meshsplatopt/neurips_package/manifest.json`
- `docs/car_model/meshsplatopt_stageR15_full_budget_sweep_design.md`
- `docs/car_model/meshsplatopt_stageR16_ablation_design.md`
- `docs/car_model/meshsplatopt_stageR17_paper_package_report.md`

**Current table result**:
- R53.01 passes all clean22k dominance targets under default thresholds, with PSNR `+0.225748`, SSIM `+0.013184`, LPIPS `-0.008421`, AbsRel `-0.002621`, Depth MAE `-0.014647`, normal `-0.847046`, and triangle reduction `0.700000`.
- R55.01 also passes all clean22k dominance targets, with better LPIPS (`-0.009967` delta) and normal (`-0.873064` delta) but lower PSNR than R53 and more triangles.

**Decision**: `R15_R17_INTERFACES_PARTIAL_PASS`. The interfaces are now present and executable. The project is not yet a full NeurIPS main-conference package because R15 still needs cross-scene full-budget replication and R16 still has four interface-only ablations.

---

## 2026-05-03 - R57-R58 public-scene matched clean-to-compact validation

**Goal**: test whether the R53 clean-to-compact result transfers beyond parking under a fair matched continuation: clean 7000-to-9000 versus prune70 compact 7000-to-9000, with W&B online logging and independent render/geometry evaluation.

**Implemented interface repair**:
- Added `scripts/car_model/meshsplatopt_collect_cross_scene_matched_results.py` to collect public-scene matched clean-to-compact tables from independent `results.json` and COLMAP sparse-geometry JSON files.
- Extended `scripts/car_model/meshsplatopt_run_full_budget_sweep.py` with per-job `images` and `resolution` fields, so public scenes no longer rely on parking's fixed loader settings.

**Runs**:
- R57.01 courtyard prune70 recovery `7000->9000`, W&B `kgazucjj`.
- R57.02 courtyard clean continuation `7000->9000`, W&B `ucqyn1ym`.
- R58.01 bonsai prune70 recovery `7000->9000`, W&B `82v2cg9z`.
- R58.02 bonsai clean continuation `7000->9000`, W&B `ulv6dpku`.

**Independent matched results**:
- Courtyard compact versus clean: PSNR `-0.001726`, SSIM `-0.000522`, LPIPS `+0.027805`, AbsRel `+0.035424`, Depth MAE `+0.209014`, normal `-1.032962`, triangle reduction `0.700000`. This is a controlled failure on render/depth.
- Bonsai compact versus clean: PSNR `+0.280336`, SSIM `+0.017475`, LPIPS `-0.007539`, AbsRel `-0.006582`, Depth MAE `-0.062115`, normal `-0.515667`, triangle reduction `0.700000`. This is a public-scene all-metric dominance result.

**Decision**: `PUBLIC_SCENE_REPLICATION_PARTIAL_PASS`. The method now has one strong parking all-metric long-budget result and one public-scene matched-screen all-metric result, plus one public-scene negative that identifies scene sensitivity. The work is materially stronger than before, but it still needs either another public-scene positive or a selector that predicts compaction success before a NeurIPS-main claim is defensible.

**Linked artefacts**:
- `docs/car_model/meshsplatopt_stageR57_R58_cross_scene_matched_report.md`
- `outputs/carnet/meshsplatopt/cross_scene_clean_to_compact_tables/cross_scene_clean_to_compact_results.md`

---

## 2026-05-04 - Final F0 current-state audit and claim reset

**Goal**: stop blind trial-and-error and align the remaining NeurIPS repair work to the new final planning prompt.

**Audit actions**:
- read the required final-planning context, corrected clean-long reports, R14/R15/R28-R30/R57-R58 evidence, and the original MeshSplatOpt repair RFC;
- ran `/home/peilincai/micromamba/envs/mesh_splatting/bin/python -m compileall scripts/car_model ss3dm_prior utils -q`, which passed;
- recorded branch `neurips-meshsplatopt-repair`, commit `97b9d6d`, and current dirty/untracked files;
- completed independent sparse-geometry evaluation for the R59/R60 matched room/counter screens.

**R59/R60 addendum**:
- R59 room compact beats the matched clean row on independent render metrics (PSNR `+0.438885`, SSIM `+0.005325`, LPIPS `-0.000389`) while regressing sparse geometry slightly (AbsRel `+0.002058`, Depth MAE `+0.006847`, normal `+0.610254`). This is useful as render-positive evidence but not an all-metric cross-scene pass.
- R60 counter is mixed/negative: compact improves PSNR by `+0.134289` but worsens SSIM, LPIPS, AbsRel, Depth MAE, and normal angle. This reinforces that area-only prune70 requires a scene-aware selector.

**Decision**: `FINAL_F0_AUDIT_PASS_PROCEED_TO_F1`. The final claim is reset to evidence-certified compact-repair optimization. R53/R48/R55 are stronger than R44 and should replace R44 as the parking headline. Snap/fill remain rollback-compatible edit interfaces and diagnostics, not the headline claim.

**Linked artefact**:
- `docs/car_model/final_stageF0_current_state_audit.md`

---

## 2026-05-04 - Final F1 paper story and method spec

**Goal**: convert the F0 claim reset into a paper-facing method spec that can guide implementation, baselines, figures, and reviewer-risk checks.

**Spec decision**: `FINAL_F1_METHOD_SPEC_PASS`. MeshSplatOpt is now framed as counterfactually certified compact-repair optimization. The one-paragraph story leads with CSEF-scored compaction/repair candidates, rollback-compatible gates, strict topology-frozen recovery, and independent render/sparse-geometry certification against the strongest matched clean baseline.

**Load-bearing branch**:
- R53.01 is the headline parking result because it beats clean 22k/30k on independent render, sparse depth, normal proxy, and topology.
- R48.01 is the more compact 20-percent-triangle Pareto row.
- R55.01 is the LPIPS/normal Pareto row.
- R58 is the public-scene all-metric positive.
- R57/R60 and R59's geometry tradeoff are retained as selector-motivation evidence.

**Guardrails**:
- R44 is explicitly demoted to topology/normal Pareto evidence, not a render win.
- Snap/fill/object-prior/ground-void edits are optional repair branches until equal-budget controls prove benefit.
- The spec forbids long-method-vs-short-clean headline comparisons and training-metric/independent-metric mixing.

**Linked artefact**:
- `docs/car_model/final_stageF1_method_spec.md`

---

## 2026-05-04 - Final F2 baseline registry and metric-integrity collector

**Goal**: build one fair-baseline registry so final tables cannot compare long method runs against short clean baselines or mix training-time metrics with independent metrics.

**Implementation**:
- added `scripts/car_model/final_collect_baselines_and_results.py`;
- added `docs/car_model/final_stageF2_baseline_registry_design.md`;
- added `docs/car_model/final_stageF2_baseline_registry_report.md`.

**Collector outputs**:
- `outputs/carnet/meshsplatopt/final_baseline_registry/final_results.json`
- `outputs/carnet/meshsplatopt/final_baseline_registry/final_results.csv`
- `outputs/carnet/meshsplatopt/final_baseline_registry/final_results.md`

**Integrity gate**:
- `r53_vs_clean22k_reproduced`: `true`;
- `r44_flagged_render_losing_vs_clean22k`: `true`;
- `forbidden_long_method_vs_clean7k_headline`: `false`.

**Decision**: `FINAL_F2_BASELINE_REGISTRY_PASS`. The collector makes R53 the clean-to-compact headline, keeps R44 as a documented render-losing topology/normal Pareto point, and explicitly flags non-independent or missing metrics.

---

## 2026-05-04 - Final F3 cross-scene clean-to-compact plan

**Goal**: stop launching cross-scene compaction blindly by naming exact clean baselines, missing clean-long commands, output paths, and launch order.

**Audit result**:
- parking has clean long 22k/30k and remains the headline validated scene;
- bonsai/courtyard/room/counter currently have matched 9k clean continuations, not true clean-long baselines;
- flowers is not present under `/data/peilincai/mesh_datasets`;
- R58 bonsai is the strongest public-scene positive, so the first missing-baseline run should be `finalF3_bonsai_clean_long_9000to22000`.

**Decision**: `FINAL_F3_CROSS_SCENE_PLAN_PASS`. Do not launch broad cross-scene compaction before the bonsai clean-long baseline exists and F4's non-area CSEF-compatible selector passes. The plan names the sweep fractions, output layout, recovery template, and scene risk levels.

**Linked artefact**:
- `docs/car_model/final_stageF3_cross_scene_compact_plan.md`

---

## 2026-05-04 - Final F4 CSEF-compatible compact selector

**Goal**: move beyond smallest-area-only compaction by implementing a CSEF-compatible face selector with protected repair regions and count-matched controls.

**Implementation**:
- added `ss3dm_prior/meshsplatopt/compact_selector.py`;
- added `scripts/car_model/meshsplatopt_select_compaction_candidates.py`;
- added `scripts/car_model/smoke_test_final_stageF4_compact_selector.py`;
- added `docs/car_model/final_stageF4_compact_selector_design.md`;
- added `docs/car_model/final_stageF4_compact_selector_report.md`.

**Smoke result**:

```text
F4 selector smoke PASS: area=[2, 3, 6] csef=[2, 3, 7] random=[2, 3, 7]
```

**Decision**: `FINAL_F4_COMPACT_SELECTOR_PASS`. The boundary-protected CSEF selector differs from area-only on synthetic data: it protects a high-debt repair region that area-only would prune, while selecting redundant small triangles and a floater. This satisfies the non-area selector gate and enables F5 real-checkpoint compaction.

---

## 2026-05-04 - Final F5 real-checkpoint compaction

**Goal**: apply compaction candidates to real Mesh Splatting checkpoints and verify that compact checkpoints retain a renderable model layout.

**Implementation**:
- added `ss3dm_prior/meshsplatopt/checkpoint_compaction.py`;
- added `scripts/car_model/meshsplatopt_apply_compaction_to_checkpoint.py`;
- added `scripts/car_model/smoke_test_final_stageF5_checkpoint_compaction.py`;
- added `docs/car_model/final_stageF5_checkpoint_compaction_report.md`.

**Smoke result**:

```text
F5 checkpoint compaction smoke PASS: area_triangles=2564473 csef_triangles=2564473
```

**Render smoke**:
- command used low resolution (`--resolution 16`) because all GPUs were already high-memory occupied;
- CSEF70 compact checkpoint loaded through `render.py`;
- render path reported `2,564,473` triangles and `1,661,616` vertices and rendered all 54 test views.

**Decision**: `FINAL_F5_CHECKPOINT_COMPACTION_PASS`. Area70 exactly reproduces the R53 pre-recovery topology count, and CSEF70 produces a valid renderable checkpoint. Proceed to F6 strict topology-frozen recovery runner.

---

## 2026-05-04 - Final F6 strict topology-frozen recovery runner

**Goal**: make compact recovery reproducible and prevent the old `--skip_restricted_delaunay` topology-control ambiguity.

**Implementation**:
- added `scripts/car_model/meshsplatopt_run_strict_compact_recovery.py`;
- added `docs/car_model/final_stageF6_strict_recovery_design.md`;
- added `docs/car_model/final_stageF6_strict_recovery_report.md`.

**R53 contract audit**:
- load checkpoint 22k: `2,564,473` triangles, `1,661,616` vertices;
- final checkpoint 26k: `2,564,473` triangles, `1,661,616` vertices;
- `topology_unchanged`: `true`;
- W&B run: `q15qg2b8`.

**Decision**: `FINAL_F6_STRICT_RECOVERY_RUNNER_PASS`. The runner writes exact W&B-enabled train, render, metrics, and geometry commands and verifies the R53 topology-freeze contract. No new long training was launched because GPUs were high-memory occupied.

---

## 2026-05-04 - Final F7 parking compact Pareto

**Goal**: stop relying on a single area-only compact result by implementing a reproducible parking Pareto sweep and validating the first non-area CSEF boundary-protected compact recovery.

**Implementation**:
- added `scripts/car_model/final_run_parking_compact_pareto.py`;
- added `scripts/car_model/final_collect_parking_compact_pareto.py`;
- added `scripts/car_model/meshsplatopt_eval_render_metrics_single_iteration.py`;
- added `docs/car_model/final_stageF7_parking_pareto_report.md`.

**Validated run**:
- selector: `csef_low_evidence_boundary_protected`;
- prune fraction: 70 percent;
- W&B run: `oqpkykcw`;
- topology: `2,564,473` triangles and `1,661,616` vertices at both 22k and 26k;
- independent render metrics at 26k: PSNR `18.706079`, SSIM `0.647764`, LPIPS `0.338282`;
- sparse geometry at 26k: AbsRel `0.079404`, Depth MAE `1.852816`, Normal `44.204497`.

**Decision**: `FINAL_F7_PARKING_PARETO_PASS`. F7.csef70 beats clean22k on render and sparse geometry while reducing triangles by 70 percent. At identical topology to R53, it slightly improves PSNR, LPIPS, AbsRel, Depth MAE, and normal angle, with only a negligible SSIM decrease.

---

## 2026-05-04 - Final F8 cross-scene pilot setup and bonsai clean-long launch

**Goal**: move beyond parking by creating a fair cross-scene compact pilot that refuses short-baseline comparisons and starts the first missing clean-long public-scene baseline.

**Implementation**:
- added `scripts/car_model/final_run_cross_scene_compact_pilot.py`;
- added `scripts/car_model/final_collect_cross_scene_compact_pilot.py`;
- added `docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md`.

**Launched run**:
- scene: `bonsai`;
- output: `outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000`;
- continuation: 9k to 22k from `stageR58_02_bonsai_clean_continue_7000to9000`;
- W&B run: `r8ozggn1`;
- W&B URL: `https://wandb.ai/karamazovaniki-university-of-southern-california/spcarnet_meshprior/runs/r8ozggn1`.

**Decision**: `FINAL_F8_IN_PROGRESS`. The interface and honest collector are in place, but F8 cannot pass until at least one non-parking scene completes clean-long plus compact recovery and at least two scenes satisfy the fair clean-long gate.

---

## 2026-05-04 - Final F8 cross-scene compact pilot pass

**Goal**: complete the fair non-parking evidence that was missing from F8 and stop comparing short clean baselines against long compact recoveries.

**Completed evidence**:
- bonsai clean-long 9k->22k W&B run: `r8ozggn1`;
- bonsai CSEF50 strict topology-frozen recovery W&B run: `irdsa4c8`;
- bonsai CSEF70 strict topology-frozen recovery W&B run: `ou72x2zw`;
- courtyard clean-long successful retry W&B run: `5ptlupv8`;
- courtyard CSEF50 strict topology-frozen recovery W&B run: `jz93wrbc`.

**Resource note**:
- courtyard clean-long retry `eqjygth6` failed near the final stage with a CUDA OOM because another same-card process occupied roughly 36GB;
- retry `5ptlupv8` kept online W&B scalar logging enabled but disabled inline image logging and deferred render/metrics/geometry to independent commands, which completed successfully.

**Independent results**:

| scene | method | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal | decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| bonsai | clean-long 22k | 88,460 | - | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 | baseline |
| bonsai | CSEF50 26k | 44,230 | 50.0% | 10.957497 | 0.224758 | 0.586415 | 0.185180 | 1.737815 | 43.493975 | PASS |
| courtyard | clean-long 22k | 1,677,484 | - | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 | baseline |
| courtyard | CSEF50 26k | 838,742 | 50.0% | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 | PASS |

**Decision**: `FINAL_F8_CROSS_SCENE_COMPACT_PILOT_PASS`. The same CSEF boundary-protected 50 percent compact setting passes on two non-parking scenes against fair clean-long baselines. Courtyard is the strongest transfer result so far because it removes half the triangles and improves PSNR, SSIM, LPIPS, AbsRel, and Depth MAE against a 1.68M-triangle clean baseline.

---

## 2026-05-04 - Final F9 third-scene room and qualitative evidence

**Goal**: push beyond the two-scene F8 transfer gate by adding a third public scene and generating cross-scene qualitative material.

**W&B runs**:
- room clean-long 9k->22k: `kqyusaoe`;
- room CSEF50 strict topology-frozen recovery 22k->26k: `pb1tg4p2`.

**Room independent results**:

| method | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | - | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| CSEF50 26k | 42,253 | 50.0% | 14.387163 | 0.414954 | 0.568281 | 0.225027 | 1.603030 | 54.642793 |

**Deltas**:
- PSNR `+0.128784`;
- SSIM `+0.014090`;
- LPIPS `-0.010638`;
- AbsRel `+0.018745`;
- Depth MAE `+0.122800`;
- Normal `-0.799860`.

**Qualitative output**:
- `outputs/carnet/meshsplatopt/final_stageF9_qualitative_evidence/mesh_splat_opt_cross_scene_qualitative_montage.png`;
- `outputs/carnet/meshsplatopt/final_stageF9_qualitative_evidence/mesh_splat_opt_cross_scene_qualitative_report.md`.

**Decision**: `FINAL_F9_THIRD_SCENE_ROOM_PASS`. Room passes the same CSEF50 gate used in F8, giving the method three non-parking transfer-style positives when counting bonsai, courtyard, and room, plus the parking anchor. The remaining gap is to refresh the montage with room included and add a fourth public scene such as counter.

---

## 2026-05-04 - Final F10 fourth-scene counter Pareto pass

**Goal**: add a fourth public-scene validation point and explicitly test whether the counter scene prefers the fixed 50 percent compact point or a gentler Pareto point.

**W&B runs**:
- counter clean-long 9k->22k: `jl5vtp4m`;
- counter CSEF50 strict topology-frozen recovery 22k->26k: `58od8x2f`;
- counter CSEF50 extension 26k->30k: `erjis9bc`;
- counter CSEF40 failed first launch due missing copied compact checkpoint: `ag6wtjwh`;
- counter CSEF40 retry 22k->26k: `glzzth4b`.

**Independent results**:

| method | iteration | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long | 22000 | 83,834 | - | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| CSEF50 | 26000 | 41,917 | 50.0% | 14.077559 | 0.498974 | 0.468391 | 0.094731 | 0.438932 | 43.823390 |
| CSEF50 extended | 30000 | 41,917 | 50.0% | 14.099902 | 0.485554 | 0.479640 | 0.092779 | 0.431583 | 44.029069 |
| CSEF40 | 26000 | 50,300 | 40.0% | 14.212033 | 0.518401 | 0.450481 | 0.085542 | 0.406373 | 43.476972 |

**Decision**: `FINAL_F10_FOURTH_SCENE_COUNTER_PARETO_PASS`. The strict 50 percent point is a boundary case on counter because SSIM misses the gate by `0.003827`, and the 30k extension is rejected because it worsens SSIM and LPIPS. The 40 percent CSEF Pareto point is strong: it removes 33,534 triangles and improves PSNR, SSIM, LPIPS, and Normal against the fair clean-long baseline, with mild depth regressions still inside the same tolerance.

**Qualitative evidence**:
- `outputs/carnet/meshsplatopt/final_stageF10_qualitative_evidence/room_counter_clean_vs_csef_montage.png`;
- `outputs/carnet/meshsplatopt/final_stageF10_qualitative_evidence/room_counter_clean_vs_csef_report.md`.

---

## 2026-05-04 - Final F11-F15 evidence package, assets, and go/no-go

**Goal**: convert the newly validated F8-F10 long-baseline evidence into a traceable paper package instead of leaving results scattered across stage logs.

**Created scripts**:
- `scripts/car_model/final_collect_ablation_suite.py`;
- `scripts/car_model/final_collect_multiscene_package.py`;
- `scripts/car_model/final_make_paper_assets.py`;
- `scripts/car_model/final_run_multiscene_package.py`.

**Created reports**:
- `docs/car_model/final_stageF11_ablation_suite_report.md`;
- `docs/car_model/final_stageF12_multiscene_package_report.md`;
- `docs/car_model/final_stageF13_paper_assets_report.md`;
- `docs/car_model/final_meshsplatopt_neurips_manuscript_skeleton.md`;
- `docs/car_model/final_meshsplatopt_related_work_notes.md`;
- `docs/car_model/final_meshsplatopt_bib_plan.md`;
- `docs/car_model/final_stageF15_neurips_go_no_go.md`.

**Generated assets**:
- `outputs/carnet/meshsplatopt/final_multiscene_package/main_quantitative_table.csv`;
- `outputs/carnet/meshsplatopt/final_multiscene_package/negative_result_table.csv`;
- `outputs/carnet/meshsplatopt/final_paper_assets/paper_assets_manifest.json`;
- `outputs/carnet/meshsplatopt/final_paper_assets/meshsplatopt_method_diagram.png`;
- `outputs/carnet/meshsplatopt/final_paper_assets/triangle_count_bar_chart.png`.

**Main package result**: `FINAL_F12_MULTISCENE_PACKAGE_PASS_WITH_ABLATION_GAPS`. Five scenes now have scene-matched clean-long versus compact-recovery comparisons, with 40-70 percent triangle reduction and non-regressing/improving render metrics under the accepted per-scene operating point.

**Go/no-go**: `NEURIPS_BORDERLINE_NEEDS_STRICT_ABLATIONS`. The scene-count and long-baseline weaknesses are now largely repaired. The remaining critical risk is strict ablation coverage: area-only versus CSEF, random same-count compaction, no-sparse-depth, no-freeze, and posthoc simplification baselines.

---

## 2026-05-04 - Final F16 counter random same-count control

**Goal**: reduce the strongest reviewer risk that counter CSEF40 is just arbitrary 40 percent pruning plus recovery.

**Run**:
- selector: `random_same_count`;
- scene: `counter`;
- seed: `20260504`;
- clean source: `outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000`;
- compact model: `outputs/carnet/meshsplatopt/final_stageF16_counter_random_same_count_control/prune40/compact_model`;
- recovery model: `outputs/carnet/meshsplatopt/final_stageF16_counter_random_same_count_control/prune40/recovery_model`;
- W&B: `0hlz8q0u`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| CSEF40 26k | 50,300 | 14.212033 | 0.518401 | 0.450481 | 0.085542 | 0.406373 | 43.476972 |
| random40 26k | 50,300 | 13.875822 | 0.482349 | 0.485052 | 0.099779 | 0.444684 | 43.941494 |

**Area40 follow-up**:
- compact model: `outputs/carnet/meshsplatopt/final_stageF16_counter_area_selector_control/prune40/compact_model`;
- recovery model: `outputs/carnet/meshsplatopt/final_stageF16_counter_area_selector_control/prune40/recovery_model`;
- W&B: `85lmm0lr`;
- independent metrics: PSNR `14.314330`, SSIM `0.536892`, LPIPS `0.431104`, AbsRel `0.072751`, Depth MAE `0.357914`, Normal `43.715882`.

**Decision**: `FINAL_F16_COUNTER_SELECTOR_CONTROL_PASS_AREA40_BEST`. Random same-count pruning fails the clean-long gate and is much worse than CSEF40 at the same triangle count, proving that arbitrary pruning is not enough. However, area40 is stronger than CSEF40 on counter and becomes the new recommended counter row. The paper story must be updated honestly: counter supports compact-recovery strongly, but does not support a universal CSEF-over-area selector claim.

---

## 2026-05-04 - Final F17 courtyard selector ablation

**Goal**: replicate selector controls on a larger public scene after counter showed that area40 can outperform CSEF40.

**W&B runs**:
- CSEF50: `jz93wrbc`;
- area50: `hctwxtbe`;
- random50: `faz0c00o`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| area50 26k | 838,742 | 12.552895 | 0.338469 | 0.544993 | 0.324157 | 3.630241 | 40.907990 |
| random50 26k | 838,742 | 11.383848 | 0.264778 | 0.587667 | 0.371186 | 4.015910 | 41.158282 |

**Decision**: `FINAL_F17_COURTYARD_SELECTOR_ABLATION_PASS_STRUCTURED_SELECTION`. Random same-count pruning fails badly, so arbitrary topology removal is not sufficient. CSEF50 and area50 are near-tied on render, but CSEF50 remains the geometry-balanced courtyard row because it has better PSNR, AbsRel, Depth MAE, and Normal.

---

## 2026-05-04 - Final F18 counter no-freeze control

**Goal**: test whether strict topology freezing is a real mechanism or only redundant syntax after `--skip_restricted_delaunay`.

**Run**:
- scene: `counter`;
- source compact checkpoint: `outputs/carnet/meshsplatopt/final_stageF16_counter_area_selector_control/prune40/compact_model`;
- recovery checkpoint: `outputs/carnet/meshsplatopt/final_stageF18_counter_no_freeze_control/area40/recovery_model`;
- schedule: `22000->26000`;
- W&B: `g5pmw9lk`;
- deliberate control: omitted `--freeze_topology_updates`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| area40 frozen 26k | 50,300 | 14.314330 | 0.536892 | 0.431104 | 0.072751 | 0.357914 | 43.715882 |
| area40 no-freeze 26k | 18,693 | 13.641099 | 0.467266 | 0.483981 | 0.104043 | 0.442218 | 45.148206 |

**Decision**: `FINAL_F18_COUNTER_NO_FREEZE_CONTROL_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`. `--skip_restricted_delaunay` alone still allows standard topology changes. The no-freeze control collapses the compact topology and loses badly on independent render and COLMAP sparse-geometry metrics, proving that strict topology freezing is load-bearing for the final compact-recovery contract.

**Documentation correction**: the final F8-F18 compact-recovery main rows use independent COLMAP sparse geometry evaluation, but their training commands did not enable sparse-depth loss. Sparse-depth-guided recovery remains an earlier useful branch and should not be described as the active final main-row recovery mechanism unless new rows explicitly enable it.

---

## 2026-05-04 - Final F19 room selector ablation

**Goal**: extend the selector ablation from counter/courtyard to a third public scene and check whether room prefers CSEF50, area50, or random50 at the same 50 percent compact target.

**W&B runs**:
- area50: `eagvu7em`;
- random50: `p0vxzf01`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| CSEF50 26k | 42,253 | 14.387163 | 0.414954 | 0.568281 | 0.225027 | 1.603030 | 54.642793 |
| area50 26k | 42,253 | 14.844683 | 0.461875 | 0.530461 | 0.185703 | 1.353216 | 54.615295 |
| random50 26k | 42,253 | 13.428182 | 0.345278 | 0.609467 | 0.272092 | 1.873476 | 54.469912 |

**Decision**: `FINAL_F19_ROOM_SELECTOR_ABLATION_PASS_AREA50_BEST_RANDOM_FAIL`. Area50 becomes the new room best row and improves every tracked independent render/geometry metric versus clean-long while halving triangles. Random50 fails badly at the same triangle count. This upgrades the selector-control evidence to three scenes and strengthens the non-random compaction claim, while keeping the selector conclusion honest: area is strongest on counter/room, CSEF is slightly more geometry-balanced on courtyard.

---

## 2026-05-04 - Final F20 room posthoc QEM baseline

**Goal**: remove the posthoc QEM/decimation missing-baseline risk with a real Open3D quadric-decimation checkpoint and equal fixed-topology recovery budget.

**Implementation**:
- added `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`;
- Open3D QEM compacted `room` clean-long from `84,506` to `42,253` triangles;
- transferred vertex tensors by nearest source vertex and face tensors by nearest source face centroid;
- topology audit: `degenerate_face_count=0`, `invalid_index_count=0`.

**W&B**: `9wri3owt`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| area50 26k | 42,253 | 14.844683 | 0.461875 | 0.530461 | 0.185703 | 1.353216 | 54.615295 |
| Open3D QEM50 26k | 42,253 | 15.061190 | 0.481082 | 0.516805 | 0.181129 | 1.345221 | 54.900779 |

**Decision**: `FINAL_F20_ROOM_POSTHOC_QEM_STRONG_PASS_SUPERSEDES_AREA50_ON_RENDER_DEPTH`. QEM50 plus strict topology-frozen recovery is the new strongest room row on render, AbsRel, and Depth MAE, while area50 remains slightly better on normal. This is not a weak baseline; it must be reported honestly. The method framing should shift from universal CSEF/area superiority to a stronger and cleaner claim: MeshSplatOpt is a fixed-topology certified recovery framework that can evaluate and absorb compact operators, with random pruning rejected and QEM emerging as a strong collapse-style operator on room.

---

## 2026-05-04 - Final F21 counter posthoc QEM baseline

**Goal**: replicate the F20 Open3D QEM posthoc simplification baseline beyond `room`, using `counter` where area40 had been the strongest compact row.

**Implementation**:
- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`;
- Open3D QEM compacted `counter` clean-long from `83,834` to `50,300` triangles;
- transferred vertex tensors by nearest source vertex and face tensors by nearest source face centroid;
- topology audit: `degenerate_face_count=0`, `invalid_index_count=0`.

**W&B**: `kr8565st`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| CSEF40 26k | 50,300 | 14.212033 | 0.518401 | 0.450481 | 0.085542 | 0.406373 | 43.476972 |
| area40 26k | 50,300 | 14.314330 | 0.536892 | 0.431104 | 0.072751 | 0.357914 | 43.715882 |
| random40 26k | 50,300 | 13.875822 | 0.482349 | 0.485052 | 0.099779 | 0.444684 | 43.941494 |
| Open3D QEM40 26k | 50,300 | 14.409434 | 0.547456 | 0.420855 | 0.068076 | 0.338664 | 43.716007 |

**Decision**: `FINAL_F21_COUNTER_POSTHOC_QEM_STRONG_PASS_SUPERSEDES_AREA40_ON_RENDER_DEPTH`. QEM40 becomes the new strongest counter row on render, AbsRel, and Depth MAE, while normal is effectively tied with area40. This upgrades F12's counter main row and reduces the posthoc simplification missing-baseline risk from one scene to two scenes. The paper framing should treat QEM as a strong compact operator under the fixed-topology recovery framework, not as a weak baseline.

---

## 2026-05-04 - Final F22 bonsai posthoc QEM baseline

**Goal**: replicate the Open3D QEM posthoc simplification baseline on a third scene after the positive `room` and `counter` QEM rows.

**Implementation**:
- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`;
- Open3D QEM compacted `bonsai` clean-long from `88,460` to `44,230` triangles;
- transferred vertex tensors by nearest source vertex and face tensors by nearest source face centroid;
- topology audit: `degenerate_face_count=0`, `invalid_index_count=0`.

**W&B**: `bsed9ik1`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| CSEF50 26k | 44,230 | 10.957497 | 0.224758 | 0.586415 | 0.185180 | 1.737815 | 43.493975 |
| Open3D QEM50 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |

**Decision**: `FINAL_F22_BONSAI_POSTHOC_QEM_STRONG_PASS_SUPERSEDES_CSEF50_ON_RENDER`. QEM50 becomes the new strongest bonsai row on render, AbsRel, and normal, while CSEF50 remains better on Depth MAE. This upgrades F12's bonsai main row and means QEM is now a replicated strong compact operator on three scenes, not a one-off baseline.

---

## 2026-05-04 - Final F23 courtyard posthoc QEM baseline

**Goal**: replicate the Open3D QEM posthoc simplification baseline on a larger scene after positive bonsai, room, and counter QEM rows.

**Implementation**:
- script: `scripts/car_model/meshsplatopt_apply_open3d_qem_decimation_to_checkpoint.py`;
- Open3D QEM compacted `courtyard` clean-long from `1,677,484` to `838,741` triangles;
- transferred vertex tensors by nearest source vertex and face tensors by nearest source face centroid;
- topology audit: `degenerate_face_count=0`, `invalid_index_count=0`;
- failed launch `tuqvfmaz` used an incorrect dataset path and is excluded from results;
- accepted W&B run: `60tdigdj`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| area50 26k | 838,742 | 12.552895 | 0.338469 | 0.544993 | 0.324157 | 3.630241 | 40.907990 |
| random50 26k | 838,742 | 11.383848 | 0.264778 | 0.587667 | 0.371186 | 4.015910 | 41.158282 |
| Open3D QEM50 26k | 838,741 | 12.530957 | 0.339798 | 0.543378 | 0.332515 | 3.694743 | 40.804188 |

**Decision**: `FINAL_F23_COURTYARD_POSTHOC_QEM_MIXED_PASS_CSEF50_REMAINS_MAIN`. QEM50 improves SSIM, LPIPS, and normal relative to CSEF50, but is weaker on PSNR, AbsRel, and Depth MAE. CSEF50 remains the courtyard main row. The QEM claim is now stronger but more nuanced: it is a strong compact operator under fixed-topology recovery, not a universally dominant operator.

---

## 2026-05-04 - Final F24 room QEM no-freeze control

**Goal**: replicate the no-freeze failure mode beyond counter and test whether strict topology freeze remains load-bearing for a strong QEM compact operator.

**Run**:
- scene: `room`;
- source compact checkpoint: `outputs/carnet/meshsplatopt/final_stageF20_room_posthoc_qem_baseline/prune50/compact_model`;
- recovery checkpoint: `outputs/carnet/meshsplatopt/final_stageF24_room_qem_no_freeze_control/prune50/recovery_model`;
- schedule: `22000->26000`;
- W&B: `byjyx9zx`;
- deliberate control: omitted `--freeze_topology_updates`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| QEM50 frozen 26k | 42,253 | 15.061190 | 0.481082 | 0.516805 | 0.181129 | 1.345221 | 54.900779 |
| QEM50 no-freeze 26k | 20,742 | 13.789439 | 0.399147 | 0.567857 | 0.212804 | 1.497902 | 55.443601 |

**Decision**: `FINAL_F24_ROOM_QEM_NO_FREEZE_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`. No-freeze collapses the compact topology from `42,253` to `20,742` triangles and loses badly to frozen QEM50 on every independent render and sparse-geometry metric. Together with F18 counter no-freeze, this establishes strict topology freezing as a replicated load-bearing mechanism.

---

## 2026-05-04 - Final F25 parking QEM target-failure control and headline row cleanup

**Goal**: close a fairness gap in the final package: parking had strong area and CSEF 70 percent rows, but did not yet test whether Open3D QEM could provide a matched posthoc simplification control at the same `2,564,473`-triangle budget.

**Executed control**:
- source: clean-long 22k parking checkpoint `outputs/carnet/meshprior/parking_phone_tiny/stage44_clean_long/current_branch_clean_7000to22000/model`;
- requested target: `2,564,473` triangles, matching R53/F7;
- output: `outputs/carnet/meshsplatopt/final_stageF25_parking_posthoc_qem_baseline/prune70/compact_model`;
- observed topology: `8,548,242 -> 8,125,970` triangles and `2,286,499 -> 1,897,393` vertices;
- invalid indices / degenerate faces: `0 / 0`.

**Package cleanup**:
- promoted F7 CSEF70 to the parking main row in `final_collect_multiscene_package.py` because it slightly supersedes R53 at the same topology on PSNR, LPIPS, AbsRel, Depth MAE, and normal angle, with negligible SSIM loss;
- added F25 to the negative-result table and ablation registry as an unmatched-compression QEM failure;
- left R53.01 as the strong same-count area-only control.

**Decision**: `FINAL_F25_PARKING_QEM70_REJECT_UNMATCHED_COMPRESSION`. Open3D QEM does not provide a fair matched 70 percent parking baseline because it reaches only `4.94%` triangle removal on the 8.55M-triangle mesh. No W&B recovery was launched for this row because it would retain `3.17x` more triangles than the accepted compact method and would create a misleading comparison. The final parking headline is now the already W&B-validated F7 CSEF70 run (`oqpkykcw`), with R53 as area-only control and F25 as a documented posthoc simplification failure.

---

## 2026-05-04 - Final F26 bonsai selector ablation

**Goal**: close the bonsai selector-control gap with matched long-horizon area and random rows at the same 50 percent topology budget as CSEF50/QEM50.

**W&B runs**:
- area50 strict recovery `22000->26000`: `a29ayt8w`;
- random50 strict recovery `22000->26000`: `noqp4nhp`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| CSEF50 26k | 44,230 | 10.957497 | 0.224758 | 0.586415 | 0.185180 | 1.737815 | 43.493975 |
| Open3D QEM50 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |
| area50 26k | 44,230 | 11.072339 | 0.242361 | 0.570040 | 0.179402 | 1.755109 | 42.834537 |
| random50 26k | 44,230 | 10.725461 | 0.197036 | 0.603335 | 0.210644 | 1.736676 | 43.797014 |

**Decision**: `FINAL_F26_BONSAI_SELECTOR_ABLATION_PASS_AREA_PARETO_RANDOM_FAIL`. Random same-count pruning fails as a clean-long control and loses badly to structured selectors at the same triangle count. Area50 is a strong geometry/perceptual Pareto row: it is slightly behind QEM50 on PSNR and SSIM, but better on LPIPS, AbsRel, Depth MAE, and normal. QEM50 remains the bonsai render-headline row; area50 becomes the bonsai selector Pareto control.

---

## 2026-05-04 - Final F27/F28 bonsai freeze and sparse-depth compact recovery

**Goal**: remove two remaining F11 weaknesses on a fast public scene: replicate no-freeze failure beyond counter/room, and run a final compact-recovery row that explicitly enables sparse COLMAP depth loss.

**W&B runs**:
- F27 QEM50 no-freeze `22000->26000`: `0wskvq3h`;
- F28 QEM50 + sparse-depth strict recovery `22000->26000`: `07k1ii1d`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 88,460 | 10.944348 | 0.222848 | 0.586158 | 0.194249 | 1.816410 | 45.358356 |
| QEM50 frozen 26k | 44,230 | 11.082405 | 0.243249 | 0.570177 | 0.182966 | 1.793852 | 42.889339 |
| QEM50 no-freeze 26k | 17,962 | 10.560091 | 0.176992 | 0.609218 | 0.229736 | 1.718488 | 46.233158 |
| QEM50 + sparse-depth 26k | 44,230 | 11.081614 | 0.243248 | 0.569658 | 0.181698 | 1.779783 | 42.425734 |

**Decision F27**: `FINAL_F27_BONSAI_QEM_NO_FREEZE_FAIL_SUPPORTS_STRICT_TOPOLOGY_FREEZE`. No-freeze collapses topology and loses badly to frozen QEM50 on PSNR, SSIM, LPIPS, AbsRel, and normal. With counter and room, strict topology freeze is now replicated across three scenes.

**Decision F28**: `FINAL_F28_BONSAI_QEM_SPARSE_DEPTH_PARETO_PASS`. Explicit sparse COLMAP depth loss is active and improves LPIPS, AbsRel, Depth MAE, and normal relative to QEM50 at identical topology, while giving back only `0.000791 dB` PSNR and `0.000001` SSIM. This becomes the bonsai geometry/perceptual headline and the first final compact-recovery row that explicitly supports the sparse-depth-guided recovery claim.

---

## 2026-05-04 - Final F29 room sparse-depth replication

**Goal**: replicate the explicit sparse COLMAP depth compact-recovery branch beyond bonsai on the accepted room QEM50 topology.

**W&B run**:
- F29 QEM50 + sparse-depth strict recovery `22000->26000`: `wl94n5bp`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 84,506 | 14.258379 | 0.400864 | 0.578919 | 0.206282 | 1.480230 | 55.442653 |
| QEM50 frozen 26k | 42,253 | 15.061190 | 0.481082 | 0.516805 | 0.181129 | 1.345221 | 54.900779 |
| QEM50 + sparse-depth 26k | 42,253 | 15.060190 | 0.481189 | 0.516350 | 0.181065 | 1.344086 | 54.841056 |

**Decision**: `FINAL_F29_ROOM_QEM_SPARSE_DEPTH_MIXED_GEOMETRY_PASS_QEM_REMAINS_MAIN`. Sparse depth improves SSIM, LPIPS, AbsRel, Depth MAE, and normal at identical topology, but gives back `0.001000 dB` PSNR, so the pure QEM50 frozen row remains the room PSNR headline.

---

## 2026-05-04 - Final F30/F31 courtyard sparse-depth controls

**Goal**: address courtyard's remaining normal-angle weakness and test whether sparse-depth recovery can improve CSEF50 or QEM50 without breaking the accepted CSEF50 main row.

**W&B runs**:
- F30 CSEF50 + sparse-depth strict recovery `22000->26000`: `9aaku1yn`;
- F31 QEM50 + sparse-depth strict recovery `22000->26000`, `lambda=0.0005`: `hbt9x0kg`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 1,677,484 | 12.103508 | 0.296648 | 0.569308 | 0.354648 | 3.829044 | 40.821649 |
| CSEF50 26k | 838,742 | 12.555809 | 0.338273 | 0.545077 | 0.322233 | 3.608432 | 40.830157 |
| QEM50 26k | 838,741 | 12.530957 | 0.339798 | 0.543378 | 0.332515 | 3.694743 | 40.804188 |
| CSEF50 + sparse-depth 26k | 838,742 | 12.552447 | 0.338854 | 0.545612 | 0.321690 | 3.618295 | 40.613745 |
| QEM50 + sparse-depth 26k | 838,741 | 12.531974 | 0.340074 | 0.543645 | 0.330244 | 3.689526 | 40.810260 |

**Decision**: `FINAL_F30_F31_COURTYARD_SPARSE_DEPTH_MIXED_CONTROLS_CSEF_REMAINS_MAIN`. F30 fixes the CSEF50 normal regression and improves AbsRel, but gives back small PSNR/LPIPS/Depth margins. F31 improves QEM50 on PSNR, SSIM, AbsRel, and Depth MAE, but remains weaker than CSEF50 on PSNR and sparse depth. Sparse depth is now replicated on bonsai, room, and courtyard as a geometry/perceptual regularizer, not a universal PSNR improver.

---

## 2026-05-04 - Final F32 counter sparse-depth compact recovery

**Goal**: replicate explicit sparse COLMAP depth compact recovery on a fourth accepted final scene and test whether the counter QEM40 row can be improved without changing topology.

**W&B run**:
- F32 QEM40 + sparse-depth strict recovery `22000->26000`: `x9b89ssf`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 83,834 | 14.136182 | 0.512802 | 0.452049 | 0.076996 | 0.369973 | 44.287035 |
| QEM40 frozen 26k | 50,300 | 14.409434 | 0.547456 | 0.420855 | 0.068076 | 0.338664 | 43.716007 |
| QEM40 + sparse-depth 26k | 50,300 | 14.408769 | 0.547570 | 0.420202 | 0.068014 | 0.339115 | 43.585215 |

**Decision**: `FINAL_F32_COUNTER_QEM_SPARSE_DEPTH_PARETO_PASS_PROMOTE_GEOMETRY_PERCEPTUAL`. F32 improves SSIM, LPIPS, AbsRel, and normal relative to QEM40 at identical topology while giving back only `0.000665 dB` PSNR and `0.000451` Depth MAE. It remains an all-metric clean-long win and becomes the counter geometry/perceptual headline.

---

## 2026-05-04 - Final F33 parking sparse-depth compact recovery and qualitative assets

**Goal**: close the explicit sparse-depth replication gap on the final remaining headline scene and improve the paper-facing qualitative package.

**W&B run**:
- F33 CSEF70 + sparse-depth strict recovery `22000->26000`: `x6rmhhlp`.

**Independent result**:

| method | triangles | PSNR | SSIM | LPIPS | AbsRel | Depth MAE | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean-long 22k | 8,548,242 | 18.480000 | 0.635000 | 0.347000 | 0.082000 | 1.868000 | 45.108000 |
| CSEF70 26k | 2,564,473 | 18.706079 | 0.647764 | 0.338282 | 0.079404 | 1.852816 | 44.204497 |
| CSEF70 + sparse-depth 26k | 2,564,473 | 18.712330 | 0.647730 | 0.338259 | 0.079071 | 1.854015 | 44.035708 |

**Qualitative update**:
- extended `scripts/car_model/final_make_paper_assets.py` to build per-scene GT / clean-long / strong-control / ours / error panels from independent renders;
- generated `outputs/carnet/meshsplatopt/final_paper_assets/qualitative_panels/final_multiscene_qualitative_montage.png`;
- manifest records every source image and selected frame.

**Decision**: `FINAL_F33_PARKING_CSEF_SPARSE_DEPTH_PARETO_PASS_PROMOTE`. F33 is promoted as the parking Pareto headline because it improves PSNR, LPIPS, AbsRel, and normal over F7 at identical topology, with negligible SSIM cost and a small Depth MAE tradeoff. Explicit sparse-depth compact recovery is now replicated on all five final scenes.

---

# SP-CarNet Research Log

Single source of truth for "what was tried under the SP-CarNet research line and how it went". Date-stamped, append-only. Each entry links to the relevant design / implementation / smoke / failure documents per the policy in `SPCarNet_radical_RFC.md` §8.

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

# SPCarNet / MeshSplatOpt

**Train-only evidence-guided compact Mesh Splatting with geometry-safe reconstruction repair.**

[中文](README.zh.md) | [Current archive](docs/car_model/5-7-Archive-Full9-CompactELA.md) | [May 7 update](docs/car_model/5-7-Update.md) | [Upgrade plan](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md) | [ECSR audit](docs/car_model/5-8-ECSR-CurrentStateAudit.md) | [Phase-A evidence](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md) | [Phase-B graph](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md) | [Policy split](docs/car_model/5-8-ECSR-PolicySplit.md) | [Phase-C preflight](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md) | [Execution log](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md) | [Research log](docs/car_model/SPCarNet_research_log.md) | [Legacy README](docs/car_model/archive/README_legacy_before_full9_2026-05-07.md)

SPCarNet is a research branch built on Mesh Splatting. The current version does not try to win by a hand-tuned prune ratio. It uses train-split evidence to decide how much geometry can be safely compacted, then repairs the held-out render with a train-calibrated Evidence Lumigraph Adapter (ELA). The current checkpoint is archived as:

```text
archive/full9-compact-ela-ssim-peak-20260507
commit fae7942
```

This is a strong and clean version, but not the final paper-ready endpoint: it wins RGB quality on all selected Mip-NeRF360 scenes while preserving geometry under the current geometry-safe criterion, yet its average triangle reduction is still conservative.

## Current Result

**Protocol.** Mip-NeRF360 same-protocol reproduction. For every scene, the clean MeshSplatting baseline is selected from clean `26000` and `30000` checkpoints using held-out test metrics only:

```text
score = PSNR + 20 * SSIM - 20 * LPIPS
```

Train metrics are not used to pick the baseline or the final method result.

**Final report.**

- Report: `outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean_report.md`
- W&B collector: `rp0d5gr3`
- Scenes: `9 / 9`
- RGB + compact + geometry-safe pass: `9 / 9`
- Strict all-axis pass: `5 / 9`
- Mean delta vs selected clean MeshSplatting baseline: `+0.4979 PSNR`, `+0.0158 SSIM`, `-0.0234 LPIPS`
- Mean triangle reduction: `5.7632%`

| scene | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS | triangle reduction | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| bicycle | 23.9127 | 0.6937 | 0.2803 | +0.6111 | +0.0338 | -0.0518 | 10.01% | strict pass |
| flowers | 20.1828 | 0.5473 | 0.3510 | +0.5005 | +0.0355 | -0.0436 | 10.02% | strict pass |
| garden | 26.0348 | 0.8171 | 0.1523 | +1.0056 | +0.0371 | -0.0490 | 1.50% | geometry-safe |
| stump | 25.3625 | 0.7125 | 0.2817 | +0.1575 | +0.0074 | -0.0123 | 10.02% | strict pass |
| treehill | 21.1984 | 0.5882 | 0.3581 | +0.2642 | +0.0237 | -0.0479 | 10.01% | strict pass |
| room | 29.1310 | 0.8849 | 0.2487 | +0.3837 | +0.0000 | -0.0012 | 0.10% | geometry-safe |
| counter | 27.2404 | 0.8641 | 0.2497 | +0.4886 | +0.0021 | -0.0023 | 0.10% | geometry-safe |
| kitchen | 27.9996 | 0.8769 | 0.1989 | +0.1810 | +0.0005 | -0.0002 | 0.10% | geometry-safe |
| bonsai | 29.7844 | 0.8982 | 0.2574 | +0.8892 | +0.0018 | -0.0021 | 10.00% | strict pass |

## ECSR Upgrade Status

The next method track is **ECSR: Evidence-Certified Surface Relocation**. Its goal is to move SPCarNet from image-space residual repair toward representation-level surface compression and appearance recovery.

Current execution artifacts:

- Current-state audit: [`docs/car_model/5-8-ECSR-CurrentStateAudit.md`](docs/car_model/5-8-ECSR-CurrentStateAudit.md)
- Phase-A train-only surface evidence: [`docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md`](docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md)
- Phase-B view-support graph: [`docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md`](docs/car_model/5-8-ECSR-PhaseB-ViewSupportGraph.md)
- Phase-A/B cached-view policy split: [`docs/car_model/5-8-ECSR-PolicySplit.md`](docs/car_model/5-8-ECSR-PolicySplit.md)
- Phase-C candidate preflight: [`docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md`](docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md)
- Execution log: [`docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md`](docs/car_model/5-8-ECSR-FinalDecisionExecutionLog.md)
- Combined Phase-A contact sheet: `outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/phase_a_surface_evidence_contact_sheet.png`

Phase-A result: `9 / 9` scenes pass surface addressability, but only `4 / 9` pass the current top-support multiview consistency check. This means the residual signal is real and surface-addressable, but a naive single-face residual delta is not yet a safe final method.

Phase-B result: the fixed graph policy finds `123` train-only local support clusters across full9, including `23` certificate-contraction candidates and `99` surface-attribute recovery candidates. The direct triangle-reduction upper bound of residual-hot clusters is tiny, so the next method step must separate compression candidates from appearance-recovery candidates instead of treating residual hotspots as the compression target.

Phase-C preflight result: `21 / 123` Phase-B clusters pass the train-only fitting/policy-val support-mask preflight (`13` contraction-type, `8` attribute-recovery-type). These are not accepted ECSR edits yet; they are the first eligible set for topology smoke tests and before/after local rendering certificates.

## Additional Evaluation Views

All tables below are derived from the same full9 report. Lower is better for LPIPS, AbsRel, DepthMAE, and Normal.

| evaluation view | result |
|---|---|
| selected clean MeshSplatting baseline | `9 / 9` RGB wins, mean `+0.4979` PSNR, `+0.0158` SSIM, `-0.0234` LPIPS |
| MeshSplatting paper table | `9 / 9` RGB wins, mean `+0.8685` PSNR, `+0.0366` SSIM, `-0.0465` LPIPS |
| clean checkpoint envelope | clean `26000` is selected over clean `30000` on all `9 / 9` scenes; mean score gap `+1.1029` |
| geometry / topology | `5 / 9` strict all-axis pass, `9 / 9` RGB + compact + geometry-safe pass, mean triangle reduction `5.7632%` |
| local qualitative crops | outdoor local MAE drop `12.8%` to `32.0%`; mixed indoor/outdoor local MAE drop up to `43.6%` |

**Against the MeshSplatting paper table.**

| scene | paper PSNR/SSIM/LPIPS | ours PSNR/SSIM/LPIPS | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|---:|---:|
| bicycle | 23.04 / 0.641 / 0.348 | 23.91 / 0.694 / 0.280 | +0.87 | +0.0527 | -0.0677 |
| flowers | 19.34 / 0.480 / 0.417 | 20.18 / 0.547 / 0.351 | +0.84 | +0.0673 | -0.0660 |
| garden | 24.70 / 0.762 / 0.217 | 26.03 / 0.817 / 0.152 | +1.33 | +0.0551 | -0.0647 |
| stump | 24.78 / 0.678 / 0.316 | 25.36 / 0.713 / 0.282 | +0.58 | +0.0345 | -0.0343 |
| treehill | 20.53 / 0.540 / 0.428 | 21.20 / 0.588 / 0.358 | +0.67 | +0.0482 | -0.0699 |
| room | 28.52 / 0.873 / 0.271 | 29.13 / 0.885 / 0.249 | +0.61 | +0.0119 | -0.0223 |
| counter | 26.51 / 0.846 / 0.279 | 27.24 / 0.864 / 0.250 | +0.73 | +0.0181 | -0.0293 |
| kitchen | 27.42 / 0.858 / 0.227 | 28.00 / 0.877 / 0.199 | +0.58 | +0.0189 | -0.0281 |
| bonsai | 28.19 / 0.876 / 0.294 | 29.78 / 0.898 / 0.257 | +1.59 | +0.0222 | -0.0366 |

**Clean `26000` / `30000` baseline envelope.**

| scene | selected | score 26000 | score 30000 | score gap | clean26000 PSNR/SSIM/LPIPS | clean30000 PSNR/SSIM/LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 26000 | 29.857 | 28.894 | +0.963 | 23.30 / 0.660 / 0.332 | 23.02 / 0.641 / 0.347 |
| flowers | 26000 | 22.027 | 21.060 | +0.968 | 19.68 / 0.512 / 0.395 | 19.39 / 0.492 / 0.408 |
| garden | 26000 | 36.604 | 35.623 | +0.981 | 25.03 / 0.780 / 0.201 | 24.71 / 0.762 / 0.216 |
| stump | 26000 | 33.428 | 32.347 | +1.081 | 25.21 / 0.705 / 0.294 | 24.87 / 0.684 / 0.309 |
| treehill | 26000 | 24.104 | 23.124 | +0.980 | 20.93 / 0.565 / 0.406 | 20.65 / 0.545 / 0.421 |
| room | 26000 | 41.446 | 40.575 | +0.871 | 28.75 / 0.885 / 0.250 | 28.48 / 0.873 / 0.268 |
| counter | 26000 | 38.953 | 37.772 | +1.181 | 26.75 / 0.862 / 0.252 | 26.41 / 0.846 / 0.278 |
| kitchen | 26000 | 41.364 | 39.940 | +1.424 | 27.82 / 0.876 / 0.199 | 27.30 / 0.858 / 0.226 |
| bonsai | 26000 | 41.633 | 40.156 | +1.477 | 28.90 / 0.896 / 0.259 | 28.38 / 0.879 / 0.290 |

**Geometry and topology.**

| scene | dAbsRel | dDepthMAE | dNormal | triangle red. | vertex red. | status |
|---|---:|---:|---:|---:|---:|---|
| bicycle | -0.000241 | -0.0204 | -0.0119 | 10.01% | 4.57% | strict all-axis pass |
| flowers | -0.003356 | -0.1250 | -0.0439 | 10.02% | 4.64% | strict all-axis pass |
| garden | -0.000007 | -0.0002 | -0.0010 | 1.50% | 2.69% | geometry-safe |
| stump | -0.005878 | -0.3507 | -0.0260 | 10.02% | 4.57% | strict all-axis pass |
| treehill | -0.001246 | -0.0747 | -0.0122 | 10.01% | 4.86% | strict all-axis pass |
| room | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.03% | geometry-safe |
| counter | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.10% | geometry-safe |
| kitchen | +0.000000 | +0.0000 | +0.0000 | 0.10% | 2.29% | geometry-safe |
| bonsai | -0.000368 | -0.0045 | -0.0254 | 10.00% | 3.16% | strict all-axis pass |

## Qualitative Comparison

The first panel is the fair full-frame view comparison from real held-out renders. It is useful for checking that the comparison uses the same test views and the selected clean MeshSplatting baseline, but the improvement is often residual-level and therefore visually subtle at full-frame scale.

<p align="center">
  <img src="assets/spcarnet_m360_full9_qualitative_gallery.png" width="980" alt="SPCarNet full-frame qualitative comparison against clean MeshSplatting">
</p>

The stronger qualitative evidence is the local held-out error-reduction view below. It is generated by [`scripts/car_model/generate_spcarnet_advantage_showcase.py`](scripts/car_model/generate_spcarnet_advantage_showcase.py): for each scene, the script first requires full-view `dPSNR > 0`, `dSSIM > 0`, and `dLPIPS < 0` under the same full9 protocol, then searches that view for textured crops where SPCarNet reduces RGB error against GT. Green means SPCarNet is closer to GT than clean MeshSplatting; magenta marks pixels where it is worse.

<p align="center">
  <img src="assets/spcarnet_m360_outdoor_detail_showcase.png" width="980" alt="SPCarNet outdoor local held-out error reduction against clean MeshSplatting">
</p>

The outdoor crops make the practical visual gain clearer: clean MeshSplatting often shows local triangular/blocky smoothing on foliage, ground texture, bench slats, and bark, while SPCarNet recovers sharper residual detail. A mixed indoor/outdoor version is also provided:

<p align="center">
  <img src="assets/spcarnet_m360_where_it_helps_showcase.png" width="980" alt="SPCarNet mixed local held-out error reduction against clean MeshSplatting">
</p>

Selection manifests: `assets/spcarnet_m360_outdoor_detail_selection.json`, `assets/spcarnet_m360_where_it_helps_selection.json`, and the earlier full-frame manifest `assets/spcarnet_m360_full9_gallery_selection.json`.

| qualitative crop | full-view delta PSNR/SSIM/LPIPS | local dPSNR | local MAE drop |
|---|---:|---:|---:|
| flowers / `00014.png` | +0.99 / +0.0616 / -0.0682 | +2.05 | 24.2% |
| garden / `00008.png` | +1.27 / +0.0432 / -0.0551 | +2.70 | 27.6% |
| treehill / `00010.png` | +0.59 / +0.0491 / -0.0881 | +3.03 | 32.0% |
| bicycle / `00021.png` | +1.13 / +0.0385 / -0.0615 | +1.88 | 17.5% |
| stump / `00007.png` | +0.26 / +0.0122 / -0.0208 | +0.81 | 12.8% |
| bonsai / `00001.png` | +2.79 / +0.0063 / -0.0007 | +3.82 | 43.6% |

## Method

The current method has three train-only stages.

1. **Sparse-occlusion protected compaction.** A CSEF/SOR selector scores triangles using train-view evidence. Outdoor scenes can remove around 10% of faces when evidence is stable. Indoor scenes with very low geometry error are protected by a micro-budget guard instead of being forced into destructive pruning.

2. **Checkpoint-safe topology rewrite.** The selected faces are removed from the Mesh Splatting checkpoint while keeping tensor shapes and face-index remapping consistent. The current version fixes a real room failure caused by trailing unused vertices in the checkpoint.

3. **Evidence Lumigraph Adapter.** ELA uses train-rendered RGB/depth/camera evidence to transfer local residual information to held-out views. Indoor scenes use low-resolution evidence and then upsample the residual to full resolution. The upsample alpha is selected only on train views with a strict PSNR/SSIM/LPIPS filter plus an SSIM-peak guard.

This is a research method rather than a post-hoc engineering patch because the main claim is a constrained decision policy: compact only when geometry evidence permits it, repair only when train evidence certifies the residual, and otherwise prefer a no-op or micro-edit over an unsafe apparent improvement.

## Why It Improves MeshSplatting

MeshSplatting already produces strong meshes, but its clean checkpoints still show view-dependent texture blur, local residual color errors, and overfitting sensitivity across iterations. SPCarNet adds two controls around the baseline:

- **Geometry-aware conservatism.** It does not assume every scene should be pruned equally. Garden and indoor scenes demonstrate that aggressive deletion can look attractive as a compression number but harm the fair claim.
- **Train-only view repair.** ELA improves RGB quality without selecting from held-out test metrics. It recovers residual visual detail while the compact checkpoint keeps the geometry accounting honest.

The result is not simply "train longer" or "pick a nicer checkpoint": clean `30000` is often worse than clean `26000` under held-out scoring, and the method still improves over the selected clean baseline.

## Ablation Summary

| variant | what it tests | outcome |
|---|---|---|
| Clean MeshSplatting `26000/30000` | fair baseline envelope | clean `26000` is selected on all 9 scenes by held-out score |
| Compact-only checkpoint | whether deletion alone is enough | safe but not enough for headline RGB gains |
| Compact + ELA without SSIM-peak alpha guard | whether scalar score alone is enough | room improves PSNR/LPIPS but loses held-out SSIM |
| Compact + ELA with SSIM-peak guard | current policy | restores room and keeps all indoor scenes fair under one train-only policy |
| Aggressive pruning branches | whether high compression can be forced | rejected; caused render/geometry regressions on sensitive scenes |

More detailed ablations and failed branches are archived in the research log and historical reports linked below.

## Reproduce Current Table

The archived run used the fixed method root:

```bash
OUT_ROOT=outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
POLICY_TAG=sor_adaptive_geo \
METHOD_NAME=ours_26000_sor_adaptive_geo_compact_ela \
CLEAN_ROOT=outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k \
DATA_ROOT=/data/peilincai/mesh_datasets/mipnerf360 \
SPARSE_OCCLUDER_POLICY=1 \
SPARSE_ADAPTIVE_GEOMETRY_BUDGET=1 \
INDOOR_POLICY_IMAGE_ARG=images_8 \
INDOOR_EVIDENCE_IMAGE_ARG=images_8 \
EVIDENCE_SKIP_FAILED_VIEWS=1 \
WANDB_GROUP=paper_m360_compact_ela_sor_adaptive_geo_26k \
bash scripts/car_model/run_paper_m360_compact_ela_policy_available7.sh
```

Collect the final table:

```bash
/home/peilincai/miniconda3/envs/Difix/bin/python \
  scripts/car_model/collect_paper_m360_compact_ela_policy_metrics.py \
  --method_root outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --policy_tag sor_adaptive_geo \
  --method_name ours_26000_sor_adaptive_geo_compact_ela \
  --method_iteration 26000 \
  --out_dir outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k \
  --scenes bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai \
  --wandb --wandb_project spcarnet_meshprior
```

## Limitations And Next Work

This version is promising, but it is not yet a complete "fully dominates MeshSplatting" endpoint.

- Average triangle reduction is only `5.76%` because room, counter, and kitchen are intentionally micro-pruned at `0.1%`.
- Strict all-axis pass is `5 / 9`, not `9 / 9`; the remaining scenes are geometry-safe or geometry-neutral rather than strict geometry wins.
- The next research target is a stronger geometry-preserving compaction mechanism that can raise indoor/garden compression without breaking RGB, sparse depth, or normal metrics.

The concrete improvement plan is recorded in [`docs/car_model/5-7-Archive-Full9-CompactELA.md`](docs/car_model/5-7-Archive-Full9-CompactELA.md) and the representation-level upgrade roadmap [`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md).

## Historical Material

Historical development logs are intentionally kept out of the top-level README:

- Legacy English README: [`docs/car_model/archive/README_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_legacy_before_full9_2026-05-07.md)
- Legacy Chinese README: [`docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md`](docs/car_model/archive/README_zh_legacy_before_full9_2026-05-07.md)
- Research log: [`docs/car_model/SPCarNet_research_log.md`](docs/car_model/SPCarNet_research_log.md)
- May 7 method story: [`docs/car_model/5-7-Update.md`](docs/car_model/5-7-Update.md)
- Representation-level upgrade plan: [`docs/car_model/5-7-Representation-Level-Upgrade-Plan.md`](docs/car_model/5-7-Representation-Level-Upgrade-Plan.md)

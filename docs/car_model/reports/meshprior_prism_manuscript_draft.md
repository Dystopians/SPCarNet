# PRISM: Auditable Topology Control for Mesh Splatting with Scene-Evidence Gates

Date: 2026-05-02

## Abstract

Mesh-splatting methods can reconstruct visually plausible scenes, but the resulting mesh topology is often redundant, unstable, or difficult to audit. We present PRISM, a conservative topology-control layer for mesh-splatting optimization. PRISM proposes local topology edits during training and accepts them only when scene-evidence gates, rollback validation, and final retained-topology audits agree. This makes the prior a proposal mechanism rather than an unconditional geometry writer. On a long-budget parking-phone scene, PRISM reduces topology while preserving independent render quality. On Mip-NeRF 360 `bonsai`, the retained relaxed refresh variant reduces final topology versus the Stage33 reference and improves independent PSNR, SSIM, and LPIPS. On ETH3D `courtyard`, PRISM improves topology, PSNR, and SSIM among selected rows, while exposing an LPIPS tradeoff. The current evidence supports PRISM as an auditable topology-control framework for mesh-splatting, with explicit limitations around universal metric dominance and geometry-observability assumptions.

## 1. Introduction

Mesh splatting provides a practical bridge between differentiable rendering and explicit scene topology. It can produce useful renderings from calibrated images and COLMAP reconstructions, but the learned scene representation can accumulate more triangles than necessary or retain geometry that is weakly supported by scene evidence. This matters for downstream inspection, editing, storage, and geometric reasoning. A visually plausible reconstruction with uncontrolled topology is not enough when the goal is to optimize a mesh scene that remains auditable.

The central design constraint in this work is conservative editability. A learned or hand-designed prior may suggest where the mesh should change, but it should not directly overwrite scene geometry. We use the rule:

> prior proposes, scene evidence disposes.

PRISM operationalizes this rule as a training-time topology controller. It collects per-triangle statistics, proposes candidate pruning edits, evaluates them under counterfactual render and sparse-geometry gates, commits only accepted edits, and rolls back edits that fail later validation. Later stages add post-commit relaxed discovery for the case where topology synchronization protects all surviving triangles and prevents further candidates. That relaxed path is default-off and is controlled by a retained-edit cap, strict proxy gates, and final audit metadata.

This draft reports the current strongest evidence line. Earlier object-prior and SP-CarNet modules remain useful engineering context, but the most defensible current contribution is PRISM as an auditable topology-control layer for mesh-splatting optimization.

## 2. Related Work

Neural radiance fields introduced a differentiable view-synthesis formulation in which a scene is represented by a continuous radiance field optimized from calibrated images [Mildenhall2020]. Instant-NGP later showed that multiresolution hash encodings and fused CUDA kernels can make neural graphics primitive optimization dramatically faster [Mueller2022]. These methods motivate scene optimization from posed images, but they do not directly provide explicit, auditable mesh topology.

3D Gaussian Splatting replaced neural volume queries with optimized anisotropic Gaussian primitives initialized from sparse calibration points, enabling high-quality real-time rendering with interleaved optimization and density control [Kerbl2023]. This line of work is highly effective for novel-view synthesis, but the representation is not a conventional triangle mesh and its density-control heuristics are not designed as rollback-audited topology edits. PRISM borrows the spirit of interleaved optimization and density control, but applies it to explicit mesh-splatting topology with scene-evidence gates.

Recent Gaussian-to-mesh and mesh-aligned Gaussian methods highlight the need for editable or structured geometry. SuGaR encourages Gaussians to align with scene surfaces and extracts meshes for efficient reconstruction and rendering [Guédon2024]. MeshGS aligns Gaussian splats to a mesh surface and removes redundant splats that do not contribute to rendering [Wang2024MeshGS]. Mesh-embedded Gaussian avatar work similarly uses a mesh as a structural carrier for Gaussian appearance primitives in a human-specific setting [Shao2024]. These systems support the broader point that mesh structure matters, but PRISM addresses a different problem: conservative topology edits inside mesh-splatting optimization, with rollback and retained-edit accounting.

PRISM also depends on classical scene geometry. COLMAP's incremental Structure-from-Motion pipeline provides calibrated cameras and sparse tracks [Schönberger2016], and its MVS component estimates depth and normal information with view selection and geometric consistency [Schönberger2016MVS]. In this project, COLMAP-style sparse geometry is not treated as ground truth; it is a scene-evidence proxy used for proposal validation and diagnostics.

Mesh simplification and remeshing are also relevant, but PRISM is not a generic post-hoc simplifier. Earlier rows in this project include checkpoint-copy pruning diagnostics, but the final method direction is training-time topology control: candidate edits are proposed during optimization, tested under counterfactual render/sparse-geometry gates, and rolled back if recovery validation fails.

### Citation TODO

- [Mildenhall2020] NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020. Verified source: https://arxiv.org/abs/2003.08934
- [Mueller2022] Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, SIGGRAPH 2022. Verified source: https://arxiv.org/abs/2201.05989
- [Kerbl2023] 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023. Verified source: https://arxiv.org/abs/2308.04079
- [Guédon2024] SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering, CVPR 2024. Verified source: https://arxiv.org/abs/2311.12775
- [Wang2024MeshGS] MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for High-Quality Rendering, 2024. Verified source: https://arxiv.org/abs/2410.08941
- [Shao2024] SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting, 2024. Verified source: https://arxiv.org/abs/2403.05087
- [Schönberger2016] Structure-from-Motion Revisited, CVPR 2016. Verified source: https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html
- [Schönberger2016MVS] Pixelwise View Selection for Unstructured Multi-View Stereo, ECCV 2016. Verified source: https://www.microsoft.com/en-us/research/?p=610152

## 3. Method

PRISM is inserted into the mesh-splatting training loop as an edit controller.

### 3.1 Inputs and Outputs

Inputs:

- calibrated images and COLMAP-style sparse geometry;
- a current mesh-splatting state;
- per-triangle render, geometry, orientation, uncertainty, and protection statistics;
- optional proposal metadata from earlier MeshPrior modules.

Outputs:

- accepted or rejected topology edits;
- rollback checkpoints and validation metadata;
- final cleanup summaries;
- retained-topology audits;
- independent render metrics and paper-facing evidence tables.

### 3.2 Candidate Proposal

PRISM collects per-triangle statistics during optimization and ranks candidate triangles for conservative pruning. Candidate selection is intentionally capped and auditable. The candidate set can be limited by score thresholds, maximum count, measured candidate impact, calibration-view diversity, and topology-protection rules.

### 3.3 Counterfactual Gate

Before an edit is committed, PRISM evaluates the proposed candidate edit on calibration views. A candidate is rejected if it causes unacceptable render, sparse-depth, normal, or changed-pixel degradation. The accepted edit is committed to the topology, and a rollback snapshot is retained.

### 3.4 Recovery Validation

After a committed edit, optimization enters a recovery window. At the end of the window, PRISM validates whether the edit remains acceptable. If the validation gate fails, the controller restores the rollback snapshot and records the rollback event.

### 3.5 Retained Relaxed Refresh

Stage34 showed that after a topology commit, synchronization can mark all surviving triangles as recent, causing `recent_t` to protect every triangle and zero normal prune scores through risk accumulation. Stage35 adds a conservative retained relaxed refresh:

- the relaxed path remains default-off;
- relaxed commits are capped separately from ordinary candidate rounds;
- strict proxy gates are required before relaxed commits;
- validation rollbacks decrement the active retained relaxed count;
- final topology audits distinguish total relaxed attempts from active retained relaxed commits.

This mechanism is central to the current best `bonsai` result: several relaxed commits were attempted and rolled back before one survived to the final checkpoint.

## 4. Experimental Setup

Datasets:

- `parking_phone_tiny`: local phone-captured parking scene with COLMAP reconstruction;
- Mip-NeRF 360 `bonsai`: public COLMAP-compatible scene;
- ETH3D `courtyard`: public COLMAP-compatible scene.

Metrics:

- paper-facing image metrics are independent `render.py + metrics.py` PSNR, SSIM, and LPIPS;
- topology is final checkpoint triangle count from final-cleanup summaries;
- audit metrics include active PRISM commits, relaxed commit attempts, validation rollbacks, and cap-reached metadata;
- training-time metrics are diagnostic only and are not mixed into paper-facing tables.

Reproducibility details are listed in `docs/car_model/reports/meshprior_prism_reproducibility_appendix.md`.

## 5. Results

All image metrics below come from independent rendering and metrics scripts. Lower LPIPS is better.

| scene | row | topology | PSNR | SSIM | LPIPS | audit note |
|---|---|---:|---:|---:|---:|---|
| parking_phone_tiny | late_prism_freeze_after_first_commit | 254491 | 17.314823 | 0.559230 | 0.442099 | long-budget parking topology-retention evidence |
| mipnerf360_bonsai | diverse_calib_measured_rank_cap512 | 633787 | 12.199921 | 0.276533 | 0.612583 | Stage33 reference |
| mipnerf360_bonsai | retained_relaxed_cap1_strict_gate | 633275 | 12.267367 | 0.277617 | 0.611939 | 1 active relaxed edit; 4 validation rollbacks recorded |
| eth3d_courtyard | measured_rank_cap512 | 102404 | 15.138977 | 0.484960 | 0.579188 | selected earlier row |
| eth3d_courtyard | retained_relaxed_cap1_strict_gate | 101913 | 15.383161 | 0.508091 | 0.584694 | 1 active relaxed edit; cap reached later |

### 5.1 Parking Scene

The parking row provides the long-budget single-scene topology-retention evidence. It demonstrates that PRISM can remain stable in a longer optimization path and produce a substantially reduced final topology while preserving independent render quality. This row should be treated as single-scene evidence, not a public benchmark claim.

### 5.2 Mip-NeRF 360 Bonsai

The Stage35 retained relaxed row improves over the Stage33 reference on all selected independent metrics while lowering topology. The audit is important: five relaxed commit attempts are recorded, four are validation-rolled back, and one remains active in the final checkpoint. This supports the need for retained-topology accounting rather than only reporting the final triangle count.

### 5.3 ETH3D Courtyard

On `courtyard`, Stage35 improves topology, PSNR, and SSIM relative to the selected M32 row. LPIPS is worse, so the correct claim is a topology/PSNR/SSIM improvement with an explicit perceptual tradeoff, not universal metric dominance.

## 6. Failure Cases and Diagnostics

The paper should include failure modes as part of the method evidence.

| failure type | evidence | implication |
|---|---|---|
| post-commit no-candidate | Stage34 post-commit diagnostics | recent protection can erase the normal candidate pool after topology sync |
| validation rollback | Stage35 `bonsai` retained audit | local proxy acceptance is not sufficient; recovery validation is necessary |
| relaxed cap reached | Stage35 `courtyard` metadata | cap-1 is conservative and may under-prune easy scenes |
| metric-path mismatch | Stage36 reconciliation table | training-time and independent metrics must remain separate |
| dataset geometry observability | Stage25 Tanks note | the current Tanks mirror is trainable but not geometry-observable enough for sparse-track claims |
| perceptual tradeoff | Stage36 selected rows | topology and PSNR/SSIM gains may not imply LPIPS gains |

## 7. Limitations

PRISM is conservative and can leave removable geometry when the retained relaxed cap is low. Current public-scene evidence is medium-budget rather than full-budget. The Tanks and Temples mirror available locally lacks the sparse tracks needed for paper-grade geometry claims. The method should not be described as radar-only reconstruction; the current assumption is calibrated images plus COLMAP-style scene geometry. Finally, the object-prior modules are not the strongest current story and should be framed as historical proposal infrastructure rather than the paper's headline result.

## 8. Conclusion

PRISM provides an auditable topology-control layer for mesh-splatting optimization. The strongest evidence is not that PRISM universally improves every render metric, but that topology edits can be proposed, gated, rolled back, audited, and retained under scene evidence. This is a defensible foundation for a paper about safe topology control in mesh-splatting systems. The next empirical step should be a full-budget public-scene Stage35 run only if a concrete missing table row is identified.

## 9. Final Evidence Gap Review

| gap | current status | immediate action |
|---|---|---|
| full-budget public Stage35 row | missing, but not yet tied to a specific table need | no-go for now |
| Tanks geometry tracks | current mirror lacks true sparse tracks | rebuild COLMAP tracks before geometry claims |
| `courtyard` LPIPS tradeoff | known and documented | report explicitly |
| object-prior narrative | weaker than topology-control narrative | keep as background/proposal context |
| manuscript completeness | draft exists but needs human editing and citations | continue writing, then decide if a targeted run is needed |

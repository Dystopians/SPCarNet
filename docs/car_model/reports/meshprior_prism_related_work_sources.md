# PRISM Related Work Source Notes

Date: 2026-05-02

This file records the verified sources used to replace the related-work placeholder in `meshprior_prism_manuscript_draft.md`. Exact BibTeX should be generated before submission, but titles, venues/years, and URLs are verified enough for draft writing.

## Sources

| key | title | venue/year | verified URL | why it matters |
|---|---|---|---|---|
| Mildenhall2020 | NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | ECCV 2020 | https://arxiv.org/abs/2003.08934 | Differentiable scene optimization from posed images. |
| Mueller2022 | Instant Neural Graphics Primitives with a Multiresolution Hash Encoding | SIGGRAPH 2022 | https://arxiv.org/abs/2201.05989 | Fast neural graphics primitive optimization. |
| Kerbl2023 | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | SIGGRAPH 2023 | https://arxiv.org/abs/2308.04079 | Interleaved optimization and density control for splatted radiance fields. |
| Guedon2024 | SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering | CVPR 2024 | https://arxiv.org/abs/2311.12775 | Mesh reconstruction and mesh binding from Gaussian splats. |
| Wang2024MeshGS | MeshGS: Adaptive Mesh-Aligned Gaussian Splatting for High-Quality Rendering | 2024 | https://arxiv.org/abs/2410.08941 | Mesh-aligned splats and redundant-splat removal. |
| Shao2024 | SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting | 2024 | https://arxiv.org/abs/2403.05087 | Mesh-embedded Gaussian splats in a structured domain. |
| Schonberger2016 | Structure-from-Motion Revisited | CVPR 2016 | https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html | COLMAP-style camera and sparse point reconstruction. |
| Schonberger2016MVS | Pixelwise View Selection for Unstructured Multi-View Stereo | ECCV 2016 | https://www.microsoft.com/en-us/research/?p=610152 | COLMAP-style depth/normal MVS evidence. |

## Claim Alignment

These sources support the following positioning:

- PRISM is downstream of posed-image scene optimization and splatting.
- PRISM is not a replacement for 3DGS or NeRF.
- PRISM differs from Gaussian-to-mesh reconstruction by emphasizing conservative topology edits, rollback, and retained-edit audits during mesh-splatting optimization.
- COLMAP evidence is a proxy for scene consistency, not ground-truth geometry.

## Remaining Citation Tasks

- Generate BibTeX entries from official proceedings/arXiv pages.
- Add exact citations for the local Mesh Splatting codebase/paper if a formal paper page is available.
- Add classic mesh simplification/remeshing references after choosing the exact framing.


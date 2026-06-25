# SPCarNet Rate-Distortion Frontier 2026-06-25

| method | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS | tri red. | ckpt red. | VRAM red. | render FPS ratio | adapter ms/view | integrated ms/view | integrated FPS | integrated/render-only compact | claim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MeshSplatting official clean30k reproduction | 24.800172 | 0.731005 | 0.307177 | n/a | n/a | n/a | 0 | 0 | n/a | n/a | 0 | n/a | n/a | n/a | paper-table bridge only |
| Selected clean MeshSplatting baseline | 25.151682 | 0.749018 | 0.287621 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000000 | 0 | n/a | n/a | n/a | same-protocol baseline |
| Compact-ELA support | 25.649623 | 0.764774 | 0.264247 | 0.497941 | 0.015755 | -0.023373 | 0.057632 | 0.039467 | not_measured | not_measured | not_measured | not_measured | not_measured | not_measured | RGB+compact support; not strongest headline |
| Phase-J SPCarNet | 26.482766 | 0.783720 | 0.224261 | 1.331084 | 0.034702 | -0.063359 | 0.076479 | 0.046753 | 0.025733 | 0.946023 | 1061.298183 | 951.410896 | 1.051071 | 27.044247 | strong RGB+triangles; memory/size positive; integrated no-IO runtime speed-negative |

Interpretation: Phase-J is strong on RGB and triangle reduction, but current render-time adapter dominates runtime. The new integrated runner confirms `951.410896 ms/view` and `1.051071 FPS` without PNG writes or metrics, about `27.044247x` slower than compact render-only. This frontier supports quality/compactness/memory claims, not speedup.

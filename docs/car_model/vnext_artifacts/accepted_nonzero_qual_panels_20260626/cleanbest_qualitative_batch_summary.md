# vNext Clean-Best Qualitative Panel Batch

- schema version: `2`
- run root: `/dev/shm/peilincai_spcarnet_vnext_garden_qual_20260626_131403`
- output root: `docs/car_model/vnext_artifacts/accepted_nonzero_qual_panels_20260626`
- scenes: `garden`
- clean selection policy: `composite_psnr_ssim_lpips`
- explicit clean method: ``

| scene | status | clean best | vNext PSNR | vNext SSIM | vNext LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR base | dSSIM base | dLPIPS base | panel |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| garden | FRAME_CONTRACT_MISMATCH | ours_26000 | 24.741142273 | 0.754052162 | 0.248015299 | -0.288068771 | -0.025982916 | +0.046700805 | -0.286394119 | -0.025978506 | +0.046693772 | `` |

Interpretation note: LPIPS deltas are better when negative. Panels use the clean checkpoint selection policy stated above from local test metrics and are intended for qualitative diagnosis, not parameter search.

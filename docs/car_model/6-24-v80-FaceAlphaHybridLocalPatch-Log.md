# v80 Face-Alpha + Local-Patch + Bin-Gain Hybrid Counter Probe

Date: 2026-06-24

## Purpose

v80 tested whether the strongest reproducible counter anchor could be improved by combining:

- policy-val face-alpha calibration;
- local-patch surface prior candidates with blend `0,0.5,1.0`;
- policy-val prior bin-gain hybrid selection;
- W&B online logging under group `v80_facealpha_hybrid_localpatch`.

This was a real train/eval pipeline probe, not a README-only change. It was run on `counter` as a promotion gate before any full9 expansion.

## Command And Artifacts

Output root:

```text
/dev/shm/peilincai_spcarnet_v80_facealpha_hybrid_localpatch_20260624
```

Persisted small artifacts:

```text
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/counter/results.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/counter/per_view.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/counter/surface_residual_region_texture_adapter_audit.json
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/counter/run_counter.log
outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v80_facealpha_hybrid_localpatch_20260624/counter/apply_metrics_counter.log
```

The large render/GT image tree remains in `/dev/shm` and was intentionally not copied because `/data` is almost full.

## Result

| line | PSNR | SSIM | LPIPS | status |
|---|---:|---:|---:|---|
| v79 / v56 / v64 anchor | `26.756130219` | `0.862126231` | `0.251691371` | best fixed representation-level counter anchor |
| v80 face-alpha + local-patch + bin-gain hybrid | `26.756135941` | `0.862126231` | `0.251691461` | not promoted |

Delta versus v79 anchor:

```text
dPSNR  = +0.000005722
dSSIM  = +0.000000000
dLPIPS = +0.000000089  (worse; LPIPS is lower-is-better)
```

The adapter accepted an atlas with `selected_alpha=0.5`, wrote all `30` test views, and changed `6.3901%` of pixels. However, it did not strict-win all three RGB metrics over the v56/v64/v79 anchor.

## Verdict

v80 is a useful diagnostic but not a promoted method. It shows that adding local-patch prior and bin-gain hybrid on top of the face-alpha anchor can recover the anchor almost exactly, but the remaining improvement is numerical noise and LPIPS is marginally worse. The next representation-level attempt should therefore not keep expanding this exact candidate family blindly; it needs a stronger mechanism or a stricter source-wise promotion policy.


# Stage ELA12 Corrected Clean Baseline Audit

This report supersedes the earlier train-selected checkpoint audit. The reviewer-facing baseline is now the coherent clean Mesh Splatting checkpoint with the best held-out test score, where `score = PSNR + 20 * SSIM - 20 * LPIPS`. The train score table is retained only as a diagnostic because train metrics can favor longer overfit runs.

Strict full-pass rows against the held-out-test-selected clean baseline: `5/5`.

## Held-Out-Test-Selected Baseline Comparison

| scene | selected clean baseline | method | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth MAE | dNormal | tri reduction | full |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bonsai | clean9000@9000 | SOR10 + ELA safe | 2.838371 | 0.163376 | -0.099541 | -0.105169 | -1.032433 | -2.410058 | 10.25% | `True` |
| courtyard | clean9000@9000 | SOR10 + ELA safe | 0.969368 | 0.028828 | -0.056569 | -0.104763 | -1.288431 | -2.711335 | 10.34% | `True` |
| room | clean9000@9000 | QEM50 parent-rollback + ELA safe | 3.304691 | 0.050085 | -0.062170 | -0.002331 | -0.019509 | -1.824378 | 50.00% | `True` |
| counter | clean9000@9000 | QEM50 parent-rollback + ELA safe | 3.157017 | 0.069925 | -0.070661 | -0.000686 | -0.008253 | -2.080537 | 50.00% | `True` |
| parking_phone_tiny | clean22000@22000 | CSEF70 sparse-depth + train-p15 local parent-gated ELA | 0.496731 | 0.026720 | -0.033581 | -0.003106 | -0.014383 | -1.072729 | 70.00% | `True` |

## Clean Baseline Candidate Table

| scene | candidate | train score | test score | train PSNR | train SSIM | train LPIPS | test PSNR | test SSIM | test LPIPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bonsai | clean7000@7000 | 18.081908 | 17.601224 | 18.500816 | 0.464979 | 0.485924 | 18.303303 | 0.455556 | 0.490660 |
| bonsai | clean9000@9000 | 18.561093 | 18.145757 | 18.700453 | 0.472101 | 0.479069 | 18.541124 | 0.463496 | 0.483265 |
| bonsai | clean22000@22000 | 5.246265 | 3.678145 | 11.324788 | 0.264448 | 0.568374 | 10.944348 | 0.222848 | 0.586158 |
| courtyard | clean7000@7000 | 26.058600 | 21.406315 | 20.804838 | 0.683681 | 0.420992 | 18.321131 | 0.594281 | 0.440022 |
| courtyard | clean9000@9000 | 27.723876 | 22.066040 | 21.388170 | 0.709467 | 0.392682 | 18.494551 | 0.602439 | 0.423865 |
| courtyard | clean22000@22000 | 21.882735 | 6.650293 | 17.586325 | 0.627036 | 0.412216 | 12.103508 | 0.296648 | 0.569308 |
| room | clean7000@7000 | 40.004548 | 37.792803 | 26.288824 | 0.868990 | 0.183204 | 24.913910 | 0.845804 | 0.201860 |
| room | clean9000@9000 | 43.825732 | 41.302776 | 27.825556 | 0.911629 | 0.111620 | 26.217100 | 0.889372 | 0.135088 |
| room | clean22000@22000 | 12.308334 | 10.697283 | 14.981968 | 0.431373 | 0.565055 | 14.258379 | 0.400864 | 0.578919 |
| counter | clean7000@7000 | 35.885164 | 34.925578 | 24.085732 | 0.807145 | 0.217174 | 23.594526 | 0.794066 | 0.227514 |
| counter | clean9000@9000 | 39.751650 | 38.506212 | 25.459196 | 0.860541 | 0.145919 | 24.801929 | 0.844451 | 0.159236 |
| counter | clean22000@22000 | 15.857680 | 15.351248 | 14.101978 | 0.528315 | 0.440530 | 14.136182 | 0.512802 | 0.452049 |
| parking_phone_tiny | clean22000@22000 | 30.818566 | 24.234202 | 21.148010 | 0.753121 | 0.269593 | 18.479990 | 0.634623 | 0.346913 |
| parking_phone_tiny | clean30000@30000 | 31.411739 | 24.019566 | 21.364588 | 0.764637 | 0.262280 | 18.408827 | 0.631504 | 0.350967 |

## Per-View RGB Stress Test

| scene | views | RGB full-pass views | min dPSNR | mean dPSNR | worst dLPIPS |
|---|---:|---:|---:|---:|---:|
| bonsai | 37 | 37 | 0.837978 | 2.838371 | -0.056966 |
| courtyard | 5 | 5 | 0.210857 | 0.969366 | -0.031749 |
| room | 39 | 39 | 0.453001 | 3.304691 | -0.013167 |
| counter | 30 | 30 | 1.043489 | 3.157015 | -0.034378 |
| parking_phone_tiny | 54 | 53 | -0.049734 | 0.496731 | -0.003703 |

## Per-View RGB Envelope Stress Test

This stricter stress test compares each held-out view to the best clean checkpoint separately for PSNR, SSIM, and LPIPS. It is not the main coherent-checkpoint baseline, but it exposes remaining visual tails.

| scene | views | RGB envelope full-pass views | min dPSNR | mean dPSNR | worst dLPIPS |
|---|---:|---:|---:|---:|---:|
| bonsai | 37 | 37 | 0.685295 | 2.824528 | -0.056966 |
| courtyard | 5 | 4 | -0.021313 | 0.843062 | -0.031749 |
| room | 39 | 39 | 0.453001 | 3.304691 | -0.013167 |
| counter | 30 | 30 | 1.043489 | 3.157015 | -0.034378 |
| parking_phone_tiny | 54 | 53 | -0.049734 | 0.458938 | -0.003703 |

## Scope Note

This audit covers the current validated scene set with complete method artifacts: `parking_phone_tiny`, `bonsai`, `courtyard`, `room`, and `counter`. Raw dataset folders that do not yet have complete method artifacts are intentionally not folded into the headline table. For Mip-NeRF360 paper-level comparison, this is still only a scene subset and must not be presented as the full nine-scene Mip-NeRF360 benchmark mean.

## Artifacts

- summary JSON: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/fair_baseline_audit.json`
- baseline candidate CSV: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/baseline_candidate_rows.csv`
- comparison CSV: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/fair_selected_baseline_comparison.csv`
- per-view CSV: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/per_view_rgb_deltas.csv`
- per-view RGB envelope CSV: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/per_view_rgb_envelope_deltas.csv`
- qualitative gallery: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/qualitative_gallery/gallery.html`
- qualitative manifest: `outputs/carnet/meshsplatopt/stageELA12_fair_baseline_audit/qualitative_gallery/gallery_manifest.md`

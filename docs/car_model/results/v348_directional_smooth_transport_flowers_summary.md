# v346-v348 Representation Transport Summary
Date: 2026-07-02
Status: **NOT COMPLETE**. v346-v348 did not surpass Phase-J.
## Implemented Changes
- source-heldout image/patch proxy loss for training supervision
- source-heldout calibrated texture anchor inside SurfaceResidualDecoder
- support-normalized smooth residual transport at apply/eval time
- raw/view-gated/applied residual diagnostics in policy evaluation

## Phase-J Reference Boundary
Phase-J remains the strongest local method family in current evidence: 9/9 strict RGB scene wins vs clean, 244/246 per-view strict RGB wins, mean +1.331084 PSNR, +0.034702 SSIM, -0.063359 LPIPS, and 7.6479% mean triangle reduction. v346-v348 did not approach this level.

## Flowers Policy-Val Results
| run | steps | faces | all-axis | phasej gate | alpha | PSNR gain | SSIM gain | LPIPS gain | raw pred | applied delta | smooth |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| v346a_v297_like_transport_no_patch_24step | 24 | 32 | false | false | 0.125 | 8.11184286853e-06 | -5.76178232829e-07 | 2.0923713843e-07 |  |  | off |
| v346b_source_heldout_patch_transport_24step | 24 | 32 | false | false | 0.03125 | 1.93131492902e-06 | -1.44044558207e-07 | 2.53940622012e-07 |  |  | off |
| v347a_energy_transport_no_anchor_24step | 24 | 32 | false | false | 0.03125 | 3.75981900043e-06 | -2.98023223877e-07 | 1.22313698133e-07 | 0.0084864435515 | 1.0875660202e-07 | off |
| v347b_texture_anchor_24step | 24 | 32 | false | false | 0 | 0 | 0 | 0 | 0.0176902754087 | 0 | off |
| v347c_texture_anchor_source_image_24step | 24 | 32 | false | false | 0 | 0 | 0 | 0 | 0.0176526025452 | 0 | off |
| v347d_anchor_structure_gate_policy_probe | 0 | 32 | false | false | 0.015625 | 3.94366681267e-06 | -3.22858492533e-07 | 2.53940622012e-07 | 0.0176902754087 | 1.14084231543e-07 | off |
| v347e_128face_no_anchor_48step | 48 | 128 | false | false | 0.125 | 6.06853681265e-05 | -1.17719173431e-06 | -9.03382897377e-07 | 0.0102284620821 | 1.34567198273e-06 | off |
| v347f_128face_anchor_48step | 48 | 128 | false | false | 0.0625 | 3.95106753196e-05 | -9.08970832825e-07 | 1.17346644402e-07 | 0.0137829521906 | 8.67257676636e-07 | off |
| v347g_128face_anchor_structure_safe_48step | 48 | 128 | false | false | 0.03125 | 1.58043896083e-05 | -4.66903050741e-07 | 5.32095630964e-07 | 0.0118941590317 | 3.84056799992e-07 | off |
| v347h_no_anchor_smooth_r1_probe | 0 | 128 | false | false | 0.125 | 7.50028014105e-05 | -1.23182932536e-06 | -3.3217171828e-07 | 0.0102284620821 | 2.16576409665e-06 | r1 |
| v347i_anchor_smooth_r1_probe | 0 | 128 | false | false | 0.00390625 | 3.26529441767e-06 | -3.97364298503e-08 | 4.91738319397e-07 | 0.0137829521906 | 8.97728832013e-08 | r1 |
| v347j_anchor_smooth_r2_probe | 0 | 128 | false | false | 0.0625 | 5.12595767844e-05 | -1.14242235819e-06 | 1.52053932349e-06 | 0.0137829521906 | 2.08378147685e-06 | r2 |
| v348a_128face_directional_smooth_no_anchor_600step | 600 | 128 | false | false | 0.0625 | 5.21192973162e-05 | -8.19563865662e-07 | 7.28294253349e-07 | 0.0176960811009 | 2.56649254027e-06 | r2 |
| v348b_128face_directional_smooth_anchor_600step | 600 | 128 | false | false | 0.0625 | 5.78156919939e-05 | -8.99036725362e-07 | 1.22437874476e-06 | 0.0179988269431 | 2.61266410462e-06 | r2 |

## Oracle Diagnostic
A teacher-residual oracle using the same 128 candidate faces is positive on PSNR/SSIM and mostly positive on LPIPS. This proves the candidate support/apply path still has headroom; the current learned decoder is failing to recover the correct residual direction and structure.
| oracle setting | PSNR gain | SSIM gain | LPIPS gain | changed fraction | positive view fractions |
|---|---:|---:|---:|---:|---|
| radius2_alpha0.125 | 0.00103598579046 | 2.20785538356e-05 | 1.38903657595e-05 | 0.00129689476597 | 1.0/1.0/0.6667 |
| radius2_alpha1.0 | 0.00610695796717 | 0.000113919377327 | 0.000171336034934 | 0.0016712622549 | 1.0/1.0/0.75 |
| radius3_alpha1.0 | 0.00646381108986 | 8.74598821004e-05 | 0.000240119795005 | 0.00227585388994 | 1.0/1.0/0.8333 |

## Interpretation
The negative result is useful: source-heldout image proxy, texture anchor, and residual smoothing increase amplitude/coverage, but they do not produce an all-axis policy-val pass. Longer 600-step training increases raw residual magnitude, yet SSIM remains negative. The next real method change should replace the low-bandwidth per-face mean residual decoder with a high-bandwidth support-view residual transport representation or a neighbor/view-conditioned field that can preserve structure rather than average residuals.

## Evidence Paths
- `outputs/carnet/spcarnet_v346_source_heldout_image_proxy_20260702/v346a_v297_like_transport_no_patch_24step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v346_source_heldout_image_proxy_20260702/v346b_source_heldout_patch_transport_24step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_texture_anchor_flowers_20260702/v347a_energy_transport_no_anchor_24step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_texture_anchor_flowers_20260702/v347b_texture_anchor_24step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_texture_anchor_flowers_20260702/v347c_texture_anchor_source_image_24step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_texture_anchor_gate_probe_20260702/v347d_anchor_structure_gate_policy_probe/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_128face_texture_anchor_flowers_20260702/v347e_128face_no_anchor_48step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_128face_texture_anchor_flowers_20260702/v347f_128face_anchor_48step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_128face_texture_anchor_flowers_20260702/v347g_128face_anchor_structure_safe_48step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_smooth_transport_probe_flowers_20260702/v347h_no_anchor_smooth_r1_probe/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_smooth_transport_probe_flowers_20260702/v347i_anchor_smooth_r1_probe/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v347_smooth_transport_probe_flowers_20260702/v347j_anchor_smooth_r2_probe/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v348_directional_smooth_transport_flowers_20260702/v348a_128face_directional_smooth_no_anchor_600step/v180_perceptual_surface_decoder_audit.json`
- `outputs/carnet/spcarnet_v348_directional_smooth_transport_flowers_20260702/v348b_128face_directional_smooth_anchor_600step/v180_perceptual_surface_decoder_audit.json`

Final status: **NOT COMPLETE**.

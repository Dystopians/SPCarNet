# v169 Policy-Val Carrier Upper-Bound Diagnostic

- fit evidence: `/dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence`
- region carrier: `/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626/flowers/carrier.json`
- candidate faces: `342`
- view count: `46`
- verdict: `current_carrier_improves_policy_val_all_axis`
- robust verdict: `current_carrier_too_weak_for_tail_robust_policy_val_ssim_lpips`

## Best All-Axis Candidate

- texture size: `8`
- teacher low-rank texture rank: `4`
- alpha: `0.03125000`
- relative gain: `+0.01903233`
- PSNR gain: `+0.00016382`
- SSIM gain: `+0.00000039`
- LPIPS gain: `+0.00000096`

## Best Robust All-Axis Candidate

- no candidate passes the tail-robust gate over positive-view fraction, min-view, and CVaR metrics.

## Candidate Rows

| texture | rank | alpha | rel gain | PSNR gain | SSIM gain | LPIPS gain | all-axis |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 2 | 0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |  |
| 8 | 2 | 0.03125000 | +0.01913459 | +0.00016366 | +0.00000030 | +0.00000081 | yes |
| 8 | 2 | 0.06250000 | +0.03690628 | +0.00032308 | -0.00000009 | +0.00000064 |  |
| 8 | 2 | 0.12500000 | +0.06836146 | +0.00062903 | -0.00000294 | -0.00000644 |  |
| 8 | 2 | 0.25000000 | +0.11491814 | +0.00118958 | -0.00001631 | -0.00003309 |  |
| 8 | 2 | 0.50000000 | +0.14250178 | +0.00210332 | -0.00006724 | -0.00012604 |  |
| 8 | 4 | 0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |  |
| 8 | 4 | 0.03125000 | +0.01903233 | +0.00016382 | +0.00000039 | +0.00000096 | yes |
| 8 | 4 | 0.06250000 | +0.03670266 | +0.00032339 | +0.00000010 | +0.00000070 |  |
| 8 | 4 | 0.12500000 | +0.06795707 | +0.00062963 | -0.00000256 | -0.00000565 |  |
| 8 | 4 | 0.25000000 | +0.11412145 | +0.00119053 | -0.00001559 | -0.00003272 |  |
| 8 | 4 | 0.50000000 | +0.14104594 | +0.00210646 | -0.00006615 | -0.00012565 |  |
| 8 | 8 | 0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |  |
| 8 | 8 | 0.03125000 | +0.01915527 | +0.00016320 | +0.00000031 | +0.00000067 | yes |
| 8 | 8 | 0.06250000 | +0.03693258 | +0.00032207 | -0.00000008 | +0.00000001 |  |
| 8 | 8 | 0.12500000 | +0.06835382 | +0.00062669 | -0.00000291 | -0.00000756 |  |
| 8 | 8 | 0.25000000 | +0.11466193 | +0.00118382 | -0.00001635 | -0.00003512 |  |
| 8 | 8 | 0.50000000 | +0.14113921 | +0.00208921 | -0.00006792 | -0.00013219 |  |
| 16 | 2 | 0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |  |
| 16 | 2 | 0.03125000 | +0.01911639 | +0.00016733 | +0.00000041 | -0.00000078 |  |
| 16 | 2 | 0.06250000 | +0.03655185 | +0.00032948 | +0.00000004 | -0.00000353 |  |
| 16 | 2 | 0.12500000 | +0.06638004 | +0.00063757 | -0.00000327 | -0.00001495 |  |
| 16 | 2 | 0.25000000 | +0.10586480 | +0.00119005 | -0.00001908 | -0.00004949 |  |
| 16 | 2 | 0.50000000 | +0.10414859 | +0.00203975 | -0.00007909 | -0.00015889 |  |
| 16 | 4 | 0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |  |
| 16 | 4 | 0.03125000 | +0.01916583 | +0.00016804 | +0.00000040 | -0.00000028 |  |
| 16 | 4 | 0.06250000 | +0.03662647 | +0.00033075 | +0.00000000 | -0.00000280 |  |
| 16 | 4 | 0.12500000 | +0.06643218 | +0.00064002 | -0.00000337 | -0.00001288 |  |
| 16 | 4 | 0.25000000 | +0.10558212 | +0.00119374 | -0.00001942 | -0.00004594 |  |
| 16 | 4 | 0.50000000 | +0.10202843 | +0.00204219 | -0.00008015 | -0.00015214 |  |
| 16 | 8 | 0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 | +0.00000000 |  |
| 16 | 8 | 0.03125000 | +0.01939441 | +0.00017065 | +0.00000045 | -0.00000023 |  |
| 16 | 8 | 0.06250000 | +0.03704597 | +0.00033594 | +0.00000008 | -0.00000249 |  |
| 16 | 8 | 0.12500000 | +0.06712015 | +0.00064969 | -0.00000325 | -0.00001266 |  |
| 16 | 8 | 0.25000000 | +0.10635330 | +0.00121142 | -0.00001935 | -0.00004433 |  |
| 16 | 8 | 0.50000000 | +0.10118516 | +0.00206957 | -0.00008053 | -0.00015286 |  |

## Interpretation

- This diagnostic uses train-fit evidence for fitting and train-policy-val GT for certification only.
- It does not read target/test GT and does not write model artifacts.
- The nominal all-axis pass is not a useful promotion signal here: the best
  full-image PSNR gain is only `+0.00016382 dB`, while SSIM/LPIPS gains are
  around `1e-6`.
- The robust gate fails: SSIM positive-view fraction is only `0.5`, LPIPS
  positive-view fraction is only `0.66666667`, and both SSIM/LPIPS CVaR tails
  are negative.
- This should block flowers exact/full9 promotion for this projection route.
  The current carrier is not strong enough; the next step must be a stronger
  cross-view residual-direction representation, not another alpha or rank scan.

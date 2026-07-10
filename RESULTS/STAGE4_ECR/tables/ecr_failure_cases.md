# ECR transport — E9-style failure cases (frozen selection rules)

| case | scene | view | type | covered_fraction | dPSNR (final-base) | planes |
|---|---|---|---|---|---|---|
| FC01 | garden | DSC07988 | coverage-gap | 0.922 | +0.348 | `analysis/quals/garden_final/DSC07988/` |
| FC02 | garden | DSC07988 | transport-worst | 0.922 | +0.348 | `analysis/quals/garden_final/DSC07988/` |
| — | garden | (0/24 views transport-negative: none) | summary | | | |
| FC03 | bicycle | _DSC8784 | coverage-gap | 0.467 | +0.128 | `analysis/quals/bicycle_final/_DSC8784/` |
| FC04 | bicycle | _DSC8784 | transport-worst | 0.467 | +0.128 | `analysis/quals/bicycle_final/_DSC8784/` |
| — | bicycle | (0/25 views transport-negative: none) | summary | | | |
| FC05 | bonsai | DSCF5813 | coverage-gap | 0.925 | +2.561 | `analysis/quals/bonsai_final/DSCF5813/` |
| FC06 | bonsai | DSCF5789 | transport-worst | 0.954 | +0.850 | `analysis/quals/bonsai_final/DSCF5789/` |
| — | bonsai | (0/37 views transport-negative: none) | summary | | | |
| FC07 | treehill | _DSC8898 | coverage-gap | 0.471 | +0.664 | `analysis/quals/treehill_final/_DSC8898/` |
| FC08 | treehill | _DSC8946 | transport-worst | 0.825 | -0.059 | `analysis/quals/treehill_final/_DSC8946/` |
| — | treehill | (1/18 views transport-negative: _DSC8946) | summary | | | |
| FC09 | kitchen | DSCF0760 | coverage-gap | 0.935 | +4.523 | `analysis/quals/kitchen_final/DSCF0760/` |
| FC10 | kitchen | DSCF0728 | transport-worst | 0.984 | +0.691 | `analysis/quals/kitchen_final/DSCF0728/` |
| — | kitchen | (0/35 views transport-negative: none) | summary | | | |

Types are SELECTION rules; the occlusion/seam/coverage characterization per case comes from inspecting the dumped planes (base/final/err/conf/count/beta):

## Case narratives

- **garden / DSC07988** — Garden's weakest coverage view is still 92% covered and IMPROVES +0.35 dB — no failure mode present; kept as the scene's honest worst case.
- **bicycle / _DSC8784** — COVERAGE GAP (the archetype): a low viewpoint whose entire foreground ground plane (lower half-frame) has ZERO warp support — the training trajectory never observes it from a compatible pose (conf.png: near-black lower half). The confidence gate withholds the transport there (β·valid = 0, base passes through untouched) while the supported upper half (bench/bicycle/vegetation) is corrected: net +0.13 dB at 47% coverage. Graceful degradation, no hallucination.
- **bonsai / DSCF5813** — Lowest-coverage bonsai view still gains +2.56 dB — the indoor ring trajectory keeps even the worst view well supported.
- **bonsai / DSCF5789** — Scene-worst transport delta is +0.85 dB (still strongly positive); no failure mode.
- **treehill / _DSC8898** — COVERAGE GAP: 47% covered (frame edges + near ground unsupported); transport corrects the covered remainder for +0.66 dB.
- **kitchen / DSCF0760** — Lowest-coverage kitchen view is 94% covered and gains +4.52 dB — no failure mode in this indoor ring scene.
- **kitchen / DSCF0728** — Scene-worst transport delta is +0.69 dB (strongly positive); the view where the base render is already best (29.1 dB).
- **treehill / _DSC8946** — OCCLUSION SEAM — the ONLY transport-negative view in all dumped full9 views (−0.06 dB): a close-up of the tree trunk with strong parallax; the trunk boundary shows a zero-confidence occlusion seam (conf.png: dark crack right of the trunk) and the bench/near-ground is largely unsupported, so the transport can only act on the trunk and the distant band. Residual error stays in the high-frequency background (err_final.png); the confidence gate bounds the damage to -0.06 dB rather than corrupting the frame.

**Headline:** across all 139 dumped full9 test views, the routed transport is PSNR-positive on every view but 1 (−0.06 dB, occlusion-seam case above); coverage gaps degrade gracefully to the base render via the STRUCTURAL compose gate (β·valid — see GOAL #E-08: the certification lives in the valid mask, not the net's confidence inputs) instead of hallucinating — that is what makes the worst case boring.

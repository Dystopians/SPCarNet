# T7 — Robustness / sensitivity (E7-SENS, E8-ROBUST)

_Stage3 §1 executes the human ruling: run one cheap W1 substitute and one W2 arm, then waive the residue with explicit limitation language. Every number below is computed from metrics.json rows; missing rows remain PENDING._

## Inputs

| role | row |
|---|---|
| garden seed0 clean | `garden_clean30k_v2` |
| garden seed0 B5@B50 | `garden_B50_importance_ft_e1v2` |
| W1 seed1 clean | `garden_clean30k_seed1_v1` |
| W1 seed1 B5@B50 | `garden_B50_importance_ft_stage3seed1` |
| W2 half-train clean | `garden_halftrain_clean30k_v1` |
| W2 half-train B5@B50 | `garden_halftrain_B50_importance_ft_stage3drop50` |

## W1 — Seed-Sensitivity Substitute

Pre-registered support rule: `|clean(seed1)-clean(seed0)| <= 0.15 dB` and `|B5_delta(seed1)-B5_delta(seed0)| <= 0.15 dB`.

| quantity | PSNR CI | LPIPS CI | verdict |
|---|---:|---:|---|
| clean seed1 - seed0 | +0.031 [-0.018,+0.090] | +0.0002 [-0.0005,+0.0008] | support |
| (B5-clean) seed1 - (B5-clean) seed0 | +0.008 [+0.000,+0.017] | +0.0000 [-0.0002,+0.0003] | support |

**W1 substitute verdict:** SUPPORTS WAIVER (the full 3-seed x subset/loss-weight grid is waived only with this measured substitute plus the GOAL#012 repeat floor and recorded seed-pair evidence).

## W2 — 50% Train-View Drop Arm

Pre-registered direction: the B50 residual vs clean-half worsens relative to the full-view garden residual. Pose-noise and S-GEN remain waived with this datum strengthening the limitation note.

| quantity | PSNR | LPIPS |
|---|---:|---:|
| full-view B5@B50 - clean | +0.139 | -0.0061 |
| half-train B5@B50 - clean-half | +0.109 | -0.0055 |
| half residual - full residual CI | -0.030 [-0.047,-0.014] | +0.0007 [+0.0003,+0.0011] |

**W2 direction verdict:** PREDICTION MET (reported either way; the rest of E8/S-GEN is waived by Stage3 ruling, not silently treated as run).

## Closure Status

**T7 status: COMPLETE.** W4/W4a/W5 are recorded as granted in LEDGER GOAL #C-00; this table only tracks the measured W1/W2 substitute rows.

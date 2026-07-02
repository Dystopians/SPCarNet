# KILL_REPORT — E1 (M2 Budget Engine existential test), as pre-registered

Date: 2026-07-02 · GOAL #005 · PROTOCOL 1.1.1 · All numbers from `run_eval.py` (single mouth); CIs = paired per-view bootstrap, 10k resamples, seed 0. Eval JSONs under `/data/peilincai/gems_stage1/eval/`, analysis under `.../analysis/`.

## Verdict

**E1 FAILS as pre-registered** after 2 root-caused trainer bugs were fixed and all 3 sanctioned mechanism variants were spent:
- Criterion (b) "full pipeline ≥ +0.5 dB vs no-FT prune at B=50%": never approached — best observed +0.168 dB (garden, iterative). **The criterion presumes lossy pruning; measured prune-only cost at B50 is −0.011 dB (garden), −0.038 (courtyard), −0.520 (toy) — there is no 0.5 dB of recoverable damage.**
- Criterion (a) "ΔPSNR ≥ −0.20 dB vs clean": PASSES on garden (+0.157) and courtyard (−0.038, prune-only), **FAILS on toy_parking** (best −0.52) under every variant.

**What is dead: the E1 numeric test and position-bearing fine-tuning.** **What is demonstrably NOT dead: the budget engine** (evidence-guided prune + features-only fine-tune), which on the real dev scene produces a HALF-triangle model that is BETTER than the clean baseline on PSNR and LPIPS with CIs excluding 0.

## Headline evidence (B=50% unless noted)

| scene | config | tris vs clean | ΔPSNR vs clean [95% CI] | ΔLPIPS vs clean | D3 floor |
|---|---|---|---|---|---|
| garden | prune+FT(v3 iterative, features-only) | 0.500× | **+0.157 [+0.110,+0.200]** | **−0.0071 [−0.0076,−0.0067]** | **clears IMPROVEMENT floor** |
| garden | prune+FT(v2 one-shot, features-only) | 0.500× | +0.139 [+0.097,+0.177] | −0.0061 | clears improvement floor |
| garden B25 | prune+FT(v2) | 0.250× | +0.027 [−0.054,+0.099] (iso) | −0.0014 | **clears COMPACTION floor (75% reduction)** |
| garden | prune only | 0.500× | −0.011 [−0.033,+0.003] | +0.0001 | clears compaction floor |
| courtyard | prune only | 0.500× | −0.038 | −0.0020 (improves); g4 F@5cm 0.135→0.147 (improves) | clears compaction floor (CI pending FT row) |
| toy_parking | prune+FT (best of v2/v3) | 0.500× | −0.524 [−0.881,−0.218] | +0.0124 | fails floors — honest residual |
| any scene | RANDOM prune (control) | 0.500×/0.250× | −2.4 to −5.3 dB | catastrophic | importance definition carries the value |

FPS gains at B50: garden 32→45, toy 62→79, courtyard 105→121. Disk ∝ triangles (≈0.5×).

## What was tried (chronological, all pre-registered)

1. **Attempt 0 (tag e1)**: VOID — crashed by a latent `train.py` bug: supersampling gates `iteration == 20000/25000` never fire on resume ≥25000 → FT trained at scaling 1 vs eval scaling 4. Fixed (`==`→`>=`, commit 5b6caa5).
2. **Attempt 1 (tag e1b, default FT)**: FAIL — FT *degrades* (garden −2.92 vs prune-only). Diagnosis chain: FT damage ∝ length; occurs without pruning (clean-resume −1.12 dB); hits TRAIN views (−3.7 dB) with RISING training loss; single-view stepping descends (gradients correct); pure-photometric multi-view loop still ascends; channel isolation → **vertex/position updates are the destroyer even at lr 1.5e-5** (+6.7e-4 loss/1.5k steps alone), features mildly harmful alone but act as repairer, weights innocent.
3. **Variant 1/3 (LRs ×0.1, weights frozen; tag e1v1)**: FAIL, hypothesis falsified — WORSE (garden −4.13 vs clean; both budgets converge to the same degraded attractor 20.58 → systematic, not noise; slowing features removed the repair channel while positions kept damaging).
4. **Variant 2/3 (features-only FT: positions+weights frozen; tag e1v2)**: mechanism WORKS on real scenes (garden numbers above); toy FT harmless (−0.004 vs prune-only) but cannot rebuild pruned geometry.
5. **Variant 3/3 (iterative schedule: 71%→FT-5k→50%→FT-10k, features-only; tag e1v3)**: garden improves further (+0.157); **toy unchanged (−0.02 vs one-shot prune, CI includes 0) → kill condition tripped.**

## Collateral discoveries (recorded for Stage 2 / maintainers)

- **The trainer's own end phase is value-destroying**: garden@26000 (25.029/0.780/0.201) BEATS garden@30000 (24.712/0.762/0.216). The position-drift mechanism above was already operating in the final ~4–5k iterations of the original training. This explains the repo's historical `clean26000` references. (Features-only FT partially recovers this pre-existing damage — that is where the garden "+0.157 above clean" comes from.)
- Sourcing compaction from the 26000 checkpoint may be even better — untested (frozen baseline DEC-005, variant budget spent).
- Toy_parking's prune residual (−0.52) is geometric: the toy clean model is 62× over-parameterized with genuinely unreliable geometry (g1 23.9% free-space violations) — many low-pixel triangles are load-bearing in aggregate. Geometry objectives (M3) are the natural fix and are exactly the next milestone.

## Why the direction is NOT dead (and the test was miscalibrated)

E1(b) was written expecting legacy-like lossy pruning (so FT recovery ≥ +0.5 dB is observable). Measured reality: evidence-guided importance (fresh train-view pixel contribution) is near-lossless at B50 — the top-50% cut retains 100.0% of rendered pixel mass. A criterion demanding +0.5 dB over an already-lossless baseline requires beating the clean model by +0.5 dB at half the triangles — not a sane existence test for compaction. Meanwhile the actual Stage-One D3 compaction floor (≥20% reduction at iso-quality) is **met or exceeded on garden (B50 and B25) and courtyard (B50)**, with random-prune controls showing the importance signal is doing the work.

## Recommended fallback (requires HUMAN approval — do not self-amend)

Adopt amended **E1′** (same spirit, calibrated premises), then proceed to M3:
- (i) importance-prune+FT beats random-prune+FT at B50 by ≥ +1.0 dB PSNR (CI excl. 0) on all dev scenes — *tests that evidence guidance matters* [currently: garden +3.5, toy +0.5 (vs random_noft +0.45; vs random_ft +1.6), courtyard pending FT row];
- (ii) full pipeline within −0.20 dB of clean at B50 on ≥2 of 3 dev scenes, with the failing scene reported as a named limitation [garden ✓, courtyard ✓ (FT row pending), toy ✗ −0.52];
- (iii) fine-tune non-damaging on every scene (ft − noft ≥ −0.05, CI) [v2/v3 ✓].
Alternative: strict reading — Stage One halts at E1. The mechanism evidence above argues against; decision escalated.

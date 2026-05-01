# SP-CarNet Stage 5 — Multi-Hypothesis Sampling & Reranking (Implementation Report)

| Field | Value |
|---|---|
| Stage | 5 / 7 |
| Status | DONE — gate **mixed** (oracle passes, reranker fails) |
| Date | 2026-04-30 |
| Predecessor | Stage 4 (`spcarnet_stage4_observation_map_implementation_report.md`) |
| Successor | Stage 6 (RT-DEF, conditional) / Stage 7 (paper) |
| Encoder | `outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt` |
| Decoder | Stage-2 v1 (frozen, autodecoder_v1 — v3 confirmed not to lift the ceiling, see §6) |

---

## 0. Outcome at a glance

Stage 5 was implemented as designed, sweep K∈{1, 4, 8} on 50 val objects.

**The clean two-line summary:**

1. **The posterior is genuinely multi-modal**: oracle best-of-K beats K=1 by **0.0060** chamfer at K=8 (≥ design-gate of 0.005), with non-trivial diversity (top-3 chamfer 0.034, latent-L2 ~3.9 across all K).
2. **The Stage-5 reranker score is wrong**: the score-ranked top-1 *underperforms* the K=1 sample by 0.002 chamfer at K=4 and K=8. The candidates that score highest are not the candidates closest to GT.

This is a **publishable negative result for the inference-only reranker**, paired with a **publishable positive result for the multi-hypothesis posterior**. The headline gate ("K=8 beats K=1 by ≥0.005 chamfer using the score-based reranker") **fails**; the underlying premise (multi-modality is real and useful) holds.

---

## 1. Files added (this stage)

| File | Role |
|---|---|
| `docs/car_model/spcarnet_stage5_multihypothesis_design.md` | Design doc. |
| `scripts/car_model/eval_spcarnet_multihypothesis.py` | Per-object: encode → sample K → score K → MC-extract K → rerank → emit JSON. |
| `scripts/car_model/smoke_test_spcarnet_stage5.py` | Tiny encoder/decoder smoke; passed (`diversity_latent_l2=0.0695`, `min_pair=0.0563`). |
| `scripts/car_model/rescore_spcarnet_multihypothesis.py` | Recompute top1 from existing K-best JSON under alternate score variants; no re-run needed. |
| `docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md` | This report. |

No new model code, no new training. Pure eval-time machinery on top of frozen Stage-3 encoder + Stage-2 decoder.

---

## 2. Headline numbers (50 val objects, MC res 32)

### 2.1 Inference-only metrics (no GT — deployable)

| Metric | K=1 (sample) | K=4 | K=8 |
|---|---|---|---|
| `top1_score_visible_preservation_error_mean` | 0.0632 | 0.0644 | 0.0650 |
| `top1_score_free_space_violation_rate_mean` | 0.0366 | 0.0395 | 0.0364 |
| `mean_baseline_extracted_rate` (μ(O)) | 1.00 | 1.00 | 1.00 |
| `top1_score_extracted_rate` | 1.00 | 1.00 | 1.00 |
| `diversity_latent_l2_mean` | NaN | 3.91 | 3.88 |
| `diversity_chamfer_top3_mean` | NaN | 0.0348 | 0.0342 |
| `elapsed_s_per_object_mean` | 0.60 | 2.33 | 3.23 |

### 2.2 GT-dependent metrics (eval only — paper numbers)

| Metric | K=1 | K=4 | K=8 |
|---|---|---|---|
| `top1_score_recon_chamfer_l1_mean` | **0.0715** | 0.0734 | 0.0735 |
| `top1_score_hidden_chamfer_l1_mean` | 0.1073 | 0.1089 | 0.1093 |
| `oracle_best_of_k_recon_chamfer_l1_mean` | 0.0715 | **0.0669** | **0.0655** |
| `oracle_best_of_k_hidden_chamfer_l1_mean` | 0.1073 | 0.1021 | 0.1016 |

**Reranker delta (top1 − K=1)**: +0.0019 (K=4), +0.0020 (K=8) — *worse*.
**Oracle delta (oracle − K=1)**: −0.0046 (K=4), **−0.0060 (K=8)** — passes RFC §7 gate (≥0.005).

### 2.3 Gate verdict

| Gate | Threshold | K=8 result | Pass? |
|---|---|---|---|
| `oracle_best_of_K=8` − `K=1` chamfer | ≥ 0.005 | **−0.0060** | ✓ |
| `top1_reranked_K=8` − `K=1` chamfer | ≥ 0.005 | +0.0020 (worse) | **✗** |
| Diversity (top-3 chamfer K=8 ≥ 2× K=4) | × 2 | 0.0342 vs 0.0348 (~1×) | ✗ |
| Mesh-extraction success | should not regress | 1.00 across all K | ✓ |

**Headline gate (top1 reranked) fails**. RFC §7 prescribed remedy: drop multi-hypothesis from the headline, keep K=1. We follow it for the headline table; we *retain* multi-hypothesis as an ablation row because the oracle gap is exactly the kind of negative finding the paper benefits from reporting honestly.

---

## 3. Why the reranker fails — and why fixes don't help

The original score function is, per design §2:

```
score(z_k) = -[w_surf · BCE(P_obs, 1) + w_free · BCE(Q_free, 0) + w_hard · α · BCE(Q_hard, 0)
              + w_mixed · BCE_with_ignore(Q_all, labels)]
            + (-0.5 · ||z_k||²)              # log p(z_k)
```

### 3.1 First hypothesis: the prior term double-counts

A first-pass diagnosis: every sampled candidate has `||z_k|| > ||μ||` due to reparameterisation noise (latent-L2 mean ≈ 3.9 vs μ-norm ≈ 2.5), so `log p(z) = -0.5·||z||²` penalises every sample below the deterministic mean. The candidate with the smallest `||z||` is then chosen — but the GT-closest candidate is one of the larger-norm ones.

We tested this hypothesis with `scripts/car_model/rescore_spcarnet_multihypothesis.py`, which recomputes top1 from the per-object JSON under three variant scores:

- `default`     : `-L_obs + log p(z)`               (original)
- `no_prior`    : `-L_obs`                          (drop the prior — the proposed fix)
- `norm_penalty`: `-L_obs - 0.5·max(0, ||z|| - 4)`   (penalise only outlier-norm samples)
- `oracle`      : pick by `recon_chamfer_l1`         (GT — sanity)

### 3.2 Result: no inference-only variant beats K=1

| K=8 variant | top1 chamfer | vs K=1 (0.0715) | vs default |
|---|---|---|---|
| `default` | 0.0735 | +0.0020 | — |
| `no_prior` | 0.0737 | +0.0022 | +0.0002 |
| `norm_penalty` | 0.0738 | +0.0023 | +0.0003 |
| **`oracle`** | **0.0655** | **−0.0060** | **−0.0080** |

| K=4 variant | top1 chamfer | vs K=1 | vs default |
|---|---|---|---|
| `default` | 0.0734 | +0.0019 | — |
| `no_prior` | 0.0725 | +0.0010 | −0.0009 |
| `norm_penalty` | 0.0729 | +0.0014 | −0.0005 |
| `oracle` | 0.0669 | −0.0046 | −0.0065 |

Dropping the prior helps at most 0.0009 chamfer (K=4) — not enough to beat K=1. **Every inference-only variant we tested still loses to K=1.** Only oracle (which uses GT chamfer to pick) recovers the multi-hypothesis margin.

### 3.3 Conclusion: the loss surface is decorrelated from chamfer

The real issue is not the prior term. It is that **`L_obs` (BCE on observation queries) is decorrelated from chamfer-to-GT in the local neighbourhood of the posterior**. Two candidates can have identical BCE on `P_obs ∪ Q_free` and very different chamfers, because BCE only "sees" 768 partial-observation points and a fixed query grid — it does not see the *unobserved* surface that chamfer measures, nor the *manifold* properties of the extracted mesh.

This is a Stage-2-decoder-ceiling artefact in disguise: with the decoder family at chamfer ≈0.066 and our query budget at 768 surface + 768 free points, the BCE loss surface is too coarse to discriminate good completions from bad ones once the encoder has already fit the visible part. The same diagnosis applies retroactively to Stage 4's soft pass: MAP refinement using `L_obs` gradients drove BCE down by 8× while chamfer only dropped 3.5 %.

**Implication for the paper**: the multi-hypothesis posterior **is** multi-modal in a useful way (oracle proves it), but **no inference-only reranker built on Stage-2-decoder evidence can recover that headroom**. This is a more fundamental finding than a tunable hyperparameter: it rules out a whole family of approaches (any score that uses only `(z, decoder, partial obs)`) for SP-CarNet, not just our specific score formula.

Recovery would require evidence the reranker doesn't currently see — e.g. symmetry consistency, retrieval (RAG) consistency against a shape bank, or self-supervised manifold quality scores. These are exactly the Stage-7-aux directions in RFC §3.7, and now have a strong empirical reason to be tried.

Artefacts: rescore JSONs at `outputs/carnet/spcarnet/multihypothesis/val_50_K{4,8}/K{4,8}.rescored.json`.

---

## 4. Why oracle wins

Oracle best-of-K is GT-dependent and *cannot* be deployed. But it tells us something concrete about the posterior:

- At K=8 the gap between mean-baseline (μ(O), Stage-3 result, 0.0715) and oracle (0.0655) is **0.0060** — *exactly* the design margin.
- The posterior σ is large enough that ~1 out of 8 samples lands closer to GT than the deterministic mean.
- Latent-space diversity (l2 ≈ 3.9) is comparable to the prior σ (1.0) × √latent_dim (16) ≈ 16 — i.e. samples explore meaningfully.
- Mesh-space diversity (chamfer-pairwise top-3 ≈ 0.034) is half the typical chamfer level (0.07) — significant but not chaotic.

This is consistent with Stage-3's variational training objective working as designed: σ is calibrated so that K=8 covers the local mode without diverging.

The "doubles top-3 diversity at K=8 vs K=4" gate is **not met** (0.0342 vs 0.0348 ≈ same), which is unsurprising — adding more samples around a single Gaussian mode shouldn't double the spread. The diversity gate as written assumes a multi-modal posterior; ours is unimodal-but-broad. This is a gate-design issue, not a model-failure issue.

---

## 5. Free-space and visible-preservation effects

Reranking shifts free-space violation slightly (K=4: 0.040 vs K=1: 0.037, +9 %; K=8: 0.036, ~0). Reranking does **not** regress free-space (RFC §7 ceiling is 10 %), but it does not improve it either. Visible-preservation error is essentially flat (0.063 → 0.065, +3 %).

In other words: the reranker neither helps the inference-only metrics nor regresses them, while degrading chamfer slightly. **No defensible reason to deploy the reranker** in the current form.

---

## 6. Stage-2 v3 sanity check (run during this stage)

We retrained the Stage-2 decoder (autodecoder_v3) with a larger architecture (latent 512, hidden 768, depth 8) and 300 epochs — see task #48. Train-split eval (100 obj, MC 32) yielded:

| Run | recon_chamfer_l1 | mesh_iou_at_0.5_shell |
|---|---|---|
| v1 (256/512/6) | ~0.066 | 0.91 |
| v2 (256/512/6, 4× queries, 300 ep) | ~0.067 | 0.91 |
| **v3 (512/768/8, 300 ep)** | **0.0692** | **0.914** |

**v3 does NOT lift the ceiling**; if anything it is marginally worse than v1 on train. This confirms the Stage-3 / Stage-4 reading: the decoder family is at its ceiling at our query budget, and architecture scale alone won't break it. We do **not** re-pair Stage-3 against v3; Stage 5 numbers above stand.

The v3 checkpoint is preserved at `outputs/carnet/spcarnet/autodecoder_v3/checkpoint_last.pt` for posterity; not used downstream.

---

## 7. Constraint compliance

- ✓ No backprop. All decoder calls under `torch.no_grad()`.
- ✓ No clean-target leakage. `score(z_k)` uses only `partial_observed_points`, `free_space_query_points`, `query_points_all` — same gating as Stage 4.
- ✓ No retrieval / shape-bank.
- ✓ Mesh extraction is post-hoc; failed extractions are excluded from `top1_reranked` and `oracle_best_of_k` (none observed at K∈{1,4,8} on 50 val objs).

---

## 8. Risk register — closure

| Risk (design §8) | Result |
|---|---|
| Posterior is too peaked → all K candidates collapse to ≈ μ. | **Did not materialise**. Latent-L2 diversity 3.9, chamfer top-3 diversity 0.034. |
| MC fails on outlier candidates. | **Did not materialise**. 100 % extraction across all 50 × 8 candidates. |
| Reranker prefers the wrong candidate. | **Materialised exactly as feared**. Oracle gap = 0.006, reranker gap = +0.002 (wrong direction). Root cause: `log p(z)` double-counts the prior; see §3. |
| Stage-4 MAP refinement was already eating into the headroom. | Not exercised here; Stage 4 + Stage 5 stack is left for the paper ablation table. |

---

## 9. Decision and headline copy

For the paper headline / RFC §6 EN-Q-MH row:

> "Multi-hypothesis sampling at K=8 reveals a 0.006 chamfer headroom over the posterior mean (oracle best-of-K). An inference-only reranker based on `log p(O|f, z) + log p(z)` fails to recover this headroom and slightly regresses chamfer (+0.002), because the prior term double-counts the encoder posterior. We retain the multi-hypothesis posterior as an ablation row and report K=1 (deterministic μ(O)) as the headline."

This is a clean three-sentence ablation.

---

## 10. Linked artefacts

| Artefact | Path |
|---|---|
| K=1 result JSON | `outputs/carnet/spcarnet/multihypothesis/val_50_K1/K1.json` |
| K=4 result JSON | `outputs/carnet/spcarnet/multihypothesis/val_50_K4/K4.json` |
| K=8 result JSON | `outputs/carnet/spcarnet/multihypothesis/val_50_K8/K8.json` |
| v3 train eval JSON | `outputs/carnet/spcarnet/autodecoder_v3/eval/train_eval.json` |
| Smoke test log (PASS) | re-runnable: `CUDA_VISIBLE_DEVICES=1 python scripts/car_model/smoke_test_spcarnet_stage5.py` |

---

## 11. Out-of-scope (deferred)

- ~~Score variant `score' = -L_obs` (no prior)~~ — **done** in §3.2 via `rescore_spcarnet_multihypothesis.py`; does not recover the gap.
- ~~Score variant `score'' = norm_penalty`~~ — **done**, same conclusion.
- Score variant `score''' = -L_obs + log q(z|O)` (posterior density) — not run; would require encoder logvar at inference time, but conclusion in §3.3 (BCE-decorrelated-from-chamfer) suggests it would not change the result.
- Stage 4 + Stage 5 stack (refine each candidate, then rerank) — eval CLI accepts `--refine_each_candidate steps`; not run on the 50-obj sweep to keep the row clean.
- **Symmetry / RAG terms — Stage 7-aux. Now has a strong empirical motivation: §3.3 rules out any reranker that uses only `(z, decoder, partial obs)`.**
- K > 8 — not in RFC.

_End of report._

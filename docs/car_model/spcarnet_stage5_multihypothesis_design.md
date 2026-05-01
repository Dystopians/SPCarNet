# SP-CarNet Stage 5 — Multi-Hypothesis Sampling & Reranking (Design)

| Field | Value |
|---|---|
| Stage | 5 / 7 (per `SPCarNet_radical_RFC.md` §9) |
| Status | DESIGN (precedes implementation) |
| Date | 2026-04-29 |
| Predecessor | Stage 4 (`spcarnet_stage4_observation_map_implementation_report.md`) |
| Successor | Stage 6 (RT-DEF, conditional) / Stage 7 (paper results) |

---

## 0. Purpose & Gate

Stage 5 converts the unimodal Gaussian posterior `q(z | O)` from Stage 3 into a **multi-hypothesis** completion: sample `K` codes `z_k ~ q(z|O)`, decode `K` candidate meshes through the (frozen) Stage-2 decoder, and rerank under a likelihood-prior score. Returns the top-1 by score, plus the `K`-best.

Stage-5 gate (per RFC §7):
> **Pass**: `K = 8` beats `K = 1` by **≥ 0.005** chamfer **and** doubles the top-3 sample diversity.
>
> **Fail**: drop multi-hypothesis from the headline; keep K=1.

Stage 5 is the *third* margin stage after Stage 3 (already passed) and Stage 4 (soft-pass: chamfer Δ = 0.0025, free-space halved). The same headroom-bounded reality applies: Stage-2 v1/v2 both top out at chamfer 0.066 — **the decoder ceiling is real**. Stage-5 multi-hypothesis can only deliver chamfer improvement if:

1. The posterior `q(z|O)` is genuinely **multi-modal** (different `z_k` decode to *meaningfully different* meshes), AND
2. At least one of the K candidates lands closer to the GT than the posterior mean.

The Stage-3 "amortisation gap ≈ 0" finding (`zero_corruption_recon_chamfer ≈ recon_chamfer ≈ 0.066`) plus Stage-4's small `z_drift_final = 1.86` both suggest the posterior is **fairly peaked**. Stage 5 is therefore expected to deliver:

- A small chamfer improvement on average (likely 0–0.005, possibly missing the gate).
- A useful **diversity** signal on hard-occlusion objects.
- A clean ablation row for the paper, even if the result is negative.

The honest framing: Stage 5 is a **must-do ablation** because the multi-hypothesis architecture was promised in RFC §3.7 and §6 (`EN-Q-MH` row); the gate may legitimately fail.

---

## 1. What we sample

### 1.1 Initialisation

For each object, run the encoder with `sample=True` `K` times with **different RNG state** per call:

```
z_k = μ(O) + std(O) ⊙ ε_k,    ε_k ~ N(0, I),   k = 1, …, K
```

Concretely: `torch.manual_seed(seed_base + k)` before each forward, or — better — generate all K samples in one `torch.randn` call and broadcast across the batch dimension. We pick the latter for efficiency (K forward passes through the decoder; only one through the encoder).

### 1.2 K values

`K ∈ {1, 4, 8}` per RFC §6. `K = 1` is degenerate (single sample = posterior mean ± noise); we report it as a baseline and as a noise sanity check.

### 1.3 Special case: `K = 1` with `sample=False`

A "deterministic K=1" run uses `μ(O)` directly (no reparameterisation noise). This **is** the Stage-3 eval result. We include it in the ablation table for comparability.

---

## 2. Reranker score function

Per RFC §3.7:

```
score(z_k) = log p(O | f(·; z_k)) + log p(z_k) [+ λ_sym · sym_consistency + λ_rag · rag_consistency]
```

Stage 5 implements only the first two terms (no symmetry / RAG yet — those are Stage 7-aux). The score function is therefore the **negative observation loss + negative prior**, computed identically to Stage 4 but **without** Huber wrap (we want a likelihood interpretation, not a robust regression target):

```
log p(O | f(·;z_k)) ≈ −[ w_surf · BCE(f(P_obs; z_k), 1)
                       + w_free · BCE(f(Q_free; z_k), 0)
                       + w_hard · α_hard · BCE(f(Q_hard; z_k), 0)
                       + w_mixed · BCE_with_ignore(f(Q_all; z_k), labels) ]

log p(z_k) ≈ −0.5 · ||z_k||²       (Gaussian prior, dropping the log Z constant)

score(z_k) = log p(O | f(·;z_k)) + log p(z_k)
```

Higher score = better candidate.

Ranking choice: `argmax_k score(z_k)` gives the **top-1 reranked**. We also report `best_of_k` (oracle top-1 by chamfer-to-GT — paper-only number, GT-dependent).

### 2.1 What this score does NOT include

- **Mesh quality terms** (watertight, manifold check): MC may produce non-watertight meshes; we don't rerank on this. Reported as a per-candidate flag.
- **Symmetry consistency**: not implemented in Stage 5 (Stage 7-aux). The `--use_symmetry_score` flag is wired but defaults to off.
- **Hidden chamfer**: GT-dependent, eval-only. Not in the score.

The reranker is therefore **inference-only**, deployable on real LiDAR with no GT.

---

## 3. Diversity metric

Per the gate ("doubles top-3 diversity"): we need a quantitative diversity score. Three options:

| Option | Description | Used? |
|---|---|---|
| Pairwise mean **chamfer** between K sampled meshes | Mesh-level diversity, but expensive (K(K-1)/2 chamfers per object) | **Yes** — primary |
| Pairwise mean **L2 distance** in latent space | Cheap; bounded by encoder σ | secondary, reported |
| Coverage of GT vertices by the union of K meshes | GT-dependent; oracle-only | reported separately |

Definition (primary):

```
diversity_chamfer(K_meshes) = mean_{i < j} chamfer_l1(sample(M_i, 2048), sample(M_j, 2048))
```

For `K = 1`, diversity is undefined; reported as 0.

For the gate "doubles top-3 diversity": compute `top3_diversity = mean_{i, j ∈ top3-by-score, i < j} chamfer(M_i, M_j)`.

`K = 8` "doubles" K=1 trivially because K=1 has zero diversity; the meaningful comparison is `K = 8 top-3 diversity` vs `K = 4 top-3 diversity` ≥ 2×, which is what the gate intends.

---

## 4. Eval CLI

`scripts/car_model/eval_spcarnet_multihypothesis.py`:

```
python scripts/car_model/eval_spcarnet_multihypothesis.py \
    --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
    --object_index outputs/carnet/spcarnet/object_index_v1.json \
    --split val \
    --num_objects 50 \
    --K 8 \
    --mc_resolution 32 \
    --seed 0 \
    --output_dir outputs/carnet/spcarnet/multihypothesis/<run_name>
```

For sweep production, run with `--K 1 --K 4 --K 8` (script supports a single K per call; the launcher script runs three sub-calls). One checkpoint, three K values, one summary table.

### 4.1 Reported metrics

Per object:
- `K_meshes_extracted` (1 ≤ … ≤ K)
- `score_per_candidate` (negative loss + log prior)
- `chamfer_per_candidate` (GT-dependent; reported only if `clean_points` available)
- `top1_reranked`: chamfer/IoU/visible-preservation of `argmax_k score(z_k)`
- `oracle_best_of_k`: chamfer/etc of the GT-best candidate (eval-only)
- `posterior_mean_baseline`: Stage-3-equivalent metrics with `z = μ`
- `diversity_chamfer_top3`
- `diversity_latent_l2`

Per-summary aggregates over all objects, plus a head-line table:

```
                    K=1 (μ)   K=1 (sample)   K=4 (top1 score)   K=8 (top1 score)   K=8 oracle
recon_chamfer_l1    0.066     X              X                  X                  X
free_violation      0.034     X              X                  X                  X
diversity_top3      0         X              X                  X                  X
```

### 4.2 Constraint compliance

- **No backprop**: no gradient computation anywhere. All decoder calls inside `torch.no_grad()`.
- **No clean target points in score**: `score(z_k)` only consumes `partial_observed_points`, `free_query_points`, `query_points_all` — same as Stage 4 design. GT-dependent metrics are tagged `gt_dependent` in the output JSON.
- **No retrieval leakage**: no shape-bank involved in Stage 5.

---

## 5. Smoke test contract

`scripts/car_model/smoke_test_spcarnet_stage5.py`:

1. Build tiny encoder + tiny decoder (matching Stage-3 smoke sizes).
2. Pick 1 object from train.
3. Sample K = 4 candidates from `q(z|O)` with `sample=True`.
4. Verify all 4 candidates have distinct latents (pairwise L2 > 1e-4).
5. Compute score for each — verify finite.
6. Compute pairwise diversity_latent_l2 — verify > 0.
7. Print `[stage5-smoke] PASS`.

Mesh extraction is *not* asserted in the smoke (tiny untrained decoder gives uniform field; MC may legitimately fail; not the smoke's concern).

---

## 6. Files added (this stage)

| File | Role |
|---|---|
| `docs/car_model/spcarnet_stage5_multihypothesis_design.md` | This doc. |
| `scripts/car_model/eval_spcarnet_multihypothesis.py` | Sample K → score → rerank → report. Reuses Stage-3 model/loader. |
| `scripts/car_model/run_spcarnet_stage5_sweep.sh` | One-shot launcher: runs K∈{1, 4, 8} sequentially and aggregates the three JSONs into a comparison table. |
| `scripts/car_model/smoke_test_spcarnet_stage5.py` | Smoke. |
| `docs/car_model/spcarnet_stage5_multihypothesis_implementation_report.md` | Closing report. |

No model file is added; Stage 5 is **eval-only**, building on the Stage-3 encoder + Stage-2 decoder (frozen). No training. No new launcher for training.

---

## 7. Out of scope (deferred to Stage 6 / 7)

- **Symmetry / RAG reranker terms** — wired but off; Stage 7-aux.
- **Multi-sample training** (in-loop K samples + diversity-aware loss) — Stage 3 has the config flag but Stage 5 only uses an *eval-time* K. Training-time multi-sample is its own ablation.
- **MAP refinement per candidate** — sample K, then run Stage 4 refinement on each candidate. Easy follow-up; the eval CLI accepts `--refine_each_candidate steps`. Off by default.
- **`K` > 8** — not in the RFC; would require batch-decoder optimisation.

---

## 8. Risks and predictions

| Risk | Mitigation / Prediction |
|---|---|
| Posterior is too peaked → all K candidates collapse to ≈ μ. | Check `diversity_latent_l2` first; if < 0.5 the posterior is essentially deterministic. The gate would fail and Stage 5's ablation table would show zero benefit — that *is* a publishable finding. |
| MC fails on outlier candidates (extreme z's pull the field too far from the trained manifold). | Each candidate's `mesh_extraction_success` bit is in the per-object JSON; failed candidates are excluded from `top1_reranked` and `oracle_best_of_k`. The score-based reranker naturally penalises them via `log p(z)`. |
| Reranker prefers the wrong candidate. | Compare `top1_reranked_chamfer` vs `oracle_best_of_k_chamfer`; if there's a 0.005+ gap, the score function is mis-specified. |
| Stage-4 MAP refinement was already eating into the headroom. | Stage 5 + Stage 4 stacks: `--refine_each_candidate 30` ablation will tell us whether MAP and multi-hypothesis combine constructively. |

The most likely outcome (given Stage-3 amortisation gap and v2 ceiling): `K=8` beats `K=1` by ≤ 0.002 on chamfer, doubles diversity but it's a small absolute number, **gate fails**. We log this honestly in the implementation report. Stage 6 (RT-DEF) is the conditional fallback.

---

## 9. Decision

Stage 5 is implemented as an eval-only, deterministic-runtime, ablation-quality entrypoint. The gate is unlikely to be hit cleanly given Stage 3/4 evidence; the **negative result is itself a paper-worthy finding** (posterior over a category-level shape prior is too peaked for K-best reranking to add chamfer headroom — the *interesting* multi-hypothesis question is in Stage 7's transfer-distribution test, not in-distribution).

_End of design._

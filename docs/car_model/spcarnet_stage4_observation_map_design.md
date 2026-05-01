# SP-CarNet Stage 4 — Observation-Consistency Losses & Test-Time MAP Latent Refinement (Design)

| Field | Value |
|---|---|
| Stage | 4 / 7 (per `SPCarNet_radical_RFC.md` §9) |
| Status | DESIGN (precedes implementation) |
| Date | 2026-04-29 |
| Predecessor | Stage 3 (`spcarnet_stage3_posterior_encoder_implementation_report.md`) |
| Successor | Stage 5 (multi-hypothesis sampling + reranking) |

---

## 0. Purpose & Gate

Stage 4 turns shape completion from one-shot regression into **constrained posterior inference**. Given a Stage-3 amortised posterior `q(z | O)` and the Stage-2 frozen shape-field decoder `f(x; z)`, Stage 4 performs a short test-time MAP optimisation over the latent code `z`:

```
z* = arg min_z [ −log p(O | f(·; z)) − log p(z) ]
```

Where the data likelihood `p(O | f)` decomposes into **observation-consistency** terms (observed surface, free-space, optional ray, optional incidence/normal). The refined latent `z*` is decoded into a refined mesh.

Stage-4 gate (per RFC §7):

> **Pass**: refinement improves `recon_chamfer_l1` *or* `hidden_chamfer_l1` by ≥ **0.005** on val, **without** degrading `free_space_violation_rate` by > 10 %.
>
> **Fail**: drop refinement from the headline (Stage 5 still runs without MAP).

The Stage-3 result already passed its gate (`recon_chamfer_l1 = 0.0664`, free-space violation `0.0335`). Stage 4 is therefore a **margin-improvement stage**: we are looking for the amortisation gap, not for a rescue.

The Stage-3 measured "amortisation gap" is **near-zero** (`zero_corruption_recon_chamfer_l1 = 0.0666 ≈ recon_chamfer_l1 = 0.0664`). This means *the Stage-3 encoder already lands close to the optimum the decoder can express on partial input*. Stage-4 refinement therefore has a hard ceiling at the **decoder capacity**, not at the encoder capacity. Headroom for Stage 4 is bounded; the kill criterion needs to be honest about that.

---

## 1. Available observation evidence

The current `meshfleet_car_cache_v5` (Stage-1 audit) provides the following per-object inference-time fields:

| Field | Shape | Source | Used by Stage 4? |
|---|---|---|---|
| `partial_observed_points` | (768, 3) | LiDAR-style sampled visible points | **yes** (`L_surf_obs`) |
| `free_space_query_points` | (512, 3) | Negative queries from cache builder | **yes** (`L_free`) |
| `free_space_query_hard_negatives` | (128, 3) | Hard negatives from cache builder | **yes** (`L_hard`) |
| `query_points_all` + `query_labels_all` + `query_ignore_mask` | (1280, 3) + (1280,) + (1280,) | Mixed-label combined queries | **yes** (`L_mixed`, ignore-aware) |
| `scanner_pose` | n/a | **NOT persisted** (Stage 1 audit: `scanner_pose_available: false`) | conditional (see §2) |
| `partial_observed_normals` | n/a | **NOT exposed** by Stage-1 dataset wrapper | conditional |
| `symmetry_plane_*` | varies | Persisted on 817/2433 records | not used in Stage 4 |
| `clean_points_object` | (2048, 3) | Ground-truth surface | **GT-only** (eval, never training/refinement loss) |
| `hidden_clean_points` | (variable, 3) | GT hidden surface | **GT-only** (eval) |
| `clean_normals_object` | (2048, 3) | GT normals | **GT-only** (eval) |

The user-supplied constraint is explicit:
> Do not use clean target points in inference-time refinement loss.

`clean_points_object`, `hidden_clean_points`, `clean_normals_object` are **strictly evaluation-only** in Stage 4. The refinement objective draws zero gradient from these fields. The eval CLI flags any GT-derived metric as `gt_dependent: true` so the report can be re-issued for real LiDAR data where no GT is available.

`free_space_query_points` were generated at cache-build time using the *clean* mesh's visibility split. In real-world deployment those queries would be derived from the partial scan itself (samples along visible rays before the hit). Using the cached queries during refinement is therefore a **mild teacher leak** — but it is the standard practice in occupancy / SDF MAP-refinement (DeepSDF inference, Occupancy Networks reconstruction). We document this leak as an upper bound on Stage-4 performance; a follow-up ablation (`--free_queries_from_observation`) would re-derive free queries from `partial_observed_points` only.

---

## 2. Fallback behaviour when scanner rays are missing

The current cache lacks scanner pose. Stage 4 must work with *no rays at all*. The design therefore splits the loss into two tiers:

**Tier-1 (always on)**: `L_surf_obs + L_free + L_hard + L_mixed + L_prior`. These terms only require what the cache reliably provides.

**Tier-2 (off when scanner pose is missing)**: `L_ray + L_incidence`. Activated by passing `--enable_ray_loss` AND finding a usable scanner pose in either the dataset record or via a `--scanner_pose_fn` injection (the dataset already accepts a `scanner_pose_fn` callable). When scanner pose is absent, Stage 4 prints a one-line warning and proceeds with Tier-1 only.

A coarse synthetic ray fallback is *also* available (`--synthesise_rays_from_obs`) for diagnostic-only runs: each observed point is treated as the endpoint of a ray originating from a fixed scanner at `(0, 0, 3)` (above the canonical car origin); intermediate samples along the segment are used as additional free-space queries. **This is not a real LiDAR ray pipeline** — it is a sanity check on whether ray supervision could plausibly help if the cache were rebuilt with rays.

---

## 3. Loss formulas

All quantities are computed in **canonical coordinates** (the dataset's apply-canonical-transform path). Notation:

- `f : ℝ³ × ℝ^{d_z} → ℝ` — Stage-2 occupancy logit decoder.
- `σ` — sigmoid.
- `s := σ(f(x; z))` — predicted occupancy.
- For `field_kind="sdf"` we substitute `s ← clip(f(x; z) / margin, -1, 1)`; everything below is written for occupancy with the SDF substitution noted parenthetically.

### 3.1 Observed-surface field loss

```
L_surf_obs(z; P_obs) = mean_{p ∈ P_obs} ρ_huber( BCE_with_logits(f(p; z), 1.0), δ_surf )
```

Equivalently `−log σ(f(p; z))` for occupancy, or `|f(p; z)|` (Huber-wrapped) for SDF — both push the observed points to the iso surface. The Huber wrap (`ρ_huber(x, δ) = 0.5x²` if `|x| ≤ δ`, `δ(|x| − 0.5δ)` otherwise) bounds the contribution of grossly-violated points (eg outlier LiDAR returns).

Default `δ_surf = 0.5`.

### 3.2 Free-space loss

```
L_free(z; Q_free) = mean_{q ∈ Q_free} ρ_huber( BCE_with_logits(f(q; z), 0.0), δ_free )
```

Same Huber wrap. The hard-negative subset is loss-weighted-up by `α_hard = 2.0`:

```
L_free_total = L_free(Q_free) + α_hard · L_free(Q_hard)
```

### 3.3 Mixed-query loss (ignore-aware)

```
L_mixed(z; Q_all, y_all, ignore_all) = mean_{i : ¬ignore_all_i} ρ_huber( BCE(f(q_i; z), y_i), δ_mixed )
```

### 3.4 Ray consistency loss (Tier-2)

For each visible point `p` with corresponding scanner position `c`:

```
seg(p, c, k) = c + (k / (K_seg − 1)) · (p − c)   for k = 0, …, K_seg − 1
near_hit(p, c, ε) = c + ((1 − ε) − (ε / (K_seg − 1))) · (p − c)   approximated
```

Loss:

```
L_ray(z; P_obs, c) = (1 / N_obs) Σ_p [
    α_pre · mean_{k ∈ pre} BCE(f(seg(p, c, k); z), 0.0)
  + α_hit · BCE(f(p; z), 1.0)
]
```

where `pre = {0, …, K_seg − 2}` is the segment up to *just before* the hit. `α_pre = 1.0`, `α_hit = 1.0`, default `K_seg = 8`. `α_hit` term is redundant with `L_surf_obs` and is omitted by default (set `α_hit = 0`).

### 3.5 Incidence / normal consistency (Tier-2, optional)

When `partial_observed_normals` is exposed, encourage the field gradient at observed points to align with the partial normal:

```
g(p; z) = ∇_x f(x; z) | x=p   (computed via torch.autograd.grad)
L_incidence(z; P_obs, n_obs) = 1 − mean_p (g(p; z) · n_obs(p))² / (||g(p; z)||² · ||n_obs(p)||²)
```

Squared cosine ignores sign-flip ambiguity. **Off by default in Stage 4** because partial normals are not exposed by the current dataset wrapper.

### 3.6 Latent prior

```
L_prior(z) = ||z||² / d_z
```

Equivalent to a centred unit-Gaussian negative log prior, scaled by dimension. Default `λ_prior = 1e-3`.

### 3.7 Total

```
L_obs(z) = w_surf · L_surf_obs + w_free · L_free_total + w_mixed · L_mixed
         + w_ray · L_ray              [Tier-2, if scanner_pose present]
         + w_incidence · L_incidence  [Tier-2, off by default]

L_total(z) = L_obs(z) + λ_prior · L_prior(z)
```

Default weights:

| Term | Weight |
|---|---|
| `w_surf` | 1.0 |
| `w_free` | 1.0 |
| `α_hard` (intra-`L_free_total`) | 2.0 |
| `w_mixed` | 0.5 |
| `w_ray` | 0.5 |
| `w_incidence` | 0.0 (off) |
| `λ_prior` | 1e-3 |

The Huber thresholds `δ_*` default to 0.5. They guard against runaway gradients on heavily-violated queries (outlier LiDAR returns, or queries that the encoder's mean has placed deep inside / outside the field).

---

## 4. Test-time refinement procedure

### 4.1 Initialisation

For each object:

1. Run the Stage-3 encoder on `partial_observed_points` with `sample=False` to obtain the **posterior mean** `μ(O)`.
2. Initialise `z ← μ(O).clone().detach().requires_grad_(True)`.
3. **Do not** sample `z`. The optimiser should start at the posterior mode, not at a random sample. (A multi-sample variant — `--start_from_K K` — initialises K candidates from `q(z|O)` and refines each; reported as a Stage-4-aux table for diagnosis.)

### 4.2 Optimiser

Adam on `[z]` only. Decoder remains frozen at all times — its gradient graph is used to compute `∂L_obs/∂z` but *no decoder parameter receives an update step*.

Defaults:

| Hyperparameter | Default |
|---|---|
| `--steps` | 50 |
| `--lr` | 1e-2 |
| Adam betas | `(0.9, 0.999)` |
| `--lr_schedule` | `cosine` (warmup 5 steps, decay to 0.1× at step `--steps`) |

Refinement is per-object (no batching across objects). 50 steps × ~0.7 ms / step ≈ 35 ms per object on a single GPU. For 206 val objects: ~ 7 s plus mesh-extraction overhead.

### 4.3 Keep-best tracking

After every step, evaluate `L_obs(z)` (no `L_prior`) on a held-out subset of the per-object queries — specifically, **a fresh random partition of the `query_points_all` not used in the loss this step**. Track the lowest such held-out `L_obs` across steps; the returned `z*` is whichever step minimised this score. This is "validation-on-the-instance" — defends against overfitting to the training subset of queries.

If no held-out partition exists (very small query budgets), fall back to the in-loss `L_obs` minimum.

### 4.4 Early stopping

Stop refinement early if any of the following triggers:

1. Held-out `L_obs` does not improve over the last 10 steps (patience).
2. `free_space_violation_rate` (computed on `Q_free`, threshold 0.5) increases by > 10 % vs the initial encoder estimate. Log a `early_stop_reason: "free_space_increase"`.
3. `||z* − μ(O)||_2 > 5 · σ_prior` where `σ_prior = 1.0`. Log `early_stop_reason: "z_too_far_from_prior"`.
4. NaN or Inf in `L_obs`.

Early stopping defends against the **noisy-observation overfit** failure mode: heavy LiDAR jitter or corrupted normal flips can push refinement to push the surface through visible noise rather than through the underlying object.

### 4.5 Mesh extraction (post-refinement)

Marching Cubes is run **after** refinement on the refined `z*`. The MC call is *not* in the gradient graph (constraint: no backprop through MC). Extraction is identical to Stage 2 / Stage 3 eval — `extract_patch_mesh(occupancy_fn, resolution=32 or 64, iso_level=0.5)`.

---

## 5. Step count, LR, compute cost

| Setting | Value | Justification |
|---|---|---|
| `--steps` | 50 default | DeepSDF inference uses 800 steps; for our setting 50 is plenty because the encoder already lands on the right manifold ridge. Larger values cause `z` to drift further from `q(z|O).μ` and trigger the prior-distance early-stop. Sweep `{0, 10, 30, 50, 100}` planned. |
| `--lr` | 1e-2 default | DeepSDF defaults are 1e-3 / 1e-4; we go higher because we have a frozen decoder (no destabilisation risk) and a well-calibrated initial `z`. Sweep `{1e-3, 5e-3, 1e-2, 5e-2}` planned. |
| Per-object compute | ~35 ms refinement + ~50 ms MC | Single forward+backward of decoder on ~1024 query points per step. |
| Full val (206 obj) | ~7 s refinement + ~10 s MC + ~3 s metric assembly | Order of seconds. |

---

## 6. Constraint compliance audit

| Constraint | Enforced by |
|---|---|
| **Do not backprop through Marching-Cubes** | MC is called only after `z*` is finalised; the call is wrapped in `torch.no_grad()` in the eval CLI. The refinement loss is over decoder field values at fixed query coordinates only. |
| **Refine latent z, not the full decoder weights** | Optimiser is constructed as `Adam([z])`. Decoder parameters retain `requires_grad_(False)` from the Stage-3 frozen-decoder pathway. A unit test in the smoke verifies decoder parameters never receive gradients. |
| **Do not use clean target points in inference-time refinement loss** | The loss API takes only: `partial_observed_points`, `free_query_points`, `free_space_query_hard_negatives`, `query_points_all` + labels + ignore mask. There is no parameter for `clean_points` or `hidden_clean_points`. The CLI extracts these only for the *eval* metric block, never for the loss. |
| **GT metrics are for evaluation only** | Output JSON splits metrics into `inference_only_metrics` (free-space violation, latent prior, refinement time) and `gt_dependent_metrics` (chamfers, mesh IoU). Real-data deployment can suppress the GT block. |

---

## 7. Failure modes and how to diagnose

### 7.1 Refinement *increases* chamfer

**Symptom**: `after_refine_recon_chamfer_l1 > before_refine_recon_chamfer_l1`.

**Likely cause**: free-queries-derived-from-clean leak (§1) is exactly tight enough that the encoder mean is already aligned with the leaky GT signal; refinement then pushes against `partial_observed_points` *only*, which are noisier than the leaky free-queries — so the surface migrates *away* from GT. Run the `--free_queries_from_observation` ablation (re-derives free queries from partial obs) to confirm.

### 7.2 Free-space violation goes *up*

**Symptom**: `after_refine_free_space_violation_rate > 1.10 · before_*`.

**Likely cause**: `w_surf` too high; refinement is pulling the surface toward observed points so aggressively that surrounding voxels become occupied. Drop `w_surf` to 0.5 or raise `w_free` to 2.0. Stage-4 early-stop (§4.4 trigger 2) will fire and the run is reported as `early_stop=free_space_increase`.

### 7.3 `z` drifts far from prior

**Symptom**: `||z* − μ(O)||_2 / d_z^{0.5} > 0.5` for many objects.

**Likely cause**: `λ_prior` too low. Raise to 1e-2.

### 7.4 No movement at all

**Symptom**: `L_obs` plateaus from step 1; `||z* − μ(O)||_2 ≈ 0`.

**Likely cause**: LR is too small *or* the encoder mean is already at a local optimum of `L_obs`. Either is consistent with the Stage-3 finding that the amortisation gap is near zero. **This is the expected outcome on the current cache.** Stage 4 may legitimately report "no measurable improvement"; this is itself useful information (kill criterion).

---

## 8. Files to be added (this stage)

| File | Role |
|---|---|
| `ss3dm_prior/losses_spcarnet_observation.py` | Pure-functional loss API: `observed_surface_field_loss`, `free_space_loss`, `ray_consistency_loss`, `normal_incidence_consistency`, `compute_observation_loss` (combined). |
| `scripts/car_model/refine_spcarnet_latent_map.py` | Per-object MAP refinement CLI with the user-specified signature. |
| `scripts/car_model/smoke_test_spcarnet_stage4.py` | 2-object × 3-step smoke. |
| `docs/car_model/spcarnet_stage4_observation_map_implementation_report.md` | Closing report. |

No existing file is modified. Stage-1 dataset, Stage-2 trainer / decoder, Stage-3 trainer / encoder / configs / launcher all remain untouched. CarNet v0.x is unaffected.

---

## 9. Smoke-test contract

`scripts/car_model/smoke_test_spcarnet_stage4.py` must:

1. Build / reuse the Stage-1 object index.
2. Construct a tiny encoder + tiny decoder (matching Stage-3 smoke sizes).
3. Pick 2 objects from the train split.
4. Run **3 refinement steps** on each:
   - Init `z` from the encoder.
   - Compute `L_obs` (Tier-1 only, `--enable_ray_loss=false`).
   - Backward; verify finite gradient on `z`.
   - Step Adam on `[z]`.
5. Verify decoder gradients remain zero throughout.
6. Verify the script tolerates `scanner_pose=None` without crash.
7. Print `[stage4-smoke] PASS`.

---

## 10. Out of scope (deferred)

- **Multi-sample candidate refinement (`--start_from_K K > 1`)** — wired but not benchmarked in Stage 4. Stage 5 builds on this for proper multi-hypothesis.
- **Cache rebuild with persisted scanner pose / ray endpoints** — would unlock real `L_ray`. Stage-4 implementation supports it via the `scanner_pose_fn` dataset hook; the rebuild itself is a separate Stage-1 follow-up.
- **Online corruption-robustness sweep** — applying explicit corruption pipelines (gaussian, dropout, beam_occlusion) at refinement time and reporting per-corruption-type chamfer recovery. Slated for Stage 7 paper-results.

---

## 11. Decision

Stage 4 is a **headroom-bounded margin stage**. Stage 3 already passes its gate, and the Stage-3 amortisation-gap diagnostic (`zero_corruption_recon_chamfer ≈ recon_chamfer`) tells us the encoder is not what is leaving margin on the table — the decoder ceiling is. We therefore expect Stage 4 to:

1. Show a small but real chamfer improvement on val (target ≥ 0.005, the gate).
2. Preserve or improve free-space violation.
3. Move `z*` modestly from `μ(O)` (a sign of useful inference) but not catastrophically (a sign of overfitting).

If (1) fails, the report documents the negative result and Stage 5 (multi-hypothesis) proceeds without MAP — that is the explicit RFC §7 fall-through.

_End of design._

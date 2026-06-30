# 2026-06-30 Thorough Investigation: Why The v169/vNext Line Stalls Below Phase-J

This report is a hard reflection on why the current SPCarNet representation
line keeps staying below the Phase-J flowers reference, despite many real code
changes, medium runs, exact runs, audits, and negative-result logs.

## Executive Answer

The current method is stuck below Phase-J because it still cannot transport
enough correct, view-dependent RGB residual energy from train-fit views to
target views.

The problem is not one missing alpha, rank, gate threshold, W&B run, or full9
promotion. The problem is representational and objective-level:

- Phase-J has a strong image-space teacher correction.
- Our current v169/vNext route tries to bake that correction into a sparse
  face/UV/bin surface residual carrier.
- The carrier has weak cross-view residual direction consistency.
- Reliability gates can detect unsafe residuals, but then they suppress the
  same residual energy needed to close the PSNR gap.
- Relaxing gates raises PSNR only slightly and hurts SSIM/LPIPS tails.
- Tightening gates makes the method safe but close to no-op.

This explains why many versions produce real but tiny gains over the parent,
while still remaining about `0.45 dB` below Phase-J on flowers PSNR.

## The Gap Is An Order-Of-Magnitude Gap

Current strict flowers numbers:

| method | PSNR | SSIM | LPIPS | role |
|---|---:|---:|---:|---|
| parent | 19.832054 | 0.619910 | 0.180335 | direct parent |
| v292d | 19.851452 | 0.620343 | 0.180212 | best balanced recent route |
| v293a | 19.853420 | 0.620328 | 0.180312 | best recent PSNR |
| Phase-J gate | 20.304358 | 0.557770 | 0.329222 | required reference |

The SSIM/LPIPS thresholds are not the blocker under this recorded gate. The
blocker is PSNR.

Converting PSNR to MSE:

| item | PSNR | MSE |
|---|---:|---:|
| parent | 19.832054 | 0.010394285 |
| v293a | 19.853420 | 0.010343274 |
| Phase-J | 20.304358 | 0.009323183 |

Required MSE reduction from parent to Phase-J:

```text
0.010394285 - 0.009323183 = 0.001071102
```

v293a MSE reduction from parent:

```text
0.010394285 - 0.010343274 = 0.000051011
```

So v293a captures only about:

```text
0.000051011 / 0.001071102 = 4.76%
```

That is about `1/21` of the required MSE reduction. This is why the project
feels stuck: many changes are real, but the effect size is too small.

## Historical Frontier Table

| method | PSNR | PSNR gain vs parent | Phase-J PSNR gap | captured MSE reduction | SSIM gain | LPIPS gain |
|---|---:|---:|---:|---:|---:|---:|
| v258a policy gain | 19.838304 | +0.006250 | -0.466054 | 0.0140 | +0.000109 | +0.000139 |
| v266c hybrid conservative | 19.845698 | +0.013644 | -0.458660 | 0.0304 | +0.000291 | +0.000420 |
| v282b fixed alpha 0.50 | 19.850666 | +0.018612 | -0.453692 | 0.0415 | -0.000165 | -0.000285 |
| v285b heldout calibration | 19.842752 | +0.010698 | -0.461606 | 0.0239 | +0.000216 | +0.000317 |
| v286b recalibrated heldout | 19.840910 | +0.008856 | -0.463448 | 0.0198 | +0.000273 | +0.000235 |
| v289c target-compatible weighting | 19.841839 | +0.009785 | -0.462519 | 0.0218 | +0.000304 | +0.000255 |
| v292d PatchViewMoE + view support | 19.851452 | +0.019398 | -0.452906 | 0.0432 | +0.000433 | +0.000123 |
| v293a texture latent | 19.853420 | +0.021366 | -0.450938 | 0.0476 | +0.000418 | +0.000023 |

The frontier is improving, but the best PSNR route still captures less than
`5%` of the Phase-J MSE reduction. That is not a near-miss.

## Evidence 1: Teacher Signal Exists

The failure is not that Phase-J has no useful correction. Earlier v249-v252
diagnostics showed strong teacher headroom on policy-val:

| source | PSNR gain | SSIM gain | LPIPS gain |
|---|---:|---:|---:|
| Phase-J teacher headroom | about +0.913279 | about +0.065512 | about +0.017600 |

Therefore the missing element is not teacher availability. The missing element
is correct transport into a target-view-safe representation.

## Evidence 2: Current Carrier Has A Tiny Projection Upper Bound

The v294 projection upper-bound diagnostic gave the current face/UV/low-rank
carrier direct access to train-fit Phase-J residuals and certified only on
train-policy-val.

Best candidate:

| field | value |
|---|---:|
| texture size | 8 |
| rank | 4 |
| alpha | 0.031250 |
| full-image PSNR gain | +0.000163820 |
| SSIM gain | +0.000000392 |
| LPIPS gain | +0.000000956 |
| SSIM positive-view fraction | 0.500000 |
| LPIPS positive-view fraction | 0.666667 |
| robust all-axis pass | false |

This is the strongest root-cause evidence. Even under a favorable projection
setup, the useful full-image gain is near noise level. Raising alpha gives tiny
extra PSNR but quickly damages SSIM/LPIPS tails.

Conclusion: continuing rank/alpha scans on this carrier is not a credible path
to Phase-J.

## Evidence 3: Cross-View Residual Direction Is Weak

The v285/v286 source-heldout diagnostics measured whether source residuals have
stable direction across source-view splits:

| statistic | value |
|---|---:|
| heldout residual direction cosine | 0.214671 |
| heldout error ratio | 2.078181 |
| v285b target PSNR gain | +0.010698 |
| v286b target PSNR gain | +0.008856 |
| Phase-J PSNR gap after v286b | -0.463448 |

A cosine around `0.21` is too low for confident residual transport. The error
ratio above `2.0` says the residual predictor fitted from one source split is
substantially wrong on heldout source views. Calibration can reduce harm; it
does not create missing residual direction.

## Evidence 4: Policy-Val Does Not Transfer Enough To Target Exact

Many versions were positive on policy-val but shrank on target exact:

| run | policy-val PSNR gain | target PSNR gain | target issue |
|---|---:|---:|---|
| v289c | about +0.048 | +0.009785 | source weighting effect too small |
| v292d | +0.024338 | +0.019398 | safe but weak |
| v293a | +0.028777 | +0.021366 | LPIPS/tails worsen |

This means policy-val is useful, but it is not enough to certify high-frequency
cross-view residual direction. Current policy-val mostly helps choose how much
to shrink. It does not teach the representation how to move Phase-J energy into
new views.

## Evidence 5: v296 Reduced Comparison Was Negative

After the root-cause diagnosis, v296 added a real representation scaffold:

```text
--surface_texture_mode lowrank_view_holdout_v3
```

It exposes source-heldout residual direction features to the neural decoder:

- `holdout_cosine`
- `holdout_error_confidence`
- `holdout_support_balance`
- `holdout_confidence`

Reduced same-budget comparison:

| run | output | W&B offline |
|---|---|---|
| v2 lowrank_view_v2 | `/tmp/peilincai_spcarnet_v296_reduced_v2_20260630` | `/tmp/peilincai_spcarnet_v296_reduced_v2_20260630/wandb/offline-run-20260630_162127-m8dqjv69` |
| v3 lowrank_view_holdout_v3 | `/tmp/peilincai_spcarnet_v296_reduced_v3_20260630` | `/tmp/peilincai_spcarnet_v296_reduced_v3_20260630/wandb/offline-run-20260630_162127-we9r6yry` |

Shared reduced command shape:

```text
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/train_perceptual_surface_residual_decoder.py \
  --fit_evidence_dir /dev/shm/peilincai_spcarnet_20260629_v168_direct_teacher_lowcopy_exact/flowers/teacher_surface_evidence \
  --target_eval_mode never \
  --surface_texture_uv_bins 4 \
  --surface_texture_max_samples_per_view 50000 \
  --decoder_output_mode patch_view_moe \
  --max_candidate_faces 64 \
  --max_candidate_face_samples_per_view 1024 \
  --steps 80 \
  --batch_size 4096 \
  --policy_val_stride 8 \
  --alpha_grid 0,0.03125,0.0625,0.125 \
  --eval_chunk_size 32768 \
  --compute_lpips \
  --policy_val_ssim_max_side 384 \
  --policy_val_lpips_max_side 192 \
  --enable_wandb
```

v296 reduced result:

| mode | feature dim | covered bins | mean holdout cosine | mean holdout confidence | selected alpha | policy-val all-axis | target exact |
|---|---:|---:|---:|---:|---:|---|---|
| lowrank_view_v2 | 43 | 0.507812 | n/a | n/a | 0.000000 | false | not run |
| lowrank_view_holdout_v3 | 47 | 0.507812 | 0.356491 | 0.286819 | 0.000000 | false | not run |

Nonzero-alpha rows were also too weak:

| mode | alpha | PSNR gain | SSIM gain | LPIPS gain | changed fraction |
|---|---:|---:|---:|---:|---:|
| v2 | 0.03125 | +0.000002765 | -0.000000020 | -0.000001945 | 0.000005139 |
| v2 | 0.06250 | +0.000005307 | -0.000000070 | -0.000001685 | 0.000014429 |
| v2 | 0.12500 | +0.000010541 | -0.000000089 | -0.000004155 | 0.000033108 |
| v3 | 0.03125 | +0.000002483 | +0.000000000 | -0.000000847 | 0.000004645 |
| v3 | 0.06250 | +0.000004747 | -0.000000050 | -0.000001592 | 0.000015615 |
| v3 | 0.12500 | +0.000009116 | -0.000000089 | -0.000004034 | 0.000029056 |

Interpretation:

v296 confirms that simply adding heldout-direction statistics as input features
is not enough. The decoder needs an explicit training objective that forces
source-heldout residual prediction and penalizes wrong residual direction. A
diagnostic feature without a transport loss becomes another weak confidence
signal, and policy-val correctly shrinks it to no-op.

## What Actually Went Wrong In The Research Loop

The loop made real engineering progress:

- target-GT leakage controls;
- no-target-GT apply and exact evaluation separation;
- W&B offline logging for medium/exact runs;
- parent, clean MeshSplatting, v106, Phase-J, vNext records;
- policy-val and tail-risk audits;
- projection upper-bound diagnostics;
- source-heldout diagnostics;
- documented negative results and reproducible commands.

But it over-invested in making weak residual mechanisms safe. Safety made the
failures trustworthy; it did not make the method strong.

The failed assumption was:

```text
If we make the surface residual carrier safer, more adaptive, and more
feature-rich, it will eventually carry Phase-J-like corrections.
```

The evidence now says:

```text
The carrier/objective itself is not learning a stable cross-view residual
transport function. More gates and scalar confidence features mostly decide
when not to apply it.
```

## Why This Is Not Just A Prompt Or Execution Problem

The prompts correctly forced stronger fairness:

- no target GT during apply;
- exact flowers before full9;
- W&B logging;
- baseline/current/improved comparisons;
- documented commands and failures;
- refusal to promote weak runs.

Those requirements exposed the truth. The weakness is primarily in the current
method family, not in the existence of the gate.

However, the prompt-driven loop did have a research failure mode: it encouraged
many incremental mechanism variants around the same carrier. Once v294 showed
the carrier upper bound was near zero, continuing local gate/alpha/feature scans
became low-value.

## What Should Stop

These should stop being the main route:

- alpha/rank/floor scans on the same carrier;
- more scalar reliability thresholds as the central novelty;
- target-compatible source weighting as the main method;
- adding diagnostic features without a heldout transport loss;
- footprint expansion without a stronger residual objective;
- full9 promotion before flowers exact closes the Phase-J PSNR gap.

## What The Next Real Method Must Do

The next credible paper-level attempt must directly train residual transport.

Minimum viable direction:

1. Split train-fit source views into source-A and heldout-source-B.
2. Build residual evidence from source-A only.
3. Predict residuals on heldout-source-B pixels/views.
4. Train with explicit RGB residual loss, direction cosine loss, and magnitude
   calibration loss on heldout-source-B.
5. Predict confidence as an auxiliary output, but do not use confidence as the
   main mechanism.
6. Use policy-val only as certification and alpha selection after the transport
   model has learned a nontrivial residual field.
7. Apply to stripped target no-GT evidence only after policy-val passes.

Success criteria before full9:

- nonzero policy-val alpha selected without SSIM/LPIPS tail damage;
- changed fraction large enough to be visually meaningful;
- target exact flowers improves PSNR by at least `+0.10 dB` over v293a before
  spending full9 resources;
- no-target-GT audit remains true;
- Phase-J flowers PSNR gap decreases materially, not by `0.002 dB`.

## Final Verdict

Current status:

```text
Final status: NOT COMPLETE.
```

The project has a strong evaluation and audit shell, and v292/v293 are genuine
parent-improving representation changes. But the current method family has not
solved the core Phase-J bottleneck.

The core blocker is:

```text
weak cross-view residual direction learning plus insufficient residual-energy
transport in the current face/UV/bin surface carrier.
```

The next work should be treated as a new objective-level method, not another
parameter sweep.

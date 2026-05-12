# SP-CarNet RAG/Symmetry Reranker Audit

Date: 2026-05-12

## Purpose

Stage 5 showed useful K-best posterior headroom, but the deployable K=8
reranker was worse than K=1.  This audit tests the next inference-safe reranker
family: retrieval distance to the train latent bank (RAG) and mesh
self-symmetry consistency.

No clean target or GT metric is used for scoring except the explicit `oracle`
diagnostic row.

## Code Change

Updated:

- `scripts/car_model/rescore_spcarnet_multihypothesis.py`

New variants:

- `rag_only`: pick the candidate with lowest `rag_dist_mean`;
- `sym_only`: pick the candidate with lowest `sym_residual_norm`;
- `rag_sym`: equal-rank fusion of RAG and symmetry;
- `obs_rag_sym`: equal-rank fusion of observation loss, RAG, and symmetry.

The variants consume only candidate fields emitted by
`eval_spcarnet_multihypothesis.py --enable_symmetry_score --enable_rag_score`.

## Bounded K=8 Result

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/eval_spcarnet_multihypothesis.py \
  --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --split val \
  --num_objects 50 \
  --K 8 \
  --mc_resolution 32 \
  --enable_symmetry_score \
  --enable_rag_score \
  --latent_bank_checkpoint outputs/carnet/spcarnet/autodecoder_v2/checkpoint_last.pt \
  --output_dir outputs/carnet/spcarnet/multihypothesis/val_50_K8_rag_sym_20260512
```

Rescore command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/rescore_spcarnet_multihypothesis.py \
  --input outputs/carnet/spcarnet/multihypothesis/val_50_K8_rag_sym_20260512/K8.json \
  --output outputs/carnet/spcarnet/multihypothesis/val_50_K8_rag_sym_20260512/K8_rag_sym_rescored.json \
  --variants default no_prior norm_penalty rag_only sym_only rag_sym obs_rag_sym oracle
```

| variant | top1 recon Chamfer | hidden Chamfer | free violation | visible preservation |
|---|---:|---:|---:|---:|
| default | 0.07347 | 0.10925 | 0.03641 | 0.06499 |
| no_prior | 0.07379 | 0.10927 | 0.03797 | 0.06462 |
| norm_penalty | 0.07388 | 0.10916 | 0.03797 | 0.06462 |
| rag_only | 0.07322 | 0.10940 | 0.03754 | 0.06432 |
| sym_only | 0.07213 | 0.10879 | 0.03773 | 0.06397 |
| rag_sym | 0.07231 | 0.10871 | 0.03707 | 0.06468 |
| obs_rag_sym | 0.07223 | 0.10787 | 0.03652 | 0.06391 |
| oracle | 0.06548 | 0.10156 | 0.03270 | 0.05792 |

Interpretation:

- RAG/symmetry ranking is better than the original K=8 default reranker.
- `obs_rag_sym` is the best deployable row in this 50-object probe.
- It still does not beat the previous K=1 baseline near `0.0715` Chamfer, so it
  is not a closed headline method.
- Oracle remains far better, which confirms posterior headroom exists but is
  still not recovered by inference-safe scoring.

## Full-Val Run

Command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/eval_spcarnet_multihypothesis.py \
  --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --split val \
  --num_objects 206 \
  --K 8 \
  --mc_resolution 32 \
  --enable_symmetry_score \
  --enable_rag_score \
  --latent_bank_checkpoint outputs/carnet/spcarnet/autodecoder_v2/checkpoint_last.pt \
  --output_dir outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_20260512
```

Rescore command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/rescore_spcarnet_multihypothesis.py \
  --input outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_20260512/K8.json \
  --output outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_20260512/K8_rag_sym_rescored.json \
  --variants default no_prior norm_penalty rag_only sym_only rag_sym obs_rag_sym oracle
```

Full-val summary from `206` validation objects:

| variant | top1 recon Chamfer | hidden Chamfer | free violation | visible preservation |
|---|---:|---:|---:|---:|
| default | 0.06802 | 0.10058 | 0.03581 | 0.06317 |
| no_prior | 0.06738 | 0.09989 | 0.03503 | 0.06263 |
| norm_penalty | 0.06761 | 0.10019 | 0.03541 | 0.06286 |
| rag_only | 0.06785 | 0.10049 | 0.03613 | 0.06271 |
| sym_only | 0.06673 | 0.09948 | 0.03563 | 0.06259 |
| rag_sym | 0.06744 | 0.10040 | 0.03610 | 0.06304 |
| obs_rag_sym | 0.06745 | 0.10016 | 0.03535 | 0.06330 |
| oracle | 0.06088 | 0.09317 | 0.03124 | 0.05625 |

Artifacts:

- raw K-best result:
  `outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_20260512/K8.json`
- rescored result:
  `outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_20260512/K8_rag_sym_rescored.json`
- W&B post-hoc summary/artifact:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/ygw72x7r`

Interpretation:

- The full-val conclusion is stronger than the 50-object probe: `sym_only`
  improves the deployable default from `0.06802` to `0.06673` recon Chamfer and
  from `0.10058` to `0.09948` hidden Chamfer.
- `no_prior` is also consistently better than the original default, so the
  original Gaussian prior term is too conservative for top-1 deployment.
- RAG alone helps less than symmetry on full val; equal fusion does not beat
  the cleaner symmetry-only rule.
- Oracle remains much better at `0.06088`, so the posterior has real sampled
  headroom that the current inference-safe selector still does not fully
  recover.

## Full-Val K=1 Baseline

The previous K=1 reference near `0.0715` was a 50-object bounded probe.  For a
fair full-val comparison, the K=1 baseline was rerun on all `206` validation
objects with the same extraction protocol and W&B logging.

Command:

```bash
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/eval_spcarnet_multihypothesis.py \
  --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --split val \
  --num_objects 206 \
  --K 1 \
  --mc_resolution 32 \
  --enable_symmetry_score \
  --enable_rag_score \
  --latent_bank_checkpoint outputs/carnet/spcarnet/autodecoder_v2/checkpoint_last.pt \
  --output_dir outputs/carnet/spcarnet/multihypothesis/val_full_K1_rag_sym_20260512 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group spcarnet_fullval_k1_k8_rag_sym_20260512 \
  --wandb_name spcarnet_val_full_K1_rag_sym_20260512 \
  --wandb_tags spcarnet fullval K1 baseline rag_sym 20260512
```

Result:

| protocol | recon Chamfer | hidden Chamfer | free violation | visible preservation |
|---|---:|---:|---:|---:|
| K=1 baseline | 0.06637 | 0.09884 | 0.03515 | 0.06131 |
| K=8 default | 0.06802 | 0.10058 | 0.03581 | 0.06317 |
| K=8 `sym_only` | 0.06673 | 0.09948 | 0.03563 | 0.06259 |
| K=8 oracle | 0.06088 | 0.09317 | 0.03124 | 0.05625 |

Artifacts:

- K=1 result:
  `outputs/carnet/spcarnet/multihypothesis/val_full_K1_rag_sym_20260512/K1.json`
- W&B K=1 run:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/kuty9usn`

Full-val conclusion:

- `sym_only` repairs a large part of the K8 reranker failure, but does not beat
  the fair K=1 baseline: recon Chamfer is worse by `+0.00036`, hidden Chamfer
  by `+0.00064`, free-space violation by `+0.00048`, and visible preservation
  by `+0.00128`.
- Therefore the deployable paper headline should still use K=1 unless a stronger
  selector closes the gap to the oracle row.

## Nested-Seed Fairness Fix

The first full-val K=1/K=8 comparison used the original Stage-5 seed schedule:

`seed_base = seed * 1024 + object_index * K`

That schedule makes K=1 and K=8 non-nested: the K=8 candidate `k=0` is not the
same sample as the corresponding K=1 run.  The eval script now defaults to a
K-invariant per-object stride:

`seed_base = seed * seed_object_stride * n_total + object_index * max(seed_object_stride, K + 1)`

so K=8 contains the same first candidate as K=1.  The old behavior remains
available via `--legacy_k_dependent_seed`.

Nested full-val K=1 command:

```bash
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/eval_spcarnet_multihypothesis.py \
  --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --split val \
  --num_objects 206 \
  --K 1 \
  --mc_resolution 32 \
  --enable_symmetry_score \
  --enable_rag_score \
  --latent_bank_checkpoint outputs/carnet/spcarnet/autodecoder_v2/checkpoint_last.pt \
  --output_dir outputs/carnet/spcarnet/multihypothesis/val_full_K1_rag_sym_nestedseed_20260512 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group spcarnet_fullval_nested_seed_k1_k8_20260512 \
  --wandb_name spcarnet_val_full_K1_rag_sym_nestedseed_20260512 \
  --wandb_tags spcarnet fullval K1 baseline rag_sym nested_seed 20260512
```

Nested full-val K=8 command:

```bash
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=online PYTHONUNBUFFERED=1 \
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/eval_spcarnet_multihypothesis.py \
  --posterior_checkpoint outputs/carnet/spcarnet/posterior_encoder_v1/checkpoint_last.pt \
  --object_index outputs/carnet/spcarnet/object_index_v1.json \
  --split val \
  --num_objects 206 \
  --K 8 \
  --mc_resolution 32 \
  --enable_symmetry_score \
  --enable_rag_score \
  --latent_bank_checkpoint outputs/carnet/spcarnet/autodecoder_v2/checkpoint_last.pt \
  --output_dir outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512 \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group spcarnet_fullval_nested_seed_k1_k8_20260512 \
  --wandb_name spcarnet_val_full_K8_rag_sym_nestedseed_20260512 \
  --wandb_tags spcarnet fullval K8 rag_sym nested_seed 20260512
```

Nested K=8 rescore command:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/rescore_spcarnet_multihypothesis.py \
  --input outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8.json \
  --output outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_rag_sym_guarded_rescored.json \
  --variants first default no_prior norm_penalty rag_only sym_only rag_sym obs_rag_sym sym_if_loss_le_first sym_if_score_ge_first sym_if_loss_and_score_first oracle
```

Nested full-val result:

| variant | recon Chamfer | hidden Chamfer | free violation | visible preservation |
|---|---:|---:|---:|---:|
| K=1 nested baseline | 0.06782 | 0.10011 | 0.03643 | 0.06243 |
| K=8 `first` | 0.06786 | 0.10013 | 0.03643 | 0.06246 |
| K=8 default | 0.06816 | 0.10061 | 0.03629 | 0.06326 |
| K=8 `no_prior` | 0.06763 | 0.10014 | 0.03426 | 0.06287 |
| K=8 `rag_only` | 0.06744 | 0.09997 | 0.03620 | 0.06260 |
| K=8 `sym_only` | 0.06726 | 0.10010 | 0.03590 | 0.06310 |
| K=8 `rag_sym` | 0.06700 | 0.09971 | 0.03546 | 0.06294 |
| K=8 `obs_rag_sym` | 0.06746 | 0.10009 | 0.03539 | 0.06319 |
| K=8 oracle | 0.06132 | 0.09357 | 0.03114 | 0.05670 |

Visible-preserving follow-up:

After the nested full-val run, I added deployable visible-surface selector
variants to the rescore script.  These variants use
`visible_preservation_error`, which is computed from the observed partial input
points and candidate mesh, not from clean/test target geometry.

Command:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
scripts/car_model/rescore_spcarnet_multihypothesis.py \
  --input outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8.json \
  --output outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_visible_rescored.json \
  --variants first rag_sym visible_only visible_rag_sym oracle \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group spcarnet_fullval_nested_visible_selector_20260512 \
  --wandb_name spcarnet_val_full_K8_visible_selector_rescore_20260512 \
  --wandb_tags spcarnet fullval K8 visible_selector 20260512
```

Result:

| variant | recon Chamfer | hidden Chamfer | free violation | visible preservation | z norm |
|---|---:|---:|---:|---:|---:|
| K=8 `first` | 0.06786 | 0.10013 | 0.03643 | 0.06246 | 3.8385 |
| K=8 `rag_sym` | 0.06700 | 0.09971 | 0.03546 | 0.06294 | 3.7067 |
| K=8 `visible_only` | 0.06259 | 0.09425 | 0.03217 | 0.05592 | 3.8744 |
| K=8 `visible_rag_sym` | 0.06426 | 0.09630 | 0.03353 | 0.05950 | 3.7379 |
| K=8 oracle | 0.06132 | 0.09357 | 0.03114 | 0.05670 | 3.8634 |

Artifacts:

- `outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_visible_rescored.json`
- W&B run: `dxgqvwn0`

This changes the selector conclusion.  `rag_sym` remains the geometry-prior
selector, but `visible_only` is the stronger deployable selector on this
full-val evidence package: it improves recon, hidden, free-space, and visible
preservation versus the contained K=1/first candidate.  The oracle gap remains
nonzero, so this is a meaningful selector upgrade rather than a finished shape
completion story.

Artifacts:

- nested K=1:
  `outputs/carnet/spcarnet/multihypothesis/val_full_K1_rag_sym_nestedseed_20260512/K1.json`
- nested K=8:
  `outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8.json`
- nested K=8 rescore:
  `outputs/carnet/spcarnet/multihypothesis/val_full_K8_rag_sym_nestedseed_20260512/K8_rag_sym_guarded_rescored.json`
- W&B K=1:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/zp8yhp21`
- W&B K=8:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/dmsj04t6`
- W&B nested K=8 rescore/artifact:
  `https://wandb.ai/karamazovaniki-university-of-southern-california/mesh-splatting-ecsr/runs/b3n3t53f`

Nested conclusion:

- `visible_only` is the best deployable nested selector in the current evidence
  package.  It beats the contained K=1/first candidate on recon, hidden,
  free-space, and visible-preservation metrics.
- `rag_sym` remains a useful geometry-prior ablation: it improves recon, hidden,
  and free-space metrics versus the contained first candidate, but it worsens
  visible preservation.
- Oracle is still much better, which keeps selector design as an open research
  gap rather than a closed headline breakthrough.

## Review

This is an honest improvement to the failed Stage-5 reranker, but not a
paper-level closure by itself.  The strongest current version is the
nested-seed `visible_only` selector because it is inference-safe and improves all
four reported validation metrics versus the contained first/K=1 candidate.  The
weakness is also clear: oracle remains better on recon, hidden geometry, and
free-space metrics.  A paper-facing SPCarNet selector still needs a stronger
observation-grounded certificate and a deployment argument explaining when the
visible-surface score can be trusted.

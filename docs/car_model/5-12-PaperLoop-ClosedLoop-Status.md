# Paper-Loop Closed-Loop Status

Date: 2026-05-12

Gaincert v1 update: the latest Phase-S policy now adds a train-only per-face gain certificate on top of face/view consensus. This is a real method change and it fixes one concrete outdoor strict-gate failure: `flowers` now passes the fixed four-offset train-val gate. The loop is still not complete because `bicycle`, `counter`, and `treehill` reject under frozen policies, `bonsai` is only threshold-accepted with tiny SSIM/LPIPS train-val regressions, and the visual/metric gains are still low amplitude.

Full9 status collector update: `scripts/car_model/meshsplatopt_collect_full9_paper_loop_status.py` now mechanically joins the existing clean MeshSplatting clean-best rows, Phase-J full9 rows, and Phase-S decision/gate rows. It writes `docs/car_model/5-12-Full9-PaperLoop-Evidence-Status.md` plus JSON/CSV artifacts under `outputs/carnet/meshsplatopt/full9_paper_loop_status/`, and logs W&B run `6g09l2ul`. The collector confirms clean-best `9 / 9`, Phase-J `9 / 9`, and Phase-J strict RGB wins vs clean-best `9 / 9`, but marks full9 clean/Phase-J/Phase-S closure as `False`: Phase-S strict four-offset gates exist for `7 / 9`, accept `6 / 9`, reject `1 / 9`, and have only `3 / 7` all-axis train-val wins.

## What Changed

This update contains real method changes rather than only parameter scanning.

### 1. Phase-S Face-Local Consensus Surface Code

The Phase-R bottleneck was not residual strength.  Shared-vertex SH1 edits were
too conservative because one vertex is shared by several faces, so a repair
learned for one face can leak into neighboring surfaces.  Phase-S changes the
operator:

1. collect train-only residual evidence per face;
2. split policy-validation train views by camera/view;
3. accept a face only when enough train views agree on the residual direction;
4. require enough train-only policy-validation views to certify a positive local residual-MSE gain;
5. duplicate the accepted face's three vertices and write a bounded local SH1
   appearance code onto those local vertices.

In plain terms: the method first asks "does this triangle consistently need the
same correction from multiple training views, and does it actually reduce local residual error on train-held views?", then gives that triangle its own small appearance patch so the repair does not spill into adjacent triangles.

This is implemented in:

- `scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py`
- `scripts/car_model/ecsr_apply_surface_residual_barycentric_sh1_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`

### 2. SP-CarNet Nested-Seed RAG/Symmetry Selector

The previous K-best evaluation had a fairness flaw: K=1 and K=8 used different
per-object seeds, so K=8 did not necessarily contain the K=1 candidate.  The
eval script now uses a K-invariant seed schedule by default, and keeps the old
behavior behind `--legacy_k_dependent_seed`.

On top of that, the reranker now includes inference-safe variants that use:

- latent-bank retrieval distance (`rag_only`);
- mesh self-symmetry residual (`sym_only`);
- rank fusion between retrieval and symmetry (`rag_sym`);
- observed partial-to-mesh visible preservation (`visible_only`);
- guarded variants that fall back to the first candidate unless observation
  evidence is no worse.

This is implemented in:

- `scripts/car_model/eval_spcarnet_multihypothesis.py`
- `scripts/car_model/rescore_spcarnet_multihypothesis.py`

## Current Evidence

### Full9 Paper-Loop Collector

Generated report:

- `docs/car_model/5-12-Full9-PaperLoop-Evidence-Status.md`
- `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_paper_loop_status.json`
- `outputs/carnet/meshsplatopt/full9_paper_loop_status/full9_paper_loop_status.csv`
- W&B run: `6g09l2ul` (`full9_paper_loop_status_20260512`)

Summary:

| evidence | status |
|---|---:|
| clean-best rows | `9 / 9` |
| Phase-J full9 rows | `9 / 9` |
| Phase-J strict RGB wins vs clean-best | `9 / 9` |
| Phase-S single-gate decisions | `9 / 9`, accepted `6 / 9` |
| Phase-S strict four-offset gates | `7 / 9`, accepted `6 / 9`, rejected `1 / 9` |
| Phase-S strict all-axis train-val wins | `3 / 7` |
| missing strict evidence | `counter`, `treehill` |
| full9 clean/Phase-J/Phase-S closure | `False` |

This collector is deliberately stricter than a success-only summary. Missing rows count as missing evidence, and report-only held-out Phase-S test deltas are not used for acceptance.

### Phase-S

`garden` was a v11 hard fallback scene.  The new face-local carrier passes the
strict four-offset train-only gate:

| scene | gate | accepted | mean dPSNR | mean dSSIM | mean dLPIPS | test use |
|---|---|---:|---:|---:|---:|---|
| garden | 4-offset | true | +0.000572681 | +0.000022039 | -0.000095792 | none |

Fresh-cache and cached-evidence runs agree, reducing the risk that the result
comes from stale evidence reuse.

The gain-certified v1 policy was then run under the same strict four-offset train-val protocol where available:

| scene | policy | gate | accepted | mean dPSNR | mean dSSIM | mean dLPIPS | report-only test |
|---|---|---|---:|---:|---:|---:|---|
| garden | gaincert v1 | 4-offset | true | +0.000519753 | +0.000017226 | -0.000081759 | +0.000063 / +0.000001 / -0.000001 |
| flowers | gaincert v1 | 4-offset | true | +0.000030041 | +0.000000477 | +0.000000164 | +0.001677 / +0.000158 / -0.000305 |
| bicycle | gaincert v1 | 4-offset | false | +0.000142574 | +0.000001207 | +0.000023305 | +0.000374 / +0.000035 / -0.000115 |
| bonsai | gaincert v1 | 4-offset | true | +0.000156403 | -0.000000522 | +0.000019606 | +0.000715 / +0.000016 / -0.000047 |
| kitchen | gaincert v1 | 4-offset | true | +0.000072479 | +0.000000224 | -0.000000775 | +0.000084 / +0.000000 / -0.000001 |
| room | gaincert v1 | 4-offset | true | +0.000050545 | +0.000000015 | -0.000000205 | +0.000046 / +0.000000 / +0.000000 |
| stump | gaincert v1 | 4-offset | true | +0.000001431 | -0.000000015 | -0.000000022 | +0.000000 / -0.000000 / +0.000000 |

The important change is `flowers`: consensus-only min2 and min3 variants failed strict four-offset gates, while gaincert v1 passes all four train-val offsets. The hard blocker is `bicycle`, which still fails under gaincert v1 and also fails under the stronger centroid patch-certified follow-up.

Single-gate expansion on the remaining Mip-NeRF360 scenes is broader but less
conclusive than strict four-offset validation:

| scene | v1 single gate | train-val dPSNR | dSSIM | dLPIPS | report-only test dPSNR | dSSIM | dLPIPS | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bonsai | accept | +0.000210 | -0.000005 | +0.000004 | +0.000715 | +0.000016 | -0.000047 | strict four-offset accepted by tolerance; not all-axis clean |
| counter | reject | -0.000172 | -0.000038 | +0.000088 | +0.000340 | +0.000008 | -0.000178 | v3 low-strength follow-up rejected |
| kitchen | accept | +0.000105 | +0.000000 | -0.000001 | +0.000084 | +0.000000 | -0.000001 | strict four-offset accepted |
| room | accept | +0.000069 | +0.000000 | -0.000000 | +0.000046 | +0.000000 | +0.000000 | strict four-offset accepted; near no-op |
| stump | accept | +0.000000 | +0.000000 | -0.000000 | +0.000000 | -0.000000 | +0.000000 | strict four-offset accepted; effect is near no-op |
| treehill | reject | -0.000338 | +0.000001 | -0.000006 | -0.000261 | +0.000001 | -0.000004 | v3 low-strength follow-up rejected |

Updated statuses after the continuation batch:

- `bonsai`, `kitchen`, `room`, and `stump` all have completed strict four-offset
  files under `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512/`.
- `counter` and `treehill` remain blocked at single-gate rejection; running a
  strict gate would promote a candidate that the frozen policy already rejected.
- The v3 low-strength face-shrink diagnostic rejects `bicycle`, `counter`, and
  `treehill`.
- The `bicycle` centroid patch-certified attempt accepts single-gate train-val
  but rejects strict four-offset: mean `-0.000041` PSNR, `-0.000016` SSIM, and
  `+0.000023` LPIPS, with offset2/offset3 PSNR failures. Its strict JSON is
  `outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_patchcert_v4_centroid_v2_cached_dense16_20260512/bicycle/multifold_trainval_gate.json`.

The qualitative held-out gallery for accepted v1 single-gate scenes is:

- `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/qualitative_gallery/gallery.html`
- manifest: `outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/qualitative_gallery/selected_views.json`

The shared-vertex SH1 consensus ablation failed the same four-offset gate:

| carrier | scene | gate | accepted | mean dPSNR | mean dSSIM | mean dLPIPS | reason |
|---|---|---|---:|---:|---:|---:|---|
| face-local SH1 | garden | 4-offset | true | +0.000572681 | +0.000022039 | -0.000095792 | pass |
| shared-vertex SH1 | garden | 4-offset | false | +0.000010967 | -0.000012904 | +0.000053991 | offset1 dPSNR < 0 |

This supports the claim that local ownership, not only consensus filtering, is
the useful method component.

### SP-CarNet

Nested-seed full-val results on 206 validation objects:

| variant | recon Chamfer | hidden Chamfer | free violation | visible preservation |
|---|---:|---:|---:|---:|
| K=1 nested baseline | 0.06782 | 0.10011 | 0.03643 | 0.06243 |
| K=8 default | 0.06816 | 0.10061 | 0.03629 | 0.06326 |
| K=8 `rag_sym` | 0.06700 | 0.09971 | 0.03546 | 0.06294 |
| K=8 `visible_only` | 0.06259 | 0.09425 | 0.03217 | 0.05592 |
| K=8 `visible_rag_sym` | 0.06426 | 0.09630 | 0.03353 | 0.05950 |
| K=8 oracle | 0.06132 | 0.09357 | 0.03114 | 0.05670 |

`rag_sym` fairly beats the contained K=1 candidate on reconstruction Chamfer,
hidden Chamfer, and free-space violation, but slightly worsens visible
preservation.  The newer `visible_only` selector is stronger on this evidence:
it improves all four reported inference-time metrics versus the contained first
candidate while staying close to oracle.  It is still not an oracle selector:
oracle remains better on recon, hidden, and free-space metrics.

## Paper Story

The coherent story is:

> Geometry-aware neural scene compression should not blindly delete structure
> and then rely on a renderer to hide the damage.  SPCarNet/Phase-S introduces
> certified local repair: the system only writes persistent representation
> changes when independent train views agree, and it isolates those changes at
> the surface element that owns the evidence.  For shape completion, the same
> principle appears as inference-time candidate certification: sampled
> hypotheses must be ranked by deployable structural priors rather than by a
> brittle likelihood score.

The research contribution is not "we tuned a threshold".  It is a move from
global or shared residual edits to certificate-carrying, ownership-aware local
representation edits, plus a fair nested multi-hypothesis protocol for the
shape-completion side.

The latest Phase-S story is stronger than the earlier consensus-only story because the certificate is not merely geometric agreement. It asks whether each local edit has train-only evidence of positive residual gain before the representation is changed. That is why this update is a method change, not a threshold-only retry.

## Weaknesses

- Phase-S effects are still low amplitude.  The numerical wins are robust but
  often too subtle for obvious visual figures.
- `garden`, `flowers`, `kitchen`, and `room` have completed accepted four-offset
  Phase-S certificates, `stump` is accepted but near no-op, and `bonsai` is
  threshold-accepted but not all-axis clean because mean SSIM/LPIPS regress
  slightly within tolerance.
- `bicycle` still rejects under gaincert v1 and under the centroid
  patch-certified follow-up. `counter` and `treehill` reject before strict
  four-offset validation.
- Face-local SH1 increases vertex count on accepted faces.  Rate-distortion
  reporting must include vertices/attributes, not only triangle count.
- SP-CarNet `visible_only` fixes the visible-preservation weakness in the
  nested K=8 package, but it needs a stronger theory section explaining why
  observed-visible scoring is inference-safe and not a hidden GT oracle.
- Full nine-scene fixed-policy closure is missing. The clean-best/Phase-J
  table accounting is now mechanically reconciled by the full9 collector, but
  the same collector proves Phase-S is not closed. The Stage ELA12 collector was
  rerun with W&B and remains positive on its existing five-scene selected-clean
  artifact set, but that report explicitly does not cover the full nine-scene
  Mip-NeRF360 benchmark.
- This is not yet a full paper closure.  It is a meaningful method upgrade with honest evidence and clear remaining blockers.

## Next Required Commands

Do not rerun the completed continuation probes unless auditing reproducibility.
The Stage ELA12 selected-clean collector has already been rerun in this
continuation and wrote W&B run `rmpikjz2`; it confirms the existing five-scene
selected-clean audit, not a full9 benchmark. The full9 status collector wrote
W&B run `6g09l2ul` and resolves the clean-best/Phase-J table-accounting issue.
The next defensible action is to design a new representation operator for
`bicycle`, `counter`, and `treehill`, then rerun this collector as the gate:

```bash
WANDB_MODE=online /home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/meshsplatopt_collect_full9_paper_loop_status.py \
  --doc-out docs/car_model/5-12-Full9-PaperLoop-Evidence-Status.md \
  --fail-on-missing \
  --wandb \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group full9_paper_loop_status_20260512
```

This gate currently fails by design because `counter/treehill` lack strict
Phase-S rows and `bicycle` rejects. It should only pass after a stronger method
creates accepted strict rows for all full9 scenes.

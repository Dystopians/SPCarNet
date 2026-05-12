# Paper-Loop Closed-Loop Status

Date: 2026-05-12

Gaincert v1 update: the latest Phase-S policy now adds a train-only per-face gain certificate on top of face/view consensus. This is a real method change and it fixes one concrete outdoor strict-gate failure: `flowers` now passes the fixed four-offset train-val gate. The closed loop is still not complete because `bicycle` remains rejected, full-scene validation is unfinished, and the visual/metric gains are still low amplitude.

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
- guarded variants that fall back to the first candidate unless observation
  evidence is no worse.

This is implemented in:

- `scripts/car_model/eval_spcarnet_multihypothesis.py`
- `scripts/car_model/rescore_spcarnet_multihypothesis.py`

## Current Evidence

### Phase-S

`garden` was a v11 hard fallback scene.  The new face-local carrier passes the
strict four-offset train-only gate:

| scene | gate | accepted | mean dPSNR | mean dSSIM | mean dLPIPS | test use |
|---|---|---:|---:|---:|---:|---|
| garden | 4-offset | true | +0.000572681 | +0.000022039 | -0.000095792 | none |

Fresh-cache and cached-evidence runs agree, reducing the risk that the result
comes from stale evidence reuse.

The gain-certified v1 policy was then run under the same strict four-offset train-val protocol:

| scene | policy | gate | accepted | mean dPSNR | mean dSSIM | mean dLPIPS | report-only test |
|---|---|---|---:|---:|---:|---:|---|
| garden | gaincert v1 | 4-offset | true | +0.000519753 | +0.000017226 | -0.000081759 | +0.000063 / +0.000001 / -0.000001 |
| flowers | gaincert v1 | 4-offset | true | +0.000030041 | +0.000000477 | +0.000000164 | +0.001677 / +0.000158 / -0.000305 |
| bicycle | gaincert v1 | 4-offset | false | +0.000142574 | +0.000001207 | +0.000023305 | +0.000374 / +0.000035 / -0.000115 |

The important change is `flowers`: consensus-only min2 and min3 variants failed strict four-offset gates, while gaincert v1 passes all four train-val offsets. The hard blocker is `bicycle`, which still fails on offset0 and offset2 PSNR.

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
| K=8 oracle | 0.06132 | 0.09357 | 0.03114 | 0.05670 |

`rag_sym` fairly beats the contained K=1 candidate on reconstruction Chamfer,
hidden Chamfer, and free-space violation.  It still slightly worsens visible
preservation and remains far from oracle.

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
- `garden` and `flowers` now have completed accepted four-offset Phase-S certificates, but `bicycle` still rejects under the same fixed policy.
- Face-local SH1 increases vertex count on accepted faces.  Rate-distortion
  reporting must include vertices/attributes, not only triangle count.
- SP-CarNet `rag_sym` improves the nested K=8 selector, but visible
  preservation worsens slightly and oracle remains much stronger.
- Full nine-scene fixed-policy closure is missing.
- This is not yet a full paper closure.  It is a meaningful method upgrade with honest evidence and clear remaining blockers.

## Next Required Commands

If interrupted, do not resume the old running-probe commands. They have finished. First parse and summarize the gaincert v1 gate JSONs:

```bash
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
$PY - <<'PY'
import json
from pathlib import Path
root=Path('outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512')
for scene in ['garden','flowers','bicycle']:
    p=root/scene/'multifold_trainval_gate.json'
    d=json.loads(p.read_text())
    print(scene)
    print(json.dumps({
        'accepted': d.get('accepted'),
        'selection_uses_test': d.get('selection_uses_test'),
        'aggregate': d.get('aggregate'),
        'reasons': d.get('reasons') or d.get('decision_reasons'),
    }, indent=2))
PY
```

Then run remaining scenes with the frozen gaincert v1 policy, or explicitly mark a scene blocked if the required Phase-J policy root/checkpoint is missing:

```bash
PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python
scene=bonsai
gpu=0

$PY scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --policy_root outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix \
  --dataset_root /data/peilincai/mesh_datasets/mipnerf360 \
  --scenes "$scene" \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16 \
  --iteration 26000 \
  --gpu "$gpu" \
  --delta_operator facelocal_sh1 \
  --delta_uniform_barycentric \
  --evidence_max_views 16 \
  --evidence_view_stride 3 \
  --evidence_high_error_quantile 0.70 \
  --delta_top_k 16384 \
  --delta_min_view_hits 2 \
  --delta_min_consistency 0.65 \
  --delta_min_pixel_count 32 \
  --delta_strength 0.08 \
  --delta_max_abs_rgb 0.12 \
  --delta_max_faces_to_apply 4096 \
  --delta_min_policy_val_relative_gain 0.02 \
  --delta_min_policy_val_samples 512 \
  --delta_min_policy_val_unique_faces 16 \
  --delta_min_face_policy_val_relative_gain 0.0 \
  --delta_min_face_policy_val_samples 8 \
  --delta_min_face_view_consensus 0.67 \
  --delta_min_face_consensus_views 2 \
  --delta_min_face_consensus_view_samples 4 \
  --delta_face_consensus_min_cosine 0.0 \
  --delta_min_face_gain_certificate_views 2 \
  --delta_min_face_gain_certificate_relative_gain 0.0 \
  --delta_min_face_gain_certificate_view_samples 4 \
  --delta_min_face_gain_certificate_fraction 0.67 \
  --candidate_label facelocal_gaincert_v1_cached_dense16_20260512 \
  --candidate_base_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_base \
  --candidate_test_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela \
  --candidate_trainval_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_trainval_gate \
  --wandb_project mesh-splatting-ecsr \
  --wandb_group phase_s_facelocal_gaincert_v1_cached_dense16_20260512 \
  --wandb_name "phase_s_gaincert_v1_${scene}"

$PY scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py \
    --scene "$scene" \
    --phasej_model "outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/${scene}/ratio_0200/compact_model" \
    --candidate_model "outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/${scene}/model" \
    --candidate_audit_json "outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512/${scene}/model/surface_residual_facelocal_sh1_delta_audit.json" \
    --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512 \
    --candidate_label facelocal_gaincert_v1_cached_dense16_20260512 \
    --candidate_base_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_base \
    --candidate_test_method ours_26000_facelocal_gaincert_v1_cached_dense16_20260512_phasej_ela \
    --phasej_trainval_method_prefix ours_26000_phasej_gaincert_v1_20260512 \
    --candidate_trainval_method_prefix ours_26000_facelocal_gaincert_v1_20260512_multifold \
    --offsets 0,1,2,3 \
    --iteration 26000 \
    --gpu "$gpu" \
    --policy_holdout_fraction 0.25 \
    --calib_sampler uniform \
    --calib_max_views 32 \
    --calib_stride 1 \
    --alpha_feature_mode confidence_magnitude_edge \
    --alpha_default 0.0 \
    --gate_min_psnr_gain 0.0 \
    --gate_max_ssim_regression 0.00005 \
    --gate_max_lpips_regression 0.00015 \
    --wandb_project mesh-splatting-ecsr \
    --wandb_group phase_s_facelocal_gaincert_v1_cached_dense16_20260512_multifold \
    --wandb_name "${scene}_facelocal_gaincert_v1_cached_dense16_20260512_multifold"
```

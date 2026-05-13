# Phase-S Patch-Cluster Face-Filter Audit

Date: 2026-05-13

This audit records the first fixed-policy surface-patch residual carrier after
the vertex-delta line saturated.  It uses Phase-B train-only view-support
clusters as anchors, materializes topology-preserving DC feature deltas on the
existing mesh vertices, and evaluates against the fixed Phase-J guarded adaptive
edge endpoint.  Held-out test metrics remain report-only.

## Method Change

Implemented in:

- `scripts/car_model/ecsr_apply_surface_residual_barycentric_delta.py`
- `scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py`
- `scripts/car_model/ecsr_decide_phasek_trainval_gate.py`

The new carrier is not a hand-picked per-scene parameter table:

1. Phase-B builds candidate local support clusters from train evidence only.
2. The DC barycentric residual solver uses those clusters as anchors and can
   expand deterministically with top residual supports.
3. The new `--policy_val_filter_faces` pass evaluates each face on train
   holdout samples, keeps only faces with positive local policy-val gain, and
   refits the residual carrier on the kept subset.
4. The Phase-K decision gate now defaults `min_balanced_delta` to `0.0`, so a
   candidate cannot pass only because PSNR is slightly positive while LPIPS or
   SSIM make the balanced objective worse.

## Evidence Paths

| item | path |
|---|---|
| Phase-B graph report | `docs/car_model/5-13-PhaseS-PhaseJ-Bary-ViewSupportGraph.md` |
| v2 broad cluster run, bicycle | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v2_expand256_20260513_bicycle/phasek_barycentric_gate_summary.md` |
| v2 broad cluster run, flowers | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v2_expand256_20260513_flowers/phasek_barycentric_gate_summary.md` |
| v3 face-filter run, bicycle | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v3_facefilter_20260513_bicycle/phasek_barycentric_gate_summary.md` |
| v3 face-filter run, flowers | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v3_facefilter_20260513_flowers/phasek_barycentric_gate_summary.md` |
| v4 high-confidence run, bicycle | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v4_highconf_20260513_bicycle/phasek_barycentric_gate_summary.md` |
| v4 high-confidence run, flowers | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v4_highconf_20260513_flowers/phasek_barycentric_gate_summary.md` |
| qualitative HTML audit | `outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v4_highconf_qualitative_20260513/gallery.html` |

## Quantitative Summary

All rows compare against the fixed Phase-J guarded adaptive edge method.
Train-val decides acceptance; test is report-only.

| variant | scene | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | train-val balanced | test dPSNR | test dSSIM | test dLPIPS | test balanced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 broad cluster | bicycle | false | +0.000422 | +0.000029 | +0.000246 | -0.003917 | -0.000870 | -0.000120 | +0.000283 | -0.008934 |
| v2 broad cluster | flowers | old-gate true | +0.000517 | -0.000003 | +0.000056 | -0.000671 | -0.003870 | -0.000363 | +0.000385 | -0.018841 |
| v3 face filter | bicycle | false | +0.000198 | +0.000004 | +0.000033 | -0.000382 | +0.000551 | +0.000035 | -0.000058 | +0.002420 |
| v3 face filter | flowers | false | +0.000027 | -0.000000 | +0.000001 | -0.000005 | -0.003756 | -0.000312 | +0.000281 | -0.015631 |
| v4 high confidence | bicycle | false | +0.000101 | +0.000003 | +0.000033 | -0.000499 | +0.000597 | +0.000038 | -0.000038 | +0.002130 |
| v4 high confidence | flowers | false | +0.000011 | -0.000000 | +0.000002 | -0.000030 | -0.003752 | -0.000312 | +0.000282 | -0.015628 |

## Operator-Level Audit

| variant | scene | input policy-val faces | kept faces | modified vertices | policy-val proxy gain | topology changed |
|---|---|---:|---:|---:|---:|---|
| v3 face filter | bicycle | 114 | 67 | 183 | 0.847396 | no |
| v3 face filter | flowers | 48 | 29 | 83 | 0.608252 | no |
| v4 high confidence | bicycle | 114 | 24 | 68 | 0.992633 | no |
| v4 high confidence | flowers | 48 | 8 | 24 | 0.960800 | no |

## Interpretation

The fixed-policy patch carrier is a real method change and it fixed one major
failure mode: broad cluster writes were harmful, while face-level policy-val
filtering turned bicycle held-out test deltas positive in all three RGB
metrics.  However, this is not a paper-level closed loop yet:

- bicycle improves on held-out test, but the train-val gate still rejects it
  because LPIPS increases slightly on policy-val views;
- flowers remains negative on held-out test even after high-confidence
  filtering;
- the magnitude is still around `1e-4` to `1e-3`, far below the level needed
  for a visible qualitative claim or a robust top-conference headline.

Current status: `NOT COMPLETE`.

## Next Required Step

The next experiment should test whether the high-confidence carrier is too
strong for train-val LPIPS, or whether the local support itself is misaligned.
The most useful next branch is a fixed low-amplitude v5 policy, not a free
parameter sweep:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py \
  --scenes bicycle,flowers \
  --gpu 1 \
  --output_root outputs/carnet/meshsplatopt/ecsr_phase_s/phasepatch_cluster_dc_v5_highconf_lowamp_20260513 \
  --evidence_root outputs/carnet/meshsplatopt/ecsr_phase_d/surface_evidence_bary_v2wide_phasej_model \
  --candidate_label patchcluster_dc_v5_highconf_lowamp \
  --candidate_base_method ours_26000_patchcluster_dc_v5_highconf_lowamp_base \
  --candidate_test_method ours_26000_patchcluster_dc_v5_highconf_lowamp_phasej_ela \
  --candidate_trainval_method ours_26000_patchcluster_dc_v5_highconf_lowamp_phasej_trainval_gate \
  --delta_operator dc \
  --delta_candidate_cluster_json 'outputs/carnet/meshsplatopt/ecsr_phase_s/view_support_graph_phasej_bary_v2wide_20260513/{scene}/view_support_graph.json' \
  --delta_cluster_operator_types certificate_cluster_contraction_candidate,surface_attached_attribute_recovery_candidate \
  --delta_max_clusters 16 \
  --delta_cluster_min_redundancy_score 0.55 \
  --delta_cluster_expand_with_top_residual_faces \
  --delta_cluster_expand_target_faces 256 \
  --delta_policy_val_filter_faces \
  --delta_policy_val_face_min_samples 8 \
  --delta_policy_val_face_min_relative_gain 0.90 \
  --delta_policy_val_face_max_keep 24 \
  --delta_strength 0.04 \
  --delta_max_abs_rgb 0.004 \
  --delta_min_policy_val_samples 128 \
  --delta_min_policy_val_unique_faces 8 \
  --gate_min_balanced_delta 0.0 \
  --skip_failed_views \
  --wandb_group phase_s_patchcluster_dc_v5_highconf_lowamp_20260513
```

The runner now expands the quoted `{scene}` placeholder per scene.

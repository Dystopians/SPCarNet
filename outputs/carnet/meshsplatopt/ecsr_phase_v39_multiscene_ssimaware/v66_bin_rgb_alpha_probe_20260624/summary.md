# v66 Bin-RGB Alpha Probe Persistent Summary

Date: 2026-06-24

Status: `NOT_PROMOTED_NEGATIVE_DIAGNOSTIC`

This directory mirrors the key v66 probe artifacts from the original `/dev/shm` run into a persistent output path.

## Results

| scene | reference | v66 PSNR | v66 SSIM | v66 LPIPS | decision |
|---|---:|---:|---:|---:|---|
| counter | v56/v64: `26.756130 / 0.862126 / 0.251691` | 26.751209 | 0.862078 | 0.251961 | reject |
| kitchen | v64: `27.822626 / 0.876538 / 0.198849` | 27.822626 | 0.876538 | 0.198849 | tie, not promoted |

## Artifacts

```text
counter/results.json
counter/surface_residual_region_texture_adapter_audit.json
kitchen/results.json
kitchen/surface_residual_region_texture_adapter_audit.json
logs/apply_metrics_counter.log
logs/apply_metrics_kitchen.log
```

## Interpretation

On the probed cap-hit / accepted-policy scenes, RGB channel-wise bin alpha does not improve over the current selected policy. It slightly regresses `counter` and ties `kitchen`, so the current promoted stack remains:

```text
headline endpoint: Phase-J guarded adaptive Evidence Lumigraph Adapter
best fixed representation-level policy: v64 fixed auto bin-alpha policy
negative diagnostics: v65 teacher basis, v66 bin-RGB alpha
```


# Final Stage F2 Baseline Registry Design

Date: 2026-05-04

## Purpose

The final result pipeline needs one source of truth for fair comparisons. The registry prevents the previous failure mode where a long method checkpoint was compared against a short clean baseline.

## Collector

Script:

```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python scripts/car_model/final_collect_baselines_and_results.py
```

Outputs:

```text
outputs/carnet/meshsplatopt/final_baseline_registry/final_results.json
outputs/carnet/meshsplatopt/final_baseline_registry/final_results.csv
outputs/carnet/meshsplatopt/final_baseline_registry/final_results.md
```

## Row Contract

Each row records:

- scene;
- method label;
- source checkpoint;
- training start iteration;
- final iteration;
- triangle count;
- vertex count;
- independent PSNR/SSIM/LPIPS;
- sparse AbsRel, Depth MAE, and normal angle;
- W&B URL;
- exact command or report path;
- metric source path;
- metric source type;
- topology freeze flag;
- sparse-depth loss flag;
- sparse sampling mode, fraction, lambda, and decay;
- edit primitive flag;
- edit class;
- prior-only flag;
- decision;
- integrity warnings.

## Integrity Checks

The collector checks:

1. R53.01 reproduces the all-metric table against clean 22k.
2. R44.01 is flagged as render-losing against clean 22k.
3. No headline comparison uses long method versus parking clean 7k.
4. Missing independent metrics are flagged per row.
5. Training-time or manual metrics are flagged per row.
6. Training rows without W&B URLs are flagged.
7. Checkpoint topology is loaded from `point_cloud_state_dict.pt` when available and compared against reported triangles/vertices.

## Initial Coverage

The initial registry includes:

- parking clean 7k, 22k, and 30k;
- R44.01, R43.01b, R48.01, R50.01, R53.01, R55.01, and R56.01;
- courtyard R40.02, R43.02b, and R44.02;
- bonsai R31.03, R41.01, and R42.01;
- Stage35 courtyard and bonsai public baselines;
- R15 multi-scene medium rows;
- R57/R58 matched clean-to-compact public-scene rows;
- R59/R60 matched room/counter rows from the F0 addendum.

Rows with missing sparse geometry or training-only metrics remain in the registry, but are explicitly flagged and cannot silently become headline evidence.

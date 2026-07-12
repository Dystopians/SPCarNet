# FREEZE MANIFEST — pre-submission code/config/seed freeze

- git commit: `bf3ee65a3feef135b2c462efb09b010a18df95f3` (branch neurips-meshsplatopt-repair)
- python: 3.11.14  · torch: 2.7.1+cu126
- env: /home/peilincai/micromamba/envs/mesh_splatting (frozen; R1/IBR/Difix cells use layered venvs, never modifying it)

## Seeds (universal)
- ALL training/eval/bootstrap seeds = 0 (train.py --seed 0; fusion trainer TRAIN_CONFIG['seed']=0; bootstrap seed 0, 10k; toy generator --seed 0; problem sampler SEED=0).
- Deterministic caveat (documented in EDIT_AWARE_ECR_PROTOCOL amendment 4): renders are deterministic within a process, not across processes; all paired CIs are within-run.

## Frozen configs (sha256 of sorted JSON)
- ELA/PJ-2026 transport config: `48676a5bf8c16d9c` = {"depth_abs_tol": 0.02, "depth_rel_tol": 0.03, "direction_weight": 0.35, "edge_gate": false, "edge_gate_dilate": 0, "edge_gate_min": 0.0, "edge_gate_quantile": -1.0, "evidence_max_side": 0, "k": 4, "local_trust_agreement_scale": 0.04, "local_trust_confidence_quantile": -1.0, "local_trust_gate": false, "local_trust_max_residual_std": -1.0, "local_trust_min_agreement": 0.0, "local_trust_min_confidence": 0.0, "local_trust_min_supports": 2, "local_trust_min_weight": 0.0, "local_trust_mode": "hard", "min_confidence": 0.0001, "mode": "residual", "residual_clip": 0.25}
- alpha calibration policy: {"alpha_grid": [0.0, 0.125, 0.25, 0.5, 0.75, 1.0], "calib_max_views": 16, "calib_sampler": "stride_first", "calib_stride": 16, "policy_objective": "psnr"}
- fusion trainer: {"batch": 4, "crop": 256, "dssim_weight": 0.2, "feature_dtype": "float16", "loss": "l1+0.2*dssim", "lr": 0.0001, "seed": 0, "steps": 3000, "weight_decay": 0.0}
- final-stack transport kwargs hash (banked garden routed row): `c186be1c37e63b19` (net sha) — per-row config hashes live in each metrics.json ecr block.

## Key checkpoint fingerprints (sha256 first 16 MiB)
- garden_cleanfixed30k: `796d17687bc2c2ae3010eb78f6180f4b…`
- toy_parking_clean30k: `7d68968deb1bf0ab2b0661c005edb5ce…`
- bonsai_cleanfixed30k: `3926f4e0db328301b3952a2bb3db5f31…`

## Evidence pack
- RESULTS/STAGE4_ECR/sha256_manifest.txt is the authoritative per-artifact hash list (byte-verified at fold time); regenerate with tools/ecr/fold_pack.sh.
- Paper tables: RESULTS/tables_tex/ (regenerate: tools/analysis/paper_tables.py). Figures: RESULTS/figures/{ecr_paper,ecr_qual,edit_aware}/ (regenerate: plot_ladder.py, plot_rd.py, ecr_qual_grids.py, edit_grids.py).
- Reference map: docs/PUBLIC_REFERENCE_MAP.md.

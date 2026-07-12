# Stage-4 ECR Reproducibility Notes

## Environment

Use this frozen interpreter for local plotting and report regeneration:

`/home/peilincai/micromamba/envs/mesh_splatting/bin/python`

Do not install into that environment for this subset.

Repository root:

`/data/peilincai/mesh-splatting`

## Artifact map

Evidence cache manifests:

- `/data/peilincai/gems_stage1/ecr_cache/garden_cleanfixed30k_l4routed/manifest.json`
- `/data/peilincai/gems_stage1/ecr_cache/bicycle_cleanfixed30k_l4routed/manifest.json`
- `/data/peilincai/gems_stage1/ecr_cache/kitchen_cleanfixed30k_l4routed/manifest.json`
- `/data/peilincai/gems_stage1/ecr_cache/garden_B50_final/manifest.json`
- `/data/peilincai/gems_stage1/ecr_cache/bicycle_B50_final/manifest.json`
- `/data/peilincai/gems_stage1/ecr_cache/kitchen_B50_final/manifest.json`

Eval row metrics. Key schema highlights observed in these files:
`rendering.mean.psnr`, `rendering.per_view.psnr`,
`rendering.per_view.lpips`, `cost.disk_mb`, `cost.total_artifact_mb`,
`ecr.cache_dir`, `ecr.manifest_sha256`, `ecr.config_hash`.

- `/data/peilincai/gems_stage1/eval/l4_garden_cleanfixed30k_routed_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/l4_bicycle_cleanfixed30k_routed_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/l4_kitchen_cleanfixed30k_routed_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/final_garden_B50_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/final_bicycle_B50_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/final_kitchen_B50_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/garden_cleanfixed30k_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/bicycle_cleanfixed30k_v1/metrics.json`
- `/data/peilincai/gems_stage1/eval/kitchen_cleanfixed30k_v1/metrics.json`

Audit output directories:

- `/data/peilincai/gems_stage1/eval/l4_garden_cleanfixed30k_routed_audit`
- `/data/peilincai/gems_stage1/eval/l4_bicycle_cleanfixed30k_routed_audit`
- `/data/peilincai/gems_stage1/eval/l4_kitchen_cleanfixed30k_routed_audit`
- `/data/peilincai/gems_stage1/eval/final_garden_B50_audit`
- `/data/peilincai/gems_stage1/eval/final_bicycle_B50_audit`
- `/data/peilincai/gems_stage1/eval/final_kitchen_B50_audit`

Analysis outputs:

- `/data/peilincai/gems_stage1/analysis/e0_pj2026/l1_gate.json`
- `/data/peilincai/gems_stage1/analysis/e0_pj2026/l2_gate.json`
- `/data/peilincai/gems_stage1/analysis/e0_pj2026/l3_gate.json`
- `/data/peilincai/gems_stage1/analysis/e0_pj2026/l4_gate.json`
- `/data/peilincai/gems_stage1/analysis/e0_pj2026/l4_vs_floor.json`
- `/data/peilincai/gems_stage1/analysis/final_stack/l5_pareto.json`
- `/data/peilincai/gems_stage1/analysis/final_stack/e07_matched_total_3dgs.json`
- `/data/peilincai/gems_stage1/analysis/final_stack/final_stack_tables.md`
- `/data/peilincai/gems_stage1/analysis/final_stack/hierarchical_cis.md`
- `/data/peilincai/gems_stage1/analysis/difix_cell/difix_garden.json`
- `/data/peilincai/gems_stage1/analysis/difix_cell/difix_bicycle.json`
- `/data/peilincai/gems_stage1/analysis/difix_cell/difix_kitchen.json`
- `/data/peilincai/gems_stage1/analysis/difix_cell/difix_table.md`

Figure scripts and outputs:

- `/data/peilincai/mesh-splatting/tools/analysis/plot_ladder.py`
- `/data/peilincai/mesh-splatting/tools/analysis/plot_rd.py`
- `/data/peilincai/mesh-splatting/RESULTS/figures/ecr_paper/ladder_ci.pdf`
- `/data/peilincai/mesh-splatting/RESULTS/figures/ecr_paper/ladder_ci.png`
- `/data/peilincai/mesh-splatting/RESULTS/figures/ecr_paper/rd_master.pdf`
- `/data/peilincai/mesh-splatting/RESULTS/figures/ecr_paper/rd_master.png`

## One-command examples

The referenced repo paths exist:
`/data/peilincai/mesh-splatting/tools/ecr/build_cache.py`,
`/data/peilincai/mesh-splatting/run_eval.py`,
`/data/peilincai/mesh-splatting/tools/audit_test_path.py`,
`/data/peilincai/mesh-splatting/tools/ecr/train_fusion.py`.

Build cache:

`python -m tools.ecr.build_cache --checkpoint <ckpt> --scene <scene> --out <cache> --gpu N`

Eval row:

`python run_eval.py --checkpoint <ckpt> --scene <scene> --out <dir> --gpu N --renderer ecr --ecr-cache <cache>`

Audit:

`python tools/audit_test_path.py ... --ecr --ecr-cache <cache> --fast`

Fusion training:

`python -m tools.ecr.train_fusion --cache <cache> --gpu N --routed`

Report generators:

- `/data/peilincai/mesh-splatting/tools/ecr/final_report.py` - writes final stack markdown/json tables from banked rows and gate JSONs.
- `/data/peilincai/mesh-splatting/tools/ecr/l5_report.py` - writes L5 cache Pareto markdown/json from banked variant metrics.
- `/data/peilincai/mesh-splatting/tools/ecr/e07_build.py` - writes matched-total 3DGS comparison outputs from R1 and L4 artifacts.
- `/data/peilincai/mesh-splatting/tools/analysis/hboot_report.py` - writes hierarchical bootstrap re-analysis for headline aggregates.

## Determinism notes

- Bootstrap reports use seed 0 and 10k resamples in the observed Stage-4
  scripts (`tools/ecr/e0_report.py`, `tools/ecr/rung_gate.py`,
  `/data/peilincai/mesh-splatting/tools/analysis/hboot_report.py`).
- Fusion training records `seed: 0` in `tools/ecr/train_fusion.py`; its
  docstring states the last iterate is the model used for the cache manifest.
- Eval metrics record the executed command, git commit, checkpoint fingerprint,
  `ecr.manifest_sha256`, and `ecr.config_hash`.
- ECR cache manifests record the frozen transport config, checkpoint metadata,
  train-view list, alpha source, and provenance command.

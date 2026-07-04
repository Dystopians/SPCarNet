# GEMS — REPRO_PACK (Stage-2/3 §8, pack v4)

One-command regeneration of every table/figure in the evidence pack from the
durable eval corpus. Assembled 2026-07-03; Stage3 pack v4 refreshed
2026-07-04.

## Environment

- Host: Linux 5.15, 8× NVIDIA RTX 6000 Ada (49 GB); CUDA 12.6 driver stack.
- Python: micromamba env `mesh_splatting`
  (`/home/peilincai/micromamba/envs/mesh_splatting/bin/python`, Python 3.11.14)
- Key packages: torch 2.7.1+cu126, numpy 2.4.2, scipy 1.17.1,
  matplotlib 3.10.8 (full spec: `pip freeze` in this env).
- Repo: `/data/peilincai/mesh-splatting`. The citable release is the tag
  `gems-evidence-v1.0`; verify the exact commit with
  `git rev-parse gems-evidence-v1.0^{}` after tagging.
- Rasterizer submodule pin: `submodules/diff-triangle-mesh-rasterization @
  b27f283` (pristine-build check: LEDGER GOAL#R-07, bit-exact).
- Eval-time commits per row are recorded in `RESULTS/aggregate/all_rows.json`
  (`provenance.eval_git_commit`, 32 distinct commits across the corpus) plus
  per-row `config_hash` and checkpoint `sha256_first16mb`.

## Data / durable inputs (read-only for this pack)

- Eval corpus (single mouth `run_eval.py`, PROTOCOL v1.1.x):
  `/data/peilincai/gems_stage1/eval/<row>/{metrics.json,row.json,geometry/,downstream/,panels/}`
- Analysis artifacts: `/data/peilincai/gems_stage1/analysis/{r3a_occupancy_routes,r3c_planner,r3b_submesh,r3final_three_state_v1,r08_sgeo,e9_failure_taxonomy,...}`
- Checkpoints: `/data/peilincai/gems_stage1/models/` (146 GB) and
  `outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/` (23 GB).
- Datasets: `/data/peilincai/mesh_datasets/{mipnerf360,SS3DM,eth3d_colmap}`,
  `/data/peilincai/gems_stage1/datasets/` (toy_parking, ss3dm ingests).

## Seeds (all frozen by PROTOCOL/pre-registration)

- Paired bootstrap: seed 0, 10,000 resamples (`tools/gems/paired_bootstrap.py`).
- d2 trajectory sampler: seed 0 (200 trajectories).
- R3 planner problems: seed 0 (100 paired problems/scene).
- g4 sampling: deterministic per PROTOCOL (1M GT samples).
- Training/FT runs: repo-default seeds unless pre-registered otherwise.
  Stage3 W1 uses train seed 1 for clean and B5@B50; W2 uses the frozen
  garden_halftrain file split and seed 0. Configs are hashed per row
  (`row.json config_hash`).

## One command per deliverable

Run from the repo root with `PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python`:

| Deliverable | Command |
|---|---|
| `RESULTS/aggregate/all_rows.json` | `$PY tools/gems/report/collect.py` |
| T1–T7 (`RESULTS/aggregate/T*.{md,csv}`) | `$PY tools/gems/report/tables.py` |
| F1/F2/F7 (`RESULTS/figures/*.{png,pdf}`) | `$PY tools/gems/report/figures.py` |
| F3/F6/F8 + captions (`RESULTS/figures/F{3,6,8}_*`) | `$PY tools/gems/report/figures_composite.py` |
| F4 qual grids (`RESULTS/figures/qual/`, GPU) | `$PY tools/gems/report/qual_grids.py` (crop-rule invocations logged in qual/manifest.json) |
| E4 half-res FPS bench (GPU, ~15 min) | `tools/gems/run_supervised.sh fpsbench -- $PY tools/gems/report/fps_bench.py --gpu 4` |
| everything (CPU parts) | `bash RESULTS/REPRO_PACK/regenerate_all.sh` |

Order matters only in that `collect.py` must run before `tables.py`/
`figures.py`; the FPS bench json is optional input to T4 (T4 renders "—"
without it and states so).

Custom output locations: every script takes `--rows/--out/--outdir` flags, so
tables can be regenerated into a scratch dir and diffed against the shipped
ones (that is exactly what `verify_t1.sh` does).

## Verification protocol (T1 from scratch)

`bash RESULTS/REPRO_PACK/verify_t1.sh` — in a FRESH process: re-runs
`collect.py` into a temp dir, re-runs `tables.py` on that fresh corpus dump,
and byte-diffs the fresh `T1_main_pareto.{md,csv}` + `T1_per_scene_detail.csv`
against the shipped ones (ignoring only the generation-timestamp line in the
.md). Result at pack-assembly time: **PASS** (see
`verify_t1_result.txt`, written by the script).

## Storage / GPU requirements

- Regenerating tables+figures: CPU-only, <1 GB RAM, seconds–minutes; the pack
  itself is ~10 MB under `RESULTS/`.
- FPS bench: 1 GPU (≈5 GB VRAM peak), ~15 min for 48 checkpoints.
- Re-running the underlying evals (not needed for the pack): 1 GPU,
  ~80 s (rendering-only scenes) to ~8 min (geometry+downstream) per row;
  corpus total ≈ 234 rows.
- Re-training everything from scratch: see T4's measured SS3DM 30k wall-clocks
  (~31–48 min/town) and LEDGER GOAL#002 estimates for M360 (40–80 min/scene);
  ~170 GB for checkpoints.

## Honesty notes carried with the pack

- `RESULTS/aggregate/all_rows.json` encodes the LEDGER VOID list
  (courtyard_clean30k_v2.g4; all pre-R-08 SS3DM g4 fields) — regenerated
  tables inherit it; do not consume voided fields from raw metrics.json.
- Pack v2 (final assembly, 2026-07-03): the v1 'ablation-pending' quarantine
  is retired — B3 (tag b3), E6 importance-family (abl_blend/abl_ckptimp),
  D-2 variants (toy_parking_v2/occl) and b6r_* rows are folded into the
  corpus/tables now that GOALs #012/#013/#014/#016 are closed (corpus:
  217 rows, 179 canonical). H1 and R1 are quoted as clearly-marked CONTEXT
  appendices in T1/T4 (script-read from
  `analysis/{h1_v106_context,r1_3dgs_reference}/`), never as corpus rows
  (R1 is the sanctioned outside-single-mouth exception, GOAL#017).
- Pack v4 (Stage3 closure): T7 is complete. W1 seed-1 retrain clean+B5@B50
  supports the seed waiver; W2 50% train-view-drop arm was run and the
  pose-noise/S-GEN residue is waived by Stage3 ruling. R3-FINAL V1 is folded
  into T5b and `RESULTS/CONSUMPTION_IMPOSSIBILITY.md`.
- The half-res FPS column is bench-only (non-protocol resolution) and was
  measured under partial GPU contention (documented in T4's header).

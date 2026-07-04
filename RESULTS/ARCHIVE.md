# GEMS Evidence Archive

Generated 2026-07-04 for Stage3 closure.

## Release Identity

- Release tag: `gems-evidence-v1.0`
- Exact release commit: resolve the tag with
  `git rev-parse gems-evidence-v1.0^{}`
- Repo path at assembly: `/data/peilincai/mesh-splatting`
- Evidence root: `/data/peilincai/gems_stage1/`

The tag is the citable source reference. The file cannot contain the hash of
the same commit that contains this file without a self-reference cycle, so the
tag resolution command above is the authoritative exact-commit lookup.

## Pack Contents

- `RESULTS/aggregate/`: regenerated T1-T7 and `all_rows.json`
- `RESULTS/figures/`: regenerated F1-F8 and qualitative/video manifests
- `RESULTS/CONSUMPTION_IMPOSSIBILITY.md`: R3-FINAL closure addendum
- `RESULTS/NEGATIVE_RESULTS.md`: first-class negative result ledger
- `RESULTS/CLAIMS_EVIDENCE_MATRIX.md`: claim-to-evidence audit
- `RESULTS/REPRO_PACK/`: regeneration and verification scripts
- `SUBMISSION_HANDOFF/`: venue memo, rebuttal bank, figure notes, abstract skeleton

Durable external artifacts are under `/data/peilincai/gems_stage1/`, especially:

- Eval corpus: `/data/peilincai/gems_stage1/eval/`
- Analysis artifacts: `/data/peilincai/gems_stage1/analysis/`
- Models/checkpoints: `/data/peilincai/gems_stage1/models/`
- Stage3 R3-FINAL: `/data/peilincai/gems_stage1/analysis/r3final_three_state_v1/`

## Verification Record

- Pack v4 regeneration command:
  `PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python bash RESULTS/REPRO_PACK/regenerate_all.sh`
- T1 byte-diff verifier:
  `PY=/home/peilincai/micromamba/envs/mesh_splatting/bin/python bash RESULTS/REPRO_PACK/verify_t1.sh`
- Verification result:
  `RESULTS/REPRO_PACK/verify_t1_result.txt`
- Latest verifier verdict:
  `VERDICT: PASS` at `2026-07-04T07:48:36Z`

## Environment Pins

- Python executable: `/home/peilincai/micromamba/envs/mesh_splatting/bin/python`
- Python version recorded by REPRO_PACK: 3.11.14
- Key packages recorded by REPRO_PACK: torch 2.7.1+cu126, numpy 2.4.2,
  scipy 1.17.1, matplotlib 3.10.8
- Rasterizer submodule pin: `submodules/diff-triangle-mesh-rasterization @ b27f283`

## Freeze Procedure

1. Commit only the Stage3 closure files and generated pack updates.
2. Create the release tag:
   `git tag -a gems-evidence-v1.0 -m "GEMS evidence pack v1.0"`
3. Record the tag target:
   `git rev-parse gems-evidence-v1.0^{}`
4. Push the commit and tag to the intended remote.
5. Create an archive tarball if needed:
   `git archive --format=tar.gz -o gems-evidence-v1.0.tar.gz gems-evidence-v1.0`

The large checkpoints, eval renders, and analysis artifacts are not embedded in
the git archive; preserve `/data/peilincai/gems_stage1/` or rsync it alongside
the source release when creating a citable artifact.

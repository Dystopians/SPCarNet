# PRISM MeshPrior Final Handoff

Date: 2026-05-02

## Current Method in Plain Terms

PRISM is a conservative topology-control layer for mesh-splatting optimization. It watches the current mesh during training, proposes small topology edits, tests those edits against scene evidence, commits only safe edits, and rolls back edits that fail later validation. The best current variant is M35 retained relaxed refresh: after a normal PRISM commit, it can attempt one extra relaxed post-commit edit, but only behind strict gates and final retained-topology auditing.

The strongest story is not "object prior fixes cars" or "radar reconstructs mesh scenes". The strongest story is:

> mesh-splatting topology can be edited safely when every edit is proposed, gated, validated, rolled back when needed, and audited.

## Best Evidence

- Parking long-budget topology-retention row:
  - `254491` triangles
  - independent PSNR `17.314823`, SSIM `0.559230`, LPIPS `0.442099`
  - source: `outputs/carnet/meshprior/parking_phone_tiny/stage24_2_topology_retention/freeze_after_first_commit_7000iter/model`
- Mip-NeRF 360 `bonsai` M35:
  - `633275` triangles versus Stage33 `633787`
  - independent PSNR `12.267367`, SSIM `0.277617`, LPIPS `0.611939`
  - one active retained relaxed edit; four validation-rolled-back relaxed edits recorded
- ETH3D `courtyard` M35:
  - `101913` triangles
  - independent PSNR `15.383161`, SSIM `0.508091`, LPIPS `0.584694`
  - improves topology/PSNR/SSIM among selected rows, but LPIPS is a tradeoff

## Main Artifacts

- manuscript draft: `docs/car_model/reports/meshprior_prism_manuscript_draft.md`
- reproducibility appendix: `docs/car_model/reports/meshprior_prism_reproducibility_appendix.md`
- final paper table: `outputs/carnet/meshprior/stage38_paper_assets/final_paper_table.md`
- metric reconciliation table: `outputs/carnet/meshprior/stage36_metric_reconciliation/metric_reconciliation_table.md`
- visual panels: `outputs/carnet/meshprior/stage37_visual_failure_package/visual_panels/`
- failure cases: `outputs/carnet/meshprior/stage37_visual_failure_package/failure_case_table.md`
- reviewer risks: `docs/car_model/reports/meshprior_prism_reviewer_risk_checklist.md`
- bibliography draft: `docs/car_model/reports/meshprior_prism_bibliography_draft.bib`

## What Not To Claim

- Do not claim universal image-quality dominance.
- Do not hide `courtyard` LPIPS regression.
- Do not claim radar-only reconstruction.
- Do not treat COLMAP sparse geometry as ground truth.
- Do not use Tanks and Temples geometry claims until true sparse tracks are rebuilt.
- Do not mix training-time metrics with independent `render.py + metrics.py` values.

## Experiment Trigger

Full-budget public Stage35 training is `NO_GO_FOR_NOW`.

Run one full-budget public Stage35 experiment only if all of the following are true:

1. The manuscript table has a named missing row that cannot be answered by existing artifacts.
2. The row is expected to change the core claim or reviewer risk, not merely add another diagnostic.
3. The target scene is geometry-observable with COLMAP tracks.
4. W&B can be run online.
5. GPU availability is checked immediately before launch.

Preferred target if this trigger fires: one public COLMAP-compatible scene already present locally, likely Mip-NeRF 360 `bonsai` or ETH3D `courtyard`, using the M35 retained relaxed schedule.

## Next Human Editing Tasks

1. Replace TODO authors in `meshprior_prism_bibliography_draft.bib`.
2. Add exact citations into the manuscript draft.
3. Draw the PRISM method overview figure.
4. Build the relaxed-commit timeline figure from the M35 `bonsai` audit JSON.
5. Tighten the abstract after deciding final page budget.


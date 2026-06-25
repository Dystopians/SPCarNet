# v101 Subagent Review and Paper Story

Date: 2026-06-25

Scope: Subagent 5 review and paper-story synthesis for the current v101 method. This note summarizes existing local docs/results only. It does not replace the detailed evidence log in `docs/car_model/6-25-v101-RenderPyEndpoint-EvidenceBank-Log.md`.

Primary sources:

- `docs/car_model/6-25-v101-RenderPyEndpoint-EvidenceBank-Log.md`
- `docs/car_model/6-25-v100-FixedFull9-CheckpointAttachedELA-Sidecar.md`
- `docs/car_model/6-25-v96-Subagent-Paper-Story-And-Review.md`
- `README.md` and `README.zh.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_v101_renderpy_endpoint_full9_bankfp16_fixed_20260625/v101_renderpy_endpoint_full9_summary.md`
- `outputs/carnet/meshsplatopt/ecsr_phase_v101_detached_package_20260625/counter_detached_package_report.json`
- `outputs/carnet/meshsplatopt/ecsr_phase_v101_runtime_audit_20260625/counter_runtime_audit.md`

## 0. Review Verdict

v101 is a meaningful paper-story improvement over v100 because it moves the Phase-J/v100 checkpoint-attached ELA endpoint into the normal `render.py` entrypoint and adds a forceable train-derived evidence bank. The strongest current claim is:

> SPCarNet v101 packages the Phase-J guarded ELA repair as a `render.py`-consumable checkpoint endpoint with a train-derived residual/depth/camera evidence bank, preserving the local full9 gains over the selected clean MeshSplatting baseline while making the deployment artifact more auditable.

The claim must stay narrow:

- v101 is not an independent metric improvement over Phase-J.
- v101 is not a fully baked vanilla MeshSplatting checkpoint.
- v101 still evaluates endpoint logic at render time.
- v101 does not support a speedup claim; the counter runtime audit shows a `1.885235x` wall slowdown versus standard `render.py`.

## 1. Method Modules

| Module | Role | Current evidence |
|---|---|---|
| MeshSplatting base checkpoint | Starting representation and fair local clean baseline | local selected clean baseline remains the main comparison |
| Phase-J guarded ELA | Strong RGB repair endpoint inherited by v100/v101 | full9 Phase-J/clean evidence already closed in prior docs |
| v100 checkpoint-attached ELA sidecar | Packages Phase-J as an auditable endpoint artifact | full9 sidecar passes; still not directly consumed by vanilla `render.py` |
| v101 `render.py` endpoint hook | Loads endpoint report, recomputes target base renders, applies guarded residual transfer, writes endpoint renders | full9 auto endpoint and require-bank fp16 endpoint both complete with zero render/eval return codes |
| v101 evidence bank builder | Stores train-derived residuals, depths, cameras, hashes, and manifest beside the endpoint | require-bank fp16 full9 builds and consumes scene-specific banks for all 9 scenes |
| Detached-package validator | Tests whether the package can run without reading original train evidence folders | counter detached package passes with `used_required_bank=true` and `30/30` PNG SHA-256 matches after forcing the package-local bank and overriding the endpoint base model to a nonexistent path |
| Runtime audit | Measures deploy-time overhead | counter standard render: `2.238598 sec/view`; v101 require-bank render: `4.220285 sec/view` |

The key paper-facing distinction from generic image postprocessing is that the residual evidence is selected from train/support views and transferred through the existing MeshSplatting surface/camera/depth evidence path. The key limitation is that the endpoint still performs this transfer at render time.

## 2. Quantitative Evidence

Strongest current v101 row: `ours_26000_v101_bankfp16_renderpy_endpoint_full9_fixed`.

| Evidence item | Status |
|---|---|
| Full9 scenes | `9 / 9` complete |
| Bank build / render / eval return codes | all zero |
| Mean PSNR / SSIM / LPIPS | `26.481309 / 0.783675 / 0.224305` |
| Mean delta vs selected clean MeshSplatting | `+1.329627 PSNR`, `+0.034657 SSIM`, `-0.063316 LPIPS` |
| Scene-level wins vs selected clean | all 9 rows are positive in PSNR/SSIM and negative in LPIPS |
| Mean drift vs Phase-J | `-0.001457 PSNR`, `-0.000044 SSIM`, `+0.000043 LPIPS` |

Interpretation:

- The bank-backed v101 endpoint preserves the main local full9 Phase-J gains over the selected clean MeshSplatting baseline.
- The tiny Phase-J drift should be described as deployment/quantization/path drift, not as a new quality claim.
- The auto endpoint run is useful for `render.py` entrypoint reproduction, but the require-bank fp16 run is the cleaner evidence-bank claim because all scenes force bank consumption.

Counter-specific supporting checks:

- Float32 bank/non-bank render.py endpoint paths reproduce v100 on counter with `30/30` render PNG SHA-256 matches.
- Target-GT non-use smoke reports `max_abs_output_diff=0.0` after replacing the target GT path with a dummy nonexistent path.
- Detached-package counter validation reports identical metrics to the reference bankfp16 output and `30/30` PNG hash matches.

## 3. Qualitative Evidence

Current v101 qualitative asset:

- panel: `assets/spcarnet_v101_bankfp16_full9_qualitative_panel.png`
- manifest: `assets/spcarnet_v101_bankfp16_full9_qualitative_panel_manifest.json`

The panel compares local clean MeshSplatting, v101 bankfp16, and GT on selected held-out crops. The manifest records scene/view/crop paths plus LPIPS improvement and crop absolute-error reduction. This is useful for paper/PPT because full-frame differences are often subtle; captions should state that crops were selected by held-out LPIPS/crop-error improvement and should keep the manifest path attached.

## 4. Relation to MeshSplatting Baseline

Standard MeshSplatting:

```text
train checkpoint -> direct render.py render -> held-out evaluation
```

SPCarNet v101:

```text
trained MeshSplatting/compact checkpoint
  -> Phase-J/v100 endpoint report
  -> train-derived residual/depth/camera evidence bank
  -> render.py endpoint hook
  -> guarded residual transfer at render time
  -> held-out evaluation
```

The fair comparison remains the local selected clean MeshSplatting baseline under the same split/evaluator. v101 inherits the Phase-J improvement pattern over that baseline: quality gains are broad and strong, but geometry/compactness come from the compact parent and evidence-gated pipeline rather than from v101 adding new geometry improvements.

Paper wording should say that v101 "builds on MeshSplatting" or "adds an evidence-certified post-training endpoint to MeshSplatting checkpoints." It should not say that v101 replaces MeshSplatting training or that the repair is absorbed into a vanilla checkpoint.

## 5. Weaknesses and Reviewer Risks

1. Endpoint, not baked representation.
   v101 still needs endpoint logic in `render.py`. A standard checkpoint render without the endpoint hook will not reproduce the repaired output.

2. Runtime is worse than standard render.
   The counter audit measures `4.220285 sec/view` for v101 require-bank versus `2.238598 sec/view` for standard render on the detached package. This blocks speed/deployment-efficiency claims.

3. Full9 detached packaging is not closed.
   Counter detached-package validation is strong, but full9 detached-package validation is still missing.

4. fp16 bank has small Phase-J drift.
   The biggest observed dPSNR drifts are on `counter`, `flowers`, and `kitchen`. If exact Phase-J parity matters, add float32-bank checks for those scenes or a full9 float32 bank run.

5. Qualitative improvements need traceability.
   Crop panels are useful, but full-frame differences can be subtle. Every qualitative claim should cite the manifest and exact scene/view/crop.

6. Version sprawl remains a paper risk.
   v95/v96/v99/v100/v101 should not appear as a version-number story in the paper. The clean story is Phase-J evidence repair, v100 sidecar packaging, and v101 render.py/evidence-bank deployment closure.

## 6. Paper Story

One-sentence positioning:

> SPCarNet uses MeshSplatting's trained surface as an address space for post-training evidence: it audits which regions can be compacted, transfers stable train-derived residuals through guarded surface correspondence, and falls back when evidence is insufficient.

v101-specific contribution wording:

> v101 closes the artifact gap between an offline Phase-J sidecar and a reproducible renderer endpoint: the repaired output is generated by `render.py` from a checkpoint-attached endpoint report plus a train-derived evidence bank, with no held-out target GT used for policy or support evidence.

Recommended paper hierarchy:

1. Main method: evidence-certified post-training repair and compaction for MeshSplatting checkpoints.
2. Main local result: full9 gains over selected clean MeshSplatting under the same local evaluator.
3. Deployment artifact closure: v101 `render.py` endpoint plus forceable evidence bank.
4. Limitations: render-time overhead, not a vanilla baked checkpoint, full9 detached package still pending.

## 7. Next Actions

P0:

- Run full9 detached-package validation, or at least hard-triad detached validation on `counter/kitchen/bonsai`.
- Add float32-bank targeted checks for `counter`, `flowers`, and `kitchen` if exact Phase-J parity is needed.
- Create a final claim-to-artifact manifest that maps each table/figure sentence to an exact output path and command.

P1:

- Benchmark whether preprojected residual banks or lower-cost endpoint kernels can reduce the `1.885235x` detached-package slowdown.
- Keep representation-level baking work separate from the v101 claim unless it beats the predeclared anchors.
- Prepare reviewer ablations around no bank, fail-closed require-bank, compact-only, ELA without gate/fallback, and target-GT non-use.

P2:

- Refresh paper/PPT figures from the v101 bankfp16 panel only after captions are tied to the manifest.
- Update higher-level README/mentor docs in a separate owned edit if the team wants v101 to replace v100 as the current deployment-closure milestone.

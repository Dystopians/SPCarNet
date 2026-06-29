# SPCarNet Claim Readiness Auto Report

Generated from current local artifacts. Missing volatile `/dev/shm` files are treated as missing evidence.

Manual freshness note: this auto report predates the v168 low-copy/direct-teacher unblock patch and the currently running exact flowers attempt. For the freshest handoff, read `feedback.md` and `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md`. The conservative paper verdict remains `NOT COMPLETE`.

## Storage

- `/data`: 26.4T / 27.8T used, 0.01G free, 95.0%
- `/dev/shm`: 0.2T / 0.2T used, 6.43G free, 97.4%
- `/tmp`: 7.1T / 13.8T used, 6199.55G free, 51.2%

## Claim Matrix

| claim | status | evidence | blocker |
|---|---|---|---|
| Phase-J is strongest local RGB endpoint | PASS_LOCAL | scene wins 9/9; outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.json | Clarify that this is a render-time endpoint, not baked representation. |
| v106 is strongest verified baked representation over selected clean | PARTIAL_PASS | v106 25.831280/0.760830/0.268435 vs clean 25.151682/0.749018/0.287621 | Still weaker than Phase-J; qualitative gain is subtle. |
| v166 flowers all-axis beats Phase-J | FAIL | metrics 20.452814/0.549059/0.355544; manifest COMPLETE, protocol=True, errors=0 | Wins PSNR only; loses SSIM/LPIPS vs Phase-J flowers. |
| v167 flowers all-axis beats Phase-J | FAIL | metrics 20.452776/0.549059/0.355544; manifest COMPLETE, protocol=True, errors=0 | Affine/patch candidate was policy-val rejected and fell back to no-op. |
| v168 Phase-J distillation profile is exact metric win | NOT_RUN | dry-run manifest status=DRY_RUN, protocol=True, errors=0; log_exists=True | Exact flowers validation not run; storage is unsafe. |
| vNext/new prompt is paper-main method | FAIL | v165-v167 negative flowers evidence; vNext full9 below clean/v106/Phase-J. | Needs flowers all-axis win vs Phase-J, then fixed-policy full9 and ablations. |

## Flowers Gate Against Phase-J

- Phase-J flowers reference: 20.304358 / 0.557770 / 0.329222
- v166: 20.452814 / 0.549059 / 0.355544; all-axis=False
- v167: 20.452776 / 0.549059 / 0.355544; all-axis=False
- v168: no exact metrics yet.

## Key Artifact Index

- Phase-J closure JSON: `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.json` exists=True
- Phase-J closure CSV: `outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv` exists=True
- vNext structure full9 summary: `docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/vnext_manifest_summary_enhanced.md` exists=True
- vNext effective-margin full9 summary: `docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/vnext_manifest_summary_enhanced.md` exists=True
- v166 manifest: `/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json` exists=True
- v167 manifest: `/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json` exists=True
- v168 dry-run manifest: `/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers/reports/flowers_vnext_certified_residual_texture_manifest.json` exists=True
- v168 durable log: `docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md` exists=True

## Verdict

Final status: NOT COMPLETE.

The current repo has strong engineering scaffolding and local Phase-J/v106 evidence, but the vNext/new-prompt route is not paper-main ready until a fixed, no-target-GT, Phase-J-distilled baked representation beats Phase-J on flowers all-axis and then survives full9 promotion.

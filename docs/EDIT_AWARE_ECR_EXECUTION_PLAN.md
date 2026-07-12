# EDIT-AWARE ECR — EXECUTION PLAN

Opened 2026-07-12. Deletion prototype ONLY (per audit: classes 3–4 out of scope; class 2 a corollary).
Stop-rule: if the C2 stale-ECR control shows NO material ghosting on both scenes, Route A's premise
fails → NO-GO and stop before further implementation.

## Steps & owners

1. **Warp mask hook** (me — audited-path code): `FrameRecord.mask_path` (default None) + confined
   `mask()` in both loaders + `support_valid` sampling in `warp_support_residual` (+ threading through
   `adapt_frame`, `adapt_frame_l2`, `compute_transport_features`). Default-off; verify the banked garden
   routed config hash is bit-identical after the edit before anything else runs.
2. **Edit tool** (me): `tools/edit/delete_faces.py` — box → face set → edited checkpoint + edit spec json.
3. **Edited-cache builder** (me): `tools/edit/build_edited_cache.py` — original-ckpt rend_ids pass →
   masks + affected list; edited-ckpt renders/depths for affected views; hardlinks otherwise; manifest
   with edit block; net + (K,α) copied.
4. **Eval harness** (me): `tools/edit/edit_eval.py` per protocol (C1/C2/C4/C5 + metrics + panels).
5. **Box selection** (me, one manual iteration allowed per scene, recorded in the spec json BEFORE any
   metric run): render one view of the edited checkpoint to confirm the object is gone and collateral
   deletion is acceptable; box adjustments after seeing METRICS are forbidden.
6. **Runs:** E1 toy_parking then E2 garden; C4 full rebuilds; temporal on E2/C5. GPUs 5/7-class.
7. **Codex (delegated):** visualization grid assembler for the panels; doc sync sweeps; (optional)
   mask-dilation utility tests. All numeric work stays in the harness (mine).
8. **Verdict:** finalize `EDIT_AWARE_ECR_VALUE_REPORT.md` with per-class scores, controls table,
   reviewer-attack analysis, and the GO/CONDITIONAL-GO/NO-GO + next-action recommendation.

## Frozen decisions (made now, before numbers)

- Reuse original (K, α) and fusion net in C5 (isolate invalidation; retraining would confound).
- Mask dilation: 1 px at source views (splat bleed); 8 px eval-region context; 16 px exclusion for U.
- No generative infill anywhere; disocclusions fall to the edited base render by the structural gate.
- One manual box iteration per scene max; box frozen in the edit spec before metrics.
- Deletion only; recolor/translation/deformation get scores from the audit, not prototypes, this cycle.

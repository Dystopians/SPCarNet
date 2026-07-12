# EDIT-AWARE ECR (Route A) — FEASIBILITY AUDIT

Opened 2026-07-12. Mission: evidence-backed GO / CONDITIONAL-GO / NO-GO on Route A
(triangle-level edit → identify affected evidence → invalidate/transform stale cache entries → local
rebuild → rerender without stale-appearance leakage → preserve ECR quality elsewhere).
Companions: `EDIT_AWARE_ECR_PROTOCOL.md` (frozen eval), `EDIT_AWARE_ECR_EXECUTION_PLAN.md`,
`EDIT_AWARE_ECR_VALUE_REPORT.md` (verdict). Banked Stage-4/5 evidence untouched throughout.

## 1. Data-model audit (code-verified, with line pointers)

| Trace question | Verdict | Evidence |
|---|---|---|
| source pixels → face IDs | **YES, free** | every render returns the per-pixel winning face id: `triangle_renderer/__init__.py:283` (`'rend_ids': render_id`); consumed at scale by ECSR (`scripts/car_model/ecsr_build_surface_evidence_cache.py:408`). Note: produced at render resolution (×4 supersampling), nearest-interpolated to image size by existing consumers. |
| source pixels → barycentric coords | **NOT stored; 2D-approx reconstructable** | the rasterizer does not expose barycentrics (documented at `ecsr_build_surface_evidence_cache.py:252`); `_compute_visible_face_barycentric` reconstructs image-plane barycentrics from projected vertices (approximate, chunked). NOT needed for deletion/recolor; needed (in 3D-exact form) for deformation. |
| cached evidence → checkpoint fingerprint | **YES, audited** | cache manifest pins `checkpoint` fingerprint (sha256-16MiB + size); `run_eval` refuses mismatches; `--ecr` audit re-verifies per row. Edit lineage requires only an added provenance block (parent fingerprint + edit spec), not a mechanism change. |
| edited faces → affected source views & transport paths | **Derivable exactly** | face identity = row index of `_triangle_indices`; one render pass per train view on the ORIGINAL checkpoint gives `rend_ids`; affected view ⟺ any pixel ∈ deleted-face set; per-view stale-pixel mask = `isin(rend_ids, deleted)` (+1 px dilation for splat edges). Transport paths follow from the frozen support selection. |
| edit application to the artifact | **Trivial** | checkpoint = vertices (1.97M) + `_triangle_indices` (6.59M×3) + per-VERTEX features + per-TRIANGLE stats (`importance_score`, `image_size`, `pixel_count`). Deletion = boolean-index the triangle-indexed tensors; orphaned vertices are harmless; the B-zoo prune machinery is precedent that subset checkpoints render correctly. |
| single injection point for invalidation | **YES — one function** | ALL fuse modes sample support evidence through `utils/evidence_lumigraph_adapter.py::warp_support_residual` (single: adapter:808; L2: `transport_l2.py:136`; L3/L4 features incl. the color warp: `fusion.py:66,77`). Sampling a per-source-pixel validity mask with the same grid and multiplying into `confidence` (~20 lines, default-off) invalidates evidence structurally — masked evidence gets zero weight, and the existing β·valid compose gate already handles "no evidence ⇒ base render" (banked: 1-in-139 failure census, coverage-gap grace). |

**The pivotal scientific fact found by this audit:** a FULL cache rebuild from the edited checkpoint does
NOT fix staleness — the GT photographs still contain the deleted object, so the rebuilt residuals
(GT − edited-render) encode the object as a large "correction" at its old location and the transport
paints it back. Staleness lives in the *photographs*, not the cache files. Masking is therefore not an
optimization of rebuild — it is the only correct mechanism, and "full rebuild" becomes a CONTROL that
demonstrates the problem is fundamental. This makes the hypothesis falsifiable and the contribution
non-obvious (see control C4 in the protocol).

## 2. Edit-class assessment

| Class | Feasibility | Sci. value | Impl. cost | Cache-consistency difficulty | Objective evaluability | Top-conf likelihood |
|---|---|---|---|---|---|---|
| 1. Triangle/object deletion | **85** | **75** | LOW (edit tool + mask pass + 1 warp hook + cache builder) | LOW-MED (masks per affected view; renders/depths regenerated for affected views only; GT retained+masked) | **HIGH** (leakage vs edited-base/original-render; masked-GT preservation outside the region is a TRUE-GT metric) | **MED-HIGH** — as the anchor of the "why retain the mesh" story |
| 2. Appearance-only (recolor) | 70 | 45 | LOW (same mask machinery; recolored-face evidence masked → base shows the recolor, no transport gain in-region) | LOW | MED (edited reference = edited base render only) | LOW-MED standalone — mechanism identical to deletion; a corollary, not a second contribution. "Analytic transform" of evidence colors is shading-approximate → hallucination-adjacent; NOT pursued |
| 3. Rigid translation | 55 | 65 | HIGH (object-pixel evidence transform by T; dual occlusion update; depth edits both locations) | HIGH (moved evidence must re-enter the depth-consistency test at the new location; old location = deletion problem; new location shadows/baked lighting are physically wrong) | MED | MED — honest version = "transform vs mask" ablation; high demo-ware risk |
| 4. Deformation / topology | 35 | 60 | VERY HIGH (needs stored 3D-exact per-pixel face+barycentric attachment; only 2D-approx exists) | VERY HIGH (shading validity breaks with normals; stretch regions lose photographic support) | LOW-MED | LOW this cycle — **declared boundary**: not supportable from original photographs without hallucination beyond small deformations |

## 3. Evidence disposition taxonomy (per the mission's step 2)

- **Safely retained:** all evidence of unaffected views (bit-identical renders/depths, unmasked GT);
  unmasked pixels of affected views (their scene content is unchanged; the depth-consistency test
  continues to police occlusion changes near the edit).
- **Must be masked:** source pixels whose winning face ∈ deleted set (their GT photographs image the
  deleted object) — for deletion AND for recolor (old color) — plus a 1-px dilation for splat bleed.
- **Analytically transformable:** rigid-translation object pixels in principle (backprojected points
  moved by T) — class 3 only, deferred; NOT color transforms (shading-approximate).
- **Must be regenerated:** renders + median depths of affected views (the edited checkpoint renders
  differently there); the residuals follow automatically (loader computes GT − render on the fly).
- **Unsupportable without hallucination:** background newly disoccluded at the old object location in
  views where the object occluded it (no photograph ever saw it) — falls back to the edited BASE render
  via the structural gate, and this is the honest behavior (no generative infill, per mission constraint);
  large deformations; relit/shadow-consistent translation.

## 4. Minimum missing capabilities (exact)

1. `mask_path: Optional` on `FrameRecord` + confined `mask()` loader method + `support_valid` sampling in
   `warp_support_residual` (default-off; existing rows bit-stable).
2. An edit tool: face selection (3D box → centroid test → face-id set) + checkpoint row-drop + edit spec json.
3. An edited-cache builder: original-ckpt rend_ids pass → masks + affected-view list; edited-ckpt
   render/depth regeneration for affected views; hardlinks for the rest; manifest with edit-lineage block.
4. An evaluation harness computing the frozen metrics over the 5 methods (protocol doc) — edited scenes
   have no GT, so this is a documented outside-the-mouth measurement cell (masked-GT metrics outside the
   edit region ARE true-GT metrics and need no exception).

## 5. Initial overall scores (to be finalized in the VALUE_REPORT after the prototype)

- Technical feasibility (deletion path): **85/100**.
- Scientific value (as the "why mesh" anchor): **70/100 provisional** — hinges on the prototype showing
  (a) stale-cache ghosting is real and visually/metrically large, (b) full rebuild does NOT fix it,
  (c) local invalidation fixes it at a small fraction of rebuild bytes/time with unaffected-region
  preservation ≈ exact. If (b) holds, the contribution is non-obvious; if ghosting is negligible,
  Route A collapses to engineering.
- Working verdict at audit close: **CONDITIONAL-GO** — proceed to the deletion prototype
  (classes 3–4 explicitly out of scope for the prototype; class 2 documented as corollary).

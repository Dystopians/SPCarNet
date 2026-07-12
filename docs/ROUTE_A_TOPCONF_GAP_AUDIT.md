# ROUTE-A TOPCONF GAP AUDIT — independent red-team

Opened 2026-07-12. Mission: close every addressable reviewer attack on Route A (edit-aware ECR);
readiness 100/100 = no unresolved P0 / feasible P1 (not guaranteed acceptance). Banked Stage-4/5/Route-A
evidence is never overwritten; every new number comes from new banked rows. Companions:
`ROUTE_A_TOPCONF_EXECUTION_PLAN.md` (frozen specs), `ROUTE_A_TOPCONF_READINESS_REPORT.md` (verdict).

## 1. Scorecard (0–100 per suspected gap; evidence-based)

| # | Gap | Initial | Current | Audit finding |
|---|---|---|---|---|
| 1 | Edited-region validity (do leak_R/ghost correlate with TRUE edited GT?) | **35** | 35 | P0. All current in-region metrics are reference-free proxies (deviation-from-edited-base; similarity-to-original). Fatal reviewer line: "your main metric rewards falling back to the base render — a method that deletes ALL evidence in the region scores perfectly." CLOSEABLE: `tools/gems/build_toy_parking.py` is a single-seeded generator rendering GT through the repo's own rasterizer; **no shadow baking** (Lambertian direct only, verified by grep) and **cameras depend only on constant CAR_CENTERS** (verified line ~103/472-495) → dropping car_0's faces post-build yields a bit-identical-camera oracle dataset (true edited photographs). Validates or replaces leak_R by correlation against oracle error. |
| 2 | Breadth (scenes / object sizes / visibility / masks) | **40** | 40 | P1. Currently 3 edit cells (garden table del, toy car_0 del, garden table recolor) — 2 scenes, both edits central/large. Missing: small/peripheral object, indoor scene, second object per scene. Minimum defensible: +3 cells → 6 total across 3 scenes with size/visibility spread. |
| 3 | Novelty vs simpler alternatives | **30** | 30 | P0 for the method claim. ZERO ablations exist against: (a) target-side masking (zero the transport signal in the edit region R at compose — no source provenance needed), (b) 2D source-view bounding-box masks, (c) coarse dilation of provenance masks, (d) dropping affected views, (e) global ECR disable. Scientific crux: source provenance PRESERVES valid disoccluded-background evidence inside R (photographed in views where the object never occluded it) — target masking discards it, box masks over-discard around it. Only the ORACLE benchmark can arbitrate in-region quality → gaps 1 and 3 close together. If simple masks tie, the claim must be weakened honestly. |
| 4 | Locality/cost wording | **55** | 55 | P1. "Local update" currently rewrites full renders+depths for ALL views (161/161, 72/72; 231 MB–1.05 GB) because both edited objects are centrally visible. Two fixes: (i) a peripheral-object edit measuring true view-locality; (ii) a **sparse patch sidecar** (store only changed-pixel bounding-box patches per affected view; reconstruct at cache load) — bytes become region-proportional. Wording is corrected either way to "region-proportional update", never "local" without measurement. |
| 5 | Disocclusion boundary (why deletion/recolor yes, translation/deformation no) | **60** | 60 | P1. Currently stated, not evidenced. Precise statement: supported edits only ever need content that (a) exists in the edited base render or (b) was genuinely photographed (disoccluded background). Translation moves content into configurations never photographed (contact shadows, baked lighting direction) — bank an explicit wrong-shadow failure figure (translate car_0; the baked ground shadow stays at the old location and the moved car lacks one) instead of asserting the boundary. |
| 6 | Closest prior art & baselines | **50** | 50 | P1 (writing + one context row). A10 covers the IBR lineage; missing an editing-specific related-work matrix: editable IBR / source-view masking, neural mesh editing (NeuMesh/SEAL-3D class), Gaussian editing (GaussianEditor/Gaussian-Grouping class — no evidence cache exists there: the stale-evidence problem is ECR-specific; their deletion is a REPRESENTATION edit ≈ our C1). Matrix + honest positioning paragraph; no new experiment justified (nothing comparable has an evidence cache to invalidate). |
| 7 | Temporal & repeated-edit stability | **70** | 70 | P2→P1. Temporal-after-edit banked (0.982 PASS). Missing: edit COMPOSITION (sequential deletion+recolor through the parent-chain; masks must union — current builder ignores parent masks: REAL BUG for chained edits, found by this audit). |
| 8 | One coherent contribution? | **65** | 65 | P1 (framing). The unified mechanism exists but is unwritten: ONE primitive — per-pixel evidence gating on mesh-face provenance — yields quality (β·valid transport), safety (structural worst-case + audits), and editability (provenance invalidation). Write it into the claims/paper plan; the coherence attack dissolves if the same gate is shown doing all three jobs. |

**Initial overall: 48/100.** (Route-A-specific; the static-ECR system keeps its banked 87/100.)

## 2. Blocker register

- **P0-A (gaps 1+3): oracle edited-GT benchmark + mask-strategy ablations.** Build `toy_parking_nocar0`
  via a `--drop-elements` generator flag (same seed, same RNG consumption, faces filtered post-build);
  acceptance: `sparse/0/images.txt` byte-identical to the original build's, car_0 absent from the
  census, pixel-identical renders on car_0-free views. Then score ALL methods and ALL mask strategies
  in-region against TRUE GT; validate leak_R by per-view correlation with oracle in-region error.
- **P0-B (gap 3): ablation grid** {provenance-1px (ours), dilate-4, dilate-16, 2D-box, target-mask,
  view-drop, C1} × {toy oracle, garden} — no open sweep: 7 fixed strategies, 2 scenes, frozen metrics.
- **P1-C (gap 2): breadth cells** — toy car_1 deletion (oracle-able too), garden peripheral-pot
  deletion (locality), kitchen recolor (indoor). 6 cells total.
- **P1-D (gap 4): sparse patch sidecar** + corrected wording; peripheral-edit locality numbers.
- **P1-E (gap 5): translation wrong-shadow boundary figure** (checkpoint edit + base renders only).
- **P1-F (gap 7): chained-edit support** (builder parent-mask union fix — bugfix regardless) + one
  composed-edit eval.
- **P1-G (gaps 6+8): related-work matrix + unified-mechanism claim text** (writing; synchronized into
  CLAIMS/handoff).

## 3. Honest pre-registered risks

- The oracle may reveal that in-region, ALL masking strategies ≈ edited base (if disoccluded background
  is rarely photographed) — then the novelty claim shrinks from "better in-region quality" to "equal
  quality at exact provenance + lower over-deletion collateral OUTSIDE the region" (measurable on U).
- Target-side masking may tie ours on both regions for DELETION (its true weakness may only show on
  recolor or near-region evidence loss); if it ties everywhere, the honest claim becomes about
  render-time cost (target masking needs a per-view original-checkpoint ID pass at EVERY render) and
  composability — and the paper says so.
- The sidecar adds a read-path branch to the audited renderer; it must be default-off and bit-stable
  like every prior hook.

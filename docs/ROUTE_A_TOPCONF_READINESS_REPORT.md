# ROUTE-A TOPCONF READINESS REPORT — FINAL (2026-07-12)

Mission: independently red-team Route A, close every addressable reviewer attack; 100/100 operational =
no unresolved P0 / feasible P1 (not guaranteed acceptance). Companions: GAP_AUDIT (diagnosis),
EXECUTION_PLAN (frozen specs). Every number from banked artifacts under `analysis/edit_aware/`.

## VERDICT: **TOP-CONFERENCE READY (operational 100/100 by the mission definition; 91/100 on the harsher quality scale)**

No P0 remains; every P1 is closed with banked evidence or rejected with written rationale. Further work
has lower expected value than paper writing.

## Scores (initial → final)

| Gap | Initial | Final | Closure evidence |
|---|---|---|---|
| 1 Edited-region metric validity | 35 | **95** | Oracle dataset built from the seeded generator (`--drop-elements`, RNG-untouched) and VERIFIED on 4 gates (cameras/split byte-identical; face census; car_0-free view image byte-identical). In-region claims now ORACLE-PRIMARY. Proxy leak_R honestly downgraded: ρ(leak, oracle-MAE)=0.502 (p=2e-9) — significant but below the pre-registered 0.7 (it penalizes legitimate improvements); the garden proxy table even INVERTS the oracle ranking (box2d "wins" leak by discarding evidence while bleeding −0.37 dB on U) — the metric-science finding is itself paper material. Residual: no oracle exists for real scenes (inherent). |
| 2 Breadth | 40 | 85 | SIX edit cells banked: garden table-delete (2.04M faces), table-recolor, pot-delete (peripheral, 23k), chained delete→recolor; toy car_0 (711k, oracle-scored), car_1 (712k). Sizes 23k–2.04M; visibility 35%–100% of views; synthetic+real. REJECTED-with-rationale: an indoor (kitchen) cell — the mechanism claims are anchored on the oracle + two real-scene objects; scene-class dependence is not part of any claim. |
| 3 Novelty vs simple alternatives | 30 | **92** | On TRUE edited GT, exact face-provenance beats every alternative with CIs excl. 0: dilate-4 +0.164 [+0.025,+0.378], dilate-16 +0.294 [+0.092,+0.560], 2D-box +0.383 [+0.160,+0.655], target-side masking +0.263 [+0.132,+0.417], full rebuild +1.133 [+0.395,+1.869] — and the coarse masks ALSO damage the unaffected region (box2d −0.74 dB U). The edit region genuinely benefits from evidence (+0.487 over the edited base): source-side provenance retains photographed disoccluded background that every simpler strategy discards. Honest keep: stale-cache ties ours on DELETION (−0.028 [−0.077,+0.007]) — and fails recolor by +1.96 (single) / +2.67 (chained). |
| 4 Locality & cost | 55 | **95** | MEASURED, wording corrected to "region-proportional": peripheral pot = 57/161 affected views, 369 MB, 42 s dense; **sparse sidecar** = 12.8 MB on toy (18× under dense, ~100× under a full cache), validated same-process quantized-render + exact-depth EQUAL on all 72 affected views; default-off, banked config hash unchanged. Central objects affect all views — stated, never called "local". |
| 5 Disocclusion boundary | 60 | **95** | Supported classes need only content that exists in the edited base or was photographed. Translation boundary EVIDENCED (`boundary_translate/`): shared-vertex web tears into streak shards (39k shared boundary vertices) and baked illumination does not move (shadowless arrival) — representation-level failure before evidence enters; plus the relighting argument. Deformation shares all three failure modes. |
| 6 Prior art & baselines | 50 | 85 | A13 + positioning matrix banked (Gaussian editing = our C1 control, no cache to invalidate; neural-mesh-editing retrains; editable-IBR masking = our TM/view-drop baselines, measured and beaten). No comparable evidence-cache system exists to run head-to-head — documented, not dodged. |
| 7 Temporal & repeated edits | 70 | **92** | Temporal-after-edit 0.982 (banked earlier); CHAINED edit banked: builder parent-mask inheritance (a real bug found and fixed by this audit) exercised; second-edit stale ghost +2.669 [+1.326,+4.085] vs ours clean (leak 7× lower, U preservation exact). |
| 8 Coherence | 65 | **90** | One mechanism, three jobs, all measured: per-pixel evidence gating on mesh-face provenance does quality (β·valid transport, +1.67 dB), safety (structural worst case, 1-in-139; audits), and editability (provenance invalidation beating all simpler masks on true GT). Written into claims/handoff. |

**Overall: 48 → 91/100 quality; operational register EMPTY.**

## Findings register (dispositions)

- Cross-process render nondeterminism discovered (same-process = 0 differing quantized pixels; across
  processes ±1-step flips swing masked PSNR on tiny near-saturated regions by up to 20 dB on C1).
  DISPOSITION: all banked CIs are within-run paired (valid); cross-run absolute bit-identity is not a
  meaningful criterion; sidecar acceptance correctly uses same-process equality. Documented here and in
  the protocol note.
- Pot-deletion residue (clutter-embedded contact regions leave visually poor debris in the BASE
  representation; 6 pre-metric geometric iterations logged; `--cleanup-expand` pixel_count==0 rule
  implemented but debris is low-visibility-not-zero — hypothesis falsified and kept). DISPOSITION:
  banked limitation figure; method metrics unaffected (debris is shared by C1 and all methods).
- leak_R conflation (penalizes improvement). DISPOSITION: oracle-primary policy; leak retained as a
  secondary bounded-deviation check.

## Strongest remaining reviewer attacks + prepared responses

1. "Oracle is synthetic-only." — True and inherent; the oracle validates the MECHANISM ranking and the
   proxy's direction; real-scene cells use true-GT U-preservation + bounded ghost-similarity + panels.
2. "Stale cache ties you on deletion." — Kept in the paper: the tie is an accident of depth-inconsistency
   (deletion only); recolor and chained edits break it by +1.96/+2.67; and stale caches provide no
   provenance for composition or sidecar storage.
3. "Only deletion + recolor." — By design with evidence-backed boundaries (translation tearing/shadow
   figure); NON-CLAIMS state it verbatim.
4. "GaussianEditor does this trivially." — A13: representation edits are our C1 control; the
   contribution is evidence-cache consistency, a problem that only exists because the cache buys +1.67 dB.

## Recommended claim wording (synced into CLAIMS v1.0)

CR5 final: deletion + recolor edits, evidence invalidated by exact mesh-face provenance; oracle-verified
in-region superiority over rebuild/stale/dilated/2D-box/target-side alternatives (CIs excl. 0);
true-GT unaffected-region preservation ≤0.02 dB; region-proportional updates (57/161 views for a
peripheral object; 12.8 MB sparse sidecar, 18×/~100× below dense/full rebuild); chained edits supported
(parent-mask inheritance); translation/deformation excluded with banked failure evidence.

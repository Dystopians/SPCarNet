# ROUTE-A TOPCONF EXECUTION PLAN — frozen specs

Opened 2026-07-12. Each experiment: hypothesis, frozen protocol, acceptance criterion, collector,
artifact path, paired uncertainty. No open-ended sweeps. Owners: me = protocols, provenance/invalidation
code, review, interpretation; Codex = generators, launchers, mask transforms, collectors, plots, doc
plumbing (every diff inspected).

## EXP-ORACLE (closes P0-A with EXP-ABL)

- **Hypothesis:** (H1) leak_R rank-correlates with TRUE in-region error (Spearman ρ ≥ 0.7 over
  methods×views), so the real-scene proxy is valid; (H2) C5 in-region TRUE-GT quality ≥ every naive
  strategy (paired CIs), because provenance masking retains photographed disoccluded background.
- **Protocol (frozen):** `build_toy_parking.py --drop-elements car_0` (RNG stream untouched — faces
  filtered after mesh assembly, before render/export), seed 0, out `datasets/toy_parking_nocar0`.
  Acceptance gates BEFORE use: (a) `sparse/0/images.txt` and split file byte-identical to the banked
  original build; (b) element census lacks car_0, all other counts unchanged; (c) renders of the 3
  test views with zero car_0 coverage (from the banked coverage census) pixel-identical (max |Δ| = 0).
  Oracle metrics: per test view, PSNR/SSIM in R (the frozen 8-px edit region) between each method's
  output and the ORACLE GT image; U-metrics unchanged (real GT already valid there).
- **Collector:** `tools/edit/oracle_eval.py` → `analysis/edit_aware/oracle_toy/{oracle_eval.{md,json}}`
  (method table + leak_R↔oracle-error correlation + paired CIs).
- Owner: Codex writes the generator flag + verification script (no GPU in its sandbox — I run the GPU
  build); I write oracle_eval.py (claim-bearing metric code).

## EXP-ABL (closes P0-B)

- **Strategies (7, frozen):** ours = provenance 1-px; provenance dilate-4; dilate-16; 2D bbox of the
  provenance mask per source view; TARGET-mask (zero transport signal inside R at compose — needs the
  original checkpoint at render time; implemented in the eval harness, not the cache); VIEW-DROP
  (manifest without affected views); C1 (global disable). Same frozen transport/net everywhere.
- **Hypothesis:** ours ≥ all on in-region ORACLE quality; ours ≥ box/dilate-16/view-drop on U
  preservation (they over-delete valid evidence); target-mask ties in R for deletion but loses
  disoccluded-background quality in R and adds per-render ID-pass cost (reported).
- **Protocol:** toy car_0 deletion (oracle-scored) + garden table deletion (proxy-scored, correlation
  from EXP-ORACLE justifies). Mask variants generated from the banked C5 masks by pure transforms
  (Codex tool + tests); caches share bytes via hardlinks. Metrics + paired CIs per the edit protocol.
- **Acceptance:** claims written to MATCH the outcome (pre-registered weakenings in the audit §3).
- **Collector:** `tools/edit/abl_report.py` → `analysis/edit_aware/ablations/abl_table.{md,json}`.

## EXP-BREADTH (closes P1-C)

- Cells: toy car_1 deletion (GT-box; oracle build `--drop-elements car_1` optional if cheap — else
  proxy), garden POT deletion (peripheral small object — locality measurement), kitchen recolor
  (indoor; object = a counter item selected by cylinder, one pre-metric visual iteration allowed,
  logged). Same frozen pipeline (C1/C2/C5; C4 only for the pot — cheap scene? No: C4 skipped for
  breadth cells, the failure mode is established; documented).
- **Acceptance:** C5 leak ≤ 2× its garden/toy values on every new cell; preservation CI within
  ±0.05 dB; ANY violation reported as a boundary finding, not hidden.

## EXP-SIDECAR (closes P1-D)

- **Design (mine):** `build_edited_cache.py --sparse`: per affected view store
  `patches/<name>.npz {bbox, render_patch(uint8 png bytes), depth_patch(f32)}` instead of full
  renders/depths; EcrRenderer reconstructs in memory at load (original hardlink + patch compose),
  default-off, read-confined, patch files manifest-listed; bit-stability required when absent.
- **Hypothesis:** bytes rewritten become region-proportional (garden ≤ ~15% of the current 1053 MB;
  peripheral pot ≪ that), outputs BIT-IDENTICAL to the dense edited cache.
- **Acceptance:** per-view outputs identical (max|Δ|=0) dense-vs-sparse; bytes + wall-clock banked;
  locality wording in all docs corrected to measured numbers.

## EXP-BOUNDARY (closes P1-E) & EXP-CHAIN (closes P1-F)

- BOUNDARY: `delete→translate car_0 by (0, +2.5m)` checkpoint edit (vertex translation of selected
  faces); render edited BASE views; bank the wrong-shadow/old-shadow figure + a one-paragraph analysis.
  NO cache/transport work (the boundary is in the representation's baked shading, shown at C1 level).
- CHAIN: fix `build_edited_cache.py` to inherit parent masks (AND/union with new masks — bugfix);
  compose garden deletion→recolor(pot? no—recolor of a REMAINING object, the black pot) as cache2 built
  on cache1; eval leak/preservation; acceptance = metrics within 1.5× single-edit values.

## EXP-RELWORK + COHERENCE (closes P1-G)

- Related-work matrix (editable IBR, source-masking IBR, neural mesh editing, Gaussian editing) with
  the one-line honest position each; the unified-mechanism paragraph ("one gate, three jobs") written
  into CLAIMS positioning + paper plan; REBUTTAL entry A13 ("why is editing hard if GaussianEditor
  exists") drafted from the matrix.

## Sequencing

1. Codex: generator flag + verification + mask-variant tool (parallel) | me: oracle_eval.py + harness
   target-mask mode + sidecar design.
2. GPU: oracle dataset build → EXP-ORACLE/ABL runs (toy first, garden ablations after).
3. EXP-BREADTH cells | EXP-SIDECAR | EXP-BOUNDARY | EXP-CHAIN.
4. Docs/claims sync at every bank; readiness report finalized when register is empty.

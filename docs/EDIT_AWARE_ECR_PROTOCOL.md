# EDIT-AWARE ECR — FROZEN EVALUATION PROTOCOL (v1, 2026-07-12)

Frozen BEFORE any prototype number exists. Edited scenes have no ground truth, so this cell is a
documented outside-the-mouth measurement harness (`tools/edit/edit_eval.py`) — EXCEPT the
unaffected-region metrics, which are computed against REAL GT (valid there) with the frozen edit-region
mask excluded. No test-GT enters any render path (masks and edits are derived from checkpoints and
train views only; test-view edit-region masks are derived from POSES + the original checkpoint, never
from test images). Banked Stage-4/5 rows untouched.

## Scenes and edits (frozen)

- **E1 (controlled, synthetic):** toy_parking — delete ONE parked car (3D box in meters from the
  dataset's own frame; box recorded in the edit spec json). Synthetic scene, metric units, ring
  trajectory → moderate view-locality.
- **E2 (real):** garden — delete the central table+vase assembly (3D box around it; box recorded).
  Iconic object; nearly all views affected → the locality metric is expected to be WEAK here and is
  reported as such (locality is E1's job).
- Face set = triangles whose centroid ∈ box on the ORIGINAL checkpoint; the same box defines the
  edit-region mask on test views via the original checkpoint's rend_ids.

## Methods compared (all on the SAME edited checkpoint unless stated)

- **C1 edited-base:** edited checkpoint, base renderer (no ECR).
- **C2 stale-ECR:** edited checkpoint + the ORIGINAL unmodified cache (fingerprint guard bypassed ONLY
  inside this harness, documented — this is the failure mode Route A exists to fix).
- **C3 global-disable:** ≡ C1 (listed separately to mirror the mission's comparison list; no extra run).
- **C4 full-rebuild-no-mask:** `build_cache` run from scratch on the edited checkpoint (all renders/
  depths regenerated, α recalibrated by the standard builder, GT photographs UNMASKED). Hypothesis: does
  NOT fix ghosting (the photographs still contain the object).
- **C5 ours (local invalidation):** edited cache = affected-view renders/depths regenerated + per-view
  stale-pixel masks (rend_ids ∈ deleted, 1-px dilation) honored at the single warp site; unaffected view
  files hardlinked; original (K, α) and fusion net REUSED (frozen decision — isolates the invalidation
  mechanism; the net consumes content planes, not scene identity).

## Metrics (per test view; means over views; automatic collector)

Let R = edit-region mask (test-view pixels whose ORIGINAL-checkpoint rend_ids ∈ deleted set, dilated
8 px for context), U = complement of the 16-px-dilated region.

1. **Ghosting / stale-evidence leakage (primary):**
   `ghost(m) = PSNR_R(I_m, I_origECR)` — similarity of method m's edit region to the ORIGINAL unedited
   ECR output (which shows the object). HIGHER = MORE ghost. Success criterion (frozen): C5's ghost is
   statistically indistinguishable from C1's (paired per-view CI of the difference includes 0 or C5 is
   lower), while C2's exceeds C1's by a CI excluding 0 (the problem is real) — and reported for C4.
   Secondary form: `leak(m) = mean_R |I_m − I_C1|` (deviation from the edited base in-region; the
   no-hallucination policy means the honest target IS the edited base there).
2. **Unaffected-region preservation (TRUE-GT):** PSNR/SSIM/LPIPS over U vs the REAL test GT, for C5 vs
   the banked original ECR output. Success: per-view paired ΔPSNR_U CI includes 0 (edit is local).
3. **Cost/locality:** bytes rewritten and wall-clock for C5 vs C4; affected-view counts.
4. **Temporal stability after edit:** `ecr_temporal.py` on the C5 cache (E2), ratio bar ≤ 1.5 as before.
5. Panels: per-view strips {C1, C2, C4, C5, original-ECR} full frame + R-crop for the failure/success
   figure; no hand-picked-only reporting — every test view enters the means.

## Statistics

Paired per-view bootstrap (10k, seed 0) on the per-view metric arrays, exactly the banked machinery.
All numbers emitted by `tools/edit/edit_eval.py` into `analysis/edit_aware/<scene>/edit_eval.{md,json}`
— nothing hand-typed.

## Purity / audit statement

The C5 cache passes the standard `--ecr` audit checks with ONE amendment: the manifest carries an
`edit` provenance block (parent fingerprint, box, deleted-face count, affected views, mask list) and its
checkpoint fingerprint is the EDITED checkpoint's (cache built from it — the fingerprint match holds by
construction). Masks only REMOVE evidence (multiplicative ≤1 on confidence) — they cannot inject
information; the audit wall (read confinement, pose-primitives boundary, GT sentinel, frozen kwargs)
is unchanged. `ablate`-style default-off law: with no masks present, the transport is bit-identical
(config-hash check required before any run).

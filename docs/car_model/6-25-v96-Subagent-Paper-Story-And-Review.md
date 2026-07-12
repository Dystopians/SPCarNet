# v96 Subagent Paper Story and Review Draft

Date: 2026-06-25

Scope: review + paper-story synthesis for the current `docs/car_model` state. This draft only reads and synthesizes existing local docs/artifacts. It does not promote unfinished representation-level work.

Primary source docs:

- `docs/car_model/README.md`
- `docs/car_model/6-25-v95-SPCarNet-Technical-Report-for-Mentor.md`
- `docs/car_model/6-25-v94-TargetGridRuntimeOptimization-Log.md`
- `docs/car_model/6-25-v95-Rejected-And-v96-CheckpointBaked-Launch.md`
- supporting context from `docs/car_model/6-25-OfficialProtocol-Refresh-And-PaperLoop-Gap.md` and `docs/car_model/6-25-SPCarNet-Mentor-PPT-Final-Technical-Report-v92.zh.md`

## 0. Claim Boundary First

Current paper-safe endpoint:

```text
SPCarNet Phase-J guarded adaptive Evidence Lumigraph Adapter
+ geometry-safe compact checkpoint
```

Current paper-safe claim:

> SPCarNet is an evidence-certified post-training repair and compaction layer for MeshSplatting. It uses train/policy-val surface evidence to decide where a trained checkpoint can remove low-risk triangles, where stable residuals can be transferred through mesh surface correspondence, and where the system should fall back to the clean checkpoint.

Do not claim:

- Do not claim the method has "fully surpassed MeshSplatting" in every axis.
- Do not claim deployment speedup. Current integrated Phase-J is speed-negative.
- Do not claim a promoted checkpoint-baked representation endpoint. v96 is still running/under validation.
- Do not put v95 in the main result as a success. v95 is completed and rejected.
- Do not treat train/policy-val gate numbers as held-out test metrics.

Status table:

| Item | Current status | Paper interpretation |
|---|---|---|
| Phase-J guarded adaptive ELA + compaction | closed full9 evidence | current headline endpoint |
| v94 target-grid runtime cleanup | kept, exact, small improvement | implementation cleanup, not speed solution |
| v95 region-texture representation candidate | completed, rejected | negative diagnostic only |
| v96 checkpoint-baked certified recovery | running / being repaired and validated | active next attempt, not a result |

## 1. Method Story

### 1.1 Core Problem

MeshSplatting gives an explicit surface/triangle-aware representation, but after the checkpoint is trained the standard pipeline does not ask:

- which triangles are low-risk enough to remove;
- which surface regions have stable, multi-view appearance residuals that can repair held-out renders;
- which proposed repairs only look good on train/policy-val views but are unsafe on held-out views;
- which regions lack evidence and should remain unchanged.

SPCarNet turns this into the main research question:

```text
Can train/policy-val surface evidence certify where a trained MeshSplatting
checkpoint can be compacted and where its appearance residuals can be safely repaired?
```

### 1.2 Current Narrative

The strongest current story is not "we replace MeshSplatting training." It is:

```text
trained MeshSplatting checkpoint
  -> render train/policy-val views with surface evidence
  -> build evidence cache over residual, visibility, face/bin support, risk
  -> remove low-risk triangles
  -> transfer stable surface-bound residuals through guarded ELA
  -> policy-val gate chooses branch/alpha/fallback
  -> evaluate held-out views
```

One-sentence paper positioning:

> SPCarNet adds a post-training self-audit layer to MeshSplatting: the trained surface is used not only for rendering, but also as an address space for certified compaction, guarded residual repair, and fallback.

More precise abstract-style version:

> Starting from a trained MeshSplatting checkpoint, SPCarNet builds a train/policy-val surface evidence cache containing residuals, visibility, support, and risk statistics. It uses this evidence to remove low-risk triangles and to transfer stable residuals through mesh-surface correspondence with policy-val gates and fallback. On local Mip-NeRF360 full9 under the same evaluator and selected clean MeshSplatting baseline, the current Phase-J endpoint improves held-out RGB metrics while removing a moderate fraction of triangles. The current limitation is that the strongest RGB gain is still delivered by a render-time adapter; checkpoint-baked recovery remains under validation.

### 1.3 Method Modules To Present

| Module | Story role | Evidence status |
|---|---|---|
| Surface evidence cache | makes train/policy-val views auditable after training | used by current Phase-J and policy gates |
| Geometry-safe compaction | quality-first triangle removal, not aggressive compression | full9 triangle reduction with RGB wins |
| Guarded Evidence Lumigraph Adapter | transfers stable residuals through surface correspondence | current largest RGB gain |
| Policy-val gate and fallback | prevents risky edits from being forced | branch/alpha/fallback selected before held-out evaluation |
| v94 target-grid reuse | exact implementation cleanup for adapter runtime | kept, small runtime gain |
| v96 checkpoint-baked repair | attempt to bake ELA-style recovery into checkpoint | running; no claim yet |

## 2. Difference From MeshSplatting

| Dimension | MeshSplatting baseline | SPCarNet current endpoint |
|---|---|---|
| Starting point | trained MeshSplatting checkpoint | same trained checkpoint as base |
| Use of train views | optimization signal during training | persistent surface evidence after training |
| Geometry | render checkpoint as-is | delete low-risk triangles under evidence gates |
| Appearance repair | direct checkpoint render | guarded surface residual transfer via ELA |
| Safety policy | mostly relies on training convergence | policy-val gate, tail-risk checks, fallback/no-op |
| Held-out use | final evaluation | final evaluation only; not used for policy selection |
| Output claim | high-quality render | high-quality render + moderate compactness + audit trail |
| Runtime status | normal render path | current Phase-J adapter is much slower |
| Representation status | baked checkpoint | strongest current SPCarNet RGB result is render-time adapter, not yet baked |

Important wording:

- "SPCarNet is a post-training evidence-certified repair/compaction layer for MeshSplatting."
- "It builds on MeshSplatting rather than replacing the base optimizer."
- "It improves held-out RGB quality and compactness under the local same-protocol full9 audit."
- "Paper-table comparisons are contextual; the main claim uses the stronger local selected clean baseline."

What makes ELA more than a generic image postprocess:

- residuals are indexed by mesh surface evidence, not only image coordinates;
- support comes from train/policy-val views through face/bin/barycentric correspondence;
- branch/alpha/fallback are selected before held-out GT metrics;
- uncertain regions can remain clean rather than being forced through a repair.

## 3. Existing Evidence

### 3.1 Main Same-Protocol Full9 Evidence

Protocol summarized from the mentor report:

- dataset: local Mip-NeRF360 full9;
- evaluator/split: same local evaluator and split;
- baseline: selected clean MeshSplatting envelope over clean `26000/30000`, chosen by held-out score;
- method: SPCarNet Phase-J guarded adaptive ELA plus compact checkpoint.

Aggregate result:

| Metric | Clean MeshSplatting | SPCarNet Phase-J | Delta / status |
|---|---:|---:|---:|
| Mean PSNR | `25.1517` | `26.4828` | `+1.331084` |
| Mean SSIM | `0.7490` | `0.7837` | `+0.034702` |
| Mean LPIPS | `0.2876` | `0.2243` | `-0.063359` |
| Scene-level strict RGB wins | n/a | n/a | `9 / 9` |
| Held-out view strict RGB wins | n/a | n/a | `244 / 246` |
| Mean triangle reduction | `0%` | `7.6479%` removed | moderate quality-preserving compaction |

Evidence paths:

```text
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean_report.md
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.json
outputs/carnet/meshsplatopt/paper_m360_repro/phasej_guarded_adaptedge_official_refresh_20260625_v94/compact_ela_vs_clean.csv
```

Interpretation:

- This supports the strongest current quality + compactness headline.
- It does not prove deployment speed.
- It does not prove the repair has been baked into a standard checkpoint.

### 3.2 MeshSplatting Paper-Table Bridge

The local official clean30k reproduction is close to the MeshSplatting paper table:

| Method / protocol | PSNR | SSIM | LPIPS | Role |
|---|---:|---:|---:|---|
| MeshSplatting paper table | `24.78` | `0.728` | `0.310` | external reference |
| Local official clean30k reproduction | `24.8002` | `0.7310` | `0.3072` | protocol sanity check |
| Local selected clean MeshSplatting | `25.1517` | `0.7490` | `0.2876` | fair stronger local baseline |
| SPCarNet Phase-J | `26.4828` | `0.7837` | `0.2243` | current strongest endpoint |

Safe interpretation:

- local clean30k is close enough to make the local evaluator credible;
- the selected clean baseline is stricter than clean30k and should be the main comparison;
- the paper-table delta is useful context, not the primary official claim.

### 3.3 Official-Style Compact-ELA Support Evidence

The official-style Compact-ELA support table is weaker than the Phase-J headline but useful because it gives a stricter paper-protocol bridge:

| Item | Value |
|---|---:|
| Available scenes | `9 / 9` |
| Strict all-axis pass | `5 / 9` |
| RGB + compact + geometry-safe pass | `9 / 9` |
| Mean dPSNR vs selected clean | `+0.497941` |
| Mean dSSIM vs selected clean | `+0.015755` |
| Mean dLPIPS vs selected clean | `-0.023373` |
| Mean triangle reduction | `5.7632%` |

Interpretation:

- Use this as supporting evidence for compactness and protocol sanity.
- Do not confuse it with the stronger Phase-J result.
- Do not claim all-axis dominance unless the table actually supports it.

### 3.4 Qualitative Evidence

Available qualitative assets:

```text
assets/spcarnet_phasej_where_it_helps_showcase_20260622.png
assets/spcarnet_m360_full9_qualitative_gallery.png
assets/spcarnet_m360_outdoor_detail_showcase.png
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.md
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.json
outputs/carnet/spcarnet/qualitative_traceability_20260625/manifest.csv
```

Current qualitative story:

- local crop/error-reduction panels show where ELA helps;
- full-frame gallery is useful for fairness but may look subtle;
- outdoor detail panel helps when full-frame differences are hard to see;
- traceability manifest exists and should be used to avoid cherry-pick concerns.

TODO:

- Verify final paper figure captions against exact scene/view/crop/source paths before submission.
- If v96 produces a real checkpoint-baked result, regenerate or add a separate baked-result qualitative panel instead of reusing Phase-J-only figures.

### 3.5 Runtime and Rate Evidence

v94 target-grid-only runtime cleanup:

| Runtime item | Old integrated v2 | v94 target-grid-only | Delta |
|---|---:|---:|---:|
| Weighted integrated ms/view | `951.410896` | `944.945199` | `-0.68%` |
| Weighted adapter ms/view | `913.855245` | `907.552261` | `-0.69%` |
| Integrated/render-only compact ratio | `27.044247x` | `26.860457x` | slightly better, still bad |
| Max alloc | `17703.596 MiB` | `17701.383 MiB` | essentially unchanged |

v94 interpretation:

- exact code cleanup;
- leaves policy, support, residual transfer, gates, alpha maps, renders, and full-resolution metrics unchanged;
- rejects fused/cache and batch-warp variants because they were slower and/or heavier;
- does not solve deployment speed.

Rate/frontier status:

- Phase-J reduces triangles by `7.6479%` on average;
- checkpoint bytes and CUDA peak memory improve in the static/render-only evidence;
- render-only FPS is not improved;
- integrated render+adapter is speed-negative because adapter dominates runtime.

Safe runtime claim:

```text
SPCarNet currently supports quality, compactness, memory, and checkpoint-size discussion.
It does not support an FPS speedup or deployment-speed claim.
```

### 3.6 Representation-Level Negative Evidence

Current representation-level line has not produced a promoted endpoint:

| Run | Status | Meaning |
|---|---|---|
| v87 source mixture | finished, not promoted | accepted edit but below v84/v86 anchor |
| v88 anchor-dominance tail-risk | finished, not promoted | PSNR/LPIPS signal, SSIM regression blocks promotion |
| v89b L1-proxy bin-dominance | finished, not promoted | tiny PSNR signal, LPIPS gate fails |
| v90 adaptive source mixture | no promotion evidence | do not use as headline |
| v91 residual-debt support | interrupted / no valid held-out result | do not use as headline |
| v95 region-texture candidate | completed, rejected | negative diagnostic |

v95 explicit result:

| Field | Value |
|---|---:|
| v84/v86 anchor floor | `>26.7561378479 / >0.8621263504 / <0.2516906559` |
| v95 held-out counter | `26.7500514984 / 0.8620513678 / 0.2519962788` |
| accepted atlas | `true` |
| selected alpha | `0.03125` |
| target changed fraction | `0.0184769104` |
| promotion verdict | rejected on PSNR, SSIM, LPIPS and risk-gain floors |

Interpretation:

- v95 proves the gate and audit machinery can run through a real counter probe.
- It does not beat the pre-declared anchor and must not be expanded to hard-triad/full9 as a success.

### 3.7 v96 Current Evidence

v96 changes the method form:

```text
compact checkpoint 26000
  -> train-only Phase-J/ELA teacher render loss
  -> parent render rollback
  -> checkpoint render depth/normal anchors
  -> train sparse-depth sentinel cache
  -> topology-frozen checkpoint 30000
```

Current confirmed status:

| Item | Status |
|---|---|
| New runner | `scripts/car_model/run_v96_checkpoint_baked_certified_repair_scene.py` |
| Dry-run validation | succeeded |
| Sentinel cache | built: `24` train views, `12000` sentinels, no test leakage |
| Live counter probe | launched under `/dev/shm/peilincai_spcarnet_v96_checkpoint_baked_20260625` |
| Promotion status | not promoted; running / validating |

Current v96 claim:

```text
v96 is an active checkpoint-baked repair attempt.
It is not yet a paper result.
```

TODO:

- Fill in v96 final counter training status.
- Fill in held-out PSNR/SSIM/LPIPS.
- Fill in geometry metrics and sparse-depth sentinel/rollback audit.
- Fill in pass/fail versus v84/v86 counter anchor.
- Only if counter passes, run hard-triad and full9.

## 4. Unfinished Evidence

The most important missing pieces are:

| Missing evidence | Why it matters | Required before claim |
|---|---|---|
| v96 final counter held-out metrics | first gate for checkpoint-baked repair | beat v84/v86 anchor on PSNR, SSIM, LPIPS |
| v96 geometry evaluation | prevents RGB-only baked repair from damaging geometry | depth/normal/sparse-depth non-regression or declared boundary |
| v96 hard-triad expansion | tests whether counter result generalizes | pass counter/kitchen/bonsai or explicitly report limitation |
| v96 full9 expansion | required for paper headline if baked endpoint is claimed | full9 same-protocol table |
| v96 runtime/profile | baked repair should remove render-time adapter bottleneck | render-only and integrated profile against clean/compact/Phase-J |
| Adapter acceleration alternative | needed if v96 fails or is too weak | materially reduce adapter/render gap, not micro-optimization only |
| Ablation set | reviewers will ask what component matters | no evidence cache, no gate, no fallback, no compaction, no ELA/residual transfer |
| Geometry/all-axis table clarity | current ELA collector has geometry `nan` for ELA layer | separate RGB+compact claim from all-axis geometry claim |
| Final artifact manifest | avoids version sprawl and stale metrics | single manifest mapping claims to exact artifacts |

## 5. Review Findings and Weak Points

### 5.1 Main Strength

The current story is coherent if framed as post-training evidence-certified repair and compaction. The full9 Phase-J evidence is strong under the local same-protocol selected clean MeshSplatting baseline, and the method has a defensible reason to be surface-aware rather than a generic 2D postprocess.

### 5.2 Main Weaknesses

1. Render-time adapter bottleneck.
   v94 improves integrated runtime by only about `0.68%`; Phase-J remains about `26.86x` slower than compact render-only. This blocks deployment-speed claims.

2. Representation-baked endpoint not closed.
   v87/v88/v89b/v95 are not promoted, and v96 is still running/validating. The current strongest RGB endpoint is therefore an adapter endpoint.

3. v95 is an explicit negative result.
   It accepted an atlas and changed target pixels, but missed all three anchor metrics. It should be used as evidence of honest gating, not as progress toward a positive result.

4. Triangle reduction is moderate.
   `7.6479%` is useful for quality-preserving compactness, but not enough for an aggressive compression paper by itself.

5. Geometry wording needs care.
   The Phase-J ELA collector supports RGB quality plus compactness; geometry columns for the ELA layer are `nan` in the v94 paper-protocol bridge. Do not write "all-axis geometry win" unless a separate table supports it.

6. Paper-table comparison is contextual.
   The local clean30k reproduction is close to the MeshSplatting paper table, but official details can differ. The main comparison should remain the local selected clean baseline.

7. Version sprawl is a reviewer risk.
   Many branches are negative diagnostics. The paper needs a small number of named modules and a single claim/artifact manifest, not a version-number narrative.

8. Qualitative full-frame differences can be subtle.
   Use crop/error-map panels for impact and full-frame panels for fairness. Avoid over-selling visual examples without traceability.

## 6. Next Experiments

### P0: Finish v96 Counter Gate

Goal:

```text
Determine whether checkpoint-baked certified repair can beat the v84/v86 counter anchor.
```

Required outputs:

- final training/render status;
- held-out counter PSNR/SSIM/LPIPS;
- geometry evaluation;
- sparse-depth sentinel/parent rollback audit;
- promotion-gate verdict against:

```text
PSNR > 26.7561378479
SSIM > 0.8621263504
LPIPS < 0.2516906559
```

Decision:

- if v96 fails: archive as negative and do not expand;
- if v96 passes RGB but fails geometry: report as incomplete and diagnose geometry rollback;
- if v96 passes RGB and geometry: expand to hard-triad.

### P1: Hard-Triad Then Full9 Only After Counter Pass

Sequence:

1. Counter gate.
2. Hard-triad: `counter,kitchen,bonsai`.
3. Full9 same-protocol selected clean comparison.
4. Full9 qualitative and artifact manifest.

Reason:

- Previous representation branches show tiny or scene-specific gains. Full9 should not be launched until the counter anchor is actually beaten.

### P1: Baked Runtime Profile

If v96 produces a valid checkpoint:

- run render-only benchmark against clean and compact Phase-J checkpoint;
- run any integrated profile needed to show the adapter is no longer required;
- report FPS, ms/view, peak memory, checkpoint bytes, triangles;
- compare against v94 Phase-J integrated `944.945199 ms/view` and adapter `907.552261 ms/view`.

Success criterion:

- at minimum, do not add the Phase-J adapter runtime bottleneck;
- for deployment claims, show real FPS improvement or explicitly keep runtime out of the headline.

### P1: Minimal Reviewer Ablations

Needed ablations, if not already cleanly materialized:

- compact checkpoint only vs clean;
- ELA without compaction;
- compaction + ELA without policy-val gate;
- no fallback or forced repair;
- no surface correspondence / image-space residual control, if feasible;
- v94 runtime cleanup on/off for exactness/runtime only;
- v96 teacher loss without geometry/sparse-depth rollback, if v96 becomes central.

TODO:

- Map each ablation to existing artifacts if already present; otherwise mark as missing.

### P2: Rate-Distortion Extension

Current triangle reduction is moderate. If the paper needs a stronger compression angle:

- add 1-2 more compaction targets;
- keep same selected-clean baseline and held-out evaluator;
- require RGB non-regression or explicitly present a rate-distortion frontier;
- avoid weakening the current strongest quality claim.

### P2: Final Claim Manifest

Create one manifest before paper writing:

```text
claim -> table/figure -> source artifact -> command/protocol -> status
```

This should separate:

- main Phase-J adapter result;
- Compact-ELA support result;
- MeshSplatting paper-table bridge;
- runtime/rate evidence;
- negative representation diagnostics;
- v96 final status.

## 7. Suggested Paper Claim Wording

Safe title direction:

```text
SPCarNet: Evidence-Certified Repair and Compaction for MeshSplatting
```

Safe main claim:

> On local Mip-NeRF360 full9 with the same evaluator and a selected clean MeshSplatting baseline, SPCarNet's current Phase-J endpoint improves PSNR, SSIM, and LPIPS on all 9 scenes and 244/246 held-out views while removing 7.65% triangles on average.

Safe limitation sentence:

> The strongest current RGB improvement comes from a render-time guarded adapter; checkpoint-baked recovery is under validation and current evidence does not support deployment-speed claims.

Unsafe wording to avoid:

- "fully surpasses MeshSplatting";
- "faster than MeshSplatting";
- "baked representation solved";
- "all-axis geometry win" for the Phase-J ELA table;
- "v95 improves representation" without the rejected status.

## 8. Current Draft Verdict

The paper story is viable as a systems/reconstruction-method story centered on evidence-certified post-training repair and quality-preserving compaction. The Phase-J evidence is strong enough for a careful internal/mentor narrative and possibly a paper direction if the claim boundary stays honest.

The major blocker for a stronger top-conference story is that the current best result is still a render-time adapter. v96 is the right next attempt because it attacks the representation-baking weakness directly, but it must finish training, held-out metrics, geometry evaluation, and expansion gates before it can be used as a main result.

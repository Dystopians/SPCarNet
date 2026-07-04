# GEMS — STAGE 1R INSERT DIRECTIVE
## Anchor Integrity · Geometry Reopening (Opacity Release) · Downstream Closure · Stage-2 Amendments

Version 1.0 · 2026-07-03 · **This directive AMENDS — does not replace — `docs/GEMS_Stage1_Prompt.md` and `docs/GEMS_Stage2_Prompt.md`.** It constitutes the explicit HUMAN SIGN-OFF that the Stage-Two scope-freeze requires for the specific, bounded reopenings defined below, and nothing else. All Prime Directives D1–D9, effect-size floors, sunset rules, pre-registration, CI discipline, the single mouth (`run_eval.py`), and the /goal output template remain in force verbatim. Log all work from this directive in `LEDGER.md` as `GOAL #R-xx`, alongside (not replacing) the Stage-2 `GOAL #NNN` sequence.

Before your first R-goal, re-read: `STAGE1_REPORT.md` (esp. §4, §7b, §8), `KILL_REPORT.md`, `LEDGER.md` GOALs #005–#009, `CLAIMS.md`, `MATRIX.md`.

---

## 0. WHY THIS INSERT EXISTS (context — internalize, do not re-litigate)

Stage One was executed with discipline and its negative verdicts STAND. This insert does not relitigate E1/E1′, the two falsified E2 loss routes, the E2v3 load-bearing falsification, the E3 sunset, or the 26k-sourcing refutation. It exists because Stage One's own evidence exposes four gaps between the current frozen claim set and the final objective (a top-conference paper with a real parking/low-speed-driving downstream result):

- **G1 — The claim anchor is decayed.** `clean@30k` sits below its own `@26k` (garden +0.32 dB) due to the measured end-phase position-drift pathology. Every "vs clean" comparison — including the C1 headline "garden B50 BETTER than clean" — is currently subsidized by the baseline's self-inflicted damage. Full9 evidence is being banked against this anchor right now. A reviewer re-anchors in one sentence; we must re-anchor first. (Fact: iteration-26000 AND 30000 checkpoints exist for all 9 scenes on disk.)
- **G2 — The geometry axis was demoted with one representation-level unlock never tested.** The opacity floor (`triangle_model.py` pins render opacity to [0.999, 1] for every triangle) was treated as a structural given. It is hereby declared a MODIFIABLE REPRESENTATION PARAMETER, not protocol. It is exactly what blocked the transmittance/fade route AND what forced the delete-vs-keep dilemma that the load-bearing falsification exposed.
- **G3 — The downstream story is hollow for the stated application.** C4 currently claims *preservation* of models whose d1 false-free rates are 59–65%. For a parking objective this is a measurement contribution, not an application. SS3DM is absent (D-1 blocked); the on-disk, domain-perfect `parking_phone_tiny_anonymized` capture is unused; no occupancy-extraction or planner result exists.
- **G4 — E1′ reshapes the paper's center of gravity.** Under safe FT at moderate budgets on small scenes, evidence-importance ≈ random (+0.13, CI incl. 0). Importance is decisive at aggressive budgets / no-FT (random collapses 2.4–5.3 dB). The compaction claim must live where the mechanism actually matters.

Execution order: **R0 (sequencing, immediate) → R1 (anchor) → R2 ∥ R3 → R4 (integration).** Stage-2 breadth continues in parallel only as R0 specifies.

---

## R0 — SEQUENCING RULES (apply immediately, before any new launch)

1. In-flight S-REND B25 / B12.5 chains: let them FINISH. GEMS method rows are anchor-independent; nothing already computed is wasted.
2. Do NOT launch new comparative rows whose interpretation depends on the anchor (H1 v106 row, R1 3DGS-reference row, S-GEN comparative claims, any new "vs clean" claim statements) until R1 delivers a frozen anchor verdict. Training new B0 baselines (T&T, toy variants) is allowed — training is anchor-independent.
3. Launch **D-1 (SS3DM acquisition)** NOW as a background task (download/verify is I/O, not GPU). If unobtainable after a genuine attempt (document sources tried), file the INFEASIBLE note per MATRIX rules and activate the fallback suite {toy variants ×2 (D-2), courtyard, parking_phone_tiny qualitative} with an honest scope statement staged for CLAIMS.
4. Open new iteration-budget lines in `LEDGER.md` for mechanisms R1 / R2 / R3 (same ≤2 tuning-flavored rule, same sunset rule).
5. Storage discipline unchanged: `/data` remains near quota — preflight every launch; GB-scale outputs go to the DEC-007 root.

---

## R1 — ANCHOR INTEGRITY (Tier-1-blocking; cheap; do first)

**Purpose.** Make every "vs clean" statement honest against the strongest fair baseline, and convert the end-phase-decline discovery from a hidden subsidy into a reported baseline property (consistent with the existing NON-CLAIM: the finding is a property of the baseline trainer, not a GEMS contribution).

- **R1.a — Anchor rows (evaluation only, no training):** run `clean@26000` for all 9 mipnerf360 scenes + toy + courtyard through the single mouth. (~11 evals; hours.) Durable paths, standard template.
- **R1.b — Freeze ONE anchor policy** (pre-register before looking at more than the garden numbers already known; D4-legal — no test-view influence; one rule for ALL scenes, D7):
  - **Option (i) fixed-rule checkpoint:** anchor = the topology-freeze checkpoint (iteration 26000, where densification stops; census confirms 26000 ≡ 30000 topology). Zero training cost.
  - **Option (ii) trainer end-phase fix (preferred if wall-clock permits):** from each scene's 26000 checkpoint, re-run the final 4k iterations with positions+weights frozen (the validated safe channel) → `clean-fixed@30k`. ≈ 9 × ≤10 min. This upgrades the discovery into a one-line, paper-reportable trainer recommendation and yields a single canonical anchor.
- **R1-T existence test (both outcomes acceptable — this closes either way):** the chosen anchor ≥ clean@30k on ≥7/9 S-REND scenes with CI excl. 0 → adopt as PRIMARY anchor. Otherwise → the end-phase finding is scene-limited; clean@30k stands; report honestly. ≤2 policy variants total; per-scene anchor choice is FORBIDDEN.
- **R1.c — Dual-row re-expression:** all comparative tables carry `vs clean30k` (legacy, continuity) AND `vs PRIMARY anchor`. `CLAIMS.md` C1/C3 are re-instantiated against the primary anchor via the claim-edit rule (expect garden B50 to move from "+0.157 better" toward "iso at half triangles" — that IS the honest claim; do not mourn the headline). No GEMS rows are re-run; re-expression is arithmetic over existing per-view JSONs plus the new anchor evals.

---

## R2 — GEOMETRY AXIS REOPENING: OPACITY-FLOOR RELEASE (existence test E2R; bounded)

**Human sanction (this paragraph is the authorization):** modifying the opacity floor is D1-legal representation work. This is a NEW mechanism class (representation parameterization), distinct from the two falsified loss routes and the falsified deletion route; that distinction is why exactly this — and nothing else on the geometry axis — is reopened.

- **E2R-v1 (pre-register fully before code):** replace the pinned floor with learnable per-triangle opacity `o = sigmoid(logit)` on `[o_min, 1]`, `o_min ∈ {0, 0.01}` (pick one, justify); initialize logits to reproduce current behavior bit-for-bit at step 0. Add (a) an opacity sparsity/decay regularizer and (b) **fade-and-prune**: triangles with `o < τ_fade` for T consecutive check intervals are REMOVED at the next prune step (topology op; budget `≤ B` respected; removal only). Fine-tune from the B5@50 model with the validated safe channel extended to {features, opacity} — positions stay frozen. **Rationale:** converts the delete-vs-keep dilemma into a differentiable trade — junk fades under the regularizer; load-bearing content resists via the photometric gradient (directly addressing the E2v3 selection-effect falsification).
- **E2R-v2 (only if v1 trains stably but geometry effect is weak):** v1 + the existing free-space hinge `L_fs` with gradients routed to OPACITY LOGITS ONLY (not positions; implementation exists in-tree, default-off).
- **Mandatory sanity sub-check (report in the same goal):** the tile compositor sorts by triangle-center depth with no backface culling; verify rendering correctness with genuinely semi-transparent triangles (compositing-order artifacts) on a small probe BEFORE full runs; report findings either way.
- **E2R PASS iff** at B50 on `toy_parking` AND `courtyard`: (g1 OR d1-false-free OR g3) improves ≥30% relative vs the B5 model (CI excl. 0), ΔPSNR ≥ −0.10 dB vs B5 (paired CI), AND before/after panels show visible cleanup.
- **Bounds:** ≤2 variants. If dead → the geometry demotion becomes PERMANENT, strengthened by the floor-release falsification; NO further geometry reopenings in this project, ever. E2R models enter MATRIX as NEW rows (`B6R`); banked B5 rows are never silently altered.

---

## R3 — DOWNSTREAM CLOSURE (the parking deliverable; mostly no retraining)

**Purpose.** Turn C4 from "preservation of an unreliable model" into a consumable artifact plus an application result. Framing guard against old pathologies: every product here is a ONE-TIME, GLOBAL, TRAIN-EVIDENCE-ONLY artifact (D4). Nothing varies per view or per query; all thresholds are calibrated ONCE on toy GT, then FROZEN and applied unchanged elsewhere (calibrate-once/test-elsewhere). If you notice per-view logic creeping in, stop and say so.

- **R3.a — Occupancy-extraction study (Tier-1; no retraining):** from the SAME checkpoints (primary anchor clean, B5@50, B5@25, B6R if it exists), compare two occupancy routes through d1 / d2 / ESDF error: (i) triangle voxelization (current) vs (ii) TSDF fusion of TRAIN-VIEW rendered depths (robust/median fusion; train views only). Pre-register on toy: route (ii) reduces d1 false-free by ≥50% relative at ≤2× false-occupied. Verdict on courtyard (+SS3DM when present) with toy-frozen fusion parameters. Either verdict is a paper result ("how to consume mesh splats for planning").
- **R3.b — Certified structural sub-mesh (Tier-2):** a one-time global triangle labeling (multi-view support ≥ k, train-depth consistency, opacity if R2 landed) producing a "collision-grade sub-mesh" consumed by route (i). Calibrated once on toy, frozen, evaluated elsewhere. This is an artifact-generation step, not a selector — say so in the report, and log the frozen thresholds.
- **R3.c — Planner closed loop v0 (Tier-1; small):** Hybrid-A* (or A* + footprint) on the ESDF/costmap from each occupancy route; ≥100 planned parking maneuvers per (scene × route × model), plans checked against GT geometry (toy GT mesh; courtyard laser scans; SS3DM when present). Report **collisions per 100 plans** and conservatism (path-length inflation / spurious infeasibility). This cell IS the "downstream application implementation".
- **R3.d — Real-scene demo (Tier-2):** ingest `parking_phone_tiny_anonymized` end-to-end (train clean → B5@50 → both occupancy routes → planner demo → flythrough panels). Qualitative/teaser; no GT mesh exists — state that plainly; metrics limited to rendering + cost.
- **R3.e — SS3DM integration (when D-1 lands):** ≥4 sequences through the full pipeline: g1–g4, d1/d2, R3.a and R3.c. This is the paper's driving-domain evidence; if D-1 is INFEASIBLE, the CLAIMS scope statement from R0.3 applies.

---

## R4 — STAGE-2 INTEGRATION AMENDMENTS

- **MATRIX additions (commit in one goal):** anchor rows (R1.a); `B6R` rows (conditional on E2R); ensure B12.5 across suites and add **B6.25 as Tier-2 on 3 scenes** (the regime where evidence-guidance is decisive per E1′ — the compaction claim's center of gravity moves here); occupancy-route cells (R3.a); planner cells (R3.c); parking_phone_tiny cells (R3.d); **pristine-submodule build check (Tier-1 hygiene):** build the rasterizer from the pinned submodule commit in a fresh env, compare one eval bit-for-bit; if it differs, freeze the local diff as an in-repo patch file and document — this is currently an open reproducibility landmine.
- **CLAIMS re-instantiation** (claim-edit rule, dated pointers; instantiate from evidence only, never aspiration):
  - C1′ — compaction at iso-quality vs the PRIMARY anchor at B50/B25, plus the aggressive-budget regime (B12.5/B6.25) where evidence-guidance dominates random by measured dB margins;
  - C2′ — only if E2R lands: measured geometry improvement; otherwise C2 stays a measurement claim, now STRENGTHENED by the floor-release falsification (a complete, mechanism-level account of why geometry repair fails in this representation);
  - C4′ — occupancy-route + planner results: correctly consumed compact mesh splats support parking-grade collision checking, with false-free reductions and collisions-per-100-plans quantified (numbers from R3, whatever they are).
- **RED-TEAM section in `EXPERIMENT_REPORT.md` (Tier-1):** answer the four anticipated attacks with evidence pointers: (A1) "your clean is decayed — re-anchor" → R1 dual rows; (A2) "importance ≈ random under FT" → E1′ verdict reported verbatim + aggressive-budget results; (A3) "modest ratios vs GS-compression literature" → positioning as reliability/deployment for mesh-based splatting + R1 3DGS reference row + FPS/VRAM story; (A4) "'for parking' without driving data or geometry improvement" → R3 results + SS3DM status + E2R verdict either way.
- **Paper-loop cadence:** after every 5 goals (R- or Stage-2), emit a one-page **CLAIMS-vs-EVIDENCE delta memo** (what strengthened, what weakened, what is still naked) so the humans can steer writing early. This memo is a deliverable, not chat.

---

## WHAT YOU MUST NOT REOPEN (verbatim boundary)

E1/E1′ numeric verdicts · 26k-sourcing for GEMS rows (refuted, −0.020 CI excl. 0) · E3 teacher distillation (stays diagnostic-only; citable) · the two falsified E2 loss routes and the falsified floater-deletion route · anything in the selector / per-view arbitration / gate / rollback species · metric-mouth forks · per-scene tuning. A proposal touching any of these is answered by naming this section and choosing a compliant action.

---

## COMPLETION OF THIS INSERT

Stage 1R is complete when: R1 has a frozen anchor verdict and CLAIMS re-instantiation is committed; E2R has a PASS-or-permanent-tombstone verdict; R3.a and R3.c have verdicts on toy + courtyard and the SS3DM status (integrated or INFEASIBLE-documented) is resolved; R4 amendments (MATRIX, CLAIMS, red-team skeleton, pristine-build check) are committed. Then Stage 2 resumes under the amended CLAIMS/MATRIX until its §10 completion criteria hold.

Honest negative verdicts on R1-T, E2R, and R3.a are acceptable closures of this insert; decorated ones are not. The paper is written from whatever survives.

— END OF STAGE 1R INSERT DIRECTIVE —

# GEMS — FULL-PROGRAM STATUS AUDIT
Audit time: 2026-07-09T03:03Z (UTC) · Auditor: read-only /goal session · Labels: VERIFIED-NOW / REPORTED / MISSING-UNKNOWN. Every item carries an evidence pointer. Where documents disagree, both values are shown.

## 1. SNAPSHOT

- Branch: `neurips-meshsplatopt-repair` — VERIFIED-NOW (`git branch --show-current`).
- HEAD: `6ab4c71 "Add GEMS method figures and sources" | 2026-07-08T19:50:01-07:00` — VERIFIED-NOW. NOTE: this commit post-dates program closure (2026-07-04) and adds `docs/figures/` method-overview assets only (`git show --stat 6ab4c71`).
- Last 10 commits — VERIFIED-NOW (`git log --oneline -10`):
  `6ab4c71` figures/sources · `7c415dc` record release tag · `74ee0d1` close Stage3 evidence pack · `328928e` C-00 closure activated · `097fba8` GOAL#020 pack v3 + conditional declaration · `9cf0301` GOAL#019 gap closures · `6a3ae28` flythrough videos · `96f3959` GOAL#018 final assembly · `3225c5a` GOAL#017 R1 context row · `0e55e56` Delta Memo #002.
- Tags: `gems-evidence-v1.0` (annotated `ead6d54`, peeled target `6ab4c71`) + 6 pre-GEMS tags (`archive/full9-…`, `f79/f82/f84/f89…`) — VERIFIED-NOW (`git tag`; `git rev-parse gems-evidence-v1.0^{}`; `git ls-remote spcarnet`). Remote also has `refs/heads/main` at `6ab4c71`.
- Dirty tree: 36 modified + 187 untracked files — VERIFIED-NOW (`git status --porcelain`). Modified set is pre-GEMS user WIP (e.g. `arguments/__init__.py`, `render.py`, `docs/car_model/*`, `scripts/car_model/*`); untracked set is SPCarNet-era assets/configs/docs. None are GEMS state files.
- GPUs: all 8 RTX 6000 Ada busy 58–100% util — VERIFIED-NOW (`nvidia-smi`); GEMS-owned running processes: **0** — VERIFIED-NOW (`ps` filter on train.py/gems_pipeline/run_eval/carved/planner). No zombies found.
- Storage: `/data` 26T/28T used, 867G free (df) / preflight ok=true free_gb=930.6 ≥ 50 — VERIFIED-NOW (`df -h`; `tools/storage_preflight.py /data/peilincai/gems_stage1`). `/` volume 5.8T free (df) but user quota ~100 GiB known-full — REPORTED (LEDGER DEC-007; flythrough tool note, LEDGER #018 addendum).

## 2. STAGE-BY-STAGE KEY-NODE CHECKLIST

### 2a. Stage 1
| node | verdict | key numbers (verbatim) | label | pointer |
|---|---|---|---|---|
| M0 Reproduce&Audit | DONE | garden eval reproduced exactly 24.7120/0.7618/0.2163; FT 19.3 it/s; census 9.28–11.57M tris | REPORTED | LEDGER milestone board; commit `16b3bb2` |
| M1 Protocol&Harness | DONE | GT-calibration row g1 0.028%, chamfer 1.8 cm, d2 1.000 vs toy clean g1 23.9%, d1 ff 58.9% | REPORTED | LEDGER #003/#004; `gems_stage1/eval/toy_parking_GTmodel_v1/metrics.json` |
| M2 Budget engine / E1 | E1 FAIL as written; mechanism validated | best FT-over-noFT +0.17; toy −0.52 vs clean; garden B50 noft −0.011 CI[−0.033,+0.003] | REPORTED | `KILL_REPORT.md`; LEDGER #005 |
| E1′ (amended) | FAIL | courtyard imp−rnd(+safe FT) +0.130 CI[−0.551,+0.640] < +1.0 bar | REPORTED | LEDGER #008 RESULTS |
| E2 (3 variants) | FAIL ×3 → demoted evaluation-only | base g3 −21.8% guard −0.245; routing −1.21/−0.75 guards blown; floater-prune g3 −100% but ΔPSNR −0.513/−0.235 | REPORTED | LEDGER #006, #008 Pre-reg C result; PROTOCOL §0 demotions |
| E3 (teacher) | SUNSET → diagnostic-only | garden distill−control +0.039/+0.051/+0.125 dB (all CI excl. 0, all < +0.15 floor) | REPORTED | LEDGER #007; evals `e3*_v1` |
| KILL_REPORT.md | exists | — | VERIFIED-NOW | `KILL_REPORT.md` (repo root) |
| STAGE1_REPORT.md | exists; candidate-complete + demotions | — | VERIFIED-NOW | `STAGE1_REPORT.md` |
| Fresh-clone repro | PASS | PSNR 30.3700 vs recorded 30.3705, \|Δ\|=0.0005 dB ≤ 0.05 | REPORTED | STAGE1_REPORT §7; `gems_stage1/logs/repro_check3.log` |

### 2b. Stage 1R
| node | verdict | key numbers | label | pointer |
|---|---|---|---|---|
| R1 anchor policy | Option (ii) clean-fixed@30k pre-registered; ADOPTED | fixed>clean30k 9/9 CI excl. 0 (+0.034..+0.314 dB); honest note: clean@26k > fixed@30k 9/9 | REPORTED | LEDGER #R-00/#R-01; evals `*_cleanfixed30k_v1` |
| CLAIMS v1.1 | committed | B50 floor 4/9 vs primary; 8/9 vs legacy; garden +0.070 CI excl. 0 above primary | VERIFIED-NOW (file) | `CLAIMS.md` edit log v1.1 |
| E2R / B6R | joint bar FAIL; courtyard bounded positive; SS3DM generalization DONE-FAIL | courtyard g1 −28.5% CI excl. 0, g3 −34/−41%, ΔPSNR +0.46; toy guard −0.383; SS3DM g3-fraction 0/3 (−14.3/−22.7/−18.6% vs ≥30%), guard held 3/3 | REPORTED | LEDGER #R-04, #014; `analysis/b6r_ss3dm/` |
| R3.a occupancy routes | DONE-FAIL (falsified) | TSDF worsens false-free 26–41% rel (toy clean 0.589→0.832); false-occ ×0.35–0.54 | REPORTED | LEDGER #R-02; `analysis/r3a_occupancy_routes/` |
| R3.c planner v0 | DONE | B50 preservation exact CI[0,0]; route-i 93–100% spurious infeasible; route-ii 10.7 coll/100 (courtyard) | REPORTED | LEDGER #R-03; `analysis/r3c_planner/` |
| R3.b certified sub-mesh | DONE-FAIL (0/4) | courtyard 42/100 found at 16.7 coll/100; kept 7.8–11.4%; kept-sets EXACTLY identical clean↔B50 | REPORTED | LEDGER #R-06; `analysis/r3b_submesh/` |
| Frozen fix-target | frozen | courtyard ≥30/100 found AND ≤3.0 coll/100 simultaneously | VERIFIED-NOW (quoted in LEDGER #C-02 bar) | LEDGER #R-06/#C-02 |
| D-1 / D-1b SS3DM | DONE-PASS / DONE | 4 towns + GT meshes, ~15 GB of 137 GB via Zip64 range extraction; left-handed frame mirrored | REPORTED | MATRIX rows; `mesh_datasets/SS3DM/ACQUISITION_LOG.md` |
| Pristine build | PASS (Outcome A) | bit-for-bit eval reproduction from pinned `b27f283`; zero source drift | REPORTED | LEDGER #R-07; `analysis/r4_pristine_build/` |
| CLAIMS v1.2 | committed | C4′ three components; fix-target frozen | VERIFIED-NOW (file) | `CLAIMS.md` edit log v1.2 |

### 2c. Stage 2 (MATRIX final statuses — extracted VERIFIED-NOW from `MATRIX.md`)
| cell | status |
|---|---|
| D-1 | **DONE-PASS** · D-1b **DONE** · D-2 **DONE** (GOAL#016) |
| E1-PARETO S-REND | B50 DONE-PASS (8/9 within −0.10 legacy; 5/9 above clean; LPIPS 9/9); B25/B12.5 DONE; B12.5 dominance +5.23 dB 9/9 CI |
| E1-PARETO S-GEO | B50/B25 × {B5,B4} × 4 towns DONE; B6R-on-SS3DM DONE-FAIL as pre-registered |
| E2-GEO | **DONE** (tables + evidence-vs-error analysis, GOAL#015A) |
| E3-REND / E4-EFF | **DONE** (pack; half-res bench w/ contention caveat; laptop bench waived) |
| E5-DOWN | **DONE** (N=500 + ESDF, GOAL#015B) |
| E6-ABL | **DONE** (importance families flat ±0.05 dB; other axes mapped to Stage-1 variants) |
| E7-SENS / E8-ROBUST | **DONE / RESIDUE WAIVED (Stage3 W1 / W2)** |
| E9-FAIL | **DONE** (13 cases, 5 families) · E10-STATS **DONE** · E11-QUAL **grids DONE + T2 videos DONE (W3 withdrawn)** |
| H1 | **DONE** (context-only, never CI-compared) · R1 **DONE** (context row) · B3 **DONE-PASS** (B3<B5 3/3) |
| PACK+§10 | **DONE (Stage3 pack v4)**: 234-row corpus (195 canonical) |
| R2 (T3, 2DGS-style) | **TODO** — only open cell; Tier-3 optional |
| R3.a / R3.c / R3.b / R3-FINAL | DONE-FAIL · DONE · DONE-FAIL 0/4 · **DONE-FAIL / IMPOSSIBILITY ×4** |

Waivers (draft → ruling → execution):
- W1 E7-SENS: draft → Stage3 ruling "run cheap substitute then waive" → RUN: seed-1 full retrain pair; clean Δ +0.031 dB CI[−0.018,+0.090]; B5-residual shift +0.008 dB CI[+0.000,+0.017]; both ≤ 0.15 pre-reg bar → residue WAIVED — VERIFIED-NOW (LEDGER #C-01; `RESULTS/aggregate/T7_robustness.md`).
- W2 E8-ROBUST+S-GEN: ruling "run one arm then waive" → RUN: garden 50% train-view drop (81/161 train, test unchanged 24); half-train B50 residual worsens −0.030 dB CI[−0.047,−0.014] (prediction met) → rest WAIVED — VERIFIED-NOW (LEDGER #C-01).
- W3 videos: WITHDRAWN — delivered; files exist — VERIFIED-NOW (`RESULTS/figures/videos/{garden,ss3dm_town01}_flythrough.mp4`).
- W4/W4a/W5: GRANTED as drafted (scope-freeze; B6.25 and S-GEO B2 ran in GOAL#019) — REPORTED (Stage3 prompt §1; LEDGER #019).
- W6 SS3DM planner cells: SUPERSEDED by §2.4; R3-FINAL failed → never run — REPORTED (Stage3 §1; LEDGER #C-02).
- §10 table: "All §10 rows are PASS; zero PARTIAL rows remain." — VERIFIED-NOW (`EXPERIMENT_REPORT.md:246`).
- Evidence pack: v4; `verify_t1.sh` re-run THIS AUDIT: "VERDICT: PASS" (byte-identical modulo timestamp) — VERIFIED-NOW.
- DECLARATION issued: "Stage Two complete — evidence pack ready for paper writing." — VERIFIED-NOW (LEDGER:481, GOAL #C-03).

### 2d. Stage 3 closure
| node | verdict | key numbers | label | pointer |
|---|---|---|---|---|
| W1/W2 execution | DONE (see 2c) | numbers above | VERIFIED-NOW | LEDGER #C-01 RESULT |
| R3-FINAL variants run | V1 only (mechanisms 1/3); V2/V3 correctly NOT run (no near-miss per pre-registration) | frozen: θ_free=−0.5, θ_occ=1.0, v_min=1, r_inf=1.0; votes: FREE −1 along [cam,0.95·d], OCC +2 at hit, stride-16, α≥0.5 | VERIFIED-NOW | LEDGER #C-02; `analysis/r3final_three_state_v1/summary.json` |
| P1 (toy FREE-set ff ≤10%) | MET | free_at_gt_occ_rate 0.0975 | VERIFIED-NOW | summary.json calibration.selected.confusion |
| P2 (courtyard fix-target) | FAILED | 0/100 found (toy and courtyard, clean and B50) | VERIFIED-NOW | LEDGER #C-02 RESULT; summary.json planner_metrics.plans_found=0 |
| P3 (infeasibility collapses) | FAILED | UNKNOWN-as-obstacle blocks 65.17% of GT-free toy space, 76.49% courtyard; toy band_occupied_fraction 0.788 vs GTREF 0.0227 | VERIFIED-NOW | LEDGER #C-02; summary.json |
| State fractions (toy clean, selected params) | — | FREE 0.3375 / OCC 0.0163 / UNKNOWN 0.6462 | VERIFIED-NOW | summary.json state_fractions |
| Inflation / coll per 100 | UNAVAILABLE | no common plans found (0/100) | VERIFIED-NOW | LEDGER #C-02 |
| Final verdict | **IMPOSSIBILITY ×4 route families** | `RESULTS/CONSUMPTION_IMPOSSIBILITY.md` (86 lines) | VERIFIED-NOW (file) | RESULTS/ |
| C4″ | NOT instantiated | CLAIMS v1.3 records the closure | VERIFIED-NOW | `CLAIMS.md` edit log v1.3 |
| SS3DM planner cells | not run (conditional on PASS) | — | VERIFIED-NOW | LEDGER #C-02 |
| parking_phone_tiny demo | not run (conditional on PASS, §2.4 item 4) | — | REPORTED | Stage3 §2.4 |
| Pack v4 | regenerated; corpus 234 rows (195 canonical); T7 complete | verify_t1 PASS 2026-07-04T07:48:36Z + re-verified now | VERIFIED-NOW | LEDGER #C-03; `RESULTS/REPRO_PACK/verify_t1_result.txt` |
| Purity audits | garden B5 headline artifact AUDIT GREEN re-run THIS AUDIT; carving-tool audit ok=true | — | VERIFIED-NOW | `gems_stage1/eval/status_audit_20260709/audit_report.json`; `eval/c02_purity_audit_fast/audit_report.json` |
| Release tag + ARCHIVE.md | tag exists locally+remote; ARCHIVE.md 66 lines | tag target `6ab4c71` (see §6 note) | VERIFIED-NOW | `git ls-remote spcarnet`; `RESULTS/ARCHIVE.md` |
| SUBMISSION_HANDOFF/ | 4 docs present: VENUE_MEMO (68 ln), REBUTTAL_BANK (100 ln), FIGURE_NOTES (50 ln), ABSTRACT_SKELETON (41 ln) | — | VERIFIED-NOW (existence+size; contents not line-audited) | `SUBMISSION_HANDOFF/` |
| Final closure line | ISSUED per ledger ("declaration = issued"; closure line in the closing /goal of 2026-07-04 session) | — | REPORTED (LEDGER #C-03 board) | LEDGER #C-03 |

## 3. HEADLINE NUMBERS (verbatim)

- Anchors: clean@30k (legacy); clean-fixed@30k PRIMARY — beats clean@30k **9/9 scenes, CIs excl. 0, +0.034..+0.314 dB**; context: clean@26k > fixed@30k 9/9 (not compute-matched). REPORTED (LEDGER #R-01).
- B50 slice (S-REND): B5 within −0.10 of clean@30k **8/9** (4/9 vs primary; garden +0.070 CI[+0.037,+0.100] above primary); LPIPS better 9/9. REPORTED (LEDGER #009/#R-01; T1).
- B25: floor 3/9 legacy / 2/9 primary; FT positive 9/9. B12.5: 0/9 floor (−0.31..−2.37), FT +0.09..+0.27 9/9 CI. REPORTED (LEDGER #010).
- B12.5 evidence-vs-random: **+5.23 dB mean, range +3.63..+8.19, 9/9 CI excl. 0**. REPORTED (LEDGER #011). S-GEO B50 imp−rnd: +0.77..+1.30 dB 4/4 CI. REPORTED (LEDGER #019).
- B3 QEM margin: B3−B5 = **−3.394 [−3.62,−3.15] (garden), −2.723 [−3.52,−1.94] (toy), −0.135 [−1.21,+1.31] (courtyard, 5-view)**. REPORTED (LEDGER #013).
- R1 3DGS context: at matched storage, 3DGS renders **2.1–3.4 dB above GEMS B5@B50 at 3.1–4.4× FPS** (garden 27.503 vs 24.851; bicycle 25.241 vs 23.135; kitchen 30.803 vs 27.449); kitchen vanilla already under target (no prune). REPORTED (`analysis/r1_3dgs_reference/r1_table.md`).
- E2R courtyard: g1 −28.5% CI excl. 0; g3 comps −34.3% / fraction −41.1%; ΔPSNR +0.460 [−0.151,+1.605]; ΔLPIPS −0.0063 [−0.0108,−0.0023]; toy guard −0.383 [−0.586,−0.201]. REPORTED (LEDGER #R-04).
- R3 trilogy: TSDF false-free +26–41% rel (falsified); route-i spurious infeasibility 88–100% (N=500: toy 0.884, courtyard 0.996; courtyard route-i's only 2/500 plans BOTH collide); R3.b 42/100 found at 16.7 coll/100; ESDF clearance under-estimated 1.0–2.9 m mean. REPORTED (LEDGER #R-02/#R-03/#R-06/#015).
- R3-FINAL V1: **0/100 found on toy and courtyard, clean and B50**; P1 met (0.0975 ≤ 0.10); UNKNOWN blocks 65.17%/76.49% of GT-free space. VERIFIED-NOW (LEDGER #C-02; summary.json).
- W1: clean seed1−seed0 **+0.031 dB CI[−0.018,+0.090]**; B5-residual shift **+0.008 dB CI[+0.000,+0.017]**. W2: half-train residual worsens **−0.030 dB CI[−0.047,−0.014]**. VERIFIED-NOW (LEDGER #C-01).

## 4. CLAIMS FINAL STATE (verbatim from `CLAIMS.md`, VERIFIED-NOW this session)

- **C1 — Compactness–quality.** "At matched triangle budgets, GEMS-core (evidence prune + drift-safe features-only fine-tune) Pareto-dominates random pruning and QEM-style decimation on rendering metrics, and achieves ≥50% triangle reduction at iso-quality vs clean on real scenes (Stage-One instantiation: garden B50 +0.157 dB / −0.0071 LPIPS BETTER than clean, CIs excl. 0; garden B25 iso; courtyard B50 iso). Known bound: on a heavily over-parameterized synthetic scene (toy_parking) B50 costs −0.52 dB (train-coverage selection effect, measured)."
- **C2 — Geometry reliability: DEMOTED to measurement claim.** "…GEMS compaction preserves geometry-reliability metrics (g1–g4) at half budget, and the GEMS metric suite exposes that photometric quality masks geometric unreliability in splatting models (toy clean: 30.9 dB PSNR yet 23.9% free-space violations, 59% d1 false-free; courtyard 65%; GT-calibration row proves the metrics). NON-claim: GEMS does not improve geometry vs clean."
- **C3 — Rendering.** "At B ≥ 50%, GEMS renders within −0.10 dB of clean on real dev scenes (measured: garden ABOVE clean; courtyard −0.04 prune-only). Teacher part DEMOTED (E3 sunset): distillation is reported as a diagnostic channel (+0.04..+0.13 dB, CIs excl. 0, below floor; recovered fraction ~5–16%…)."
- **C4 — Downstream proxy: BOUNDED to preservation.** "…GEMS preserves downstream proxies exactly (courtyard d2 collision agreement 0.895, unsafe-disagreement 0.000, identical clean/B50/B25; toy d2 0.625→0.625). NON-claim: GEMS does not lower false-free-space rates vs clean… The safety-relevant contribution is the measurement suite + the false-free-space exposure." (C4″ NOT instantiated — v1.3.)
- **NON-CLAIMS (verbatim):** "This is a per-scene optimization setting; the teacher is train-only and absent at test time; no claim of state-of-the-art novel-view quality versus the 3DGS family; no claim about high-speed driving; downstream results are proxies unless the closed-loop stretch item was executed. Additionally (Stage-One additions): no claim that GEMS improves geometry or downstream metrics vs clean; no claim that evidence-guided importance dominates random pruning under safe fine-tuning at moderate budgets on small scenes (courtyard B50: +0.13 CI incl. 0); the end-phase-decline finding (clean26k > clean30k) is reported as a property of the baseline trainer, not a GEMS contribution. Stage3 addition: no tested one-time train-evidence occupancy consumer supports parking-grade closed-loop planning on these checkpoints; the closed-loop stretch is an impossibility result, not a positive application claim."
- Last 3 claim-edit log entries: **v1.3 (2026-07-04)** R3-FINAL closure — C4″ not instantiated; IMPOSSIBILITY×4 route families. **v1.2 (2026-07-03)** C4′ three components + frozen fix-target. **v1.1 (2026-07-03)** primary-anchor re-instantiation (dual rows; B50 4/9 vs primary).

## 5. DEVIATIONS & INCIDENTS (from LEDGER, checked this session)

1. Latent trainer bug fixed mid-program: supersampling `==` resume gate (commit `5b6caa5`) — voided the first E1 FT wave (tag e1 → e1b). REPORTED (LEDGER #005).
2. ETH3D scan misalignment: PROTOCOL 1.1.1 GT-asset correction; one g4 row VOIDed. REPORTED (PROTOCOL changelog; LEDGER #005).
3. PROTOCOL version bumps: 1.0.0 → 1.1.0 (pre-first-row amendments) → 1.1.1 (scan transforms). REPORTED (PROTOCOL.md changelogs).
4. Pipeline crash (stage_evidence str/dict, `ee5cf64`); zsh word-splitting no-op B25 launch (silent; caught by supervision); repro-check double silent death (improperly detached launches); 3 self-matching pkill incidents (protocolized in `tools/gems/run_supervised.sh` + `jobs_wait.sh`); R1 chain `sys.path` + `weights_only` crashes (fixed in-run); C-02 save-dir bug (fixed, relaunched). REPORTED (LEDGER #005/#009/#C-02; memory notes).
5. Sanctioned single-mouth exception: R1 3DGS rows computed outside `run_eval.py` (representation differs), documented per-row. REPORTED (LEDGER #017).
6. Iteration budgets: E1 3/3 variants; E2 3/3; E3 sunset at 3 below-floor; E2R closed at 1/2 (v2 trigger unmet); R3.a 0/2 tuning variants; R3-FINAL 1/3 mechanisms (no near-miss → no V2/V3). No budget violations found. VERIFIED-NOW (LEDGER budget lines).
7. Two Stage-3 launch attempts died on API errors (internal error; 529 overloaded) and one on usage-credit exhaustion; work resumed and completed 2026-07-04. REPORTED (session record; LEDGER #C-02 "relaunched").

## 6. CROSS-DOCUMENT CONSISTENCY CHECK

1. **Tag target vs ARCHIVE.md**: `RESULTS/ARCHIVE.md` (2026-07-04) says the tag was created "for the final evidence-pack commit"; the tag now resolves to `6ab4c71` (2026-07-08 figures commit), i.e. it was created/moved AFTER closure to include post-closure figure assets. Also LEDGER #C-03 says "pushed to `spcarnet/main`" while all program work is on `neurips-meshsplatopt-repair`; remote `refs/heads/main` exists at `6ab4c71`. Both facts VERIFIED-NOW; flagged, not reconciled.
2. **MATRIX status-note drift**: several rows carry notes that originated from neighboring rows during successive edits (e.g. the `E3-REND` row cites "evidence pack v1 commit 3dce47c" which is accurate, but `H1 row` carries text beginning "13 cases, 5 mechanism families…" in one historical revision; current file text VERIFIED-NOW is self-consistent after the Stage-3 rewrite except cosmetic duplication of R3.b in one earlier revision, since deduplicated). No numeric contradictions found between MATRIX and LEDGER.
3. **corpus row counts**: pack v2 "217 rows (179 canonical)" (GOAL#018) vs pack v4 "234 rows (195 canonical)" (LEDGER #C-03) — consistent growth, not a mismatch; noted for the reviewer.
4. **EXPERIMENT_REPORT `PARTIAL` string**: single remaining occurrence is the sentence "All §10 rows are PASS; zero PARTIAL rows remain." — not a status row. VERIFIED-NOW (grep).
5. LEDGER vs CLAIMS vs CONSUMPTION_IMPOSSIBILITY on R3-FINAL: all three agree (0/100; IMPOSSIBILITY×4). VERIFIED-NOW.
6. No claim in `CLAIMS.md` lacks an evidence pointer in `RESULTS/CLAIMS_EVIDENCE_MATRIX.md` — REPORTED (per GOAL#018/#C-03 audits; matrix file exists, VERIFIED-NOW, contents not re-derived).

## 7. OPEN BLOCKERS TO A SUBMITTABLE PAPER

1. Paper prose itself — human-only by program design (§4 handoff exists). Cost: human writing time (est. 2–4 weeks). Blocks: submission.
2. Venue decision + deadline verification — human-only; `SUBMISSION_HANDOFF/VENUE_MEMO.md` carries [HUMAN-VERIFY] deadline tags. Cost: ~1 human-h.
3. R2 row (2DGS-style on S-GEO) — MATRIX TODO, Tier-3 optional; agent-doable (~1 day setup+runs) or drop with a note. Blocks: nothing mandatory.
4. A8 rebuttal (SuGaR/2DGS meshing comparison) — declared limitation/future work in REBUTTAL_BANK; running it would be new science (out of closed scope; human call). Cost if run: multi-day.
5. Figure polish (F1/F3 are drafts by design) — human art direction; FIGURE_NOTES.md exists. Cost: human-h.
6. Optional: reconcile the tag/branch note in ARCHIVE.md with the actual `main` branch + tag target (§6.1) — agent-doable in minutes, but modifies RESULTS/ (outside this audit's write scope).

## 8. SELF-ASSESSMENT

- Percent-complete: engineering system **100%** (pipeline, supervision, single mouth, audits, repro all verified); evidence pack **100%** (v4, byte-verified, §10 all-PASS); downstream application **100% closed as negative** (IMPOSSIBILITY×4 with quantified mechanisms and a frozen fix-target — a citable result, not a working demo; 0% as a positive deliverable); paper-support materials **~90%** (handoff docs exist; figures F1/F3 drafts; prose 0% by design).
- Top-3 submission risks: (1) the downstream story is a negative result — venues wanting a working robotics demo (e.g. ICRA application track) will read C4 as preservation+impossibility, not capability; (2) rendering-quality gap vs 3DGS family (2.1–3.4 dB at matched storage) invites "why this representation" pushback — answered but not neutralized by A3; (3) breadth waivers (E7 full grid, E8 pose-noise/S-GEN) leave a reviewer opening on robustness, partially mitigated by the seed-pair and view-drop arms.
- Single highest-value next action: human writing pass starting from `RESULTS/HANDOFF.md` §narrative order, with venue chosen per `VENUE_MEMO.md` (the FAIL branch of its logic: 3DV / ICRA-analysis / WACV primary).
- Honest summary: the program is closed and internally verified. What exists on disk is a reproducible, CI-disciplined evidence pack showing that evidence-guided compaction of mesh-splatting checkpoints is essentially free at 50% (and dominant over random/QEM at aggressive budgets), that fine-tuning any channel except appearance features damages converged checkpoints, that the baseline trainer's own final iterations destroy value, and that every tested one-time route for consuming these checkpoints as planning-grade occupancy — including the final three-state carving — fails, with measured mechanisms. There is no working parking demo; there is a quantified explanation of why not, plus preservation-exactness under compaction. The declaration line is issued, the release is tagged, and the remaining work is human: choose a venue, write the paper, polish two draft figures.

STATUS AUDIT COMPLETE — report at STATUS_AUDIT_20260709.md

# GEMS — Consumption Impossibility Addendum

Generated 2026-07-04 for Stage3 closure.

## Verdict

**R3-FINAL V1 FAILS as a hard non-near-miss.** The final untested consumer class,
three-state train-evidence visibility carving, does not meet the frozen
parking-grade bar on either calibration or transfer scenes. No V2/V3 mechanism
is launched because Stage3 permits V2/V3 only for a near-miss; V1 produced
0/100 feasible plans on all reported cells.

Therefore the downstream-consumption axis closes as:

> No tested one-time train-evidence consumer among surface voxelization,
> TSDF fusion, certified sub-mesh, and visibility-carved three-state occupancy
> meets parking-grade closed-loop planning bars on these checkpoints. The
> blocker is baseline checkpoint geometry plus train-coverage limits, invariant
> to B50 compaction.

R3-FINAL mechanisms consumed: **1/3** (`V1 log-odds visibility carving`).

## R3-FINAL V1 Evidence

Frozen calibration was performed once on `toy_parking` clean, then applied
unchanged to toy B50 and courtyard clean/B50.

Selected frozen parameters:

```json
{"theta_free": -0.5, "theta_occ": 1.0, "v_min": 1, "r_inf": 1.0}
```

| scene | model | FREE frac | UNKNOWN frac | FREE@GT-occ | blocked@GT-free | found/100 | coll/100 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| toy_parking | clean30k | 0.3375 | 0.6462 | 0.0975 | 0.6517 | 0 | n/a | FAIL |
| toy_parking | B50 | 0.3375 | 0.6462 | 0.0975 | 0.6517 | 0 | n/a | FAIL |
| courtyard | clean30k | 0.2259 | 0.7649 | 0.0621 | 0.7649 | 0 | n/a | FAIL |
| courtyard | B50 | 0.2259 | 0.7649 | 0.0621 | 0.7649 | 0 | n/a | FAIL |

Important nuance: V1 does achieve the intended toy false-free safety filter
(`FREE@GT-occ <= 10%`). The failure is instead the conservative side of the
trade: treating UNKNOWN as obstacle blocks 65-76% of GT-free space, so the
planner cannot find feasible paths.

Evidence paths:

- `/data/peilincai/gems_stage1/analysis/r3final_three_state_v1/summary.json`
- `/data/peilincai/gems_stage1/analysis/r3final_three_state_v1/R3_FINAL_V1_REPORT.md`
- `/data/peilincai/gems_stage1/analysis/r3final_three_state_v1/calibration/toy_clean_calibration.json`
- Purity audit: `/data/peilincai/gems_stage1/eval/c02_purity_audit_fast/audit_report.json`

## Route-Family Closure

| route family | best observed useful behavior | decisive blocker | evidence |
|---|---|---|---|
| raw surface voxelization | compaction-preservation exact; clean/B50 planner outcomes invariant | route-i causes spurious infeasibility: toy 7/100, courtyard 0/100 in the first-100 cells | `RESULTS/aggregate/T5b_r3_trilogy.md`, R3.c |
| TSDF fusion | raises courtyard feasibility to 28/100 | unsafe: courtyard collision rate 10.7/100, above the 3.0 cap; false-free worsens | `RESULTS/aggregate/T5b_r3_trilogy.md`, R3.a/R3.c |
| certified sub-mesh | courtyard found rate 42/100 | unsafe: courtyard 16.7 coll/100; toy only 14/100; supported kept sets still shed/load wrong structure | `RESULTS/aggregate/T5b_r3_trilogy.md`, R3.b |
| three-state visibility carving | toy FREE false-free <=10%; clean/B50 invariant | over-conservative UNKNOWN blocks 65-76% GT-free space; 0/100 found on toy and courtyard | `summary.json`, `R3_FINAL_V1_REPORT.md` |

## Mechanism Diagnosis

The failure is not a threshold accident. The explored routes cover the two
natural extremes and the train-evidence middle:

- declaring broad unobserved space FREE creates false-free hazards or real
  collisions;
- declaring unobserved space UNKNOWN avoids false-free but blocks too much
  legitimate drivable/free space;
- support/depth-certified surfaces cannot separate supported junk from
  load-bearing surfaces reliably enough for collision-grade planning.

The consistent clean/B50 invariance across depth, certification sets, grids, and
planner outcomes means B50 compaction is not the limiting factor. The limiting
factor is the checkpoint geometry and train-view coverage: the rendered
evidence does not contain a collision-grade world model in the vehicle band.

## Claim Handling

Positive downstream claims remain bounded to preservation/proxy statements.
The paper should not claim closed-loop parking-grade planning from these
checkpoints. The strongest honest statement is negative and diagnostic:
GEMS exposes where splatting-style checkpoint geometry remains unsafe for
one-time occupancy consumers, while B50 compaction preserves those downstream
outcomes exactly under the tested consumers.

## DS-1 Addendum (2026-07-10, Stage-4 prompt §6 — the ONE permitted retry): STRENGTHENED

The closure was narrowly reopened for one pre-registered mechanism (LEDGER
GOAL #E-10): ray stride 16→2, FREE dilation by r_inf, and UNKNOWN =
traversable at ×5 cost (never free); V1's frozen params reused, NO
recalibration; GTREF/problems/collision accounting byte-identical.

**Verdict: FAIL — and the diagnosis is sharper than V1's.** Courtyard stays
0/100 found on both checkpoints, but the failure MODE changed: with UNKNOWN
traversable, nothing is starved — instead the dense stride-2 carve floods
the map with depth-noise votes, so the model's own OCCUPIED set invalidates
the episode poses (courtyard: 92 start_invalid + 8 goal_invalid; lethal
fraction 88.6% after the r_inf inflation). Critically, P1 no longer holds
at dense sampling: raw FREE false-free rises from 6.2% (stride 16) to
19.1% (stride 2) — denser sampling of the same rendered depth makes the
map WORSE on both safety axes simultaneously. Toy moves 0→6/100 (0
collisions, far under the bar; 57 start_invalid / 31 goal_invalid / 6
disconnected).

The V1 diagnosis ("the map was safe; the semantics starved it") is
therefore amended: at the sampling density needed to defeat starvation, the
map is NOT safe. The binding constraint is the metric accuracy of the
checkpoint's own rendered depth in the vehicle band — not ray density, not
UNKNOWN semantics, not planner cost design. This closes the §6 reopening
with the impossibility STRENGTHENED: no configuration of this one-time
carve family reaches the frozen fix-target, and the failure now has a
single named cause. (Note the Stage-4 contrast: the SAME rendered evidence
that cannot form a collision-grade world model is highly effective as
RENDER-TIME photometric evidence — ECR's +1.67 dB — which is exactly the
representation-vs-consumption boundary this program documents.)

Evidence: `analysis/ds1_dense_carve/{summary.json,cells/*.json,maps/*}`;
mechanism + hook: `tools/gems/ds1_dense_carve.py`,
`tools/gems/planner_loop.py::astar(cell_cost_mult=None)` (default-off).

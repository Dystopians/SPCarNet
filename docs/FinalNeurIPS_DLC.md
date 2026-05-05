# MeshSplatOpt Surface-Ledger Expansion Pack

Date: 2026-05-04  
Purpose: strengthen the MeshSplatOpt story so it is not a pile of modules, but one simple, memorable research idea.

This file is an **expansion package** for `FinalNeurIPSPrompts_MeshSplatOpt.md`. It should be given to Codex after the current F0-F3 audit/planning stages, or immediately if Codex has not yet started F4. If F4 is already implemented, use this package to replace/extend F4-F11.

---

## 0. New core story

Current story is too modular:

```text
CSEF + compaction + rollback + sparse depth + topology freeze + snap + fill + recovery
```

A reviewer may read this as engineering accumulation.

The new story should be:

> **Surface Evidence Ledger: every surface must pay rent.**

A Mesh Splatting model naturally grows an overcomplete surface hypothesis bank. This is useful: it finds many possible surfaces, colors, and view explanations. But the final representation should not keep every hypothesis. MeshSplatOpt treats topology as a budgeted resource and runs a ledger audit:

```text
surfaces that explain evidence earn rent;
redundant or unsupported surfaces pay topology tax;
holes, dents, and missing ground are unpaid evidence debt;
the optimizer transfers topology credits from low-rent surfaces to high-debt regions;
every transaction is counterfactually certified before it is committed.
```

This gives a simple intuition:

```text
Let Mesh Splatting overbuild.
Audit every surface.
Evict surfaces that do not pay rent.
Invest topology where the scene has unpaid evidence debt.
Freeze the settled topology.
Recover appearance and geometry.
```

This is more exciting than "safe pruning" because it unifies compaction and repair:

- pruning is eviction,
- merging/collapse is consolidation,
- snap is debt correction,
- split/fill is investment,
- topology freeze is closing the ledger,
- recovery is refinancing appearance after topology settlement,
- counterfactual gate is the auditor.

---

## 1. Why this is a stronger NeurIPS story

### What existing papers usually do

Compact / pruning methods ask:

```text
Which primitives can be removed?
```

Geometry-regularized methods ask:

```text
How can we make the learned surface more geometrically plausible?
```

Classical mesh processing asks:

```text
How can we simplify / remesh / fill a mesh?
```

MeshSplatOpt should ask:

```text
Given a finite topology budget, where should surface capacity be spent so that the final representation explains the most multi-view evidence with the least hallucination?
```

This is the key distinction.

### Why the current R53 result fits perfectly

R53 already proves the "eviction + recovery" half:

```text
clean 22k overcomplete teacher
-> remove 70% smallest-area triangles
-> strict topology freeze
-> recovery
-> better PSNR / SSIM / LPIPS / sparse geometry than clean 22k with 30% triangles
```

The expansion makes it more principled:

```text
R53 area pruning is the first crude ledger policy.
The final method should replace area-only rent with Surface Evidence Rent.
```

### Why this also handles giant holes

A giant parking-lot void is not just a hole. It is **unpaid evidence debt**.

The method should not hallucinate it for free. It should require:

```text
debt certificate:
  boundary support
  neighboring ground/plane support
  sparse/depth support
  multi-view coverage
  low free-space conflict
  low uncertainty or explicit prior-only label

topology funding:
  donor triangles removed/merged from redundant low-rent surfaces

counterfactual certificate:
  render / sparse geometry / normal / changed-pixel / free-space gate passes
```

Thus the method can say:

> We do not merely add geometry to fill a hole. We transfer topology budget from overbuilt surfaces to under-explained scene regions, and the transaction must pass a counterfactual audit.

This is intuitive, solid, and interesting.

---

## 2. New method vocabulary

Use these terms consistently.

### Surface Evidence Ledger

A table over primitives and regions.

For each primitive `i`:

```text
rent_i = evidence_value_i - topology_tax_i - risk_i
```

For each region `r`:

```text
debt_r = unexplained_evidence_r + missing_surface_evidence_r - free_space_conflict_r - uncertainty_r
```

### Surface rent

How much a primitive earns by explaining the scene.

Signals:

```text
multi-view visibility
render contribution
gradient/sensitivity
sparse COLMAP support
normal agreement
teacher-student agreement
low changed-pixel risk
boundary / thin-structure protection
```

### Topology tax

How expensive a primitive is to keep.

Signals:

```text
triangle count
vertex count
projected area / raster cost
memory / attribute cost
redundancy with neighbors
low contribution after recovery
```

### Evidence debt

Where the scene is under-explained.

Signals:

```text
high render residual
missing sparse depth support
boundary loop / hole
ground-plane discontinuity
vehicle surface discontinuity
large void
rough broken surface
```

### Topology credits

Capacity recovered by deleting or merging low-rent surfaces.

### Topology investment

Capacity spent on filling, splitting, snapping, or preserving high-debt regions.

### Ledger transaction

A compound edit:

```text
donor action:
  delete / collapse / merge low-rent primitives

recipient action:
  fill / split / snap / protect high-debt region

recovery action:
  topology-frozen appearance/geometry recovery

audit:
  counterfactual certificates
```

A transaction may be donor-only, recipient-only, or budget-neutral. The most exciting paper branch is **budget-neutral repair**:

```text
repair a hole or misalignment while keeping or reducing total primitive count.
```

---

## 3. Add this to the paper narrative

Suggested introduction paragraph:

> Mesh-splatting optimization is excellent at discovering surface hypotheses, but its final topology is not a principled allocation of scene capacity. Some triangles over-explain already easy regions, while holes and broken surfaces remain under-explained. We view this as a surface-accounting problem: every primitive must pay rent by explaining multi-view evidence, and every unexplained region creates evidence debt. MeshSplatOpt introduces a Surface Evidence Ledger that audits an overcomplete mesh, evicts low-rent topology, invests capacity into high-debt repairs, and certifies every transaction through counterfactual rendering and sparse-geometry validation. The result is a compact, topology-frozen mesh-splat representation that can outperform the overcomplete clean model while using far fewer primitives.

Suggested one-line slogan:

```text
Grow first, audit later: every surface must pay rent.
```

Suggested paper contribution bullets:

1. **Surface Evidence Ledger**, a unified accounting framework for mesh-splat topology and repair decisions.
2. **Budget-neutral certified mesh surgery**, which transfers topology capacity from redundant regions to under-explained defects under rollback-certified validation.
3. **Strict topology-frozen recovery**, which converts a settled topology into a compact high-quality representation.
4. **Evidence-calibrated Pareto evaluation**, comparing clean long training, posthoc compaction, sparse-depth recovery, PRISM, and repair transactions under fair baselines.

---

## 4. Implementation extension stages

The following stages are designed to be appended to `FinalNeurIPSPrompts_MeshSplatOpt.md`.

Use prefix `L` for Ledger stages.

---

# Prompt L0 — Story lock: Surface Evidence Ledger manifesto

```text
You are working inside Dystopians/SPCarNet.

Mission:
Write a short manifesto that replaces the scattered "CSEF + edits + recovery" story with a single Surface Evidence Ledger story.

Read:
- README.md
- docs/car_model/parking_clean_to_compact_repair_report.md
- docs/car_model/parking_best_clean_long_vs_method_long_report.md
- docs/car_model/meshsplatopt_stageR2_related_work_matrix.md
- docs/car_model/meshsplatopt_stageR1_repair_RFC.md
- docs/car_model/final_stageF1_method_spec.md if present

Write:
- docs/car_model/ledger_stageL0_surface_ledger_manifesto.md

The manifesto must include:
1. One-sentence method:
   "MeshSplatOpt audits an overcomplete mesh-splat scene with a Surface Evidence Ledger, evicts low-rent topology, invests capacity into high-debt repairs, and certifies every transaction by counterfactual rendering and sparse geometry."

2. Definitions:
   - surface rent
   - topology tax
   - evidence debt
   - topology credit
   - topology investment
   - ledger transaction

3. Why this is not just pruning:
   - pruning is only one kind of ledger transaction;
   - holes are evidence debt, not just missing triangles;
   - fill/snap/split must be funded and certified;
   - topology budget can be transferred rather than only reduced.

4. How it explains R53:
   - R53 area pruning is the first crude rent policy;
   - the final method should test whether evidence rent beats area-only rent.

5. How it explains giant-hole repair:
   - a giant void can be repaired only if debt evidence is strong;
   - if unobserved, it must be marked prior-only diagnostic;
   - budget-neutral repair should transfer topology from low-rent donors to the void patch.

6. What not to claim:
   - do not claim snap/fill are load-bearing unless L6/L7 prove it;
   - do not claim Surface Ledger beats area-only unless L4/L5 prove it.

Append research-log entry.
Commit and push.

Gate:
PASS only if the manifesto can be understood by a non-specialist reviewer in one reading.
```

---

# Prompt L1 — Surface Ledger data model

```text
Stage L0 must be PASS.

Mission:
Implement the Surface Evidence Ledger data model on top of existing CSEF.

Create:
- ss3dm_prior/meshsplatopt/surface_ledger.py
- scripts/car_model/meshsplatopt_build_surface_ledger.py
- scripts/car_model/smoke_test_ledger_stageL1.py
- docs/car_model/ledger_stageL1_surface_ledger_design.md
- docs/car_model/ledger_stageL1_surface_ledger_report.md

Data structures:

PrimitiveLedgerEntry:
- primitive_id / face_id
- area
- projected_area_estimate
- visibility_score
- render_contribution_score
- sparse_support_score
- normal_support_score
- boundary_risk_score
- redundancy_score
- uncertainty_score
- topology_tax
- evidence_value
- surface_rent
- donor_score
- protect_score
- notes

RegionDebtEntry:
- region_id
- defect_type
- explanation_debt
- missing_surface_score
- render_residual_score
- sparse_depth_gap_score
- boundary_loop_score
- ground_plane_support
- object_prior_support
- free_space_conflict
- uncertainty
- investment_score
- allowed_edit_types
- prior_only_flag
- notes

LedgerSummary:
- total_surface_rent
- total_topology_tax
- total_evidence_debt
- donor_budget_available
- investment_budget_requested
- number_low_rent_primitives
- number_high_debt_regions

Implement:
- build_primitive_ledger_from_checkpoint_or_mesh(...)
- build_region_debt_from_defects(...)
- compute_surface_rent(...)
- compute_investment_score(...)
- export_surface_ledger_json_csv_npz(...)

Initial formula:
surface_rent =
  + visibility
  + render_contribution
  + sparse_support
  + normal_support
  - topology_tax
  - redundancy
  - uncertainty
  - free_space_conflict

donor_score =
  topology_tax + redundancy + uncertainty - evidence_value - boundary_risk

investment_score =
  explanation_debt + missing_surface + boundary_loop + ground/object_prior
  - free_space_conflict - uncertainty

Smoke:
- synthetic scene with:
  - good supported surface,
  - redundant low-area surface,
  - floater,
  - hole,
  - giant ground void.
- Verify:
  - good surface has positive rent and low donor_score;
  - redundant/floater has high donor_score;
  - hole/void has high investment_score;
  - unknown void is high uncertainty and prior_only if applicable.

Append research-log entry.
Commit and push.

Gate:
PASS only if synthetic tests show donor and investment rankings behave correctly.
```

---

# Prompt L2 — Topology credit bank and budget-neutral transaction planner

```text
Stage L1 must be PASS.

Mission:
Implement a topology credit bank that pairs donor primitives with repair/investment regions.

Create:
- ss3dm_prior/meshsplatopt/topology_credit_bank.py
- scripts/car_model/meshsplatopt_plan_ledger_transactions.py
- scripts/car_model/smoke_test_ledger_stageL2_credit_bank.py
- docs/car_model/ledger_stageL2_credit_bank_design.md
- docs/car_model/ledger_stageL2_credit_bank_report.md

Definitions:
- donor: primitive or group that can be deleted/collapsed/merged.
- recipient: defect region that can receive snap/split/fill/protect.
- credit: topology budget released by donor.
- cost: topology budget required by recipient edit.
- transaction: donor actions + recipient actions + expected recovery/audit.

Implement transaction modes:
1. donor_only_compaction
2. recipient_only_repair
3. budget_neutral_repair
4. budget_positive_repair, allowed only as diagnostic unless explicitly enabled
5. budget_negative_compact_repair, preferred if possible

Planner:
- sort donors by donor_score;
- sort recipients by investment_score;
- allocate credits to highest-confidence recipients;
- protect high-rent or high-boundary-risk surfaces;
- enforce max topology cost;
- enforce no prior-only recipient in normal mode;
- output transaction JSON.

Transaction fields:
- transaction_id
- donor_edit_ids
- recipient_edit_ids
- topology_credit
- topology_cost
- net_topology_delta
- expected_debt_reduction
- expected_rent_loss
- expected_risk
- mode
- prior_only_flag
- requires_counterfactual_gate

Smoke:
- synthetic donor pool with 100 removable triangles;
- synthetic giant void fill requiring 40 triangles;
- planner should propose budget_neutral_repair with net topology <= 0.
- unknown void should be rejected in normal mode.

Append research-log entry.
Commit and push.

Gate:
PASS only if the planner can fund a giant-void repair from low-rent donors without increasing total topology.
```

---

# Prompt L3 — Amortized Counterfactual Auditor

```text
Stage L2 must be PASS.

Mission:
Build a lightweight predictor that learns from existing counterfactual gate logs to predict which ledger transactions are likely to pass.

This is the second research upgrade. Existing pruning papers often use analytic or score-based importance. MeshSplatOpt should learn an audit prior from actual measured counterfactual outcomes, while still requiring real gates for final acceptance.

Create:
- ss3dm_prior/meshsplatopt/amortized_auditor.py
- scripts/car_model/meshsplatopt_collect_audit_training_data.py
- scripts/car_model/meshsplatopt_train_amortized_auditor.py
- scripts/car_model/meshsplatopt_score_transactions_with_auditor.py
- scripts/car_model/smoke_test_ledger_stageL3_auditor.py
- docs/car_model/ledger_stageL3_amortized_auditor_design.md
- docs/car_model/ledger_stageL3_amortized_auditor_report.md

Training data sources:
- PRISM counterfactual JSONs
- MeshSplatOpt render-backed gate JSONs
- R14/R17/R18/R19/R21/R22/R26 gate reports if available
- synthetic R13 gate outputs

Features:
- transaction mode
- edit type counts
- donor_score stats
- investment_score stats
- net topology delta
- expected rent loss
- expected debt reduction
- uncertainty stats
- prior_only flag
- free_space conflict
- projected changed pixel proxy
- scene id / scene type if available

Labels:
- accept/reject
- delta_psnr
- delta_ssim
- delta_lpips
- delta_absrel
- changed_pixel_ratio
- post_recovery_success if available

Model:
- Start with logistic regression / ridge regression / random forest if sklearn available.
- If sklearn unavailable, implement a simple numpy linear/ridge model.
- The auditor is advisory only.
- Final edits still require real counterfactual gates.

Outputs:
- audit_dataset.csv
- auditor_model.pkl or auditor_model.npz
- auditor_report.md
- calibration plot data JSON

Smoke:
- generate tiny synthetic dataset with pass/fail labels;
- train auditor;
- score transactions;
- verify it ranks known-safe transaction above known-bad transaction.

Append research-log entry.
Commit and push.

Gate:
PASS only if auditor training and scoring run end-to-end and the report states that auditor cannot replace real gates.
```

---

# Prompt L4 — Ledger compaction selector for real checkpoints

```text
Stage L3 must be PASS.

Mission:
Replace/extend area-only compaction with Surface Ledger compaction on real Mesh Splatting checkpoints.

Create:
- scripts/car_model/meshsplatopt_select_ledger_compaction.py
- scripts/car_model/meshsplatopt_apply_ledger_compaction.py
- scripts/car_model/smoke_test_ledger_stageL4_real_compaction.py
- docs/car_model/ledger_stageL4_real_compaction_selector_report.md

Modes:
1. area_smallest
2. ledger_rent_lowest
3. ledger_donor_highest
4. ledger_auditor_ranked
5. hybrid_area_ledger
6. random_same_count

For each mode:
- select target prune fraction
- preserve high-rent / high-protect primitives
- avoid high-debt regions
- write candidate JSON
- apply to checkpoint
- verify schema
- render/evaluate if requested

First real test:
- parking clean 22k
- prune fractions 65%, 70%, 75%, 80%
- compare immediately after compaction to R47/R53 area-only where possible.

Outputs:
- compact checkpoint directories
- ledger_compaction_candidates.json
- ledger_compaction_report.md
- immediate render/geometry metrics where run

Append research-log entry.
Commit and push.

Gate:
PASS only if at least one ledger-based mode produces a valid renderable compact checkpoint and area-only reproduction remains available as a control.
```

---

# Prompt L5 — Ledger compaction recovery sweep

```text
Stage L4 must be PASS.

Mission:
Run strict topology-frozen recovery for ledger-compacted checkpoints and compare against R53 area-only.

Runs:
- parking clean 22k
- modes:
  - area_smallest
  - ledger_rent_lowest
  - ledger_donor_highest
  - ledger_auditor_ranked
  - hybrid_area_ledger
  - random_same_count
- prune fractions:
  - 65%
  - 70%
  - 75%
  - 80%

Use:
- --freeze_topology_updates
- --skip_restricted_delaunay
- recovery 22k -> 26k
- same evaluation as R53

Create:
- scripts/car_model/meshsplatopt_run_ledger_compaction_recovery_sweep.py
- scripts/car_model/meshsplatopt_collect_ledger_compaction_recovery.py
- docs/car_model/ledger_stageL5_compaction_recovery_sweep_report.md

Required comparisons:
- best ledger vs R53.01
- best ledger vs R55.01
- best ledger vs R48.01
- random same-count vs ledger
- compaction-only vs recovery

Hard decision:
- If ledger selector beats area-only at same triangle count, promote Surface Ledger as load-bearing.
- If ledger selector ties but gives better protection / repair diagnostics, report as auxiliary.
- If ledger selector loses, keep area-only as baseline and use ledger for repair transactions only.

Append research-log entry.
Commit and push.

Gate:
PASS only if all completed rows are fairly compared and no overclaim is made.
```

---

# Prompt L6 — Budget-neutral giant-hole repair benchmark

```text
Stage L5 must be PASS or SOFT_PASS.

Mission:
Demonstrate the most exciting capability: repair a giant hole while keeping total topology fixed or reduced.

This is the key "not just pruning" benchmark.

Create:
- scripts/car_model/meshsplatopt_run_budget_neutral_giant_void_benchmark.py
- scripts/car_model/meshsplatopt_collect_budget_neutral_giant_void_results.py
- docs/car_model/ledger_stageL6_budget_neutral_giant_void_report.md

Benchmark types:
1. synthetic giant parking-ground void
2. injected real checkpoint void:
   - choose a ground-like region in parking or courtyard;
   - remove a patch to create a known void;
   - keep pre-damage checkpoint as oracle only for evaluation;
   - do not use oracle for proposal selection.
3. optional natural real void if defect miner finds one.

Methods:
- no repair
- donor-only compaction
- classical fill without donor accounting
- plane fill without render/free-space gate
- CSEF fill recipient-only
- Ledger budget-neutral fill:
  donor low-rent topology -> recipient giant-void patch
- Ledger budget-neutral fill + topology-frozen recovery
- prior-only diagnostic fill

Metrics:
- net triangle delta
- repaired void area
- boundary closure
- render PSNR/SSIM/LPIPS
- sparse geometry metrics
- changed pixel ratio
- free-space violation
- oracle surface error for injected benchmark only
- hallucination/prior-only flag

Rules:
- Budget-neutral fill must have net topology delta <= 0 or a clearly stated topology cap.
- Unknown/unobserved void must reject in normal mode.
- Prior-only fill cannot be headline.

Append research-log entry.
Commit and push.

Gate:
PASS only if budget-neutral fill beats no repair and classical fill on at least synthetic and injected-real benchmarks without net topology growth.
```

---

# Prompt L7 — Real-scene ledger transactions

```text
Stage L6 must be PASS.

Mission:
Run the full ledger transaction system on real public scenes.

Scenes:
- parking_phone_tiny
- courtyard
- bonsai
- optional extra COLMAP scene

Pipeline:
1. Build Surface Ledger.
2. Mine debts and donors.
3. Train/load amortized auditor.
4. Plan ledger transactions.
5. Apply transaction candidates to checkpoint copies.
6. Run render-backed counterfactual gate.
7. Run topology-frozen recovery for accepted transactions.
8. Run equal-budget no-transaction controls.
9. Collect metrics.

Create:
- scripts/car_model/meshsplatopt_run_real_ledger_transactions.py
- scripts/car_model/meshsplatopt_collect_real_ledger_transactions.py
- docs/car_model/ledger_stageL7_real_scene_transactions_report.md

Transaction types to test:
- donor-only compact
- budget-neutral repair
- snap transaction
- fill transaction
- compact+recovery
- compact+repair+recovery

Required decision:
- If real budget-neutral repairs improve a scene, promote them.
- If not, keep ledger transaction system as diagnostic and keep clean-to-compact as headline.

Append research-log entry.
Commit and push.

Gate:
PASS only if at least one real scene produces an accepted transaction that beats equal-budget control, or the report explicitly demotes real repair transactions.
```

---

# Prompt L8 — Final narrative and figure rewrite

```text
Stage L5 must be PASS; L6/L7 should have decisions.

Mission:
Rewrite final paper assets around Surface Evidence Ledger.

Create/update:
- docs/car_model/final_surface_ledger_paper_story.md
- scripts/car_model/final_make_surface_ledger_figures.py
- outputs/carnet/meshsplatopt/final_surface_ledger_assets/

Figures:
1. "Every surface must pay rent" overview:
   overcomplete mesh -> ledger audit -> donors/debts -> certified transaction -> compact recovered mesh.
2. Ledger table visualization:
   high-rent keep, low-rent donor, high-debt recipient.
3. Topology credit transfer:
   redundant triangles removed from easy region, budget invested in defect / or saved as compaction.
4. Counterfactual auditor:
   proposed transaction -> temporary edit -> render/geometry gate -> commit/rollback.
5. Pareto:
   clean long, area-only, ledger selector, random, PRISM, sparse-only.
6. Giant void:
   no repair, classical fill, ledger budget-neutral fill, prior-only rejection if L6 passes.

Tables:
- Main compact-recovery table.
- Ledger selector ablation.
- Auditor ablation.
- Transaction ablation.
- Negative results.

Append research-log entry.
Commit and push.

Gate:
PASS only if the paper story can be understood as a single ledger/audit idea, not a module list.
```

---

# Prompt L9 — Surface Ledger go/no-go

```text
Stage L8 must be PASS.

Mission:
Make a final decision on whether Surface Ledger is the main NeurIPS story.

Write:
- docs/car_model/ledger_stageL9_go_no_go.md

Decision categories:
1. LEDGER_MAIN_READY
   - ledger selector beats area-only or real budget-neutral repair succeeds;
   - at least two scenes have strong compact-recovery results;
   - ablations show ledger/auditor/gate/freeze are load-bearing.

2. LEDGER_STORY_READY_BUT_AREA_BASELINE_MAIN
   - area-only still wins, but ledger provides useful repair diagnostics;
   - paper can still be about certified compact-recovery, with ledger as framework.

3. COMPACT_RECOVERY_ONLY
   - ledger does not improve over area;
   - use R53-style clean-to-compact + freeze recovery as main story;
   - demote ledger to future work.

4. NOT_NEURIPS_READY
   - only one scene works or baselines are unfair.

The report must quote exact tables and decisions.

Append research-log entry.
Commit and push.

Gate:
PASS only if the go/no-go decision is brutally honest.
```

---

## 5. What to tell reviewers if this works

If L5 works:

> Area pruning showed the first compact-recovery result, but it was only a crude proxy for topology rent. Surface Ledger improves this by scoring primitives according to their evidence rent and topology tax, producing a stronger Pareto frontier.

If L6 works:

> MeshSplatOpt does not merely compress. It can perform budget-neutral repair: topology removed from low-rent regions is reinvested into high-debt defects such as giant ground voids, and every transaction is certified by counterfactual rendering and sparse geometry.

If L3 works:

> Because exact counterfactual validation is expensive, MeshSplatOpt learns an amortized auditor from prior counterfactual outcomes. The auditor proposes promising transactions, but final acceptance remains certified by actual gates.

If L5/L6 do not work:

> The ledger framework remains an interpretation and diagnostic layer. The honest final paper should center on clean-to-compact topology-frozen recovery, with Surface Ledger and repair transactions presented as future extensions or negative-results analysis.

---

## 6. Minimal exciting final story if only R53 remains strongest

Even if Ledger selectors do not beat area-only, use this simplified story:

> Mesh Splatting learns overcomplete scene surfaces. Instead of treating this as waste, we treat the overcomplete model as a teacher surface bank. MeshSplatOpt audits the bank, removes a large fraction of low-cost surface hypotheses, freezes topology, and recovers appearance. Surprisingly, the settled compact mesh can outperform the overcomplete teacher while using far fewer primitives. This reveals a new training principle: grow freely, then settle topology under counterfactual and geometry-aware recovery.

This is still much better than a module pile.

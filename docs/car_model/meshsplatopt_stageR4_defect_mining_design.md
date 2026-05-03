# MeshSplatOpt Stage R4 Defect Mining Design

Date: 2026-05-02

## Goal

Turn CSEF diagnostics into actionable repair regions. R4 still does not edit geometry. It produces defect records with severity, confidence, affected faces, allowed edit types, evidence summaries, uncertainty summaries, and explicit no-repair reasons when evidence is insufficient.

## Defect Types

The miner emits:

- `FLOATER_COMPONENT`
- `LOCAL_DENT`
- `ROUGH_BROKEN_SURFACE`
- `VEHICLE_DISCONTINUITY`
- `GROUND_WALL_MISALIGNMENT`
- `SMALL_BOUNDARY_HOLE`
- `GIANT_GROUND_VOID`
- `UNKNOWN_UNOBSERVED_VOID`
- `APPEARANCE_GHOSTING_REGION`

R4 implements geometry/CSEF-based mining for floaters, boundary holes, giant ground voids, and unknown void hints. Other types are present in the contract and will be activated by later evidence modules.

## Giant Void Logic

The first implementation uses three signals:

1. boundary-loop support from CSEF boundary scores;
2. neighboring surface support from connected component area and face count;
3. optional ground/coverage hints passed by smoke or future scene modules.

The separation is:

- `GIANT_GROUND_VOID`: large boundary support plus neighboring surface/ground support;
- `SMALL_BOUNDARY_HOLE`: boundary support but area below giant threshold;
- `UNKNOWN_UNOBSERVED_VOID`: a void hint with weak/no boundary support and insufficient coverage.

Unknown voids are emitted with no allowed repair in normal mode.

## Outputs

- `defects.json`
- `defects_summary.csv`
- `defect_mining_report.md`

Debug PLY/OBJ labels are optional and deferred.

## Gate

`PASS` requires a synthetic parking-ground mesh with a large rectangular missing ground patch to become `GIANT_GROUND_VOID`, while a synthetic out-of-trajectory void with no boundary support becomes `UNKNOWN_UNOBSERVED_VOID`.

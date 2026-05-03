# MeshSplatOpt Stage R9 Object-Prior Repair Design

Date: 2026-05-02

## Goal

Integrate SP-CarNet object posterior as an optional proposal generator for vehicle-region repair without allowing the object prior to commit geometry directly.

## Contract

The object prior may propose:

- vehicle protect masks;
- floater delete candidates;
- surface snap candidates;
- discontinuity fill candidates;
- boundary split candidates.

Every proposal records:

```text
prior_proposes_evidence_disposes = true
requires_scene_counterfactual_validation = true
```

## Safety Rules

- Posterior uncertainty downweights all aggressive proposals.
- Low canonicalization confidence disables object-prior fill.
- Object-prior proposals cannot bypass scene gates.
- Prior-only vehicle fills are proposal candidates, not committed repairs.

## Gate

`PASS` requires a synthetic car-like box with a missing side panel to produce a bounded fill/snap/protect package, while an uncertain prior case emits only protect/prune or no proposal.

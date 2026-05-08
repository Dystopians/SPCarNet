# ECSR Phase-C Candidate Preflight

This is a pre-contraction certificate pass. It does not modify
checkpoints and does not claim final acceptance. It only verifies
that Phase-B candidates have train-only fitting and policy-val
surface support masks before any topology or appearance recovery
experiment is allowed to spend GPU time.

## Fixed Rules

- min fitting-train support views: `1`
- min policy-val support views: `1`
- min policy-val support pixels: `64`
- held-out test usage: `none`
- checkpoint edits: `none`

## Aggregate

| metric | value |
|---|---|
| candidates checked | 123 |
| preflight pass | 21 |
| preflight reject | 102 |
| contraction preflight pass | 13 |
| attribute-recovery preflight pass | 8 |

## Per-Scene

| scene | checked | pass | reject | contraction pass | attribute pass | certificate dir |
|---|---|---|---|---|---|---|
| bicycle | 20 | 3 | 17 | 3 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/bicycle/certificates` |
| flowers | 11 | 1 | 10 | 0 | 1 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/flowers/certificates` |
| garden | 16 | 1 | 15 | 1 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/garden/certificates` |
| stump | 14 | 1 | 13 | 1 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/stump/certificates` |
| treehill | 7 | 0 | 7 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/treehill/certificates` |
| room | 15 | 2 | 13 | 1 | 1 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/room/certificates` |
| counter | 15 | 3 | 12 | 2 | 1 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/counter/certificates` |
| kitchen | 12 | 5 | 7 | 2 | 3 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/kitchen/certificates` |
| bonsai | 13 | 5 | 8 | 3 | 2 | `outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight/bonsai/certificates` |

## Interpretation

Candidates that pass this preflight are not yet accepted ECSR edits.
They are merely eligible for the next expensive step: static topology
smoke testing, local rendering certificates, and policy-val before/after
metrics. Rejected candidates should not be manually revived.


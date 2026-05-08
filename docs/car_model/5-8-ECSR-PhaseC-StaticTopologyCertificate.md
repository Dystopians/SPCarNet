# ECSR Phase-C Static Topology Certificate

This is the first static Layer-1 certificate for ECSR contraction
candidates. It simulates a conservative edge-collapse operator on
the real checkpoint topology and rejects candidates that would create
local zero-area faces, normal flips, invalid edge lengths, or no
triangle reduction. No checkpoint is modified and no held-out test
view is used.

## Aggregate

| metric | value |
|---|---|
| candidates checked | 21 |
| static pass | 7 |
| static reject | 14 |
| contraction static pass | 3 |

## Per-Scene

| scene | checked | static pass | static reject | contraction pass | certificate dir |
|---|---|---|---|---|---|
| bicycle | 3 | 2 | 1 | 2 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/bicycle/certificates` |
| flowers | 1 | 1 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/flowers/certificates` |
| garden | 1 | 0 | 1 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/garden/certificates` |
| stump | 1 | 0 | 1 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/stump/certificates` |
| treehill | 0 | 0 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/treehill/certificates` |
| room | 2 | 1 | 1 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/room/certificates` |
| counter | 3 | 0 | 3 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/counter/certificates` |
| kitchen | 5 | 2 | 3 | 1 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/kitchen/certificates` |
| bonsai | 5 | 1 | 4 | 0 | `outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate/bonsai/certificates` |

## Interpretation

A PASS_STATIC candidate is still not an accepted ECSR edit. It is only
eligible for materialized checkpoint smoke rendering and policy-val
before/after metrics. This prevents expensive long runs on candidates
that are already topologically unsafe.


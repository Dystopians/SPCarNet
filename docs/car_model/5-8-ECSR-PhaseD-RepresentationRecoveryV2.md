# ECSR Phase-D Representation Recovery V2

Date: 2026-05-08

This log records the next representation-level recovery attempt after the
failed attribute-only and direct DC-delta pilots. The goal was to move beyond
renderer-side ELA by attaching recovery state to the MeshSplatting
representation itself.

## Added Interfaces

| script | purpose | topology |
|---|---|---|
| `scripts/car_model/ecsr_apply_surface_residual_ridge_delta.py` | Solves a bounded ridge/smoothness SH-DC residual over train-evidence surface supports. | unchanged |
| `scripts/car_model/ecsr_apply_surface_residual_microfacets.py` | Adds a tiny set of train-evidence residual carrier triangles attached to stable high-error faces. | adds audited microfacets |

Both operators use train-only surface evidence, write checkpoint-level state,
emit JSON/Markdown audits, and do not inspect held-out test views during
materialization.

## Protocol

Two variants were tested:

1. `surface_residual_ridge_delta_v1`
   - source: archived SOR compact checkpoints
   - evidence: original Phase-A surface evidence
   - scenes: `bicycle`, `flowers`, `treehill`, `garden`

2. `surface_residual_ridge_delta_v2_phasej_model`
   - source: Phase-J selected `ratio_0200` checkpoints
   - evidence: rebuilt directly on those Phase-J checkpoint topologies
   - scenes: `bicycle`, `flowers`

The second variant is the fairer integration test because face IDs are aligned
with the current promoted Phase-J topology.

## Ridge Delta Findings

Bare checkpoint ridge deltas are safe but too weak. On four outdoor scenes,
V1 changed compact-only test metrics only at numerical-noise scale:

| scene | dPSNR vs compact | dSSIM vs compact | dLPIPS vs compact |
|---|---:|---:|---:|
| bicycle | +0.000038 | +0.000001 | -0.000000 |
| flowers | +0.000067 | -0.000001 | -0.000000 |
| treehill | +0.000031 | -0.000001 | +0.000002 |
| garden | +0.000242 | +0.000000 | -0.000000 |

When composed with the Phase-J-style ELA adapter on the source checkpoint,
the method beat the older source SOR+ELA row on the two tested scenes, but it
did not consistently beat the current Phase-J selected method.

| scene | dPSNR vs Phase-J | dSSIM vs Phase-J | dLPIPS vs Phase-J |
|---|---:|---:|---:|
| bicycle source+ridge+ELA | +0.007858 | +0.001040 | -0.001458 |
| flowers source+ridge+ELA | -0.105659 | -0.009450 | +0.013393 |

After rebuilding evidence on the Phase-J selected checkpoints and applying the
ridge delta there, the result became essentially neutral on `bicycle` and
clearly harmful on `flowers`:

| scene | dPSNR vs Phase-J | dSSIM vs Phase-J | dLPIPS vs Phase-J |
|---|---:|---:|---:|
| bicycle Phase-J-aligned ridge+ELA | +0.000011 | +0.000000 | -0.000001 |
| flowers Phase-J-aligned ridge+ELA | -0.108633 | -0.009351 | +0.012901 |

## Microfacet Findings

The microfacet operator adds a small number of residual carrier triangles:

| scene | added triangles | added vertices | degenerate | invalid |
|---|---:|---:|---:|---:|
| bicycle | 41 | 123 | 0 | 0 |
| flowers | 29 | 87 | 0 | 0 |

As a bare representation update, it also had negligible global effect:

| scene | dPSNR vs Phase-F bare compact | dSSIM | dLPIPS |
|---|---:|---:|---:|
| bicycle | -0.000071 | -0.000001 | -0.000003 |
| flowers | -0.000025 | -0.000001 | -0.000000 |

## Decision

`REJECT_AS_FINAL_METHOD`.

The new interfaces are useful and auditable, but neither V2 mechanism is strong
enough to promote as the paper's representation-level recovery contribution.
The repeated failure pattern is now specific:

- Sparse residual supports cover only a tiny part of the full image metric.
- Per-face residual aggregates lack barycentric/per-pixel residual detail, so
  SH-DC relocation collapses to a very weak local color nudge.
- Adding a handful of microfacets is topologically valid but does not carry
  enough rendering mass to move global or qualitative metrics.
- Phase-J's current gains remain dominated by train-only ELA, not by persistent
  representation state.

## Rich Evidence Interface Update

After the V2 rejection, the Surface Evidence Cache was extended so the next
representation-level method no longer has to infer residual structure from
per-face averages alone.

Implemented fields in `scripts/car_model/ecsr_build_surface_evidence_cache.py`:

- always stored with `--save_view_npz`: `face_id`, `residual_l1`, `texture`,
  `alpha`, `depth`, `normal`;
- optional with `--save_residual_rgb`: per-pixel `residual_rgb`;
- optional with `--save_rgb`: per-pixel `rgb_render` and `rgb_gt`;
- summary metadata: `per_view_npz_fields` and `barycentric_available`.

Smoke validation:

- command: rich cache smoke on `bicycle` Phase-J `ratio_0200` checkpoint;
- output:
  `outputs/carnet/meshsplatopt/ecsr_phase_d/surface_evidence_rich_smoke/bicycle/`;
- verified NPZ fields:
  `face_id`, `residual_l1`, `texture`, `alpha`, `depth`, `normal`,
  `residual_rgb`;
- current `barycentric_available`: `False`.

This does not by itself improve metrics. It closes the missing data-interface
piece needed for a stronger Phase-D method: residual fitting can now inspect
per-pixel residual vectors, normals, depth, alpha, and face IDs jointly instead
of relying only on aggregated face statistics.

## Next Required Upgrade

The next representation-level attempt should not keep increasing these local
DC edits. It should use the rich evidence interface above and add a real
fitting/validation mechanism:

1. Add barycentric coordinates or an equivalent stable local surface coordinate
   to the per-view cache when the renderer exposes it.
2. Fit per-face or per-cluster residual basis functions from fitting-train
   views and validate them on policy-val train views before materialization.
3. Apply residual relocation on Phase-B clusters rather than isolated top-error
   faces.
4. Promote only if the representation-attached result beats Phase-J on
   policy-val and then holds on the final test split.

Until that upgrade exists, the honest paper position is that Phase-J closes the
current selected-scene RGB-plus-compactness claim, while Phase-D
representation-level recovery remains the main unresolved research bottleneck.

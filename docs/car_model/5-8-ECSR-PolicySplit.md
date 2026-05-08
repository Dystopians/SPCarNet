# ECSR Phase-A/B Policy Split

This file fixes a deterministic fitting/policy-val split for the cached
Phase-A/B train views. It is sufficient for local certificate smoke tests
on the current cache, but it is not the final full-train split required
before long Phase-C/D optimization.

- seed: `20260508`
- policy-val fraction: `0.5`
- split scope: `phase_ab_cached_train_views`
- held-out test usage: `none`

| scene | cached views | fitting | policy-val | fitting train indices | policy-val train indices |
|---|---|---|---|---|---|
| bicycle | 8 | 4 | 4 | 0, 6, 12, 36 | 18, 24, 30, 42 |
| flowers | 8 | 4 | 4 | 12, 30, 36, 42 | 0, 6, 18, 24 |
| garden | 8 | 4 | 4 | 18, 24, 30, 42 | 0, 6, 12, 36 |
| stump | 8 | 4 | 4 | 0, 6, 12, 18 | 24, 30, 36, 42 |
| treehill | 8 | 4 | 4 | 6, 12, 18, 24 | 0, 30, 36, 42 |
| room | 8 | 4 | 4 | 0, 24, 36, 42 | 6, 12, 18, 30 |
| counter | 8 | 4 | 4 | 0, 12, 18, 42 | 6, 24, 30, 36 |
| kitchen | 8 | 4 | 4 | 0, 18, 30, 42 | 6, 12, 24, 36 |
| bonsai | 8 | 4 | 4 | 0, 12, 30, 36 | 6, 18, 24, 42 |

Per-scene JSON files are saved under
`outputs/carnet/meshsplatopt/ecsr_policy_splits/phase_ab_cached_views`.

Before Phase-C/D full optimization, regenerate this with all train views
from the scene loader and keep held-out test views excluded from all
candidate, strength, crop, and rollback decisions.

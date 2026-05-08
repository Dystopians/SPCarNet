# ECSR Full-Train Policy Split

This is the deterministic fitting/policy-val split for Phase-C/D
candidate acceptance. It is generated from the scene loader's full train
camera list. Held-out test views are not used for candidate generation,
strength selection, crop selection, rollback, or acceptance.

- split scope: `full_train_scene_loader`
- held-out test usage: `none`

| scene | train views | fitting | policy-val | seed | split JSON |
|---|---|---|---|---|---|
| bicycle | 169 | 135 | 34 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/bicycle/policy_split.json` |
| flowers | 151 | 121 | 30 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/flowers/policy_split.json` |
| garden | 161 | 129 | 32 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/garden/policy_split.json` |
| stump | 109 | 87 | 22 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/stump/policy_split.json` |
| treehill | 123 | 98 | 25 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/treehill/policy_split.json` |
| room | 272 | 218 | 54 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/room/policy_split.json` |
| counter | 210 | 168 | 42 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/counter/policy_split.json` |
| kitchen | 244 | 195 | 49 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/kitchen/policy_split.json` |
| bonsai | 255 | 204 | 51 | 20260508 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train/bonsai/policy_split.json` |

The earlier cached-view split is retained only for smoke tests.

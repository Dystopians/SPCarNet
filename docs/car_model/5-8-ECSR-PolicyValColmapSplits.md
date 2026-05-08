# ECSR Policy-Val COLMAP Split Files

These split files convert the deterministic full-train fitting/policy-val
records into the native COLMAP loader format. They are for Phase-D
policy certificates: fitting views are loader train views, policy-val
views are loader test views, and the original LLFF held-out test views
are dropped so they cannot affect candidate acceptance.

| scene | fitting train | policy-val | dropped | missing train | missing val | split file |
|---|---|---|---|---|---|---|
| bicycle | 135 | 34 | 25 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/bicycle/split_file.json` |
| flowers | 121 | 30 | 22 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/flowers/split_file.json` |
| garden | 129 | 32 | 24 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/garden/split_file.json` |
| stump | 87 | 22 | 16 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/stump/split_file.json` |
| treehill | 98 | 25 | 18 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/treehill/split_file.json` |
| room | 218 | 54 | 39 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/room/split_file.json` |
| counter | 168 | 42 | 30 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/counter/split_file.json` |
| kitchen | 195 | 49 | 35 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/kitchen/split_file.json` |
| bonsai | 204 | 51 | 37 | 0 | 0 | `outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file/bonsai/split_file.json` |

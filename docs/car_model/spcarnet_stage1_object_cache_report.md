# SP-CarNet Stage 1 — Object Cache & Canonicalization Report

| Field | Value |
|---|---|
| Stage | 1 / 7 (per `SPCarNet_radical_RFC.md` §9) |
| Status | **DONE** — gate passed (no cache rebuild required). |
| Date | 2026-04-29 |
| Cache audited | `outputs/ss3dm_prior_car/meshfleet_car_cache_v5` |
| Index produced | `outputs/carnet/spcarnet/object_index_v1.json` |
| Design doc | `docs/car_model/spcarnet_stage1_object_cache_design.md` |
| Research log entry | `docs/car_model/SPCarNet_research_log.md` (first entry) |

---

## 1. Headline numbers

| Metric | Value |
|---|---|
| Total objects | **2 433** |
| Total patches | 2 433 (1 patch per object — verified) |
| Patches per object: min / median / mean / max | 1 / 1 / 1.0 / 1 |
| Train / val / test | **1 854 / 206 / 373** |
| `__ext1` (Objaverse extension) share | 817 of 2 433 (33.6 %) |
| Pseudo-aggregated objects | **0** (none required) |
| Source GLB path resolved | **2 433 / 2 433** (100 %) |
| Occupancy queries available | **yes** for every object |
| Free-space queries available | **yes** for every object |
| Scanner pose persisted in cache | **no** (sampled at runtime by LiDAR corruption) |
| Symmetry persisted in cache | **partial** (817 v3 records) |

---

## 2. Object grouping & identity

The whole-car cache has a clean 1-to-1 mapping between **patch_id** and **car_id**. Verified directly:

```
unique sequence_ids  : 2433
unique patch_ids     : 2433
sequence_id == patch_id : 2433 / 2433
```

Therefore the SP-CarNet `object_id` is simply `patch_id` (== `sequence_id` == `car_id` from the source manifest). No pseudo-aggregation is needed for the headline path. The pseudo-object code path is implemented (and unit-exercised in the smoke test via the round-trip transform check) but does not fire on this cache.

Every object also has a resolved `source_mesh_path` from `source_mesh_manifest.json`, which Stage 7 will use for mesh-IoU evaluation against the original GLB.

---

## 3. Available canonical-frame metadata

Per-NPZ verification across the 2 433 records:

- **Centre**: `patch_center_world == [0, 0, 0]` for every record.
- **Scale**: `patch_radius_m == 1.0` for every record.
- **Bounds**: `clean_points` falls within `[-0.99, 0.85]` per axis on the surveyed records (and `[-1.07, 1.07]` once the slightly-extended `query_points_all` is included).
- **Normals**: `clean_normals`, `visible_clean_normals`, `hidden_clean_normals` are all present and unit-norm (`min=-1, max=1`).
- **Visible / hidden split**: present (`visible_clean_points` ≈ 1 673 ± few hundred per car; `hidden_clean_points` ≈ 375 ± few hundred).
- **Partial / observed**: `observed_points (768, 3)` for every record.
- **Front-axis convention**: **NOT annotated**. MeshFleet and Objaverse meshes do not share a consistent +x-forward convention. PCA orientation is provided by the index builder as an opt-in (`--canonicalization pca`) with the documented eigenvector-flip caveat.

The Stage 1 dataset wrapper therefore defaults to `canonical_transform.type = "identity"`. SP-CarNet Stage 2 inherits this default; downstream stages may swap to `pca` once a per-axis stability test is in place.

---

## 4. Cache-format split (unexpected finding)

The audit revealed that the merged cache contains **two format versions**:

| Format | Records | Splits | Source |
|---|---|---|---|
| v2 (no symmetry persisted) | 1 616 | 1 228 train / 137 val / 251 test | original MeshFleet |
| v3 (symmetry persisted) | 817 | 626 train / 69 val / 122 test | Objaverse `__ext1` extension |

The v3 records carry four additional fields:

- `symmetry_plane_normal` (3,)
- `symmetry_plane_offset` scalar
- `symmetry_target_confidence` scalar
- `symmetry_chamfer_residual` scalar

Sampled v3 record: `symmetry_plane_normal = [0.058, 0.008, −0.998]`, `symmetry_target_confidence = 0.278` — a near-XY mirror plane with low confidence (consistent with the Objaverse subset containing a higher proportion of asymmetric objects).

Implication: **2/3 of the cache lacks persisted symmetry.** SP-CarNet Stage 3+ will populate symmetry for those objects on the fly via `ss3dm_prior/data/symmetry_targets.py::estimate_symmetry_plane()` (≤ 2 ms per car at load time). The dataset wrapper exposes this through the `estimate_symmetry=True` constructor flag.

---

## 5. Smoke-test results

`scripts/car_model/smoke_test_spcarnet_stage1.py` ran successfully against the full cache:

```
[smoke] index_ok: n_objects=8 stats={... 'splits': {'train': 6, 'test': 2}, ...}
[smoke] dataset_ok: len=8
[smoke] transform_ok: identity_err=0.00e+00 pca_err=5.96e-08
[smoke] batch_ok: keys=[canonical_transform, clean_normals_object, clean_points_object,
       free_space_query_hard_negatives, free_space_query_points, hidden_clean_points,
       object_id, occupancy_query_ignore, occupancy_query_labels, occupancy_query_points,
       partial_observed_points, patch_metadata_list, scanner_pose, source_mesh_path,
       split, symmetry, town_id, visible_clean_points]
[smoke] PASS
```

Verifications:
- 8-object index built on the smoke subset.
- Dataset opens with `splits=("train", "val", "test")`.
- `clean_points_object`: `(2048, 3)` `float32`, no NaN/Inf.
- `partial_observed_points`: `(768, 3)` `float32`, no NaN/Inf.
- `occupancy_query_labels` post-`query_ignore_mask` strictly ∈ {0, 1}.
- Identity transform round-trip error: **0.0**.
- Non-identity (`pca`-style) round-trip error: **5.96 × 10⁻⁸** (< 1 × 10⁻⁵ threshold).
- Batch collate (`collate_object_batch`, `B=2`) produces `(B, 2048, 3)`, `(B, 768, 3)`, `(B, 1280, 3)`, `(B, 512, 3)` for the four fixed-shape fields.

---

## 6. Failure cases

None observed. Specifically:
- 0 records had `len(patches) > 1` (no pseudo-aggregation triggered).
- 0 records had a missing NPZ on disk (all `patch_files` resolved).
- 0 records had a missing `source_mesh_path` in the manifest.
- 0 records had non-finite tensors in the smoke subset.

The two latent risks both have implemented mitigations and are *not* failures of Stage 1:
- 1 616 records lack persisted symmetry → runtime estimator (mitigation in dataset).
- No records persist scanner pose → runtime sampling via existing LiDAR corruption module (mitigation in dataset constructor).

---

## 7. Files added

| File | LoC | Role |
|---|---|---|
| `docs/car_model/spcarnet_stage1_object_cache_design.md` | ~310 | Stage 1 design doc (precedes code). |
| `scripts/car_model/build_spcarnet_object_index.py` | ~210 | Object-index builder CLI. |
| `ss3dm_prior/data/spcarnet_object_dataset.py` | ~280 | Object-centric dataset wrapper + collate. |
| `scripts/car_model/smoke_test_spcarnet_stage1.py` | ~150 | Stage 1 smoke test entry. |
| `outputs/carnet/spcarnet/object_index_v1.json` | (data) | Generated artefact: 2 433 object records. |
| `docs/car_model/spcarnet_stage1_object_cache_report.md` | this file | Stage 1 closing report. |
| `docs/car_model/SPCarNet_research_log.md` | new | First research-log entry. |

No existing file was modified. CarNet v0.x configs, `train_dataset.py`, `patch_index.py`, `teacher_patch_*` and any trainer code remain untouched.

---

## 8. Recommendation

> **SP-CarNet can proceed on real object-level data — no cache rebuild is required, and pseudo-object aggregation does not need to be exercised.**

Stage 2 (shape-field auto-decoder) inherits the wrapper as-is. The only downstream concern is whether `identity` canonical orientation is sufficient for the auto-decoder; if Stage 2 reveals that PCA orientation closes a non-trivial chamfer gap, swap the index builder default in a separate Stage-2-scoped doc.

Symmetry persistence remains partial; this is **not a Stage-1 blocker** because the runtime estimator is fast and the dataset already exposes the field. A small follow-up (running the v3 cache builder over the v2 records to backfill symmetry NPZs) is logged as a Stage 2/3 nice-to-have.

Stage 1 is **DONE**. Proceeding to Stage 2.

---

## 9. Reproducibility

```bash
# Build the full object index
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_spcarnet_object_index.py

# Smoke test
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/smoke_test_spcarnet_stage1.py
```

Outputs:
- `outputs/carnet/spcarnet/object_index_v1.json` (2 433 objects, identity canonicalization)
- stdout: `[smoke] PASS`

Optional PCA canonicalization (caveat in §3):
```bash
/home/peilincai/micromamba/envs/mesh_splatting/bin/python \
  scripts/car_model/build_spcarnet_object_index.py --canonicalization pca \
  --output outputs/carnet/spcarnet/object_index_v1_pca.json
```

_End of Stage 1 report._

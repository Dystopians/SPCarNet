# SP-CarNet Stage 1 — Object Cache & Canonicalization Audit (Design)

| Field | Value |
|---|---|
| Stage | 1 / 7 (per `SPCarNet_radical_RFC.md` §9) |
| Status | DESIGN (precedes implementation) |
| Date | 2026-04-29 |
| Cache audited | `outputs/ss3dm_prior_car/meshfleet_car_cache_v5` |
| Source | MeshFleet v4 + Objaverse vehicle extension (`__ext1`) |

---

## 0. Purpose

Stage 1 must answer one question: **can SP-CarNet train on real object-level data without rebuilding the cache?**

The four sub-questions are:
1. What identifiers exist in the current cache to group patches by car / object / mesh?
2. Can we recover whole-object point clouds or meshes from existing MeshFleet / Objaverse records?
3. What canonicalization metadata (orientation, scale, bbox, normals, scanner pose, visible/hidden split) is available?
4. What is missing and how do we add object-centric data without breaking patch-centric training?

A short answer: **the existing cache is already organised one-NPZ-per-car**, and each NPZ is already in a canonical unit-bounding-cube. The Stage 1 work is therefore **wrapping**, not rebuilding.

---

## 1. Audit findings

### 1.1 Identifier structure

Every record in `patch_index.jsonl` has:
- `patch_id` — 64-character SHA-256 hash, unique per car.
- `sequence_id` — equal to `patch_id` for whole-car caches (verified: 2433 of 2433 records).
- `town_id` — split bucket (`MeshFleetTrain`, `MeshFleetVal`, `MeshFleetTest`, plus `__ext1` variants).
- `tile_id`, `scale_id` — both 0; the patch-tiling subsystem is inactive on this cache.

In the source `source_mesh_manifest.json`:
- `car_id` — equals the `patch_id`.
- `assigned_split_name` — one of `train` / `val` / `test`.
- `file_identifier` — the original GLB URL (Objaverse) or MeshFleet metadata pointer.
- `local_path` — relative path under the GLB raw store.
- `bucket_suffix` — `""` for MeshFleet, `__ext1` for Objaverse.

**Conclusion**: the canonical object identifier is `car_id == patch_id == sequence_id`. Object grouping is **1 patch ↔ 1 object**; no pseudo-aggregation is required for the headline path. SP-CarNet will adopt `car_id` as its `object_id`.

### 1.2 Whole-object data availability

Each per-car NPZ (e.g. `0075272ea4...npz`) already stores the **complete object geometry** in canonical coordinates. Verified tensor schema (cache format v2, 2433 cars):

| Field | Shape | Dtype | Role |
|---|---|---|---|
| `clean_points` | `(2048, 3)` | f32 | Full clean surface sample — **the object point cloud**. |
| `clean_normals` | `(2048, 3)` | f32 | Per-point normals. |
| `visible_clean_points` | `(N_v, 3)` (mean ~1673) | f32 | Visibility-split surface (subset of clean). |
| `hidden_clean_points` | `(N_h, 3)` (mean ~375) | f32 | Occluded-side surface. |
| `visible_clean_normals`, `hidden_clean_normals` | matching | f32 | Per-side normals. |
| `observed_points` | `(768, 3)` | f32 | Partial / camera-sparse observation (the corrupted source signal). |
| `query_points_all` | `(1280, 3)` | f32 | Concatenated surface + free + unknown queries. |
| `query_labels_all` | `(1280,)` | i8 | 1 = surface, 0 = free, −1 / `query_ignore_mask` = unknown. |
| `query_ignore_mask` | `(1280,)` | bool | Skip-flag for occupancy supervision. |
| `surface_query_points` / `surface_query_labels` | `(512, 3)` / `(512,)` | f32 / i8 | Surface BCE supervision. |
| `free_query_points` / `free_query_labels` | `(512, 3)` / `(512,)` | f32 / i8 | Free-space BCE supervision. |
| `free_space_query_hard_negatives` | `(128, 3)` | f32 | Hard-negative free-space samples. |
| `unknown_query_points` | `(256, 3)` | f32 | Unknown / margin samples. |
| `patch_center_world` | `(3,)` | f32 | Always `[0, 0, 0]` — already centred. |
| `patch_radius_m` | scalar | f32 | Always `1.0` — already unit-radius. |
| `patch_metadata_json` | str | — | Raw JSON of patch construction parameters. |

**Conclusion**: SP-CarNet has direct access to whole-object clean / visible / hidden / observed point clouds plus **occupancy and free-space query points and labels** for every training object. Nothing needs to be reconstructed.

### 1.3 Canonical orientation, scale, bbox, normals

| Property | Status | Source |
|---|---|---|
| **Centre** | `(0, 0, 0)` | `patch_center_world` (verified across all 2433 records) |
| **Scale** | unit ball, radius 1 | `patch_radius_m` (constant 1.0) |
| **Empirical bbox** | within `[-1.07, 1.07]` per axis | observed in surveyed records (small overshoot from query points) |
| **Normals** | available per surface point | `clean_normals`, `visible_clean_normals`, `hidden_clean_normals` |
| **Canonical orientation** | **NOT explicitly annotated**: cars are *not* guaranteed front-axis-aligned | reverse-engineered from MeshFleet / Objaverse (no consistent +x convention) |

Because every cache file is already centred and unit-scaled, the Stage 1 wrapper treats the **object-centric canonical transform as the identity** by default. PCA-based orientation is provided as an *optional fallback* with a documented stability caveat (see §3.4).

### 1.4 Scanner pose

- **Not persisted** in the v2 cache.
- Used only by LiDAR-realistic corruption (`ss3dm_prior/data/corruptions.py`, added 2026-04-17 per `CarNet_v0_update_log.md`), which **samples a scanner pose at runtime** and applies beam-occlusion / incidence-angle dropout / range noise.
- Implication: when the LiDAR pipeline is active, the dataset can request a per-sample scanner pose by invoking the corruption module's pose sampler. When inactive, scanner pose is `None`.

The Stage 1 dataset wrapper exposes a `scanner_pose` field that is `None` by default and a free hook (callable) for downstream LiDAR integration in Stage 3+.

### 1.5 Visible / hidden split

- Already split per record: `visible_clean_points` (~75 % of surface, camera-supported) and `hidden_clean_points` (~25 %, cosine-threshold occluded).
- These are the canonical SP-CarNet supervision targets for the `L_surf` / `L_hidden` likelihood breakdown — see RFC §3.5.

### 1.6 Symmetry plane

- **NOT persisted** in v2 cache. The CarNet_v0 update log (2026-04-17, Phase 2) describes a v3 cache that adds `symmetry_plane_normal`, `symmetry_plane_offset`, `symmetry_target_confidence`, `symmetry_chamfer_residual` — but the active cache is **format version 2** (`patch_cache_format_version: 2`). The v3 cache was not built for the merged whole-car dataset.
- Workaround: SP-CarNet recomputes symmetry targets at runtime via the existing `ss3dm_prior/data/symmetry_targets.py::estimate_symmetry_plane()` (closed-form PCA + Chamfer fit). Cost: ~2 ms per car at load time.
- Stage 1 wrapper exposes a `symmetry` field; populated lazily on first access (cached on the dataset instance).

### 1.7 Splits

- Driven by `town_id` via `outputs/ss3dm_prior_car/meshfleet_car_cache_v5/split_meshfleet_car.yaml`.
- Maps: `MeshFleetTrain[*__ext1] → train`, `MeshFleetVal[*__ext1] → val`, `MeshFleetTest[*__ext1] → test`.
- 1228 + 626 = **1854 train**, 137 + 69 = **206 val**, 251 + 122 = **373 test**. Total 2433.
- Stage 1 inherits this split unchanged.

### 1.8 What is missing

| Missing field | Severity | Workaround |
|---|---|---|
| Canonical orientation (front-axis +x convention) | medium | Optional PCA fallback (§3.4) — flagged unstable on near-symmetric cars. |
| Scanner pose persisted at build time | low | Runtime sampling via existing LiDAR corruption module. |
| Symmetry plane persisted (cache v3) | low | Runtime estimation via existing `estimate_symmetry_plane()`. |
| Per-shape latent slot | n/a | Stage 2 introduces this (auto-decoder); not a cache concern. |
| Mesh on disk for MC reference | low | `source_mesh_manifest.json` records the GLB path; reload on demand for visualization eval. |
| Multi-patch object aggregation | not applicable | Cache is already 1 NPZ per car; pseudo-aggregation API kept for future schema variants. |

**Verdict**: SP-CarNet can proceed on real object-level data **without** any cache rebuild. PCA orientation is the only piece that requires a documented stability caveat.

---

## 2. Non-disruptive integration

The constraint from the user prompt: **do not break existing patch-centric datasets or configs**.

Approach:

1. **New module path**: `ss3dm_prior/data/spcarnet_object_dataset.py`. The existing `train_dataset.py`, `patch_index.py`, and `teacher_patch_*` modules are not edited.
2. **Read-only access** to the existing cache files. The new module loads NPZ via `numpy.load` and `patch_index.jsonl` via the existing `read_patch_index_jsonl`.
3. **New index file**: `outputs/carnet/spcarnet/object_index_v1.json`. The patch index, manifest, and split YAML are not modified.
4. **No new config**: the dataset is parameterised via constructor arguments. CarNet_v0.x configs remain untouched.
5. **No trainer change**: Stage 1 ships the dataset only. Stage 2 will introduce the auto-decoder trainer entry point.
6. **Backwards-compatible API**: the new dataset returns a strict-superset record. Old patch-centric callers continue to use `TeacherPatchTrainDataset` unchanged.

---

## 3. Stage 1 implementation specification

### 3.1 Object index file format (`object_index_v1.json`)

```json
{
  "version": 1,
  "source_patch_index": "<absolute path>",
  "source_split_yaml": "<absolute path>",
  "source_manifest": "<absolute path>",
  "stats": {
    "n_objects": 2433,
    "n_patches": 2433,
    "patches_per_object": {"min": 1, "median": 1, "mean": 1.0, "max": 1},
    "splits": {"train": 1854, "val": 206, "test": 373},
    "scanner_pose_available": false,
    "symmetry_persisted": false,
    "occupancy_queries_available": true,
    "free_space_queries_available": true,
    "cache_format_version": 2
  },
  "objects": [
    {
      "object_id": "<patch_id>",
      "split": "train|val|test",
      "town_id": "MeshFleetTrain__ext1",
      "patch_files": ["<absolute path .npz>"],
      "patch_ids": ["<patch_id>"],
      "canonical_transform": {
        "type": "identity",
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      },
      "source_mesh_path": "<glb path or null>",
      "n_clean_points": 2048,
      "patch_format_version": 2
    },
    ...
  ]
}
```

### 3.2 Object index builder

`scripts/car_model/build_spcarnet_object_index.py`. CLI args:

- `--patch_cache_dir` (default: meshfleet v5)
- `--split_config` (default: split yaml)
- `--manifest` (optional override; otherwise `<cache>/source_mesh_manifest.json`)
- `--output` (default: `outputs/carnet/spcarnet/object_index_v1.json`)
- `--canonicalization` ∈ {`identity`, `pca`, `mesh-axis`} (default: `identity`)
- `--limit N` (cap for smoke runs; default: full)

Algorithm (concise):

1. Read `patch_index.jsonl`. Group records by `sequence_id` (== `car_id`). Verify "1 patch per object"; if any object has > 1 patch, log a warning and aggregate (pseudo-object code path, §3.3).
2. Read `split_meshfleet_car.yaml`. Build `town → split` map.
3. Read `source_mesh_manifest.json`. Build `car_id → glb path` map.
4. For each object, compute canonical transform per `--canonicalization`:
   - `identity` (default): center=0, scale=1, rot=I.
   - `pca`: center = mean(`clean_points`), scale = max-axis std × 2, rot from `np.linalg.eigh(cov(clean_points))`. **Documented unstable on near-symmetric cars** (eigenvectors flip sign).
   - `mesh-axis`: reload GLB and use its native axes (deferred to Stage 2; current builder errors out with a clear message).
5. Emit the JSON document.

Smoke-friendly runtime: full build < 30 s on 2433 objects (no GLB loads).

### 3.3 Object-centric dataset class

`ss3dm_prior/data/spcarnet_object_dataset.py`. Class: `SPCarObjectDataset`.

Constructor:
```python
SPCarObjectDataset(
    object_index_path: str | Path,
    *,
    splits: Iterable[str] = ("train",),
    return_observed: bool = True,
    return_queries: bool = True,
    return_normals: bool = True,
    apply_canonical_transform: bool = True,
    estimate_symmetry: bool = False,
    scanner_pose_fn: Callable[[dict], np.ndarray] | None = None,
)
```

`__len__` returns the number of objects in the requested splits.

`__getitem__(i)` returns a dict:

```python
{
    "object_id": str,                       # car_id
    "split": str,                           # train / val / test
    "town_id": str,
    "clean_points_object": np.ndarray,      # (2048, 3) f32, canonical frame
    "clean_normals_object": np.ndarray,     # (2048, 3) f32, optional
    "visible_clean_points": np.ndarray,     # (N_v, 3) f32
    "hidden_clean_points": np.ndarray,      # (N_h, 3) f32
    "partial_observed_points": np.ndarray,  # (768, 3) f32, partial obs
    "occupancy_query_points": np.ndarray,   # (1280, 3) f32 (None if missing)
    "occupancy_query_labels": np.ndarray,   # (1280,) i8
    "occupancy_query_ignore": np.ndarray,   # (1280,) bool
    "free_space_query_points": np.ndarray,  # (512, 3) f32
    "free_space_query_hard_negatives": np.ndarray,  # (128, 3) f32
    "scanner_pose": np.ndarray | None,      # (4, 4) f32 from scanner_pose_fn
    "canonical_transform": dict,            # {center, scale, rotation, type}
    "symmetry": dict | None,                # {n, d, sigma} when estimated lazily
    "patch_metadata_list": list[dict],      # raw patch_index entries
    "source_mesh_path": str | None,
}
```

### 3.4 Pseudo-object fallback

If the index builder finds an object with **multiple patches** (does not occur on the current cache but the API is future-proof):

- Aggregate `clean_points` by concatenation, then deterministically downsample to 2048 via farthest-point sampling.
- Aggregate `visible_clean_points` / `hidden_clean_points` similarly with their target counts.
- For occupancy queries, **concatenate without downsampling** (queries from different patches probe different volumes).
- Set `canonical_transform.type = "pseudo-aggregated"` and emit a per-object warning entry to a `pseudo_object_warnings.jsonl` log next to the index.

PCA-orientation caveat (when `--canonicalization pca` is used):
- Eigenvalues are returned in ascending order by `np.linalg.eigh`. The largest eigenvector defines the longest axis (≈ car length), which is reasonable for cars but flips sign across the train set (no front-vs-back convention).
- A simple sign convention (`+x` direction = direction of larger mean point coordinate along the eigenvector) reduces flips but does not eliminate them on near-symmetric vehicles (e.g. some buses).
- The dataset documents this as a known instability and recommends `identity` (the default) for SP-CarNet Stage 2.

### 3.5 Canonical-transform invertibility

Stored as `(center, scale, rotation)` such that:

```
canonical = rotation @ ((world - center) / scale)
world     = scale * (rotation^T @ canonical) + center
```

The dataset exposes `apply_canonical_transform(points: np.ndarray) -> np.ndarray` and `invert_canonical_transform(points: np.ndarray) -> np.ndarray` as static / instance methods. Smoke test asserts round-trip equivalence to `< 1e-5` per-coordinate.

### 3.6 Smoke tests

`scripts/car_model/smoke_test_spcarnet_stage1.py`:

1. Build a tiny index with `--limit 8`.
2. Construct `SPCarObjectDataset(splits=["train", "val", "test"])`.
3. Iterate one batch (DataLoader with `batch_size=2`, `num_workers=0`, custom collate that stacks variable-length tensors with padding).
4. Assertions:
   - `clean_points_object.shape == (B, 2048, 3)` and dtype `float32`.
   - No `NaN` or `Inf` in any returned tensor.
   - `partial_observed_points.shape[1] == 768`.
   - Every `occupancy_query_labels` ∈ `{0, 1}` after applying `~query_ignore_mask`.
   - `canonical_transform` round-trip: `||invert(apply(p)) - p|| < 1e-5`.
5. Print a one-line summary on success.

Exit non-zero on any assertion failure.

---

## 4. Files added

Listed here for the implementation report to cross-check:

| File | Role |
|---|---|
| `docs/car_model/spcarnet_stage1_object_cache_design.md` | This document. |
| `scripts/car_model/build_spcarnet_object_index.py` | Index builder CLI. |
| `ss3dm_prior/data/spcarnet_object_dataset.py` | Dataset wrapper class. |
| `scripts/car_model/smoke_test_spcarnet_stage1.py` | Smoke test entry. |
| `outputs/carnet/spcarnet/object_index_v1.json` | Generated index artefact (data, not code). |
| `docs/car_model/spcarnet_stage1_object_cache_report.md` | Stage 1 closing report. |
| `docs/car_model/SPCarNet_research_log.md` | Research log, first entry created in Stage 1. |

---

## 5. Out of scope (deferred)

- Canonical orientation supervised by mesh axes: requires GLB reload pipeline, deferred to Stage 2 if Stage 2 demands it.
- Symmetry-plane persistence in cache: deferred until cache rebuild is otherwise warranted (the runtime estimator is fast enough at training time).
- Scanner pose persistence: deferred unless the LiDAR pipeline grows to need batch-deterministic poses (currently sampled per-step).
- Mesh-IoU metric eval against the original GLB: a Stage 7 (benchmark) concern.

_End of design._

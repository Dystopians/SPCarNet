"""SP-CarNet Stage 1 — object-centric dataset wrapper.

Reads the per-object index produced by ``scripts/car_model/build_spcarnet_object_index.py``
and exposes whole-object tensors (clean / visible / hidden / observed point clouds plus
occupancy queries and free-space queries) for the SP-CarNet shape-field auto-decoder.

Design contract: ``docs/car_model/spcarnet_stage1_object_cache_design.md``.

The wrapper is read-only with respect to the existing patch cache. It does not subclass
or mutate ``TeacherPatchTrainDataset``. Patch-centric callers continue to use that
class unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


_TENSOR_FIELDS = (
    "clean_points",
    "clean_normals",
    "visible_clean_points",
    "visible_clean_normals",
    "hidden_clean_points",
    "hidden_clean_normals",
    "observed_points",
    "query_points_all",
    "query_labels_all",
    "query_ignore_mask",
    "surface_query_points",
    "surface_query_labels",
    "free_query_points",
    "free_query_labels",
    "free_space_query_hard_negatives",
    "unknown_query_points",
)


@dataclass
class CanonicalTransform:
    """Affine canonical-frame transform.

    Forward:  canonical = rotation @ ((world - center) / scale)
    Inverse:  world     = scale * (rotation.T @ canonical) + center
    """

    type: str
    center: np.ndarray  # (3,)
    scale: float
    rotation: np.ndarray  # (3, 3)

    @staticmethod
    def from_dict(spec: dict[str, Any]) -> "CanonicalTransform":
        return CanonicalTransform(
            type=str(spec.get("type", "identity")),
            center=np.asarray(spec["center"], dtype=np.float32),
            scale=float(spec["scale"]),
            rotation=np.asarray(spec["rotation"], dtype=np.float32),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "center": [float(c) for c in self.center.tolist()],
            "scale": float(self.scale),
            "rotation": [[float(v) for v in row] for row in self.rotation.tolist()],
        }

    def is_identity(self) -> bool:
        if self.type == "identity":
            return True
        return (
            np.allclose(self.center, 0.0, atol=1e-6)
            and abs(self.scale - 1.0) < 1e-6
            and np.allclose(self.rotation, np.eye(3, dtype=np.float32), atol=1e-6)
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        if self.is_identity():
            return points.astype(np.float32, copy=False)
        centred = (points - self.center) / max(self.scale, 1e-12)
        return (centred @ self.rotation.T).astype(np.float32)

    def invert(self, points: np.ndarray) -> np.ndarray:
        if self.is_identity():
            return points.astype(np.float32, copy=False)
        rotated = points @ self.rotation
        return (rotated * self.scale + self.center).astype(np.float32)


class SPCarObjectDataset:
    """Object-level wrapper over the existing whole-car patch cache.

    Each ``__getitem__`` returns a dict with whole-object point clouds and
    occupancy / free-space query supervision in canonical coordinates.

    Parameters
    ----------
    object_index_path:
        JSON produced by ``build_spcarnet_object_index.py``.
    splits:
        Subset selector — any combination of ``("train", "val", "test")``.
    return_observed / return_queries / return_normals:
        Field selectors. Disabling reduces I/O on large epochs.
    apply_canonical_transform:
        When True (default) all returned point fields are mapped through the
        canonical transform recorded in the index. When False the raw cache
        values are returned (useful for verifying round-trip equivalence).
    estimate_symmetry:
        When True the wrapper lazily computes a symmetry plane via
        ``ss3dm_prior.data.symmetry_targets.estimate_symmetry_plane`` per object
        and caches the result. Disabled by default — Stage 2 onwards may opt in.
    scanner_pose_fn:
        Optional callable ``(record_dict) -> np.ndarray (4, 4)`` used to inject
        a runtime-sampled scanner pose. The current cache does not persist
        scanner poses; LiDAR pipelines should provide their own sampler here.
    """

    def __init__(
        self,
        object_index_path: str | Path,
        *,
        splits: Iterable[str] = ("train",),
        return_observed: bool = True,
        return_queries: bool = True,
        return_normals: bool = True,
        apply_canonical_transform: bool = True,
        estimate_symmetry: bool = False,
        scanner_pose_fn: Callable[[dict[str, Any]], np.ndarray] | None = None,
    ) -> None:
        index_path = Path(object_index_path).expanduser().resolve()
        with index_path.open() as f:
            doc = json.load(f)
        if int(doc.get("version", 0)) != 1:
            raise ValueError(f"Unsupported object_index version: {doc.get('version')!r}")
        self._index_path = index_path
        self._stats: dict[str, Any] = doc.get("stats", {})
        self._return_observed = bool(return_observed)
        self._return_queries = bool(return_queries)
        self._return_normals = bool(return_normals)
        self._apply_canonical = bool(apply_canonical_transform)
        self._estimate_symmetry = bool(estimate_symmetry)
        self._scanner_pose_fn = scanner_pose_fn
        wanted = set(splits) if splits is not None else None
        all_objects: list[dict[str, Any]] = doc.get("objects", []) or []
        if wanted is not None:
            self._objects = [o for o in all_objects if o.get("split") in wanted]
        else:
            self._objects = list(all_objects)
        self._symmetry_cache: dict[str, dict[str, Any] | None] = {}

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    @property
    def index_path(self) -> Path:
        return self._index_path

    def __len__(self) -> int:
        return len(self._objects)

    def get_object_record(self, idx: int) -> dict[str, Any]:
        return self._objects[idx]

    def _maybe_estimate_symmetry(self, object_id: str, points: np.ndarray) -> dict[str, Any] | None:
        if not self._estimate_symmetry:
            return None
        if object_id in self._symmetry_cache:
            return self._symmetry_cache[object_id]
        try:
            from ss3dm_prior.data.symmetry_targets import estimate_symmetry_plane

            result = estimate_symmetry_plane(points.astype(np.float32))
            payload = {
                "normal": np.asarray(result.normal, dtype=np.float32),
                "offset": float(result.offset),
                "confidence": float(result.confidence),
            }
        except Exception as exc:  # pragma: no cover — runtime estimator may evolve
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        self._symmetry_cache[object_id] = payload
        return payload

    def _load_primary_npz(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        patch_files = record.get("patch_files") or []
        if not patch_files:
            raise FileNotFoundError(f"No patch files for object {record.get('object_id')!r}")
        path = Path(patch_files[0])
        if not path.is_file():
            raise FileNotFoundError(f"Missing NPZ for object {record.get('object_id')!r}: {path}")
        with np.load(path) as data:
            return {k: data[k] for k in data.files if k in _TENSOR_FIELDS}

    def _aggregate_pseudo_object(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        """Concatenate tensors across multiple patches and downsample large fields.

        Not exercised by the current cache (each car already maps to one NPZ).
        Implemented for future schema variants.
        """
        target_counts = {
            "clean_points": 2048,
            "clean_normals": 2048,
            "visible_clean_points": None,
            "hidden_clean_points": None,
            "observed_points": 768,
        }
        accumulated: dict[str, list[np.ndarray]] = {}
        for path_str in record.get("patch_files", []):
            with np.load(path_str) as data:
                for key in _TENSOR_FIELDS:
                    if key in data.files:
                        accumulated.setdefault(key, []).append(data[key])
        merged: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed=hash(record.get("object_id", "")) & 0xFFFFFFFF)
        for key, parts in accumulated.items():
            arr = np.concatenate(parts, axis=0)
            tgt = target_counts.get(key)
            if tgt is not None and arr.shape[0] > tgt:
                idx = rng.choice(arr.shape[0], size=tgt, replace=False)
                arr = arr[idx]
            merged[key] = arr
        return merged

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._objects[idx]
        npz_fields = (
            self._aggregate_pseudo_object(record)
            if record.get("canonical_transform", {}).get("type") == "pseudo-aggregated"
            else self._load_primary_npz(record)
        )
        canonical = CanonicalTransform.from_dict(record["canonical_transform"])

        def _xform(points: np.ndarray) -> np.ndarray:
            if not self._apply_canonical or canonical.is_identity():
                return points.astype(np.float32, copy=False)
            return canonical.apply(points)

        clean_points = _xform(np.asarray(npz_fields["clean_points"], dtype=np.float32))
        out: dict[str, Any] = {
            "object_id": record["object_id"],
            "split": record["split"],
            "town_id": record["town_id"],
            "clean_points_object": clean_points,
            "canonical_transform": canonical.to_dict(),
            "patch_metadata_list": [{"patch_id": pid} for pid in record.get("patch_ids", [])],
            "source_mesh_path": record.get("source_mesh_path"),
        }

        if self._return_normals and "clean_normals" in npz_fields:
            out["clean_normals_object"] = np.asarray(npz_fields["clean_normals"], dtype=np.float32)
        else:
            out["clean_normals_object"] = None

        if "visible_clean_points" in npz_fields:
            out["visible_clean_points"] = _xform(np.asarray(npz_fields["visible_clean_points"], dtype=np.float32))
        else:
            out["visible_clean_points"] = None
        if "hidden_clean_points" in npz_fields:
            out["hidden_clean_points"] = _xform(np.asarray(npz_fields["hidden_clean_points"], dtype=np.float32))
        else:
            out["hidden_clean_points"] = None

        if self._return_observed and "observed_points" in npz_fields:
            out["partial_observed_points"] = _xform(
                np.asarray(npz_fields["observed_points"], dtype=np.float32)
            )
        else:
            out["partial_observed_points"] = None

        if self._return_queries:
            qpts = npz_fields.get("query_points_all")
            qlab = npz_fields.get("query_labels_all")
            qign = npz_fields.get("query_ignore_mask")
            if qpts is not None:
                out["occupancy_query_points"] = _xform(np.asarray(qpts, dtype=np.float32))
                out["occupancy_query_labels"] = np.asarray(qlab, dtype=np.int8) if qlab is not None else None
                out["occupancy_query_ignore"] = (
                    np.asarray(qign, dtype=bool) if qign is not None else None
                )
            else:
                out["occupancy_query_points"] = None
                out["occupancy_query_labels"] = None
                out["occupancy_query_ignore"] = None

            fpts = npz_fields.get("free_query_points")
            out["free_space_query_points"] = (
                _xform(np.asarray(fpts, dtype=np.float32)) if fpts is not None else None
            )
            fhard = npz_fields.get("free_space_query_hard_negatives")
            out["free_space_query_hard_negatives"] = (
                _xform(np.asarray(fhard, dtype=np.float32)) if fhard is not None else None
            )
        else:
            out["occupancy_query_points"] = None
            out["occupancy_query_labels"] = None
            out["occupancy_query_ignore"] = None
            out["free_space_query_points"] = None
            out["free_space_query_hard_negatives"] = None

        if self._scanner_pose_fn is not None:
            try:
                out["scanner_pose"] = np.asarray(self._scanner_pose_fn(record), dtype=np.float32)
            except Exception as exc:  # pragma: no cover
                out["scanner_pose"] = None
                out["scanner_pose_error"] = f"{type(exc).__name__}: {exc}"
        else:
            out["scanner_pose"] = None

        out["symmetry"] = self._maybe_estimate_symmetry(record["object_id"], clean_points)
        return out

    def iter_records(self) -> Iterable[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]


def collate_object_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack object dicts into a batch dict.

    Variable-length tensors (visible/hidden split) are returned as Python lists rather
    than padded — Stage 2 trainers operate per-object and can resample within their
    forward pass. Fixed-length tensors (clean_points 2048, observed_points 768,
    occupancy queries 1280, free queries 512, hard negatives 128) are stacked.
    """
    if not items:
        return {}
    fixed_fields = (
        ("clean_points_object", np.float32),
        ("clean_normals_object", np.float32),
        ("partial_observed_points", np.float32),
        ("occupancy_query_points", np.float32),
        ("occupancy_query_labels", np.int8),
        ("occupancy_query_ignore", np.bool_),
        ("free_space_query_points", np.float32),
        ("free_space_query_hard_negatives", np.float32),
    )
    variable_fields = ("visible_clean_points", "hidden_clean_points")
    out: dict[str, Any] = {
        "object_id": [item["object_id"] for item in items],
        "split": [item["split"] for item in items],
        "town_id": [item["town_id"] for item in items],
        "canonical_transform": [item["canonical_transform"] for item in items],
        "patch_metadata_list": [item["patch_metadata_list"] for item in items],
        "source_mesh_path": [item.get("source_mesh_path") for item in items],
        "scanner_pose": [item.get("scanner_pose") for item in items],
        "symmetry": [item.get("symmetry") for item in items],
    }
    for key, dtype in fixed_fields:
        values = [item.get(key) for item in items]
        if any(v is None for v in values):
            out[key] = values
            continue
        out[key] = np.stack([np.asarray(v, dtype=dtype) for v in values], axis=0)
    for key in variable_fields:
        out[key] = [item.get(key) for item in items]
    return out


__all__ = [
    "CanonicalTransform",
    "SPCarObjectDataset",
    "collate_object_batch",
]

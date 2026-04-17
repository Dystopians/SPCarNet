"""Build whole-car mesh samples as a patch-cache-compatible dataset."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import trimesh

from ss3dm_prior.data.patch_index import write_patch_index_jsonl
from ss3dm_prior.data.patch_types import PatchIndexRecord, TeacherPatchSample
from ss3dm_prior.utils.io import dump_json, dump_yaml


TRAIN_TOWN_ID = "MeshFleetTrain"
VAL_TOWN_ID = "MeshFleetVal"
TEST_TOWN_ID = "MeshFleetTest"


def _safe_stem(path: str) -> str:
    candidate = Path(path)
    if candidate.suffix:
        return candidate.stem
    return candidate.name


def _load_metadata_rows(metadata_path: Path, *, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = dict(row)
            record["metadata_path"] = str(metadata_path.resolve())
            record["split_name"] = split_name
            sha256 = str(record.get("sha256", "")).strip()
            record["car_id"] = sha256 or _safe_stem(str(record.get("local_path", "")))
            rows.append(record)
    return rows


def _resolve_mesh_path(
    row: dict[str, Any],
    *,
    dataset_root: Path,
    mesh_root: Path,
) -> Path | None:
    local_path = str(row.get("local_path", "")).strip().lstrip("./")
    split_name = str(row.get("split_name", "")).strip()
    car_id = str(row.get("car_id", "")).strip()
    candidate_ids = []
    for candidate_id in [car_id, _safe_stem(local_path)]:
        if candidate_id and candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)
    candidates = []
    if local_path:
        candidates.extend(
            [
                mesh_root / split_name / local_path,
                mesh_root / local_path,
                dataset_root / split_name / local_path,
                dataset_root / local_path,
            ]
        )
    for candidate_id in candidate_ids:
        for base_root in [mesh_root, dataset_root]:
            candidates.extend(
                [
                    base_root / split_name / "mesh_normalized" / candidate_id / "mesh.glb",
                    base_root / split_name / "mesh_normalized" / candidate_id / "mesh.obj",
                    base_root / split_name / "mesh_normalized" / candidate_id / "mesh.ply",
                    base_root / split_name / "mesh_normalized" / candidate_id / "mesh.fbx",
                    base_root / "mesh_normalized" / candidate_id / "mesh.glb",
                    base_root / "mesh_normalized" / candidate_id / "mesh.obj",
                    base_root / "mesh_normalized" / candidate_id / "mesh.ply",
                    base_root / "mesh_normalized" / candidate_id / "mesh.fbx",
                ]
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _merge_scene_geometry(loaded: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    geometries: list[trimesh.Trimesh] = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph.get(node_name)
        geometry = loaded.geometry[geom_name].copy()
        geometry.apply_transform(transform)
        geometries.append(geometry)
    if not geometries:
        raise ValueError("Scene contains no mesh geometry.")
    merged = trimesh.util.concatenate(geometries)
    if not isinstance(merged, trimesh.Trimesh):
        raise ValueError("Failed to merge scene geometry into a mesh.")
    return merged


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = _merge_scene_geometry(loaded)
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {path}")
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "nondegenerate_faces"):
        nondegenerate = np.asarray(mesh.nondegenerate_faces(), dtype=bool)
        if nondegenerate.shape[0] == len(mesh.faces):
            mesh.update_faces(nondegenerate)
            mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "unique_faces"):
        unique = np.asarray(mesh.unique_faces(), dtype=bool)
        if unique.shape[0] == len(mesh.faces):
            mesh.update_faces(unique)
            mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh became empty after cleanup: {path}")
    return mesh


def _normalize_mesh(mesh: trimesh.Trimesh, *, target_radius: float) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    normalized = mesh.copy()
    vertices = np.asarray(normalized.vertices, dtype=np.float32)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid[None, :]
    radius = float(np.linalg.norm(centered, axis=1).max())
    radius = max(radius, 1e-6)
    scale = float(target_radius) / radius
    normalized.vertices = centered * scale
    return normalized, centroid.astype(np.float32), radius


def _sample_surface(
    mesh: trimesh.Trimesh,
    *,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    points, face_indices = trimesh.sample.sample_surface(mesh, sample_count, seed=rng)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(normals_norm, 1e-6, None)
    return np.asarray(points, dtype=np.float32), normals.astype(np.float32)


def _planarity_hint(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
    denom = float(np.sum(eigvals))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - eigvals[0] / denom)


def _synthetic_camera_dirs(num_views: int) -> np.ndarray:
    num_views = max(int(num_views), 1)
    dirs = []
    for idx in range(num_views):
        azimuth = 2.0 * np.pi * idx / num_views
        elevation = np.deg2rad(15.0 if idx % 2 == 0 else -10.0)
        direction = np.asarray(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float32,
        )
        direction /= np.clip(np.linalg.norm(direction), 1e-6, None)
        dirs.append(direction)
    return np.stack(dirs, axis=0)


def _build_observed_points(
    mesh: trimesh.Trimesh,
    *,
    observed_sample_count: int,
    clean_sample_count: int,
    observed_view_count: int,
    min_visibility_cosine: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    candidate_count = max(int(observed_sample_count) * 6, int(clean_sample_count) * 2)
    candidate_points, candidate_normals = _sample_surface(mesh, sample_count=candidate_count, rng=rng)
    camera_dirs = _synthetic_camera_dirs(observed_view_count)
    view_scores = candidate_normals @ (-camera_dirs).T
    best_scores = np.max(view_scores, axis=1)
    visible_mask = best_scores >= float(min_visibility_cosine)
    visible_points = candidate_points[visible_mask]
    visible_fraction = float(np.mean(visible_mask)) if visible_mask.size else 0.0
    if len(visible_points) == 0:
        visible_points = candidate_points
    if len(visible_points) >= observed_sample_count:
        indices = rng.choice(len(visible_points), size=int(observed_sample_count), replace=False)
        observed_points = visible_points[indices]
    else:
        fallback_indices = rng.choice(len(candidate_points), size=int(observed_sample_count), replace=len(candidate_points) < int(observed_sample_count))
        augmented = np.concatenate([visible_points, candidate_points[fallback_indices]], axis=0)
        if len(augmented) >= observed_sample_count:
            observed_points = augmented[: int(observed_sample_count)]
        else:
            indices = rng.choice(len(augmented), size=int(observed_sample_count), replace=True)
            observed_points = augmented[indices]
    return observed_points.astype(np.float32), visible_fraction


def _build_sample(
    row: dict[str, Any],
    *,
    mesh_path: Path,
    out_dir: Path,
    clean_sample_count: int,
    observed_sample_count: int,
    normalized_radius: float,
    observed_view_count: int,
    min_visibility_cosine: float,
    seed: int,
) -> PatchIndexRecord:
    car_id = str(row["car_id"])
    split_name = str(row["split_name"])
    town_id = TRAIN_TOWN_ID if split_name == "train" else VAL_TOWN_ID if split_name == "val" else TEST_TOWN_ID
    rng = np.random.default_rng(int(seed) + (zlib.crc32(car_id.encode("utf-8")) % (2**32)))
    mesh = _load_mesh(mesh_path)
    normalized_mesh, original_centroid, original_radius = _normalize_mesh(mesh, target_radius=normalized_radius)
    clean_points, clean_normals = _sample_surface(normalized_mesh, sample_count=clean_sample_count, rng=rng)
    observed_points, visible_fraction = _build_observed_points(
        normalized_mesh,
        observed_sample_count=observed_sample_count,
        clean_sample_count=clean_sample_count,
        observed_view_count=observed_view_count,
        min_visibility_cosine=min_visibility_cosine,
        rng=rng,
    )
    teacher_area_local = float(normalized_mesh.area)
    planarity = _planarity_hint(clean_points)
    intrinsic_target = float(np.clip(1.0 - visible_fraction, 0.0, 1.0))
    metadata = {
        "asset_id": car_id,
        "source_mesh_path": str(mesh_path),
        "file_identifier": row.get("file_identifier", ""),
        "captions": row.get("captions", ""),
        "original_centroid_world": [float(value) for value in original_centroid],
        "original_radius_world": float(original_radius),
        "normalized_radius": float(normalized_radius),
        "planarity_hint": planarity,
        "source_split_name": split_name,
    }

    sample = TeacherPatchSample(
        clean_points=clean_points,
        clean_normals=clean_normals,
        observed_points=observed_points,
        patch_center_world=np.zeros((3,), dtype=np.float32),
        patch_radius_m=float(normalized_radius),
        town_id=town_id,
        sequence_id=car_id,
        tile_id=0,
        patch_id=car_id,
        num_local_faces=int(len(normalized_mesh.faces)),
        num_observed_points_raw=int(len(observed_points)),
        teacher_area_local=teacher_area_local,
        source_town_mesh_cache_dir=str(mesh_path),
        source_sequence_observed_cache="synthetic_whole_car_views",
        patch_cache_format_version=1,
        camera_support_count=int(observed_view_count),
        lidar_support_count=0,
        visible_surface_fraction=visible_fraction,
        free_space_fraction=0.0,
        unknown_fraction=0.0,
        intrinsic_patch_difficulty_target=intrinsic_target,
        difficulty_components_json={"one_minus_visible_surface_fraction": intrinsic_target},
        metadata=metadata,
    )
    patch_dir = out_dir / town_id / car_id
    patch_path = patch_dir / f"{car_id}.npz"
    sample.save(patch_path)
    return PatchIndexRecord(
        patch_id=car_id,
        town_id=town_id,
        sequence_id=car_id,
        tile_id=0,
        patch_file=str(patch_path.resolve()),
        num_local_faces=int(len(normalized_mesh.faces)),
        num_observed_points_raw=int(len(observed_points)),
        num_clean_points=int(len(clean_points)),
        num_observed_points=int(len(observed_points)),
        teacher_area_local=teacher_area_local,
        planarity_hint=planarity,
        patch_cache_format_version=1,
        camera_support_count=int(observed_view_count),
        lidar_support_count=0,
        visible_surface_fraction=visible_fraction,
        intrinsic_patch_difficulty_target=intrinsic_target,
        difficulty_components_json={"one_minus_visible_surface_fraction": intrinsic_target},
    )


def _partition_train_rows(train_rows: list[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not train_rows:
        return [], []
    ordered = sorted(train_rows, key=lambda row: str(row["car_id"]))
    val_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * float(val_fraction))))) if len(ordered) > 1 and val_fraction > 0 else 0
    if val_count <= 0:
        return ordered, []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    val_indices = set(int(idx) for idx in perm[:val_count])
    train_split: list[dict[str, Any]] = []
    val_split: list[dict[str, Any]] = []
    for idx, row in enumerate(ordered):
        target = val_split if idx in val_indices else train_split
        copied = dict(row)
        copied["assigned_split_name"] = "val" if idx in val_indices else "train"
        target.append(copied)
    return train_split, val_split


def _write_split_config(out_dir: Path) -> Path:
    split_path = out_dir / "split_meshfleet_car.yaml"
    dump_yaml(
        split_path,
        {
            "split_name": "meshfleet_car_whole_mesh",
            "strategy": "preassigned_car_split",
            "unit_of_split": "car_mesh",
            "forbid_random_patch_split": True,
            "forbid_random_frame_split": True,
            "train_towns": [TRAIN_TOWN_ID],
            "val_towns": [VAL_TOWN_ID],
            "test_towns": [TEST_TOWN_ID],
            "notes": [
                "Whole-car mesh route: each asset is one training sample rather than a local street patch.",
                "town_id is repurposed as a split bucket label for compatibility with the existing trainer.",
            ],
        },
    )
    return split_path


def _process_row(
    *,
    row: dict[str, Any],
    dataset_root: str,
    mesh_root: str,
    out_dir: str,
    clean_sample_count: int,
    observed_sample_count: int,
    normalized_radius: float,
    observed_view_count: int,
    min_visibility_cosine: float,
    seed: int,
    skip_existing: bool,
) -> dict[str, Any]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    mesh_root_path = Path(mesh_root).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()
    mesh_path = _resolve_mesh_path(row, dataset_root=dataset_root_path, mesh_root=mesh_root_path)
    if mesh_path is None:
        return {"status": "missing_mesh", "car_id": row["car_id"], "split_name": row["split_name"]}
    assigned_split_name = str(row.get("assigned_split_name", row["split_name"]))
    town_id = TRAIN_TOWN_ID if assigned_split_name == "train" else VAL_TOWN_ID if assigned_split_name == "val" else TEST_TOWN_ID
    patch_path = out_dir_path / town_id / str(row["car_id"]) / f"{row['car_id']}.npz"
    if skip_existing and patch_path.exists():
        return {"status": "reused", "car_id": row["car_id"], "split_name": assigned_split_name, "patch_file": str(patch_path.resolve())}
    patched_row = dict(row)
    patched_row["split_name"] = assigned_split_name
    record = _build_sample(
        patched_row,
        mesh_path=mesh_path,
        out_dir=out_dir_path,
        clean_sample_count=clean_sample_count,
        observed_sample_count=observed_sample_count,
        normalized_radius=normalized_radius,
        observed_view_count=observed_view_count,
        min_visibility_cosine=min_visibility_cosine,
        seed=seed,
    )
    return {
        "status": "built",
        "car_id": row["car_id"],
        "split_name": assigned_split_name,
        "patch_file": record.patch_file,
    }


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a whole-car mesh patch cache compatible with SS3DM prior training.")
    parser.add_argument("--dataset_root", required=True, help="Root directory containing MeshFleet-style `train/` and `test/` metadata.")
    parser.add_argument("--mesh_root", default=None, help="Optional separate root used to resolve mesh paths from metadata local_path.")
    parser.add_argument("--train_metadata_csv", default=None, help="Optional explicit train metadata CSV path.")
    parser.add_argument("--test_metadata_csv", default=None, help="Optional explicit test metadata CSV path.")
    parser.add_argument("--out_dir", required=True, help="Output directory for the generated patch cache.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of train metadata rows to hold out as validation.")
    parser.add_argument("--clean_sample_count", type=int, default=2048, help="Surface samples per whole-car target mesh.")
    parser.add_argument("--observed_sample_count", type=int, default=768, help="Observed samples per whole-car input.")
    parser.add_argument("--normalized_radius", type=float, default=1.0, help="Target radius after centering/scaling each mesh.")
    parser.add_argument("--observed_view_count", type=int, default=3, help="Number of synthetic camera directions used to form observed points.")
    parser.add_argument("--min_visibility_cosine", type=float, default=0.05, help="Visibility threshold for synthetic observed point selection.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing per-car NPZ samples when present.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argparser().parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    mesh_root = Path(args.mesh_root).expanduser().resolve() if args.mesh_root else dataset_root
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_metadata_csv = Path(args.train_metadata_csv).expanduser().resolve() if args.train_metadata_csv else dataset_root / "train" / "metadata.csv"
    test_metadata_csv = Path(args.test_metadata_csv).expanduser().resolve() if args.test_metadata_csv else dataset_root / "test" / "metadata.csv"
    if not train_metadata_csv.exists():
        raise FileNotFoundError(f"Train metadata CSV not found: {train_metadata_csv}")
    if not test_metadata_csv.exists():
        raise FileNotFoundError(f"Test metadata CSV not found: {test_metadata_csv}")

    train_rows_raw = _load_metadata_rows(train_metadata_csv, split_name="train")
    test_rows = _load_metadata_rows(test_metadata_csv, split_name="test")
    train_rows, val_rows = _partition_train_rows(train_rows_raw, val_fraction=float(args.val_fraction), seed=int(args.seed))
    for row in train_rows:
        row["assigned_split_name"] = "train"
    for row in val_rows:
        row["assigned_split_name"] = "val"
    for row in test_rows:
        row["assigned_split_name"] = "test"
    all_rows = [*train_rows, *val_rows, *test_rows]

    records_by_car_id: dict[str, dict[str, Any]] = {}
    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        results = [
            _process_row(
                row=row,
                dataset_root=str(dataset_root),
                mesh_root=str(mesh_root),
                out_dir=str(out_dir),
                clean_sample_count=int(args.clean_sample_count),
                observed_sample_count=int(args.observed_sample_count),
                normalized_radius=float(args.normalized_radius),
                observed_view_count=int(args.observed_view_count),
                min_visibility_cosine=float(args.min_visibility_cosine),
                seed=int(args.seed),
                skip_existing=bool(args.skip_existing),
            )
            for row in all_rows
        ]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_row,
                    row=row,
                    dataset_root=str(dataset_root),
                    mesh_root=str(mesh_root),
                    out_dir=str(out_dir),
                    clean_sample_count=int(args.clean_sample_count),
                    observed_sample_count=int(args.observed_sample_count),
                    normalized_radius=float(args.normalized_radius),
                    observed_view_count=int(args.observed_view_count),
                    min_visibility_cosine=float(args.min_visibility_cosine),
                    seed=int(args.seed),
                    skip_existing=bool(args.skip_existing),
                )
                for row in all_rows
            ]
            results = [future.result() for future in as_completed(futures)]

    missing_count = 0
    for result in results:
        if result["status"] == "missing_mesh":
            missing_count += 1
            print(f"missing_mesh: {result['car_id']} split={result['split_name']}")
            continue
        print(f"{result['status']}: {result['car_id']} split={result['split_name']}")
        patch_file = Path(result["patch_file"]).expanduser().resolve()
        payload = np.load(patch_file)
        try:
            patch_id = str(payload["patch_id"].item())
            records_by_car_id[patch_id] = {
                "patch_id": patch_id,
                "town_id": str(payload["town_id"].item()),
                "sequence_id": str(payload["sequence_id"].item()),
                "tile_id": int(payload["tile_id"].item()),
                "patch_file": str(patch_file),
                "num_local_faces": int(payload["num_local_faces"].item()),
                "num_observed_points_raw": int(payload["num_observed_points_raw"].item()),
                "num_clean_points": int(payload["clean_points"].shape[0]),
                "num_observed_points": int(payload["observed_points"].shape[0]),
                "teacher_area_local": float(payload["teacher_area_local"].item()),
                "planarity_hint": float(json.loads(str(payload["patch_metadata_json"].item())).get("planarity_hint", 0.0)),
                "patch_cache_format_version": int(payload["patch_cache_format_version"].item()),
                "camera_support_count": int(payload["camera_support_count"].item()),
                "lidar_support_count": int(payload["lidar_support_count"].item()),
                "visible_surface_fraction": float(payload["visible_surface_fraction"].item()),
                "free_space_fraction": float(payload["free_space_fraction"].item()),
                "unknown_fraction": float(payload["unknown_fraction"].item()),
                "intrinsic_patch_difficulty_target": float(payload["intrinsic_patch_difficulty_target"].item()),
                "difficulty_components_json": json.loads(str(payload["difficulty_components_json"].item())),
            }
        finally:
            payload.close()

    patch_records = [
        PatchIndexRecord(**record)
        for record in sorted(records_by_car_id.values(), key=lambda item: (item["town_id"], item["sequence_id"]))
    ]
    index_path = write_patch_index_jsonl(out_dir / "patch_index.jsonl", patch_records)
    split_path = _write_split_config(out_dir)
    manifest_path = out_dir / "source_mesh_manifest.json"
    dump_json(
        manifest_path,
        {
            "dataset_name": "meshfleet_car_whole_mesh",
            "dataset_root": str(dataset_root),
            "mesh_root": str(mesh_root),
            "generated_patch_index": str(index_path),
            "generated_split_config": str(split_path),
            "missing_mesh_count": missing_count,
            "records": [
                {
                    "car_id": row["car_id"],
                    "assigned_split_name": row["assigned_split_name"],
                    "metadata_path": row["metadata_path"],
                    "local_path": row.get("local_path", ""),
                    "file_identifier": row.get("file_identifier", ""),
                }
                for row in all_rows
            ],
        },
    )
    print(f"written_samples: {len(patch_records)}")
    print(f"missing_meshes: {missing_count}")
    print(f"patch_index: {index_path}")
    print(f"split_config: {split_path}")
    print(f"source_manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

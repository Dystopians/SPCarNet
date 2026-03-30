from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ColmapSparsePointsData:
    point_ids: np.ndarray  # [N] int64
    xyz: np.ndarray  # [N,3] float64
    errors: np.ndarray  # [N] float64
    pid_to_xyz: Dict[int, np.ndarray]
    pid_to_error: Dict[int, float]


def extract_colmap_sparse_points(
    colmap_points3d: Optional[Dict[int, object]],
    error_max: float = -1.0,
) -> ColmapSparsePointsData:
    """
    Convert COLMAP points3D dictionary into array/mapping views.

    Args:
        colmap_points3d: scene.scene_info.colmap_points3d dictionary.
        error_max: keep only points with error <= error_max when > 0.
    """
    if not colmap_points3d:
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_xyz = np.zeros((0, 3), dtype=np.float64)
        empty_err = np.zeros((0,), dtype=np.float64)
        return ColmapSparsePointsData(
            point_ids=empty_ids,
            xyz=empty_xyz,
            errors=empty_err,
            pid_to_xyz={},
            pid_to_error={},
        )

    pid_list = []
    xyz_list = []
    err_list = []
    for pid, pt in colmap_points3d.items():
        pid_i = int(pid)
        err = float(getattr(pt, "error", 0.0))
        if error_max > 0 and err > float(error_max):
            continue
        xyz = np.asarray(getattr(pt, "xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
        if xyz.shape != (3,):
            continue
        pid_list.append(pid_i)
        xyz_list.append(xyz)
        err_list.append(err)

    if len(pid_list) == 0:
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_xyz = np.zeros((0, 3), dtype=np.float64)
        empty_err = np.zeros((0,), dtype=np.float64)
        return ColmapSparsePointsData(
            point_ids=empty_ids,
            xyz=empty_xyz,
            errors=empty_err,
            pid_to_xyz={},
            pid_to_error={},
        )

    point_ids = np.asarray(pid_list, dtype=np.int64)
    xyz = np.stack(xyz_list, axis=0).astype(np.float64)
    errors = np.asarray(err_list, dtype=np.float64)

    pid_to_xyz = {int(point_ids[i]): xyz[i] for i in range(point_ids.shape[0])}
    pid_to_error = {int(point_ids[i]): float(errors[i]) for i in range(point_ids.shape[0])}

    return ColmapSparsePointsData(
        point_ids=point_ids,
        xyz=xyz,
        errors=errors,
        pid_to_xyz=pid_to_xyz,
        pid_to_error=pid_to_error,
    )

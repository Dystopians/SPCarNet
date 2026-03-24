import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def point_to_plane_signed_distance(points: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    Signed distance to plane n^T x + d = 0.
    `normal` is expected to be unit-length.
    """
    return points @ normal + offset


def signed_height_relative_to_plane(points: torch.Tensor, normal: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """
    Alias for signed point-to-plane distance; positive means along +normal direction.
    """
    return point_to_plane_signed_distance(points=points, normal=normal, offset=offset)


def _fit_plane_from_points(points_xyz: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    if points_xyz.shape[0] < 3:
        return None
    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-12:
        return None
    normal = normal / normal_norm
    offset = -float(np.dot(normal, centroid))
    return normal.astype(np.float64), float(offset)


def fit_plane_ransac(
    points_xyz: np.ndarray,
    max_iters: int,
    distance_threshold: float,
    min_points: int,
    axis_hint: Optional[np.ndarray] = None,
    axis_consistency_min: float = 0.0,
) -> Dict:
    stats = {
        "ok": False,
        "reason": "unknown",
        "normal": None,
        "offset": None,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "residual_mean": 0.0,
        "residual_median": 0.0,
        "residual_q95": 0.0,
    }
    n_points = int(points_xyz.shape[0])
    if n_points < max(min_points, 3):
        stats["reason"] = "insufficient_points"
        return stats

    best_inlier_mask = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(int(max_iters)):
        sample_idx = rng.choice(n_points, size=3, replace=False)
        sample = points_xyz[sample_idx]
        fit = _fit_plane_from_points(sample)
        if fit is None:
            continue
        normal, offset = fit
        if axis_hint is not None and float(np.abs(np.dot(normal, axis_hint))) < float(axis_consistency_min):
            continue
        residual = np.abs(points_xyz @ normal + offset)
        inlier_mask = residual <= float(distance_threshold)
        count = int(inlier_mask.sum())
        if count > best_count:
            best_count = count
            best_inlier_mask = inlier_mask

    if best_inlier_mask is None or best_count < max(min_points, 3):
        stats["reason"] = "ransac_failed"
        return stats

    fit_full = _fit_plane_from_points(points_xyz[best_inlier_mask])
    if fit_full is None:
        stats["reason"] = "refit_failed"
        return stats

    normal, offset = fit_full
    if axis_hint is not None and np.dot(normal, axis_hint) < 0:
        normal = -normal
        offset = -offset

    residual_signed = points_xyz @ normal + offset
    residual_abs = np.abs(residual_signed)
    inlier_mask = residual_abs <= float(distance_threshold)
    inlier_ratio = float(inlier_mask.mean())

    stats.update(
        {
            "ok": True,
            "reason": "ok",
            "normal": normal,
            "offset": float(offset),
            "inlier_count": int(inlier_mask.sum()),
            "inlier_ratio": inlier_ratio,
            "residual_mean": float(residual_abs[inlier_mask].mean()) if inlier_mask.any() else float(residual_abs.mean()),
            "residual_median": float(np.median(residual_abs[inlier_mask])) if inlier_mask.any() else float(np.median(residual_abs)),
            "residual_q95": float(np.quantile(residual_abs[inlier_mask], 0.95)) if inlier_mask.any() else float(np.quantile(residual_abs, 0.95)),
            "inlier_mask": inlier_mask,
            "residual_signed": residual_signed,
        }
    )
    return stats


@dataclass
class GroundPlaneConfig:
    source_priority: List[str]
    min_points: int
    ransac_iters: int
    ransac_dist_thresh: float
    inlier_ratio_min: float
    track_len_min: int
    obs_min: int
    obs_ratio_min: float
    colmap_error_max: float
    depth_max_samples_per_view: int
    depth_sample_stride: int
    depth_inv_min: float
    mesh_sample_max: int
    axis_consistency_min: float
    outlier_quantile: float
    use_if_poor: bool
    cache_file: str
    recompute_interval: int
    force_recompute: bool
    diag_save: bool
    diag_dir: str


def _to_numpy_points(points: List[np.ndarray]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(points, axis=0).astype(np.float64)


def _camera_axis_hint(scene) -> np.ndarray:
    cams = scene.getTrainCameras()
    if len(cams) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dirs = []
    for cam in cams:
        c2w = torch.inverse(cam.world_view_transform.T).detach().cpu().numpy()
        up_vec = c2w[:3, 1]
        n = np.linalg.norm(up_vec)
        if n > 1e-9:
            dirs.append(up_vec / n)
    if len(dirs) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mean_dir = np.mean(np.stack(dirs, axis=0), axis=0)
    n = np.linalg.norm(mean_dir)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (mean_dir / n).astype(np.float64)


def _collect_colmap_ground_points(scene, cfg: GroundPlaneConfig) -> np.ndarray:
    if scene.scene_info is None or scene.scene_info.colmap_points3d is None:
        return np.zeros((0, 3), dtype=np.float64)
    raw_cam_infos = scene.scene_info.train_cameras
    cam_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    points3d = scene.scene_info.colmap_points3d

    point_ground_hits = {}
    point_total_hits = {}

    for cam_info in raw_cam_infos:
        cam = cam_by_name.get(cam_info.image_name, None)
        if cam is None or getattr(cam, "ground_mask", None) is None:
            continue
        if cam_info.colmap_xys is None or cam_info.colmap_point3D_ids is None:
            continue

        xys = cam_info.colmap_xys
        pids = cam_info.colmap_point3D_ids
        if xys is None or pids is None or len(xys) == 0:
            continue

        orig_w = max(int(cam_info.width), 1)
        orig_h = max(int(cam_info.height), 1)
        new_w = int(cam.image_width)
        new_h = int(cam.image_height)
        mask = cam.ground_mask.detach().cpu().numpy()

        sx = (new_w - 1.0) / max(orig_w - 1.0, 1.0)
        sy = (new_h - 1.0) / max(orig_h - 1.0, 1.0)
        xs = np.clip(np.round(xys[:, 0] * sx).astype(np.int64), 0, new_w - 1)
        ys = np.clip(np.round(xys[:, 1] * sy).astype(np.int64), 0, new_h - 1)
        is_ground = mask[ys, xs]

        for idx, pid in enumerate(pids):
            if int(pid) < 0:
                continue
            pid_int = int(pid)
            point_total_hits[pid_int] = point_total_hits.get(pid_int, 0) + 1
            if bool(is_ground[idx]):
                point_ground_hits[pid_int] = point_ground_hits.get(pid_int, 0) + 1

    selected_points = []
    for pid, gh in point_ground_hits.items():
        th = point_total_hits.get(pid, 0)
        if th <= 0:
            continue
        if gh < int(cfg.obs_min):
            continue
        if (float(gh) / float(th)) < float(cfg.obs_ratio_min):
            continue
        pt = points3d.get(pid, None)
        if pt is None:
            continue
        track_len = len(pt.image_ids) if hasattr(pt, "image_ids") else 0
        if track_len < int(cfg.track_len_min):
            continue
        if float(pt.error) > float(cfg.colmap_error_max):
            continue
        selected_points.append(np.asarray(pt.xyz, dtype=np.float64))
    return _to_numpy_points(selected_points)


def _collect_depth_ground_points(scene, cfg: GroundPlaneConfig) -> np.ndarray:
    points = []
    for cam in scene.getTrainCameras():
        mask = getattr(cam, "ground_mask", None)
        inv = getattr(cam, "invdepthmap", None)
        if mask is None or inv is None or not getattr(cam, "depth_reliable", False):
            continue
        inv_np = inv.detach().cpu().numpy().squeeze(0)
        mask_np = mask.detach().cpu().numpy()
        valid = mask_np & np.isfinite(inv_np) & (inv_np > float(cfg.depth_inv_min))
        ys, xs = np.where(valid)
        if ys.size == 0:
            continue
        stride = max(int(cfg.depth_sample_stride), 1)
        ys = ys[::stride]
        xs = xs[::stride]
        if ys.size > int(cfg.depth_max_samples_per_view):
            sel = np.linspace(0, ys.size - 1, int(cfg.depth_max_samples_per_view), dtype=np.int64)
            ys = ys[sel]
            xs = xs[sel]

        z = 1.0 / np.clip(inv_np[ys, xs], float(cfg.depth_inv_min), None)
        H, W = int(cam.image_height), int(cam.image_width)
        fx = 0.5 * W / np.tan(0.5 * float(cam.FoVx))
        fy = 0.5 * H / np.tan(0.5 * float(cam.FoVy))
        cx = 0.5 * (W - 1.0)
        cy = 0.5 * (H - 1.0)

        x_cam = ((xs.astype(np.float64) - cx) / fx) * z
        y_cam = ((ys.astype(np.float64) - cy) / fy) * z
        z_cam = z.astype(np.float64)
        ones = np.ones_like(z_cam)
        p_cam_h = np.stack([x_cam, y_cam, z_cam, ones], axis=1)
        c2w = torch.inverse(cam.world_view_transform.T).detach().cpu().numpy()
        p_world = p_cam_h @ c2w
        points.append(p_world[:, :3])
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(points, axis=0).astype(np.float64)


def _collect_mesh_ground_points(triangles, sample_max: int, axis_hint: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        verts = triangles.vertices.detach().cpu().numpy().astype(np.float64)
        faces = triangles._triangle_indices.detach().cpu().numpy().astype(np.int64)
    if faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    tri = verts[faces]
    centroids = tri.mean(axis=1)
    ab = tri[:, 1] - tri[:, 0]
    ac = tri[:, 2] - tri[:, 0]
    normals = np.cross(ab, ac)
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = n_norm[:, 0] > 1e-12
    normals[valid] = normals[valid] / n_norm[valid]

    # Keep near-horizontal triangles and lower-height band only.
    horizontal = np.abs(normals @ axis_hint) >= 0.7
    heights = centroids @ axis_hint
    h_cut = np.quantile(heights, 0.40)
    low_band = heights <= h_cut
    keep = horizontal & low_band
    if keep.any():
        centroids = centroids[keep]

    if centroids.shape[0] > int(sample_max):
        sel = np.linspace(0, centroids.shape[0] - 1, int(sample_max), dtype=np.int64)
        centroids = centroids[sel]
    return centroids.astype(np.float64)


def _cache_path(scene, cfg: GroundPlaneConfig) -> str:
    if os.path.isabs(cfg.cache_file):
        return cfg.cache_file
    return os.path.join(scene.model_path, cfg.cache_file)


def _load_cached_plane(scene, cfg: GroundPlaneConfig) -> Optional[Dict]:
    path = _cache_path(scene, cfg)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        expected_sig = {
            "source_priority": [s.strip().lower() for s in cfg.source_priority],
            "ransac_dist_thresh": float(cfg.ransac_dist_thresh),
            "inlier_ratio_min": float(cfg.inlier_ratio_min),
            "obs_ratio_min": float(cfg.obs_ratio_min),
            "colmap_error_max": float(cfg.colmap_error_max),
        }
        if payload.get("config_signature", None) != expected_sig:
            return None
        n = np.array(payload["normal"], dtype=np.float64)
        d = float(payload["offset"])
        if n.shape != (3,):
            return None
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return None
        n = n / nn
        d = d / nn
        payload["normal"] = n.tolist()
        payload["offset"] = d
        return payload
    except Exception:
        return None


def _save_cached_plane(scene, cfg: GroundPlaneConfig, payload: Dict):
    path = _cache_path(scene, cfg)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_diag(scene, cfg: GroundPlaneConfig, residual_signed: np.ndarray, tag: str):
    if not cfg.diag_save:
        return
    out_dir = cfg.diag_dir if cfg.diag_dir else os.path.join(scene.model_path, "ground_plane_diag")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 4))
        plt.hist(residual_signed, bins=80)
        plt.title("Ground plane residual histogram")
        plt.xlabel("signed residual (m)")
        plt.ylabel("count")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"residual_hist_{tag}.png"), dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[GroundPlane] Failed to save residual histogram: {exc}")


def estimate_or_load_ground_plane(scene, triangles, cfg: GroundPlaneConfig, iteration: int, force_recompute: bool = False) -> Dict:
    if (not cfg.force_recompute) and (not force_recompute):
        cached = _load_cached_plane(scene, cfg)
        if cached is not None:
            cached["from_cache"] = True
            return cached

    source_arrays = {}
    axis_hint = _camera_axis_hint(scene)
    for source_name in cfg.source_priority:
        src = source_name.strip().lower()
        if src == "colmap":
            source_arrays["colmap"] = _collect_colmap_ground_points(scene, cfg)
        elif src == "depth":
            source_arrays["depth"] = _collect_depth_ground_points(scene, cfg)
        elif src == "mesh":
            source_arrays["mesh"] = _collect_mesh_ground_points(triangles, cfg.mesh_sample_max, axis_hint=axis_hint)

    fused = np.zeros((0, 3), dtype=np.float64)
    used_sources = []
    for source_name in cfg.source_priority:
        src = source_name.strip().lower()
        arr = source_arrays.get(src, np.zeros((0, 3), dtype=np.float64))
        if arr.shape[0] <= 0:
            continue
        fused = np.concatenate([fused, arr], axis=0) if fused.shape[0] > 0 else arr
        used_sources.append(src)
        if fused.shape[0] >= int(cfg.min_points):
            break

    fit = fit_plane_ransac(
        points_xyz=fused,
        max_iters=cfg.ransac_iters,
        distance_threshold=cfg.ransac_dist_thresh,
        min_points=cfg.min_points,
        axis_hint=axis_hint,
        axis_consistency_min=cfg.axis_consistency_min,
    )
    if not fit["ok"]:
        return {
            "ok": False,
            "enabled_for_loss": False,
            "reason": fit["reason"],
            "points_total": int(fused.shape[0]),
            "source_counts": {k: int(v.shape[0]) for k, v in source_arrays.items()},
        }

    inlier_ratio = float(fit["inlier_ratio"])
    enabled = inlier_ratio >= float(cfg.inlier_ratio_min)
    if (not enabled) and cfg.use_if_poor:
        enabled = True

    residual_signed = fit["residual_signed"]
    q = float(np.clip(cfg.outlier_quantile, 0.5, 0.999))
    abs_limit = float(np.quantile(np.abs(residual_signed), q))
    keep = np.abs(residual_signed) <= abs_limit
    filtered_signed = residual_signed[keep] if keep.any() else residual_signed

    n = np.array(fit["normal"], dtype=np.float64)
    d = float(fit["offset"])
    n_norm = np.linalg.norm(n)
    n = n / max(n_norm, 1e-12)
    d = d / max(n_norm, 1e-12)

    payload = {
        "ok": True,
        "enabled_for_loss": bool(enabled),
        "from_cache": False,
        "iteration": int(iteration),
        "normal": n.tolist(),
        "offset": float(d),
        "plane_equation": "n^T x + d = 0, with ||n||_2 = 1",
        "inlier_count": int(fit["inlier_count"]),
        "inlier_ratio": float(inlier_ratio),
        "residual_mean": float(fit["residual_mean"]),
        "residual_median": float(fit["residual_median"]),
        "residual_q95": float(fit["residual_q95"]),
        "height_mean": float(filtered_signed.mean()) if filtered_signed.size > 0 else 0.0,
        "height_std": float(filtered_signed.std()) if filtered_signed.size > 0 else 0.0,
        "height_q05": float(np.quantile(filtered_signed, 0.05)) if filtered_signed.size > 0 else 0.0,
        "height_q95": float(np.quantile(filtered_signed, 0.95)) if filtered_signed.size > 0 else 0.0,
        "source_counts": {k: int(v.shape[0]) for k, v in source_arrays.items()},
        "sources_used": used_sources,
        "points_total": int(fused.shape[0]),
        "config_signature": {
            "source_priority": [s.strip().lower() for s in cfg.source_priority],
            "ransac_dist_thresh": float(cfg.ransac_dist_thresh),
            "inlier_ratio_min": float(cfg.inlier_ratio_min),
            "obs_ratio_min": float(cfg.obs_ratio_min),
            "colmap_error_max": float(cfg.colmap_error_max),
        },
    }

    _save_cached_plane(scene, cfg, payload)
    _save_diag(scene, cfg, residual_signed=filtered_signed, tag=f"iter_{int(iteration)}")
    return payload

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.colmap_sparse_utils import extract_colmap_sparse_points
from utils.geometry_metrics_utils import depth_metrics, normal_metrics_from_abs_cos

try:
    from scipy.spatial import cKDTree  # type: ignore

    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False


def normalize_image_key(name: str) -> str:
    base = name.split("/")[-1]
    stem = base.rsplit(".", 1)[0]
    return stem.lower()


def _safe_unit(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return v / n


def _try_build_kdtree(points_xyz: np.ndarray):
    if (not _HAS_CKDTREE) or points_xyz.shape[0] == 0:
        return None
    try:
        return cKDTree(points_xyz)
    except Exception:
        return None


@dataclass
class GeometryProxyConfig:
    max_points_per_view: int = 3000
    point_error_max: float = 2.0
    normal_knn: int = 24
    compute_normal: bool = True
    seed: int = 7
    sample_mode: str = "random"
    low_error_fraction: float = 1.0


@dataclass
class GeometryProxyContext:
    cam_info_by_name: Dict[str, object]
    point_ids: np.ndarray
    point_xyz: np.ndarray
    pid_to_xyz: Dict[int, np.ndarray]
    pid_to_error: Dict[int, float]
    pid_to_index: Dict[int, int]
    normal_knn: int
    compute_normal: bool
    kdtree: Optional[object]
    normal_cache: Dict[int, np.ndarray]


def build_cam_info_lookup(cam_infos: Sequence[object]) -> Dict[str, object]:
    by_name: Dict[str, object] = {}
    for c in cam_infos:
        key = normalize_image_key(getattr(c, "image_name", ""))
        if key:
            by_name[key] = c
    return by_name


def build_geometry_proxy_context(
    colmap_points3d: Optional[Dict[int, object]],
    cam_infos: Sequence[object],
    cfg: GeometryProxyConfig,
) -> GeometryProxyContext:
    sparse = extract_colmap_sparse_points(colmap_points3d=colmap_points3d, error_max=-1.0)
    point_ids = sparse.point_ids.astype(np.int64, copy=False)
    point_xyz = sparse.xyz.astype(np.float64, copy=False)
    pid_to_index = {int(point_ids[i]): int(i) for i in range(point_ids.shape[0])}
    return GeometryProxyContext(
        cam_info_by_name=build_cam_info_lookup(cam_infos=cam_infos),
        point_ids=point_ids,
        point_xyz=point_xyz,
        pid_to_xyz=sparse.pid_to_xyz,
        pid_to_error=sparse.pid_to_error,
        pid_to_index=pid_to_index,
        normal_knn=int(max(1, cfg.normal_knn)),
        compute_normal=bool(cfg.compute_normal),
        kdtree=_try_build_kdtree(point_xyz) if bool(cfg.compute_normal) else None,
        normal_cache={},
    )


def _estimate_point_normal(pid: int, ctx: GeometryProxyContext) -> Optional[np.ndarray]:
    if pid in ctx.normal_cache:
        return ctx.normal_cache[pid]
    if (not bool(ctx.compute_normal)) or (ctx.kdtree is None):
        return None
    idx = ctx.pid_to_index.get(int(pid), None)
    if idx is None:
        return None
    k = min(int(ctx.normal_knn) + 1, int(ctx.point_xyz.shape[0]))
    if k < 4:
        return None
    _, nn = ctx.kdtree.query(ctx.point_xyz[idx], k=k)
    if np.isscalar(nn):
        return None
    nn = np.array(nn, dtype=np.int64).reshape(-1)
    if nn.shape[0] < 4:
        return None
    neigh = ctx.point_xyz[nn[1:]]
    centered = neigh - neigh.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    n = v[:, int(np.argmin(w))]
    n = n / max(np.linalg.norm(n), 1e-10)
    ctx.normal_cache[int(pid)] = n
    return n


def _subsample_sparse_points(
    xys_valid: np.ndarray,
    pids_valid: np.ndarray,
    ctx: GeometryProxyContext,
    cfg: GeometryProxyConfig,
    rng: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray]:
    max_points = int(cfg.max_points_per_view)
    if max_points <= 0 or xys_valid.shape[0] <= max_points:
        return xys_valid, pids_valid
    mode = str(getattr(cfg, "sample_mode", "random") or "random").strip().lower()
    if rng is None:
        rng = np.random.default_rng(int(cfg.seed))
    if mode in {"low_error", "lowest_error", "trusted"}:
        errors = np.array([ctx.pid_to_error.get(int(pid), 1e9) for pid in pids_valid], dtype=np.float64)
        order = np.argsort(errors, kind="stable")
        pick = order[:max_points]
    elif mode in {"mixed_low_error", "mixed"}:
        fraction = float(getattr(cfg, "low_error_fraction", 0.5))
        fraction = min(1.0, max(0.0, fraction))
        trusted_count = min(max_points, int(round(float(max_points) * fraction)))
        errors = np.array([ctx.pid_to_error.get(int(pid), 1e9) for pid in pids_valid], dtype=np.float64)
        order = np.argsort(errors, kind="stable")
        trusted = order[:trusted_count]
        if trusted_count < max_points:
            remaining = order[trusted_count:]
            random_count = min(max_points - trusted_count, remaining.shape[0])
            random_pick = rng.choice(remaining, size=random_count, replace=False) if random_count > 0 else np.zeros((0,), dtype=np.int64)
            pick = np.concatenate([trusted, random_pick], axis=0)
        else:
            pick = trusted
    else:
        pick = rng.choice(xys_valid.shape[0], size=max_points, replace=False)
    return xys_valid[pick], pids_valid[pick]


def estimate_view_sparse_observability(
    view,
    ctx: GeometryProxyContext,
    cfg: GeometryProxyConfig,
) -> Dict[str, float]:
    key = normalize_image_key(getattr(view, "image_name", ""))
    cam_info = ctx.cam_info_by_name.get(key, None)
    if cam_info is None:
        return {
            "score": 0.0,
            "depth_matches": 0.0,
            "normal_matches": 0.0,
            "reason": "missing_cam_info",
        }
    xys = getattr(cam_info, "colmap_xys", None)
    pids = getattr(cam_info, "colmap_point3D_ids", None)
    if xys is None or pids is None:
        return {
            "score": 0.0,
            "depth_matches": 0.0,
            "normal_matches": 0.0,
            "reason": "no_sparse_matches",
        }
    pids_np = np.asarray(pids, dtype=np.int64)
    valid = pids_np > 0
    valid &= np.array([int(pid) in ctx.pid_to_xyz for pid in pids_np], dtype=bool)
    if float(cfg.point_error_max) > 0:
        valid &= np.array(
            [ctx.pid_to_error.get(int(pid), 1e9) <= float(cfg.point_error_max) for pid in pids_np],
            dtype=bool,
        )
    if not np.any(valid):
        return {
            "score": 0.0,
            "depth_matches": 0.0,
            "normal_matches": 0.0,
            "reason": "no_sparse_matches",
        }
    pids_valid = pids_np[valid]
    depth_n = int(pids_valid.shape[0])
    normal_n = 0
    if bool(cfg.compute_normal):
        for pid in pids_valid.tolist():
            if _estimate_point_normal(int(pid), ctx) is not None:
                normal_n += 1
    # Prefer depth observability first, then normal observability.
    score = float(depth_n) + 0.2 * float(normal_n)
    return {
        "score": float(score),
        "depth_matches": float(depth_n),
        "normal_matches": float(normal_n),
        "reason": "ok",
    }


def _extract_render_outputs(render_pkg: Dict) -> Dict[str, Optional[np.ndarray]]:
    surf_depth = render_pkg.get("surf_depth", None)
    rend_normal = render_pkg.get("rend_normal", None)
    pred_depth = None
    pred_normal = None
    if surf_depth is not None:
        try:
            pred_depth = surf_depth[0].detach().cpu().numpy()
        except Exception:
            pred_depth = None
    if rend_normal is not None:
        try:
            pred_normal = rend_normal.detach().cpu().numpy().transpose(1, 2, 0)
            pred_normal = _safe_unit(pred_normal)
        except Exception:
            pred_normal = None
    return {"pred_depth": pred_depth, "pred_normal": pred_normal}


def evaluate_view_sparse_geometry_proxy(
    view,
    render_pkg: Dict,
    ctx: GeometryProxyContext,
    cfg: GeometryProxyConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    def _empty(reason: str) -> Dict:
        return {
            "image_name": getattr(view, "image_name", ""),
            "depth_points": 0,
            "normal_points": 0,
            "depth_stats": None,
            "normal_stats": None,
            "reason": reason,
            "_depth_pred": None,
            "_depth_gt": None,
            "_cos_abs": None,
        }

    key = normalize_image_key(getattr(view, "image_name", ""))
    cam_info = ctx.cam_info_by_name.get(key, None)
    if cam_info is None:
        return _empty("missing_cam_info")

    xys = getattr(cam_info, "colmap_xys", None)
    pids = getattr(cam_info, "colmap_point3D_ids", None)
    if xys is None or pids is None:
        return _empty("no_sparse_matches")

    xys_np = np.asarray(xys, dtype=np.float64)
    pids_np = np.asarray(pids, dtype=np.int64)
    if xys_np.shape[0] == 0:
        return _empty("no_sparse_matches")

    valid = pids_np > 0
    valid &= np.array([int(pid) in ctx.pid_to_xyz for pid in pids_np], dtype=bool)
    if float(cfg.point_error_max) > 0:
        valid &= np.array(
            [ctx.pid_to_error.get(int(pid), 1e9) <= float(cfg.point_error_max) for pid in pids_np],
            dtype=bool,
        )
    if not np.any(valid):
        return _empty("no_sparse_matches")

    xys_valid = xys_np[valid]
    pids_valid = pids_np[valid]
    if int(cfg.max_points_per_view) > 0 and xys_valid.shape[0] > int(cfg.max_points_per_view):
        xys_valid, pids_valid = _subsample_sparse_points(
            xys_valid=xys_valid,
            pids_valid=pids_valid,
            ctx=ctx,
            cfg=cfg,
            rng=rng,
        )

    outputs = _extract_render_outputs(render_pkg=render_pkg)
    pred_depth = outputs["pred_depth"]
    pred_normal = outputs["pred_normal"]
    if pred_depth is None:
        return _empty("render_missing_output")

    h, w = pred_depth.shape
    cam_w = int(getattr(cam_info, "width", 0))
    cam_h = int(getattr(cam_info, "height", 0))
    vw = int(getattr(view, "image_width", getattr(view, "width", 0)))
    vh = int(getattr(view, "image_height", getattr(view, "height", 0)))
    if cam_w <= 0 or cam_h <= 0 or vw <= 0 or vh <= 0:
        return _empty("missing_camera_intrinsics")

    sx = float(vw) / float(cam_w)
    sy = float(vh) / float(cam_h)
    px = np.clip(np.round(xys_valid[:, 0] * sx).astype(np.int64), 0, w - 1)
    py = np.clip(np.round(xys_valid[:, 1] * sy).astype(np.int64), 0, h - 1)

    xyz = np.stack([ctx.pid_to_xyz[int(pid)] for pid in pids_valid], axis=0)
    r = np.asarray(getattr(cam_info, "R", None), dtype=np.float64)
    t = np.asarray(getattr(cam_info, "T", None), dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3,):
        return _empty("missing_camera_pose")
    xyz_cam = xyz @ r + t[None, :]
    gt_depth = xyz_cam[:, 2]
    pd = pred_depth[py, px]
    keep = (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pd) & (pd > 1e-6)
    if not np.any(keep):
        return _empty("no_valid_depth_matches")

    gt_depth = gt_depth[keep]
    pd = pd[keep]
    px = px[keep]
    py = py[keep]
    pids_keep = pids_valid[keep]

    d_stats = depth_metrics(pred=pd.astype(np.float64), gt=gt_depth.astype(np.float64))
    depth_points = int(gt_depth.shape[0])

    normal_arr = None
    n_stats = None
    normal_points = 0
    reason = "ok"
    if bool(cfg.compute_normal):
        if pred_normal is None:
            reason = "render_missing_output"
        else:
            cos_vals: List[float] = []
            for i, pid in enumerate(pids_keep.tolist()):
                gn = _estimate_point_normal(int(pid), ctx)
                if gn is None:
                    continue
                pn = pred_normal[py[i], px[i]]
                c = float(abs(np.dot(pn, gn)))
                if np.isfinite(c):
                    cos_vals.append(max(0.0, min(1.0, c)))
            if len(cos_vals) > 0:
                normal_arr = np.asarray(cos_vals, dtype=np.float64)
                n_stats = normal_metrics_from_abs_cos(normal_arr)
                normal_points = int(normal_arr.shape[0])
                reason = "ok"
            else:
                reason = "no_valid_normal_matches"

    return {
        "image_name": getattr(view, "image_name", ""),
        "depth_points": depth_points,
        "normal_points": int(normal_points),
        "depth_stats": d_stats,
        "normal_stats": n_stats,
        "reason": reason,
        "_depth_pred": pd.astype(np.float64),
        "_depth_gt": gt_depth.astype(np.float64),
        "_cos_abs": normal_arr,
    }


def collect_view_sparse_depth_correspondences(
    view,
    ctx: GeometryProxyContext,
    cfg: GeometryProxyConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Collect sparse COLMAP depth correspondences for one view.

    Returns:
    - reason
    - num_matches
    - point3D_id (int64 numpy array)
    - px, py (int64 numpy arrays in rendered image coordinates)
    - gt_depth (float64 numpy array)
    """

    def _empty(reason: str) -> Dict:
        return {
            "image_name": getattr(view, "image_name", ""),
            "reason": reason,
            "num_matches": 0,
            "point3D_id": np.zeros((0,), dtype=np.int64),
            "px": np.zeros((0,), dtype=np.int64),
            "py": np.zeros((0,), dtype=np.int64),
            "gt_depth": np.zeros((0,), dtype=np.float64),
        }

    key = normalize_image_key(getattr(view, "image_name", ""))
    cam_info = ctx.cam_info_by_name.get(key, None)
    if cam_info is None:
        return _empty("missing_cam_info")

    xys = getattr(cam_info, "colmap_xys", None)
    pids = getattr(cam_info, "colmap_point3D_ids", None)
    if xys is None or pids is None:
        return _empty("no_sparse_matches")

    xys_np = np.asarray(xys, dtype=np.float64)
    pids_np = np.asarray(pids, dtype=np.int64)
    if xys_np.shape[0] == 0:
        return _empty("no_sparse_matches")

    valid = pids_np > 0
    valid &= np.array([int(pid) in ctx.pid_to_xyz for pid in pids_np], dtype=bool)
    if float(cfg.point_error_max) > 0:
        valid &= np.array(
            [ctx.pid_to_error.get(int(pid), 1e9) <= float(cfg.point_error_max) for pid in pids_np],
            dtype=bool,
        )
    if not np.any(valid):
        return _empty("no_sparse_matches")

    xys_valid = xys_np[valid]
    pids_valid = pids_np[valid]
    if int(cfg.max_points_per_view) > 0 and xys_valid.shape[0] > int(cfg.max_points_per_view):
        xys_valid, pids_valid = _subsample_sparse_points(
            xys_valid=xys_valid,
            pids_valid=pids_valid,
            ctx=ctx,
            cfg=cfg,
            rng=rng,
        )

    cam_w = int(getattr(cam_info, "width", 0))
    cam_h = int(getattr(cam_info, "height", 0))
    vw = int(getattr(view, "image_width", getattr(view, "width", 0)))
    vh = int(getattr(view, "image_height", getattr(view, "height", 0)))
    if cam_w <= 0 or cam_h <= 0 or vw <= 0 or vh <= 0:
        return _empty("missing_camera_intrinsics")

    sx = float(vw) / float(cam_w)
    sy = float(vh) / float(cam_h)
    px = np.clip(np.round(xys_valid[:, 0] * sx).astype(np.int64), 0, vw - 1)
    py = np.clip(np.round(xys_valid[:, 1] * sy).astype(np.int64), 0, vh - 1)

    xyz = np.stack([ctx.pid_to_xyz[int(pid)] for pid in pids_valid], axis=0)
    r = np.asarray(getattr(cam_info, "R", None), dtype=np.float64)
    t = np.asarray(getattr(cam_info, "T", None), dtype=np.float64)
    if r.shape != (3, 3) or t.shape != (3,):
        return _empty("missing_camera_pose")
    xyz_cam = xyz @ r + t[None, :]
    gt_depth = xyz_cam[:, 2]
    keep = (gt_depth > 1e-6) & np.isfinite(gt_depth)
    if not np.any(keep):
        return _empty("no_valid_depth_matches")
    px = px[keep]
    py = py[keep]
    pids_valid = pids_valid[keep]
    gt_depth = gt_depth[keep]

    return {
        "image_name": getattr(view, "image_name", ""),
        "reason": "ok",
        "num_matches": int(gt_depth.shape[0]),
        "point3D_id": pids_valid.astype(np.int64),
        "px": px.astype(np.int64),
        "py": py.astype(np.int64),
        "gt_depth": gt_depth.astype(np.float64),
    }


def evaluate_views_sparse_geometry_proxy(
    views: Sequence,
    triangles,
    render_func,
    pipe,
    background: torch.Tensor,
    ctx: GeometryProxyContext,
    cfg: GeometryProxyConfig,
) -> Dict:
    rng = np.random.default_rng(int(cfg.seed))
    all_depth_pred: List[np.ndarray] = []
    all_depth_gt: List[np.ndarray] = []
    all_cos_abs: List[np.ndarray] = []
    per_view: List[Dict] = []
    dropped_reasons: Dict[str, int] = {}

    for v in views:
        pkg = render_func(v, triangles, pipe, background)
        view_res = evaluate_view_sparse_geometry_proxy(
            view=v,
            render_pkg=pkg,
            ctx=ctx,
            cfg=cfg,
            rng=rng,
        )
        reason = str(view_res.get("reason", "unknown"))
        if reason != "ok":
            dropped_reasons[reason] = int(dropped_reasons.get(reason, 0)) + 1
        dp = view_res.get("_depth_pred", None)
        dg = view_res.get("_depth_gt", None)
        if isinstance(dp, np.ndarray) and isinstance(dg, np.ndarray) and dp.shape[0] > 0:
            all_depth_pred.append(dp)
            all_depth_gt.append(dg)
        ca = view_res.get("_cos_abs", None)
        if isinstance(ca, np.ndarray) and ca.shape[0] > 0:
            all_cos_abs.append(ca)
        per_view.append(view_res)

    depth_views = [p for p in per_view if int(p.get("depth_points", 0)) > 0 and p.get("depth_stats") is not None]
    normal_views = [p for p in per_view if int(p.get("normal_points", 0)) > 0 and p.get("normal_stats") is not None]
    total_depth = int(sum(int(p.get("depth_points", 0)) for p in depth_views))
    total_normal = int(sum(int(p.get("normal_points", 0)) for p in normal_views))

    depth_stats = None
    if len(all_depth_pred) > 0:
        depth_stats = depth_metrics(
            pred=np.concatenate(all_depth_pred, axis=0),
            gt=np.concatenate(all_depth_gt, axis=0),
        )

    normal_stats = None
    if bool(cfg.compute_normal) and len(all_cos_abs) > 0:
        normal_stats = normal_metrics_from_abs_cos(np.concatenate(all_cos_abs, axis=0))

    per_view_summary: List[Dict] = []
    for p in per_view:
        d_stats = p.get("depth_stats", None)
        per_view_summary.append(
            {
                "image_name": p.get("image_name", ""),
                "depth_points": int(p.get("depth_points", 0)),
                "depth_mae": float(d_stats["mae"]) if isinstance(d_stats, dict) else float("nan"),
                "depth_abs_rel": float(d_stats["abs_rel"]) if isinstance(d_stats, dict) else float("nan"),
                "normal_points": int(p.get("normal_points", 0)),
                "reason": str(p.get("reason", "unknown")),
            }
        )

    return {
        "num_views": int(len(views)),
        "num_depth_views_used": int(len(depth_views)),
        "num_normal_views_used": int(len(normal_views)),
        "total_valid_depth_matches": int(total_depth),
        "total_valid_normal_matches": int(total_normal),
        "dropped_views_reason_breakdown": dropped_reasons,
        "depth": depth_stats,
        "normal": normal_stats,
        "normal_note": "COLMAP does not provide ground-truth normals; normals are estimated by local PCA on sparse 3D points."
        if bool(cfg.compute_normal)
        else "normal evaluation disabled",
        "per_view": per_view_summary,
    }

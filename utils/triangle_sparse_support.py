from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from utils.colmap_sparse_utils import extract_colmap_sparse_points

try:
    from scipy.spatial import cKDTree  # type: ignore

    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False


@dataclass
class SparseSupportConfig:
    # Query policy
    radius: float = -1.0
    radius_factor: float = 0.02
    knn: int = 32
    min_support_points: int = 6
    # Local PCA normal
    pca_min_points: int = 10
    # Score shaping
    residual_scale: float = -1.0
    angle_scale_deg: float = 20.0
    # COLMAP point filtering
    max_point_error: float = 2.0
    # Backend behavior
    prefer_ckdtree: bool = True


@dataclass
class TriangleSparseSupportResult:
    support_count: torch.Tensor  # [T] int32
    plane_residual_mean: torch.Tensor  # [T] float32
    plane_residual_median: torch.Tensor  # [T] float32
    normal_angle_residual_deg: torch.Tensor  # [T] float32
    normal_angle_valid: torch.Tensor  # [T] bool
    confidence: torch.Tensor  # [T] float32, count-based confidence
    score_count_term: torch.Tensor  # [T] float32
    score_plane_term: torch.Tensor  # [T] float32
    score_normal_term: torch.Tensor  # [T] float32
    geometry_support_score_base: torch.Tensor  # [T] float32
    query_radius: float
    scene_scale: float


def _estimate_scene_scale(point_xyz: np.ndarray, scene=None) -> float:
    if scene is not None:
        ext = float(getattr(scene, "cameras_extent", 0.0))
        if ext > 1e-9:
            return max(2.0 * ext, 1e-6)
    if point_xyz.shape[0] > 0:
        mn = point_xyz.min(axis=0)
        mx = point_xyz.max(axis=0)
        diag = float(np.linalg.norm(mx - mn))
        if diag > 1e-9:
            return diag
    return 1.0


def _triangle_plane_and_centroid(
    vertices: torch.Tensor, triangle_indices: torch.Tensor
):
    tri = triangle_indices.to(torch.int64)
    pts = vertices[tri]  # [T,3,3]
    ab = pts[:, 1] - pts[:, 0]
    ac = pts[:, 2] - pts[:, 0]
    cross = torch.cross(ab, ac, dim=1)
    n = cross / torch.clamp(torch.linalg.norm(cross, dim=1, keepdim=True), min=1e-12)
    c = pts.mean(dim=1)
    d = -torch.sum(n * c, dim=1)
    return n, d, c


class TriangleSparseSupportEstimator:
    """
    Local sparse COLMAP support estimator (triangle-level).

    This module computes geometry support signals only; it does not perform pruning.
    """

    def __init__(
        self,
        point_xyz: np.ndarray,
        point_errors: np.ndarray,
        scene_scale: float,
        cfg: SparseSupportConfig,
    ):
        self.point_xyz = np.asarray(point_xyz, dtype=np.float64)
        self.point_errors = np.asarray(point_errors, dtype=np.float64)
        self.scene_scale = float(max(scene_scale, 1e-6))
        self.cfg = cfg
        if float(cfg.radius) > 0:
            self.query_radius = float(cfg.radius)
        else:
            self.query_radius = float(max(cfg.radius_factor, 1e-6) * self.scene_scale)
        if float(cfg.residual_scale) > 0:
            self.residual_scale = float(cfg.residual_scale)
        else:
            self.residual_scale = 0.01 * self.scene_scale

        self._kdtree = None
        if bool(cfg.prefer_ckdtree) and _HAS_CKDTREE and self.point_xyz.shape[0] > 0:
            self._kdtree = cKDTree(self.point_xyz)

    @classmethod
    def from_scene(cls, scene, cfg: Optional[SparseSupportConfig] = None):
        cfg = cfg or SparseSupportConfig()
        sparse = extract_colmap_sparse_points(
            colmap_points3d=getattr(scene.scene_info, "colmap_points3d", None),
            error_max=float(cfg.max_point_error),
        )
        scale = _estimate_scene_scale(point_xyz=sparse.xyz, scene=scene)
        return cls(
            point_xyz=sparse.xyz,
            point_errors=sparse.errors,
            scene_scale=scale,
            cfg=cfg,
        )

    def _query_support_indices(self, centers_np: np.ndarray):
        t = centers_np.shape[0]
        support_idx = []
        if self.point_xyz.shape[0] == 0:
            return [[] for _ in range(t)]

        if self._kdtree is not None:
            # radius-first for true local support
            support_idx = self._kdtree.query_ball_point(centers_np, r=self.query_radius)
            # kNN fallback for sparse local neighborhoods
            if int(self.cfg.knn) > 0:
                for i in range(t):
                    if len(support_idx[i]) >= int(self.cfg.min_support_points):
                        continue
                    k = min(int(self.cfg.knn), self.point_xyz.shape[0])
                    _, idx = self._kdtree.query(centers_np[i], k=k)
                    if np.isscalar(idx):
                        idx = [int(idx)]
                    else:
                        idx = [int(v) for v in np.asarray(idx).reshape(-1)]
                    support_idx[i] = sorted(set(support_idx[i] + idx))
            return support_idx

        # Brute-force fallback (no scipy): still functional, but slower.
        points = torch.from_numpy(self.point_xyz).to(torch.float32)
        centers = torch.from_numpy(centers_np).to(torch.float32)
        for i in range(t):
            d = torch.linalg.norm(points - centers[i].unsqueeze(0), dim=1)
            ids = torch.nonzero(d <= float(self.query_radius), as_tuple=True)[0].tolist()
            if len(ids) < int(self.cfg.min_support_points) and int(self.cfg.knn) > 0:
                k = min(int(self.cfg.knn), points.shape[0])
                _, topk = torch.topk(-d, k=k, largest=True, sorted=False)
                ids = sorted(set(ids + topk.tolist()))
            support_idx.append(ids)
        return support_idx

    def compute(self, vertices: torch.Tensor, triangle_indices: torch.Tensor) -> TriangleSparseSupportResult:
        tri = triangle_indices.to(torch.int64).contiguous()
        device = tri.device
        t = int(tri.shape[0])

        support_count = torch.zeros((t,), dtype=torch.int32, device=device)
        plane_residual_mean = torch.full((t,), float(self.scene_scale), dtype=torch.float32, device=device)
        plane_residual_median = torch.full((t,), float(self.scene_scale), dtype=torch.float32, device=device)
        normal_angle_residual_deg = torch.full((t,), 180.0, dtype=torch.float32, device=device)
        normal_angle_valid = torch.zeros((t,), dtype=torch.bool, device=device)

        if t == 0:
            zf = torch.zeros((0,), dtype=torch.float32, device=device)
            zb = torch.zeros((0,), dtype=torch.bool, device=device)
            zi = torch.zeros((0,), dtype=torch.int32, device=device)
            return TriangleSparseSupportResult(
                support_count=zi,
                plane_residual_mean=zf,
                plane_residual_median=zf,
                normal_angle_residual_deg=zf,
                normal_angle_valid=zb,
                confidence=zf,
                score_count_term=zf,
                score_plane_term=zf,
                score_normal_term=zf,
                geometry_support_score_base=zf,
                query_radius=float(self.query_radius),
                scene_scale=float(self.scene_scale),
            )

        n_tri, d_tri, c_tri = _triangle_plane_and_centroid(vertices=vertices, triangle_indices=tri)
        centers_np = c_tri.detach().cpu().numpy().astype(np.float64)
        support_lists = self._query_support_indices(centers_np)

        for i in range(t):
            idx = support_lists[i]
            cnt = len(idx)
            support_count[i] = int(cnt)
            if cnt <= 0:
                continue
            p = self.point_xyz[idx]  # [K,3]
            p_t = torch.from_numpy(p).to(device=device, dtype=vertices.dtype)
            signed = torch.abs(torch.sum(p_t * n_tri[i].unsqueeze(0), dim=1) + d_tri[i])
            plane_residual_mean[i] = torch.mean(signed).to(torch.float32)
            plane_residual_median[i] = torch.median(signed).to(torch.float32)

            if cnt >= int(self.cfg.pca_min_points):
                p0 = p - p.mean(axis=0, keepdims=True)
                cov = (p0.T @ p0) / max(1, p0.shape[0] - 1)
                try:
                    w, v = np.linalg.eigh(cov)
                    n_pca = v[:, int(np.argmin(w))]
                    n_pca = n_pca / max(np.linalg.norm(n_pca), 1e-12)
                    n_tri_i = n_tri[i].detach().cpu().numpy()
                    c = float(np.clip(abs(np.dot(n_tri_i, n_pca)), 0.0, 1.0))
                    angle = float(np.degrees(np.arccos(c)))
                    normal_angle_residual_deg[i] = angle
                    normal_angle_valid[i] = True
                except Exception:
                    pass

        confidence = torch.clamp(
            support_count.to(torch.float32) / max(float(self.cfg.min_support_points), 1.0),
            0.0,
            1.0,
        )
        score_count_term = confidence
        score_plane_term = torch.exp(-plane_residual_median / max(float(self.residual_scale), 1e-8))
        angle_scale = max(float(self.cfg.angle_scale_deg), 1e-6)
        score_normal_term = torch.exp(-normal_angle_residual_deg / angle_scale)
        # If no reliable normal estimate, make the normal term neutral instead of punitive.
        score_normal_term = torch.where(normal_angle_valid, score_normal_term, torch.ones_like(score_normal_term))
        geometry_support_score_base = score_count_term * score_plane_term * score_normal_term

        return TriangleSparseSupportResult(
            support_count=support_count,
            plane_residual_mean=plane_residual_mean,
            plane_residual_median=plane_residual_median,
            normal_angle_residual_deg=normal_angle_residual_deg,
            normal_angle_valid=normal_angle_valid,
            confidence=confidence,
            score_count_term=score_count_term,
            score_plane_term=score_plane_term,
            score_normal_term=score_normal_term,
            geometry_support_score_base=geometry_support_score_base,
            query_radius=float(self.query_radius),
            scene_scale=float(self.scene_scale),
        )

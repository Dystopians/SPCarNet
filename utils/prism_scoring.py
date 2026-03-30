from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from utils.triangle_stats import TriangleState


@dataclass
class PrismScoreConfig:
    # Utility weights
    utility_w_vis: float = 0.30
    utility_w_sens: float = 0.25
    utility_w_geo: float = 0.20
    utility_w_viewdiv: float = 0.15
    utility_w_edge: float = 0.10
    # Redundancy weights
    redund_w_flat: float = 0.70
    redund_w_coplanar: float = 0.30
    # Robust normalization
    norm_percentile_low: float = 5.0
    norm_percentile_high: float = 95.0
    norm_eps: float = 1e-6
    # Classifier thresholds
    thresh_protected_edge: float = 0.60
    thresh_protected_geo: float = 0.75
    thresh_protected_sens: float = 0.80
    thresh_protected_unc: float = 0.65
    thresh_dead_vis: float = 0.02
    thresh_dead_sens: float = 0.03
    thresh_dead_geo: float = 0.05
    thresh_dead_edge: float = 0.10
    thresh_suspicious_vis: float = 0.05
    thresh_suspicious_geo: float = 0.15
    thresh_suspicious_unc: float = 0.50
    # Risk controls
    boundary_risk_value: float = 1.0
    nonmanifold_risk_value: float = 1.0
    recent_age_iters: int = 500
    ground_protect_bonus: float = 1.0
    roi_protect_bonus: float = 1.0
    use_ground_protect: bool = False
    use_roi_protect: bool = False
    # Geometry/orientation keep signals (protect-oriented, not direct prune score terms)
    keep_support_count_min: float = 12.0
    keep_plane_residual_max: float = 0.02
    keep_normal_residual_max_deg: float = 15.0
    keep_orientation_dihedral_min_deg: float = 12.0
    keep_orientation_local_var_min: float = 0.25
    keep_geometry_threshold: float = 0.6
    keep_orientation_threshold: float = 0.6
    keep_render_threshold: float = 0.6
    keep_geometry_bonus: float = 1.0
    keep_orientation_bonus: float = 1.0
    keep_render_bonus: float = 1.0
    # Candidate gating
    candidate_block_geometry_keep_threshold: float = 0.6
    protected_dilation_rings: int = 1


@dataclass
class PrismScoreInputs:
    vis_count_ema: torch.Tensor
    grad_pos_norm_ema: torch.Tensor
    grad_app_norm_ema: torch.Tensor
    grad_norm_var_ema: torch.Tensor
    view_direction_histogram: torch.Tensor  # [T, B]
    birth_iter: torch.Tensor
    geometry_support_score_base: torch.Tensor
    boundary_edge_count: torch.Tensor
    nonmanifold_edge_count: torch.Tensor
    flatness_score: torch.Tensor
    coplanar_neighbor_fraction: torch.Tensor
    mean_abs_dihedral_deg: Optional[torch.Tensor] = None
    sparse_support_count: Optional[torch.Tensor] = None
    sparse_plane_residual: Optional[torch.Tensor] = None
    sparse_normal_residual_deg: Optional[torch.Tensor] = None
    render_keep_t: Optional[torch.Tensor] = None
    tri_neighbors: Optional[List[List[int]]] = None
    ground_protect_t: Optional[torch.Tensor] = None
    roi_protect_t: Optional[torch.Tensor] = None


@dataclass
class PrismScoreOutputs:
    vis_t: torch.Tensor
    sens_t: torch.Tensor
    geo_t: torch.Tensor
    viewdiv_t: torch.Tensor
    edge_t: torch.Tensor
    unc_t: torch.Tensor
    recent_t: torch.Tensor
    boundary_t: torch.Tensor
    nonmanifold_t: torch.Tensor
    optional_groundprotect_t: torch.Tensor
    optional_roiprotect_t: torch.Tensor
    geometry_keep_t: torch.Tensor
    orientation_keep_t: torch.Tensor
    render_keep_t: torch.Tensor
    utility_t: torch.Tensor
    risk_t: torch.Tensor
    redund_t: torch.Tensor
    prune_score_t: torch.Tensor
    triangle_state: torch.Tensor  # int64 TriangleState
    protected_mask: torch.Tensor
    protected_mask_raw: torch.Tensor
    protected_mask_dilated: torch.Tensor
    candidate_blocked_by_geometry_keep: torch.Tensor
    candidate_blocked_by_dilated_protect: torch.Tensor
    dead_mask: torch.Tensor
    suspicious_mask: torch.Tensor
    candidate_mask: torch.Tensor


def _robust_normalize(x: torch.Tensor, p_lo: float, p_hi: float, eps: float) -> torch.Tensor:
    if x.numel() == 0:
        return x
    q_lo = torch.quantile(x, q=float(max(0.0, min(1.0, p_lo / 100.0))))
    q_hi = torch.quantile(x, q=float(max(0.0, min(1.0, p_hi / 100.0))))
    scale = torch.clamp(q_hi - q_lo, min=float(eps))
    y = (x - q_lo) / scale
    return torch.clamp(y, 0.0, 1.0)


def _view_diversity_from_hist(hist: torch.Tensor, eps: float) -> torch.Tensor:
    # hist: [T, B], can be EMA counts.
    if hist.numel() == 0:
        return hist.sum(dim=1) if hist.dim() == 2 else hist
    p = hist / torch.clamp(hist.sum(dim=1, keepdim=True), min=float(eps))
    p = torch.clamp(p, min=float(eps))
    ent = -torch.sum(p * torch.log(p), dim=1)
    ent_max = torch.log(torch.tensor(float(hist.shape[1]), device=hist.device, dtype=hist.dtype))
    return torch.clamp(ent / torch.clamp(ent_max, min=float(eps)), 0.0, 1.0)


def _expand_protected_mask(
    protected_mask: torch.Tensor,
    tri_neighbors: Optional[List[List[int]]],
    rings: int,
) -> torch.Tensor:
    if (tri_neighbors is None) or (int(rings) <= 0) or (protected_mask.numel() == 0):
        return protected_mask
    out = protected_mask.clone()
    t = int(protected_mask.numel())
    frontier = torch.nonzero(protected_mask, as_tuple=True)[0].tolist()
    visited = set(frontier)
    for _ in range(int(rings)):
        nxt = []
        for tid in frontier:
            if tid < 0 or tid >= len(tri_neighbors):
                continue
            for n in tri_neighbors[tid]:
                ni = int(n)
                if ni < 0 or ni >= t or ni in visited:
                    continue
                visited.add(ni)
                nxt.append(ni)
        if len(nxt) == 0:
            break
        out[torch.tensor(nxt, dtype=torch.int64, device=out.device)] = True
        frontier = nxt
    return out


def compute_prism_scores(
    inputs: PrismScoreInputs,
    current_iter: int,
    cfg: PrismScoreConfig,
) -> PrismScoreOutputs:
    device = inputs.vis_count_ema.device
    n = int(inputs.vis_count_ema.numel())

    p_lo = float(cfg.norm_percentile_low)
    p_hi = float(cfg.norm_percentile_high)
    eps = float(cfg.norm_eps)

    vis_t = _robust_normalize(inputs.vis_count_ema.to(torch.float32), p_lo=p_lo, p_hi=p_hi, eps=eps)
    sens_raw = 0.5 * inputs.grad_pos_norm_ema.to(torch.float32) + 0.5 * inputs.grad_app_norm_ema.to(torch.float32)
    sens_t = _robust_normalize(sens_raw, p_lo=p_lo, p_hi=p_hi, eps=eps)
    geo_t = torch.clamp(inputs.geometry_support_score_base.to(torch.float32), 0.0, 1.0)
    viewdiv_t = _view_diversity_from_hist(inputs.view_direction_histogram.to(torch.float32), eps=eps)
    edge_raw = torch.clamp(inputs.boundary_edge_count.to(torch.float32) / 3.0, 0.0, 1.0)
    edge_t = edge_raw

    unc_t = _robust_normalize(inputs.grad_norm_var_ema.to(torch.float32), p_lo=p_lo, p_hi=p_hi, eps=eps)
    age = int(current_iter) - inputs.birth_iter.to(torch.int64)
    recent_t = (age < int(cfg.recent_age_iters)).to(torch.float32)
    boundary_t = (inputs.boundary_edge_count > 0).to(torch.float32) * float(cfg.boundary_risk_value)
    nonmanifold_t = (inputs.nonmanifold_edge_count > 0).to(torch.float32) * float(cfg.nonmanifold_risk_value)
    boundary_t = torch.clamp(boundary_t, 0.0, 1.0)
    nonmanifold_t = torch.clamp(nonmanifold_t, 0.0, 1.0)

    optional_groundprotect_t = torch.zeros((n,), dtype=torch.float32, device=device)
    if bool(cfg.use_ground_protect) and inputs.ground_protect_t is not None:
        optional_groundprotect_t = torch.clamp(
            inputs.ground_protect_t.to(torch.float32) * float(cfg.ground_protect_bonus), 0.0, 1.0
        )
    optional_roiprotect_t = torch.zeros((n,), dtype=torch.float32, device=device)
    # Geometry keep from sparse support quality.
    geometry_keep_t = torch.zeros((n,), dtype=torch.float32, device=device)
    if (
        inputs.sparse_support_count is not None
        and inputs.sparse_plane_residual is not None
        and inputs.sparse_normal_residual_deg is not None
    ):
        support_count = inputs.sparse_support_count.to(torch.float32)
        plane_res = inputs.sparse_plane_residual.to(torch.float32)
        normal_res = inputs.sparse_normal_residual_deg.to(torch.float32)
        support_term = torch.clamp(
            support_count / max(float(cfg.keep_support_count_min), 1e-6),
            0.0,
            1.0,
        )
        plane_term = torch.clamp(
            1.0 - (plane_res / max(float(cfg.keep_plane_residual_max), 1e-6)),
            0.0,
            1.0,
        )
        normal_term = torch.clamp(
            1.0 - (normal_res / max(float(cfg.keep_normal_residual_max_deg), 1e-6)),
            0.0,
            1.0,
        )
        geometry_keep_t = torch.clamp(
            support_term * torch.maximum(plane_term, normal_term) * float(cfg.keep_geometry_bonus),
            0.0,
            1.0,
        )

    # Orientation keep from structure signals.
    orientation_keep_t = torch.zeros((n,), dtype=torch.float32, device=device)
    if inputs.mean_abs_dihedral_deg is not None:
        dihedral = inputs.mean_abs_dihedral_deg.to(torch.float32)
        dihedral_term = torch.clamp(
            dihedral / max(float(cfg.keep_orientation_dihedral_min_deg), 1e-6),
            0.0,
            1.0,
        )
        local_var_term = torch.clamp(
            (1.0 - inputs.coplanar_neighbor_fraction.to(torch.float32))
            / max(float(cfg.keep_orientation_local_var_min), 1e-6),
            0.0,
            1.0,
        )
        orientation_keep_t = torch.clamp(
            torch.maximum(dihedral_term, local_var_term) * float(cfg.keep_orientation_bonus),
            0.0,
            1.0,
        )

    # Optional render-space keep (already computed in train when available).
    render_keep_t = torch.zeros((n,), dtype=torch.float32, device=device)
    if inputs.render_keep_t is not None:
        render_keep_t = torch.clamp(
            inputs.render_keep_t.to(torch.float32) * float(cfg.keep_render_bonus),
            0.0,
            1.0,
        )

    if bool(cfg.use_roi_protect) and inputs.roi_protect_t is not None:
        optional_roiprotect_t = torch.clamp(
            inputs.roi_protect_t.to(torch.float32) * float(cfg.roi_protect_bonus), 0.0, 1.0
        )

    utility_t = (
        float(cfg.utility_w_vis) * vis_t
        + float(cfg.utility_w_sens) * sens_t
        + float(cfg.utility_w_geo) * geo_t
        + float(cfg.utility_w_viewdiv) * viewdiv_t
        + float(cfg.utility_w_edge) * edge_t
    )
    utility_t = torch.clamp(utility_t, 0.0, 1.0)

    risk_t = torch.stack(
        [
            unc_t,
            recent_t,
            boundary_t,
            nonmanifold_t,
            optional_groundprotect_t,
            optional_roiprotect_t,
            (geometry_keep_t > float(cfg.keep_geometry_threshold)).to(torch.float32),
            (orientation_keep_t > float(cfg.keep_orientation_threshold)).to(torch.float32),
            (render_keep_t > float(cfg.keep_render_threshold)).to(torch.float32),
        ],
        dim=1,
    ).max(dim=1).values
    risk_t = torch.clamp(risk_t, 0.0, 1.0)

    flat_t = torch.clamp(inputs.flatness_score.to(torch.float32), 0.0, 1.0)
    coplanar_t = torch.clamp(inputs.coplanar_neighbor_fraction.to(torch.float32), 0.0, 1.0)
    redund_t = float(cfg.redund_w_flat) * flat_t + float(cfg.redund_w_coplanar) * coplanar_t
    redund_t = torch.clamp(redund_t, 0.0, 1.0)

    prune_score_t = redund_t * (1.0 - utility_t) * (1.0 - risk_t)
    prune_score_t = torch.clamp(prune_score_t, 0.0, 1.0)

    protected_mask_raw = (
        (edge_t > float(cfg.thresh_protected_edge))
        | (geo_t > float(cfg.thresh_protected_geo))
        | (sens_t > float(cfg.thresh_protected_sens))
        | (unc_t > float(cfg.thresh_protected_unc))
        | (recent_t > 0.0)
        | (geometry_keep_t > float(cfg.keep_geometry_threshold))
        | (orientation_keep_t > float(cfg.keep_orientation_threshold))
        | (render_keep_t > float(cfg.keep_render_threshold))
    )
    protected_mask_dilated = _expand_protected_mask(
        protected_mask=protected_mask_raw,
        tri_neighbors=inputs.tri_neighbors,
        rings=int(max(0, cfg.protected_dilation_rings)),
    )
    protected_mask = protected_mask_dilated
    dead_mask = (
        (vis_t < float(cfg.thresh_dead_vis))
        & (sens_t < float(cfg.thresh_dead_sens))
        & (geo_t < float(cfg.thresh_dead_geo))
        & (edge_t < float(cfg.thresh_dead_edge))
    ) & (~protected_mask)
    suspicious_mask = (
        (vis_t > float(cfg.thresh_suspicious_vis))
        & (geo_t < float(cfg.thresh_suspicious_geo))
        & (unc_t > float(cfg.thresh_suspicious_unc))
    ) & (~protected_mask) & (~dead_mask)

    triangle_state = torch.full((n,), int(TriangleState.ACTIVE), dtype=torch.int64, device=device)
    triangle_state[protected_mask] = int(TriangleState.PROTECTED)
    triangle_state[dead_mask] = int(TriangleState.DEAD)
    triangle_state[suspicious_mask] = int(TriangleState.SUSPICIOUS)
    candidate_mask = triangle_state == int(TriangleState.ACTIVE)
    candidate_blocked_by_geometry_keep = (
        geometry_keep_t > float(cfg.candidate_block_geometry_keep_threshold)
    )
    candidate_blocked_by_dilated_protect = protected_mask_dilated
    candidate_mask = candidate_mask & (~candidate_blocked_by_geometry_keep) & (~candidate_blocked_by_dilated_protect)

    return PrismScoreOutputs(
        vis_t=vis_t,
        sens_t=sens_t,
        geo_t=geo_t,
        viewdiv_t=viewdiv_t,
        edge_t=edge_t,
        unc_t=unc_t,
        recent_t=recent_t,
        boundary_t=boundary_t,
        nonmanifold_t=nonmanifold_t,
        optional_groundprotect_t=optional_groundprotect_t,
        optional_roiprotect_t=optional_roiprotect_t,
        geometry_keep_t=geometry_keep_t,
        orientation_keep_t=orientation_keep_t,
        render_keep_t=render_keep_t,
        utility_t=utility_t,
        risk_t=risk_t,
        redund_t=redund_t,
        prune_score_t=prune_score_t,
        triangle_state=triangle_state,
        protected_mask=protected_mask,
        protected_mask_raw=protected_mask_raw,
        protected_mask_dilated=protected_mask_dilated,
        candidate_blocked_by_geometry_keep=candidate_blocked_by_geometry_keep,
        candidate_blocked_by_dilated_protect=candidate_blocked_by_dilated_protect,
        dead_mask=dead_mask,
        suspicious_mask=suspicious_mask,
        candidate_mask=candidate_mask,
    )


def summarize_prism_scores(scores: PrismScoreOutputs) -> Dict[str, float]:
    def _mean(x: torch.Tensor) -> float:
        return float(x.mean().item()) if x.numel() > 0 else 0.0

    return {
        "num_triangles": float(scores.prune_score_t.numel()),
        "mean_utility": _mean(scores.utility_t),
        "mean_risk": _mean(scores.risk_t),
        "mean_redund": _mean(scores.redund_t),
        "mean_prune_score": _mean(scores.prune_score_t),
        "num_protected": float(scores.protected_mask.sum().item()),
        "num_protected_raw": float(scores.protected_mask_raw.sum().item()),
        "num_protected_dilated": float(scores.protected_mask_dilated.sum().item()),
        "num_dead": float(scores.dead_mask.sum().item()),
        "num_suspicious": float(scores.suspicious_mask.sum().item()),
        "num_candidates": float(scores.candidate_mask.sum().item()),
    }

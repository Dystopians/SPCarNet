from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os

import torch
import torch.nn.functional as F

from utils.ground_plane_utils import point_to_plane_signed_distance


@dataclass
class GroundRegConfig:
    enabled: bool
    start_iter: int
    warmup_iters: int
    lambda_plane: float
    lambda_normal: float
    lambda_smoothness: float
    global_scale: float
    target_ratio: float
    adaptive_ema_decay: float
    adaptive_min_scale: float
    adaptive_max_scale: float
    max_total: float
    smooth_start_iter: int
    huber_delta: float
    assignment_min_pixels: int
    assignment_ema_decay: float
    plane_min_ratio: float
    plane_max_abs_height: float
    normal_min_ratio: float
    normal_max_abs_height: float
    smooth_min_ratio: float
    smooth_max_abs_height: float
    smooth_max_pairs: int
    smooth_fallback_edges_max: int


def _safe_huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    abs_x = x.abs()
    quadratic = torch.minimum(abs_x, abs_x.new_tensor(delta))
    linear = abs_x - quadratic
    return 0.5 * quadratic * quadratic + delta * linear


def _resize_mask_to_ids(mask_hw: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if mask_hw.shape[0] == out_h and mask_hw.shape[1] == out_w:
        return mask_hw
    mask = mask_hw.float().unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(out_h, out_w), mode="nearest")
    return (mask.squeeze(0).squeeze(0) > 0.5)


def _ensure_ground_state(ground_state: Optional[Dict], num_triangles: int, device: torch.device) -> Dict:
    if ground_state is None:
        ground_state = {}
    ratio = ground_state.get("ema_ratio", None)
    pixels = ground_state.get("ema_pixels", None)
    seen = ground_state.get("ema_seen", None)
    if ratio is None or pixels is None or seen is None or ratio.numel() != num_triangles:
        ground_state = {
            "ema_ratio": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
            "ema_pixels": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
            "ema_seen": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
        }
    return ground_state


def assign_ground_labels_to_mesh_elements(
    render_pkg: Dict,
    viewpoint_cam,
    num_triangles: int,
    ground_state: Optional[Dict],
    min_pixels: int,
    ema_decay: float,
) -> Tuple[Dict, Dict]:
    device = render_pkg["rend_ids"].device
    ground_state = _ensure_ground_state(ground_state, num_triangles=num_triangles, device=device)

    if getattr(viewpoint_cam, "ground_mask", None) is None:
        return {
            "available": False,
            "reason": "no_ground_mask",
            "valid_triangles": torch.zeros((num_triangles,), dtype=torch.bool, device=device),
            "ground_ratio": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
            "pixel_count": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
        }, ground_state

    ids = render_pkg["rend_ids"].squeeze(0).long()
    h, w = int(ids.shape[0]), int(ids.shape[1])
    mask = _resize_mask_to_ids(viewpoint_cam.ground_mask, out_h=h, out_w=w).to(device)

    valid = (ids >= 0) & (ids < num_triangles)
    if not torch.any(valid):
        return {
            "available": False,
            "reason": "no_valid_ids",
            "valid_triangles": torch.zeros((num_triangles,), dtype=torch.bool, device=device),
            "ground_ratio": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
            "pixel_count": torch.zeros((num_triangles,), dtype=torch.float32, device=device),
        }, ground_state

    ids_valid = ids[valid]
    ids_ground = ids[valid & mask]

    total = torch.bincount(ids_valid, minlength=num_triangles).to(torch.float32)
    ground = torch.bincount(ids_ground, minlength=num_triangles).to(torch.float32)
    ratio = ground / torch.clamp(total, min=1.0)
    observed = total > 0

    decay = float(max(0.0, min(ema_decay, 0.9999)))
    one_minus = 1.0 - decay
    ground_state["ema_ratio"][observed] = decay * ground_state["ema_ratio"][observed] + one_minus * ratio[observed]
    ground_state["ema_pixels"][observed] = decay * ground_state["ema_pixels"][observed] + one_minus * total[observed]
    ground_state["ema_seen"][observed] = decay * ground_state["ema_seen"][observed] + one_minus
    ground_state["ema_seen"][~observed] = decay * ground_state["ema_seen"][~observed]
    ground_state["ema_pixels"][~observed] = decay * ground_state["ema_pixels"][~observed]

    reliable = (ground_state["ema_pixels"] >= float(min_pixels)) & (ground_state["ema_seen"] > 0.05)
    return {
        "available": True,
        "reason": "ok",
        "valid_triangles": reliable,
        "ground_ratio": ground_state["ema_ratio"].clone(),
        "pixel_count": ground_state["ema_pixels"].clone(),
    }, ground_state


def compute_ground_plane_loss(
    triangles,
    plane_normal: torch.Tensor,
    plane_offset: torch.Tensor,
    assignment: Dict,
    cfg: GroundRegConfig,
) -> Tuple[torch.Tensor, Dict]:
    device = triangles.vertices.device
    zero = torch.tensor(0.0, device=device)
    valid = assignment["valid_triangles"] & (assignment["ground_ratio"] >= float(cfg.plane_min_ratio))
    if not torch.any(valid):
        return zero, {"count": 0}

    tri_ids = torch.nonzero(valid, as_tuple=True)[0]
    tri = triangles._triangle_indices[tri_ids]
    pts = triangles.vertices[tri]
    centroids = pts.mean(dim=1)
    signed_h = point_to_plane_signed_distance(centroids, plane_normal, plane_offset)
    if float(cfg.plane_max_abs_height) > 0:
        keep = signed_h.abs() <= float(cfg.plane_max_abs_height)
        if torch.any(keep):
            signed_h = signed_h[keep]
        else:
            return zero, {"count": 0}

    loss = _safe_huber(signed_h, float(cfg.huber_delta)).mean()
    if not torch.isfinite(loss):
        return zero, {"count": 0, "nan_guard": 1}
    return loss, {"count": int(signed_h.numel())}


def compute_ground_normal_loss(
    triangles,
    plane_normal: torch.Tensor,
    plane_offset: torch.Tensor,
    assignment: Dict,
    cfg: GroundRegConfig,
) -> Tuple[torch.Tensor, Dict]:
    device = triangles.vertices.device
    zero = torch.tensor(0.0, device=device)
    valid = assignment["valid_triangles"] & (assignment["ground_ratio"] >= float(cfg.normal_min_ratio))
    if not torch.any(valid):
        return zero, {"count": 0}

    tri_ids = torch.nonzero(valid, as_tuple=True)[0]
    tri = triangles._triangle_indices[tri_ids]
    pts = triangles.vertices[tri]
    if float(cfg.normal_max_abs_height) > 0:
        centroids = pts.mean(dim=1)
        signed_h = point_to_plane_signed_distance(centroids, plane_normal, plane_offset)
        keep_h = signed_h.abs() <= float(cfg.normal_max_abs_height)
        if torch.any(keep_h):
            pts = pts[keep_h]
        else:
            return zero, {"count": 0}
    ab = pts[:, 1] - pts[:, 0]
    ac = pts[:, 2] - pts[:, 0]
    normals = torch.cross(ab, ac, dim=1)
    n_norm = torch.linalg.norm(normals, dim=1, keepdim=True)
    good = n_norm.squeeze(1) > 1e-10
    if not torch.any(good):
        return zero, {"count": 0}
    normals = normals[good] / n_norm[good]
    align = torch.abs(normals @ plane_normal)
    residual = 1.0 - torch.clamp(align, 0.0, 1.0)
    loss = _safe_huber(residual, float(cfg.huber_delta)).mean()
    if not torch.isfinite(loss):
        return zero, {"count": 0, "nan_guard": 1}
    return loss, {"count": int(residual.numel())}


def _triangle_adjacency_pairs(triangles_idx: torch.Tensor, max_pairs: int) -> torch.Tensor:
    if triangles_idx.numel() == 0 or int(max_pairs) <= 0:
        return torch.empty((0, 2), dtype=torch.long, device=triangles_idx.device)
    e01 = triangles_idx[:, [0, 1]]
    e12 = triangles_idx[:, [1, 2]]
    e02 = triangles_idx[:, [0, 2]]
    edges = torch.cat([e01, e12, e02], dim=0)
    tri_ids = torch.arange(triangles_idx.shape[0], device=triangles_idx.device).repeat(3)
    edges = torch.sort(edges, dim=1).values

    edges_cpu = edges.detach().cpu()
    tri_cpu = tri_ids.detach().cpu()
    pairs = []
    edge_map = {}
    for idx in range(edges_cpu.shape[0]):
        key = (int(edges_cpu[idx, 0]), int(edges_cpu[idx, 1]))
        t = int(tri_cpu[idx])
        prev = edge_map.get(key, None)
        if prev is None:
            edge_map[key] = t
        else:
            if prev != t:
                pairs.append((prev, t))
                edge_map[key] = -1
    if len(pairs) == 0:
        return torch.empty((0, 2), dtype=torch.long, device=triangles_idx.device)
    pairs_t = torch.tensor(pairs, dtype=torch.long, device=triangles_idx.device)
    if pairs_t.shape[0] > int(max_pairs):
        step = max(1, pairs_t.shape[0] // int(max_pairs))
        pairs_t = pairs_t[::step][: int(max_pairs)]
    return pairs_t


def compute_ground_smoothness_loss(
    triangles,
    plane_normal: torch.Tensor,
    plane_offset: torch.Tensor,
    assignment: Dict,
    cfg: GroundRegConfig,
) -> Tuple[torch.Tensor, Dict]:
    device = triangles.vertices.device
    zero = torch.tensor(0.0, device=device)

    valid = assignment["valid_triangles"] & (assignment["ground_ratio"] >= float(cfg.smooth_min_ratio))
    if not torch.any(valid):
        return zero, {"count": 0, "mode": "none"}

    global_tri_ids = torch.nonzero(valid, as_tuple=True)[0]
    local_tri = triangles._triangle_indices[global_tri_ids]
    local_pts = triangles.vertices[local_tri]
    centroids = local_pts.mean(dim=1)
    heights = point_to_plane_signed_distance(centroids, plane_normal, plane_offset)
    if float(cfg.smooth_max_abs_height) > 0:
        keep = heights.abs() <= float(cfg.smooth_max_abs_height)
        if torch.any(keep):
            local_tri = local_tri[keep]
            centroids = centroids[keep]
            heights = heights[keep]
        else:
            return zero, {"count": 0, "mode": "none"}

    max_pairs = int(cfg.smooth_max_pairs)
    # Triangle-adjacency building uses CPU-side hashing and can dominate runtime
    # once ground triangles grow. Keep it only for small local sets.
    max_tri_for_adj = 4096
    if max_pairs > 0 and local_tri.shape[0] <= max_tri_for_adj:
        pairs = _triangle_adjacency_pairs(local_tri, max_pairs=max_pairs)
        if pairs.shape[0] > 0:
            diffs = heights[pairs[:, 0]] - heights[pairs[:, 1]]
            loss = _safe_huber(diffs, float(cfg.huber_delta)).mean()
            if not torch.isfinite(loss):
                return zero, {"count": 0, "mode": "tri_adj", "nan_guard": 1}
            return loss, {"count": int(diffs.numel()), "mode": "tri_adj"}

    # Fallback: vertex-edge smoothing within selected triangles.
    e01 = local_tri[:, [0, 1]]
    e12 = local_tri[:, [1, 2]]
    e02 = local_tri[:, [0, 2]]
    edges = torch.cat([e01, e12, e02], dim=0)
    edges = torch.sort(edges, dim=1).values
    edges = torch.unique(edges, dim=0)
    if edges.shape[0] == 0:
        return zero, {"count": 0, "mode": "none"}
    if edges.shape[0] > int(cfg.smooth_fallback_edges_max):
        step = max(1, edges.shape[0] // int(cfg.smooth_fallback_edges_max))
        edges = edges[::step][: int(cfg.smooth_fallback_edges_max)]

    v = triangles.vertices
    h0 = point_to_plane_signed_distance(v[edges[:, 0]], plane_normal, plane_offset)
    h1 = point_to_plane_signed_distance(v[edges[:, 1]], plane_normal, plane_offset)
    diffs = h0 - h1
    loss = _safe_huber(diffs, float(cfg.huber_delta)).mean()
    if not torch.isfinite(loss):
        return zero, {"count": 0, "mode": "vert_edge", "nan_guard": 1}
    return loss, {"count": int(diffs.numel()), "mode": "vert_edge"}


def aggregate_ground_regularization_losses(
    triangles,
    render_pkg: Dict,
    viewpoint_cam,
    plane_payload: Optional[Dict],
    ground_state: Optional[Dict],
    association_stats: Optional[Dict],
    cfg: GroundRegConfig,
    iteration: int,
    image_loss_ref: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict, Dict]:
    device = triangles.vertices.device
    zero = torch.tensor(0.0, device=device)
    if ground_state is None:
        ground_state = {}
    logs = {
        "enabled": 0.0,
        "available": 0.0,
        "warmup": 0.0,
        "Lground_plane_pure": 0.0,
        "Lground_normal_pure": 0.0,
        "Lground_smooth_pure": 0.0,
        "Lground_plane": 0.0,
        "Lground_normal": 0.0,
        "Lground_smooth": 0.0,
        "Lground_raw": 0.0,
        "Lground_total": 0.0,
        "ground_reg_global_scale": 1.0,
        "ground_reg_adaptive_scale": 1.0,
        "ground_reg_adaptive_scale_raw": 1.0,
        "ground_reg_adaptive_target": 0.0,
        "ground_reg_adaptive_ema": 0.0,
        "ground_smooth_warmup": 0.0,
        "ground_triangles_reliable": 0.0,
        "ground_plane_elements": 0.0,
        "ground_normal_elements": 0.0,
        "ground_smooth_elements": 0.0,
        "ground_smooth_mode": 0.0,
    }
    if not cfg.enabled:
        return zero, logs, ground_state

    logs["enabled"] = 1.0
    if plane_payload is None or (not plane_payload.get("ok", False)) or (not plane_payload.get("enabled_for_loss", False)):
        return zero, logs, ground_state

    if association_stats is not None:
        assignment = {
            "available": True,
            "reason": "from_tracker",
            "valid_triangles": association_stats["is_ground_mask"].to(device),
            "ground_ratio": association_stats["ground_support_ratio"].to(device),
            "pixel_count": association_stats["observations"].to(device),
        }
    else:
        assignment, ground_state = assign_ground_labels_to_mesh_elements(
            render_pkg=render_pkg,
            viewpoint_cam=viewpoint_cam,
            num_triangles=int(triangles._triangle_indices.shape[0]),
            ground_state=ground_state,
            min_pixels=int(cfg.assignment_min_pixels),
            ema_decay=float(cfg.assignment_ema_decay),
        )
    if not assignment["available"]:
        return zero, logs, ground_state
    logs["available"] = 1.0
    logs["ground_triangles_reliable"] = float(assignment["valid_triangles"].sum().item())

    start = int(cfg.start_iter)
    warm_iters = max(int(cfg.warmup_iters), 1)
    if iteration < start:
        warmup = 0.0
    else:
        warmup = min(float(iteration - start) / float(warm_iters), 1.0)
    logs["warmup"] = warmup
    if warmup <= 0:
        return zero, logs, ground_state

    plane_normal = torch.tensor(plane_payload["normal"], dtype=triangles.vertices.dtype, device=device)
    plane_normal = plane_normal / torch.clamp(torch.linalg.norm(plane_normal), min=1e-10)
    plane_offset = torch.tensor(float(plane_payload["offset"]), dtype=triangles.vertices.dtype, device=device)

    Lp_pure, info_p = compute_ground_plane_loss(
        triangles=triangles,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        assignment=assignment,
        cfg=cfg,
    )
    Ln_pure, info_n = compute_ground_normal_loss(
        triangles=triangles,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        assignment=assignment,
        cfg=cfg,
    )
    Ls_pure, info_s = compute_ground_smoothness_loss(
        triangles=triangles,
        plane_normal=plane_normal,
        plane_offset=plane_offset,
        assignment=assignment,
        cfg=cfg,
    )

    logs["Lground_plane_pure"] = float(Lp_pure.detach().item())
    logs["Lground_normal_pure"] = float(Ln_pure.detach().item())
    logs["Lground_smooth_pure"] = float(Ls_pure.detach().item())
    logs["ground_plane_elements"] = float(info_p.get("count", 0))
    logs["ground_normal_elements"] = float(info_n.get("count", 0))
    logs["ground_smooth_elements"] = float(info_s.get("count", 0))
    logs["ground_smooth_mode"] = 1.0 if info_s.get("mode", "none") == "tri_adj" else (0.5 if info_s.get("mode", "none") == "vert_edge" else 0.0)

    smooth_start = int(cfg.smooth_start_iter)
    if smooth_start < 0:
        smooth_start = start
    if iteration < smooth_start:
        smooth_warmup = 0.0
    else:
        smooth_warmup = min(float(iteration - smooth_start) / float(warm_iters), 1.0)
    logs["ground_smooth_warmup"] = smooth_warmup

    Lp = warmup * float(cfg.lambda_plane) * Lp_pure
    Ln = warmup * float(cfg.lambda_normal) * Ln_pure
    Ls = smooth_warmup * float(cfg.lambda_smoothness) * Ls_pure
    raw_total = Lp + Ln + Ls
    if not torch.isfinite(raw_total):
        return zero, logs, ground_state

    logs["Lground_raw"] = float(raw_total.detach().item())
    global_scale = float(cfg.global_scale)
    scaled_total = raw_total * global_scale
    logs["ground_reg_global_scale"] = global_scale

    adaptive_scale = 1.0
    adaptive_scale_raw = 1.0
    target_value = 0.0
    adaptive_ema = float(scaled_total.detach().item())
    if float(cfg.target_ratio) > 0.0 and image_loss_ref is not None:
        if torch.is_tensor(image_loss_ref):
            image_ref_val = float(image_loss_ref.detach().item())
        else:
            image_ref_val = float(image_loss_ref)
        target_value = float(cfg.target_ratio) * max(image_ref_val, 0.0)
        decay = float(max(0.0, min(cfg.adaptive_ema_decay, 0.9999)))
        prev_ema = ground_state.get("reg_total_ema", None)
        if prev_ema is None:
            adaptive_ema = float(scaled_total.detach().item())
        else:
            adaptive_ema = float(prev_ema)
        adaptive_ema = decay * adaptive_ema + (1.0 - decay) * float(scaled_total.detach().item())
        ground_state["reg_total_ema"] = adaptive_ema

        denom = max(adaptive_ema, 1e-12)
        adaptive_scale_raw = target_value / denom
        adaptive_scale = max(float(cfg.adaptive_min_scale), min(float(cfg.adaptive_max_scale), adaptive_scale_raw))
        scaled_total = scaled_total * adaptive_scale

    logs["ground_reg_adaptive_scale_raw"] = float(adaptive_scale_raw)
    logs["ground_reg_adaptive_scale"] = float(adaptive_scale)
    logs["ground_reg_adaptive_target"] = float(target_value)
    logs["ground_reg_adaptive_ema"] = float(adaptive_ema)

    max_total = float(cfg.max_total)
    if max_total > 0.0:
        total_abs = torch.abs(scaled_total.detach())
        if float(total_abs.item()) > max_total:
            scaled_total = scaled_total * (max_total / float(total_abs.item()))

    if not torch.isfinite(scaled_total):
        return zero, logs, ground_state

    logs["Lground_plane"] = float(Lp.detach().item())
    logs["Lground_normal"] = float(Ln.detach().item())
    logs["Lground_smooth"] = float(Ls.detach().item())
    logs["Lground_total"] = float(scaled_total.detach().item())
    return scaled_total, logs, ground_state


def maybe_save_ground_geometry_diagnostics(
    triangles,
    plane_payload: Optional[Dict],
    association_stats: Optional[Dict],
    out_dir: str,
    iteration: int,
):
    if plane_payload is None or (not plane_payload.get("ok", False)):
        return
    if association_stats is None:
        return
    device = triangles.vertices.device
    ground_mask = association_stats["is_ground_mask"].to(device)
    if not torch.any(ground_mask):
        return

    tri_ids = torch.nonzero(ground_mask, as_tuple=True)[0]
    tri = triangles._triangle_indices[tri_ids]
    pts = triangles.vertices[tri]
    centroids = pts.mean(dim=1)

    plane_normal = torch.tensor(plane_payload["normal"], dtype=triangles.vertices.dtype, device=device)
    plane_normal = plane_normal / torch.clamp(torch.linalg.norm(plane_normal), min=1e-10)
    plane_offset = torch.tensor(float(plane_payload["offset"]), dtype=triangles.vertices.dtype, device=device)

    signed_h = point_to_plane_signed_distance(centroids, plane_normal, plane_offset).detach().cpu().numpy()

    ab = pts[:, 1] - pts[:, 0]
    ac = pts[:, 2] - pts[:, 0]
    normals = torch.cross(ab, ac, dim=1)
    n_norm = torch.linalg.norm(normals, dim=1, keepdim=True)
    valid = n_norm.squeeze(1) > 1e-10
    if torch.any(valid):
        normals = normals[valid] / n_norm[valid]
        cosang = torch.clamp(torch.abs(normals @ plane_normal), 0.0, 1.0)
        angles = torch.rad2deg(torch.arccos(cosang)).detach().cpu().numpy()
    else:
        angles = None

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"geom_diag_iter_{int(iteration):06d}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"iteration={int(iteration)}\n")
        f.write(f"ground_triangles={int(tri_ids.numel())}\n")
        f.write(f"signed_height_mean={float(signed_h.mean()):.6f}\n")
        f.write(f"signed_height_std={float(signed_h.std()):.6f}\n")
        f.write(f"signed_height_q95={float(float(torch.tensor(signed_h).abs().quantile(0.95))):.6f}\n")
        if angles is not None and len(angles) > 0:
            f.write(f"normal_angle_mean_deg={float(angles.mean()):.6f}\n")
            f.write(f"normal_angle_q95_deg={float(float(torch.tensor(angles).quantile(0.95))):.6f}\n")

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 4))
        plt.hist(signed_h, bins=80)
        plt.xlabel("signed point-to-plane distance")
        plt.ylabel("count")
        plt.title(f"Ground signed distances @iter {int(iteration)}")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"height_hist_iter_{int(iteration):06d}.png"), dpi=180)
        plt.close(fig)

        if angles is not None and len(angles) > 0:
            fig = plt.figure(figsize=(7, 4))
            plt.hist(angles, bins=80)
            plt.xlabel("normal alignment angle (deg)")
            plt.ylabel("count")
            plt.title(f"Ground normal angles @iter {int(iteration)}")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"normal_angle_hist_iter_{int(iteration):06d}.png"), dpi=180)
            plt.close(fig)
    except Exception as exc:
        print(f"[GroundReg] Failed to save geometry diagnostics: {exc}")

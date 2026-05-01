"""SP-CarNet Stage 4 — observation-consistency losses for test-time MAP refinement.

Pure-functional loss API. Each public function takes the **frozen** Stage-2
``SPCarShapeFieldDecoder`` plus a single-object batch of query tensors, and
returns a scalar loss.

Constraints (per Stage-4 design §6):
- No backprop through Marching-Cubes (none invoked here).
- Refine ``z``, not the decoder. Loss never sees decoder params.
- No clean target points appear in any signature below.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Robust penalties
# ---------------------------------------------------------------------------


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Element-wise Huber penalty with threshold ``delta``."""
    abs_x = x.abs()
    quad = 0.5 * x.pow(2)
    lin = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quad, lin)


def _bce_with_huber(logits: torch.Tensor, target_value: float, delta: float) -> torch.Tensor:
    """BCE-with-logits wrapped in Huber. Targets are scalar ``target_value``."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    target = torch.full_like(logits, target_value)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return huber(raw, delta).mean()


# ---------------------------------------------------------------------------
# Tier-1 losses (always available)
# ---------------------------------------------------------------------------


def observed_surface_field_loss(
    decoder,
    z: torch.Tensor,
    observed_points: torch.Tensor,
    *,
    delta: float = 0.5,
    field_kind: str = "occupancy",
    sdf_margin: float = 0.05,
) -> torch.Tensor:
    """Push observed surface points toward the iso level.

    Parameters
    ----------
    decoder:
        ``SPCarShapeFieldDecoder`` (frozen).
    z:
        Latent code of shape ``(B, d_z)``.
    observed_points:
        ``(B, N_obs, 3)`` partial-observation point cloud.
    delta:
        Huber threshold.
    field_kind:
        ``"occupancy"`` (BCE→1) or ``"sdf"`` (Huber on |f|).
    """
    if observed_points.numel() == 0:
        return z.new_zeros(())
    field = decoder(observed_points, z)  # (B, N_obs)
    if field_kind == "occupancy":
        return _bce_with_huber(field, 1.0, delta)
    if field_kind == "sdf":
        scaled = (field / max(sdf_margin, 1e-6)).clamp(min=-2.0, max=2.0)
        return huber(scaled, delta).mean()
    raise ValueError(f"Unknown field_kind: {field_kind!r}")


def free_space_loss(
    decoder,
    z: torch.Tensor,
    free_points: torch.Tensor,
    hard_negatives: torch.Tensor | None,
    *,
    delta: float = 0.5,
    alpha_hard: float = 2.0,
    field_kind: str = "occupancy",
    sdf_margin: float = 0.05,
) -> torch.Tensor:
    """Push known-empty queries to ``f < iso``.

    ``hard_negatives`` is up-weighted by ``alpha_hard``.
    """
    if free_points.numel() == 0 and (hard_negatives is None or hard_negatives.numel() == 0):
        return z.new_zeros(())

    def _loss(points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return z.new_zeros(())
        field = decoder(points, z)
        if field_kind == "occupancy":
            return _bce_with_huber(field, 0.0, delta)
        if field_kind == "sdf":
            # SDF should be > +sdf_margin: penalise deviations below.
            scaled = ((sdf_margin - field) / max(sdf_margin, 1e-6)).clamp(min=-2.0, max=2.0)
            return huber(F.relu(scaled), delta).mean()
        raise ValueError(f"Unknown field_kind: {field_kind!r}")

    main = _loss(free_points)
    if hard_negatives is None or hard_negatives.numel() == 0:
        return main
    return main + alpha_hard * _loss(hard_negatives)


def mixed_query_loss(
    decoder,
    z: torch.Tensor,
    points: torch.Tensor,
    labels: torch.Tensor,
    ignore_mask: torch.Tensor | None,
    *,
    delta: float = 0.5,
    field_kind: str = "occupancy",
    sdf_margin: float = 0.05,
) -> torch.Tensor:
    """BCE on combined queries with the dataset's ignore mask honoured."""
    if points.numel() == 0:
        return z.new_zeros(())
    field = decoder(points, z)  # (B, Q)
    if ignore_mask is not None:
        keep = ~ignore_mask
        field = field[keep]
        target = labels[keep].float()
    else:
        target = labels.float()
    if field.numel() == 0:
        return z.new_zeros(())
    if field_kind == "occupancy":
        raw = F.binary_cross_entropy_with_logits(field, target, reduction="none")
        return huber(raw, delta).mean()
    if field_kind == "sdf":
        # +1 means inside, 0 means outside in mixed labels — convert to SDF target.
        sdf_target = torch.where(target > 0.5, -sdf_margin, sdf_margin).float()
        diff = (field - sdf_target) / max(sdf_margin, 1e-6)
        return huber(diff, delta).mean()
    raise ValueError(f"Unknown field_kind: {field_kind!r}")


# ---------------------------------------------------------------------------
# Tier-2 losses (need scanner pose / partial normals)
# ---------------------------------------------------------------------------


def ray_consistency_loss(
    decoder,
    z: torch.Tensor,
    observed_points: torch.Tensor,
    scanner_pose: torch.Tensor,
    *,
    num_samples: int = 8,
    pre_hit_margin: float = 0.0,
    alpha_hit: float = 0.0,
    delta: float = 0.5,
    field_kind: str = "occupancy",
    sdf_margin: float = 0.05,
) -> torch.Tensor:
    """Free-space along visible rays from ``scanner_pose`` to each observed point.

    Parameters
    ----------
    scanner_pose:
        ``(B, 3)`` scanner position per object (canonical coords).
    num_samples:
        ``K_seg`` — number of samples along the ray segment.
    pre_hit_margin:
        Fraction (0–1) by which to retract the last sample from the hit point;
        avoids placing a free-space sample exactly at the surface.
    alpha_hit:
        Weight for an extra surface BCE at the hit; redundant with
        ``observed_surface_field_loss``, off (0) by default.
    """
    if observed_points.numel() == 0 or num_samples <= 1:
        return z.new_zeros(())
    B, N, _ = observed_points.shape
    device = observed_points.device
    # Linearly-spaced parameters t ∈ [0, 1 − pre_hit_margin / (num_samples - 1)].
    end = 1.0 - pre_hit_margin / max(num_samples - 1, 1)
    ts = torch.linspace(0.0, end, num_samples, device=device, dtype=observed_points.dtype)
    # (B, 1, 1, 3) + ts·(B, N, 1, 3) → (B, N, K_seg, 3)
    c = scanner_pose.view(B, 1, 1, 3)
    direction = (observed_points - scanner_pose.view(B, 1, 3)).unsqueeze(2)
    seg = c + ts.view(1, 1, num_samples, 1) * direction
    seg = seg.view(B, N * num_samples, 3)

    field = decoder(seg, z)  # (B, N*K_seg)
    pre_field = field.view(B, N, num_samples)[:, :, :-1]  # all but last sample
    if field_kind == "occupancy":
        l_pre = _bce_with_huber(pre_field, 0.0, delta)
    elif field_kind == "sdf":
        scaled = ((sdf_margin - pre_field) / max(sdf_margin, 1e-6)).clamp(min=-2.0, max=2.0)
        l_pre = huber(F.relu(scaled), delta).mean()
    else:
        raise ValueError(f"Unknown field_kind: {field_kind!r}")

    if alpha_hit > 0:
        hit_field = decoder(observed_points, z)
        if field_kind == "occupancy":
            l_hit = _bce_with_huber(hit_field, 1.0, delta)
        else:
            scaled = (hit_field / max(sdf_margin, 1e-6)).clamp(min=-2.0, max=2.0)
            l_hit = huber(scaled, delta).mean()
        return l_pre + alpha_hit * l_hit
    return l_pre


def normal_incidence_consistency(
    decoder,
    z: torch.Tensor,
    observed_points: torch.Tensor,
    observed_normals: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cosine alignment of decoder field gradient with observed normals.

    Returns ``1 - cos²``; squared cosine ignores sign-flip ambiguity.
    """
    if observed_points.numel() == 0:
        return z.new_zeros(())
    pts = observed_points.detach().clone().requires_grad_(True)
    field = decoder(pts, z).sum()
    grad = torch.autograd.grad(field, pts, create_graph=True, retain_graph=True)[0]
    dot = (grad * observed_normals).sum(dim=-1)
    g_norm = grad.norm(dim=-1).clamp_min(eps)
    n_norm = observed_normals.norm(dim=-1).clamp_min(eps)
    cos2 = (dot / (g_norm * n_norm)) ** 2
    return (1.0 - cos2).mean()


# ---------------------------------------------------------------------------
# Combined loss + diagnostics
# ---------------------------------------------------------------------------


def latent_prior_l2(z: torch.Tensor) -> torch.Tensor:
    """``||z||² / d_z`` averaged over the batch."""
    if z.numel() == 0:
        return z.new_zeros(())
    return z.pow(2).sum(dim=-1).mean() / max(z.shape[-1], 1)


def compute_observation_loss(
    *,
    decoder,
    z: torch.Tensor,
    observed_points: torch.Tensor,
    free_points: torch.Tensor | None = None,
    hard_negatives: torch.Tensor | None = None,
    mixed_points: torch.Tensor | None = None,
    mixed_labels: torch.Tensor | None = None,
    mixed_ignore: torch.Tensor | None = None,
    scanner_pose: torch.Tensor | None = None,
    observed_normals: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
    deltas: dict[str, float] | None = None,
    field_kind: str = "occupancy",
    sdf_margin: float = 0.05,
    enable_ray_loss: bool = False,
    enable_incidence: bool = False,
    ray_num_samples: int = 8,
    alpha_hard: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined Tier-1 + Tier-2 observation loss.

    Returns
    -------
    (total_loss, scalar_metrics)
    """
    weights = weights or {}
    deltas = deltas or {}
    w_surf = float(weights.get("w_surf", 1.0))
    w_free = float(weights.get("w_free", 1.0))
    w_mixed = float(weights.get("w_mixed", 0.5))
    w_ray = float(weights.get("w_ray", 0.5))
    w_incidence = float(weights.get("w_incidence", 0.0))
    lambda_prior = float(weights.get("lambda_prior", 1e-3))
    delta_surf = float(deltas.get("delta_surf", 0.5))
    delta_free = float(deltas.get("delta_free", 0.5))
    delta_mixed = float(deltas.get("delta_mixed", 0.5))

    metrics: dict[str, float] = {}
    total = z.new_zeros(())

    l_surf = observed_surface_field_loss(
        decoder, z, observed_points,
        delta=delta_surf, field_kind=field_kind, sdf_margin=sdf_margin,
    )
    metrics["loss_surf_obs"] = float(l_surf.detach().item())
    total = total + w_surf * l_surf

    if free_points is not None:
        l_free = free_space_loss(
            decoder, z, free_points, hard_negatives,
            delta=delta_free, alpha_hard=alpha_hard,
            field_kind=field_kind, sdf_margin=sdf_margin,
        )
        metrics["loss_free"] = float(l_free.detach().item())
        total = total + w_free * l_free

    if mixed_points is not None and mixed_labels is not None:
        l_mixed = mixed_query_loss(
            decoder, z, mixed_points, mixed_labels, mixed_ignore,
            delta=delta_mixed, field_kind=field_kind, sdf_margin=sdf_margin,
        )
        metrics["loss_mixed"] = float(l_mixed.detach().item())
        total = total + w_mixed * l_mixed

    if enable_ray_loss and scanner_pose is not None:
        l_ray = ray_consistency_loss(
            decoder, z, observed_points, scanner_pose,
            num_samples=ray_num_samples, delta=delta_free,
            field_kind=field_kind, sdf_margin=sdf_margin,
        )
        metrics["loss_ray"] = float(l_ray.detach().item())
        total = total + w_ray * l_ray
    elif enable_ray_loss:
        metrics["loss_ray_skipped"] = 1.0

    if enable_incidence and observed_normals is not None:
        l_inc = normal_incidence_consistency(decoder, z, observed_points, observed_normals)
        metrics["loss_incidence"] = float(l_inc.detach().item())
        total = total + w_incidence * l_inc

    l_prior = latent_prior_l2(z)
    metrics["loss_prior"] = float(l_prior.detach().item())
    total = total + lambda_prior * l_prior

    metrics["loss_total"] = float(total.detach().item())
    return total, metrics


def free_space_violation_rate(
    decoder,
    z: torch.Tensor,
    free_points: torch.Tensor,
    *,
    threshold: float = 0.5,
    field_kind: str = "occupancy",
    sdf_margin: float = 0.05,
) -> float:
    """Fraction of ``free_points`` predicted occupied. Used as an early-stop signal."""
    if free_points.numel() == 0:
        return float("nan")
    with torch.no_grad():
        field = decoder(free_points, z)
        if field_kind == "occupancy":
            occupied = (torch.sigmoid(field) > threshold).float()
        elif field_kind == "sdf":
            occupied = (field < 0.0).float()
        else:
            raise ValueError(f"Unknown field_kind: {field_kind!r}")
    return float(occupied.mean().item())


__all__ = [
    "huber",
    "observed_surface_field_loss",
    "free_space_loss",
    "mixed_query_loss",
    "ray_consistency_loss",
    "normal_incidence_consistency",
    "latent_prior_l2",
    "compute_observation_loss",
    "free_space_violation_rate",
]

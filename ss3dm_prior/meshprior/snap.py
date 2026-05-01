"""Conservative snap proposal utilities for MeshPrior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class SnapProposal:
    vertices_before: np.ndarray
    vertices_after: np.ndarray
    displacement: np.ndarray
    eligible_mask: np.ndarray
    max_disp: float
    metadata: dict


def compute_field_gradient(
    decoder: Callable[..., torch.Tensor],
    z: torch.Tensor | None,
    points: torch.Tensor,
    *,
    iso_level: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gradient of squared occupancy-iso loss wrt points."""
    pts = points.detach().clone().requires_grad_(True)
    if z is None:
        logits = decoder(pts)
    else:
        zz = z
        if zz.ndim == 1:
            zz = zz.unsqueeze(0)
        logits = decoder(pts.unsqueeze(0), zz).reshape(-1)
    occ = torch.sigmoid(logits)
    loss = (occ - float(iso_level)).pow(2).sum()
    grad = torch.autograd.grad(loss, pts, create_graph=False, retain_graph=False)[0]
    return grad, occ.detach()


def _boundary_vertex_mask(faces: np.ndarray, num_vertices: int) -> np.ndarray:
    from collections import defaultdict

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for face in faces:
        for u, v in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            counts[tuple(sorted((int(u), int(v))))] += 1
    mask = np.zeros(num_vertices, dtype=bool)
    for (u, v), count in counts.items():
        if count == 1:
            mask[u] = True
            mask[v] = True
    return mask


def propose_vertex_snap(
    vertices: np.ndarray,
    faces: np.ndarray,
    decoder: Callable[..., torch.Tensor],
    z: torch.Tensor | None = None,
    *,
    confidence: float = 1.0,
    max_disp: float = 0.02,
    iso_level: float = 0.5,
    protect_mask: np.ndarray | None = None,
    observed_support: np.ndarray | None = None,
    uncertainty: float = 0.0,
    allow_boundary: bool = False,
    device: str | torch.device = "cpu",
) -> SnapProposal:
    verts = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    eligible = np.ones(len(verts), dtype=bool)
    if not allow_boundary and len(faces) > 0:
        eligible &= ~_boundary_vertex_mask(faces, len(verts))
    if protect_mask is not None:
        eligible &= ~np.asarray(protect_mask, dtype=bool)
    if observed_support is not None:
        eligible &= np.asarray(observed_support, dtype=np.float32) < 0.95
    if float(uncertainty) >= 0.75 or float(confidence) <= 0.0:
        eligible[:] = False

    pts = torch.from_numpy(verts).to(torch.device(device))
    grad, _ = compute_field_gradient(decoder, z.to(torch.device(device)) if z is not None else None, pts, iso_level=iso_level)
    step = -grad.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(step, axis=1, keepdims=True)
    step = np.where(norms > 1e-12, step / np.maximum(norms, 1e-12), 0.0)
    step *= float(max_disp) * float(np.clip(confidence, 0.0, 1.0)) * (1.0 - float(np.clip(uncertainty, 0.0, 1.0)))
    step[~eligible] = 0.0
    disp_norm = np.linalg.norm(step, axis=1, keepdims=True)
    too_large = disp_norm[:, 0] > float(max_disp)
    if too_large.any():
        step[too_large] *= float(max_disp) / np.maximum(disp_norm[too_large], 1e-12)
    after = verts + step
    return SnapProposal(
        vertices_before=verts,
        vertices_after=after,
        displacement=step,
        eligible_mask=eligible,
        max_disp=float(max_disp),
        metadata={
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
            "iso_level": float(iso_level),
            "allow_boundary": bool(allow_boundary),
        },
    )


def apply_snap_proposal(mesh: tuple[np.ndarray, np.ndarray], proposal: SnapProposal) -> tuple[np.ndarray, np.ndarray]:
    """Return a snapped mesh copy; does not mutate input arrays."""
    _, faces = mesh
    return proposal.vertices_after.copy(), np.asarray(faces, dtype=np.int64).copy()


def evaluate_snap_risk(
    proposal: SnapProposal,
    *,
    distance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    free_space_violation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    disp_norm = np.linalg.norm(proposal.displacement, axis=1)
    out = {
        "mean_displacement": float(disp_norm.mean()) if len(disp_norm) else 0.0,
        "max_displacement": float(disp_norm.max()) if len(disp_norm) else 0.0,
        "moved_vertex_fraction": float((disp_norm > 1e-9).mean()) if len(disp_norm) else 0.0,
    }
    if distance_fn is not None:
        before = np.asarray(distance_fn(proposal.vertices_before), dtype=np.float64)
        after = np.asarray(distance_fn(proposal.vertices_after), dtype=np.float64)
        out["surface_distance_before_mean"] = float(before.mean())
        out["surface_distance_after_mean"] = float(after.mean())
        out["surface_distance_delta_mean"] = float(before.mean() - after.mean())
    if free_space_violation_fn is not None:
        before_free = np.asarray(free_space_violation_fn(proposal.vertices_before), dtype=np.float64)
        after_free = np.asarray(free_space_violation_fn(proposal.vertices_after), dtype=np.float64)
        out["free_space_violation_before_mean"] = float(before_free.mean())
        out["free_space_violation_after_mean"] = float(after_free.mean())
        out["free_space_violation_delta"] = float(after_free.mean() - before_free.mean())
    else:
        out["free_space_violation_delta"] = 0.0
    return out


def accept_snap_proposal(
    risk: dict[str, float],
    *,
    max_free_space_delta: float = 0.0,
    max_visible_preservation_drop: float = 0.05,
    visible_preservation_drop: float = 0.0,
    require_surface_improvement: bool = True,
) -> bool:
    """Return whether a snap proposal may be applied by a downstream gate."""
    if risk.get("free_space_violation_delta", 0.0) > float(max_free_space_delta):
        return False
    if float(visible_preservation_drop) > float(max_visible_preservation_drop):
        return False
    if require_surface_improvement and risk.get("surface_distance_delta_mean", 0.0) <= 0.0:
        return False
    return True

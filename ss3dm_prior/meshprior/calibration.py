"""Post-hoc MeshPrior proposal calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ss3dm_prior.meshprior.protect_prune import compute_triangle_scores
from ss3dm_prior.meshprior.snap import evaluate_snap_risk, propose_vertex_snap


@dataclass(frozen=True)
class SurfaceCalibrationProfile:
    name: str
    snap_max_disp: float
    max_visible_protect_drop: float
    max_free_space_delta: float
    require_surface_improvement: bool = True


UNCALIBRATED_PROFILE = SurfaceCalibrationProfile(
    name="none",
    snap_max_disp=0.02,
    max_visible_protect_drop=0.05,
    max_free_space_delta=0.0,
)

SURFACE_SUPPORT_V1 = SurfaceCalibrationProfile(
    name="surface_support_v1",
    snap_max_disp=0.005,
    max_visible_protect_drop=0.05,
    max_free_space_delta=0.0,
)


def get_calibration_profile(name: str) -> SurfaceCalibrationProfile:
    if name in {"", "none", "uncalibrated"}:
        return UNCALIBRATED_PROFILE
    if name == "surface_support_v1":
        return SURFACE_SUPPORT_V1
    raise ValueError(f"unknown calibration profile: {name}")


def calibrated_snap_max_disp(profile_name: str) -> float:
    return float(get_calibration_profile(profile_name).snap_max_disp)


def valid_surface_protect_recall(
    vertices: np.ndarray,
    faces: np.ndarray,
    decoder: Callable[..., torch.Tensor],
    valid_face_mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    table = compute_triangle_scores(vertices=vertices, faces=faces, decoder=decoder, z=None, samples_per_face=4)
    protect = np.asarray(table.protect_scores, dtype=np.float32) >= float(threshold)
    valid = np.asarray(valid_face_mask, dtype=bool)
    return float(np.logical_and(protect, valid).sum() / max(int(valid.sum()), 1))


def evaluate_snap_calibration_profile(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    valid_face_mask: np.ndarray,
    support_decoder: Callable[..., torch.Tensor],
    occupancy_decoder: Callable[..., torch.Tensor],
    surface_distance_fn: Callable[[np.ndarray], np.ndarray],
    profile_name: str,
) -> dict[str, float | str | bool]:
    profile = get_calibration_profile(profile_name)
    baseline_recall = valid_surface_protect_recall(vertices, faces, support_decoder, valid_face_mask)
    proposal = propose_vertex_snap(
        vertices,
        faces,
        occupancy_decoder,
        z=None,
        max_disp=profile.snap_max_disp,
        allow_boundary=False,
    )
    snapped_recall = valid_surface_protect_recall(proposal.vertices_after, faces, support_decoder, valid_face_mask)
    risk = evaluate_snap_risk(proposal, distance_fn=surface_distance_fn)
    protect_drop = baseline_recall - snapped_recall
    accepted = (
        protect_drop <= profile.max_visible_protect_drop
        and float(risk.get("free_space_violation_delta", 0.0)) <= profile.max_free_space_delta
        and (not profile.require_surface_improvement or float(risk.get("surface_distance_delta_mean", 0.0)) > 0.0)
    )
    return {
        "profile": profile.name,
        "snap_max_disp": float(profile.snap_max_disp),
        "baseline_valid_surface_protect_recall": float(baseline_recall),
        "snapped_valid_surface_protect_recall": float(snapped_recall),
        "valid_surface_protect_recall_delta": float(snapped_recall - baseline_recall),
        "visible_protect_drop": float(protect_drop),
        "surface_distance_delta_mean": float(risk.get("surface_distance_delta_mean", 0.0)),
        "max_displacement": float(risk["max_displacement"]),
        "free_space_violation_delta": float(risk.get("free_space_violation_delta", 0.0)),
        "accepted_by_profile": bool(accepted),
    }

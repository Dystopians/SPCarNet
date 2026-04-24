"""Patch-level Marching-Cubes mesh extraction (CarNet_v0 D4 / auxiliary output).

Given a trained model that exposes an occupancy query head, this module
queries that head on a uniform grid inside the patch's canonical unit ball
and extracts a triangle mesh via ``trimesh.voxel.ops.matrix_to_marching_cubes``.

The module is eval-only: it does not enter the training loop. Training
remains point-cloud + occupancy-BCE + free-space as before. At evaluation
time (``ss3dm_prior.eval``) a single call per patch produces a ``.glb``
alongside the point-cloud output, and per-patch mesh metrics are computed
against the ground-truth mesh when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import trimesh
from trimesh.voxel import ops as voxel_ops


@dataclass
class MeshExtractionResult:
    """Output of :func:`extract_patch_mesh`.

    Attributes
    ----------
    mesh : trimesh.Trimesh | None
        The extracted mesh in the patch's canonical frame (unit ball). None
        when Marching Cubes could not produce any vertices (e.g. the
        occupancy field never crossed the iso-level).
    vertex_count : int
    face_count : int
    watertight : bool
    iso_level : float
    resolution : int
    """

    mesh: trimesh.Trimesh | None
    vertex_count: int
    face_count: int
    watertight: bool
    iso_level: float
    resolution: int


@torch.no_grad()
def extract_patch_mesh(
    *,
    occupancy_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    patch_radius: float = 1.0,
    resolution: int = 64,
    iso_level: float = 0.5,
    chunk_size: int = 65536,
    bounds_padding: float = 1.05,
) -> MeshExtractionResult:
    """Extract a mesh from the model's occupancy head on a regular grid.

    Parameters
    ----------
    occupancy_fn : callable
        A zero-argument closure returning occupancy *probabilities* (not
        logits) in ``[0, 1]`` for a given tensor of query points
        ``(N, 3)`` on the given ``device``. The caller is responsible for
        applying ``torch.sigmoid`` on the model's raw logits before passing
        to this function — this keeps the iso-level interpretation
        unambiguous.
    device : torch.device
    patch_radius : float
        Radius of the patch's canonical bounding ball (default 1.0 after
        normalisation in the car cache builder).
    resolution : int
        Grid resolution per axis. Memory grows as ``O(resolution^3)``; 64
        is a good default for whole-car patches.
    iso_level : float
        Occupancy probability threshold; 0.5 treats the occupancy head as
        a binary indicator.
    chunk_size : int
        Number of query points evaluated per forward call.
    bounds_padding : float
        Scale factor on ``patch_radius`` to slightly extend the grid so
        the extracted mesh isn't clipped at the bounding box.

    Returns
    -------
    MeshExtractionResult
    """
    axis = torch.linspace(
        -patch_radius * bounds_padding,
        patch_radius * bounds_padding,
        resolution,
        dtype=torch.float32,
    )
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1).to(device)

    occupancy_values = torch.empty(grid.shape[0], dtype=torch.float32)
    for start in range(0, grid.shape[0], chunk_size):
        chunk = grid[start : start + chunk_size]
        value = occupancy_fn(chunk)
        occupancy_values[start : start + chunk.shape[0]] = value.detach().cpu().to(torch.float32)

    volume = occupancy_values.reshape(resolution, resolution, resolution).numpy()

    # Normalise sign of the iso-level: matrix_to_marching_cubes expects a
    # binary "pitch matrix" (truthy = inside). Use our iso_level threshold.
    binary_volume = (volume >= float(iso_level)).astype(np.uint8)
    if binary_volume.sum() < 8:
        return MeshExtractionResult(
            mesh=None,
            vertex_count=0,
            face_count=0,
            watertight=False,
            iso_level=float(iso_level),
            resolution=int(resolution),
        )

    pitch = float(2.0 * patch_radius * bounds_padding / max(resolution - 1, 1))
    try:
        mesh = voxel_ops.matrix_to_marching_cubes(matrix=binary_volume, pitch=pitch)
    except ModuleNotFoundError as exc:
        # trimesh.voxel.ops.matrix_to_marching_cubes imports skimage on demand;
        # if it isn't installed we degrade gracefully — the point cloud and
        # volume-IoU pathways still work, only the triangulated mesh is gone.
        import warnings as _warnings
        _warnings.warn(
            "marching_cubes backend unavailable (%s); mesh extraction skipped."
            " Install `scikit-image` to enable." % exc,
            stacklevel=2,
        )
        return MeshExtractionResult(
            mesh=None,
            vertex_count=0,
            face_count=0,
            watertight=False,
            iso_level=float(iso_level),
            resolution=int(resolution),
        )
    # matrix_to_marching_cubes returns mesh in voxel coords starting at
    # origin; shift so the patch is centred at origin.
    if mesh.vertices is not None and len(mesh.vertices) > 0:
        mesh.apply_translation(np.asarray([-patch_radius * bounds_padding] * 3, dtype=np.float64))

    vertex_count = int(len(mesh.vertices)) if mesh.vertices is not None else 0
    face_count = int(len(mesh.faces)) if mesh.faces is not None else 0
    watertight = bool(getattr(mesh, "is_watertight", False))
    return MeshExtractionResult(
        mesh=mesh if vertex_count > 0 else None,
        vertex_count=vertex_count,
        face_count=face_count,
        watertight=watertight,
        iso_level=float(iso_level),
        resolution=int(resolution),
    )


def save_patch_mesh(mesh: trimesh.Trimesh, output_path: str | Path) -> Path:
    """Save a mesh as ``.glb`` (preferred) or ``.obj`` depending on suffix."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return output_path


def mesh_iou_at_iso(
    predicted_volume: np.ndarray,
    target_volume: np.ndarray,
    iso_level: float = 0.5,
) -> float:
    """Volume-level IoU at a given iso-level.

    ``predicted_volume`` and ``target_volume`` are 3-D arrays of occupancy
    probabilities (or binary occupancy) in ``[0, 1]``. The two grids must
    be the same shape; the caller is responsible for registering them in
    the same canonical frame.
    """
    if predicted_volume.shape != target_volume.shape:
        raise ValueError(
            "predicted_volume and target_volume must have the same shape; "
            f"got {predicted_volume.shape} vs {target_volume.shape}"
        )
    pred_mask = (np.asarray(predicted_volume) >= iso_level).astype(bool)
    targ_mask = (np.asarray(target_volume) >= iso_level).astype(bool)
    intersection = float(np.logical_and(pred_mask, targ_mask).sum())
    union = float(np.logical_or(pred_mask, targ_mask).sum())
    if union <= 0.0:
        return float("nan")
    return intersection / union


def surface_normal_consistency(
    predicted_mesh: trimesh.Trimesh | None,
    target_mesh: trimesh.Trimesh | None,
    *,
    sample_count: int = 4096,
    seed: int = 0,
) -> float:
    """Mean absolute cosine similarity between nearest-face normals of two meshes.

    Returns NaN when either mesh is empty/degenerate.
    """
    if predicted_mesh is None or target_mesh is None:
        return float("nan")
    if len(predicted_mesh.faces) == 0 or len(target_mesh.faces) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    pred_points, pred_face_idx = trimesh.sample.sample_surface(predicted_mesh, sample_count, seed=rng)
    pred_normals = predicted_mesh.face_normals[pred_face_idx]
    # Nearest target face for each predicted sample.
    try:
        _, _, target_face_idx = target_mesh.nearest.on_surface(pred_points)
    except Exception:  # noqa: BLE001 — trimesh raises on pathological meshes
        return float("nan")
    target_normals = target_mesh.face_normals[target_face_idx]
    cos = np.sum(pred_normals * target_normals, axis=1)
    return float(np.mean(np.abs(cos)))

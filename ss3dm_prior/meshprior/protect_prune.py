"""Protect/prune proposal scoring for MeshPrior."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from ss3dm_prior.meshprior.proposals import MeshPriorProposal, ProposalBatch, TriangleScoreTable


def sample_triangle_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    samples_per_face: int = 4,
) -> np.ndarray:
    """Deterministically sample barycentric points on each triangle.

    Returns an array of shape ``(F, samples_per_face, 3)``.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    if samples_per_face <= 1:
        bary = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32)
    else:
        base = [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ]
        while len(base) < samples_per_face:
            t = len(base) + 1
            a = (math.sin(t * 12.9898) * 43758.5453) % 1.0
            b = (math.sin(t * 78.233) * 12345.6789) % 1.0
            if a + b > 1.0:
                a = 1.0 - a
                b = 1.0 - b
            base.append([1.0 - a - b, a, b])
        bary = np.asarray(base[:samples_per_face], dtype=np.float32)
    return np.einsum("sk,nkd->nsd", bary, tri).astype(np.float32)


@torch.no_grad()
def compute_shape_field_support(
    decoder: Callable[..., torch.Tensor],
    z: torch.Tensor | None,
    samples: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 65536,
) -> np.ndarray:
    """Compute occupancy support probabilities for sampled triangle points."""
    dev = torch.device(device)
    pts = torch.as_tensor(samples, dtype=torch.float32, device=dev)
    flat = pts.reshape(-1, 3)
    outs = []
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        if z is None:
            logits = decoder(chunk)
        else:
            zz = z.to(dev)
            if zz.ndim == 1:
                zz = zz.unsqueeze(0)
            logits = decoder(chunk.unsqueeze(0), zz).reshape(-1)
        outs.append(torch.sigmoid(logits).detach().cpu())
    support = torch.cat(outs, dim=0).numpy().reshape(pts.shape[:-1])
    return support.astype(np.float32)


def compute_triangle_scores(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    decoder: Callable[..., torch.Tensor],
    z: torch.Tensor | None,
    face_indices: list[int] | np.ndarray | None = None,
    samples_per_face: int = 4,
    observed_support: float = 1.0,
    uncertainty_penalty: float = 0.0,
    free_space_violation: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> TriangleScoreTable:
    all_faces = np.asarray(faces, dtype=np.int64)
    if face_indices is None:
        face_indices_arr = np.arange(len(all_faces), dtype=np.int64)
    else:
        face_indices_arr = np.asarray(face_indices, dtype=np.int64)
    selected_faces = all_faces[face_indices_arr]
    samples = sample_triangle_points(vertices, selected_faces, samples_per_face=samples_per_face)
    support_samples = compute_shape_field_support(decoder, z, samples, device=device)
    surface_support = support_samples.mean(axis=1)
    prior_violation = 1.0 - surface_support
    uncertainty = float(np.clip(uncertainty_penalty, 0.0, 1.0))
    observed = float(np.clip(observed_support, 0.0, 1.0))
    free = np.zeros_like(surface_support) if free_space_violation is None else np.asarray(free_space_violation)
    protect = surface_support * observed * (1.0 - uncertainty)
    low_observed = 1.0 - observed
    prune = np.clip(prior_violation + free + low_observed - protect, 0.0, 1.0)
    return TriangleScoreTable(
        region_id="",
        face_indices=[int(x) for x in face_indices_arr.tolist()],
        protect_scores=[float(x) for x in protect.tolist()],
        prune_scores=[float(x) for x in prune.tolist()],
        surface_support=[float(x) for x in surface_support.tolist()],
        prior_violation=[float(x) for x in prior_violation.tolist()],
        uncertainty_penalty=uncertainty,
    )


def build_protect_prune_proposals(
    score_table: TriangleScoreTable,
    *,
    region_id: str,
    protect_threshold: float = 0.5,
    prune_threshold: float = 0.5,
) -> ProposalBatch:
    protect_faces = [
        f for f, s in zip(score_table.face_indices, score_table.protect_scores, strict=True) if s >= protect_threshold
    ]
    prune_faces = [
        f for f, s in zip(score_table.face_indices, score_table.prune_scores, strict=True) if s >= prune_threshold
    ]
    proposals: list[MeshPriorProposal] = []
    if protect_faces:
        vals = [s for s in score_table.protect_scores if s >= protect_threshold]
        proposals.append(
            MeshPriorProposal(
                proposal_id=f"{region_id}_protect_0000",
                proposal_type="protect",
                region_id=region_id,
                face_indices=[int(x) for x in protect_faces],
                confidence=float(np.mean(vals)),
                score_mean=float(np.mean(vals)),
                score_max=float(np.max(vals)),
                metadata={"threshold": float(protect_threshold)},
            )
        )
    if prune_faces:
        vals = [s for s in score_table.prune_scores if s >= prune_threshold]
        proposals.append(
            MeshPriorProposal(
                proposal_id=f"{region_id}_prune_0000",
                proposal_type="prune",
                region_id=region_id,
                face_indices=[int(x) for x in prune_faces],
                confidence=float(np.mean(vals)),
                score_mean=float(np.mean(vals)),
                score_max=float(np.max(vals)),
                metadata={"threshold": float(prune_threshold)},
            )
        )
    score_table.region_id = region_id
    return ProposalBatch(proposals=proposals, score_tables=[score_table])

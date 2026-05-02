"""Retrieval-deformation fallback for MeshPrior proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ss3dm_prior.meshprior.proposals import MeshPriorProposal, ProposalBatch, TriangleScoreTable
from ss3dm_prior.meshprior.protect_prune import sample_triangle_points


@dataclass
class AnchorBank:
    object_ids: list[str]
    splits: list[str]
    points: np.ndarray
    metadata: dict[str, Any]

    def to_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            object_ids=np.asarray(self.object_ids, dtype=object),
            splits=np.asarray(self.splits, dtype=object),
            points=np.asarray(self.points, dtype=np.float32),
            metadata=np.asarray([self.metadata], dtype=object),
        )

    @staticmethod
    def from_npz(path: str | Path) -> "AnchorBank":
        with np.load(path, allow_pickle=True) as data:
            object_ids = [str(x) for x in data["object_ids"].tolist()]
            splits = [str(x) for x in data["splits"].tolist()]
            points = np.asarray(data["points"], dtype=np.float32)
            metadata = dict(data["metadata"].tolist()[0])
        if any(split != "train" for split in splits):
            raise ValueError("Anchor bank contains non-train anchors; refusing leakage-prone retrieval.")
        return AnchorBank(object_ids=object_ids, splits=splits, points=points, metadata=metadata)


@dataclass
class RetrievalResult:
    anchor_index: int
    object_id: str
    score: float
    second_score: float
    margin: float
    uncertainty: float
    nearest_distance_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    scale = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    scale = max(scale, 1e-6)
    return ((pts - center) / scale).astype(np.float32)


def deterministic_sample(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError("cannot sample an empty point cloud")
    if len(pts) >= count:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=count, replace=False)
    else:
        idx = np.arange(count) % len(pts)
    return pts[idx].astype(np.float32)


def chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    d = np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def build_anchor_bank(
    records: list[dict[str, Any]],
    *,
    points_per_anchor: int = 512,
    max_anchors: int = 128,
    seed: int = 0,
) -> AnchorBank:
    object_ids: list[str] = []
    splits: list[str] = []
    anchors: list[np.ndarray] = []
    for rec in records:
        split = str(rec.get("split", ""))
        if split != "train":
            continue
        pts = np.asarray(rec["points"], dtype=np.float32)
        oid = str(rec.get("object_id", f"anchor_{len(anchors):04d}"))
        sampled = deterministic_sample(normalize_points(pts), points_per_anchor, seed=seed + len(anchors))
        object_ids.append(oid)
        splits.append(split)
        anchors.append(sampled)
        if len(anchors) >= max_anchors:
            break
    if not anchors:
        raise ValueError("no train anchors available")
    return AnchorBank(
        object_ids=object_ids,
        splits=splits,
        points=np.stack(anchors, axis=0).astype(np.float32),
        metadata={
            "train_only": True,
            "points_per_anchor": int(points_per_anchor),
            "max_anchors": int(max_anchors),
            "seed": int(seed),
        },
    )


def retrieve_anchor(
    observed_points: np.ndarray,
    bank: AnchorBank,
    *,
    query_object_id: str = "",
) -> RetrievalResult:
    observed = normalize_points(observed_points)
    scores: list[tuple[float, int]] = []
    for idx, anchor in enumerate(bank.points):
        if query_object_id and bank.object_ids[idx] == query_object_id:
            continue
        scores.append((chamfer_l1(observed, anchor), idx))
    if not scores:
        raise ValueError("no eligible anchors after leakage filtering")
    scores.sort(key=lambda x: x[0])
    best_score, best_idx = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else best_score
    margin = max(0.0, float(second_score - best_score))
    uncertainty = float(1.0 / (1.0 + margin / max(best_score, 1e-6)))
    d = np.linalg.norm(observed[:, None, :] - bank.points[best_idx][None, :, :], axis=-1)
    return RetrievalResult(
        anchor_index=int(best_idx),
        object_id=bank.object_ids[best_idx],
        score=float(best_score),
        second_score=float(second_score),
        margin=float(margin),
        uncertainty=uncertainty,
        nearest_distance_mean=float(d.min(axis=1).mean()),
    )


def anchor_surface_support(samples: np.ndarray, anchor_points: np.ndarray, *, radius: float = 0.08) -> np.ndarray:
    pts = normalize_points(samples.reshape(-1, 3))
    anchor = normalize_points(anchor_points)
    d = np.linalg.norm(pts[:, None, :] - anchor[None, :, :], axis=-1).min(axis=1)
    support = np.exp(-d / max(radius, 1e-6))
    return support.reshape(samples.shape[:-1]).astype(np.float32)


def compute_retrieval_triangle_scores(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    anchor_points: np.ndarray,
    retrieval: RetrievalResult,
    face_indices: list[int] | np.ndarray | None = None,
    samples_per_face: int = 4,
    support_radius: float = 0.08,
) -> TriangleScoreTable:
    all_faces = np.asarray(faces, dtype=np.int64)
    face_idx = np.arange(len(all_faces), dtype=np.int64) if face_indices is None else np.asarray(face_indices, dtype=np.int64)
    samples = sample_triangle_points(vertices, all_faces[face_idx], samples_per_face=samples_per_face)
    support_samples = anchor_surface_support(samples, anchor_points, radius=support_radius)
    support = support_samples.mean(axis=1)
    uncertainty = float(np.clip(retrieval.uncertainty, 0.0, 1.0))
    protect = np.clip(support * (1.0 - 0.5 * uncertainty), 0.0, 1.0)
    prune = np.clip((1.0 - support) + 0.25 * uncertainty - protect, 0.0, 1.0)
    return TriangleScoreTable(
        region_id="",
        face_indices=[int(x) for x in face_idx.tolist()],
        protect_scores=[float(x) for x in protect.tolist()],
        prune_scores=[float(x) for x in prune.tolist()],
        surface_support=[float(x) for x in support.tolist()],
        prior_violation=[float(x) for x in (1.0 - support).tolist()],
        uncertainty_penalty=uncertainty,
    )


def propose_retrieval_snap(
    vertices: np.ndarray,
    anchor_points: np.ndarray,
    *,
    max_disp: float = 0.005,
) -> tuple[np.ndarray, dict[str, float]]:
    verts = np.asarray(vertices, dtype=np.float32)
    anchor = np.asarray(anchor_points, dtype=np.float32)
    d = np.linalg.norm(verts[:, None, :] - anchor[None, :, :], axis=-1)
    nearest = anchor[d.argmin(axis=1)]
    delta = nearest - verts
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_disp / np.maximum(norms, 1e-8))
    moved = verts + delta * scale
    disp = np.linalg.norm(moved - verts, axis=1)
    return moved.astype(np.float32), {
        "snap_mean_displacement": float(disp.mean()),
        "snap_max_displacement": float(disp.max(initial=0.0)),
        "snap_moved_vertex_fraction": float((disp > 1e-8).mean()),
    }


def smooth_deform_anchor_to_observed(
    anchor_points: np.ndarray,
    observed_points: np.ndarray,
    *,
    blend: float = 0.25,
    max_disp: float = 0.02,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a conservative nearest-observation displacement to the anchor.

    This is deliberately small and smooth-ish: each anchor point moves toward its
    nearest observed point by a clipped fraction. Full neural deformation should
    be added only after this retrieval-only baseline is measured.
    """
    anchor = np.asarray(anchor_points, dtype=np.float32)
    observed = np.asarray(observed_points, dtype=np.float32)
    d = np.linalg.norm(anchor[:, None, :] - observed[None, :, :], axis=-1)
    nearest = observed[d.argmin(axis=1)]
    delta = (nearest - anchor) * float(np.clip(blend, 0.0, 1.0))
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_disp / np.maximum(norms, 1e-8))
    moved = anchor + delta * scale
    disp = np.linalg.norm(moved - anchor, axis=1)
    return moved.astype(np.float32), {
        "deform_mean_displacement": float(disp.mean()),
        "deform_max_displacement": float(disp.max(initial=0.0)),
        "deform_moved_fraction": float((disp > 1e-8).mean()),
    }


def build_retrieval_proposals(
    table: TriangleScoreTable,
    *,
    region_id: str,
    retrieval: RetrievalResult,
    protect_threshold: float = 0.5,
    prune_threshold: float = 0.5,
    include_fill_candidate: bool = True,
) -> ProposalBatch:
    proposals: list[MeshPriorProposal] = []
    protect_faces = [f for f, s in zip(table.face_indices, table.protect_scores, strict=True) if s >= protect_threshold]
    prune_faces = [f for f, s in zip(table.face_indices, table.prune_scores, strict=True) if s >= prune_threshold]
    meta = {"retrieval": retrieval.to_dict(), "source": "retrieval_deformation"}
    protect_vals = [s for s in table.protect_scores if s >= protect_threshold]
    prune_vals = [s for s in table.prune_scores if s >= prune_threshold]
    proposals.append(
        MeshPriorProposal(
            proposal_id=f"{region_id}_retrieval_protect_0000",
            proposal_type="protect",
            region_id=region_id,
            face_indices=protect_faces,
            confidence=float(np.mean(protect_vals)) if protect_vals else 0.0,
            score_mean=float(np.mean(protect_vals)) if protect_vals else 0.0,
            score_max=float(np.max(protect_vals)) if protect_vals else 0.0,
            metadata={**meta, "threshold": float(protect_threshold), "empty": not bool(protect_faces)},
        )
    )
    proposals.append(
        MeshPriorProposal(
            proposal_id=f"{region_id}_retrieval_prune_0000",
            proposal_type="prune",
            region_id=region_id,
            face_indices=prune_faces,
            confidence=float(np.mean(prune_vals)) if prune_vals else 0.0,
            score_mean=float(np.mean(prune_vals)) if prune_vals else 0.0,
            score_max=float(np.max(prune_vals)) if prune_vals else 0.0,
            metadata={**meta, "threshold": float(prune_threshold), "empty": not bool(prune_faces)},
        )
    )
    proposals.append(
        MeshPriorProposal(
            proposal_id=f"{region_id}_retrieval_snap_0000",
            proposal_type="snap",
            region_id=region_id,
            face_indices=[],
            confidence=float(max(0.0, 1.0 - retrieval.uncertainty)),
            score_mean=float(max(0.0, 1.0 - retrieval.uncertainty)),
            score_max=float(max(0.0, 1.0 - retrieval.uncertainty)),
            metadata={**meta, "requires_scene_gate": True, "bounded": True},
        )
    )
    if include_fill_candidate:
        proposals.append(
            MeshPriorProposal(
                proposal_id=f"{region_id}_retrieval_fill_candidate_0000",
                proposal_type="fill_candidate",
                region_id=region_id,
                face_indices=[],
                confidence=float(max(0.0, 1.0 - retrieval.uncertainty)),
                score_mean=float(max(0.0, 1.0 - retrieval.uncertainty)),
                score_max=float(max(0.0, 1.0 - retrieval.uncertainty)),
                metadata={**meta, "requires_scene_gate": True},
            )
        )
    proposals.append(
        MeshPriorProposal(
            proposal_id=f"{region_id}_retrieval_uncertainty_0000",
            proposal_type="uncertainty",
            region_id=region_id,
            face_indices=[],
            confidence=float(1.0 - retrieval.uncertainty),
            score_mean=float(retrieval.uncertainty),
            score_max=float(retrieval.uncertainty),
            metadata=meta,
        )
    )
    table.region_id = region_id
    return ProposalBatch(proposals=proposals, score_tables=[table], notes=["retrieval-only baseline is measured before deformation"])

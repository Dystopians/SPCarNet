"""Proposal contracts for MeshPrior outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MeshPriorProposal:
    proposal_id: str
    proposal_type: str
    region_id: str
    face_indices: list[int]
    confidence: float
    score_mean: float
    score_max: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriangleScoreTable:
    region_id: str
    face_indices: list[int]
    protect_scores: list[float]
    prune_scores: list[float]
    surface_support: list[float]
    prior_violation: list[float]
    uncertainty_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalBatch:
    proposals: list[MeshPriorProposal] = field(default_factory=list)
    score_tables: list[TriangleScoreTable] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [p.to_dict() for p in self.proposals],
            "score_tables": [t.to_dict() for t in self.score_tables],
            "notes": list(self.notes),
        }

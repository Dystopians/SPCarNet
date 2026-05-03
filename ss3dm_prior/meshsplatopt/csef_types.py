from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CSEFSample:
    sample_id: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    region_id: str
    positive_surface_evidence: float
    negative_free_space_evidence: float
    explanation_debt: float
    prior_support: float
    topology_cost: float
    uncertainty: float
    evidence_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CSEFRegion:
    region_id: str
    defect_type_candidates: list[str]
    bbox: dict[str, list[float]]
    boundary_loop_ids: list[str]
    mesh_face_indices: list[int]
    image_evidence_refs: list[str]
    sparse_point_refs: list[str]
    summary_stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CSEFBuildResult:
    scene_model: str
    scene_source: str
    mesh_path: str
    regions: list[CSEFRegion]
    global_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["regions"] = [region.to_dict() for region in self.regions]
        return data

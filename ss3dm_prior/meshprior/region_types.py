"""JSON-serialisable contracts for MeshPrior scene region mining."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RegionEvidence:
    """Evidence used to decide whether a scene region may be object-like."""

    segmentation_available: bool = False
    segmentation_score: float = 0.0
    geometry_score: float = 0.0
    ground_rejection_score: float = 0.0
    observed_support_score: float = 0.0
    car_likeness_score: float = 0.0
    eligible_for_posterior: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectCanonicalization:
    """Canonical transform metadata for later SP-CarNet posterior inference."""

    mode: str = "unknown"
    center: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    rotation: list[list[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneMeshRegion:
    """A mined mesh region and its geometry diagnostics."""

    region_id: str
    source_mesh_path: str | None
    component_id: int
    face_indices: list[int]
    triangle_count: int
    vertex_count: int
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_extent: list[float]
    centroid: list[float]
    surface_area: float
    vertex_density: float
    boundary_edge_count: int
    connected_components: int
    approximate_hole_boundary_score: float
    evidence: RegionEvidence = field(default_factory=RegionEvidence)
    canonicalization: ObjectCanonicalization = field(default_factory=ObjectCanonicalization)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["evidence"] = self.evidence.to_dict()
        out["canonicalization"] = self.canonicalization.to_dict()
        return out


@dataclass
class RegionMiningResult:
    """Top-level output of the MeshPrior region miner."""

    scene_model: str
    scene_source: str
    mode: str
    mesh_path: str | None
    regions: list[SceneMeshRegion] = field(default_factory=list)
    segmentation_artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_model": self.scene_model,
            "scene_source": self.scene_source,
            "mode": self.mode,
            "mesh_path": self.mesh_path,
            "segmentation_artifacts": list(self.segmentation_artifacts),
            "notes": list(self.notes),
            "regions": [r.to_dict() for r in self.regions],
        }

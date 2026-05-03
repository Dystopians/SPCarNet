from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class DefectType(StrEnum):
    FLOATER_COMPONENT = "FLOATER_COMPONENT"
    LOCAL_DENT = "LOCAL_DENT"
    ROUGH_BROKEN_SURFACE = "ROUGH_BROKEN_SURFACE"
    VEHICLE_DISCONTINUITY = "VEHICLE_DISCONTINUITY"
    GROUND_WALL_MISALIGNMENT = "GROUND_WALL_MISALIGNMENT"
    SMALL_BOUNDARY_HOLE = "SMALL_BOUNDARY_HOLE"
    GIANT_GROUND_VOID = "GIANT_GROUND_VOID"
    UNKNOWN_UNOBSERVED_VOID = "UNKNOWN_UNOBSERVED_VOID"
    APPEARANCE_GHOSTING_REGION = "APPEARANCE_GHOSTING_REGION"


@dataclass(frozen=True)
class DefectRecord:
    defect_id: str
    defect_type: str
    severity: float
    confidence: float
    affected_faces: list[int] = field(default_factory=list)
    affected_vertices: list[int] = field(default_factory=list)
    boundary_loops: list[str] = field(default_factory=list)
    candidate_edit_types_allowed: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    uncertainty_summary: dict[str, Any] = field(default_factory=dict)
    no_repair_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

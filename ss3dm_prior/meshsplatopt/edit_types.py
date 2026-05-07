from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any

import numpy as np


class MeshSplatOptEditType(StrEnum):
    PROTECT = "PROTECT"
    DELETE_TRIANGLES = "DELETE_TRIANGLES"
    EDGE_COLLAPSE = "EDGE_COLLAPSE"
    FACE_MERGE = "FACE_MERGE"
    SNAP_VERTICES = "SNAP_VERTICES"
    SPLIT_TRIANGLES = "SPLIT_TRIANGLES"
    FILL_PATCH = "FILL_PATCH"
    APPEARANCE_RESET = "APPEARANCE_RESET"


@dataclass
class MeshState:
    vertices: np.ndarray
    faces: np.ndarray
    attributes: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "MeshState":
        return MeshState(
            vertices=np.asarray(self.vertices).copy(),
            faces=np.asarray(self.faces).copy(),
            attributes=dict(self.attributes),
        )


@dataclass(frozen=True)
class MeshEdit:
    edit_id: str
    edit_type: str
    defect_id: str
    affected_vertices: list[int] = field(default_factory=list)
    affected_faces: list[int] = field(default_factory=list)
    inserted_vertices: list[list[float]] = field(default_factory=list)
    inserted_faces: list[list[int]] = field(default_factory=list)
    deleted_vertices: list[int] = field(default_factory=list)
    deleted_faces: list[int] = field(default_factory=list)
    attribute_changes: dict[str, Any] = field(default_factory=dict)
    topology_cost_delta: float = 0.0
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    risk_summary: dict[str, Any] = field(default_factory=dict)
    rollback_snapshot_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

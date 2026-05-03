"""MeshSplatOpt research utilities."""

from .csef_builder import build_csef, write_csef_outputs
from .csef_types import CSEFBuildResult, CSEFRegion, CSEFSample
from .defect_mining import mine_defects, write_defect_outputs
from .defect_types import DefectRecord, DefectType
from .edit_apply import apply_edit, summarize_topology_delta, verify_mesh_integrity
from .edit_snapshot import create_snapshot, rollback_edit
from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from .topology_baselines import TopologyBaselineRun, run_topology_baselines

__all__ = [
    "CSEFBuildResult",
    "CSEFRegion",
    "CSEFSample",
    "DefectRecord",
    "DefectType",
    "MeshEdit",
    "MeshSplatOptEditType",
    "MeshState",
    "TopologyBaselineRun",
    "apply_edit",
    "build_csef",
    "create_snapshot",
    "mine_defects",
    "run_topology_baselines",
    "rollback_edit",
    "summarize_topology_delta",
    "verify_mesh_integrity",
    "write_defect_outputs",
    "write_csef_outputs",
]

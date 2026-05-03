"""MeshSplatOpt research utilities."""

from .csef_builder import build_csef, write_csef_outputs
from .csef_types import CSEFBuildResult, CSEFRegion, CSEFSample
from .defect_mining import mine_defects, write_defect_outputs
from .defect_types import DefectRecord, DefectType
from .edit_apply import apply_edit, summarize_topology_delta, verify_mesh_integrity
from .edit_snapshot import create_snapshot, rollback_edit
from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from .ground_void_fill import make_ground_plane_void_fill
from .hole_fill import FillProposal, find_boundary_loops, make_boundary_loop_fill
from .snap_proposals import SnapProposal, make_snap_proposals
from .topology_baselines import TopologyBaselineRun, run_topology_baselines

__all__ = [
    "CSEFBuildResult",
    "CSEFRegion",
    "CSEFSample",
    "DefectRecord",
    "DefectType",
    "FillProposal",
    "MeshEdit",
    "MeshSplatOptEditType",
    "MeshState",
    "SnapProposal",
    "TopologyBaselineRun",
    "apply_edit",
    "build_csef",
    "create_snapshot",
    "find_boundary_loops",
    "make_boundary_loop_fill",
    "make_ground_plane_void_fill",
    "make_snap_proposals",
    "mine_defects",
    "run_topology_baselines",
    "rollback_edit",
    "summarize_topology_delta",
    "verify_mesh_integrity",
    "write_defect_outputs",
    "write_csef_outputs",
]

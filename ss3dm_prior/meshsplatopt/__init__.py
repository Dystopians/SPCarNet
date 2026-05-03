"""MeshSplatOpt research utilities."""

from .csef_builder import build_csef, write_csef_outputs
from .csef_types import CSEFBuildResult, CSEFRegion, CSEFSample
from .checkpoint_adapter import CheckpointEditReport, apply_edit_to_checkpoint_copy
from .counterfactual_edit_gate import CounterfactualGateReport, validate_edit_counterfactual
from .defect_mining import mine_defects, write_defect_outputs
from .defect_types import DefectRecord, DefectType
from .edit_apply import apply_edit, summarize_topology_delta, verify_mesh_integrity
from .edit_portfolio import PortfolioItem, rank_portfolio
from .edit_snapshot import create_snapshot, rollback_edit
from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from .ground_void_fill import make_ground_plane_void_fill
from .hole_fill import FillProposal, find_boundary_loops, make_boundary_loop_fill
from .object_prior_repair import ObjectRepairProposal, make_object_prior_repair_proposals
from .repair_state_machine import run_repair_state_machine
from .snap_proposals import SnapProposal, make_snap_proposals
from .synthetic_damage import run_synthetic_repair_benchmark
from .teacher_recovery import TeacherRecoveryPlan, run_teacher_recovery_contract
from .topology_baselines import TopologyBaselineRun, run_topology_baselines

__all__ = [
    "CSEFBuildResult",
    "CSEFRegion",
    "CSEFSample",
    "CheckpointEditReport",
    "CounterfactualGateReport",
    "DefectRecord",
    "DefectType",
    "FillProposal",
    "MeshEdit",
    "MeshSplatOptEditType",
    "MeshState",
    "ObjectRepairProposal",
    "PortfolioItem",
    "SnapProposal",
    "TeacherRecoveryPlan",
    "TopologyBaselineRun",
    "apply_edit",
    "apply_edit_to_checkpoint_copy",
    "build_csef",
    "create_snapshot",
    "find_boundary_loops",
    "make_boundary_loop_fill",
    "make_ground_plane_void_fill",
    "make_object_prior_repair_proposals",
    "make_snap_proposals",
    "mine_defects",
    "rank_portfolio",
    "run_repair_state_machine",
    "run_synthetic_repair_benchmark",
    "run_topology_baselines",
    "run_teacher_recovery_contract",
    "rollback_edit",
    "summarize_topology_delta",
    "validate_edit_counterfactual",
    "verify_mesh_integrity",
    "write_defect_outputs",
    "write_csef_outputs",
]

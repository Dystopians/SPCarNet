from __future__ import annotations

import numpy as np

from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState
from .hole_fill import FillProposal


def make_ground_plane_void_fill(
    state: MeshState,
    *,
    bbox_min: tuple[float, float],
    bbox_max: tuple[float, float],
    z: float = 0.0,
    grid_resolution: int = 3,
    proposal_id: str = "ground_void_fill_0000",
    allow_prior_only: bool = False,
    observed_support: bool = True,
) -> FillProposal:
    if not observed_support and not allow_prior_only:
        return FillProposal(
            proposal_id=proposal_id,
            fill_mode="ground_plane_void_fill",
            edit=None,
            certificate={
                "boundary_loop_support": False,
                "neighboring_surface_support": False,
                "sparse_depth_support": False,
                "free_space_risk": 0.5,
                "semantic_ground_object_support": "ground_prior",
                "camera_coverage_score": 0.0,
                "prior_only_flag": False,
                "expected_topology_cost": 0,
                "expected_area_repaired": 0.0,
            },
            rejected_reason="unknown_unobserved_void_normal_mode_reject",
        )
    xs = np.linspace(bbox_min[0], bbox_max[0], grid_resolution + 1)
    ys = np.linspace(bbox_min[1], bbox_max[1], grid_resolution + 1)
    verts = [[float(x), float(y), float(z)] for y in ys for x in xs]
    faces = []
    stride = grid_resolution + 1
    for y in range(grid_resolution):
        for x in range(grid_resolution):
            v0 = y * stride + x
            v1 = y * stride + x + 1
            v2 = (y + 1) * stride + x
            v3 = (y + 1) * stride + x + 1
            faces.extend([[v0, v1, v3], [v0, v3, v2]])
    area = float(abs((bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1])))
    cert = {
        "boundary_loop_support": bool(observed_support),
        "neighboring_surface_support": bool(observed_support),
        "sparse_depth_support": False,
        "free_space_risk": 0.15 if observed_support else 0.45,
        "semantic_ground_object_support": "ground_prior",
        "camera_coverage_score": 0.55 if observed_support else 0.0,
        "prior_only_flag": bool(not observed_support),
        "expected_topology_cost": len(faces),
        "expected_area_repaired": area,
    }
    edit = MeshEdit(
        edit_id=f"{proposal_id}_edit",
        edit_type=MeshSplatOptEditType.FILL_PATCH.value,
        defect_id="unknown",
        inserted_vertices=verts,
        inserted_faces=faces,
        topology_cost_delta=float(len(faces)),
        evidence_summary=cert,
        risk_summary={"free_space_risk": cert["free_space_risk"], "prior_only_flag": cert["prior_only_flag"]},
    )
    return FillProposal(proposal_id=proposal_id, fill_mode="ground_plane_void_fill", edit=edit, certificate=cert)

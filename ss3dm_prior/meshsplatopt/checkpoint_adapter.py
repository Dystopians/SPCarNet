from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState


FACE_FIELDS = ("importance_score", "image_size", "pixel_count")
VERTEX_FIELDS = ("triangles_points", "vertex_weight", "features_dc", "features_rest")


@dataclass(frozen=True)
class CheckpointEditReport:
    status: str
    input_path: str
    output_path: str
    edit_id: str
    edit_type: str
    triangles_before: int
    triangles_after: int
    vertices_before: int
    vertices_after: int
    supported: bool
    reason: str
    schema_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def checkpoint_to_mesh_state(payload: dict[str, Any]) -> MeshState:
    vertices = payload["triangles_points"].detach().cpu().numpy().astype(np.float64)
    faces = payload["_triangle_indices"].detach().cpu().numpy().astype(np.int64)
    return MeshState(vertices=vertices, faces=faces)


def validate_checkpoint_schema(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in ("triangles_points", "_triangle_indices"):
        if key not in payload:
            errors.append(f"missing {key}")
    if errors:
        return False, errors
    v = int(payload["triangles_points"].shape[0])
    f = int(payload["_triangle_indices"].shape[0])
    for key in VERTEX_FIELDS:
        if key in payload and hasattr(payload[key], "shape") and int(payload[key].shape[0]) != v:
            errors.append(f"{key} length mismatch")
    for key in FACE_FIELDS:
        if key in payload and hasattr(payload[key], "shape") and int(payload[key].shape[0]) != f:
            errors.append(f"{key} length mismatch")
    if f > 0:
        tri = payload["_triangle_indices"]
        if int(tri.min()) < 0 or int(tri.max()) >= v:
            errors.append("triangle index out of range")
    return not errors, errors


def _delete_checkpoint_faces(payload: dict[str, Any], face_ids: list[int]) -> dict[str, Any]:
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in payload.items()}
    f = int(out["_triangle_indices"].shape[0])
    mask = torch.ones((f,), dtype=torch.bool)
    valid = [i for i in face_ids if 0 <= int(i) < f]
    if valid:
        mask[torch.as_tensor(valid, dtype=torch.long)] = False
    out["_triangle_indices"] = out["_triangle_indices"][mask].clone()
    for key in FACE_FIELDS:
        if key in out and torch.is_tensor(out[key]) and int(out[key].shape[0]) == f:
            out[key] = out[key][mask].clone()
    return out


def _snap_checkpoint_vertices(payload: dict[str, Any], target_positions: dict[str, list[float]]) -> dict[str, Any]:
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in payload.items()}
    pts = out["triangles_points"].clone()
    for vid_text, pos in target_positions.items():
        vid = int(vid_text)
        if vid < 0 or vid >= int(pts.shape[0]):
            raise ValueError(f"SNAP_VERTICES vertex index out of range: {vid}")
        pts[vid] = torch.as_tensor(pos, dtype=pts.dtype)
    out["triangles_points"] = pts
    return out


def apply_edit_to_checkpoint_copy(
    checkpoint_path: str | Path,
    edit: MeshEdit,
    output_dir: str | Path,
) -> CheckpointEditReport:
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_state(checkpoint_path)
    before_valid, before_errors = validate_checkpoint_schema(payload)
    if not before_valid:
        raise ValueError(f"Invalid input checkpoint schema: {before_errors}")
    vertices_before = int(payload["triangles_points"].shape[0])
    triangles_before = int(payload["_triangle_indices"].shape[0])
    edit_type = MeshSplatOptEditType(edit.edit_type)
    supported = True
    reason = ""
    if edit_type == MeshSplatOptEditType.DELETE_TRIANGLES:
        payload = _delete_checkpoint_faces(payload, edit.affected_faces or edit.deleted_faces)
    elif edit_type == MeshSplatOptEditType.SNAP_VERTICES:
        payload = _snap_checkpoint_vertices(payload, edit.attribute_changes.get("target_positions", {}))
    elif edit_type in {MeshSplatOptEditType.PROTECT, MeshSplatOptEditType.APPEARANCE_RESET}:
        reason = "metadata_only_no_checkpoint_geometry_change"
    else:
        supported = False
        reason = f"{edit.edit_type} requires radiance/optimizer attribute initialization; deferred"
    output_path = out_dir / "point_cloud_state_dict.pt"
    if supported:
        torch.save(payload, output_path)
    after_valid, after_errors = validate_checkpoint_schema(payload)
    report = CheckpointEditReport(
        status="PASS" if supported and after_valid else "UNSUPPORTED",
        input_path=str(checkpoint_path),
        output_path=str(output_path) if supported else "",
        edit_id=edit.edit_id,
        edit_type=edit.edit_type,
        triangles_before=triangles_before,
        triangles_after=int(payload["_triangle_indices"].shape[0]),
        vertices_before=vertices_before,
        vertices_after=int(payload["triangles_points"].shape[0]),
        supported=supported,
        reason=reason or ("; ".join(after_errors) if after_errors else ""),
        schema_valid=after_valid,
    )
    (out_dir / "checkpoint_edit_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    lines = [
        "# Checkpoint Edit Report",
        "",
        f"- status: `{report.status}`",
        f"- edit: `{report.edit_id}` `{report.edit_type}`",
        f"- triangles: `{report.triangles_before}` -> `{report.triangles_after}`",
        f"- vertices: `{report.vertices_before}` -> `{report.vertices_after}`",
        f"- supported: `{report.supported}`",
        f"- reason: `{report.reason}`",
    ]
    (out_dir / "checkpoint_edit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report

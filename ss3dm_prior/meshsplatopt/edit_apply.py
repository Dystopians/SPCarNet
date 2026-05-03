from __future__ import annotations

from typing import Any

import numpy as np

from .edit_types import MeshEdit, MeshSplatOptEditType, MeshState


def verify_mesh_integrity(mesh_or_state: MeshState) -> dict[str, Any]:
    vertices = np.asarray(mesh_or_state.vertices)
    faces = np.asarray(mesh_or_state.faces)
    errors: list[str] = []
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        errors.append("vertices must have shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        errors.append("faces must have shape [M, 3]")
    if len(vertices) == 0 and len(faces) > 0:
        errors.append("faces exist without vertices")
    if len(faces) > 0:
        if int(faces.min()) < 0:
            errors.append("face index below zero")
        if int(faces.max()) >= len(vertices):
            errors.append("face index exceeds vertex count")
        degenerate = np.any(faces[:, 0:1] == faces[:, 1:], axis=1) | (faces[:, 1] == faces[:, 2])
        if bool(np.any(degenerate)):
            errors.append("degenerate faces present")
    if not np.all(np.isfinite(vertices)):
        errors.append("non-finite vertex coordinate")
    return {"valid": not errors, "errors": errors, "vertex_count": int(len(vertices)), "face_count": int(len(faces))}


def summarize_topology_delta(before: MeshState, after: MeshState) -> dict[str, int]:
    return {
        "vertices_before": int(len(before.vertices)),
        "vertices_after": int(len(after.vertices)),
        "vertices_delta": int(len(after.vertices) - len(before.vertices)),
        "faces_before": int(len(before.faces)),
        "faces_after": int(len(after.faces)),
        "faces_delta": int(len(after.faces) - len(before.faces)),
    }


def _remove_faces(faces: np.ndarray, remove_ids: list[int]) -> np.ndarray:
    if not remove_ids:
        return faces.copy()
    mask = np.ones((len(faces),), dtype=bool)
    valid_ids = [i for i in remove_ids if 0 <= i < len(faces)]
    mask[np.asarray(valid_ids, dtype=np.int64)] = False
    return faces[mask].copy()


def _remove_degenerate_faces(faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return faces.reshape(0, 3)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    return faces[keep].copy()


def apply_edit(mesh_or_state: MeshState, edit: MeshEdit) -> MeshState:
    before = mesh_or_state.copy()
    vertices = np.asarray(mesh_or_state.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh_or_state.faces, dtype=np.int64).copy()
    edit_type = MeshSplatOptEditType(edit.edit_type)

    if edit_type in {MeshSplatOptEditType.PROTECT, MeshSplatOptEditType.APPEARANCE_RESET}:
        mesh_or_state.attributes.setdefault("edit_notes", []).append(edit.to_dict())
    elif edit_type == MeshSplatOptEditType.DELETE_TRIANGLES:
        faces = _remove_faces(faces, edit.affected_faces or edit.deleted_faces)
    elif edit_type == MeshSplatOptEditType.EDGE_COLLAPSE:
        if len(edit.affected_vertices) < 2:
            raise ValueError("EDGE_COLLAPSE requires two affected vertices: keep, remove")
        keep, remove = int(edit.affected_vertices[0]), int(edit.affected_vertices[1])
        if keep < 0 or keep >= len(vertices) or remove < 0 or remove >= len(vertices):
            raise ValueError("EDGE_COLLAPSE vertex index out of range")
        vertices[keep] = 0.5 * (vertices[keep] + vertices[remove])
        faces[faces == remove] = keep
        faces = _remove_degenerate_faces(faces)
    elif edit_type == MeshSplatOptEditType.FACE_MERGE:
        remove = edit.affected_faces[1:] if len(edit.affected_faces) > 1 else []
        faces = _remove_faces(faces, remove)
    elif edit_type == MeshSplatOptEditType.SNAP_VERTICES:
        targets = edit.attribute_changes.get("target_positions", {})
        for vid_text, pos in targets.items():
            vid = int(vid_text)
            if vid < 0 or vid >= len(vertices):
                raise ValueError(f"SNAP_VERTICES vertex index out of range: {vid}")
            vertices[vid] = np.asarray(pos, dtype=np.float64)
    elif edit_type == MeshSplatOptEditType.SPLIT_TRIANGLES:
        replace_ids = sorted({int(i) for i in edit.affected_faces if 0 <= int(i) < len(faces)})
        new_faces = []
        remove_set = set(replace_ids)
        for fi, face in enumerate(faces):
            if fi not in remove_set:
                new_faces.append(face.tolist())
                continue
            centroid = vertices[face].mean(axis=0)
            cid = len(vertices)
            vertices = np.vstack([vertices, centroid.reshape(1, 3)])
            new_faces.extend([[int(face[0]), int(face[1]), cid], [int(face[1]), int(face[2]), cid], [int(face[2]), int(face[0]), cid]])
        faces = np.asarray(new_faces, dtype=np.int64)
    elif edit_type == MeshSplatOptEditType.FILL_PATCH:
        patch_vertices = np.asarray(edit.inserted_vertices, dtype=np.float64)
        patch_faces = np.asarray(edit.inserted_faces, dtype=np.int64)
        if patch_vertices.size == 0 or patch_faces.size == 0:
            raise ValueError("FILL_PATCH requires inserted vertices and faces")
        offset = len(vertices)
        vertices = np.vstack([vertices, patch_vertices.reshape(-1, 3)])
        faces = np.vstack([faces, patch_faces.reshape(-1, 3) + offset])
    else:
        raise ValueError(f"Unsupported edit type: {edit.edit_type}")

    mesh_or_state.vertices = vertices
    mesh_or_state.faces = faces
    mesh_or_state.attributes["last_topology_delta"] = summarize_topology_delta(before, mesh_or_state)
    integrity = verify_mesh_integrity(mesh_or_state)
    if not integrity["valid"]:
        raise ValueError(f"Mesh integrity failed after {edit.edit_type}: {integrity['errors']}")
    return mesh_or_state

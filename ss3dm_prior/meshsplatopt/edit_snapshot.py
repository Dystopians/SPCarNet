from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .edit_types import MeshState


def mesh_checksum(state: MeshState) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(state.vertices, dtype=np.float64).tobytes())
    h.update(np.asarray(state.faces, dtype=np.int64).tobytes())
    h.update(json.dumps(state.attributes, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def create_snapshot(mesh_or_state: MeshState, snapshot_path: str | Path) -> str:
    path = Path(snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = mesh_or_state.copy()
    attrs = json.dumps(state.attributes, sort_keys=True, default=str)
    checksum = mesh_checksum(state)
    np.savez(path, vertices=state.vertices, faces=state.faces, attributes_json=attrs, checksum=checksum)
    return str(path)


def rollback_edit(mesh_or_state: MeshState, snapshot_path: str | Path) -> MeshState:
    data = np.load(snapshot_path, allow_pickle=False)
    restored = MeshState(
        vertices=np.asarray(data["vertices"], dtype=np.float64).copy(),
        faces=np.asarray(data["faces"], dtype=np.int64).copy(),
        attributes=json.loads(str(data["attributes_json"])),
    )
    expected = str(data["checksum"])
    actual = mesh_checksum(restored)
    if expected != actual:
        raise ValueError(f"Snapshot checksum mismatch: expected {expected}, got {actual}")
    mesh_or_state.vertices = restored.vertices
    mesh_or_state.faces = restored.faces
    mesh_or_state.attributes = restored.attributes
    return mesh_or_state

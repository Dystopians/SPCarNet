#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import apply_edit_to_checkpoint_copy, load_checkpoint_state, validate_checkpoint_schema
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType


def make_synthetic_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = 4
    f = 2
    payload = {
        "triangles_points": torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float32),
        "_triangle_indices": torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32),
        "vertex_weight": torch.ones((v, 1), dtype=torch.float32),
        "sigma": 0.0,
        "active_sh_degree": 0,
        "features_dc": torch.zeros((v, 1, 3), dtype=torch.float32),
        "features_rest": torch.zeros((v, 15, 3), dtype=torch.float32),
        "importance_score": torch.ones((f,), dtype=torch.float32),
        "image_size": torch.ones((f,), dtype=torch.float32),
        "pixel_count": torch.ones((f,), dtype=torch.int32),
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/stageR14_1_checkpoint_adapter_smoke")
    args = parser.parse_args()
    out = Path(args.output_dir)
    ckpt = out / "input" / "point_cloud_state_dict.pt"
    make_synthetic_checkpoint(ckpt)
    delete = MeshEdit("delete_face", MeshSplatOptEditType.DELETE_TRIANGLES.value, "smoke", affected_faces=[0])
    snap = MeshEdit(
        "snap_vertex",
        MeshSplatOptEditType.SNAP_VERTICES.value,
        "smoke",
        affected_vertices=[2],
        attribute_changes={"target_positions": {"2": [1.0, 1.0, 0.25]}},
    )
    fill = MeshEdit(
        "fill_unsupported",
        MeshSplatOptEditType.FILL_PATCH.value,
        "smoke",
        inserted_vertices=[[2, 0, 0], [2, 1, 0], [3, 0, 0]],
        inserted_faces=[[0, 1, 2]],
    )
    delete_report = apply_edit_to_checkpoint_copy(ckpt, delete, out / "delete")
    snap_report = apply_edit_to_checkpoint_copy(ckpt, snap, out / "snap")
    fill_report = apply_edit_to_checkpoint_copy(ckpt, fill, out / "fill")
    delete_payload = load_checkpoint_state(out / "delete" / "point_cloud_state_dict.pt")
    snap_payload = load_checkpoint_state(out / "snap" / "point_cloud_state_dict.pt")
    fill_payload = load_checkpoint_state(out / "fill" / "point_cloud_state_dict.pt")
    delete_valid, _ = validate_checkpoint_schema(delete_payload)
    snap_valid, _ = validate_checkpoint_schema(snap_payload)
    fill_valid, _ = validate_checkpoint_schema(fill_payload)
    checks = {
        "delete_updates_face_arrays": delete_report.triangles_after == 1 and int(delete_payload["importance_score"].shape[0]) == 1,
        "snap_updates_vertex": abs(float(snap_payload["triangles_points"][2, 2]) - 0.25) < 1e-6,
        "fill_appends_vertices_faces": fill_report.supported and fill_report.vertices_after == 7 and fill_report.triangles_after == 3,
        "fill_schema_valid": fill_valid,
        "delete_schema_valid": delete_valid,
        "snap_schema_valid": snap_valid,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "delete_report": delete_report.to_dict(),
        "snap_report": snap_report.to_dict(),
        "fill_report": fill_report.to_dict(),
    }
    (out / "checkpoint_adapter_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(f"Checkpoint adapter smoke failed: {report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

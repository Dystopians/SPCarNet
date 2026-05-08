from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np
import torch


VERTEX_KEYS = ("triangles_points", "vertex_weight", "features_dc", "features_rest")
FACE_KEYS = ("importance_score", "image_size", "pixel_count")


@dataclass(frozen=True)
class CompactionAudit:
    source_model: str
    source_checkpoint: str
    output_model: str
    output_checkpoint: str
    iteration: int
    selector_mode: str
    pre_triangles: int
    post_triangles: int
    pre_vertices: int
    post_vertices: int
    removed_triangles: int
    removed_fraction: float
    degenerate_face_count: int
    invalid_index_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_path(model_path: str | Path, iteration: int) -> Path:
    model = Path(model_path)
    direct = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if direct.is_file():
        return direct
    nested = model / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"point_cloud_state_dict.pt not found for {model} iteration {iteration}")


def copy_model_metadata(source_model: str | Path, output_model: str | Path) -> None:
    src_model = Path(source_model)
    out_model = Path(output_model)
    out_model.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = src_model / name
        if src.is_file():
            shutil.copy2(src, out_model / name)


def _compact_state(
    state: dict[str, Any],
    selected_faces: np.ndarray,
    *,
    keep_unused_vertices: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    faces = state["_triangle_indices"].detach().cpu().long()
    vertices = state["triangles_points"].detach().cpu()
    face_count_before = int(faces.shape[0])
    remove_mask = torch.zeros((face_count_before,), dtype=torch.bool)
    selected = torch.as_tensor(np.asarray(selected_faces, dtype=np.int64), dtype=torch.long)
    if selected.numel():
        if int(selected.min()) < 0 or int(selected.max()) >= face_count_before:
            raise ValueError("selected face ids are out of range")
        remove_mask[selected] = True
    keep_mask = ~remove_mask
    kept_faces = faces[keep_mask]
    if bool(keep_unused_vertices):
        out: dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                if key == "_triangle_indices":
                    out[key] = kept_faces.to(dtype=state["_triangle_indices"].dtype).clone()
                elif key in FACE_KEYS and value.shape[0] == face_count_before:
                    out[key] = value.detach().cpu()[keep_mask].clone()
                else:
                    out[key] = value.detach().cpu().clone()
            else:
                out[key] = value
        return out, {
            "pre_triangles": face_count_before,
            "post_triangles": int(kept_faces.shape[0]),
            "pre_vertices": int(vertices.shape[0]),
            "post_vertices": int(vertices.shape[0]),
        }

    vertex_count_before = int(vertices.shape[0])
    if kept_faces.numel():
        if int(kept_faces.min().item()) < 0 or int(kept_faces.max().item()) >= vertex_count_before:
            raise ValueError("checkpoint face indices are out of vertex range")
    used_vertices = torch.unique(kept_faces.reshape(-1), sorted=True)
    remap = torch.full((vertex_count_before,), -1, dtype=torch.long)
    remap[used_vertices] = torch.arange(len(used_vertices), dtype=torch.long)
    new_faces = remap[kept_faces].to(dtype=state["_triangle_indices"].dtype)

    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            if key == "_triangle_indices":
                out[key] = new_faces.clone()
            elif key in VERTEX_KEYS and value.shape[0] == vertex_count_before:
                out[key] = value.detach().cpu()[used_vertices].clone()
            elif key in FACE_KEYS and value.shape[0] == face_count_before:
                out[key] = value.detach().cpu()[keep_mask].clone()
            else:
                out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out, {
        "pre_triangles": face_count_before,
        "post_triangles": int(new_faces.shape[0]),
        "pre_vertices": int(vertices.shape[0]),
        "post_vertices": int(out["triangles_points"].shape[0]),
    }


def validate_faces(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[int, int]:
    faces_long = faces.detach().cpu().long()
    invalid = int(((faces_long < 0) | (faces_long >= int(vertices.shape[0]))).sum().item())
    repeated = (
        (faces_long[:, 0] == faces_long[:, 1])
        | (faces_long[:, 1] == faces_long[:, 2])
        | (faces_long[:, 0] == faces_long[:, 2])
    )
    degenerate = int(repeated.sum().item())
    return degenerate, invalid


def apply_compaction(
    source_model: str | Path,
    output_model: str | Path,
    iteration: int,
    selected_faces: np.ndarray,
    selector_mode: str,
    *,
    keep_unused_vertices: bool = False,
) -> CompactionAudit:
    src_model = Path(source_model)
    out_model = Path(output_model)
    src_checkpoint = checkpoint_path(src_model, iteration)
    out_checkpoint = out_model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copy_model_metadata(src_model, out_model)

    state = torch.load(src_checkpoint, map_location="cpu")
    compact_state, stats = _compact_state(state, selected_faces, keep_unused_vertices=bool(keep_unused_vertices))
    degenerate, invalid = validate_faces(compact_state["triangles_points"], compact_state["_triangle_indices"])
    torch.save(compact_state, out_checkpoint)

    removed = int(stats["pre_triangles"] - stats["post_triangles"])
    audit = CompactionAudit(
        source_model=str(src_model),
        source_checkpoint=str(src_checkpoint),
        output_model=str(out_model),
        output_checkpoint=str(out_checkpoint),
        iteration=int(iteration),
        selector_mode=selector_mode,
        pre_triangles=int(stats["pre_triangles"]),
        post_triangles=int(stats["post_triangles"]),
        pre_vertices=int(stats["pre_vertices"]),
        post_vertices=int(stats["post_vertices"]),
        removed_triangles=removed,
        removed_fraction=float(removed / max(int(stats["pre_triangles"]), 1)),
        degenerate_face_count=degenerate,
        invalid_index_count=invalid,
    )
    (out_model / "topology_audit.json").write_text(json.dumps(audit.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out_model / "topology_audit.md").write_text(
        "\n".join(
            [
                "# Checkpoint Compaction Topology Audit",
                "",
                f"- selector mode: `{selector_mode}`",
                f"- iteration: `{iteration}`",
                f"- triangles: `{audit.pre_triangles}` -> `{audit.post_triangles}`",
                f"- vertices: `{audit.pre_vertices}` -> `{audit.post_vertices}`",
                f"- removed fraction: `{audit.removed_fraction:.6f}`",
                f"- keep unused vertices: `{bool(keep_unused_vertices)}`",
                f"- degenerate face count: `{audit.degenerate_face_count}`",
                f"- invalid index count: `{audit.invalid_index_count}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return audit

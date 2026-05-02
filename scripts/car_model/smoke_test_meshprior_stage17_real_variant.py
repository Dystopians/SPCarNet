"""Smoke checks for the Stage 17 real MeshPrior variant artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_model(model_dir: Path, iteration: int) -> dict:
    checkpoint = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    assert checkpoint.is_file(), f"missing checkpoint: {checkpoint}"
    for name in ("cfg_args", "cameras.json", "input.ply"):
        assert (model_dir / name).is_file(), f"missing model layout file: {model_dir / name}"
    state = torch.load(checkpoint, map_location="cpu")
    triangles = int(state["_triangle_indices"].shape[0])
    vertices = int(state["triangles_points"].shape[0])
    assert triangles > 0, "empty triangle checkpoint"
    assert vertices > 0, "empty vertex checkpoint"
    return {"checkpoint": str(checkpoint), "triangles": triangles, "vertices": vertices}


def main() -> None:
    root = Path("outputs/carnet/meshprior/parking_phone_tiny/stage17_real_variant_2000iter")
    init_report = root / "model" / "meshprior_recovery_model_report.json"
    assert init_report.is_file(), f"missing initialization report: {init_report}"
    report = _load_json(init_report)
    assert report.get("source_model_edited") is False
    assert report.get("recovery_model_written") is True
    init_stats = _check_model(root / "model", 200)
    assert int(report["triangles"]) == int(init_stats["triangles"])
    assert int(report["vertices"]) == int(init_stats["vertices"])
    print(
        json.dumps(
            {
                "stage17_init_ok": True,
                "model": str(root / "model"),
                "iteration": 200,
                "triangles": init_stats["triangles"],
                "vertices": init_stats["vertices"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

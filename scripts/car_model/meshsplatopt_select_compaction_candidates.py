#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.compact_selector import (  # noqa: E402
    SELECTOR_MODES,
    CompactionSignals,
    select_faces,
    write_selector_outputs,
)


def _checkpoint_path(model_path: Path, iteration: int) -> Path:
    direct = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if direct.is_file():
        return direct
    nested = model_path / "model" / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"Could not find point_cloud_state_dict.pt for {model_path} iteration {iteration}")


def _load_checkpoint_signals(model_path: Path, iteration: int) -> CompactionSignals:
    import torch

    checkpoint = _checkpoint_path(model_path, iteration)
    state = torch.load(checkpoint, map_location="cpu")
    vertices = state["triangles_points"].detach().cpu().numpy()
    faces = state["_triangle_indices"].detach().cpu().numpy()
    face_count = faces.shape[0]

    def face_signal(name: str) -> np.ndarray | None:
        value = state.get(name)
        if value is None or not hasattr(value, "shape") or int(value.shape[0]) != face_count:
            return None
        return value.detach().cpu().float().reshape(face_count, -1).mean(dim=1).numpy()

    importance = face_signal("importance_score")
    pixel_count = face_signal("pixel_count")
    image_size = face_signal("image_size")
    render_contribution = None
    if importance is not None:
        render_contribution = importance
    elif pixel_count is not None and image_size is not None:
        render_contribution = pixel_count / np.maximum(image_size, 1e-6)

    return CompactionSignals(
        vertices=vertices,
        faces=faces,
        render_contribution=render_contribution,
        sparse_support=None,
        normal_support=None,
        positive_surface_evidence=render_contribution,
        negative_free_space=None,
        explanation_debt=None,
        topology_cost=None,
        uncertainty=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Select MeshSplatOpt compaction candidates.")
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--mode", choices=SELECTOR_MODES, default="csef_low_evidence_boundary_protected")
    parser.add_argument("--target_prune_fraction", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    signals = _load_checkpoint_signals(Path(args.source_model), args.iteration)
    selected, table = select_faces(
        signals,
        mode=args.mode,
        target_prune_fraction=args.target_prune_fraction,
        seed=args.seed,
    )
    payload = write_selector_outputs(args.out_dir, selected, table, args.mode, args.target_prune_fraction)
    print(f"Wrote {len(payload['selected_faces'])} candidates to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

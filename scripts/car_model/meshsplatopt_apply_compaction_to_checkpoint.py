#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import apply_compaction, checkpoint_path  # noqa: E402
from ss3dm_prior.meshsplatopt.compact_selector import (  # noqa: E402
    CompactionSignals,
    SELECTOR_MODES,
    decide_adaptive_compaction_policy,
    select_faces,
    write_selector_outputs,
)


def _load_checkpoint_signals(source_model: Path, iteration: int) -> CompactionSignals:
    import torch

    state = torch.load(checkpoint_path(source_model, iteration), map_location="cpu")
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
        vertices=state["triangles_points"].detach().cpu().numpy(),
        faces=faces,
        render_contribution=render_contribution,
        positive_surface_evidence=render_contribution,
    )


def _selected_from_json(path: Path) -> tuple[np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected_faces_path" in payload:
        selected_path = path.parent / str(payload["selected_faces_path"])
        return np.load(selected_path).astype(np.int64, copy=False), str(payload.get("mode", "from_json"))
    return np.asarray(payload["selected_faces"], dtype=np.int64), str(payload.get("mode", "from_json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply MeshSplatOpt compaction candidates to a checkpoint copy.")
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--candidates_json", default="")
    parser.add_argument("--selector_mode", choices=SELECTOR_MODES, default="area_smallest")
    parser.add_argument("--target_prune_fraction", type=float, default=0.70)
    parser.add_argument("--selector_out_dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep_unused_vertices",
        action="store_true",
        help="Delete selected faces without remapping vertices. This is faster for very large checkpoints but keeps vertex count unchanged.",
    )
    args = parser.parse_args()

    if args.candidates_json:
        selected, mode = _selected_from_json(Path(args.candidates_json))
    else:
        signals = _load_checkpoint_signals(Path(args.source_model), args.iteration)
        policy = None
        target_prune_fraction = float(args.target_prune_fraction)
        if args.selector_mode == "csef_adaptive_policy":
            policy, table = decide_adaptive_compaction_policy(signals, seed=args.seed)
            target_prune_fraction = float(policy.target_prune_fraction)
            selected, table = select_faces(signals, args.selector_mode, target_prune_fraction, seed=args.seed)
        else:
            selected, table = select_faces(signals, args.selector_mode, target_prune_fraction, seed=args.seed)
        mode = args.selector_mode
        selector_out = Path(args.selector_out_dir) if args.selector_out_dir else Path(args.output_model) / "selector"
        write_selector_outputs(selector_out, selected, table, args.selector_mode, target_prune_fraction, policy_decision=policy)
    audit = apply_compaction(
        args.source_model,
        args.output_model,
        args.iteration,
        selected,
        mode,
        keep_unused_vertices=bool(args.keep_unused_vertices),
    )
    print(json.dumps(audit.to_dict(), indent=2))
    return 0 if audit.invalid_index_count == 0 and audit.degenerate_face_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

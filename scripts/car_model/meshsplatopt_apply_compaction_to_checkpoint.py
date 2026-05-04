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
from ss3dm_prior.meshsplatopt.compact_selector import CompactionSignals, SELECTOR_MODES, select_faces, write_selector_outputs  # noqa: E402


def _load_checkpoint_signals(source_model: Path, iteration: int) -> CompactionSignals:
    import torch

    state = torch.load(checkpoint_path(source_model, iteration), map_location="cpu")
    return CompactionSignals(
        vertices=state["triangles_points"].detach().cpu().numpy(),
        faces=state["_triangle_indices"].detach().cpu().numpy(),
    )


def _selected_from_json(path: Path) -> tuple[np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
    args = parser.parse_args()

    if args.candidates_json:
        selected, mode = _selected_from_json(Path(args.candidates_json))
    else:
        signals = _load_checkpoint_signals(Path(args.source_model), args.iteration)
        selected, table = select_faces(signals, args.selector_mode, args.target_prune_fraction, seed=args.seed)
        mode = args.selector_mode
        selector_out = Path(args.selector_out_dir) if args.selector_out_dir else Path(args.output_model) / "selector"
        write_selector_outputs(selector_out, selected, table, args.selector_mode, args.target_prune_fraction)
    audit = apply_compaction(args.source_model, args.output_model, args.iteration, selected, mode)
    print(json.dumps(audit.to_dict(), indent=2))
    return 0 if audit.invalid_index_count == 0 and audit.degenerate_face_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

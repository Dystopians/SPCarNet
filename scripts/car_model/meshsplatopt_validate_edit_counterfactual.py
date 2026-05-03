#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.counterfactual_edit_gate import validate_edit_counterfactual, write_counterfactual_report
from ss3dm_prior.meshsplatopt.csef_builder import load_mesh
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--edit_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--commit_on_accept", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vertices, faces = load_mesh(args.mesh_path)
    edit = MeshEdit(**json.loads(Path(args.edit_json).read_text(encoding="utf-8")))
    state = MeshState(vertices, faces)
    report = validate_edit_counterfactual(
        state,
        edit,
        snapshot_path=Path(args.output_json).with_suffix(".snapshot.npz"),
        commit_on_accept=args.commit_on_accept,
    )
    write_counterfactual_report(report, args.output_json)
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.csef_builder import build_csef, load_mesh, write_csef_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MeshSplatOpt CSEF diagnostics for a mesh.")
    parser.add_argument("--mesh_path", required=True, help="Input mesh path. PLY/OBJ supported when trimesh is available.")
    parser.add_argument("--output_dir", required=True, help="Directory for CSEF artifacts.")
    parser.add_argument("--scene_model", default="unknown")
    parser.add_argument("--scene_source", default="mesh")
    parser.add_argument("--external_evidence_available", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh_path)
    vertices, faces = load_mesh(mesh_path)
    result, samples = build_csef(
        vertices,
        faces,
        scene_model=args.scene_model,
        scene_source=args.scene_source,
        mesh_path=str(mesh_path),
        external_evidence_available=bool(args.external_evidence_available),
    )
    write_csef_outputs(result, samples, args.output_dir)
    print(f"Wrote CSEF artifacts to {args.output_dir}")
    print(f"regions={len(result.regions)} samples={len(samples)}")


if __name__ == "__main__":
    main()

"""Convert one town-level OBJ into a binary cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ss3dm_prior.data.obj_converter import convert_obj_to_cache
from ss3dm_prior.utils.io import load_yaml


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a large town OBJ mesh into compact binary cache files."
    )
    parser.add_argument("--town_id", required=True, help="Town id, for example Town10.")
    parser.add_argument("--obj_path", required=True, help="Path to the source OBJ file.")
    parser.add_argument("--out_dir", required=True, help="Output root directory or town cache dir.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config with dtype preferences.",
    )
    return parser


def main() -> int:
    args = make_argparser().parse_args()
    vertex_dtype = "float32"
    face_dtype = "int32"
    if args.config:
        config = load_yaml(args.config)
        mesh_config = config.get("town_mesh_cache", config)
        vertex_dtype = str(mesh_config.get("vertex_dtype", vertex_dtype))
        face_dtype = str(mesh_config.get("face_dtype", face_dtype))

    out_root = Path(args.out_dir).expanduser().resolve()
    town_out_dir = out_root if out_root.name == args.town_id else out_root / args.town_id
    conversion_command = " ".join(sys.argv)
    converted = convert_obj_to_cache(
        obj_path=args.obj_path,
        out_dir=town_out_dir,
        town_id=args.town_id,
        conversion_command=conversion_command,
        vertex_dtype=vertex_dtype,
        face_dtype=face_dtype,
    )

    print(f"town_id: {args.town_id}")
    print(f"cache_dir: {town_out_dir}")
    print(f"num_vertices: {len(converted.vertices)}")
    print(f"num_faces: {len(converted.faces)}")
    print(f"bbox_min: {converted.bbox['min']}")
    print(f"bbox_max: {converted.bbox['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

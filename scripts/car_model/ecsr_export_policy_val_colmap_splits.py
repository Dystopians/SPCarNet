#!/usr/bin/env python3
"""Export ECSR fitting/policy-val splits as COLMAP loader split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train_colmap_file"),
    )
    parser.add_argument("--images", default="images_2")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PolicyValColmapSplits.md"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm(name: str) -> str:
    return Path(str(name)).stem


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def all_image_names(source_path: Path, images: str) -> list[str]:
    image_dir = source_path / images
    if not image_dir.exists():
        image_dir = source_path / "images"
    names = sorted(norm(p.name) for p in image_dir.iterdir() if p.is_file())
    if not names:
        raise RuntimeError(f"no image files found under {image_dir}")
    return names


def convert_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    policy = load_json(args.split_root / scene / "policy_split.json")
    source_path = Path(policy["source_path"])
    fitting = [norm(v["image_name"]) for v in policy["fitting_train"]]
    policy_val = [norm(v["image_name"]) for v in policy["policy_val"]]
    fitting_set = set(fitting)
    policy_set = set(policy_val)
    all_names = all_image_names(source_path, args.images)
    dropped = [name for name in all_names if name not in fitting_set and name not in policy_set]
    missing_fitting = sorted(fitting_set.difference(all_names))
    missing_policy = sorted(policy_set.difference(all_names))
    payload = {
        "train": fitting,
        "test": policy_val,
        "dropped": dropped,
        "metadata": {
            "scene": scene,
            "source_policy_split": str(args.split_root / scene / "policy_split.json"),
            "source_path": str(source_path),
            "images": args.images,
            "test_usage": "none; this split uses only original train cameras",
            "role": "ECSR fitting_train as train, policy_val as loader test, original held-out cameras dropped",
            "missing_fitting": missing_fitting,
            "missing_policy_val": missing_policy,
        },
    }
    scene_dir = args.out_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_path = scene_dir / "split_file.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "scene": scene,
        "path": str(out_path),
        "train": len(fitting),
        "policy_val": len(policy_val),
        "dropped": len(dropped),
        "missing_fitting": len(missing_fitting),
        "missing_policy_val": len(missing_policy),
    }


def main() -> int:
    args = parse_args()
    scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [convert_scene(args, scene) for scene in scenes]
    summary = {
        "split_root": str(args.split_root),
        "out_root": str(args.out_root),
        "images": args.images,
        "rows": rows,
    }
    (args.out_root / "full_train_colmap_file_splits_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    md_rows = [
        [
            row["scene"],
            row["train"],
            row["policy_val"],
            row["dropped"],
            row["missing_fitting"],
            row["missing_policy_val"],
            f"`{row['path']}`",
        ]
        for row in rows
    ]
    md = [
        "# ECSR Policy-Val COLMAP Split Files",
        "",
        "These split files convert the deterministic full-train fitting/policy-val",
        "records into the native COLMAP loader format. They are for Phase-D",
        "policy certificates: fitting views are loader train views, policy-val",
        "views are loader test views, and the original LLFF held-out test views",
        "are dropped so they cannot affect candidate acceptance.",
        "",
        md_table(
            ["scene", "fitting train", "policy-val", "dropped", "missing train", "missing val", "split file"],
            md_rows,
        ),
        "",
    ]
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.out_root / 'full_train_colmap_file_splits_summary.json'}")
    print(f"[ECSR] wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

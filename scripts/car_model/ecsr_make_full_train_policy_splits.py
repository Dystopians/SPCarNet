#!/usr/bin/env python3
"""Create deterministic full-train fitting/policy-val splits for ECSR."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from triangle_renderer import TriangleModel


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


def scene_seed(seed: int, scene: str) -> int:
    digest = hashlib.sha256(scene.encode("utf-8")).hexdigest()
    return int(seed) + int(digest[:8], 16)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def view_record(index: int, view: Any) -> dict[str, Any]:
    return {
        "train_view_index": int(index),
        "image_name": str(getattr(view, "image_name", f"{index:05d}")),
        "width": int(getattr(view, "image_width", 0) or getattr(view, "width", 0)),
        "height": int(getattr(view, "image_height", 0) or getattr(view, "height", 0)),
    }


def make_split(args: argparse.Namespace, dataset: Any, scene_name: str) -> dict[str, Any]:
    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = int(args.internal_upsample)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=int(args.iteration),
        shuffle=False,
    )
    views = [view_record(i, view) for i, view in enumerate(scene.getTrainCameras())]
    if len(views) < 2:
        raise RuntimeError(f"{scene_name}: need at least two train views, got {len(views)}")
    rng = random.Random(scene_seed(int(args.seed), scene_name))
    order = list(range(len(views)))
    rng.shuffle(order)
    val_count = max(1, min(len(views) - 1, round(len(views) * float(args.policy_val_fraction))))
    policy_positions = set(order[:val_count])
    fitting = [view for pos, view in enumerate(views) if pos not in policy_positions]
    policy = [view for pos, view in enumerate(views) if pos in policy_positions]
    return {
        "scene": scene_name,
        "seed": int(args.seed),
        "scene_seed": scene_seed(int(args.seed), scene_name),
        "split_scope": "full_train_scene_loader",
        "model_path": str(dataset.model_path),
        "source_path": str(dataset.source_path),
        "iteration": int(args.iteration),
        "test_usage": "none",
        "num_train_views": len(views),
        "num_fitting_train": len(fitting),
        "num_policy_val": len(policy),
        "fitting_train": fitting,
        "policy_val": policy,
    }


def write_outputs(args: argparse.Namespace, splits: list[dict[str, Any]]) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        scene_dir = out_root / str(split["scene"])
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "policy_split.json").write_text(
            json.dumps(split, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "seed": int(args.seed),
        "policy_val_fraction": float(args.policy_val_fraction),
        "split_scope": "full_train_scene_loader",
        "test_usage": "none",
        "splits": splits,
    }
    (out_root / "full_train_policy_splits_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        [
            split["scene"],
            split["num_train_views"],
            split["num_fitting_train"],
            split["num_policy_val"],
            f"`{out_root / str(split['scene']) / 'policy_split.json'}`",
        ]
        for split in splits
    ]
    md = [
        "# ECSR Full-Train Policy Split",
        "",
        "This is the deterministic fitting/policy-val split required before",
        "Phase-C/D candidate acceptance. It is generated from the scene loader's",
        "full train camera list. Held-out test views are not used.",
        "",
        f"- seed: `{int(args.seed)}`",
        f"- policy-val fraction: `{float(args.policy_val_fraction):.3f}`",
        "- split scope: `full_train_scene_loader`",
        "- held-out test usage: `none`",
        "",
        md_table(["scene", "train views", "fitting", "policy-val", "split JSON"], rows),
        "",
        "Use this split for long Phase-C/D optimization and candidate acceptance.",
        "The earlier cached-view split remains only a smoke-test split.",
        "",
    ]
    doc_out = Path(args.doc_out)
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    doc_out.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create ECSR full-train policy splits.")
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--policy_val_fraction", type=float, default=0.20)
    parser.add_argument("--internal_upsample", type=int, default=4)
    parser.add_argument(
        "--out_root",
        default="outputs/carnet/meshsplatopt/ecsr_policy_splits/full_train",
    )
    parser.add_argument(
        "--doc_out",
        default="docs/car_model/5-8-ECSR-FullTrainPolicySplit.md",
    )
    args = get_combined_args(parser)
    if not args.scene_name:
        scene_name = Path(args.model_path).parts[-3]
    else:
        scene_name = str(args.scene_name)
    split = make_split(args, model.extract(args), scene_name)
    write_outputs(args, [split])
    print(json.dumps({"scene": scene_name, "num_train_views": split["num_train_views"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create fixed fitting/policy-val splits for ECSR Phase-A/B cached train views."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
        "--surface_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_policy_splits/phase_ab_cached_views"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PolicySplit.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--policy_val_fraction", type=float, default=0.5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_seed(seed: int, scene: str) -> int:
    digest = hashlib.sha256(scene.encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def build_split(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    summary_path = args.surface_root / scene / "surface_evidence_summary.json"
    summary = load_json(summary_path)
    views = []
    by_index = {int(v["view_index"]): v for v in summary["view_summaries"]}
    for idx in summary["selected_views"]:
        view = by_index[int(idx)]
        views.append(
            {
                "train_view_index": int(idx),
                "image_name": str(view["image_name"]),
                "width": int(view["width"]),
                "height": int(view["height"]),
            }
        )
    rng = random.Random(scene_seed(args.seed, scene))
    order = list(range(len(views)))
    rng.shuffle(order)
    val_count = max(1, min(len(views) - 1, round(len(views) * args.policy_val_fraction)))
    policy_positions = set(order[:val_count])
    fitting = [view for pos, view in enumerate(views) if pos not in policy_positions]
    policy = [view for pos, view in enumerate(views) if pos in policy_positions]
    return {
        "scene": scene,
        "seed": args.seed,
        "scene_seed": scene_seed(args.seed, scene),
        "split_scope": "phase_ab_cached_train_views",
        "test_usage": "none",
        "warning": (
            "This split covers the cached Phase-A/B train views only. "
            "Regenerate a full-train split before long Phase-C/D optimization."
        ),
        "num_cached_train_views": len(views),
        "num_fitting_train": len(fitting),
        "num_policy_val": len(policy),
        "fitting_train": fitting,
        "policy_val": policy,
    }


def main() -> int:
    args = parse_args()
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    splits = []
    for scene in scenes:
        split = build_split(args, scene)
        scene_dir = args.out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "policy_split.json").write_text(
            json.dumps(split, indent=2) + "\n", encoding="utf-8"
        )
        splits.append(split)

    rows = [
        [
            split["scene"],
            split["num_cached_train_views"],
            split["num_fitting_train"],
            split["num_policy_val"],
            ", ".join(str(v["train_view_index"]) for v in split["fitting_train"]),
            ", ".join(str(v["train_view_index"]) for v in split["policy_val"]),
        ]
        for split in splits
    ]
    summary = {
        "seed": args.seed,
        "policy_val_fraction": args.policy_val_fraction,
        "split_scope": "phase_ab_cached_train_views",
        "test_usage": "none",
        "warning": "This is not the final full-train split for long Phase-C/D optimization.",
        "splits": splits,
    }
    (args.out_root / "phase_ab_policy_splits_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# ECSR Phase-A/B Policy Split",
        "",
        "This file fixes a deterministic fitting/policy-val split for the cached",
        "Phase-A/B train views. It is sufficient for local certificate smoke tests",
        "on the current cache, but it is not the final full-train split required",
        "before long Phase-C/D optimization.",
        "",
        f"- seed: `{args.seed}`",
        f"- policy-val fraction: `{args.policy_val_fraction}`",
        "- split scope: `phase_ab_cached_train_views`",
        "- held-out test usage: `none`",
        "",
        md_table(
            [
                "scene",
                "cached views",
                "fitting",
                "policy-val",
                "fitting train indices",
                "policy-val train indices",
            ],
            rows,
        ),
        "",
        "Per-scene JSON files are saved under",
        f"`{args.out_root}`.",
        "",
        "Before Phase-C/D full optimization, regenerate this with all train views",
        "from the scene loader and keep held-out test views excluded from all",
        "candidate, strength, crop, and rollback decisions.",
        "",
    ]
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.out_root / 'phase_ab_policy_splits_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

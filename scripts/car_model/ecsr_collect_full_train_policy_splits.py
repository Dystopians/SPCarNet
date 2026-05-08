#!/usr/bin/env python3
"""Collect per-scene ECSR full-train policy split files into one report."""

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
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-FullTrainPolicySplit.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    splits = []
    missing = []
    for scene in scenes:
        path = args.split_root / scene / "policy_split.json"
        if not path.exists():
            missing.append(scene)
            continue
        splits.append(load_json(path))
    summary = {
        "split_scope": "full_train_scene_loader",
        "test_usage": "none",
        "split_root": str(args.split_root),
        "missing": missing,
        "splits": splits,
    }
    (args.split_root / "full_train_policy_splits_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        [
            split["scene"],
            split["num_train_views"],
            split["num_fitting_train"],
            split["num_policy_val"],
            split["seed"],
            f"`{args.split_root / str(split['scene']) / 'policy_split.json'}`",
        ]
        for split in splits
    ]
    md = [
        "# ECSR Full-Train Policy Split",
        "",
        "This is the deterministic fitting/policy-val split for Phase-C/D",
        "candidate acceptance. It is generated from the scene loader's full train",
        "camera list. Held-out test views are not used for candidate generation,",
        "strength selection, crop selection, rollback, or acceptance.",
        "",
        "- split scope: `full_train_scene_loader`",
        "- held-out test usage: `none`",
        "",
        md_table(["scene", "train views", "fitting", "policy-val", "seed", "split JSON"], rows),
        "",
        "The earlier cached-view split is retained only for smoke tests.",
        "",
    ]
    if missing:
        md.extend(["## Missing", "", ", ".join(missing), ""])
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.split_root / 'full_train_policy_splits_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

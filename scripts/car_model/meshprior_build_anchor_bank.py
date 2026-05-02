"""Build a train-only retrieval anchor bank for MeshPrior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.retrieval_deformation import AnchorBank, build_anchor_bank
from ss3dm_prior.meshprior.synthetic_damage import make_box_mesh


def _records_from_object_index(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset

    dataset = SPCarObjectDataset(path, splits=("train",), return_observed=False, return_queries=False, return_normals=False)
    records: list[dict[str, Any]] = []
    for idx in range(min(len(dataset), max_records)):
        item = dataset[idx]
        records.append(
            {
                "object_id": item["object_id"],
                "split": item["split"],
                "points": np.asarray(item["clean_points_object"], dtype=np.float32),
            }
        )
    return records


def _synthetic_records() -> list[dict[str, Any]]:
    vertices, _ = make_box_mesh()
    return [{"object_id": "synthetic_train_box", "split": "train", "points": vertices}]


def run(args: argparse.Namespace) -> dict[str, object]:
    object_index = Path(args.object_index) if args.object_index else None
    if args.synthetic_smoke or object_index is None or not object_index.is_file():
        records = _synthetic_records()
        source = "synthetic_smoke"
    else:
        records = _records_from_object_index(object_index, max_records=args.max_anchors)
        source = str(object_index)
    bank = build_anchor_bank(records, points_per_anchor=args.points_per_anchor, max_anchors=args.max_anchors, seed=args.seed)
    out = Path(args.output)
    bank.metadata.update({"source": source})
    bank.to_npz(out)
    summary = {
        "anchor_bank": str(out),
        "source": source,
        "anchor_count": len(bank.object_ids),
        "points_per_anchor": int(bank.points.shape[1]),
        "train_only": all(split == "train" for split in bank.splits),
        "object_ids": bank.object_ids[:10],
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MeshPrior retrieval anchor bank.")
    parser.add_argument("--object_index", default=str(REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_anchors", type=int, default=128)
    parser.add_argument("--points_per_anchor", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic_smoke", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

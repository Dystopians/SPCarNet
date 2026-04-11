"""Validate teacher patch caches and render sample visualizations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.patch_types import load_patch_npz


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check teacher patch cache statistics and render sample PNGs."
    )
    parser.add_argument("--patch_cache_dir", required=True, help="Root directory of teacher patch caches.")
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=3,
        help="Number of patch visualizations to save.",
    )
    return parser


def _to_scalar(value: np.ndarray | object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _render_patch_png(patch_path: Path, png_path: Path) -> None:
    patch = load_patch_npz(patch_path)
    observed_points = np.asarray(patch["observed_points"], dtype=np.float32)
    clean_points = np.asarray(patch["clean_points"], dtype=np.float32)

    fig = plt.figure(figsize=(12, 4))
    titles = ["Observed Patch", "Clean Patch", "Overlay"]
    point_sets = [
        (observed_points, None),
        (clean_points, None),
        (observed_points, clean_points),
    ]

    for idx, (points_a, points_b) in enumerate(point_sets, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.set_title(titles[idx - 1])
        if len(points_a):
            ax.scatter(points_a[:, 0], points_a[:, 1], points_a[:, 2], s=2, alpha=0.6, c="tab:blue")
        if points_b is not None and len(points_b):
            ax.scatter(points_b[:, 0], points_b[:, 1], points_b[:, 2], s=2, alpha=0.4, c="tab:orange")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.suptitle(str(_to_scalar(patch["patch_id"])))
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = make_argparser().parse_args()
    patch_cache_dir = Path(args.patch_cache_dir).expanduser().resolve()
    index_path = patch_cache_dir / "patch_index.jsonl"
    records = read_patch_index_jsonl(index_path)
    if not records:
        print(f"no patch index records found in {index_path}")
        return 1

    town_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    clean_counts: list[int] = []
    observed_counts: list[int] = []
    observed_raw_counts: list[int] = []

    for record in records:
        town_counts[str(record["town_id"])] += 1
        sequence_counts[str(record["sequence_id"])] += 1
        clean_counts.append(int(record["num_clean_points"]))
        observed_counts.append(int(record["num_observed_points"]))
        observed_raw_counts.append(int(record["num_observed_points_raw"]))

    print(f"patch_count_total: {len(records)}")
    print("patch_count_by_town:")
    for town_id in sorted(town_counts):
        print(f"  - {town_id}: {town_counts[town_id]}")
    print("patch_count_by_sequence:")
    for sequence_id in sorted(sequence_counts):
        print(f"  - {sequence_id}: {sequence_counts[sequence_id]}")
    print(
        "clean_point_count_stats: "
        f"min={min(clean_counts)} max={max(clean_counts)} mean={float(np.mean(clean_counts)):.2f}"
    )
    print(
        "observed_point_count_stats: "
        f"min={min(observed_counts)} max={max(observed_counts)} mean={float(np.mean(observed_counts)):.2f}"
    )
    print(
        "observed_raw_point_count_stats: "
        f"min={min(observed_raw_counts)} max={max(observed_raw_counts)} "
        f"mean={float(np.mean(observed_raw_counts)):.2f}"
    )

    viz_dir = patch_cache_dir / "visualizations"
    for render_idx, record in enumerate(records[: int(args.num_visualizations)]):
        patch_path = Path(record["patch_file"])
        png_path = viz_dir / f"{record['patch_id']}.png"
        _render_patch_png(patch_path, png_path)
        print(f"visualization_png: {png_path}")

    summary_path = patch_cache_dir / "check_summary.json"
    summary = {
        "patch_count_total": len(records),
        "patch_count_by_town": dict(sorted(town_counts.items())),
        "patch_count_by_sequence": dict(sorted(sequence_counts.items())),
        "clean_point_count_stats": {
            "min": int(min(clean_counts)),
            "max": int(max(clean_counts)),
            "mean": float(np.mean(clean_counts)),
        },
        "observed_point_count_stats": {
            "min": int(min(observed_counts)),
            "max": int(max(observed_counts)),
            "mean": float(np.mean(observed_counts)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary_json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

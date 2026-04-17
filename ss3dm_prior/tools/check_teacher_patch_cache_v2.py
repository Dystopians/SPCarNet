"""Validate teacher patch cache v2 and render visibility query overlays."""

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
        description="Check teacher patch cache v2 statistics and render visibility/free-space PNGs."
    )
    parser.add_argument("--patch_cache_dir", required=True, help="Root directory of teacher patch cache v2.")
    parser.add_argument("--num_visualizations", type=int, default=3, help="Number of visualizations to save.")
    parser.add_argument("--patch_id", default=None, help="Optional specific patch id to visualize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampled visualizations.")
    return parser


def _to_scalar(value: np.ndarray | object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _scatter_points(ax, points: np.ndarray, *, color: str, label: str, size: float, alpha: float) -> None:
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, alpha=alpha, c=color, label=label)


def _render_patch_png_v2(patch_path: Path, png_path: Path) -> None:
    patch = load_patch_npz(patch_path)
    observed_points = np.asarray(patch["observed_points"], dtype=np.float32)
    clean_points = np.asarray(patch["clean_points"], dtype=np.float32)
    surface_query_points = np.asarray(patch["surface_query_points"], dtype=np.float32)
    free_query_points = np.asarray(patch["free_query_points"], dtype=np.float32)
    unknown_query_points = np.asarray(patch["unknown_query_points"], dtype=np.float32)

    fig = plt.figure(figsize=(16, 4))
    axes = [fig.add_subplot(1, 4, idx + 1, projection="3d") for idx in range(4)]
    titles = [
        "Observed Patch",
        "Clean Patch",
        "Support Overlay",
        "Query Labels",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _scatter_points(axes[0], observed_points, color="tab:blue", label="observed", size=2.0, alpha=0.7)
    _scatter_points(axes[1], clean_points, color="tab:orange", label="clean", size=2.0, alpha=0.6)

    _scatter_points(axes[2], clean_points, color="tab:orange", label="clean", size=1.5, alpha=0.15)
    _scatter_points(axes[2], observed_points, color="tab:blue", label="observed", size=2.0, alpha=0.4)
    _scatter_points(axes[2], surface_query_points, color="tab:green", label="surface_q", size=3.0, alpha=0.7)
    _scatter_points(axes[2], free_query_points, color="tab:red", label="free_q", size=3.0, alpha=0.7)
    _scatter_points(axes[2], unknown_query_points, color="tab:gray", label="unknown_q", size=3.0, alpha=0.5)

    _scatter_points(axes[3], surface_query_points, color="tab:green", label="surface=1", size=3.0, alpha=0.75)
    _scatter_points(axes[3], free_query_points, color="tab:red", label="free=0", size=3.0, alpha=0.75)
    _scatter_points(axes[3], unknown_query_points, color="tab:gray", label="unknown(ignore)", size=3.0, alpha=0.6)

    for ax in axes[2:]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7)

    title = (
        f"{_to_scalar(patch['patch_id'])} | "
        f"visible={float(_to_scalar(patch['visible_surface_fraction'])):.3f} "
        f"free={float(_to_scalar(patch['free_space_fraction'])):.3f} "
        f"unknown={float(_to_scalar(patch['unknown_fraction'])):.3f} "
        f"difficulty={float(_to_scalar(patch['intrinsic_patch_difficulty_target'])):.3f}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = make_argparser().parse_args()
    patch_cache_dir = Path(args.patch_cache_dir).expanduser().resolve()
    records = read_patch_index_jsonl(patch_cache_dir / "patch_index.jsonl")
    if not records:
        print(f"no patch index records found in {patch_cache_dir / 'patch_index.jsonl'}")
        return 1

    town_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    difficulty_values: list[float] = []
    visible_values: list[float] = []
    free_values: list[float] = []
    unknown_values: list[float] = []

    for record in records:
        town_counts[str(record["town_id"])] += 1
        sequence_counts[str(record["sequence_id"])] += 1
        difficulty_values.append(float(record.get("intrinsic_patch_difficulty_target", 0.0)))
        visible_values.append(float(record.get("visible_surface_fraction", 0.0)))
        free_values.append(float(record.get("free_space_fraction", 0.0)))
        unknown_values.append(float(record.get("unknown_fraction", 0.0)))

    print(f"patch_count_total: {len(records)}")
    print(f"patch_cache_format_versions: {sorted({int(record.get('patch_cache_format_version', 1)) for record in records})}")
    print("patch_count_by_town:")
    for town_id in sorted(town_counts):
        print(f"  - {town_id}: {town_counts[town_id]}")
    print("patch_count_by_sequence:")
    for sequence_id in sorted(sequence_counts):
        print(f"  - {sequence_id}: {sequence_counts[sequence_id]}")
    print(f"visible_surface_fraction_mean: {float(np.mean(visible_values)):.4f}")
    print(f"free_space_fraction_mean: {float(np.mean(free_values)):.4f}")
    print(f"unknown_fraction_mean: {float(np.mean(unknown_values)):.4f}")
    print(f"intrinsic_patch_difficulty_mean: {float(np.mean(difficulty_values)):.4f}")

    viz_dir = patch_cache_dir / "visualizations_v2"
    if args.patch_id is not None:
        selected_records = [record for record in records if str(record["patch_id"]) == str(args.patch_id)]
    else:
        rng = np.random.default_rng(int(args.seed))
        sample_count = min(int(args.num_visualizations), len(records))
        selected_indices = np.sort(rng.choice(len(records), size=sample_count, replace=False))
        selected_records = [records[idx] for idx in selected_indices]
    if not selected_records:
        print("no matching records selected for visualization")
        return 1

    for record in selected_records[: int(args.num_visualizations)]:
        patch_path = Path(record["patch_file"])
        png_path = viz_dir / f"{record['patch_id']}.png"
        _render_patch_png_v2(patch_path, png_path)
        print(f"visualization_png: {png_path}")

    summary = {
        "patch_count_total": len(records),
        "patch_cache_format_versions": sorted({int(record.get("patch_cache_format_version", 1)) for record in records}),
        "patch_count_by_town": dict(sorted(town_counts.items())),
        "patch_count_by_sequence": dict(sorted(sequence_counts.items())),
        "visible_surface_fraction_mean": float(np.mean(visible_values)),
        "free_space_fraction_mean": float(np.mean(free_values)),
        "unknown_fraction_mean": float(np.mean(unknown_values)),
        "intrinsic_patch_difficulty_mean": float(np.mean(difficulty_values)),
    }
    summary_path = patch_cache_dir / "check_summary_v2.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary_json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate teacher patch cache v3 and render semantic/multi-scale PNGs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ss3dm_prior.data.patch_index import read_patch_index_jsonl
from ss3dm_prior.data.patch_types import load_patch_npz


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check teacher patch cache v3 statistics and render semantic PNGs.")
    parser.add_argument("--patch_cache_dir", required=True, help="Root directory of teacher patch cache v3.")
    parser.add_argument("--num_visualizations", type=int, default=3, help="Number of semantic visualizations to save.")
    parser.add_argument("--patch_id", default=None, help="Optional specific patch id to visualize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampled visualizations.")
    return parser


def _scatter_points(ax, points: np.ndarray, *, color: str, label: str, size: float, alpha: float) -> None:
    if len(points) == 0:
        return
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, alpha=alpha, c=color, label=label)


def _render_patch_png_v3(patch_path: Path, png_path: Path) -> None:
    patch = load_patch_npz(patch_path)
    observed_points = np.asarray(patch["observed_points"], dtype=np.float32)
    visible_clean_points = np.asarray(patch.get("visible_clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    hidden_clean_points = np.asarray(patch.get("hidden_clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    free_hard_negatives = np.asarray(
        patch.get("free_space_query_hard_negatives", np.zeros((0, 3), dtype=np.float32)),
        dtype=np.float32,
    )
    free_query_points = np.asarray(patch["free_query_points"], dtype=np.float32)

    fig = plt.figure(figsize=(16, 4))
    axes = [fig.add_subplot(1, 4, idx + 1, projection="3d") for idx in range(4)]
    titles = [
        "Visible vs Hidden Clean",
        "Observed vs Visible Clean",
        "Free-Space Hard Negatives",
        "Semantic Overview",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _scatter_points(axes[0], visible_clean_points, color="tab:green", label="visible_clean", size=2.0, alpha=0.75)
    _scatter_points(axes[0], hidden_clean_points, color="tab:orange", label="hidden_clean", size=2.0, alpha=0.65)

    _scatter_points(axes[1], observed_points, color="tab:blue", label="observed", size=2.0, alpha=0.6)
    _scatter_points(axes[1], visible_clean_points, color="tab:green", label="visible_clean", size=2.0, alpha=0.5)

    _scatter_points(axes[2], free_query_points, color="tab:red", label="free_query", size=1.5, alpha=0.12)
    _scatter_points(axes[2], free_hard_negatives, color="black", label="hard_negative", size=6.0, alpha=0.85)

    _scatter_points(axes[3], observed_points, color="tab:blue", label="observed", size=1.5, alpha=0.25)
    _scatter_points(axes[3], visible_clean_points, color="tab:green", label="visible_clean", size=2.0, alpha=0.7)
    _scatter_points(axes[3], hidden_clean_points, color="tab:orange", label="hidden_clean", size=2.0, alpha=0.55)
    _scatter_points(axes[3], free_hard_negatives, color="black", label="free_hard_negative", size=6.0, alpha=0.8)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"{patch['patch_id'].item()} | scale={int(patch.get('scale_id', np.asarray(0)).item())} "
        f"| radius={float(patch['patch_radius_m'].item()):.2f} "
        f"| visible_support={float(patch.get('visible_support_fraction', np.asarray(0.0)).item()):.3f} "
        f"| hidden_surface={float(patch.get('hidden_surface_fraction', np.asarray(0.0)).item()):.3f}"
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _render_multiscale_png(patch_paths: list[Path], png_path: Path) -> None:
    fig = plt.figure(figsize=(5 * max(1, len(patch_paths)), 4))
    axes = [fig.add_subplot(1, len(patch_paths), idx + 1, projection="3d") for idx in range(len(patch_paths))]
    for ax, patch_path in zip(axes, patch_paths):
        patch = load_patch_npz(patch_path)
        visible_clean_points = np.asarray(patch.get("visible_clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        hidden_clean_points = np.asarray(patch.get("hidden_clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        _scatter_points(ax, visible_clean_points, color="tab:green", label="visible", size=2.0, alpha=0.7)
        _scatter_points(ax, hidden_clean_points, color="tab:orange", label="hidden", size=2.0, alpha=0.55)
        ax.set_title(
            f"scale={int(patch.get('scale_id', np.asarray(0)).item())}\n"
            f"radius={float(patch['patch_radius_m'].item()):.2f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
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
    scale_counts: Counter[int] = Counter()
    visible_support_values: list[float] = []
    hidden_surface_values: list[float] = []
    hard_negative_values: list[float] = []
    grouped_by_tile: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        town_counts[str(record["town_id"])] += 1
        scale_counts[int(record.get("scale_id", 0))] += 1
        visible_support_values.append(float(record.get("visible_support_fraction", 0.0)))
        hidden_surface_values.append(float(record.get("hidden_surface_fraction", 0.0)))
        hard_negative_values.append(float(record.get("free_space_hard_negative_count", 0)))
        grouped_by_tile[(str(record["sequence_id"]), int(record["tile_id"]))].append(record)

    print(f"patch_count_total: {len(records)}")
    print(f"patch_cache_format_versions: {sorted({int(record.get('patch_cache_format_version', 1)) for record in records})}")
    print("patch_count_by_town:")
    for town_id in sorted(town_counts):
        print(f"  - {town_id}: {town_counts[town_id]}")
    print("patch_count_by_scale:")
    for scale_id in sorted(scale_counts):
        print(f"  - scale_{scale_id}: {scale_counts[scale_id]}")
    print(f"visible_support_fraction_mean: {float(np.mean(visible_support_values)):.4f}")
    print(f"hidden_surface_fraction_mean: {float(np.mean(hidden_surface_values)):.4f}")
    print(f"free_space_hard_negative_count_mean: {float(np.mean(hard_negative_values)):.4f}")

    viz_dir = patch_cache_dir / "visualizations_v3"
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

    written_pngs: list[Path] = []
    for record in selected_records[: int(args.num_visualizations)]:
        patch_path = Path(record["patch_file"])
        png_path = viz_dir / f"{record['patch_id']}__semantics.png"
        _render_patch_png_v3(patch_path, png_path)
        written_pngs.append(png_path)
        print(f"visualization_png: {png_path}")

    multiscale_groups = [
        (key, sorted(group, key=lambda item: int(item.get("scale_id", 0))))
        for key, group in grouped_by_tile.items()
        if len(group) > 1
    ]
    for (sequence_id, tile_id), group in multiscale_groups[: max(1, int(args.num_visualizations))]:
        patch_paths = [Path(item["patch_file"]) for item in group]
        png_path = viz_dir / f"{sequence_id}__tile_{tile_id:06d}__multiscale.png"
        _render_multiscale_png(patch_paths, png_path)
        written_pngs.append(png_path)
        print(f"visualization_png: {png_path}")

    summary = {
        "patch_count_total": len(records),
        "patch_cache_format_versions": sorted({int(record.get("patch_cache_format_version", 1)) for record in records}),
        "patch_count_by_town": dict(sorted(town_counts.items())),
        "patch_count_by_scale": {str(key): value for key, value in sorted(scale_counts.items())},
        "visible_support_fraction_mean": float(np.mean(visible_support_values)),
        "hidden_surface_fraction_mean": float(np.mean(hidden_surface_values)),
        "free_space_hard_negative_count_mean": float(np.mean(hard_negative_values)),
        "written_pngs": [str(path) for path in written_pngs],
    }
    summary_path = patch_cache_dir / "check_summary_v3.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary_json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

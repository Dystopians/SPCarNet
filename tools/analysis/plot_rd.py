#!/usr/bin/env python
"""Unified Stage-4 ECR rate-distortion scatter plot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


REPO = Path("/data/peilincai/mesh-splatting")
G1 = Path("/data/peilincai/gems_stage1")
OUT_DIR = REPO / "RESULTS" / "figures" / "ecr_paper"

SCENES = ["garden", "bicycle", "kitchen"]
L5_PATH = G1 / "analysis" / "final_stack" / "l5_pareto.json"
MATCHED_3DGS_PATH = G1 / "analysis" / "final_stack" / "e07_matched_total_3dgs.json"

INK = "#101418"
MUTED = "#5d6670"
GRID = "#d9dee3"
SURFACE = "#fbfbfa"
C_L5 = "#2f6f9f"
C_L6 = "#1b7f5a"
C_3DGS = "#111111"
C_DIFIX = "#b13b73"
C_BASE = "#7a8087"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
})


def skip(scene: str, series: str, reason: str) -> None:
    print(f"[skip] {scene} {series}: {reason}")


def load_json(path: Path, scene: str, series: str) -> dict[str, Any] | None:
    if not path.exists():
        skip(scene, series, f"file not found: {path}")
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - defensive for partial jobs.
        skip(scene, series, f"could not read json: {path}: {exc}")
        return None
    if not isinstance(data, dict):
        skip(scene, series, f"json root is not an object: {path}")
        return None
    return data


def get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(".".join(keys))
        cur = cur[key]
    return cur


def add_point(
    points: list[tuple[float, float, str]],
    scene: str,
    series: str,
    path: Path,
    x: Any,
    y: Any,
    label: str,
) -> None:
    try:
        points.append((float(x), float(y), label))
    except (TypeError, ValueError) as exc:
        skip(scene, series, f"non-numeric value in {path}: {exc}")


def efficient_subset(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    efficient = []
    for i, (x_i, y_i, label_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j, _) in enumerate(points):
            if i == j:
                continue
            if x_j <= x_i and y_j >= y_i and (x_j < x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            efficient.append((x_i, y_i, label_i))
    return sorted(efficient, key=lambda p: (p[0], p[1]))


def metric_point(path: Path, scene: str, series: str, x_keys: tuple[str, ...]) -> tuple[float, float] | None:
    data = load_json(path, scene, series)
    if data is None:
        return None
    try:
        x = get_nested(data, x_keys)
        y = get_nested(data, ("rendering", "mean", "psnr"))
        return float(x), float(y)
    except (KeyError, TypeError, ValueError) as exc:
        skip(scene, series, f"missing or invalid key in {path}: {exc}")
        return None


def annotate(ax, x: float, y: float, text: str, dx: int = 5, dy: int = 5) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.5,
        color=INK,
    )


def plot_scene(ax, scene: str, all_y: list[float]) -> None:
    l5_data = load_json(L5_PATH, scene, "L5 variants")
    if l5_data is not None:
        scene_rows = l5_data.get(scene)
        if not isinstance(scene_rows, dict):
            skip(scene, "L5 variants", f"scene not found in {L5_PATH}")
        else:
            l5_points: list[tuple[float, float, str]] = []
            for label, row in scene_rows.items():
                if not isinstance(row, dict):
                    skip(scene, f"L5 {label}", f"entry is not an object in {L5_PATH}")
                    continue
                try:
                    add_point(
                        l5_points,
                        scene,
                        f"L5 {label}",
                        L5_PATH,
                        row["total_mb_raw"],
                        row["psnr"],
                        str(label),
                    )
                except KeyError as exc:
                    skip(scene, f"L5 {label}", f"missing key in {L5_PATH}: {exc}")
            if l5_points:
                xs = [p[0] for p in l5_points]
                ys = [p[1] for p in l5_points]
                all_y.extend(ys)
                ax.scatter(xs, ys, s=38, color=C_L5, marker="o", edgecolor="white", linewidth=0.7, zorder=4)
                for x, y, label in l5_points:
                    annotate(ax, x, y, label, dx=4, dy=4)
                front = efficient_subset(l5_points)
                if len(front) >= 2:
                    ax.plot([p[0] for p in front], [p[1] for p in front], color=C_L5, lw=1.5, zorder=3)

    l6_path = G1 / "eval" / f"final_{scene}_B50_v1" / "metrics.json"
    l6 = metric_point(l6_path, scene, "L6 compact", ("cost", "total_artifact_mb"))
    if l6 is not None:
        x, y = l6
        all_y.append(y)
        ax.scatter([x], [y], s=54, color=C_L6, marker="s", edgecolor="white", linewidth=0.7, zorder=5)
        annotate(ax, x, y, "L6 compact", dx=5, dy=-10)

    matched = load_json(MATCHED_3DGS_PATH, scene, "3DGS-30k")
    if matched is not None:
        try:
            row = get_nested(matched, ("scenes", scene, "3dgs_matched_total"))
            x, y = float(row["disk_mb"]), float(row["psnr"])
            all_y.append(y)
            ax.scatter([x], [y], s=82, color=C_3DGS, marker="*", edgecolor="white", linewidth=0.7, zorder=6)
            annotate(ax, x, y, "3DGS-30k", dx=5, dy=5)
        except (KeyError, TypeError, ValueError) as exc:
            skip(scene, "3DGS-30k", f"missing or invalid key in {MATCHED_3DGS_PATH}: {exc}")

    difix_path = G1 / "analysis" / "difix_cell" / f"difix_{scene}.json"
    l4_path = G1 / "eval" / f"l4_{scene}_cleanfixed30k_routed_v1" / "metrics.json"
    difix = load_json(difix_path, scene, "Difix")
    l4_for_difix = load_json(l4_path, scene, "Difix x-size")
    if difix is not None and l4_for_difix is not None:
        try:
            x = float(get_nested(l4_for_difix, ("cost", "disk_mb")))
            y = float(get_nested(difix, ("difix", "psnr")))
            all_y.append(y)
            ax.scatter([x], [y], s=62, color=C_DIFIX, marker="X", edgecolor="white", linewidth=0.7, zorder=6)
            annotate(ax, x, y, "base+Difix (no cache)", dx=6, dy=5)
        except (KeyError, TypeError, ValueError) as exc:
            skip(scene, "Difix", f"missing or invalid key in {difix_path} or {l4_path}: {exc}")

    base_path = G1 / "eval" / f"{scene}_cleanfixed30k_v1" / "metrics.json"
    base = metric_point(base_path, scene, "base anchor", ("cost", "disk_mb"))
    if base is not None:
        x, y = base
        all_y.append(y)
        ax.scatter([x], [y], s=56, color=C_BASE, marker="D", edgecolor="white", linewidth=0.7, zorder=5)
        annotate(ax, x, y, "base (ckpt only)", dx=5, dy=-10)

    ax.set_title(scene)
    ax.set_xlabel("Total artifact MB")
    ax.grid(axis="both")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), sharey=True)
    all_y: list[float] = []

    for ax, scene in zip(axes, SCENES):
        plot_scene(ax, scene, all_y)

    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        pad = max((y_max - y_min) * 0.12, 0.5)
        axes[0].set_ylim(y_min - pad, y_max + pad)

    axes[0].set_ylabel("PSNR (dB)")
    legend_handles = [
        Line2D([], [], color=C_L5, marker="o", linestyle="-", lw=1.5, markersize=6, label="L5 variants + Pareto front"),
        Line2D([], [], color=C_L6, marker="s", linestyle="None", markersize=6, label="L6 compact"),
        Line2D([], [], color=C_3DGS, marker="*", linestyle="None", markersize=9, label="3DGS-30k"),
        Line2D([], [], color=C_DIFIX, marker="X", linestyle="None", markersize=7, label="base+Difix (no cache)"),
        Line2D([], [], color=C_BASE, marker="D", linestyle="None", markersize=6, label="base (ckpt only)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Stage-4 ECR rate-distortion scatter", y=0.98, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    pdf = OUT_DIR / "rd_master.pdf"
    png = OUT_DIR / "rd_master.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()

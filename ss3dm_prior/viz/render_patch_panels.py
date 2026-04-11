"""Static patch-level visualization panels for denoising runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _scatter_3d(ax, points: np.ndarray, *, color=None, cmap=None, title: str = "", vmin=None, vmax=None) -> None:
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=color, cmap=cmap, alpha=0.75, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect((1, 1, 1))


def _shared_limits(*point_sets: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    valid = [np.asarray(points, dtype=np.float32) for points in point_sets if len(points)]
    if not valid:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
    merged = np.concatenate(valid, axis=0)
    mins = merged.min(axis=0)
    maxs = merged.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_extent = max(float(np.max(maxs - mins)) * 0.55, 1e-3)
    return (
        (float(center[0] - half_extent), float(center[0] + half_extent)),
        (float(center[1] - half_extent), float(center[1] + half_extent)),
        (float(center[2] - half_extent), float(center[2] + half_extent)),
    )


def _scatter_projection(
    ax,
    points: np.ndarray,
    *,
    dims: tuple[int, int],
    title: str = "",
    color: str = "tab:blue",
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    if len(points):
        ax.scatter(points[:, dims[0]], points[:, dims[1]], s=4, c=color, alpha=0.75, linewidths=0)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


def render_patch_triptych(
    *,
    corrupted_points: np.ndarray,
    recon_points: np.ndarray,
    clean_points: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limits = _shared_limits(corrupted_points, recon_points, clean_points)
    axis_pairs = [
        ((0, 1), "XY"),
        ((0, 2), "XZ"),
        ((1, 2), "YZ"),
    ]
    point_sets = [
        ("Input Corrupted", np.asarray(corrupted_points, dtype=np.float32), "tab:orange"),
        ("Model Output", np.asarray(recon_points, dtype=np.float32), "tab:green"),
        ("GT Clean", np.asarray(clean_points, dtype=np.float32), "tab:purple"),
    ]

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.2, hspace=0.25)

    for row_idx, (dims, row_name) in enumerate(axis_pairs):
        for col_idx, (col_name, points, color) in enumerate(point_sets):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            title = col_name if row_idx == 0 else ""
            _scatter_projection(
                ax,
                points,
                dims=dims,
                title=title,
                color=color,
                xlim=limits[dims[0]],
                ylim=limits[dims[1]],
            )
            ax.set_xlabel(["x", "x", "y"][row_idx])
            ax.set_ylabel(["y", "z", "z"][row_idx])
            ax.text(0.02, 0.98, row_name, transform=ax.transAxes, va="top", ha="left", fontsize=9)

    text_ax = fig.add_subplot(grid[:, 3])
    text_ax.axis("off")
    text_ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_patch_denoise_panel(
    *,
    observed_points: np.ndarray,
    corrupted_points: np.ndarray,
    recon_points: np.ndarray,
    clean_points: np.ndarray,
    defect_scores: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    axes = [fig.add_subplot(2, 3, idx + 1, projection="3d" if idx < 5 else None) for idx in range(6)]
    _scatter_3d(axes[0], observed_points, color="tab:blue", title="Observed")
    _scatter_3d(axes[1], corrupted_points, color="tab:orange", title="Corrupted")
    _scatter_3d(axes[2], recon_points, color="tab:green", title="Reconstructed")
    _scatter_3d(axes[3], clean_points, color="tab:purple", title="Clean GT")
    defect_vmax = float(np.max(defect_scores)) if len(defect_scores) else 1.0
    _scatter_3d(
        axes[4],
        corrupted_points,
        color=defect_scores,
        cmap="inferno",
        title="Predicted Defect Heatmap",
        vmin=0.0,
        vmax=max(defect_vmax, 1e-6),
    )
    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_retrieval_gallery(
    *,
    query_corrupted_points: np.ndarray,
    target_clean_points: np.ndarray,
    nearest_clean_points: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4.5))
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax4 = fig.add_subplot(1, 4, 4)
    _scatter_3d(ax1, query_corrupted_points, color="tab:orange", title="Query Corrupted")
    _scatter_3d(ax2, target_clean_points, color="tab:purple", title="Target Clean")
    _scatter_3d(ax3, nearest_clean_points, color="tab:green", title="Nearest Clean")
    ax4.axis("off")
    ax4.text(0.02, 0.98, "\n".join(info_lines), va="top", ha="left", fontsize=10, family="monospace")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

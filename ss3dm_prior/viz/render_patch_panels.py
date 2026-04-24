"""Static patch-level visualization panels for denoising runs."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib emits two recurring UserWarnings from our 3D + info-text layouts:
#   1. "This figure includes Axes that are not compatible with tight_layout"
#      — a known limitation when mixing mpl_toolkits.mplot3d axes with 2D axes.
#   2. "No artists with labels found to put in legend" — fires even after the
#      `if handles and labels` guard on some matplotlib versions when no
#      plotted artist ended up with a visible label.
# Both are cosmetic; suppress them at the module level so training logs stay
# readable instead of being drowned in per-panel warnings.
warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="No artists with labels found to put in legend.*",
    category=UserWarning,
)


def _scatter_3d(ax, points: np.ndarray, *, color=None, cmap=None, title: str = "", vmin=None, vmax=None) -> None:
    if len(points):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=color, cmap=cmap, alpha=0.75, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect((1, 1, 1))


def _scatter_labeled_queries(
    ax,
    *,
    clean_points: np.ndarray,
    observed_points: np.ndarray | None = None,
    surface_query_points: np.ndarray | None = None,
    free_query_points: np.ndarray | None = None,
    unknown_query_points: np.ndarray | None = None,
    title: str,
) -> None:
    if clean_points is not None and len(clean_points):
        ax.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], s=1.5, c="tab:purple", alpha=0.12)
    if observed_points is not None and len(observed_points):
        ax.scatter(observed_points[:, 0], observed_points[:, 1], observed_points[:, 2], s=2, c="tab:blue", alpha=0.35)
    if surface_query_points is not None and len(surface_query_points):
        ax.scatter(surface_query_points[:, 0], surface_query_points[:, 1], surface_query_points[:, 2], s=3, c="tab:green", alpha=0.8)
    if free_query_points is not None and len(free_query_points):
        ax.scatter(free_query_points[:, 0], free_query_points[:, 1], free_query_points[:, 2], s=3, c="tab:red", alpha=0.8)
    if unknown_query_points is not None and len(unknown_query_points):
        ax.scatter(unknown_query_points[:, 0], unknown_query_points[:, 1], unknown_query_points[:, 2], s=3, c="tab:gray", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect((1, 1, 1))


def _text_panel(ax, info_lines: list[str]) -> None:
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )


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
    _text_panel(axes[5], info_lines)
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
    _text_panel(ax4, info_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_visibility_panel(
    *,
    clean_points: np.ndarray,
    observed_points: np.ndarray,
    surface_query_points: np.ndarray,
    free_query_points: np.ndarray,
    unknown_query_points: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    axes = [fig.add_subplot(2, 3, idx + 1, projection="3d" if idx < 5 else None) for idx in range(6)]
    _scatter_3d(axes[0], clean_points, color="tab:purple", title="Clean Patch")
    _scatter_3d(axes[1], observed_points, color="tab:blue", title="Observed Patch")
    _scatter_labeled_queries(
        axes[2],
        clean_points=clean_points,
        observed_points=observed_points,
        surface_query_points=surface_query_points,
        title="Visible Surface Queries",
    )
    _scatter_labeled_queries(
        axes[3],
        clean_points=clean_points,
        observed_points=observed_points,
        free_query_points=free_query_points,
        title="Free-Space Queries",
    )
    _scatter_labeled_queries(
        axes[4],
        clean_points=clean_points,
        observed_points=observed_points,
        unknown_query_points=unknown_query_points,
        title="Unknown / Ignore Queries",
    )
    _text_panel(axes[5], info_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_visible_vs_hidden_panel(
    *,
    observed_points: np.ndarray,
    clean_points: np.ndarray,
    visible_clean_points: np.ndarray,
    hidden_clean_points: np.ndarray,
    recon_points: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    axes = [fig.add_subplot(2, 3, idx + 1, projection="3d" if idx < 5 else None) for idx in range(6)]
    _scatter_3d(axes[0], observed_points, color="tab:blue", title="Observed")
    _scatter_3d(axes[1], visible_clean_points, color="tab:green", title="Visible Clean")
    _scatter_3d(axes[2], hidden_clean_points, color="tab:red", title="Hidden Clean")
    _scatter_3d(axes[3], recon_points, color="tab:orange", title="Reconstruction")
    overlay = axes[4]
    if len(clean_points):
        overlay.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], s=1.0, c="tab:purple", alpha=0.08)
    if len(visible_clean_points):
        overlay.scatter(
            visible_clean_points[:, 0],
            visible_clean_points[:, 1],
            visible_clean_points[:, 2],
            s=3,
            c="tab:green",
            alpha=0.8,
            label="visible",
        )
    if len(hidden_clean_points):
        overlay.scatter(
            hidden_clean_points[:, 0],
            hidden_clean_points[:, 1],
            hidden_clean_points[:, 2],
            s=3,
            c="tab:red",
            alpha=0.8,
            label="hidden",
        )
    if len(recon_points):
        overlay.scatter(
            recon_points[:, 0],
            recon_points[:, 1],
            recon_points[:, 2],
            s=2,
            c="tab:orange",
            alpha=0.35,
            label="recon",
        )
    overlay.set_title("Visible / Hidden / Recon Overlay")
    overlay.set_xlabel("x")
    overlay.set_ylabel("y")
    overlay.set_zlabel("z")
    overlay.view_init(elev=25, azim=45)
    overlay.set_box_aspect((1, 1, 1))
    handles, labels = overlay.get_legend_handles_labels()
    if handles and labels:
        overlay.legend(loc="upper right", fontsize=8)
    _text_panel(axes[5], info_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_free_space_error_panel(
    *,
    corrupted_points: np.ndarray,
    recon_points: np.ndarray,
    clean_points: np.ndarray,
    free_query_points: np.ndarray,
    free_query_violation_scores: np.ndarray | None,
    free_space_hard_negatives: np.ndarray | None,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 8))
    axes = [fig.add_subplot(2, 3, idx + 1, projection="3d" if idx < 5 else None) for idx in range(6)]
    _scatter_3d(axes[0], corrupted_points, color="tab:orange", title="Corrupted")
    _scatter_3d(axes[1], recon_points, color="tab:green", title="Reconstruction")
    _scatter_3d(axes[2], clean_points, color="tab:purple", title="Clean")
    if free_query_violation_scores is None or len(free_query_points) == 0:
        _scatter_3d(axes[3], free_query_points, color="tab:red", title="Free Queries")
    else:
        _scatter_3d(
            axes[3],
            free_query_points,
            color=free_query_violation_scores,
            cmap="inferno",
            title="Free-Space FP Scores",
            vmin=0.0,
            vmax=max(float(np.max(free_query_violation_scores)), 1e-6),
        )
    hard_negatives = (
        np.asarray(free_space_hard_negatives, dtype=np.float32)
        if free_space_hard_negatives is not None
        else np.zeros((0, 3), dtype=np.float32)
    )
    overlay = axes[4]
    if len(recon_points):
        overlay.scatter(recon_points[:, 0], recon_points[:, 1], recon_points[:, 2], s=2, c="tab:green", alpha=0.35)
    if len(free_query_points):
        overlay.scatter(
            free_query_points[:, 0],
            free_query_points[:, 1],
            free_query_points[:, 2],
            s=3,
            c="tab:red",
            alpha=0.5,
            label="free_queries",
        )
    if len(hard_negatives):
        overlay.scatter(
            hard_negatives[:, 0],
            hard_negatives[:, 1],
            hard_negatives[:, 2],
            s=8,
            c="gold",
            alpha=0.9,
            label="hard_negatives",
        )
    overlay.set_title("Free-Space Error Overlay")
    overlay.set_xlabel("x")
    overlay.set_ylabel("y")
    overlay.set_zlabel("z")
    overlay.view_init(elev=25, azim=45)
    overlay.set_box_aspect((1, 1, 1))
    handles, labels = overlay.get_legend_handles_labels()
    if handles and labels:
        overlay.legend(loc="upper right", fontsize=8)
    _text_panel(axes[5], info_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_difficulty_calibration_panel(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    if predicted.size and target.size:
        axes[0].scatter(target, predicted, s=18, c=np.abs(predicted - target), cmap="magma", alpha=0.85)
        axes[0].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", alpha=0.4)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("pred")
    axes[0].set_title("Calibration Scatter")
    axes[0].grid(True, alpha=0.2)

    abs_error = np.abs(predicted - target) if predicted.size and target.size else np.zeros((0,), dtype=np.float32)
    axes[1].hist(abs_error, bins=min(10, max(int(abs_error.size), 1)), color="tab:orange", alpha=0.8)
    axes[1].set_title("Abs Error Histogram")
    axes[1].set_xlabel("|pred-target|")
    axes[1].grid(True, alpha=0.2)

    _text_panel(axes[2], info_lines)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_stochastic_candidate_gallery(
    *,
    corrupted_points: np.ndarray,
    clean_points: np.ndarray,
    hidden_clean_points: np.ndarray,
    candidate_recon_points: list[np.ndarray],
    candidate_info_lines: list[list[str]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_candidates = max(len(candidate_recon_points), 1)
    fig = plt.figure(figsize=(4 * (num_candidates + 1), 8))
    base_ax = fig.add_subplot(2, num_candidates + 1, 1, projection="3d")
    _scatter_3d(base_ax, corrupted_points, color="tab:orange", title="Corrupted")
    hidden_ax = fig.add_subplot(2, num_candidates + 1, num_candidates + 2, projection="3d")
    _scatter_3d(hidden_ax, hidden_clean_points, color="tab:red", title="Hidden Clean")
    for idx, candidate_points in enumerate(candidate_recon_points, start=1):
        ax_top = fig.add_subplot(2, num_candidates + 1, idx + 1, projection="3d")
        _scatter_3d(ax_top, candidate_points, color="tab:green", title=f"Candidate {idx}")
        ax_bottom = fig.add_subplot(2, num_candidates + 1, num_candidates + 2 + idx)
        _text_panel(ax_bottom, candidate_info_lines[idx - 1] if idx - 1 < len(candidate_info_lines) else [])
    overlay_ax = fig.add_subplot(2, num_candidates + 1, num_candidates + 1, projection="3d")
    if len(clean_points):
        overlay_ax.scatter(clean_points[:, 0], clean_points[:, 1], clean_points[:, 2], s=1.5, c="tab:purple", alpha=0.10)
    for candidate_points in candidate_recon_points:
        if len(candidate_points):
            overlay_ax.scatter(candidate_points[:, 0], candidate_points[:, 1], candidate_points[:, 2], s=2, alpha=0.28)
    overlay_ax.set_title("Candidate Overlay")
    overlay_ax.set_xlabel("x")
    overlay_ax.set_ylabel("y")
    overlay_ax.set_zlabel("z")
    overlay_ax.view_init(elev=25, azim=45)
    overlay_ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_hybrid_reconstruction_panel(
    *,
    corrupted_points: np.ndarray,
    recon_points: np.ndarray,
    clean_points: np.ndarray,
    free_query_points: np.ndarray,
    free_query_violation_scores: np.ndarray | None,
    intrinsic_pred: float,
    intrinsic_target: float,
    prototype_summary_lines: list[str],
    info_lines: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 8))
    axes = [fig.add_subplot(2, 3, idx + 1, projection="3d" if idx < 4 else None) for idx in range(6)]
    _scatter_3d(axes[0], corrupted_points, color="tab:orange", title="Corrupted")
    _scatter_3d(axes[1], recon_points, color="tab:green", title="Reconstructed")
    _scatter_3d(axes[2], clean_points, color="tab:purple", title="Clean")
    if free_query_violation_scores is None or len(free_query_points) == 0:
        _scatter_3d(axes[3], free_query_points, color="tab:red", title="Free-Space Queries")
    else:
        _scatter_3d(
            axes[3],
            free_query_points,
            color=free_query_violation_scores,
            cmap="inferno",
            title="Free-Space False Positive Heatmap",
            vmin=0.0,
            vmax=max(float(np.max(free_query_violation_scores)), 1e-6),
        )
    difficulty_ax = axes[4]
    difficulty_ax.bar(["pred", "target"], [intrinsic_pred, intrinsic_target], color=["tab:green", "tab:purple"])
    difficulty_ax.set_ylim(0.0, 1.05)
    difficulty_ax.set_title("Intrinsic Difficulty")
    difficulty_ax.grid(True, alpha=0.2)
    _text_panel(axes[5], [*info_lines, "", *prototype_summary_lines])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_prototype_usage_gallery(
    *,
    prototype_examples: list[dict[str, object]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_examples = max(len(prototype_examples), 1)
    fig = plt.figure(figsize=(4 * max_examples, 6))
    for idx, example in enumerate(prototype_examples, start=1):
        ax = fig.add_subplot(2, max_examples, idx, projection="3d")
        clean_points = np.asarray(example.get("clean_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        _scatter_3d(ax, clean_points, color="tab:purple", title=f"Prototype {example.get('code_index', 'n/a')}")
        info_ax = fig.add_subplot(2, max_examples, max_examples + idx)
        info_lines = [
            f"patch: {example.get('patch_id', 'n/a')}",
            f"code: {example.get('code_index', 'n/a')}",
            f"pred_intr: {example.get('intrinsic_pred', 'n/a')}",
            f"target_intr: {example.get('intrinsic_target', 'n/a')}",
        ]
        _text_panel(info_ax, info_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

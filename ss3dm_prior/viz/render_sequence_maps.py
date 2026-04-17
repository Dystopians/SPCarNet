"""Static sequence-level XY map visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_sequence_improvement_map(
    *,
    patch_centers_world: np.ndarray,
    predicted_scores: np.ndarray,
    actual_gains: np.ndarray,
    sequence_id: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xy = np.asarray(patch_centers_world, dtype=np.float32)[:, :2]
    pred = np.asarray(predicted_scores, dtype=np.float32)
    gain = np.asarray(actual_gains, dtype=np.float32)
    pred_vmin, pred_vmax = float(np.min(pred)), float(np.max(pred))
    gain_vmin, gain_vmax = float(np.min(gain)), float(np.max(gain))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sc1 = axes[0].scatter(xy[:, 0], xy[:, 1], c=pred, cmap="viridis", s=45, vmin=pred_vmin, vmax=pred_vmax)
    sc2 = axes[1].scatter(xy[:, 0], xy[:, 1], c=gain, cmap="coolwarm", s=45, vmin=gain_vmin, vmax=gain_vmax)
    axes[0].set_title(f"{sequence_id} Predicted Difficulty")
    axes[1].set_title(f"{sequence_id} Actual Denoise Gain")
    for ax in axes:
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    fig.colorbar(sc1, ax=axes[0], shrink=0.85)
    fig.colorbar(sc2, ax=axes[1], shrink=0.85)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_sequence_visibility_map(
    *,
    patch_centers_world: np.ndarray,
    visible_surface_fraction: np.ndarray,
    free_space_fraction: np.ndarray,
    intrinsic_targets: np.ndarray,
    actual_gains: np.ndarray,
    sequence_id: str,
    map_title: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xy = np.asarray(patch_centers_world, dtype=np.float32)[:, :2]
    visible = np.asarray(visible_surface_fraction, dtype=np.float32)
    free = np.asarray(free_space_fraction, dtype=np.float32)
    intrinsic = np.asarray(intrinsic_targets, dtype=np.float32)
    gain = np.asarray(actual_gains, dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    entries = [
        (visible, "viridis", "Visible Surface Fraction"),
        (free, "Reds", "Free-Space Fraction"),
        (intrinsic, "magma", "Intrinsic Difficulty Target"),
        (gain, "coolwarm", "Actual Denoise Gain"),
    ]
    for ax, (values, cmap, title) in zip(axes.ravel(), entries):
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=values,
            cmap=cmap,
            s=45,
            vmin=float(np.min(values)) if len(values) else 0.0,
            vmax=float(np.max(values)) if len(values) else 1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle(f"{map_title}: {sequence_id}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

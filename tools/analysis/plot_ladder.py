#!/usr/bin/env python
"""Plot Stage-4 ECR ladder confidence intervals from banked gate JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO = Path("/data/peilincai/mesh-splatting")
G1 = Path("/data/peilincai/gems_stage1")
GATE_DIR = G1 / "analysis" / "e0_pj2026"
OUT_DIR = REPO / "RESULTS" / "figures" / "ecr_paper"

GATES = [
    ("L1 (banked negative)", "l1_gate.json"),
    ("L2 multiband", "l2_gate.json"),
    ("L3 learned fusion", "l3_gate.json"),
    ("L4 routing", "l4_gate.json"),
    ("FINAL vs PJ-2026 floor", "l4_vs_floor.json"),
]

INK = "#101418"
MUTED = "#5d6670"
GRID = "#d9dee3"
SURFACE = "#fbfbfa"
L1_COLOR = "#b84a4a"
RUN_COLOR = "#2f6f9f"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
})


def load_gate(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def metric_arrays(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([row[metric]["mean"] for row in rows], dtype=float)
    ci_lo = np.array([row[metric]["ci_lo"] for row in rows], dtype=float)
    ci_hi = np.array([row[metric]["ci_hi"] for row in rows], dtype=float)
    err = np.vstack((means - ci_lo, ci_hi - means))
    return means, err


def set_metric_limits(ax, means: np.ndarray, err: np.ndarray, floor: float) -> None:
    lo = min(float(np.min(means - err[0])), floor, 0.0)
    hi = max(float(np.max(means + err[1])), floor, 0.0)
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo - 0.08 * span, hi + 0.18 * span)


def draw_panel(ax, rows: list[dict], metric: str, xlabel: str, floor: float) -> None:
    labels = [label for label, _ in GATES]
    y = np.arange(len(labels))
    means, xerr = metric_arrays(rows, metric)
    colors = [L1_COLOR] + [RUN_COLOR] * (len(labels) - 1)

    ax.barh(y, means, color=colors, height=0.62, edgecolor="white", linewidth=0.8)
    ax.errorbar(
        means,
        y,
        xerr=xerr,
        fmt="none",
        ecolor=INK,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=3,
    )
    ax.axvline(floor, color=INK, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.axvline(0.0, color=MUTED, linewidth=0.8, alpha=0.55)
    set_metric_limits(ax, means, xerr, floor)
    ymax = len(labels) - 0.35
    ax.text(
        floor,
        ymax,
        "promotion floor",
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
        color=INK,
        backgroundcolor=SURFACE,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)


def main() -> None:
    rows = [load_gate(GATE_DIR / filename) for _, filename in GATES]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    draw_panel(axes[0], rows, "full9_mean_dpsnr", "dPSNR (dB)", 0.10)
    draw_panel(axes[1], rows, "full9_mean_dlpips", "dLPIPS (LPIPS units)", -0.004)
    axes[0].set_title("PSNR ladder")
    axes[1].set_title("LPIPS ladder")
    fig.suptitle("Stage-4 ECR ladder: full9 paired CIs", y=0.98, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    pdf = OUT_DIR / "ladder_ci.pdf"
    png = OUT_DIR / "ladder_ci.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()

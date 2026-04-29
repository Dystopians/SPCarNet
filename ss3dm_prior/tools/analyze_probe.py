"""Analyze residual-direction probe arrays from diagnose_carnet.

Input: probe_points.npz with arrays r_in (P, N) and r_out (P, N).
    r_in  = per-point distance from corrupted point to its nearest clean point.
    r_out = per-point distance from recon point to its nearest clean point.

Writes:
    residual_ratio_histogram.png  - log-scale histogram of ratio = r_out / r_in
    residual_scatter.png          - scatter of r_out vs r_in, with y=x
    residual_stats.json           - median / mean / quantiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    data = np.load(args.probe_npz)
    r_in = data["r_in"]
    r_out = data["r_out"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eps = 1e-6
    ratio = r_out / np.clip(r_in, eps, None)
    ratio_flat = ratio.reshape(-1)
    r_in_flat = r_in.reshape(-1)
    r_out_flat = r_out.reshape(-1)

    stats = {
        "n_points": int(ratio_flat.size),
        "n_patches": int(r_in.shape[0]),
        "r_in_mean": float(np.mean(r_in_flat)),
        "r_in_median": float(np.median(r_in_flat)),
        "r_out_mean": float(np.mean(r_out_flat)),
        "r_out_median": float(np.median(r_out_flat)),
        "ratio_mean": float(np.mean(ratio_flat)),
        "ratio_median": float(np.median(ratio_flat)),
        "ratio_q25": float(np.quantile(ratio_flat, 0.25)),
        "ratio_q75": float(np.quantile(ratio_flat, 0.75)),
        "ratio_q95": float(np.quantile(ratio_flat, 0.95)),
        "pct_ratio_lt_0_2": float(np.mean(ratio_flat < 0.2)),
        "pct_ratio_0_2_to_0_6": float(np.mean((ratio_flat >= 0.2) & (ratio_flat < 0.6))),
        "pct_ratio_0_6_to_1_0": float(np.mean((ratio_flat >= 0.6) & (ratio_flat < 1.0))),
        "pct_ratio_gt_1_0": float(np.mean(ratio_flat >= 1.0)),
    }
    (output_dir / "residual_stats.json").write_text(json.dumps(stats, indent=2))

    # Histogram: ratio distribution (clip at 3 for display)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 3, 61)
    ratio_clipped = np.clip(ratio_flat, 0, 3)
    ax.hist(ratio_clipped, bins=bins, color="C0", alpha=0.85)
    ax.axvline(1.0, color="red", linestyle="--", label="ratio=1 (no fix)")
    ax.axvline(stats["ratio_median"], color="black", linestyle=":", label=f"median={stats['ratio_median']:.3f}")
    ax.set_xlabel("ratio = ||recon - clean_NN|| / ||corrupted - clean_NN||")
    ax.set_ylabel("count")
    ax.set_title(f"Residual direction ratio (n_patches={stats['n_patches']}, n_points={stats['n_points']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_ratio_histogram.png", dpi=100)
    plt.close(fig)

    # Scatter: r_out vs r_in (subsample for readability)
    rng = np.random.default_rng(0)
    sample_n = min(20000, r_in_flat.size)
    idx = rng.choice(r_in_flat.size, sample_n, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(r_in_flat[idx], r_out_flat[idx], s=1, alpha=0.25, color="C0")
    vmax = float(np.quantile(np.concatenate([r_in_flat, r_out_flat]), 0.99))
    diag = np.linspace(0, vmax, 100)
    ax.plot(diag, diag, "r--", linewidth=1.5, label="y=x (no fix)")
    ax.plot(diag, 0.5 * diag, "g--", linewidth=1.0, label="y=0.5x (half fix)")
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_xlabel("r_in  = ||corrupted - clean_NN||")
    ax.set_ylabel("r_out = ||recon - clean_NN||")
    ax.set_title("Per-point error: input vs output")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residual_scatter.png", dpi=100)
    plt.close(fig)

    print(json.dumps(stats, indent=2), flush=True)
    print(f"[probe] wrote residual_stats.json + 2 plots to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Stream a huge ASCII .obj and render a lightweight point preview (vertices only).
Skips faces (f), vt, vn — memory stays ~O(sample budget), not mesh size.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stream_sample_vertices(
    path: str,
    max_points: int,
    seed: int,
    stride: int,
) -> np.ndarray:
    """
    Collect vertices with:
    - take every `stride`-th 'v ' line (fast path for huge files)
    - if still > max_points, random subsample without storing all
    """
    rng = random.Random(seed)
    buf: list[list[float]] = []
    n_v = 0
    with open(path, "r", errors="replace", encoding="utf-8", buffering=1024 * 1024) as f:
        for line in f:
            if not line.startswith("v "):
                continue
            n_v += 1
            if stride > 1 and (n_v % stride) != 0:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if len(buf) < max_points * 2:
                buf.append([x, y, z])
            else:
                # reservoir-ish: replace randomly to cap memory
                j = rng.randint(0, n_v - 1)
                if j < len(buf):
                    buf[j] = [x, y, z]

    if not buf:
        raise RuntimeError("No vertices parsed (wrong format or binary OBJ?)")

    v = np.asarray(buf, dtype=np.float64)
    if v.shape[0] > max_points:
        idx = rng.sample(range(v.shape[0]), max_points)
        v = v[idx]
    return v


def render_png(
    v: np.ndarray,
    title: str,
    out: str,
    clip_lo: float | None = 0.5,
    clip_hi: float | None = 99.5,
) -> None:
    if clip_lo is not None and clip_hi is not None and v.shape[0] > 100:
        lo = np.percentile(v, clip_lo, axis=0)
        hi = np.percentile(v, clip_hi, axis=0)
        m = np.all((v >= lo) & (v <= hi), axis=1)
        if m.sum() > 5000:
            v = v[m]
    v = v - v.mean(axis=0)
    span = np.ptp(v, axis=0)
    span = np.where(span < 1e-9, 1.0, span)
    fig = plt.figure(figsize=(9, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    z = v[:, 2]
    sc = ax.scatter(
        v[:, 0], v[:, 1], v[:, 2], s=0.35, c=z, cmap="turbo", alpha=0.75, linewidths=0, rasterized=True
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    try:
        ax.set_box_aspect(tuple(span / span.max()))
    except Exception:
        pass
    plt.colorbar(sc, ax=ax, shrink=0.55, label="Z (centered)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream-sample huge ASCII OBJ -> PNG preview")
    ap.add_argument("obj", help="Path to .obj")
    ap.add_argument("-o", "--output", default="", help="Output PNG (default: <obj>.preview.png)")
    ap.add_argument("-n", "--max-points", type=int, default=80_000)
    ap.add_argument("--stride", type=int, default=50, help="Keep every Nth vertex line (raise if still too slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not percentile-clip outliers before plotting",
    )
    args = ap.parse_args()

    obj = os.path.abspath(args.obj)
    if not os.path.isfile(obj):
        print(f"Not a file: {obj}", file=sys.stderr)
        return 1
    out = args.output or (obj + ".preview.png")

    print(f"Streaming vertices from {obj} (stride={args.stride}, cap~{args.max_points}) ...", flush=True)
    v = stream_sample_vertices(obj, max_points=args.max_points, seed=args.seed, stride=args.stride)
    print(f"Plotting {v.shape[0]} points -> {out}", flush=True)
    clip = (None, None) if args.no_clip else (0.5, 99.5)
    render_png(
        v,
        f"{os.path.basename(obj)}\n{v.shape[0]} sampled verts",
        out,
        clip_lo=clip[0],
        clip_hi=clip[1],
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

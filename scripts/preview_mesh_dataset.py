#!/usr/bin/env python3
"""Render small PNG previews of representative PLY meshes (headless)."""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_vertices(path: str) -> np.ndarray:
    import trimesh

    g = trimesh.load(path, process=False, force="mesh")
    if isinstance(g, trimesh.Scene):
        geoms = [mesh for mesh in g.geometry.values() if hasattr(mesh, "vertices")]
        if not geoms:
            raise RuntimeError(f"No geometry in scene: {path}")
        g = trimesh.util.concatenate(geoms)
    if isinstance(g, trimesh.PointCloud):
        return np.asarray(g.vertices, dtype=np.float64)
    if isinstance(g, trimesh.Trimesh):
        return np.asarray(g.vertices, dtype=np.float64)
    raise TypeError(f"Unsupported type {type(g)} for {path}")


def subsample(v: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    n = v.shape[0]
    if n <= max_n:
        return v
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return v[idx]


def render_scatter(
    v: np.ndarray,
    title: str,
    out_png: str,
    max_points: int = 35_000,
) -> None:
    v = subsample(v, max_points)
    # center for nicer view
    c = v.mean(axis=0)
    v = v - c
    span = np.ptp(v, axis=0)
    span = np.where(span < 1e-9, 1.0, span)

    fig = plt.figure(figsize=(9, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    z = v[:, 2]
    sc = ax.scatter(
        v[:, 0],
        v[:, 1],
        v[:, 2],
        s=0.35,
        c=z,
        cmap="turbo",
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    try:
        ax.set_box_aspect(tuple(span / span.max()))
    except Exception:
        pass
    plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label="Z (centered)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    base = "/data2/peilincai/Mesh_Dataset"
    out_dir = os.path.join(base, "_previews")
    os.makedirs(out_dir, exist_ok=True)

    samples: list[tuple[str, str]] = [
        (
            "nerf_loam__mesh_final",
            f"{base}/nerf_loam/Town10_1000/2023-12-27-23-36-52/mesh/final_mesh_transformed.ply",
        ),
        (
            "nerf_loam__meshes_pred",
            f"{base}/nerf_loam/Town10_1000/2023-12-27-23-36-52/meshes/cleaned_pred_mesh.ply",
        ),
        (
            "r3d3__cleaned_pred_mesh",
            f"{base}/r3d3/Town10/Town10_1000/meshes/cleaned_pred_mesh.ply",
        ),
        (
            "streetsurf__full_cfg",
            f"{base}/streetsurf/Town10_1000/withmask_withlidar_withnormal_all_cameras/meshes/cleaned_pred_mesh.ply",
        ),
        (
            "sugar__cleaned_pred_mesh",
            f"{base}/sugar/Town10_1000/meshes/cleaned_pred_mesh.ply",
        ),
        (
            "urban_nerf__cleaned_pred_mesh",
            f"{base}/urban_nerf/Town10_1000/withmask_withlidar_withnormal_all_cameras/meshes/cleaned_pred_mesh.ply",
        ),
    ]

    for name, ply in samples:
        if not os.path.isfile(ply):
            print(f"SKIP missing: {ply}", file=sys.stderr)
            continue
        out = os.path.join(out_dir, f"{name}.png")
        print(f"Rendering {name} ...")
        verts = load_vertices(ply)
        title = f"{name}\n{os.path.basename(ply)} ({verts.shape[0]:,} verts)"
        render_scatter(verts, title, out)
        print(f"  -> {out}")

    print(f"Done. Open: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

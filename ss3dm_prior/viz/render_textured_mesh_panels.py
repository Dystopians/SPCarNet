"""Photorealistic whole-mesh rendering panels for textured assets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def _camera_pose(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    eye = np.asarray(eye, dtype=np.float32)
    up_hint = np.asarray(up_hint if up_hint is not None else [0.0, 0.0, 1.0], dtype=np.float32)
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up_hint))
    if np.linalg.norm(right) <= 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    up = _normalize(np.cross(right, forward))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _load_trimesh_scene(mesh_path: str | Path):
    import trimesh

    loaded = trimesh.load(Path(mesh_path).expanduser().resolve(), process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    scene = trimesh.Scene()
    scene.add_geometry(loaded)
    return scene


def _displacements_from_targets(
    clean_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[cKDTree, np.ndarray]:
    clean_points = np.asarray(clean_points, dtype=np.float32)
    target_points = np.asarray(target_points, dtype=np.float32)
    clean_tree = cKDTree(clean_points)
    target_tree = cKDTree(target_points)
    _, target_nn = target_tree.query(clean_points, k=1)
    target_nn = np.asarray(target_nn, dtype=np.int64)
    displacements = target_points[target_nn] - clean_points
    return clean_tree, displacements.astype(np.float32)


def _deform_scene_geometry(
    scene,
    *,
    clean_points: np.ndarray,
    target_points: np.ndarray,
):
    import trimesh

    clean_tree, displacements = _displacements_from_targets(clean_points, target_points)
    deformed_meshes: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph.get(node_name)
        geometry = scene.geometry[geom_name].copy()
        geometry.apply_transform(transform)
        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        _, clean_nn = clean_tree.query(vertices, k=1)
        clean_nn = np.asarray(clean_nn, dtype=np.int64)
        geometry.vertices = vertices + displacements[clean_nn]
        deformed_meshes.append(geometry)
    return deformed_meshes


def _make_render_scene(meshes: Iterable):
    import pyrender

    scene = pyrender.Scene(bg_color=[245, 247, 250, 255], ambient_light=[0.45, 0.45, 0.45])
    for mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    return scene


def _view_specs(radius: float) -> list[tuple[str, np.ndarray, np.ndarray | None, float]]:
    distance = max(float(radius) * 1.75, 1.45)
    return [
        ("Hero 3/4", np.asarray([0.92, -1.0, 0.48], dtype=np.float32) * distance, None, 34.0),
        ("Front", np.asarray([0.0, -1.12, 0.18], dtype=np.float32) * distance, None, 30.0),
        ("Top-Down", np.asarray([0.0, -0.08, 1.08], dtype=np.float32) * distance, np.asarray([0.0, 1.0, 0.0], dtype=np.float32), 28.0),
        ("Low Angle", np.asarray([0.78, -0.92, -0.18], dtype=np.float32) * distance, np.asarray([0.0, 0.0, 1.0], dtype=np.float32), 32.0),
    ]


def _render_scene_rgb(
    meshes: Iterable,
    *,
    center: np.ndarray,
    image_size: tuple[int, int],
    eye_offset: np.ndarray,
    up_hint: np.ndarray | None,
    yfov_deg: float,
) -> np.ndarray:
    import pyrender

    scene = _make_render_scene(meshes)
    width, height = image_size
    eye = center + np.asarray(eye_offset, dtype=np.float32)
    camera_pose = _camera_pose(eye=eye, target=center, up_hint=up_hint)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(float(yfov_deg)))
    scene.add(camera, pose=camera_pose)

    base_distance = float(np.linalg.norm(eye_offset))
    key_light_pose = _camera_pose(
        center + np.asarray([0.9, -0.55, 1.25], dtype=np.float32) * base_distance,
        center,
        up_hint=up_hint,
    )
    fill_light_pose = _camera_pose(
        center + np.asarray([-0.65, 0.18, 0.75], dtype=np.float32) * base_distance,
        center,
        up_hint=up_hint,
    )
    rim_light_pose = _camera_pose(
        center + np.asarray([-1.05, 0.85, 0.52], dtype=np.float32) * base_distance,
        center,
        up_hint=up_hint,
    )
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.0), pose=key_light_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=fill_light_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.4), pose=rim_light_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=int(width), viewport_height=int(height))
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return np.asarray(color, dtype=np.uint8)


def render_textured_whole_mesh_triptych(
    *,
    source_mesh_path: str | Path,
    clean_points: np.ndarray,
    corrupted_points: np.ndarray,
    recon_points: np.ndarray,
    info_lines: list[str],
    output_path: str | Path,
    image_size: tuple[int, int] = (760, 540),
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_scene = _load_trimesh_scene(source_mesh_path)
    clean_points = np.asarray(clean_points, dtype=np.float32)
    corrupted_points = np.asarray(corrupted_points, dtype=np.float32)
    recon_points = np.asarray(recon_points, dtype=np.float32)

    all_points = np.concatenate([clean_points, corrupted_points, recon_points], axis=0)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.linalg.norm(maxs - mins)), 1.0) * 0.5

    corrupted_meshes = _deform_scene_geometry(source_scene, clean_points=clean_points, target_points=corrupted_points)
    recon_meshes = _deform_scene_geometry(source_scene, clean_points=clean_points, target_points=recon_points)
    clean_meshes = _deform_scene_geometry(source_scene, clean_points=clean_points, target_points=clean_points)

    rendered_sets = [
        ("Corrupt Render", corrupted_meshes),
        ("Repaired Render", recon_meshes),
        ("Ground Truth Render", clean_meshes),
    ]
    view_specs = _view_specs(radius)

    fig = plt.figure(figsize=(17, 12.5))
    grid = fig.add_gridspec(
        len(view_specs) + 1,
        len(rendered_sets),
        height_ratios=[4.2, 4.2, 4.2, 4.2, 1.35],
        hspace=0.04,
        wspace=0.02,
    )

    for row_idx, (view_name, eye_offset, up_hint, yfov_deg) in enumerate(view_specs):
        for col_idx, (title, meshes) in enumerate(rendered_sets):
            image = _render_scene_rgb(
                meshes,
                center=center,
                image_size=image_size,
                eye_offset=eye_offset,
                up_hint=up_hint,
                yfov_deg=yfov_deg,
            )
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.imshow(image)
            if row_idx == 0:
                ax.set_title(title, fontsize=12)
            ax.text(
                0.02,
                0.04,
                view_name,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                color="white",
                bbox={"facecolor": (0.05, 0.05, 0.05, 0.72), "edgecolor": "none", "pad": 2.5},
            )
            ax.axis("off")

    text_ax = fig.add_subplot(grid[-1, :])
    text_ax.axis("off")
    text_ax.text(
        0.01,
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

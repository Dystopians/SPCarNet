#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams  # noqa: E402
from scene import Scene  # noqa: E402
from triangle_renderer import TriangleModel, render  # noqa: E402
from utils.prism_geometry_proxy import (  # noqa: E402
    GeometryProxyConfig,
    build_geometry_proxy_context,
    collect_view_sparse_depth_correspondences,
    normalize_image_key,
)
from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402
from ss3dm_prior.meshsplatopt.compact_selector import (  # noqa: E402
    CompactionSignals,
    select_faces,
)


def _clone_dataset(dataset, model_path: str):
    out = copy.copy(dataset)
    out.model_path = str(model_path)
    return out


def _load_scene(dataset, model_path: str, iteration: int):
    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = 4
    scene = Scene(
        args=_clone_dataset(dataset, model_path),
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=int(iteration),
        shuffle=False,
    )
    return scene, triangles


def _split_views(scene, split: str):
    split = str(split).lower()
    if split == "test":
        raise RuntimeError("Sparse occluder candidate mining refuses test split to prevent leakage.")
    train = list(scene.getTrainCameras())
    if split == "train":
        return train
    if split == "calibration":
        return train[:: max(1, len(train) // 32)] if len(train) > 32 else train
    raise ValueError(f"unsupported split: {split}")


def _select_views(views, num_views: int):
    views = list(views)
    if num_views <= 0 or len(views) <= num_views:
        return views
    ids = np.linspace(0, len(views) - 1, num=int(num_views))
    return [views[int(round(float(i)))] for i in ids]


def _sample_depth(render_pkg: dict[str, Any], px: np.ndarray, py: np.ndarray) -> np.ndarray:
    surf_depth = render_pkg.get("surf_depth", None)
    if surf_depth is None:
        return np.full(px.shape, np.nan, dtype=np.float64)
    depth = surf_depth[0].detach().cpu().numpy()
    h, w = depth.shape
    x = np.clip(px.astype(np.int64), 0, w - 1)
    y = np.clip(py.astype(np.int64), 0, h - 1)
    return depth[y, x].astype(np.float64)


def _sample_render_ids(render_pkg: dict[str, Any], px: np.ndarray, py: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    ids_t = render_pkg.get("rend_ids", None)
    if ids_t is None:
        return np.full(px.shape, -1, dtype=np.int64)
    ids = ids_t[0].detach().cpu().numpy().astype(np.int64, copy=False)
    hid, wid = ids.shape
    sx = float(wid) / max(float(width), 1.0)
    sy = float(hid) / max(float(height), 1.0)
    cx = np.clip(np.floor((px.astype(np.float64) + 0.5) * sx).astype(np.int64), 0, wid - 1)
    cy = np.clip(np.floor((py.astype(np.float64) + 0.5) * sy).astype(np.int64), 0, hid - 1)
    out = np.full(px.shape, -1, dtype=np.int64)
    r = max(0, int(radius))
    for i, (x, y) in enumerate(zip(cx.tolist(), cy.tolist())):
        if r <= 0:
            out[i] = int(ids[y, x])
            continue
        patch = ids[max(0, y - r) : min(hid, y + r + 1), max(0, x - r) : min(wid, x + r + 1)].reshape(-1)
        patch = patch[patch >= 0]
        if patch.size == 0:
            continue
        values, counts = np.unique(patch, return_counts=True)
        out[i] = int(values[int(np.argmax(counts))])
    return out


def _load_checkpoint_signals(source_model: Path, iteration: int) -> CompactionSignals:
    state = torch.load(checkpoint_path(source_model, iteration), map_location="cpu")
    faces = state["_triangle_indices"].detach().cpu().numpy()
    face_count = int(faces.shape[0])

    def face_signal(name: str) -> np.ndarray | None:
        value = state.get(name)
        if value is None or not hasattr(value, "shape") or int(value.shape[0]) != face_count:
            return None
        return value.detach().cpu().float().reshape(face_count, -1).mean(dim=1).numpy()

    importance = face_signal("importance_score")
    pixel_count = face_signal("pixel_count")
    image_size = face_signal("image_size")
    render_contribution = importance
    if render_contribution is None and pixel_count is not None and image_size is not None:
        render_contribution = pixel_count / np.maximum(image_size, 1e-6)
    return CompactionSignals(
        vertices=state["triangles_points"].detach().cpu().numpy(),
        faces=faces,
        render_contribution=render_contribution,
        positive_surface_evidence=render_contribution,
    )


def _write_candidates(
    output_dir: Path,
    selected: np.ndarray,
    *,
    face_count: int,
    base_selected: np.ndarray,
    risk_selected: np.ndarray,
    stats: dict[str, Any],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = np.sort(np.unique(selected.astype(np.int64, copy=False)))
    payload: dict[str, Any] = {
        "mode": "sparse_occluder_low_evidence_union",
        "target_prune_fraction": float(selected.shape[0] / max(face_count, 1)),
        "selected_faces_path": "selected_faces.npy",
        "selected_faces_count": int(selected.shape[0]),
        "selected_faces_head": [int(x) for x in selected[:32].tolist()],
        "selected_faces_tail": [int(x) for x in selected[-32:].tolist()],
        "summary": {
            "face_count": int(face_count),
            "selected_count": int(selected.shape[0]),
            "selected_fraction": float(selected.shape[0] / max(face_count, 1)),
            "base_low_evidence_count": int(base_selected.shape[0]),
            "sparse_occluder_count": int(risk_selected.shape[0]),
            "union_overlap_count": int(base_selected.shape[0] + risk_selected.shape[0] - selected.shape[0]),
        },
        "sparse_occluder_stats": stats,
        "config": config,
    }
    np.save(output_dir / "selected_faces.npy", selected)
    (output_dir / "compaction_candidates.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Sparse Occluder Prune Candidate Report",
        "",
        f"- mode: `{payload['mode']}`",
        f"- selected faces: `{selected.shape[0]}` / `{face_count}`",
        f"- selected fraction: `{payload['summary']['selected_fraction']:.9f}`",
        f"- effective base prune fraction: `{config.get('effective_base_prune_fraction', config.get('base_prune_fraction'))}`",
        f"- effective sparse occluder cap: `{config.get('effective_max_sparse_occluder_fraction', config.get('max_sparse_occluder_fraction'))}`",
        f"- base low-evidence faces: `{base_selected.shape[0]}`",
        f"- sparse occluder faces: `{risk_selected.shape[0]}`",
        f"- union overlap: `{payload['summary']['union_overlap_count']}`",
        f"- train sparse points used: `{stats.get('valid_sparse_points', 0)}`",
        f"- front-occluder sparse points: `{stats.get('front_occluder_points', 0)}`",
        f"- front-occluder rate: `{stats.get('front_occluder_rate', 0.0):.8f}`",
        f"- mean sparse AbsRel: `{stats.get('mean_sparse_absrel', 0.0):.8f}`",
        f"- adaptive decision: `{json.dumps(stats.get('adaptive_geometry_budget', {}), sort_keys=True)}`",
        f"- touched faces: `{stats.get('touched_faces', 0)}`",
        "",
        "This selector uses only train/calibration sparse COLMAP correspondences. It targets faces that are the rendered front surface where the model depth is closer than COLMAP by a fixed relative margin, then unions them with a small low-evidence compression base.",
    ]
    (output_dir / "compaction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _estimated_vertex_reduction(faces: np.ndarray, vertex_count: int, selected: np.ndarray) -> float:
    if selected.size == 0:
        return 0.0
    face_count = int(faces.shape[0])
    remove_mask = np.zeros((face_count,), dtype=bool)
    remove_mask[np.asarray(selected, dtype=np.int64)] = True
    kept = faces[~remove_mask].reshape(-1)
    used = np.unique(kept)
    return float(1.0 - float(used.shape[0]) / max(float(vertex_count), 1.0))


def run(args) -> int:
    dataset = args.dataset
    pipe = args.pipe
    source_model = Path(args.source_model)
    scene, triangles = _load_scene(dataset, str(source_model), int(args.iteration))
    if len(scene.scene_info.colmap_points3d or {}) == 0:
        raise RuntimeError("COLMAP sparse points are unavailable; cannot mine sparse occluder candidates.")
    views = _select_views(_split_views(scene, args.split), int(args.num_views))
    if not views:
        raise RuntimeError(f"no views selected for split={args.split}")

    signals = _load_checkpoint_signals(source_model, int(args.iteration))
    face_count = int(signals.faces.shape[0])

    hit_count = np.zeros((face_count,), dtype=np.int32)
    front_excess_sum = np.zeros((face_count,), dtype=np.float64)
    absrel_sum = np.zeros((face_count,), dtype=np.float64)
    low_error_count = np.zeros((face_count,), dtype=np.int32)
    absrel_total = 0.0
    low_error_total = 0

    proxy_cfg = GeometryProxyConfig(
        max_points_per_view=int(args.max_points_per_view),
        point_error_max=float(args.point_error_max),
        normal_knn=24,
        compute_normal=False,
        seed=int(args.seed),
        sample_mode=str(args.sample_mode),
        low_error_fraction=float(args.low_error_fraction),
    )
    cam_infos = []
    cam_infos.extend(list(getattr(scene.scene_info, "train_cameras", []) or []))
    cam_infos.extend(list(getattr(scene.scene_info, "test_cameras", []) or []))
    proxy_ctx = build_geometry_proxy_context(
        colmap_points3d=scene.scene_info.colmap_points3d,
        cam_infos=cam_infos,
        cfg=proxy_cfg,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rng = np.random.default_rng(int(args.seed))
    dropped: dict[str, int] = {}
    valid_sparse_points = 0
    front_occluder_points = 0
    view_manifest: list[dict[str, Any]] = []
    with torch.no_grad():
        for view in views:
            pkg = render(view, triangles, pipe, background)
            corr = collect_view_sparse_depth_correspondences(view=view, ctx=proxy_ctx, cfg=proxy_cfg, rng=rng)
            n = int(corr.get("num_matches", 0))
            image_key = normalize_image_key(str(getattr(view, "image_name", "")))
            if n <= 0:
                reason = str(corr.get("reason", "unknown"))
                dropped[reason] = int(dropped.get(reason, 0)) + 1
                view_manifest.append({"image_key": image_key, "matches": 0, "reason": reason})
                continue
            px = np.asarray(corr["px"], dtype=np.int64)
            py = np.asarray(corr["py"], dtype=np.int64)
            gt = np.asarray(corr["gt_depth"], dtype=np.float64)
            pred = _sample_depth(pkg, px, py)
            depth_image = pkg["surf_depth"][0]
            height = int(depth_image.shape[0])
            width = int(depth_image.shape[1])
            face_ids = _sample_render_ids(pkg, px, py, width, height, int(args.id_patch_radius))
            valid = (
                np.isfinite(gt)
                & np.isfinite(pred)
                & (gt > 1e-8)
                & (pred > 1e-8)
                & (face_ids >= 0)
                & (face_ids < face_count)
            )
            if not np.any(valid):
                view_manifest.append({"image_key": image_key, "matches": int(n), "valid": 0})
                continue
            gt_v = gt[valid]
            pred_v = pred[valid]
            ids_v = face_ids[valid].astype(np.int64, copy=False)
            rel_signed = (gt_v - pred_v) / np.maximum(gt_v, 1e-8)
            abs_rel = np.abs(pred_v - gt_v) / np.maximum(gt_v, 1e-8)
            front_excess = np.maximum(rel_signed - float(args.front_rel_margin), 0.0)
            low_error = abs_rel <= float(args.low_absrel_support)
            np.add.at(hit_count, ids_v, 1)
            np.add.at(absrel_sum, ids_v, abs_rel)
            np.add.at(front_excess_sum, ids_v, front_excess)
            np.add.at(low_error_count, ids_v, low_error.astype(np.int32))
            valid_sparse_points += int(ids_v.shape[0])
            front_occluder_points += int(np.count_nonzero(front_excess > 0.0))
            absrel_total += float(np.sum(abs_rel))
            low_error_total += int(np.count_nonzero(low_error))
            view_manifest.append(
                {
                    "image_key": image_key,
                    "matches": int(n),
                    "valid": int(ids_v.shape[0]),
                    "front_occluder": int(np.count_nonzero(front_excess > 0.0)),
                    "mean_abs_rel": float(np.mean(abs_rel)) if abs_rel.size else math.nan,
                }
            )

    touched = hit_count > 0
    mean_front = np.zeros((face_count,), dtype=np.float64)
    mean_absrel = np.zeros((face_count,), dtype=np.float64)
    low_ratio = np.zeros((face_count,), dtype=np.float64)
    mean_front[touched] = front_excess_sum[touched] / np.maximum(hit_count[touched], 1)
    mean_absrel[touched] = absrel_sum[touched] / np.maximum(hit_count[touched], 1)
    low_ratio[touched] = low_error_count[touched] / np.maximum(hit_count[touched], 1)
    score = mean_front * np.log1p(hit_count.astype(np.float64)) * (1.0 - 0.75 * low_ratio)
    eligible = (hit_count >= int(args.min_face_hits)) & (score > 0.0)
    eligible_ids = np.flatnonzero(eligible)
    front_occluder_rate = float(front_occluder_points / max(valid_sparse_points, 1))
    mean_sparse_absrel = float(absrel_total / max(valid_sparse_points, 1))
    sparse_low_error_ratio = float(low_error_total / max(valid_sparse_points, 1))
    effective_base_prune_fraction = float(args.base_prune_fraction)
    effective_max_sparse_occluder_fraction = float(args.max_sparse_occluder_fraction)
    adaptive_decision: dict[str, Any] = {
        "enabled": bool(args.adaptive_geometry_budget),
        "mode": "fixed_sparse_occluder_budget",
        "reason": "adaptive geometry budget disabled",
        "input_base_prune_fraction": float(args.base_prune_fraction),
        "input_max_sparse_occluder_fraction": float(args.max_sparse_occluder_fraction),
        "front_occluder_rate": front_occluder_rate,
        "mean_sparse_absrel": mean_sparse_absrel,
        "sparse_low_error_ratio": sparse_low_error_ratio,
        "valid_sparse_points": int(valid_sparse_points),
    }
    if bool(args.adaptive_geometry_budget):
        high_geometry_confidence = (
            valid_sparse_points >= int(args.adaptive_min_valid_sparse_points)
            and front_occluder_rate <= float(args.adaptive_front_occluder_rate_threshold)
            and mean_sparse_absrel <= float(args.adaptive_mean_absrel_threshold)
        )
        if high_geometry_confidence:
            effective_base_prune_fraction = min(
                effective_base_prune_fraction,
                float(args.adaptive_high_conf_base_prune_fraction),
            )
            effective_max_sparse_occluder_fraction = min(
                effective_max_sparse_occluder_fraction,
                float(args.adaptive_high_conf_max_sparse_occluder_fraction),
            )
            adaptive_decision.update(
                {
                    "mode": "high_geometry_confidence_guard",
                    "reason": (
                        "train sparse geometry is already reliable; preserve geometry with a "
                        "conservative low-evidence-only budget"
                    ),
                }
            )
            ultra_stable_geometry = (
                mean_sparse_absrel <= float(args.adaptive_ultra_stable_mean_absrel_threshold)
                and sparse_low_error_ratio >= float(args.adaptive_ultra_stable_low_error_ratio_floor)
            )
            if ultra_stable_geometry:
                effective_base_prune_fraction = min(
                    effective_base_prune_fraction,
                    float(args.adaptive_render_stability_base_prune_fraction),
                )
                effective_max_sparse_occluder_fraction = 0.0
                adaptive_decision.update(
                    {
                        "mode": "high_geometry_confidence_ultra_stable_guard",
                        "reason": (
                            "train sparse geometry is already ultra stable; use a micro "
                            "topology budget to preserve normals while keeping the ELA recovery path"
                        ),
                        "ultra_stable_mean_absrel_threshold": float(
                            args.adaptive_ultra_stable_mean_absrel_threshold
                        ),
                        "ultra_stable_low_error_ratio_floor": float(
                            args.adaptive_ultra_stable_low_error_ratio_floor
                        ),
                    }
                )
        else:
            adaptive_decision.update(
                {
                    "mode": "sparse_occluder_repair_budget",
                    "reason": "train sparse geometry has enough occluder/depth residual evidence for SOR budget",
                }
            )
    base_selected, _ = select_faces(
        signals,
        "csef_low_evidence_boundary_protected",
        effective_base_prune_fraction,
        seed=int(args.seed),
    )
    estimated_base_vertex_reduction = _estimated_vertex_reduction(
        signals.faces,
        int(signals.vertices.shape[0]),
        base_selected.astype(np.int64, copy=False),
    )
    if (
        bool(args.adaptive_geometry_budget)
        and adaptive_decision.get("mode") in {"high_geometry_confidence_guard", "high_geometry_confidence_ultra_stable_guard"}
        and effective_base_prune_fraction > float(args.adaptive_render_stability_base_prune_fraction)
    ):
        topology_no_shrink_risk = (
            estimated_base_vertex_reduction
            < float(args.adaptive_render_stability_vertex_reduction_threshold)
        )
        train_occlusion_risk = (
            front_occluder_rate >= float(args.adaptive_render_stability_front_occluder_rate_floor)
            and sparse_low_error_ratio <= float(args.adaptive_render_stability_low_error_ratio_ceiling)
        )
        if topology_no_shrink_risk or train_occlusion_risk:
            effective_base_prune_fraction = min(
                effective_base_prune_fraction,
                float(args.adaptive_render_stability_base_prune_fraction),
            )
            effective_max_sparse_occluder_fraction = 0.0
            adaptive_decision.update(
                {
                    "mode": "high_geometry_confidence_render_stability_guard",
                    "reason": (
                        "train-only sparse evidence indicates high-confidence geometry but a "
                        "rasterization/overdraw risk; use a micro topology budget"
                    ),
                    "estimated_base_vertex_reduction": estimated_base_vertex_reduction,
                    "topology_no_shrink_risk": bool(topology_no_shrink_risk),
                    "train_occlusion_risk": bool(train_occlusion_risk),
                    "render_stability_front_occluder_rate_floor": float(
                        args.adaptive_render_stability_front_occluder_rate_floor
                    ),
                    "render_stability_low_error_ratio_ceiling": float(
                        args.adaptive_render_stability_low_error_ratio_ceiling
                    ),
                }
            )
            base_selected, _ = select_faces(
                signals,
                "csef_low_evidence_boundary_protected",
                effective_base_prune_fraction,
                seed=int(args.seed),
            )
            estimated_base_vertex_reduction = _estimated_vertex_reduction(
                signals.faces,
                int(signals.vertices.shape[0]),
                base_selected.astype(np.int64, copy=False),
            )
    risk_cap = min(int(round(face_count * effective_max_sparse_occluder_fraction)), int(eligible_ids.shape[0]))
    if risk_cap > 0:
        eligible_scores = score[eligible_ids]
        if risk_cap < eligible_ids.shape[0]:
            part = np.argpartition(eligible_scores, eligible_scores.shape[0] - risk_cap)[-risk_cap:]
            risk_selected = eligible_ids[part]
        else:
            risk_selected = eligible_ids
        risk_selected = np.sort(risk_selected.astype(np.int64, copy=False))
    else:
        risk_selected = np.zeros((0,), dtype=np.int64)
    selected = np.union1d(base_selected.astype(np.int64, copy=False), risk_selected)

    stats = {
        "split": str(args.split),
        "num_views": int(len(views)),
        "valid_sparse_points": int(valid_sparse_points),
        "front_occluder_points": int(front_occluder_points),
        "touched_faces": int(np.count_nonzero(touched)),
        "eligible_sparse_occluder_faces": int(eligible_ids.shape[0]),
        "selected_sparse_occluder_faces": int(risk_selected.shape[0]),
        "front_occluder_rate": front_occluder_rate,
        "mean_sparse_absrel": mean_sparse_absrel,
        "sparse_low_error_ratio": sparse_low_error_ratio,
        "max_score": float(np.max(score)) if score.size else 0.0,
        "mean_selected_score": float(np.mean(score[risk_selected])) if risk_selected.size else 0.0,
        "adaptive_geometry_budget": adaptive_decision,
        "estimated_base_vertex_reduction": float(estimated_base_vertex_reduction),
        "dropped_views": dropped,
        "view_manifest": view_manifest,
    }
    config = {
        "source_model": str(source_model),
        "iteration": int(args.iteration),
        "base_prune_fraction": float(args.base_prune_fraction),
        "max_sparse_occluder_fraction": float(args.max_sparse_occluder_fraction),
        "effective_base_prune_fraction": float(effective_base_prune_fraction),
        "effective_max_sparse_occluder_fraction": float(effective_max_sparse_occluder_fraction),
        "adaptive_geometry_budget": adaptive_decision,
        "front_rel_margin": float(args.front_rel_margin),
        "low_absrel_support": float(args.low_absrel_support),
        "min_face_hits": int(args.min_face_hits),
        "max_points_per_view": int(args.max_points_per_view),
        "sample_mode": str(args.sample_mode),
        "low_error_fraction": float(args.low_error_fraction),
        "point_error_max": float(args.point_error_max),
        "seed": int(args.seed),
        "id_patch_radius": int(args.id_patch_radius),
        "no_test_leakage": str(args.split).lower() != "test",
    }
    out = Path(args.output_dir)
    _write_candidates(out, selected, face_count=face_count, base_selected=base_selected, risk_selected=risk_selected, stats=stats, config=config)
    np.savez_compressed(
        out / "sparse_occluder_face_scores.npz",
        hit_count=hit_count,
        mean_front_excess=mean_front,
        mean_absrel=mean_absrel,
        low_error_ratio=low_ratio,
        score=score,
        risk_selected=risk_selected,
        base_selected=base_selected.astype(np.int64, copy=False),
        selected=selected.astype(np.int64, copy=False),
    )
    print(json.dumps({"summary": stats, "selected_count": int(selected.shape[0]), "selected_fraction": float(selected.shape[0] / max(face_count, 1))}, indent=2))
    print(f"[SparseOccluder] Saved candidates to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine train-split sparse-depth front-occluder faces and union them with low-evidence compaction.")
    model = ModelParams(parser, sentinel=False)
    pipe = PipelineParams(parser)
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "test"), default="train")
    parser.add_argument("--num_views", type=int, default=48)
    parser.add_argument("--base_prune_fraction", type=float, default=0.10)
    parser.add_argument("--max_sparse_occluder_fraction", type=float, default=0.01)
    parser.add_argument("--front_rel_margin", type=float, default=0.04)
    parser.add_argument("--low_absrel_support", type=float, default=0.03)
    parser.add_argument("--min_face_hits", type=int, default=1)
    parser.add_argument("--max_points_per_view", type=int, default=500)
    parser.add_argument("--point_error_max", type=float, default=2.0)
    parser.add_argument("--sample_mode", default="mixed_low_error")
    parser.add_argument("--low_error_fraction", type=float, default=0.5)
    parser.add_argument("--id_patch_radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adaptive_geometry_budget", action="store_true")
    parser.add_argument("--adaptive_min_valid_sparse_points", type=int, default=5000)
    parser.add_argument("--adaptive_front_occluder_rate_threshold", type=float, default=0.025)
    parser.add_argument("--adaptive_mean_absrel_threshold", type=float, default=0.015)
    parser.add_argument("--adaptive_high_conf_base_prune_fraction", type=float, default=0.015)
    parser.add_argument("--adaptive_high_conf_max_sparse_occluder_fraction", type=float, default=0.000005)
    parser.add_argument("--adaptive_ultra_stable_mean_absrel_threshold", type=float, default=0.010)
    parser.add_argument("--adaptive_ultra_stable_low_error_ratio_floor", type=float, default=0.95)
    parser.add_argument("--adaptive_render_stability_vertex_reduction_threshold", type=float, default=0.001)
    parser.add_argument("--adaptive_render_stability_base_prune_fraction", type=float, default=0.001)
    parser.add_argument("--adaptive_render_stability_front_occluder_rate_floor", type=float, default=0.02)
    parser.add_argument("--adaptive_render_stability_low_error_ratio_ceiling", type=float, default=0.95)
    parser.add_argument("--output_dir", required=True)
    parser.set_defaults(_model_group=model, _pipe_group=pipe)
    return parser


def main() -> int:
    parser = build_parser()
    parsed = parser.parse_args()
    parsed.dataset = parser.get_default("_model_group").extract(parsed)
    parsed.pipe = parser.get_default("_pipe_group").extract(parsed)
    return run(parsed)


if __name__ == "__main__":
    raise SystemExit(main())

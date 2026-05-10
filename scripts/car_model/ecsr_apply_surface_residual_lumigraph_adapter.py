#!/usr/bin/env python3
"""Apply a train-certified surface-attached residual lumigraph.

This adapter is intentionally different from the image-space ELA path.  Train
residuals are aggregated on persistent face ids, a train-only policy split
selects the global residual strength and optional face gate, and held-out
renders are corrected only through the target pixel's rendered face id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import gc
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpipsPyTorch.modules.lpips import LPIPS
from utils.loss_utils import ssim


_LPIPS_CACHE: dict[tuple[str, str], LPIPS] = {}


@dataclass
class FaceResidualField:
    residuals: dict[int, np.ndarray]
    centers: dict[int, np.ndarray]
    weights: dict[int, np.ndarray]
    source_views: dict[int, list[str]]
    selected_faces: set[int]
    distance_scale: float
    source_view_count: int
    observation_count: int


@dataclass
class PreparedPolicyView:
    name: str
    base: np.ndarray
    gt: np.ndarray
    face_id: np.ndarray
    signal: np.ndarray
    confidence: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--target_surface_map_dir", type=Path, required=True)
    parser.add_argument("--output_model_path", type=Path, default=None)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--base_method_name", required=True)
    parser.add_argument("--method_name", default="ours_26000_surface_residual_lumigraph")
    parser.add_argument("--top_k", type=int, default=8192)
    parser.add_argument("--min_view_hits", type=int, default=2)
    parser.add_argument("--min_consistency", type=float, default=0.85)
    parser.add_argument("--min_pixel_count", type=float, default=6.0)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--high_error_quantile", type=float, default=0.55)
    parser.add_argument("--max_abs_residual", type=float, default=0.25)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--distance_scale", type=float, default=0.0)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument(
        "--consensus_policy_strides",
        default="",
        help=(
            "Optional comma-separated extra policy-val strides. If set, the "
            "adapter accepts a nonzero residual only when the primary split and "
            "all extra train-only policy splits pass the same guards."
        ),
    )
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--policy_objective", choices=("psnr", "balanced"), default="balanced")
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--min_policy_dpsnr", type=float, default=0.0)
    parser.add_argument("--min_policy_dssim", type=float, default=0.0)
    parser.add_argument("--max_policy_dlpips", type=float, default=0.0)
    parser.add_argument("--face_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_face_gate_pixels", type=int, default=128)
    parser.add_argument("--min_face_gate_gain", type=float, default=0.0)
    parser.add_argument(
        "--neighbor_rings",
        type=int,
        default=0,
        help=(
            "Topology-propagation radius. Zero keeps the strict V8 behavior. "
            "A positive value allows train-certified residual source faces to "
            "propose adjacent target faces through checkpoint triangle topology."
        ),
    )
    parser.add_argument("--neighbor_max_targets_per_source", type=int, default=128)
    parser.add_argument("--neighbor_chunk_size", type=int, default=262144)
    parser.add_argument(
        "--consensus_face_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Intersect the primary face gate with gates fitted on all consensus policy splits.",
    )
    parser.add_argument(
        "--provisional_face_gate_recalibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If the raw policy is rejected despite a positive provisional alpha, "
            "fit a train-only face gate with that provisional alpha and rerun "
            "policy calibration on the gated signal."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "mesh-splatting-ecsr"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def _parse_alpha_grid(text: str) -> list[float]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _parse_stride_list(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text or "").split(","):
        token = token.strip()
        if not token:
            continue
        stride = max(int(token), 2)
        if stride not in out:
            out.append(stride)
    return out


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def read_selected_faces(
    evidence_dir: Path,
    *,
    top_k: int,
    min_view_hits: int,
    min_consistency: float,
    min_pixel_count: float,
    max_face_id: int | None = None,
) -> set[int]:
    csv_path = evidence_dir / "top_residual_supports.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(_float(row, "view_hits")) < int(min_view_hits):
                continue
            if _float(row, "residual_consistency") < float(min_consistency):
                continue
            if _float(row, "pixel_count") < float(min_pixel_count):
                continue
            face_id = int(_float(row, "face_id"))
            if max_face_id is not None and (face_id < 0 or face_id >= int(max_face_id)):
                continue
            rows.append(
                {
                    "face_id": face_id,
                    "score": _float(row, "score"),
                    "pixel_count": _float(row, "pixel_count"),
                }
            )
    rows.sort(key=lambda r: (float(r["score"]), float(r["pixel_count"])), reverse=True)
    return {int(row["face_id"]) for row in rows[: int(top_k)]}


def _view_paths(evidence_dir: Path) -> list[Path]:
    view_dir = evidence_dir / "views"
    if not view_dir.is_dir():
        view_dir = evidence_dir / "per_view_npz"
    return sorted(view_dir.glob("*.npz"))


def split_policy_views(paths: list[Path], stride: int) -> tuple[list[Path], list[Path]]:
    if len(paths) < 3:
        return paths, paths
    stride = max(int(stride), 2)
    fit: list[Path] = []
    val: list[Path] = []
    for idx, path in enumerate(paths):
        if idx % stride == 0:
            val.append(path)
        else:
            fit.append(path)
    if not fit or not val:
        return paths, paths
    return fit, val


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    cache_root = path.parent / ".ecsr_npz_memmap_cache" / path.stem
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        source_mtime = path.stat().st_mtime
        out: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(path, "r") as zf:
            members = [name for name in zf.namelist() if name.endswith(".npy")]
            for member in members:
                key = Path(member).stem
                target = cache_root / member
                if (not target.is_file()) or target.stat().st_mtime < source_mtime:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
                out[key] = np.lib.format.open_memmap(target, mode="r")
        return out
    except Exception:
        with np.load(path) as z:
            return {name: z[name] for name in z.files}


def _sorted_int_candidates(candidates: set[int] | np.ndarray | list[int]) -> np.ndarray:
    if isinstance(candidates, np.ndarray):
        arr = candidates.astype(np.int64, copy=False).reshape(-1)
    else:
        arr = np.asarray(list(candidates), dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    return np.unique(arr)


def _fast_isin_int(values: np.ndarray, candidates: set[int] | np.ndarray | list[int]) -> np.ndarray:
    candidate_arr = _sorted_int_candidates(candidates)
    if candidate_arr.size == 0:
        return np.zeros(values.shape, dtype=bool)
    vals = np.array(values, copy=True) if isinstance(values, np.memmap) else np.asarray(values)
    if vals.size == 0:
        return np.zeros(values.shape, dtype=bool)
    value_min = int(np.min(vals))
    value_max = int(np.max(vals))
    candidate_arr = candidate_arr[(candidate_arr >= value_min) & (candidate_arr <= value_max)]
    if candidate_arr.size == 0:
        return np.zeros(values.shape, dtype=bool)
    min_candidate = int(candidate_arr[0])
    max_candidate = int(candidate_arr[-1])
    span = max_candidate - min_candidate
    if span <= 50_000_000:
        table = np.zeros(span + 1, dtype=bool)
        table[candidate_arr - min_candidate] = True
        in_range = (vals >= min_candidate) & (vals <= max_candidate)
        out = np.zeros(values.shape, dtype=bool)
        if np.any(in_range):
            out[in_range] = table[vals[in_range] - min_candidate]
        return out
    unique_values, inverse = np.unique(vals.reshape(-1), return_inverse=True)
    idx = np.searchsorted(unique_values, candidate_arr)
    valid = idx < unique_values.size
    member_unique = np.zeros(unique_values.shape, dtype=bool)
    if np.any(valid):
        valid_candidates = candidate_arr[valid]
        valid_idx = idx[valid]
        matched = unique_values[valid_idx] == valid_candidates
        member_unique[valid_idx[matched]] = True
    return member_unique[inverse].reshape(values.shape)


def _weighted_face_observations(
    payload: dict[str, np.ndarray],
    selected_faces: set[int],
    *,
    min_alpha: float,
    high_error_quantile: float,
    max_abs_residual: float,
) -> dict[int, tuple[np.ndarray, float]]:
    required = {"face_id", "residual_rgb", "residual_l1", "alpha"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"surface evidence view missing required fields: {sorted(missing)}")
    face_id = np.array(payload["face_id"], copy=True)
    residual = payload["residual_rgb"].astype(np.float32)
    residual_l1 = payload["residual_l1"].astype(np.float32)
    alpha = payload["alpha"].astype(np.float32)
    if alpha.ndim == 3:
        alpha = np.squeeze(alpha, axis=0)
    if residual.shape[0] != 3:
        raise RuntimeError(f"expected residual_rgb with shape [3,H,W], got {residual.shape}")

    valid = (face_id >= 0) & (alpha >= float(min_alpha))
    if selected_faces:
        valid &= _fast_isin_int(face_id, selected_faces)
    if float(high_error_quantile) > 0.0:
        threshold = float(np.quantile(residual_l1.reshape(-1), min(max(float(high_error_quantile), 0.0), 0.999)))
        valid &= residual_l1 >= threshold
    if not np.any(valid):
        return {}

    fids = face_id[valid].astype(np.int64)
    res = np.moveaxis(residual, 0, -1)[valid].astype(np.float32)
    res = np.clip(res, -float(max_abs_residual), float(max_abs_residual))
    weights = np.maximum(residual_l1[valid].astype(np.float32), 1e-4) * np.maximum(alpha[valid], 1e-4)
    order = np.argsort(fids)
    fids = fids[order]
    res = res[order]
    weights = weights[order]
    unique, starts = np.unique(fids, return_index=True)
    ends = np.r_[starts[1:], len(fids)]
    out: dict[int, tuple[np.ndarray, float]] = {}
    for fid, start, end in zip(unique, starts, ends):
        w = weights[start:end]
        r = res[start:end]
        denom = float(np.sum(w))
        if denom <= 0.0:
            continue
        out[int(fid)] = ((r * w[:, None]).sum(axis=0) / denom, float(max(denom, end - start)))
    return out


def build_field(
    view_paths: list[Path],
    selected_faces: set[int],
    *,
    min_alpha: float,
    high_error_quantile: float,
    max_abs_residual: float,
    distance_scale: float,
) -> FaceResidualField:
    residuals: dict[int, list[np.ndarray]] = {}
    centers: dict[int, list[np.ndarray]] = {}
    weights: dict[int, list[float]] = {}
    source_views: dict[int, list[str]] = {}
    all_centers: list[np.ndarray] = []
    observation_count = 0
    for path in view_paths:
        payload = _load_npz(path)
        if "camera_center" not in payload:
            raise RuntimeError(f"{path} missing camera_center; rebuild surface evidence cache")
        center = payload["camera_center"].astype(np.float32).reshape(3)
        all_centers.append(center)
        obs = _weighted_face_observations(
            payload,
            selected_faces,
            min_alpha=min_alpha,
            high_error_quantile=high_error_quantile,
            max_abs_residual=max_abs_residual,
        )
        for fid, (mean_residual, weight) in obs.items():
            residuals.setdefault(fid, []).append(mean_residual.astype(np.float32))
            centers.setdefault(fid, []).append(center)
            weights.setdefault(fid, []).append(float(weight))
            source_views.setdefault(fid, []).append(path.stem)
            observation_count += 1
    packed_residuals = {fid: np.stack(vals, axis=0).astype(np.float32) for fid, vals in residuals.items()}
    packed_centers = {fid: np.stack(vals, axis=0).astype(np.float32) for fid, vals in centers.items()}
    packed_weights = {fid: np.asarray(vals, dtype=np.float32) for fid, vals in weights.items()}
    if float(distance_scale) > 0.0:
        scale = float(distance_scale)
    elif len(all_centers) > 1:
        centers_np = np.stack(all_centers, axis=0)
        scale = float(np.median(np.linalg.norm(centers_np - centers_np.mean(axis=0, keepdims=True), axis=1)))
        scale = max(scale, 1e-3)
    else:
        scale = 1.0
    return FaceResidualField(
        residuals=packed_residuals,
        centers=packed_centers,
        weights=packed_weights,
        source_views=source_views,
        selected_faces=set(selected_faces),
        distance_scale=scale,
        source_view_count=len(view_paths),
        observation_count=observation_count,
    )


def load_checkpoint_faces(model_path: Path, iteration: int) -> np.ndarray:
    state_path = Path(model_path) / "point_cloud" / f"iteration_{int(iteration)}" / "point_cloud_state_dict.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint topology not found: {state_path}")
    try:
        state = torch.load(state_path, map_location="cpu", mmap=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    if "_triangle_indices" not in state:
        raise KeyError(f"{state_path} missing `_triangle_indices`")
    faces = state["_triangle_indices"].detach().cpu().to(dtype=torch.long).numpy().astype(np.int64, copy=False)
    faces = np.ascontiguousarray(faces)
    del state
    gc.collect()
    return faces


def build_topology_neighbor_alias(
    field: FaceResidualField,
    faces: np.ndarray,
    *,
    neighbor_rings: int,
    max_targets_per_source: int,
    chunk_size: int,
    candidate_faces: set[int] | None = None,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Map target face ids to train-observed residual source face ids.

    The alias is topology-only: target faces can borrow a residual source only
    through shared checkpoint vertices, and the train policy face gate later
    decides whether those proposed target faces are actually allowed.
    """
    n_faces = int(faces.shape[0])
    source_faces = {int(fid) for fid in field.residuals if 0 <= int(fid) < n_faces}
    alias: dict[int, int] = {fid: fid for fid in source_faces}
    source_quality = {fid: float(np.sum(field.weights[fid])) for fid in source_faces}
    source_target_counts: dict[int, int] = {fid: 0 for fid in source_faces}
    frontier: set[int] = set(source_faces)
    ring_growth: list[dict[str, Any]] = []
    rings = max(int(neighbor_rings), 0)
    max_targets = max(int(max_targets_per_source), 0)
    chunk = max(int(chunk_size), 1024)
    if rings <= 0 or not source_faces or max_targets <= 0:
        return alias, {
            "enabled": bool(rings > 0),
            "rings": int(rings),
            "source_faces": int(len(source_faces)),
            "alias_faces": int(len(alias)),
            "propagated_faces": 0,
            "ring_growth": ring_growth,
        }

    candidate_face_list: list[int] | None = None
    if candidate_faces is not None:
        candidate_face_list = sorted(int(fid) for fid in candidate_faces if 0 <= int(fid) < n_faces)

    for ring in range(1, rings + 1):
        valid_frontier = [fid for fid in frontier if 0 <= int(fid) < n_faces]
        if not valid_frontier:
            break
        vertex_to_sources: dict[int, set[int]] = {}
        for target_fid in valid_frontier:
            root = int(alias.get(int(target_fid), int(target_fid)))
            tri = faces[int(target_fid)]
            for vertex in tri.tolist():
                vertex_to_sources.setdefault(int(vertex), set()).add(root)
        if not vertex_to_sources:
            break
        vertex_keys = np.asarray(sorted(vertex_to_sources), dtype=np.int64)
        new_alias: dict[int, int] = {}
        if candidate_face_list is not None:
            candidate_iter = ((target_fid, faces[target_fid]) for target_fid in candidate_face_list)
            for target_fid, tri in candidate_iter:
                if target_fid in alias or target_fid in new_alias:
                    continue
                counts: dict[int, int] = {}
                for vertex in tri.tolist():
                    for root in vertex_to_sources.get(int(vertex), ()):
                        counts[int(root)] = counts.get(int(root), 0) + 1
                if not counts:
                    continue
                ranked = sorted(
                    counts,
                    key=lambda root: (int(counts[root]), float(source_quality.get(root, 0.0)), -int(source_target_counts.get(root, 0))),
                    reverse=True,
                )
                chosen: int | None = None
                for root in ranked:
                    if source_target_counts.get(root, 0) < max_targets:
                        chosen = int(root)
                        break
                if chosen is None:
                    continue
                new_alias[target_fid] = chosen
                source_target_counts[chosen] = source_target_counts.get(chosen, 0) + 1
        else:
            for start in range(0, n_faces, chunk):
                stop = min(start + chunk, n_faces)
                tri_chunk = faces[start:stop]
                candidate_mask = _fast_isin_int(tri_chunk, vertex_keys).any(axis=1)
                if not np.any(candidate_mask):
                    continue
                local_ids = np.nonzero(candidate_mask)[0]
                for local_id in local_ids.tolist():
                    target_fid = int(start + local_id)
                    if target_fid in alias or target_fid in new_alias:
                        continue
                    counts: dict[int, int] = {}
                    for vertex in tri_chunk[local_id].tolist():
                        for root in vertex_to_sources.get(int(vertex), ()):
                            counts[int(root)] = counts.get(int(root), 0) + 1
                    if not counts:
                        continue
                    ranked = sorted(
                        counts,
                        key=lambda root: (int(counts[root]), float(source_quality.get(root, 0.0)), -int(source_target_counts.get(root, 0))),
                        reverse=True,
                    )
                    chosen: int | None = None
                    for root in ranked:
                        if source_target_counts.get(root, 0) < max_targets:
                            chosen = int(root)
                            break
                    if chosen is None:
                        continue
                    new_alias[target_fid] = chosen
                    source_target_counts[chosen] = source_target_counts.get(chosen, 0) + 1
        alias.update(new_alias)
        frontier = set(new_alias)
        ring_growth.append(
            {
                "ring": int(ring),
                "frontier_faces": int(len(valid_frontier)),
                "new_alias_faces": int(len(new_alias)),
                "total_alias_faces": int(len(alias)),
            }
        )
        if not new_alias:
            break

    propagated = len(alias) - len(source_faces)
    return alias, {
        "enabled": True,
        "rings": int(rings),
        "source_faces": int(len(source_faces)),
        "alias_faces": int(len(alias)),
        "propagated_faces": int(propagated),
        "max_targets_per_source": int(max_targets),
        "chunk_size": int(chunk),
        "candidate_faces": int(len(candidate_face_list)) if candidate_face_list is not None else None,
        "ring_growth": ring_growth,
    }


def collect_visible_faces(paths: list[Path], *, max_face_id: int) -> set[int]:
    visible: set[int] = set()
    limit = int(max_face_id)
    for path in paths:
        payload = _load_npz(path)
        if "face_id" not in payload:
            continue
        face = np.array(payload["face_id"], copy=True)
        valid = (face >= 0) & (face < limit)
        if np.any(valid):
            visible.update(int(fid) for fid in np.unique(face[valid]).tolist())
    return visible


def compute_surface_signal(
    face_id: np.ndarray,
    camera_center: np.ndarray,
    field: FaceResidualField,
    *,
    k: int,
    max_abs_residual: float,
    allowed_faces: set[int] | None = None,
    alias_source: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    face_id = np.array(face_id, copy=True) if isinstance(face_id, np.memmap) else np.asarray(face_id)
    h, w = face_id.shape
    signal = np.zeros((3, h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    if allowed_faces is None:
        allowed_faces = set(alias_source) if alias_source is not None else set(field.residuals)
    if not allowed_faces:
        return signal, confidence, {"used_faces": 0, "covered_fraction": 0.0, "mean_confidence": 0.0}
    valid = (face_id >= 0) & _fast_isin_int(face_id, allowed_faces)
    if not np.any(valid):
        return signal, confidence, {"used_faces": 0, "covered_fraction": 0.0, "mean_confidence": 0.0}
    flat_face = np.asarray(face_id).reshape(-1)
    flat_indices = np.flatnonzero(valid.reshape(-1))
    active_fids = flat_face[flat_indices].astype(np.int64, copy=False)
    order = np.argsort(active_fids)
    active_fids = active_fids[order]
    flat_indices = flat_indices[order]
    present, starts = np.unique(active_fids, return_index=True)
    ends = np.r_[starts[1:], len(active_fids)]
    target_center = camera_center.astype(np.float32).reshape(3)
    signal_flat = signal.reshape(3, -1)
    confidence_flat = confidence.reshape(-1)
    used = 0
    propagated = 0
    source_faces_used: set[int] = set()
    source_cache: dict[int, tuple[np.ndarray, float]] = {}
    for fid, start, end in zip(present.tolist(), starts.tolist(), ends.tolist()):
        target_fid = int(fid)
        source_fid = int(alias_source.get(target_fid, target_fid)) if alias_source is not None else target_fid
        if source_fid not in field.residuals:
            continue
        cached = source_cache.get(source_fid)
        if cached is None:
            centers = field.centers[source_fid]
            residuals = field.residuals[source_fid]
            weights = field.weights[source_fid]
            if centers.size == 0:
                continue
            dist = np.linalg.norm(centers - target_center[None, :], axis=1)
            take_count = min(max(int(k), 1), len(dist))
            take = np.argsort(dist)[:take_count]
            view_w = np.exp(-dist[take] / max(float(field.distance_scale), 1e-6)) * np.sqrt(np.maximum(weights[take], 1e-6))
            denom = float(np.sum(view_w))
            if denom <= 0.0:
                continue
            residual = (residuals[take] * view_w[:, None]).sum(axis=0) / denom
            residual = np.clip(residual, -float(max_abs_residual), float(max_abs_residual)).astype(np.float32)
            cached = (residual, float(denom))
            source_cache[source_fid] = cached
        residual, denom = cached
        pixels = flat_indices[start:end]
        signal_flat[:, pixels] = residual[:, None]
        confidence_flat[pixels] = float(denom)
        used += 1
        source_faces_used.add(source_fid)
        if source_fid != target_fid:
            propagated += 1
    covered = confidence > 0.0
    return (
        signal,
        confidence,
        {
            "used_faces": int(used),
            "source_faces_used": int(len(source_faces_used)),
            "propagated_faces_used": int(propagated),
            "covered_fraction": float(np.mean(covered)),
            "mean_confidence": float(np.mean(confidence[covered])) if np.any(covered) else 0.0,
        },
    )


def _image_to_np(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.moveaxis(arr, -1, 0)


def _save_np_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hwc = np.moveaxis(np.clip(image, 0.0, 1.0), 0, -1)
    Image.fromarray((hwc * 255.0 + 0.5).astype(np.uint8)).save(path)


def _torch_image(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device=device).unsqueeze(0)


def _metrics(pred: np.ndarray, gt: np.ndarray, *, device: torch.device, compute_lpips: bool) -> dict[str, float]:
    pred_t = _torch_image(pred, device)
    gt_t = _torch_image(gt, device)
    with torch.no_grad():
        mse = float(torch.mean((pred_t - gt_t) ** 2).detach().cpu().item())
        ssim_value = float(ssim(pred_t, gt_t).detach().cpu().item())
    out = {
        "PSNR": -10.0 * math.log10(max(mse, 1e-12)),
        "SSIM": ssim_value,
    }
    if compute_lpips:
        key = ("vgg", str(device))
        criterion = _LPIPS_CACHE.get(key)
        if criterion is None:
            criterion = LPIPS("vgg").to(device).eval()
            _LPIPS_CACHE[key] = criterion
        with torch.no_grad():
            out["LPIPS"] = float(criterion(pred_t, gt_t).detach().cpu().item())
    return out


def prepare_policy_views(
    field: FaceResidualField,
    policy_paths: list[Path],
    *,
    k: int,
    max_abs_residual: float,
    alias_source: dict[int, int] | None = None,
    fallback_render_dir: Path | None = None,
    fallback_gt_dir: Path | None = None,
) -> list[PreparedPolicyView]:
    prepared: list[PreparedPolicyView] = []
    for path in tqdm(policy_paths, desc="Surface policy-val signals"):
        payload = _load_npz(path)
        required = {"face_id", "camera_center"}
        missing = required - set(payload)
        if missing:
            raise RuntimeError(f"{path} missing policy fields: {sorted(missing)}")
        face_id = np.array(payload["face_id"], copy=True)
        if "rgb_render" in payload:
            base = payload["rgb_render"].astype(np.float32)
        elif fallback_render_dir is not None and (fallback_render_dir / f"{path.stem}.png").is_file():
            base = _image_to_np(fallback_render_dir / f"{path.stem}.png")
        else:
            raise RuntimeError(f"{path} missing rgb_render and no fallback render exists for {path.stem}.png")
        if "rgb_gt" in payload:
            gt = payload["rgb_gt"].astype(np.float32)
        elif fallback_gt_dir is not None and (fallback_gt_dir / f"{path.stem}.png").is_file():
            gt = _image_to_np(fallback_gt_dir / f"{path.stem}.png")
        else:
            raise RuntimeError(f"{path} missing rgb_gt and no fallback gt exists for {path.stem}.png")
        center = payload["camera_center"].astype(np.float32).reshape(3)
        signal, confidence, _ = compute_surface_signal(
            face_id,
            center,
            field,
            k=k,
            max_abs_residual=max_abs_residual,
            alias_source=alias_source,
        )
        prepared.append(
            PreparedPolicyView(
                name=path.stem,
                base=base,
                gt=gt,
                face_id=face_id,
                signal=signal,
                confidence=confidence,
            )
        )
    return prepared


def fit_face_gate(
    prepared: list[PreparedPolicyView],
    *,
    alpha: float,
    min_pixels: int,
    min_gain: float,
) -> tuple[set[int], dict[str, Any]]:
    gain_sum: dict[int, float] = {}
    count_sum: dict[int, int] = {}
    for view in prepared:
        active = view.confidence > 0.0
        if not np.any(active):
            continue
        pred = np.clip(view.base + float(alpha) * view.signal, 0.0, 1.0)
        gain = np.mean((view.base - view.gt) ** 2 - (pred - view.gt) ** 2, axis=0)
        flat_active = np.flatnonzero(active.reshape(-1))
        flat_face = np.asarray(view.face_id).reshape(-1)
        flat_gain = gain.reshape(-1)
        active_fids = flat_face[flat_active].astype(np.int64, copy=False)
        order = np.argsort(active_fids)
        active_fids = active_fids[order]
        flat_active = flat_active[order]
        present, starts = np.unique(active_fids, return_index=True)
        ends = np.r_[starts[1:], len(active_fids)]
        for fid, start, end in zip(present.tolist(), starts.tolist(), ends.tolist()):
            pixels = flat_active[start:end]
            n = int(len(pixels))
            if n <= 0:
                continue
            gain_sum[int(fid)] = gain_sum.get(int(fid), 0.0) + float(np.sum(flat_gain[pixels]))
            count_sum[int(fid)] = count_sum.get(int(fid), 0) + n
    accepted = {
        fid
        for fid, total in gain_sum.items()
        if count_sum.get(fid, 0) >= int(min_pixels) and (total / max(count_sum.get(fid, 0), 1)) > float(min_gain)
    }
    means = [gain_sum[fid] / max(count_sum.get(fid, 0), 1) for fid in accepted]
    return accepted, {
        "candidate_faces": int(len(gain_sum)),
        "accepted_faces": int(len(accepted)),
        "min_pixels": int(min_pixels),
        "min_gain": float(min_gain),
        "mean_accepted_gain": float(np.mean(means)) if means else 0.0,
    }


def mask_prepared_views_by_faces(prepared: list[PreparedPolicyView], allowed_faces: set[int]) -> list[PreparedPolicyView]:
    masked: list[PreparedPolicyView] = []
    for view in prepared:
        keep = (view.confidence > 0.0) & _fast_isin_int(view.face_id, allowed_faces)
        signal = np.array(view.signal, copy=True)
        confidence = np.array(view.confidence, copy=True)
        signal[:, ~keep] = 0.0
        confidence[~keep] = 0.0
        masked.append(
            PreparedPolicyView(
                name=view.name,
                base=view.base,
                gt=view.gt,
                face_id=view.face_id,
                signal=signal,
                confidence=confidence,
            )
        )
    return masked


def maybe_recalibrate_with_provisional_face_gate(
    prepared: list[PreparedPolicyView],
    alpha: float,
    calibration: dict[str, Any],
    alpha_grid: list[float],
    *,
    enabled: bool,
    min_pixels: int,
    min_gain: float,
    objective: str,
    ssim_weight: float,
    lpips_weight: float,
    min_dpsnr: float,
    min_dssim: float,
    max_dlpips: float,
    compute_lpips: bool,
    device: torch.device,
) -> tuple[float, dict[str, Any], dict[str, Any] | None]:
    if not enabled or float(alpha) > 0.0:
        return alpha, calibration, None
    chosen = calibration.get("chosen_row") or {}
    provisional_alpha = float(chosen.get("alpha") or 0.0)
    provisional_dpsnr = float(chosen.get("dPSNR") or 0.0)
    if provisional_alpha <= 0.0 or provisional_dpsnr <= 0.0:
        return alpha, calibration, None
    gated, gate = fit_face_gate(
        prepared,
        alpha=provisional_alpha,
        min_pixels=min_pixels,
        min_gain=min_gain,
    )
    report: dict[str, Any] = {
        "enabled": True,
        "provisional_alpha": provisional_alpha,
        "provisional_dPSNR": provisional_dpsnr,
        "gate": gate,
    }
    if not gated:
        report["reason"] = "no_provisional_faces"
        updated = dict(calibration)
        updated["provisional_face_gate_recalibration"] = report
        return alpha, updated, report
    masked = mask_prepared_views_by_faces(prepared, gated)
    gated_alpha, gated_calibration = calibrate_policy(
        masked,
        alpha_grid,
        objective=objective,
        ssim_weight=ssim_weight,
        lpips_weight=lpips_weight,
        min_dpsnr=min_dpsnr,
        min_dssim=min_dssim,
        max_dlpips=max_dlpips,
        compute_lpips=compute_lpips,
        device=device,
    )
    report["gated_calibration"] = gated_calibration
    updated = dict(calibration)
    updated["provisional_face_gate_recalibration"] = report
    if float(gated_alpha) > 0.0:
        updated["reason"] = "provisional_face_gate_recalibrated"
        updated["accepted"] = True
        updated["alpha"] = float(gated_alpha)
        return float(gated_alpha), updated, report
    return alpha, updated, report


def calibrate_policy(
    prepared: list[PreparedPolicyView],
    alpha_grid: list[float],
    *,
    objective: str,
    ssim_weight: float,
    lpips_weight: float,
    min_dpsnr: float,
    min_dssim: float,
    max_dlpips: float,
    compute_lpips: bool,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    rows = []
    if not prepared:
        return 0.0, {"reason": "no_policy_views", "rows": []}
    base_metrics = []
    for view in prepared:
        base_metrics.append(_metrics(view.base, view.gt, device=device, compute_lpips=compute_lpips))
    base_mean = {
        key: float(np.mean([m[key] for m in base_metrics]))
        for key in ("PSNR", "SSIM", *(() if not compute_lpips else ("LPIPS",)))
    }
    best_raw: tuple[float, float, dict[str, Any]] | None = None
    best_pass: tuple[float, float, dict[str, Any]] | None = None
    for alpha in alpha_grid:
        metric_rows = []
        for view in prepared:
            pred = np.clip(view.base + float(alpha) * view.signal, 0.0, 1.0)
            metric_rows.append(_metrics(pred, view.gt, device=device, compute_lpips=compute_lpips))
        mean = {
            key: float(np.mean([m[key] for m in metric_rows]))
            for key in ("PSNR", "SSIM", *(() if not compute_lpips else ("LPIPS",)))
        }
        dpsnr = mean["PSNR"] - base_mean["PSNR"]
        dssim = mean["SSIM"] - base_mean["SSIM"]
        dlpips = mean.get("LPIPS", base_mean.get("LPIPS", 0.0)) - base_mean.get("LPIPS", 0.0)
        if objective == "balanced":
            score = dpsnr + float(ssim_weight) * dssim - (float(lpips_weight) * dlpips if compute_lpips else 0.0)
        else:
            score = dpsnr
        row = {
            "alpha": float(alpha),
            "PSNR": mean["PSNR"],
            "SSIM": mean["SSIM"],
            "LPIPS": mean.get("LPIPS"),
            "base_PSNR": base_mean["PSNR"],
            "base_SSIM": base_mean["SSIM"],
            "base_LPIPS": base_mean.get("LPIPS"),
            "dPSNR": dpsnr,
            "dSSIM": dssim,
            "dLPIPS": dlpips if compute_lpips else None,
            "selection_score": score,
        }
        rows.append(row)
        rank = (float(score), float(alpha))
        if best_raw is None or rank > (best_raw[0], best_raw[1]):
            best_raw = (float(score), float(alpha), row)
        passes = (
            row["dPSNR"] >= float(min_dpsnr)
            and row["dSSIM"] >= float(min_dssim)
            and (not compute_lpips or row["dLPIPS"] <= float(max_dlpips))
            and float(row["alpha"]) > 0.0
        )
        if passes and (best_pass is None or rank > (best_pass[0], best_pass[1])):
            best_pass = (float(score), float(alpha), row)
    assert best_raw is not None
    accepted = best_pass is not None
    chosen = dict((best_pass or best_raw)[2])
    raw_chosen = dict(best_raw[2])
    alpha = float(chosen["alpha"]) if accepted else 0.0
    reason = "policy_val_pass" if accepted else "policy_val_guard_rejected"
    if accepted and raw_chosen.get("alpha") != chosen.get("alpha"):
        reason = "policy_val_best_guarded_pass"
    return alpha, {
        "reason": reason,
        "alpha": alpha,
        "accepted": bool(accepted),
        "chosen_row": chosen,
        "raw_chosen_row": raw_chosen,
        "rows": rows,
        "base_mean": base_mean,
    }


def _copy_gt(base_method_dir: Path, out_method_dir: Path) -> None:
    src = base_method_dir / "gt"
    dst = out_method_dir / "gt"
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.is_file():
            target = dst / path.name
            if not target.exists():
                shutil.copy2(path, target)


def apply_target(
    args: argparse.Namespace,
    field: FaceResidualField,
    *,
    alpha: float,
    allowed_faces: set[int] | None,
    alias_source: dict[int, int] | None = None,
) -> dict[str, Any]:
    base_model = Path(args.base_model_path)
    output_model = Path(args.output_model_path) if args.output_model_path else base_model
    base_method_dir = base_model / args.target_split / args.base_method_name
    out_method_dir = output_model / args.target_split / args.method_name
    out_render_dir = out_method_dir / "renders"
    out_render_dir.mkdir(parents=True, exist_ok=True)
    _copy_gt(base_method_dir, out_method_dir)

    render_paths = {path.stem: path for path in (base_method_dir / "renders").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    map_paths = sorted(Path(args.target_surface_map_dir).glob("*.npz"))
    infos = []
    for map_path in tqdm(map_paths, desc=f"Surface lumigraph {args.target_split}"):
        payload = _load_npz(map_path)
        face_id = np.array(payload["face_id"], copy=True)
        center = payload["camera_center"].astype(np.float32).reshape(3)
        render_path = render_paths.get(map_path.stem)
        if render_path is None:
            raise FileNotFoundError(f"missing base render for surface map {map_path.name} under {base_method_dir / 'renders'}")
        base = _image_to_np(render_path)
        if base.shape[1:] != face_id.shape:
            raise RuntimeError(f"base render and face map shape mismatch for {map_path}: {base.shape[1:]} vs {face_id.shape}")
        signal, confidence, info = compute_surface_signal(
            face_id,
            center,
            field,
            k=int(args.k),
            max_abs_residual=float(args.max_abs_residual),
            allowed_faces=allowed_faces,
            alias_source=alias_source,
        )
        adapted = np.clip(base + float(alpha) * signal, 0.0, 1.0)
        _save_np_image(adapted, out_render_dir / render_path.name)
        infos.append({"frame": render_path.stem, **info})
    return {
        "target_frames": int(len(infos)),
        "mean_covered_fraction": float(np.mean([x["covered_fraction"] for x in infos])) if infos else 0.0,
        "mean_confidence": float(np.mean([x["mean_confidence"] for x in infos])) if infos else 0.0,
        "mean_used_faces": float(np.mean([x["used_faces"] for x in infos])) if infos else 0.0,
        "mean_propagated_faces_used": float(np.mean([x.get("propagated_faces_used", 0) for x in infos])) if infos else 0.0,
        "frames": infos,
        "output_method_dir": str(out_method_dir),
    }


def _maybe_wandb(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[SurfaceLumigraph] W&B unavailable, skipping log: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=args.wandb_mode,
        config={
            "base_model_path": str(args.base_model_path),
            "evidence_dir": str(args.evidence_dir),
            "target_surface_map_dir": str(args.target_surface_map_dir),
            "target_split": args.target_split,
            "base_method_name": args.base_method_name,
            "method_name": args.method_name,
            "top_k": args.top_k,
            "min_view_hits": args.min_view_hits,
            "min_consistency": args.min_consistency,
            "min_pixel_count": args.min_pixel_count,
            "high_error_quantile": args.high_error_quantile,
            "k": args.k,
            "policy_val_stride": args.policy_val_stride,
            "consensus_policy_strides": args.consensus_policy_strides,
            "face_gate": args.face_gate,
            "neighbor_rings": args.neighbor_rings,
            "neighbor_max_targets_per_source": args.neighbor_max_targets_per_source,
            "consensus_face_gate": args.consensus_face_gate,
            "provisional_face_gate_recalibrate": args.provisional_face_gate_recalibrate,
        },
    )
    flat = {
        "surface_lumigraph/alpha": float(report.get("alpha", 0.0)),
        "surface_lumigraph/selected_faces": int(report.get("selected_faces", 0)),
        "surface_lumigraph/field_faces": int(report.get("field_faces", 0)),
        "surface_lumigraph/accepted_faces": int(report.get("accepted_faces", 0)),
        "surface_lumigraph/alias_faces": int(report.get("topology_alias", {}).get("alias_faces", 0)),
        "surface_lumigraph/propagated_faces": int(report.get("topology_alias", {}).get("propagated_faces", 0)),
        "surface_lumigraph/target_frames": int(report.get("target", {}).get("target_frames", 0)),
        "surface_lumigraph/mean_covered_fraction": float(report.get("target", {}).get("mean_covered_fraction", 0.0)),
        "surface_lumigraph/mean_used_faces": float(report.get("target", {}).get("mean_used_faces", 0.0)),
        "surface_lumigraph/mean_propagated_faces_used": float(
            report.get("target", {}).get("mean_propagated_faces_used", 0.0)
        ),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    topology_faces: np.ndarray | None = None
    topology_report: dict[str, Any] = {"enabled": False}
    if int(args.neighbor_rings) > 0:
        topology_faces = load_checkpoint_faces(args.base_model_path, int(args.iteration))
    selected_faces = read_selected_faces(
        args.evidence_dir,
        top_k=args.top_k,
        min_view_hits=args.min_view_hits,
        min_consistency=args.min_consistency,
        min_pixel_count=args.min_pixel_count,
        max_face_id=int(topology_faces.shape[0]) if topology_faces is not None else None,
    )
    all_views = _view_paths(args.evidence_dir)
    fit_views, policy_views = split_policy_views(all_views, args.policy_val_stride)
    visible_face_candidates: set[int] | None = None
    if topology_faces is not None:
        target_map_paths = sorted(Path(args.target_surface_map_dir).glob("*.npz"))
        visible_face_candidates = collect_visible_faces(
            [*all_views, *target_map_paths],
            max_face_id=int(topology_faces.shape[0]),
        )
    field = build_field(
        fit_views,
        selected_faces,
        min_alpha=args.min_alpha,
        high_error_quantile=args.high_error_quantile,
        max_abs_residual=args.max_abs_residual,
        distance_scale=args.distance_scale,
    )
    alias_source: dict[int, int] | None = None
    if int(args.neighbor_rings) > 0:
        assert topology_faces is not None
        alias_source, topology_report = build_topology_neighbor_alias(
            field,
            topology_faces,
            neighbor_rings=int(args.neighbor_rings),
            max_targets_per_source=int(args.neighbor_max_targets_per_source),
            chunk_size=int(args.neighbor_chunk_size),
            candidate_faces=visible_face_candidates,
        )
    prepared = prepare_policy_views(
        field,
        policy_views,
        k=args.k,
        max_abs_residual=args.max_abs_residual,
        alias_source=alias_source,
        fallback_render_dir=args.base_model_path / "train" / args.base_method_name / "renders",
        fallback_gt_dir=args.base_model_path / "train" / args.base_method_name / "gt",
    )
    alpha, calibration = calibrate_policy(
        prepared,
        _parse_alpha_grid(args.alpha_grid),
        objective=args.policy_objective,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        min_dpsnr=args.min_policy_dpsnr,
        min_dssim=args.min_policy_dssim,
        max_dlpips=args.max_policy_dlpips,
        compute_lpips=bool(args.calib_lpips),
        device=device,
    )
    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    alpha, calibration, _ = maybe_recalibrate_with_provisional_face_gate(
        prepared,
        alpha,
        calibration,
        alpha_grid,
        enabled=bool(args.provisional_face_gate_recalibrate and args.face_gate),
        min_pixels=args.min_face_gate_pixels,
        min_gain=args.min_face_gate_gain,
        objective=args.policy_objective,
        ssim_weight=args.ssim_weight,
        lpips_weight=args.lpips_weight,
        min_dpsnr=args.min_policy_dpsnr,
        min_dssim=args.min_policy_dssim,
        max_dlpips=args.max_policy_dlpips,
        compute_lpips=bool(args.calib_lpips),
        device=device,
    )
    consensus_reports: list[dict[str, Any]] = []
    consensus_alphas = [float(alpha)]
    consensus_gated_faces: list[set[int]] = []
    for stride in _parse_stride_list(args.consensus_policy_strides):
        if int(stride) == int(args.policy_val_stride):
            continue
        c_fit_views, c_policy_views = split_policy_views(all_views, stride)
        c_field = build_field(
            c_fit_views,
            selected_faces,
            min_alpha=args.min_alpha,
            high_error_quantile=args.high_error_quantile,
            max_abs_residual=args.max_abs_residual,
            distance_scale=args.distance_scale,
        )
        c_alias_source: dict[int, int] | None = None
        c_topology_report: dict[str, Any] = {"enabled": False}
        if topology_faces is not None and int(args.neighbor_rings) > 0:
            c_alias_source, c_topology_report = build_topology_neighbor_alias(
                c_field,
                topology_faces,
                neighbor_rings=int(args.neighbor_rings),
                max_targets_per_source=int(args.neighbor_max_targets_per_source),
                chunk_size=int(args.neighbor_chunk_size),
                candidate_faces=visible_face_candidates,
            )
        c_prepared = prepare_policy_views(
            c_field,
            c_policy_views,
            k=args.k,
            max_abs_residual=args.max_abs_residual,
            alias_source=c_alias_source,
            fallback_render_dir=args.base_model_path / "train" / args.base_method_name / "renders",
            fallback_gt_dir=args.base_model_path / "train" / args.base_method_name / "gt",
        )
        c_alpha, c_calibration = calibrate_policy(
            c_prepared,
            alpha_grid,
            objective=args.policy_objective,
            ssim_weight=args.ssim_weight,
            lpips_weight=args.lpips_weight,
            min_dpsnr=args.min_policy_dpsnr,
            min_dssim=args.min_policy_dssim,
            max_dlpips=args.max_policy_dlpips,
            compute_lpips=bool(args.calib_lpips),
            device=device,
        )
        c_alpha, c_calibration, _ = maybe_recalibrate_with_provisional_face_gate(
            c_prepared,
            c_alpha,
            c_calibration,
            alpha_grid,
            enabled=bool(args.provisional_face_gate_recalibrate and args.face_gate),
            min_pixels=args.min_face_gate_pixels,
            min_gain=args.min_face_gate_gain,
            objective=args.policy_objective,
            ssim_weight=args.ssim_weight,
            lpips_weight=args.lpips_weight,
            min_dpsnr=args.min_policy_dpsnr,
            min_dssim=args.min_policy_dssim,
            max_dlpips=args.max_policy_dlpips,
            compute_lpips=bool(args.calib_lpips),
            device=device,
        )
        c_gate_report: dict[str, Any] | None = None
        if bool(args.face_gate) and bool(args.consensus_face_gate) and float(c_alpha) > 0.0:
            c_gated, c_gate = fit_face_gate(
                c_prepared,
                alpha=c_alpha,
                min_pixels=args.min_face_gate_pixels,
                min_gain=args.min_face_gate_gain,
            )
            consensus_gated_faces.append(set(c_gated))
            c_gate_report = c_gate
        consensus_reports.append(
            {
                "stride": int(stride),
                "fit_views": [p.stem for p in c_fit_views],
                "policy_val_views": [p.stem for p in c_policy_views],
                "alpha": float(c_alpha),
                "calibration": c_calibration,
                "topology_alias": c_topology_report,
                "face_gate": c_gate_report,
            }
        )
        consensus_alphas.append(float(c_alpha))
    if consensus_reports:
        pre_consensus_alpha = float(alpha)
        calibration = dict(calibration)
        calibration["pre_consensus_alpha"] = pre_consensus_alpha
        calibration["consensus_policy_strides"] = [r["stride"] for r in consensus_reports]
        if any(a <= 0.0 for a in consensus_alphas):
            alpha = 0.0
            calibration["reason"] = "policy_consensus_rejected"
            calibration["accepted"] = False
            calibration["alpha"] = 0.0
        else:
            alpha = float(min(consensus_alphas))
            calibration["reason"] = "policy_consensus_pass"
            calibration["accepted"] = True
            calibration["alpha"] = alpha
            calibration["consensus_alpha"] = alpha
    allowed_faces: set[int] | None = set(field.residuals)
    face_gate_report: dict[str, Any] = {"enabled": bool(args.face_gate)}
    if bool(args.face_gate) and float(alpha) > 0.0:
        gated, gate = fit_face_gate(
            prepared,
            alpha=alpha,
            min_pixels=args.min_face_gate_pixels,
            min_gain=args.min_face_gate_gain,
        )
        allowed_faces = set(gated)
        face_gate_report.update(gate)
        if bool(args.consensus_face_gate) and consensus_gated_faces:
            before = len(allowed_faces)
            for gated_split in consensus_gated_faces:
                allowed_faces &= set(gated_split)
            face_gate_report.update(
                {
                    "consensus_intersection_enabled": True,
                    "pre_consensus_accepted_faces": int(before),
                    "post_consensus_accepted_faces": int(len(allowed_faces)),
                    "consensus_gate_splits": int(len(consensus_gated_faces)),
                }
            )
    elif float(alpha) <= 0.0:
        allowed_faces = set()
        face_gate_report.update({"accepted_faces": 0, "reason": "alpha_zero"})
    target_report = apply_target(args, field, alpha=alpha, allowed_faces=allowed_faces, alias_source=alias_source)
    report = {
        "method": "ECSR Surface Residual Lumigraph",
        "base_model_path": str(args.base_model_path),
        "evidence_dir": str(args.evidence_dir),
        "target_surface_map_dir": str(args.target_surface_map_dir),
        "target_split": args.target_split,
        "base_method": args.base_method_name,
        "method_name": args.method_name,
        "selected_faces": int(len(selected_faces)),
        "field_faces": int(len(field.residuals)),
        "field_observations": int(field.observation_count),
        "source_view_count": int(field.source_view_count),
        "distance_scale": float(field.distance_scale),
        "topology_alias": topology_report,
        "visible_face_candidates": int(len(visible_face_candidates or set())),
        "fit_views": [p.stem for p in fit_views],
        "policy_val_views": [p.stem for p in policy_views],
        "alpha": float(alpha),
        "calibration": calibration,
        "consensus_policy": consensus_reports,
        "face_gate": face_gate_report,
        "accepted_faces": int(len(allowed_faces or set())),
        "target": target_report,
    }
    out_method_dir = Path(target_report["output_method_dir"])
    (out_method_dir / "surface_lumigraph_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        "# ECSR Surface Residual Lumigraph Report",
        "",
        f"- base model: `{args.base_model_path}`",
        f"- evidence: `{args.evidence_dir}`",
        f"- base method: `{args.base_method_name}`",
        f"- method: `{args.method_name}`",
        f"- selected faces: `{len(selected_faces)}`",
        f"- field faces: `{len(field.residuals)}`",
        f"- topology alias faces: `{topology_report.get('alias_faces', len(field.residuals))}`",
        f"- topology propagated faces: `{topology_report.get('propagated_faces', 0)}`",
        f"- policy alpha: `{alpha:.4f}`",
        f"- calibration: `{calibration.get('reason', 'unknown')}`",
        f"- face gate accepted: `{len(allowed_faces or set())}`",
        f"- target coverage: `{target_report['mean_covered_fraction']:.4f}`",
        "",
        "This path stores the residual signal on train-observed surface face ids. With topology aliasing enabled, adjacent target faces are only proposals; train-only policy and face gates decide the final allowed target faces. Held-out GT is not used for policy selection.",
    ]
    (out_method_dir / "surface_lumigraph_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    _maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

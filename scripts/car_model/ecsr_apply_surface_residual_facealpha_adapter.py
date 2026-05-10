#!/usr/bin/env python3
"""Apply topology-propagated surface residuals with train-fitted per-face alpha.

This is a higher-capacity successor to the global-alpha surface lumigraph.  It
keeps the same train/test boundary: residuals are fitted from train evidence,
topology propagation uses checkpoint triangles only, and held-out renders use
target surface maps without held-out GT.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_lumigraph_adapter import (
    PreparedPolicyView,
    _fast_isin_int,
    _image_to_np,
    _load_npz,
    _metrics,
    _parse_stride_list,
    _save_np_image,
    _view_paths,
    build_field,
    build_topology_neighbor_alias,
    collect_visible_faces,
    compute_surface_signal,
    load_checkpoint_faces,
    prepare_policy_views,
    read_selected_faces,
    split_policy_views,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--evidence_dir", type=Path, required=True)
    parser.add_argument("--target_surface_map_dir", type=Path, required=True)
    parser.add_argument("--output_model_path", type=Path, default=None)
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--base_method_name", required=True)
    parser.add_argument("--method_name", default="ours_26000_surface_residual_facealpha")
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
    parser.add_argument("--consensus_policy_strides", default="")
    parser.add_argument("--neighbor_rings", type=int, default=1)
    parser.add_argument("--neighbor_max_targets_per_source", type=int, default=64)
    parser.add_argument("--neighbor_chunk_size", type=int, default=524288)
    parser.add_argument("--face_alpha_max", type=float, default=0.25)
    parser.add_argument("--face_alpha_min_pixels", type=int, default=32)
    parser.add_argument("--face_alpha_min_gain", type=float, default=0.0)
    parser.add_argument("--face_alpha_ridge", type=float, default=1e-4)
    parser.add_argument("--face_alpha_shrink_pixels", type=float, default=64.0)
    parser.add_argument(
        "--face_alpha_edge_weight",
        type=float,
        default=0.0,
        help=(
            "Optional structure-aware fitting weight. Positive values add a "
            "train-only local-gradient objective that prefers face residuals "
            "which also improve edge/texture differences around covered pixels."
        ),
    )
    parser.add_argument(
        "--face_alpha_edge_stride",
        type=int,
        default=4,
        help="Deterministic pixel stride for the edge-aware objective.",
    )
    parser.add_argument("--min_policy_dpsnr", type=float, default=0.0)
    parser.add_argument("--min_policy_dssim", type=float, default=0.0)
    parser.add_argument("--calib_lpips", action="store_true")
    parser.add_argument("--max_policy_dlpips", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "mesh-splatting-ecsr"))
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def _accumulate_grouped(
    num: dict[int, float],
    den: dict[int, float],
    count: dict[int, int] | None,
    fids: np.ndarray,
    dots: np.ndarray,
    norms: np.ndarray,
    *,
    count_samples: bool,
) -> int:
    if fids.size == 0:
        return 0
    fids = fids.astype(np.int64, copy=False).reshape(-1)
    valid = fids >= 0
    if not np.all(valid):
        fids = fids[valid]
        dots = dots[valid]
        norms = norms[valid]
    if fids.size == 0:
        return 0
    max_fid = int(np.max(fids))
    # Face ids are dense checkpoint triangle indices in this project.  Bincount
    # avoids repeated full-image sorts when fitting edge-aware objectives.
    if max_fid <= 20_000_000:
        sums_num = np.bincount(fids, weights=dots.astype(np.float64, copy=False), minlength=max_fid + 1)
        sums_den = np.bincount(fids, weights=norms.astype(np.float64, copy=False), minlength=max_fid + 1)
        present = np.nonzero(sums_den > 0.0)[0]
        if count_samples and count is not None:
            sums_count = np.bincount(fids, minlength=max_fid + 1)
        for fid in present.tolist():
            num[fid] = num.get(fid, 0.0) + float(sums_num[fid])
            den[fid] = den.get(fid, 0.0) + float(sums_den[fid])
            if count_samples and count is not None:
                count[fid] = count.get(fid, 0) + int(sums_count[fid])
        return int(fids.size)
    order = np.argsort(fids)
    fids = fids[order]
    dots = dots[order]
    norms = norms[order]
    unique, starts = np.unique(fids, return_index=True)
    ends = np.r_[starts[1:], len(fids)]
    samples = 0
    for fid, start, end in zip(unique.tolist(), starts.tolist(), ends.tolist()):
        n = int(end - start)
        if n <= 0:
            continue
        fid = int(fid)
        num[fid] = num.get(fid, 0.0) + float(np.sum(dots[start:end]))
        den[fid] = den.get(fid, 0.0) + float(np.sum(norms[start:end]))
        if count_samples and count is not None:
            count[fid] = count.get(fid, 0) + n
        samples += n
    return samples


def _ensure_dense(arr: np.ndarray, size: int, dtype: np.dtype | type) -> np.ndarray:
    size = int(size)
    if arr.size >= size:
        return arr
    out = np.zeros(size, dtype=dtype)
    if arr.size:
        out[: arr.size] = arr
    return out


def _accumulate_dense(
    num: np.ndarray,
    den: np.ndarray,
    count: np.ndarray,
    fids: np.ndarray,
    dots: np.ndarray,
    norms: np.ndarray,
    *,
    count_samples: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if fids.size == 0:
        return num, den, count, 0
    fids = fids.astype(np.int64, copy=False).reshape(-1)
    valid = fids >= 0
    if not np.all(valid):
        fids = fids[valid]
        dots = dots[valid]
        norms = norms[valid]
    if fids.size == 0:
        return num, den, count, 0
    max_fid = int(np.max(fids))
    size = max_fid + 1
    num = _ensure_dense(num, size, np.float64)
    den = _ensure_dense(den, size, np.float64)
    count = _ensure_dense(count, size, np.int64)
    num[:size] += np.bincount(fids, weights=dots.astype(np.float64, copy=False), minlength=size)
    den[:size] += np.bincount(fids, weights=norms.astype(np.float64, copy=False), minlength=size)
    if count_samples:
        count[:size] += np.bincount(fids, minlength=size).astype(np.int64, copy=False)
    return num, den, count, int(fids.size)


def _edge_objective_samples(
    view: PreparedPolicyView,
    target: np.ndarray,
    active: np.ndarray,
    *,
    stride: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-current-pixel edge terms assigned to the current face.

    The residual field is face-attached and mostly piecewise constant.  Same-face
    gradients therefore do not expose structural risk.  These terms compare each
    covered pixel to its four image neighbors while assigning the edge objective
    to the covered pixel's face.  This is a conservative train-only surrogate for
    SSIM: a face residual is favored only when it also moves local contrast toward
    the GT image instead of merely lowering RGB MSE.
    """
    face_id = np.asarray(view.face_id)
    signal = view.signal
    terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    step = max(int(stride), 1)
    slices = [
        ((slice(None, None, step), slice(None, -1, step)), (slice(None, None, step), slice(1, None, step))),
        ((slice(None, None, step), slice(1, None, step)), (slice(None, None, step), slice(None, -1, step))),
        ((slice(None, -1, step), slice(None, None, step)), (slice(1, None, step), slice(None, None, step))),
        ((slice(1, None, step), slice(None, None, step)), (slice(None, -1, step), slice(None, None, step))),
    ]
    for cur, neigh in slices:
        valid = active[cur]
        if not np.any(valid):
            continue
        cur_signal = signal[(slice(None), *cur)]
        cur_target = target[(slice(None), *cur)]
        neigh_target = target[(slice(None), *neigh)]
        target_edge = cur_target - neigh_target
        dots = np.sum(cur_signal * target_edge, axis=0)[valid].reshape(-1)
        norms = np.sum(cur_signal * cur_signal, axis=0)[valid].reshape(-1)
        fids = face_id[cur][valid].astype(np.int64, copy=False).reshape(-1)
        terms.append((fids, dots.astype(np.float64, copy=False), norms.astype(np.float64, copy=False)))
    return terms


def fit_face_alphas(
    prepared: list[PreparedPolicyView],
    *,
    alpha_max: float,
    min_pixels: int,
    min_gain: float,
    ridge: float,
    shrink_pixels: float,
    edge_weight: float,
    edge_stride: int,
) -> tuple[dict[int, float], dict[str, Any]]:
    num_arr = np.zeros(0, dtype=np.float64)
    den_arr = np.zeros(0, dtype=np.float64)
    count_arr = np.zeros(0, dtype=np.int64)
    edge_samples = 0
    for view in prepared:
        active = view.confidence > 0.0
        if not np.any(active):
            continue
        target = view.gt - view.base
        dot = np.sum(view.signal * target, axis=0)
        norm = np.sum(view.signal * view.signal, axis=0)
        flat_active = np.flatnonzero(active.reshape(-1))
        flat_face = np.asarray(view.face_id).reshape(-1)
        flat_dot = dot.reshape(-1)
        flat_norm = norm.reshape(-1)
        fids = flat_face[flat_active].astype(np.int64, copy=False)
        num_arr, den_arr, count_arr, _ = _accumulate_dense(
            num_arr,
            den_arr,
            count_arr,
            fids,
            flat_dot[flat_active],
            flat_norm[flat_active],
            count_samples=True,
        )
        if float(edge_weight) > 0.0:
            for edge_fids, edge_dot, edge_norm in _edge_objective_samples(view, target, active, stride=edge_stride):
                num_arr, den_arr, count_arr, samples = _accumulate_dense(
                    num_arr,
                    den_arr,
                    count_arr,
                    edge_fids,
                    float(edge_weight) * edge_dot,
                    float(edge_weight) * edge_norm,
                    count_samples=False,
                )
                edge_samples += samples

    alphas: dict[int, float] = {}
    gains: list[float] = []
    raw_alphas: list[float] = []
    candidate = np.nonzero(count_arr > 0)[0]
    if candidate.size:
        n = count_arr[candidate].astype(np.float64, copy=False)
        numerator = num_arr[candidate]
        denominator = den_arr[candidate]
        denom = denominator + float(ridge)
        raw = np.clip(numerator / np.maximum(denom, 1e-12), 0.0, float(alpha_max))
        shrink = n / (n + max(float(shrink_pixels), 0.0))
        alpha = raw * shrink
        gain_sum = 2.0 * alpha * numerator - (alpha * alpha) * denominator
        mean_gain = gain_sum / np.maximum(n, 1.0)
        keep = (n >= int(min_pixels)) & (denom > 0.0) & (raw > 0.0) & (mean_gain > float(min_gain))
        kept_faces = candidate[keep]
        kept_alpha = alpha[keep]
        kept_gain = mean_gain[keep]
        kept_raw = raw[keep]
        alphas = {int(fid): float(val) for fid, val in zip(kept_faces.tolist(), kept_alpha.tolist())}
        gains = [float(x) for x in kept_gain.tolist()]
        raw_alphas = [float(x) for x in kept_raw.tolist()]

    return alphas, {
        "candidate_faces": int(candidate.size),
        "accepted_faces": int(len(alphas)),
        "min_pixels": int(min_pixels),
        "alpha_max": float(alpha_max),
        "ridge": float(ridge),
        "shrink_pixels": float(shrink_pixels),
        "edge_weight": float(edge_weight),
        "edge_stride": int(max(edge_stride, 1)),
        "edge_samples": int(edge_samples),
        "mean_alpha": float(np.mean(list(alphas.values()))) if alphas else 0.0,
        "mean_raw_alpha": float(np.mean(raw_alphas)) if raw_alphas else 0.0,
        "mean_gain": float(np.mean(gains)) if gains else 0.0,
    }


def intersect_face_alphas(alpha_sets: list[dict[int, float]]) -> dict[int, float]:
    if not alpha_sets:
        return {}
    common = set(alpha_sets[0])
    for item in alpha_sets[1:]:
        common &= set(item)
    return {fid: min(float(item[fid]) for item in alpha_sets) for fid in common}


def scale_signal_by_face_alpha(face_id: np.ndarray, signal: np.ndarray, face_alphas: dict[int, float]) -> np.ndarray:
    if not face_alphas:
        return np.zeros_like(signal, dtype=np.float32)
    out = np.zeros_like(signal, dtype=np.float32)
    valid = (face_id >= 0) & _fast_isin_int(face_id, set(face_alphas))
    if not np.any(valid):
        return out
    flat_face = np.asarray(face_id).reshape(-1)
    flat_idx = np.flatnonzero(valid.reshape(-1))
    fids = flat_face[flat_idx].astype(np.int64, copy=False)
    order = np.argsort(fids)
    fids = fids[order]
    flat_idx = flat_idx[order]
    unique, starts = np.unique(fids, return_index=True)
    ends = np.r_[starts[1:], len(fids)]
    out_flat = out.reshape(3, -1)
    signal_flat = signal.reshape(3, -1)
    for fid, start, end in zip(unique.tolist(), starts.tolist(), ends.tolist()):
        alpha = float(face_alphas.get(int(fid), 0.0))
        if alpha <= 0.0:
            continue
        pixels = flat_idx[start:end]
        out_flat[:, pixels] = alpha * signal_flat[:, pixels]
    return out


def evaluate_policy(
    prepared: list[PreparedPolicyView],
    face_alphas: dict[int, float],
    *,
    device: torch.device,
    compute_lpips: bool,
) -> dict[str, Any]:
    base_rows = []
    adapted_rows = []
    for view in prepared:
        base_rows.append(_metrics(view.base, view.gt, device=device, compute_lpips=compute_lpips))
        delta = scale_signal_by_face_alpha(view.face_id, view.signal, face_alphas)
        adapted = np.clip(view.base + delta, 0.0, 1.0)
        adapted_rows.append(_metrics(adapted, view.gt, device=device, compute_lpips=compute_lpips))
    keys = ("PSNR", "SSIM", *(() if not compute_lpips else ("LPIPS",)))
    base = {key: float(np.mean([row[key] for row in base_rows])) for key in keys}
    adapted = {key: float(np.mean([row[key] for row in adapted_rows])) for key in keys}
    return {
        "base": base,
        "adapted": adapted,
        "dPSNR": adapted["PSNR"] - base["PSNR"],
        "dSSIM": adapted["SSIM"] - base["SSIM"],
        "dLPIPS": (adapted.get("LPIPS", 0.0) - base.get("LPIPS", 0.0)) if compute_lpips else None,
    }


def passes_policy(report: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        report["dPSNR"] >= float(args.min_policy_dpsnr)
        and report["dSSIM"] >= float(args.min_policy_dssim)
        and (not bool(args.calib_lpips) or report["dLPIPS"] <= float(args.max_policy_dlpips))
    )


def _copy_gt(base_method_dir: Path, out_method_dir: Path) -> None:
    src = base_method_dir / "gt"
    dst = out_method_dir / "gt"
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.is_file():
            target = dst / path.name
            if not target.exists():
                shutil.copy2(path, target)


def apply_target(args: argparse.Namespace, field: Any, face_alphas: dict[int, float], alias_source: dict[int, int] | None) -> dict[str, Any]:
    base_model = Path(args.base_model_path)
    output_model = Path(args.output_model_path) if args.output_model_path else base_model
    base_method_dir = base_model / args.target_split / args.base_method_name
    out_method_dir = output_model / args.target_split / args.method_name
    out_render_dir = out_method_dir / "renders"
    out_render_dir.mkdir(parents=True, exist_ok=True)
    _copy_gt(base_method_dir, out_method_dir)

    render_paths = {
        path.stem: path
        for path in (base_method_dir / "renders").iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    infos = []
    for map_path in tqdm(sorted(Path(args.target_surface_map_dir).glob("*.npz")), desc=f"Surface face-alpha {args.target_split}"):
        payload = _load_npz(map_path)
        face_id = np.array(payload["face_id"], copy=True)
        center = payload["camera_center"].astype(np.float32).reshape(3)
        render_path = render_paths.get(map_path.stem)
        if render_path is None:
            raise FileNotFoundError(f"missing base render for {map_path.name}")
        base = _image_to_np(render_path)
        signal, confidence, info = compute_surface_signal(
            face_id,
            center,
            field,
            k=int(args.k),
            max_abs_residual=float(args.max_abs_residual),
            allowed_faces=set(face_alphas),
            alias_source=alias_source,
        )
        delta = scale_signal_by_face_alpha(face_id, signal, face_alphas)
        adapted = np.clip(base + delta, 0.0, 1.0)
        _save_np_image(adapted, out_render_dir / render_path.name)
        infos.append({"frame": render_path.stem, **info})
    return {
        "target_frames": int(len(infos)),
        "mean_covered_fraction": float(np.mean([x["covered_fraction"] for x in infos])) if infos else 0.0,
        "mean_used_faces": float(np.mean([x["used_faces"] for x in infos])) if infos else 0.0,
        "mean_propagated_faces_used": float(np.mean([x.get("propagated_faces_used", 0) for x in infos])) if infos else 0.0,
        "frames": infos,
        "output_method_dir": str(out_method_dir),
    }


def maybe_wandb(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[SurfaceFaceAlpha] W&B unavailable, skipping log: {exc}")
        return
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or None,
        mode=args.wandb_mode,
        config=vars(args),
    )
    flat = {
        "surface_facealpha/faces": int(report.get("accepted_faces", 0)),
        "surface_facealpha/mean_alpha": float(report.get("face_alpha_report", {}).get("mean_alpha", 0.0)),
        "surface_facealpha/policy_dpsnr": float(report.get("policy", {}).get("dPSNR", 0.0)),
        "surface_facealpha/policy_dssim": float(report.get("policy", {}).get("dSSIM", 0.0)),
        "surface_facealpha/target_coverage": float(report.get("target", {}).get("mean_covered_fraction", 0.0)),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    topology_faces = load_checkpoint_faces(args.base_model_path, int(args.iteration))
    selected_faces = read_selected_faces(
        args.evidence_dir,
        top_k=args.top_k,
        min_view_hits=args.min_view_hits,
        min_consistency=args.min_consistency,
        min_pixel_count=args.min_pixel_count,
        max_face_id=int(topology_faces.shape[0]),
    )
    all_views = _view_paths(args.evidence_dir)
    target_maps = sorted(Path(args.target_surface_map_dir).glob("*.npz"))
    visible_faces = collect_visible_faces([*all_views, *target_maps], max_face_id=int(topology_faces.shape[0]))
    fit_views, policy_views = split_policy_views(all_views, args.policy_val_stride)
    field = build_field(
        fit_views,
        selected_faces,
        min_alpha=args.min_alpha,
        high_error_quantile=args.high_error_quantile,
        max_abs_residual=args.max_abs_residual,
        distance_scale=args.distance_scale,
    )
    alias_source, alias_report = build_topology_neighbor_alias(
        field,
        topology_faces,
        neighbor_rings=args.neighbor_rings,
        max_targets_per_source=args.neighbor_max_targets_per_source,
        chunk_size=args.neighbor_chunk_size,
        candidate_faces=visible_faces,
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
    primary_alphas, primary_alpha_report = fit_face_alphas(
        prepared,
        alpha_max=args.face_alpha_max,
        min_pixels=args.face_alpha_min_pixels,
        min_gain=args.face_alpha_min_gain,
        ridge=args.face_alpha_ridge,
        shrink_pixels=args.face_alpha_shrink_pixels,
        edge_weight=args.face_alpha_edge_weight,
        edge_stride=args.face_alpha_edge_stride,
    )
    policy_report = evaluate_policy(prepared, primary_alphas, device=device, compute_lpips=bool(args.calib_lpips))
    alpha_sets = [primary_alphas]
    consensus_reports: list[dict[str, Any]] = []
    accepted = passes_policy(policy_report, args)
    for stride in _parse_stride_list(args.consensus_policy_strides):
        if int(stride) == int(args.policy_val_stride):
            continue
        c_fit, c_policy = split_policy_views(all_views, stride)
        c_field = build_field(
            c_fit,
            selected_faces,
            min_alpha=args.min_alpha,
            high_error_quantile=args.high_error_quantile,
            max_abs_residual=args.max_abs_residual,
            distance_scale=args.distance_scale,
        )
        c_alias, c_alias_report = build_topology_neighbor_alias(
            c_field,
            topology_faces,
            neighbor_rings=args.neighbor_rings,
            max_targets_per_source=args.neighbor_max_targets_per_source,
            chunk_size=args.neighbor_chunk_size,
            candidate_faces=visible_faces,
        )
        c_prepared = prepare_policy_views(
            c_field,
            c_policy,
            k=args.k,
            max_abs_residual=args.max_abs_residual,
            alias_source=c_alias,
            fallback_render_dir=args.base_model_path / "train" / args.base_method_name / "renders",
            fallback_gt_dir=args.base_model_path / "train" / args.base_method_name / "gt",
        )
        c_alphas, c_alpha_report = fit_face_alphas(
            c_prepared,
            alpha_max=args.face_alpha_max,
            min_pixels=args.face_alpha_min_pixels,
            min_gain=args.face_alpha_min_gain,
            ridge=args.face_alpha_ridge,
            shrink_pixels=args.face_alpha_shrink_pixels,
            edge_weight=args.face_alpha_edge_weight,
            edge_stride=args.face_alpha_edge_stride,
        )
        c_policy_report = evaluate_policy(c_prepared, c_alphas, device=device, compute_lpips=bool(args.calib_lpips))
        c_accepted = passes_policy(c_policy_report, args)
        accepted = bool(accepted and c_accepted)
        if c_accepted:
            alpha_sets.append(c_alphas)
        consensus_reports.append(
            {
                "stride": int(stride),
                "accepted": bool(c_accepted),
                "face_alpha_report": c_alpha_report,
                "policy": c_policy_report,
                "topology_alias": c_alias_report,
            }
        )
    final_alphas = intersect_face_alphas(alpha_sets) if accepted else {}
    target_report = apply_target(args, field, final_alphas, alias_source)
    report = {
        "method": "ECSR Surface Residual FaceAlpha",
        "base_model_path": str(args.base_model_path),
        "evidence_dir": str(args.evidence_dir),
        "target_surface_map_dir": str(args.target_surface_map_dir),
        "target_split": args.target_split,
        "base_method": args.base_method_name,
        "method_name": args.method_name,
        "selected_faces": int(len(selected_faces)),
        "field_faces": int(len(field.residuals)),
        "topology_alias": alias_report,
        "policy": policy_report,
        "policy_accepted": bool(accepted),
        "face_alpha_report": primary_alpha_report,
        "consensus_policy": consensus_reports,
        "accepted_faces": int(len(final_alphas)),
        "mean_final_alpha": float(np.mean(list(final_alphas.values()))) if final_alphas else 0.0,
        "target": target_report,
    }
    out_method_dir = Path(target_report["output_method_dir"])
    (out_method_dir / "surface_facealpha_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_method_dir / "surface_facealpha_report.md").write_text(
        "\n".join(
            [
                "# ECSR Surface Residual FaceAlpha Report",
                "",
                f"- method: `{args.method_name}`",
                f"- policy accepted: `{accepted}`",
                f"- accepted faces: `{len(final_alphas)}`",
                f"- mean final alpha: `{report['mean_final_alpha']:.6f}`",
                f"- target coverage: `{target_report['mean_covered_fraction']:.6f}`",
                "",
                "Per-face alphas are fitted on train-policy views only and intersected across consensus splits.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

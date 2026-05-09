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

from lpipsPyTorch import lpips
from utils.loss_utils import ssim


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
            rows.append(
                {
                    "face_id": int(_float(row, "face_id")),
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
    with np.load(path) as z:
        return {name: z[name] for name in z.files}


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
    face_id = payload["face_id"].astype(np.int64)
    residual = payload["residual_rgb"].astype(np.float32)
    residual_l1 = payload["residual_l1"].astype(np.float32)
    alpha = payload["alpha"].astype(np.float32)
    if alpha.ndim == 3:
        alpha = np.squeeze(alpha, axis=0)
    if residual.shape[0] != 3:
        raise RuntimeError(f"expected residual_rgb with shape [3,H,W], got {residual.shape}")

    valid = (face_id >= 0) & (alpha >= float(min_alpha))
    if selected_faces:
        valid &= np.isin(face_id, np.fromiter(selected_faces, dtype=np.int64))
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


def compute_surface_signal(
    face_id: np.ndarray,
    camera_center: np.ndarray,
    field: FaceResidualField,
    *,
    k: int,
    max_abs_residual: float,
    allowed_faces: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = face_id.shape
    signal = np.zeros((3, h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    if allowed_faces is None:
        allowed_faces = set(field.residuals)
    if not allowed_faces:
        return signal, confidence, {"used_faces": 0, "covered_fraction": 0.0, "mean_confidence": 0.0}
    allowed_np = np.fromiter(allowed_faces, dtype=np.int64)
    valid = (face_id >= 0) & np.isin(face_id, allowed_np)
    present = np.unique(face_id[valid]).astype(np.int64) if np.any(valid) else np.empty((0,), dtype=np.int64)
    target_center = camera_center.astype(np.float32).reshape(3)
    used = 0
    for fid in present.tolist():
        if int(fid) not in field.residuals:
            continue
        centers = field.centers[int(fid)]
        residuals = field.residuals[int(fid)]
        weights = field.weights[int(fid)]
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
        mask = face_id == int(fid)
        signal[:, mask] = residual[:, None]
        confidence[mask] = float(denom)
        used += 1
    covered = confidence > 0.0
    return (
        signal,
        confidence,
        {
            "used_faces": int(used),
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
    mse = float(torch.mean((pred_t - gt_t) ** 2).detach().cpu().item())
    out = {
        "PSNR": -10.0 * math.log10(max(mse, 1e-12)),
        "SSIM": float(ssim(pred_t, gt_t).detach().cpu().item()),
    }
    if compute_lpips:
        out["LPIPS"] = float(lpips(pred_t, gt_t, net_type="vgg").detach().cpu().item())
    return out


def prepare_policy_views(
    field: FaceResidualField,
    policy_paths: list[Path],
    *,
    k: int,
    max_abs_residual: float,
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
        face_id = payload["face_id"].astype(np.int64)
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
        present = np.unique(view.face_id[active]).astype(np.int64)
        for fid in present.tolist():
            mask = active & (view.face_id == int(fid))
            n = int(np.sum(mask))
            if n <= 0:
                continue
            gain_sum[int(fid)] = gain_sum.get(int(fid), 0.0) + float(np.sum(gain[mask]))
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
    best: tuple[float, float, dict[str, Any]] | None = None
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
        if best is None or rank > (best[0], best[1]):
            best = (float(score), float(alpha), row)
    assert best is not None
    chosen = dict(best[2])
    accepted = (
        chosen["dPSNR"] >= float(min_dpsnr)
        and chosen["dSSIM"] >= float(min_dssim)
        and (not compute_lpips or chosen["dLPIPS"] <= float(max_dlpips))
        and float(chosen["alpha"]) > 0.0
    )
    alpha = float(chosen["alpha"]) if accepted else 0.0
    reason = "policy_val_pass" if accepted else "policy_val_guard_rejected"
    return alpha, {
        "reason": reason,
        "alpha": alpha,
        "accepted": bool(accepted),
        "chosen_row": chosen,
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
        face_id = payload["face_id"].astype(np.int64)
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
        )
        adapted = np.clip(base + float(alpha) * signal, 0.0, 1.0)
        _save_np_image(adapted, out_render_dir / render_path.name)
        infos.append({"frame": render_path.stem, **info})
    return {
        "target_frames": int(len(infos)),
        "mean_covered_fraction": float(np.mean([x["covered_fraction"] for x in infos])) if infos else 0.0,
        "mean_confidence": float(np.mean([x["mean_confidence"] for x in infos])) if infos else 0.0,
        "mean_used_faces": float(np.mean([x["used_faces"] for x in infos])) if infos else 0.0,
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
        },
    )
    flat = {
        "surface_lumigraph/alpha": float(report.get("alpha", 0.0)),
        "surface_lumigraph/selected_faces": int(report.get("selected_faces", 0)),
        "surface_lumigraph/field_faces": int(report.get("field_faces", 0)),
        "surface_lumigraph/accepted_faces": int(report.get("accepted_faces", 0)),
        "surface_lumigraph/target_frames": int(report.get("target", {}).get("target_frames", 0)),
        "surface_lumigraph/mean_covered_fraction": float(report.get("target", {}).get("mean_covered_fraction", 0.0)),
        "surface_lumigraph/mean_used_faces": float(report.get("target", {}).get("mean_used_faces", 0.0)),
    }
    run.log(flat)
    run.summary.update(flat)
    run.finish()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    selected_faces = read_selected_faces(
        args.evidence_dir,
        top_k=args.top_k,
        min_view_hits=args.min_view_hits,
        min_consistency=args.min_consistency,
        min_pixel_count=args.min_pixel_count,
    )
    all_views = _view_paths(args.evidence_dir)
    fit_views, policy_views = split_policy_views(all_views, args.policy_val_stride)
    field = build_field(
        fit_views,
        selected_faces,
        min_alpha=args.min_alpha,
        high_error_quantile=args.high_error_quantile,
        max_abs_residual=args.max_abs_residual,
        distance_scale=args.distance_scale,
    )
    prepared = prepare_policy_views(
        field,
        policy_views,
        k=args.k,
        max_abs_residual=args.max_abs_residual,
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
    consensus_reports: list[dict[str, Any]] = []
    consensus_alphas = [float(alpha)]
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
        c_prepared = prepare_policy_views(
            c_field,
            c_policy_views,
            k=args.k,
            max_abs_residual=args.max_abs_residual,
            fallback_render_dir=args.base_model_path / "train" / args.base_method_name / "renders",
            fallback_gt_dir=args.base_model_path / "train" / args.base_method_name / "gt",
        )
        c_alpha, c_calibration = calibrate_policy(
            c_prepared,
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
        consensus_reports.append(
            {
                "stride": int(stride),
                "fit_views": [p.stem for p in c_fit_views],
                "policy_val_views": [p.stem for p in c_policy_views],
                "alpha": float(c_alpha),
                "calibration": c_calibration,
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
    elif float(alpha) <= 0.0:
        allowed_faces = set()
        face_gate_report.update({"accepted_faces": 0, "reason": "alpha_zero"})
    target_report = apply_target(args, field, alpha=alpha, allowed_faces=allowed_faces)
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
        f"- policy alpha: `{alpha:.4f}`",
        f"- calibration: `{calibration.get('reason', 'unknown')}`",
        f"- face gate accepted: `{len(allowed_faces or set())}`",
        f"- target coverage: `{target_report['mean_covered_fraction']:.4f}`",
        "",
        "This path stores the residual signal on train-observed surface face ids and applies it to held-out renders through target face-id maps only. Held-out GT is not used for policy selection.",
    ]
    (out_method_dir / "surface_lumigraph_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    _maybe_wandb(args, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

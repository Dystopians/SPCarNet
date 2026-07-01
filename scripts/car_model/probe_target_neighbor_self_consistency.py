#!/usr/bin/env python3
"""Probe target-neighborhood render self-consistency for promoted candidates.

The self-consistency score is target-blind: it uses target split render/depth
and camera geometry only. Target GT is copied from the reference report only for
post-hoc correlation with the diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.apply_source_heldout_support_transport_calibrator import (  # noqa: E402
    _candidate_deltas,
    _load_model,
)
from scripts.car_model.train_source_heldout_support_transport_calibrator import (  # noqa: E402
    _build_features,
    _normalize,
    _split_source_heldout,
)
from utils.evidence_lumigraph_adapter import (  # noqa: E402
    FrameLoader,
    compute_evidence_signal,
    load_split_frames,
    save_image_tensor,
    select_support_frames,
    warp_support_residual,
)


def _fit_hw(height: int, width: int, max_side: int) -> tuple[int, int] | None:
    if int(max_side) <= 0 or max(height, width) <= int(max_side):
        return None
    scale = float(max_side) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _resize_chw(image: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def _resize_hw(depth: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    if size is None or tuple(depth.shape[-2:]) == tuple(size):
        return depth
    return F.interpolate(depth[None, None], size=size, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)


def _weighted_mae(a: torch.Tensor, b: torch.Tensor, confidence: torch.Tensor, threshold: float) -> tuple[float, float]:
    mask = (confidence > float(threshold)).to(device=a.device, dtype=a.dtype)
    denom = torch.clamp(mask.sum() * float(a.shape[0]), min=1.0)
    mae = (torch.abs(a - b) * mask.unsqueeze(0)).sum() / denom
    return float(mae.detach().cpu().item()), float(mask.mean().detach().cpu().item())


def _run_model_delta(
    *,
    target: Any,
    source_frames: list[Any],
    model: Any,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, torch.Tensor]]:
    ev = compute_evidence_signal(
        target,
        source_frames,
        k=int(args.k),
        mode="residual",
        residual_clip=float(args.residual_clip),
        min_confidence=float(args.min_confidence),
        depth_abs_tol=float(args.depth_abs_tol),
        depth_rel_tol=float(args.depth_rel_tol),
        direction_weight=float(args.direction_weight),
        evidence_max_side=int(args.evidence_max_side),
        loader=loader,
        device=device,
    )
    features = _build_features(ev, k=int(args.k)).unsqueeze(0).to(device=device, dtype=torch.float32)
    signal = ev.signal.unsqueeze(0).to(device=device, dtype=torch.float32)
    valid = ev.valid.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_delta = model(_normalize(features, feature_mean, feature_std), signal, valid).squeeze(0)
    return ev, _candidate_deltas(ev, pred_delta, args)


def _score_variant_against_neighbors(
    *,
    target: Any,
    target_image: torch.Tensor,
    neighbors: list[tuple[Any, float]],
    neighbor_reference_images: dict[str, torch.Tensor] | None = None,
    loader: FrameLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    current_depth_full = loader.depth(str(target.depth_path)).to(device=device, dtype=torch.float32)
    current_size = _fit_hw(int(current_depth_full.shape[0]), int(current_depth_full.shape[1]), int(args.consistency_max_side))
    current_depth = _resize_hw(current_depth_full, current_size)
    current_image = _resize_chw(target_image.to(device=device, dtype=torch.float32), current_size)

    rows: list[dict[str, Any]] = []
    weighted_error = 0.0
    total_weight = 0.0
    confidence_values: list[float] = []
    neighbor_reference_images = neighbor_reference_images or {}
    for neighbor, view_weight in neighbors:
        neighbor_depth_full = loader.depth(str(neighbor.depth_path)).to(device=device, dtype=torch.float32)
        neighbor_reference_full = neighbor_reference_images.get(str(neighbor.name))
        if neighbor_reference_full is None:
            neighbor_reference_full = loader.render(str(neighbor.render_path)).to(device=device, dtype=torch.float32)
        neighbor_reference_full = neighbor_reference_full.to(device=device, dtype=torch.float32)
        neighbor_size = _fit_hw(
            int(neighbor_depth_full.shape[0]),
            int(neighbor_depth_full.shape[1]),
            int(args.consistency_max_side),
        )
        neighbor_depth = _resize_hw(neighbor_depth_full, neighbor_size)
        neighbor_reference = _resize_chw(neighbor_reference_full, neighbor_size)
        warped, confidence = warp_support_residual(
            neighbor,
            target,
            neighbor_depth,
            current_depth,
            current_image,
            depth_abs_tol=float(args.consistency_depth_abs_tol),
            depth_rel_tol=float(args.consistency_depth_rel_tol),
            device=device,
        )
        mae, confident_fraction = _weighted_mae(
            warped,
            neighbor_reference,
            confidence,
            threshold=float(args.consistency_min_confidence),
        )
        effective_weight = float(view_weight) * max(confident_fraction, 0.0)
        rows.append(
            {
                "neighbor": str(neighbor.name),
                "view_weight": float(view_weight),
                "mae_to_neighbor_reference": mae,
                "confident_fraction": confident_fraction,
                "effective_weight": effective_weight,
            }
        )
        weighted_error += mae * effective_weight
        total_weight += effective_weight
        confidence_values.append(confident_fraction)
    mean_error = weighted_error / max(total_weight, 1.0e-12) if rows else float("nan")
    return {
        "available": bool(rows),
        "neighbor_count": int(len(rows)),
        "mean_mae_to_neighbor_reference": float(mean_error),
        "mean_confident_fraction": float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
        "total_effective_weight": float(total_weight),
        "neighbors": rows,
    }


def _target_eval_delta(row: dict[str, Any], output_variant: str, incumbent_variant: str) -> dict[str, float]:
    metrics = row.get("candidate_metrics") or {}
    output_metrics = metrics.get(output_variant) or {}
    incumbent_metrics = metrics.get(incumbent_variant) or {}
    return {
        "delta_psnr_gain": float(output_metrics.get("psnr_gain", 0.0)) - float(incumbent_metrics.get("psnr_gain", 0.0)),
        "delta_ssim_gain": float(output_metrics.get("ssim_gain", 0.0)) - float(incumbent_metrics.get("ssim_gain", 0.0)),
        "output_psnr_gain": float(output_metrics.get("psnr_gain", 0.0)),
        "incumbent_psnr_gain": float(incumbent_metrics.get("psnr_gain", 0.0)),
        "output_ssim_gain": float(output_metrics.get("ssim_gain", 0.0)),
        "incumbent_ssim_gain": float(incumbent_metrics.get("ssim_gain", 0.0)),
    }


def _safe_error(payload: dict[str, Any]) -> float:
    value = float(payload.get("mean_mae_to_neighbor_reference", float("nan")))
    return value


def _write_visuals(
    *,
    out_dir: Path,
    view: str,
    ev: Any,
    output_variant: str,
    incumbent_variant: str,
    deltas: dict[str, torch.Tensor],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image_tensor(ev.base, out_dir / f"{view}_base.png")
    if incumbent_variant in deltas:
        save_image_tensor(torch.clamp(ev.base + deltas[incumbent_variant], 0.0, 1.0), out_dir / f"{view}_incumbent_{incumbent_variant}.png")
    if output_variant in deltas:
        save_image_tensor(torch.clamp(ev.base + deltas[output_variant], 0.0, 1.0), out_dir / f"{view}_output_{output_variant}.png")
        if incumbent_variant in deltas:
            diff = torch.clamp(0.5 + 8.0 * (deltas[output_variant] - deltas[incumbent_variant]), 0.0, 1.0)
            save_image_tensor(diff, out_dir / f"{view}_output_minus_incumbent_x8.png")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, feature_mean, feature_std, _ = _load_model(Path(args.checkpoint), device)
    base_model = Path(args.base_model_path)
    train_frames = load_split_frames(base_model, "train", str(args.base_method_name))
    if str(args.support_source_mode) == "source_split":
        source_frames, _ = _split_source_heldout(train_frames, int(args.heldout_stride), int(args.heldout_offset))
    else:
        source_frames = train_frames
    target_frames = load_split_frames(base_model, str(args.target_split), str(args.base_method_name))
    target_by_name = {frame.name: frame for frame in target_frames}
    reference = json.loads(Path(args.reference_report).read_text(encoding="utf-8"))
    wanted = {part.strip() for part in str(args.views).split(",") if part.strip()}
    loader = FrameLoader(device=device)
    target_delta_cache: dict[str, tuple[Any, dict[str, torch.Tensor]]] = {}

    def target_model_delta(frame: Any) -> tuple[Any, dict[str, torch.Tensor]]:
        key = str(frame.name)
        cached = target_delta_cache.get(key)
        if cached is not None:
            return cached
        computed = _run_model_delta(
            target=frame,
            source_frames=source_frames,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            loader=loader,
            device=device,
            args=args,
        )
        target_delta_cache[key] = computed
        return computed

    rows: list[dict[str, Any]] = []
    for row in tqdm(reference.get("per_view", []), desc="target neighbor consistency"):
        view = str(row.get("view"))
        if wanted and view not in wanted:
            continue
        output_variant = str(row.get("output_variant"))
        if bool(args.promotions_only):
            if output_variant in {"noop", "__scene__", "__incumbent__"}:
                continue
            if output_variant == str(row.get("selected_variant")):
                continue
        target = target_by_name.get(view)
        if target is None:
            continue
        pairwise_diag = row.get("pairwise_dominance_diagnostics") or {}
        incumbent_variant = str(pairwise_diag.get("incumbent_variant") or row.get("selected_variant"))
        ev, deltas = target_model_delta(target)
        if output_variant not in deltas or incumbent_variant not in deltas:
            continue
        neighbors = select_support_frames(
            target,
            target_frames,
            k=int(args.neighbor_k),
            exclude_names={target.name, target.camera.image_name},
            direction_weight=float(args.neighbor_direction_weight),
        )
        variant_names = [name for name in deltas.keys() if name in (row.get("candidate_metrics") or {})]
        if output_variant not in variant_names:
            variant_names.append(output_variant)
        if incumbent_variant not in variant_names:
            variant_names.append(incumbent_variant)
        variant_consistency: dict[str, Any] = {}
        for variant in variant_names:
            if variant not in deltas:
                continue
            neighbor_reference_images: dict[str, torch.Tensor] = {}
            if str(args.neighbor_reference_mode) == "same_variant":
                for neighbor, _ in neighbors:
                    neighbor_ev, neighbor_deltas = target_model_delta(neighbor)
                    if variant in neighbor_deltas:
                        neighbor_reference_images[str(neighbor.name)] = torch.clamp(neighbor_ev.base + neighbor_deltas[variant], 0.0, 1.0)
            image = torch.clamp(ev.base + deltas[variant], 0.0, 1.0)
            variant_consistency[variant] = _score_variant_against_neighbors(
                target=target,
                target_image=image,
                neighbors=neighbors,
                neighbor_reference_images=neighbor_reference_images,
                loader=loader,
                device=device,
                args=args,
            )
        base_consistency = _score_variant_against_neighbors(
            target=target,
            target_image=ev.base,
            neighbors=neighbors,
            neighbor_reference_images={},
            loader=loader,
            device=device,
            args=args,
        )
        output_error = _safe_error(variant_consistency.get(output_variant, {}))
        incumbent_error = _safe_error(variant_consistency.get(incumbent_variant, {}))
        base_error = _safe_error(base_consistency)
        row_payload = {
            "view": view,
            "output_variant": output_variant,
            "incumbent_variant": incumbent_variant,
            "selected_variant": str(row.get("selected_variant")),
            "support_names": list(ev.support_names),
            "neighbor_names": [neighbor.name for neighbor, _ in neighbors],
            "target_eval_delta": _target_eval_delta(row, output_variant, incumbent_variant),
            "base_consistency": base_consistency,
            "variant_consistency": variant_consistency,
            "self_consistency_delta_incumbent_minus_output": float(incumbent_error - output_error),
            "self_consistency_delta_base_minus_output": float(base_error - output_error),
            "self_consistency_delta_base_minus_incumbent": float(base_error - incumbent_error),
        }
        rows.append(row_payload)
        if str(args.save_visual_dir):
            _write_visuals(
                out_dir=Path(args.save_visual_dir),
                view=view,
                ev=ev,
                output_variant=output_variant,
                incumbent_variant=incumbent_variant,
                deltas=deltas,
            )
    positive_target = [r for r in rows if float(r["target_eval_delta"]["delta_psnr_gain"]) > 0.0]
    negative_target = [r for r in rows if float(r["target_eval_delta"]["delta_psnr_gain"]) <= 0.0]
    consistency_positive = [r for r in rows if float(r["self_consistency_delta_incumbent_minus_output"]) > 0.0]
    consistency_negative = [r for r in rows if float(r["self_consistency_delta_incumbent_minus_output"]) <= 0.0]
    payload = {
        "target_gt_usage": "self-consistency uses target render/depth/camera only; reference target metrics are copied for post-hoc diagnostic correlation",
        "base_model_path": str(base_model),
        "base_method_name": str(args.base_method_name),
        "checkpoint": str(args.checkpoint),
        "reference_report": str(args.reference_report),
        "target_split": str(args.target_split),
        "support_source_mode": str(args.support_source_mode),
        "k": int(args.k),
        "neighbor_k": int(args.neighbor_k),
        "neighbor_reference_mode": str(args.neighbor_reference_mode),
        "consistency_max_side": int(args.consistency_max_side),
        "row_count": int(len(rows)),
        "summary": {
            "target_positive_count": int(len(positive_target)),
            "target_negative_count": int(len(negative_target)),
            "consistency_positive_count": int(len(consistency_positive)),
            "consistency_negative_count": int(len(consistency_negative)),
            "mean_self_consistency_delta_incumbent_minus_output": float(
                sum(float(r["self_consistency_delta_incumbent_minus_output"]) for r in rows) / max(len(rows), 1)
            ),
            "positive_target_mean_consistency_delta": float(
                sum(float(r["self_consistency_delta_incumbent_minus_output"]) for r in positive_target)
                / max(len(positive_target), 1)
            ),
            "negative_target_mean_consistency_delta": float(
                sum(float(r["self_consistency_delta_incumbent_minus_output"]) for r in negative_target)
                / max(len(negative_target), 1)
            ),
        },
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference_report", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--target_split", default="test")
    parser.add_argument("--support_source_mode", choices=["source_split", "all_train"], default="source_split")
    parser.add_argument("--heldout_stride", type=int, default=4)
    parser.add_argument("--heldout_offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--anchor_alpha", type=float, default=0.25)
    parser.add_argument("--learned_scale", type=float, default=0.5)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--enable_candidate_ladder", action="store_true")
    parser.add_argument("--candidate_ladder_blends", default="0.25,0.75")
    parser.add_argument("--residual_clip", type=float, default=0.25)
    parser.add_argument("--min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--depth_abs_tol", type=float, default=0.02)
    parser.add_argument("--depth_rel_tol", type=float, default=0.03)
    parser.add_argument("--direction_weight", type=float, default=0.35)
    parser.add_argument("--evidence_max_side", type=int, default=256)
    parser.add_argument("--views", default="")
    parser.add_argument("--promotions_only", action="store_true")
    parser.add_argument("--neighbor_k", type=int, default=2)
    parser.add_argument("--neighbor_direction_weight", type=float, default=0.35)
    parser.add_argument("--neighbor_reference_mode", choices=["base", "same_variant"], default="base")
    parser.add_argument("--consistency_max_side", type=int, default=256)
    parser.add_argument("--consistency_depth_abs_tol", type=float, default=0.03)
    parser.add_argument("--consistency_depth_rel_tol", type=float, default=0.04)
    parser.add_argument("--consistency_min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--save_visual_dir", default="")
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "row_count": payload["row_count"],
                "summary": payload["summary"],
                "output_json": str(args.output_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

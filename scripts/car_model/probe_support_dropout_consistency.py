#!/usr/bin/env python3
"""Probe target-blind support-dropout consistency for support-transport candidates."""

from __future__ import annotations

import argparse
import json
import math
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
)


def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.reshape(-1).to(torch.float32)
    bv = b.reshape(-1).to(torch.float32)
    denom = float(torch.linalg.norm(av).item() * torch.linalg.norm(bv).item())
    if denom <= 1.0e-12:
        return 0.0
    return float(torch.dot(av, bv).item() / denom)


def _mean_abs(x: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(x)).detach().cpu().item())


def _dropout_stats(samples: list[torch.Tensor], reference: torch.Tensor) -> dict[str, Any]:
    if not samples:
        return {"available": False, "sample_count": 0}
    stack = torch.stack([sample.to(torch.float32) for sample in samples], dim=0)
    std = torch.std(stack, dim=0, unbiased=False)
    mean = torch.mean(stack, dim=0)
    ref_abs = torch.abs(reference.to(torch.float32))
    sign_mask = ref_abs > 1.0e-5
    sign_flip = (
        torch.mean((torch.sign(stack[:, sign_mask]) != torch.sign(reference[sign_mask]).unsqueeze(0)).to(torch.float32))
        if bool(sign_mask.any())
        else torch.tensor(0.0, device=reference.device)
    )
    ref_mae = _mean_abs(reference)
    return {
        "available": True,
        "sample_count": int(len(samples)),
        "reference_mean_abs": ref_mae,
        "dropout_mean_abs": _mean_abs(mean),
        "dropout_std_mean": _mean_abs(std),
        "relative_std": float(_mean_abs(std) / max(ref_mae, 1.0e-8)),
        "mean_cosine_to_reference": _tensor_cosine(mean, reference),
        "sign_flip_fraction": float(sign_flip.detach().cpu().item()),
    }


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
    reference = json.loads(Path(args.reference_report).read_text())
    wanted = {part.strip() for part in str(args.views).split(",") if part.strip()}
    loader = FrameLoader(device=device)
    rows: list[dict[str, Any]] = []
    for row in tqdm(reference.get("per_view", []), desc="support dropout consistency"):
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
        if output_variant not in row.get("candidate_metrics", {}) or incumbent_variant not in row.get("candidate_metrics", {}):
            continue
        ev, deltas = _run_model_delta(
            target=target,
            source_frames=source_frames,
            model=model,
            feature_mean=feature_mean,
            feature_std=feature_std,
            loader=loader,
            device=device,
            args=args,
        )
        if output_variant not in deltas or incumbent_variant not in deltas:
            continue
        pair_reference = deltas[output_variant] - deltas[incumbent_variant]
        output_reference = deltas[output_variant]
        support_names = list(ev.support_names)
        pair_samples: list[torch.Tensor] = []
        output_samples: list[torch.Tensor] = []
        support_frames_by_name = {frame.name: frame for frame in source_frames}
        for support_name in support_names[: max(0, int(args.max_dropout_samples))]:
            drop_frame = support_frames_by_name.get(support_name)
            if drop_frame is None:
                continue
            subset = [frame for frame in source_frames if frame.name != drop_frame.name]
            _, drop_deltas = _run_model_delta(
                target=target,
                source_frames=subset,
                model=model,
                feature_mean=feature_mean,
                feature_std=feature_std,
                loader=loader,
                device=device,
                args=args,
            )
            if output_variant in drop_deltas and incumbent_variant in drop_deltas:
                pair_samples.append((drop_deltas[output_variant] - drop_deltas[incumbent_variant]).detach())
                output_samples.append(drop_deltas[output_variant].detach())
        metrics = row["candidate_metrics"]
        output_metrics = metrics[output_variant]
        incumbent_metrics = metrics[incumbent_variant]
        rows.append(
            {
                "view": view,
                "output_variant": output_variant,
                "incumbent_variant": incumbent_variant,
                "support_names": support_names,
                "target_eval_delta_psnr_gain": float(output_metrics.get("psnr_gain", 0.0))
                - float(incumbent_metrics.get("psnr_gain", 0.0)),
                "target_eval_delta_ssim_gain": float(output_metrics.get("ssim_gain", 0.0))
                - float(incumbent_metrics.get("ssim_gain", 0.0)),
                "pair_delta_dropout": _dropout_stats(pair_samples, pair_reference),
                "output_delta_dropout": _dropout_stats(output_samples, output_reference),
            }
        )
    payload = {
        "target_gt_usage": "reference report metrics are copied for post-hoc diagnostic correlation only",
        "base_model_path": str(base_model),
        "base_method_name": str(args.base_method_name),
        "checkpoint": str(args.checkpoint),
        "reference_report": str(args.reference_report),
        "target_split": str(args.target_split),
        "support_source_mode": str(args.support_source_mode),
        "k": int(args.k),
        "max_dropout_samples": int(args.max_dropout_samples),
        "row_count": int(len(rows)),
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
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
    parser.add_argument("--max_dropout_samples", type=int, default=4)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"row_count": payload["row_count"], "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose high-bandwidth source-heldout ELA residual transport.

This is a train-split diagnostic, not a target/test evaluation.  It splits train
views into source and heldout-source subsets, builds the normal ELA residual
signal from source views only, then measures how well that signal repairs the
heldout-source views.  The goal is to quantify whether Phase-J's render-time
support-view information path still has useful transport headroom after the
current baked carrier has stalled.
"""

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

from utils.evidence_lumigraph_adapter import (  # noqa: E402
    FrameLoader,
    compute_evidence_signal,
    load_split_frames,
    mse_to_psnr,
    save_image_tensor,
)
from utils.loss_utils import ssim as torch_ssim  # noqa: E402


def _parse_float_grid(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if math.isfinite(value):
            values.append(value)
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(set(values))


def _split_source_heldout(frames: list[Any], stride: int, offset: int) -> tuple[list[Any], list[Any]]:
    stride = max(int(stride), 2)
    offset = int(offset) % stride
    source, heldout = [], []
    for idx, frame in enumerate(frames):
        if idx % stride == offset:
            heldout.append(frame)
        else:
            source.append(frame)
    if not source or not heldout:
        raise ValueError(
            f"invalid source-heldout split: source={len(source)} heldout={len(heldout)} "
            f"stride={stride} offset={offset}"
        )
    return source, heldout


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _tail(values: list[float], fraction: float = 0.20) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "cvar": 0.0}
    arr = sorted(float(v) for v in values)
    count = max(1, int(math.ceil(float(fraction) * len(arr))))
    return {"min": float(arr[0]), "cvar": float(sum(arr[:count]) / count)}


def _downscale_for_metric(image: torch.Tensor, max_side: int) -> torch.Tensor:
    max_side = int(max_side)
    if max_side <= 0:
        return image
    h, w = int(image.shape[-2]), int(image.shape[-1])
    current = max(h, w)
    if current <= max_side:
        return image
    scale = float(max_side) / float(current)
    out_h = max(8, int(round(h * scale)))
    out_w = max(8, int(round(w * scale)))
    return F.interpolate(image.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)


def _ssim_value(a: torch.Tensor, b: torch.Tensor, max_side: int) -> float:
    a_small = _downscale_for_metric(a, max_side).unsqueeze(0)
    b_small = _downscale_for_metric(b, max_side).unsqueeze(0)
    return float(torch_ssim(a_small, b_small).detach().cpu().item())


def _direction_stats(signal: torch.Tensor, true_residual: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    valid3 = valid.expand_as(signal).bool()
    if not bool(valid3.any().item()):
        return {
            "direction_cosine": 0.0,
            "signal_energy_l1": 0.0,
            "true_energy_l1": 0.0,
            "energy_ratio_l1": 0.0,
        }
    sig = signal[valid3].reshape(-1).to(torch.float64)
    target = true_residual[valid3].reshape(-1).to(torch.float64)
    sig_norm = torch.linalg.vector_norm(sig)
    tgt_norm = torch.linalg.vector_norm(target)
    denom = torch.clamp(sig_norm * tgt_norm, min=1.0e-12)
    cosine = float(torch.sum(sig * target).detach().cpu().item() / float(denom.detach().cpu().item()))
    sig_l1 = float(torch.mean(torch.abs(sig)).detach().cpu().item())
    tgt_l1 = float(torch.mean(torch.abs(target)).detach().cpu().item())
    return {
        "direction_cosine": cosine,
        "signal_energy_l1": sig_l1,
        "true_energy_l1": tgt_l1,
        "energy_ratio_l1": float(sig_l1 / max(tgt_l1, 1.0e-12)),
    }


def _metrics_for_alpha(
    base: torch.Tensor,
    gt: torch.Tensor,
    signal: torch.Tensor,
    alpha: float,
    *,
    compute_ssim: bool,
    ssim_max_side: int,
) -> dict[str, float]:
    adapted = torch.clamp(base + float(alpha) * signal, 0.0, 1.0)
    base_mse = float(torch.mean(torch.square(base - gt)).detach().cpu().item())
    cand_mse = float(torch.mean(torch.square(adapted - gt)).detach().cpu().item())
    row = {
        "alpha": float(alpha),
        "base_mse": base_mse,
        "candidate_mse": cand_mse,
        "mse_reduction": float(base_mse - cand_mse),
        "base_psnr": mse_to_psnr(base_mse),
        "candidate_psnr": mse_to_psnr(cand_mse),
        "psnr_gain": float(mse_to_psnr(cand_mse) - mse_to_psnr(base_mse)),
        "changed_fraction": float(
            torch.mean((torch.any(torch.abs(float(alpha) * signal) > (0.5 / 255.0), dim=0)).to(torch.float32))
            .detach()
            .cpu()
            .item()
        ),
        "mean_abs_delta": float(torch.mean(torch.abs(float(alpha) * signal)).detach().cpu().item()),
    }
    if compute_ssim:
        base_ssim = _ssim_value(base, gt, int(ssim_max_side))
        cand_ssim = _ssim_value(adapted, gt, int(ssim_max_side))
        row.update(
            {
                "base_ssim": base_ssim,
                "candidate_ssim": cand_ssim,
                "ssim_gain": float(cand_ssim - base_ssim),
            }
        )
    return row


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_heldout_ela_transport_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = payload.get("summary", {})
    lines = [
        "# Source-Heldout ELA Transport Diagnostic",
        "",
        "This is a train-split diagnostic. Target/test GT is not used.",
        "",
        "## Summary",
        "",
        f"- source views: `{payload['split']['source_views']}`",
        f"- heldout views: `{payload['split']['heldout_views']}`",
        f"- best alpha: `{summary.get('best_alpha')}`",
        f"- best PSNR gain: `{summary.get('best_psnr_gain')}`",
        f"- best changed fraction: `{summary.get('best_changed_fraction')}`",
        f"- mean direction cosine: `{summary.get('mean_direction_cosine')}`",
        f"- mean energy ratio L1: `{summary.get('mean_energy_ratio_l1')}`",
        f"- all-axis policy-val style pass: `{summary.get('all_axis_pass')}`",
        "",
        "## Alpha Sweep",
        "",
        "| alpha | PSNR gain | changed | pos views | min PSNR gain | CVaR20 PSNR gain |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("alpha_summaries", []):
        tail = row.get("psnr_gain_tail", {})
        lines.append(
            "| {alpha:.6g} | {psnr:+.9f} | {chg:.9f} | {pos:.6f} | {minv:+.9f} | {cvar:+.9f} |".format(
                alpha=float(row.get("alpha", 0.0)),
                psnr=float(row.get("psnr_gain", 0.0)),
                chg=float(row.get("mean_changed_fraction", 0.0)),
                pos=float(row.get("positive_view_fraction", 0.0)),
                minv=float(tail.get("min", 0.0)),
                cvar=float(tail.get("cvar", 0.0)),
            )
        )
    lines += [
        "",
        "## Verdict",
        "",
        str(payload.get("verdict", "")),
    ]
    (output_dir / "source_heldout_ela_transport_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    base_model = Path(args.base_model_path)
    base_method = str(args.base_method_name)
    output_dir = Path(args.output_dir)
    alpha_grid = _parse_float_grid(args.alpha_grid)

    train_frames = load_split_frames(base_model, "train", base_method)
    source_frames, heldout_frames = _split_source_heldout(train_frames, int(args.heldout_stride), int(args.heldout_offset))
    if int(args.max_heldout_views) > 0:
        heldout_frames = heldout_frames[: int(args.max_heldout_views)]

    loader = FrameLoader(device=device)
    per_view: list[dict[str, Any]] = []
    alpha_rows: dict[float, list[dict[str, float]]] = {float(alpha): [] for alpha in alpha_grid}
    direction_rows: list[dict[str, float]] = []

    visual_dir = output_dir / "visuals"
    if int(args.save_example_views) > 0:
        visual_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, target in enumerate(tqdm(heldout_frames, desc="source-heldout ELA diagnostic")):
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
        base = ev.base.to(device=device, dtype=torch.float32)
        gt = loader.gt(str(target.gt_path)).to(device=device, dtype=torch.float32)
        true_residual = gt - base
        direction = _direction_stats(ev.signal, true_residual, ev.valid)
        direction_rows.append(direction)
        view_payload: dict[str, Any] = {
            "view": target.name,
            "support_names": list(ev.support_names),
            "covered_fraction": float(ev.valid.to(torch.float32).mean().detach().cpu().item()),
            **direction,
            "alphas": [],
        }
        for alpha in alpha_grid:
            row = _metrics_for_alpha(
                base,
                gt,
                ev.signal,
                float(alpha),
                compute_ssim=bool(args.compute_ssim),
                ssim_max_side=int(args.ssim_max_side),
            )
            alpha_rows[float(alpha)].append(row)
            view_payload["alphas"].append(row)

        if view_idx < int(args.save_example_views):
            best_view = max(view_payload["alphas"], key=lambda r: float(r.get("psnr_gain", 0.0)))
            best_alpha = float(best_view["alpha"])
            save_image_tensor(base, visual_dir / f"{target.name}_base.png")
            save_image_tensor(torch.clamp(base + best_alpha * ev.signal, 0.0, 1.0), visual_dir / f"{target.name}_best.png")
            save_image_tensor(gt, visual_dir / f"{target.name}_gt.png")
            signal_vis = torch.clamp(0.5 + 2.0 * ev.signal, 0.0, 1.0)
            save_image_tensor(signal_vis, visual_dir / f"{target.name}_signal_x2.png")
        per_view.append(view_payload)

    alpha_summaries = []
    for alpha, rows in alpha_rows.items():
        psnr_gain = [float(r["psnr_gain"]) for r in rows]
        changed = [float(r["changed_fraction"]) for r in rows]
        summary = {
            "alpha": float(alpha),
            "base_psnr": _mean([float(r["base_psnr"]) for r in rows]),
            "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in rows]),
            "psnr_gain": _mean(psnr_gain),
            "psnr_gain_tail": _tail(psnr_gain),
            "positive_view_fraction": _mean([1.0 if v > 0.0 else 0.0 for v in psnr_gain]),
            "mean_changed_fraction": _mean(changed),
            "mean_abs_delta": _mean([float(r["mean_abs_delta"]) for r in rows]),
        }
        if bool(args.compute_ssim):
            ssim_gain = [float(r["ssim_gain"]) for r in rows]
            summary.update(
                {
                    "base_ssim": _mean([float(r["base_ssim"]) for r in rows]),
                    "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in rows]),
                    "ssim_gain": _mean(ssim_gain),
                    "ssim_gain_tail": _tail(ssim_gain),
                    "ssim_positive_view_fraction": _mean([1.0 if v > 0.0 else 0.0 for v in ssim_gain]),
                }
            )
        alpha_summaries.append(summary)
    alpha_summaries.sort(key=lambda r: float(r["alpha"]))
    best = max(
        alpha_summaries,
        key=lambda r: float(r.get("psnr_gain", 0.0)) + (20.0 * float(r.get("ssim_gain", 0.0)) if args.compute_ssim else 0.0),
    )
    all_axis_pass = (
        float(best.get("psnr_gain", 0.0)) > 0.0
        and float(best.get("mean_changed_fraction", 0.0)) >= float(args.min_changed_fraction)
        and (not bool(args.compute_ssim) or float(best.get("ssim_gain", 0.0)) > 0.0)
    )
    verdict = (
        "High-bandwidth ELA transport has source-heldout headroom; distilling its target-conditioned "
        "support path is justified."
        if all_axis_pass
        else "Source-heldout ELA transport did not pass the configured headroom gate; investigate geometry/support coverage before distillation."
    )
    payload = {
        "method": "source-heldout ELA transport diagnostic",
        "base_model_path": str(base_model),
        "base_method_name": base_method,
        "target_gt_usage": "train-heldout diagnostic only; no target/test GT",
        "device": str(device),
        "split": {
            "train_views": int(len(train_frames)),
            "source_views": int(len(source_frames)),
            "heldout_views": int(len(heldout_frames)),
            "heldout_stride": int(args.heldout_stride),
            "heldout_offset": int(args.heldout_offset),
            "source_names": [frame.name for frame in source_frames],
            "heldout_names": [frame.name for frame in heldout_frames],
        },
        "config": {
            "k": int(args.k),
            "alpha_grid": [float(x) for x in alpha_grid],
            "residual_clip": float(args.residual_clip),
            "min_confidence": float(args.min_confidence),
            "depth_abs_tol": float(args.depth_abs_tol),
            "depth_rel_tol": float(args.depth_rel_tol),
            "direction_weight": float(args.direction_weight),
            "evidence_max_side": int(args.evidence_max_side),
            "compute_ssim": bool(args.compute_ssim),
            "ssim_max_side": int(args.ssim_max_side),
            "min_changed_fraction": float(args.min_changed_fraction),
        },
        "alpha_summaries": alpha_summaries,
        "summary": {
            "best_alpha": float(best.get("alpha", 0.0)),
            "best_psnr_gain": float(best.get("psnr_gain", 0.0)),
            "best_changed_fraction": float(best.get("mean_changed_fraction", 0.0)),
            "best_ssim_gain": float(best.get("ssim_gain", 0.0)) if bool(args.compute_ssim) else None,
            "mean_direction_cosine": _mean([float(r["direction_cosine"]) for r in direction_rows]),
            "mean_energy_ratio_l1": _mean([float(r["energy_ratio_l1"]) for r in direction_rows]),
            "mean_covered_fraction": _mean([float(r["covered_fraction"]) for r in per_view]),
            "all_axis_pass": bool(all_axis_pass),
        },
        "per_view": per_view,
        "verdict": verdict,
        "final_status": "DIAGNOSTIC_COMPLETE_NOT_METHOD_COMPLETE",
    }
    _write_report(output_dir, payload)

    if bool(args.enable_wandb):
        import wandb

        run = wandb.init(project=str(args.wandb_project), name=str(args.wandb_run_name), dir=str(output_dir))
        run.config.update(payload["config"])
        flat = {
            "diagnostic/best_alpha": float(payload["summary"]["best_alpha"]),
            "diagnostic/best_psnr_gain": float(payload["summary"]["best_psnr_gain"]),
            "diagnostic/best_changed_fraction": float(payload["summary"]["best_changed_fraction"]),
            "diagnostic/mean_direction_cosine": float(payload["summary"]["mean_direction_cosine"]),
            "diagnostic/mean_energy_ratio_l1": float(payload["summary"]["mean_energy_ratio_l1"]),
            "diagnostic/all_axis_pass": float(bool(payload["summary"]["all_axis_pass"])),
        }
        if payload["summary"]["best_ssim_gain"] is not None:
            flat["diagnostic/best_ssim_gain"] = float(payload["summary"]["best_ssim_gain"])
        run.log(flat)
        run.summary.update(flat)
        run.finish()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--base_method_name", default="ours_26000")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout_stride", type=int, default=4)
    parser.add_argument("--heldout_offset", type=int, default=0)
    parser.add_argument("--max_heldout_views", type=int, default=0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--alpha_grid", default="0,0.125,0.25,0.5,0.75,1")
    parser.add_argument("--residual_clip", type=float, default=0.25)
    parser.add_argument("--min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--depth_abs_tol", type=float, default=0.02)
    parser.add_argument("--depth_rel_tol", type=float, default=0.03)
    parser.add_argument("--direction_weight", type=float, default=0.35)
    parser.add_argument("--evidence_max_side", type=int, default=512)
    parser.add_argument("--compute_ssim", action="store_true")
    parser.add_argument("--ssim_max_side", type=int, default=384)
    parser.add_argument("--min_changed_fraction", type=float, default=1.0e-4)
    parser.add_argument("--save_example_views", type=int, default=0)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-transport-diagnostics")
    parser.add_argument("--wandb_run_name", default="source-heldout-ela-transport")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"summary": payload["summary"], "report": str(Path(args.output_dir) / "source_heldout_ela_transport_report.json")}, indent=2))


if __name__ == "__main__":
    main()

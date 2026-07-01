#!/usr/bin/env python3
"""Probe target-neighbor candidate reranking without using target GT for choice.

The probe recomputes every candidate variant for each target view, scores each
candidate by target-neighbor render self-consistency, and then reports the
post-hoc target-GT metrics of the candidate selected by that target-blind score.
Target GT is used only for analysis after the candidate choice is made.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model import apply_source_heldout_support_transport_calibrator as app  # noqa: E402
from utils.evidence_lumigraph_adapter import FrameLoader, compute_evidence_signal, load_split_frames  # noqa: E402


def _scene_args(report: dict[str, Any], cli: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace()
    policy = report["policy"]
    evidence = report["evidence_config"]
    tnc = report.get("target_neighbor_consistency_policy", {})

    args.device = cli.device
    args.checkpoint = str(report["checkpoint"])
    args.base_model_path = str(report["base_model_path"])
    args.base_method_name = str(report.get("base_method_name", "ours"))
    args.target_split = str(report["split"]["target_split"])
    args.max_target_views = int(cli.max_target_views)
    args.support_source_mode = str(report["split"].get("support_source_mode", "source_split"))
    args.heldout_stride = int(cli.heldout_stride)
    args.heldout_offset = int(cli.heldout_offset)

    args.anchor_alpha = float(policy["anchor_alpha"])
    args.learned_scale = float(policy["learned_scale"])
    args.blend = float(policy["blend"])
    args.enable_candidate_ladder = bool(policy.get("enable_candidate_ladder", False))
    args.candidate_ladder_blends = str(policy.get("candidate_ladder_blends", ""))

    args.k = int(evidence["k"])
    args.residual_clip = float(evidence["residual_clip"])
    args.min_confidence = float(evidence["min_confidence"])
    args.depth_abs_tol = float(evidence["depth_abs_tol"])
    args.depth_rel_tol = float(evidence["depth_rel_tol"])
    args.direction_weight = float(evidence["direction_weight"])
    args.evidence_max_side = int(evidence["evidence_max_side"])
    args.compute_ssim = bool(report.get("eval_config", {}).get("compute_ssim", True))
    args.ssim_max_side = int(report.get("eval_config", {}).get("ssim_max_side", 256))

    args.target_neighbor_consistency_neighbor_k = int(tnc.get("neighbor_k", cli.neighbor_k))
    args.target_neighbor_consistency_direction_weight = float(tnc.get("direction_weight", cli.direction_weight))
    args.target_neighbor_consistency_max_side = int(tnc.get("max_side", cli.max_side))
    args.target_neighbor_consistency_depth_abs_tol = float(tnc.get("depth_abs_tol", cli.depth_abs_tol))
    args.target_neighbor_consistency_depth_rel_tol = float(tnc.get("depth_rel_tol", cli.depth_rel_tol))
    args.target_neighbor_consistency_min_confidence = float(tnc.get("min_confidence", cli.min_confidence))
    args.target_neighbor_consistency_min_effective_weight = float(tnc.get("min_effective_weight", cli.min_effective_weight))
    return args


def _summarize(rows: list[dict[str, Any]], variant_key: str) -> dict[str, Any]:
    psnr = [float(row[variant_key]["psnr_gain"]) for row in rows]
    ssim = [float(row[variant_key].get("ssim_gain", 0.0)) for row in rows]
    return {
        "psnr_gain": float(mean(psnr)) if psnr else 0.0,
        "ssim_gain": float(mean(ssim)) if ssim else 0.0,
        "view_count": int(len(rows)),
        "positive_psnr_fraction": float(sum(x > 0.0 for x in psnr) / len(psnr)) if psnr else 0.0,
        "min_psnr_gain": float(min(psnr)) if psnr else 0.0,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Target-Neighbor Candidate Rerank Probe",
        "",
        "Target GT is used only after target-blind candidate selection for analysis.",
        "",
        "## Macro",
        "",
        "| metric | current | fixed | learned | pure_tnc | oracle | pure_tnc-current | oracle-current |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    macro = payload["macro"]
    for metric in ["psnr_gain", "ssim_gain"]:
        lines.append(
            "| {metric} | {current:.12f} | {fixed:.12f} | {learned:.12f} | {tnc:.12f} | {oracle:.12f} | {dtnc:+.12f} | {doracle:+.12f} |".format(
                metric=metric,
                current=float(macro["current"][metric]),
                fixed=float(macro["fixed"][metric]),
                learned=float(macro["learned"][metric]),
                tnc=float(macro["pure_tnc"][metric]),
                oracle=float(macro["oracle"][metric]),
                dtnc=float(macro["pure_tnc"][metric]) - float(macro["current"][metric]),
                doracle=float(macro["oracle"][metric]) - float(macro["current"][metric]),
            )
        )
    lines += [
        "",
        "## Per Scene",
        "",
        "| scene | current PSNR | pure_tnc PSNR | oracle PSNR | pure_tnc-current | oracle-current | TNC/GT match | pure_tnc best counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["scenes"]:
        current = row["summaries"]["current"]["psnr_gain"]
        pure_tnc = row["summaries"]["pure_tnc"]["psnr_gain"]
        oracle = row["summaries"]["oracle"]["psnr_gain"]
        lines.append(
            f"| {row['scene']} | {current:.12f} | {pure_tnc:.12f} | {oracle:.12f} | "
            f"{pure_tnc-current:+.12f} | {oracle-current:+.12f} | "
            f"{row['pure_tnc_matches_oracle']}/{row['view_count']} | {row['pure_tnc_variant_counts']} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        str(payload.get("verdict", "")),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    report_root = Path(args.report_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_names = [part.strip() for part in str(args.scenes).split(",") if part.strip()]

    scene_payloads: list[dict[str, Any]] = []
    for scene in scene_names:
        report_path = report_root / scene / "support_transport_apply_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scene_args = _scene_args(report, args)
        model, feature_mean, feature_std, _ = app._load_model(Path(scene_args.checkpoint), device)
        base_model = Path(scene_args.base_model_path)
        train_frames = load_split_frames(base_model, "train", scene_args.base_method_name)
        if scene_args.support_source_mode == "source_split":
            source_frames, _ = app._split_source_heldout(
                train_frames,
                int(scene_args.heldout_stride),
                int(scene_args.heldout_offset),
            )
        elif scene_args.support_source_mode == "all_train":
            source_frames = train_frames
        else:
            raise ValueError(f"unsupported support_source_mode: {scene_args.support_source_mode}")
        target_frames = load_split_frames(base_model, scene_args.target_split, scene_args.base_method_name)
        if int(scene_args.max_target_views) > 0:
            target_frames = target_frames[: int(scene_args.max_target_views)]
        loader = FrameLoader(device=device)
        variants = app._candidate_variant_names(scene_args)
        report_by_view = {str(row["view"]): row for row in report.get("per_view", [])}
        rows: list[dict[str, Any]] = []
        for target in tqdm(target_frames, desc=f"probe TNC rerank {scene}"):
            with torch.no_grad():
                ev = compute_evidence_signal(
                    target,
                    source_frames,
                    k=int(scene_args.k),
                    mode="residual",
                    residual_clip=float(scene_args.residual_clip),
                    min_confidence=float(scene_args.min_confidence),
                    depth_abs_tol=float(scene_args.depth_abs_tol),
                    depth_rel_tol=float(scene_args.depth_rel_tol),
                    direction_weight=float(scene_args.direction_weight),
                    evidence_max_side=int(scene_args.evidence_max_side),
                    loader=loader,
                    device=device,
                )
                features = app._build_features(ev, k=int(scene_args.k)).unsqueeze(0).to(device=device, dtype=torch.float32)
                signal = ev.signal.unsqueeze(0).to(device=device, dtype=torch.float32)
                valid = ev.valid.unsqueeze(0).to(device=device, dtype=torch.float32)
                pred_delta = model(app._normalize(features, feature_mean, feature_std), signal, valid).squeeze(0)
                deltas = app._candidate_deltas(ev, pred_delta, scene_args)
                scores: dict[str, Any] = {}
                for variant in variants:
                    image = torch.clamp(ev.base + deltas[variant], 0.0, 1.0)
                    score = app._target_neighbor_consistency_score(
                        target=target,
                        image=image,
                        target_frames=target_frames,
                        loader=loader,
                        device=device,
                        args=scene_args,
                    )
                    scores[variant] = score
                available = [
                    variant
                    for variant in variants
                    if scores[variant].get("mean_mae_to_neighbor_base") is not None
                    and float(scores[variant].get("total_effective_weight", 0.0))
                    >= float(scene_args.target_neighbor_consistency_min_effective_weight)
                ]
                pure_tnc_variant = min(
                    available,
                    key=lambda variant: float(scores[variant]["mean_mae_to_neighbor_base"]),
                ) if available else "fixed"
                gt = loader.gt(str(target.gt_path)).to(device=device, dtype=torch.float32)
                metrics = {
                    variant: app._image_metrics(
                        ev.base,
                        gt,
                        deltas[variant],
                        compute_ssim=bool(scene_args.compute_ssim),
                        ssim_max_side=int(scene_args.ssim_max_side),
                    )
                    for variant in variants
                }
                oracle_variant = max(variants, key=lambda variant: float(metrics[variant]["psnr_gain"]))
                current_metrics = dict((report_by_view.get(str(target.name)) or {}).get("selected", metrics["fixed"]))
                rows.append(
                    {
                        "view": str(target.name),
                        "current_variant": str((report_by_view.get(str(target.name)) or {}).get("output_variant", "unknown")),
                        "pure_tnc_variant": pure_tnc_variant,
                        "oracle_variant": oracle_variant,
                        "current": current_metrics,
                        "fixed": metrics["fixed"],
                        "learned": metrics.get("learned", metrics["fixed"]),
                        "pure_tnc": metrics[pure_tnc_variant],
                        "oracle": metrics[oracle_variant],
                        "candidate_metrics": metrics,
                        "target_neighbor_scores": scores,
                    }
                )
        pure_tnc_counts: dict[str, int] = {}
        oracle_counts: dict[str, int] = {}
        for row in rows:
            pure_tnc_counts[row["pure_tnc_variant"]] = int(pure_tnc_counts.get(row["pure_tnc_variant"], 0)) + 1
            oracle_counts[row["oracle_variant"]] = int(oracle_counts.get(row["oracle_variant"], 0)) + 1
        scene_payloads.append(
            {
                "scene": scene,
                "report": str(report_path),
                "view_count": len(rows),
                "pure_tnc_matches_oracle": int(sum(row["pure_tnc_variant"] == row["oracle_variant"] for row in rows)),
                "pure_tnc_variant_counts": pure_tnc_counts,
                "oracle_variant_counts": oracle_counts,
                "summaries": {
                    "current": _summarize(rows, "current"),
                    "fixed": _summarize(rows, "fixed"),
                    "learned": _summarize(rows, "learned"),
                    "pure_tnc": _summarize(rows, "pure_tnc"),
                    "oracle": _summarize(rows, "oracle"),
                },
                "rows": rows,
            }
        )

    macro: dict[str, Any] = {}
    for key in ["current", "fixed", "learned", "pure_tnc", "oracle"]:
        macro[key] = {
            "psnr_gain": float(mean(scene["summaries"][key]["psnr_gain"] for scene in scene_payloads)),
            "ssim_gain": float(mean(scene["summaries"][key]["ssim_gain"] for scene in scene_payloads)),
        }
    payload = {
        "method": "target-neighbor candidate rerank probe",
        "report_root": str(report_root),
        "scenes": scene_payloads,
        "macro": macro,
        "target_gt_usage": "Target GT is used only after target-blind pure_tnc selection for analysis.",
    }
    payload["verdict"] = (
        "pure_tnc is useful for measuring candidate-selection headroom. Promote it only if full9 macro improves "
        "over current without unacceptable scene regressions."
    )
    json_path = output_dir / "target_neighbor_candidate_rerank_probe.json"
    md_path = output_dir / "target_neighbor_candidate_rerank_probe.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)

    if bool(args.enable_wandb):
        try:
            import wandb

            run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, dir=str(output_dir), config=vars(args))
            flat = {}
            for key, row in macro.items():
                flat[f"tnc_rerank/{key}/psnr_gain"] = row["psnr_gain"]
                flat[f"tnc_rerank/{key}/ssim_gain"] = row["ssim_gain"]
            flat["tnc_rerank/pure_tnc_minus_current_psnr_gain"] = macro["pure_tnc"]["psnr_gain"] - macro["current"]["psnr_gain"]
            flat["tnc_rerank/oracle_minus_current_psnr_gain"] = macro["oracle"]["psnr_gain"] - macro["current"]["psnr_gain"]
            run.log(flat)
            run.finish()
        except Exception as exc:  # pragma: no cover - wandb should not break artifacts.
            payload["wandb_error"] = repr(exc)
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            _write_markdown(md_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--scenes",
        default="bicycle,flowers,garden,stump,counter,treehill,bonsai,room,kitchen",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_target_views", type=int, default=0)
    parser.add_argument("--heldout_stride", type=int, default=4)
    parser.add_argument("--heldout_offset", type=int, default=0)
    parser.add_argument("--neighbor_k", type=int, default=2)
    parser.add_argument("--direction_weight", type=float, default=0.35)
    parser.add_argument("--max_side", type=int, default=256)
    parser.add_argument("--depth_abs_tol", type=float, default=0.03)
    parser.add_argument("--depth_rel_tol", type=float, default=0.04)
    parser.add_argument("--min_confidence", type=float, default=1.0e-4)
    parser.add_argument("--min_effective_weight", type=float, default=0.01)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-transport-diagnostics")
    parser.add_argument("--wandb_run_name", default="target_neighbor_candidate_rerank_probe")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"macro": payload["macro"], "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()

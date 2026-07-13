#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(model_dir: Path, method_name: str | None = None) -> dict[str, float]:
    path = model_dir / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing metrics file: {path}")
    payload = read_json(path)
    if method_name and method_name in payload:
        row = payload[method_name]
    elif len(payload) == 1:
        row = next(iter(payload.values()))
    else:
        raise KeyError(f"{path} has multiple methods; pass an explicit method name")
    missing = [key for key in METRICS if key not in row]
    if missing:
        raise KeyError(f"{path} row is missing metric(s): {','.join(missing)}")
    return {key: float(row[key]) for key in METRICS}


def audit_summary(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "surface_residual_region_texture_adapter_audit.json"
    if not path.is_file():
        return {}
    audit = read_json(path)
    fit = audit.get("fit_summary", {})
    apply = audit.get("target_apply", {})
    policy = audit.get("policy_val", {})
    return {
        "accepted": bool(audit.get("accepted", False)),
        "selected_alpha": float(audit.get("selected_alpha", 0.0)),
        "selected_texture_size": int(fit.get("selected_texture_size", fit.get("texture_size", 0))),
        "selected_fill": str(fit.get("selected_atlas_empty_bin_fill_mode", fit.get("atlas_empty_bin_fill_mode", ""))),
        "selected_support_mode": str(fit.get("selected_support_mode", fit.get("support_mode", ""))),
        "selected_support_added_faces": int(fit.get("selected_support_added_faces", fit.get("support_added_faces", 0))),
        "atlas_faces": int(fit.get("atlas_faces", 0)),
        "fit_samples": int(fit.get("fit_samples", 0)),
        "changed_fraction": float(apply.get("changed_fraction", 0.0)),
        "policy_val_relative_gain": float((policy.get("best") or {}).get("relative_gain", 0.0)),
        "policy_val_ssim_gain": float((policy.get("best") or {}).get("ssim_gain", 0.0)),
        "accepted_candidate_count": int((policy.get("fill_mode_selection") or {}).get("accepted_candidate_count", 0)),
    }


def strict_win(delta: dict[str, float], eps: float) -> bool:
    return (
        delta.get("PSNR", 0.0) > eps
        and delta.get("SSIM", 0.0) > eps
        and delta.get("LPIPS", 0.0) < -eps
    )


def nonregressive(delta: dict[str, float], eps: float) -> bool:
    return (
        delta.get("PSNR", 0.0) >= -eps
        and delta.get("SSIM", 0.0) >= -eps
        and delta.get("LPIPS", 0.0) <= eps
    )


def format_float(value: float, digits: int = 6) -> str:
    return f"{float(value):+.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize surface residual atlas multiscene results.")
    parser.add_argument("--root", default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
    parser.add_argument("--scenes", default="garden,room,counter,bonsai")
    parser.add_argument("--method_tag", required=True, help="Scene suffix after '<scene>_', e.g. v48_..._adapter.")
    parser.add_argument("--method_label", default="", help="Human-readable label. Defaults to method_tag.")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Comparison as label=tag. Use tag 'evidence_noop_compact_baseline' for the no-op rows.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    parser.add_argument("--eps", type=float, default=1.0e-12)
    args = parser.parse_args()

    root = Path(args.root)
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    method_label = str(args.method_label or args.method_tag)
    compares: list[tuple[str, str]] = []
    for item in args.compare:
        if "=" not in item:
            raise ValueError(f"--compare expects label=tag, got {item}")
        label, tag = item.split("=", 1)
        compares.append((label.strip(), tag.strip()))

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for scene in scenes:
        method_dir = root / f"{scene}_{args.method_tag}"
        method_metrics = metric_row(method_dir)
        audit = audit_summary(method_dir)
        row: dict[str, Any] = {
            "scene": scene,
            "method_dir": str(method_dir),
            "metrics": method_metrics,
            "audit": audit,
            "comparisons": {},
        }
        for label, tag in compares:
            compare_dir = root / f"{scene}_{tag}"
            compare_metrics = metric_row(compare_dir)
            delta = {
                key: float(method_metrics.get(key, 0.0) - compare_metrics.get(key, 0.0))
                for key in METRICS
                if key in method_metrics and key in compare_metrics
            }
            row["comparisons"][label] = {
                "compare_dir": str(compare_dir),
                "metrics": compare_metrics,
                "delta": delta,
                "strict_win": strict_win(delta, float(args.eps)),
                "nonregressive_or_tie": nonregressive(delta, float(args.eps)),
            }
        rows.append(row)

    for label, _tag in compares:
        deltas = [row["comparisons"][label]["delta"] for row in rows if label in row["comparisons"]]
        scene_count = len(deltas)
        if scene_count:
            summary[label] = {
                "scene_count": int(scene_count),
                "strict_wins": int(sum(strict_win(delta, float(args.eps)) for delta in deltas)),
                "nonregressive_or_tie": int(sum(nonregressive(delta, float(args.eps)) for delta in deltas)),
                "mean_dPSNR": float(sum(delta.get("PSNR", 0.0) for delta in deltas) / scene_count),
                "mean_dSSIM": float(sum(delta.get("SSIM", 0.0) for delta in deltas) / scene_count),
                "mean_dLPIPS": float(sum(delta.get("LPIPS", 0.0) for delta in deltas) / scene_count),
            }

    payload = {
        "method": method_label,
        "method_tag": str(args.method_tag),
        "root": str(root),
        "scenes": scenes,
        "comparisons": [{"label": label, "tag": tag} for label, tag in compares],
        "rows": rows,
        "summary": summary,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if args.output_md:
        lines = [
            f"# {method_label} Multiscene Summary",
            "",
            f"- root: `{root}`",
            f"- method tag: `{args.method_tag}`",
            f"- scenes: `{','.join(scenes)}`",
            "",
            "## Method Rows",
            "",
            "| scene | PSNR | SSIM | LPIPS | support | +faces | texture | fill | alpha | changed |",
            "|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
        ]
        for row in rows:
            audit = row.get("audit", {})
            metrics = row.get("metrics", {})
            lines.append(
                f"| {row['scene']} | {metrics.get('PSNR', 0.0):.6f} | {metrics.get('SSIM', 0.0):.8f} | "
                f"{metrics.get('LPIPS', 0.0):.8f} | {audit.get('selected_support_mode', '')} | "
                f"{audit.get('selected_support_added_faces', 0)} | {audit.get('selected_texture_size', 0)} | "
                f"{audit.get('selected_fill', '')} | {audit.get('selected_alpha', 0.0):.4f} | "
                f"{100.0 * float(audit.get('changed_fraction', 0.0)):.4f}% |"
            )
        lines.extend(["", "## Comparisons", ""])
        for label, stats in summary.items():
            lines.extend(
                [
                    f"### vs {label}",
                    "",
                    f"- strict scene wins: `{stats['strict_wins']} / {stats['scene_count']}`",
                    f"- nonregressive/tie: `{stats['nonregressive_or_tie']} / {stats['scene_count']}`",
                    f"- mean dPSNR: `{format_float(stats['mean_dPSNR'])}`",
                    f"- mean dSSIM: `{format_float(stats['mean_dSSIM'], 8)}`",
                    f"- mean dLPIPS: `{format_float(stats['mean_dLPIPS'], 8)}`",
                    "",
                    "| scene | dPSNR | dSSIM | dLPIPS | strict | nonreg/tie |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in rows:
                comp = row["comparisons"].get(label, {})
                delta = comp.get("delta", {})
                lines.append(
                    f"| {row['scene']} | {format_float(delta.get('PSNR', 0.0))} | "
                    f"{format_float(delta.get('SSIM', 0.0), 8)} | "
                    f"{format_float(delta.get('LPIPS', 0.0), 8)} | "
                    f"{int(bool(comp.get('strict_win', False)))} | "
                    f"{int(bool(comp.get('nonregressive_or_tie', False)))} |"
                )
            lines.append("")
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_json": str(output_json), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

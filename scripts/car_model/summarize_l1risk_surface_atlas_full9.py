#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_metric_row(path: Path) -> dict[str, float]:
    payload = read_json(path)
    if not payload:
        raise ValueError(f"empty metrics file: {path}")
    row = next(iter(payload.values()))
    return {key: float(row[key]) for key in METRICS}


def metric_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a[key] - b[key]) for key in METRICS}


def strict_win(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] > eps and delta["SSIM"] > eps and delta["LPIPS"] < -eps


def nonreg(delta: dict[str, float], eps: float) -> bool:
    return delta["PSNR"] >= -eps and delta["SSIM"] >= -eps and delta["LPIPS"] <= eps


def find_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def audit_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    audit = read_json(path)
    fit = audit.get("fit_summary", {}) or {}
    apply = audit.get("target_apply", {}) or {}
    policy = audit.get("policy_val", {}) or {}
    best = policy.get("best", {}) or {}
    return {
        "accepted": bool(audit.get("accepted", False)),
        "effective_policy": str(audit.get("effective_policy", "")),
        "fallback_written": bool(audit.get("fallback_written", False)),
        "reject_reason": str(audit.get("reject_reason", "")),
        "selected_alpha": float(audit.get("selected_alpha", 0.0)),
        "support": str(fit.get("selected_support_mode", fit.get("support_mode", ""))),
        "added_faces": int(fit.get("selected_support_added_faces", fit.get("support_added_faces", 0))),
        "texture": int(fit.get("selected_texture_size", fit.get("texture_size", 0))),
        "fill": str(fit.get("selected_atlas_empty_bin_fill_mode", fit.get("atlas_empty_bin_fill_mode", ""))),
        "changed_fraction": float(apply.get("changed_fraction", 0.0)),
        "policy_val_relative_gain": float(best.get("relative_gain", 0.0)),
        "policy_val_positive_view_fraction": float(best.get("positive_view_fraction", 0.0)),
        "policy_val_min_view_relative_gain": float(best.get("min_view_relative_gain", 0.0)),
        "policy_val_cvar20_relative_gain": float(best.get("cvar20_view_relative_gain", 0.0)),
        "policy_val_ssim_gain": float(best.get("ssim_gain", 0.0)),
        "policy_val_ssim_positive_view_fraction": float(best.get("ssim_positive_view_fraction", 0.0)),
        "policy_val_ssim_min_view_gain": float(best.get("ssim_min_view_gain", 0.0)),
        "policy_val_image_l1_gain": float(best.get("image_l1_gain", 0.0)),
        "policy_val_image_l1_positive_view_fraction": float(best.get("image_l1_positive_view_fraction", 0.0)),
        "policy_val_image_l1_min_view_gain": float(best.get("image_l1_min_view_gain", 0.0)),
        "policy_val_image_l1_cvar20_view_gain": float(best.get("image_l1_cvar20_view_gain", 0.0)),
    }


def scene_paths(
    scene: str,
    v49_root: Path,
    v49_tag: str,
    v48_root: Path,
    v48_tag: str,
    durable_root: Path,
) -> dict[str, Path | None]:
    v49_dir = v49_root / f"{scene}_{v49_tag}"
    v48_dir = v48_root / f"{scene}_{v48_tag}"
    if not (v48_dir / "results.json").is_file():
        v48_dir = durable_root / f"{scene}_{v48_tag}"
    if not (v48_dir / "results.json").is_file():
        v48_dir = durable_root / "v48_full9_missing_scene_small_artifacts_20260623" / "scenes" / scene
    noop_dir = durable_root / f"{scene}_evidence_noop_compact_baseline"
    return {
        "v49_dir": v49_dir,
        "v49_results": find_first([v49_dir / "results.json"]),
        "v49_audit": find_first([v49_dir / "surface_residual_region_texture_adapter_audit.json"]),
        "v48_dir": v48_dir,
        "v48_results": find_first([v48_dir / "results.json"]),
        "v48_audit": find_first([v48_dir / "surface_residual_region_texture_adapter_audit.json"]),
        "noop_dir": noop_dir,
        "noop_results": find_first([noop_dir / "results.json"]),
    }


def summarize(rows: list[dict[str, Any]], label: str, eps: float) -> dict[str, Any]:
    deltas = [row["comparisons"][label]["delta"] for row in rows if label in row.get("comparisons", {})]
    if not deltas:
        return {
            "scene_count": 0,
            "strict_wins": 0,
            "nonregressive_or_tie": 0,
            "mean_dPSNR": 0.0,
            "mean_dSSIM": 0.0,
            "mean_dLPIPS": 0.0,
        }
    count = len(deltas)
    return {
        "scene_count": int(count),
        "strict_wins": int(sum(strict_win(delta, eps) for delta in deltas)),
        "nonregressive_or_tie": int(sum(nonreg(delta, eps) for delta in deltas)),
        "mean_dPSNR": float(sum(delta["PSNR"] for delta in deltas) / count),
        "mean_dSSIM": float(sum(delta["SSIM"] for delta in deltas) / count),
        "mean_dLPIPS": float(sum(delta["LPIPS"] for delta in deltas) / count),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# v49 L1-risk auto-noop surface atlas summary",
        "",
        f"- scenes requested: `{len(payload['scenes'])}`",
        f"- completed scenes: `{len(payload['rows'])}`",
        f"- missing scenes: `{','.join(payload['missing_scenes']) or 'none'}`",
        f"- v49 root: `{payload['v49_root']}`",
        f"- v48 root: `{payload['v48_root']}`",
        f"- durable root: `{payload['durable_root']}`",
        "",
        "## Aggregate",
        "",
        "| comparison | scenes | strict | nonreg/tie | mean dPSNR | mean dSSIM | mean dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in payload["summary"].items():
        lines.append(
            f"| v49 vs {label} | {stats['scene_count']} | {stats['strict_wins']} | "
            f"{stats['nonregressive_or_tie']} | {fmt(stats['mean_dPSNR'])} | "
            f"{fmt(stats['mean_dSSIM'], 8)} | {fmt(stats['mean_dLPIPS'], 8)} |"
        )
    lines.extend(
        [
            "",
            "## Per-scene",
            "",
            "| scene | accepted | effective | support | tex | fill | alpha | changed | PSNR | SSIM | LPIPS | dPSNR noop | dSSIM noop | dLPIPS noop | dPSNR v48 | dSSIM v48 | dLPIPS v48 |",
            "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        audit = row["audit"]
        metrics = row["metrics"]
        noop = row["comparisons"].get("noop", {}).get("delta", {})
        v48 = row["comparisons"].get("v48", {}).get("delta", {})
        lines.append(
            f"| {row['scene']} | {int(audit.get('accepted', False))} | {audit.get('effective_policy', '')} | "
            f"{audit.get('support', '')} | {audit.get('texture', 0)} | {audit.get('fill', '')} | "
            f"{audit.get('selected_alpha', 0.0):.5f} | {100.0 * float(audit.get('changed_fraction', 0.0)):.4f}% | "
            f"{metrics['PSNR']:.6f} | {metrics['SSIM']:.8f} | {metrics['LPIPS']:.8f} | "
            f"{fmt(noop.get('PSNR', 0.0))} | {fmt(noop.get('SSIM', 0.0), 8)} | {fmt(noop.get('LPIPS', 0.0), 8)} | "
            f"{fmt(v48.get('PSNR', 0.0))} | {fmt(v48.get('SSIM', 0.0), 8)} | {fmt(v48.get('LPIPS', 0.0), 8)} |"
        )
    lines.extend(["", "## Policy Diagnostics", ""])
    for row in payload["rows"]:
        audit = row["audit"]
        lines.extend(
            [
                f"### {row['scene']}",
                "",
                f"- accepted: `{audit.get('accepted', False)}`",
                f"- effective policy: `{audit.get('effective_policy', '')}`",
                f"- fallback written: `{audit.get('fallback_written', False)}`",
                f"- reject reason: `{audit.get('reject_reason', '')}`",
                f"- policy-val relative gain: `{audit.get('policy_val_relative_gain', 0.0):.9f}`",
                f"- policy-val ssim gain: `{audit.get('policy_val_ssim_gain', 0.0):.9f}`",
                f"- policy-val image L1 gain: `{audit.get('policy_val_image_l1_gain', 0.0):.9f}`",
                f"- policy-val image L1 positive-view fraction: `{audit.get('policy_val_image_l1_positive_view_fraction', 0.0):.6f}`",
                f"- policy-val image L1 min gain: `{audit.get('policy_val_image_l1_min_view_gain', 0.0):.9f}`",
                f"- policy-val image L1 CVaR20 gain: `{audit.get('policy_val_image_l1_cvar20_view_gain', 0.0):.9f}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize v49 L1-risk auto-noop atlas outputs against v48 and no-op.")
    parser.add_argument(
        "--scenes",
        default="bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai",
    )
    parser.add_argument("--v49_root", default="/dev/shm/peilincai_spcarnet_v49_l1risk_autonoop_20260623")
    parser.add_argument(
        "--v49_tag",
        default="v49_l1risk_autonoop_autosupport_autocap_guarded_v42calib_region_texture_adapter",
    )
    parser.add_argument("--v48_root", default="/dev/shm/peilincai_spcarnet_v48_full9_20260623")
    parser.add_argument(
        "--v48_tag",
        default="v48_autosupport_autocap_guarded_v42calib_region_texture_adapter",
    )
    parser.add_argument("--durable_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--eps", type=float, default=1.0e-12)
    args = parser.parse_args()

    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    v49_root = Path(args.v49_root)
    v48_root = Path(args.v48_root)
    durable_root = Path(args.durable_root)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for scene in scenes:
        paths = scene_paths(scene, v49_root, args.v49_tag, v48_root, args.v48_tag, durable_root)
        required = [paths["v49_results"], paths["v48_results"], paths["noop_results"]]
        if any(path is None for path in required):
            missing.append(scene)
            if not args.allow_missing:
                missing_text = {key: str(value) for key, value in paths.items()}
                raise FileNotFoundError(f"missing required file for {scene}: {missing_text}")
            continue
        v49_metrics = first_metric_row(paths["v49_results"])  # type: ignore[arg-type]
        comparisons: dict[str, Any] = {}
        for label, key in (("noop", "noop_results"), ("v48", "v48_results")):
            base_metrics = first_metric_row(paths[key])  # type: ignore[arg-type]
            delta = metric_delta(v49_metrics, base_metrics)
            comparisons[label] = {
                "metrics": base_metrics,
                "delta": delta,
                "strict_win": strict_win(delta, float(args.eps)),
                "nonregressive_or_tie": nonreg(delta, float(args.eps)),
            }
        rows.append(
            {
                "scene": scene,
                "paths": {key: str(value) if value is not None else "" for key, value in paths.items()},
                "metrics": v49_metrics,
                "audit": audit_summary(paths["v49_audit"]),
                "comparisons": comparisons,
            }
        )

    payload = {
        "scenes": scenes,
        "rows": rows,
        "missing_scenes": missing,
        "v49_root": str(v49_root),
        "v48_root": str(v48_root),
        "durable_root": str(durable_root),
        "summary": {
            "noop": summarize(rows, "noop", float(args.eps)),
            "v48": summarize(rows, "v48", float(args.eps)),
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps({"output_json": str(output_json), "output_md": str(args.output_md), "missing": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

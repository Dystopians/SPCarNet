#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric(results: dict[str, Any], method: str) -> dict[str, float] | None:
    value = results.get(method)
    if not isinstance(value, dict):
        return None
    return {k: float(value[k]) for k in ("PSNR", "SSIM", "LPIPS")}


def delta(method: dict[str, float] | None, base: dict[str, float] | None) -> dict[str, float | None]:
    if method is None or base is None:
        return {"dPSNR": None, "dSSIM": None, "dLPIPS": None}
    return {
        "dPSNR": method["PSNR"] - base["PSNR"],
        "dSSIM": method["SSIM"] - base["SSIM"],
        "dLPIPS": method["LPIPS"] - base["LPIPS"],
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def strict_rgb_win(d: dict[str, float | None]) -> bool:
    return d["dPSNR"] is not None and d["dPSNR"] > 0 and d["dSSIM"] > 0 and d["dLPIPS"] < 0


def read_ela_report(model: Path, split: str, method: str) -> dict[str, Any]:
    return load_json(model / split / method / "ela_report.json")


def collect(args: argparse.Namespace) -> dict[str, Any]:
    clean_root = Path(args.clean9000_root)
    f82_root = Path(args.f82_root)
    ela7_root = Path(args.ela7_root)
    rows = []

    clean_results = load_json(clean_root / "results.json")
    clean_base = metric(clean_results, "ours_9000")
    clean_method = metric(clean_results, "ours_9000_phasej_external_clean9000_micro_autoedge_ela")
    clean_report = read_ela_report(clean_root, "test", "ours_9000_phasej_external_clean9000_micro_autoedge_ela")
    ela7_metric = metric(load_json(ela7_root / "results.json"), "ours_9000_ela7_pareto_portfolio")
    rows.append(
        {
            "scene": "courtyard",
            "protocol": "ETH3D courtyard clean9000",
            "base_method": "ours_9000",
            "method": "ours_9000_phasej_external_clean9000_micro_autoedge_ela",
            "comparison": "vs clean9000",
            "wandb_run": "vne962ci",
            "alpha": clean_report.get("alpha"),
            "edge_q": (clean_report.get("policy") or {}).get("edge_gate_quantile"),
            "k": (clean_report.get("policy") or {}).get("k"),
            "depth_rel_tol": (clean_report.get("policy") or {}).get("depth_rel_tol"),
            "covered_fraction": clean_report.get("mean_covered_fraction"),
            "method_PSNR": None if clean_method is None else clean_method["PSNR"],
            "method_SSIM": None if clean_method is None else clean_method["SSIM"],
            "method_LPIPS": None if clean_method is None else clean_method["LPIPS"],
            "base_PSNR": None if clean_base is None else clean_base["PSNR"],
            "base_SSIM": None if clean_base is None else clean_base["SSIM"],
            "base_LPIPS": None if clean_base is None else clean_base["LPIPS"],
            **delta(clean_method, clean_base),
            "strict_rgb_win": strict_rgb_win(delta(clean_method, clean_base)),
            "note": "positive external clean-checkpoint validation; test metrics used only after train-only policy materialization",
        }
    )
    d_ela7 = delta(clean_method, ela7_metric)
    rows.append(
        {
            "scene": "courtyard",
            "protocol": "ETH3D courtyard clean9000",
            "base_method": "ours_9000_ela7_pareto_portfolio",
            "method": "ours_9000_phasej_external_clean9000_micro_autoedge_ela",
            "comparison": "vs older ELA7 portfolio",
            "wandb_run": "vne962ci",
            "alpha": clean_report.get("alpha"),
            "edge_q": (clean_report.get("policy") or {}).get("edge_gate_quantile"),
            "k": (clean_report.get("policy") or {}).get("k"),
            "depth_rel_tol": (clean_report.get("policy") or {}).get("depth_rel_tol"),
            "covered_fraction": clean_report.get("mean_covered_fraction"),
            "method_PSNR": None if clean_method is None else clean_method["PSNR"],
            "method_SSIM": None if clean_method is None else clean_method["SSIM"],
            "method_LPIPS": None if clean_method is None else clean_method["LPIPS"],
            "base_PSNR": None if ela7_metric is None else ela7_metric["PSNR"],
            "base_SSIM": None if ela7_metric is None else ela7_metric["SSIM"],
            "base_LPIPS": None if ela7_metric is None else ela7_metric["LPIPS"],
            **d_ela7,
            "strict_rgb_win": strict_rgb_win(d_ela7),
            "note": "improves PSNR/SSIM over ELA7 but has slightly worse LPIPS, so this is not a strict dominance claim",
        }
    )

    f82_results = load_json(f82_root / "results.json")
    f82_base = metric(f82_results, "ours_26000")
    f82_methods = [
        ("ours_26000_phasej_external_fixededge_ela", "e7aqkn3j", "fixed edge fallback"),
        ("ours_26000_phasej_external_micro_autoedge_ela", "0plpo822", "micro auto-edge no-op diagnostic"),
        ("ours_26000_phasej_external_autoedge_ela", "m651uff6", "auto-edge with LPIPS calibration"),
        ("ours_26000_phasej_external_fast_autoedge_ela", "d7gckkmu", "auto-edge without LPIPS calibration"),
    ]
    for method_name, wandb_run, note in f82_methods:
        method_metric = metric(f82_results, method_name)
        report = read_ela_report(f82_root, "test", method_name)
        d = delta(method_metric, f82_base)
        rows.append(
            {
                "scene": "courtyard",
                "protocol": "ETH3D courtyard F82 degraded checkpoint",
                "base_method": "ours_26000",
                "method": method_name,
                "comparison": "vs F82 base",
                "wandb_run": wandb_run,
                "alpha": report.get("alpha"),
                "edge_q": (report.get("policy") or {}).get("edge_gate_quantile"),
                "k": (report.get("policy") or {}).get("k"),
                "depth_rel_tol": (report.get("policy") or {}).get("depth_rel_tol"),
                "covered_fraction": report.get("mean_covered_fraction"),
                "method_PSNR": None if method_metric is None else method_metric["PSNR"],
                "method_SSIM": None if method_metric is None else method_metric["SSIM"],
                "method_LPIPS": None if method_metric is None else method_metric["LPIPS"],
                "base_PSNR": None if f82_base is None else f82_base["PSNR"],
                "base_SSIM": None if f82_base is None else f82_base["SSIM"],
                "base_LPIPS": None if f82_base is None else f82_base["LPIPS"],
                **d,
                "strict_rgb_win": strict_rgb_win(d),
                "note": note,
            }
        )

    return {
        "method": "Phase-J external courtyard validation",
        "test_usage": "held-out metrics are reported after train-only calibration; they are not used for method selection",
        "rows": rows,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase-J External Courtyard Validation",
        "",
        "This report checks whether the Phase-J train-only ELA policy transfers outside the selected Mip-NeRF360 full9 protocol. It contains both a positive clean-checkpoint validation and a degraded-checkpoint diagnostic.",
        "",
        "## Results",
        "",
        "| protocol | method | base | dPSNR | dSSIM | dLPIPS | alpha | edge q | strict RGB | W&B |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {protocol} | `{method}` | `{base}` | {dpsnr} | {dssim} | {dlpips} | {alpha} | {edge_q} | {strict} | `{wandb}` |".format(
                protocol=row["protocol"],
                method=row["method"],
                base=row["base_method"],
                dpsnr=fmt(row["dPSNR"]),
                dssim=fmt(row["dSSIM"]),
                dlpips=fmt(row["dLPIPS"]),
                alpha=fmt(row["alpha"], 3),
                edge_q=fmt(row["edge_q"], 3),
                strict="yes" if row["strict_rgb_win"] else "no",
                wandb=row["wandb_run"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- On the fair clean9000 courtyard checkpoint, Phase-J micro auto-edge improves all three RGB metrics over the clean baseline: `+0.244770` PSNR, `+0.013113` SSIM, `-0.015389` LPIPS.",
            "- Against the older ELA7 courtyard portfolio, the same method improves PSNR and SSIM but not LPIPS, so it should be reported as a mixed replacement rather than a strict dominance result.",
            "- On the F82 degraded checkpoint, full auto-edge produces only a very small strict RGB improvement. Fixed and micro policies correctly no-op. This is useful negative evidence: the policy is conservative, but severe checkpoint degradation is not solved by render-time residual transfer alone.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean9000_root", default="outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000")
    parser.add_argument(
        "--f82_root",
        default="outputs/carnet/meshsplatopt/final_stageF82_fixed_adaptive_policy_multiscene/courtyard/adaptive_global_policy_v5_seed0/recovery_model",
    )
    parser.add_argument("--ela7_root", default="outputs/carnet/meshsplatopt/stageELA7_portfolio/courtyard/evidence_pareto_portfolio")
    parser.add_argument("--output_dir", default="outputs/carnet/meshsplatopt/ecsr_phase_j_external_validation")
    parser.add_argument("--doc_out", default="docs/car_model/5-8-ECSR-PhaseJ-ExternalCourtyardValidation.md")
    args = parser.parse_args()
    report = collect(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phasej_external_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(report["rows"], out_dir / "phasej_external_validation.csv")
    write_md(report, out_dir / "phasej_external_validation.md")
    write_md(report, Path(args.doc_out))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

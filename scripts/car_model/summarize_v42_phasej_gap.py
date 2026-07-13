#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCENES = ("garden", "room", "counter", "bonsai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v42_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware"),
    )
    parser.add_argument(
        "--phasej_csv",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv"),
    )
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/car_model/6-23-v42-PhaseJ-Gap-Diagnostic.md"),
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/v42_phasej_gap_diagnostic.json"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def one_metric(path: Path) -> dict[str, float]:
    payload = read_json(path)
    keys = list(payload.keys())
    if len(keys) != 1:
        raise ValueError(f"expected one method row in {path}, got {keys}")
    row = payload[keys[0]]
    return {k: float(row[k]) for k in ("PSNR", "SSIM", "LPIPS")}


def read_phasej(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scene = row["scene"]
            rows[scene] = row
    return rows


def ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def main() -> None:
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    phasej = read_phasej(args.phasej_csv)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        noop = one_metric(args.v42_root / f"{scene}_evidence_noop_compact_baseline" / "results.json")
        v42 = one_metric(args.v42_root / f"{scene}_v42_ssimgate_confidence_weighted_region_texture_adapter" / "results.json")
        pj = phasej[scene]
        d_v42 = {
            "dPSNR": v42["PSNR"] - noop["PSNR"],
            "dSSIM": v42["SSIM"] - noop["SSIM"],
            "dLPIPS": v42["LPIPS"] - noop["LPIPS"],
        }
        d_phasej = {
            "dPSNR": float(pj["dPSNR"]),
            "dSSIM": float(pj["dSSIM"]),
            "dLPIPS": float(pj["dLPIPS"]),
        }
        rows.append(
            {
                "scene": scene,
                "v42": v42,
                "v42_noop": noop,
                "phasej": {
                    "PSNR": float(pj["PSNR"]),
                    "SSIM": float(pj["SSIM"]),
                    "LPIPS": float(pj["LPIPS"]),
                    "clean_PSNR": float(pj["clean_PSNR"]),
                    "clean_SSIM": float(pj["clean_SSIM"]),
                    "clean_LPIPS": float(pj["clean_LPIPS"]),
                    "model_path": pj["model_path"],
                    "method_name": pj["method_name"],
                },
                "d_v42_vs_noop": d_v42,
                "d_phasej_vs_clean": d_phasej,
                "effect_size_ratio_phasej_over_v42": {
                    "PSNR": ratio(d_phasej["dPSNR"], d_v42["dPSNR"]),
                    "SSIM": ratio(d_phasej["dSSIM"], d_v42["dSSIM"]),
                    "LPIPS": ratio(abs(d_phasej["dLPIPS"]), abs(d_v42["dLPIPS"])),
                },
            }
        )

    mean_v42 = {
        "dPSNR": sum(r["d_v42_vs_noop"]["dPSNR"] for r in rows) / len(rows),
        "dSSIM": sum(r["d_v42_vs_noop"]["dSSIM"] for r in rows) / len(rows),
        "dLPIPS": sum(r["d_v42_vs_noop"]["dLPIPS"] for r in rows) / len(rows),
    }
    mean_phasej = {
        "dPSNR": sum(r["d_phasej_vs_clean"]["dPSNR"] for r in rows) / len(rows),
        "dSSIM": sum(r["d_phasej_vs_clean"]["dSSIM"] for r in rows) / len(rows),
        "dLPIPS": sum(r["d_phasej_vs_clean"]["dLPIPS"] for r in rows) / len(rows),
    }
    summary = {
        "mean_v42_vs_noop": mean_v42,
        "mean_phasej_vs_clean": mean_phasej,
        "mean_effect_ratio_phasej_over_v42": {
            "PSNR": ratio(mean_phasej["dPSNR"], mean_v42["dPSNR"]),
            "SSIM": ratio(mean_phasej["dSSIM"], mean_v42["dSSIM"]),
            "LPIPS": ratio(abs(mean_phasej["dLPIPS"]), abs(mean_v42["dLPIPS"])),
        },
    }

    lines: list[str] = []
    lines.append("# v42 vs Phase-J Gap Diagnostic")
    lines.append("")
    lines.append("Date: 2026-06-23")
    lines.append("")
    lines.append("Status: diagnostic only. Phase-J and v42 use different comparison baselines, so the ratios below quantify effect-size gap, not a strict method-vs-method fairness claim.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- v42 root: `{args.v42_root}`")
    lines.append(f"- Phase-J closure CSV: `{args.phasej_csv}`")
    lines.append("- v42 delta: v42-SSIMGate minus same-evidence no-op compact baseline.")
    lines.append("- Phase-J delta: Phase-J minus selected clean MeshSplatting baseline.")
    lines.append("")
    lines.append("## Four-Scene Gap Table")
    lines.append("")
    lines.append("| scene | v42 dPSNR | Phase-J dPSNR | PSNR ratio | v42 dSSIM | Phase-J dSSIM | SSIM ratio | v42 dLPIPS | Phase-J dLPIPS | LPIPS ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        dv = row["d_v42_vs_noop"]
        dp = row["d_phasej_vs_clean"]
        rr = row["effect_size_ratio_phasej_over_v42"]
        lines.append(
            f"| {row['scene']} | {dv['dPSNR']:+.6f} | {dp['dPSNR']:+.6f} | {fmt(rr['PSNR'], 1)}x | "
            f"{dv['dSSIM']:+.8f} | {dp['dSSIM']:+.6f} | {fmt(rr['SSIM'], 1)}x | "
            f"{dv['dLPIPS']:+.8f} | {dp['dLPIPS']:+.6f} | {fmt(rr['LPIPS'], 1)}x |"
        )
    rr = summary["mean_effect_ratio_phasej_over_v42"]
    lines.append(
        f"| **mean** | {mean_v42['dPSNR']:+.6f} | {mean_phasej['dPSNR']:+.6f} | {fmt(rr['PSNR'], 1)}x | "
        f"{mean_v42['dSSIM']:+.8f} | {mean_phasej['dSSIM']:+.6f} | {fmt(rr['SSIM'], 1)}x | "
        f"{mean_v42['dLPIPS']:+.8f} | {mean_phasej['dLPIPS']:+.6f} | {fmt(rr['LPIPS'], 1)}x |"
    )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- v42 is a real representation-level step, but its four-scene mean effect is still about three orders of magnitude smaller than Phase-J in PSNR and LPIPS effect size.")
    lines.append("- This gap explains why v42's RGB crops remain visually subtle even when error-reduction maps show local positive action.")
    lines.append("- The next representation-level method must increase residual support and expressivity while retaining the train-only SSIM/tail safety gate.")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("Do not present the ratio table as a fair head-to-head benchmark. It compares two different deltas: Phase-J over selected clean MeshSplatting, and v42 over same-evidence no-op compact. Its purpose is to quantify the remaining representation-internalization gap.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

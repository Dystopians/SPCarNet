#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {"lpips", "absrel", "depth_mae", "normal"}
HIGHER_IS_BETTER = {"psnr", "ssim"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb_metrics(model_path: Path, iteration: int) -> dict[str, float]:
    path = model_path / "results.json"
    if not path.is_file():
        return {}
    payload = _load_json(path)
    row = payload.get(f"ours_{iteration}", {})
    return {
        "psnr": float(row.get("PSNR", float("nan"))),
        "ssim": float(row.get("SSIM", float("nan"))),
        "lpips": float(row.get("LPIPS", float("nan"))),
    }


def _geometry_metrics(model_path: Path, iteration: int) -> dict[str, float]:
    candidates = [
        model_path / "geometry_eval_colmap" / f"iter_{iteration}_max500.json",
        model_path / "geometry_eval_colmap" / f"iter_{iteration}.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return {}
    payload = _load_json(path)
    return {
        "absrel": float(payload.get("depth", {}).get("abs_rel", float("nan"))),
        "depth_mae": float(payload.get("depth", {}).get("mae", float("nan"))),
        "normal": float(payload.get("normal", {}).get("mean_ang_deg", float("nan"))),
    }


def _metrics(model_path: Path, iteration: int) -> dict[str, float]:
    out = {}
    out.update(_rgb_metrics(model_path, iteration))
    out.update(_geometry_metrics(model_path, iteration))
    return out


def _pass_metric(candidate: float, baseline: float, key: str) -> bool:
    if key in HIGHER_IS_BETTER:
        return candidate >= baseline
    if key in LOWER_IS_BETTER:
        return candidate <= baseline
    raise KeyError(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect SCE8 multiscene fixed-policy tables.")
    parser.add_argument("--entry", action="append", nargs=5, metavar=("SCENE", "BASELINE_MODEL", "BASELINE_ITER", "CANDIDATE_MODEL", "CANDIDATE_ITER"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = []
    for scene, baseline_model, baseline_iter, candidate_model, candidate_iter in args.entry or []:
        b_iter = int(baseline_iter)
        c_iter = int(candidate_iter)
        baseline = _metrics(Path(baseline_model), b_iter)
        candidate = _metrics(Path(candidate_model), c_iter)
        row = {"scene": scene, "baseline_model": baseline_model, "baseline_iter": b_iter, "candidate_model": candidate_model, "candidate_iter": c_iter}
        all_pass = True
        for key in ["psnr", "ssim", "lpips", "absrel", "depth_mae", "normal"]:
            b = baseline.get(key, float("nan"))
            c = candidate.get(key, float("nan"))
            row[f"baseline_{key}"] = b
            row[f"candidate_{key}"] = c
            row[f"delta_{key}"] = c - b
            passed = _pass_metric(c, b, key) if c == c and b == b else False
            row[f"pass_{key}"] = int(passed)
            all_pass = all_pass and passed
        row["all_metric_pass"] = int(all_pass)
        rows.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "stageSCE8_multiscene_policy_table.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out / "stageSCE8_multiscene_policy_table.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ["# Stage SCE8 Multiscene Policy Table", "", f"- rows: `{len(rows)}`", f"- all-pass rows: `{sum(int(r['all_metric_pass']) for r in rows)}`", ""]
    report.append("| scene | all-pass | dPSNR | dSSIM | dLPIPS | dAbsRel | dMAE | dNormal |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        report.append(
            f"| {r['scene']} | {r['all_metric_pass']} | {r['delta_psnr']:.6f} | {r['delta_ssim']:.6f} | {r['delta_lpips']:.6f} | {r['delta_absrel']:.6f} | {r['delta_depth_mae']:.6f} | {r['delta_normal']:.6f} |"
        )
    (out / "stageSCE8_multiscene_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"rows": len(rows), "all_pass": sum(int(r["all_metric_pass"]) for r in rows), "output_dir": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


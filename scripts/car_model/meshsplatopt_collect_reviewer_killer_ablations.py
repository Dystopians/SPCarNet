#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _metrics(model: Path, iteration: int) -> dict[str, float]:
    out: dict[str, float] = {}
    res = _load(str(model / "results.json"))
    rgb = res.get(f"ours_{iteration}", {})
    out["psnr"] = float(rgb.get("PSNR", float("nan")))
    out["ssim"] = float(rgb.get("SSIM", float("nan")))
    out["lpips"] = float(rgb.get("LPIPS", float("nan")))
    geom = _load(str(model / "geometry_eval_colmap" / f"iter_{iteration}_max500.json"))
    if not geom:
        geom = _load(str(model / "geometry_eval_colmap" / f"iter_{iteration}.json"))
    out["absrel"] = float(geom.get("depth", {}).get("abs_rel", float("nan")))
    out["depth_mae"] = float(geom.get("depth", {}).get("mae", float("nan")))
    out["normal"] = float(geom.get("normal", {}).get("mean_ang_deg", float("nan")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect reviewer-killer ablation rows from existing artifacts.")
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--baseline_iter", type=int, required=True)
    parser.add_argument("--row", action="append", nargs=4, metavar=("AXIS", "MODEL", "ITER", "VALIDITY"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    baseline = _metrics(Path(args.baseline_model), int(args.baseline_iter))
    rows = []
    for axis, model, iteration, validity in args.row or []:
        iteration_i = int(iteration)
        m = _metrics(Path(model), iteration_i)
        row = {"axis": axis, "model": model, "iteration": iteration_i, "validity": validity}
        all_pass = True
        for key in ["psnr", "ssim", "lpips", "absrel", "depth_mae", "normal"]:
            b = baseline.get(key, float("nan"))
            c = m.get(key, float("nan"))
            row[f"baseline_{key}"] = b
            row[f"candidate_{key}"] = c
            row[f"delta_{key}"] = c - b
            if key in {"psnr", "ssim"}:
                passed = c >= b
            else:
                passed = c <= b
            row[f"pass_{key}"] = int(passed)
            all_pass = all_pass and bool(passed)
        row["parent_pareto_pass"] = int(all_pass and validity == "valid")
        rows.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "reviewer_killer_ablations.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out / "reviewer_killer_ablations.json").write_text(json.dumps({"baseline": baseline, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Reviewer-Killer Ablation Table", "", "| axis | valid | pareto | dPSNR | dAbsRel | dMAE | dNormal |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['axis']} | {r['validity']} | {r['parent_pareto_pass']} | {r['delta_psnr']:.6f} | {r['delta_absrel']:.6f} | {r['delta_depth_mae']:.6f} | {r['delta_normal']:.6f} |"
        )
    (out / "reviewer_killer_ablations_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "pareto_pass": sum(int(r["parent_pareto_pass"]) for r in rows), "output_dir": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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


def _metrics(model_path: Path, iteration: int) -> dict[str, float]:
    out: dict[str, float] = {}
    results_path = model_path / "results.json"
    if results_path.is_file():
        row = _load_json(results_path).get(f"ours_{iteration}", {})
        out.update(
            {
                "psnr": float(row.get("PSNR", float("nan"))),
                "ssim": float(row.get("SSIM", float("nan"))),
                "lpips": float(row.get("LPIPS", float("nan"))),
            }
        )
    for geom_path in (
        model_path / "geometry_eval_colmap" / f"iter_{iteration}_max500.json",
        model_path / "geometry_eval_colmap" / f"iter_{iteration}.json",
    ):
        if geom_path.is_file():
            payload = _load_json(geom_path)
            out.update(
                {
                    "absrel": float(payload.get("depth", {}).get("abs_rel", float("nan"))),
                    "depth_mae": float(payload.get("depth", {}).get("mae", float("nan"))),
                    "normal": float(payload.get("normal", {}).get("mean_ang_deg", float("nan"))),
                }
            )
            break
    return out


def _decision(path: str) -> dict[str, Any]:
    if not path:
        return {"action": "accept_candidate", "execute_recovery": True, "reason": "no_policy_decision_json"}
    payload = _load_json(Path(path))
    return dict(payload.get("decision", payload))


def _pass_metric(candidate: float, baseline: float, key: str) -> bool:
    if key in HIGHER_IS_BETTER:
        return candidate >= baseline
    if key in LOWER_IS_BETTER:
        return candidate <= baseline
    raise KeyError(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect guarded SCE policy results with explicit no-op rows.")
    parser.add_argument(
        "--entry",
        action="append",
        nargs=6,
        metavar=("SCENE", "BASELINE_MODEL", "BASELINE_ITER", "CANDIDATE_MODEL", "CANDIDATE_ITER", "POLICY_DECISION_JSON"),
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for scene, baseline_model, baseline_iter, candidate_model, candidate_iter, decision_json in args.entry or []:
        b_iter = int(baseline_iter)
        c_iter = int(candidate_iter)
        decision = _decision(decision_json)
        abstained = not bool(decision.get("execute_recovery", True)) or decision.get("action") == "accept_parent_noop"
        effective_model = Path(baseline_model) if abstained else Path(candidate_model)
        effective_iter = b_iter if abstained else c_iter
        baseline = _metrics(Path(baseline_model), b_iter)
        effective = _metrics(effective_model, effective_iter)
        row: dict[str, Any] = {
            "scene": scene,
            "abstained_to_parent": int(abstained),
            "policy_action": str(decision.get("action", "")),
            "policy_reason": str(decision.get("reason", "")),
            "baseline_model": baseline_model,
            "baseline_iter": b_iter,
            "candidate_model": candidate_model,
            "candidate_iter": c_iter,
            "effective_model": str(effective_model),
            "effective_iter": effective_iter,
        }
        all_pass = True
        strict_win = False
        for key in ["psnr", "ssim", "lpips", "absrel", "depth_mae", "normal"]:
            b = baseline.get(key, float("nan"))
            e = effective.get(key, float("nan"))
            delta = e - b
            row[f"baseline_{key}"] = b
            row[f"effective_{key}"] = e
            row[f"delta_{key}"] = delta
            passed = _pass_metric(e, b, key) if e == e and b == b else False
            row[f"pass_{key}"] = int(passed)
            all_pass = all_pass and passed
            if key in HIGHER_IS_BETTER:
                strict_win = strict_win or delta > 0.0
            else:
                strict_win = strict_win or delta < 0.0
        row["all_metric_nonregression"] = int(all_pass)
        row["has_strict_improvement"] = int(strict_win)
        rows.append(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "stageSCE20_guarded_policy_table.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out / "stageSCE20_guarded_policy_table.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Stage SCE20 Guarded Policy Table",
        "",
        f"- rows: `{len(rows)}`",
        f"- non-regression rows: `{sum(int(r['all_metric_nonregression']) for r in rows)}`",
        f"- strict-improvement rows: `{sum(int(r['has_strict_improvement']) for r in rows)}`",
        "",
        "| scene | abstain | non-regress | strict-improve | action | reason | dPSNR | dSSIM | dLPIPS | dAbsRel | dMAE | dNormal |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(
            f"| {r['scene']} | {r['abstained_to_parent']} | {r['all_metric_nonregression']} | {r['has_strict_improvement']} | "
            f"{r['policy_action']} | {r['policy_reason']} | {r['delta_psnr']:.6f} | {r['delta_ssim']:.6f} | "
            f"{r['delta_lpips']:.6f} | {r['delta_absrel']:.6f} | {r['delta_depth_mae']:.6f} | {r['delta_normal']:.6f} |"
        )
    (out / "stageSCE20_guarded_policy_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"rows": len(rows), "output_dir": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

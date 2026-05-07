#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


PAPER = {
    "bicycle": {"PSNR": 23.04, "LPIPS": 0.348, "SSIM": 0.641},
    "flowers": {"PSNR": 19.34, "LPIPS": 0.417, "SSIM": 0.480},
    "garden": {"PSNR": 24.70, "LPIPS": 0.217, "SSIM": 0.762},
    "stump": {"PSNR": 24.78, "LPIPS": 0.316, "SSIM": 0.678},
    "treehill": {"PSNR": 20.53, "LPIPS": 0.428, "SSIM": 0.540},
    "room": {"PSNR": 28.52, "LPIPS": 0.271, "SSIM": 0.873},
    "counter": {"PSNR": 26.51, "LPIPS": 0.279, "SSIM": 0.846},
    "kitchen": {"PSNR": 27.42, "LPIPS": 0.227, "SSIM": 0.858},
    "bonsai": {"PSNR": 28.19, "LPIPS": 0.294, "SSIM": 0.876},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _scene_method(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not results:
        raise RuntimeError("empty results")
    if "ours_30000" in results:
        return "ours_30000", dict(results["ours_30000"])
    key = sorted(results.keys())[-1]
    return key, dict(results[key])


def collect(root: Path, scenes: list[str]) -> list[dict[str, Any]]:
    rows = []
    for scene in scenes:
        result_path = root / scene / "results.json"
        if not result_path.exists():
            rows.append({"scene": scene, "status": "missing_results"})
            continue
        method, metrics = _scene_method(_read_json(result_path))
        paper = PAPER.get(scene, {})
        row = {
            "scene": scene,
            "status": "ok",
            "method": method,
            "psnr": _finite(metrics.get("PSNR")),
            "ssim": _finite(metrics.get("SSIM")),
            "lpips": _finite(metrics.get("LPIPS")),
            "paper_psnr": _finite(paper.get("PSNR")),
            "paper_ssim": _finite(paper.get("SSIM")),
            "paper_lpips": _finite(paper.get("LPIPS")),
        }
        row.update(
            {
                "d_psnr_vs_paper": row["psnr"] - row["paper_psnr"],
                "d_ssim_vs_paper": row["ssim"] - row["paper_ssim"],
                "d_lpips_vs_paper": row["lpips"] - row["paper_lpips"],
            }
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_finite(row.get(key)) for row in rows if row.get("status") == "ok"]
    vals = [x for x in vals if math.isfinite(x)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def maybe_wandb(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={"root": args.root, "scenes": args.scenes},
    )
    payload = {f"paper_m360/{key}": value for key, value in summary.items() if isinstance(value, (int, float))}
    for row in rows:
        if row.get("status") != "ok":
            continue
        scene = str(row["scene"])
        for key in ("psnr", "ssim", "lpips", "d_psnr_vs_paper", "d_ssim_vs_paper", "d_lpips_vs_paper"):
            payload[f"paper_m360/{scene}/{key}"] = _finite(row.get(key))
    run.log(payload)
    run.summary.update(payload)
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect same-protocol Mip-NeRF360 reproduction metrics.")
    parser.add_argument("--root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--scenes", default="bicycle,flowers,garden,stump,treehill,room,counter,kitchen,bonsai")
    parser.add_argument("--out-csv", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper.csv")
    parser.add_argument("--out-json", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k/repro_metrics_vs_paper.json")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default="paper_m360_official_clean30k")
    parser.add_argument("--wandb_name", default="paper_m360_clean30k_metric_collect")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()]
    rows = collect(Path(args.root), scenes)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "completed_scenes": len(ok_rows),
        "requested_scenes": len(rows),
        "mean_psnr": _mean(rows, "psnr"),
        "mean_ssim": _mean(rows, "ssim"),
        "mean_lpips": _mean(rows, "lpips"),
        "mean_d_psnr_vs_paper": _mean(rows, "d_psnr_vs_paper"),
        "mean_d_ssim_vs_paper": _mean(rows, "d_ssim_vs_paper"),
        "mean_d_lpips_vs_paper": _mean(rows, "d_lpips_vs_paper"),
    }
    payload = {"summary": summary, "rows": rows, "paper_reference": PAPER}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(Path(args.out_csv), rows)
    maybe_wandb(args, rows, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _metrics(model: Path, iteration: int) -> dict[str, float | int]:
    results = _load_json(model / "results.json").get(f"ours_{iteration}", {})
    geom = _load_json(model / "geometry_eval_colmap" / f"iter_{iteration}_max500.json")
    if not geom:
        geom = _load_json(model / "geometry_eval_colmap" / f"iter_{iteration}.json")
    topo_path = model / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    return {
        "psnr": float(results.get("PSNR", float("nan"))),
        "ssim": float(results.get("SSIM", float("nan"))),
        "lpips": float(results.get("LPIPS", float("nan"))),
        "absrel": float(geom.get("depth", {}).get("abs_rel", float("nan"))),
        "depth_mae": float(geom.get("depth", {}).get("mae", float("nan"))),
        "normal": float(geom.get("normal", {}).get("mean_ang_deg", float("nan"))),
        "checkpoint_exists": int(topo_path.is_file()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Stage SCE10 ablation package rows.")
    parser.add_argument("--row", action="append", nargs=5, metavar=("LABEL", "MODEL", "ITER", "WANDB", "DECISION"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    rows = []
    for label, model, iteration, wandb, decision in args.row or []:
        row = {"label": label, "model": model, "iteration": int(iteration), "wandb": wandb, "decision": decision}
        row.update(_metrics(Path(model), int(iteration)))
        rows.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stageSCE10_ablation_package.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with (out / "stageSCE10_ablation_package.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print({"rows": len(rows), "output_dir": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


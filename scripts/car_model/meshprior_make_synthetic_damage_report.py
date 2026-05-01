"""Generate a markdown report from MeshPrior synthetic damage metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args: argparse.Namespace) -> Path:
    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    rows = metrics.get("inference_time_metrics", [])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Synthetic Damage Report\n\n")
        f.write("| Damage | Floater Precision | Floater Recall | Protect Recall | Hole Boundary | Tri Delta |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| {damage} | {prec:.3f} | {rec:.3f} | {prot:.3f} | {hole:.3f} | {delta} |\n".format(
                    damage=r["damage_type"],
                    prec=float(r["floater_prune_precision"]),
                    rec=float(r["floater_prune_recall"]),
                    prot=float(r["valid_surface_protect_recall"]),
                    hole=float(r["hole_boundary_score"]),
                    delta=int(r["triangle_count_delta"]),
                )
            )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make MeshPrior synthetic damage report.")
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument("--output", default="docs/car_model/reports/meshprior_synthetic_damage_report.md")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

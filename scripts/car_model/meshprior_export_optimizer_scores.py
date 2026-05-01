"""Export MeshPrior triangle scores in optimizer-facing formats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.optimizer_adapter import (
    export_generic_meshprior_score_npz,
    export_prism_score_json,
    load_triangle_scores,
    normalize_scores_per_region,
    prism_present,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    scores = load_triangle_scores(args.triangle_scores)
    norm = normalize_scores_per_region(scores)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generic_path = None
    prism_path = None
    if args.format in ("generic_npz", "both"):
        generic_path = export_generic_meshprior_score_npz(norm, out_dir / "meshprior_scores.npz")
    if args.format in ("prism_json", "both"):
        prism_path = export_prism_score_json(norm, out_dir / "meshprior_prism_scores.json", alpha=args.alpha, beta=args.beta)
    summary = {
        "triangle_scores": str(args.triangle_scores),
        "output_dir": str(out_dir),
        "format": args.format,
        "rows": int(len(norm)),
        "prism_present": prism_present(REPO_ROOT),
        "generic_npz": str(generic_path) if generic_path else None,
        "prism_json": str(prism_path) if prism_path else None,
    }
    with (out_dir / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MeshPrior scores for optimizers.")
    parser.add_argument("--triangle_scores", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", choices=["generic_npz", "prism_json", "both"], default="generic_npz")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.25)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

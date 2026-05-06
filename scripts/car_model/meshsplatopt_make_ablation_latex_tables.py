#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert reviewer-killer ablation CSV to a compact LaTeX table.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_tex", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open("r", encoding="utf-8")))
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Axis & $\Delta$PSNR & $\Delta$AbsRel & $\Delta$MAE & Pareto \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['axis'].replace('_', '-') } & {float(r['delta_psnr']):+.3f} & {float(r['delta_absrel']):+.4f} & {float(r['delta_depth_mae']):+.4f} & {int(float(r['parent_pareto_pass']))} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    out = Path(args.output_tex)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print({"rows": len(rows), "output_tex": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage SCE10 Markdown tables from collected CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.csv).open("r", encoding="utf-8")))
    lines = ["# Stage SCE10 Ablation Table", "", "| label | PSNR | SSIM | LPIPS | AbsRel | MAE | Normal | decision |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(
            f"| {row['label']} | {float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {float(row['lpips']):.6f} | {float(row['absrel']):.6f} | {float(row['depth_mae']):.6f} | {float(row['normal']):.6f} | {row['decision']} |"
        )
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"rows": len(rows), "output_md": str(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import torchvision.transforms.functional as tf


def _load_rgb(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    return tf.to_tensor(Image.open(path).convert("RGB")).to(dtype=torch.float64)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:+.8f}" if signed else f"{value:.8f}"
    return str(value)


def _method_dirs(model_path: Path, split: str, base_method: str, candidate_method: str) -> tuple[Path, Path]:
    split_dir = model_path / split
    base_dir = split_dir / base_method
    candidate_dir = split_dir / candidate_method
    for label, path in [("base", base_dir), ("candidate", candidate_dir)]:
        if not (path / "renders").is_dir() or not (path / "gt").is_dir():
            raise FileNotFoundError(f"{label} method is missing renders/gt directories: {path}")
    return base_dir, candidate_dir


def diagnose(model_path: Path, split: str, base_method: str, candidate_method: str) -> dict[str, Any]:
    base_dir, candidate_dir = _method_dirs(model_path, split, base_method, candidate_method)
    base_renders = base_dir / "renders"
    candidate_renders = candidate_dir / "renders"
    gt_dir = candidate_dir / "gt"
    names = [
        path.name
        for path in sorted(candidate_renders.iterdir())
        if path.is_file() and (base_renders / path.name).is_file() and (gt_dir / path.name).is_file()
    ]
    rows: list[dict[str, Any]] = []
    for name in names:
        gt = _load_rgb(gt_dir / name)
        base = _load_rgb(base_renders / name)
        candidate = _load_rgb(candidate_renders / name)
        if tuple(gt.shape) != tuple(base.shape) or tuple(gt.shape) != tuple(candidate.shape):
            raise RuntimeError(f"shape mismatch for {name}: gt={tuple(gt.shape)} base={tuple(base.shape)} candidate={tuple(candidate.shape)}")
        base_error = base - gt
        candidate_error = candidate - gt
        delta = candidate - base
        base_mse = float(base_error.square().mean().item())
        candidate_mse = float(candidate_error.square().mean().item())
        cross_term = float((2.0 * base_error * delta).mean().item())
        delta_energy = float(delta.square().mean().item())
        delta_mse = candidate_mse - base_mse
        rows.append(
            {
                "image": name,
                "base_mse": base_mse,
                "candidate_mse": candidate_mse,
                "delta_mse": delta_mse,
                "cross_term_2ed": cross_term,
                "delta_energy_d2": delta_energy,
                "mean_abs_delta": float(delta.abs().mean().item()),
                "candidate_improves_mse": delta_mse <= 0.0,
            }
        )
    improves = [row for row in rows if row["candidate_improves_mse"]]
    worsens = [row for row in rows if not row["candidate_improves_mse"]]
    summary = {
        "schema_version": 1,
        "model_path": str(model_path),
        "split": split,
        "base_method": base_method,
        "candidate_method": candidate_method,
        "view_count": len(rows),
        "mse_improved_views": len(improves),
        "mse_worse_views": len(worsens),
        "mean_base_mse": _mean([row["base_mse"] for row in rows]),
        "mean_candidate_mse": _mean([row["candidate_mse"] for row in rows]),
        "mean_delta_mse": _mean([row["delta_mse"] for row in rows]),
        "mean_cross_term_2ed": _mean([row["cross_term_2ed"] for row in rows]),
        "mean_delta_energy_d2": _mean([row["delta_energy_d2"] for row in rows]),
        "mean_abs_delta": _mean([row["mean_abs_delta"] for row in rows]),
        "worst_delta_mse_views": sorted(rows, key=lambda row: row["delta_mse"], reverse=True)[:5],
        "best_delta_mse_views": sorted(rows, key=lambda row: row["delta_mse"])[:5],
        "rows": rows,
        "interpretation": (
            "delta_mse = 2 * (base - gt) * (candidate - base) + (candidate - base)^2. "
            "Positive mean_delta_mse means the candidate render increases MSE relative to the base render."
        ),
    }
    return summary


def write_outputs(summary: dict[str, Any], out_dir: Path, prefix: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}.json"
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = summary["rows"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Render Delta MSE Diagnostic",
        "",
        f"- model_path: `{summary['model_path']}`",
        f"- split: `{summary['split']}`",
        f"- base_method: `{summary['base_method']}`",
        f"- candidate_method: `{summary['candidate_method']}`",
        f"- view_count: `{summary['view_count']}`",
        f"- mse_improved_views: `{summary['mse_improved_views']}`",
        f"- mse_worse_views: `{summary['mse_worse_views']}`",
        f"- mean_delta_mse: `{_fmt(summary['mean_delta_mse'], True)}`",
        f"- mean_cross_term_2ed: `{_fmt(summary['mean_cross_term_2ed'], True)}`",
        f"- mean_delta_energy_d2: `{_fmt(summary['mean_delta_energy_d2'])}`",
        f"- mean_abs_delta: `{_fmt(summary['mean_abs_delta'])}`",
        "",
        "## Worst MSE Views",
        "",
        "| image | delta MSE | 2ed | d2 | mean abs delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["worst_delta_mse_views"]:
        lines.append(
            f"| {row['image']} | {_fmt(row['delta_mse'], True)} | {_fmt(row['cross_term_2ed'], True)} | {_fmt(row['delta_energy_d2'])} | {_fmt(row['mean_abs_delta'])} |"
        )
    lines.extend(["", "## Best MSE Views", "", "| image | delta MSE | 2ed | d2 | mean abs delta |", "|---|---:|---:|---:|---:|"])
    for row in summary["best_delta_mse_views"]:
        lines.append(
            f"| {row['image']} | {_fmt(row['delta_mse'], True)} | {_fmt(row['cross_term_2ed'], True)} | {_fmt(row['delta_energy_d2'])} | {_fmt(row['mean_abs_delta'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MSE direction of a candidate render relative to a base render.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--base_method", required=True)
    parser.add_argument("--candidate_method", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="render_delta_mse_diagnostic")
    args = parser.parse_args()
    outputs = write_outputs(
        diagnose(Path(args.model_path), str(args.split), str(args.base_method), str(args.candidate_method)),
        Path(args.out_dir),
        str(args.prefix),
    )
    print(json.dumps({"outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

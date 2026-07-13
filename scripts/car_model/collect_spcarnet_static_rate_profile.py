#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def file_bytes(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def method_metrics(row: dict[str, Any], key: str) -> dict[str, float]:
    payload = row.get(key) or {}
    return {
        "psnr": num(payload.get("PSNR")),
        "ssim": num(payload.get("SSIM")),
        "lpips": num(payload.get("LPIPS")),
    }


def iter_from_method(method: str, default: int) -> int:
    parts = str(method).split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return int(default)


def topology_from_audit(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    return {
        "pre_triangles": audit.get("pre_triangles"),
        "post_triangles": audit.get("post_triangles"),
        "pre_vertices": audit.get("pre_vertices"),
        "post_vertices": audit.get("post_vertices"),
        "removed_triangles": audit.get("removed_triangles"),
        "removed_fraction": num(audit.get("removed_fraction")),
    }


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.repo_root)
    phasej = read_json(root / args.phasej_summary)
    compact = read_json(root / args.compact_ela_summary)
    compact_rows = {str(row.get("scene")): row for row in compact.get("rows", []) if row.get("scene")}
    out: list[dict[str, Any]] = []
    for row in phasej.get("rows", []):
        scene = str(row.get("scene"))
        if scene not in args.scenes:
            continue
        compact_row = compact_rows.get(scene, {})
        clean_method = str(row.get("clean_baseline_method", "ours_26000"))
        clean_iter = iter_from_method(clean_method, args.default_clean_iteration)
        clean_ckpt = root / args.clean_root / scene / "point_cloud" / f"iteration_{clean_iter}" / "point_cloud_state_dict.pt"

        phasej_model = root / str(row.get("model", ""))
        phasej_ckpt = phasej_model / "point_cloud" / f"iteration_{args.method_iteration}" / "point_cloud_state_dict.pt"
        phasej_topology = topology_from_audit(phasej_model / "topology_audit.json")

        compact_model = root / args.compact_ela_method_root / scene / args.compact_ela_policy_tag / "compact_model"
        compact_ckpt = compact_model / "point_cloud" / f"iteration_{args.method_iteration}" / "point_cloud_state_dict.pt"

        clean_metrics = method_metrics(row, "clean")
        phasej_metrics = method_metrics(row, "method")
        compact_metrics = {
            "psnr": num(compact_row.get("method_psnr")),
            "ssim": num(compact_row.get("method_ssim")),
            "lpips": num(compact_row.get("method_lpips")),
        }
        clean_triangles = compact_row.get("baseline_triangles")
        clean_vertices = compact_row.get("baseline_vertices")
        compact_triangles = compact_row.get("method_triangles")
        compact_vertices = compact_row.get("method_vertices")
        phasej_triangles = phasej_topology.get("post_triangles")
        phasej_vertices = phasej_topology.get("post_vertices")
        clean_bytes = file_bytes(clean_ckpt)
        compact_bytes = file_bytes(compact_ckpt)
        phasej_bytes = file_bytes(phasej_ckpt)
        out.append(
            {
                "scene": scene,
                "clean_iteration": clean_iter,
                "clean_checkpoint_bytes": clean_bytes,
                "compact_ela_checkpoint_bytes": compact_bytes,
                "phasej_checkpoint_bytes": phasej_bytes,
                "compact_ela_checkpoint_byte_delta": (compact_bytes - clean_bytes) if clean_bytes is not None and compact_bytes is not None else None,
                "phasej_checkpoint_byte_delta": (phasej_bytes - clean_bytes) if clean_bytes is not None and phasej_bytes is not None else None,
                "clean_triangles": clean_triangles,
                "compact_ela_triangles": compact_triangles,
                "phasej_triangles": phasej_triangles,
                "clean_vertices": clean_vertices,
                "compact_ela_vertices": compact_vertices,
                "phasej_vertices": phasej_vertices,
                "compact_ela_triangle_reduction": num(compact_row.get("triangle_reduction")),
                "phasej_triangle_reduction": num(row.get("total_removed_fraction")),
                "compact_ela_vertex_reduction": num(compact_row.get("vertex_reduction")),
                "phasej_vertex_reduction": (1.0 - float(phasej_vertices) / float(clean_vertices))
                if phasej_vertices and clean_vertices
                else math.nan,
                "clean_psnr": clean_metrics["psnr"],
                "clean_ssim": clean_metrics["ssim"],
                "clean_lpips": clean_metrics["lpips"],
                "compact_ela_psnr": compact_metrics["psnr"],
                "compact_ela_ssim": compact_metrics["ssim"],
                "compact_ela_lpips": compact_metrics["lpips"],
                "phasej_psnr": phasej_metrics["psnr"],
                "phasej_ssim": phasej_metrics["ssim"],
                "phasej_lpips": phasej_metrics["lpips"],
                "compact_ela_dpsnr": compact_metrics["psnr"] - clean_metrics["psnr"],
                "compact_ela_dssim": compact_metrics["ssim"] - clean_metrics["ssim"],
                "compact_ela_dlpips": compact_metrics["lpips"] - clean_metrics["lpips"],
                "phasej_dpsnr": phasej_metrics["psnr"] - clean_metrics["psnr"],
                "phasej_dssim": phasej_metrics["ssim"] - clean_metrics["ssim"],
                "phasej_dlpips": phasej_metrics["lpips"] - clean_metrics["lpips"],
                "fps_measured": False,
                "peak_vram_measured": False,
            }
        )
    return out


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [num(row.get(key)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return sum(vals) / len(vals) if vals else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        out = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(out):
        return "nan"
    return f"{out:.6f}"


def write_md(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# SPCarNet Static Rate and Model-Size Profile",
        "",
        "This is a static profile collected from existing artifacts. It does not measure render FPS or peak VRAM; those fields are explicitly marked as missing and must be filled by a separate render benchmark.",
        "",
        "## Summary",
        "",
        f"- scenes: `{summary['scenes']}`",
        f"- mean Compact-ELA triangle reduction: `{fmt(summary['mean_compact_ela_triangle_reduction'])}`",
        f"- mean Phase-J triangle reduction: `{fmt(summary['mean_phasej_triangle_reduction'])}`",
        f"- mean Compact-ELA dPSNR/dSSIM/dLPIPS: `{fmt(summary['mean_compact_ela_dpsnr'])}` / `{fmt(summary['mean_compact_ela_dssim'])}` / `{fmt(summary['mean_compact_ela_dlpips'])}`",
        f"- mean Phase-J dPSNR/dSSIM/dLPIPS: `{fmt(summary['mean_phasej_dpsnr'])}` / `{fmt(summary['mean_phasej_dssim'])}` / `{fmt(summary['mean_phasej_dlpips'])}`",
        "- FPS measured: `false`",
        "- peak VRAM measured: `false`",
        "",
        "## Per-Scene",
        "",
        "| scene | clean bytes | compact bytes | phasej bytes | compact tri red | phasej tri red | compact dPSNR | phasej dPSNR | compact dSSIM | phasej dSSIM | compact dLPIPS | phasej dLPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['clean_checkpoint_bytes']} | {row['compact_ela_checkpoint_bytes']} | {row['phasej_checkpoint_bytes']} | "
            f"{fmt(row['compact_ela_triangle_reduction'])} | {fmt(row['phasej_triangle_reduction'])} | "
            f"{fmt(row['compact_ela_dpsnr'])} | {fmt(row['phasej_dpsnr'])} | "
            f"{fmt(row['compact_ela_dssim'])} | {fmt(row['phasej_dssim'])} | "
            f"{fmt(row['compact_ela_dlpips'])} | {fmt(row['phasej_dlpips'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This profile supports triangle-count and checkpoint-byte discussion.",
            "- It must not be used as FPS or deployment-speed evidence.",
            "- The next required profiling step is a controlled render benchmark that records ms/view, FPS, and peak VRAM for clean, compact-only, Compact-ELA, Phase-J, and any promoted representation-baked candidate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect static SPCarNet rate/model-size profile from existing full9 artifacts.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--phasej_summary", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json")
    parser.add_argument("--compact_ela_summary", default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k_refresh_20260625_correct/compact_ela_vs_clean.json")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--compact_ela_method_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k")
    parser.add_argument("--compact_ela_policy_tag", default="sor_adaptive_geo")
    parser.add_argument("--method_iteration", type=int, default=26000)
    parser.add_argument("--default_clean_iteration", type=int, default=26000)
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--out_dir", default="outputs/carnet/spcarnet/static_rate_profile_20260625")
    args = parser.parse_args()
    args.scenes = {scene.strip() for scene in str(args.scenes).replace(" ", ",").split(",") if scene.strip()}

    rows = collect(args)
    summary = {
        "scenes": len(rows),
        "mean_compact_ela_triangle_reduction": mean(rows, "compact_ela_triangle_reduction"),
        "mean_phasej_triangle_reduction": mean(rows, "phasej_triangle_reduction"),
        "mean_compact_ela_dpsnr": mean(rows, "compact_ela_dpsnr"),
        "mean_compact_ela_dssim": mean(rows, "compact_ela_dssim"),
        "mean_compact_ela_dlpips": mean(rows, "compact_ela_dlpips"),
        "mean_phasej_dpsnr": mean(rows, "phasej_dpsnr"),
        "mean_phasej_dssim": mean(rows, "phasej_dssim"),
        "mean_phasej_dlpips": mean(rows, "phasej_dlpips"),
        "fps_measured": False,
        "peak_vram_measured": False,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "per_scene.csv", rows)
    write_md(out_dir / "summary.md", rows, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

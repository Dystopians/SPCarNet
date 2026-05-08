#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _score(metrics: dict[str, Any]) -> float:
    psnr = _num(metrics.get("PSNR"))
    ssim = _num(metrics.get("SSIM"))
    lpips = _num(metrics.get("LPIPS"))
    if not all(math.isfinite(x) for x in (psnr, ssim, lpips)):
        return -math.inf
    return psnr + 20.0 * ssim - 20.0 * lpips


def _metric(payload: dict[str, Any], key: str) -> dict[str, float]:
    item = payload.get(key, {})
    return {"PSNR": _num(item.get("PSNR")), "SSIM": _num(item.get("SSIM")), "LPIPS": _num(item.get("LPIPS"))}


def _select_clean_baseline(clean_results: dict[str, Any], iterations: list[int]) -> tuple[str, dict[str, float]]:
    candidates = []
    for iteration in iterations:
        key = f"ours_{int(iteration)}"
        metrics = _metric(clean_results, key)
        candidates.append((key, metrics, _score(metrics)))
    finite = [row for row in candidates if math.isfinite(row[2])]
    if not finite:
        return "", {"PSNR": math.nan, "SSIM": math.nan, "LPIPS": math.nan}
    key, metrics, _ = max(finite, key=lambda row: (row[2], row[1]["PSNR"]))
    return key, metrics


def _image_set(scene: str, outdoor_images: str, indoor_images: str) -> str:
    return outdoor_images if scene in OUTDOOR_SCENES else indoor_images


def _run(cmd: list[str], *, gpu: int, log_path: Path) -> None:
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def _render_and_eval(
    *,
    scene: str,
    model_path: Path,
    source_path: Path,
    method_name: str,
    iteration: int,
    images: str,
    gpu: int,
    log_path: Path,
    no_depth: bool,
    skip_failed_views: bool,
) -> dict[str, float]:
    results_path = model_path / "results.json"
    existing = _metric(_read_json(results_path), method_name)
    if all(math.isfinite(existing[k]) for k in ("PSNR", "SSIM", "LPIPS")):
        return existing

    render_cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_render_evidence_maps.py",
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "-i",
        images,
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(iteration),
        "--method_name",
        method_name,
        "--skip_train",
        "--quiet",
    ]
    if no_depth:
        render_cmd.append("--no_depth")
    if skip_failed_views:
        render_cmd.append("--skip_failed_views")
    _run(render_cmd, gpu=gpu, log_path=log_path)

    eval_cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model_path),
        "--split",
        "test",
        "--methods",
        method_name,
        "--merge_model_results",
    ]
    _run(eval_cmd, gpu=gpu, log_path=log_path)
    return _metric(_read_json(results_path), method_name)


def _topology_from_audit(model_path: Path) -> dict[str, int | float | None]:
    audit = _read_json(model_path / "topology_audit.json")
    return {
        "pre_triangles": audit.get("pre_triangles"),
        "post_triangles": audit.get("post_triangles"),
        "pre_vertices": audit.get("pre_vertices"),
        "post_vertices": audit.get("post_vertices"),
        "removed_triangles": audit.get("removed_triangles"),
        "removed_fraction": audit.get("removed_fraction"),
        "degenerate_face_count": audit.get("degenerate_face_count"),
        "invalid_index_count": audit.get("invalid_index_count"),
    }


def _best_method(summary: dict[str, Any]) -> Path:
    selected = summary.get("selected") or {}
    model = selected.get("model_path")
    if not model:
        raise RuntimeError(f"summary has no selected model: {summary}")
    path = ROOT / model
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _delta(method: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {
        "dPSNR": method["PSNR"] - base["PSNR"],
        "dSSIM": method["SSIM"] - base["SSIM"],
        "dLPIPS": method["LPIPS"] - base["LPIPS"],
    }


def _rgb_safe(delta: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        delta["dPSNR"] >= float(args.min_delta_psnr)
        and delta["dSSIM"] >= float(args.min_delta_ssim)
        and delta["dLPIPS"] <= float(args.max_delta_lpips)
    )


def _status(row: dict[str, Any], args: argparse.Namespace) -> str:
    clean = row["delta_vs_clean"]
    source = row["delta_vs_source_ela"]
    compact = float(row.get("total_removed_fraction") or 0.0) > 0.0
    finite = all(math.isfinite(_num(clean[k])) for k in ("dPSNR", "dSSIM", "dLPIPS"))
    if not finite:
        return "PENDING_OR_MISSING_EVAL"
    clean_rgb = _rgb_safe(clean, args)
    source_rgb = all(math.isfinite(_num(source[k])) for k in ("dPSNR", "dSSIM", "dLPIPS")) and _rgb_safe(source, args)
    if compact and clean_rgb and source_rgb:
        return "COMPACT_HELDOUT_RGB_SAFE_VS_CLEAN_AND_SOURCE"
    if compact and clean_rgb:
        return "COMPACT_HELDOUT_RGB_SAFE_VS_CLEAN"
    if compact:
        return "COMPACT_HELDOUT_RGB_MIXED"
    return "FAIL"


def _row(scene: str, args: argparse.Namespace) -> dict[str, Any]:
    scene_dir = ROOT / args.policy_root / scene
    summary = _read_json(scene_dir / "summary.json")
    selected_model = _best_method(summary)
    source_model = ROOT / args.source_root / scene / args.policy_tag / "compact_model"
    clean_model = ROOT / args.clean_root / scene
    clean_results = _read_json(clean_model / "results.json")
    source_results = _read_json(source_model / "results.json")
    clean_key, clean_metrics = _select_clean_baseline(clean_results, args.clean_iterations)

    images = _image_set(scene, args.outdoor_images, args.indoor_images)
    method_metrics = _render_and_eval(
        scene=scene,
        model_path=selected_model,
        source_path=ROOT / args.dataset_root / scene,
        method_name=args.method_name,
        iteration=args.iteration,
        images=images,
        gpu=args.gpu,
        log_path=scene_dir / "heldout_eval.log",
        no_depth=args.no_depth,
        skip_failed_views=args.skip_failed_views,
    )

    source_compact = _metric(source_results, args.source_compact_method)
    source_ela = _metric(source_results, args.source_ela_method)
    selected = summary.get("selected") or {}
    topology = _topology_from_audit(selected_model)
    source_audit = _read_json(source_model / "topology_audit.json")
    source_removed = _num(source_audit.get("removed_fraction"))
    additional_removed = _num((selected.get("audit") or {}).get("removed_fraction"))
    if not math.isfinite(additional_removed):
        additional_removed = _num(selected.get("additional_removed_fraction"))
    clean_triangles = int(source_audit.get("pre_triangles", 0) or 0)
    post_triangles = int((selected.get("audit") or {}).get("post_triangles", 0) or topology.get("post_triangles") or 0)
    if clean_triangles > 0 and post_triangles > 0:
        total_removed = 1.0 - float(post_triangles) / float(clean_triangles)
    else:
        total_removed = (
            1.0 - (1.0 - source_removed) * (1.0 - additional_removed)
            if math.isfinite(source_removed) and math.isfinite(additional_removed)
            else math.nan
        )

    row: dict[str, Any] = {
        "scene": scene,
        "selected_model": str(selected_model.relative_to(ROOT)),
        "method_name": args.method_name,
        "clean_baseline_method": clean_key,
        "source_ela_method": args.source_ela_method,
        "selected_ratio": selected.get("ratio"),
        "additional_removed_fraction": additional_removed,
        "source_removed_fraction": source_removed,
        "total_removed_fraction": total_removed,
        "heldout_psnr": method_metrics["PSNR"],
        "heldout_ssim": method_metrics["SSIM"],
        "heldout_lpips": method_metrics["LPIPS"],
        "clean_psnr": clean_metrics["PSNR"],
        "clean_ssim": clean_metrics["SSIM"],
        "clean_lpips": clean_metrics["LPIPS"],
        "source_compact_psnr": source_compact["PSNR"],
        "source_compact_ssim": source_compact["SSIM"],
        "source_compact_lpips": source_compact["LPIPS"],
        "source_ela_psnr": source_ela["PSNR"],
        "source_ela_ssim": source_ela["SSIM"],
        "source_ela_lpips": source_ela["LPIPS"],
        "topology": topology,
    }
    row["delta_vs_clean"] = _delta(method_metrics, clean_metrics)
    row["delta_vs_source_compact"] = _delta(method_metrics, source_compact)
    row["delta_vs_source_ela"] = _delta(method_metrics, source_ela)
    row["status"] = _status(row, args)
    (scene_dir / "heldout_eval_summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def _fmt(value: Any, digits: int = 6) -> str:
    value = _num(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_root = ROOT / args.policy_root
    out_json = out_root / "phasef_heldout_eval_summary.json"
    out_csv = out_root / "phasef_heldout_eval_summary.csv"
    out_md = out_root / "phasef_heldout_eval_summary.md"
    payload = {"rows": rows, "args": vars(args)}
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        flat_rows = []
        for row in rows:
            flat = {k: v for k, v in row.items() if k not in {"topology", "delta_vs_clean", "delta_vs_source_compact", "delta_vs_source_ela"}}
            for prefix in ("delta_vs_clean", "delta_vs_source_compact", "delta_vs_source_ela"):
                for key, value in row[prefix].items():
                    flat[f"{prefix}_{key}"] = value
            flat_rows.append(flat)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    compact_safe = sum(row["status"] == "COMPACT_HELDOUT_RGB_SAFE_VS_CLEAN_AND_SOURCE" for row in rows)
    clean_safe = sum(row["status"] in {"COMPACT_HELDOUT_RGB_SAFE_VS_CLEAN_AND_SOURCE", "COMPACT_HELDOUT_RGB_SAFE_VS_CLEAN"} for row in rows)
    mean_removed = sum(_num(row["total_removed_fraction"]) for row in rows) / len(rows) if rows else math.nan
    lines = [
        "# ECSR Phase-F Held-Out Evaluation",
        "",
        "This report renders the fixed policy-val-selected Phase-F checkpoints on the original held-out LLFF test split. It does not choose ratios, thresholds, crops, or fallbacks from held-out metrics.",
        "",
        f"- scenes: `{len(rows)}`",
        f"- compact + RGB-safe vs clean and source Compact-ELA/SOR: `{compact_safe}/{len(rows)}`",
        f"- compact + RGB-safe vs clean: `{clean_safe}/{len(rows)}`",
        f"- mean total triangle removal fraction: `{_fmt(mean_removed)}`",
        "",
        "| scene | clean | PSNR | SSIM | LPIPS | dPSNR clean | dSSIM clean | dLPIPS clean | dPSNR source ELA | dSSIM source ELA | dLPIPS source ELA | total tri red. | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        clean = row["delta_vs_clean"]
        source = row["delta_vs_source_ela"]
        lines.append(
            f"| {row['scene']} | `{row['clean_baseline_method']}` | {_fmt(row['heldout_psnr'])} | {_fmt(row['heldout_ssim'])} | {_fmt(row['heldout_lpips'])} | "
            f"{clean['dPSNR']:+.6f} | {clean['dSSIM']:+.6f} | {clean['dLPIPS']:+.6f} | "
            f"{source['dPSNR']:+.6f} | {source['dSSIM']:+.6f} | {source['dLPIPS']:+.6f} | "
            f"{100.0 * _num(row['total_removed_fraction']):.2f}% | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Phase-F should be treated as a compactness certificate unless the held-out RGB deltas also clear the source Compact-ELA/SOR comparison. If it is only safe versus clean, the visual claim remains with the archived Compact-ELA/SOR branch and Phase-F is a representation compactness extension.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_wandb(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.wandb:
        return
    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None, group=args.wandb_group, name=args.wandb_name, config=vars(args))
    if rows:
        summary = {
            "scenes": len(rows),
            "mean_total_removed_fraction": sum(_num(row["total_removed_fraction"]) for row in rows) / len(rows),
            "mean_dpsnr_clean": sum(row["delta_vs_clean"]["dPSNR"] for row in rows) / len(rows),
            "mean_dssim_clean": sum(row["delta_vs_clean"]["dSSIM"] for row in rows) / len(rows),
            "mean_dlpips_clean": sum(row["delta_vs_clean"]["dLPIPS"] for row in rows) / len(rows),
            "mean_dpsnr_source_ela": sum(row["delta_vs_source_ela"]["dPSNR"] for row in rows) / len(rows),
            "mean_dssim_source_ela": sum(row["delta_vs_source_ela"]["dSSIM"] for row in rows) / len(rows),
            "mean_dlpips_source_ela": sum(row["delta_vs_source_ela"]["dLPIPS"] for row in rows) / len(rows),
        }
        wandb.log(summary)
        run.summary.update(summary)
        for row in rows:
            prefix = f"scene/{row['scene']}"
            wandb.log(
                {
                    f"{prefix}/dpsnr_clean": row["delta_vs_clean"]["dPSNR"],
                    f"{prefix}/dssim_clean": row["delta_vs_clean"]["dSSIM"],
                    f"{prefix}/dlpips_clean": row["delta_vs_clean"]["dLPIPS"],
                    f"{prefix}/dpsnr_source_ela": row["delta_vs_source_ela"]["dPSNR"],
                    f"{prefix}/dssim_source_ela": row["delta_vs_source_ela"]["dSSIM"],
                    f"{prefix}/dlpips_source_ela": row["delta_vs_source_ela"]["dLPIPS"],
                    f"{prefix}/total_removed_fraction": row["total_removed_fraction"],
                }
            )
    run.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render and evaluate fixed Phase-F selected models on the original held-out test split.")
    parser.add_argument("--policy_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--source_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--dataset_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--clean_iterations", default="26000,30000")
    parser.add_argument("--source_compact_method", default="ours_26000")
    parser.add_argument("--source_ela_method", default="ours_26000_sor_adaptive_geo_compact_ela")
    parser.add_argument("--method_name", default="ours_26000_phasef_extra_compact_heldout")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--min_delta_psnr", type=float, default=-0.03)
    parser.add_argument("--min_delta_ssim", type=float, default=-0.0015)
    parser.add_argument("--max_delta_lpips", type=float, default=0.0020)
    parser.add_argument("--no_depth", action="store_true")
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="phase_f_heldout_eval")
    parser.add_argument("--wandb_name", default="phase_f_heldout_eval")
    args = parser.parse_args()

    args.clean_iterations = [int(item) for item in str(args.clean_iterations).replace(" ", ",").split(",") if item.strip()]
    scenes = [scene.strip() for scene in args.scenes.replace(" ", ",").split(",") if scene.strip()]
    rows = [_row(scene, args) for scene in scenes]
    _write_outputs(rows, args)
    _log_wandb(rows, args)
    print(json.dumps({"rows": len(rows), "report": str(ROOT / args.policy_root / "phasef_heldout_eval_summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

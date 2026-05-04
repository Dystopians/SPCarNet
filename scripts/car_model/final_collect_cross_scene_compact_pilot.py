#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.final_run_cross_scene_compact_pilot import scene_configs  # noqa: E402
from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402
from ss3dm_prior.meshsplatopt.evaluation_contracts import load_geometry_metrics, load_render_metrics  # noqa: E402


@dataclass(frozen=True)
class SceneStatus:
    scene: str
    role: str
    model_path: str
    iteration: int
    triangles: int | None
    vertices: int | None
    psnr: float
    ssim: float
    lpips: float
    abs_rel: float
    depth_mae: float
    normal_angle: float
    delta_clean_psnr: float
    delta_clean_ssim: float
    delta_clean_lpips: float
    delta_clean_abs_rel: float
    delta_clean_depth_mae: float
    delta_clean_normal: float
    triangle_reduction: float
    status: str
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _topology(model_path: Path, iteration: int) -> tuple[int | None, int | None]:
    try:
        import torch

        state = torch.load(checkpoint_path(model_path, iteration), map_location="cpu")
        return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])
    except Exception:
        return None, None


def _metrics(model_path: Path, iteration: int) -> dict[str, float]:
    render = load_render_metrics(model_path, iteration)
    geom = load_geometry_metrics(model_path, iteration)
    return {
        "psnr": render["psnr"],
        "ssim": render["ssim"],
        "lpips": render["lpips"],
        "abs_rel": geom["abs_rel"],
        "depth_mae": geom["depth_mae"],
        "normal_angle": geom["normal_mean_ang_deg"],
    }


def _available(values: list[float], triangles: int | None) -> bool:
    return triangles is not None and all(value == value for value in values)


def _row(scene: str, role: str, model_path: str, iteration: int, clean: SceneStatus | None = None) -> SceneStatus:
    model = ROOT / model_path
    metrics = _metrics(model, iteration)
    triangles, vertices = _topology(model, iteration)
    values = [metrics["psnr"], metrics["ssim"], metrics["lpips"], metrics["abs_rel"], metrics["depth_mae"], metrics["normal_angle"]]
    status = "AVAILABLE" if _available(values, triangles) else "MISSING_EVAL"
    if clean and clean.status == "AVAILABLE":
        delta_clean_psnr = metrics["psnr"] - clean.psnr
        delta_clean_ssim = metrics["ssim"] - clean.ssim
        delta_clean_lpips = metrics["lpips"] - clean.lpips
        delta_clean_abs_rel = metrics["abs_rel"] - clean.abs_rel
        delta_clean_depth_mae = metrics["depth_mae"] - clean.depth_mae
        delta_clean_normal = metrics["normal_angle"] - clean.normal_angle
        triangle_reduction = 1.0 - float(triangles) / float(clean.triangles) if triangles and clean.triangles else math.nan
    else:
        delta_clean_psnr = math.nan
        delta_clean_ssim = math.nan
        delta_clean_lpips = math.nan
        delta_clean_abs_rel = math.nan
        delta_clean_depth_mae = math.nan
        delta_clean_normal = math.nan
        triangle_reduction = math.nan

    gate = (
        status == "AVAILABLE"
        and clean is not None
        and clean.status == "AVAILABLE"
        and triangle_reduction >= 0.50
        and delta_clean_psnr >= -0.20
        and delta_clean_ssim >= -0.01
        and delta_clean_lpips <= 0.02
        and delta_clean_abs_rel <= 0.02
        and delta_clean_depth_mae <= 0.20
        and delta_clean_normal <= 2.0
    )
    decision = "PASS_COMPACT_VS_CLEAN_LONG" if gate else ("BASELINE" if clean is None else "PENDING_OR_FAIL")
    return SceneStatus(
        scene=scene,
        role=role,
        model_path=model_path,
        iteration=iteration,
        triangles=triangles,
        vertices=vertices,
        psnr=metrics["psnr"],
        ssim=metrics["ssim"],
        lpips=metrics["lpips"],
        abs_rel=metrics["abs_rel"],
        depth_mae=metrics["depth_mae"],
        normal_angle=metrics["normal_angle"],
        delta_clean_psnr=delta_clean_psnr,
        delta_clean_ssim=delta_clean_ssim,
        delta_clean_lpips=delta_clean_lpips,
        delta_clean_abs_rel=delta_clean_abs_rel,
        delta_clean_depth_mae=delta_clean_depth_mae,
        delta_clean_normal=delta_clean_normal,
        triangle_reduction=triangle_reduction,
        status=status,
        decision=decision,
    )


def _compact_candidates(scene: str) -> list[tuple[str, str, int]]:
    root = ROOT / "outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot" / scene
    out: list[tuple[str, str, int]] = []
    if root.is_dir():
        for model in sorted(root.glob("*/*/recovery_model")):
            role = "/".join(model.relative_to(root).parts[:2])
            out.append((role, str(model.relative_to(ROOT)), 26000))
    return out


def collect() -> list[SceneStatus]:
    rows: list[SceneStatus] = []
    for config in scene_configs():
        clean_model = ROOT / config.clean_model
        if clean_model.is_dir():
            clean = _row(config.scene, "clean_long", config.clean_model, config.clean_iteration)
            rows.append(clean)
        else:
            clean = SceneStatus(
                scene=config.scene,
                role="clean_long",
                model_path=config.clean_model,
                iteration=config.clean_iteration,
                triangles=None,
                vertices=None,
                psnr=math.nan,
                ssim=math.nan,
                lpips=math.nan,
                abs_rel=math.nan,
                depth_mae=math.nan,
                normal_angle=math.nan,
                delta_clean_psnr=math.nan,
                delta_clean_ssim=math.nan,
                delta_clean_lpips=math.nan,
                delta_clean_abs_rel=math.nan,
                delta_clean_depth_mae=math.nan,
                delta_clean_normal=math.nan,
                triangle_reduction=math.nan,
                status=config.baseline_status,
                decision="MISSING_BASELINE",
            )
            rows.append(clean)
        for role, model_path, iteration in _compact_candidates(config.scene):
            rows.append(_row(config.scene, role, model_path, iteration, clean))
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[SceneStatus]) -> None:
    lines = [
        "# Final F8 Cross-Scene Compact Pilot",
        "",
        "| scene | role | status | triangles | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.scene} | {row.role} | {row.status} | {_fmt(row.triangles)} | {_fmt(row.psnr)} | "
            f"{_fmt(row.ssim)} | {_fmt(row.lpips)} | {_fmt(row.abs_rel)} | {_fmt(row.depth_mae)} | "
            f"{_fmt(row.normal_angle)} | {_fmt(row.triangle_reduction)} | {_fmt(row.delta_clean_psnr)} | "
            f"{_fmt(row.delta_clean_ssim)} | {_fmt(row.delta_clean_lpips)} | {_fmt(row.delta_clean_abs_rel)} | "
            f"{_fmt(row.delta_clean_depth_mae)} | {row.decision} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect final F8 cross-scene compact pilot evidence.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF8_cross_scene_compact_pilot")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect()
    payload = [row.to_dict() for row in rows]
    (out_dir / "cross_scene_compact_pilot_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "cross_scene_compact_pilot_results.csv", payload)
    _write_md(out_dir / "cross_scene_compact_pilot_results.md", rows)
    pass_scenes = sorted({row.scene for row in rows if row.decision == "PASS_COMPACT_VS_CLEAN_LONG"})
    print(f"Wrote {len(rows)} rows to {out_dir}; passing compact scenes: {pass_scenes}")
    return 0 if len(pass_scenes) >= 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())

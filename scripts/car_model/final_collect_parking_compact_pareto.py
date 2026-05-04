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

from ss3dm_prior.meshsplatopt.evaluation_contracts import load_geometry_metrics, load_render_metrics  # noqa: E402
from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402


@dataclass(frozen=True)
class ParetoRow:
    selector: str
    prune_fraction: float
    row_id: str
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
    delta_clean22_psnr: float
    delta_clean22_ssim: float
    delta_clean22_lpips: float
    delta_clean22_abs_rel: float
    delta_clean22_depth_mae: float
    delta_clean22_normal: float
    delta_clean30_psnr: float
    delta_clean30_ssim: float
    delta_clean30_lpips: float
    delta_r53_psnr: float
    delta_r53_ssim: float
    delta_r53_lpips: float
    wandb: str
    decision: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BASELINES = {
    "clean22": {
        "psnr": 18.479990,
        "ssim": 0.634623,
        "lpips": 0.346913,
        "abs_rel": 0.082177,
        "depth_mae": 1.868398,
        "normal": 45.108437,
    },
    "clean30": {
        "psnr": 18.408827,
        "ssim": 0.631504,
        "lpips": 0.350967,
    },
    "r53": {
        "psnr": 18.705738,
        "ssim": 0.647807,
        "lpips": 0.338492,
    },
}


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


def _row(
    selector: str,
    prune_fraction: float,
    row_id: str,
    model_path: str,
    iteration: int,
    wandb: str,
    decision: str,
) -> ParetoRow:
    model = ROOT / model_path
    render = load_render_metrics(model, iteration)
    geom = load_geometry_metrics(model, iteration)
    triangles, vertices = _topology(model, iteration)
    psnr = render["psnr"]
    ssim = render["ssim"]
    lpips = render["lpips"]
    abs_rel = geom["abs_rel"]
    depth_mae = geom["depth_mae"]
    normal = geom["normal_mean_ang_deg"]
    values = [psnr, ssim, lpips, abs_rel, depth_mae, normal]
    status = "AVAILABLE" if all(v == v for v in values) else "MISSING_EVAL"
    return ParetoRow(
        selector=selector,
        prune_fraction=prune_fraction,
        row_id=row_id,
        model_path=model_path,
        iteration=iteration,
        triangles=triangles,
        vertices=vertices,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        abs_rel=abs_rel,
        depth_mae=depth_mae,
        normal_angle=normal,
        delta_clean22_psnr=psnr - BASELINES["clean22"]["psnr"],
        delta_clean22_ssim=ssim - BASELINES["clean22"]["ssim"],
        delta_clean22_lpips=lpips - BASELINES["clean22"]["lpips"],
        delta_clean22_abs_rel=abs_rel - BASELINES["clean22"]["abs_rel"],
        delta_clean22_depth_mae=depth_mae - BASELINES["clean22"]["depth_mae"],
        delta_clean22_normal=normal - BASELINES["clean22"]["normal"],
        delta_clean30_psnr=psnr - BASELINES["clean30"]["psnr"],
        delta_clean30_ssim=ssim - BASELINES["clean30"]["ssim"],
        delta_clean30_lpips=lpips - BASELINES["clean30"]["lpips"],
        delta_r53_psnr=psnr - BASELINES["r53"]["psnr"],
        delta_r53_ssim=ssim - BASELINES["r53"]["ssim"],
        delta_r53_lpips=lpips - BASELINES["r53"]["lpips"],
        wandb=wandb,
        decision=decision,
        status=status,
    )


def default_rows() -> list[ParetoRow]:
    rows = [
        _row(
            "area_smallest",
            0.80,
            "R48.01",
            "outputs/carnet/meshsplatopt/stageR48_01_prune80_clean_recovery_22000to26000/recovery_model",
            26000,
            "1n6jv232",
            "compact_pareto_lpips_slightly_worse",
        ),
        _row(
            "area_smallest",
            0.70,
            "R53.01",
            "outputs/carnet/meshsplatopt/stageR53_01_prune70_clean_recovery_22000to26000/recovery_model",
            26000,
            "q15qg2b8",
            "headline_quality_dominating",
        ),
        _row(
            "area_smallest",
            0.65,
            "R55.01",
            "outputs/carnet/meshsplatopt/stageR55_01_prune65_clean_recovery_22000to26000/recovery_model",
            26000,
            "ja7t57cx",
            "lpips_normal_pareto",
        ),
        _row(
            "csef_low_evidence_boundary_protected",
            0.70,
            "F7.csef70",
            "outputs/carnet/meshsplatopt/final_stageF7_parking_pareto/csef_low_evidence_boundary_protected/prune70/recovery_model",
            26000,
            "oqpkykcw",
            "csef_same_topology_quality_gain",
        ),
    ]
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_md(path: Path, rows: list[ParetoRow]) -> None:
    lines = [
        "# Final F7 Parking Compact Pareto",
        "",
        "| row | selector | prune | triangles | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | dPSNR clean22 | dSSIM clean22 | dLPIPS clean22 | dAbsRel clean22 | dDepth clean22 | dPSNR R53 | dLPIPS R53 | status | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.row_id} | {row.selector} | {row.prune_fraction:.2f} | {_fmt(row.triangles)} | "
            f"{_fmt(row.psnr)} | {_fmt(row.ssim)} | {_fmt(row.lpips)} | {_fmt(row.abs_rel)} | "
            f"{_fmt(row.depth_mae)} | {_fmt(row.normal_angle)} | {_fmt(row.delta_clean22_psnr)} | "
            f"{_fmt(row.delta_clean22_ssim)} | {_fmt(row.delta_clean22_lpips)} | {_fmt(row.delta_clean22_abs_rel)} | "
            f"{_fmt(row.delta_clean22_depth_mae)} | {_fmt(row.delta_r53_psnr)} | {_fmt(row.delta_r53_lpips)} | "
            f"{row.status} | {row.decision} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect final F7 parking compact Pareto rows.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF7_parking_pareto")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = default_rows()
    payload = [row.to_dict() for row in rows]
    (out_dir / "parking_compact_pareto_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "parking_compact_pareto_results.csv", payload)
    _write_md(out_dir / "parking_compact_pareto_results.md", rows)
    pass_rows = [
        row
        for row in rows
        if row.status == "AVAILABLE"
        and row.delta_clean22_psnr >= 0.0
        and row.delta_clean22_ssim >= 0.0
        and row.delta_clean22_lpips <= 0.0
        and row.delta_clean22_abs_rel <= 0.0
        and row.delta_clean22_depth_mae <= 0.0
        and row.triangles is not None
        and row.triangles <= 4_274_121
    ]
    print(f"Wrote {len(rows)} rows to {out_dir}; clean22-dominating compact rows: {len(pass_rows)}")
    return 0 if pass_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

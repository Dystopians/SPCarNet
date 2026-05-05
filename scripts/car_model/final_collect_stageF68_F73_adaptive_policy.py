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

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402
from ss3dm_prior.meshsplatopt.evaluation_contracts import load_geometry_metrics, load_render_metrics  # noqa: E402


BASELINES: dict[str, dict[str, float | int]] = {
    "clean22k": {
        "psnr": 18.479990,
        "ssim": 0.634623,
        "lpips": 0.346913,
        "abs_rel": 0.082177,
        "depth_mae": 1.868398,
        "normal": 45.108437,
        "triangles": 8_548_242,
    },
    "R53.01": {
        "psnr": 18.705738,
        "ssim": 0.647807,
        "lpips": 0.338492,
        "abs_rel": 0.079555,
        "depth_mae": 1.853751,
        "normal": 44.261391,
        "triangles": 2_564_473,
    },
    "F7.csef70": {
        "psnr": 18.706079,
        "ssim": 0.647764,
        "lpips": 0.338282,
        "abs_rel": 0.079404,
        "depth_mae": 1.852816,
        "normal": 44.204497,
        "triangles": 2_564_473,
    },
    "F33.csef70_sparse": {
        "psnr": 18.712330,
        "ssim": 0.647730,
        "lpips": 0.338259,
        "abs_rel": 0.079071,
        "depth_mae": 1.854015,
        "normal": 44.035708,
        "triangles": 2_564_473,
    },
}


@dataclass(frozen=True)
class AdaptiveRow:
    row_id: str
    method: str
    model_path: str
    iteration: int
    wandb: str
    triangles: int | None
    vertices: int | None
    psnr: float
    ssim: float
    lpips: float
    abs_rel: float
    depth_mae: float
    normal: float
    topology_unchanged: bool | None
    sparse_depth_enabled: bool | None
    status: str
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _topology(model_path: Path, iteration: int) -> tuple[int | None, int | None]:
    try:
        import torch

        state = torch.load(checkpoint_path(model_path, iteration), map_location="cpu")
        return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])
    except Exception:
        return None, None


def _audit(model_path: Path) -> tuple[bool | None, bool | None]:
    payload = _read_json(model_path / "topology_audit.json")
    if not payload:
        payload = _read_json(model_path.parent / "recovery_contract" / "topology_audit.json")
    if not payload:
        return None, None
    return payload.get("topology_unchanged"), payload.get("sparse_depth_enabled")


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _row(row_id: str, method: str, path: str, wandb: str, decision: str) -> AdaptiveRow:
    model = ROOT / path
    render = load_render_metrics(model, 26000)
    geom = load_geometry_metrics(model, 26000)
    triangles, vertices = _topology(model, 26000)
    topology_unchanged, sparse_depth_enabled = _audit(model)
    values = [
        render["psnr"],
        render["ssim"],
        render["lpips"],
        geom["abs_rel"],
        geom["depth_mae"],
        geom["normal_mean_ang_deg"],
    ]
    status = "AVAILABLE" if _finite(values) else "PENDING_OR_MISSING_EVAL"
    return AdaptiveRow(
        row_id=row_id,
        method=method,
        model_path=path,
        iteration=26000,
        wandb=wandb,
        triangles=triangles,
        vertices=vertices,
        psnr=render["psnr"],
        ssim=render["ssim"],
        lpips=render["lpips"],
        abs_rel=geom["abs_rel"],
        depth_mae=geom["depth_mae"],
        normal=geom["normal_mean_ang_deg"],
        topology_unchanged=topology_unchanged,
        sparse_depth_enabled=sparse_depth_enabled,
        status=status,
        decision=decision,
    )


def default_rows() -> list[AdaptiveRow]:
    return [
        _row(
            "F68",
            "adaptive_v4_area_primary",
            "outputs/carnet/meshsplatopt/final_stageF68_parking_adaptive_csef_policy_v4_area_primary/recovery_model",
            "lm2nzbrs",
            "clean22k_win_area_primary_control",
        ),
        _row(
            "F69",
            "adaptive_v4_sparse_lam0p001",
            "outputs/carnet/meshsplatopt/final_stageF69_parking_adaptive_csef_policy_v4_sparse_depth_lam0p001/recovery_model",
            "qetzit46",
            "strong_geometry_row_lpips_near_miss_vs_F7",
        ),
        _row(
            "F70",
            "adaptive_v4_sparse_lam0p0005",
            "outputs/carnet/meshsplatopt/final_stageF70_parking_adaptive_csef_policy_v4_sparse_depth_lam0p0005/recovery_model",
            "0m320q88",
            "best_depth_mae_row",
        ),
        _row(
            "F71",
            "adaptive_v4_sparse_lam0p001_lpips0p002",
            "outputs/carnet/meshsplatopt/final_stageF71_parking_adaptive_csef_policy_v4_sparse_lpips/recovery_model",
            "cqdpevk8",
            "best_perceptual_row_depth_tradeoff",
        ),
        _row(
            "F72",
            "adaptive_v4_sparse_lam0p001_lpips0p0005",
            "outputs/carnet/meshsplatopt/final_stageF72_parking_adaptive_csef_policy_v4_sparse_lpips0p0005/recovery_model",
            "gafbl2m7",
            "pending_low_lpips_weight_balance",
        ),
        _row(
            "F73",
            "adaptive_v4_sparse_lam0p001_lpips0p001",
            "outputs/carnet/meshsplatopt/final_stageF73_parking_adaptive_csef_policy_v4_sparse_lpips0p001/recovery_model",
            "j811febm",
            "pending_mid_lpips_weight_balance",
        ),
        _row(
            "F74",
            "adaptive_v4_sparse_lam0p001_lpips0p0001",
            "outputs/carnet/meshsplatopt/final_stageF74_parking_adaptive_csef_policy_v4_sparse_lpips0p0001/recovery_model",
            "fs3u0p4h",
            "accepted_all_metric_F7_win_tiny_lpips_weight",
        ),
        _row(
            "F75",
            "adaptive_v4_sparse_lam0p001_lpips0p00025",
            "outputs/carnet/meshsplatopt/final_stageF75_parking_adaptive_csef_policy_v4_sparse_lpips0p00025/recovery_model",
            "hhyy475d",
            "accepted_all_metric_F7_win_small_lpips_weight",
        ),
    ]


def _delta(row: AdaptiveRow, baseline: str) -> dict[str, float]:
    base = BASELINES[baseline]
    return {
        "psnr": row.psnr - float(base["psnr"]),
        "ssim": row.ssim - float(base["ssim"]),
        "lpips": row.lpips - float(base["lpips"]),
        "abs_rel": row.abs_rel - float(base["abs_rel"]),
        "depth_mae": row.depth_mae - float(base["depth_mae"]),
        "normal": row.normal - float(base["normal"]),
        "triangle_reduction": (
            1.0 - float(row.triangles) / float(base["triangles"])
            if row.triangles and base.get("triangles")
            else math.nan
        ),
    }


def _dominates(delta: dict[str, float], require_compression: bool) -> bool:
    checks = [
        delta["psnr"] >= 0.0,
        delta["ssim"] >= 0.0,
        delta["lpips"] <= 0.0,
        delta["abs_rel"] <= 0.0,
        delta["depth_mae"] <= 0.0,
        delta["normal"] <= 0.0,
    ]
    if require_compression:
        checks.append(delta["triangle_reduction"] >= 0.5)
    return all(checks)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[AdaptiveRow]) -> None:
    lines = [
        "# Stage F68-F75 Adaptive Policy Evidence",
        "",
        "| row | method | triangles | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | beats clean22k | beats R53.01 | beats F7 | beats F33 | status | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        clean = _delta(row, "clean22k")
        r53 = _delta(row, "R53.01")
        f7 = _delta(row, "F7.csef70")
        f33 = _delta(row, "F33.csef70_sparse")
        lines.append(
            f"| {row.row_id} | {row.method} | {_fmt(row.triangles)} | {_fmt(row.psnr)} | {_fmt(row.ssim)} | "
            f"{_fmt(row.lpips)} | {_fmt(row.abs_rel)} | {_fmt(row.depth_mae)} | {_fmt(row.normal)} | "
            f"{_dominates(clean, True)} | {_dominates(r53, False)} | {_dominates(f7, False)} | "
            f"{_dominates(f33, False)} | {row.status} | {row.decision} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect F68-F75 adaptive policy evidence.")
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/final_stageF75_adaptive_policy_evidence")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = default_rows()
    payload = []
    for row in rows:
        item = row.to_dict()
        item["deltas"] = {name: _delta(row, name) for name in BASELINES}
        item["dominates"] = {
            "clean22k": _dominates(item["deltas"]["clean22k"], True),
            "R53.01": _dominates(item["deltas"]["R53.01"], False),
            "F7.csef70": _dominates(item["deltas"]["F7.csef70"], False),
            "F33.csef70_sparse": _dominates(item["deltas"]["F33.csef70_sparse"], False),
        }
        payload.append(item)
    (out_dir / "adaptive_policy_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "adaptive_policy_results.csv", payload)
    _write_md(out_dir / "adaptive_policy_results.md", rows)
    available = sum(row.status == "AVAILABLE" for row in rows)
    print(f"Wrote {len(rows)} adaptive rows to {out_dir}; available={available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

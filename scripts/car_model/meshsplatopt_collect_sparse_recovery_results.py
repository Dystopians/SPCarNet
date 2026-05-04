"""Collect MeshSplatOpt sparse-recovery paper rows from local artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WANDB_ENTITY = "karamazovaniki-university-of-southern-california"


@dataclass(frozen=True)
class SparseRecoveryRow:
    row_id: str
    scene: str
    stage: str
    method: str
    model_path: str
    iteration: int
    wandb_run: str
    sample_mode: str
    trusted_fraction: float | None
    sparse_lambda: float
    role: str
    wandb_project: str = "spcarnet_meshprior"


ROWS: tuple[SparseRecoveryRow, ...] = (
    SparseRecoveryRow("R30.02", "parking_phone_tiny", "R30", "random_sparse_depth", "outputs/carnet/meshsplatopt/stageR30_02_parking_baseline_sparse_depth_lam0p005_12000to16000/recovery_model", 16000, "6gsab26p", "random", None, 0.005, "previous random baseline"),
    SparseRecoveryRow("R32.02b", "parking_phone_tiny", "R32", "mixed_trusted_random", "outputs/carnet/meshsplatopt/stageR32_02b_parking_sparse_depth_mixed_low_error_lam0p005_12000to16000/recovery_model", 16000, "j58gdh9q", "mixed_low_error", 0.50, 0.005, "first trusted-sampling render best"),
    SparseRecoveryRow("R35.01", "parking_phone_tiny", "R35", "mixed_trusted_random", "outputs/carnet/meshsplatopt/stageR35_01_parking_sparse_depth_mixed_low_error_frac0p625_lam0p005_12000to16000/recovery_model", 16000, "t8y6ryn9", "mixed_low_error", 0.625, 0.005, "geometry-balanced trusted fraction"),
    SparseRecoveryRow("R38.01", "parking_phone_tiny", "R38", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR38_01_parking_mixed_frac0p50_lam0p003_12000to16000/recovery_model", 16000, "yo6oxofn", "mixed_low_error", 0.50, 0.003, "current render and geometry best"),
    SparseRecoveryRow("R38.02", "parking_phone_tiny", "R38", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR38_02_parking_mixed_frac0p625_lam0p003_12000to16000/recovery_model", 16000, "j8t2tyc9", "mixed_low_error", 0.625, 0.003, "geometry-biased lambda-refined row"),
    SparseRecoveryRow("R39.01", "parking_phone_tiny", "R39", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR39_01_parking_mixed_frac0p50_lam0p002_12000to16000/recovery_model", 16000, "jqcn7cwc", "mixed_low_error", 0.50, 0.002, "current render/depth best"),
    SparseRecoveryRow("R39.02", "parking_phone_tiny", "R39", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR39_02_parking_mixed_frac0p50_lam0p004_12000to16000/recovery_model", 16000, "o9f9e03g", "mixed_low_error", 0.50, 0.004, "lambda upper-side check"),
    SparseRecoveryRow("R40.01", "parking_phone_tiny", "R40", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR40_01_parking_mixed_frac0p50_lam0p001_12000to16000/recovery_model", 16000, "czebaxco", "mixed_low_error", 0.50, 0.001, "parking render/LPIPS Pareto best"),
    SparseRecoveryRow("R43.01b", "parking_phone_tiny", "R43", "mixed_trusted_random_lambda_refined_long", "outputs/carnet/meshsplatopt/stageR43_01b_parking_mixed_frac0p50_lam0p001_16000to30000_long/recovery_model", 30000, "mhz6t8ps", "mixed_low_error", 0.50, 0.001, "parking long-run overtraining boundary"),
    SparseRecoveryRow("R44.01", "parking_phone_tiny", "R44", "mixed_trusted_random_sparse_decay_long", "outputs/carnet/meshsplatopt/stageR44_01_parking_decay_sparse_frac0p50_lam0p001_16000to22000/recovery_model", 22000, "c1rxa6q6", "mixed_low_error", 0.50, 0.001, "parking sparse-decay long-run repair"),
    SparseRecoveryRow("R31.02", "eth3d_courtyard", "R31", "random_sparse_depth", "outputs/carnet/meshsplatopt/stageR31_02_courtyard_stage35_sparse_depth_lam0p005_2000to7000/recovery_model", 7000, "s35bmzau", "random", None, 0.005, "cross-scene random baseline"),
    SparseRecoveryRow("R33.01", "eth3d_courtyard", "R33", "mixed_trusted_random", "outputs/carnet/meshsplatopt/stageR33_01_courtyard_stage35_mixed_sparse_depth_lam0p005_2000to7000/recovery_model", 7000, "s1po8x07", "mixed_low_error", 0.50, 0.005, "cross-scene trusted geometry check"),
    SparseRecoveryRow("R36.01b", "eth3d_courtyard", "R36", "mixed_trusted_random", "outputs/carnet/meshsplatopt/stageR36_01b_courtyard_stage35_mixed_sparse_depth_frac0p625_lam0p005_2000to7000/recovery_model", 7000, "qguqasou", "mixed_low_error", 0.625, 0.005, "courtyard tuned render best"),
    SparseRecoveryRow("R40.02", "eth3d_courtyard", "R40", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR40_02_courtyard_mixed_frac0p625_lam0p002_2000to7000/recovery_model", 7000, "coqls9rm", "mixed_low_error", 0.625, 0.002, "courtyard all-metric best"),
    SparseRecoveryRow("R43.02b", "eth3d_courtyard", "R43", "mixed_trusted_random_lambda_refined_long", "outputs/carnet/meshsplatopt/stageR43_02b_courtyard_mixed_frac0p625_lam0p002_7000to20000_long/recovery_model", 20000, "cla3utia", "mixed_low_error", 0.625, 0.002, "courtyard long-run render best, depth tradeoff"),
    SparseRecoveryRow("R44.02", "eth3d_courtyard", "R44", "mixed_trusted_random_sparse_decay_long", "outputs/carnet/meshsplatopt/stageR44_02_courtyard_decay_sparse_frac0p625_lam0p002_7000to20000/recovery_model", 20000, "5tleod3c", "mixed_low_error", 0.625, 0.002, "courtyard sparse-decay long-run repair"),
    SparseRecoveryRow("R31.03", "mipnerf360_bonsai", "R31", "random_sparse_depth", "outputs/carnet/meshsplatopt/stageR31_03_bonsai_stage35_sparse_depth_lam0p005_2000to7000/recovery_model", 7000, "3wygm9u4", "random", None, 0.005, "cross-scene random baseline"),
    SparseRecoveryRow("R33.02", "mipnerf360_bonsai", "R33", "mixed_trusted_random", "outputs/carnet/meshsplatopt/stageR33_02_bonsai_stage35_mixed_sparse_depth_lam0p005_2000to7000/recovery_model", 7000, "xj2ng1s1", "mixed_low_error", 0.50, 0.005, "cross-scene trusted geometry check"),
    SparseRecoveryRow("R36.02b", "mipnerf360_bonsai", "R36", "mixed_trusted_random", "outputs/carnet/meshsplatopt/stageR36_02b_bonsai_stage35_mixed_sparse_depth_frac0p625_lam0p005_2000to7000/recovery_model", 7000, "xq21lzsm", "mixed_low_error", 0.625, 0.005, "bonsai fraction stress test"),
    SparseRecoveryRow("R41.01", "mipnerf360_bonsai", "R41", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR41_01_bonsai_mixed_frac0p50_lam0p002_2000to7000/recovery_model", 7000, "poh8k4be", "mixed_low_error", 0.50, 0.002, "bonsai render breakthrough, geometry tradeoff", "mesh-splatting"),
    SparseRecoveryRow("R42.01", "mipnerf360_bonsai", "R42", "mixed_trusted_random_lambda_refined", "outputs/carnet/meshsplatopt/stageR42_01_bonsai_mixed_frac0p625_lam0p002_2000to7000/recovery_model", 7000, "l2inxutg", "mixed_low_error", 0.625, 0.002, "bonsai fraction repair check"),
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _metrics(model: Path, iteration: int) -> dict[str, float]:
    payload = _load_json(model / "results.json").get(f"ours_{int(iteration)}", {})
    return {
        "psnr": _float(payload.get("PSNR")),
        "ssim": _float(payload.get("SSIM")),
        "lpips": _float(payload.get("LPIPS")),
    }


def _geometry(model: Path, iteration: int) -> dict[str, float]:
    geom_dir = model / "geometry_eval_colmap"
    candidates = [
        geom_dir / f"iter_{int(iteration)}_max500.json",
        geom_dir / f"iter_{int(iteration)}.json",
    ]
    payload: dict[str, Any] = {}
    for path in candidates:
        payload = _load_json(path)
        if payload:
            break
    depth = payload.get("depth") or {}
    normal = payload.get("normal") or {}
    return {
        "abs_rel": _float(depth.get("abs_rel")),
        "depth_mae": _float(depth.get("mae")),
        "normal_mean_ang_deg": _float(normal.get("mean_ang_deg")),
    }


def _row(spec: SparseRecoveryRow) -> dict[str, Any]:
    model = ROOT / spec.model_path
    metrics = _metrics(model, spec.iteration)
    geometry = _geometry(model, spec.iteration)
    status = "AVAILABLE" if all(not math.isnan(v) for v in [*metrics.values(), *geometry.values()]) else "MISSING"
    return {
        "row_id": spec.row_id,
        "scene": spec.scene,
        "stage": spec.stage,
        "method": spec.method,
        "sample_mode": spec.sample_mode,
        "trusted_fraction": spec.trusted_fraction,
        "sparse_lambda": spec.sparse_lambda,
        "iteration": spec.iteration,
        "status": status,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "abs_rel": geometry["abs_rel"],
        "depth_mae": geometry["depth_mae"],
        "normal_mean_ang_deg": geometry["normal_mean_ang_deg"],
        "wandb_run": spec.wandb_run,
        "wandb_url": f"https://wandb.ai/{WANDB_ENTITY}/{spec.wandb_project}/runs/{spec.wandb_run}" if spec.wandb_run else "",
        "model_path": spec.model_path,
        "role": spec.role,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["row_id", "scene", "sample_mode", "trusted_fraction", "sparse_lambda", "psnr", "ssim", "lpips", "abs_rel", "depth_mae", "normal_mean_ang_deg", "wandb_run"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/sparse_recovery_tables")
    args = parser.parse_args()
    out_dir = ROOT / args.out_dir
    rows = [_row(spec) for spec in ROWS]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse_recovery_results.json").write_text(json.dumps(_clean(rows), indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "sparse_recovery_results.csv", rows)
    _write_markdown(out_dir / "sparse_recovery_results.md", rows)
    print(f"Wrote {len(rows)} rows to {out_dir}")
    missing = [row["row_id"] for row in rows if row["status"] != "AVAILABLE"]
    if missing:
        print("Missing rows:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

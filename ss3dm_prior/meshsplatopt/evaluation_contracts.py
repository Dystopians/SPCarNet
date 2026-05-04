from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math


@dataclass(frozen=True)
class MetricTargets:
    psnr_margin: float = 0.0
    ssim_margin: float = 0.0
    lpips_margin: float = 0.0
    abs_rel_margin: float = 0.0
    depth_mae_margin: float = 0.0
    normal_margin: float = 0.0
    triangle_reduction_min: float = 0.0


@dataclass(frozen=True)
class MethodResult:
    row_id: str
    scene: str
    method: str
    model_path: str
    iteration: int
    role: str = ""
    wandb_run: str = ""
    psnr: float = math.nan
    ssim: float = math.nan
    lpips: float = math.nan
    abs_rel: float = math.nan
    depth_mae: float = math.nan
    normal_mean_ang_deg: float = math.nan
    triangles: int | None = None
    vertices: int | None = None
    status: str = "UNKNOWN"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseComparison:
    candidate_id: str
    baseline_id: str
    scene: str
    pass_all_targets: bool
    deltas: dict[str, float]
    failed_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def load_render_metrics(model_path: str | Path, iteration: int) -> dict[str, float]:
    model = Path(model_path)
    payload = read_json(model / "results.json").get(f"ours_{int(iteration)}", {})
    return {
        "psnr": number(payload.get("PSNR")),
        "ssim": number(payload.get("SSIM")),
        "lpips": number(payload.get("LPIPS")),
    }


def load_geometry_metrics(model_path: str | Path, iteration: int) -> dict[str, float]:
    model = Path(model_path)
    geom_dir = model / "geometry_eval_colmap"
    candidates = [
        geom_dir / f"iter_{int(iteration)}_max500.json",
        geom_dir / f"iter_{int(iteration)}.json",
    ]
    payload: dict[str, Any] = {}
    for path in candidates:
        payload = read_json(path)
        if payload:
            break
    depth = payload.get("depth") or {}
    normal = payload.get("normal") or {}
    return {
        "abs_rel": number(depth.get("abs_rel")),
        "depth_mae": number(depth.get("mae")),
        "normal_mean_ang_deg": number(normal.get("mean_ang_deg")),
    }


def compare_to_baseline(candidate: MethodResult, baseline: MethodResult, targets: MetricTargets) -> PairwiseComparison:
    deltas = {
        "psnr": candidate.psnr - baseline.psnr,
        "ssim": candidate.ssim - baseline.ssim,
        "lpips": candidate.lpips - baseline.lpips,
        "abs_rel": candidate.abs_rel - baseline.abs_rel,
        "depth_mae": candidate.depth_mae - baseline.depth_mae,
        "normal_mean_ang_deg": candidate.normal_mean_ang_deg - baseline.normal_mean_ang_deg,
        "triangle_reduction": (
            1.0 - float(candidate.triangles) / float(baseline.triangles)
            if candidate.triangles and baseline.triangles
            else math.nan
        ),
    }
    failed: list[str] = []
    checks = {
        "psnr": deltas["psnr"] >= targets.psnr_margin,
        "ssim": deltas["ssim"] >= targets.ssim_margin,
        "lpips": deltas["lpips"] <= -targets.lpips_margin,
        "abs_rel": deltas["abs_rel"] <= -targets.abs_rel_margin,
        "depth_mae": deltas["depth_mae"] <= -targets.depth_mae_margin,
        "normal_mean_ang_deg": deltas["normal_mean_ang_deg"] <= -targets.normal_margin,
        "triangle_reduction": deltas["triangle_reduction"] >= targets.triangle_reduction_min,
    }
    for key, passed in checks.items():
        if not passed:
            failed.append(key)
    return PairwiseComparison(
        candidate_id=candidate.row_id,
        baseline_id=baseline.row_id,
        scene=candidate.scene,
        pass_all_targets=not failed,
        deltas=deltas,
        failed_targets=failed,
    )


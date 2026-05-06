from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


HIGHER_IS_BETTER = {"PSNR", "SSIM", "psnr", "ssim"}
LOWER_IS_BETTER = {"LPIPS", "lpips", "AbsRel", "absrel", "DepthMAE", "depth_mae", "Normal", "normal"}


@dataclass(frozen=True)
class CertifiedSelectionConfig:
    psnr_tolerance: float = 0.0
    ssim_tolerance: float = 0.0
    lpips_tolerance: float = 0.0
    min_score_delta: float = 0.0


def _metric_delta(candidate: Mapping[str, float], parent: Mapping[str, float], key: str) -> float | None:
    if key not in candidate or key not in parent:
        return None
    return float(candidate[key]) - float(parent[key])


def render_pareto_guard(
    *,
    parent: Mapping[str, float],
    candidate: Mapping[str, float],
    cfg: CertifiedSelectionConfig,
) -> dict:
    deltas = {}
    reasons = []
    for key in ("PSNR", "SSIM", "LPIPS"):
        delta = _metric_delta(candidate, parent, key)
        if delta is not None:
            deltas[f"delta_{key}"] = float(delta)
    if deltas.get("delta_PSNR", 0.0) < -float(cfg.psnr_tolerance):
        reasons.append("psnr_regression")
    if deltas.get("delta_SSIM", 0.0) < -float(cfg.ssim_tolerance):
        reasons.append("ssim_regression")
    if deltas.get("delta_LPIPS", 0.0) > float(cfg.lpips_tolerance):
        reasons.append("lpips_regression")
    score_delta = (
        float(deltas.get("delta_PSNR", 0.0))
        + float(deltas.get("delta_SSIM", 0.0))
        - float(deltas.get("delta_LPIPS", 0.0))
    )
    deltas["render_score_delta"] = float(score_delta)
    if score_delta < float(cfg.min_score_delta):
        reasons.append("render_score_below_threshold")
    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "deltas": deltas,
        "config": asdict(cfg),
    }


def select_certified_render_candidate(
    *,
    parent: Mapping[str, float],
    candidate: Mapping[str, float],
    cfg: CertifiedSelectionConfig,
) -> dict:
    guard = render_pareto_guard(parent=parent, candidate=candidate, cfg=cfg)
    selected = "candidate" if bool(guard["pass"]) else "parent"
    return {
        "selected": selected,
        "guard": guard,
        "parent": dict(parent),
        "candidate": dict(candidate),
    }

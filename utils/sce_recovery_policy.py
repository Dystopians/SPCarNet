from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


LOWER_IS_BETTER = {"lpips", "absrel", "depth_mae", "normal"}
HIGHER_IS_BETTER = {"psnr", "ssim"}


@dataclass(frozen=True)
class SCEPolicyConfig:
    policy_name: str = "sce_v1"
    visual_probe_iters: int = 500
    recovery_phase_iters: int = 500
    sentinel_check_interval: int = 500
    rollback_activation_absrel_delta: float = 0.0
    rollback_activation_depth_delta: float = 0.0
    rollback_lambda_base: float = 1.0
    rollback_loss_space: str = "absrel"
    rollback_combined_mae_beta: float = 1.0
    rollback_cluster_top_k: int = 0
    rollback_regressed_only: bool = True
    rollback_aggregation: str = "mean"
    rollback_cvar_fraction: float = 0.2
    rollback_cvar_min_points: int = 16
    rollback_pixel_radius: int = 0
    rollback_patch_reduce: str = "center"
    sparse_lambda: float = 0.003
    render_normal_anchor_lambda: float = 0.01
    render_depth_anchor_lambda: float = 0.0
    lr_triangles_points_init: float = 0.015
    early_stop_patience: int = 1
    parent_tolerance: float = 0.0
    require_sentinel_gate_for_recovery: bool = False
    require_measured_candidate_for_recovery: bool = False
    max_psnr_drop: float = 0.0
    max_ssim_drop: float = 0.0
    max_lpips_increase: float = 0.0
    min_render_score_delta: float = 0.0
    require_parent_pareto_for_acceptance: bool = False


def _metric_delta(candidate: Mapping[str, float], parent: Mapping[str, float], key: str) -> float:
    if key not in candidate or key not in parent:
        return 0.0
    c = float(candidate[key])
    p = float(parent[key])
    return c - p


def sentinel_gate_degrades(gate: Mapping[str, Any], cfg: SCEPolicyConfig) -> bool:
    metrics = dict(gate.get("metrics", gate))
    delta_absrel = float(metrics.get("delta_absrel", 0.0))
    delta_mae = float(metrics.get("delta_mae", metrics.get("delta_depth_mae", 0.0)))
    checks = dict(gate.get("checks", {}))
    return bool(
        delta_absrel > float(cfg.rollback_activation_absrel_delta)
        or delta_mae > float(cfg.rollback_activation_depth_delta)
        or checks.get("mean_absrel_nonregression") is False
        or checks.get("mean_mae_nonregression") is False
    )


def render_guard_pass(
    candidate: Mapping[str, float] | None,
    parent: Mapping[str, float] | None,
    cfg: SCEPolicyConfig,
) -> tuple[bool, list[str], dict[str, float]]:
    if not candidate or not parent:
        if bool(cfg.require_measured_candidate_for_recovery):
            return False, ["missing_measured_candidate_metrics"], {}
        return True, [], {}
    deltas: dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips"):
        if key in candidate and key in parent:
            deltas[f"delta_{key}"] = _metric_delta(candidate, parent, key)
    reasons: list[str] = []
    if deltas.get("delta_psnr", 0.0) < -float(cfg.max_psnr_drop):
        reasons.append("psnr_drop_exceeds_guard")
    if deltas.get("delta_ssim", 0.0) < -float(cfg.max_ssim_drop):
        reasons.append("ssim_drop_exceeds_guard")
    if deltas.get("delta_lpips", 0.0) > float(cfg.max_lpips_increase):
        reasons.append("lpips_increase_exceeds_guard")
    render_score = (
        float(deltas.get("delta_psnr", 0.0))
        + float(deltas.get("delta_ssim", 0.0))
        - float(deltas.get("delta_lpips", 0.0))
    )
    deltas["render_score_delta"] = render_score
    if render_score < float(cfg.min_render_score_delta):
        reasons.append("render_score_below_guard")
    return len(reasons) == 0, reasons, deltas


def parent_pareto_guard_pass(
    candidate: Mapping[str, float] | None,
    parent: Mapping[str, float] | None,
    cfg: SCEPolicyConfig,
) -> tuple[bool, list[str], dict[str, float]]:
    if not candidate or not parent:
        if bool(cfg.require_parent_pareto_for_acceptance):
            return False, ["missing_parent_pareto_metrics"], {}
        return True, [], {}
    deltas: dict[str, float] = {}
    reasons: list[str] = []
    tol = float(cfg.parent_tolerance)
    for key in sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER):
        if key not in candidate or key not in parent:
            continue
        delta = _metric_delta(candidate, parent, key)
        deltas[f"delta_{key}"] = delta
        if key in HIGHER_IS_BETTER and delta + tol < 0.0:
            reasons.append(f"{key}_below_parent")
        if key in LOWER_IS_BETTER and delta > tol:
            reasons.append(f"{key}_above_parent")
    return len(reasons) == 0, reasons, deltas


def decide_sce_policy_action(
    *,
    sentinel_gate: Mapping[str, Any] | None,
    cfg: SCEPolicyConfig,
    candidate_metrics: Mapping[str, float] | None = None,
    parent_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if sentinel_gate is None:
        if bool(cfg.require_sentinel_gate_for_recovery):
            return {
                "action": "accept_parent_noop",
                "activate_rollback": False,
                "execute_recovery": False,
                "reason": "missing_sentinel_gate_policy_guard",
                "policy": asdict(cfg),
            }
        return {
            "action": "run_visual_probe_then_gate",
            "activate_rollback": False,
            "execute_recovery": True,
            "reason": "missing_sentinel_gate",
            "policy": asdict(cfg),
        }
    degraded = sentinel_gate_degrades(sentinel_gate, cfg)
    render_ok, render_reasons, render_deltas = render_guard_pass(candidate_metrics, parent_metrics, cfg)
    if not render_ok:
        return {
            "action": "accept_parent_noop",
            "activate_rollback": False,
            "execute_recovery": False,
            "reason": "render_guard_failed",
            "render_guard_reasons": render_reasons,
            "render_deltas": render_deltas,
            "policy": asdict(cfg),
        }
    pareto_ok, pareto_reasons, pareto_deltas = parent_pareto_guard_pass(candidate_metrics, parent_metrics, cfg)
    if not pareto_ok:
        return {
            "action": "accept_parent_noop",
            "activate_rollback": False,
            "execute_recovery": False,
            "reason": "parent_pareto_guard_failed",
            "parent_pareto_guard_reasons": pareto_reasons,
            "parent_pareto_deltas": pareto_deltas,
            "render_guard_reasons": render_reasons,
            "render_deltas": render_deltas,
            "policy": asdict(cfg),
        }
    return {
        "action": "run_targeted_rollback" if degraded else "continue_or_accept_visual_recovery",
        "activate_rollback": bool(degraded),
        "execute_recovery": True,
        "reason": "sentinel_degraded" if degraded else "sentinel_non_degrading",
        "render_guard_reasons": render_reasons,
        "render_deltas": render_deltas,
        "parent_pareto_guard_reasons": pareto_reasons,
        "parent_pareto_deltas": pareto_deltas,
        "policy": asdict(cfg),
    }


def candidate_parent_pareto_pass(
    candidate: Mapping[str, float],
    parent: Mapping[str, float],
    *,
    tolerance: float = 0.0,
) -> bool:
    for key in HIGHER_IS_BETTER:
        if key in candidate and key in parent and float(candidate[key]) + tolerance < float(parent[key]):
            return False
    for key in LOWER_IS_BETTER:
        if key in candidate and key in parent and float(candidate[key]) > float(parent[key]) + tolerance:
            return False
    return True


def _score_candidate(candidate: Mapping[str, float], parent: Mapping[str, float]) -> float:
    score = 0.0
    for key in HIGHER_IS_BETTER:
        score += _metric_delta(candidate, parent, key)
    for key in LOWER_IS_BETTER:
        score -= _metric_delta(candidate, parent, key)
    return float(score)


def select_early_stop_candidate(
    history: Sequence[Mapping[str, Any]],
    *,
    parent_metrics: Mapping[str, float],
    cfg: SCEPolicyConfig,
) -> dict[str, Any]:
    rows = [dict(row) for row in history]
    if not rows:
        return {"decision": "NO_CANDIDATES", "selected": None, "reason": "empty_history"}
    passing = [
        row
        for row in rows
        if candidate_parent_pareto_pass(
            row.get("metrics", row),
            parent_metrics,
            tolerance=float(cfg.parent_tolerance),
        )
        and bool(row.get("sentinel_pass", True))
        and render_guard_pass(row.get("metrics", row), parent_metrics, cfg)[0]
    ]
    pool = passing if passing else rows
    selected = max(pool, key=lambda row: _score_candidate(row.get("metrics", row), parent_metrics))
    return {
        "decision": "PARENT_PARETO_EARLY_STOP" if passing else "BEST_AVAILABLE_PARTIAL",
        "selected": selected,
        "reason": "selected_parent_pareto_candidate" if passing else "no_full_parent_pareto_candidate",
        "num_candidates": len(rows),
        "num_parent_pareto": len(passing),
    }


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))

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
    sparse_lambda: float = 0.003
    render_normal_anchor_lambda: float = 0.01
    render_depth_anchor_lambda: float = 0.0
    lr_triangles_points_init: float = 0.015
    early_stop_patience: int = 1
    parent_tolerance: float = 0.0


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


def decide_sce_policy_action(
    *,
    sentinel_gate: Mapping[str, Any] | None,
    cfg: SCEPolicyConfig,
) -> dict[str, Any]:
    if sentinel_gate is None:
        return {
            "action": "run_visual_probe_then_gate",
            "activate_rollback": False,
            "reason": "missing_sentinel_gate",
            "policy": asdict(cfg),
        }
    degraded = sentinel_gate_degrades(sentinel_gate, cfg)
    return {
        "action": "run_targeted_rollback" if degraded else "continue_or_accept_visual_recovery",
        "activate_rollback": bool(degraded),
        "reason": "sentinel_degraded" if degraded else "sentinel_non_degrading",
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

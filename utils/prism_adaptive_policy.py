from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AdaptiveCSEFPolicyConfig:
    enabled: bool = False
    min_ratio: float = 0.006
    max_ratio: float = 0.020
    initial_ratio: float = 0.012
    target_accept_margin: float = 0.55
    rollback_decay: float = 0.55
    accept_growth: float = 1.18
    no_candidate_decay: float = 0.75
    cooldown_iters: int = 20
    max_candidate_count: int = 0
    min_candidate_count: int = 512
    depth_degrade_absrel: float = 0.004
    normal_degrade_deg: float = 0.10
    render_degrade_psnr: float = -0.05
    uncertainty_high: float = 0.35
    geometry_keep_high: float = 0.04
    orientation_keep_high: float = 0.04
    reliable_absrel_max: float = 2.0
    strict_gate_after_rejects: int = 1
    normal_repair_penalty_boost: float = 0.8
    geometry_repair_penalty_boost: float = 0.8
    uncertainty_penalty_boost: float = 0.6
    cold_start_rounds: int = 1
    cold_start_gate_scale: float = 0.70
    cold_start_ratio_damping: float = 0.96
    cold_start_quality_rank: bool = False
    enable_measured_rank: bool = True
    enable_microbatch_gate: bool = True
    microbatch_size: int = 512
    microbatch_max_batches: int = 0


@dataclass
class AdaptiveCSEFPolicyDecision:
    enabled: bool
    ratio: float
    candidate_max_count: int
    use_quality_rank: bool
    render_penalty: float
    geometry_penalty: float
    orientation_penalty: float
    utility_penalty: float
    uncertainty_penalty: float
    gate_scale: float
    cooldown_iters: int
    use_measured_rank: bool
    use_microbatch_gate: bool
    microbatch_size: int
    microbatch_max_batches: int
    confidence: float
    mode: str
    reason: str
    risk: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _last_decision_payload(policy_state: Dict[str, Any]) -> Dict[str, Any]:
    decision = policy_state.get("last_counterfactual_decision", None)
    if decision is None:
        return {}
    deltas = getattr(decision, "deltas", None)
    baseline = getattr(decision, "baseline", None)
    if isinstance(decision, dict):
        deltas = decision.get("deltas", deltas)
        baseline = decision.get("baseline", baseline)
    return {
        "deltas": dict(deltas or {}),
        "baseline": dict(baseline or {}),
    }


def decide_adaptive_csef_policy(
    *,
    cfg: AdaptiveCSEFPolicyConfig,
    prism_state: Dict[str, Any],
    scores_summary: Optional[Dict[str, Any]],
    iteration: int,
    total_triangles: int,
) -> AdaptiveCSEFPolicyDecision:
    if not bool(cfg.enabled):
        ratio = _finite(prism_state.get("adaptive_candidate_prune_ratio", cfg.initial_ratio), cfg.initial_ratio)
        return AdaptiveCSEFPolicyDecision(
            enabled=False,
            ratio=ratio,
            candidate_max_count=0,
            use_quality_rank=False,
            render_penalty=0.5,
            geometry_penalty=0.5,
            orientation_penalty=0.25,
            utility_penalty=0.25,
            uncertainty_penalty=0.25,
            gate_scale=1.0,
            cooldown_iters=0,
            use_measured_rank=False,
            use_microbatch_gate=False,
            microbatch_size=0,
            microbatch_max_batches=0,
            confidence=1.0,
            mode="disabled",
            reason="adaptive_policy_disabled",
            risk={},
        )

    scores_summary = dict(scores_summary or {})
    state = prism_state.setdefault("adaptive_csef_policy", {})
    current_ratio = _finite(state.get("ratio", cfg.initial_ratio), cfg.initial_ratio)
    current_ratio = min(max(current_ratio, cfg.min_ratio), cfg.max_ratio)

    rollback_streak = int(state.get("rollback_streak", 0))
    accept_streak = int(state.get("accept_streak", 0))
    no_candidate_streak = int(state.get("no_candidate_streak", 0))
    commit_count = int(prism_state.get("candidate_commit_count", state.get("commit_count", 0)))
    cold_start = commit_count < int(cfg.cold_start_rounds)
    last = _last_decision_payload(prism_state)
    deltas = last.get("deltas", {})
    baseline = last.get("baseline", {})

    delta_psnr = _finite(deltas.get("delta_psnr", 0.0))
    delta_absrel = _finite(deltas.get("delta_absrel", 0.0))
    delta_normal = _finite(deltas.get("delta_mean_angle", 0.0))
    changed_pixel = _finite(deltas.get("changed_pixel_ratio", 0.0))
    baseline_absrel = _finite(baseline.get("absrel", np.inf), np.inf)
    geometry_reliable = baseline_absrel <= float(cfg.reliable_absrel_max)

    unc_frac = _finite(scores_summary.get("uncertain_fraction", scores_summary.get("unc_nonzero_fraction", 0.0)))
    geom_keep_frac = _finite(scores_summary.get("geometry_keep_nonzero_fraction", 0.0))
    orient_keep_frac = _finite(scores_summary.get("orientation_keep_nonzero_fraction", 0.0))
    if "candidate_fraction" in scores_summary:
        candidate_frac = _finite(scores_summary.get("candidate_fraction", 0.0))
    else:
        candidate_frac = _finite(scores_summary.get("num_candidates", 0.0)) / max(
            1.0, _finite(scores_summary.get("num_triangles", 1.0), 1.0)
        )
    heavy_frac = _finite(scores_summary.get("heavy_eval_fraction", 0.0))

    depth_risk = 1.0 if geometry_reliable and delta_absrel > float(cfg.depth_degrade_absrel) else 0.0
    normal_risk = 1.0 if delta_normal > float(cfg.normal_degrade_deg) else 0.0
    render_risk = 1.0 if delta_psnr < float(cfg.render_degrade_psnr) else 0.0
    uncertainty_risk = min(1.0, max(0.0, unc_frac / max(float(cfg.uncertainty_high), 1e-6)))
    geometry_keep_risk = min(1.0, max(0.0, geom_keep_frac / max(float(cfg.geometry_keep_high), 1e-6)))
    orientation_keep_risk = min(1.0, max(0.0, orient_keep_frac / max(float(cfg.orientation_keep_high), 1e-6)))
    rollback_risk = min(1.0, rollback_streak / 3.0)
    no_candidate_risk = min(1.0, no_candidate_streak / 3.0)

    risk_score = (
        0.22 * depth_risk
        + 0.18 * normal_risk
        + 0.16 * render_risk
        + 0.14 * uncertainty_risk
        + 0.12 * geometry_keep_risk
        + 0.10 * orientation_keep_risk
        + 0.06 * rollback_risk
        + 0.02 * no_candidate_risk
    )
    confidence = float(min(1.0, max(0.05, 1.0 - risk_score)))

    ratio = current_ratio
    reasons = []
    if rollback_streak > 0:
        ratio *= float(cfg.rollback_decay) ** rollback_streak
        reasons.append(f"rollback_streak={rollback_streak}")
    if no_candidate_streak > 0:
        ratio *= float(cfg.no_candidate_decay) ** no_candidate_streak
        reasons.append(f"no_candidate_streak={no_candidate_streak}")
    if accept_streak > 0 and risk_score < (1.0 - float(cfg.target_accept_margin)):
        ratio *= float(cfg.accept_growth) ** min(accept_streak, 3)
        reasons.append(f"accept_streak={accept_streak}")
    if cold_start:
        ratio *= float(cfg.cold_start_ratio_damping)
        reasons.append("cold_start_damping")

    ratio *= max(0.45, min(1.15, confidence + 0.15))
    ratio = float(min(max(ratio, cfg.min_ratio), cfg.max_ratio))

    cap_from_ratio = int(max(1, round(ratio * max(1, int(total_triangles)))))
    candidate_max_count = cap_from_ratio
    if int(cfg.max_candidate_count) > 0:
        candidate_max_count = min(candidate_max_count, int(cfg.max_candidate_count))
    candidate_max_count = max(int(cfg.min_candidate_count), candidate_max_count)

    gate_scale = 1.0
    if cold_start:
        gate_scale = float(cfg.cold_start_gate_scale)
        reasons.append("cold_start_gate")
    elif rollback_streak >= int(cfg.strict_gate_after_rejects):
        gate_scale = 0.65
        reasons.append("strict_after_reject")
    elif risk_score < 0.18 and accept_streak > 0:
        gate_scale = 1.15
        reasons.append("stable_accept_margin")

    geometry_penalty = 0.5 + float(cfg.geometry_repair_penalty_boost) * max(depth_risk, geometry_keep_risk)
    orientation_penalty = 0.25 + float(cfg.normal_repair_penalty_boost) * max(normal_risk, orientation_keep_risk)
    render_penalty = 0.5 + 0.5 * render_risk + 0.25 * changed_pixel
    uncertainty_penalty = 0.25 + float(cfg.uncertainty_penalty_boost) * uncertainty_risk
    utility_penalty = 0.25 + 0.25 * max(render_risk, depth_risk, normal_risk)

    if cold_start:
        mode = "cold_start_pareto_probe"
        cooldown_iters = 0
    elif risk_score >= 0.55:
        mode = "conservative_repair"
        cooldown_iters = int(cfg.cooldown_iters)
    elif risk_score <= 0.20 and candidate_frac > 0.0:
        mode = "opportunistic_compact"
        cooldown_iters = 0
    else:
        mode = "balanced"
        cooldown_iters = 0

    risk = {
        "risk_score": float(risk_score),
        "depth_risk": float(depth_risk),
        "normal_risk": float(normal_risk),
        "render_risk": float(render_risk),
        "uncertainty_risk": float(uncertainty_risk),
        "geometry_keep_risk": float(geometry_keep_risk),
        "orientation_keep_risk": float(orientation_keep_risk),
        "rollback_risk": float(rollback_risk),
        "no_candidate_risk": float(no_candidate_risk),
        "candidate_fraction": float(candidate_frac),
        "heavy_eval_fraction": float(heavy_frac),
        "geometry_reliable": float(1.0 if geometry_reliable else 0.0),
        "baseline_absrel": float(baseline_absrel if np.isfinite(baseline_absrel) else -1.0),
        "cold_start": float(1.0 if cold_start else 0.0),
        "commit_count": float(commit_count),
    }

    use_quality_rank = True
    if cold_start:
        use_quality_rank = bool(cfg.cold_start_quality_rank)
    use_measured_rank = bool(cfg.enable_measured_rank and (cold_start or risk_score >= 0.35))
    use_microbatch_gate = bool(cfg.enable_microbatch_gate and cold_start)

    decision = AdaptiveCSEFPolicyDecision(
        enabled=True,
        ratio=ratio,
        candidate_max_count=int(candidate_max_count),
        use_quality_rank=bool(use_quality_rank),
        render_penalty=float(render_penalty),
        geometry_penalty=float(geometry_penalty),
        orientation_penalty=float(orientation_penalty),
        utility_penalty=float(utility_penalty),
        uncertainty_penalty=float(uncertainty_penalty),
        gate_scale=float(gate_scale),
        cooldown_iters=int(cooldown_iters),
        use_measured_rank=bool(use_measured_rank),
        use_microbatch_gate=bool(use_microbatch_gate),
        microbatch_size=int(cfg.microbatch_size),
        microbatch_max_batches=int(cfg.microbatch_max_batches),
        confidence=float(confidence),
        mode=str(mode),
        reason=",".join(reasons) if reasons else "evidence_balanced",
        risk=risk,
    )
    state["last_decision"] = decision.to_dict()
    state["ratio"] = float(ratio)
    state["commit_count"] = int(commit_count)
    return decision


def update_adaptive_csef_policy_after_prune(
    *,
    prism_state: Dict[str, Any],
    committed: bool,
    rollback: bool,
    no_candidates: bool,
    ratio: float,
) -> None:
    state = prism_state.setdefault("adaptive_csef_policy", {})
    if bool(rollback):
        state["rollback_streak"] = int(state.get("rollback_streak", 0)) + 1
        state["accept_streak"] = 0
    elif bool(committed):
        state["accept_streak"] = int(state.get("accept_streak", 0)) + 1
        state["rollback_streak"] = 0
        state["commit_count"] = int(state.get("commit_count", 0)) + 1
    else:
        state["accept_streak"] = 0
        state["rollback_streak"] = 0
    if bool(no_candidates):
        state["no_candidate_streak"] = int(state.get("no_candidate_streak", 0)) + 1
    else:
        state["no_candidate_streak"] = 0
    state["ratio"] = float(ratio)

#!/usr/bin/env python3
"""Run a train-val render-risk selector over face-local residual plans.

The script is an outer-loop selector for Phase-S face-local residual plans. It
does not change the underlying render/eval gate. For legacy non-strict plans it
can build a fixed set of face subsets from a train-only candidate plan. For
strict PatchCert carrier plans it defaults to a certification-preserving full
plan replay: no face subset, no coefficient rescale, and no alpha refit. Held-out
test deltas are copied as report-only evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_PLAN_TEMPLATE = (
    "outputs/carnet/meshsplatopt/ecsr_phase_s/"
    "facelocal_rendercalib_v1_plan_20260513/{scene}/facelocal_sh3_candidate_plan.json"
)
DEFAULT_EVIDENCE_ROOT = "outputs/carnet/meshsplatopt/ecsr_phase_r/surface_evidence_uniform_sh1_v6_dense16"
DEFAULT_PHASEJ_TEST_METHOD = "ours_26000_phasej_guarded_adaptedge_ela_replay_rendercalib_v1_top1_s2_fair"
DEFAULT_PHASEJ_TRAINVAL_METHOD = "ours_26000_phasej_trainval_gate_rendercalib_v1_top1_s2_fair"


@dataclass(frozen=True)
class TrialSpec:
    label: str
    mode: str
    count: int
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default="bicycle")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument(
        "--output_root",
        default="outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_coupled_selector_v1_20260513",
    )
    parser.add_argument(
        "--reuse_trials_root",
        default="",
        help=(
            "Optional existing coupled-selector output root whose per-trial Phase-K decisions "
            "are reused for a re-decision pass. This lets stricter selector thresholds be "
            "audited without rerendering the same trials."
        ),
    )
    parser.add_argument("--plan_template", default=DEFAULT_PLAN_TEMPLATE)
    parser.add_argument("--evidence_root", default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--trial_specs",
        default="top1x2,score2x1,score4x1,score8x0.5",
        help=(
            "Comma separated trials. Grammar: topNxS, scoreNxS, riskNxS, georiskNxS, "
            "or patchriskNxS, for example top1x2,score4x1,risk8x0.5,georisk4x1,"
            "patchrisk2x0.75. score/risk/georisk/patchrisk use train-only plan "
            "certificates."
        ),
    )
    parser.add_argument(
        "--risk_pair_lambda",
        type=float,
        default=0.65,
        help="Penalty strength for risk-mode greedy view-overlap redundancy.",
    )
    parser.add_argument(
        "--georisk_pair_lambda",
        type=float,
        default=-1.0,
        help="Pair-risk penalty for georisk mode. Negative means reuse --risk_pair_lambda.",
    )
    parser.add_argument(
        "--georisk_geometry_lambda",
        type=float,
        default=0.35,
        help="Penalty for selecting geometrically adjacent residual faces in georisk mode.",
    )
    parser.add_argument(
        "--georisk_tail_lambda",
        type=float,
        default=0.55,
        help="Penalty for low-CVaR/negative per-view train certificate tails in georisk mode.",
    )
    parser.add_argument(
        "--georisk_error_lambda",
        type=float,
        default=0.12,
        help="Small bonus for train-only local residual-error concentration in georisk mode.",
    )
    parser.add_argument(
        "--georisk_tail_fraction",
        type=float,
        default=0.30,
        help="Worst-view fraction used for per-face CVaR certificate in georisk mode.",
    )
    parser.add_argument(
        "--georisk_min_view_gain",
        type=float,
        default=0.02,
        help="Soft minimum per-view relative gain used by georisk tail risk.",
    )
    parser.add_argument(
        "--georisk_load_adjacency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load source checkpoint triangle indices to build candidate face adjacency for georisk mode.",
    )
    parser.add_argument(
        "--patchrisk_rings",
        type=int,
        default=1,
        help="Neighborhood rings grown from georisk seed faces for patchrisk mode.",
    )
    parser.add_argument(
        "--patchrisk_max_patch_faces",
        type=int,
        default=6,
        help="Maximum materialized candidate-plan faces per patchrisk seed.",
    )
    parser.add_argument(
        "--patchrisk_max_total_faces",
        type=int,
        default=24,
        help="Maximum total materialized candidate-plan faces per patchrisk trial.",
    )
    parser.add_argument(
        "--patchrisk_neighbor_mode",
        choices=("topology", "centroid", "both"),
        default="both",
        help="Candidate-neighborhood source used by patchrisk growth.",
    )
    parser.add_argument(
        "--patchrisk_centroid_candidates_per_seed",
        type=int,
        default=32,
        help="Nearest candidate faces considered when patchrisk uses centroid neighbors.",
    )
    parser.add_argument(
        "--patchrisk_min_direction_cosine",
        type=float,
        default=0.35,
        help="Minimum signed residual coefficient cosine for adding a patchrisk neighbor.",
    )
    parser.add_argument(
        "--patchrisk_min_policy_gain",
        type=float,
        default=0.0,
        help="Minimum train-only policy-val relative gain for patchrisk neighbors.",
    )
    parser.add_argument(
        "--patchrisk_min_policy_samples",
        type=int,
        default=8,
        help="Minimum train-only policy-val samples for patchrisk neighbors.",
    )
    parser.add_argument(
        "--patchrisk_max_tail_risk",
        type=float,
        default=0.85,
        help="Maximum per-face georisk tail risk for patchrisk neighbors.",
    )
    parser.add_argument("--candidate_prefix", default="facelocal_coupled_v1")
    parser.add_argument("--phasej_test_method", default=DEFAULT_PHASEJ_TEST_METHOD)
    parser.add_argument("--phasej_trainval_method", default=DEFAULT_PHASEJ_TRAINVAL_METHOD)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=0.0)
    parser.add_argument(
        "--selector_balanced_ssim_weight",
        type=float,
        default=20.0,
        help="SSIM weight for selector train-val per-view balanced tail diagnostics.",
    )
    parser.add_argument(
        "--selector_balanced_lpips_weight",
        type=float,
        default=20.0,
        help="LPIPS weight for selector train-val per-view balanced tail diagnostics.",
    )
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument(
        "--selector_min_trainval_balanced_delta",
        type=float,
        default=0.0,
        help="Minimum train-val balanced delta for outer-loop promotion after the inner gate accepts.",
    )
    parser.add_argument(
        "--selector_min_trainval_psnr_gain",
        type=float,
        default=0.0,
        help="Minimum train-val PSNR gain for outer-loop promotion after the inner gate accepts.",
    )
    parser.add_argument(
        "--selector_max_trainval_ssim_regression",
        type=float,
        default=5e-5,
        help="Maximum train-val SSIM regression for outer-loop promotion after the inner gate accepts.",
    )
    parser.add_argument(
        "--selector_max_trainval_lpips_regression",
        type=float,
        default=1.5e-4,
        help="Maximum train-val LPIPS regression for outer-loop promotion after the inner gate accepts.",
    )
    parser.add_argument(
        "--selector_enable_tail_stable_promotion",
        action="store_true",
        help="Allow a lower train-val mean threshold when train-val per-view tails are stable.",
    )
    parser.add_argument("--selector_tail_min_trainval_balanced_delta", type=float, default=1.8e-5)
    parser.add_argument("--selector_tail_max_psnr_negative_fraction", type=float, default=0.20)
    parser.add_argument("--selector_tail_max_balanced_negative_fraction", type=float, default=0.40)
    parser.add_argument("--selector_tail_max_worst_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--selector_tail_cvar_fraction", type=float, default=0.20)
    parser.add_argument("--selector_tail_max_balanced_cvar_loss", type=float, default=math.inf)
    parser.add_argument("--selector_tail_min_mean_to_cvar_ratio", type=float, default=0.0)
    parser.add_argument("--selector_tail_max_lpips_positive_fraction", type=float, default=1.0)
    parser.add_argument(
        "--selector_enable_region_stable_promotion",
        action="store_true",
        help=(
            "Allow promotion when the full-frame train-val gate is non-regressive and the "
            "train-only render-region filter certifies a strong local ROI improvement."
        ),
    )
    parser.add_argument("--selector_region_min_trainval_balanced_delta", type=float, default=0.0)
    parser.add_argument("--selector_region_min_mean_core_balanced_delta", type=float, default=0.01)
    parser.add_argument("--selector_region_min_mean_delta_psnr", type=float, default=0.001)
    parser.add_argument("--selector_region_min_changed_fraction", type=float, default=0.50)
    parser.add_argument("--selector_region_max_negative_core_balanced_fraction", type=float, default=0.35)
    parser.add_argument("--selector_region_max_context_mse_regression", type=float, default=1.0e-6)
    parser.add_argument(
        "--selector_fit_plan_alphas",
        action="store_true",
        help=(
            "Before each materialization trial, fit train-only per-face alpha multipliers for "
            "the selected plan rows and pass them to the materializer."
        ),
    )
    parser.add_argument(
        "--selector_strict_fit_plan_alphas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow strict PatchCert full-plan replay to fit train-only per-face alpha shrink "
            "multipliers. This keeps the face set fixed and requires a render-trust certificate."
        ),
    )
    parser.add_argument(
        "--selector_allow_uncertified_plan",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicit legacy-ablation escape hatch for old face-local plans that lack strict "
            "PatchCert carrier metadata. When enabled, both alpha refit and materialization "
            "are labeled as uncertified plan replay."
        ),
    )
    parser.add_argument(
        "--selector_strict_cert_replay_mode",
        choices=("full_plan", "reject"),
        default="full_plan",
        help=(
            "How to handle strict PatchCert carrier plans when uncertified replay is disabled. "
            "full_plan preserves certification by replaying the complete plan with scale 1 and "
            "without alpha refit; reject records a Phase-J fallback instead of running unsafe "
            "subset/scale/alpha trials."
        ),
    )
    parser.add_argument("--ela_alpha_holdout_safe_zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ela_alpha_risk_tail_fraction", type=float, default=0.20)
    parser.add_argument("--ela_alpha_max_negative_gain_fraction", type=float, default=1.0)
    parser.add_argument("--ela_alpha_min_tail_gain", type=float, default=-math.inf)
    parser.add_argument("--ela_alpha_view_tail_scale_grid", default="")
    parser.add_argument("--ela_alpha_view_tail_cvar_fraction", type=float, default=0.25)
    parser.add_argument("--ela_alpha_view_tail_min_gain", type=float, default=-math.inf)
    parser.add_argument("--ela_alpha_view_tail_max_negative_fraction", type=float, default=1.0)
    parser.add_argument("--ela_alpha_region_risk_enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ela_alpha_region_risk_json_template", default="")
    parser.add_argument("--ela_alpha_region_risk_objective_bad_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ela_alpha_region_risk_objective_max_balanced_delta", type=float, default=0.0)
    parser.add_argument("--ela_alpha_region_risk_objective_max_delta_ssim", type=float, default=0.0)
    parser.add_argument("--ela_alpha_region_risk_objective_min_delta_lpips", type=float, default=0.0)
    parser.add_argument("--ela_alpha_region_risk_min_tail_gain", type=float, default=0.0)
    parser.add_argument("--ela_alpha_region_risk_max_negative_fraction", type=float, default=1.0)
    parser.add_argument("--ela_alpha_region_risk_min_regions", type=int, default=1)
    parser.add_argument("--ela_local_trust_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ela_local_trust_min_supports", type=int, default=2)
    parser.add_argument("--ela_local_trust_max_residual_std", type=float, default=-1.0)
    parser.add_argument("--ela_local_trust_min_agreement", type=float, default=0.0)
    parser.add_argument("--ela_local_trust_agreement_scale", type=float, default=0.04)
    parser.add_argument("--ela_local_trust_confidence_quantile", type=float, default=-1.0)
    parser.add_argument("--ela_local_trust_min_confidence", type=float, default=0.0)
    parser.add_argument("--ela_local_trust_mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--ela_local_trust_min_weight", type=float, default=0.0)
    parser.add_argument(
        "--selector_strict_replay_scales",
        default="1.0",
        help=(
            "Comma/space separated full-carrier scales for strict PatchCert plans. "
            "Only complete-plan replay is used; no subset or alpha refit is enabled."
        ),
    )
    parser.add_argument(
        "--selector_strict_adaptive_scale_policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For strict PatchCert plans, expand the replay scale list with a fixed train-only "
            "risk policy derived from candidate per-view certificate tails."
        ),
    )
    parser.add_argument("--selector_strict_adaptive_scale_min", type=float, default=0.10)
    parser.add_argument("--selector_strict_adaptive_scale_max_extra", type=int, default=5)
    parser.add_argument("--selector_strict_adaptive_scale_tail_fraction", type=float, default=0.30)
    parser.add_argument("--selector_alpha_max", type=float, default=1.0)
    parser.add_argument("--selector_alpha_steps", type=int, default=450)
    parser.add_argument("--selector_alpha_lr", type=float, default=0.06)
    parser.add_argument("--selector_alpha_max_total_samples", type=int, default=240000)
    parser.add_argument("--selector_alpha_device", default="cuda")
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phase_s_facelocal_coupled_selector_v1_20260513")
    parser.add_argument("--skip_failed_views", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if float(args.risk_pair_lambda) < 0.0:
        parser.error("--risk_pair_lambda must be non-negative")
    if float(args.georisk_pair_lambda) < 0.0:
        args.georisk_pair_lambda = float(args.risk_pair_lambda)
    if float(args.georisk_pair_lambda) < 0.0:
        parser.error("--georisk_pair_lambda must be non-negative")
    for name in ("georisk_geometry_lambda", "georisk_tail_lambda", "georisk_error_lambda"):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name} must be non-negative")
    if not 0.0 < float(args.georisk_tail_fraction) <= 1.0:
        parser.error("--georisk_tail_fraction must be in (0, 1]")
    if int(args.patchrisk_rings) < 0:
        parser.error("--patchrisk_rings must be non-negative")
    if int(args.patchrisk_max_patch_faces) <= 0:
        parser.error("--patchrisk_max_patch_faces must be positive")
    if int(args.patchrisk_max_total_faces) <= 0:
        parser.error("--patchrisk_max_total_faces must be positive")
    if int(args.patchrisk_centroid_candidates_per_seed) < 0:
        parser.error("--patchrisk_centroid_candidates_per_seed must be non-negative")
    if not -1.0 <= float(args.patchrisk_min_direction_cosine) <= 1.0:
        parser.error("--patchrisk_min_direction_cosine must be in [-1, 1]")
    if int(args.patchrisk_min_policy_samples) < 0:
        parser.error("--patchrisk_min_policy_samples must be non-negative")
    if not 0.0 <= float(args.patchrisk_max_tail_risk) <= 1.0:
        parser.error("--patchrisk_max_tail_risk must be in [0, 1]")
    if not 0.0 < float(args.selector_tail_cvar_fraction) <= 1.0:
        parser.error("--selector_tail_cvar_fraction must be in (0, 1]")
    if float(args.selector_tail_max_balanced_cvar_loss) < 0.0:
        parser.error("--selector_tail_max_balanced_cvar_loss must be non-negative")
    if float(args.selector_tail_min_mean_to_cvar_ratio) < 0.0:
        parser.error("--selector_tail_min_mean_to_cvar_ratio must be non-negative")
    if not 0.0 <= float(args.selector_tail_max_lpips_positive_fraction) <= 1.0:
        parser.error("--selector_tail_max_lpips_positive_fraction must be in [0, 1]")
    if float(args.selector_region_min_mean_core_balanced_delta) < 0.0:
        parser.error("--selector_region_min_mean_core_balanced_delta must be non-negative")
    if float(args.selector_region_min_mean_delta_psnr) < 0.0:
        parser.error("--selector_region_min_mean_delta_psnr must be non-negative")
    if not 0.0 <= float(args.selector_region_min_changed_fraction) <= 1.0:
        parser.error("--selector_region_min_changed_fraction must be in [0, 1]")
    if not 0.0 <= float(args.selector_region_max_negative_core_balanced_fraction) <= 1.0:
        parser.error("--selector_region_max_negative_core_balanced_fraction must be in [0, 1]")
    if (
        not math.isfinite(float(args.ela_alpha_risk_tail_fraction))
        or float(args.ela_alpha_risk_tail_fraction) <= 0.0
        or float(args.ela_alpha_risk_tail_fraction) > 1.0
    ):
        parser.error("--ela_alpha_risk_tail_fraction must be in (0, 1]")
    if (
        not math.isfinite(float(args.ela_alpha_max_negative_gain_fraction))
        or float(args.ela_alpha_max_negative_gain_fraction) < 0.0
        or float(args.ela_alpha_max_negative_gain_fraction) > 1.0
    ):
        parser.error("--ela_alpha_max_negative_gain_fraction must be in [0, 1]")
    if (
        not math.isfinite(float(args.ela_alpha_region_risk_max_negative_fraction))
        or float(args.ela_alpha_region_risk_max_negative_fraction) < 0.0
        or float(args.ela_alpha_region_risk_max_negative_fraction) > 1.0
    ):
        parser.error("--ela_alpha_region_risk_max_negative_fraction must be in [0, 1]")
    if not math.isfinite(float(args.ela_alpha_region_risk_min_tail_gain)):
        parser.error("--ela_alpha_region_risk_min_tail_gain must be finite")
    for name in (
        "ela_alpha_region_risk_objective_max_balanced_delta",
        "ela_alpha_region_risk_objective_max_delta_ssim",
        "ela_alpha_region_risk_objective_min_delta_lpips",
    ):
        if not math.isfinite(float(getattr(args, name))):
            parser.error(f"--{name} must be finite")
    if int(args.ela_alpha_region_risk_min_regions) <= 0:
        parser.error("--ela_alpha_region_risk_min_regions must be > 0")
    if int(args.ela_local_trust_min_supports) < 0:
        parser.error("--ela_local_trust_min_supports must be >= 0")
    if not math.isfinite(float(args.ela_local_trust_max_residual_std)):
        parser.error("--ela_local_trust_max_residual_std must be finite; use a negative value to disable")
    if not 0.0 <= float(args.ela_local_trust_min_agreement) <= 1.0:
        parser.error("--ela_local_trust_min_agreement must be in [0, 1]")
    if float(args.ela_local_trust_agreement_scale) <= 0.0:
        parser.error("--ela_local_trust_agreement_scale must be > 0")
    if not -1.0 <= float(args.ela_local_trust_confidence_quantile) < 1.0:
        parser.error("--ela_local_trust_confidence_quantile must be in [-1, 1)")
    if float(args.ela_local_trust_min_confidence) < 0.0:
        parser.error("--ela_local_trust_min_confidence must be >= 0")
    if float(args.ela_local_trust_min_weight) < 0.0:
        parser.error("--ela_local_trust_min_weight must be >= 0")
    if float(args.selector_region_max_context_mse_regression) < 0.0:
        parser.error("--selector_region_max_context_mse_regression must be non-negative")
    try:
        parse_scale_list(args.selector_strict_replay_scales)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 < float(args.selector_strict_adaptive_scale_min) <= 1.0:
        parser.error("--selector_strict_adaptive_scale_min must be in (0, 1]")
    if int(args.selector_strict_adaptive_scale_max_extra) < 0:
        parser.error("--selector_strict_adaptive_scale_max_extra must be non-negative")
    if not 0.0 < float(args.selector_strict_adaptive_scale_tail_fraction) <= 1.0:
        parser.error("--selector_strict_adaptive_scale_tail_fraction must be in (0, 1]")
    return args


def scene_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]


def safe_scale(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def parse_trial_specs(raw: str) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for item in str(raw).replace(";", ",").split(","):
        token = item.strip()
        if not token:
            continue
        match = re.fullmatch(r"(top|score|risk|georisk|patchrisk)(\d+)x([0-9]*\.?[0-9]+)", token)
        if not match:
            raise ValueError(f"invalid trial spec: {token}")
        mode = match.group(1)
        count = int(match.group(2))
        scale = float(match.group(3))
        if count <= 0:
            raise ValueError(f"trial count must be positive: {token}")
        label = f"{mode}{count}_s{safe_scale(scale)}"
        specs.append(TrialSpec(label=label, mode=mode, count=count, scale=scale))
    if not specs:
        raise ValueError("no trial specs")
    return specs


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def is_strict_patchcert_plan(plan: dict[str, Any]) -> bool:
    if bool(plan.get("strict_patchcert_carrier")):
        return True
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        return False
    for row in candidates:
        if not isinstance(row, dict):
            continue
        if str(row.get("certification_source", "")).startswith("strict_"):
            return True
        if isinstance(row.get("carrier_holdout_certificate"), dict):
            return True
    return False


def parse_scale_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).replace(",", " ").split():
        if not item.strip():
            continue
        value = float(item)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise ValueError(f"strict replay scales must be in (0, 1]: {item}")
        values.append(float(value))
    if not values:
        raise ValueError("no strict replay scales")
    out: list[float] = []
    for value in values:
        if not any(abs(value - existing) < 1.0e-12 for existing in out):
            out.append(value)
    return out


def strict_full_plan_spec(candidates: list[dict[str, Any]], scale: float = 1.0) -> TrialSpec:
    count = max(1, len(candidates))
    return TrialSpec(label=f"strictfull_s{safe_scale(scale)}", mode="strictfull", count=count, scale=float(scale))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_block(payload: dict[str, Any] | None) -> dict[str, float]:
    payload = payload or {}
    out: dict[str, float] = {}
    for key in METRICS:
        try:
            value = float(payload.get(key))
        except Exception:
            value = math.nan
        out[key] = value if math.isfinite(value) else math.nan
    return out


def num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def train_certificate_score(row: dict[str, Any]) -> float:
    """Score a plan candidate using train-only certificate fields."""
    rel_gain = max(num(nested(row, "policy_val_proxy", "relative_gain")), 0.0)
    samples = max(num(nested(row, "policy_val_proxy", "samples")), 1.0)
    shrink = max(num(nested(row, "validation_shrink", "scale"), 1.0), 0.0)
    consensus = max(num(nested(row, "face_view_consensus", "consensus"), 0.0), 0.0)
    view_fraction = max(num(nested(row, "face_view_gain_certificate", "beneficial_fraction"), 0.0), 0.0)
    min_view_gain = max(num(nested(row, "face_view_gain_certificate", "min_relative_gain"), 0.0), 0.0)
    consistency = max(num(nested(row, "face_stats", "consistency"), 0.0), 0.0)
    pixels = max(num(nested(row, "face_stats", "pixel_count"), 1.0), 1.0)
    view_hits = max(num(nested(row, "face_stats", "view_hits"), 1.0), 1.0)
    support = math.log1p(samples) * math.log1p(pixels) * math.sqrt(view_hits)
    return float(rel_gain * shrink * consensus * view_fraction * (0.5 + 0.5 * min_view_gain) * consistency * support)


def view_support(row: dict[str, Any]) -> dict[str, float]:
    support: dict[str, float] = {}
    cert_views = nested(row, "face_view_gain_certificate", "views")
    if isinstance(cert_views, list):
        for item in cert_views:
            if not isinstance(item, dict):
                continue
            name = str(item.get("view_name", ""))
            if not name:
                continue
            support[name] = support.get(name, 0.0) + max(num(item.get("samples"), 1.0), 1.0)
    if not support:
        names = nested(row, "face_view_consensus", "view_names")
        counts = nested(row, "face_view_consensus", "view_sample_counts")
        if isinstance(names, list):
            for idx, name in enumerate(names):
                count = 1.0
                if isinstance(counts, list) and idx < len(counts):
                    count = max(num(counts[idx], 1.0), 1.0)
                support[str(name)] = support.get(str(name), 0.0) + count
    total = sum(support.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in support.items()}


def coeff_direction(row: dict[str, Any]) -> list[float]:
    coeff = row.get("delta_coeff")
    values: list[float] = []
    if isinstance(coeff, list):
        stack = [coeff]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            else:
                values.append(num(item, 0.0))
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return []
    return [value / norm for value in values]


def cosine_abs(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    count = min(len(a), len(b))
    return abs(sum(a[idx] * b[idx] for idx in range(count)))


def cosine_signed(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    count = min(len(a), len(b))
    return float(max(min(sum(a[idx] * b[idx] for idx in range(count)), 1.0), -1.0))


def view_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    numerator = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    denominator = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator > 0 else 0.0


def pair_risk(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = view_overlap(view_support(a), view_support(b))
    direction = cosine_abs(coeff_direction(a), coeff_direction(b))
    return float(overlap * (0.5 + 0.5 * direction))


def view_relative_gains(row: dict[str, Any]) -> list[float]:
    out: list[float] = []
    cert_views = nested(row, "face_view_gain_certificate", "views")
    if not isinstance(cert_views, list):
        return out
    for item in cert_views:
        if not isinstance(item, dict):
            continue
        value = num(item.get("relative_gain"), math.nan)
        if math.isfinite(value):
            out.append(float(value))
    return out


def cvar_tail(values: list[float], fraction: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    count = max(1, int(math.ceil(len(finite) * max(min(float(fraction), 1.0), 1e-6))))
    return float(sum(finite[:count]) / count)


def georisk_tail_info(row: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    gains = view_relative_gains(row)
    if not gains:
        return {
            "georisk_view_count": 0.0,
            "georisk_tail_cvar_relative_gain": math.nan,
            "georisk_tail_min_relative_gain": math.nan,
            "georisk_tail_negative_fraction": 1.0,
            "georisk_tail_low_gain_fraction": 1.0,
            "georisk_tail_risk": 1.0,
        }
    cvar = cvar_tail(gains, float(args.georisk_tail_fraction))
    min_gain = min(gains)
    negative_fraction = sum(1 for value in gains if value < 0.0) / len(gains)
    low_fraction = sum(1 for value in gains if value < float(args.georisk_min_view_gain)) / len(gains)
    min_gain_deficit = max(0.0, float(args.georisk_min_view_gain) - min_gain) / max(float(args.georisk_min_view_gain), 1e-8)
    cvar_deficit = max(0.0, float(args.georisk_min_view_gain) - cvar) / max(float(args.georisk_min_view_gain), 1e-8)
    support_penalty = 1.0 / max(float(len(gains)), 1.0)
    risk = min(1.0, 0.35 * cvar_deficit + 0.35 * negative_fraction + 0.20 * low_fraction + 0.10 * support_penalty + 0.15 * min_gain_deficit)
    return {
        "georisk_view_count": float(len(gains)),
        "georisk_tail_cvar_relative_gain": float(cvar),
        "georisk_tail_min_relative_gain": float(min_gain),
        "georisk_tail_negative_fraction": float(negative_fraction),
        "georisk_tail_low_gain_fraction": float(low_fraction),
        "georisk_tail_risk": float(risk),
    }


def strict_adaptive_scale_policy(
    base_scales: list[float],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, Any]]:
    """Derive certification-preserving full-plan scales from train-only tail risk."""
    min_scale = float(args.selector_strict_adaptive_scale_min)
    base = [float(value) for value in base_scales if float(value) >= min_scale - 1.0e-12]
    tail_infos = [georisk_tail_info(row, args) for row in candidates]
    risks = [float(item.get("georisk_tail_risk", 1.0)) for item in tail_infos]
    cvars = []
    for item in tail_infos:
        cvar_value = num(item.get("georisk_tail_cvar_relative_gain"), math.nan)
        if math.isfinite(cvar_value):
            cvars.append(float(cvar_value))
    view_counts = [float(item.get("georisk_view_count", 0.0)) for item in tail_infos]
    if risks:
        risk_mean = float(mean(risks))
        risk_cvar = float(cvar_tail(risks, float(args.selector_strict_adaptive_scale_tail_fraction)))
        risk_max = float(max(risks))
    else:
        risk_mean = 1.0
        risk_cvar = 1.0
        risk_max = 1.0
    if cvars:
        cvar_mean = float(mean(cvars))
        cvar_tail_value = float(cvar_tail(cvars, float(args.selector_strict_adaptive_scale_tail_fraction)))
    else:
        cvar_mean = math.nan
        cvar_tail_value = math.nan
    low_support_fraction = (
        float(sum(1 for value in view_counts if value < 3.0) / len(view_counts)) if view_counts else 1.0
    )
    severity = max(risk_mean, risk_cvar, 0.35 + 0.65 * low_support_fraction)
    if severity >= 0.72:
        anchors = [1.0, 0.85, 0.70, 0.55, 0.42, 0.32, 0.24, 0.18, 0.12]
        policy_band = "high_tail_risk"
    elif severity >= 0.48:
        anchors = [1.0, 0.88, 0.76, 0.64, 0.52, 0.40, 0.30, 0.22]
        policy_band = "medium_tail_risk"
    else:
        anchors = [1.0, 0.90, 0.80, 0.68, 0.56, 0.44]
        policy_band = "low_tail_risk"
    requested = list(base)
    max_extra = int(args.selector_strict_adaptive_scale_max_extra)
    extra: list[float] = []
    for value in anchors:
        if value < min_scale - 1.0e-12:
            continue
        if any(abs(value - existing) < 1.0e-9 for existing in requested + extra):
            continue
        extra.append(float(value))
        if len(extra) >= max_extra:
            break
    merged: list[float] = []
    for value in sorted(requested + extra, reverse=True):
        if value < min_scale - 1.0e-12:
            continue
        if not any(abs(value - existing) < 1.0e-9 for existing in merged):
            merged.append(float(value))
    audit = {
        "enabled": True,
        "policy": "strict_patchcert_train_only_tail_risk_scale_v1",
        "selection_uses_test": False,
        "candidate_count": int(len(candidates)),
        "base_scales": [float(value) for value in base_scales],
        "added_scales": [float(value) for value in extra],
        "final_scales": [float(value) for value in merged],
        "min_scale": float(min_scale),
        "max_extra": int(max_extra),
        "tail_fraction": float(args.selector_strict_adaptive_scale_tail_fraction),
        "risk_mean": float(risk_mean),
        "risk_cvar": float(risk_cvar),
        "risk_max": float(risk_max),
        "tail_cvar_relative_gain_mean": float(cvar_mean) if math.isfinite(cvar_mean) else None,
        "tail_cvar_relative_gain_tail": float(cvar_tail_value) if math.isfinite(cvar_tail_value) else None,
        "low_support_fraction": float(low_support_fraction),
        "severity": float(severity),
        "policy_band": policy_band,
    }
    return merged, audit


def local_error_concentration(row: dict[str, Any]) -> float:
    mean_l1 = max(num(nested(row, "face_stats", "mean_l1_error"), 0.0), 0.0)
    pixels = max(num(nested(row, "face_stats", "pixel_count"), 1.0), 1.0)
    view_hits = max(num(nested(row, "face_stats", "view_hits"), 1.0), 1.0)
    consistency = max(num(nested(row, "face_stats", "consistency"), 0.0), 0.0)
    return float(mean_l1 * math.log1p(pixels) * math.sqrt(view_hits) * (0.25 + 0.75 * consistency))


def face_adjacency_geometry(
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    enabled: bool,
) -> tuple[dict[int, set[int]], dict[int, tuple[float, float, float]], dict[str, Any]]:
    face_ids = sorted({int(row.get("face_id", -1)) for row in candidates if int(row.get("face_id", -1)) >= 0})
    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "available": False,
        "candidate_faces": int(len(face_ids)),
        "edge_count": 0,
        "source_model": str(plan.get("source_model", "")),
        "iteration": int(plan.get("iteration", 0) or 0),
    }
    adjacency: dict[int, set[int]] = {face_id: set() for face_id in face_ids}
    centers: dict[int, tuple[float, float, float]] = {}
    if not enabled or not face_ids:
        return adjacency, centers, meta
    source_model = str(plan.get("source_model", "")).strip()
    if not source_model:
        meta["error"] = "missing_source_model"
        return adjacency, centers, meta
    try:
        import torch
        from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path

        iteration = int(plan.get("iteration", 26000) or 26000)
        state = torch.load(checkpoint_path(Path(source_model), iteration), map_location="cpu")
        faces = state["_triangle_indices"].detach().cpu().long()
        vertices = state.get("triangles_points")
        if vertices is not None:
            vertices = vertices.detach().cpu().float()
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return adjacency, centers, meta

    vertex_to_faces: dict[int, list[int]] = {}
    valid_count = 0
    for face_id in face_ids:
        if face_id < 0 or face_id >= int(faces.shape[0]):
            continue
        valid_count += 1
        if vertices is not None:
            center = vertices[faces[face_id]].mean(dim=0).tolist()
            centers[int(face_id)] = (float(center[0]), float(center[1]), float(center[2]))
        for vertex_id in faces[face_id].tolist():
            vertex_to_faces.setdefault(int(vertex_id), []).append(face_id)
    for incident in vertex_to_faces.values():
        if len(incident) < 2:
            continue
        for idx, left in enumerate(incident):
            for right in incident[idx + 1 :]:
                if left == right:
                    continue
                adjacency[int(left)].add(int(right))
                adjacency[int(right)].add(int(left))
    edge_count = sum(len(value) for value in adjacency.values()) // 2
    meta.update(
        {
            "available": True,
            "valid_candidate_faces": int(valid_count),
            "edge_count": int(edge_count),
            "mean_degree": float(sum(len(value) for value in adjacency.values()) / max(len(adjacency), 1)),
            "center_count": int(len(centers)),
        }
    )
    return adjacency, centers, meta


def risk_greedy_rows(candidates: list[dict[str, Any]], count: int, *, pair_lambda: float) -> list[dict[str, Any]]:
    pool = sorted(candidates, key=train_certificate_score, reverse=True)
    selected: list[dict[str, Any]] = []
    while pool and len(selected) < count:
        best_idx = 0
        best_score = -math.inf
        selected_views = set().union(*(set(view_support(item)) for item in selected)) if selected else set()
        for idx, row in enumerate(pool):
            base = train_certificate_score(row)
            redundancy = max((pair_risk(row, other) for other in selected), default=0.0)
            coverage_bonus = 0.05 * len(set(view_support(row)) - selected_views) if selected else 0.0
            adjusted = base * max(0.05, 1.0 - float(pair_lambda) * redundancy) * (1.0 + coverage_bonus)
            if adjusted > best_score:
                best_score = adjusted
                best_idx = idx
        selected.append(pool.pop(best_idx))
    return selected


def georisk_adjusted_score(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    face_adjacency: dict[int, set[int]],
    args: argparse.Namespace,
    max_concentration: float,
) -> dict[str, float]:
    base = train_certificate_score(row)
    fid = int(row.get("face_id", -1))
    selected_ids = {int(item.get("face_id", -1)) for item in selected}
    selected_views = set().union(*(set(view_support(item)) for item in selected)) if selected else set()
    new_supported_views = len(set(view_support(row)) - selected_views) if selected else 0
    pair_redundancy = max((pair_risk(row, other) for other in selected), default=0.0)
    adjacent_selected = sorted(face_adjacency.get(fid, set()) & selected_ids)
    adjacent_rows = [item for item in selected if int(item.get("face_id", -1)) in set(adjacent_selected)]
    adjacent_pair = max((pair_risk(row, other) for other in adjacent_rows), default=0.0)
    adjacent_fraction = float(len(adjacent_selected) / max(len(selected), 1)) if selected else 0.0
    geometry_redundancy = max(adjacent_pair, min(1.0, adjacent_fraction))
    tail = georisk_tail_info(row, args)
    concentration = local_error_concentration(row)
    concentration_norm = concentration / max(float(max_concentration), 1e-8)
    pair_factor = max(0.05, 1.0 - float(args.georisk_pair_lambda) * pair_redundancy)
    geometry_factor = max(0.05, 1.0 - float(args.georisk_geometry_lambda) * geometry_redundancy)
    tail_factor = max(0.05, 1.0 - float(args.georisk_tail_lambda) * float(tail["georisk_tail_risk"]))
    coverage_bonus = 0.04 * float(new_supported_views)
    local_bonus = float(args.georisk_error_lambda) * min(1.0, concentration_norm)
    adjusted = base * pair_factor * geometry_factor * tail_factor * (1.0 + coverage_bonus + local_bonus)
    return {
        "train_certificate_score": float(base),
        "risk_max_pair_risk_to_previous": float(pair_redundancy),
        "risk_new_supported_view_count": float(new_supported_views),
        "risk_coverage_bonus": float(coverage_bonus),
        "risk_adjusted_selection_score": float(adjusted),
        "georisk_pair_factor": float(pair_factor),
        "georisk_geometry_adjacent_previous_count": float(len(adjacent_selected)),
        "georisk_geometry_redundancy": float(geometry_redundancy),
        "georisk_geometry_factor": float(geometry_factor),
        "georisk_tail_factor": float(tail_factor),
        "georisk_local_error_concentration": float(concentration),
        "georisk_local_error_concentration_norm": float(concentration_norm),
        "georisk_local_error_bonus": float(local_bonus),
        "georisk_adjusted_selection_score": float(adjusted),
        **tail,
    }


def georisk_greedy_rows(
    candidates: list[dict[str, Any]],
    count: int,
    *,
    face_adjacency: dict[int, set[int]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pool = sorted(candidates, key=train_certificate_score, reverse=True)
    selected: list[dict[str, Any]] = []
    max_concentration = max((local_error_concentration(row) for row in candidates), default=1.0)
    while pool and len(selected) < count:
        best_idx = 0
        best_score = -math.inf
        for idx, row in enumerate(pool):
            score = georisk_adjusted_score(
                row,
                selected,
                face_adjacency=face_adjacency,
                args=args,
                max_concentration=max_concentration,
            )["georisk_adjusted_selection_score"]
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(pool.pop(best_idx))
    return selected


def centroid_neighbors(
    face_id: int,
    *,
    face_centers: dict[int, tuple[float, float, float]],
    candidate_ids: set[int],
    max_candidates: int,
) -> list[int]:
    if int(max_candidates) <= 0 or int(face_id) not in face_centers:
        return []
    center = face_centers[int(face_id)]
    scored: list[tuple[float, int]] = []
    for other in candidate_ids:
        other_id = int(other)
        if other_id == int(face_id) or other_id not in face_centers:
            continue
        other_center = face_centers[other_id]
        dist2 = sum((float(center[idx]) - float(other_center[idx])) ** 2 for idx in range(3))
        scored.append((float(dist2), other_id))
    scored.sort()
    return [face for _, face in scored[: int(max_candidates)]]


def patchrisk_neighbor_ok(
    *,
    seed: dict[str, Any],
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    policy_gain = num(nested(row, "policy_val_proxy", "relative_gain"), math.nan)
    policy_samples = num(nested(row, "policy_val_proxy", "samples"), 0.0)
    signed_cosine = cosine_signed(coeff_direction(seed), coeff_direction(row))
    tail = georisk_tail_info(row, args)
    tail_risk = num(tail.get("georisk_tail_risk"), 1.0)
    if policy_samples < int(args.patchrisk_min_policy_samples):
        reasons.append("low_policy_samples")
    if not math.isfinite(policy_gain) or policy_gain < float(args.patchrisk_min_policy_gain):
        reasons.append("low_policy_gain")
    if signed_cosine < float(args.patchrisk_min_direction_cosine):
        reasons.append("low_direction_cosine")
    if tail_risk > float(args.patchrisk_max_tail_risk):
        reasons.append("high_tail_risk")
    return not reasons, reasons, {
        "policy_val_relative_gain": float(policy_gain) if math.isfinite(policy_gain) else math.nan,
        "policy_val_samples": float(policy_samples),
        "direction_cosine_to_seed": float(signed_cosine),
        "georisk_tail_risk": float(tail_risk),
    }


def patchrisk_expand_rows(
    candidates: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    *,
    face_adjacency: dict[int, set[int]],
    face_centers: dict[int, tuple[float, float, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_by_face = {int(row.get("face_id", -1)): row for row in candidates if int(row.get("face_id", -1)) >= 0}
    candidate_ids = set(row_by_face)
    assigned: set[int] = set()
    expanded: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    total_limit = max(int(args.patchrisk_max_total_faces), 1)
    max_patch = max(int(args.patchrisk_max_patch_faces), 1)
    rings = max(int(args.patchrisk_rings), 0)

    for seed in seed_rows:
        seed_id = int(seed.get("face_id", -1))
        if seed_id < 0 or seed_id not in row_by_face or seed_id in assigned:
            continue
        patch_ids: list[int] = [seed_id]
        assigned.add(seed_id)
        frontier = [seed_id]
        rejected_neighbors: list[dict[str, Any]] = []
        for _ in range(rings):
            next_frontier: list[int] = []
            for face_id in frontier:
                if len(patch_ids) >= max_patch or len(expanded) + len(patch_ids) >= total_limit:
                    break
                neighbor_ids: list[int] = []
                if str(args.patchrisk_neighbor_mode) in {"topology", "both"}:
                    neighbor_ids.extend(sorted(face_adjacency.get(int(face_id), set())))
                if str(args.patchrisk_neighbor_mode) in {"centroid", "both"}:
                    neighbor_ids.extend(
                        centroid_neighbors(
                            int(face_id),
                            face_centers=face_centers,
                            candidate_ids=candidate_ids,
                            max_candidates=int(args.patchrisk_centroid_candidates_per_seed),
                        )
                    )
                seen_neighbors: set[int] = set()
                for neighbor_id in neighbor_ids:
                    neighbor_id = int(neighbor_id)
                    if neighbor_id in seen_neighbors:
                        continue
                    seen_neighbors.add(neighbor_id)
                    if neighbor_id in assigned or neighbor_id in patch_ids or neighbor_id not in row_by_face:
                        continue
                    ok, reasons, audit = patchrisk_neighbor_ok(seed=seed, row=row_by_face[neighbor_id], args=args)
                    if not ok:
                        if len(rejected_neighbors) < 12:
                            rejected_neighbors.append({"face_id": neighbor_id, "reasons": reasons, **audit})
                        continue
                    patch_ids.append(neighbor_id)
                    assigned.add(neighbor_id)
                    next_frontier.append(neighbor_id)
                    if len(patch_ids) >= max_patch or len(expanded) + len(patch_ids) >= total_limit:
                        break
            frontier = next_frontier
            if not frontier or len(patch_ids) >= max_patch or len(expanded) + len(patch_ids) >= total_limit:
                break
        expanded.extend(row_by_face[face_id] for face_id in patch_ids)
        patches.append(
            {
                "seed_face": seed_id,
                "faces": patch_ids,
                "patch_size": int(len(patch_ids)),
                "rejected_neighbor_preview": rejected_neighbors,
            }
        )
        if len(expanded) >= total_limit:
            break

    meta = {
        "enabled": True,
        "seed_count": int(len(seed_rows)),
        "rings": int(rings),
        "max_patch_faces": int(max_patch),
        "max_total_faces": int(total_limit),
        "neighbor_mode": str(args.patchrisk_neighbor_mode),
        "centroid_candidates_per_seed": int(args.patchrisk_centroid_candidates_per_seed),
        "min_direction_cosine": float(args.patchrisk_min_direction_cosine),
        "min_policy_gain": float(args.patchrisk_min_policy_gain),
        "min_policy_samples": int(args.patchrisk_min_policy_samples),
        "max_tail_risk": float(args.patchrisk_max_tail_risk),
        "seed_face_ids": [int(row.get("face_id", -1)) for row in seed_rows],
        "expanded_face_ids": [int(row.get("face_id", -1)) for row in expanded],
        "expanded_face_count": int(len(expanded)),
        "patch_count": int(len(patches)),
        "mean_patch_size": float(mean([float(item["patch_size"]) for item in patches])) if patches else 0.0,
        "patches": patches,
        "uses_test": False,
    }
    return expanded, meta


def risk_adjusted_score(
    row: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    pair_lambda: float,
) -> dict[str, float]:
    base = train_certificate_score(row)
    redundancy = max((pair_risk(row, other) for other in selected), default=0.0)
    selected_views = set().union(*(set(view_support(item)) for item in selected)) if selected else set()
    new_supported_views = len(set(view_support(row)) - selected_views) if selected else 0
    coverage_bonus = 0.05 * new_supported_views
    adjusted = base * max(0.05, 1.0 - float(pair_lambda) * redundancy) * (1.0 + coverage_bonus)
    return {
        "train_certificate_score": float(base),
        "risk_max_pair_risk_to_previous": float(redundancy),
        "risk_new_supported_view_count": float(new_supported_views),
        "risk_coverage_bonus": float(coverage_bonus),
        "risk_adjusted_selection_score": float(adjusted),
    }


def face_score_entries(
    rows: list[dict[str, Any]],
    mode: str,
    *,
    pair_lambda: float,
    face_adjacency: dict[int, set[int]] | None = None,
    args: argparse.Namespace | None = None,
    max_concentration: float | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    concentration_norm_base = (
        float(max_concentration)
        if max_concentration is not None and math.isfinite(float(max_concentration))
        else max((local_error_concentration(row) for row in rows), default=1.0)
    )
    for row in rows:
        if mode == "risk":
            score_info = risk_adjusted_score(row, selected, pair_lambda=pair_lambda)
        elif mode == "georisk" and args is not None:
            score_info = georisk_adjusted_score(
                row,
                selected,
                face_adjacency=face_adjacency or {},
                args=args,
                max_concentration=concentration_norm_base,
            )
        else:
            score_info = {
                "train_certificate_score": train_certificate_score(row),
                "risk_max_pair_risk_to_previous": 0.0,
                "risk_new_supported_view_count": 0.0,
                "risk_coverage_bonus": 0.0,
                "risk_adjusted_selection_score": train_certificate_score(row),
            }
        entries.append(
            {
                "face_id": int(row["face_id"]),
                "rank": int(row.get("rank", -1)),
                "patchrisk_role": str(row.get("_patchrisk_role", "")),
                "patchrisk_seed_face": int(row.get("_patchrisk_seed_face", -1)),
                "patchrisk_patch_size": int(row.get("_patchrisk_patch_size", 0)),
                **score_info,
                "policy_val_relative_gain": num(nested(row, "policy_val_proxy", "relative_gain"), math.nan),
                "policy_val_samples": num(nested(row, "policy_val_proxy", "samples"), math.nan),
            }
        )
        selected.append(row)
    return entries


def selected_rows(
    candidates: list[dict[str, Any]],
    spec: TrialSpec,
    *,
    pair_lambda: float,
    face_adjacency: dict[int, set[int]] | None = None,
    args: argparse.Namespace | None = None,
) -> list[dict[str, Any]]:
    if spec.mode == "strictfull":
        return list(candidates)
    if spec.mode == "top":
        rows = list(candidates)
    elif spec.mode == "score":
        rows = sorted(candidates, key=train_certificate_score, reverse=True)
    elif spec.mode == "risk":
        return risk_greedy_rows(candidates, spec.count, pair_lambda=pair_lambda)
    elif spec.mode == "georisk":
        if args is None:
            raise ValueError("georisk mode requires selector args")
        return georisk_greedy_rows(
            candidates,
            spec.count,
            face_adjacency=face_adjacency or {},
            args=args,
        )
    elif spec.mode == "patchrisk":
        if args is None:
            raise ValueError("patchrisk mode requires selector args")
        return georisk_greedy_rows(
            candidates,
            spec.count,
            face_adjacency=face_adjacency or {},
            args=args,
        )
    else:
        raise ValueError(f"unknown trial mode: {spec.mode}")
    return rows[: spec.count]


def plan_path(template: str, scene: str) -> Path:
    return ROOT / template.format(scene=scene)


def run_command(cmd: list[str], *, gpu: int, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WANDB_MODE"] = "online"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        if dry_run:
            handle.write("[dry_run] skipped\n")
            return 0
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
        return int(proc.returncode)


def decision_path(root: Path, spec: TrialSpec, scene: str) -> Path:
    return root / "trials" / spec.label / "decisions" / f"{scene}_decision.json"


def alpha_json_path(root: Path, scene: str, spec: TrialSpec) -> Path:
    return root / scene / "alpha_refit" / f"{spec.label}_alpha_refit.json"


def render_trust_json_path(root: Path, scene: str, spec: TrialSpec) -> Path:
    return root / scene / "render_trust" / f"{spec.label}_render_trust.json"


def plan_materialize_alpha_path(plan: dict[str, Any], *, plan_file: Path) -> Path | None:
    raw = str(plan.get("render_cvar_aggregate_subset_materialize_alpha_json", "")).strip()
    if not raw:
        nested = plan.get("render_cvar_aggregate_subset")
        if isinstance(nested, dict):
            raw = str(nested.get("materialize_alpha_json", "")).strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates = [plan_file.parent / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def write_strict_shrink_render_trust(
    *,
    path: Path,
    plan_file: Path,
    scene: str,
    spec: TrialSpec,
    alpha_json: Path | None = None,
) -> None:
    scale = float(spec.scale)
    if not (math.isfinite(scale) and 0.0 < scale <= 1.0):
        raise ValueError(f"strict shrink render trust requires scale in (0, 1], got {scale}")
    payload = {
        "protocol": "strict_full_carrier_monotone_shrink_render_trust_v2",
        "accepted": True,
        "selection_uses_test": False,
        "scene": scene,
        "scale": scale,
        "accepted_scale": scale,
        "plan_path": str(plan_file),
        "plan_sha256": file_sha256(plan_file),
        "allow_alpha_shrink": bool(alpha_json is not None),
        "alpha_json": str(alpha_json) if alpha_json is not None else "",
        "alpha_json_sha256": file_sha256(alpha_json) if alpha_json is not None else "",
        "trust_basis": (
            "full strict PatchCert carrier replay with monotone coefficient shrink; "
            "no face subset or coefficient amplification is allowed. Promotion still requires "
            "the post-materialization train-val render gate."
        ),
        "decision_json": "",
    }
    write_json(path, payload)


def fit_alpha_command(args: argparse.Namespace, scene: str, spec: TrialSpec, face_ids: list[int], alpha_path: Path) -> list[str]:
    selector_mode = spec.mode if spec.mode in {"top", "score", "risk"} else "risk"
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_fit_facelocal_plan_alphas.py",
        "--candidate_plan",
        str(args.plan_template).format(scene=scene),
        "--evidence_dir",
        str(Path(args.evidence_root) / scene),
        "--output_json",
        str(alpha_path),
        "--face_ids",
        ",".join(str(fid) for fid in face_ids),
        "--selector_mode",
        selector_mode,
        "--selector_count",
        str(spec.count),
        "--risk_pair_lambda",
        str(args.risk_pair_lambda),
        "--alpha_max",
        str(args.selector_alpha_max),
        "--steps",
        str(args.selector_alpha_steps),
        "--lr",
        str(args.selector_alpha_lr),
        "--max_total_samples",
        str(args.selector_alpha_max_total_samples),
        "--device",
        str(args.selector_alpha_device),
    ]
    if bool(args.selector_allow_uncertified_plan):
        cmd.append("--allow_uncertified_plan_rows")
    return cmd


def build_trial_command(
    args: argparse.Namespace,
    scene: str,
    spec: TrialSpec,
    face_ids: list[int],
    *,
    alpha_json: Path | None = None,
    render_trust_json: Path | None = None,
    materialize_face_filter: bool = True,
    materialize_scale: float | None = None,
) -> list[str]:
    label = f"{args.candidate_prefix}_{spec.label}"
    output_root = Path(args.output_root) / "trials" / spec.label
    scale = float(spec.scale if materialize_scale is None else materialize_scale)
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_run_phasek_barycentric_gate_scene.py",
        "--scenes",
        scene,
        "--gpu",
        str(args.gpu),
        "--output_root",
        str(output_root),
        "--evidence_root",
        str(args.evidence_root),
        "--delta_operator",
        "facelocal_sh1",
        "--delta_uniform_barycentric",
        "--delta_sh_degree",
        "3",
        "--delta_facelocal_materialize_plan_in",
        str(args.plan_template),
        "--delta_facelocal_materialize_plan_limit",
        "0",
        "--delta_facelocal_materialize_plan_scale",
        str(scale),
        "--phasej_test_method",
        str(args.phasej_test_method),
        "--phasej_trainval_method",
        str(args.phasej_trainval_method),
        "--candidate_label",
        label,
        "--candidate_base_method",
        f"ours_26000_{label}_base",
        "--candidate_test_method",
        f"ours_26000_{label}_phasej_ela",
        "--candidate_trainval_method",
        f"ours_26000_{label}_trainval_gate",
        "--gate_min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--gate_max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--gate_max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--gate_min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        str(args.wandb_group),
        "--wandb_name",
        f"{label}_{scene}",
    ]
    if materialize_face_filter and face_ids:
        cmd.extend(
            [
                "--delta_facelocal_materialize_plan_face_ids",
                ",".join(str(fid) for fid in face_ids),
            ]
        )
    if bool(args.selector_allow_uncertified_plan):
        cmd.append("--delta_facelocal_materialize_allow_uncertified_plan")
    if alpha_json is not None:
        cmd.extend(["--delta_facelocal_materialize_plan_alpha_json", str(alpha_json)])
    if render_trust_json is not None:
        cmd.extend(["--delta_facelocal_materialize_plan_render_trust_json", str(render_trust_json)])
    if bool(args.ela_alpha_holdout_safe_zero):
        cmd.append("--ela_alpha_holdout_safe_zero")
    cmd.extend(
        [
            "--ela_alpha_risk_tail_fraction",
            str(float(args.ela_alpha_risk_tail_fraction)),
            "--ela_alpha_max_negative_gain_fraction",
            str(float(args.ela_alpha_max_negative_gain_fraction)),
            f"--ela_alpha_min_tail_gain={float(args.ela_alpha_min_tail_gain)}",
            "--ela_alpha_view_tail_scale_grid",
            str(args.ela_alpha_view_tail_scale_grid),
            "--ela_alpha_view_tail_cvar_fraction",
            str(float(args.ela_alpha_view_tail_cvar_fraction)),
            f"--ela_alpha_view_tail_min_gain={float(args.ela_alpha_view_tail_min_gain)}",
            "--ela_alpha_view_tail_max_negative_fraction",
            str(float(args.ela_alpha_view_tail_max_negative_fraction)),
        ]
    )
    if bool(args.ela_local_trust_gate):
        cmd.append("--ela_local_trust_gate")
    cmd.extend(
        [
            "--ela_local_trust_min_supports",
            str(int(args.ela_local_trust_min_supports)),
            "--ela_local_trust_max_residual_std",
            str(float(args.ela_local_trust_max_residual_std)),
            "--ela_local_trust_min_agreement",
            str(float(args.ela_local_trust_min_agreement)),
            "--ela_local_trust_agreement_scale",
            str(float(args.ela_local_trust_agreement_scale)),
            "--ela_local_trust_confidence_quantile",
            str(float(args.ela_local_trust_confidence_quantile)),
            "--ela_local_trust_min_confidence",
            str(float(args.ela_local_trust_min_confidence)),
            "--ela_local_trust_mode",
            str(args.ela_local_trust_mode),
            "--ela_local_trust_min_weight",
            str(float(args.ela_local_trust_min_weight)),
        ]
    )
    if bool(args.ela_alpha_region_risk_enable):
        risk_template = str(args.ela_alpha_region_risk_json_template).strip()
        if not risk_template:
            raise ValueError("--ela_alpha_region_risk_enable requires --ela_alpha_region_risk_json_template")
        risk_path = Path(risk_template.format(scene=scene))
        if not risk_path.is_file():
            raise FileNotFoundError(f"ELA alpha region-risk JSON not found for {scene}: {risk_path}")
        cmd.extend(
            [
                "--ela_alpha_region_risk_enable",
                "--ela_alpha_region_risk_json",
                str(risk_path),
                "--ela_alpha_region_risk_objective_bad_only"
                if bool(args.ela_alpha_region_risk_objective_bad_only)
                else "--no-ela_alpha_region_risk_objective_bad_only",
                "--ela_alpha_region_risk_objective_max_balanced_delta",
                str(float(args.ela_alpha_region_risk_objective_max_balanced_delta)),
                "--ela_alpha_region_risk_objective_max_delta_ssim",
                str(float(args.ela_alpha_region_risk_objective_max_delta_ssim)),
                "--ela_alpha_region_risk_objective_min_delta_lpips",
                str(float(args.ela_alpha_region_risk_objective_min_delta_lpips)),
                "--ela_alpha_region_risk_min_tail_gain",
                str(float(args.ela_alpha_region_risk_min_tail_gain)),
                "--ela_alpha_region_risk_max_negative_fraction",
                str(float(args.ela_alpha_region_risk_max_negative_fraction)),
                "--ela_alpha_region_risk_min_regions",
                str(int(args.ela_alpha_region_risk_min_regions)),
            ]
        )
    if bool(args.skip_failed_views):
        cmd.append("--skip_failed_views")
    if bool(args.force):
        cmd.append("--force")
    return cmd


def render_region_promotion_summary(plan: dict[str, Any]) -> dict[str, Any]:
    """Aggregate train-only render-region filter evidence from a filtered plan."""
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        return {"present": False, "reason": "no_plan_candidates"}

    carriers: dict[str, dict[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, dict):
            continue
        decision = row.get("render_region_filter_decision")
        if not isinstance(decision, dict):
            continue
        stats = decision.get("render_region_stats")
        if not isinstance(stats, dict):
            continue
        carrier_id = str(
            decision.get("plan_carrier_id")
            or row.get("carrier_id")
            or row.get("carrier_seed_face")
            or row.get("face_id")
            or len(carriers)
        )
        if carrier_id in carriers:
            continue
        carriers[carrier_id] = {
            "carrier_id": carrier_id,
            "accepted": bool(decision.get("accepted", False)),
            "decision_reasons": decision.get("decision_reasons", []),
            "decision_notes": decision.get("decision_notes", []),
            "tail_safe_shrink_applied": bool(decision.get("tail_safe_shrink_applied", False)),
            "tail_safe_shrink_scale": num(decision.get("tail_safe_shrink_scale"), math.nan),
            "stats": {
                "regions": num(stats.get("regions"), 0.0),
                "changed_regions": num(stats.get("changed_regions"), 0.0),
                "changed_fraction": num(stats.get("changed_fraction"), 0.0),
                "mean_core_balanced_delta": num(stats.get("mean_core_balanced_delta"), math.nan),
                "mean_delta_core_psnr": num(stats.get("mean_delta_core_psnr"), math.nan),
                "mean_delta_core_ssim": num(stats.get("mean_delta_core_ssim"), math.nan),
                "mean_delta_core_lpips": num(stats.get("mean_delta_core_lpips"), math.nan),
                "tail_core_balanced_delta": num(stats.get("tail_core_balanced_delta"), math.nan),
                "negative_core_balanced_fraction": num(stats.get("negative_core_balanced_fraction"), math.inf),
                "max_context_mse_regression": num(stats.get("max_context_mse_regression"), math.inf),
                "mean_crop_abs_diff": num(stats.get("mean_crop_abs_diff"), 0.0),
                "max_crop_abs_diff": num(stats.get("max_crop_abs_diff"), 0.0),
            },
        }

    accepted = [row for row in carriers.values() if bool(row.get("accepted", False))]
    if not carriers:
        return {"present": False, "reason": "missing_render_region_filter_decision"}
    if not accepted:
        return {
            "present": True,
            "accepted_carriers": 0,
            "carrier_count": len(carriers),
            "carriers": list(carriers.values()),
        }

    def stats_values(key: str) -> list[float]:
        values: list[float] = []
        for row in accepted:
            value = num((row.get("stats") or {}).get(key), math.nan)
            if math.isfinite(value):
                values.append(float(value))
        return values

    changed_fractions = stats_values("changed_fraction")
    mean_balanced = stats_values("mean_core_balanced_delta")
    mean_psnr = stats_values("mean_delta_core_psnr")
    negative_fractions = stats_values("negative_core_balanced_fraction")
    context_regressions = stats_values("max_context_mse_regression")
    tail_values = stats_values("tail_core_balanced_delta")
    return {
        "present": True,
        "accepted_carriers": len(accepted),
        "carrier_count": len(carriers),
        "mean_changed_fraction": mean(changed_fractions),
        "min_changed_fraction": min(changed_fractions) if changed_fractions else math.nan,
        "mean_core_balanced_delta": mean(mean_balanced),
        "min_mean_core_balanced_delta": min(mean_balanced) if mean_balanced else math.nan,
        "mean_delta_core_psnr": mean(mean_psnr),
        "min_mean_delta_core_psnr": min(mean_psnr) if mean_psnr else math.nan,
        "max_negative_core_balanced_fraction": max(negative_fractions) if negative_fractions else math.inf,
        "max_context_mse_regression": max(context_regressions) if context_regressions else math.inf,
        "min_tail_core_balanced_delta": min(tail_values) if tail_values else math.nan,
        "tail_safe_shrink_carriers": sum(1 for row in accepted if bool(row.get("tail_safe_shrink_applied", False))),
        "carriers": accepted,
    }


def selector_pass(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    train = metric_block(row.get("trainval_delta"))
    balanced = num(row.get("trainval_balanced_delta"), -math.inf)
    if not bool(row.get("accepted", False)):
        reasons.append("inner_gate_rejected")
    if train["PSNR"] < float(args.selector_min_trainval_psnr_gain):
        reasons.append(f"selector_psnr_gain_below_{args.selector_min_trainval_psnr_gain:g}")
    if train["SSIM"] < -float(args.selector_max_trainval_ssim_regression):
        reasons.append(f"selector_ssim_regression_exceeds_{args.selector_max_trainval_ssim_regression:g}")
    if train["LPIPS"] > float(args.selector_max_trainval_lpips_regression):
        reasons.append(f"selector_lpips_regression_exceeds_{args.selector_max_trainval_lpips_regression:g}")
    if balanced < float(args.selector_min_trainval_balanced_delta):
        reasons.append(f"selector_balanced_delta_below_{args.selector_min_trainval_balanced_delta:g}")
    if not reasons:
        row["selector_pass_mode"] = "strict_mean"
        return True, reasons

    if bool(args.selector_enable_tail_stable_promotion) and bool(row.get("accepted", False)):
        tail = row.get("trainval_per_view_tail") if isinstance(row.get("trainval_per_view_tail"), dict) else {}
        tail_reasons: list[str] = []
        if train["PSNR"] < float(args.selector_min_trainval_psnr_gain):
            tail_reasons.append(f"tail_psnr_gain_below_{args.selector_min_trainval_psnr_gain:g}")
        if train["SSIM"] < -float(args.selector_max_trainval_ssim_regression):
            tail_reasons.append(f"tail_ssim_regression_exceeds_{args.selector_max_trainval_ssim_regression:g}")
        if train["LPIPS"] > float(args.selector_max_trainval_lpips_regression):
            tail_reasons.append(f"tail_lpips_regression_exceeds_{args.selector_max_trainval_lpips_regression:g}")
        if balanced < float(args.selector_tail_min_trainval_balanced_delta):
            tail_reasons.append(f"tail_balanced_delta_below_{args.selector_tail_min_trainval_balanced_delta:g}")
        if num(tail.get("view_count"), 0.0) <= 0:
            tail_reasons.append("tail_per_view_missing")
        if num(tail.get("psnr_negative_fraction"), math.inf) > float(args.selector_tail_max_psnr_negative_fraction):
            tail_reasons.append(f"tail_psnr_negative_fraction_exceeds_{args.selector_tail_max_psnr_negative_fraction:g}")
        if num(tail.get("balanced_negative_fraction"), math.inf) > float(args.selector_tail_max_balanced_negative_fraction):
            tail_reasons.append(f"tail_balanced_negative_fraction_exceeds_{args.selector_tail_max_balanced_negative_fraction:g}")
        if num(tail.get("worst_lpips_regression"), math.inf) > float(args.selector_tail_max_worst_lpips_regression):
            tail_reasons.append(f"tail_worst_lpips_regression_exceeds_{args.selector_tail_max_worst_lpips_regression:g}")
        if num(tail.get("lpips_positive_fraction"), math.inf) > float(args.selector_tail_max_lpips_positive_fraction):
            tail_reasons.append(f"tail_lpips_positive_fraction_exceeds_{args.selector_tail_max_lpips_positive_fraction:g}")
        if num(tail.get("balanced_cvar_loss"), math.inf) > float(args.selector_tail_max_balanced_cvar_loss):
            tail_reasons.append(f"tail_balanced_cvar_loss_exceeds_{args.selector_tail_max_balanced_cvar_loss:g}")
        if num(tail.get("mean_to_cvar_ratio"), math.inf) < float(args.selector_tail_min_mean_to_cvar_ratio):
            tail_reasons.append(f"tail_mean_to_cvar_ratio_below_{args.selector_tail_min_mean_to_cvar_ratio:g}")
        if not tail_reasons:
            row["selector_pass_mode"] = "tail_stable"
            return True, []
        row["selector_tail_reasons"] = tail_reasons

    if bool(args.selector_enable_region_stable_promotion) and bool(row.get("accepted", False)):
        region = row.get("selector_region_promotion")
        region = region if isinstance(region, dict) else {}
        region_reasons: list[str] = []
        if train["PSNR"] < float(args.selector_min_trainval_psnr_gain):
            region_reasons.append(f"region_psnr_gain_below_{args.selector_min_trainval_psnr_gain:g}")
        if train["SSIM"] < -float(args.selector_max_trainval_ssim_regression):
            region_reasons.append(f"region_ssim_regression_exceeds_{args.selector_max_trainval_ssim_regression:g}")
        if train["LPIPS"] > float(args.selector_max_trainval_lpips_regression):
            region_reasons.append(f"region_lpips_regression_exceeds_{args.selector_max_trainval_lpips_regression:g}")
        if balanced < float(args.selector_region_min_trainval_balanced_delta):
            region_reasons.append(f"region_balanced_delta_below_{args.selector_region_min_trainval_balanced_delta:g}")
        if not bool(region.get("present", False)):
            region_reasons.append("region_promotion_missing_filter_evidence")
        if int(region.get("accepted_carriers", 0) or 0) <= 0:
            region_reasons.append("region_promotion_no_accepted_carrier")
        if num(region.get("mean_core_balanced_delta"), -math.inf) < float(args.selector_region_min_mean_core_balanced_delta):
            region_reasons.append(f"region_mean_core_balanced_below_{args.selector_region_min_mean_core_balanced_delta:g}")
        if num(region.get("mean_delta_core_psnr"), -math.inf) < float(args.selector_region_min_mean_delta_psnr):
            region_reasons.append(f"region_mean_delta_psnr_below_{args.selector_region_min_mean_delta_psnr:g}")
        if num(region.get("min_changed_fraction"), -math.inf) < float(args.selector_region_min_changed_fraction):
            region_reasons.append(f"region_changed_fraction_below_{args.selector_region_min_changed_fraction:g}")
        if num(region.get("max_negative_core_balanced_fraction"), math.inf) > float(
            args.selector_region_max_negative_core_balanced_fraction
        ):
            region_reasons.append(
                f"region_negative_fraction_exceeds_{args.selector_region_max_negative_core_balanced_fraction:g}"
            )
        if num(region.get("max_context_mse_regression"), math.inf) > float(
            args.selector_region_max_context_mse_regression
        ):
            region_reasons.append(f"region_context_mse_regression_exceeds_{args.selector_region_max_context_mse_regression:g}")
        if not region_reasons:
            row["selector_pass_mode"] = "region_stable"
            row["selector_region_reasons"] = []
            return True, []
        row["selector_region_reasons"] = region_reasons

    row["selector_pass_mode"] = "rejected"
    return False, reasons


def only_method_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    first = next(iter(payload.values()))
    return first if isinstance(first, dict) else {}


def trainval_per_view_tail(
    root: Path,
    spec: TrialSpec,
    scene: str,
    *,
    cvar_fraction: float = 0.20,
    ssim_weight: float = 20.0,
    lpips_weight: float = 20.0,
) -> dict[str, float]:
    trial_root = root / "trials" / spec.label / scene
    base_path = trial_root / "phasej_trainval_gate_per_view.json"
    candidate_path = trial_root / "model" / "trainval_gate_per_view.json"
    if not base_path.is_file() or not candidate_path.is_file():
        return {"view_count": 0.0}
    base = only_method_metrics(read_json(base_path))
    candidate = only_method_metrics(read_json(candidate_path))
    if not base or not candidate:
        return {"view_count": 0.0}
    view_names = sorted(set((base.get("PSNR") or {}).keys()) & set((candidate.get("PSNR") or {}).keys()))
    rows: list[dict[str, float]] = []
    for name in view_names:
        dpsnr = num((candidate.get("PSNR") or {}).get(name)) - num((base.get("PSNR") or {}).get(name))
        dssim = num((candidate.get("SSIM") or {}).get(name)) - num((base.get("SSIM") or {}).get(name))
        dlpips = num((candidate.get("LPIPS") or {}).get(name)) - num((base.get("LPIPS") or {}).get(name))
        if not all(math.isfinite(value) for value in (dpsnr, dssim, dlpips)):
            continue
        rows.append(
            {
                "PSNR": dpsnr,
                "SSIM": dssim,
                "LPIPS": dlpips,
                "balanced": dpsnr + float(ssim_weight) * dssim - float(lpips_weight) * dlpips,
            }
        )
    if not rows:
        return {"view_count": 0.0}
    balanced_values = sorted(row["balanced"] for row in rows)
    psnr_values = sorted(row["PSNR"] for row in rows)
    lpips_values = sorted((row["LPIPS"] for row in rows), reverse=True)
    cvar_count = max(1, int(math.ceil(len(rows) * max(min(float(cvar_fraction), 1.0), 1e-6))))
    balanced_cvar_delta = mean(balanced_values[:cvar_count])
    balanced_cvar_loss = max(0.0, -balanced_cvar_delta) if math.isfinite(balanced_cvar_delta) else math.inf
    mean_balanced_delta = mean([row["balanced"] for row in rows])
    if balanced_cvar_loss <= 1e-12:
        mean_to_cvar_ratio = math.inf if mean_balanced_delta > 0.0 else 0.0
    else:
        mean_to_cvar_ratio = max(0.0, mean_balanced_delta) / balanced_cvar_loss
    return {
        "view_count": float(len(rows)),
        "mean_psnr_delta": mean([row["PSNR"] for row in rows]),
        "mean_abs_psnr_delta": mean([abs(row["PSNR"]) for row in rows]),
        "mean_balanced_delta": float(mean_balanced_delta),
        "psnr_negative_fraction": float(sum(1 for row in rows if row["PSNR"] < 0.0) / len(rows)),
        "balanced_negative_fraction": float(sum(1 for row in rows if row["balanced"] < 0.0) / len(rows)),
        "lpips_positive_fraction": float(sum(1 for row in rows if row["LPIPS"] > 0.0) / len(rows)),
        "worst_lpips_regression": max(row["LPIPS"] for row in rows),
        "worst_balanced_delta": min(row["balanced"] for row in rows),
        "cvar_fraction": float(cvar_fraction),
        "cvar_view_count": float(cvar_count),
        "balanced_ssim_weight": float(ssim_weight),
        "balanced_lpips_weight": float(lpips_weight),
        "psnr_cvar_delta": mean(psnr_values[:cvar_count]),
        "lpips_worst_cvar_regression": mean(lpips_values[:cvar_count]),
        "balanced_cvar_delta": float(balanced_cvar_delta),
        "balanced_cvar_loss": float(balanced_cvar_loss),
        "mean_to_cvar_ratio": float(mean_to_cvar_ratio),
    }


def decision_row(
    root: Path,
    spec: TrialSpec,
    scene: str,
    face_ids: list[int],
    exit_code: int,
    args: argparse.Namespace,
    region_promotion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = decision_path(root, spec, scene)
    decision = read_json(path)
    train_delta = metric_block(decision.get("trainval_delta"))
    test_delta = metric_block(decision.get("test_delta_report_only"))
    row = {
        "trial": spec.label,
        "mode": spec.mode,
        "count": spec.count,
        "scale": spec.scale,
        "face_ids": face_ids,
        "exit_code": int(exit_code),
        "decision_path": path_label(path),
        "present": bool(decision),
        "accepted": bool(decision.get("accepted", False)),
        "selected_label": decision.get("selected_label", ""),
        "decision_reasons": decision.get("decision_reasons", []),
        "trainval_delta": train_delta,
        "trainval_balanced_delta": num(decision.get("trainval_balanced_delta"), -math.inf),
        "trainval_per_view_tail": trainval_per_view_tail(
            root,
            spec,
            scene,
            cvar_fraction=float(args.selector_tail_cvar_fraction),
            ssim_weight=float(args.selector_balanced_ssim_weight),
            lpips_weight=float(args.selector_balanced_lpips_weight),
        ),
        "selector_region_promotion": region_promotion or {},
        "report_only_test_delta": test_delta,
        "test_balanced_delta_report_only": num(decision.get("test_balanced_delta_report_only"), math.nan),
    }
    passed, reasons = selector_pass(row, args)
    row["selector_pass"] = bool(passed)
    row["selector_reasons"] = reasons
    return row


def effective_delta(row: dict[str, Any] | None) -> dict[str, float]:
    if not row or not bool(row.get("accepted", False)):
        return {key: 0.0 for key in METRICS}
    return metric_block(row.get("report_only_test_delta"))


def run_scene(args: argparse.Namespace, scene: str, specs: list[TrialSpec]) -> dict[str, Any]:
    root = ROOT / args.output_root
    reuse_trials_root = ROOT / str(args.reuse_trials_root) if str(args.reuse_trials_root).strip() else None
    trial_decision_root = reuse_trials_root or root
    scene_plan_path = plan_path(args.plan_template, scene)
    plan = read_json(scene_plan_path)
    plan_alpha_path = plan_materialize_alpha_path(plan, plan_file=scene_plan_path)
    region_promotion = render_region_promotion_summary(plan)
    candidates = plan.get("candidates") if isinstance(plan.get("candidates"), list) else []
    strict_patchcert_plan = is_strict_patchcert_plan(plan)
    strict_safe_replay = (
        bool(strict_patchcert_plan)
        and not bool(args.selector_allow_uncertified_plan)
        and str(args.selector_strict_cert_replay_mode) == "full_plan"
    )
    scene_log = root / scene / "facelocal_coupled_selector.log"
    rows: list[dict[str, Any]] = []
    if not candidates:
        payload = {
            "scene": scene,
            "plan_path": path_label(scene_plan_path),
            "plan_materialize_alpha_json": path_label(plan_alpha_path) if plan_alpha_path else "",
            "candidate_count": 0,
            "strict_patchcert_plan": bool(strict_patchcert_plan),
            "strict_safe_replay": bool(strict_safe_replay),
            "reuse_trials_root": path_label(reuse_trials_root) if reuse_trials_root else "",
            "selected_trial": "phasej_fallback",
            "accepted": False,
            "selection_uses_test": False,
            "decision_reasons": ["no_plan_candidates"],
            "effective_report_only_test_delta": {key: 0.0 for key in METRICS},
            "trials": rows,
        }
        write_json(root / scene / "coupled_selector_decision.json", payload)
        return payload

    if (
        bool(strict_patchcert_plan)
        and not bool(args.selector_allow_uncertified_plan)
        and str(args.selector_strict_cert_replay_mode) == "reject"
    ):
        payload = {
            "scene": scene,
            "plan_path": path_label(scene_plan_path),
            "plan_materialize_alpha_json": path_label(plan_alpha_path) if plan_alpha_path else "",
            "candidate_count": int(len(candidates)),
            "strict_patchcert_plan": True,
            "strict_safe_replay": False,
            "trial_specs": [spec.label for spec in specs],
            "selection_uses_test": False,
            "reuse_trials_root": path_label(reuse_trials_root) if reuse_trials_root else "",
            "accepted": False,
            "selected_trial": "phasej_fallback",
            "selected_trainval_balanced_delta": 0.0,
            "decision_reasons": ["strict_patchcert_plan_replay_rejected_by_selector_mode"],
            "effective_report_only_test_delta": {key: 0.0 for key in METRICS},
            "trials": rows,
        }
        write_json(root / scene / "coupled_selector_decision.json", payload)
        return payload

    strict_scale_policy: dict[str, Any] = {"enabled": False}
    if strict_safe_replay:
        replay_scales = parse_scale_list(args.selector_strict_replay_scales)
        if bool(args.selector_strict_adaptive_scale_policy):
            replay_scales, strict_scale_policy = strict_adaptive_scale_policy(replay_scales, candidates, args)
        active_specs = [strict_full_plan_spec(candidates, scale=scale) for scale in replay_scales]
    else:
        active_specs = specs
    uses_georisk = any(spec.mode in {"georisk", "patchrisk"} for spec in active_specs)
    face_adjacency, face_centers, geometry_meta = face_adjacency_geometry(
        plan,
        candidates,
        enabled=bool(uses_georisk and args.georisk_load_adjacency),
    )
    score_types = {
        "top": "rank",
        "score": "train_certificate_score",
        "risk": "risk_greedy_train_certificate_pair_penalty",
        "georisk": "geometry_neighborhood_cvar_train_certificate",
        "patchrisk": "patch_level_georisk_neighborhood_residual_carrier",
        "strictfull": "strict_patchcert_full_plan_replay",
    }
    max_concentration = max((local_error_concentration(row) for row in candidates), default=1.0)
    for spec in active_specs:
        trial_rows = selected_rows(
            candidates,
            spec,
            pair_lambda=float(args.risk_pair_lambda),
            face_adjacency=face_adjacency,
            args=args,
        )
        patchrisk_meta: dict[str, Any] = {}
        score_rows = trial_rows
        if spec.mode == "patchrisk":
            seed_rows = trial_rows
            expanded_rows, patchrisk_meta = patchrisk_expand_rows(
                candidates,
                seed_rows,
                face_adjacency=face_adjacency,
                face_centers=face_centers,
                args=args,
            )
            patch_by_face: dict[int, dict[str, Any]] = {}
            for patch in patchrisk_meta.get("patches", []):
                if not isinstance(patch, dict):
                    continue
                seed_id = int(patch.get("seed_face", -1))
                faces_in_patch = patch.get("faces", [])
                for face_id in faces_in_patch if isinstance(faces_in_patch, list) else []:
                    patch_by_face[int(face_id)] = {
                        "seed_face": seed_id,
                        "patch_size": int(patch.get("patch_size", 0)),
                    }
            seed_ids = {int(row.get("face_id", -1)) for row in seed_rows}
            trial_rows = []
            for row in expanded_rows:
                face_id = int(row.get("face_id", -1))
                annotated = dict(row)
                patch_info = patch_by_face.get(face_id, {})
                annotated["_patchrisk_role"] = "seed" if face_id in seed_ids else "neighbor"
                annotated["_patchrisk_seed_face"] = int(patch_info.get("seed_face", face_id))
                annotated["_patchrisk_patch_size"] = int(patch_info.get("patch_size", 1))
                trial_rows.append(annotated)
            score_rows = trial_rows
        face_ids = [int(row["face_id"]) for row in trial_rows]
        manifest = {
            "scene": scene,
            "trial": spec.label,
            "mode": spec.mode,
            "count": spec.count,
            "scale": spec.scale,
            "strict_patchcert_plan": bool(strict_patchcert_plan),
            "strict_safe_replay": bool(strict_safe_replay),
            "strict_scale_policy": strict_scale_policy,
            "requested_trial_specs": [item.label for item in specs],
            "selection_uses_test": False,
            "plan_materialize_alpha_json": path_label(plan_alpha_path) if plan_alpha_path else "",
            "score_type": score_types[spec.mode],
            "risk_pair_lambda": float(args.risk_pair_lambda) if spec.mode == "risk" else 0.0,
            "georisk_pair_lambda": float(args.georisk_pair_lambda) if spec.mode in {"georisk", "patchrisk"} else 0.0,
            "georisk_geometry_lambda": float(args.georisk_geometry_lambda) if spec.mode in {"georisk", "patchrisk"} else 0.0,
            "georisk_tail_lambda": float(args.georisk_tail_lambda) if spec.mode in {"georisk", "patchrisk"} else 0.0,
            "georisk_error_lambda": float(args.georisk_error_lambda) if spec.mode in {"georisk", "patchrisk"} else 0.0,
            "georisk_tail_fraction": float(args.georisk_tail_fraction) if spec.mode in {"georisk", "patchrisk"} else 0.0,
            "georisk_min_view_gain": float(args.georisk_min_view_gain) if spec.mode in {"georisk", "patchrisk"} else 0.0,
            "georisk_geometry": geometry_meta if spec.mode in {"georisk", "patchrisk"} else {},
            "patchrisk": patchrisk_meta if spec.mode == "patchrisk" else {},
            "selector_tail_cvar_fraction": float(args.selector_tail_cvar_fraction),
            "selector_balanced_ssim_weight": float(args.selector_balanced_ssim_weight),
            "selector_balanced_lpips_weight": float(args.selector_balanced_lpips_weight),
            "selector_tail_max_balanced_cvar_loss": float(args.selector_tail_max_balanced_cvar_loss),
            "selector_tail_min_mean_to_cvar_ratio": float(args.selector_tail_min_mean_to_cvar_ratio),
            "selector_tail_max_lpips_positive_fraction": float(args.selector_tail_max_lpips_positive_fraction),
            "alpha_refit": bool(args.selector_fit_plan_alphas and (not strict_safe_replay or args.selector_strict_fit_plan_alphas)),
            "strict_alpha_shrink": bool(strict_safe_replay and args.selector_strict_fit_plan_alphas),
            "allow_uncertified_plan": bool(args.selector_allow_uncertified_plan),
            "materialize_face_filter": bool(not strict_safe_replay),
            "materialize_scale": float(spec.scale),
            "reuse_trials_root": path_label(reuse_trials_root) if reuse_trials_root else "",
            "face_ids": face_ids,
            "face_scores": face_score_entries(
                score_rows,
                "georisk" if spec.mode == "patchrisk" else spec.mode,
                pair_lambda=float(args.risk_pair_lambda),
                face_adjacency=face_adjacency,
                args=args,
                max_concentration=max_concentration,
            ),
        }
        manifest_path = root / scene / "trial_manifests" / f"{spec.label}.json"
        write_json(manifest_path, manifest)
        reused_trial = False
        if reuse_trials_root is not None:
            reuse_decision = decision_path(reuse_trials_root, spec, scene)
            if reuse_decision.is_file():
                rows.append(decision_row(reuse_trials_root, spec, scene, face_ids, 0, args, region_promotion))
                reused_trial = True
        if reused_trial:
            continue
        decision = decision_path(root, spec, scene)
        if decision.is_file() and not bool(args.force):
            rows.append(decision_row(root, spec, scene, face_ids, 0, args, region_promotion))
            continue
        if bool(args.force) or not decision.is_file():
            alpha_path: Path | None = None
            if bool(args.selector_fit_plan_alphas and (not strict_safe_replay or args.selector_strict_fit_plan_alphas)):
                alpha_path = alpha_json_path(root, scene, spec)
                alpha_cmd = fit_alpha_command(args, scene, spec, face_ids, alpha_path)
                alpha_exit = run_command(alpha_cmd, gpu=int(args.gpu), log_path=scene_log, dry_run=bool(args.dry_run))
                if alpha_exit != 0:
                    rows.append(decision_row(root, spec, scene, face_ids, alpha_exit, args, region_promotion))
                    continue
            elif strict_safe_replay and plan_alpha_path is not None:
                alpha_path = plan_alpha_path
            render_trust_path: Path | None = None
            if strict_safe_replay and (abs(float(spec.scale) - 1.0) > 1.0e-12 or alpha_path is not None):
                render_trust_path = render_trust_json_path(root, scene, spec)
                write_strict_shrink_render_trust(
                    path=render_trust_path,
                    plan_file=scene_plan_path,
                    scene=scene,
                    spec=spec,
                    alpha_json=alpha_path,
                )
            cmd = build_trial_command(
                args,
                scene,
                spec,
                face_ids,
                alpha_json=alpha_path,
                render_trust_json=render_trust_path,
                materialize_face_filter=bool(not strict_safe_replay),
                materialize_scale=spec.scale if strict_safe_replay else None,
            )
            exit_code = run_command(cmd, gpu=int(args.gpu), log_path=scene_log, dry_run=bool(args.dry_run))
            if exit_code != 0:
                rows.append(decision_row(root, spec, scene, face_ids, exit_code, args, region_promotion))
                continue
        rows.append(decision_row(root, spec, scene, face_ids, 0, args, region_promotion))

    accepted = [
        row
        for row in rows
        if row["selector_pass"] and math.isfinite(float(row.get("trainval_balanced_delta", math.nan)))
    ]
    selected = max(accepted, key=lambda row: float(row["trainval_balanced_delta"])) if accepted else None
    payload = {
        "scene": scene,
        "plan_path": path_label(scene_plan_path),
        "plan_materialize_alpha_json": path_label(plan_alpha_path) if plan_alpha_path else "",
        "candidate_count": int(len(candidates)),
        "strict_patchcert_plan": bool(strict_patchcert_plan),
        "strict_safe_replay": bool(strict_safe_replay),
        "trial_specs": [spec.label for spec in active_specs],
        "requested_trial_specs": [spec.label for spec in specs],
        "selection_uses_test": False,
        "reuse_trials_root": path_label(reuse_trials_root) if reuse_trials_root else "",
        "georisk_geometry": geometry_meta if uses_georisk else {},
        "strict_scale_policy": strict_scale_policy,
        "accepted": bool(selected),
        "selected_trial": selected["trial"] if selected else "phasej_fallback",
        "selected_selector_pass_mode": selected.get("selector_pass_mode", "") if selected else "",
        "selected_trainval_balanced_delta": float(selected["trainval_balanced_delta"]) if selected else 0.0,
        "effective_report_only_test_delta": effective_delta(selected),
        "trials": rows,
    }
    write_json(root / scene / "coupled_selector_decision.json", payload)
    return payload


def fmt(value: Any, digits: int = 9) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f}"


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    present = [row for row in rows if row.get("candidate_count", 0) > 0]
    accepted = [row for row in rows if row.get("accepted")]
    mean_effective = {
        key: (sum(float(row["effective_report_only_test_delta"][key]) for row in present) / len(present) if present else math.nan)
        for key in METRICS
    }
    payload = {
        "scene_count": len(rows),
        "present_candidate_scene_count": len(present),
        "accepted_count": len(accepted),
        "mean_effective_report_only_test_delta": mean_effective,
        "rows": rows,
    }
    write_json(root / "coupled_selector_summary.json", payload)
    lines = [
        "# Phase-S Face-Local Coupled Selector Summary",
        "",
        "Selection uses train-val render metrics only. Held-out test deltas are report-only; rejected scenes fall back to Phase-J with zero effective test delta.",
        "",
        f"- scenes: `{len(rows)}`",
        f"- scenes with plan candidates: `{len(present)}`",
        f"- accepted scenes: `{len(accepted)}`",
        f"- mean effective report-only dPSNR: `{fmt(mean_effective['PSNR'])}`",
        f"- mean effective report-only dSSIM: `{fmt(mean_effective['SSIM'])}`",
        f"- mean effective report-only dLPIPS: `{fmt(mean_effective['LPIPS'])}`",
        "",
        "| scene | candidates | selected trial | accepted | train-val balanced | effective test dPSNR | effective test dSSIM | effective test dLPIPS |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        eff = row.get("effective_report_only_test_delta", {})
        lines.append(
            f"| {row['scene']} | {int(row.get('candidate_count', 0))} | {row.get('selected_trial')} | "
            f"{str(bool(row.get('accepted'))).lower()} | {fmt(row.get('selected_trainval_balanced_delta'))} | "
            f"{fmt(eff.get('PSNR'))} | {fmt(eff.get('SSIM'))} | {fmt(eff.get('LPIPS'))} |"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "coupled_selector_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    specs = parse_trial_specs(args.trial_specs)
    rows = [run_scene(args, scene, specs) for scene in scene_list(args.scenes)]
    root = ROOT / args.output_root
    write_summary(root, rows)
    print(json.dumps({"rows": len(rows), "output_root": str(root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

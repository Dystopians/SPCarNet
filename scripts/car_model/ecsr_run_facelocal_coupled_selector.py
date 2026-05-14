#!/usr/bin/env python3
"""Run a train-val render-risk selector over face-local residual plan subsets.

The script is an outer-loop selector for Phase-S face-local residual plans. It
does not change the underlying render/eval gate. Instead, it builds a fixed set
of face subsets from a train-only candidate plan, runs the existing Phase-K/S
gate for each subset, then promotes the accepted subset with the best train-val
balanced delta. Held-out test deltas are copied as report-only evidence only.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--plan_template", default=DEFAULT_PLAN_TEMPLATE)
    parser.add_argument("--evidence_root", default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--trial_specs",
        default="top1x2,score2x1,score4x1,score8x0.5",
        help=(
            "Comma separated trials. Grammar: topNxS or scoreNxS, for example "
            "top1x2,score4x1,score8x0.5. score uses train-only plan certificates."
        ),
    )
    parser.add_argument("--candidate_prefix", default="facelocal_coupled_v1")
    parser.add_argument("--phasej_test_method", default=DEFAULT_PHASEJ_TEST_METHOD)
    parser.add_argument("--phasej_trainval_method", default=DEFAULT_PHASEJ_TRAINVAL_METHOD)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=0.0)
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
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phase_s_facelocal_coupled_selector_v1_20260513")
    parser.add_argument("--skip_failed_views", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


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
        match = re.fullmatch(r"(top|score)(\d+)x([0-9]*\.?[0-9]+)", token)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def selected_rows(candidates: list[dict[str, Any]], spec: TrialSpec) -> list[dict[str, Any]]:
    if spec.mode == "top":
        rows = list(candidates)
    elif spec.mode == "score":
        rows = sorted(candidates, key=train_certificate_score, reverse=True)
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


def build_trial_command(args: argparse.Namespace, scene: str, spec: TrialSpec, face_ids: list[int]) -> list[str]:
    label = f"{args.candidate_prefix}_{spec.label}"
    output_root = Path(args.output_root) / "trials" / spec.label
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
        "--delta_facelocal_materialize_plan_face_ids",
        ",".join(str(fid) for fid in face_ids),
        "--delta_facelocal_materialize_plan_scale",
        str(spec.scale),
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
    if bool(args.skip_failed_views):
        cmd.append("--skip_failed_views")
    if bool(args.force):
        cmd.append("--force")
    return cmd


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
    return not reasons, reasons


def decision_row(
    root: Path,
    spec: TrialSpec,
    scene: str,
    face_ids: list[int],
    exit_code: int,
    args: argparse.Namespace,
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
        "decision_path": str(path.relative_to(ROOT)) if path.is_file() else str(path),
        "present": bool(decision),
        "accepted": bool(decision.get("accepted", False)),
        "selected_label": decision.get("selected_label", ""),
        "decision_reasons": decision.get("decision_reasons", []),
        "trainval_delta": train_delta,
        "trainval_balanced_delta": num(decision.get("trainval_balanced_delta"), -math.inf),
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
    plan = read_json(plan_path(args.plan_template, scene))
    candidates = plan.get("candidates") if isinstance(plan.get("candidates"), list) else []
    scene_log = root / scene / "facelocal_coupled_selector.log"
    rows: list[dict[str, Any]] = []
    if not candidates:
        payload = {
            "scene": scene,
            "plan_path": str(plan_path(args.plan_template, scene).relative_to(ROOT)),
            "candidate_count": 0,
            "selected_trial": "phasej_fallback",
            "accepted": False,
            "selection_uses_test": False,
            "decision_reasons": ["no_plan_candidates"],
            "effective_report_only_test_delta": {key: 0.0 for key in METRICS},
            "trials": rows,
        }
        write_json(root / scene / "coupled_selector_decision.json", payload)
        return payload

    for spec in specs:
        trial_rows = selected_rows(candidates, spec)
        face_ids = [int(row["face_id"]) for row in trial_rows]
        manifest = {
            "scene": scene,
            "trial": spec.label,
            "mode": spec.mode,
            "count": spec.count,
            "scale": spec.scale,
            "selection_uses_test": False,
            "score_type": "rank" if spec.mode == "top" else "train_certificate_score",
            "face_ids": face_ids,
            "face_scores": [
                {
                    "face_id": int(row["face_id"]),
                    "rank": int(row.get("rank", -1)),
                    "train_certificate_score": train_certificate_score(row),
                    "policy_val_relative_gain": num(nested(row, "policy_val_proxy", "relative_gain"), math.nan),
                    "policy_val_samples": num(nested(row, "policy_val_proxy", "samples"), math.nan),
                }
                for row in trial_rows
            ],
        }
        manifest_path = root / scene / "trial_manifests" / f"{spec.label}.json"
        write_json(manifest_path, manifest)
        decision = decision_path(root, spec, scene)
        if bool(args.force) or not decision.is_file():
            cmd = build_trial_command(args, scene, spec, face_ids)
            exit_code = run_command(cmd, gpu=int(args.gpu), log_path=scene_log, dry_run=bool(args.dry_run))
            if exit_code != 0:
                rows.append(decision_row(root, spec, scene, face_ids, exit_code, args))
                continue
        rows.append(decision_row(root, spec, scene, face_ids, 0, args))

    accepted = [
        row
        for row in rows
        if row["selector_pass"] and math.isfinite(float(row.get("trainval_balanced_delta", math.nan)))
    ]
    selected = max(accepted, key=lambda row: float(row["trainval_balanced_delta"])) if accepted else None
    payload = {
        "scene": scene,
        "plan_path": str(plan_path(args.plan_template, scene).relative_to(ROOT)),
        "candidate_count": int(len(candidates)),
        "trial_specs": [spec.label for spec in specs],
        "selection_uses_test": False,
        "accepted": bool(selected),
        "selected_trial": selected["trial"] if selected else "phasej_fallback",
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

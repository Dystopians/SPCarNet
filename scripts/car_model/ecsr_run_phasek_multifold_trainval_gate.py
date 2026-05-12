#!/usr/bin/env python3
"""Run a multi-offset train-heldout gate for an existing Phase-K candidate.

This is a stricter diagnostic/selection layer on top of
ecsr_run_phasek_barycentric_gate_scene.py. It does not rebuild the candidate
checkpoint. Instead it re-runs train-heldout ELA validation on complementary
deterministic train-view slices and accepts a candidate only when every slice
passes the representation gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASEJ_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
BASE_METHOD = "ours_26000_phasef_extra_compact_base"
METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(path: Path, method: str) -> dict[str, float]:
    row = _read_json(path).get(method, {})
    out: dict[str, float] = {}
    for key in METRICS:
        try:
            value = float(row.get(key))
        except Exception:
            value = math.nan
        out[key] = value if math.isfinite(value) else math.nan
    return out


def _has_metric(path: Path, method: str) -> bool:
    values = _metric(path, method)
    return all(math.isfinite(values[key]) for key in METRICS)


def _delta(candidate: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - base[key]) for key in METRICS}


def _balanced(delta: dict[str, float], *, ssim_weight: float, lpips_weight: float) -> float:
    return float(delta["PSNR"] + float(ssim_weight) * delta["SSIM"] - float(lpips_weight) * delta["LPIPS"])


def _run(cmd: list[str], *, gpu: int, log_path: Path, wandb_online: bool = False) -> None:
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if wandb_online:
        env["WANDB_MODE"] = "online"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def _policy_args(report: dict[str, Any], *, args: argparse.Namespace, offset: int) -> list[str]:
    policy = report.get("policy") or {}
    out = [
        "--mode",
        str(policy.get("mode", "residual")),
        "--k",
        str(int(policy.get("k", 4))),
        "--residual_clip",
        str(float(policy.get("residual_clip", 0.25))),
        "--depth_abs_tol",
        str(float(policy.get("depth_abs_tol", 0.02))),
        "--depth_rel_tol",
        str(float(policy.get("depth_rel_tol", 0.06))),
        "--direction_weight",
        str(float(policy.get("direction_weight", 0.35))),
        "--alpha",
        "0",
        "--skip_fixed_alpha_calibration",
        "--alpha_policy",
        "adaptive_bins",
        "--alpha_feature_mode",
        str(args.alpha_feature_mode),
        "--alpha_default",
        str(args.alpha_default),
        "--policy_holdout_fraction",
        str(args.policy_holdout_fraction),
        "--policy_holdout_offset",
        str(offset),
        "--support_policy_fit_only",
        "--calib_sampler",
        args.calib_sampler,
        "--calib_max_views",
        str(args.calib_max_views),
        "--calib_stride",
        str(args.calib_stride),
    ]
    if bool(policy.get("edge_gate", False)):
        out.extend(
            [
                "--edge_gate",
                "--edge_gate_quantile",
                str(float(policy.get("edge_gate_quantile", 0.5))),
                "--edge_gate_dilate",
                str(int(policy.get("edge_gate_dilate", 1))),
                "--edge_gate_min",
                str(float(policy.get("edge_gate_min", 0.0))),
            ]
        )
    return out


def _apply_train_ela(
    args: argparse.Namespace,
    *,
    scene: str,
    model: Path,
    base_method: str,
    method: str,
    phasej_report: dict[str, Any],
    offset: int,
    log_path: Path,
) -> Path:
    report_path = model / "train" / method / "ela_report.json"
    if not bool(args.force) and report_path.is_file():
        return report_path
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_apply_evidence_lumigraph_adapter.py",
        "--base_model_path",
        str(model),
        "--iteration",
        str(args.iteration),
        "--base_method_name",
        base_method,
        "--target_split",
        "train",
        "--method_name",
        method,
        "--wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        f"{args.wandb_name}_{scene}_{method}_o{offset}_train",
    ]
    cmd.extend(_policy_args(phasej_report, args=args, offset=offset))
    _run(cmd, gpu=int(args.gpu), log_path=log_path, wandb_online=True)
    return report_path


def _evaluate_trainval(
    args: argparse.Namespace,
    *,
    model: Path,
    method: str,
    view_names_file: Path,
    output: Path,
    per_view_output: Path,
    log_path: Path,
) -> None:
    if not bool(args.force) and _has_metric(output, method):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model),
        "--split",
        "train",
        "--methods",
        method,
        "--view_names_file",
        str(view_names_file),
        "--view_names_key",
        "policy_val_views",
        "--output",
        str(output),
        "--per_view_output",
        str(per_view_output),
    ]
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _passes(delta: dict[str, float], balanced_delta: float, args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if delta["PSNR"] < float(args.gate_min_psnr_gain):
        reasons.append(f"psnr_gain_below_{args.gate_min_psnr_gain:g}")
    if delta["SSIM"] < -float(args.gate_max_ssim_regression):
        reasons.append(f"ssim_regression_exceeds_{args.gate_max_ssim_regression:g}")
    if delta["LPIPS"] > float(args.gate_max_lpips_regression):
        reasons.append(f"lpips_regression_exceeds_{args.gate_max_lpips_regression:g}")
    if balanced_delta < float(args.gate_min_balanced_delta):
        reasons.append(f"balanced_delta_below_{args.gate_min_balanced_delta:g}")
    return reasons


def _candidate_audit_ok(path: Path) -> tuple[bool, dict[str, Any]]:
    audit = _read_json(path)
    ok = bool(audit) and bool(audit.get("accepted", False)) and not bool(audit.get("no_op_copy", False))
    return ok, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene = args.scene
    phasej_model = ROOT / args.phasej_model
    candidate_model = ROOT / args.candidate_model
    output_root = ROOT / args.output_root
    log_path = output_root / scene / "multifold_trainval_gate.log"
    phasej_report = _read_json(phasej_model / "test" / PHASEJ_METHOD / "ela_report.json")
    if not phasej_report:
        raise FileNotFoundError(phasej_model / "test" / PHASEJ_METHOD / "ela_report.json")
    offsets = [int(x.strip()) for x in str(args.offsets).split(",") if x.strip()]
    if not offsets:
        raise ValueError("at least one offset is required")

    rows: list[dict[str, Any]] = []
    for offset in offsets:
        base_method = f"{args.phasej_trainval_method_prefix}_o{offset}"
        cand_method = f"{args.candidate_trainval_method_prefix}_o{offset}"
        base_report = _apply_train_ela(
            args,
            scene=scene,
            model=phasej_model,
            base_method=BASE_METHOD,
            method=base_method,
            phasej_report=phasej_report,
            offset=offset,
            log_path=log_path,
        )
        cand_report = _apply_train_ela(
            args,
            scene=scene,
            model=candidate_model,
            base_method=args.candidate_base_method,
            method=cand_method,
            phasej_report=phasej_report,
            offset=offset,
            log_path=log_path,
        )
        base_results = output_root / scene / f"base_trainval_o{offset}_results.json"
        cand_results = output_root / scene / f"candidate_trainval_o{offset}_results.json"
        _evaluate_trainval(
            args,
            model=phasej_model,
            method=base_method,
            view_names_file=base_report,
            output=base_results,
            per_view_output=output_root / scene / f"base_trainval_o{offset}_per_view.json",
            log_path=log_path,
        )
        _evaluate_trainval(
            args,
            model=candidate_model,
            method=cand_method,
            view_names_file=cand_report,
            output=cand_results,
            per_view_output=output_root / scene / f"candidate_trainval_o{offset}_per_view.json",
            log_path=log_path,
        )
        base = _metric(base_results, base_method)
        cand = _metric(cand_results, cand_method)
        delta = _delta(cand, base)
        balanced = _balanced(delta, ssim_weight=float(args.ssim_weight), lpips_weight=float(args.lpips_weight))
        reasons = _passes(delta, balanced, args)
        rows.append(
            {
                "offset": offset,
                "base_method": base_method,
                "candidate_method": cand_method,
                "base_metrics": base,
                "candidate_metrics": cand,
                "delta": delta,
                "balanced_delta": balanced,
                "accepted": not reasons,
                "decision_reasons": reasons,
            }
        )

    audit_ok, candidate_audit = _candidate_audit_ok(ROOT / args.candidate_audit_json)
    per_metric = {
        key: {
            "mean": float(sum(row["delta"][key] for row in rows) / len(rows)),
            "min": float(min(row["delta"][key] for row in rows)),
            "max": float(max(row["delta"][key] for row in rows)),
        }
        for key in METRICS
    }
    balanced_values = [float(row["balanced_delta"]) for row in rows]
    accepted = bool(audit_ok and all(row["accepted"] for row in rows))
    reasons: list[str] = []
    if not audit_ok:
        reasons.append("candidate_checkpoint_operator_rejected_or_noop")
    for row in rows:
        for reason in row["decision_reasons"]:
            reasons.append(f"offset{row['offset']}:{reason}")

    test_delta: dict[str, float] | None = None
    if args.candidate_test_method:
        base_test = _metric(phasej_model / "results.json", PHASEJ_METHOD)
        cand_test = _metric(candidate_model / "results.json", args.candidate_test_method)
        if all(math.isfinite(base_test[key]) and math.isfinite(cand_test[key]) for key in METRICS):
            test_delta = _delta(cand_test, base_test)

    payload: dict[str, Any] = {
        "scene": scene,
        "candidate_label": args.candidate_label,
        "fallback_label": args.fallback_label,
        "selected_label": args.candidate_label if accepted else args.fallback_label,
        "accepted": accepted,
        "selection_uses_test": False,
        "decision_reasons": reasons,
        "offsets": offsets,
        "rows": rows,
        "trainval_delta_summary": per_metric,
        "trainval_balanced_delta_summary": {
            "mean": float(sum(balanced_values) / len(balanced_values)),
            "min": float(min(balanced_values)),
            "max": float(max(balanced_values)),
        },
        "candidate_operator_audit": {
            "path": str(args.candidate_audit_json),
            "available": bool(candidate_audit),
            "accepted": bool(candidate_audit.get("accepted", False)) if candidate_audit else None,
            "no_op_copy": bool(candidate_audit.get("no_op_copy", False)) if candidate_audit else None,
            "policy_pass": bool(candidate_audit.get("policy_pass", False)) if candidate_audit else None,
        },
        "test_delta_report_only": test_delta,
        "thresholds": {
            "min_psnr_gain": float(args.gate_min_psnr_gain),
            "max_ssim_regression": float(args.gate_max_ssim_regression),
            "max_lpips_regression": float(args.gate_max_lpips_regression),
            "min_balanced_delta": float(args.gate_min_balanced_delta),
            "ssim_weight": float(args.ssim_weight),
            "lpips_weight": float(args.lpips_weight),
        },
    }
    out_json = output_root / scene / "multifold_trainval_gate.json"
    out_md = output_root / scene / "multifold_trainval_gate.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Multi-Offset Train-Val Gate",
        "",
        f"- scene: `{scene}`",
        f"- candidate: `{args.candidate_label}`",
        f"- selected: `{payload['selected_label']}`",
        f"- accepted: `{payload['accepted']}`",
        f"- offsets: `{','.join(str(x) for x in offsets)}`",
        f"- decision reasons: `{', '.join(reasons) or 'pass'}`",
        "",
        "| offset | accepted | dPSNR | dSSIM | dLPIPS | balanced | reasons |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        delta = row["delta"]
        lines.append(
            f"| {row['offset']} | {str(row['accepted']).lower()} | "
            f"{delta['PSNR']:+.9f} | {delta['SSIM']:+.9f} | {delta['LPIPS']:+.9f} | "
            f"{row['balanced_delta']:+.9f} | {', '.join(row['decision_reasons']) or 'pass'} |"
        )
    if test_delta is not None:
        lines.extend(
            [
                "",
                "Held-out test metrics below are report-only and were not used for selection.",
                f"- test delta PSNR/SSIM/LPIPS: `{test_delta['PSNR']:.9f}` / `{test_delta['SSIM']:.9f}` / `{test_delta['LPIPS']:.9f}`",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": accepted, "output_json": str(out_json)}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--phasej_model", required=True)
    parser.add_argument("--candidate_model", required=True)
    parser.add_argument("--candidate_audit_json", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--candidate_label", required=True)
    parser.add_argument("--fallback_label", default="phasej_guarded_adaptedge")
    parser.add_argument("--candidate_base_method", required=True)
    parser.add_argument("--candidate_test_method", default="")
    parser.add_argument("--phasej_trainval_method_prefix", default="ours_26000_phasej_multifold_trainval_gate")
    parser.add_argument("--candidate_trainval_method_prefix", default="ours_26000_candidate_multifold_trainval_gate")
    parser.add_argument("--offsets", default="0,1,2,3")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--policy_holdout_fraction", type=float, default=0.25)
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="uniform")
    parser.add_argument("--calib_max_views", type=int, default=32)
    parser.add_argument("--calib_stride", type=int, default=1)
    parser.add_argument("--alpha_feature_mode", choices=("confidence_magnitude", "confidence_magnitude_edge"), default="confidence_magnitude_edge")
    parser.add_argument("--alpha_default", type=float, default=0.0)
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=-1.0e9)
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phasek_multifold_trainval_gate")
    parser.add_argument("--wandb_name", default="phasek_multifold_trainval_gate")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

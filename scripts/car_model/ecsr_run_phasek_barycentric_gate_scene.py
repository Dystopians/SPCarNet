#!/usr/bin/env python3
"""Run Phase-K barycentric residual recovery and train-val gate for scenes.

This is an orchestration script. It uses the fixed Phase-J selected compact
checkpoint and the fixed Phase-J ELA policy for each scene, then evaluates a
single barycentric residual candidate with a train-heldout representation gate.
Held-out test metrics are collected as report-only evidence.
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
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}
PHASEJ_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
BASE_METHOD = "ours_26000_phasef_extra_compact_base"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(path: Path, method: str) -> dict[str, float]:
    row = _read_json(path).get(method, {})
    out: dict[str, float] = {}
    for key in ("PSNR", "SSIM", "LPIPS"):
        try:
            value = float(row.get(key))
        except Exception:
            value = math.nan
        out[key] = value if math.isfinite(value) else math.nan
    return out


def _has_metric(path: Path, method: str) -> bool:
    values = _metric(path, method)
    return all(math.isfinite(values[key]) for key in ("PSNR", "SSIM", "LPIPS"))


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


def _selected_model(policy_root: Path, scene: str) -> Path:
    selected = _read_json(policy_root / scene / "summary.json").get("selected", {})
    model_path = selected.get("model_path")
    if not model_path:
        raise RuntimeError(f"missing selected model for {scene}: {policy_root / scene / 'summary.json'}")
    model = ROOT / model_path
    if not model.is_dir():
        raise FileNotFoundError(model)
    return model


def _image_set(scene: str, args: argparse.Namespace) -> str:
    return args.outdoor_images if scene in OUTDOOR_SCENES else args.indoor_images


def _render_maps(
    args: argparse.Namespace,
    *,
    scene: str,
    model: Path,
    method_name: str,
    log_path: Path,
) -> None:
    train_dir = model / "train" / method_name
    test_dir = model / "test" / method_name
    if (
        not bool(args.force)
        and (train_dir / "camera_index.json").is_file()
        and (test_dir / "camera_index.json").is_file()
        and (train_dir / "depths").is_dir()
        and (test_dir / "depths").is_dir()
    ):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_render_evidence_maps.py",
        "-s",
        str(Path(args.dataset_root) / scene),
        "-m",
        str(model),
        "-i",
        _image_set(scene, args),
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(args.iteration),
        "--method_name",
        method_name,
        "--quiet",
    ]
    if bool(args.skip_failed_views):
        cmd.append("--skip_failed_views")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _build_evidence(args: argparse.Namespace, *, scene: str, phasej_model: Path, evidence_dir: Path, log_path: Path) -> None:
    summary = evidence_dir / "surface_evidence_summary.json"
    existing_summary = _read_json(summary)
    has_camera_center = "camera_center" in existing_summary.get("per_view_npz_fields", [])
    rich_surface_ok = existing_summary.get("barycentric_available") is True or bool(args.delta_uniform_barycentric)
    if (
        not bool(args.force)
        and summary.is_file()
        and rich_surface_ok
        and (str(args.delta_operator) not in {"sh1", "facelocal_sh1"} or has_camera_center)
    ):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_build_surface_evidence_cache.py",
        "-s",
        str(Path(args.dataset_root) / scene),
        "-m",
        str(phasej_model),
        "-i",
        _image_set(scene, args),
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(args.iteration),
        "--scene_name",
        scene,
        "--out_dir",
        str(evidence_dir.parent),
        "--base_method_name",
        BASE_METHOD,
        "--final_method_name",
        PHASEJ_METHOD,
        "--max_views",
        str(args.evidence_max_views),
        "--view_stride",
        str(args.evidence_view_stride),
        "--view_offset",
        str(args.evidence_view_offset),
        "--high_error_quantile",
        str(args.evidence_high_error_quantile),
        "--top_k_faces",
        str(args.delta_top_k),
        "--save_view_npz",
        "--save_residual_rgb",
        "--quiet",
    ]
    if not bool(args.delta_uniform_barycentric):
        cmd.append("--save_barycentric")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _apply_delta(args: argparse.Namespace, *, phasej_model: Path, evidence_dir: Path, candidate_model: Path, log_path: Path) -> None:
    if str(args.delta_operator) == "facelocal_sh1":
        apply_script = "scripts/car_model/ecsr_apply_surface_residual_facelocal_sh1_delta.py"
    elif str(args.delta_operator) == "sh1":
        apply_script = "scripts/car_model/ecsr_apply_surface_residual_barycentric_sh1_delta.py"
    else:
        apply_script = "scripts/car_model/ecsr_apply_surface_residual_barycentric_delta.py"
    audit = _candidate_audit_path(args, candidate_model)
    checkpoint = candidate_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "point_cloud_state_dict.pt"
    if not bool(args.force) and audit.is_file() and checkpoint.is_file():
        return
    cmd = [
        sys.executable,
        apply_script,
        "--source_model",
        str(phasej_model),
        "--evidence_dir",
        str(evidence_dir),
        "--output_model",
        str(candidate_model),
        "--iteration",
        str(args.iteration),
        "--top_k",
        str(args.delta_top_k),
        "--min_view_hits",
        str(args.delta_min_view_hits),
        "--min_consistency",
        str(args.delta_min_consistency),
        "--min_pixel_count",
        str(args.delta_min_pixel_count),
        "--max_samples_per_face_view",
        str(args.delta_max_samples_per_face_view),
        "--max_total_samples",
        str(args.delta_max_total_samples),
        "--high_error_quantile",
        str(args.delta_high_error_quantile),
        "--strength",
        str(args.delta_strength),
        "--max_abs_delta_rgb",
        str(args.delta_max_abs_rgb),
        "--lambda_mag",
        str(args.delta_lambda_mag),
        "--lambda_smooth",
        str(args.delta_lambda_smooth),
        "--steps",
        str(args.delta_steps),
        "--min_policy_val_relative_gain",
        str(args.delta_min_policy_val_relative_gain),
        "--min_policy_val_samples",
        str(args.delta_min_policy_val_samples),
        "--min_policy_val_unique_faces",
        str(args.delta_min_policy_val_unique_faces),
    ]
    if str(args.delta_operator) in {"sh1", "facelocal_sh1"}:
        cmd.extend(
            [
                "--max_abs_sh_coeff",
                str(args.delta_max_abs_sh_coeff),
                "--lambda_sh1_mag",
                str(args.delta_lambda_sh1_mag),
            ]
        )
        if bool(args.delta_uniform_barycentric):
            cmd.append("--uniform_barycentric")
    sh1_view_consensus = (
        str(args.delta_operator) in {"sh1", "facelocal_sh1"}
        and float(args.delta_min_face_view_consensus) > 0.0
    )
    facelocal_view_gain_certificate = (
        str(args.delta_operator) == "facelocal_sh1"
        and int(args.delta_min_face_gain_certificate_views) > 0
    )
    if str(args.delta_operator) == "facelocal_sh1" or (
        str(args.delta_operator) == "sh1" and (bool(args.delta_sh1_face_policy) or sh1_view_consensus)
    ):
        cmd.extend(
            [
                "--max_faces_to_apply",
                str(args.delta_max_faces_to_apply),
                "--min_face_policy_val_relative_gain",
                str(args.delta_min_face_policy_val_relative_gain),
                "--min_face_policy_val_samples",
                str(args.delta_min_face_policy_val_samples),
            ]
        )
    if sh1_view_consensus:
        cmd.extend(
            [
                "--min_face_view_consensus",
                str(args.delta_min_face_view_consensus),
                "--min_face_consensus_views",
                str(args.delta_min_face_consensus_views),
                "--min_face_consensus_view_samples",
                str(args.delta_min_face_consensus_view_samples),
                "--face_consensus_min_cosine",
                str(args.delta_face_consensus_min_cosine),
            ]
        )
    if facelocal_view_gain_certificate:
        cmd.extend(
            [
                "--min_face_gain_certificate_views",
                str(args.delta_min_face_gain_certificate_views),
                "--min_face_gain_certificate_relative_gain",
                str(args.delta_min_face_gain_certificate_relative_gain),
                "--min_face_gain_certificate_view_samples",
                str(args.delta_min_face_gain_certificate_view_samples),
                "--min_face_gain_certificate_fraction",
                str(args.delta_min_face_gain_certificate_fraction),
            ]
        )
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _candidate_audit_path(args: argparse.Namespace, candidate_model: Path) -> Path:
    if str(args.delta_operator) == "facelocal_sh1":
        return candidate_model / "surface_residual_facelocal_sh1_delta_audit.json"
    if str(args.delta_operator) == "sh1":
        return candidate_model / "surface_residual_barycentric_sh1_delta_audit.json"
    return candidate_model / "surface_residual_barycentric_delta_audit.json"


def _policy_args(report: dict[str, Any], *, trainval: bool, args: argparse.Namespace) -> list[str]:
    policy = report.get("policy") or {}
    out = [
        "--mode",
        str(policy.get("mode", "residual")),
        "--k",
        str(int(policy.get("k", 4))),
        "--residual_clip",
        str(float(policy.get("residual_clip", 0.2))),
        "--depth_abs_tol",
        str(float(policy.get("depth_abs_tol", 0.02))),
        "--depth_rel_tol",
        str(float(policy.get("depth_rel_tol", 0.06))),
        "--direction_weight",
        str(float(policy.get("direction_weight", 0.35))),
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
    alpha_policy = str(report.get("alpha_policy", "global"))
    if alpha_policy == "adaptive_bins":
        out.extend(
            [
                "--alpha",
                "0",
                "--skip_fixed_alpha_calibration",
                "--alpha_policy",
                "adaptive_bins",
                "--alpha_feature_mode",
                str(args.alpha_feature_mode),
                "--alpha_default",
                str(args.alpha_default),
            ]
        )
    else:
        out.extend(["--alpha", str(float(report.get("alpha", 0.0))), "--skip_fixed_alpha_calibration"])
    if trainval:
        out.extend(
            [
                "--policy_holdout_fraction",
                str(args.policy_holdout_fraction),
                "--policy_holdout_offset",
                str(args.policy_holdout_offset),
                "--support_policy_fit_only",
                "--calib_sampler",
                args.calib_sampler,
                "--calib_max_views",
                str(args.calib_max_views),
                "--calib_stride",
                str(args.calib_stride),
            ]
        )
    return out


def _apply_ela(
    args: argparse.Namespace,
    *,
    scene: str,
    model: Path,
    base_method: str,
    method_name: str,
    phasej_report: dict[str, Any],
    target_split: str,
    log_path: Path,
) -> None:
    report = model / target_split / method_name / "ela_report.json"
    if not bool(args.force) and report.is_file():
        return
    trainval = target_split == "train"
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
        target_split,
        "--method_name",
        method_name,
        "--wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        f"{args.wandb_name}_{scene}_{method_name}_{target_split}",
    ]
    cmd.extend(_policy_args(phasej_report, trainval=trainval, args=args))
    _run(cmd, gpu=int(args.gpu), log_path=log_path, wandb_online=True)


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


def _evaluate_test(args: argparse.Namespace, *, model: Path, method: str, log_path: Path) -> None:
    if not bool(args.force) and _has_metric(model / "results.json", method):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(model),
        "--split",
        "test",
        "--methods",
        method,
        "--merge_model_results",
    ]
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _decide(
    args: argparse.Namespace,
    *,
    scene: str,
    phasej_model: Path,
    candidate_model: Path,
    output_root: Path,
    log_path: Path,
) -> dict[str, Any]:
    decision_json = output_root / "decisions" / f"{scene}_decision.json"
    decision_md = output_root / "decisions" / f"{scene}_decision.md"
    if not bool(args.force) and decision_json.is_file():
        return _read_json(decision_json)
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_decide_phasek_trainval_gate.py",
        "--scene",
        scene,
        "--candidate_label",
        args.candidate_label,
        "--fallback_label",
        "phasej_guarded_adaptedge",
        "--base_trainval_results",
        str(phasej_model / "trainval_gate_results.json"),
        "--base_trainval_method",
        args.phasej_trainval_method,
        "--candidate_trainval_results",
        str(candidate_model / "trainval_gate_results.json"),
        "--candidate_trainval_method",
        args.candidate_trainval_method,
        "--candidate_audit_json",
        str(_candidate_audit_path(args, candidate_model)),
        "--base_test_results",
        str(phasej_model / "results.json"),
        "--base_test_method",
        PHASEJ_METHOD,
        "--candidate_test_results",
        str(candidate_model / "results.json"),
        "--candidate_test_method",
        args.candidate_test_method,
        "--min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--output_json",
        str(decision_json),
        "--output_md",
        str(decision_md),
    ]
    _run(cmd, gpu=-1, log_path=log_path)
    return _read_json(decision_json)


def run_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    policy_root = ROOT / args.policy_root
    output_root = ROOT / args.output_root
    evidence_root = ROOT / args.evidence_root
    phasej_model = _selected_model(policy_root, scene)
    phasej_report_path = phasej_model / "test" / PHASEJ_METHOD / "ela_report.json"
    phasej_report = _read_json(phasej_report_path)
    if not phasej_report:
        raise FileNotFoundError(phasej_report_path)

    evidence_dir = evidence_root / scene
    candidate_model = output_root / scene / "model"
    log_path = output_root / scene / "phasek_barycentric_gate.log"
    _render_maps(args, scene=scene, model=phasej_model, method_name=BASE_METHOD, log_path=log_path)
    _build_evidence(args, scene=scene, phasej_model=phasej_model, evidence_dir=evidence_dir, log_path=log_path)
    _apply_delta(args, phasej_model=phasej_model, evidence_dir=evidence_dir, candidate_model=candidate_model, log_path=log_path)
    _render_maps(args, scene=scene, model=candidate_model, method_name=args.candidate_base_method, log_path=log_path)
    _evaluate_test(args, model=phasej_model, method=BASE_METHOD, log_path=log_path)
    _evaluate_test(args, model=candidate_model, method=args.candidate_base_method, log_path=log_path)

    _apply_ela(
        args,
        scene=scene,
        model=phasej_model,
        base_method=BASE_METHOD,
        method_name=args.phasej_trainval_method,
        phasej_report=phasej_report,
        target_split="train",
        log_path=log_path,
    )
    _evaluate_trainval(
        args,
        model=phasej_model,
        method=args.phasej_trainval_method,
        view_names_file=phasej_model / "train" / args.phasej_trainval_method / "ela_report.json",
        output=phasej_model / "trainval_gate_results.json",
        per_view_output=phasej_model / "trainval_gate_per_view.json",
        log_path=log_path,
    )
    _apply_ela(
        args,
        scene=scene,
        model=candidate_model,
        base_method=args.candidate_base_method,
        method_name=args.candidate_test_method,
        phasej_report=phasej_report,
        target_split="test",
        log_path=log_path,
    )
    _evaluate_test(args, model=candidate_model, method=args.candidate_test_method, log_path=log_path)
    _apply_ela(
        args,
        scene=scene,
        model=candidate_model,
        base_method=args.candidate_base_method,
        method_name=args.candidate_trainval_method,
        phasej_report=phasej_report,
        target_split="train",
        log_path=log_path,
    )
    _evaluate_trainval(
        args,
        model=candidate_model,
        method=args.candidate_trainval_method,
        view_names_file=phasej_model / "train" / args.phasej_trainval_method / "ela_report.json",
        output=candidate_model / "trainval_gate_results.json",
        per_view_output=candidate_model / "trainval_gate_per_view.json",
        log_path=log_path,
    )
    decision = _decide(
        args,
        scene=scene,
        phasej_model=phasej_model,
        candidate_model=candidate_model,
        output_root=output_root,
        log_path=log_path,
    )
    scene_summary = {
        "scene": scene,
        "phasej_model": str(phasej_model.relative_to(ROOT)),
        "candidate_model": str(candidate_model.relative_to(ROOT)),
        "evidence_dir": str(evidence_dir.relative_to(ROOT)),
        "decision": decision,
        "log_path": str(log_path.relative_to(ROOT)),
    }
    (output_root / scene / "phasek_scene_summary.json").write_text(
        json.dumps(scene_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scene_summary


def _write_aggregate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    accepted = [row for row in rows if row["decision"].get("accepted")]
    payload = {"rows": rows, "accepted_count": len(accepted), "total_count": len(rows), "args": vars(args)}
    (output_root / "phasek_barycentric_gate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase-K Barycentric Gate Summary",
        "",
        f"- scenes: `{len(rows)}`",
        f"- accepted: `{len(accepted)}`",
        "",
        "| scene | selected | accepted | raw dPSNR | raw dSSIM | raw dLPIPS | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decision = row["decision"]
        td = decision.get("trainval_delta", {})
        hd = decision.get("test_delta_report_only", {})
        raw_delta = {"PSNR": math.nan, "SSIM": math.nan, "LPIPS": math.nan}
        try:
            phasej_model = ROOT / row["phasej_model"]
            candidate_model = ROOT / row["candidate_model"]
            base_raw = _metric(phasej_model / "results.json", BASE_METHOD)
            candidate_raw = _metric(candidate_model / "results.json", args.candidate_base_method)
            raw_delta = {key: candidate_raw[key] - base_raw[key] for key in ("PSNR", "SSIM", "LPIPS")}
        except Exception:
            pass
        lines.append(
            f"| {row['scene']} | {decision.get('selected_label')} | {str(decision.get('accepted')).lower()} | "
            f"{float(raw_delta.get('PSNR', math.nan)):+.6f} | {float(raw_delta.get('SSIM', math.nan)):+.6f} | {float(raw_delta.get('LPIPS', math.nan)):+.6f} | "
            f"{float(td.get('PSNR', math.nan)):+.6f} | {float(td.get('SSIM', math.nan)):+.6f} | {float(td.get('LPIPS', math.nan)):+.6f} | "
            f"{float(hd.get('PSNR', math.nan)):+.6f} | {float(hd.get('SSIM', math.nan)):+.6f} | {float(hd.get('LPIPS', math.nan)):+.6f} |"
        )
    (output_root / "phasek_barycentric_gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_root", default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix")
    parser.add_argument("--dataset_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument("--output_root", default="outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded")
    parser.add_argument("--evidence_root", default="outputs/carnet/meshsplatopt/ecsr_phase_k/surface_evidence_bary_v2wide")
    parser.add_argument("--scenes", default="garden")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evidence_max_views", type=int, default=8)
    parser.add_argument("--evidence_view_stride", type=int, default=6)
    parser.add_argument("--evidence_view_offset", type=int, default=0)
    parser.add_argument("--evidence_high_error_quantile", type=float, default=0.70)
    parser.add_argument("--delta_top_k", type=int, default=4096)
    parser.add_argument("--delta_min_view_hits", type=int, default=2)
    parser.add_argument("--delta_min_consistency", type=float, default=0.85)
    parser.add_argument("--delta_min_pixel_count", type=float, default=6.0)
    parser.add_argument("--delta_max_samples_per_face_view", type=int, default=64)
    parser.add_argument("--delta_max_total_samples", type=int, default=300000)
    parser.add_argument("--delta_high_error_quantile", type=float, default=0.70)
    parser.add_argument("--delta_strength", type=float, default=0.08)
    parser.add_argument("--delta_max_abs_rgb", type=float, default=0.008)
    parser.add_argument("--delta_operator", choices=("dc", "sh1", "facelocal_sh1"), default="dc")
    parser.add_argument("--candidate_label", default="bary_delta_v2wide_s08")
    parser.add_argument("--delta_max_abs_sh_coeff", type=float, default=0.0)
    parser.add_argument(
        "--delta_uniform_barycentric",
        action="store_true",
        help="For SH1 deltas, use equal face-vertex weights so a broader train-evidence support list can be used without barycentric rerendering.",
    )
    parser.add_argument("--delta_lambda_mag", type=float, default=0.03)
    parser.add_argument("--delta_lambda_sh1_mag", type=float, default=0.06)
    parser.add_argument("--delta_lambda_smooth", type=float, default=0.10)
    parser.add_argument("--delta_steps", type=int, default=800)
    parser.add_argument("--delta_min_policy_val_relative_gain", type=float, default=0.02)
    parser.add_argument("--delta_min_policy_val_samples", type=int, default=512)
    parser.add_argument("--delta_min_policy_val_unique_faces", type=int, default=16)
    parser.add_argument("--delta_max_faces_to_apply", type=int, default=2048)
    parser.add_argument("--delta_min_face_policy_val_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_min_face_policy_val_samples", type=int, default=8)
    parser.add_argument(
        "--delta_sh1_face_policy",
        action="store_true",
        help="For shared-vertex SH1 deltas, only write faces that pass fixed policy-val per-face certificates.",
    )
    parser.add_argument(
        "--delta_min_face_view_consensus",
        type=float,
        default=0.0,
        help=(
            "For SH1-family deltas, require this fraction of policy-val train views "
            "to agree with a face residual direction before applying the face update."
        ),
    )
    parser.add_argument("--delta_min_face_consensus_views", type=int, default=2)
    parser.add_argument("--delta_min_face_consensus_view_samples", type=int, default=4)
    parser.add_argument("--delta_face_consensus_min_cosine", type=float, default=0.0)
    parser.add_argument(
        "--delta_min_face_gain_certificate_views",
        type=int,
        default=0,
        help=(
            "For face-local SH1 deltas, require each accepted face to have predicted "
            "residual MSE gain on at least this many policy-val train views. 0 disables it."
        ),
    )
    parser.add_argument("--delta_min_face_gain_certificate_relative_gain", type=float, default=0.0)
    parser.add_argument("--delta_min_face_gain_certificate_view_samples", type=int, default=4)
    parser.add_argument("--delta_min_face_gain_certificate_fraction", type=float, default=0.0)
    parser.add_argument("--phasej_trainval_method", default="ours_26000_phasej_trainval_gate")
    parser.add_argument("--candidate_base_method", default="ours_26000_bary_delta_v2wide_s08_base")
    parser.add_argument("--candidate_test_method", default="ours_26000_bary_delta_v2wide_s08_phasej_ela")
    parser.add_argument("--candidate_trainval_method", default="ours_26000_bary_delta_v2wide_s08_phasej_trainval_gate")
    parser.add_argument("--policy_holdout_fraction", type=float, default=0.25)
    parser.add_argument("--policy_holdout_offset", type=int, default=0)
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="uniform")
    parser.add_argument("--calib_max_views", type=int, default=32)
    parser.add_argument("--calib_stride", type=int, default=1)
    parser.add_argument("--alpha_feature_mode", choices=("confidence_magnitude", "confidence_magnitude_edge"), default="confidence_magnitude_edge")
    parser.add_argument("--alpha_default", type=float, default=0.0)
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=-1.0e9)
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phasek_barycentric_multiscene")
    parser.add_argument("--wandb_name", default="phasek_barycentric")
    args = parser.parse_args()
    scenes = [scene.strip() for scene in str(args.scenes).replace(" ", ",").split(",") if scene.strip()]
    rows = [run_scene(args, scene) for scene in scenes]
    _write_aggregate(args, rows)
    print(json.dumps({"rows": len(rows), "output_root": str(ROOT / args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

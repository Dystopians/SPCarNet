#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT_METHOD = "ours_26000_v100_checkpoint_attached_ela_endpoint"
EXPECTED_FIELD_TYPE = "v102_surface_residual_field"
EXPECTED_BASIS_TYPE = "affine_barycentric_viewdir_mixture"
EXPECTED_BUILDER_VARIANT = "v105_evidence_gated_residual_mixture"
V108_METHOD_VERSION = "v108_mse_descent_locked_pod_moe"
V108_BUILDER_VARIANT = "v108_mse_descent_locked_pod_moe"
V108_EXPERT_MSE_CERTIFICATE = "joint_two_expert_weighted_normal_equation_box_qp_descent_lock"


def _expected_basis_type(args: argparse.Namespace) -> str:
    return "affine_barycentric_viewdir_pod_mixture" if str(args.field_variant) == "pod_moe" else EXPECTED_BASIS_TYPE


def _expected_builder_variant(args: argparse.Namespace) -> str:
    if _method_version(args) == V108_METHOD_VERSION:
        return V108_BUILDER_VARIANT
    return "v106_perceptual_occlusion_detail_moe" if str(args.field_variant) == "pod_moe" else EXPECTED_BUILDER_VARIANT


def _method_version(args: argparse.Namespace) -> str:
    if str(getattr(args, "method_version", "auto") or "auto") == V108_METHOD_VERSION:
        return V108_METHOD_VERSION
    if str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk":
        return "v107_crossfit_pod_moe_expert_reliability"
    if str(args.field_variant) == "pod_moe":
        return "v106_perceptual_occlusion_detail_moe"
    return EXPECTED_BUILDER_VARIANT


def _expected_pod_view_gate_mode(args: argparse.Namespace) -> str:
    if str(args.field_variant) != "pod_moe":
        return ""
    if str(args.gate_source) == "crossfit_risk":
        return "temperature_controlled"
    return "implicit_unit_temperature"


def _report_stem(args: argparse.Namespace) -> str:
    if _method_version(args) == V108_METHOD_VERSION:
        return "v108_mse_descent_locked_pod_moe_report"
    if str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk":
        return "v107_crossfit_pod_moe_report"
    if str(args.field_variant) == "pod_moe":
        return "v106_pod_moe_report"
    return "v105_evidence_gated_mixture_report"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_token(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "p").replace("-", "m")


def _field_filename(args: argparse.Namespace) -> str:
    if _method_version(args) == V108_METHOD_VERSION:
        prefix = "v108_mse_descent_locked_podmoe"
    elif str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk":
        prefix = "v107_podmoe_crossfit"
    else:
        prefix = "v106_podmoe" if str(args.field_variant) == "pod_moe" else "v105_egmix"
    return (
        f"{prefix}_mc1_mv1"
        f"_r{_float_token(args.ridge)}"
        f"_clip{_float_token(args.residual_clip)}"
        f"_vs{_float_token(args.view_std_floor)}"
        f"_rr{_float_token(args.rank_rtol)}"
        f"_cm{_float_token(args.condition_max)}"
        f"_gb{_float_token(args.gate_boost)}"
        f"_{args.gate_source}"
        f"_vgt{_float_token(args.view_gate_temperature)}"
        f"_{args.residual_dtype}"
        f"_s{int(args.renderer_scaling)}"
        "_field.pt"
    )


def _same_float(actual: Any, expected: float) -> bool:
    try:
        actual_f = float(actual)
    except (TypeError, ValueError):
        return False
    expected_f = float(expected)
    return abs(actual_f - expected_f) <= max(1e-12, abs(expected_f) * 1e-6)


def _strict_pod_identity(args: argparse.Namespace) -> bool:
    return str(args.field_variant) == "pod_moe" and (
        _method_version(args) == V108_METHOD_VERSION or str(args.gate_source) == "crossfit_risk"
    )


def _expected_expert_mse_certificate(args: argparse.Namespace) -> str:
    if _method_version(args) == V108_METHOD_VERSION:
        return V108_EXPERT_MSE_CERTIFICATE
    if str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk":
        return "v107_crossfit_heldout_weighted_normal_equation_lambda_star"
    return ""


def _validate_v105_field_manifest(manifest: dict[str, Any], field: Path, args: argparse.Namespace) -> dict[str, bool]:
    solve_stats = manifest.get("solve_stats", {}) if isinstance(manifest.get("solve_stats"), dict) else {}
    expected_certificate = _expected_expert_mse_certificate(args)
    checks = {
        "manifest_present": bool(manifest),
        "field_type": str(manifest.get("field_type", "")) == EXPECTED_FIELD_TYPE,
        "basis_type": str(manifest.get("basis_type", "")) == _expected_basis_type(args),
        "builder_variant": str(manifest.get("builder_variant", "")) == _expected_builder_variant(args),
        "field_variant": str(manifest.get("field_variant", "")) == str(args.field_variant),
        "method_version": (
            str(manifest.get("method_version", "")) == _method_version(args)
            if _strict_pod_identity(args)
            else str(manifest.get("method_version", _method_version(args)) or _method_version(args)) == _method_version(args)
        ),
        "field_sha256": bool(field.is_file())
        and str(manifest.get("field_sha256", "")) == _sha256(field),
        "min_count": int(manifest.get("min_count", -1) or -1) == 1,
        "min_views": int(manifest.get("min_views", -1) or -1) == 1,
        "ridge": _same_float(manifest.get("ridge"), args.ridge),
        "residual_clip": _same_float(manifest.get("residual_clip"), args.residual_clip),
        "view_std_floor": _same_float(manifest.get("view_std_floor"), args.view_std_floor),
        "rank_rtol": _same_float(manifest.get("rank_rtol"), args.rank_rtol),
        "condition_max": _same_float(manifest.get("condition_max"), args.condition_max),
        "gate_boost": _same_float(manifest.get("gate_boost"), args.gate_boost),
        "gate_source": str(manifest.get("gate_source", "")) == str(args.gate_source),
        "view_gate_temperature": _same_float(manifest.get("view_gate_temperature"), args.view_gate_temperature),
        "pod_view_gate_mode": (
            str(manifest.get("pod_view_gate_mode", "")) == _expected_pod_view_gate_mode(args)
            if _strict_pod_identity(args)
            else True
        ),
        "expert_mse_certificate": (
            str(manifest.get("expert_mse_certificate", "") or solve_stats.get("expert_mse_certificate", ""))
            == expected_certificate
            if expected_certificate
            else True
        ),
        "renderer_scaling": int(manifest.get("renderer_scaling", -1) or -1) == int(args.renderer_scaling),
        "residual_dtype": str(manifest.get("residual_dtype", "")) == str(args.residual_dtype),
        "endpoint_method": str(manifest.get("endpoint_method", "")) == str(args.endpoint_method),
    }
    return checks


def _run_cmd(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - started
        handle.write(f"\n[exit_code] {proc.returncode}\n[wall_sec] {elapsed:.6f}\n")
    return int(proc.returncode), float(elapsed)


def _metrics_delta(lhs: dict[str, Any], rhs: dict[str, Any], key: str) -> float | None:
    if key not in lhs or key not in rhs:
        return None
    return float(lhs[key]) - float(rhs[key])


def _load_clean_metrics(scene: str, clean_root: Path) -> dict[str, Any]:
    payload = _read_json(clean_root / scene / "results.json")
    metrics = payload.get("ours_26000", {})
    return metrics if isinstance(metrics, dict) else {}


def run_scene(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    if int(args.renderer_scaling) != 4:
        raise RuntimeError("v105 runner requires renderer_scaling=4 because render.py uses TriangleModel.scaling=4")
    model = Path(args.package_root) / scene / "detached_model"
    if not model.is_dir():
        raise FileNotFoundError(model)
    v102_bank = Path(args.v102_bank_root) / scene / "v102_preprojected_delta_bank.pt"
    field = Path(args.field_root) / scene / _field_filename(args)
    report_root = Path(args.report_root) / scene
    report_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if str(args.gpu).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TMPDIR", "/tmp")

    v102_rc = 0
    v102_wall = 0.0
    if (args.force_v102 or not v102_bank.is_file()) and bool(args.build_v102_if_missing):
        v102_cmd = [
            sys.executable,
            "scripts/car_model/run_v102_preprojected_delta_scene.py",
            "--scene",
            scene,
            "--package_root",
            str(args.package_root),
            "--v102_bank_root",
            str(args.v102_bank_root),
            "--report_root",
            str(args.v102_report_root),
            "--endpoint_method",
            str(args.endpoint_method),
            "--iteration",
            str(args.iteration),
            "--delta_dtype",
            "float32",
            "--gpu",
            str(args.gpu),
            "--no_intermediate_outputs",
        ]
        if args.force_v102:
            v102_cmd.append("--force")
        v102_rc, v102_wall = _run_cmd(v102_cmd, report_root / f"{scene}_v102.log", env)
    elif not v102_bank.is_file():
        raise FileNotFoundError(f"missing v102 bank and --build_v102_if_missing not set: {v102_bank}")

    field_rc = 999
    field_wall = 0.0
    if v102_rc == 0 and (args.force_field or not field.is_file()):
        field_rc, field_wall = _run_cmd(
            [
                sys.executable,
                "scripts/car_model/build_v105_evidence_gated_mixture_field.py",
                "--model_path",
                str(model),
                "--delta_bank_path",
                str(v102_bank),
                "--output_field",
                str(field),
                "--endpoint_method",
                str(args.endpoint_method),
                "--iteration",
                str(args.iteration),
                "--split",
                "test",
                "--renderer_scaling",
                str(args.renderer_scaling),
                "--residual_dtype",
                str(args.residual_dtype),
                "--field_variant",
                str(args.field_variant),
                "--method_version",
                str(getattr(args, "method_version", "auto") or "auto"),
                "--min_count",
                "1",
                "--min_views",
                "1",
                "--ridge",
                str(args.ridge),
                "--residual_clip",
                str(args.residual_clip),
                "--view_std_floor",
                str(args.view_std_floor),
                "--rank_rtol",
                str(args.rank_rtol),
                "--condition_max",
                str(args.condition_max),
                "--gate_boost",
                str(args.gate_boost),
                "--gate_source",
                str(args.gate_source),
                "--view_gate_temperature",
                str(args.view_gate_temperature),
                "--chunk_pixels",
                str(args.chunk_pixels),
            ],
            report_root / f"{scene}_field.log",
            env,
        )
    elif field.is_file():
        field_rc = 0

    manifest = _read_json(field.with_suffix(".manifest.json")) if field_rc == 0 else {}
    field_identity_checks = _validate_v105_field_manifest(manifest, field, args) if field_rc == 0 else {}
    failed_field_checks = [name for name, ok in field_identity_checks.items() if not ok]
    if field_rc == 0 and failed_field_checks:
        raise RuntimeError(
            "v105 field manifest failed identity/parameter checks: "
            + ", ".join(failed_field_checks)
            + f" ({field.with_suffix('.manifest.json')})"
        )

    output_label = _method_version(args) if str(args.field_variant) == "pod_moe" else f"{args.field_variant}_{args.gate_source}"
    output_method = args.output_method or f"ours_{int(args.iteration)}_{output_label}_surface_field_{scene}"
    render_rc = 999
    render_wall = 0.0
    render_report_path = model / "test" / output_method / "render_py_endpoint_report.json"
    if field_rc == 0 and (args.force_render or not render_report_path.is_file()):
        render_rc, render_wall = _run_cmd(
            [
                sys.executable,
                "render.py",
                "-m",
                str(model),
                "--iteration",
                str(args.iteration),
                "--skip_train",
                "--checkpoint_endpoint_method",
                str(args.endpoint_method),
                "--checkpoint_endpoint_output_method",
                output_method,
                "--checkpoint_endpoint_surface_field_path",
                str(field),
                "--checkpoint_endpoint_require_surface_field",
                "--checkpoint_endpoint_no_intermediate_outputs",
                "--quiet",
            ],
            report_root / f"{scene}_render.log",
            env,
        )
    elif render_report_path.is_file():
        render_rc = 0

    eval_rc = 999
    eval_wall = 0.0
    results_before = _read_json(model / "results.json")
    if render_rc == 0 and (args.force_eval or output_method not in results_before):
        eval_rc, eval_wall = _run_cmd(
            [
                sys.executable,
                "scripts/car_model/evaluate_render_split_metrics.py",
                "-m",
                str(model),
                "--split",
                "test",
                "--methods",
                output_method,
                "--merge_model_results",
            ],
            report_root / f"{scene}_eval.log",
            env,
        )
    elif output_method in results_before:
        eval_rc = 0

    results = _read_json(model / "results.json")
    metrics = results.get(output_method, {}) if isinstance(results.get(output_method), dict) else {}
    clean_metrics = _load_clean_metrics(scene, Path(args.clean_root))
    reference_key = args.reference_method or f"ours_{int(args.iteration)}_v101_detached_package_full9_{scene}"
    reference_metrics = results.get(reference_key, {}) if isinstance(results.get(reference_key), dict) else {}
    v102_key = f"ours_{int(args.iteration)}_v102_delta_fast_{scene}"
    v102_metrics = results.get(v102_key, {}) if isinstance(results.get(v102_key), dict) else {}
    v104c_key = f"ours_{int(args.iteration)}_v104c_shrink_view_affine_min1_minviews1_{scene}"
    v104c_metrics = results.get(v104c_key, {}) if isinstance(results.get(v104c_key), dict) else {}
    render_report = _read_json(render_report_path)
    render_field = render_report.get("surface_residual_field", {})
    render_field = render_field if isinstance(render_field, dict) else {}
    expected_certificate = _expected_expert_mse_certificate(args)
    render_identity_checks = {
        "mode": str(render_report.get("mode")) == "surface_residual_field_endpoint",
        "support_source": str(render_report.get("support_source", "")).startswith("v102_surface_residual_field:"),
        "no_test_gt_used_for_policy": bool(render_report.get("no_test_gt_used_for_policy")),
        "field_sha256": str(render_field.get("field_sha256", "")) == str(manifest.get("field_sha256", "")),
        "field_type": str(render_field.get("field_type", "")) == EXPECTED_FIELD_TYPE,
        "basis_type": str(render_field.get("basis_type", "")) == _expected_basis_type(args),
        "builder_variant": str(render_field.get("builder_variant", "")) == _expected_builder_variant(args),
        "field_variant": str(render_field.get("field_variant", "")) == str(args.field_variant),
        "method_version": (
            str(render_field.get("method_version", "")) == _method_version(args)
            if _strict_pod_identity(args)
            else str(render_field.get("method_version", _method_version(args)) or _method_version(args))
            == _method_version(args)
        ),
        "renderer_scaling": int(render_field.get("renderer_scaling", -1) or -1) == int(args.renderer_scaling),
        "residual_clip": _same_float(render_field.get("residual_clip"), args.residual_clip),
        "gate_boost": _same_float(render_field.get("gate_boost"), args.gate_boost),
        "gate_source": str(render_field.get("gate_source", "")) == str(args.gate_source),
        "view_gate_temperature": _same_float(render_field.get("view_gate_temperature"), args.view_gate_temperature),
        "pod_view_gate_mode": (
            str(render_field.get("pod_view_gate_mode", "")) == _expected_pod_view_gate_mode(args)
            if _strict_pod_identity(args)
            else True
        ),
        "pod_crossfit_split": (
            str(render_field.get("pod_crossfit_split", "")) == "target_view_even_odd"
            if str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk"
            else True
        ),
        "expert_reliability_variant": (
            str(render_field.get("expert_reliability_variant", "")) == "v107_crossfit_heldout_weighted_risk"
            if str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk"
            else True
        ),
        "expert_mse_certificate": (
            str(render_field.get("expert_mse_certificate", "")) == expected_certificate
            if expected_certificate
            else True
        ),
    }
    solve_stats = manifest.get("solve_stats", {}) if isinstance(manifest.get("solve_stats"), dict) else {}
    required_metrics = {"PSNR", "SSIM", "LPIPS"}
    field_stats = {
        "field_variant": manifest.get("field_variant"),
        "method_version": manifest.get("method_version"),
        "pod_expert_reliability_variant": manifest.get("pod_expert_reliability_variant"),
        "basis_type": manifest.get("basis_type"),
        "builder_variant": manifest.get("builder_variant"),
        "base_variant": solve_stats.get("base_variant"),
        "expert_names": solve_stats.get("expert_names"),
        "expert_reliability_variant": solve_stats.get("expert_reliability_variant"),
        "expert_reliability_combine": solve_stats.get("expert_reliability_combine"),
        "expert_mse_certificate": solve_stats.get("expert_mse_certificate"),
        "pod_crossfit_split": solve_stats.get("pod_crossfit_split"),
        "pod_base_keep_mode": solve_stats.get("pod_base_keep_mode", manifest.get("pod_base_keep_mode")),
        "pod_view_gate_mode": solve_stats.get("pod_view_gate_mode", manifest.get("pod_view_gate_mode")),
        "valid_triangles": manifest.get("valid_triangles"),
        "total_accumulated_pixels": manifest.get("total_accumulated_pixels"),
        "mixture_triangles": solve_stats.get("mixture_triangles"),
        "fallback_only_triangles": solve_stats.get("fallback_only_triangles"),
        "gate_mean": solve_stats.get("gate_mean"),
        "gain_score_mean": solve_stats.get("gain_score_mean"),
        "gate_source": solve_stats.get("gate_source", manifest.get("gate_source")),
        "crossfit_gain_mean": solve_stats.get("crossfit_gain_mean"),
        "crossfit_gain_supported_triangles": solve_stats.get("crossfit_gain_supported_triangles"),
        "stability_score_mean": solve_stats.get("stability_score_mean"),
        "debt_guard_mean": solve_stats.get("debt_guard_mean"),
        "detail_triangles": solve_stats.get("detail_triangles"),
        "boundary_triangles": solve_stats.get("boundary_triangles"),
        "detail_reliability_mean": solve_stats.get("detail_reliability_mean"),
        "boundary_reliability_mean": solve_stats.get("boundary_reliability_mean"),
        "detail_gain_mean": solve_stats.get("detail_gain_mean"),
        "boundary_gain_mean": solve_stats.get("boundary_gain_mean"),
        "detail_full_gain_mean": solve_stats.get("detail_full_gain_mean"),
        "boundary_full_gain_mean": solve_stats.get("boundary_full_gain_mean"),
        "detail_mse_scale_mean": solve_stats.get("detail_mse_scale_mean"),
        "boundary_mse_scale_mean": solve_stats.get("boundary_mse_scale_mean"),
        "detail_prelock_mse_scale_mean": solve_stats.get("detail_prelock_mse_scale_mean"),
        "boundary_prelock_mse_scale_mean": solve_stats.get("boundary_prelock_mse_scale_mean"),
        "joint_descent_supported_triangles": solve_stats.get("joint_descent_supported_triangles"),
        "joint_descent_active_triangles": solve_stats.get("joint_descent_active_triangles"),
        "joint_descent_detail_active_triangles": solve_stats.get("joint_descent_detail_active_triangles"),
        "joint_descent_boundary_active_triangles": solve_stats.get("joint_descent_boundary_active_triangles"),
        "joint_descent_scale_mean": solve_stats.get("joint_descent_scale_mean"),
        "joint_descent_active_scale_mean": solve_stats.get("joint_descent_active_scale_mean"),
        "joint_descent_gain_mean": solve_stats.get("joint_descent_gain_mean"),
        "joint_descent_active_gain_mean": solve_stats.get("joint_descent_active_gain_mean"),
        "joint_descent_objective_delta_mean": solve_stats.get("joint_descent_objective_delta_mean"),
        "detail_debt_guard_mean": solve_stats.get("detail_debt_guard_mean"),
        "boundary_debt_guard_mean": solve_stats.get("boundary_debt_guard_mean"),
        "detail_weighted_pixels": solve_stats.get("detail_weighted_pixels"),
        "boundary_weighted_pixels": solve_stats.get("boundary_weighted_pixels"),
        "detail_crossfit_supported_triangles": solve_stats.get("detail_crossfit_supported_triangles"),
        "boundary_crossfit_supported_triangles": solve_stats.get("boundary_crossfit_supported_triangles"),
        "detail_crossfit_gain_mean": solve_stats.get("detail_crossfit_gain_mean"),
        "boundary_crossfit_gain_mean": solve_stats.get("boundary_crossfit_gain_mean"),
        "detail_crossfit_mse_scale_mean": solve_stats.get("detail_crossfit_mse_scale_mean"),
        "boundary_crossfit_mse_scale_mean": solve_stats.get("boundary_crossfit_mse_scale_mean"),
        "base_observed_triangles": solve_stats.get("base_observed_triangles"),
        "base_mixture_triangles": solve_stats.get("base_mixture_triangles"),
        "base_gate_mean": solve_stats.get("base_gate_mean"),
        "base_gain_score_mean": solve_stats.get("base_gain_score_mean"),
        "base_debt_guard_mean": solve_stats.get("base_debt_guard_mean"),
        "elapsed_sec": manifest.get("elapsed_sec"),
    }
    if _method_version(args) == V108_METHOD_VERSION:
        report_label = "v108 MSE-Descent-Locked POD-MoE Surface Field"
        claim_boundary = (
            "v108 POD-MoE diagnostic target-delta surface field uses v102 target-camera endpoint deltas as teacher. "
            "It stores a v104c-like base plus full-data detail and occlusion-boundary experts, and scales the rendered "
            "expert corrections with a joint two-expert weighted normal-equation MSE descent box-QP certificate. It "
            "uses no held-out target GT for the policy, but it is not a train-only unseen-camera field."
        )
    elif str(args.field_variant) == "pod_moe" and str(args.gate_source) == "crossfit_risk":
        report_label = "v107 Cross-Fitted POD-MoE Surface Field"
        claim_boundary = (
            "v107 POD-MoE diagnostic target-delta surface field uses v102 target-camera endpoint deltas as teacher. "
            "It stores a v104c-like base plus full-data detail and occlusion-boundary experts, with expert "
            "reliability/gain/scale certified by even/odd held-out weighted risk. It uses no held-out target GT "
            "for the policy, but it is not a train-only unseen-camera field."
        )
    elif str(args.field_variant) == "pod_moe":
        report_label = "v106 POD-MoE Surface Field"
        claim_boundary = (
            "v106 POD-MoE diagnostic target-delta surface field uses v102 target-camera endpoint deltas as teacher. "
            "It stores a v104c-like base plus detail and occlusion-boundary experts. It uses no held-out target GT "
            "for the policy, but it is not a train-only unseen-camera field."
        )
    else:
        report_label = "v105 Evidence-Gated Mixture"
        claim_boundary = (
            "v105 diagnostic target-delta mixture field uses v102 target-camera endpoint deltas as teacher. "
            "It uses no held-out target GT for the policy, but it is not a train-only unseen-camera field."
        )
    payload = {
        "schema_version": 1,
        "scene": scene,
        "model": str(model),
        "endpoint_method": str(args.endpoint_method),
        "iteration": int(args.iteration),
        "output_method": output_method,
        "v102_bank": str(v102_bank),
        "field": str(field),
        "field_manifest": str(field.with_suffix(".manifest.json")),
        "field_identity": {
            "checks": field_identity_checks,
            "expected": {
                "field_type": EXPECTED_FIELD_TYPE,
                "basis_type": _expected_basis_type(args),
                "builder_variant": _expected_builder_variant(args),
                "field_variant": str(args.field_variant),
                "method_version": _method_version(args),
                "renderer_scaling": int(args.renderer_scaling),
                "residual_dtype": str(args.residual_dtype),
                "ridge": float(args.ridge),
                "residual_clip": float(args.residual_clip),
                "view_std_floor": float(args.view_std_floor),
                "rank_rtol": float(args.rank_rtol),
                "condition_max": float(args.condition_max),
                "gate_boost": float(args.gate_boost),
                "gate_source": str(args.gate_source),
                "view_gate_temperature": float(args.view_gate_temperature),
                "pod_view_gate_mode": _expected_pod_view_gate_mode(args),
                "expert_mse_certificate": _expected_expert_mse_certificate(args),
            },
            "manifest": {
                "field_sha256": manifest.get("field_sha256"),
                "field_type": manifest.get("field_type"),
                "basis_type": manifest.get("basis_type"),
                "builder_variant": manifest.get("builder_variant"),
                "field_variant": manifest.get("field_variant"),
                "method_version": manifest.get("method_version"),
                "pod_view_gate_mode": manifest.get("pod_view_gate_mode"),
                "expert_mse_certificate": manifest.get("expert_mse_certificate"),
                "renderer_scaling": manifest.get("renderer_scaling"),
                "residual_dtype": manifest.get("residual_dtype"),
                "gate_source": manifest.get("gate_source"),
            },
        },
        "render_report": str(render_report_path),
        "build_v102_if_missing": bool(args.build_v102_if_missing),
        "return_codes": {
            "v102": int(v102_rc),
            "field": int(field_rc),
            "render": int(render_rc),
            "eval": int(eval_rc),
        },
        "wall_sec": {
            "v102": float(v102_wall),
            "field": float(field_wall),
            "render": float(render_wall),
            "eval": float(eval_wall),
        },
        "metrics": metrics,
        "clean_metrics": clean_metrics,
        "v104c_key": v104c_key,
        "v104c_metrics": v104c_metrics,
        "reference_key": reference_key,
        "reference_metrics": reference_metrics,
        "v102_key": v102_key,
        "v102_metrics": v102_metrics,
        "deltas": {
            "vs_clean": {
                "PSNR": _metrics_delta(metrics, clean_metrics, "PSNR"),
                "SSIM": _metrics_delta(metrics, clean_metrics, "SSIM"),
                "LPIPS": _metrics_delta(metrics, clean_metrics, "LPIPS"),
            },
            "vs_v104c": {
                "PSNR": _metrics_delta(metrics, v104c_metrics, "PSNR"),
                "SSIM": _metrics_delta(metrics, v104c_metrics, "SSIM"),
                "LPIPS": _metrics_delta(metrics, v104c_metrics, "LPIPS"),
            },
            "vs_v101_v102a": {
                "PSNR": _metrics_delta(metrics, reference_metrics, "PSNR"),
                "SSIM": _metrics_delta(metrics, reference_metrics, "SSIM"),
                "LPIPS": _metrics_delta(metrics, reference_metrics, "LPIPS"),
            },
        },
        "field_stats": field_stats,
        "render_stats": {
            "mode": render_report.get("mode"),
            "support_source": render_report.get("support_source"),
            "target_frames": render_report.get("target_frames"),
            "elapsed_sec": render_report.get("elapsed_sec"),
            "mean_abs_delta": render_report.get("mean_abs_delta"),
            "mean_changed_fraction": render_report.get("mean_changed_fraction"),
            "mean_surface_valid_fraction": render_report.get("mean_surface_valid_fraction"),
            "no_test_gt_used_for_policy": render_report.get("no_test_gt_used_for_policy"),
            "surface_residual_field": render_field,
            "identity_checks": render_identity_checks,
        },
        "report_label": report_label,
        "claim_boundary": claim_boundary,
        "logs": {
            "v102": str(report_root / f"{scene}_v102.log"),
            "field": str(report_root / f"{scene}_field.log"),
            "render": str(report_root / f"{scene}_render.log"),
            "eval": str(report_root / f"{scene}_eval.log"),
        },
    }
    payload["passed"] = bool(
        v102_rc == 0
        and field_rc == 0
        and render_rc == 0
        and eval_rc == 0
        and all(field_identity_checks.values())
        and all(render_identity_checks.values())
        and required_metrics.issubset(metrics.keys())
        and required_metrics.issubset(clean_metrics.keys())
        and required_metrics.issubset(v104c_metrics.keys())
    )
    report_stem = _report_stem(args)
    out_json = report_root / f"{scene}_{report_stem}.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = report_root / f"{scene}_{report_stem}.md"
    out_md.write_text(
        "\n".join(
            [
                f"# {report_label} Report: {scene}",
                "",
                f"- passed: `{payload['passed']}`",
                f"- metrics: `{metrics}`",
                f"- delta vs clean: `{payload['deltas']['vs_clean']}`",
                f"- delta vs v104c: `{payload['deltas']['vs_v104c']}`",
                f"- delta vs v101/v102a: `{payload['deltas']['vs_v101_v102a']}`",
                f"- field stats: `{payload['field_stats']}`",
                f"- render stats: `{payload['render_stats']}`",
                f"- claim boundary: {payload['claim_boundary']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out_json), "passed": payload["passed"], "metrics": metrics}, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/render/evaluate v105 evidence-gated mixture field.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--package_root", default="/dev/shm/peilincai_spcarnet_v101_detached_package_full9_20260625")
    parser.add_argument("--v102_bank_root", default="/dev/shm/peilincai_spcarnet_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--field_root", default="/dev/shm/peilincai_spcarnet_v105_evidence_gated_mixture_field_20260625")
    parser.add_argument("--report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v105_evidence_gated_mixture_hardtriad_20260625")
    parser.add_argument("--v102_report_root", default="outputs/carnet/meshsplatopt/ecsr_phase_v102_preprojected_delta_bank_20260625")
    parser.add_argument("--clean_root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--endpoint_method", default=ENDPOINT_METHOD)
    parser.add_argument("--reference_method", default="")
    parser.add_argument("--output_method", default="")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--renderer_scaling", type=int, default=4)
    parser.add_argument("--residual_dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--field_variant", default="residual_mixture", choices=("residual_mixture", "pod_moe"))
    parser.add_argument("--method_version", default="auto", choices=("auto", V108_METHOD_VERSION))
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--residual_clip", type=float, default=0.08)
    parser.add_argument("--view_std_floor", type=float, default=1e-4)
    parser.add_argument("--rank_rtol", type=float, default=1e-7)
    parser.add_argument("--condition_max", type=float, default=1e8)
    parser.add_argument("--gate_boost", type=float, default=0.5)
    parser.add_argument(
        "--gate_source",
        default=None,
        choices=("normal_equation", "crossfit_risk", "optimal_risk"),
        help=(
            "Gate/reliability source. Defaults to normal_equation for pod_moe "
            "to preserve v106 behavior, and crossfit_risk for residual_mixture."
        ),
    )
    parser.add_argument("--view_gate_temperature", type=float, default=0.0)
    parser.add_argument("--chunk_pixels", type=int, default=262144)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--build_v102_if_missing", action="store_true")
    parser.add_argument("--force_v102", action="store_true")
    parser.add_argument("--force_field", action="store_true")
    parser.add_argument("--force_render", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    args = parser.parse_args()
    if args.gate_source is None:
        if str(args.method_version) == V108_METHOD_VERSION:
            args.gate_source = "crossfit_risk"
        else:
            args.gate_source = "normal_equation" if str(args.field_variant) == "pod_moe" else "crossfit_risk"
    if str(args.method_version) == V108_METHOD_VERSION and str(args.field_variant) != "pod_moe":
        parser.error(f"--method_version {V108_METHOD_VERSION} requires --field_variant pod_moe")
    if str(args.method_version) == V108_METHOD_VERSION and str(args.gate_source) not in {"normal_equation", "crossfit_risk"}:
        parser.error(f"--method_version {V108_METHOD_VERSION} requires --gate_source normal_equation or crossfit_risk")
    return args


def main() -> int:
    payload = run_scene(parse_args())
    return 0 if payload.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())

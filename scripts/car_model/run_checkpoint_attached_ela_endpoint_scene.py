#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageChops, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.benchmark_ela_postprocess_runtime import (
    _alpha_calibrator_from_report,
    _alpha_from_report,
    _benefit_calibrator_from_report,
    _json_safe,
    _policy_from_report,
    _read_report,
    _select_support_frames,
)
from utils.evidence_lumigraph_adapter import (
    FrameLoader,
    adapt_frame,
    load_split_frames,
    save_image_tensor,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _metric(payload: dict[str, Any], method: str) -> dict[str, float]:
    item = payload.get(method, {})
    return {
        "PSNR": _finite_float(item.get("PSNR")),
        "SSIM": _finite_float(item.get("SSIM")),
        "LPIPS": _finite_float(item.get("LPIPS")),
    }


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values: list[float]) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _copy_or_link(src: Path, dst: Path, *, required: bool = False) -> str:
    if not src.exists():
        if required:
            raise FileNotFoundError(f"required model asset is missing: {src}")
        return "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            same_target = dst.resolve() == src.resolve()
        except OSError:
            same_target = False
        if dst.is_symlink() and not same_target:
            raise RuntimeError(f"stale symlink at {dst}: points to {dst.resolve()}, expected {src.resolve()}")
        return "existing"
    try:
        os.symlink(src.resolve(), dst)
        return "symlink"
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
        return "copy"


def _prepare_candidate_model(base_model: Path, output_model: Path, iteration: int) -> dict[str, Any]:
    output_model.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for name in ("cfg_args", "cameras.json", "input.ply", "topology_audit.json", "topology_audit.md"):
        src = base_model / name
        if src.exists():
            links[name] = _copy_or_link(src, output_model / name, required=name in {"cfg_args", "cameras.json"})

    ckpt_name = "point_cloud_state_dict.pt"
    src_ckpt = base_model / "point_cloud" / f"iteration_{int(iteration)}" / ckpt_name
    dst_ckpt = output_model / "point_cloud" / f"iteration_{int(iteration)}" / ckpt_name
    links[str(dst_ckpt.relative_to(output_model))] = _copy_or_link(src_ckpt, dst_ckpt, required=True)
    manifest = {
        "model_link_manifest_version": 1,
        "created_at_unix": time.time(),
        "base_model": str(base_model),
        "output_model": str(output_model),
        "iteration": int(iteration),
        "linked_or_copied": links,
        "checkpoint_attachment_note": (
            "The candidate model keeps MeshSplatting geometry/checkpoint inherited from the base model "
            "and attaches a render residual endpoint sidecar under point_cloud/iteration_*/render_residual_endpoint."
        ),
    }
    (output_model / "checkpoint_attached_model_links.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validate_source_report_provenance(
    report: dict[str, Any],
    *,
    source_report_path: Path,
    base_model: Path,
    base_method: str,
    train_frames,
) -> dict[str, Any]:
    train_names: set[str] = set()
    for frame in train_frames:
        train_names.add(str(frame.name))
        train_names.add(str(frame.camera.image_name))
        train_names.add(Path(str(frame.camera.image_name)).stem)

    def _names_subset(key: str) -> tuple[bool, list[str]]:
        value = report.get(key)
        if not isinstance(value, list):
            return True, []
        missing = [
            str(item)
            for item in value
            if str(item) not in train_names and Path(str(item)).stem not in train_names
        ]
        return not missing, missing

    checks: list[dict[str, Any]] = []
    if report.get("base_model_path"):
        source_base = Path(str(report["base_model_path"])).resolve()
        checks.append(
            {
                "name": "base_model_matches",
                "pass": source_base == base_model.resolve(),
                "source_base_model_path": str(source_base),
                "expected_base_model_path": str(base_model.resolve()),
            }
        )
    checks.append(
        {
            "name": "base_method_matches",
            "pass": str(report.get("base_method", "")) in {"", str(base_method)},
            "source_base_method": str(report.get("base_method", "")),
            "expected_base_method": str(base_method),
        }
    )
    for key in ("policy_fit_views", "policy_val_views", "adapt_support_view_names"):
        ok, missing = _names_subset(key)
        checks.append(
            {
                "name": f"{key}_subset_of_train",
                "pass": bool(ok),
                "missing": missing[:16],
                "missing_count": len(missing),
            }
        )
    leakage_flags = {
        "uses_test_gt_for_branch": report.get("uses_test_gt_for_branch"),
        "uses_test_gt_for_policy": report.get("uses_test_gt_for_policy"),
        "test_gt_used_for_policy": report.get("test_gt_used_for_policy"),
    }
    checks.append(
        {
            "name": "no_declared_test_gt_policy_flag",
            "pass": not any(bool(value) for value in leakage_flags.values()),
            "flags": leakage_flags,
        }
    )
    passed = all(bool(row.get("pass", False)) for row in checks)
    if not passed:
        failed = [row for row in checks if not row.get("pass", False)]
        raise RuntimeError(f"source ELA report provenance failed for {source_report_path}: {failed}")
    return {
        "source_report_path": str(source_report_path),
        "source_report_target_split": str(report.get("target_split", "")),
        "source_report_method_name": str(report.get("method_name", "")),
        "policy_and_support_checked_against_split": "train",
        "no_test_gt_used_for_policy": True,
        "checks": checks,
    }


def _copy_gt(target_frames, out_gt: Path) -> None:
    out_gt.mkdir(parents=True, exist_ok=True)
    for frame in target_frames:
        dst = out_gt / frame.render_path.name
        if dst.exists():
            continue
        try:
            os.link(frame.gt_path, dst)
        except OSError:
            shutil.copy2(frame.gt_path, dst)


def _assert_frame_set(expected_frames, method_dir: Path) -> None:
    expected = {frame.render_path.name for frame in expected_frames}
    renders = {p.name for p in (method_dir / "renders").glob("*.png")}
    gts = {p.name for p in (method_dir / "gt").glob("*.png")}
    missing_render = sorted(expected - renders)
    missing_gt = sorted(expected - gts)
    extra_render = sorted(renders - expected)
    extra_gt = sorted(gts - expected)
    if missing_render or missing_gt or extra_render or extra_gt:
        raise RuntimeError(
            "frame-set mismatch for metric fairness: "
            f"method_dir={method_dir}, missing_render={missing_render[:8]}, missing_gt={missing_gt[:8]}, "
            f"extra_render={extra_render[:8]}, extra_gt={extra_gt[:8]}"
        )


def _materialize_endpoint(args: argparse.Namespace) -> dict[str, Any]:
    base_model = Path(args.base_model_path).resolve()
    output_model = Path(args.output_model_path).resolve()
    source_report_path = Path(args.source_ela_report).resolve()
    if not source_report_path.is_file():
        raise FileNotFoundError(f"missing source ELA report: {source_report_path}")
    if not base_model.is_dir():
        raise FileNotFoundError(f"missing base model: {base_model}")

    link_manifest = _prepare_candidate_model(base_model, output_model, int(args.iteration))
    source_report = _read_report(str(source_report_path))
    policy, policy_source = _policy_from_report(source_report)
    alpha, alpha_source = _alpha_from_report(source_report)
    benefit_calibrator = _benefit_calibrator_from_report(source_report)
    alpha_calibrator = _alpha_calibrator_from_report(source_report)

    train_frames = load_split_frames(base_model, "train", args.base_method_name)
    target_frames = load_split_frames(base_model, args.target_split, args.base_method_name)
    provenance = _validate_source_report_provenance(
        source_report,
        source_report_path=source_report_path,
        base_model=base_model,
        base_method=args.base_method_name,
        train_frames=train_frames,
    )
    support_frames, support_source, missing_support_names = _select_support_frames(train_frames, source_report)
    if not support_frames:
        raise RuntimeError("source ELA report selected no support frames")

    out_method = output_model / args.target_split / args.method_name
    if args.force and out_method.exists():
        shutil.rmtree(out_method)
    out_render = out_method / "renders"
    out_gt = out_method / "gt"
    out_render.mkdir(parents=True, exist_ok=True)
    _copy_gt(target_frames, out_gt)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    loader = FrameLoader(device=device)
    frame_infos: list[dict[str, Any]] = []
    for target in tqdm(target_frames, desc=f"checkpoint-attached ELA {args.target_split}"):
        adapted, info = adapt_frame(
            target,
            support_frames,
            k=int(policy["k"]),
            alpha=float(alpha),
            mode=str(policy["mode"]),
            residual_clip=float(policy["residual_clip"]),
            min_confidence=float(policy["min_confidence"]),
            depth_abs_tol=float(policy["depth_abs_tol"]),
            depth_rel_tol=float(policy["depth_rel_tol"]),
            direction_weight=float(policy["direction_weight"]),
            benefit_calibrator=benefit_calibrator,
            alpha_calibrator=alpha_calibrator,
            edge_gate=bool(policy["edge_gate"]),
            edge_gate_quantile=float(policy["edge_gate_quantile"]),
            edge_gate_min=float(policy["edge_gate_min"]),
            edge_gate_dilate=int(policy["edge_gate_dilate"]),
            local_trust_gate=bool(policy["local_trust_gate"]),
            local_trust_min_supports=int(policy["local_trust_min_supports"]),
            local_trust_max_residual_std=float(policy["local_trust_max_residual_std"]),
            local_trust_min_agreement=float(policy["local_trust_min_agreement"]),
            local_trust_agreement_scale=float(policy["local_trust_agreement_scale"]),
            local_trust_confidence_quantile=float(policy["local_trust_confidence_quantile"]),
            local_trust_min_confidence=float(policy["local_trust_min_confidence"]),
            local_trust_mode=str(policy["local_trust_mode"]),
            local_trust_min_weight=float(policy["local_trust_min_weight"]),
            evidence_max_side=int(args.evidence_max_side),
            loader=loader,
            device=device,
        )
        base = loader.render(str(target.render_path))
        diff = (adapted - base).detach().abs()
        changed = diff.amax(dim=0) > float(args.changed_threshold)
        save_image_tensor(adapted, out_render / target.render_path.name)
        frame_infos.append(
            {
                "frame": target.name,
                **info,
                "changed_fraction": float(changed.float().mean().item()),
                "mean_abs_delta": float(diff.mean().item()),
                "max_abs_delta": float(diff.max().item()),
            }
        )

    _assert_frame_set(target_frames, out_method)
    report = dict(source_report)
    report.update(
        {
            "method": "Checkpoint-Attached Evidence Lumigraph Endpoint",
            "endpoint_materialization_version": 1,
            "source_ela_report": str(source_report_path),
            "base_model_path": str(base_model),
            "output_model_path": str(output_model),
            "base_method": str(args.base_method_name),
            "method_name": str(args.method_name),
            "target_split": str(args.target_split),
            "target_frames": int(len(target_frames)),
            "train_support_frames": int(len(train_frames)),
            "adapt_support_scope": str(support_source),
            "adapt_support_frames": int(len(support_frames)),
            "adapt_support_view_names": [frame.name for frame in support_frames],
            "missing_report_support_names": missing_support_names,
            "policy_source": policy_source,
            "alpha_source": alpha_source,
            "alpha": float(alpha),
            "policy": policy,
            "evidence_max_side": int(args.evidence_max_side),
            "source_report_provenance": provenance,
            "no_test_gt_used_for_policy": bool(provenance["no_test_gt_used_for_policy"]),
            "policy_statement": (
                "The endpoint replays the fixed train-derived ELA report. Held-out test GT is only copied for "
                "metric evaluation after materialization and is not read for policy, support, scale, or gate selection."
            ),
            "mean_covered_fraction": _mean([_finite_float(x.get("covered_fraction")) for x in frame_infos]),
            "mean_confidence": _mean([_finite_float(x.get("mean_confidence")) for x in frame_infos]),
            "mean_changed_fraction": _mean([_finite_float(x.get("changed_fraction")) for x in frame_infos]),
            "mean_abs_delta": _mean([_finite_float(x.get("mean_abs_delta")) for x in frame_infos]),
            "mean_alpha": _mean([_finite_float(x.get("alpha_mean")) for x in frame_infos]),
            "mean_alpha_active_fraction": _mean(
                [_finite_float(x.get("alpha_active_fraction")) for x in frame_infos]
            ),
            "frames": frame_infos,
            "command": sys.argv,
            "link_manifest": link_manifest,
        }
    )
    report = _json_safe(report)
    (out_method / "ela_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    endpoint_dir = output_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "render_residual_endpoint" / args.method_name
    if args.force and endpoint_dir.exists():
        shutil.rmtree(endpoint_dir)
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    (endpoint_dir / "ela_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_command(cmd: list[str], *, cwd: Path, log_path: Path, gpu: int) -> None:
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}; see {log_path}")


def _evaluate_if_requested(args: argparse.Namespace) -> dict[str, float]:
    output_model = Path(args.output_model_path).resolve()
    if not args.evaluate:
        return _metric(_read_json(output_model / "results.json"), args.method_name)
    cmd = [
        sys.executable,
        "scripts/car_model/evaluate_render_split_metrics.py",
        "-m",
        str(output_model),
        "--split",
        str(args.target_split),
        "--methods",
        str(args.method_name),
        "--merge_model_results",
    ]
    _run_command(cmd, cwd=ROOT, log_path=output_model / "endpoint_commands.log", gpu=int(args.gpu))
    metrics = _metric(_read_json(output_model / "results.json"), args.method_name)
    per_view = _read_json(output_model / "per_view.json").get(args.method_name, {})
    frame_count = len((per_view.get("PSNR") or {})) if isinstance(per_view, dict) else 0
    render_count = len(list((output_model / args.target_split / args.method_name / "renders").glob("*.png")))
    if frame_count != render_count:
        raise RuntimeError(
            f"metric denominator mismatch for {args.method_name}: per_view={frame_count}, renders={render_count}"
        )
    return metrics


def _delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "dPSNR": candidate["PSNR"] - baseline["PSNR"],
        "dSSIM": candidate["SSIM"] - baseline["SSIM"],
        "dLPIPS": candidate["LPIPS"] - baseline["LPIPS"],
    }


def _baseline_rows(args: argparse.Namespace, candidate: dict[str, float]) -> dict[str, Any]:
    base_model = Path(args.base_model_path).resolve()
    rows: dict[str, dict[str, Any]] = {
        "candidate": {"method": args.method_name, "metrics": candidate},
        "selected_clean_meshsplatting": {
            "method": "clean_meshsplatting_counter_iter26000_selected",
            "metrics": {"PSNR": args.clean_psnr, "SSIM": args.clean_ssim, "LPIPS": args.clean_lpips},
        },
        "strict_anchor_floor": {
            "method": "v84_v86_strict_anchor_floor",
            "metrics": {"PSNR": args.anchor_psnr, "SSIM": args.anchor_ssim, "LPIPS": args.anchor_lpips},
        },
        "compact_parent_noop": {
            "method": args.base_method_name,
            "metrics": _metric(_read_json(base_model / "results.json"), args.base_method_name),
        },
        "phasej_reference_ceiling": {
            "method": args.phasej_method,
            "metrics": _metric(_read_json(base_model / "results.json"), args.phasej_method),
        },
        "legacy_source_ela_baseline": {
            "method": args.source_ela_method,
            "metrics": {"PSNR": args.source_ela_psnr, "SSIM": args.source_ela_ssim, "LPIPS": args.source_ela_lpips},
        },
    }
    same_path = Path(args.same_evidence_noop_results)
    if args.scene == "counter" and same_path.is_file():
        rows["same_evidence_noop"] = {
            "method": args.same_evidence_noop_method,
            "metrics": _metric(_read_json(same_path), args.same_evidence_noop_method),
        }
    v98b_path = Path(args.v98b_results)
    if args.scene == "counter" and v98b_path.is_file():
        rows["v98b_negative_checkpoint_baked"] = {
            "method": args.v98b_method,
            "metrics": _metric(_read_json(v98b_path), args.v98b_method),
        }
    for name, row in rows.items():
        if name == "candidate":
            continue
        row["delta_from_candidate"] = _delta(candidate, row["metrics"])
    return rows


def _write_endpoint_contract(args: argparse.Namespace, report: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    output_model = Path(args.output_model_path).resolve()
    base_model = Path(args.base_model_path).resolve()
    endpoint_dir = output_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "render_residual_endpoint" / args.method_name
    base_topology = _read_json(base_model / "topology_audit.json")
    endpoint_topology = {
        "topology_audit_version": 1,
        "status": "PASS_INHERITED_RENDER_ENDPOINT_TOPOLOGY_UNCHANGED",
        "mode": "render_time_residual_endpoint",
        "base_model": str(base_model),
        "output_model": str(output_model),
        "iteration": int(args.iteration),
        "topology_inherited_from_base": True,
        "checkpoint_state_mutated": False,
        "base_topology_audit": base_topology,
    }
    (endpoint_dir / "topology_audit.json").write_text(
        json.dumps(endpoint_topology, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _baseline_rows(args, metrics)
    alpha_active = _finite_float(report.get("mean_alpha_active_fraction"), 0.0)
    actual_delta_active = _finite_float(report.get("mean_abs_delta"), 0.0) > float(args.min_mean_abs_delta)
    non_noop = (
        int(report.get("adapt_support_frames", 0) or 0) > 0
        and _finite_float(report.get("mean_changed_fraction"), 0.0) > float(args.min_changed_fraction)
        and (alpha_active > 0.0 or actual_delta_active)
    )
    rgb_gate = {
        "psnr_pass": metrics["PSNR"] > float(args.anchor_psnr),
        "ssim_pass": metrics["SSIM"] > float(args.anchor_ssim),
        "lpips_pass": metrics["LPIPS"] < float(args.anchor_lpips),
        "candidate": metrics,
        "threshold": {"PSNR": args.anchor_psnr, "SSIM": args.anchor_ssim, "LPIPS": args.anchor_lpips},
    }
    gate = {
        "gate_version": 1,
        "status": "PASS_COUNTER_GATE" if all(rgb_gate[k] for k in ("psnr_pass", "ssim_pass", "lpips_pass")) and non_noop else "FAIL_COUNTER_GATE",
        "scene": str(args.scene),
        "method_name": str(args.method_name),
        "base_method_name": str(args.base_method_name),
        "source_ela_report": str(Path(args.source_ela_report).resolve()),
        "no_test_gt_used_for_policy": bool(report.get("no_test_gt_used_for_policy", False)),
        "source_report_provenance": report.get("source_report_provenance", {}),
        "rgb_gate": rgb_gate,
        "non_noop_gate": {
            "pass": bool(non_noop),
            "adapt_support_frames": int(report.get("adapt_support_frames", 0) or 0),
            "mean_changed_fraction": _finite_float(report.get("mean_changed_fraction"), 0.0),
            "min_changed_fraction": float(args.min_changed_fraction),
            "mean_alpha_active_fraction": _finite_float(report.get("mean_alpha_active_fraction"), 0.0),
            "mean_covered_fraction": _finite_float(report.get("mean_covered_fraction"), 0.0),
            "mean_abs_delta": _finite_float(report.get("mean_abs_delta"), 0.0),
            "min_mean_abs_delta": float(args.min_mean_abs_delta),
            "actual_delta_active": bool(actual_delta_active),
        },
        "topology_gate": endpoint_topology,
        "comparison_rows": rows,
        "artifact_paths": {
            "output_model": str(output_model),
            "renders": str(output_model / args.target_split / args.method_name / "renders"),
            "gt": str(output_model / args.target_split / args.method_name / "gt"),
            "results_json": str(output_model / "results.json"),
            "per_view_json": str(output_model / "per_view.json"),
            "endpoint_dir": str(endpoint_dir),
        },
        "command": sys.argv,
    }
    gate = _json_safe(gate)
    (endpoint_dir / "endpoint_manifest.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_model / "endpoint_gate_report.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return gate


def _resize_keep_aspect(image: Image.Image, width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width == width:
        return image
    height = max(1, int(round(image.height * (float(width) / float(image.width)))))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _make_contact_sheet(args: argparse.Namespace) -> dict[str, Any]:
    if not args.make_contact_sheet:
        return {}
    output_model = Path(args.output_model_path).resolve()
    base_model = Path(args.base_model_path).resolve()
    method = args.method_name
    base_method = args.base_method_name
    cand_per = _read_json(output_model / "per_view.json").get(method, {})
    base_per = _read_json(base_model / "per_view.json").get(base_method, {})
    cand_psnr = cand_per.get("PSNR", {}) if isinstance(cand_per, dict) else {}
    base_psnr = base_per.get("PSNR", {}) if isinstance(base_per, dict) else {}
    cand_lpips = cand_per.get("LPIPS", {}) if isinstance(cand_per, dict) else {}
    base_lpips = base_per.get("LPIPS", {}) if isinstance(base_per, dict) else {}

    render_dir = output_model / args.target_split / method / "renders"
    gt_dir = output_model / args.target_split / method / "gt"
    base_dir = base_model / args.target_split / base_method / "renders"
    names = sorted(p.name for p in render_dir.glob("*.png"))
    scored = []
    for name in names:
        dpsnr = _finite_float(cand_psnr.get(name), 0.0) - _finite_float(base_psnr.get(name), 0.0)
        dlpips = _finite_float(base_lpips.get(name), 0.0) - _finite_float(cand_lpips.get(name), 0.0)
        scored.append((dlpips, dpsnr, name))
    scored.sort(reverse=True)
    selected = [name for _dlpips, _dpsnr, name in scored[: max(1, int(args.contact_sheet_views))]]
    if not selected:
        return {}

    tile_w = int(args.contact_sheet_tile_width)
    label_h = 28
    rows: list[Image.Image] = []
    selected_rows: list[dict[str, Any]] = []
    for name in selected:
        cand = _resize_keep_aspect(Image.open(render_dir / name), tile_w)
        base = _resize_keep_aspect(Image.open(base_dir / name), tile_w)
        gt = _resize_keep_aspect(Image.open(gt_dir / name), tile_w)
        diff = ImageChops.difference(cand, base).point(lambda x: min(255, int(x) * 4))
        h = max(gt.height, base.height, cand.height, diff.height)
        row = Image.new("RGB", (tile_w * 4, h + label_h), "white")
        draw = ImageDraw.Draw(row)
        labels = [
            f"GT {name}",
            f"MeshSplat {base_method[:20]}",
            f"Endpoint {method[:20]}",
            "|Endpoint-Base| x4",
        ]
        for col, (img, label) in enumerate(zip((gt, base, cand, diff), labels)):
            x = col * tile_w
            row.paste(img, (x, label_h))
            draw.text((x + 4, 6), label, fill=(0, 0, 0))
        rows.append(row)
        selected_rows.append(
            {
                "frame": name,
                "delta_psnr": _finite_float(cand_psnr.get(name), 0.0) - _finite_float(base_psnr.get(name), 0.0),
                "delta_lpips_improvement": _finite_float(base_lpips.get(name), 0.0)
                - _finite_float(cand_lpips.get(name), 0.0),
            }
        )

    sheet_w = max(row.width for row in rows)
    sheet_h = sum(row.height for row in rows)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    out_dir = output_model / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / f"{method}_contact_sheet.png"
    selected_path = out_dir / f"{method}_selected_views.json"
    sheet.save(sheet_path)
    payload = {"contact_sheet": str(sheet_path), "selected_views": selected_rows}
    selected_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_markdown_summary(args: argparse.Namespace, report: dict[str, Any], gate: dict[str, Any], qualitative: dict[str, Any]) -> Path:
    output_model = Path(args.output_model_path).resolve()
    rows = gate.get("comparison_rows", {})
    lines = [
        "# v100 Checkpoint-Attached ELA Endpoint Gate",
        "",
        f"- status: `{gate.get('status')}`",
        f"- scene: `{args.scene}`",
        f"- method: `{args.method_name}`",
        f"- base model: `{Path(args.base_model_path).resolve()}`",
        f"- output model: `{output_model}`",
        f"- source ELA report: `{Path(args.source_ela_report).resolve()}`",
        f"- no test GT used for policy: `{gate.get('no_test_gt_used_for_policy')}`",
        f"- mean changed fraction: `{report.get('mean_changed_fraction')}`",
        f"- mean covered fraction: `{report.get('mean_covered_fraction')}`",
        f"- mean alpha active fraction: `{report.get('mean_alpha_active_fraction')}`",
        "",
        "## Counter Metrics",
        "",
        "| row | PSNR | SSIM | LPIPS | dPSNR cand-row | dSSIM cand-row | dLPIPS cand-row |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        metrics = row.get("metrics", {})
        delta = row.get("delta_from_candidate", {"dPSNR": 0.0, "dSSIM": 0.0, "dLPIPS": 0.0})
        lines.append(
            f"| {name} | {_finite_float(metrics.get('PSNR')):.6f} | {_finite_float(metrics.get('SSIM')):.6f} | "
            f"{_finite_float(metrics.get('LPIPS')):.6f} | {_finite_float(delta.get('dPSNR'), 0.0):+.6f} | "
            f"{_finite_float(delta.get('dSSIM'), 0.0):+.6f} | {_finite_float(delta.get('dLPIPS'), 0.0):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- RGB PSNR pass: `{gate.get('rgb_gate', {}).get('psnr_pass')}`",
            f"- RGB SSIM pass: `{gate.get('rgb_gate', {}).get('ssim_pass')}`",
            f"- RGB LPIPS pass: `{gate.get('rgb_gate', {}).get('lpips_pass')}`",
            f"- non-noop pass: `{gate.get('non_noop_gate', {}).get('pass')}`",
            f"- topology status: `{gate.get('topology_gate', {}).get('status')}`",
            "",
            "## Artifacts",
            "",
            f"- results: `{output_model / 'results.json'}`",
            f"- per-view: `{output_model / 'per_view.json'}`",
            f"- endpoint manifest: `{output_model / 'endpoint_gate_report.json'}`",
            f"- qualitative contact sheet: `{qualitative.get('contact_sheet', '')}`",
        ]
    )
    out = output_model / "endpoint_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _maybe_wandb(args: argparse.Namespace, gate: dict[str, Any], qualitative: dict[str, Any]) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"[v100-endpoint] W&B unavailable, skipping: {exc}")
        return
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True) if args.wandb_dir else None
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group or None,
        name=args.wandb_name or args.method_name,
        mode=args.wandb_mode,
        dir=args.wandb_dir or None,
        config={
            "scene": args.scene,
            "base_model_path": args.base_model_path,
            "output_model_path": args.output_model_path,
            "method_name": args.method_name,
            "source_ela_report": args.source_ela_report,
            "iteration": args.iteration,
            "base_method_name": args.base_method_name,
        },
    )
    candidate = gate.get("comparison_rows", {}).get("candidate", {}).get("metrics", {})
    non_noop = gate.get("non_noop_gate", {})
    flat = {
        "gate/pass": int(gate.get("status") == "PASS_COUNTER_GATE"),
        "metrics/psnr": _finite_float(candidate.get("PSNR")),
        "metrics/ssim": _finite_float(candidate.get("SSIM")),
        "metrics/lpips": _finite_float(candidate.get("LPIPS")),
        "endpoint/mean_changed_fraction": _finite_float(non_noop.get("mean_changed_fraction"), 0.0),
        "endpoint/mean_covered_fraction": _finite_float(non_noop.get("mean_covered_fraction"), 0.0),
        "endpoint/mean_alpha_active_fraction": _finite_float(non_noop.get("mean_alpha_active_fraction"), 0.0),
    }
    run.log(flat)
    if qualitative.get("contact_sheet"):
        try:
            run.log({"qualitative/contact_sheet": wandb.Image(str(qualitative["contact_sheet"]))})
        except Exception as exc:
            print(f"[v100-endpoint] W&B image log skipped: {exc}")
    run.summary.update(flat)
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize and gate a checkpoint-attached ELA render endpoint.")
    parser.add_argument("--scene", default="counter")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_model_path", required=True)
    parser.add_argument("--source_ela_report", required=True)
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--base_method_name", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--method_name", default="ours_26000_v100_checkpoint_attached_ela_endpoint")
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--evidence_max_side", type=int, default=0)
    parser.add_argument("--changed_threshold", type=float, default=1e-5)
    parser.add_argument("--min_changed_fraction", type=float, default=1e-4)
    parser.add_argument("--min_mean_abs_delta", type=float, default=1e-7)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--make_contact_sheet", action="store_true")
    parser.add_argument("--contact_sheet_views", type=int, default=6)
    parser.add_argument("--contact_sheet_tile_width", type=int, default=320)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--anchor_psnr", type=float, default=26.7561378479)
    parser.add_argument("--anchor_ssim", type=float, default=0.8621263504)
    parser.add_argument("--anchor_lpips", type=float, default=0.2516906559)
    parser.add_argument("--clean_psnr", type=float, default=26.7517738342)
    parser.add_argument("--clean_ssim", type=float, default=0.8620552421)
    parser.add_argument("--clean_lpips", type=float, default=0.2520033121)
    parser.add_argument("--phasej_method", default="ours_26000_phasej_guarded_adaptedge_ela")
    parser.add_argument("--source_ela_method", default="ours_26000_sor_adaptive_geo_compact_ela")
    parser.add_argument("--source_ela_psnr", type=float, default=27.2404232025)
    parser.add_argument("--source_ela_ssim", type=float, default=0.8641442060)
    parser.add_argument("--source_ela_lpips", type=float, default=0.2497010678)
    parser.add_argument(
        "--same_evidence_noop_results",
        default="outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware/counter_evidence_noop_compact_baseline/results.json",
    )
    parser.add_argument("--same_evidence_noop_method", default="ours_26000_counter_evidence_noop_compact_baseline")
    parser.add_argument(
        "--v98b_results",
        default="/dev/shm/peilincai_spcarnet_v98b_delta_texture_bake_parentmask_20260625/counter_v98b_delta_texture_bake_parentmask/recovery_model/results.json",
    )
    parser.add_argument("--v98b_method", default="ours_27000")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "spcarnet_meshprior"))
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP", "v100_checkpoint_attached_ela_endpoint"))
    parser.add_argument("--wandb_name", default=os.environ.get("WANDB_NAME", ""))
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", ""))
    args = parser.parse_args()
    if int(args.evidence_max_side) < 0:
        parser.error("--evidence_max_side must be >= 0")
    if int(args.contact_sheet_views) < 1:
        parser.error("--contact_sheet_views must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    if int(args.gpu) >= 0 and str(args.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
        args.device = "cuda:0"
    report = _materialize_endpoint(args)
    metrics = _evaluate_if_requested(args)
    gate = _write_endpoint_contract(args, report, metrics)
    qualitative = _make_contact_sheet(args)
    summary = _write_markdown_summary(args, report, gate, qualitative)
    _maybe_wandb(args, gate, qualitative)
    print(
        json.dumps(
            {
                "status": gate.get("status"),
                "summary": str(summary),
                "output_model": str(Path(args.output_model_path).resolve()),
                "results": str(Path(args.output_model_path).resolve() / "results.json"),
                "contact_sheet": qualitative.get("contact_sheet", ""),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if gate.get("status") == "PASS_COUNTER_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

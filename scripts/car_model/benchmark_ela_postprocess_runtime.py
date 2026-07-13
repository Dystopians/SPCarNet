#!/usr/bin/env python3
"""Benchmark Evidence Lumigraph Adapter post-processing runtime.

This profiler times utils.evidence_lumigraph_adapter.adapt_frame over an
existing rendered split/method. It intentionally does not write PNGs; the
reported wall time is the CPU-side elapsed time around adapt_frame calls, with
CUDA synchronization and peak CUDA memory stats when a CUDA device is used.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_POLICY: dict[str, Any] = {
    "k": 4,
    "mode": "residual",
    "residual_clip": 0.25,
    "min_confidence": 1e-4,
    "depth_abs_tol": 0.02,
    "depth_rel_tol": 0.03,
    "direction_weight": 0.35,
    "edge_gate": False,
    "edge_gate_quantile": -1.0,
    "edge_gate_min": 0.0,
    "edge_gate_dilate": 0,
    "local_trust_gate": False,
    "local_trust_min_supports": 2,
    "local_trust_max_residual_std": -1.0,
    "local_trust_min_agreement": 0.0,
    "local_trust_agreement_scale": 0.04,
    "local_trust_confidence_quantile": -1.0,
    "local_trust_min_confidence": 0.0,
    "local_trust_mode": "hard",
    "local_trust_min_weight": 0.0,
}


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float(value: Any, default: float) -> float:
    out = _finite(value)
    return float(default) if out is None else float(out)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _float_tuple(value: Any, default: Sequence[float]) -> tuple[float, ...]:
    raw = value if isinstance(value, (list, tuple)) else default
    out: list[float] = []
    for item in raw:
        parsed = _finite(item)
        if parsed is not None:
            out.append(float(parsed))
    return tuple(out) if out else tuple(float(x) for x in default)


def _read_report(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ELA report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ELA report must contain a JSON object: {path}")
    return payload


def _policy_value(report: dict[str, Any], key: str, default: Any) -> Any:
    policy = report.get("policy")
    if isinstance(policy, dict) and key in policy:
        return policy[key]
    if key in report:
        return report[key]
    return default


def _policy_from_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not report:
        return dict(DEFAULT_POLICY), "defaults"

    policy = dict(DEFAULT_POLICY)
    policy["k"] = max(0, _int(_policy_value(report, "k", policy["k"]), policy["k"]))
    mode = str(_policy_value(report, "mode", policy["mode"]))
    if mode not in {"residual", "color"}:
        raise ValueError(f"Unsupported ELA mode in report: {mode}")
    policy["mode"] = mode
    for key in (
        "residual_clip",
        "min_confidence",
        "depth_abs_tol",
        "depth_rel_tol",
        "direction_weight",
        "edge_gate_quantile",
        "edge_gate_min",
        "local_trust_max_residual_std",
        "local_trust_min_agreement",
        "local_trust_agreement_scale",
        "local_trust_confidence_quantile",
        "local_trust_min_confidence",
        "local_trust_min_weight",
    ):
        policy[key] = _float(_policy_value(report, key, policy[key]), policy[key])
    for key in ("edge_gate_dilate", "local_trust_min_supports"):
        policy[key] = _int(_policy_value(report, key, policy[key]), policy[key])
    for key in ("edge_gate", "local_trust_gate"):
        policy[key] = _bool(_policy_value(report, key, policy[key]), policy[key])
    local_mode = str(_policy_value(report, "local_trust_mode", policy["local_trust_mode"])).strip().lower()
    policy["local_trust_mode"] = local_mode if local_mode in {"hard", "soft"} else "hard"
    return policy, "ela_report"


def _alpha_from_report(report: dict[str, Any]) -> tuple[float, str]:
    if not report:
        return 1.0, "default"
    alpha = _finite(report.get("alpha"))
    if alpha is not None:
        return float(alpha), "ela_report"
    calibration = report.get("calibration")
    if isinstance(calibration, dict):
        alpha = _finite(calibration.get("alpha"))
        if alpha is not None:
            return float(alpha), "ela_report_calibration"
    return 1.0, "default"


def _benefit_calibrator_from_report(report: dict[str, Any]) -> BenefitCalibrator | None:
    from utils.evidence_lumigraph_adapter import BenefitCalibrator

    row = report.get("benefit_policy")
    if not isinstance(row, dict):
        return None
    return BenefitCalibrator(
        confidence_edges=_float_tuple(row.get("confidence_edges"), (0.0, 1.0)),
        magnitude_edges=_float_tuple(row.get("magnitude_edges"), (0.0, 1.0)),
        gain_table=_nested_tuple(row.get("gain_table", ((0.0,),))),
        count_table=_nested_tuple(row.get("count_table", ((0,),))),
        accept_table=_nested_tuple(row.get("accept_table", ((False,),))),
        min_gain=_float(row.get("min_gain"), 0.0),
        min_bin_count=max(0, _int(row.get("min_bin_count"), 64)),
        edge_edges=_float_tuple(row.get("edge_edges"), ()) if row.get("edge_edges") is not None else None,
        feature_mode=str(row.get("feature_mode", "confidence_magnitude")),
    )


def _alpha_calibrator_from_report(report: dict[str, Any]) -> AlphaCalibrator | None:
    from utils.evidence_lumigraph_adapter import AlphaCalibrator

    row = report.get("alpha_calibrator")
    if not isinstance(row, dict):
        return None
    if "alpha_table" not in row or "accept_table" not in row:
        return None

    min_tail_gain = row.get("min_tail_gain")
    if min_tail_gain is None:
        min_tail_gain = -math.inf
    view_tail_min_gain = row.get("view_tail_min_gain")
    if view_tail_min_gain is None:
        view_tail_min_gain = -math.inf

    return AlphaCalibrator(
        confidence_edges=_float_tuple(row.get("confidence_edges"), (0.0, 1.0)),
        magnitude_edges=_float_tuple(row.get("magnitude_edges"), (0.0, 1.0)),
        alpha_table=_nested_tuple(row.get("alpha_table")),
        gain_table=_nested_tuple(row.get("gain_table", ((0.0,),))),
        count_table=_nested_tuple(row.get("count_table", ((0,),))),
        accept_table=_nested_tuple(row.get("accept_table")),
        tail_gain_table=_nested_tuple(row.get("tail_gain_table")) if row.get("tail_gain_table") is not None else None,
        negative_fraction_table=(
            _nested_tuple(row.get("negative_fraction_table"))
            if row.get("negative_fraction_table") is not None
            else None
        ),
        risk_zeroed_table=_nested_tuple(row.get("risk_zeroed_table")) if row.get("risk_zeroed_table") is not None else None,
        region_tail_gain_table=(
            _nested_tuple(row.get("region_tail_gain_table"))
            if row.get("region_tail_gain_table") is not None
            else None
        ),
        region_negative_fraction_table=(
            _nested_tuple(row.get("region_negative_fraction_table"))
            if row.get("region_negative_fraction_table") is not None
            else None
        ),
        region_count_table=_nested_tuple(row.get("region_count_table")) if row.get("region_count_table") is not None else None,
        region_risk_zeroed_table=(
            _nested_tuple(row.get("region_risk_zeroed_table"))
            if row.get("region_risk_zeroed_table") is not None
            else None
        ),
        default_alpha=_float(row.get("default_alpha"), 0.0),
        min_gain=_float(row.get("min_gain"), 0.0),
        min_bin_count=max(0, _int(row.get("min_bin_count"), 64)),
        risk_tail_fraction=_float(row.get("risk_tail_fraction"), 0.20),
        max_negative_gain_fraction=_float(row.get("max_negative_gain_fraction"), 1.0),
        min_tail_gain=float(min_tail_gain),
        holdout_safe_zero=_bool(row.get("holdout_safe_zero"), False),
        region_risk_enabled=_bool(row.get("region_risk_enabled"), False),
        region_risk_json=str(row.get("region_risk_json", "")),
        region_risk_objective_bad_only=_bool(row.get("region_risk_objective_bad_only"), False),
        region_risk_objective_max_balanced_delta=_float(row.get("region_risk_objective_max_balanced_delta"), 0.0),
        region_risk_objective_max_delta_ssim=_float(row.get("region_risk_objective_max_delta_ssim"), 0.0),
        region_risk_objective_min_delta_lpips=_float(row.get("region_risk_objective_min_delta_lpips"), 0.0),
        region_risk_min_tail_gain=_float(row.get("region_risk_min_tail_gain"), 0.0),
        region_risk_max_negative_fraction=_float(row.get("region_risk_max_negative_fraction"), 1.0),
        region_risk_min_regions=max(0, _int(row.get("region_risk_min_regions"), 1)),
        view_tail_scale=_float(row.get("view_tail_scale"), 1.0),
        view_tail_enabled=_bool(row.get("view_tail_enabled"), False),
        view_tail_scale_grid=(
            _nested_tuple(row.get("view_tail_scale_grid")) if row.get("view_tail_scale_grid") is not None else None
        ),
        view_tail_cvar_fraction=_float(row.get("view_tail_cvar_fraction"), 0.25),
        view_tail_min_gain=float(view_tail_min_gain),
        view_tail_max_negative_fraction=_float(row.get("view_tail_max_negative_fraction"), 1.0),
        view_tail_objective=str(row.get("view_tail_objective", "mse")),
        view_tail_ssim_weight=_float(row.get("view_tail_ssim_weight"), 20.0),
        view_tail_lpips_weight=_float(row.get("view_tail_lpips_weight"), 20.0),
        view_tail_compute_lpips=_bool(row.get("view_tail_compute_lpips"), False),
        view_tail_metric_max_side=max(1, _int(row.get("view_tail_metric_max_side"), 512)),
        view_tail_mean_gain=_float(row.get("view_tail_mean_gain"), 0.0),
        view_tail_cvar_gain=_float(row.get("view_tail_cvar_gain"), 0.0),
        view_tail_negative_fraction=_float(row.get("view_tail_negative_fraction"), 0.0),
        view_tail_safe_scale_found=_bool(row.get("view_tail_safe_scale_found"), False),
        view_tail_fallback_used=_bool(row.get("view_tail_fallback_used"), False),
        view_tail_candidate_stats=(
            _nested_tuple(row.get("view_tail_candidate_stats"))
            if row.get("view_tail_candidate_stats") is not None
            else None
        ),
        edge_edges=_float_tuple(row.get("edge_edges"), ()) if row.get("edge_edges") is not None else None,
        feature_mode=str(row.get("feature_mode", "confidence_magnitude")),
    )


def _resolve_device(text: str) -> torch.device:
    import torch

    requested = str(text).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device but torch.cuda.is_available() is false: {text}")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


def _select_targets(frames: Sequence[FrameRecord], max_views: int) -> list[FrameRecord]:
    selected = list(frames)
    if int(max_views) > 0:
        selected = selected[: int(max_views)]
    return selected


def _select_support_frames(
    train_frames: Sequence[FrameRecord],
    report: dict[str, Any],
) -> tuple[list[FrameRecord], str, list[str]]:
    names = report.get("adapt_support_view_names")
    if not isinstance(names, list):
        names = report.get("policy_fit_views") if report.get("adapt_support_scope") == "policy_fit_train_only" else None
    if not isinstance(names, list):
        return list(train_frames), "all_train", []

    by_name: dict[str, FrameRecord] = {}
    for frame in train_frames:
        by_name[str(frame.name)] = frame
        by_name[str(frame.camera.image_name)] = frame

    support: list[FrameRecord] = []
    missing: list[str] = []
    seen: set[str] = set()
    for item in names:
        name = str(item)
        frame = by_name.get(name)
        if frame is None:
            missing.append(name)
            continue
        if frame.name in seen:
            continue
        support.append(frame)
        seen.add(frame.name)
    if not support:
        return list(train_frames), "all_train_report_support_names_unmatched", missing
    return support, "ela_report_support_names", missing


def _cuda_peak_row(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"cuda_peak_allocated_mib": None, "cuda_peak_reserved_mib": None}
    import torch

    return {
        "cuda_peak_allocated_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        "cuda_peak_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _run_repeat(
    *,
    repeat_index: int,
    target_frames: Sequence[FrameRecord],
    support_frames: Sequence[FrameRecord],
    policy: dict[str, Any],
    alpha: float,
    benefit_calibrator: BenefitCalibrator | None,
    alpha_calibrator: AlphaCalibrator | None,
    evidence_max_side: int,
    device: torch.device,
) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    from utils.evidence_lumigraph_adapter import FrameLoader, adapt_frame

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    loader = FrameLoader(device=device)
    _sync(device)
    start = time.perf_counter()
    frame_infos: list[dict[str, Any]] = []
    checksum = 0.0
    for target in tqdm(target_frames, desc=f"ELA adapt repeat {repeat_index}", leave=False):
        _, info = adapt_frame(
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
            evidence_max_side=int(evidence_max_side),
            loader=loader,
            device=device,
        )
        checksum += float(info.get("mean_confidence", 0.0)) + float(info.get("covered_fraction", 0.0))
        frame_infos.append({"frame": target.name, **info})
    _sync(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    covered = [float(row.get("covered_fraction", 0.0)) for row in frame_infos]
    confidence = [float(row.get("mean_confidence", 0.0)) for row in frame_infos]
    row: dict[str, Any] = {
        "repeat": int(repeat_index),
        "cpu_wall_time_sec": float(elapsed),
        "ms_per_target_frame": float(elapsed * 1000.0 / max(len(target_frames), 1)),
        "target_frames_per_sec": float(len(target_frames) / elapsed),
        "mean_covered_fraction": _mean(covered),
        "mean_confidence": _mean(confidence),
        "checksum": float(checksum),
        "frames": frame_infos,
    }
    row.update(_cuda_peak_row(device))
    return row


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    cuda_alloc = payload.get("cuda_peak_allocated_mib_max")
    cuda_reserved = payload.get("cuda_peak_reserved_mib_max")
    lines = [
        "# ELA Postprocess Runtime Profile",
        "",
        "This profile times `utils.evidence_lumigraph_adapter.adapt_frame` without writing PNG renders.",
        "",
        "## Summary",
        "",
        f"- base model path: `{payload['base_model_path']}`",
        f"- base method name: `{payload['base_method_name']}`",
        f"- target split: `{payload['target_split']}`",
        f"- device: `{payload['device']}`",
        f"- target frame count: `{payload['target_frame_count']}`",
        f"- support frame count: `{payload['support_frame_count']}`",
        f"- repeats: `{payload['repeats']}`",
        f"- alpha: `{payload['alpha']}`",
        f"- k: `{payload['k']}`",
        f"- mode: `{payload['mode']}`",
        f"- evidence max side: `{payload['evidence_max_side']}`",
        f"- depth rel tol: `{payload['depth_rel_tol']}`",
        f"- residual clip: `{payload['residual_clip']}`",
        f"- direction weight: `{payload['direction_weight']}`",
        f"- CPU wall mean sec: `{payload['cpu_wall_time_sec_mean']:.6f}`",
        f"- mean ms/frame: `{payload['ms_per_target_frame_mean']:.6f}`",
        f"- CUDA peak allocated MiB max: `{cuda_alloc if cuda_alloc is not None else 'n/a'}`",
        f"- CUDA peak reserved MiB max: `{cuda_reserved if cuda_reserved is not None else 'n/a'}`",
        "",
        "## Repeats",
        "",
        "| repeat | CPU wall sec | ms/frame | frames/sec | CUDA peak allocated MiB | CUDA peak reserved MiB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["repeat_rows"]:
        alloc = row.get("cuda_peak_allocated_mib")
        reserved = row.get("cuda_peak_reserved_mib")
        lines.append(
            "| {repeat} | {wall:.6f} | {ms:.6f} | {fps:.6f} | {alloc} | {reserved} |".format(
                repeat=row["repeat"],
                wall=row["cpu_wall_time_sec"],
                ms=row["ms_per_target_frame"],
                fps=row["target_frames_per_sec"],
                alloc=f"{alloc:.3f}" if alloc is not None else "n/a",
                reserved=f"{reserved:.3f}" if reserved is not None else "n/a",
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- Measures ELA post-processing over existing rendered RGB/depth/GT artifacts.",
            "- Excludes PNG output writes, image metrics, renderer runtime, and policy calibration.",
            "- Uses a fresh `FrameLoader` per repeat, with its normal LRU cache active within each repeat.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    from utils.evidence_lumigraph_adapter import load_split_frames

    device = _resolve_device(args.device)
    report = _read_report(args.ela_report)
    policy, policy_source = _policy_from_report(report)
    alpha, alpha_source = _alpha_from_report(report)
    benefit_calibrator = _benefit_calibrator_from_report(report)
    alpha_calibrator = _alpha_calibrator_from_report(report)

    base_model = Path(args.base_model_path).expanduser()
    train_frames = load_split_frames(base_model, "train", args.base_method_name)
    available_targets = load_split_frames(base_model, args.target_split, args.base_method_name)
    target_frames = _select_targets(available_targets, int(args.max_views))
    if not target_frames:
        raise RuntimeError(f"No target frames selected from {base_model}/{args.target_split}/{args.base_method_name}")
    support_frames, support_source, missing_support_names = _select_support_frames(train_frames, report)
    if not support_frames:
        raise RuntimeError(f"No support frames selected from {base_model}/train/{args.base_method_name}")

    repeat_rows = [
        _run_repeat(
            repeat_index=idx + 1,
            target_frames=target_frames,
            support_frames=support_frames,
            policy=policy,
            alpha=alpha,
            benefit_calibrator=benefit_calibrator,
            alpha_calibrator=alpha_calibrator,
            evidence_max_side=int(args.evidence_max_side),
            device=device,
        )
        for idx in range(max(1, int(args.repeats)))
    ]

    wall = [float(row["cpu_wall_time_sec"]) for row in repeat_rows]
    ms = [float(row["ms_per_target_frame"]) for row in repeat_rows]
    fps = [float(row["target_frames_per_sec"]) for row in repeat_rows]
    cuda_alloc = [
        float(row["cuda_peak_allocated_mib"])
        for row in repeat_rows
        if row.get("cuda_peak_allocated_mib") is not None
    ]
    cuda_reserved = [
        float(row["cuda_peak_reserved_mib"])
        for row in repeat_rows
        if row.get("cuda_peak_reserved_mib") is not None
    ]

    payload: dict[str, Any] = {
        "benchmark": "ela_postprocess_runtime",
        "scope": "adapt_frame_no_png_no_metrics_no_renderer_no_calibration",
        "command": sys.argv,
        "base_model_path": str(base_model),
        "base_method_name": str(args.base_method_name),
        "target_split": str(args.target_split),
        "ela_report": str(args.ela_report) if args.ela_report else "",
        "policy_source": policy_source,
        "alpha_source": alpha_source,
        "requested_device": str(args.device),
        "device": str(device),
        "max_views": int(args.max_views),
        "repeats": int(len(repeat_rows)),
        "available_target_frame_count": int(len(available_targets)),
        "target_frame_count": int(len(target_frames)),
        "support_frame_count": int(len(support_frames)),
        "support_source": support_source,
        "missing_report_support_names": missing_support_names,
        "target_frame_names": [frame.name for frame in target_frames],
        "support_frame_names": [frame.name for frame in support_frames],
        "alpha": float(alpha),
        "k": int(policy["k"]),
        "mode": str(policy["mode"]),
        "depth_abs_tol": float(policy["depth_abs_tol"]),
        "depth_rel_tol": float(policy["depth_rel_tol"]),
        "residual_clip": float(policy["residual_clip"]),
        "direction_weight": float(policy["direction_weight"]),
        "min_confidence": float(policy["min_confidence"]),
        "edge_gate": bool(policy["edge_gate"]),
        "edge_gate_quantile": float(policy["edge_gate_quantile"]),
        "edge_gate_min": float(policy["edge_gate_min"]),
        "edge_gate_dilate": int(policy["edge_gate_dilate"]),
        "local_trust_gate": bool(policy["local_trust_gate"]),
        "local_trust_min_supports": int(policy["local_trust_min_supports"]),
        "local_trust_max_residual_std": float(policy["local_trust_max_residual_std"]),
        "local_trust_min_agreement": float(policy["local_trust_min_agreement"]),
        "local_trust_agreement_scale": float(policy["local_trust_agreement_scale"]),
        "local_trust_confidence_quantile": float(policy["local_trust_confidence_quantile"]),
        "local_trust_min_confidence": float(policy["local_trust_min_confidence"]),
        "local_trust_mode": str(policy["local_trust_mode"]),
        "local_trust_min_weight": float(policy["local_trust_min_weight"]),
        "evidence_max_side": int(args.evidence_max_side),
        "benefit_calibrator_loaded": benefit_calibrator is not None,
        "alpha_calibrator_loaded": alpha_calibrator is not None,
        "cpu_wall_time_sec_mean": _mean(wall),
        "cpu_wall_time_sec_stdev": _stdev(wall),
        "cpu_wall_time_sec_min": min(wall) if wall else None,
        "cpu_wall_time_sec_max": max(wall) if wall else None,
        "ms_per_target_frame_mean": _mean(ms),
        "ms_per_target_frame_stdev": _stdev(ms),
        "target_frames_per_sec_mean": _mean(fps),
        "target_frames_per_sec_stdev": _stdev(fps),
        "cuda_peak_allocated_mib_mean": _mean(cuda_alloc),
        "cuda_peak_allocated_mib_max": max(cuda_alloc) if cuda_alloc else None,
        "cuda_peak_reserved_mib_mean": _mean(cuda_reserved),
        "cuda_peak_reserved_mib_max": max(cuda_reserved) if cuda_reserved else None,
        "repeat_rows": repeat_rows,
    }
    return _json_safe(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SPCarNet ELA adapt_frame post-processing runtime without writing PNGs."
    )
    parser.add_argument("--base_model_path", required=True, help="Model artifact root containing train/test methods.")
    parser.add_argument("--base_method_name", required=True, help="Existing method directory under train/test.")
    parser.add_argument("--target_split", choices=("train", "test"), default="test")
    parser.add_argument("--ela_report", default="", help="Optional existing ela_report.json to reuse policy/alpha fields.")
    parser.add_argument("--max_views", type=int, default=1, help="0 means all target views. Default 1 is smoke-friendly.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--evidence_max_side",
        type=int,
        default=0,
        help="Optional fast adapter path: compute evidence warps at this maximum side before upsampling.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--out_json", default="ela_postprocess_runtime_profile.json")
    parser.add_argument("--out_md", default="ela_postprocess_runtime_profile.md")
    args = parser.parse_args()

    if int(args.max_views) < 0:
        parser.error("--max_views must be >= 0")
    if int(args.repeats) < 1:
        parser.error("--repeats must be >= 1")
    if int(args.evidence_max_side) < 0:
        parser.error("--evidence_max_side must be >= 0")

    payload = build_payload(args)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        _write_markdown(Path(args.out_md), payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

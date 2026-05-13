#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE_S_ROOT = ROOT / "outputs/carnet/meshsplatopt/ecsr_phase_s"
MESHSPLATOPT_ROOT = ROOT / "outputs/carnet/meshsplatopt"
VERSIONS = ("v24", "v25", "v26", "v27")
METRIC_KEYS = ("PSNR", "SSIM", "LPIPS", "AbsRel", "DepthMAE", "NormalMeanDeg")


CONTROL_ROOTS = {
    "short_2200": "outputs/carnet/meshsplatopt/stageR17_04_parking_baseline_freeze_skip_delaunay_2000to2200/recovery_model",
    "medium_4000": "outputs/carnet/meshsplatopt/stageR15_03_parking_baseline_freeze_densify_skip_delaunay_2000to4000/recovery_model",
    "sparse_medium_4000": "outputs/carnet/meshsplatopt/stageR27_04_parking_baseline_sparse_depth_lam0p005_2000to4000/recovery_model",
}


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _extract_version(text: str) -> str | None:
    match = re.search(r"\bv(2[4-7])\b|_v(2[4-7])_", text)
    if not match:
        return None
    return f"v{match.group(1) or match.group(2)}"


def _extract_stage(text: str) -> str | None:
    match = re.search(r"stageR(2[4-7])", text)
    return f"R{match.group(1)}" if match else None


def _read_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,ppid,stat,etimes,cmd"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        pid, ppid, stat, etimes, cmd = parts
        if "meshsplatopt_audit_v24_v27_outputs.py" in cmd:
            continue
        if not any(token in cmd for token in ("v24", "v25", "v26", "v27", "stageR24", "stageR25", "stageR26", "stageR27")):
            continue
        rows.append({"pid": int(pid), "ppid": int(ppid), "stat": stat, "etimes": int(etimes), "cmd": cmd})
    return rows


def _running_for(path: Path, processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abs_path = path.resolve().as_posix()
    rel_path = _rel(path) or ""
    roots = [abs_path, rel_path]
    if path.is_dir():
        roots.extend([abs_path.rstrip("/"), rel_path.rstrip("/")])
    hits = []
    for proc in processes:
        cmd = proc["cmd"]
        if any(root and root in cmd for root in roots):
            hits.append(proc)
            continue
        stage = _extract_stage(path.as_posix())
        version = _extract_version(path.as_posix())
        specific_tokens = [
            token
            for token in (path.name, path.parent.name)
            if len(token) > 8 and token not in {"multifold_trainval_gate", "ecsr_phase_s", "meshsplatopt"}
        ]
        if stage and stage in cmd and any(token in cmd for token in specific_tokens):
            hits.append(proc)
        elif version and version in cmd and any(token in cmd for token in specific_tokens):
            hits.append(proc)
    return hits


def _extract_log_commands(path: Path, limit: int = 6) -> list[str]:
    if not path.is_file():
        return []
    commands: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("$ "):
                commands.append(stripped[2:])
    except OSError:
        return []
    return commands[-limit:]


def _exit_status_from_report(report: dict[str, Any] | None) -> tuple[str | None, list[str]]:
    if not isinstance(report, dict):
        return None, []
    exit_codes = report.get("exit_codes")
    if not isinstance(exit_codes, dict):
        return None, []
    failed = [f"{name}={code}" for name, code in exit_codes.items() if code not in (0, None)]
    if failed:
        return "failed", failed
    return "complete", []


def _metric_from_summary(summary: dict[str, Any] | None, metric: str, stat: str = "mean") -> float | None:
    if not isinstance(summary, dict):
        return None
    row = summary.get(metric)
    if not isinstance(row, dict):
        return None
    return _finite_float(row.get(stat))


def _delta_dict(raw: dict[str, Any] | None) -> dict[str, float | None]:
    raw = raw if isinstance(raw, dict) else {}
    return {key: _finite_float(raw.get(key)) for key in ("PSNR", "SSIM", "LPIPS")}


def _topology_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, int | None]:
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    out: dict[str, int | None] = {}
    for key in ("vertices", "triangles"):
        old = before.get(key)
        new = after.get(key)
        out[f"d{key}"] = int(new) - int(old) if isinstance(old, int) and isinstance(new, int) else None
    return out


def _materialization_effect(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit, dict):
        return {}
    effect = audit.get("materialization_effect")
    if not isinstance(effect, dict):
        effect = {}
    topo = _topology_delta(audit.get("topology_before"), audit.get("topology_after"))
    return {
        "audit_path": None,
        "operator": audit.get("operator"),
        "accepted": audit.get("accepted"),
        "policy_pass": audit.get("policy_pass"),
        "materialized": audit.get("materialized"),
        "no_op_copy": audit.get("no_op_copy"),
        "feature_mode": audit.get("feature_mode"),
        "materialize_mode": audit.get("materialize_mode"),
        "strength": _finite_float(audit.get("strength")),
        "accepted_faces": audit.get("accepted_faces"),
        "candidate_faces": audit.get("candidate_faces"),
        "topology_changed": bool(effect.get("topology_changed")) if "topology_changed" in effect else None,
        "attribute_changed": bool(effect.get("attribute_changed")) if "attribute_changed" in effect else None,
        "max_attribute_delta": _finite_float(effect.get("max_attribute_delta")),
        "mean_delta_abs": _finite_float(audit.get("mean_delta_abs")),
        "mean_proxy_relative_gain": _finite_float(audit.get("mean_proxy_relative_gain")),
        **topo,
    }


def collect_phase_s_gates(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gate_root = PHASE_S_ROOT / "multifold_trainval_gate"
    for gate_path in sorted(gate_root.glob("**/multifold_trainval_gate.json")):
        version = _extract_version(gate_path.as_posix())
        if version not in VERSIONS:
            continue
        gate = _load_json(gate_path)
        if not isinstance(gate, dict):
            continue
        scene = str(gate.get("scene") or gate_path.parent.name)
        candidate_label = str(gate.get("candidate_label") or gate_path.parent.parent.name)
        materialized_root = PHASE_S_ROOT / gate_path.parent.parent.name / scene / "model"
        audit_path = materialized_root / "surface_residual_subdivision_delta_audit.json"
        audit = _load_json(audit_path)
        effect = _materialization_effect(audit)
        effect["audit_path"] = _rel(audit_path) if audit_path.is_file() else None
        running = _running_for(gate_path.parent.parent, processes)
        status = "running" if running else "complete"
        summary = gate.get("trainval_delta_summary")
        row = {
            "version": version,
            "family": "phase_s_multifold_gate",
            "scene": scene,
            "candidate_label": candidate_label,
            "accepted": gate.get("accepted"),
            "status": status,
            "decision_reasons": gate.get("decision_reasons") or [],
            "offset_count": len(gate.get("rows") or []),
            "mean_dPSNR": _metric_from_summary(summary, "PSNR"),
            "mean_dSSIM": _metric_from_summary(summary, "SSIM"),
            "mean_dLPIPS": _metric_from_summary(summary, "LPIPS"),
            "path": _rel(gate_path),
            "effect": effect,
            "running_commands": [proc["cmd"] for proc in running],
            "offsets": [],
        }
        for item in gate.get("rows") or []:
            if not isinstance(item, dict):
                continue
            delta = _delta_dict(item.get("delta"))
            row["offsets"].append(
                {
                    "offset": item.get("offset"),
                    "accepted": item.get("accepted"),
                    "dPSNR": delta["PSNR"],
                    "dSSIM": delta["SSIM"],
                    "dLPIPS": delta["LPIPS"],
                    "decision_reasons": item.get("decision_reasons") or [],
                }
            )
        rows.append(row)
    return rows


def collect_rendercalib_searches(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for search_path in sorted(PHASE_S_ROOT.glob("rendercalib*v2[4-7]*/*/render_calibrated_search.json")):
        payload = _load_json(search_path)
        if not isinstance(payload, dict):
            continue
        version = _extract_version(str(payload.get("candidate_label") or search_path.as_posix()))
        if version not in VERSIONS:
            continue
        run_root = search_path.parent.parent
        running = _running_for(run_root, processes)
        events = [item for item in payload.get("events") or [] if isinstance(item, dict)]
        accepted_events = [item for item in events if item.get("action") == "accept" or item.get("accepted_by_strict_gate") is True]
        log_path = search_path.with_suffix(".log")
        latest_event = events[-1] if events else {}
        status = "running" if running else "complete"
        row = {
            "version": version,
            "family": "phase_s_render_calibrated_search",
            "scene": payload.get("scene") or search_path.parent.name,
            "candidate_label": payload.get("candidate_label"),
            "accepted": payload.get("accepted"),
            "status": status,
            "event_count": len(events),
            "strict_gate_pass_events": len(accepted_events),
            "best_objective": _finite_float(payload.get("best_objective")),
            "latest_event": {
                "trial_id": latest_event.get("trial_id"),
                "action": latest_event.get("action"),
                "accepted_by_strict_gate": latest_event.get("accepted_by_strict_gate"),
                "decision_reasons": latest_event.get("decision_reasons") or [],
                **{f"mean_d{k}": v for k, v in _delta_dict(latest_event.get("trainval_delta_mean")).items()},
            },
            "path": _rel(search_path),
            "log_path": _rel(log_path) if log_path.is_file() else None,
            "commands": _extract_log_commands(log_path),
            "running_commands": [proc["cmd"] for proc in running],
        }
        rows.append(row)
    return rows


def collect_qualitative_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(PHASE_S_ROOT.glob("*v2[4-7]*qualitative*/selected_views.json")):
        payload = _load_json(path)
        if not isinstance(payload, list):
            continue
        scenes = sorted({str(item.get("scene")) for item in payload if isinstance(item, dict) and item.get("scene")})
        md_path = path.with_name("qualitative_manifest.md")
        manifests.append(
            {
                "version": _extract_version(path.as_posix()),
                "path": _rel(path),
                "markdown_path": _rel(md_path) if md_path.is_file() else None,
                "view_count": len(payload),
                "scenes": scenes,
            }
        )
    return manifests


def _load_render_metrics(model_root: Path) -> dict[str, float | None]:
    results = _load_json(model_root / "results.json")
    if not isinstance(results, dict) or not results:
        return {"PSNR": None, "SSIM": None, "LPIPS": None}
    key = sorted(results)[-1]
    row = results.get(key)
    if not isinstance(row, dict):
        return {"PSNR": None, "SSIM": None, "LPIPS": None}
    return {metric: _finite_float(row.get(metric)) for metric in ("PSNR", "SSIM", "LPIPS")}


def _load_geometry_metrics(model_root: Path) -> dict[str, float | None]:
    geom_dir = model_root / "geometry_eval_colmap"
    files = sorted(geom_dir.glob("*.json"))
    if not files:
        return {"AbsRel": None, "DepthMAE": None, "NormalMeanDeg": None}
    payload = _load_json(files[-1])
    if not isinstance(payload, dict):
        return {"AbsRel": None, "DepthMAE": None, "NormalMeanDeg": None}
    depth = payload.get("depth") if isinstance(payload.get("depth"), dict) else {}
    normal = payload.get("normal") if isinstance(payload.get("normal"), dict) else {}
    return {
        "AbsRel": _finite_float(depth.get("abs_rel")),
        "DepthMAE": _finite_float(depth.get("mae")),
        "NormalMeanDeg": _finite_float(normal.get("mean_ang_deg")),
    }


def _load_model_metrics(model_root: Path) -> dict[str, float | None]:
    return {**_load_render_metrics(model_root), **_load_geometry_metrics(model_root)}


def _deltas(candidate: dict[str, float | None], baseline: dict[str, float | None] | None) -> dict[str, float | None]:
    if baseline is None:
        return {key: None for key in METRIC_KEYS}
    out: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        c = candidate.get(key)
        b = baseline.get(key)
        out[key] = c - b if c is not None and b is not None else None
    return out


def _failed_targets(delta: dict[str, float | None]) -> list[str]:
    targets = {
        "PSNR": lambda value: value > 0.0,
        "SSIM": lambda value: value > 0.0,
        "LPIPS": lambda value: value < 0.0,
        "AbsRel": lambda value: value < 0.0,
        "DepthMAE": lambda value: value < 0.0,
        "NormalMeanDeg": lambda value: value < 0.0,
    }
    failed = []
    for key, predicate in targets.items():
        value = delta.get(key)
        if value is None or not predicate(value):
            failed.append(key)
    return failed


def _control_kind(train_until_iteration: int | None, stage: str | None, path: Path) -> str | None:
    text = path.as_posix()
    if stage == "R27" and "baseline_sparse_depth" not in text and train_until_iteration == 4000:
        return "sparse_medium_4000"
    if train_until_iteration == 2200:
        return "short_2200"
    if train_until_iteration == 4000:
        return "medium_4000"
    return None


def _topology_from_cleanup(model_root: Path) -> dict[str, int | None]:
    summary = _load_json(model_root / "prism_debug/final_cleanup_summary.json")
    if not isinstance(summary, dict):
        return {"vertices": None, "triangles": None}
    vertices = summary.get("post_prune_vertex_count") or summary.get("pre_prune_vertex_count")
    triangles = summary.get("post_prune_triangle_count") or summary.get("pre_prune_triangle_count")
    return {
        "vertices": int(vertices) if isinstance(vertices, int) else None,
        "triangles": int(triangles) if isinstance(triangles, int) else None,
    }


def collect_stage_r_outputs(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controls = {name: _load_model_metrics(ROOT / rel) for name, rel in CONTROL_ROOTS.items()}
    for run_root in sorted(MESHSPLATOPT_ROOT.glob("stageR2[4-7]*")):
        stage = _extract_stage(run_root.name)
        if stage is None:
            continue
        running = _running_for(run_root, processes)
        commands = []
        for log_path in sorted((run_root / "logs").glob("*.log")):
            commands.extend(_extract_log_commands(log_path, limit=2))
        gate_path = run_root / "render_backed_checkpoint_gate_report.json"
        recovery_report_path = run_root / "real_tiny_recovery_report.json"
        expansion_path = run_root / "boundary_grid_fill_expansion_report.json"
        smoke_path = run_root / "checkpoint_adapter_smoke_report.json"

        if gate_path.is_file():
            gate = _load_json(gate_path)
            if not isinstance(gate, dict):
                continue
            row = {
                "stage": stage,
                "family": "stageR_render_backed_gate",
                "scene": "parking",
                "run": run_root.name,
                "accepted": gate.get("accepted"),
                "status": "running" if running else str(gate.get("status") or "complete").lower(),
                "metrics": gate.get("deltas") if isinstance(gate.get("deltas"), dict) else {},
                "topology": {
                    "dvertices": (gate.get("deltas") or {}).get("vertices") if isinstance(gate.get("deltas"), dict) else None,
                    "dtriangles": (gate.get("deltas") or {}).get("triangles") if isinstance(gate.get("deltas"), dict) else None,
                },
                "attribute_effect": "render_backed_checkpoint_gate",
                "path": _rel(gate_path),
                "commands": commands,
                "running_commands": [proc["cmd"] for proc in running],
                "reasons": gate.get("reasons") or [],
            }
            rows.append(row)

        if recovery_report_path.is_file():
            report = _load_json(recovery_report_path)
            if not isinstance(report, dict):
                continue
            model_root = ROOT / str(report.get("recovery_model", run_root / "recovery_model"))
            metrics = _load_model_metrics(model_root)
            train_until = report.get("train_until_iteration")
            control_kind = _control_kind(train_until if isinstance(train_until, int) else None, stage, run_root)
            delta_vs_control = _deltas(metrics, controls.get(control_kind) if control_kind else None)
            failed_targets = _failed_targets(delta_vs_control) if control_kind else []
            is_control = "baseline" in run_root.name
            status, failures = _exit_status_from_report(report)
            status = "running" if running else (status or "complete")
            row = {
                "stage": stage,
                "family": "stageR_recovery",
                "scene": "parking",
                "run": run_root.name,
                "accepted": None if is_control or not control_kind else not failed_targets,
                "decision": "control" if is_control else ("accepted" if control_kind and not failed_targets else "rejected"),
                "status": status,
                "failures": failures,
                "load_iteration": report.get("load_iteration"),
                "train_until_iteration": train_until,
                "metrics": metrics,
                "delta_vs_control": delta_vs_control,
                "failed_targets": failed_targets,
                "control": control_kind,
                "topology": _topology_from_cleanup(model_root),
                "attribute_effect": "teacher_recovery" + (" + sparse_depth" if "sparse_depth" in run_root.name else ""),
                "train_overrides": report.get("train_overrides") if isinstance(report.get("train_overrides"), dict) else {},
                "path": _rel(recovery_report_path),
                "commands": commands,
                "running_commands": [proc["cmd"] for proc in running],
            }
            rows.append(row)

        if expansion_path.is_file():
            expansion = _load_json(expansion_path)
            if isinstance(expansion, dict):
                rows.append(
                    {
                        "stage": stage,
                        "family": "stageR_grid_fill_selection",
                        "scene": "parking",
                        "run": run_root.name,
                        "accepted": expansion.get("status") == "PASS",
                        "status": "running" if running else str(expansion.get("status") or "complete").lower(),
                        "metrics": {},
                        "topology": {
                            "dvertices": expansion.get("added_vertices"),
                            "dtriangles": expansion.get("added_faces"),
                            "loop_vertices": expansion.get("loop_vertices"),
                            "area_2d": expansion.get("area_2d"),
                            "spacing": expansion.get("spacing"),
                        },
                        "attribute_effect": "grid_fill_geometry",
                        "path": _rel(expansion_path),
                        "commands": commands,
                        "running_commands": [proc["cmd"] for proc in running],
                    }
                )

        if smoke_path.is_file():
            smoke = _load_json(smoke_path)
            if isinstance(smoke, dict):
                fill = smoke.get("fill_report") if isinstance(smoke.get("fill_report"), dict) else {}
                rows.append(
                    {
                        "stage": stage,
                        "family": "stageR_checkpoint_adapter_smoke",
                        "scene": "parking",
                        "run": run_root.name,
                        "accepted": smoke.get("status") == "PASS",
                        "status": "running" if running else str(smoke.get("status") or "complete").lower(),
                        "metrics": {},
                        "topology": {
                            "dvertices": fill.get("vertices_after") - fill.get("vertices_before")
                            if isinstance(fill.get("vertices_after"), int) and isinstance(fill.get("vertices_before"), int)
                            else None,
                            "dtriangles": fill.get("triangles_after") - fill.get("triangles_before")
                            if isinstance(fill.get("triangles_after"), int) and isinstance(fill.get("triangles_before"), int)
                            else None,
                        },
                        "attribute_effect": fill.get("reason") or "checkpoint_adapter_smoke",
                        "path": _rel(smoke_path),
                        "commands": commands,
                        "running_commands": [proc["cmd"] for proc in running],
                    }
                )
    return rows


def collect() -> dict[str, Any]:
    processes = _read_processes()
    return {
        "active_processes": processes,
        "phase_s_gates": collect_phase_s_gates(processes),
        "render_calibrated_searches": collect_rendercalib_searches(processes),
        "qualitative_manifests": collect_qualitative_manifests(),
        "stageR_outputs": collect_stage_r_outputs(processes),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    numeric = _finite_float(value)
    if numeric is not None:
        return f"{numeric:.{digits}f}"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value) if value else ""
    return str(value)


def _metric_cells(delta: dict[str, Any], prefix: str = "") -> str:
    return " | ".join(_fmt(delta.get(prefix + key if prefix else key)) for key in ("PSNR", "SSIM", "LPIPS"))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# MeshSplatOpt v24-v27 Output Audit",
        "",
        "This report is generated by `scripts/car_model/meshsplatopt_audit_v24_v27_outputs.py` from existing JSON/log artifacts.",
        "",
        "## Phase-S Gate Summary",
        "",
        "| version | scene | status | accepted | offsets | mean dPSNR | mean dSSIM | mean dLPIPS | topology/attribute effect | artifact |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["phase_s_gates"]:
        effect = row["effect"]
        effect_text = (
            f"faces={_fmt(effect.get('accepted_faces'), 0)}, "
            f"dV={_fmt(effect.get('dvertices'), 0)}, dT={_fmt(effect.get('dtriangles'), 0)}, "
            f"attr={_fmt(effect.get('attribute_changed'))}, max_attr={_fmt(effect.get('max_attribute_delta'))}"
        )
        lines.append(
            f"| {row['version']} | {row['scene']} | {row['status']} | {_fmt(row['accepted'])} | {row['offset_count']} | "
            f"{_fmt(row['mean_dPSNR'])} | {_fmt(row['mean_dSSIM'])} | {_fmt(row['mean_dLPIPS'])} | "
            f"{effect_text} | `{row['path']}` |"
        )

    lines.extend(["", "## Phase-S Offset Deltas", ""])
    lines.append("| version | scene | offset | accepted | dPSNR | dSSIM | dLPIPS | reasons |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in payload["phase_s_gates"]:
        for offset in row["offsets"]:
            lines.append(
                f"| {row['version']} | {row['scene']} | {offset.get('offset')} | {_fmt(offset.get('accepted'))} | "
                f"{_fmt(offset.get('dPSNR'))} | {_fmt(offset.get('dSSIM'))} | {_fmt(offset.get('dLPIPS'))} | "
                f"{_fmt(offset.get('decision_reasons'))} |"
            )

    lines.extend(["", "## Render-Calibrated Searches", ""])
    lines.append("| version | scene | status | accepted | events | strict-pass events | latest trial | latest action | mean dPSNR | mean dSSIM | mean dLPIPS | artifact |")
    lines.append("|---|---|---|---:|---:|---:|---|---|---:|---:|---:|---|")
    for row in payload["render_calibrated_searches"]:
        latest = row["latest_event"]
        lines.append(
            f"| {row['version']} | {row['scene']} | {row['status']} | {_fmt(row['accepted'])} | {row['event_count']} | "
            f"{row['strict_gate_pass_events']} | {latest.get('trial_id') or ''} | {latest.get('action') or ''} | "
            f"{_fmt(latest.get('mean_dPSNR'))} | {_fmt(latest.get('mean_dSSIM'))} | {_fmt(latest.get('mean_dLPIPS'))} | "
            f"`{row['path']}` |"
        )

    lines.extend(["", "## MeshSplatOpt Stage R24-R27 Outputs", ""])
    lines.append("| stage | family | run | status | decision | accepted | metrics | delta vs control | topology/effect | artifact |")
    lines.append("|---|---|---|---|---|---:|---|---|---|---|")
    for row in payload["stageR_outputs"]:
        metrics = row.get("metrics") or {}
        metric_text = ", ".join(f"{key}={_fmt(metrics.get(key))}" for key in METRIC_KEYS if metrics.get(key) is not None)
        delta = row.get("delta_vs_control") or {}
        delta_text = ", ".join(f"d{key}={_fmt(delta.get(key))}" for key in METRIC_KEYS if delta.get(key) is not None)
        topo = row.get("topology") or {}
        topo_text = ", ".join(f"{key}={_fmt(value)}" for key, value in topo.items() if value is not None)
        if row.get("attribute_effect"):
            topo_text = f"{topo_text}; {row['attribute_effect']}" if topo_text else str(row["attribute_effect"])
        decision = row.get("decision") or ("accepted" if row.get("accepted") is True else "rejected" if row.get("accepted") is False else "")
        if row.get("failed_targets") and decision != "control":
            decision = f"{decision}: {','.join(row['failed_targets'])}"
        lines.append(
            f"| {row.get('stage')} | {row.get('family')} | {row.get('run')} | {row.get('status')} | "
            f"{decision} | {_fmt(row.get('accepted'))} | {metric_text or 'n/a'} | {delta_text or row.get('control') or 'n/a'} | "
            f"{topo_text or 'n/a'} | `{row.get('path')}` |"
        )

    lines.extend(["", "## Qualitative Manifests", ""])
    if payload["qualitative_manifests"]:
        lines.append("| version | views | scenes | selected views JSON | manifest MD |")
        lines.append("|---|---:|---|---|---|")
        for item in payload["qualitative_manifests"]:
            lines.append(
                f"| {item.get('version')} | {item.get('view_count')} | {', '.join(item.get('scenes') or [])} | "
                f"`{item.get('path')}` | `{item.get('markdown_path')}` |"
            )
    else:
        lines.append("No v24-v27 qualitative manifest was found.")

    lines.extend(["", "## Running Commands", ""])
    active = payload["active_processes"]
    if active:
        lines.append("| pid | elapsed_s | command |")
        lines.append("|---:|---:|---|")
        for proc in active:
            lines.append(f"| {proc['pid']} | {proc['etimes']} | `{proc['cmd']}` |")
    else:
        lines.append("No active v24-v27 or stageR24-stageR27 commands were visible in `ps`.")

    lines.extend(["", "## Recorded Commands", ""])
    command_rows = []
    for section in ("render_calibrated_searches", "stageR_outputs"):
        for row in payload[section]:
            for cmd in row.get("commands") or []:
                command_rows.append((row.get("version") or row.get("stage"), row.get("scene"), row.get("run") or row.get("candidate_label"), cmd))
    if command_rows:
        lines.append("| source | scene | run | command |")
        lines.append("|---|---|---|---|")
        for source, scene, run, cmd in command_rows[-40:]:
            lines.append(f"| {source or ''} | {scene or ''} | {run or ''} | `{cmd}` |")
    else:
        lines.append("No `$ ...` commands were found in the scanned logs.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit current MeshSplatOpt/SPCarNet v24-v27 outputs.")
    parser.add_argument("--markdown-out", default="docs/car_model/meshsplatopt_v24_v27_audit_report.md")
    parser.add_argument("--json-out", default="docs/car_model/meshsplatopt_v24_v27_audit_report.json")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    payload = collect()
    md_path = ROOT / args.markdown_out
    write_markdown(md_path, payload)
    print(f"Wrote {md_path.relative_to(ROOT)}")
    if not args.no_json:
        json_path = ROOT / args.json_out
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {json_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

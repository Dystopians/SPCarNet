#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any


SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai")
METRICS = ("PSNR", "SSIM", "LPIPS")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _metric_row(payload: dict[str, Any] | None, method: str) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    row = payload.get(method)
    if not isinstance(row, dict):
        return None
    out: dict[str, float] = {}
    for key in METRICS:
        value = _float(row.get(key))
        if value is None:
            return None
        out[key] = value
    return out


def _score(metrics: dict[str, float]) -> float:
    return metrics["PSNR"] + 20.0 * metrics["SSIM"] - 20.0 * metrics["LPIPS"]


def _delta(method: dict[str, float] | None, base: dict[str, float] | None) -> dict[str, float | None]:
    if method is None or base is None:
        return {"dPSNR": None, "dSSIM": None, "dLPIPS": None}
    return {
        "dPSNR": method["PSNR"] - base["PSNR"],
        "dSSIM": method["SSIM"] - base["SSIM"],
        "dLPIPS": method["LPIPS"] - base["LPIPS"],
    }


def _strict_rgb_win(delta: dict[str, Any]) -> bool:
    dpsnr = _float(delta.get("dPSNR") if "dPSNR" in delta else delta.get("PSNR"))
    dssim = _float(delta.get("dSSIM") if "dSSIM" in delta else delta.get("SSIM"))
    dlpips = _float(delta.get("dLPIPS") if "dLPIPS" in delta else delta.get("LPIPS"))
    return dpsnr is not None and dssim is not None and dlpips is not None and dpsnr > 0.0 and dssim > 0.0 and dlpips < 0.0


def _summary_delta(gate: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(gate, dict):
        return {"dPSNR": None, "dSSIM": None, "dLPIPS": None}
    summary = gate.get("trainval_delta_summary")
    if isinstance(summary, dict):
        return {
            "dPSNR": _float((summary.get("PSNR") or {}).get("mean")),
            "dSSIM": _float((summary.get("SSIM") or {}).get("mean")),
            "dLPIPS": _float((summary.get("LPIPS") or {}).get("mean")),
        }
    delta = gate.get("trainval_delta")
    if isinstance(delta, dict):
        return {"dPSNR": _float(delta.get("PSNR")), "dSSIM": _float(delta.get("SSIM")), "dLPIPS": _float(delta.get("LPIPS"))}
    return {"dPSNR": None, "dSSIM": None, "dLPIPS": None}


def _test_delta(gate: dict[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(gate, dict):
        return {"dPSNR": None, "dSSIM": None, "dLPIPS": None}
    delta = gate.get("test_delta_report_only")
    if not isinstance(delta, dict):
        return {"dPSNR": None, "dSSIM": None, "dLPIPS": None}
    return {"dPSNR": _float(delta.get("PSNR")), "dSSIM": _float(delta.get("SSIM")), "dLPIPS": _float(delta.get("LPIPS"))}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    numeric = _float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.{digits}f}"


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _read_phasej_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    return {str(row.get("scene")): row for row in rows if isinstance(row, dict) and row.get("scene")}


def _clean_candidates(scene: str, clean_root: Path, clean_methods: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    results_path = clean_root / scene / "results.json"
    results = _load_json(results_path)
    candidates: list[dict[str, Any]] = []
    for method in clean_methods:
        metrics = _metric_row(results, method)
        row: dict[str, Any] = {
            "scene": scene,
            "method": method,
            "path": str(results_path),
            "available": metrics is not None,
            "PSNR": None if metrics is None else metrics["PSNR"],
            "SSIM": None if metrics is None else metrics["SSIM"],
            "LPIPS": None if metrics is None else metrics["LPIPS"],
            "clean_score": None if metrics is None else _score(metrics),
        }
        row["status"] = "complete" if metrics is not None else ("missing_results_json" if not results_path.is_file() else "missing_method_or_metrics")
        candidates.append(row)
    complete = [row for row in candidates if row["available"]]
    if not complete:
        return candidates, None
    best = max(complete, key=lambda row: (float(row["clean_score"]), float(row["PSNR"])))
    return candidates, best


def _phase_s_decision(scene: str, phase_s_root: Path) -> tuple[dict[str, Any] | None, Path]:
    direct_path = phase_s_root / "decisions" / f"{scene}_decision.json"
    payload = _load_json(direct_path)
    if isinstance(payload, dict):
        return payload, direct_path
    summary_path = phase_s_root / scene / "phasek_scene_summary.json"
    summary = _load_json(summary_path)
    if isinstance(summary, dict) and isinstance(summary.get("decision"), dict):
        return summary["decision"], summary_path
    return None, direct_path


def _phase_s_status(decision: dict[str, Any] | None, strict_gate: dict[str, Any] | None) -> str:
    if strict_gate is not None:
        return "strict_accepted" if strict_gate.get("accepted") is True else "strict_rejected"
    if decision is None:
        return "missing_single_and_strict"
    if decision.get("accepted") is True:
        return "single_accepted_strict_missing"
    return "single_rejected_strict_not_run"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    clean_methods = [method.strip() for method in args.clean_methods.split(",") if method.strip()]
    clean_root = Path(args.clean_root)
    phasej_summary = Path(args.phasej_summary_json)
    phase_s_root = Path(args.phase_s_root)
    phase_s_strict_root = Path(args.phase_s_strict_root)
    phasej_by_scene = _read_phasej_rows(phasej_summary)

    rows: list[dict[str, Any]] = []
    clean_candidate_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_missing: list[str] = []
        candidates, clean_best = _clean_candidates(scene, clean_root, clean_methods)
        clean_candidate_rows.extend(candidates)
        clean_complete = clean_best is not None
        if not clean_complete:
            scene_missing.append("clean_best")
            missing_rows.append({"scene": scene, "evidence": "clean_best", "path": str(clean_root / scene / "results.json"), "reason": "no complete clean metric row"})

        clean_best_metrics = None
        if clean_best is not None:
            clean_best_metrics = {"PSNR": clean_best["PSNR"], "SSIM": clean_best["SSIM"], "LPIPS": clean_best["LPIPS"]}

        phasej = phasej_by_scene.get(scene)
        phasej_present = phasej is not None
        if not phasej_present:
            scene_missing.append("phase_j_full9")
            missing_rows.append({"scene": scene, "evidence": "phase_j_full9", "path": str(phasej_summary), "reason": "scene absent from Phase-J full9 summary"})

        phasej_metrics = phasej.get("method") if isinstance(phasej, dict) and isinstance(phasej.get("method"), dict) else None
        phasej_clean_method = phasej.get("clean_baseline_method") if isinstance(phasej, dict) else None
        phasej_delta_clean_best = _delta(phasej_metrics, clean_best_metrics)
        phasej_delta_recorded = phasej.get("delta_vs_clean") if isinstance(phasej, dict) and isinstance(phasej.get("delta_vs_clean"), dict) else {}
        phasej_model = Path(str(phasej.get("model"))) if isinstance(phasej, dict) and phasej.get("model") else None

        decision, decision_path = _phase_s_decision(scene, phase_s_root)
        single_present = decision is not None
        if not single_present:
            scene_missing.append("phase_s_single_gate")
            missing_rows.append({"scene": scene, "evidence": "phase_s_single_gate", "path": str(decision_path), "reason": "single-gate decision JSON missing"})

        strict_path = phase_s_strict_root / scene / "multifold_trainval_gate.json"
        strict_gate = _load_json(strict_path)
        strict_present = isinstance(strict_gate, dict)
        if not strict_present:
            scene_missing.append("phase_s_strict_four_offset")
            missing_rows.append({"scene": scene, "evidence": "phase_s_strict_four_offset", "path": str(strict_path), "reason": "strict four-offset train-val gate JSON missing"})

        single_trainval_delta = _summary_delta(decision)
        single_test_delta = _test_delta(decision)
        strict_trainval_delta = _summary_delta(strict_gate)
        strict_test_delta = _test_delta(strict_gate)
        phase_s_status = _phase_s_status(decision, strict_gate if strict_present else None)
        phase_s_selected_label = None
        if isinstance(strict_gate, dict):
            phase_s_selected_label = strict_gate.get("selected_label")
        if phase_s_selected_label is None and isinstance(decision, dict):
            phase_s_selected_label = decision.get("selected_label")

        row = {
            "scene": scene,
            "clean_status": "complete" if clean_complete else "missing",
            "clean_best_method": None if clean_best is None else clean_best["method"],
            "clean_best_PSNR": None if clean_best is None else clean_best["PSNR"],
            "clean_best_SSIM": None if clean_best is None else clean_best["SSIM"],
            "clean_best_LPIPS": None if clean_best is None else clean_best["LPIPS"],
            "clean_best_score": None if clean_best is None else clean_best["clean_score"],
            "phasej_status": "complete" if phasej_present else "missing",
            "phasej_method": None if phasej is None else phasej.get("method_name"),
            "phasej_clean_baseline_method": phasej_clean_method,
            "phasej_clean_best_same_as_recorded": bool(clean_best is not None and phasej_clean_method == clean_best["method"]),
            "phasej_model_exists": bool(phasej_model is not None and phasej_model.is_dir()),
            "phasej_PSNR": None if phasej_metrics is None else phasej_metrics.get("PSNR"),
            "phasej_SSIM": None if phasej_metrics is None else phasej_metrics.get("SSIM"),
            "phasej_LPIPS": None if phasej_metrics is None else phasej_metrics.get("LPIPS"),
            "phasej_dPSNR_recorded_vs_clean": phasej_delta_recorded.get("dPSNR"),
            "phasej_dSSIM_recorded_vs_clean": phasej_delta_recorded.get("dSSIM"),
            "phasej_dLPIPS_recorded_vs_clean": phasej_delta_recorded.get("dLPIPS"),
            "phasej_strict_rgb_win_recorded_vs_clean": _strict_rgb_win(phasej_delta_recorded),
            "phasej_dPSNR_vs_clean_best": phasej_delta_clean_best["dPSNR"],
            "phasej_dSSIM_vs_clean_best": phasej_delta_clean_best["dSSIM"],
            "phasej_dLPIPS_vs_clean_best": phasej_delta_clean_best["dLPIPS"],
            "phasej_strict_rgb_win_vs_clean_best": _strict_rgb_win(phasej_delta_clean_best),
            "phase_s_status": phase_s_status,
            "phase_s_single_present": single_present,
            "phase_s_single_accepted": None if decision is None else decision.get("accepted"),
            "phase_s_single_selected_label": None if decision is None else decision.get("selected_label"),
            "phase_s_single_reasons": "" if decision is None else ";".join(str(x) for x in (decision.get("decision_reasons") or [])),
            "phase_s_single_trainval_dPSNR": single_trainval_delta["dPSNR"],
            "phase_s_single_trainval_dSSIM": single_trainval_delta["dSSIM"],
            "phase_s_single_trainval_dLPIPS": single_trainval_delta["dLPIPS"],
            "phase_s_single_trainval_all_axis_win": _strict_rgb_win(single_trainval_delta),
            "phase_s_single_test_dPSNR_report_only": single_test_delta["dPSNR"],
            "phase_s_single_test_dSSIM_report_only": single_test_delta["dSSIM"],
            "phase_s_single_test_dLPIPS_report_only": single_test_delta["dLPIPS"],
            "phase_s_strict_present": strict_present,
            "phase_s_strict_accepted": None if not isinstance(strict_gate, dict) else strict_gate.get("accepted"),
            "phase_s_strict_selected_label": phase_s_selected_label,
            "phase_s_strict_reasons": "" if not isinstance(strict_gate, dict) else ";".join(str(x) for x in (strict_gate.get("decision_reasons") or [])),
            "phase_s_strict_offsets": "" if not isinstance(strict_gate, dict) else ",".join(str(x) for x in (strict_gate.get("offsets") or [])),
            "phase_s_strict_trainval_dPSNR": strict_trainval_delta["dPSNR"],
            "phase_s_strict_trainval_dSSIM": strict_trainval_delta["dSSIM"],
            "phase_s_strict_trainval_dLPIPS": strict_trainval_delta["dLPIPS"],
            "phase_s_strict_trainval_all_axis_win": _strict_rgb_win(strict_trainval_delta),
            "phase_s_strict_test_dPSNR_report_only": strict_test_delta["dPSNR"],
            "phase_s_strict_test_dSSIM_report_only": strict_test_delta["dSSIM"],
            "phase_s_strict_test_dLPIPS_report_only": strict_test_delta["dLPIPS"],
            "phase_s_selection_uses_test": None if decision is None else decision.get("selection_uses_test"),
            "missing_evidence": ";".join(scene_missing),
        }
        rows.append(row)

    summary = {
        "scene_count": len(rows),
        "clean_complete": sum(1 for row in rows if row["clean_status"] == "complete"),
        "phasej_complete": sum(1 for row in rows if row["phasej_status"] == "complete"),
        "phasej_strict_rgb_wins_recorded_vs_clean": sum(1 for row in rows if row["phasej_strict_rgb_win_recorded_vs_clean"]),
        "phasej_strict_rgb_wins_vs_clean_best": sum(1 for row in rows if row["phasej_strict_rgb_win_vs_clean_best"]),
        "phase_s_single_present": sum(1 for row in rows if row["phase_s_single_present"]),
        "phase_s_single_accepted": sum(1 for row in rows if row["phase_s_single_accepted"] is True),
        "phase_s_strict_present": sum(1 for row in rows if row["phase_s_strict_present"]),
        "phase_s_strict_accepted": sum(1 for row in rows if row["phase_s_strict_accepted"] is True),
        "phase_s_strict_rejected": sum(1 for row in rows if row["phase_s_strict_accepted"] is False),
        "phase_s_strict_trainval_all_axis_wins": sum(1 for row in rows if row["phase_s_strict_trainval_all_axis_win"]),
        "missing_evidence_count": len(missing_rows),
        "missing_scenes": sorted({row["scene"] for row in missing_rows}),
        "mean_phasej_dPSNR_vs_clean_best": _mean(rows, "phasej_dPSNR_vs_clean_best"),
        "mean_phasej_dSSIM_vs_clean_best": _mean(rows, "phasej_dSSIM_vs_clean_best"),
        "mean_phasej_dLPIPS_vs_clean_best": _mean(rows, "phasej_dLPIPS_vs_clean_best"),
        "mean_phase_s_strict_trainval_dPSNR": _mean(rows, "phase_s_strict_trainval_dPSNR"),
        "mean_phase_s_strict_trainval_dSSIM": _mean(rows, "phase_s_strict_trainval_dSSIM"),
        "mean_phase_s_strict_trainval_dLPIPS": _mean(rows, "phase_s_strict_trainval_dLPIPS"),
    }
    summary["full9_clean_phasej_phase_s_closed"] = (
        summary["clean_complete"] == len(rows)
        and summary["phasej_complete"] == len(rows)
        and summary["phase_s_strict_present"] == len(rows)
        and summary["phase_s_strict_accepted"] == len(rows)
    )

    return {
        "collector": "meshsplatopt_collect_full9_paper_loop_status",
        "inputs": {
            "clean_root": str(clean_root),
            "clean_methods": clean_methods,
            "phasej_summary_json": str(phasej_summary),
            "phase_s_root": str(phase_s_root),
            "phase_s_strict_root": str(phase_s_strict_root),
            "scenes": scenes,
        },
        "summary": summary,
        "rows": rows,
        "clean_candidate_rows": clean_candidate_rows,
        "missing_rows": missing_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(report: dict[str, Any], path: Path, artifact_dir: Path | None = None) -> None:
    summary = report["summary"]
    rows = report["rows"]
    missing_rows = report["missing_rows"]
    artifact_dir = artifact_dir or path.parent
    lines = [
        "# Full9 Paper-Loop Evidence Status",
        "",
        "This report is generated mechanically from existing clean MeshSplatting, Phase-J, and Phase-S artifacts. It is a status collector, not a new experiment, and it treats missing rows as explicit evidence gaps.",
        "",
        "## Summary",
        "",
        f"- Clean-best rows: `{summary['clean_complete']} / {summary['scene_count']}`",
        f"- Phase-J full9 rows: `{summary['phasej_complete']} / {summary['scene_count']}`",
        f"- Phase-J strict RGB wins vs recorded clean baseline: `{summary['phasej_strict_rgb_wins_recorded_vs_clean']} / {summary['scene_count']}`",
        f"- Phase-J strict RGB wins vs clean-best row selected here: `{summary['phasej_strict_rgb_wins_vs_clean_best']} / {summary['scene_count']}`",
        f"- Phase-S single-gate decisions: `{summary['phase_s_single_present']} / {summary['scene_count']}`; accepted `{summary['phase_s_single_accepted']} / {summary['scene_count']}`",
        f"- Phase-S strict four-offset gates: `{summary['phase_s_strict_present']} / {summary['scene_count']}`; accepted `{summary['phase_s_strict_accepted']} / {summary['scene_count']}`; rejected `{summary['phase_s_strict_rejected']} / {summary['scene_count']}`",
        f"- Phase-S strict all-axis train-val wins: `{summary['phase_s_strict_trainval_all_axis_wins']} / {summary['phase_s_strict_present']}`",
        f"- Missing evidence entries: `{summary['missing_evidence_count']}`",
        f"- Full9 clean/Phase-J/Phase-S closure: `{summary['full9_clean_phasej_phase_s_closed']}`",
        "",
        "## Scene Status",
        "",
        "| scene | clean-best | Phase-J vs clean-best | Phase-J strict | Phase-S single | Phase-S strict | Phase-S strict mean delta | missing evidence |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    for row in rows:
        clean = "missing" if row["clean_status"] != "complete" else f"{row['clean_best_method']} ({_fmt(row['clean_best_PSNR'], 3)}/{_fmt(row['clean_best_SSIM'], 3)}/{_fmt(row['clean_best_LPIPS'], 3)})"
        phasej_delta = f"{_fmt(row['phasej_dPSNR_vs_clean_best'])} / {_fmt(row['phasej_dSSIM_vs_clean_best'])} / {_fmt(row['phasej_dLPIPS_vs_clean_best'])}"
        phasej_strict = "yes" if row["phasej_strict_rgb_win_vs_clean_best"] else "no"
        single = "missing"
        if row["phase_s_single_present"]:
            single = "accept" if row["phase_s_single_accepted"] is True else "reject"
        strict = "missing"
        if row["phase_s_strict_present"]:
            strict = "accept" if row["phase_s_strict_accepted"] is True else "reject"
        strict_delta = f"{_fmt(row['phase_s_strict_trainval_dPSNR'])} / {_fmt(row['phase_s_strict_trainval_dSSIM'])} / {_fmt(row['phase_s_strict_trainval_dLPIPS'])}"
        lines.append(
            f"| {row['scene']} | {clean} | {phasej_delta} | {phasej_strict} | {single} | {strict} | {strict_delta} | {row['missing_evidence'] or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Missing Rows",
            "",
        ]
    )
    if missing_rows:
        lines.extend(["| scene | evidence | path | reason |", "|---|---|---|---|"])
        for row in missing_rows:
            lines.append(f"| {row['scene']} | {row['evidence']} | `{row['path']}` | {row['reason']} |")
    else:
        lines.append("No missing evidence rows under the configured roots.")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `clean-best` is selected from the configured clean methods by `PSNR + 20 * SSIM - 20 * LPIPS`; this exposes when Phase-J used a different recorded clean row.",
            "- Phase-J is checked against both its recorded clean baseline and the clean-best row selected by this collector.",
            "- Phase-S closure is intentionally stricter than single-gate acceptance: a paper-loop row is not closed unless the strict four-offset train-val gate exists and accepts.",
            "- Held-out Phase-S test deltas are report-only and are not used to decide acceptance.",
            "",
            "## Artifacts",
            "",
            f"- summary JSON: `{artifact_dir / 'full9_paper_loop_status.json'}`",
            f"- scene CSV: `{artifact_dir / 'full9_paper_loop_status.csv'}`",
            f"- clean candidates CSV: `{artifact_dir / 'full9_clean_candidate_rows.csv'}`",
            f"- missing rows CSV: `{artifact_dir / 'full9_missing_rows.csv'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(report: dict[str, Any], out_dir: Path, doc_out: str | None = None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "full9_paper_loop_status.json"
    scene_csv_path = out_dir / "full9_paper_loop_status.csv"
    clean_csv_path = out_dir / "full9_clean_candidate_rows.csv"
    missing_csv_path = out_dir / "full9_missing_rows.csv"
    md_path = out_dir / "full9_paper_loop_status.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(scene_csv_path, report["rows"])
    _write_csv(clean_csv_path, report["clean_candidate_rows"])
    _write_csv(missing_csv_path, report["missing_rows"])
    _write_md(report, md_path)
    outputs = {
        "json": str(json_path),
        "scene_csv": str(scene_csv_path),
        "clean_candidate_csv": str(clean_csv_path),
        "missing_csv": str(missing_csv_path),
        "markdown": str(md_path),
    }
    if doc_out:
        doc_path = Path(doc_out)
        _write_md(report, doc_path, out_dir)
        outputs["doc_markdown"] = str(doc_path)
    return outputs


def _maybe_log_wandb(report: dict[str, Any], outputs: dict[str, str], args: argparse.Namespace) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        print(f"[wandb] skipped logging because wandb is unavailable: {exc}")
        return

    run = None
    try:
        config = {
            "collector": report.get("collector"),
            "inputs": report.get("inputs"),
            "out_dir": args.out_dir,
            "doc_out": args.doc_out,
            "fail_on_missing": bool(args.fail_on_missing),
        }
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=config,
        )
        summary_log: dict[str, int | float] = {}
        for key, value in report.get("summary", {}).items():
            if isinstance(value, bool):
                summary_log[key] = int(value)
            elif isinstance(value, int):
                summary_log[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                summary_log[key] = value
        wandb.log(summary_log)

        rows = report.get("rows") or []
        if rows:
            columns = list(rows[0].keys())
            table = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])
            wandb.log({"scene_status": table})

        missing_rows = report.get("missing_rows") or []
        if missing_rows:
            columns = list(missing_rows[0].keys())
            table = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in missing_rows])
            wandb.log({"missing_rows": table})

        artifact = wandb.Artifact(Path(args.out_dir).name, type="full9-paper-loop-status")
        for path in outputs.values():
            p = Path(path)
            if p.is_file():
                artifact.add_file(str(p))
        wandb.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - external service failures should not corrupt local evidence
        print(f"[wandb] skipped logging due to error: {exc}")
    finally:
        if run is not None:
            wandb.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect full9 clean-best, Phase-J, and Phase-S paper-loop evidence status.")
    parser.add_argument("--clean-root", default="outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k")
    parser.add_argument("--clean-methods", default="ours_26000,ours_30000")
    parser.add_argument(
        "--phasej-summary-json",
        default="outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix/phasef_ela_eval_summary_phasej_guarded_adaptedge_full9.json",
    )
    parser.add_argument("--phase-s-root", default="outputs/carnet/meshsplatopt/ecsr_phase_s/facelocal_gaincert_v1_cached_dense16_20260512")
    parser.add_argument(
        "--phase-s-strict-root",
        default="outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate/facelocal_gaincert_v1_cached_dense16_20260512",
    )
    parser.add_argument("--out-dir", default="outputs/carnet/meshsplatopt/full9_paper_loop_status")
    parser.add_argument("--doc-out", default="")
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="full9_paper_loop_status")
    parser.add_argument("--wandb_name", default="collect_full9_paper_loop_status")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    report = collect(args)
    outputs = write_outputs(report, Path(args.out_dir), args.doc_out or None)
    _maybe_log_wandb(report, outputs, args)
    print(json.dumps({"summary": report["summary"], "outputs": outputs}, indent=2))
    if args.fail_on_missing and report["summary"]["missing_evidence_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Select a fixed Phase-R ECSR candidate ladder without looking at test scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("PSNR", "SSIM", "LPIPS")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_candidate(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("candidate must be NAME=ROOT_TEMPLATE")
    name, template = spec.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("candidate name is empty")
    if "{scene}" not in template:
        raise argparse.ArgumentTypeError("candidate root template must contain {scene}")
    return name, template


def _parse_override(spec: str) -> tuple[tuple[str, str], str]:
    try:
        scene_name, path = spec.split("=", 1)
        scene, name = scene_name.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("override must be SCENE:CANDIDATE=ROOT") from exc
    if not scene or not name or not path:
        raise argparse.ArgumentTypeError("override has an empty scene, candidate, or root")
    return (scene, name), path


def _decision_paths(scene: str, root: Path) -> list[Path]:
    return [
        root / "decisions" / f"{scene}_decision.json",
        root / scene / "multifold_trainval_gate.json",
    ]


def _decision_path(scene: str, root: Path) -> Path:
    for path in _decision_paths(scene, root):
        if path.exists():
            return path
    return _decision_paths(scene, root)[0]


def _metric_delta(delta: dict[str, float] | None, metric: str) -> float:
    if not delta:
        return 0.0
    return float(delta.get(metric, 0.0))


def _strict_rgb_win(delta: dict[str, float] | None) -> bool:
    if not delta:
        return False
    return (
        _metric_delta(delta, "PSNR") > 0.0
        and _metric_delta(delta, "SSIM") > 0.0
        and _metric_delta(delta, "LPIPS") < 0.0
    )


def _fmt_delta(delta: dict[str, float] | None, metric: str) -> str:
    value = _metric_delta(delta, metric)
    return f"{value:+.6f}"


def _normalized_trainval_delta(decision: dict[str, Any]) -> dict[str, float]:
    if isinstance(decision.get("trainval_delta"), dict):
        return {metric: _metric_delta(decision.get("trainval_delta"), metric) for metric in METRICS}
    summary = decision.get("trainval_delta_summary")
    if isinstance(summary, dict):
        return {metric: _metric_delta(summary.get(metric), "mean") for metric in METRICS}
    return {}


def _decision_kind(path: Path) -> str:
    if path.name == "multifold_trainval_gate.json":
        return "multifold_trainval_gate"
    return "single_trainval_gate"


def _select_scene(
    scene: str,
    candidates: list[tuple[str, str]],
    overrides: dict[tuple[str, str], str],
    force_fallback_scenes: set[str],
) -> dict[str, Any]:
    if scene in force_fallback_scenes:
        return {
            "scene": scene,
            "selected": "fallback",
            "accepted": False,
            "selection_reason": "predeclared_force_fallback_scene",
            "candidate": "fallback",
            "root": "",
            "decision_path": "",
            "exists": False,
            "selected_label": "fallback",
            "candidate_label": "fallback",
            "reasons": ["predeclared_force_fallback_scene"],
            "trainval_delta": {},
            "test_delta_report_only": {},
            "selection_uses_test": False,
            "checked": [],
        }
    checked: list[dict[str, Any]] = []
    for name, template in candidates:
        root = Path(overrides.get((scene, name), template.format(scene=scene)))
        decision_path = _decision_path(scene, root)
        if not decision_path.exists():
            checked.append(
                {
                    "candidate": name,
                    "root": str(root),
                    "decision_path": str(decision_path),
                    "exists": False,
                    "accepted": False,
                    "reasons": ["missing_decision"],
                }
            )
            continue
        decision = _load_json(decision_path)
        trainval_delta = _normalized_trainval_delta(decision)
        row = {
            "candidate": name,
            "root": str(root),
            "decision_path": str(decision_path),
            "decision_kind": _decision_kind(decision_path),
            "exists": True,
            "accepted": bool(decision.get("accepted")),
            "selected_label": decision.get("selected_label"),
            "candidate_label": decision.get("candidate_label"),
            "reasons": decision.get("decision_reasons", []),
            "trainval_delta": trainval_delta,
            "test_delta_report_only": decision.get("test_delta_report_only", {}),
            "selection_uses_test": bool(decision.get("selection_uses_test")),
        }
        checked.append(row)
        if row["accepted"]:
            return {
                "scene": scene,
                "selected": name,
                "accepted": True,
                "selection_reason": "first_trainval_accepted_candidate",
                **row,
                "checked": checked,
            }
    return {
        "scene": scene,
        "selected": "fallback",
        "accepted": False,
        "selection_reason": "no_trainval_accepted_candidate",
        "candidate": "fallback",
        "root": "",
        "decision_path": "",
        "exists": False,
        "selected_label": "fallback",
        "candidate_label": "fallback",
        "reasons": ["no_trainval_accepted_candidate"],
        "trainval_delta": {},
        "test_delta_report_only": {},
        "selection_uses_test": False,
        "checked": checked,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scene",
        "selected",
        "accepted",
        "selected_label",
        "trainval_dPSNR",
        "trainval_dSSIM",
        "trainval_dLPIPS",
        "test_dPSNR_report_only",
        "test_dSSIM_report_only",
        "test_dLPIPS_report_only",
        "strict_test_rgb_win_report_only",
        "decision_kind",
        "reasons",
        "decision_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scene": row["scene"],
                    "selected": row["selected"],
                    "accepted": row["accepted"],
                    "selected_label": row["selected_label"],
                    "trainval_dPSNR": _metric_delta(row.get("trainval_delta"), "PSNR"),
                    "trainval_dSSIM": _metric_delta(row.get("trainval_delta"), "SSIM"),
                    "trainval_dLPIPS": _metric_delta(row.get("trainval_delta"), "LPIPS"),
                    "test_dPSNR_report_only": _metric_delta(row.get("test_delta_report_only"), "PSNR"),
                    "test_dSSIM_report_only": _metric_delta(row.get("test_delta_report_only"), "SSIM"),
                    "test_dLPIPS_report_only": _metric_delta(row.get("test_delta_report_only"), "LPIPS"),
                    "strict_test_rgb_win_report_only": _strict_rgb_win(row.get("test_delta_report_only")),
                    "decision_kind": row.get("decision_kind", ""),
                    "reasons": ";".join(row.get("reasons", [])),
                    "decision_path": row.get("decision_path", ""),
                }
            )


def _write_md(path: Path, rows: list[dict[str, Any]], candidates: list[tuple[str, str]]) -> None:
    test_deltas = [row.get("test_delta_report_only", {}) for row in rows]
    means = {metric: mean(_metric_delta(delta, metric) for delta in test_deltas) for metric in METRICS}
    strict_wins = sum(1 for delta in test_deltas if _strict_rgb_win(delta))
    accepted = sum(1 for row in rows if row.get("accepted"))
    lines = [
        "# Phase-R Fixed Candidate Ladder Summary",
        "",
        "Selection rule: candidates are evaluated in the fixed order below, and the first candidate accepted by its train-val gate is selected. If a candidate root contains `SCENE/multifold_trainval_gate.json`, that stricter multi-offset decision is used; otherwise the legacy `decisions/SCENE_decision.json` single-split gate is used. Held-out test deltas are report-only and are not used for selection.",
        "",
        "Candidate order:",
    ]
    for i, (name, template) in enumerate(candidates, 1):
        lines.append(f"{i}. `{name}`: `{template}`")
    lines += [
        "",
        f"- scenes: `{len(rows)}`",
        f"- train-val accepted selections: `{accepted} / {len(rows)}`",
        f"- report-only strict RGB wins: `{strict_wins} / {len(rows)}`",
        f"- mean report-only delta: PSNR `{means['PSNR']:+.6f}`, SSIM `{means['SSIM']:+.6f}`, LPIPS `{means['LPIPS']:+.6f}`",
        "",
        "| scene | selected | gate | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | test dPSNR | test dSSIM | test dLPIPS | strict test win | notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        notes = "; ".join(row.get("reasons", []))
        lines.append(
            "| {scene} | {selected} | {gate} | {accepted} | {tr_psnr} | {tr_ssim} | {tr_lpips} | {te_psnr} | {te_ssim} | {te_lpips} | {win} | {notes} |".format(
                scene=row["scene"],
                selected=row["selected"],
                gate=row.get("decision_kind", ""),
                accepted=str(row.get("accepted")).lower(),
                tr_psnr=_fmt_delta(row.get("trainval_delta"), "PSNR"),
                tr_ssim=_fmt_delta(row.get("trainval_delta"), "SSIM"),
                tr_lpips=_fmt_delta(row.get("trainval_delta"), "LPIPS"),
                te_psnr=_fmt_delta(row.get("test_delta_report_only"), "PSNR"),
                te_ssim=_fmt_delta(row.get("test_delta_report_only"), "SSIM"),
                te_lpips=_fmt_delta(row.get("test_delta_report_only"), "LPIPS"),
                win=str(_strict_rgb_win(row.get("test_delta_report_only"))).lower(),
                notes=notes,
            )
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- Positive PSNR/SSIM and negative LPIPS are improvements.",
        "- The selected policy is still a small-delta representation-attached result. It improves cross-scene stability, but it is not yet a large-margin replacement for the Phase-J render-time endpoint.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    parser.add_argument("--override", action="append", type=_parse_override, default=[])
    parser.add_argument(
        "--force_fallback_scene",
        action="append",
        default=[],
        help="Scene forced to no-op by a predeclared policy rule, not by held-out test metrics.",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    overrides = dict(args.override)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    force_fallback_scenes = set(args.force_fallback_scene)
    rows = [
        _select_scene(scene, args.candidate, overrides, force_fallback_scenes)
        for scene in args.scenes
    ]

    summary = {
        "selection_uses_test": False,
        "scenes": args.scenes,
        "candidate_order": [{"name": name, "root_template": template} for name, template in args.candidate],
        "overrides": {f"{scene}:{name}": path for (scene, name), path in overrides.items()},
        "force_fallback_scenes": sorted(force_fallback_scenes),
        "rows": rows,
    }
    (out_dir / "phase_r_fixed_candidate_ladder.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "phase_r_fixed_candidate_ladder.csv", rows)
    _write_md(out_dir / "phase_r_fixed_candidate_ladder.md", rows, args.candidate)
    print(json.dumps({"rows": len(rows), "out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect the current Phase-S vertex-delta closed-loop evidence.

This is intentionally a reporting utility. It does not render, train, select
hyperparameters, or inspect held-out images. It reads existing gate/search/audit
artifacts and writes a compact evidence table for paper-loop decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("PSNR", "SSIM", "LPIPS")


DEFAULT_GATE_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_s/multifold_trainval_gate")
DEFAULT_SEARCH_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_s")
DEFAULT_QUAL_MANIFEST = Path(
    "outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_v24_qualitative_20260513/qualitative_manifest.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate_root", type=Path, default=DEFAULT_GATE_ROOT)
    parser.add_argument("--search_root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--qualitative_manifest", type=Path, default=DEFAULT_QUAL_MANIFEST)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_s/vertexdelta_closedloop_audit_20260513/summary.json"),
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("docs/car_model/5-13-VertexDelta-ClosedLoop-Audit.md"),
    )
    parser.add_argument("--include_patterns", default="vertexdelta_v24,vertexdelta_v25,vertexdelta_v26,vertexdelta_v27")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt(value: Any, digits: int = 9) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f}"


def _summary_delta(gate: dict[str, Any]) -> dict[str, float]:
    summary = gate.get("trainval_delta_summary") or {}
    if summary:
        return {key: float((summary.get(key) or {}).get("mean", math.nan)) for key in METRICS}
    rows = gate.get("rows") or []
    if rows:
        return {
            key: float(sum(float((row.get("delta") or {}).get(key, math.nan)) for row in rows) / max(len(rows), 1))
            for key in METRICS
        }
    return {key: math.nan for key in METRICS}


def _audit_path_from_gate(gate: dict[str, Any], gate_path: Path) -> Path | None:
    audit = gate.get("candidate_operator_audit") or {}
    path = audit.get("path")
    if path:
        p = Path(path)
        if p.is_absolute():
            return p
        return ROOT / p
    return None


def gate_records(gate_root: Path, include_patterns: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((ROOT / gate_root).glob("*/**/multifold_trainval_gate.json")):
        text = str(path)
        if include_patterns and not any(pattern in text for pattern in include_patterns):
            continue
        gate = load_json(path)
        if not gate:
            continue
        delta = _summary_delta(gate)
        audit_path = _audit_path_from_gate(gate, path)
        audit = load_json(audit_path) if audit_path is not None else {}
        effect = audit.get("materialization_effect") or {}
        topology_before = audit.get("topology_before") or {}
        topology_after = audit.get("topology_after") or {}
        records.append(
            {
                "kind": "multifold_gate",
                "scene": gate.get("scene", path.parent.name),
                "label": gate.get("candidate_label", path.parent.parent.name),
                "accepted": bool(gate.get("accepted", False)),
                "selected": gate.get("selected_label", ""),
                "deltas": delta,
                "decision_reasons": gate.get("decision_reasons", []),
                "offsets": gate.get("offsets", []),
                "gate_path": rel(path),
                "audit_path": rel(audit_path) if audit_path is not None else "",
                "accepted_faces": int(audit.get("accepted_faces", 0) or 0),
                "materialize_mode": audit.get("materialize_mode", ""),
                "has_effect": bool(effect.get("has_effect", False)),
                "topology_changed": bool(effect.get("topology_changed", False)),
                "attribute_changed": bool(effect.get("attribute_changed", False)),
                "max_attribute_delta": float(effect.get("max_attribute_delta", 0.0) or 0.0),
                "triangles_before": topology_before.get("triangles", "n/a"),
                "triangles_after": topology_after.get("triangles", "n/a"),
                "vertices_before": topology_before.get("vertices", "n/a"),
                "vertices_after": topology_after.get("vertices", "n/a"),
            }
        )
    return records


def rendercalib_records(search_root: Path, include_patterns: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = ROOT / search_root
    for path in sorted(root.glob("rendercalib_vertexdelta_*/**/render_calibrated_search.json")):
        text = str(path)
        if include_patterns and not any(pattern in text for pattern in include_patterns):
            continue
        payload = load_json(path)
        events = payload.get("events") or []
        best_event = None
        if events:
            best_event = max(events, key=lambda row: float(row.get("objective", -math.inf)))
        records.append(
            {
                "kind": "rendercalib_search",
                "scene": payload.get("scene", path.parent.name),
                "label": payload.get("candidate_label", path.parent.parent.name),
                "accepted": bool(payload.get("accepted", False)),
                "accepted_faces": len(payload.get("accepted_indices", [])),
                "accepted_face_ids": payload.get("accepted_face_ids", []),
                "best_objective": float(payload.get("best_objective", 0.0) or 0.0),
                "events": len(events),
                "last_action": (events[-1].get("action") if events else "none"),
                "last_action_reasons": (events[-1].get("action_reasons") if events else []),
                "best_trial": (best_event or {}).get("trial_id", ""),
                "best_trial_strict": bool((best_event or {}).get("accepted_by_strict_gate", False)),
                "best_trial_objective": float((best_event or {}).get("objective", 0.0) or 0.0),
                "best_trial_delta": (best_event or {}).get("trainval_delta_mean", {}),
                "path": rel(path),
                "report_path": rel(path.with_suffix(".md")),
                "log_path": rel(path.with_suffix(".log")),
            }
        )
    return records


def build_markdown(payload: dict[str, Any]) -> str:
    gate_rows = []
    for record in payload["gates"]:
        delta = record["deltas"]
        gate_rows.append(
            [
                record["scene"],
                record["label"],
                str(record["accepted"]).lower(),
                record["accepted_faces"],
                record["materialize_mode"] or "n/a",
                str(record["topology_changed"]).lower(),
                str(record["attribute_changed"]).lower(),
                f"{record['max_attribute_delta']:.6g}",
                fmt(delta.get("PSNR")),
                fmt(delta.get("SSIM")),
                fmt(delta.get("LPIPS")),
                ", ".join(record["decision_reasons"]) or "pass",
            ]
        )
    search_rows = []
    for record in payload["rendercalib_searches"]:
        delta = record.get("best_trial_delta") or {}
        search_rows.append(
            [
                record["scene"],
                record["label"],
                str(record["accepted"]).lower(),
                record["accepted_faces"],
                record["events"],
                f"{record['best_objective']:+.9f}",
                record["best_trial"],
                str(record["best_trial_strict"]).lower(),
                f"{record['best_trial_objective']:+.9f}",
                fmt(delta.get("PSNR")),
                fmt(delta.get("SSIM")),
                fmt(delta.get("LPIPS")),
                ", ".join(record.get("last_action_reasons") or []) or "n/a",
            ]
        )
    lines = [
        "# Phase-S Vertex-Delta Closed-Loop Audit",
        "",
        "This collector summarizes existing Phase-S vertex-delta artifacts. It is",
        "read-only and uses only train-val gate/search outputs plus operator audits.",
        "",
        "## Summary",
        "",
        md_table(
            ["metric", "value"],
            [
                ["multifold gate rows", len(payload["gates"])],
                ["accepted multifold rows", sum(1 for r in payload["gates"] if r["accepted"])],
                ["render-calibrated searches", len(payload["rendercalib_searches"])],
                ["accepted render-calibrated searches", sum(1 for r in payload["rendercalib_searches"] if r["accepted"])],
                ["qualitative manifest", f"`{payload['qualitative_manifest']}`" if payload["qualitative_manifest"] else "missing"],
            ],
        ),
        "",
        "## Multi-Offset Gates",
        "",
        md_table(
            [
                "scene",
                "label",
                "accepted",
                "faces",
                "mode",
                "topology changed",
                "attr changed",
                "max attr delta",
                "dPSNR",
                "dSSIM",
                "dLPIPS",
                "reasons",
            ],
            gate_rows or [["n/a"] * 12],
        ),
        "",
        "## Render-Calibrated Searches",
        "",
        md_table(
            [
                "scene",
                "label",
                "accepted",
                "faces",
                "events",
                "best objective",
                "best trial",
                "strict",
                "trial objective",
                "dPSNR",
                "dSSIM",
                "dLPIPS",
                "last action reasons",
            ],
            search_rows or [["n/a"] * 13],
        ),
        "",
        "## Interpretation",
        "",
        "- v24-style vertex-delta gates show that topology-preserving feature edits can be made strict-gate safe.",
        "- The effect size is still too small for a paper-level visual claim when the qualitative manifest reports near-zero image deltas.",
        "- v25/v27-style stronger edits increase PSNR but fail LPIPS/offset stability, so they are useful negative ablations, not promoted methods.",
        "- The render-calibrated searches were stopped because completed strict passes stayed below the fixed objective thresholds and did not change this conclusion.",
        "",
        "## Artifact Index",
        "",
    ]
    artifact_rows = []
    for record in payload["gates"]:
        artifact_rows.append([record["scene"], record["label"], f"`{record['gate_path']}`", f"`{record['audit_path']}`"])
    for record in payload["rendercalib_searches"]:
        artifact_rows.append([record["scene"], record["label"], f"`{record['path']}`", f"`{record['log_path']}`"])
    lines.append(md_table(["scene", "label", "primary artifact", "audit/log"], artifact_rows or [["n/a"] * 4]))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    include_patterns = [token.strip() for token in args.include_patterns.split(",") if token.strip()]
    payload = {
        "protocol": {
            "gate_root": rel(args.gate_root),
            "search_root": rel(args.search_root),
            "include_patterns": include_patterns,
            "selection_uses_test": False,
        },
        "qualitative_manifest": rel(args.qualitative_manifest) if (ROOT / args.qualitative_manifest).is_file() else "",
        "gates": gate_records(args.gate_root, include_patterns),
        "rendercalib_searches": rendercalib_records(args.search_root, include_patterns),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": rel(args.output_json), "output_md": rel(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

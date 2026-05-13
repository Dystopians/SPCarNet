#!/usr/bin/env python3
"""Collect Phase-K barycentric gate decision JSON files into one report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_PHASEJ_TEST_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _effective_delta(decision: dict[str, Any]) -> dict[str, float]:
    if bool(decision.get("accepted")):
        source = decision.get("test_delta_report_only", {})
        return {key: _num(source.get(key)) for key in METRICS}
    return {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}


def _operator_audit(decision: dict[str, Any]) -> dict[str, Any]:
    audit = decision.get("candidate_operator_audit") or {}
    available = bool(audit.get("available", False))
    accepted = audit.get("accepted")
    no_op_copy = audit.get("no_op_copy")
    rejected_or_noop = available and (accepted is False or no_op_copy is True)
    missing = not available
    return {
        "available": available,
        "accepted": accepted,
        "no_op_copy": no_op_copy,
        "policy_pass": audit.get("policy_pass"),
        "path": audit.get("path", ""),
        "missing": missing,
        "rejected_or_noop": bool(rejected_or_noop),
        "issue": bool(missing or rejected_or_noop),
    }


def _test_replay_audit(decision: dict[str, Any]) -> dict[str, Any]:
    base_method = str(decision.get("base_test_method_report_only") or "")
    candidate_method = str(decision.get("candidate_test_method_report_only") or "")
    test_delta = decision.get("test_delta_report_only") or {}
    has_delta = all(math.isfinite(_num(test_delta.get(key))) for key in METRICS)
    default_phasej_reference = base_method == DEFAULT_PHASEJ_TEST_METHOD
    fresh_phasej_replay = bool(base_method and not default_phasej_reference)
    return {
        "base_test_method": base_method,
        "candidate_test_method": candidate_method,
        "has_report_only_test_delta": bool(has_delta),
        "default_phasej_reference": bool(default_phasej_reference),
        "fresh_phasej_replay": bool(fresh_phasej_replay),
        "candidate_matches_base_method": bool(base_method and candidate_method and base_method == candidate_method),
    }


def _mean(rows: list[dict[str, Any]], key: str, block: str) -> float:
    values = [_num(row[block].get(key)) for row in rows]
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# Phase-K Barycentric Representation Gate Summary",
        "",
        "Selection uses train-policy-val decisions only. Held-out test deltas are report-only; effective deltas set rejected scenes to zero because they fall back to Phase-J.",
        "",
        f"- scenes: `{len(rows)}`",
        f"- accepted scenes: `{payload['accepted_count']}`",
        f"- mean effective dPSNR: `{payload['mean_effective_delta']['PSNR']:.6f}`",
        f"- mean effective dSSIM: `{payload['mean_effective_delta']['SSIM']:.6f}`",
        f"- mean effective dLPIPS: `{payload['mean_effective_delta']['LPIPS']:.6f}`",
        f"- operator-audit missing scenes: `{payload['operator_audit_missing_count']}`",
        f"- operator rejected/no-op scenes: `{payload['operator_rejected_or_noop_count']}`",
        f"- accepted scenes with operator-audit issue: `{payload['accepted_with_operator_audit_issue_count']}`",
        f"- report-only test deltas available: `{payload['report_only_test_delta_count']}`",
        f"- fresh Phase-J replay references: `{payload['fresh_phasej_replay_count']}`",
        f"- default Phase-J reference uses: `{payload['default_phasej_reference_count']}`",
        "",
        "| scene | selected | accepted | operator audit | Phase-J test ref | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decision = row["decision"]
        train = decision.get("trainval_delta", {})
        test = decision.get("test_delta_report_only", {})
        effective = row["effective_test_delta"]
        operator = row["operator_audit"]
        replay = row["test_replay_audit"]
        if operator["missing"]:
            operator_status = "missing"
        elif operator["rejected_or_noop"]:
            operator_status = "rejected/no-op"
        else:
            operator_status = "ok"
        replay_status = "fresh" if replay["fresh_phasej_replay"] else ("default" if replay["default_phasej_reference"] else "missing")
        lines.append(
            f"| {row['scene']} | {decision.get('selected_label')} | {str(decision.get('accepted')).lower()} | "
            f"{operator_status} | {replay_status} | "
            f"{_num(train.get('PSNR')):+.6f} | {_num(train.get('SSIM')):+.6f} | {_num(train.get('LPIPS')):+.6f} | "
            f"{_num(test.get('PSNR')):+.6f} | {_num(test.get('SSIM')):+.6f} | {_num(test.get('LPIPS')):+.6f} | "
            f"{effective['PSNR']:+.6f} | {effective['SSIM']:+.6f} | {effective['LPIPS']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision_root", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/decisions"))
    parser.add_argument(
        "--decision_path_template",
        default="",
        help=(
            "Optional decision JSON path template. Use {scene} for the scene "
            "name when decisions live under per-scene output roots."
        ),
    )
    parser.add_argument("--scenes", default="bicycle,flowers,garden,stump,treehill")
    parser.add_argument("--output_json", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/phasek_barycentric_gate_summary_outdoor5.json"))
    parser.add_argument("--output_md", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/phasek_barycentric_gate_summary_outdoor5.md"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for scene in [item.strip() for item in args.scenes.replace(" ", ",").split(",") if item.strip()]:
        if str(args.decision_path_template).strip():
            decision_path = Path(str(args.decision_path_template).format(scene=scene))
        elif "{scene}" in str(args.decision_root):
            decision_path = Path(str(args.decision_root).format(scene=scene)) / f"{scene}_decision.json"
        else:
            decision_path = args.decision_root / f"{scene}_decision.json"
        decision = _read_json(decision_path)
        if not decision:
            raise FileNotFoundError(decision_path)
        rows.append(
            {
                "scene": scene,
                "decision_path": str(decision_path),
                "decision": decision,
                "effective_test_delta": _effective_delta(decision),
                "operator_audit": _operator_audit(decision),
                "test_replay_audit": _test_replay_audit(decision),
            }
        )

    payload = {
        "rows": rows,
        "accepted_count": sum(1 for row in rows if bool(row["decision"].get("accepted"))),
        "mean_effective_delta": {
            key: _mean(rows, key, "effective_test_delta")
            for key in METRICS
        },
        "operator_audit_missing_count": sum(1 for row in rows if row["operator_audit"]["missing"]),
        "operator_rejected_or_noop_count": sum(1 for row in rows if row["operator_audit"]["rejected_or_noop"]),
        "accepted_with_operator_audit_issue_count": sum(
            1 for row in rows if bool(row["decision"].get("accepted")) and row["operator_audit"]["issue"]
        ),
        "report_only_test_delta_count": sum(1 for row in rows if row["test_replay_audit"]["has_report_only_test_delta"]),
        "fresh_phasej_replay_count": sum(1 for row in rows if row["test_replay_audit"]["fresh_phasej_replay"]),
        "default_phasej_reference_count": sum(1 for row in rows if row["test_replay_audit"]["default_phasej_reference"]),
        "accepted_with_default_phasej_reference_count": sum(
            1 for row in rows if bool(row["decision"].get("accepted")) and row["test_replay_audit"]["default_phasej_reference"]
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(args.output_md, payload)
    print(json.dumps({"rows": len(rows), "accepted": payload["accepted_count"], "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

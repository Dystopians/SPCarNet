#!/usr/bin/env python3
"""Collect Phase-K barycentric gate decision JSON files into one report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


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
        return {key: _num(source.get(key)) for key in ("PSNR", "SSIM", "LPIPS")}
    return {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}


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
        "",
        "| scene | selected | accepted | train-val dPSNR | train-val dSSIM | train-val dLPIPS | report test dPSNR | report test dSSIM | report test dLPIPS | effective dPSNR | effective dSSIM | effective dLPIPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decision = row["decision"]
        train = decision.get("trainval_delta", {})
        test = decision.get("test_delta_report_only", {})
        effective = row["effective_test_delta"]
        lines.append(
            f"| {row['scene']} | {decision.get('selected_label')} | {str(decision.get('accepted')).lower()} | "
            f"{_num(train.get('PSNR')):+.6f} | {_num(train.get('SSIM')):+.6f} | {_num(train.get('LPIPS')):+.6f} | "
            f"{_num(test.get('PSNR')):+.6f} | {_num(test.get('SSIM')):+.6f} | {_num(test.get('LPIPS')):+.6f} | "
            f"{effective['PSNR']:+.6f} | {effective['SSIM']:+.6f} | {effective['LPIPS']:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision_root", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/decisions"))
    parser.add_argument("--scenes", default="bicycle,flowers,garden,stump,treehill")
    parser.add_argument("--output_json", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/phasek_barycentric_gate_summary_outdoor5.json"))
    parser.add_argument("--output_md", type=Path, default=Path("outputs/carnet/meshsplatopt/ecsr_phase_k/bary_delta_v2wide_s08_guarded/phasek_barycentric_gate_summary_outdoor5.md"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for scene in [item.strip() for item in args.scenes.replace(" ", ",").split(",") if item.strip()]:
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
            }
        )

    payload = {
        "rows": rows,
        "accepted_count": sum(1 for row in rows if bool(row["decision"].get("accepted"))),
        "mean_effective_delta": {
            key: _mean(rows, key, "effective_test_delta")
            for key in ("PSNR", "SSIM", "LPIPS")
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_md(args.output_md, payload)
    print(json.dumps({"rows": len(rows), "accepted": payload["accepted_count"], "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Write a train-val render-trust certificate for strict plan replay.

The certificate does not run evaluation. It converts an existing Phase-K
train-val decision JSON into a small artifact that can authorize a non-unit
facelocal plan scale during later strict materialization. Held-out test metrics
remain report-only; a certificate is accepted only when the source decision is
accepted and `selection_uses_test` is false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision_json", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--min_balanced_delta", type=float, default=0.0)
    args = parser.parse_args()

    decision = read_json(args.decision_json)
    trainval_balanced_delta = float(decision.get("trainval_balanced_delta", float("-inf")))
    selection_uses_test = bool(decision.get("selection_uses_test", True))
    accepted = (
        bool(decision.get("accepted", False))
        and not selection_uses_test
        and trainval_balanced_delta >= float(args.min_balanced_delta)
    )
    cert = {
        "certificate_type": "phase_s_render_trust_scale_v1",
        "accepted": bool(accepted),
        "selection_uses_test": selection_uses_test,
        "accepted_scale": float(args.scale),
        "min_balanced_delta": float(args.min_balanced_delta),
        "trainval_balanced_delta": trainval_balanced_delta,
        "trainval_delta": decision.get("trainval_delta", {}),
        "test_balanced_delta_report_only": decision.get("test_balanced_delta_report_only"),
        "test_delta_report_only": decision.get("test_delta", {}),
        "decision_accepted": bool(decision.get("accepted", False)),
        "decision_reasons": decision.get("decision_reasons", decision.get("rejection_reasons", [])),
        "selected_label": decision.get("selected_label", ""),
        "candidate_label": decision.get("candidate_label", ""),
        "decision_json": str(args.decision_json),
        "plan": str(args.plan),
        "plan_sha256": sha256(args.plan),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Phase-S Render-Trust Certificate",
            "",
            f"- accepted: `{cert['accepted']}`",
            f"- selection uses test: `{cert['selection_uses_test']}`",
            f"- scale: `{cert['accepted_scale']}`",
            f"- train-val balanced delta: `{cert['trainval_balanced_delta']}`",
            f"- min balanced delta: `{cert['min_balanced_delta']}`",
            f"- selected label: `{cert['selected_label']}`",
            f"- candidate label: `{cert['candidate_label']}`",
            f"- decision json: `{cert['decision_json']}`",
            f"- plan sha256: `{cert['plan_sha256']}`",
        ]
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": accepted, "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect renderer-smoke results for materialized ECSR Phase-C candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES = (
    "bicycle_C0001",
    "bicycle_C0074",
    "kitchen_C0019",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_c/materialized_static_smoke"),
    )
    parser.add_argument(
        "--phase_a_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseC-RendererSmoke.md"),
    )
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def base_scene(candidate: str) -> str:
    return candidate.split("_C", 1)[0]


def baseline_view_error(args: argparse.Namespace, scene: str, view_index: int) -> float | None:
    path = args.phase_a_root / scene / "surface_evidence_summary.json"
    if not path.exists():
        return None
    summary = load_json(path)
    for view in summary.get("view_summaries", []):
        if int(view.get("view_index", -1)) == int(view_index):
            return float(view["mean_l1_error"])
    return None


def main() -> int:
    args = parse_args()
    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    records = []
    for candidate in candidates:
        path = args.smoke_root / candidate / "surface_evidence_summary.json"
        if not path.exists():
            records.append({"candidate": candidate, "status": "missing", "path": str(path)})
            continue
        summary = load_json(path)
        view = summary["view_summaries"][0]
        scene = base_scene(candidate)
        baseline_error = baseline_view_error(args, scene, int(view["view_index"]))
        candidate_error = float(view["mean_l1_error"])
        records.append(
            {
                "candidate": candidate,
                "scene": scene,
                "status": "PASS_RENDER_SMOKE",
                "model_path": summary["model_path"],
                "view_index": int(view["view_index"]),
                "image_name": view["image_name"],
                "valid_face_id_fraction": float(summary["mean_valid_face_id_fraction"]),
                "top_error_addressable_fraction": float(summary["mean_top_error_addressable_fraction"]),
                "candidate_mean_l1_error": candidate_error,
                "baseline_mean_l1_error": baseline_error,
                "delta_mean_l1_error": None if baseline_error is None else candidate_error - baseline_error,
                "contact_sheet": summary["artifacts"]["contact_sheet"],
            }
        )
    rows = []
    for record in records:
        rows.append(
            [
                record["candidate"],
                record.get("status", "missing"),
                record.get("view_index", "n/a"),
                f"{100.0 * record.get('valid_face_id_fraction', 0.0):.3f}%" if "valid_face_id_fraction" in record else "n/a",
                f"{100.0 * record.get('top_error_addressable_fraction', 0.0):.3f}%" if "top_error_addressable_fraction" in record else "n/a",
                f"{record.get('candidate_mean_l1_error', 0.0):.6f}" if "candidate_mean_l1_error" in record else "n/a",
                f"{record.get('delta_mean_l1_error', 0.0):+.6f}" if record.get("delta_mean_l1_error") is not None else "n/a",
            ]
        )
    passed = sum(1 for record in records if record.get("status") == "PASS_RENDER_SMOKE")
    payload = {
        "smoke_root": str(args.smoke_root),
        "phase_a_root": str(args.phase_a_root),
        "candidates": records,
        "passed": passed,
        "total": len(records),
    }
    (args.smoke_root / "phase_c_renderer_smoke_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# ECSR Phase-C Renderer Smoke",
        "",
        "This smoke test loads materialized PASS_STATIC checkpoint copies and",
        "renders one train view per candidate. It verifies renderer loadability",
        "and surface evidence generation before any longer policy-val experiment.",
        "Held-out test views are not used.",
        "",
        md_table(
            [
                "candidate",
                "status",
                "train view",
                "valid face-id",
                "top-error addressable",
                "mean L1",
                "dMean L1 vs compact",
            ],
            rows,
        ),
        "",
        f"Passed: `{passed} / {len(records)}`",
        "",
        "A pass here does not accept the candidate. It only permits local",
        "before/after rendering certificates and policy-val checks.",
        "",
    ]
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.smoke_root / 'phase_c_renderer_smoke_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

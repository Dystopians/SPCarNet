#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decide_action(
    sor_report: dict[str, Any],
    *,
    high_occluder_threshold: float,
    min_occluder_faces: int,
    large_mesh_face_threshold: int,
) -> dict[str, Any]:
    summary = sor_report.get("summary", {})
    stats = sor_report.get("sparse_occluder_stats", {})
    config = sor_report.get("config", {})
    face_count = int(summary.get("face_count", stats.get("face_count", 0)) or 0)
    valid_points = int(stats.get("valid_sparse_points", 0) or 0)
    front_points = int(stats.get("front_occluder_points", 0) or 0)
    selected_occluder_faces = int(stats.get("selected_sparse_occluder_faces", summary.get("sparse_occluder_count", 0)) or 0)
    front_fraction = float(front_points / max(valid_points, 1))
    selected_fraction = float(summary.get("selected_fraction", 0.0) or 0.0)

    if front_fraction >= float(high_occluder_threshold) and selected_occluder_faces >= int(min_occluder_faces):
        action = "sparse_occluder_low_evidence_union"
        branch = "SOR"
        reason = (
            "high_train_sparse_front_occluder_fraction:"
            f"{front_fraction:.6f}>={float(high_occluder_threshold):.6f}"
        )
    elif face_count >= int(large_mesh_face_threshold):
        action = "csef_adaptive_sparse_depth_recovery"
        branch = "CSEF"
        reason = (
            "large_mesh_without_high_front_occluder_signal:"
            f"faces={face_count},front_fraction={front_fraction:.6f}"
        )
    else:
        action = "qem50_sparse_parentrollback_ela_safe"
        branch = "QEM"
        reason = (
            "low_or_moderate_train_sparse_front_occluder_fraction:"
            f"{front_fraction:.6f}<{float(high_occluder_threshold):.6f}"
        )

    return {
        "policy": "SPCarNet-AdaptiveRepair-v1",
        "action": action,
        "branch": branch,
        "reason": reason,
        "inputs": {
            "face_count": face_count,
            "valid_sparse_points": valid_points,
            "front_occluder_points": front_points,
            "front_occluder_fraction": front_fraction,
            "selected_sparse_occluder_faces": selected_occluder_faces,
            "sor_selected_fraction": selected_fraction,
            "sor_no_test_leakage": bool(config.get("no_test_leakage", False)),
        },
        "thresholds": {
            "high_occluder_threshold": float(high_occluder_threshold),
            "min_occluder_faces": int(min_occluder_faces),
            "large_mesh_face_threshold": int(large_mesh_face_threshold),
        },
        "contract": {
            "SOR": "Apply train-split sparse occluder + low-evidence union candidates, then optional train-only ELA.",
            "QEM": "Apply QEM50 compact topology, topology-frozen sparse parent-rollback recovery, then train-only ELA.",
            "CSEF": "Apply CSEF adaptive compact recovery for large meshes without a high sparse-occluder signal.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a fixed adaptive MeshSplatOpt repair action from train-split sparse-occluder evidence.")
    parser.add_argument("--sor_candidates_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--high_occluder_threshold", type=float, default=0.25)
    parser.add_argument("--min_occluder_faces", type=int, default=256)
    parser.add_argument("--large_mesh_face_threshold", type=int, default=1_000_000)
    args = parser.parse_args()

    report = _read_json(Path(args.sor_candidates_json))
    decision = decide_action(
        report,
        high_occluder_threshold=float(args.high_occluder_threshold),
        min_occluder_faces=int(args.min_occluder_faces),
        large_mesh_face_threshold=int(args.large_mesh_face_threshold),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

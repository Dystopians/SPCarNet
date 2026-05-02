"""Run a no-op/protect readiness gate for extracted parking mesh patches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.scene_gate import (  # noqa: E402
    evaluate_proposal_geometry_delta,
    evaluate_proposal_topology_delta,
    save_rollback_snapshot,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(payload: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    return json.loads(str(payload["metadata_json"].item()))


def gate_patch(row: dict[str, Any], output_dir: Path, min_faces: int) -> dict[str, Any]:
    patch_path = Path(row["patch_path"])
    payload = np.load(patch_path, allow_pickle=False)
    vertices = np.asarray(payload["vertices"], dtype=np.float32)
    faces = np.asarray(payload["faces"], dtype=np.int64)
    metadata = _metadata(payload)
    region_id = str(row["region_id"])
    metrics: dict[str, float] = {}
    metrics.update(evaluate_proposal_geometry_delta(vertices, vertices.copy()))
    metrics.update(evaluate_proposal_topology_delta(faces, faces.copy()))
    reasons: list[str] = []
    decision = "protect_ready"
    if len(faces) < min_faces:
        decision = "deferred"
        reasons.append("patch_too_small")
    if metrics["mean_matched_vertex_displacement"] != 0.0 or metrics["triangle_count_delta"] != 0.0:
        decision = "failed"
        reasons.append("noop_changed_geometry")
    if decision == "protect_ready":
        reasons.append("noop_patch_gate_stable")
        reasons.append("rollback_snapshot_written")
    snapshot = save_rollback_snapshot(
        output_dir / "rollback_snapshots" / f"{region_id}_rollback_snapshot.npz",
        vertices,
        faces,
        metadata={"region_id": region_id, "source_patch": str(patch_path), **metadata},
    )
    return {
        "region_id": region_id,
        "proposal_types": row.get("proposal_types", ""),
        "decision": decision,
        "face_count": int(len(faces)),
        "vertex_count": int(len(vertices)),
        "rollback_snapshot": str(snapshot),
        "geometry_edited": False,
        "metrics": metrics,
        "reasons": reasons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _load_json(Path(args.mesh_patch_summary))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [gate_patch(row, out, args.min_faces) for row in summary.get("patches", [])]
    protect_ready = sum(1 for row in results if row["decision"] == "protect_ready")
    report = {
        "source_mesh_patch_summary": str(args.mesh_patch_summary),
        "patch_count": len(results),
        "protect_ready_count": protect_ready,
        "deferred_count": sum(1 for row in results if row["decision"] == "deferred"),
        "failed_count": sum(1 for row in results if row["decision"] == "failed"),
        "geometry_edited": False,
        "results": results,
        "notes": [
            "This is a no-op/protect readiness gate for local patches.",
            "It verifies patch topology can be snapshotted and passed through gate plumbing before any deformation is attempted.",
        ],
    }
    (out / "patch_gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "patch_gate_results.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "region_id",
            "proposal_types",
            "decision",
            "face_count",
            "vertex_count",
            "rollback_snapshot",
            "geometry_edited",
            "reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: (";".join(row[k]) if k == "reasons" else row[k]) for k in fieldnames})
    with (out / "patch_gate_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Patch Gate Report\n\n")
        f.write("- geometry edited: `false`\n")
        f.write(f"- patches evaluated: `{len(results)}`\n")
        f.write(f"- protect ready: `{protect_ready}`\n")
        f.write(f"- deferred: `{report['deferred_count']}`\n")
        f.write(f"- failed: `{report['failed_count']}`\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-op/protect readiness gates over parking mesh patches.")
    parser.add_argument("--mesh_patch_summary", default="outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/patch_gate")
    parser.add_argument("--min_faces", type=int, default=50)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "patch_count": report["patch_count"],
                "protect_ready_count": report["protect_ready_count"],
                "deferred_count": report["deferred_count"],
                "failed_count": report["failed_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

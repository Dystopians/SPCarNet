"""Run copied-patch before/after proposal tests for parking mesh patches."""

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
    accept_or_reject,
    evaluate_proposal_free_space_delta,
    evaluate_proposal_geometry_delta,
    evaluate_proposal_topology_delta,
    save_rollback_snapshot,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _save_npz(path: Path, vertices: np.ndarray, faces: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _add_floater(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return vertices.copy(), faces.copy()
    span = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-3)
    base = vertices.max(axis=0) + span + np.asarray([0.25, 0.25, 0.25], dtype=np.float32)
    scale = float(max(span.max() * 0.05, 1e-3))
    extra = np.asarray(
        [
            base,
            base + np.asarray([scale, 0.0, 0.0], dtype=np.float32),
            base + np.asarray([0.0, scale, 0.0], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    after_vertices = np.concatenate([vertices, extra], axis=0)
    after_faces = np.concatenate([faces, np.asarray([[len(vertices), len(vertices) + 1, len(vertices) + 2]], dtype=np.int64)], axis=0)
    return after_vertices, after_faces


def _remove_smallest_faces(faces: np.ndarray, areas: np.ndarray, fraction: float) -> np.ndarray:
    if len(faces) <= 1:
        return faces.copy()
    remove_count = max(1, int(round(len(faces) * fraction)))
    remove_count = min(remove_count, len(faces) - 1)
    keep = np.ones((len(faces),), dtype=bool)
    keep[np.argsort(areas)[:remove_count]] = False
    return faces[keep]


def _evaluate(
    *,
    proposal_id: str,
    proposal_type: str,
    before_vertices: np.ndarray,
    before_faces: np.ndarray,
    after_vertices: np.ndarray,
    after_faces: np.ndarray,
    object_evidence: dict[str, float],
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    metrics.update(evaluate_proposal_geometry_delta(before_vertices, after_vertices))
    metrics.update(evaluate_proposal_free_space_delta(before_vertices, after_vertices))
    metrics.update(evaluate_proposal_topology_delta(before_faces, after_faces))
    result = accept_or_reject(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        metrics=metrics,
        object_evidence=object_evidence,
    ).to_dict()
    result["metrics"] = metrics
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = _load_json(Path(args.mesh_patch_summary))
    rows = list(summary.get("patches", []))
    if args.max_patches > 0:
        rows = rows[: args.max_patches]
    out = Path(args.output_dir)
    proposal_dir = out / "proposal_meshes"
    rollback_dir = out / "rollback_snapshots"
    results: list[dict[str, Any]] = []

    for row in rows:
        region_id = str(row["region_id"])
        patch = np.load(row["patch_path"], allow_pickle=False)
        vertices = np.asarray(patch["vertices"], dtype=np.float32)
        faces = np.asarray(patch["faces"], dtype=np.int64)
        object_evidence = {"uncertainty": 0.0, "support_score": 1.0}
        save_rollback_snapshot(
            rollback_dir / f"{region_id}_rollback_snapshot.npz",
            vertices,
            faces,
            metadata={"region_id": region_id, "source_patch": row["patch_path"]},
        )

        variants = [
            ("protect_noop", vertices.copy(), faces.copy()),
            ("component_cleanup_candidate", vertices.copy(), _remove_smallest_faces(faces, _face_areas(vertices, faces), args.cleanup_fraction)),
        ]
        fv, ff = _add_floater(vertices, faces)
        variants.append(("floater_reject", fv, ff))

        for proposal_type, after_vertices, after_faces in variants:
            proposal_id = f"{region_id}_{proposal_type}"
            before_path = proposal_dir / region_id / f"{proposal_type}_before.npz"
            after_path = proposal_dir / region_id / f"{proposal_type}_after.npz"
            _save_npz(before_path, vertices, faces, {"region_id": region_id, "proposal_type": proposal_type, "role": "before"})
            _save_npz(after_path, after_vertices, after_faces, {"region_id": region_id, "proposal_type": proposal_type, "role": "after"})
            result = _evaluate(
                proposal_id=proposal_id,
                proposal_type=proposal_type,
                before_vertices=vertices,
                before_faces=faces,
                after_vertices=after_vertices,
                after_faces=after_faces,
                object_evidence=object_evidence,
            )
            expected = {
                "protect_noop": "reject_no_scene_improvement",
                "component_cleanup_candidate": "accept_topology_cleanup_on_copy",
                "floater_reject": "reject_new_component",
            }[proposal_type]
            results.append(
                {
                    "region_id": region_id,
                    "proposal_id": proposal_id,
                    "proposal_type": proposal_type,
                    "accepted": bool(result["accepted"]),
                    "expected_behavior": expected,
                    "before_npz": str(before_path),
                    "after_npz": str(after_path),
                    "rollback_snapshot": str(rollback_dir / f"{region_id}_rollback_snapshot.npz"),
                    "copied_patch_geometry_edited": proposal_type != "protect_noop",
                    "source_model_edited": False,
                    "reasons": list(result["reasons"]),
                    "metrics": result["metrics"],
                }
            )

    counts = {
        "accepted": sum(1 for r in results if r["accepted"]),
        "rejected": sum(1 for r in results if not r["accepted"]),
        "protect_noop_rejected": sum(1 for r in results if r["proposal_type"] == "protect_noop" and not r["accepted"]),
        "cleanup_accepted": sum(1 for r in results if r["proposal_type"] == "component_cleanup_candidate" and r["accepted"]),
        "floater_rejected": sum(1 for r in results if r["proposal_type"] == "floater_reject" and not r["accepted"]),
    }
    report = {
        "source_mesh_patch_summary": str(args.mesh_patch_summary),
        "patches_tested": len(rows),
        "proposal_tests": len(results),
        "counts": counts,
        "cleanup_fraction": float(args.cleanup_fraction),
        "copied_patch_geometry_edited": True,
        "source_model_edited": False,
        "results": results,
        "notes": [
            "All before/after edits are performed on copied patch NPZ files only.",
            "component_cleanup_candidate is a gate stress test on disconnected triangle-splat patches, not an approved full-scene edit.",
            "protect_noop rejection by M9 scene gate is expected because no scene metric improves.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "patch_proposal_test_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "patch_proposal_test_results.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "region_id",
            "proposal_id",
            "proposal_type",
            "accepted",
            "expected_behavior",
            "before_npz",
            "after_npz",
            "rollback_snapshot",
            "copied_patch_geometry_edited",
            "source_model_edited",
            "reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: (";".join(row[k]) if k == "reasons" else row[k]) for k in fieldnames})
    with (out / "patch_proposal_test_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Parking Patch Proposal Test Report\n\n")
        f.write("- source model edited: `false`\n")
        f.write("- copied patch geometry edited: `true`\n")
        f.write(f"- patches tested: `{len(rows)}`\n")
        f.write(f"- proposal tests: `{len(results)}`\n")
        for key, value in counts.items():
            f.write(f"- {key}: `{value}`\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run copied-patch before/after proposal tests for parking mesh patches.")
    parser.add_argument("--mesh_patch_summary", default="outputs/carnet/meshprior/parking_phone_tiny/mesh_patches/mesh_patch_summary.json")
    parser.add_argument("--output_dir", default="outputs/carnet/meshprior/parking_phone_tiny/patch_proposal_tests")
    parser.add_argument("--cleanup_fraction", type=float, default=0.05)
    parser.add_argument("--max_patches", type=int, default=0)
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"patches_tested": report["patches_tested"], "proposal_tests": report["proposal_tests"], **report["counts"]}, indent=2))


if __name__ == "__main__":
    main()

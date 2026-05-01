"""Dry-run MeshPrior orchestration pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.fill import build_fill_proposal, find_boundary_loops
from ss3dm_prior.meshprior.protect_prune import compute_triangle_scores
from ss3dm_prior.meshprior.scene_gate import (
    accept_or_reject,
    evaluate_proposal_free_space_delta,
    evaluate_proposal_geometry_delta,
    evaluate_proposal_topology_delta,
    save_rollback_snapshot,
)
from ss3dm_prior.meshprior.snap import propose_vertex_snap
from ss3dm_prior.meshprior.synthetic_damage import damage_mesh_local_hole, make_box_mesh
from scripts.car_model.meshprior_run_synthetic_damage_benchmark import analytic_box_field, analytic_box_occupancy_field


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int64))


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return np.asarray(payload["vertices"], dtype=np.float32), np.asarray(payload["faces"], dtype=np.int64)


def _synthetic_scene(output_dir: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    vertices, faces = make_box_mesh()
    damaged = damage_mesh_local_hole(vertices, faces, remove_count=2)
    mesh_path = output_dir / "synthetic_scene" / "damaged_mesh.npz"
    _save_mesh(mesh_path, damaged.vertices, damaged.faces)
    return mesh_path, damaged.vertices, damaged.faces


def _region_mining(args: argparse.Namespace, output_dir: Path, mesh_path: Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    if args.skip_region_mining:
        if not args.regions_json:
            raise ValueError("--skip_region_mining requires --regions_json")
        return Path(args.regions_json)
    regions = {
        "regions": [
            {
                "region_id": "synthetic_box_0000",
                "mesh_npz": str(mesh_path),
                "face_count": int(len(faces)),
                "vertex_count": int(len(vertices)),
                "source": "synthetic" if args.scene_source == "synthetic" else "dry_run_placeholder",
            }
        ][: args.max_regions],
    }
    path = output_dir / "regions.json"
    _write_json(path, regions)
    return path


def _posterior(args: argparse.Namespace, output_dir: Path, regions_json: Path) -> Path:
    if args.posterior_dir:
        return Path(args.posterior_dir)
    posterior_dir = output_dir / "posterior"
    summary = {
        "mode": args.mode,
        "posterior_checkpoint": args.posterior_checkpoint,
        "regions_json": str(regions_json),
        "posterior_source": "dry_run_analytic_box",
        "uncertainty": 0.0,
    }
    _write_json(posterior_dir / "posterior_summary.json", summary)
    return posterior_dir


def _proposal_record(
    *,
    proposal_id: str,
    proposal_type: str,
    before_path: Path,
    after_path: Path,
    object_evidence: dict,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "proposal_type": proposal_type,
        "before_npz": str(before_path),
        "after_npz": str(after_path),
        "object_evidence": object_evidence,
    }


def _make_proposals(args: argparse.Namespace, output_dir: Path, mesh_path: Path) -> Path:
    if args.proposals_json:
        return Path(args.proposals_json)
    vertices, faces = _load_mesh(mesh_path)
    proposal_dir = output_dir / "proposals"
    before_path = proposal_dir / "before.npz"
    _save_mesh(before_path, vertices, faces)
    records: list[dict] = []

    if any(t in args.proposal_types for t in ("protect", "prune")):
        table = compute_triangle_scores(vertices=vertices, faces=faces, decoder=analytic_box_field, z=None, samples_per_face=4)
        _write_json(
            proposal_dir / "protect_prune_scores.json",
            {
                "face_indices": table.face_indices,
                "protect_scores": table.protect_scores,
                "prune_scores": table.prune_scores,
            },
        )

    if "snap" in args.proposal_types:
        snap = propose_vertex_snap(vertices, faces, analytic_box_occupancy_field, max_disp=0.005, allow_boundary=False)
        snap_path = proposal_dir / "snap_after.npz"
        _save_mesh(snap_path, snap.vertices_after, faces)
        records.append(
            _proposal_record(
                proposal_id="synthetic_snap_0000",
                proposal_type="snap",
                before_path=before_path,
                after_path=snap_path,
                object_evidence={"confidence": 1.0, "uncertainty": 0.0},
            )
        )

    if "fill" in args.proposal_types:
        loops = find_boundary_loops((vertices, faces))
        if loops:
            fill = build_fill_proposal((vertices, faces), loops[0], analytic_box_field, min_support=0.45)
            fill_path = proposal_dir / "fill_after.npz"
            _save_mesh(fill_path, fill.vertices_after, fill.faces_after)
            records.append(
                _proposal_record(
                    proposal_id="synthetic_fill_0000",
                    proposal_type="fill",
                    before_path=before_path,
                    after_path=fill_path,
                    object_evidence={"confidence": fill.confidence, "uncertainty": 0.0},
                )
            )

    records = records[: args.max_proposals]
    proposals_path = proposal_dir / "proposals.json"
    _write_json(proposals_path, {"proposals": records})
    return proposals_path


def _evaluate_gate(args: argparse.Namespace, output_dir: Path, proposals_json: Path) -> Path:
    payload = json.loads(proposals_json.read_text(encoding="utf-8"))
    results = []
    gate_dir = output_dir / "scene_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    for row in payload.get("proposals", []):
        before = np.load(row["before_npz"])
        after = np.load(row["after_npz"])
        metrics = {}
        metrics.update(evaluate_proposal_geometry_delta(before["vertices"], after["vertices"]))
        metrics.update(evaluate_proposal_free_space_delta(before["vertices"], after["vertices"]))
        metrics.update(evaluate_proposal_topology_delta(before["faces"], after["faces"]))
        result = accept_or_reject(
            proposal_id=row["proposal_id"],
            proposal_type=row["proposal_type"],
            metrics=metrics,
            object_evidence=row.get("object_evidence"),
        )
        save_rollback_snapshot(
            gate_dir / f"{row['proposal_id']}_rollback_snapshot.npz",
            before["vertices"],
            before["faces"],
            {"proposal_id": row["proposal_id"], "proposal_type": row["proposal_type"]},
        )
        results.append(result.to_dict())
    accepted = [r for r in results if bool(r["accepted"])]
    report = {
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "no_geometry_write": bool(args.no_geometry_write),
        "proposal_count": len(results),
        "accepted_count": len(accepted),
        "rejected_count": len(results) - len(accepted),
        "results": results,
    }
    path = gate_dir / "gate_report.json"
    _write_json(path, report)
    _write_json(output_dir / "accepted_proposals.json", {"proposals": accepted})
    return path


def _report(output_dir: Path, gate_report_path: Path) -> None:
    gate = json.loads(gate_report_path.read_text(encoding="utf-8"))
    with (output_dir / "pipeline_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Pipeline Report\n\n")
        f.write(f"mode: `{gate['mode']}`\n\n")
        f.write(f"dry_run: `{gate['dry_run']}`\n\n")
        f.write(f"no_geometry_write: `{gate['no_geometry_write']}`\n\n")
        f.write(f"accepted: `{gate['accepted_count']}` / `{gate['proposal_count']}`\n")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.apply and args.no_geometry_write:
        raise ValueError("--apply conflicts with --no_geometry_write")
    if args.apply:
        raise ValueError("M10 does not implement geometry application; use dry-run artifacts only")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", vars(args))
    try:
        mesh_path, vertices, faces = _synthetic_scene(output_dir) if args.scene_source == "synthetic" else _synthetic_scene(output_dir)
        regions_json = _region_mining(args, output_dir, mesh_path, vertices, faces)
        posterior_dir = _posterior(args, output_dir, regions_json)
        proposals_json = Path(args.proposals_json) if args.eval_only and args.proposals_json else _make_proposals(args, output_dir, mesh_path)
        gate_report = _evaluate_gate(args, output_dir, proposals_json)
        gate = json.loads(gate_report.read_text(encoding="utf-8"))
        if args.require_gate_pass and gate["accepted_count"] == 0:
            raise RuntimeError("require_gate_pass enabled but no proposals were accepted")
        _report(output_dir, gate_report)
        status = {
            "status": "PASS",
            "regions_json": str(regions_json),
            "posterior_dir": str(posterior_dir),
            "proposals_json": str(proposals_json),
            "gate_report": str(gate_report),
            "accepted_count": int(gate["accepted_count"]),
            "rejected_count": int(gate["rejected_count"]),
        }
    except Exception as exc:
        status = {"status": "FAIL", "error": str(exc)}
        _write_json(output_dir / "pipeline_status.json", status)
        raise
    _write_json(output_dir / "pipeline_status.json", status)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MeshPrior dry-run pipeline.")
    parser.add_argument("--scene_source", required=True)
    parser.add_argument("--scene_model", required=True)
    parser.add_argument("--posterior_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--proposal_types", nargs="+", default=["protect", "prune"])
    parser.add_argument("--mode", default="dry_run")
    parser.add_argument("--skip_region_mining", action="store_true")
    parser.add_argument("--regions_json", default="")
    parser.add_argument("--posterior_dir", default="")
    parser.add_argument("--proposals_json", default="")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true", default=True)
    parser.add_argument("--no_geometry_write", action="store_true", default=True)
    parser.add_argument("--max_regions", type=int, default=1)
    parser.add_argument("--max_proposals", type=int, default=16)
    parser.add_argument("--require_gate_pass", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

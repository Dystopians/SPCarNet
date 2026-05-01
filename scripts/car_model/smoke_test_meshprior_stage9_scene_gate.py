"""Smoke test for MeshPrior Stage 9 scene gates and rollback."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.fill import build_fill_proposal, find_boundary_loops
from ss3dm_prior.meshprior.scene_gate import (
    accept_or_reject,
    evaluate_proposal_free_space_delta,
    evaluate_proposal_geometry_delta,
    evaluate_proposal_topology_delta,
    restore_rollback_snapshot,
    save_rollback_snapshot,
)
from ss3dm_prior.meshprior.synthetic_damage import add_floater_triangles, damage_mesh_local_hole, make_box_mesh


def _save_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    np.savez(path, vertices=vertices, faces=faces)


def main() -> None:
    vertices, faces = make_box_mesh()
    damaged = damage_mesh_local_hole(vertices, faces, remove_count=2)
    loop = find_boundary_loops((damaged.vertices, damaged.faces))[0]
    fill = build_fill_proposal((damaged.vertices, damaged.faces), loop)
    topo = evaluate_proposal_topology_delta(damaged.faces, fill.faces_after)
    metrics = {}
    metrics.update(evaluate_proposal_geometry_delta(damaged.vertices, fill.vertices_after))
    metrics.update(evaluate_proposal_free_space_delta(damaged.vertices, fill.vertices_after))
    metrics.update(topo)
    accepted = accept_or_reject(proposal_id="fill_good", proposal_type="fill", metrics=metrics)
    assert accepted.accepted, accepted

    floater = add_floater_triangles(vertices, faces)
    bad_topo = evaluate_proposal_topology_delta(faces, floater.faces)
    bad_metrics = {}
    bad_metrics.update(evaluate_proposal_geometry_delta(vertices, floater.vertices))
    bad_metrics.update(evaluate_proposal_free_space_delta(vertices, floater.vertices))
    bad_metrics.update(bad_topo)
    rejected = accept_or_reject(proposal_id="floater_bad", proposal_type="fill", metrics=bad_metrics)
    assert not rejected.accepted, rejected
    assert "component_count_increased" in rejected.reasons

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        snap = save_rollback_snapshot(tmp / "rollback.npz", damaged.vertices, damaged.faces, {"case": "smoke"})
        rv, rf, meta = restore_rollback_snapshot(snap)
        assert np.allclose(rv, damaged.vertices)
        assert np.array_equal(rf, damaged.faces)
        assert meta["case"] == "smoke"

        before = tmp / "before.npz"
        after_good = tmp / "after_good.npz"
        after_bad = tmp / "after_bad.npz"
        proposals = tmp / "proposals.json"
        out_dir = tmp / "gate"
        _save_mesh(before, damaged.vertices, damaged.faces)
        _save_mesh(after_good, fill.vertices_after, fill.faces_after)
        _save_mesh(after_bad, floater.vertices, floater.faces)
        proposals.write_text(
            json.dumps(
                {
                    "proposals": [
                        {"proposal_id": "fill_good", "proposal_type": "fill", "before_npz": str(before), "after_npz": str(after_good)},
                        {"proposal_id": "floater_bad", "proposal_type": "fill", "before_npz": str(before), "after_npz": str(after_bad)},
                    ]
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/car_model/meshprior_evaluate_proposals.py"),
                "--scene_source",
                "synthetic",
                "--scene_model",
                "synthetic",
                "--proposals",
                str(proposals),
                "--output_dir",
                str(out_dir),
                "--mode",
                "dry_run",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        report = json.loads((out_dir / "gate_report.json").read_text(encoding="utf-8"))
        assert report["accepted_count"] == 1, report
        assert report["rejected_count"] == 1, report

    print("[meshprior-stage9-smoke] PASS")
    print(json.dumps({"accepted": accepted.to_dict(), "rejected": rejected.to_dict()}, indent=2))


if __name__ == "__main__":
    main()

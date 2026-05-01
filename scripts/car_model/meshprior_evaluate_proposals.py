"""Evaluate MeshPrior proposals with dry-run scene gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshprior.scene_gate import (
    accept_or_reject,
    evaluate_proposal_free_space_delta,
    evaluate_proposal_geometry_delta,
    evaluate_proposal_topology_delta,
    save_rollback_snapshot,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _proposal_rows(payload: dict) -> list[dict]:
    if "proposals" in payload:
        return list(payload["proposals"])
    return [payload]


def evaluate_one(row: dict, output_dir: Path) -> dict[str, object]:
    before_npz = np.load(row["before_npz"])
    after_npz = np.load(row["after_npz"])
    vertices_before = before_npz["vertices"]
    faces_before = before_npz["faces"]
    vertices_after = after_npz["vertices"]
    faces_after = after_npz["faces"]
    proposal_id = str(row.get("proposal_id", Path(row["after_npz"]).stem))
    proposal_type = str(row.get("proposal_type", "unknown"))
    metrics = {}
    metrics.update(evaluate_proposal_geometry_delta(vertices_before, vertices_after))
    metrics.update(evaluate_proposal_free_space_delta(vertices_before, vertices_after))
    metrics.update(evaluate_proposal_topology_delta(faces_before, faces_after))
    result = accept_or_reject(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        metrics=metrics,
        object_evidence=row.get("object_evidence"),
    )
    save_rollback_snapshot(
        output_dir / f"{proposal_id}_rollback_snapshot.npz",
        vertices_before,
        faces_before,
        metadata={"proposal_id": proposal_id, "proposal_type": proposal_type},
    )
    return result.to_dict()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode != "dry_run":
        raise ValueError("M9 currently implements dry_run gates only")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals = _proposal_rows(_load_json(Path(args.proposals)))
    results = [evaluate_one(row, output_dir) for row in proposals]
    accepted = sum(1 for r in results if bool(r["accepted"]))
    report = {
        "mode": args.mode,
        "scene_source": args.scene_source,
        "scene_model": args.scene_model,
        "proposal_count": len(results),
        "accepted_count": accepted,
        "rejected_count": len(results) - accepted,
        "results": results,
    }
    (output_dir / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "gate_report.md").open("w", encoding="utf-8") as f:
        f.write("# MeshPrior Scene Gate Report\n\n")
        f.write(f"mode: `{args.mode}`\n\n")
        f.write(f"accepted: `{accepted}` / `{len(results)}`\n\n")
        for item in results:
            f.write(f"## {item['proposal_id']}\n\n")
            f.write(f"- type: `{item['proposal_type']}`\n")
            f.write(f"- accepted: `{item['accepted']}`\n")
            f.write(f"- reasons: `{', '.join(item['reasons'])}`\n\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MeshPrior proposals with scene gates.")
    parser.add_argument("--scene_source", required=True)
    parser.add_argument("--scene_model", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", default="dry_run")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()

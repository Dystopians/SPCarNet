#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_adapter import checkpoint_to_mesh_state, load_checkpoint_state
from ss3dm_prior.meshsplatopt.csef_builder import triangle_geometry
from ss3dm_prior.meshsplatopt.edit_types import MeshEdit, MeshSplatOptEditType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--min_area_ratio_to_median", type=float, default=1000.0)
    parser.add_argument("--min_percentile", type=float, default=99.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = checkpoint_to_mesh_state(load_checkpoint_state(args.checkpoint_path))
    _, _, areas = triangle_geometry(state.vertices, state.faces)
    median = float(np.median(areas)) if len(areas) else 0.0
    percentile_threshold = float(np.percentile(areas, args.min_percentile)) if len(areas) else 0.0
    ratio_threshold = median * float(args.min_area_ratio_to_median)
    threshold = max(percentile_threshold, ratio_threshold)
    candidate_ids = np.where(areas >= threshold)[0]
    ranked = sorted((int(i) for i in candidate_ids), key=lambda i: float(areas[i]), reverse=True)[: int(args.top_k)]
    status = "PASS" if ranked else "NO_CANDIDATE"
    edit = None
    if ranked:
        edit = MeshEdit(
            edit_id="real_checkpoint_area_outlier_delete_topk",
            edit_type=MeshSplatOptEditType.DELETE_TRIANGLES.value,
            defect_id="checkpoint_area_outlier",
            affected_faces=ranked,
            deleted_faces=ranked,
            evidence_summary={
                "selector": "checkpoint_area_outlier",
                "area_threshold": threshold,
                "median_area": median,
                "min_percentile": float(args.min_percentile),
                "selected_areas": [float(areas[i]) for i in ranked],
                "csef_source": "checkpoint_area_statistics_not_edge_topology",
            },
            risk_summary={
                "free_space_risk": 0.0,
                "deletes_supported_surface": False,
                "requires_render_backed_gate": True,
            },
            topology_cost_delta=-float(len(ranked)),
        )
        (out / "selected_edit.json").write_text(json.dumps(edit.to_dict(), indent=2), encoding="utf-8")
    report = {
        "status": status,
        "checkpoint_path": args.checkpoint_path,
        "triangles": int(len(state.faces)),
        "vertices": int(len(state.vertices)),
        "median_area": median,
        "max_area": float(np.max(areas)) if len(areas) else 0.0,
        "percentile_threshold": percentile_threshold,
        "ratio_threshold": ratio_threshold,
        "effective_threshold": threshold,
        "candidate_count": int(len(candidate_ids)),
        "selected_faces": ranked,
        "selected_areas": [float(areas[i]) for i in ranked],
        "edit_json": str(out / "selected_edit.json") if edit is not None else "",
        "note": "checkpoint-statistics selector; render-backed gate is mandatory before acceptance",
    }
    (out / "area_outlier_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# MeshSplatOpt Checkpoint Area-Outlier Edit Selection",
        "",
        f"- status: `{status}`",
        f"- checkpoint: `{args.checkpoint_path}`",
        f"- median area: `{median}`",
        f"- max area: `{report['max_area']}`",
        f"- effective threshold: `{threshold}`",
        f"- selected faces: `{ranked}`",
        f"- selected areas: `{report['selected_areas']}`",
        "",
        "Render-backed validation is mandatory before accepting this edit.",
    ]
    (out / "area_outlier_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit("no checkpoint area-outlier edit candidate selected")


if __name__ == "__main__":
    main()

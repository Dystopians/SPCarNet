from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .csef_types import CSEFBuildResult, CSEFRegion
from .defect_types import DefectRecord, DefectType


def _region_from_dict(data: dict[str, Any]) -> CSEFRegion:
    return CSEFRegion(
        region_id=str(data["region_id"]),
        defect_type_candidates=list(data.get("defect_type_candidates", [])),
        bbox=dict(data.get("bbox", {"min": [0, 0, 0], "max": [0, 0, 0]})),
        boundary_loop_ids=list(data.get("boundary_loop_ids", [])),
        mesh_face_indices=[int(x) for x in data.get("mesh_face_indices", [])],
        image_evidence_refs=list(data.get("image_evidence_refs", [])),
        sparse_point_refs=list(data.get("sparse_point_refs", [])),
        summary_stats=dict(data.get("summary_stats", {})),
    )


def load_csef_result(path: str | Path) -> CSEFBuildResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    regions = [_region_from_dict(r) for r in data.get("regions", [])]
    return CSEFBuildResult(
        scene_model=str(data.get("scene_model", "unknown")),
        scene_source=str(data.get("scene_source", "unknown")),
        mesh_path=str(data.get("mesh_path", "")),
        regions=regions,
        global_summary=dict(data.get("global_summary", {})),
    )


def mine_defects(
    csef: CSEFBuildResult,
    *,
    giant_area_threshold: float = 8.0,
    unknown_void_hints: list[dict[str, Any]] | None = None,
) -> list[DefectRecord]:
    defects: list[DefectRecord] = []
    for region in csef.regions:
        stats = region.summary_stats
        area = float(stats.get("area_sum", 0.0))
        mean_boundary = float(stats.get("mean_boundary_edge_score", 0.0))
        max_boundary = float(stats.get("max_boundary_edge_score", 0.0))
        size_ratio = float(stats.get("component_size_ratio", 1.0))
        candidates = set(region.defect_type_candidates)

        if "FLOATER_COMPONENT" in candidates or size_ratio < 0.25:
            defects.append(
                DefectRecord(
                    defect_id=f"defect_{len(defects):04d}",
                    defect_type=DefectType.FLOATER_COMPONENT.value,
                    severity=float(min(1.0, 0.5 + (0.25 - min(size_ratio, 0.25)) * 2.0)),
                    confidence=0.85,
                    affected_faces=region.mesh_face_indices,
                    boundary_loops=region.boundary_loop_ids,
                    candidate_edit_types_allowed=["DELETE_TRIANGLES", "EDGE_COLLAPSE", "PROTECT"],
                    evidence_summary={
                        "component_size_ratio": size_ratio,
                        "area_sum": area,
                        "csef_candidates": sorted(candidates),
                    },
                    uncertainty_summary={"low_positive_evidence_expected": True, "small_component": True},
                )
            )

        has_boundary_support = max_boundary > 0.25 or mean_boundary > 0.05
        if has_boundary_support and region.boundary_loop_ids:
            if area >= giant_area_threshold and max_boundary >= 0.5:
                defects.append(
                    DefectRecord(
                        defect_id=f"defect_{len(defects):04d}",
                        defect_type=DefectType.GIANT_GROUND_VOID.value,
                        severity=float(min(1.0, 0.35 + area / max(giant_area_threshold * 2.0, 1e-6))),
                        confidence=0.78,
                        affected_faces=region.mesh_face_indices,
                        boundary_loops=region.boundary_loop_ids,
                        candidate_edit_types_allowed=["FILL_PATCH", "SPLIT_TRIANGLES", "SNAP_VERTICES", "PROTECT"],
                        evidence_summary={
                            "boundary_loop_support": True,
                            "neighboring_surface_support": True,
                            "area_sum": area,
                            "mean_boundary_edge_score": mean_boundary,
                            "max_boundary_edge_score": max_boundary,
                            "ground_plane_prior_compatible": True,
                        },
                        uncertainty_summary={"prior_only_flag": False, "camera_coverage_score": 0.5},
                    )
                )
            else:
                defects.append(
                    DefectRecord(
                        defect_id=f"defect_{len(defects):04d}",
                        defect_type=DefectType.SMALL_BOUNDARY_HOLE.value,
                        severity=float(min(1.0, 0.25 + mean_boundary + area / max(giant_area_threshold, 1e-6) * 0.2)),
                        confidence=0.72,
                        affected_faces=region.mesh_face_indices,
                        boundary_loops=region.boundary_loop_ids,
                        candidate_edit_types_allowed=["FILL_PATCH", "SNAP_VERTICES", "PROTECT"],
                        evidence_summary={
                            "boundary_loop_support": True,
                            "area_sum": area,
                            "mean_boundary_edge_score": mean_boundary,
                            "max_boundary_edge_score": max_boundary,
                        },
                        uncertainty_summary={"prior_only_flag": False},
                    )
                )

    for hint in unknown_void_hints or []:
        boundary_support = float(hint.get("boundary_loop_support", 0.0))
        coverage = float(hint.get("camera_coverage_score", 0.0))
        prior_support = float(hint.get("prior_support", 0.0))
        is_unknown = boundary_support < 0.2 and coverage < 0.2
        defects.append(
            DefectRecord(
                defect_id=f"defect_{len(defects):04d}",
                defect_type=DefectType.UNKNOWN_UNOBSERVED_VOID.value if is_unknown else DefectType.GIANT_GROUND_VOID.value,
                severity=float(hint.get("severity", 0.8)),
                confidence=0.9 if is_unknown else 0.55,
                affected_faces=[],
                boundary_loops=[],
                candidate_edit_types_allowed=[] if is_unknown else ["FILL_PATCH"],
                evidence_summary={
                    "hint_id": hint.get("hint_id", "unknown_hint"),
                    "boundary_loop_support": boundary_support,
                    "camera_coverage_score": coverage,
                    "prior_support": prior_support,
                },
                uncertainty_summary={
                    "unknown_unobserved": is_unknown,
                    "prior_only_flag": bool(prior_support > 0.0 and boundary_support < 0.2),
                },
                no_repair_reason="insufficient boundary and camera coverage evidence" if is_unknown else "",
            )
        )

    return defects


def write_defect_outputs(defects: list[DefectRecord], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "defects.json").write_text(json.dumps([d.to_dict() for d in defects], indent=2), encoding="utf-8")
    with (out / "defects_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_id", "defect_type", "severity", "confidence", "face_count", "allowed_edits", "no_repair_reason"])
        for d in defects:
            writer.writerow(
                [
                    d.defect_id,
                    d.defect_type,
                    d.severity,
                    d.confidence,
                    len(d.affected_faces),
                    " ".join(d.candidate_edit_types_allowed),
                    d.no_repair_reason,
                ]
            )
    lines = ["# Defect Mining Report", "", f"- defects: `{len(defects)}`", "", "## Defects", ""]
    for d in defects:
        lines.append(
            f"- `{d.defect_id}` `{d.defect_type}` severity `{d.severity:.3f}` "
            f"confidence `{d.confidence:.3f}` edits `{', '.join(d.candidate_edit_types_allowed) or 'none'}`"
        )
        if d.no_repair_reason:
            lines.append(f"  - no repair: {d.no_repair_reason}")
    (out / "defect_mining_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

#!/usr/bin/env python3
"""Materialize PASS_STATIC ECSR contraction candidates as checkpoint copies."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path, validate_faces


VERTEX_KEYS = ("triangles_points", "vertex_weight", "features_dc", "features_rest")
FACE_KEYS = ("importance_score", "image_size", "pixel_count")


DEFAULT_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/paper_m360_repro/compact_ela_sor_adaptive_geo_26k"),
    )
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument(
        "--static_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_c/static_topology_certificate"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_c/materialized_static_pass"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseC-MaterializedStaticPass.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument(
        "--operator_filter",
        default="certificate_cluster_contraction_candidate",
        help="Only materialize PASS_STATIC certificates with this phase_b_operator_type. Use 'all' for all.",
    )
    parser.add_argument("--max_per_scene", type=int, default=1)
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


def compact_unused_vertices(payload: dict[str, Any], keep_face_mask: torch.Tensor) -> dict[str, Any]:
    faces = payload["_triangle_indices"].detach().cpu().long()[keep_face_mask]
    vertices = payload["triangles_points"].detach().cpu()
    used = torch.unique(faces.reshape(-1), sorted=True)
    remap = torch.full((int(vertices.shape[0]),), -1, dtype=torch.long)
    remap[used] = torch.arange(int(used.shape[0]), dtype=torch.long)
    new_faces = remap[faces].to(dtype=payload["_triangle_indices"].dtype)
    out: dict[str, Any] = {}
    vertex_count = int(vertices.shape[0])
    face_count = int(payload["_triangle_indices"].shape[0])
    for key, value in payload.items():
        if torch.is_tensor(value):
            cpu = value.detach().cpu()
            if key == "_triangle_indices":
                out[key] = new_faces.clone()
            elif key in VERTEX_KEYS and int(cpu.shape[0]) == vertex_count:
                out[key] = cpu[used].clone()
            elif key in FACE_KEYS and int(cpu.shape[0]) == face_count:
                out[key] = cpu[keep_face_mask].clone()
            else:
                out[key] = cpu.clone()
        else:
            out[key] = value
    return out


def apply_edge_collapse(payload: dict[str, Any], static_cert: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    faces = payload["_triangle_indices"].detach().cpu().long().clone()
    vertices = payload["triangles_points"].detach().cpu().clone()
    edge = static_cert["edge"]
    v_keep = int(edge["v_keep"])
    v_remove = int(edge["v_remove"])
    midpoint = 0.5 * (vertices[v_keep] + vertices[v_remove])
    vertices[v_keep] = midpoint
    faces[faces == v_remove] = v_keep
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 0] == faces[:, 2])
        | (faces[:, 1] == faces[:, 2])
    )
    edited = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in payload.items()}
    edited["triangles_points"] = vertices
    edited["_triangle_indices"] = faces.to(dtype=payload["_triangle_indices"].dtype)
    out = compact_unused_vertices(edited, ~repeated)
    degenerate, invalid = validate_faces(out["triangles_points"], out["_triangle_indices"])
    audit = {
        "v_keep": v_keep,
        "v_remove": v_remove,
        "edge_length": float(edge["edge_length"]),
        "pre_triangles": int(payload["_triangle_indices"].shape[0]),
        "post_triangles": int(out["_triangle_indices"].shape[0]),
        "pre_vertices": int(payload["triangles_points"].shape[0]),
        "post_vertices": int(out["triangles_points"].shape[0]),
        "removed_triangles": int(payload["_triangle_indices"].shape[0] - out["_triangle_indices"].shape[0]),
        "removed_vertices": int(payload["triangles_points"].shape[0] - out["triangles_points"].shape[0]),
        "degenerate_face_count": int(degenerate),
        "invalid_index_count": int(invalid),
    }
    return out, audit


def copy_metadata(source_model: Path, output_model: Path) -> None:
    output_model.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = source_model / name
        if src.is_file():
            shutil.copy2(src, output_model / name)


def materialize_scene(args: argparse.Namespace, scene: str) -> list[dict[str, Any]]:
    source_model = args.method_root / scene / args.policy_tag / "compact_model"
    payload = torch.load(checkpoint_path(source_model, int(args.iteration)), map_location="cpu")
    cert_dir = args.static_root / scene / "certificates"
    certs = []
    for path in sorted(cert_dir.glob("*.json")):
        cert = load_json(path)
        if cert.get("status") != "PASS_STATIC":
            continue
        if args.operator_filter != "all" and cert.get("phase_b_operator_type") != args.operator_filter:
            continue
        certs.append((path, cert))
    certs = certs[: int(args.max_per_scene)]
    reports = []
    for path, cert in certs:
        candidate_id = str(cert["candidate_id"])
        output_model = args.out_root / scene / candidate_id / "model"
        out_ckpt = output_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "point_cloud_state_dict.pt"
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        copy_metadata(source_model, output_model)
        out_payload, audit = apply_edge_collapse(payload, cert)
        torch.save(out_payload, out_ckpt)
        report = {
            "scene": scene,
            "candidate_id": candidate_id,
            "source_model": str(source_model),
            "output_model": str(output_model),
            "output_checkpoint": str(out_ckpt),
            "source_static_certificate": str(path),
            "test_usage": "none",
            "checkpoint_schema_valid": audit["degenerate_face_count"] == 0 and audit["invalid_index_count"] == 0,
            "topology_audit": audit,
            "status": "MATERIALIZED" if audit["degenerate_face_count"] == 0 and audit["invalid_index_count"] == 0 else "INVALID",
        }
        output_model.mkdir(parents=True, exist_ok=True)
        (output_model / "topology_audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (args.out_root / scene / candidate_id / "materialization_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(report)
    return reports


def build_doc(reports: list[dict[str, Any]]) -> str:
    rows = [
        [
            r["scene"],
            r["candidate_id"],
            r["status"],
            r["topology_audit"]["removed_triangles"],
            r["topology_audit"]["removed_vertices"],
            r["checkpoint_schema_valid"],
            f"`{r['output_model']}`",
        ]
        for r in reports
    ]
    return "\n".join(
        [
            "# ECSR Phase-C Materialized Static-Pass Candidates",
            "",
            "This report lists the PASS_STATIC contraction candidates materialized",
            "as checkpoint copies. These are still not final ECSR results; they are",
            "only ready for renderer smoke and policy-val before/after checks.",
            "",
            md_table(
                [
                    "scene",
                    "candidate",
                    "status",
                    "removed triangles",
                    "removed vertices",
                    "schema valid",
                    "model",
                ],
                rows,
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    reports: list[dict[str, Any]] = []
    for scene in scenes:
        reports.extend(materialize_scene(args, scene))
    summary = {"protocol": vars(args), "reports": reports}
    (args.out_root / "phase_c_materialized_static_pass_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md = build_doc(reports)
    (args.out_root / "phase_c_materialized_static_pass_summary.md").write_text(
        md + "\n", encoding="utf-8"
    )
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(md + "\n", encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] materialized {len(reports)} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

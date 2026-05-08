#!/usr/bin/env python3
"""Run Phase-C preflight certificates for ECSR candidate clusters.

This script is intentionally pre-contraction: it does not modify checkpoints.
It checks whether Phase-B candidate clusters have train-only fitting and
policy-val support masks, and writes candidate-level JSON certificates that can
later be extended with topology smoke tests and before/after render metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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
        "--phase_a_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence"),
    )
    parser.add_argument(
        "--phase_b_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_b/view_support_graph"),
    )
    parser.add_argument(
        "--split_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_policy_splits/phase_ab_cached_views"),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_c/candidate_preflight"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseC-CandidatePreflight.md"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--min_policy_views", type=int, default=1)
    parser.add_argument("--min_fitting_views", type=int, default=1)
    parser.add_argument("--min_policy_pixels", type=int, default=64)
    parser.add_argument("--max_candidates_per_scene", type=int, default=128)
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


def load_view(scene_dir: Path, train_view_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = scene_dir / "views" / f"{train_view_index:05d}.npz"
    with np.load(path) as data:
        face_id = data["face_id"].astype(np.int64, copy=False)
        residual_l1 = data["residual_l1"].astype(np.float32, copy=False)
        texture = data["texture"].astype(np.float32, copy=False)
    return face_id, residual_l1, texture


def support_stats(
    scene_dir: Path,
    views: list[dict[str, Any]],
    faces: list[int],
) -> dict[str, Any]:
    face_arr = np.asarray(faces, dtype=np.int64)
    hits = []
    total_pixels = 0
    weighted_error = 0.0
    weighted_texture = 0.0
    for view in views:
        face_id, residual_l1, texture = load_view(scene_dir, int(view["train_view_index"]))
        mask = np.isin(face_id, face_arr)
        pixels = int(np.count_nonzero(mask))
        if pixels > 0:
            hits.append(int(view["train_view_index"]))
            total_pixels += pixels
            weighted_error += float(np.sum(residual_l1[mask]))
            weighted_texture += float(np.sum(texture[mask]))
    return {
        "views_hit": hits,
        "num_views_hit": len(hits),
        "pixels": total_pixels,
        "mean_l1_error": weighted_error / max(total_pixels, 1),
        "mean_texture": weighted_texture / max(total_pixels, 1),
    }


def preflight_status(
    candidate: dict[str, Any],
    fitting: dict[str, Any],
    policy: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str]:
    if candidate.get("operator_type") == "diagnostic_only_high_risk_cluster":
        return "reject", "diagnostic-only high-risk cluster"
    if "weak_residual_direction" in candidate.get("risk_flags", []):
        return "reject", "weak residual direction"
    if fitting["num_views_hit"] < args.min_fitting_views:
        return "reject", "insufficient fitting-train support"
    if policy["num_views_hit"] < args.min_policy_views:
        return "reject", "insufficient policy-val support"
    if policy["pixels"] < args.min_policy_pixels:
        return "reject", "policy-val support mask too small"
    return "preflight_pass", "candidate has fitting and policy-val train support"


def process_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    scene_a_dir = args.phase_a_root / scene
    graph = load_json(args.phase_b_root / scene / "view_support_graph.json")
    split = load_json(args.split_root / scene / "policy_split.json")
    candidates = graph.get("candidate_clusters", [])[: args.max_candidates_per_scene]
    out_dir = args.out_root / scene
    cert_dir = out_dir / "certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)

    certificates = []
    for candidate in candidates:
        faces = [int(face) for face in candidate["faces"]]
        fitting = support_stats(scene_a_dir, split["fitting_train"], faces)
        policy = support_stats(scene_a_dir, split["policy_val"], faces)
        status, reason = preflight_status(candidate, fitting, policy, args)
        cert = {
            "scene": scene,
            "candidate_id": candidate["candidate_id"],
            "operator_type": candidate["operator_type"],
            "affected_faces": faces,
            "num_faces": candidate["num_faces"],
            "phase_b_mean_redundancy_score": candidate["mean_redundancy_score"],
            "risk_flags": candidate.get("risk_flags", []),
            "split_scope": split["split_scope"],
            "seed": split["seed"],
            "test_usage": "none",
            "fitting_train_support": fitting,
            "policy_val_support": policy,
            "static_topology_smoke": "not_evaluated_pre_contraction",
            "local_rendering_certificate": "not_evaluated_no_candidate_checkpoint",
            "policy_validation_certificate": "support_mask_preflight_only",
            "accepted": False,
            "preflight_status": status,
            "rejection_reason": "" if status == "preflight_pass" else reason,
            "continue_reason": reason if status == "preflight_pass" else "",
        }
        (cert_dir / f"{candidate['candidate_id']}.json").write_text(
            json.dumps(cert, indent=2) + "\n", encoding="utf-8"
        )
        certificates.append(cert)

    passed = [c for c in certificates if c["preflight_status"] == "preflight_pass"]
    by_operator: dict[str, int] = {}
    for cert in passed:
        by_operator[cert["operator_type"]] = by_operator.get(cert["operator_type"], 0) + 1
    summary = {
        "scene": scene,
        "candidates_checked": len(certificates),
        "preflight_pass": len(passed),
        "preflight_reject": len(certificates) - len(passed),
        "passes_by_operator": by_operator,
        "certificate_dir": str(cert_dir),
    }
    (out_dir / "candidate_preflight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_doc(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    rows = [
        [
            s["scene"],
            s["candidates_checked"],
            s["preflight_pass"],
            s["preflight_reject"],
            s["passes_by_operator"].get("certificate_cluster_contraction_candidate", 0),
            s["passes_by_operator"].get("surface_attached_attribute_recovery_candidate", 0),
            f"`{s['certificate_dir']}`",
        ]
        for s in summaries
    ]
    total_checked = sum(s["candidates_checked"] for s in summaries)
    total_pass = sum(s["preflight_pass"] for s in summaries)
    total_reject = sum(s["preflight_reject"] for s in summaries)
    total_contraction = sum(
        s["passes_by_operator"].get("certificate_cluster_contraction_candidate", 0)
        for s in summaries
    )
    total_attribute = sum(
        s["passes_by_operator"].get("surface_attached_attribute_recovery_candidate", 0)
        for s in summaries
    )
    return "\n".join(
        [
            "# ECSR Phase-C Candidate Preflight",
            "",
            "This is a pre-contraction certificate pass. It does not modify",
            "checkpoints and does not claim final acceptance. It only verifies",
            "that Phase-B candidates have train-only fitting and policy-val",
            "surface support masks before any topology or appearance recovery",
            "experiment is allowed to spend GPU time.",
            "",
            "## Fixed Rules",
            "",
            f"- min fitting-train support views: `{args.min_fitting_views}`",
            f"- min policy-val support views: `{args.min_policy_views}`",
            f"- min policy-val support pixels: `{args.min_policy_pixels}`",
            "- held-out test usage: `none`",
            "- checkpoint edits: `none`",
            "",
            "## Aggregate",
            "",
            md_table(
                ["metric", "value"],
                [
                    ["candidates checked", total_checked],
                    ["preflight pass", total_pass],
                    ["preflight reject", total_reject],
                    ["contraction preflight pass", total_contraction],
                    ["attribute-recovery preflight pass", total_attribute],
                ],
            ),
            "",
            "## Per-Scene",
            "",
            md_table(
                [
                    "scene",
                    "checked",
                    "pass",
                    "reject",
                    "contraction pass",
                    "attribute pass",
                    "certificate dir",
                ],
                rows,
            ),
            "",
            "## Interpretation",
            "",
            "Candidates that pass this preflight are not yet accepted ECSR edits.",
            "They are merely eligible for the next expensive step: static topology",
            "smoke testing, local rendering certificates, and policy-val before/after",
            "metrics. Rejected candidates should not be manually revived.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    summaries = [process_scene(args, scene) for scene in scenes]
    aggregate = {"protocol": vars(args), "summaries": summaries}
    (args.out_root / "phase_c_candidate_preflight_summary.json").write_text(
        json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8"
    )
    md = build_doc(args, summaries)
    (args.out_root / "phase_c_candidate_preflight_summary.md").write_text(
        md + "\n", encoding="utf-8"
    )
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(md + "\n", encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.out_root / 'phase_c_candidate_preflight_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

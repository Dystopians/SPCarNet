#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def evidence_views(evidence_dir: Path) -> list[Path]:
    views_dir = evidence_dir / "views"
    if views_dir.is_dir():
        return sorted(views_dir.glob("*.npz"))
    return sorted(evidence_dir.glob("*.npz"))


def load_carrier_faces(path: Path, max_carriers: int, max_faces_per_carrier: int, max_faces: int) -> tuple[set[int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    carriers = list(payload.get("carriers") or [])
    if max_carriers > 0:
        carriers = carriers[: int(max_carriers)]
    faces: list[int] = []
    for carrier in carriers:
        raw = carrier.get("face_ids")
        if raw is None:
            raw = [row.get("face_id") for row in carrier.get("faces", []) if "face_id" in row]
        raw_faces = [int(x) for x in raw if int(x) >= 0]
        if max_faces_per_carrier > 0:
            raw_faces = raw_faces[: int(max_faces_per_carrier)]
        faces.extend(raw_faces)
        if max_faces > 0 and len(set(faces)) >= int(max_faces):
            break
    unique = []
    seen = set()
    for face in faces:
        if face not in seen:
            unique.append(face)
            seen.add(face)
        if max_faces > 0 and len(unique) >= int(max_faces):
            break
    return set(unique), {
        "carrier_count": int(len(carriers)),
        "candidate_faces": int(len(unique)),
        "max_carriers": int(max_carriers),
        "max_faces_per_carrier": int(max_faces_per_carrier),
        "max_faces": int(max_faces),
    }


def audit_coverage(args: argparse.Namespace) -> dict[str, Any]:
    evidence_dir = Path(args.target_evidence_dir)
    carrier_json = Path(args.region_carrier_json)
    candidate_faces, carrier_summary = load_carrier_faces(
        carrier_json,
        max_carriers=int(args.max_carriers),
        max_faces_per_carrier=int(args.max_faces_per_carrier),
        max_faces=int(args.max_faces),
    )
    views = evidence_views(evidence_dir)
    if not views:
        raise FileNotFoundError(f"no target npz views found in {evidence_dir}")

    candidate_arr = np.fromiter(candidate_faces, dtype=np.int64) if candidate_faces else np.empty((0,), dtype=np.int64)
    totals = {
        "views": 0,
        "views_with_barycentric": 0,
        "views_missing_barycentric": 0,
        "total_pixels": 0,
        "valid_face_pixels": 0,
        "candidate_face_pixels": 0,
        "barycentric_valid_pixels": 0,
        "candidate_barycentric_valid_pixels": 0,
        "actionable_pixels": 0,
    }
    rows: list[dict[str, Any]] = []

    for path in tqdm(views, desc="audit target atlas coverage"):
        z = np.load(path)
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        total = int(face_id.size)
        valid_face = face_id >= 0
        candidate = np.isin(face_id, candidate_arr) if candidate_arr.size else valid_face
        has_bary = "barycentric" in z
        if has_bary:
            bary = np.asarray(z["barycentric"], dtype=np.float32)
            bary_valid = np.all(np.isfinite(bary), axis=0)
            bary_valid &= np.all(bary >= -float(args.barycentric_tolerance), axis=0)
            bary_valid &= np.all(bary <= 1.0 + float(args.barycentric_tolerance), axis=0)
            if "barycentric_valid" in z:
                bary_valid &= np.asarray(z["barycentric_valid"]).astype(bool)
        else:
            bary_valid = np.zeros_like(valid_face, dtype=bool)
        actionable = candidate & bary_valid
        if "alpha" in z:
            actionable &= np.asarray(z["alpha"], dtype=np.float32) >= float(args.min_alpha)
        if str(args.residual_l1_key) in z and float(args.min_l1) > 0.0:
            actionable &= np.asarray(z[str(args.residual_l1_key)], dtype=np.float32) >= float(args.min_l1)

        row = {
            "view": path.stem,
            "has_barycentric": bool(has_bary),
            "total_pixels": total,
            "valid_face_pixels": int(valid_face.sum()),
            "candidate_face_pixels": int(candidate.sum()),
            "barycentric_valid_pixels": int(bary_valid.sum()),
            "candidate_barycentric_valid_pixels": int((candidate & bary_valid).sum()),
            "actionable_pixels": int(actionable.sum()),
            "candidate_fraction": float(candidate.sum() / max(1, total)),
            "barycentric_valid_fraction": float(bary_valid.sum() / max(1, total)),
            "actionable_fraction": float(actionable.sum() / max(1, total)),
        }
        rows.append(row)

        totals["views"] += 1
        totals["views_with_barycentric"] += int(has_bary)
        totals["views_missing_barycentric"] += int(not has_bary)
        totals["total_pixels"] += total
        totals["valid_face_pixels"] += int(valid_face.sum())
        totals["candidate_face_pixels"] += int(candidate.sum())
        totals["barycentric_valid_pixels"] += int(bary_valid.sum())
        totals["candidate_barycentric_valid_pixels"] += int((candidate & bary_valid).sum())
        totals["actionable_pixels"] += int(actionable.sum())

    total_pixels = max(1, int(totals["total_pixels"]))
    candidate_pixels = max(1, int(totals["candidate_face_pixels"]))
    summary = {
        "target_evidence_dir": str(evidence_dir),
        "region_carrier_json": str(carrier_json),
        "carrier_summary": carrier_summary,
        "settings": vars(args),
        "totals": totals,
        "valid_face_fraction": float(totals["valid_face_pixels"] / total_pixels),
        "candidate_face_fraction": float(totals["candidate_face_pixels"] / total_pixels),
        "barycentric_valid_fraction": float(totals["barycentric_valid_pixels"] / total_pixels),
        "candidate_barycentric_valid_fraction": float(totals["candidate_barycentric_valid_pixels"] / total_pixels),
        "actionable_fraction": float(totals["actionable_pixels"] / total_pixels),
        "actionable_over_candidate_fraction": float(totals["actionable_pixels"] / candidate_pixels),
        "per_view": rows,
    }
    return summary


def write_md(path: Path, summary: dict[str, Any]) -> None:
    totals = summary["totals"]
    lines = [
        "# Surface Residual Atlas Target Coverage Audit",
        "",
        f"- target evidence: `{summary['target_evidence_dir']}`",
        f"- region carrier: `{summary['region_carrier_json']}`",
        f"- views: `{totals['views']}`",
        f"- views with barycentric: `{totals['views_with_barycentric']}`",
        f"- views missing barycentric: `{totals['views_missing_barycentric']}`",
        f"- candidate faces: `{summary['carrier_summary']['candidate_faces']}`",
        f"- valid face fraction: `{summary['valid_face_fraction']:.8f}`",
        f"- candidate face fraction: `{summary['candidate_face_fraction']:.8f}`",
        f"- barycentric valid fraction: `{summary['barycentric_valid_fraction']:.8f}`",
        f"- candidate barycentric valid fraction: `{summary['candidate_barycentric_valid_fraction']:.8f}`",
        f"- actionable fraction: `{summary['actionable_fraction']:.8f}`",
        f"- actionable / candidate fraction: `{summary['actionable_over_candidate_fraction']:.8f}`",
        "",
        "## Per-View Preview",
        "",
        "| view | has bary | candidate frac | bary valid frac | actionable frac |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("per_view", [])[:40]:
        lines.append(
            f"| {row['view']} | {int(row['has_barycentric'])} | "
            f"{float(row['candidate_fraction']):.8f} | "
            f"{float(row['barycentric_valid_fraction']):.8f} | "
            f"{float(row['actionable_fraction']):.8f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit held-out target support for a residual surface atlas.")
    parser.add_argument("--target_evidence_dir", required=True)
    parser.add_argument("--region_carrier_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--max_carriers", type=int, default=64)
    parser.add_argument("--max_faces_per_carrier", type=int, default=128)
    parser.add_argument("--max_faces", type=int, default=4096)
    parser.add_argument("--barycentric_tolerance", type=float, default=0.05)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--min_l1", type=float, default=0.0)
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    args = parser.parse_args()
    summary = audit_coverage(args)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_md(Path(args.out_md), summary)
    print(json.dumps({k: summary[k] for k in ("candidate_face_fraction", "barycentric_valid_fraction", "actionable_fraction")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Collect ECSR Phase-A surface evidence diagnostics across scenes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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

OUTDOOR = {"bicycle", "flowers", "garden", "stump", "treehill"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence"),
    )
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-PhaseA-SurfaceEvidence.md"),
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/"
            "phase_a_surface_evidence_summary.json"
        ),
    )
    parser.add_argument(
        "--md_out",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/"
            "phase_a_surface_evidence_summary.md"
        ),
    )
    parser.add_argument(
        "--montage_out",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/ecsr_phase_a/surface_evidence/"
            "phase_a_surface_evidence_contact_sheet.png"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.fmean(values) if values else None


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_montage(records: list[dict[str, Any]], out_path: Path) -> None:
    images: list[tuple[str, Image.Image]] = []
    for record in records:
        path = Path(record["artifacts"]["contact_sheet"])
        if path.exists():
            images.append((str(record["scene"]), Image.open(path).convert("RGB")))
    if not images:
        return

    cell_w = 420
    label_h = 30
    gap = 14
    cols = 3
    rows = (len(images) + cols - 1) // cols
    resized: list[tuple[str, Image.Image]] = []
    for scene, image in images:
        scale = cell_w / float(image.width)
        cell_h = int(image.height * scale)
        resized.append((scene, image.resize((cell_w, cell_h), Image.Resampling.LANCZOS)))
    max_h = max(image.height for _, image in resized)
    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * gap, rows * (max_h + label_h) + (rows + 1) * gap),
        (22, 24, 28),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    for idx, (scene, image) in enumerate(resized):
        col = idx % cols
        row = idx // cols
        x = gap + col * (cell_w + gap)
        y = gap + row * (max_h + label_h + gap)
        draw.text((x, y), scene, font=font, fill=(245, 245, 245))
        canvas.paste(image, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def classify_action(record: dict[str, Any]) -> str:
    scene = str(record["scene"])
    address = record["diagnostic_a"]["surface_addressability"]
    consistency = record["diagnostic_a"]["residual_multiview_consistency"]
    relocation = record["diagnostic_b"]["verdict"]
    if address == "pass" and consistency == "pass" and relocation == "appearance-relocation-promising":
        return "direct surface residual candidate"
    if address == "pass" and consistency == "pass":
        return "cluster/attribute recovery candidate"
    if address == "pass" and scene in OUTDOOR:
        return "needs cluster-level view-support graph"
    return "geometry/topology first"


def build_summary(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    scenes = [scene.strip() for scene in args.scenes.split(",") if scene.strip()]
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for scene in scenes:
        summary_path = args.surface_root / scene / "surface_evidence_summary.json"
        if not summary_path.exists():
            missing.append(scene)
            continue
        record = load_json(summary_path)
        record["scene_type"] = "outdoor" if scene in OUTDOOR else "indoor"
        record["next_action"] = classify_action(record)
        records.append(record)

    outdoor_records = [r for r in records if r["scene_type"] == "outdoor"]
    indoor_records = [r for r in records if r["scene_type"] == "indoor"]
    address_pass = sum(
        1 for r in records if r.get("diagnostic_a", {}).get("surface_addressability") == "pass"
    )
    consistency_pass = sum(
        1 for r in records if r.get("diagnostic_a", {}).get("residual_multiview_consistency") == "pass"
    )
    relocation_promising = sum(
        1 for r in records if r.get("diagnostic_b", {}).get("verdict") == "appearance-relocation-promising"
    )
    aggregate = {
        "num_scenes": len(records),
        "missing_scenes": missing,
        "surface_addressability_pass": address_pass,
        "residual_multiview_consistency_pass": consistency_pass,
        "appearance_relocation_promising": relocation_promising,
        "mean_valid_face_id_fraction": mean(records, "mean_valid_face_id_fraction"),
        "mean_top_error_addressable_fraction": mean(records, "mean_top_error_addressable_fraction"),
        "mean_top_support_multiview_fraction": mean(records, "top_support_multiview_fraction"),
        "mean_top_support_consistency": mean(records, "top_support_mean_multiview_consistency"),
        "outdoor_mean_multiview_fraction": mean(outdoor_records, "top_support_multiview_fraction"),
        "indoor_mean_multiview_fraction": mean(indoor_records, "top_support_multiview_fraction"),
    }
    payload = {
        "protocol": {
            "split": "train",
            "views_per_scene": 8,
            "view_stride": 6,
            "test_usage": "none for evidence generation or candidate decisions",
            "surface_root": str(args.surface_root),
        },
        "aggregate": aggregate,
        "records": records,
    }

    rows: list[list[Any]] = []
    for r in records:
        rows.append(
            [
                r["scene"],
                r["scene_type"],
                r["num_views"],
                pct(r["mean_valid_face_id_fraction"], 2),
                pct(r["mean_top_error_addressable_fraction"], 2),
                pct(r["top_support_multiview_fraction"], 2),
                fmt(r["top_support_mean_multiview_consistency"], 4),
                r["diagnostic_a"]["surface_addressability"],
                r["diagnostic_a"]["residual_multiview_consistency"],
                r["diagnostic_b"]["verdict"],
                r["next_action"],
            ]
        )

    artifact_rows = []
    for r in records:
        scene_dir = args.surface_root / str(r["scene"])
        artifact_rows.append(
            [
                r["scene"],
                f"`{scene_dir / 'surface_evidence_report.md'}`",
                f"`{scene_dir / 'top_residual_supports.csv'}`",
                f"`{scene_dir / 'surface_residual_contact_sheet.png'}`",
            ]
        )

    md = [
        "# ECSR Phase-A Surface Evidence Diagnostics",
        "",
        "This report is generated from train-view renders only. It is the Phase-A",
        "acceptance artifact for the FinalDecision ECSR plan: it checks whether",
        "current residual signals are surface-addressable before any new",
        "contraction or surface-attached recovery module is promoted.",
        "",
        "## Protocol",
        "",
        "- split: `train` only",
        "- scenes: `" + ", ".join(scenes) + "`",
        "- selected views: `8` per scene, stride `6`, offset `0`",
        "- held-out test usage: `none`",
        "- per-scene cache root: `" + str(args.surface_root) + "`",
        "- generated from compact checkpoint root: `compact_ela_sor_adaptive_geo_26k/*/sor_adaptive_geo/compact_model`",
        "",
        "## Aggregate Result",
        "",
        md_table(
            ["metric", "value"],
            [
                ["scenes collected", f"{len(records)} / {len(scenes)}"],
                ["surface addressability pass", f"{address_pass} / {len(records)}"],
                ["residual multiview consistency pass", f"{consistency_pass} / {len(records)}"],
                ["appearance-relocation promising", f"{relocation_promising} / {len(records)}"],
                ["mean valid face-id fraction", pct(aggregate["mean_valid_face_id_fraction"], 2)],
                ["mean top-error addressable fraction", pct(aggregate["mean_top_error_addressable_fraction"], 2)],
                ["mean top-support multiview fraction", pct(aggregate["mean_top_support_multiview_fraction"], 2)],
                ["mean top-support consistency", fmt(aggregate["mean_top_support_consistency"], 4)],
                ["outdoor mean multiview fraction", pct(aggregate["outdoor_mean_multiview_fraction"], 2)],
                ["indoor mean multiview fraction", pct(aggregate["indoor_mean_multiview_fraction"], 2)],
            ],
        ),
        "",
        "## Per-Scene Diagnostic",
        "",
        md_table(
            [
                "scene",
                "type",
                "views",
                "valid face-id",
                "top-error addressable",
                "top support multiview",
                "top consistency",
                "A-address",
                "A-consistency",
                "B-relocation",
                "next action",
            ],
            rows,
        ),
        "",
        "## Surface Evidence Artifacts",
        "",
        md_table(["scene", "report", "top support list", "contact sheet"], artifact_rows),
        "",
        "A combined visual index is written to `" + str(args.montage_out) + "`.",
        "",
        "## Interpretation",
        "",
        "1. The residual signal is strongly surface-addressable: every collected scene",
        "   passes the addressability diagnostic, with almost all high-error pixels",
        "   carrying valid rendered face ids.",
        "2. Direct per-face residual relocation is not yet a safe universal policy.",
        "   Outdoor scenes have high addressability but sparse top-support multiview",
        "   redundancy, so single-face residual deltas would risk view-specific",
        "   artifacts. The next representation-level step must aggregate supports",
        "   into local clusters or a view-support redundancy graph.",
        "3. Indoor scenes have stronger multiview support but weak aggregate ELA",
        "   relocation signal. They are better suited for certificate contraction",
        "   plus attribute-only recovery than for high-capacity residual deltas.",
        "4. Existing README qualitative crops remain presentation evidence only.",
        "   Phase-A top support masks are the replacement source for train-defined",
        "   local evaluation and must drive future qualitative crop selection.",
        "",
        "## Phase-A Acceptance",
        "",
        "Phase A is accepted as a diagnostic foundation, not as the final method.",
        "It proves that surface addressing is technically available and that the",
        "failure mode is not random image-space noise. It also rejects a naive",
        "single-face SH-delta implementation as the next main method because the",
        "top supports are not sufficiently multiview-stable on several outdoor",
        "scenes.",
        "",
        "## Concrete Next Step",
        "",
        "Proceed to Phase B with a fixed View-Support Redundancy Graph:",
        "",
        "- nodes: local face clusters, not isolated faces",
        "- edges: adjacency plus train-view co-visibility, depth/normal compatibility, residual compatibility, and occlusion risk",
        "- candidate outputs: attribute-only merge, conservative cluster contraction, and no-topology surface residual relocation",
        "- certificate: train/policy-val only, with test reserved for final report",
        "",
        "This is the cleanest path to turn the current ELA-dominated version into",
        "a representation-level ECSR method without per-scene parameter games.",
        "",
    ]
    if missing:
        md.extend(["## Missing Scenes", "", ", ".join(missing), ""])
    return payload, "\n".join(md)


def main() -> int:
    args = parse_args()
    payload, md = build_summary(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    build_montage(payload["records"], args.montage_out)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(md + "\n", encoding="utf-8")
    args.doc_out.write_text(md + "\n", encoding="utf-8")
    print(f"[ECSR] wrote {args.doc_out}")
    print(f"[ECSR] wrote {args.json_out}")
    print(f"[ECSR] wrote {args.md_out}")
    print(f"[ECSR] wrote {args.montage_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

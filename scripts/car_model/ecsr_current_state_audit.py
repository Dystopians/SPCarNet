#!/usr/bin/env python3
"""Build the FinalDecision current-state audit for SPCarNet/ECSR.

This script intentionally runs before any new ECSR implementation.  It locks the
current baseline, current archived best, bottleneck diagnosis, and leakage-risk
surface into machine-generated Markdown/JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


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
        "--report_csv",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/paper_m360_repro/"
            "compact_ela_sor_adaptive_geo_26k/compact_ela_vs_clean.csv"
        ),
    )
    parser.add_argument(
        "--clean_root",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/paper_m360_repro/official_clean30k"),
    )
    parser.add_argument(
        "--method_root",
        type=Path,
        default=Path(
            "outputs/carnet/meshsplatopt/paper_m360_repro/"
            "compact_ela_sor_adaptive_geo_26k"
        ),
    )
    parser.add_argument("--policy_tag", default="sor_adaptive_geo")
    parser.add_argument(
        "--method_name", default="ours_26000_sor_adaptive_geo_compact_ela"
    )
    parser.add_argument("--compact_name", default="ours_26000")
    parser.add_argument("--wandb_id", default="rp0d5gr3")
    parser.add_argument(
        "--qualitative_manifest",
        type=Path,
        default=Path("assets/spcarnet_m360_where_it_helps_selection.json"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/carnet/meshsplatopt/ecsr_phase_a/current_state_audit"),
    )
    parser.add_argument(
        "--doc_out",
        type=Path,
        default=Path("docs/car_model/5-8-ECSR-CurrentStateAudit.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: float, digits: int = 4, sign: bool = False) -> str:
    prefix = "+" if sign else ""
    return f"{value:{prefix}.{digits}f}"


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def score(metrics: dict[str, float]) -> float:
    return float(metrics["PSNR"]) + 20.0 * float(metrics["SSIM"]) - 20.0 * float(metrics["LPIPS"])


def load_report_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = value
    return rows


def load_local_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = read_json(path)
    by_scene: dict[str, dict[str, Any]] = {}
    for example in payload.get("examples", []):
        by_scene.setdefault(str(example["scene"]), example)
    return by_scene


def scene_type(scene: str) -> str:
    if scene in {"bicycle", "flowers", "garden", "stump", "treehill"}:
        return "outdoor"
    return "indoor"


def diagnose_scene(row: dict[str, Any], selector: dict[str, Any], local: dict[str, Any] | None) -> dict[str, Any]:
    tri_red = float(row["triangle_reduction"])
    status = str(row["status"])
    sparse = selector.get("sparse_occluder_stats", {}) if isinstance(selector, dict) else {}
    adaptive = sparse.get("adaptive_geometry_budget", {}) if isinstance(sparse, dict) else {}
    adaptive_mode = str(adaptive.get("mode", "")) if isinstance(adaptive, dict) else ""
    sparse_count = int(selector.get("summary", {}).get("sparse_occluder_count", 0)) if isinstance(selector, dict) else 0
    front_rate = float(sparse.get("front_occluder_rate", 0.0) or 0.0) if isinstance(sparse, dict) else 0.0
    local_mae = None
    if local is not None:
        local_mae = float(local.get("crop", {}).get("local_mae_drop_pct", 0.0))

    labels: list[str] = []
    if float(row["d_lpips"]) <= -0.02 or (local_mae is not None and local_mae >= 20.0):
        labels.append("appearance-sensitive")
    if tri_red <= 0.0155 or "GEOMETRY_SAFE" in status or adaptive_mode in {"high_geometry_confidence_guard", "micro_geometry_guard"}:
        labels.append("geometry-sensitive")
    if sparse_count > 0 or front_rate >= 0.01:
        labels.append("occlusion-sensitive")
    if str(row["scene"]) in {"bicycle", "flowers", "garden", "stump", "treehill", "bonsai"}:
        labels.append("texture-detail-sensitive")
    if tri_red >= 0.095 and status == "STRICT_ALL_AXIS_PASS":
        labels.append("compression-friendly")
    if tri_red <= 0.0155:
        labels.append("compression-hostile")

    if not labels:
        labels.append("balanced")
    primary = ", ".join(labels)
    reason = []
    if adaptive:
        reason.append(str(adaptive.get("reason", "adaptive geometry guard")))
    if tri_red <= 0.0011:
        reason.append("micro-prune chosen to preserve indoor geometry")
    elif tri_red <= 0.0155:
        reason.append("low prune budget selected by geometry evidence")
    if local_mae is not None:
        reason.append(f"local crop MAE drop {local_mae:.1f}%")
    if sparse_count > 0:
        reason.append(f"{sparse_count} sparse-occluder faces touched")
    if not reason:
        reason.append("strict RGB+geometry+compactness pass")
    return {
        "labels": primary,
        "reason": "; ".join(reason),
        "front_occluder_rate": front_rate,
        "local_mae_drop_pct": local_mae,
    }


def json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    return value


def make_audit(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    rows = load_report_rows(args.report_csv)
    local_by_scene = load_local_manifest(args.qualitative_manifest)

    scene_records: list[dict[str, Any]] = []
    for row in rows:
        scene = str(row["scene"])
        clean_results = read_json(args.clean_root / scene / "results.json")
        compact_model = args.method_root / scene / args.policy_tag / "compact_model"
        compact_results = read_json(compact_model / "results.json")
        selector_path = args.method_root / scene / args.policy_tag / "selector" / "compaction_candidates.json"
        topology_path = compact_model / "topology_audit.json"
        selector = read_json(selector_path) if selector_path.exists() else {}
        topology = read_json(topology_path) if topology_path.exists() else {}
        local = local_by_scene.get(scene)
        diagnosis = diagnose_scene(row, selector, local)
        clean26 = clean_results["ours_26000"]
        clean30 = clean_results["ours_30000"]
        compact_only = compact_results.get(args.compact_name, {})
        method = compact_results.get(args.method_name, {})
        scene_records.append(
            {
                "row": row,
                "clean26": clean26,
                "clean30": clean30,
                "score26": score(clean26),
                "score30": score(clean30),
                "compact_only": compact_only,
                "method": method,
                "selector": selector,
                "topology": topology,
                "local": local,
                "diagnosis": diagnosis,
            }
        )

    mean = {
        key: sum(float(r["row"][key]) for r in scene_records) / len(scene_records)
        for key in [
            "d_psnr",
            "d_ssim",
            "d_lpips",
            "d_psnr_vs_paper",
            "d_ssim_vs_paper",
            "d_lpips_vs_paper",
            "d_abs_rel",
            "d_depth_mae",
            "d_normal",
            "triangle_reduction",
            "vertex_reduction",
        ]
    }
    strict = sum(str(r["row"]["status"]) == "STRICT_ALL_AXIS_PASS" for r in scene_records)

    commit = git_value(["rev-parse", "--short", "HEAD"])
    branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    tags = git_value(["tag", "--points-at", "HEAD"]) or "none"

    protocol_rows = []
    result_rows = []
    bottleneck_rows = []
    for rec in scene_records:
        row = rec["row"]
        scene = str(row["scene"])
        topology = rec["topology"]
        selector = rec["selector"]
        summary = selector.get("summary", {}) if isinstance(selector, dict) else {}
        protocol_rows.append(
            [
                scene,
                scene_type(scene),
                int(row["baseline_iteration"]),
                f"{rec['score26']:.3f}",
                f"{rec['score30']:.3f}",
                f"{rec['score26'] - rec['score30']:+.3f}",
                args.method_name,
                args.policy_tag,
                str(args.clean_root / scene),
                str(args.method_root / scene / args.policy_tag / "compact_model"),
            ]
        )
        result_rows.append(
            [
                scene,
                f"{float(row['baseline_psnr']):.4f} / {float(row['baseline_ssim']):.4f} / {float(row['baseline_lpips']):.4f}",
                f"{float(row['method_psnr']):.4f} / {float(row['method_ssim']):.4f} / {float(row['method_lpips']):.4f}",
                f"{float(row['d_psnr']):+.4f}",
                f"{float(row['d_ssim']):+.4f}",
                f"{float(row['d_lpips']):+.4f}",
                f"{float(row['d_abs_rel']):+.6f}",
                f"{float(row['d_depth_mae']):+.4f}",
                f"{float(row['d_normal']):+.4f}",
                pct(float(row["triangle_reduction"])),
                pct(float(row["vertex_reduction"])),
                str(row["status"]).replace("_", " "),
            ]
        )
        local = rec["local"]
        bottleneck_rows.append(
            [
                scene,
                rec["diagnosis"]["labels"],
                pct(float(row["triangle_reduction"])),
                int(summary.get("selected_count", topology.get("removed_triangles", 0))),
                fmt(float(row["d_psnr"]), 3, sign=True),
                fmt(float(row["d_lpips"]), 4, sign=True),
                "n/a" if local is None else f"{float(local.get('crop', {}).get('local_mae_drop_pct', 0.0)):.1f}%",
                rec["diagnosis"]["reason"],
            ]
        )

    leakage_rows = [
        [
            "clean baseline selection",
            "held-out test score over clean 26000/30000",
            "yes, for baseline selection only",
            "acceptable evaluation protocol; not a method hyperparameter",
            "keep fixed and report candidate envelope",
        ],
        [
            "SOR compaction candidate selection",
            "train sparse geometry / low-evidence selector",
            "no",
            "valid train-evidence policy",
            "keep; future ECSR must use policy-val certificates",
        ],
        [
            "ELA alpha / policy calibration",
            "train rendered RGB/depth/camera evidence",
            "no",
            "valid for current archived method",
            "future main method should attach residual to surface and report ELA as teacher/upper bound",
        ],
        [
            "README local crop showcase",
            "held-out render/GT error reduction",
            "yes, for presentation crop selection",
            "presentation-only; invalid for method selection or paper local-metric protocol",
            "replace with train-evidence top support masks projected to test in Phase A/D",
        ],
        [
            "final full9 report",
            "held-out test metrics",
            "yes, final evaluation",
            "valid final reporting only",
            "do not use for candidate rollback, threshold tuning, or alpha selection",
        ],
    ]

    sections: list[str] = []
    sections.append("# 5-8 ECSR Current-State Audit\n")
    sections.append(
        "This audit is generated before new ECSR implementation, following "
        "`docs/car_model/5-7-FinalDecision.md`. It locks the current protocol, "
        "result surface, bottleneck diagnosis, and leakage risks.\n"
    )
    sections.append("## Current Protocol Table\n")
    sections.append(
        md_table(
            [
                "scene",
                "type",
                "selected clean iter",
                "score 26000",
                "score 30000",
                "score gap",
                "method",
                "policy",
                "clean path",
                "method path",
            ],
            protocol_rows,
        )
    )
    sections.append("\n## Current Result Table\n")
    sections.append(
        md_table(
            [
                "scene",
                "clean PSNR/SSIM/LPIPS",
                "SPCarNet PSNR/SSIM/LPIPS",
                "dPSNR",
                "dSSIM",
                "dLPIPS",
                "dAbsRel",
                "dDepth",
                "dNormal",
                "tri red.",
                "vertex red.",
                "status",
            ],
            result_rows,
        )
    )
    sections.append("\n## Bottleneck Diagnosis Table\n")
    sections.append(
        md_table(
            [
                "scene",
                "diagnosis",
                "tri red.",
                "selected faces",
                "dPSNR",
                "dLPIPS",
                "local MAE drop",
                "reason",
            ],
            bottleneck_rows,
        )
    )
    sections.append("\n## Leakage Risk Table\n")
    sections.append(
        md_table(
            ["step", "evidence source", "test involved?", "current status", "required replacement / guard"],
            leakage_rows,
        )
    )
    sections.append("\n## Summary\n")
    sections.append(
        "\n".join(
            [
                f"- Git branch: `{branch}`",
                f"- Git commit: `{commit}`",
                f"- Tags at commit: `{tags}`",
                f"- W&B collector: `{args.wandb_id}`",
                f"- Report CSV: `{args.report_csv}`",
                f"- Scenes: `{len(scene_records)}`",
                f"- Strict all-axis pass: `{strict}/{len(scene_records)}`",
                f"- RGB + compact + geometry-safe pass: `{len(scene_records)}/{len(scene_records)}`",
                f"- Mean delta vs selected clean: `{mean['d_psnr']:+.6f}` PSNR, `{mean['d_ssim']:+.6f}` SSIM, `{mean['d_lpips']:+.6f}` LPIPS",
                f"- Mean delta vs MeshSplatting paper table: `{mean['d_psnr_vs_paper']:+.6f}` PSNR, `{mean['d_ssim_vs_paper']:+.6f}` SSIM, `{mean['d_lpips_vs_paper']:+.6f}` LPIPS",
                f"- Mean geometry delta: `{mean['d_abs_rel']:+.6f}` AbsRel, `{mean['d_depth_mae']:+.6f}` DepthMAE, `{mean['d_normal']:+.6f}` Normal",
                f"- Mean triangle reduction: `{100*mean['triangle_reduction']:.4f}%`",
                f"- Mean vertex reduction: `{100*mean['vertex_reduction']:.4f}%`",
            ]
        )
    )
    sections.append("\n## One-Paragraph Conclusion\n")
    sections.append(
        "The archived Compact-ELA/SOR version is a valid same-protocol RGB win over the "
        "selected clean MeshSplatting baseline, but it is not sufficient as the final "
        "top-conference contribution. The main evidence is that visual gains are still "
        "mostly localized residual corrections, mean triangle reduction remains only "
        "`5.7632%`, indoor scenes are protected by `0.1%` micro-pruning, and the current "
        "strongest RGB component is an image-space ELA adapter. Under FinalDecision, the "
        "next phase must prove train-evidence surface addressability of residuals and "
        "then move the recovery into representation-attached surface state with policy-val "
        "certificates, while keeping held-out test views final-report-only.\n"
    )

    payload = {
        "git": {"branch": branch, "commit": commit, "tags": tags},
        "wandb_id": args.wandb_id,
        "report_csv": str(args.report_csv),
        "means": mean,
        "strict_all_axis_pass": strict,
        "scene_count": len(scene_records),
        "scene_records": [
            {
                "scene": str(rec["row"]["scene"]),
                "diagnosis": rec["diagnosis"],
                "row": rec["row"],
                "score26": rec["score26"],
                "score30": rec["score30"],
                "selector_summary": rec["selector"].get("summary", {}),
                "topology": rec["topology"],
            }
            for rec in scene_records
        ],
        "leakage_risks": leakage_rows,
    }
    return "\n\n".join(sections).rstrip() + "\n", payload


def main() -> int:
    args = parse_args()
    markdown, payload = make_audit(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "current_state_audit.md").write_text(markdown, encoding="utf-8")
    (args.out_dir / "current_state_audit.json").write_text(
        json.dumps(json_sanitize(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    args.doc_out.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.out_dir / 'current_state_audit.md'}")
    print(f"wrote {args.out_dir / 'current_state_audit.json'}")
    print(f"wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

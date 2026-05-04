#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/car_model/final_stageF45_unified_preset_fairness_report.md"


@dataclass(frozen=True)
class Row:
    scene: str
    method: str
    triangles_clean: int | None
    triangles_method: int | None
    d_psnr: float | None
    d_ssim: float | None
    d_lpips: float | None
    d_absrel: float | None
    d_depth: float | None
    d_normal: float | None
    status: str
    evidence: str


ROWS = [
    Row(
        "parking_phone_tiny",
        "CSEF50 fixed preset",
        8548242,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "MISSING_LONG_ROW",
        "F12 currently uses CSEF70+sparse-depth, not CSEF50.",
    ),
    Row(
        "bonsai",
        "CSEF50 fixed preset",
        88460,
        44230,
        0.013148,
        0.001910,
        0.000257,
        -0.009069,
        -0.078595,
        -1.864381,
        "BORDERLINE_LPIPS_REGRESSION",
        "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
    ),
    Row(
        "courtyard",
        "CSEF50 fixed preset",
        1677484,
        838742,
        0.452301,
        0.041625,
        -0.024231,
        -0.032415,
        -0.220612,
        0.008508,
        "PASS_RENDER_DEPTH_NORMAL_TIE",
        "docs/car_model/final_stageF8_cross_scene_compact_pilot_report.md",
    ),
    Row(
        "room",
        "CSEF50 fixed preset",
        84506,
        42253,
        0.128784,
        0.014090,
        -0.010638,
        0.018745,
        0.122800,
        -0.799860,
        "MIXED_RENDER_PASS_DEPTH_FAIL",
        "docs/car_model/final_stageF20_room_posthoc_qem_baseline_report.md",
    ),
    Row(
        "counter",
        "CSEF50 fixed preset",
        83834,
        41917,
        -0.058622,
        -0.013827,
        0.016342,
        0.017735,
        0.068959,
        -0.463645,
        "FAIL",
        "docs/car_model/final_stageF10_fourth_scene_counter_report.md",
    ),
]


def fmt(v: float | int | None, digits: int = 6) -> str:
    if v is None:
        return "-"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:+.{digits}f}"


def topology_reduction(clean: int | None, method: int | None) -> str:
    if clean is None or method is None:
        return "-"
    return f"{100.0 * (1.0 - method / clean):.1f}%"


def main() -> int:
    complete = [row for row in ROWS if row.triangles_method is not None]
    pass_like = [row for row in complete if row.status.startswith("PASS")]
    mixed_like = [row for row in complete if row.status.startswith(("BORDERLINE", "MIXED"))]
    fail_like = [row for row in complete if row.status == "FAIL"]
    avg_reduction = sum(1.0 - row.triangles_method / row.triangles_clean for row in complete if row.triangles_clean) / len(complete)

    lines = [
        "# Final Stage F45 - Unified-Preset Fairness Audit",
        "",
        "Date: 2026-05-04",
        "",
        "Decision: `F45_FIXED_PRESET_AUDIT_EXPOSES_CLAIM_RISK_AND_DEFINES_F46_REPAIR`.",
        "",
        "## Goal",
        "",
        "Audit whether the current F12 multi-scene package can be honestly described as one fixed method rather than a per-scene best-row selection. This report deliberately uses only already-completed long-budget rows and does not promote any new metric.",
        "",
        "## Fixed CSEF50 Audit",
        "",
        "| scene | method | clean tris | method tris | reduction | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | status | evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in ROWS:
        lines.append(
            f"| {row.scene} | {row.method} | {fmt(row.triangles_clean)} | {fmt(row.triangles_method)} | "
            f"{topology_reduction(row.triangles_clean, row.triangles_method)} | {fmt(row.d_psnr)} | "
            f"{fmt(row.d_ssim)} | {fmt(row.d_lpips)} | {fmt(row.d_absrel)} | {fmt(row.d_depth)} | "
            f"{fmt(row.d_normal)} | `{row.status}` | `{row.evidence}` |"
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            f"The strict fixed CSEF50 evidence is not a five-scene all-metric win. Among the four completed public-scene CSEF50 long rows, it has {len(pass_like)} clear pass, {len(mixed_like)} borderline/mixed rows, and {len(fail_like)} fail. Parking does not yet have a matched CSEF50 long row; F12 uses CSEF70+sparse-depth there. The completed fixed-CSEF50 rows still reduce topology by an average of "
            f"`{100.0 * avg_reduction:.1f}%`, but fixed-CSEF50 alone is too weak for the paper's strongest claim.",
            "",
            "This means F12 must be framed as a validated per-scene operator family or as validation-selected protocol evidence, not as one universal fixed CSEF50 preset. Calling the current F12 table a single fixed method would be misleading.",
            "",
            "## F46 Repair Target",
            "",
            "The next fair repair is a unified `CSEF50 + sparse-depth strict topology-frozen recovery` protocol on the missing/weak CSEF50 scenes. The minimum useful F46 batch is bonsai, room, and counter at the same `22000->26000` schedule with online W&B, independent render metrics, COLMAP geometry, and exact topology-freeze audit. Courtyard already has this row from F30, and parking needs a separate CSEF50 run or a declared validation-selected CSEF70 branch.",
            "",
            "## Claim Rule",
            "",
            "Until F46 closes the weak rows, the safe paper language is: MeshSplatOpt is a compact-recovery operator family with a validation-selected compaction backend, not a universal fixed CSEF50 hyperparameter setting. The fixed-preset audit is intentionally kept in the record to remove hidden per-scene tuning ambiguity.",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

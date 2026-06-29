#!/usr/bin/env python3
"""Build a lightweight SPCarNet claim-readiness report from current artifacts.

The report is intentionally conservative: a claim is marked ready only when the
current files prove it. Missing volatile /dev/shm artifacts are reported as
missing rather than inferred from prose.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


PHASEJ_CLOSURE_JSON = Path("outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.json")
PHASEJ_CLOSURE_CSV = Path("outputs/carnet/meshsplatopt/ecsr_phase_j_closure_audit/phasej_closure_audit.csv")
VNEXT_STRUCTURE_MD = Path(
    "docs/car_model/vnext_artifacts/full9_structure_shrink_cleanup_20260626_1200/summary/"
    "vnext_manifest_summary_enhanced.md"
)
VNEXT_MARGIN_MD = Path(
    "docs/car_model/vnext_artifacts/full9_effective_margin_gate_20260626_1500/summary/"
    "vnext_manifest_summary_enhanced.md"
)
V166_METRICS = Path(
    "/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/"
    "flowers_ours_26000_v166_target_impact_multisample_flowers_test_results.json"
)
V166_MANIFEST = Path(
    "/dev/shm/peilincai_spcarnet_20260628_v166_target_impact_multisample_exact/flowers/reports/"
    "flowers_vnext_certified_residual_texture_manifest.json"
)
V167_METRICS = Path(
    "/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/"
    "flowers_ours_26000_v167_affine_flowers_test_results.json"
)
V167_MANIFEST = Path(
    "/dev/shm/peilincai_spcarnet_20260628_v167_affine_exact/flowers/reports/"
    "flowers_vnext_certified_residual_texture_manifest.json"
)
V168_DRYRUN_MANIFEST = Path(
    "/dev/shm/peilincai_spcarnet_20260628_distill_profile_dryrun_v2/flowers/reports/"
    "flowers_vnext_certified_residual_texture_manifest.json"
)
V168_LOG = Path("docs/car_model/6-28-v168-PhaseJDistillProfile-Protocol-Log.md")


PHASEJ_FLOWERS = {"PSNR": 20.304358, "SSIM": 0.557770, "LPIPS": 0.329222}
SELECTED_CLEAN_FULL9 = {"PSNR": 25.151682, "SSIM": 0.749018, "LPIPS": 0.287621}
V106_FULL9 = {"PSNR": 25.831280, "SSIM": 0.760830, "LPIPS": 0.268435}


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt_metric(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "missing"
    return f"{float(value):.{digits}f}"


def extract_method_metrics(path: Path) -> dict[str, float] | None:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    candidate: dict[str, Any] | None = None
    if {"PSNR", "SSIM", "LPIPS"}.issubset(payload):
        candidate = payload
    elif len(payload) == 1:
        first = next(iter(payload.values()))
        if isinstance(first, dict):
            candidate = first
    if not candidate:
        return None
    out: dict[str, float] = {}
    for key in ("PSNR", "SSIM", "LPIPS"):
        try:
            out[key] = float(candidate[key])
        except Exception:
            return None
    return out


def manifest_status(path: Path) -> tuple[str, bool | None, list[Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return ("missing", None, [])
    status = str(payload.get("status", "missing"))
    protocol = payload.get("protocol_audit", {})
    passed = protocol.get("passed") if isinstance(protocol, dict) else None
    errors = payload.get("errors", [])
    return (status, bool(passed) if passed is not None else None, errors if isinstance(errors, list) else [errors])


def phasej_summary() -> dict[str, Any]:
    payload = read_json(PHASEJ_CLOSURE_JSON)
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        summary = dict(payload["summary"])
        rows = payload.get("rows", [])
        if isinstance(rows, list) and rows:
            summary["scene_count_from_rows"] = len(rows)
            for scene in rows:
                if isinstance(scene, dict) and scene.get("scene") == "flowers":
                    summary["flowers"] = scene
                    break
        return summary
    rows: list[dict[str, str]] = []
    if PHASEJ_CLOSURE_CSV.is_file():
        with PHASEJ_CLOSURE_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {
        "scene_count": len(rows),
        "strict_rgb_wins_vs_clean": sum(row.get("strict_rgb_win_vs_clean") == "True" for row in rows),
        "flowers": next((row for row in rows if row.get("scene") == "flowers"), None),
    }


def all_axis_better(metrics: dict[str, float] | None, reference: dict[str, float]) -> bool | None:
    if metrics is None:
        return None
    return (
        metrics["PSNR"] > reference["PSNR"]
        and metrics["SSIM"] > reference["SSIM"]
        and metrics["LPIPS"] < reference["LPIPS"]
    )


def disk_line(path: str) -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return f"`{path}`: unavailable"
    total = usage.total / (1024**4)
    used = usage.used / (1024**4)
    free = usage.free / (1024**3)
    pct = 100.0 * usage.used / max(usage.total, 1)
    return f"`{path}`: {used:.1f}T / {total:.1f}T used, {free:.2f}G free, {pct:.1f}%"


def build_report() -> str:
    phasej = phasej_summary()
    v166_metrics = extract_method_metrics(V166_METRICS)
    v167_metrics = extract_method_metrics(V167_METRICS)
    v166_status, v166_protocol, v166_errors = manifest_status(V166_MANIFEST)
    v167_status, v167_protocol, v167_errors = manifest_status(V167_MANIFEST)
    v168_status, v168_protocol, v168_errors = manifest_status(V168_DRYRUN_MANIFEST)

    phasej_scene_count = int(phasej.get("scene_count", phasej.get("scene_count_from_rows", 0)) or 0)
    phasej_wins = int(phasej.get("strict_rgb_wins_vs_clean", 0) or 0)
    phasej_ready = phasej_scene_count == 9 and phasej_wins == 9

    v106_beats_clean = (
        V106_FULL9["PSNR"] > SELECTED_CLEAN_FULL9["PSNR"]
        and V106_FULL9["SSIM"] > SELECTED_CLEAN_FULL9["SSIM"]
        and V106_FULL9["LPIPS"] < SELECTED_CLEAN_FULL9["LPIPS"]
    )
    v166_gate = all_axis_better(v166_metrics, PHASEJ_FLOWERS)
    v167_gate = all_axis_better(v167_metrics, PHASEJ_FLOWERS)
    v168_has_profile = v168_status == "DRY_RUN" and bool(v168_protocol)
    v168_log_exists = V168_LOG.is_file()

    rows = [
        {
            "claim": "Phase-J is strongest local RGB endpoint",
            "status": "PASS_LOCAL" if phasej_ready else "MISSING_EVIDENCE",
            "evidence": f"scene wins {phasej_wins}/{phasej_scene_count}; {PHASEJ_CLOSURE_JSON}",
            "blocker": "Clarify that this is a render-time endpoint, not baked representation.",
        },
        {
            "claim": "v106 is strongest verified baked representation over selected clean",
            "status": "PARTIAL_PASS" if v106_beats_clean else "FAIL",
            "evidence": (
                f"v106 {fmt_metric(V106_FULL9['PSNR'])}/{fmt_metric(V106_FULL9['SSIM'])}/"
                f"{fmt_metric(V106_FULL9['LPIPS'])} vs clean "
                f"{fmt_metric(SELECTED_CLEAN_FULL9['PSNR'])}/{fmt_metric(SELECTED_CLEAN_FULL9['SSIM'])}/"
                f"{fmt_metric(SELECTED_CLEAN_FULL9['LPIPS'])}"
            ),
            "blocker": "Still weaker than Phase-J; qualitative gain is subtle.",
        },
        {
            "claim": "v166 flowers all-axis beats Phase-J",
            "status": "PASS" if v166_gate else "FAIL" if v166_gate is False else "MISSING",
            "evidence": (
                f"metrics {fmt_metric(v166_metrics.get('PSNR') if v166_metrics else None)}/"
                f"{fmt_metric(v166_metrics.get('SSIM') if v166_metrics else None)}/"
                f"{fmt_metric(v166_metrics.get('LPIPS') if v166_metrics else None)}; "
                f"manifest {v166_status}, protocol={v166_protocol}, errors={len(v166_errors)}"
            ),
            "blocker": "Wins PSNR only; loses SSIM/LPIPS vs Phase-J flowers.",
        },
        {
            "claim": "v167 flowers all-axis beats Phase-J",
            "status": "PASS" if v167_gate else "FAIL" if v167_gate is False else "MISSING",
            "evidence": (
                f"metrics {fmt_metric(v167_metrics.get('PSNR') if v167_metrics else None)}/"
                f"{fmt_metric(v167_metrics.get('SSIM') if v167_metrics else None)}/"
                f"{fmt_metric(v167_metrics.get('LPIPS') if v167_metrics else None)}; "
                f"manifest {v167_status}, protocol={v167_protocol}, errors={len(v167_errors)}"
            ),
            "blocker": "Affine/patch candidate was policy-val rejected and fell back to no-op.",
        },
        {
            "claim": "v168 Phase-J distillation profile is exact metric win",
            "status": "NOT_RUN",
            "evidence": (
                f"dry-run manifest status={v168_status}, protocol={v168_protocol}, errors={len(v168_errors)}; "
                f"log_exists={v168_log_exists}"
            ),
            "blocker": "Exact flowers validation not run; storage is unsafe.",
        },
        {
            "claim": "vNext/new prompt is paper-main method",
            "status": "FAIL",
            "evidence": "v165-v167 negative flowers evidence; vNext full9 below clean/v106/Phase-J.",
            "blocker": "Needs flowers all-axis win vs Phase-J, then fixed-policy full9 and ablations.",
        },
    ]

    lines = [
        "# SPCarNet Claim Readiness Auto Report",
        "",
        "Generated from current local artifacts. Missing volatile `/dev/shm` files are treated as missing evidence.",
        "",
        "## Storage",
        "",
        f"- {disk_line('/data')}",
        f"- {disk_line('/dev/shm')}",
        f"- {disk_line('/tmp')}",
        "",
        "## Claim Matrix",
        "",
        "| claim | status | evidence | blocker |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['claim']} | {row['status']} | {row['evidence']} | {row['blocker']} |")

    lines.extend(
        [
            "",
            "## Flowers Gate Against Phase-J",
            "",
            f"- Phase-J flowers reference: {fmt_metric(PHASEJ_FLOWERS['PSNR'])} / "
            f"{fmt_metric(PHASEJ_FLOWERS['SSIM'])} / {fmt_metric(PHASEJ_FLOWERS['LPIPS'])}",
            f"- v166: {fmt_metric(v166_metrics.get('PSNR') if v166_metrics else None)} / "
            f"{fmt_metric(v166_metrics.get('SSIM') if v166_metrics else None)} / "
            f"{fmt_metric(v166_metrics.get('LPIPS') if v166_metrics else None)}; all-axis={v166_gate}",
            f"- v167: {fmt_metric(v167_metrics.get('PSNR') if v167_metrics else None)} / "
            f"{fmt_metric(v167_metrics.get('SSIM') if v167_metrics else None)} / "
            f"{fmt_metric(v167_metrics.get('LPIPS') if v167_metrics else None)}; all-axis={v167_gate}",
            "- v168: no exact metrics yet.",
            "",
            "## Key Artifact Index",
            "",
            f"- Phase-J closure JSON: `{PHASEJ_CLOSURE_JSON}` exists={PHASEJ_CLOSURE_JSON.is_file()}",
            f"- Phase-J closure CSV: `{PHASEJ_CLOSURE_CSV}` exists={PHASEJ_CLOSURE_CSV.is_file()}",
            f"- vNext structure full9 summary: `{VNEXT_STRUCTURE_MD}` exists={VNEXT_STRUCTURE_MD.is_file()}",
            f"- vNext effective-margin full9 summary: `{VNEXT_MARGIN_MD}` exists={VNEXT_MARGIN_MD.is_file()}",
            f"- v166 manifest: `{V166_MANIFEST}` exists={V166_MANIFEST.is_file()}",
            f"- v167 manifest: `{V167_MANIFEST}` exists={V167_MANIFEST.is_file()}",
            f"- v168 dry-run manifest: `{V168_DRYRUN_MANIFEST}` exists={V168_DRYRUN_MANIFEST.is_file()}",
            f"- v168 durable log: `{V168_LOG}` exists={V168_LOG.is_file()}",
            "",
            "## Verdict",
            "",
            "Final status: NOT COMPLETE.",
            "",
            "The current repo has strong engineering scaffolding and local Phase-J/v106 evidence, but the vNext/new-prompt "
            "route is not paper-main ready until a fixed, no-target-GT, Phase-J-distilled baked representation beats "
            "Phase-J on flowers all-axis and then survives full9 promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/car_model/6-28-SPCarNet-ClaimReadiness-AutoReport.md"),
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

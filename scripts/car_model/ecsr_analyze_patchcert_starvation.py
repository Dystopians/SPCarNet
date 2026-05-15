#!/usr/bin/env python3
"""Analyze PatchCert seed starvation and rescue opportunities from artifacts.

By default the script scans existing decision/audit JSON files under the
Phase-S output root and writes the report to stdout. Optional output paths can
also archive the same machine/text report for experiment logs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("outputs/carnet/meshsplatopt/ecsr_phase_s")
AUDIT_NAME = "surface_residual_facelocal_sh1_delta_audit.json"
METRICS = ("PSNR", "SSIM", "LPIPS")


@dataclass(frozen=True)
class SourceFiles:
    decision_path: Path | None
    audit_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read PatchCert Phase-S decision/operator-audit JSON files and "
            "rank candidate starvation plus seed-rescue opportunities."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Phase-S artifact root to scan.",
    )
    parser.add_argument(
        "--label-regex",
        default="",
        help="Optional regex filter applied to the candidate label/run label.",
    )
    parser.add_argument(
        "--scenes",
        default="",
        help="Optional comma/space-separated scene filter.",
    )
    parser.add_argument(
        "--strict-only",
        action="store_true",
        help="Only show rows whose audit records strict PatchCert carrier mode.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show rows even when no starvation/rescue flag is raised.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=25,
        help="Maximum rows to print in text mode; use 0 for no limit.",
    )
    parser.add_argument(
        "--seed-thin-ratio",
        type=float,
        default=0.005,
        help="Mark seed-thin rows when seed_faces/selected_faces is below this ratio.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout instead of the text table.",
    )
    parser.add_argument(
        "--include_report_only_test_opportunities",
        action="store_true",
        help=(
            "Include retrospective report-only test deltas in opportunity labels and scores. "
            "Keep this off when using the report to choose a next method variant."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to archive the machine-readable JSON report.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=None,
        help="Optional path to archive the text/Markdown table report.",
    )
    return parser.parse_args()


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def integer(value: Any) -> int | None:
    value_num = num(value)
    if not math.isfinite(value_num):
        return None
    return int(value_num)


def ratio(part: int | None, total: int | None) -> float:
    if part is None or total in (None, 0):
        return math.nan
    return float(part) / float(total)


def split_names(raw: str) -> set[str]:
    return {item.strip() for item in raw.replace(",", " ").split() if item.strip()}


def glob_many(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: rel(path))


def decision_paths(root: Path) -> list[Path]:
    return glob_many(
        root,
        (
            "*patchcert*/decisions/*_decision.json",
            "*/*patchcert*/decisions/*_decision.json",
            "*/*/*patchcert*/decisions/*_decision.json",
        ),
    )


def audit_paths(root: Path) -> list[Path]:
    return glob_many(
        root,
        (
            f"*patchcert*/model/{AUDIT_NAME}",
            f"*patchcert*/*/model/{AUDIT_NAME}",
            f"*/*patchcert*/model/{AUDIT_NAME}",
            f"*/*patchcert*/*/model/{AUDIT_NAME}",
            f"*/*/*patchcert*/model/{AUDIT_NAME}",
            f"*/*/*patchcert*/*/model/{AUDIT_NAME}",
        ),
    )


def norm_path(path: str | Path | None, root: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        return str(p.resolve())
    except Exception:
        return str((root / path).resolve())


def candidate_run_dir(path: Path, root: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        if "patchcert" in part.lower() and part not in {"decisions", "model"}:
            return part
    return path.parent.name


def scene_from_audit_path(path: Path) -> str:
    if path.parent.name == "model":
        return path.parent.parent.name
    return ""


def scene_from_decision(path: Path, decision: dict[str, Any]) -> str:
    scene = str(decision.get("scene") or "").strip()
    if scene:
        return scene
    name = path.name
    return name[: -len("_decision.json")] if name.endswith("_decision.json") else path.stem


def label_from_run_dir(run_dir: str, scene: str) -> str:
    suffix = f"_{scene}" if scene else ""
    if suffix and run_dir.endswith(suffix):
        return run_dir[: -len(suffix)]
    return run_dir


def nested(source: dict[str, Any], *keys: str) -> Any:
    cur: Any = source
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def metric_delta(decision: dict[str, Any], metric: str) -> float:
    return num(nested(decision, "test_delta_report_only", metric))


def compact_reasons(decision: dict[str, Any]) -> list[str]:
    reasons = nested(decision, "compact_stratified_gate", "decision_reasons")
    return [str(item) for item in reasons] if isinstance(reasons, list) else []


def decision_reasons(decision: dict[str, Any]) -> list[str]:
    reasons = decision.get("decision_reasons")
    return [str(item) for item in reasons] if isinstance(reasons, list) else []


def text_bool(value: Any) -> str:
    if value is None:
        return "n/a"
    return "true" if bool(value) else "false"


def fmt_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def fmt_ratio(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.3f}%"


def fmt_float(value: float, digits: int = 6) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:+.{digits}f}"


def table_line(values: list[str]) -> str:
    return "| " + " | ".join(values) + " |"


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def build_sources(root: Path) -> list[tuple[SourceFiles, dict[str, Any], dict[str, Any]]]:
    audits_by_abs: dict[str, tuple[Path, dict[str, Any]]] = {}
    consumed_audits: set[str] = set()
    rows: list[tuple[SourceFiles, dict[str, Any], dict[str, Any]]] = []

    for path in audit_paths(root):
        payload = load_json(path)
        audits_by_abs[norm_path(path, root)] = (path, payload)

    for path in decision_paths(root):
        decision = load_json(path)
        audit_ref = nested(decision, "candidate_operator_audit", "path")
        audit_path: Path | None = None
        audit: dict[str, Any] = {}
        audit_key = norm_path(audit_ref, root)
        if audit_key in audits_by_abs:
            audit_path, audit = audits_by_abs[audit_key]
            consumed_audits.add(audit_key)
        rows.append((SourceFiles(path, audit_path), decision, audit))

    for key, (path, audit) in sorted(audits_by_abs.items(), key=lambda item: rel(item[1][0])):
        if key not in consumed_audits:
            rows.append((SourceFiles(None, path), {}, audit))
    return rows


def base_row(
    sources: SourceFiles,
    decision: dict[str, Any],
    audit: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    scene = ""
    if decision:
        scene = scene_from_decision(sources.decision_path or Path(), decision)
    if not scene and sources.audit_path is not None:
        scene = scene_from_audit_path(sources.audit_path)

    if decision:
        label = str(decision.get("candidate_label") or "").strip()
    else:
        label = ""
    if not label:
        run_dir = candidate_run_dir(sources.audit_path or sources.decision_path or root, root)
        label = label_from_run_dir(run_dir, scene)

    patch = audit.get("patch_certificate") if isinstance(audit.get("patch_certificate"), dict) else {}
    face_gain = audit.get("face_view_gain_certificate") if isinstance(audit.get("face_view_gain_certificate"), dict) else {}
    consensus = audit.get("face_view_consensus") if isinstance(audit.get("face_view_consensus"), dict) else {}
    crossfold = audit.get("crossfold_face_gain_certificate") if isinstance(audit.get("crossfold_face_gain_certificate"), dict) else {}
    compact = decision.get("compact_stratified_gate") if isinstance(decision.get("compact_stratified_gate"), dict) else {}
    operator_audit = (
        decision.get("candidate_operator_audit")
        if isinstance(decision.get("candidate_operator_audit"), dict)
        else {}
    )

    selected_faces = integer(audit.get("selected_faces"))
    face_policy_candidates = integer(audit.get("face_policy_candidates"))
    seed_faces = integer(patch.get("seed_faces"))
    accepted_patches = integer(patch.get("accepted_patches"))
    accepted_faces_after = integer(patch.get("accepted_faces_after"))
    accepted_faces = integer(audit.get("accepted_faces"))
    vertices_added = integer(audit.get("vertices_added"))

    if accepted_faces_after is None:
        accepted_faces_after = accepted_faces
    if seed_faces is None:
        seed_faces = face_policy_candidates

    fvg_pass = integer(face_gain.get("faces_passing"))
    consensus_pass = integer(consensus.get("faces_passing"))
    crossfold_pass = integer(crossfold.get("faces_passing"))
    upstream_counts = [value for value in (fvg_pass, consensus_pass, crossfold_pass) if value is not None]
    upstream_max = max(upstream_counts) if upstream_counts else None

    operator_accepted = audit.get("accepted")
    if operator_accepted is None:
        operator_accepted = operator_audit.get("accepted")
    operator_noop = audit.get("no_op_copy")
    if operator_noop is None:
        operator_noop = operator_audit.get("no_op_copy")

    strict_carrier = audit.get("strict_patchcert_carrier")
    if strict_carrier is None:
        strict_carrier = False

    return {
        "label": label,
        "scene": scene,
        "decision_present": bool(decision),
        "decision_accepted": decision.get("accepted") if decision else None,
        "operator_accepted": operator_accepted,
        "operator_noop": operator_noop,
        "strict_patchcert_carrier": strict_carrier,
        "selected_faces": selected_faces,
        "face_policy_candidates": face_policy_candidates,
        "seed_faces": seed_faces,
        "seed_ratio": ratio(seed_faces, selected_faces),
        "accepted_patches": accepted_patches,
        "accepted_faces_after": accepted_faces_after,
        "vertices_added": vertices_added,
        "face_view_gain_faces_passing": fvg_pass,
        "face_view_consensus_faces_passing": consensus_pass,
        "crossfold_faces_passing": crossfold_pass,
        "upstream_certified_faces_max": upstream_max,
        "rejected_neighbor_crossfold": integer(patch.get("rejected_neighbor_crossfold")),
        "rejected_patch_crossfold": integer(patch.get("rejected_patch_crossfold")),
        "rejected_post_shrink_policy_val": integer(patch.get("rejected_post_shrink_policy_val")),
        "rejected_patch_budget": integer(patch.get("rejected_patch_budget")),
        "decision_reasons": decision_reasons(decision),
        "compact_gate_reasons": compact_reasons(decision),
        "compact_gate_faces": integer(compact.get("accepted_faces")),
        "compact_gate_vertices": integer(compact.get("vertices_added")),
        "trainval_balanced_delta": num(decision.get("trainval_balanced_delta")) if decision else math.nan,
        "test_delta_report_only": {
            key: metric_delta(decision, key) if decision else math.nan for key in METRICS
        },
        "decision_path": rel(sources.decision_path),
        "audit_path": rel(sources.audit_path),
    }


def history_by_scene(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    history: dict[str, dict[str, Any]] = {}
    for row in rows:
        scene = row["scene"]
        item = history.setdefault(
            scene,
            {
                "max_seed_faces": 0,
                "max_seed_label": "",
                "max_patches": 0,
                "max_patches_label": "",
                "accepted_labels": [],
            },
        )
        seed = row.get("seed_faces") or 0
        if seed > item["max_seed_faces"]:
            item["max_seed_faces"] = seed
            item["max_seed_label"] = row["label"]
        patches = row.get("accepted_patches") or 0
        if patches > item["max_patches"]:
            item["max_patches"] = patches
            item["max_patches_label"] = row["label"]
        if row.get("decision_accepted") is True:
            item["accepted_labels"].append(row["label"])
    for item in history.values():
        item["accepted_labels"] = sorted(set(item["accepted_labels"]))
    return history


def annotate(
    row: dict[str, Any],
    history: dict[str, Any],
    seed_thin_ratio: float,
    *,
    include_report_only_test_opportunities: bool,
) -> dict[str, Any]:
    flags: list[str] = []
    opportunities: list[str] = []
    score = 0.0

    selected = row.get("selected_faces") or 0
    seeds = row.get("seed_faces") or 0
    patches = row.get("accepted_patches") or 0
    faces = row.get("accepted_faces_after") or 0
    upstream = row.get("upstream_certified_faces_max") or 0
    rejected_neighbor = row.get("rejected_neighbor_crossfold") or 0
    compact_reasons_list = row.get("compact_gate_reasons") or []
    decision_reasons_list = row.get("decision_reasons") or []
    test = row.get("test_delta_report_only") or {}

    if selected > 0 and seeds == 0 and faces == 0:
        flags.append("hard_seed_starved")
        score += 100.0
    elif selected > 0 and math.isfinite(row["seed_ratio"]) and row["seed_ratio"] < seed_thin_ratio:
        flags.append("seed_thin")
        score += 30.0

    if upstream > 0 and seeds == 0:
        flags.append("upstream_cert_but_zero_seed")
        opportunities.append("upstream certified faces exist but no PatchCert seed survived")
        score += 60.0 + min(upstream, 100) / 5.0

    if seeds > 0 and patches == 0 and faces <= seeds:
        flags.append("patch_growth_collapsed")
        opportunities.append("seed faces exist but grown patch carriers collapsed")
        score += 35.0

    if rejected_neighbor > max(faces, seeds, 0):
        flags.append("neighbor_crossfold_choked")
        opportunities.append(f"neighbor crossfold rejected {rejected_neighbor} candidate neighbors")
        score += 25.0 + min(rejected_neighbor, 200) / 10.0

    budget_reasons = [
        reason
        for reason in compact_reasons_list
        if any(token in reason for token in ("faces_exceed", "vertices_exceed", "face_ratio_exceed"))
    ]
    if budget_reasons:
        flags.append("compact_budget_blocked")
        opportunities.append("operator found carriers but compact topology budget blocked promotion")
        score += 15.0

    metric_positive = (
        bool(include_report_only_test_opportunities)
        and num(test.get("PSNR")) > 0.0
        and num(test.get("SSIM")) >= 0.0
        and num(test.get("LPIPS")) <= 0.0
    )
    if bool(include_report_only_test_opportunities) and row.get("decision_accepted") is False and row.get("operator_accepted") is True and metric_positive:
        flags.append("report_positive_gate_rejected")
        opportunities.append("held-out report delta is positive but train-val gate rejected")
        score += 15.0

    if row.get("decision_accepted") is False and decision_reasons_list:
        tail_reasons = [reason for reason in decision_reasons_list if "tail" in reason or "balanced" in reason]
        if tail_reasons and len(tail_reasons) == len(decision_reasons_list):
            flags.append("tail_risk_gate_blocked")
            score += 5.0

    max_seed = history.get("max_seed_faces", 0)
    if max_seed > seeds:
        source = history.get("max_seed_label", "")
        opportunities.append(f"same scene reached {max_seed} seeds in {source}")
        score += min(max_seed - seeds, 100) / 4.0

    max_patches = history.get("max_patches", 0)
    if max_patches > patches:
        source = history.get("max_patches_label", "")
        opportunities.append(f"same scene reached {max_patches} accepted patches in {source}")
        score += min(max_patches - patches, 100) / 8.0

    accepted_labels = history.get("accepted_labels") or []
    if row.get("decision_accepted") is False and accepted_labels:
        opportunities.append("scene has accepted PatchCert precedent: " + ",".join(accepted_labels[:3]))
        score += 10.0

    row["starvation_flags"] = sorted(set(flags))
    row["rescue_opportunities"] = sorted(set(opportunities))
    row["rescue_score"] = score
    return row


def collect(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root
    label_re = re.compile(args.label_regex) if args.label_regex else None
    scene_filter = split_names(args.scenes)

    all_rows: list[dict[str, Any]] = []
    for sources, decision, audit in build_sources(root):
        row = base_row(sources, decision, audit, root)
        all_rows.append(row)

    history = history_by_scene(all_rows)
    loaded: list[dict[str, Any]] = []
    for row in all_rows:
        if label_re is not None and not label_re.search(row["label"]):
            continue
        if scene_filter and row["scene"] not in scene_filter:
            continue
        if args.strict_only and row.get("strict_patchcert_carrier") is not True:
            continue
        loaded.append(row)

    rows = [
        annotate(
            row,
            history.get(row["scene"], {}),
            args.seed_thin_ratio,
            include_report_only_test_opportunities=bool(args.include_report_only_test_opportunities),
        )
        for row in loaded
    ]
    if not args.show_all:
        rows = [row for row in rows if row["starvation_flags"] or row["rescue_opportunities"]]

    rows = sorted(
        rows,
        key=lambda row: (
            -float(row.get("rescue_score") or 0.0),
            row["scene"],
            row["label"],
            row["decision_path"],
            row["audit_path"],
        ),
    )

    summary = {
        "root": rel(root),
        "decision_file_count": len(decision_paths(root)),
        "audit_file_count": len(audit_paths(root)),
        "analyzed_row_count": len(loaded),
        "reported_row_count": len(rows),
        "hard_seed_starved_count": sum(1 for row in rows if "hard_seed_starved" in row["starvation_flags"]),
        "seed_thin_count": sum(1 for row in rows if "seed_thin" in row["starvation_flags"]),
        "neighbor_crossfold_choked_count": sum(
            1 for row in rows if "neighbor_crossfold_choked" in row["starvation_flags"]
        ),
        "compact_budget_blocked_count": sum(
            1 for row in rows if "compact_budget_blocked" in row["starvation_flags"]
        ),
        "decision_accepted_count": sum(1 for row in loaded if row.get("decision_accepted") is True),
        "include_report_only_test_opportunities": bool(args.include_report_only_test_opportunities),
    }
    return {"summary": summary, "rows": rows}


def render_text(payload: dict[str, Any], max_rows: int) -> str:
    summary = payload["summary"]
    rows = payload["rows"]
    shown = rows if max_rows == 0 else rows[: max(max_rows, 0)]
    lines = [
        "PatchCert starvation scan",
        f"root: {summary['root']}",
        (
            "files: "
            f"decisions={summary['decision_file_count']} "
            f"audits={summary['audit_file_count']} "
            f"analyzed={summary['analyzed_row_count']} "
            f"reported={summary['reported_row_count']}"
        ),
        (
            "flags: "
            f"hard_seed_starved={summary['hard_seed_starved_count']} "
            f"seed_thin={summary['seed_thin_count']} "
            f"neighbor_crossfold_choked={summary['neighbor_crossfold_choked_count']} "
            f"compact_budget_blocked={summary['compact_budget_blocked_count']} "
            f"decision_accepted={summary['decision_accepted_count']}"
        ),
        "",
        table_line(
            [
                "score",
                "scene",
                "label",
                "decision",
                "strict",
                "seeds/selected",
                "patches",
                "faces",
                "upstream",
                "rej_neighbor",
                "flags",
                "top opportunity",
            ]
        ),
        table_line(["---"] * 12),
    ]
    for row in shown:
        top_opportunity = row["rescue_opportunities"][0] if row["rescue_opportunities"] else ""
        upstream_bits = [
            f"fvg={fmt_int(row['face_view_gain_faces_passing'])}",
            f"cons={fmt_int(row['face_view_consensus_faces_passing'])}",
            f"cf={fmt_int(row['crossfold_faces_passing'])}",
        ]
        lines.append(
            table_line(
                [
                    f"{row['rescue_score']:.1f}",
                    row["scene"],
                    row["label"],
                    text_bool(row["decision_accepted"]),
                    text_bool(row["strict_patchcert_carrier"]),
                    f"{fmt_int(row['seed_faces'])}/{fmt_int(row['selected_faces'])} ({fmt_ratio(row['seed_ratio'])})",
                    fmt_int(row["accepted_patches"]),
                    fmt_int(row["accepted_faces_after"]),
                    ",".join(upstream_bits),
                    fmt_int(row["rejected_neighbor_crossfold"]),
                    ",".join(row["starvation_flags"]),
                    top_opportunity,
                ]
            )
        )

    hidden = len(rows) - len(shown)
    if hidden > 0:
        lines.extend(["", f"... {hidden} additional rows hidden by --max-rows"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    payload = collect(args)
    json_text = json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
    text = render_text(payload, args.max_rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text, encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(text, encoding="utf-8")
    if args.json:
        print(json_text, end="")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

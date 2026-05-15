#!/usr/bin/env python3
"""Select bounded Phase-S PatchCert carriers from train-side evidence.

The utility is intentionally read-only. It scans PatchCert/Phase-S decision,
operator-audit, candidate-plan, and trial-manifest JSON artifacts, then emits a
deterministic carrier/face selection that can be replayed later as a fixed
policy ablation. Held-out test metrics are report-only by default.
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
class ArtifactPaths:
    decision: Path | None = None
    audit: Path | None = None
    plan: Path | None = None
    manifest: Path | None = None
    trainval_per_view: Path | None = None
    test_per_view: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bounded, deterministic Phase-S PatchCert carrier/face "
            "selection from candidate/audit/plan JSON artifacts. Selection "
            "uses train-side evidence by default; test inputs are report-only."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Phase-S artifact root to scan.")
    parser.add_argument("--scenes", default="", help="Optional comma/space-separated scene filter.")
    parser.add_argument("--label-regex", default="", help="Optional regex filter for candidate/trial labels.")
    parser.add_argument(
        "--candidate-json",
        action="append",
        type=Path,
        default=[],
        help="Additional decision/candidate JSON path. May be repeated.",
    )
    parser.add_argument("--audit-json", action="append", type=Path, default=[], help="Additional audit JSON path.")
    parser.add_argument("--plan-json", action="append", type=Path, default=[], help="Additional candidate plan JSON path.")
    parser.add_argument(
        "--manifest-json",
        action="append",
        type=Path,
        default=[],
        help="Additional coupled-selector trial manifest JSON path.",
    )
    parser.add_argument(
        "--trainval-metrics-json",
        action="append",
        type=Path,
        default=[],
        help="Optional trainval per-view metric JSON. Used only as train-side support evidence.",
    )
    parser.add_argument(
        "--test-metrics-json",
        action="append",
        type=Path,
        default=[],
        help="Optional held-out test per-view metric JSON. Report-only unless --allow-test-for-selection is set.",
    )
    parser.add_argument("--max-carriers-per-scene", type=int, default=3, help="Bound carrier rows per scene.")
    parser.add_argument("--max-faces-per-carrier", type=int, default=64, help="Bound selected faces per carrier.")
    parser.add_argument("--max-faces-per-scene", type=int, default=128, help="Bound total unique faces per scene.")
    parser.add_argument(
        "--include-rejected-carriers",
        action="store_true",
        help="Allow train-val rejected carriers into the bounded list after strict/rescued carriers.",
    )
    parser.add_argument(
        "--allow-test-for-selection",
        action="store_true",
        help="Unsafe retrospective mode: allow test deltas to affect ranking. Off by default.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Write machine-readable selection JSON.")
    parser.add_argument("--output-md", type=Path, default=None, help="Write Markdown summary.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout instead of Markdown.")
    return parser.parse_args()


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def resolve_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {"_non_dict_payload": payload}


def safe_load(path: Path | None, blocked: list[str], kind: str) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return load_json(path)
    except Exception as exc:
        blocked.append(f"{kind}_load_failed:{rel(path)}:{type(exc).__name__}")
        return {}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def optional_int(value: Any) -> int | None:
    out = optional_float(value)
    return None if out is None else int(out)


def nested(source: dict[str, Any], *keys: str) -> Any:
    cur: Any = source
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def split_names(raw: str) -> set[str]:
    return {item.strip() for item in raw.replace(",", " ").split() if item.strip()}


def glob_many(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths, key=rel)


def discover_decisions(root: Path) -> list[Path]:
    return glob_many(
        root,
        (
            "**/decisions/*_decision.json",
            "**/*/coupled_selector_decision.json",
        ),
    )


def discover_audits(root: Path) -> list[Path]:
    return glob_many(root, (f"**/{AUDIT_NAME}",))


def discover_plans(root: Path) -> list[Path]:
    return glob_many(root, ("**/*candidate_plan.json",))


def discover_manifests(root: Path) -> list[Path]:
    return glob_many(root, ("**/trial_manifests/*.json",))


def discover_metric_files(root: Path, name: str) -> list[Path]:
    return glob_many(root, (f"**/{name}",))


def norm_abs(path: str | Path | None, root: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def infer_scene_from_path(path: Path | None) -> str:
    if path is None:
        return ""
    parts = path.parts
    if path.name.endswith("_decision.json"):
        return path.name[: -len("_decision.json")]
    if path.name == "coupled_selector_decision.json" and len(parts) >= 2:
        return path.parent.name
    if path.name == AUDIT_NAME:
        if len(parts) >= 3 and path.parent.name in {"model", "plan_model"}:
            return path.parent.parent.name
    if path.name.endswith("_per_view.json") and len(parts) >= 2:
        return path.parent.name
    return ""


def infer_label(path: Path | None, scene: str, decision: dict[str, Any], audit: dict[str, Any]) -> str:
    for value in (
        decision.get("selected_label"),
        decision.get("candidate_label"),
        audit.get("candidate_label"),
        audit.get("operator"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    if path is None:
        return "unknown"
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "trials" and idx + 1 < len(parts):
            return parts[idx + 1]
    for part in reversed(parts):
        if scene and part.endswith(f"_{scene}"):
            return part[: -len(scene) - 1]
        if any(token in part.lower() for token in ("patchcert", "patchrisk", "georisk", "facelocal")):
            return part
    return path.parent.name


def trial_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    cur = path if path.is_dir() else path.parent
    while cur != cur.parent:
        if cur.parent.name == "trials":
            return cur
        cur = cur.parent
    return None


def scene_root_from_trial(path: Path | None, scene: str) -> Path | None:
    tdir = trial_dir(path)
    if tdir is None:
        return None
    candidate = tdir / scene
    return candidate if candidate.exists() else tdir


def path_key(scene: str, label: str) -> str:
    return f"{scene}\0{label}"


def index_plans(paths: list[Path], root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in paths:
        scene = path.parent.name
        if scene:
            out.setdefault(scene, path)
        payload = safe_load(path, [], "plan")
        if isinstance(payload.get("scene"), str):
            out.setdefault(str(payload["scene"]), path)
    return out


def index_manifests(paths: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in paths:
        scene = path.parent.parent.name if path.parent.name == "trial_manifests" else ""
        label = path.stem
        if scene and label:
            out.setdefault(path_key(scene, label), path)
    return out


def metric_file_key(path: Path) -> str:
    scene = infer_scene_from_path(path)
    tdir = trial_dir(path)
    label = tdir.name if tdir is not None else ""
    if scene and label:
        return path_key(scene, label)
    return rel(path)


def metric_indexes(paths: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    by_carrier: dict[str, Path] = {}
    by_scene: dict[str, Path] = {}
    for path in paths:
        scene = infer_scene_from_path(path)
        if scene:
            by_scene.setdefault(scene, path)
        key = metric_file_key(path)
        by_carrier.setdefault(key, path)
    return by_carrier, by_scene


def nearby_manifest_path(decision_path: Path | None, scene: str) -> Path | None:
    tdir = trial_dir(decision_path)
    if tdir is None:
        return None
    root = tdir.parent.parent if tdir.parent.name == "trials" else None
    if root is None:
        return None
    path = root / scene / "trial_manifests" / f"{tdir.name}.json"
    return path if path.exists() else None


def nearby_trainval_per_view_path(decision_path: Path | None, scene: str) -> Path | None:
    scene_root = scene_root_from_trial(decision_path, scene)
    if scene_root is None:
        return None
    path = scene_root / "phasej_trainval_gate_per_view.json"
    return path if path.exists() else None


def decision_train_delta(decision: dict[str, Any]) -> dict[str, float]:
    delta = decision.get("trainval_delta")
    if isinstance(delta, dict):
        return {metric: finite_float(delta.get(metric)) for metric in METRICS}
    summary = decision.get("trainval_delta_summary")
    if isinstance(summary, dict):
        return {metric: finite_float(nested(summary, metric, "mean")) for metric in METRICS}
    return {}


def decision_test_delta(decision: dict[str, Any]) -> dict[str, float]:
    delta = decision.get("test_delta_report_only")
    if isinstance(delta, dict):
        return {metric: finite_float(delta.get(metric)) for metric in METRICS}
    return {}


def balanced_delta(delta: dict[str, float]) -> float:
    return (
        finite_float(delta.get("PSNR"))
        + 20.0 * finite_float(delta.get("SSIM"))
        - 20.0 * finite_float(delta.get("LPIPS"))
    )


def per_view_delta(
    payload: dict[str, Any],
    base_method: str | None,
    candidate_method: str | None,
) -> dict[str, Any]:
    if not base_method or not candidate_method:
        return {"available": False, "blocked_reason": "missing_method_names"}
    if base_method not in payload or candidate_method not in payload:
        return {"available": False, "blocked_reason": "method_not_found"}
    base = payload.get(base_method)
    cand = payload.get(candidate_method)
    if not isinstance(base, dict) or not isinstance(cand, dict):
        return {"available": False, "blocked_reason": "method_payload_not_dict"}

    metric_summary: dict[str, Any] = {}
    balanced_values: list[float] = []
    for metric in METRICS:
        bmap = base.get(metric)
        cmap = cand.get(metric)
        if not isinstance(bmap, dict) or not isinstance(cmap, dict):
            continue
        values: list[float] = []
        common = sorted(set(bmap) & set(cmap))
        for view in common:
            raw = finite_float(cmap[view]) - finite_float(bmap[view])
            values.append(-raw if metric == "LPIPS" else raw)
        if values:
            metric_summary[metric] = {
                "count": len(values),
                "mean_improvement": sum(values) / len(values),
                "min_improvement": min(values),
                "positive_fraction": sum(1 for value in values if value > 0.0) / len(values),
            }
    counts = [item["count"] for item in metric_summary.values()]
    if set(METRICS).issubset(metric_summary):
        # Approximate per-view balanced support only over views common to all metrics.
        common_views = (
            set(base["PSNR"])
            & set(cand["PSNR"])
            & set(base["SSIM"])
            & set(cand["SSIM"])
            & set(base["LPIPS"])
            & set(cand["LPIPS"])
        )
        for view in sorted(common_views):
            balanced_values.append(
                finite_float(cand["PSNR"][view])
                - finite_float(base["PSNR"][view])
                + 20.0 * (finite_float(cand["SSIM"][view]) - finite_float(base["SSIM"][view]))
                - 20.0 * (finite_float(cand["LPIPS"][view]) - finite_float(base["LPIPS"][view]))
            )
    return {
        "available": bool(metric_summary),
        "view_count_min": min(counts) if counts else 0,
        "metric_summary": metric_summary,
        "balanced_mean": sum(balanced_values) / len(balanced_values) if balanced_values else None,
        "balanced_min": min(balanced_values) if balanced_values else None,
        "balanced_positive_fraction": (
            sum(1 for value in balanced_values if value > 0.0) / len(balanced_values)
            if balanced_values
            else None
        ),
    }


def face_rows_from_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("face_scores")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and optional_int(row.get("face_id")) is not None]


def face_rows_from_audit(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("accepted_preview")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        face_id = optional_int(row.get("face_id"))
        if face_id is None:
            continue
        flat = {
            "face_id": face_id,
            "rank": row.get("rank"),
            "policy_val_relative_gain": nested(row, "policy_val_proxy", "relative_gain"),
            "policy_val_samples": nested(row, "policy_val_proxy", "samples"),
            "train_certificate_score": nested(row, "face_stats", "score"),
            "view_hits": nested(row, "face_stats", "view_hits"),
            "pixel_count": nested(row, "face_stats", "pixel_count"),
            "carrier_role": "audit_accepted_preview",
        }
        out.append(flat)
    return out


def face_rows_from_plan(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rows", "faces", "candidate_faces", "face_scores"):
        rows = payload.get(key)
        if isinstance(rows, list):
            out = []
            for row in rows:
                if isinstance(row, dict) and optional_int(row.get("face_id")) is not None:
                    out.append(row)
            if out:
                return out
    return []


def face_score(row: dict[str, Any]) -> float:
    score = 0.0
    score += finite_float(row.get("train_certificate_score"))
    score += 0.25 * finite_float(row.get("risk_adjusted_selection_score"))
    score += 0.25 * finite_float(row.get("georisk_adjusted_selection_score"))
    score += 5.0 * finite_float(row.get("policy_val_relative_gain"))
    score += 0.02 * finite_float(row.get("policy_val_samples"))
    score += 0.01 * finite_float(row.get("view_hits") or row.get("georisk_view_count"))
    rank = optional_float(row.get("rank"))
    if rank is not None:
        score += 1.0 / (1.0 + max(rank, 0.0))
    return score


def normalize_face_row(row: dict[str, Any], source: str, carrier_label: str) -> dict[str, Any]:
    face_id = optional_int(row.get("face_id"))
    role = str(row.get("patchrisk_role") or row.get("carrier_role") or "").strip()
    seed_face = optional_int(row.get("patchrisk_seed_face"))
    out = {
        "face_id": face_id,
        "carrier_label": carrier_label,
        "carrier_role": role or "unknown",
        "patch_seed_face": seed_face,
        "patch_size": optional_int(row.get("patchrisk_patch_size")),
        "rank": optional_int(row.get("rank")),
        "evidence_score": face_score(row),
        "train_certificate_score": optional_float(row.get("train_certificate_score")),
        "risk_adjusted_selection_score": optional_float(row.get("risk_adjusted_selection_score")),
        "georisk_adjusted_selection_score": optional_float(row.get("georisk_adjusted_selection_score")),
        "policy_val_relative_gain": optional_float(row.get("policy_val_relative_gain")),
        "policy_val_samples": optional_float(row.get("policy_val_samples")),
        "view_support": optional_float(row.get("georisk_view_count") or row.get("view_hits")),
        "provenance": {"source": source},
    }
    return out


def strict_provenance(decision: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    decision_accepted = decision.get("accepted")
    selection_uses_test = bool(decision.get("selection_uses_test", False))
    audit_accepted = audit.get("accepted")
    if audit_accepted is None:
        audit_accepted = nested(decision, "candidate_operator_audit", "accepted")
    test_usage = str(audit.get("test_usage") or "").lower()
    strict_carrier = audit.get("strict_patchcert_carrier")
    if strict_carrier is None:
        strict_carrier = "patchcert" in str(decision.get("candidate_label") or "").lower()
    strict = (
        decision_accepted is True
        and not selection_uses_test
        and audit_accepted is True
        and test_usage in {"", "none"}
    )
    return {
        "strict": bool(strict),
        "decision_accepted": decision_accepted,
        "operator_accepted": audit_accepted,
        "operator_policy_pass": audit.get("policy_pass", nested(decision, "candidate_operator_audit", "policy_pass")),
        "selection_uses_test": selection_uses_test,
        "audit_test_usage": audit.get("test_usage"),
        "strict_patchcert_carrier": strict_carrier,
        "decision_reasons": decision.get("decision_reasons", []),
    }


def carrier_score(
    strict: dict[str, Any],
    train_delta: dict[str, float],
    train_per_view: dict[str, Any],
    selected_faces: int,
    best_face_score: float,
    test_delta: dict[str, float],
    allow_test: bool,
) -> float:
    score = 0.0
    if strict["strict"]:
        score += 1_000_000.0
    elif strict["decision_accepted"] is True and not strict["selection_uses_test"]:
        score += 500_000.0
    score += 1_000.0 * balanced_delta(train_delta)
    if train_per_view.get("balanced_mean") is not None:
        score += 500.0 * finite_float(train_per_view.get("balanced_mean"))
    if train_per_view.get("balanced_min") is not None:
        score += 100.0 * finite_float(train_per_view.get("balanced_min"))
    score += 0.1 * min(selected_faces, 10_000)
    score += 0.01 * best_face_score
    if allow_test:
        score += 100.0 * balanced_delta(test_delta)
    return score


def build_carriers(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    root = args.root
    blocked: list[str] = []

    decisions = sorted({*(resolve_path(p, root) for p in args.candidate_json), *discover_decisions(root)}, key=rel)
    audits = sorted({*(resolve_path(p, root) for p in args.audit_json), *discover_audits(root)}, key=rel)
    plans = sorted({*(resolve_path(p, root) for p in args.plan_json), *discover_plans(root)}, key=rel)
    manifests = sorted({*(resolve_path(p, root) for p in args.manifest_json), *discover_manifests(root)}, key=rel)
    train_metric_paths = sorted(
        {*(resolve_path(p, root) for p in args.trainval_metrics_json), *discover_metric_files(root, "phasej_trainval_gate_per_view.json")},
        key=rel,
    )
    test_metric_paths = sorted({*(resolve_path(p, root) for p in args.test_metrics_json)}, key=rel)

    audit_by_abs = {norm_abs(path, root): path for path in audits}
    plan_by_scene = index_plans(plans, root)
    manifest_by_key = index_manifests(manifests)
    train_by_carrier, train_by_scene = metric_indexes(train_metric_paths)
    test_by_carrier, test_by_scene = metric_indexes(test_metric_paths)

    scene_filter = split_names(args.scenes)
    label_re = re.compile(args.label_regex) if args.label_regex else None
    carriers: list[dict[str, Any]] = []
    consumed_audits: set[Path] = set()

    for decision_path in decisions:
        decision = safe_load(decision_path, blocked, "decision")
        scene = str(decision.get("scene") or infer_scene_from_path(decision_path))
        if scene_filter and scene not in scene_filter:
            continue
        audit_ref = nested(decision, "candidate_operator_audit", "path")
        audit_path = audit_by_abs.get(norm_abs(audit_ref, root)) if audit_ref else None
        if audit_path is None:
            candidate = scene_root_from_trial(decision_path, scene)
            if candidate is not None:
                candidate_audit = candidate / "model" / AUDIT_NAME
                if candidate_audit.exists():
                    audit_path = candidate_audit
        audit = safe_load(audit_path, blocked, "audit")
        if audit_path is not None:
            consumed_audits.add(audit_path)

        label = infer_label(decision_path, scene, decision, audit)
        if label_re is not None and not label_re.search(label):
            continue
        short_trial = trial_dir(decision_path).name if trial_dir(decision_path) is not None else label
        manifest_path = nearby_manifest_path(decision_path, scene) or manifest_by_key.get(path_key(scene, short_trial))
        plan_path = plan_by_scene.get(scene)
        train_path = (
            nearby_trainval_per_view_path(decision_path, scene)
            or train_by_carrier.get(path_key(scene, short_trial))
            or train_by_scene.get(scene)
        )
        test_path = test_by_carrier.get(path_key(scene, short_trial)) or test_by_scene.get(scene)
        carrier = build_carrier(
            ArtifactPaths(decision_path, audit_path, plan_path, manifest_path, train_path, test_path),
            decision,
            audit,
            safe_load(plan_path, blocked, "plan"),
            safe_load(manifest_path, blocked, "manifest"),
            safe_load(train_path, blocked, "trainval_metrics"),
            safe_load(test_path, blocked, "test_metrics"),
            blocked,
            args.allow_test_for_selection,
        )
        carriers.append(carrier)

    for audit_path in audits:
        if audit_path in consumed_audits:
            continue
        audit = safe_load(audit_path, blocked, "audit")
        scene = infer_scene_from_path(audit_path)
        if scene_filter and scene not in scene_filter:
            continue
        label = infer_label(audit_path, scene, {}, audit)
        if label_re is not None and not label_re.search(label):
            continue
        plan_path = plan_by_scene.get(scene)
        carrier = build_carrier(
            ArtifactPaths(None, audit_path, plan_path, None, None, None),
            {},
            audit,
            safe_load(plan_path, blocked, "plan"),
            {},
            {},
            {},
            blocked,
            args.allow_test_for_selection,
        )
        carriers.append(carrier)

    return carriers, sorted(set(blocked))


def build_carrier(
    paths: ArtifactPaths,
    decision: dict[str, Any],
    audit: dict[str, Any],
    plan: dict[str, Any],
    manifest: dict[str, Any],
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    blocked: list[str],
    allow_test: bool,
) -> dict[str, Any]:
    scene = str(decision.get("scene") or infer_scene_from_path(paths.decision) or infer_scene_from_path(paths.audit))
    label = infer_label(paths.decision or paths.audit, scene, decision, audit)
    strict = strict_provenance(decision, audit)
    train_delta = decision_train_delta(decision)
    test_delta = decision_test_delta(decision)
    train_per_view = per_view_delta(
        train_metrics,
        decision.get("base_trainval_method"),
        decision.get("candidate_trainval_method"),
    ) if train_metrics else {"available": False, "blocked_reason": "missing_trainval_per_view_json"}
    test_per_view = per_view_delta(
        test_metrics,
        decision.get("base_test_method_report_only"),
        decision.get("candidate_test_method_report_only"),
    ) if test_metrics else {"available": False, "blocked_reason": "missing_test_per_view_json"}

    face_sources = [
        ("manifest_face_scores", face_rows_from_manifest(manifest)),
        ("audit_accepted_preview", face_rows_from_audit(audit)),
        ("candidate_plan", face_rows_from_plan(plan)),
    ]
    by_face: dict[int, dict[str, Any]] = {}
    for source, rows in face_sources:
        for row in rows:
            face_id = optional_int(row.get("face_id"))
            if face_id is None:
                continue
            normalized = normalize_face_row(row, source, label)
            prev = by_face.get(face_id)
            if prev is None or normalized["evidence_score"] > prev["evidence_score"]:
                by_face[face_id] = normalized
            elif prev is not None:
                prev.setdefault("provenance", {}).setdefault("additional_sources", []).append(source)

    if not by_face:
        if audit.get("materialize_plan_face_ids"):
            for text in str(audit.get("materialize_plan_face_ids")).split(","):
                face_id = optional_int(text.strip())
                if face_id is not None:
                    by_face[face_id] = {
                        "face_id": face_id,
                        "carrier_label": label,
                        "carrier_role": "materialized_face_id",
                        "patch_seed_face": None,
                        "patch_size": None,
                        "rank": None,
                        "evidence_score": 0.0,
                        "train_certificate_score": None,
                        "risk_adjusted_selection_score": None,
                        "georisk_adjusted_selection_score": None,
                        "policy_val_relative_gain": None,
                        "policy_val_samples": None,
                        "view_support": None,
                        "provenance": {"source": "audit_materialize_plan_face_ids"},
                    }
        else:
            blocked.append(f"faces_unavailable:{scene}:{label}")

    faces = sorted(
        by_face.values(),
        key=lambda row: (
            -finite_float(row.get("evidence_score")),
            optional_int(row.get("rank")) if optional_int(row.get("rank")) is not None else 10**12,
            optional_int(row.get("face_id")) or 10**12,
        ),
    )
    selected_faces = optional_int(audit.get("selected_faces")) or len(faces)
    best_face_score = finite_float(faces[0].get("evidence_score")) if faces else 0.0
    rescued = any((row.get("carrier_role") or "").lower() == "neighbor" for row in faces) or "rescue" in label.lower()
    score = carrier_score(strict, train_delta, train_per_view, selected_faces, best_face_score, test_delta, allow_test)
    return {
        "scene": scene,
        "label": label,
        "rank_score": score,
        "strict_provenance": strict,
        "rescued_provenance": {
            "rescued": bool(rescued),
            "reason": "patch_neighbor_or_rescue_label" if rescued else "none_detected",
            "seed_faces": optional_int(nested(audit, "patch_certificate", "seed_faces")),
            "accepted_patches": optional_int(nested(audit, "patch_certificate", "accepted_patches")),
        },
        "carrier_provenance": {
            "operator": audit.get("operator"),
            "plan_source_operator": audit.get("plan_source_operator"),
            "materialize_plan_in": audit.get("materialize_plan_in"),
            "materialize_plan_scale": audit.get("materialize_plan_scale"),
            "materialize_plan_alpha_json": audit.get("materialize_plan_alpha_json"),
            "selected_faces": selected_faces,
            "accepted_faces": optional_int(audit.get("accepted_faces")),
            "vertices_added": optional_int(audit.get("vertices_added")),
            "no_op_copy": audit.get("no_op_copy"),
        },
        "train_evidence": {
            "trainval_delta": train_delta,
            "trainval_balanced_delta": optional_float(decision.get("trainval_balanced_delta")),
            "per_view": train_per_view,
        },
        "test_report_only": {
            "test_delta_report_only": test_delta,
            "test_balanced_delta_report_only": optional_float(decision.get("test_balanced_delta_report_only")),
            "per_view": test_per_view,
            "used_for_selection": bool(allow_test),
        },
        "faces_ranked": faces,
        "paths": {
            "decision": rel(paths.decision),
            "audit": rel(paths.audit),
            "plan": rel(paths.plan),
            "manifest": rel(paths.manifest),
            "trainval_per_view": rel(paths.trainval_per_view),
            "test_per_view": rel(paths.test_per_view),
        },
    }


def select_bounded(args: argparse.Namespace, carriers: list[dict[str, Any]], blocked: list[str]) -> dict[str, Any]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for carrier in carriers:
        if not carrier["scene"]:
            blocked.append(f"scene_unavailable:{carrier['label']}")
            continue
        if not args.include_rejected_carriers and carrier["strict_provenance"]["decision_accepted"] is False:
            continue
        by_scene.setdefault(carrier["scene"], []).append(carrier)

    scenes: list[dict[str, Any]] = []
    for scene in sorted(by_scene):
        rows = sorted(
            by_scene[scene],
            key=lambda row: (
                -finite_float(row.get("rank_score")),
                row["label"],
                row["paths"].get("decision") or row["paths"].get("audit"),
            ),
        )
        selected_carriers: list[dict[str, Any]] = []
        scene_faces_seen: set[int] = set()
        for carrier in rows:
            if len(selected_carriers) >= max(args.max_carriers_per_scene, 0):
                break
            carrier_faces: list[dict[str, Any]] = []
            for face in carrier["faces_ranked"]:
                face_id = optional_int(face.get("face_id"))
                if face_id is None or face_id in scene_faces_seen:
                    continue
                if len(carrier_faces) >= max(args.max_faces_per_carrier, 0):
                    break
                if len(scene_faces_seen) >= max(args.max_faces_per_scene, 0):
                    break
                carrier_faces.append(face)
                scene_faces_seen.add(face_id)
            if carrier_faces or not carrier["faces_ranked"]:
                item = dict(carrier)
                item["selected_faces_bounded"] = carrier_faces
                item["bounded_face_count"] = len(carrier_faces)
                selected_carriers.append(item)
        scenes.append(
            {
                "scene": scene,
                "selected_carrier_count": len(selected_carriers),
                "selected_face_count": len(scene_faces_seen),
                "selected_carriers": selected_carriers,
                "candidate_carrier_count": len(rows),
                "candidate_labels_ranked": [row["label"] for row in rows],
            }
        )

    return {
        "schema": "ecsr_phase_s_render_certified_carrier_selection_v1",
        "selection_uses_test": bool(args.allow_test_for_selection),
        "test_policy": (
            "unsafe_allow_test_for_selection_enabled"
            if args.allow_test_for_selection
            else "report_only_not_used_for_selection"
        ),
        "root": rel(args.root),
        "bounds": {
            "max_carriers_per_scene": args.max_carriers_per_scene,
            "max_faces_per_carrier": args.max_faces_per_carrier,
            "max_faces_per_scene": args.max_faces_per_scene,
        },
        "filters": {
            "scenes": sorted(split_names(args.scenes)),
            "label_regex": args.label_regex,
            "include_rejected_carriers": bool(args.include_rejected_carriers),
        },
        "summary": {
            "input_carrier_count": len(carriers),
            "scene_count": len(scenes),
            "selected_carrier_count": sum(scene["selected_carrier_count"] for scene in scenes),
            "selected_face_count": sum(scene["selected_face_count"] for scene in scenes),
            "blocked_input_count": len(set(blocked)),
        },
        "blocked_inputs": sorted(set(blocked)),
        "scenes": scenes,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def fmt_float(value: Any, digits: int = 6) -> str:
    out = optional_float(value)
    return "n/a" if out is None else f"{out:+.{digits}f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase-S Render-Certified Carrier Selection",
        "",
        (
            "Selection rule: carriers are ranked deterministically from strict "
            "train-side provenance, train-val deltas, optional trainval per-view "
            "support, and carrier/face evidence. Held-out test inputs are "
            f"`{payload['test_policy']}`."
        ),
        "",
        f"- root: `{payload['root']}`",
        f"- scenes: `{payload['summary']['scene_count']}`",
        f"- selected carriers: `{payload['summary']['selected_carrier_count']}`",
        f"- selected faces: `{payload['summary']['selected_face_count']}`",
        f"- blocked inputs: `{payload['summary']['blocked_input_count']}`",
        "",
        "| scene | carrier | strict | rescued | bounded faces | train dPSNR | train dSSIM | train dLPIPS | train balanced | test dPSNR report-only | paths |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scene in payload["scenes"]:
        for carrier in scene["selected_carriers"]:
            train_delta = carrier["train_evidence"]["trainval_delta"]
            test_delta = carrier["test_report_only"]["test_delta_report_only"]
            paths = carrier["paths"]
            path_text = paths.get("decision") or paths.get("audit") or ""
            lines.append(
                "| {scene} | `{label}` | {strict} | {rescued} | {faces} | {dpsnr} | {dssim} | {dlpips} | {bal} | {tdpsnr} | `{path}` |".format(
                    scene=carrier["scene"],
                    label=carrier["label"],
                    strict=str(carrier["strict_provenance"]["strict"]).lower(),
                    rescued=str(carrier["rescued_provenance"]["rescued"]).lower(),
                    faces=carrier["bounded_face_count"],
                    dpsnr=fmt_float(train_delta.get("PSNR")),
                    dssim=fmt_float(train_delta.get("SSIM")),
                    dlpips=fmt_float(train_delta.get("LPIPS")),
                    bal=fmt_float(carrier["train_evidence"].get("trainval_balanced_delta")),
                    tdpsnr=fmt_float(test_delta.get("PSNR")),
                    path=path_text,
                )
            )
    if payload["blocked_inputs"]:
        lines.extend(["", "## Blocked Or Partial Inputs", ""])
        for item in payload["blocked_inputs"][:100]:
            lines.append(f"- `{item}`")
        if len(payload["blocked_inputs"]) > 100:
            lines.append(f"- ... `{len(payload['blocked_inputs']) - 100}` additional blocked inputs")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    carriers, blocked = build_carriers(args)
    payload = select_bounded(args, carriers, blocked)
    payload = json_safe(payload)
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = render_md(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text, encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_text, encoding="utf-8")
    if args.json:
        print(json_text, end="")
    else:
        print(md_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

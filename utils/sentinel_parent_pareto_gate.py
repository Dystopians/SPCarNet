from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from utils.sparse_depth_regression import (
    cluster_summary_rows,
    per_view_summary_rows,
    summarize_sparse_depth_regressions,
    write_csv,
)


@dataclass(frozen=True)
class SentinelParentParetoGateConfig:
    tolerance_absrel: float = 0.0
    tolerance_mae: float = 0.0
    worst_view_regression_count_threshold: int = 0
    cluster_delta_absrel_threshold: float = 0.0
    cluster_weight_threshold: float = 0.0


def evaluate_sentinel_parent_pareto_gate(
    table: Mapping[str, np.ndarray],
    cfg: SentinelParentParetoGateConfig,
) -> dict[str, Any]:
    summary = summarize_sparse_depth_regressions(table)
    metrics = summary.get("valid_depth_metrics", {}) or {}
    parent = metrics.get("parent", {}) or {}
    candidate = metrics.get("candidate", {}) or {}
    parent_absrel = float(parent.get("abs_rel", np.inf))
    candidate_absrel = float(candidate.get("abs_rel", np.inf))
    parent_mae = float(parent.get("mae", np.inf))
    candidate_mae = float(candidate.get("mae", np.inf))
    absrel_pass = candidate_absrel <= parent_absrel + float(cfg.tolerance_absrel)
    mae_pass = candidate_mae <= parent_mae + float(cfg.tolerance_mae)

    per_view = per_view_summary_rows(table)
    worst_view_regression_count = 0
    for row in per_view:
        if float(row.get("delta_absrel", 0.0)) > float(cfg.tolerance_absrel) or float(row.get("delta_mae", 0.0)) > float(cfg.tolerance_mae):
            worst_view_regression_count = max(worst_view_regression_count, int(row.get("regressed_rel_count", 0)))
    worst_view_pass = worst_view_regression_count <= int(cfg.worst_view_regression_count_threshold)

    clusters = cluster_summary_rows(table)
    cluster_failures = []
    for row in clusters:
        count = int(row.get("count", 0))
        delta_absrel = float(row.get("delta_absrel", 0.0))
        if count > int(cfg.cluster_weight_threshold) and delta_absrel > float(cfg.cluster_delta_absrel_threshold):
            cluster_failures.append({"cluster_id": int(row.get("cluster_id", -1)), "count": count, "delta_absrel": delta_absrel})
    cluster_pass = len(cluster_failures) == 0

    passed = bool(absrel_pass and mae_pass and worst_view_pass and cluster_pass)
    return {
        "pass": passed,
        "decision": "PASS_SENTINEL_PARENT_PARETO" if passed else "FAIL_SENTINEL_PARENT_PARETO",
        "config": asdict(cfg),
        "checks": {
            "mean_absrel_nonregression": bool(absrel_pass),
            "mean_mae_nonregression": bool(mae_pass),
            "worst_view_regression_count": int(worst_view_regression_count),
            "worst_view_pass": bool(worst_view_pass),
            "cluster_failure_count": int(len(cluster_failures)),
            "cluster_pass": bool(cluster_pass),
        },
        "metrics": {
            "parent_absrel": parent_absrel,
            "candidate_absrel": candidate_absrel,
            "delta_absrel": candidate_absrel - parent_absrel,
            "parent_mae": parent_mae,
            "candidate_mae": candidate_mae,
            "delta_mae": candidate_mae - parent_mae,
        },
        "summary": summary,
        "cluster_failures": cluster_failures[:50],
    }


def write_sentinel_parent_pareto_gate_outputs(
    *,
    output_dir: Path,
    table: Mapping[str, np.ndarray],
    gate: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_view = per_view_summary_rows(table)
    clusters = cluster_summary_rows(table)
    write_csv(output_dir / "sentinel_per_view_summary.csv", per_view)
    write_csv(output_dir / "sentinel_cluster_summary.csv", clusters)
    payload = {"manifest": dict(manifest), "gate": dict(gate)}
    (output_dir / "sentinel_parent_pareto_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sentinel_gate_report.md").write_text(format_sentinel_gate_report(payload), encoding="utf-8")


def format_sentinel_gate_report(payload: Mapping[str, Any]) -> str:
    gate = dict(payload.get("gate", {}))
    metrics = dict(gate.get("metrics", {}))
    manifest = dict(payload.get("manifest", {}))
    return (
        "# Sentinel Parent-Pareto Gate Report\n\n"
        f"Decision: `{gate.get('decision', '')}`\n\n"
        f"- split: `{manifest.get('split', '')}`\n"
        f"- sentinel_cache: `{manifest.get('sentinel_cache', '')}`\n"
        f"- parent: `{manifest.get('parent_model_path', '')}` @ `{manifest.get('parent_iteration', '')}`\n"
        f"- candidate: `{manifest.get('candidate_model_path', '')}` @ `{manifest.get('candidate_iteration', '')}`\n\n"
        "| metric | parent | candidate | candidate - parent |\n"
        "|---|---:|---:|---:|\n"
        f"| Sentinel AbsRel | {metrics.get('parent_absrel', float('nan')):.9f} | {metrics.get('candidate_absrel', float('nan')):.9f} | {metrics.get('delta_absrel', float('nan')):+.9f} |\n"
        f"| Sentinel Depth MAE | {metrics.get('parent_mae', float('nan')):.9f} | {metrics.get('candidate_mae', float('nan')):.9f} | {metrics.get('delta_mae', float('nan')):+.9f} |\n\n"
        f"Checks: `{json.dumps(gate.get('checks', {}), sort_keys=True)}`\n"
    )

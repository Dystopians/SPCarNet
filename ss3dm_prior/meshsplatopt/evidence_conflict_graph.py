from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ECGConfig:
    margin_abs: float = 0.0
    margin_rel: float = 0.0
    render_gain_weight: float = 0.25
    depth_conflict_weight: float = 1.0
    absrel_conflict_weight: float = 5.0
    invalid_candidate_weight: float = 2.0
    protected_positive_evidence: float = 0.0


def _arr(payload: Mapping[str, np.ndarray], key: str, default: Any, dtype=None) -> np.ndarray:
    if key not in payload:
        return np.asarray(default, dtype=dtype)
    return np.asarray(payload[key], dtype=dtype)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _node(node_id: str, node_type: str, **attrs: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, **attrs}


def _edge(src: str, dst: str, edge_type: str, **attrs: Any) -> dict[str, Any]:
    return {"src": src, "dst": dst, "type": edge_type, **attrs}


def load_correspondence_npz(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def build_evidence_conflict_graph(
    correspondences: Mapping[str, np.ndarray],
    *,
    cfg: ECGConfig | None = None,
    source: str = "",
    split: str = "unknown",
) -> dict[str, Any]:
    cfg = cfg or ECGConfig()
    n = int(len(_arr(correspondences, "gt_depth", [], np.float64)))
    image_key = _arr(correspondences, "image_key", np.asarray(["unknown"] * n, dtype=object), object).reshape(-1)
    point_id = _arr(correspondences, "point3D_id", np.arange(n), np.int64).reshape(-1)
    cluster_id = _arr(correspondences, "cluster_id", np.full(n, -1), np.int64).reshape(-1)
    px = _arr(correspondences, "px", np.zeros(n), np.int64).reshape(-1)
    py = _arr(correspondences, "py", np.zeros(n), np.int64).reshape(-1)
    gt_depth = _arr(correspondences, "gt_depth", np.zeros(n), np.float64).reshape(-1)
    parent_abs = _arr(correspondences, "parent_abs_error", np.zeros(n), np.float64).reshape(-1)
    candidate_abs = _arr(correspondences, "candidate_abs_error", np.zeros(n), np.float64).reshape(-1)
    parent_rel = _arr(correspondences, "parent_abs_rel", np.zeros(n), np.float64).reshape(-1)
    candidate_rel = _arr(correspondences, "candidate_abs_rel", np.zeros(n), np.float64).reshape(-1)
    parent_valid = _arr(correspondences, "parent_valid", np.ones(n, dtype=bool), bool).reshape(-1)
    candidate_valid = _arr(correspondences, "candidate_valid", np.ones(n, dtype=bool), bool).reshape(-1)
    parent_rgb = _arr(correspondences, "parent_rgb_residual", np.full(n, np.nan), np.float64).reshape(-1)
    candidate_rgb = _arr(correspondences, "candidate_rgb_residual", np.full(n, np.nan), np.float64).reshape(-1)
    gate_critical = _arr(correspondences, "gate_critical", np.zeros(n, dtype=bool), bool).reshape(-1)
    depth_bin = _arr(correspondences, "depth_bin", np.asarray(["unknown"] * n, dtype=object), object).reshape(-1)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    view_ids: set[str] = set()
    sparse_ids: set[str] = set()
    cluster_ids: set[str] = set()
    certificate_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    for i in range(n):
        view_id = f"view:{str(image_key[i])}"
        sparse_id = f"sparse:{int(point_id[i])}"
        pixel_id = f"pixel:{i}"
        cluster = int(cluster_id[i])
        cluster_node_id = f"cluster:{cluster}" if cluster >= 0 else "cluster:unknown"
        depth_conflict = max(0.0, _safe_float(candidate_abs[i]) - _safe_float(parent_abs[i]) - float(cfg.margin_abs))
        absrel_conflict = max(0.0, _safe_float(candidate_rel[i]) - _safe_float(parent_rel[i]) - float(cfg.margin_rel))
        render_gain = 0.0
        if np.isfinite(parent_rgb[i]) and np.isfinite(candidate_rgb[i]):
            render_gain = max(0.0, float(parent_rgb[i] - candidate_rgb[i]))
        invalid_penalty = 1.0 if bool(parent_valid[i]) and not bool(candidate_valid[i]) else 0.0
        bad_tradeoff = bool(render_gain > 0.0 and (depth_conflict > 0.0 or absrel_conflict > 0.0))
        certificate_pressure = (
            float(cfg.depth_conflict_weight) * depth_conflict
            + float(cfg.absrel_conflict_weight) * absrel_conflict
            + float(cfg.invalid_candidate_weight) * invalid_penalty
            + float(cfg.render_gain_weight) * render_gain * float(bad_tradeoff)
        )
        certificate_id = f"cert:depth_nonregression:{i}"

        if view_id not in view_ids:
            nodes.append(_node(view_id, "view_node", image_key=str(image_key[i]), split=split))
            view_ids.add(view_id)
        if sparse_id not in sparse_ids:
            nodes.append(_node(sparse_id, "sparse_point_node", point3D_id=int(point_id[i]), gt_depth=_safe_float(gt_depth[i])))
            sparse_ids.add(sparse_id)
        if cluster_node_id not in cluster_ids:
            nodes.append(_node(cluster_node_id, "mesh_cluster_node", cluster_id=cluster))
            cluster_ids.add(cluster_node_id)
        if certificate_id not in certificate_ids:
            nodes.append(
                _node(
                    certificate_id,
                    "certificate_node",
                    certificate="depth_absrel_parent_nonregression",
                    violated=bool(depth_conflict > 0.0 or absrel_conflict > 0.0 or invalid_penalty > 0.0),
                    pressure=certificate_pressure,
                )
            )
            certificate_ids.add(certificate_id)
        nodes.append(
            _node(
                pixel_id,
                "pixel_sample_node",
                index=i,
                px=int(px[i]),
                py=int(py[i]),
                depth_bin=str(depth_bin[i]),
                depth_conflict=depth_conflict,
                absrel_conflict=absrel_conflict,
                render_gain=render_gain,
                bad_tradeoff=bad_tradeoff,
                gate_critical=bool(gate_critical[i]),
            )
        )
        edges.extend(
            [
                _edge(view_id, sparse_id, "view_observes_sparse_point"),
                _edge(sparse_id, pixel_id, "sparse_point_projects_to_pixel"),
                _edge(pixel_id, cluster_node_id, "pixel_explained_by_cluster", approximate=True),
                _edge(certificate_id, pixel_id, "certificate_constrains_pixel"),
                _edge(certificate_id, sparse_id, "certificate_constrains_sparse_point"),
            ]
        )
        rows.append(
            {
                "cluster_id": cluster,
                "image_key": str(image_key[i]),
                "count": 1,
                "depth_conflict": depth_conflict,
                "absrel_conflict": absrel_conflict,
                "render_gain": render_gain,
                "bad_tradeoff": int(bad_tradeoff),
                "gate_critical": int(bool(gate_critical[i])),
                "candidate_invalid": int(invalid_penalty > 0.0),
                "certificate_pressure": certificate_pressure,
            }
        )

    cluster_summary = summarize_ecg_clusters(rows)
    action_nodes = []
    for row in cluster_summary[: min(50, len(cluster_summary))]:
        action = "rollback-only"
        if row["certificate_pressure"] <= 0.0 and row["render_gain"] <= 0.0:
            action = "reject_unobserved"
        elif row["bad_tradeoff_count"] == 0 and row["render_gain"] > 0.0:
            action = "appearance_reset"
        action_id = f"action:{action}:{int(row['cluster_id'])}"
        action_nodes.append(_node(action_id, "edit_action_node", action=action, cluster_id=int(row["cluster_id"])))
        edges.append(_edge(action_id, f"cluster:{int(row['cluster_id'])}", "edit_action_targets_cluster"))
    nodes.extend(action_nodes)

    return {
        "metadata": {"source": source, "split": split, "config": asdict(cfg), "num_correspondences": n},
        "nodes": nodes,
        "edges": edges,
        "cluster_summary": cluster_summary,
        "certificate_summary": {
            "active_conflict_count": int(sum(1 for r in rows if r["certificate_pressure"] > 0.0)),
            "bad_tradeoff_count": int(sum(int(r["bad_tradeoff"]) for r in rows)),
            "gate_critical_count": int(sum(int(r["gate_critical"]) for r in rows)),
        },
    }


def summarize_ecg_clusters(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        cid = int(row.get("cluster_id", -1))
        dst = grouped.setdefault(
            cid,
            {
                "cluster_id": cid,
                "count": 0,
                "depth_conflict": 0.0,
                "absrel_conflict": 0.0,
                "render_gain": 0.0,
                "bad_tradeoff_count": 0,
                "gate_critical_count": 0,
                "candidate_invalid_count": 0,
                "certificate_pressure": 0.0,
                "suggested_action": "ROLLBACK_ONLY",
            },
        )
        dst["count"] += int(row.get("count", 1))
        dst["depth_conflict"] += _safe_float(row.get("depth_conflict", 0.0))
        dst["absrel_conflict"] += _safe_float(row.get("absrel_conflict", 0.0))
        dst["render_gain"] += _safe_float(row.get("render_gain", 0.0))
        dst["bad_tradeoff_count"] += int(row.get("bad_tradeoff", 0))
        dst["gate_critical_count"] += int(row.get("gate_critical", 0))
        dst["candidate_invalid_count"] += int(row.get("candidate_invalid", 0))
        dst["certificate_pressure"] += _safe_float(row.get("certificate_pressure", 0.0))
    for dst in grouped.values():
        if dst["certificate_pressure"] <= 0.0:
            dst["suggested_action"] = "APPEARANCE_ONLY" if dst["render_gain"] > 0.0 else "REJECT_UNOBSERVED"
        elif dst["candidate_invalid_count"] > 0:
            dst["suggested_action"] = "ROLLBACK_ONLY"
        elif dst["bad_tradeoff_count"] > 0:
            dst["suggested_action"] = "ROLLBACK_ONLY"
    out = list(grouped.values())
    out.sort(
        key=lambda r: (
            int(r["cluster_id"]) < 0,
            -float(r["certificate_pressure"]),
            -int(r["gate_critical_count"]),
            int(r["cluster_id"]),
        )
    )
    return out


def write_ecg_outputs(graph: Mapping[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence_conflict_graph.json").write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(graph.get("cluster_summary", []))
    if rows:
        with (out / "ecg_cluster_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(
        out / "evidence_conflict_graph.npz",
        cluster_id=np.asarray([r["cluster_id"] for r in rows], dtype=np.int64),
        certificate_pressure=np.asarray([r["certificate_pressure"] for r in rows], dtype=np.float64),
        gate_critical_count=np.asarray([r["gate_critical_count"] for r in rows], dtype=np.int64),
    )
    top = rows[:20]
    report = ["# Evidence Conflict Graph Report", "", f"- correspondences: `{graph.get('metadata', {}).get('num_correspondences', 0)}`", ""]
    report.append("## Top Conflict Clusters")
    report.append("")
    report.append("| cluster | pressure | gate-critical | action |")
    report.append("|---:|---:|---:|---|")
    for row in top:
        report.append(
            f"| {int(row['cluster_id'])} | {float(row['certificate_pressure']):.6f} | {int(row['gate_critical_count'])} | {row['suggested_action']} |"
        )
    report.append("")
    report.append("ECG is not a cherry-picking metric. Train/calibration ECGs may drive policy; held-out/test ECGs are diagnostic audit only.")
    (out / "ecg_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

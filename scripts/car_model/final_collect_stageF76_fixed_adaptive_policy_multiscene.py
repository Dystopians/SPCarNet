#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ss3dm_prior.meshsplatopt.checkpoint_compaction import checkpoint_path  # noqa: E402
from ss3dm_prior.meshsplatopt.evaluation_contracts import load_geometry_metrics, load_render_metrics  # noqa: E402


@dataclass(frozen=True)
class SceneSpec:
    scene: str
    clean_model: str
    clean_iteration: int
    method_model: str
    method_iteration: int
    wandb: str


@dataclass(frozen=True)
class Row:
    scene: str
    wandb: str
    clean_triangles: int | None
    method_triangles: int | None
    reduction: float
    selected_fraction: float
    adaptive_fraction: float
    topology_unchanged: bool | None
    psnr: float
    ssim: float
    lpips: float
    abs_rel: float
    depth_mae: float
    normal: float
    clean_psnr: float
    clean_ssim: float
    clean_lpips: float
    clean_abs_rel: float
    clean_depth_mae: float
    clean_normal: float
    d_psnr: float
    d_ssim: float
    d_lpips: float
    d_abs_rel: float
    d_depth_mae: float
    d_normal: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SPECS = [
    SceneSpec(
        "bonsai",
        "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000",
        22000,
        "outputs/carnet/meshsplatopt/final_stageF76_fixed_adaptive_policy_multiscene/bonsai/adaptive_f75_policy/recovery_model",
        26000,
        "36hnyxkj",
    ),
    SceneSpec(
        "courtyard",
        "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000",
        22000,
        "outputs/carnet/meshsplatopt/final_stageF76_fixed_adaptive_policy_multiscene/courtyard/adaptive_f75_policy/recovery_model",
        26000,
        "d8xi1h50",
    ),
    SceneSpec(
        "room",
        "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000",
        22000,
        "outputs/carnet/meshsplatopt/final_stageF76_fixed_adaptive_policy_multiscene/room/adaptive_f75_policy/recovery_model",
        26000,
        "0mtv6wjp",
    ),
    SceneSpec(
        "counter",
        "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000",
        22000,
        "outputs/carnet/meshsplatopt/final_stageF76_fixed_adaptive_policy_multiscene/counter/adaptive_f75_policy/recovery_model",
        26000,
        "h06zst15",
    ),
]

CLEAN_MODELS = {
    "bonsai": "outputs/carnet/meshsplatopt/finalF3_bonsai_clean_long_9000to22000",
    "courtyard": "outputs/carnet/meshsplatopt/finalF3_courtyard_clean_long_9000to22000",
    "room": "outputs/carnet/meshsplatopt/finalF9_room_clean_long_9000to22000",
    "counter": "outputs/carnet/meshsplatopt/finalF10_counter_clean_long_9000to22000",
}


def _build_specs(stage_group: str, policy_tag: str) -> list[SceneSpec]:
    return [
        SceneSpec(
            scene,
            clean_model,
            22000,
            f"outputs/carnet/meshsplatopt/{stage_group}/{scene}/{policy_tag}/recovery_model",
            26000,
            "",
        )
        for scene, clean_model in CLEAN_MODELS.items()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _topology(model: Path, iteration: int) -> tuple[int | None, int | None]:
    try:
        import torch

        state = torch.load(checkpoint_path(model, iteration), map_location="cpu")
        return int(state["_triangle_indices"].shape[0]), int(state["triangles_points"].shape[0])
    except Exception:
        return None, None


def _selector_summary(method_model: Path) -> tuple[float, float]:
    selector = method_model.parent / "selector" / "compaction_candidates.json"
    payload = _read_json(selector)
    summary = payload.get("summary") or {}
    selected_fraction = float(summary.get("selected_fraction", math.nan))
    policy = payload.get("adaptive_policy_decision") or {}
    adaptive_fraction = float(policy.get("target_prune_fraction", math.nan))
    return selected_fraction, adaptive_fraction


def _topology_audit(method_model: Path) -> bool | None:
    payload = _read_json(method_model.parent / "recovery_contract" / "topology_audit.json")
    if not payload:
        payload = _read_json(method_model / "topology_audit.json")
    value = payload.get("topology_unchanged")
    return bool(value) if value is not None else None


def _wandb_id(method_model: Path, fallback: str) -> str:
    wandb_dir = method_model / "wandb"
    if not wandb_dir.is_dir():
        return fallback
    runs = sorted(wandb_dir.glob("run-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return fallback
    name = runs[0].name
    return name.rsplit("-", 1)[-1] if "-" in name else fallback


def _status(row: Row) -> str:
    values = [row.psnr, row.ssim, row.lpips, row.abs_rel, row.depth_mae, row.normal]
    if not all(math.isfinite(v) for v in values):
        return "PENDING_OR_MISSING_EVAL"
    checks = [
        row.d_psnr > 0.0,
        row.d_ssim > 0.0,
        row.d_lpips < 0.0,
        row.d_abs_rel < 0.0,
        row.d_depth_mae < 0.0,
        row.d_normal < 0.0,
    ]
    if all(checks):
        return "PASS_ALL_METRIC_CLEAN_WIN"
    render = checks[0] and checks[1] and checks[2]
    geometry = checks[3] and checks[4] and checks[5]
    if render or geometry:
        return "MIXED"
    return "FAIL"


def _row(spec: SceneSpec) -> Row:
    clean_model = ROOT / spec.clean_model
    method_model = ROOT / spec.method_model
    clean_render = load_render_metrics(clean_model, spec.clean_iteration)
    clean_geom = load_geometry_metrics(clean_model, spec.clean_iteration)
    method_render = load_render_metrics(method_model, spec.method_iteration)
    method_geom = load_geometry_metrics(method_model, spec.method_iteration)
    clean_triangles, _ = _topology(clean_model, spec.clean_iteration)
    method_triangles, _ = _topology(method_model, spec.method_iteration)
    selected_fraction, adaptive_fraction = _selector_summary(method_model)
    reduction = (
        1.0 - float(method_triangles) / float(clean_triangles)
        if clean_triangles and method_triangles
        else math.nan
    )
    row = Row(
        scene=spec.scene,
        wandb=_wandb_id(method_model, spec.wandb),
        clean_triangles=clean_triangles,
        method_triangles=method_triangles,
        reduction=reduction,
        selected_fraction=selected_fraction,
        adaptive_fraction=adaptive_fraction,
        topology_unchanged=_topology_audit(method_model),
        psnr=method_render["psnr"],
        ssim=method_render["ssim"],
        lpips=method_render["lpips"],
        abs_rel=method_geom["abs_rel"],
        depth_mae=method_geom["depth_mae"],
        normal=method_geom["normal_mean_ang_deg"],
        clean_psnr=clean_render["psnr"],
        clean_ssim=clean_render["ssim"],
        clean_lpips=clean_render["lpips"],
        clean_abs_rel=clean_geom["abs_rel"],
        clean_depth_mae=clean_geom["depth_mae"],
        clean_normal=clean_geom["normal_mean_ang_deg"],
        d_psnr=method_render["psnr"] - clean_render["psnr"],
        d_ssim=method_render["ssim"] - clean_render["ssim"],
        d_lpips=method_render["lpips"] - clean_render["lpips"],
        d_abs_rel=method_geom["abs_rel"] - clean_geom["abs_rel"],
        d_depth_mae=method_geom["depth_mae"] - clean_geom["depth_mae"],
        d_normal=method_geom["normal_mean_ang_deg"] - clean_geom["normal_mean_ang_deg"],
        status="",
    )
    return Row(**{**row.to_dict(), "status": _status(row)})


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[Row], stage_id: str) -> None:
    lines = [
        f"# Final Stage {stage_id} Fixed Adaptive Policy Multiscene Validation",
        "",
        "All rows use the same policy: `csef_adaptive_policy` compaction, strict topology freeze, sparse-depth lambda `0.001`, LPIPS lambda `0.00025`, and `22000->26000` recovery.",
        "",
        "| scene | W&B | adaptive prune | triangles | reduction | PSNR | SSIM | LPIPS | AbsRel | DepthMAE | Normal | dPSNR | dSSIM | dLPIPS | dAbsRel | dDepth | dNormal | topology | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.scene} | `{row.wandb}` | {_fmt(row.adaptive_fraction)} | {_fmt(row.method_triangles)} | "
            f"{100.0 * row.reduction:.1f}% | {_fmt(row.psnr)} | {_fmt(row.ssim)} | {_fmt(row.lpips)} | "
            f"{_fmt(row.abs_rel)} | {_fmt(row.depth_mae)} | {_fmt(row.normal)} | "
            f"{row.d_psnr:+.6f} | {row.d_ssim:+.6f} | {row.d_lpips:+.6f} | "
            f"{row.d_abs_rel:+.6f} | {row.d_depth_mae:+.6f} | {row.d_normal:+.6f} | "
            f"{row.topology_unchanged} | `{row.status}` |"
        )
    pass_count = sum(row.status == "PASS_ALL_METRIC_CLEAN_WIN" for row in rows)
    available = sum(row.status != "PENDING_OR_MISSING_EVAL" for row in rows)
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- available rows: `{available}` / `{len(rows)}`",
            f"- all-metric clean wins: `{pass_count}` / `{len(rows)}`",
            "- this table is a fixed-policy fairness validation; failed or mixed rows are not hidden by scene-specific retuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect F76 fixed adaptive policy multiscene validation.")
    parser.add_argument("--stage-id", default="F76")
    parser.add_argument("--stage-group", default="")
    parser.add_argument("--policy-tag", default="adaptive_f75_policy")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    stage_group = args.stage_group or f"final_stage{args.stage_id}_fixed_adaptive_policy_multiscene"
    out_dir = ROOT / (args.out_dir or f"outputs/carnet/meshsplatopt/{stage_group}")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs(stage_group, args.policy_tag)
    rows = [_row(spec) for spec in specs]
    payload = [row.to_dict() for row in rows]
    (out_dir / "fixed_adaptive_policy_multiscene_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "fixed_adaptive_policy_multiscene_results.csv", payload)
    _write_md(out_dir / "fixed_adaptive_policy_multiscene_results.md", rows, args.stage_id)
    print(f"Wrote {len(rows)} rows to {out_dir}; available={sum(r.status != 'PENDING_OR_MISSING_EVAL' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

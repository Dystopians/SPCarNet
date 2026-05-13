#!/usr/bin/env python3
"""Strict-render-calibrated greedy search for subdivision candidate subsets.

The subdivision operator produces a train-only local proxy ranking.  This
script adds a second train-only selection layer that materializes small
candidate subsets, renders them, runs the existing multi-offset train-val gate,
and accepts a subset only when the real render gate passes.

Held-out test metrics are not used by this script.  The search objective is
computed only from the strict train-val gate payload.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASEJ_METHOD = "ours_26000_phasej_guarded_adaptedge_ela"
BASE_METHOD = "ours_26000_phasef_extra_compact_base"
OUTDOOR_SCENES = {"bicycle", "flowers", "garden", "stump", "treehill"}
METRICS = ("PSNR", "SSIM", "LPIPS")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _run(cmd: list[str], *, gpu: int, log_path: Path, wandb_online: bool = False) -> None:
    env = os.environ.copy()
    if int(gpu) >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if wandb_online:
        env["WANDB_MODE"] = "online"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
        handle.write(f"\n[exit_code] {proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def _candidate_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("candidates")
    if not isinstance(rows, list):
        rows = plan.get("accepted")
    if not isinstance(rows, list):
        rows = plan.get("accepted_preview")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            face_id = int(row["face_id"])
        except Exception:
            continue
        if face_id in seen:
            continue
        seen.add(face_id)
        out.append(row)
    return out


def _write_subset_plan(
    *,
    source_plan: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    materialize_mode: str,
    output: Path,
) -> None:
    selected = [rows[index] for index in indices]
    payload = {
        "operator": "surface_residual_subdivision_delta_render_calibrated_subset",
        "source_plan": str(source_plan),
        "source_operator": plan.get("operator"),
        "source_model": plan.get("source_model"),
        "iteration": plan.get("iteration"),
        "feature_mode": plan.get("feature_mode", "dc"),
        "materialize_mode": materialize_mode,
        "selected_indices": [int(index) for index in indices],
        "selected_face_ids": [int(row.get("face_id", -1)) for row in selected],
        "candidate_count": int(len(selected)),
        "candidates": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _image_set(scene: str, args: argparse.Namespace) -> str:
    return str(args.outdoor_images if scene in OUTDOOR_SCENES else args.indoor_images)


def _render_maps(
    args: argparse.Namespace,
    *,
    scene: str,
    model: Path,
    method_name: str,
    log_path: Path,
) -> None:
    train_dir = model / "train" / method_name
    test_dir = model / "test" / method_name
    if (
        not bool(args.force)
        and (train_dir / "camera_index.json").is_file()
        and (test_dir / "camera_index.json").is_file()
        and (train_dir / "depths").is_dir()
        and (test_dir / "depths").is_dir()
    ):
        return
    cmd = [
        sys.executable,
        "scripts/car_model/meshsplatopt_render_evidence_maps.py",
        "-s",
        str(Path(args.dataset_root) / scene),
        "-m",
        str(model),
        "-i",
        _image_set(scene, args),
        "--resolution",
        "-1",
        "--eval",
        "--iteration",
        str(args.iteration),
        "--method_name",
        method_name,
        "--quiet",
    ]
    if bool(args.skip_failed_views):
        cmd.append("--skip_failed_views")
    _run(cmd, gpu=int(args.gpu), log_path=log_path)


def _materialize_trial(
    args: argparse.Namespace,
    *,
    subset_plan: Path,
    trial_model: Path,
    feature_mode: str,
    materialize_mode: str,
    log_path: Path,
) -> Path:
    audit = trial_model / "surface_residual_subdivision_delta_audit.json"
    checkpoint = trial_model / "point_cloud" / f"iteration_{int(args.iteration)}" / "point_cloud_state_dict.pt"
    if not bool(args.force) and audit.is_file() and checkpoint.is_file():
        return audit
    if trial_model.exists() and bool(args.force):
        shutil.rmtree(trial_model)
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_apply_surface_residual_subdivision_delta.py",
        "--source_model",
        str(_as_path(args.phasej_model)),
        "--evidence_dir",
        str(_as_path(args.evidence_dir)),
        "--output_model",
        str(trial_model),
        "--iteration",
        str(args.iteration),
        "--feature_mode",
        feature_mode,
        "--materialize_mode",
        materialize_mode,
        "--materialize_plan_in",
        str(subset_plan),
        "--min_effective_mean_relative_gain",
        str(args.min_effective_mean_relative_gain),
        "--min_effective_min_relative_gain",
        str(args.min_effective_min_relative_gain),
        "--min_effective_delta_abs_mean",
        str(args.min_effective_delta_abs_mean),
        "--min_materialized_attribute_delta",
        str(args.min_materialized_attribute_delta),
    ]
    if str(materialize_mode) == "vertex_delta":
        cmd.extend(
            [
                "--vertex_delta_min_incident_support_fraction",
                str(args.vertex_delta_min_incident_support_fraction),
                "--vertex_delta_max_incident_faces",
                str(args.vertex_delta_max_incident_faces),
            ]
        )
    if bool(args.allow_no_effect_accept):
        cmd.append("--allow_no_effect_accept")
    _run(cmd, gpu=-1, log_path=log_path)
    return audit


def _run_gate(
    args: argparse.Namespace,
    *,
    scene: str,
    trial_model: Path,
    trial_id: str,
    candidate_base_method: str,
    audit: Path,
    log_path: Path,
) -> dict[str, Any]:
    gate_root = _as_path(args.output_root) / scene / "gates" / trial_id
    gate_json = gate_root / scene / "multifold_trainval_gate.json"
    if bool(args.reuse_trial_gate_cache) and not bool(args.force) and gate_json.is_file():
        return _read_json(gate_json)
    gate_work_root = _as_path(args.output_root) / scene / "gates" / "_shared_work"
    cmd = [
        sys.executable,
        "scripts/car_model/ecsr_run_phasek_multifold_trainval_gate.py",
        "--scene",
        scene,
        "--phasej_model",
        str(_as_path(args.phasej_model)),
        "--candidate_model",
        str(trial_model),
        "--candidate_audit_json",
        str(audit),
        "--output_root",
        str(gate_work_root),
        "--candidate_label",
        f"{args.candidate_label}_{trial_id}",
        "--fallback_label",
        "phasej_guarded_adaptedge",
        "--candidate_base_method",
        candidate_base_method,
        "--phasej_trainval_method_prefix",
        str(args.phasej_trainval_method_prefix),
        "--candidate_trainval_method_prefix",
        f"{args.candidate_trainval_method_prefix}_{trial_id}",
        "--offsets",
        str(args.offsets),
        "--iteration",
        str(args.iteration),
        "--gpu",
        str(args.gpu),
        "--policy_holdout_fraction",
        str(args.policy_holdout_fraction),
        "--calib_sampler",
        str(args.calib_sampler),
        "--calib_max_views",
        str(args.calib_max_views),
        "--calib_stride",
        str(args.calib_stride),
        "--alpha_feature_mode",
        str(args.alpha_feature_mode),
        "--alpha_default",
        str(args.alpha_default),
        "--gate_min_psnr_gain",
        str(args.gate_min_psnr_gain),
        "--gate_max_ssim_regression",
        str(args.gate_max_ssim_regression),
        "--gate_max_lpips_regression",
        str(args.gate_max_lpips_regression),
        "--gate_min_balanced_delta",
        str(args.gate_min_balanced_delta),
        "--ssim_weight",
        str(args.ssim_weight),
        "--lpips_weight",
        str(args.lpips_weight),
        "--wandb_project",
        str(args.wandb_project),
        "--wandb_group",
        str(args.wandb_group),
        "--wandb_name",
        f"{args.wandb_name}_{scene}_{trial_id}",
        "--early_stop_on_failure",
    ]
    if bool(args.force):
        cmd.append("--force")
    _run(cmd, gpu=int(args.gpu), log_path=log_path, wandb_online=True)
    work_scene = gate_work_root / scene
    gate_json.parent.mkdir(parents=True, exist_ok=True)
    work_json = work_scene / "multifold_trainval_gate.json"
    work_md = work_scene / "multifold_trainval_gate.md"
    work_log = work_scene / "multifold_trainval_gate.log"
    if work_json.is_file():
        shutil.copy2(work_json, gate_json)
    if work_md.is_file():
        shutil.copy2(work_md, gate_json.with_suffix(".md"))
    if work_log.is_file():
        shutil.copy2(work_log, gate_json.with_suffix(".log"))
    return _read_json(gate_json)


def _objective(gate: dict[str, Any], args: argparse.Namespace) -> float:
    summary = gate.get("trainval_delta_summary") or {}
    psnr = float((summary.get("PSNR") or {}).get("mean", math.nan))
    ssim = float((summary.get("SSIM") or {}).get("mean", math.nan))
    lpips = float((summary.get("LPIPS") or {}).get("mean", math.nan))
    if not all(math.isfinite(x) for x in (psnr, ssim, lpips)):
        return -math.inf
    mode = str(args.objective)
    if mode == "lpips_reduction":
        return -lpips
    if mode == "balanced":
        return psnr + float(args.ssim_weight) * ssim - float(args.lpips_weight) * lpips
    if mode == "psnr_lpips":
        return psnr - float(args.lpips_weight) * lpips
    raise ValueError(f"unknown objective: {mode}")


def _delta_summary(gate: dict[str, Any]) -> dict[str, float]:
    summary = gate.get("trainval_delta_summary") or {}
    out: dict[str, float] = {}
    for key in METRICS:
        try:
            out[key] = float((summary.get(key) or {}).get("mean"))
        except Exception:
            out[key] = math.nan
    return out


def _topology(audit_path: Path) -> dict[str, Any]:
    audit = _read_json(audit_path)
    return {
        "accepted_faces": int(audit.get("accepted_faces", 0) or 0),
        "topology_before": audit.get("topology_before", {}),
        "topology_after": audit.get("topology_after", {}),
    }


def _trial(
    args: argparse.Namespace,
    *,
    plan_path: Path,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    indices: list[int],
    trial_id: str,
    log_path: Path,
) -> dict[str, Any]:
    scene = str(args.scene)
    feature_mode = str(plan.get("feature_mode", args.feature_mode))
    materialize_mode = str(plan.get("materialize_mode", args.materialize_mode))
    trial_root = _as_path(args.output_root) / scene / "trials" / trial_id
    trial_model = trial_root / "model"
    subset_plan = trial_root / "candidate_subset_plan.json"
    candidate_base_method = f"{args.candidate_base_method_prefix}_{trial_id}_base"
    _write_subset_plan(
        source_plan=plan_path,
        plan=plan,
        rows=rows,
        indices=indices,
        materialize_mode=materialize_mode,
        output=subset_plan,
    )
    audit = _materialize_trial(
        args,
        subset_plan=subset_plan,
        trial_model=trial_model,
        feature_mode=feature_mode,
        materialize_mode=materialize_mode,
        log_path=log_path,
    )
    _render_maps(args, scene=scene, model=trial_model, method_name=candidate_base_method, log_path=log_path)
    gate = _run_gate(
        args,
        scene=scene,
        trial_model=trial_model,
        trial_id=trial_id,
        candidate_base_method=candidate_base_method,
        audit=audit,
        log_path=log_path,
    )
    delta = _delta_summary(gate)
    return {
        "trial_id": trial_id,
        "indices": [int(index) for index in indices],
        "face_ids": [int(rows[index].get("face_id", -1)) for index in indices],
        "trial_model": _rel(trial_model),
        "subset_plan": _rel(subset_plan),
        "audit_json": _rel(audit),
        "gate_json": _rel(_as_path(args.output_root) / scene / "gates" / trial_id / scene / "multifold_trainval_gate.json"),
        "materialize_mode": materialize_mode,
        "accepted_by_strict_gate": bool(gate.get("accepted", False)),
        "decision_reasons": gate.get("decision_reasons", []),
        "objective": float(_objective(gate, args)),
        "trainval_delta_mean": delta,
        "topology": _topology(audit),
    }


def _candidate_groups(
    *,
    row_count: int,
    batch_size: int,
    search_mode: str,
    max_trials: int,
) -> list[list[int]]:
    batch_size = max(1, int(batch_size))
    groups: list[list[int]] = []
    if search_mode in {"greedy_batches", "standalone_batches"}:
        for start in range(0, row_count, batch_size):
            groups.append(list(range(start, min(start + batch_size, row_count))))
    elif search_mode == "sliding_windows":
        if batch_size > row_count:
            groups.append(list(range(row_count)))
        else:
            for start in range(0, row_count - batch_size + 1):
                groups.append(list(range(start, start + batch_size)))
    elif search_mode == "all_pairs":
        if batch_size != 2:
            raise ValueError("--search_mode all_pairs requires --batch_size 2")
        groups = [list(pair) for pair in itertools.combinations(range(row_count), 2)]
    else:
        raise ValueError(f"unknown search mode: {search_mode}")
    if int(max_trials) > 0:
        groups = groups[: int(max_trials)]
    return groups


def _write_report(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    out_root = _as_path(args.output_root) / str(args.scene)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "render_calibrated_search.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Render-Calibrated Candidate Search",
        "",
        f"- scene: `{payload['scene']}`",
        f"- candidate label: `{payload['candidate_label']}`",
        f"- materialize mode: `{payload.get('materialize_mode', 'subdivision')}`",
        f"- strict accepted final subset: `{payload['accepted']}`",
        f"- selected faces: `{len(payload['accepted_indices'])}`",
        f"- best objective: `{payload['best_objective']:.9f}`",
        f"- candidate plan: `{payload['candidate_plan']}`",
        f"- selection uses test: `false`",
        "",
        "| step | trial | action | strict pass | objective | dPSNR | dSSIM | dLPIPS | faces | gate reasons | action reasons |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["events"]:
        delta = row.get("trainval_delta_mean", {})
        lines.append(
            f"| {int(row['step'])} | {row['trial_id']} | {row['action']} | "
            f"{str(row['accepted_by_strict_gate']).lower()} | {float(row['objective']):+.9f} | "
            f"{float(delta.get('PSNR', math.nan)):+.9f} | {float(delta.get('SSIM', math.nan)):+.9f} | "
            f"{float(delta.get('LPIPS', math.nan)):+.9f} | {len(row.get('indices', []))} | "
            f"{', '.join(str(x) for x in row.get('decision_reasons', [])) or 'pass'} | "
            f"{', '.join(str(x) for x in row.get('action_reasons', [])) or 'n/a'} |"
        )
    (out_root / "render_calibrated_search.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--phasej_model", required=True)
    parser.add_argument("--evidence_dir", required=True)
    parser.add_argument("--candidate_plan", required=True)
    parser.add_argument("--output_root", default="outputs/carnet/meshsplatopt/ecsr_phase_s/render_calibrated_search")
    parser.add_argument("--dataset_root", default="/data/peilincai/mesh_datasets/mipnerf360")
    parser.add_argument("--iteration", type=int, default=26000)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--outdoor_images", default="images_4")
    parser.add_argument("--indoor_images", default="images_2")
    parser.add_argument("--skip_failed_views", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse_trial_gate_cache",
        action="store_true",
        help="Reuse copied per-trial gate JSON if it already exists. Defaults off to avoid stale threshold/offset caches.",
    )
    parser.add_argument("--feature_mode", choices=("dc", "sh1"), default="sh1")
    parser.add_argument("--materialize_mode", choices=("subdivision", "vertex_delta"), default="subdivision")
    parser.add_argument("--min_effective_mean_relative_gain", type=float, default=-1.0e30)
    parser.add_argument("--min_effective_min_relative_gain", type=float, default=-1.0e30)
    parser.add_argument("--min_effective_delta_abs_mean", type=float, default=0.0)
    parser.add_argument("--min_materialized_attribute_delta", type=float, default=1.0e-9)
    parser.add_argument("--vertex_delta_min_incident_support_fraction", type=float, default=0.0)
    parser.add_argument("--vertex_delta_max_incident_faces", type=int, default=0)
    parser.add_argument("--allow_no_effect_accept", action="store_true")
    parser.add_argument("--candidate_label", default="rendercalib_greedy")
    parser.add_argument("--max_candidates", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--search_mode",
        choices=("greedy_batches", "standalone_batches", "sliding_windows", "all_pairs"),
        default="greedy_batches",
        help=(
            "Candidate subset schedule. greedy_batches preserves the original incremental "
            "greedy behavior; the standalone modes evaluate each candidate group independently "
            "and keep the strict-passing group with the best train-val objective."
        ),
    )
    parser.add_argument("--max_trials", type=int, default=0, help="Optional cap on scheduled trial groups; 0 means no cap.")
    parser.add_argument("--objective", choices=("lpips_reduction", "balanced", "psnr_lpips"), default="lpips_reduction")
    parser.add_argument("--min_objective_gain", type=float, default=0.0)
    parser.add_argument("--candidate_base_method_prefix", default="ours_26000_rendercalib")
    parser.add_argument("--phasej_trainval_method_prefix", default="ours_26000_phasej_rendercalib_multifold")
    parser.add_argument("--candidate_trainval_method_prefix", default="ours_26000_rendercalib_multifold")
    parser.add_argument("--offsets", default="0,1,2,3")
    parser.add_argument("--policy_holdout_fraction", type=float, default=0.25)
    parser.add_argument("--calib_sampler", choices=("stride_first", "uniform"), default="uniform")
    parser.add_argument("--calib_max_views", type=int, default=32)
    parser.add_argument("--calib_stride", type=int, default=1)
    parser.add_argument("--alpha_feature_mode", choices=("confidence_magnitude", "confidence_magnitude_edge"), default="confidence_magnitude_edge")
    parser.add_argument("--alpha_default", type=float, default=0.0)
    parser.add_argument("--gate_min_psnr_gain", type=float, default=0.0)
    parser.add_argument("--gate_max_ssim_regression", type=float, default=5e-5)
    parser.add_argument("--gate_max_lpips_regression", type=float, default=1.5e-4)
    parser.add_argument("--gate_min_balanced_delta", type=float, default=-1.0e9)
    parser.add_argument("--ssim_weight", type=float, default=20.0)
    parser.add_argument("--lpips_weight", type=float, default=20.0)
    parser.add_argument("--wandb_project", default="mesh-splatting-ecsr")
    parser.add_argument("--wandb_group", default="phase_s_render_calibrated_search")
    parser.add_argument("--wandb_name", default="phase_s_rendercalib")
    args = parser.parse_args()

    plan_path = _as_path(args.candidate_plan)
    plan = _read_json(plan_path)
    rows = _candidate_rows(plan)
    if not rows:
        raise RuntimeError(f"candidate plan has no usable rows: {plan_path}")
    rows = rows[: max(0, int(args.max_candidates))]
    if not rows:
        raise RuntimeError("--max_candidates removed all rows")

    log_path = _as_path(args.output_root) / str(args.scene) / "render_calibrated_search.log"
    _render_maps(args, scene=str(args.scene), model=_as_path(args.phasej_model), method_name=BASE_METHOD, log_path=log_path)

    accepted_indices: list[int] = []
    best_objective = 0.0
    events: list[dict[str, Any]] = []
    groups = _candidate_groups(
        row_count=len(rows),
        batch_size=int(args.batch_size),
        search_mode=str(args.search_mode),
        max_trials=int(args.max_trials),
    )
    for step, batch in enumerate(groups):
        if str(args.search_mode) == "greedy_batches":
            trial_indices = accepted_indices + batch
        else:
            trial_indices = batch
        batch_label = "-".join(str(int(rows[i].get("face_id", -1))) for i in batch)
        trial_id = f"s{step:03d}_add{'-'.join(str(i) for i in batch)}_{batch_label}"
        row = _trial(
            args,
            plan_path=plan_path,
            plan=plan,
            rows=rows,
            indices=trial_indices,
            trial_id=trial_id,
            log_path=log_path,
        )
        row["step"] = int(step)
        improves = bool(float(row["objective"]) > float(best_objective) + float(args.min_objective_gain))
        if bool(row["accepted_by_strict_gate"]) and improves:
            accepted_indices = trial_indices
            best_objective = float(row["objective"])
            row["action"] = "accept"
            row["action_reasons"] = ["strict_gate_passed", "objective_improved"]
        else:
            row["action"] = "reject"
            action_reasons: list[str] = []
            if not bool(row["accepted_by_strict_gate"]):
                action_reasons.append("strict_gate_rejected")
            if bool(row["accepted_by_strict_gate"]) and not improves:
                action_reasons.append(
                    f"objective_gain_below_{float(args.min_objective_gain):g}"
                )
            row["action_reasons"] = action_reasons
        row["accepted_indices_after"] = [int(index) for index in accepted_indices]
        events.append(row)
        _write_report(
            args,
            {
                "scene": str(args.scene),
                "candidate_label": str(args.candidate_label),
                "candidate_plan": _rel(plan_path),
                "materialize_mode": str(plan.get("materialize_mode", args.materialize_mode)),
                "accepted": bool(accepted_indices),
                "accepted_indices": [int(index) for index in accepted_indices],
                "accepted_face_ids": [int(rows[index].get("face_id", -1)) for index in accepted_indices],
                "best_objective": float(best_objective),
                "selection_uses_test": False,
                "search_mode": str(args.search_mode),
                "events": events,
            },
        )

    final_plan = _as_path(args.output_root) / str(args.scene) / "accepted_candidate_plan.json"
    if accepted_indices:
        _write_subset_plan(
            source_plan=plan_path,
            plan=plan,
            rows=rows,
            indices=accepted_indices,
            materialize_mode=str(plan.get("materialize_mode", args.materialize_mode)),
            output=final_plan,
        )

    payload = {
        "scene": str(args.scene),
        "candidate_label": str(args.candidate_label),
        "candidate_plan": _rel(plan_path),
        "materialize_mode": str(plan.get("materialize_mode", args.materialize_mode)),
        "accepted": bool(accepted_indices),
        "accepted_indices": [int(index) for index in accepted_indices],
        "accepted_face_ids": [int(rows[index].get("face_id", -1)) for index in accepted_indices],
        "accepted_candidate_plan": _rel(final_plan) if accepted_indices else "",
        "best_objective": float(best_objective),
        "selection_uses_test": False,
        "objective": str(args.objective),
        "search_mode": str(args.search_mode),
        "scheduled_trial_count": int(len(groups)),
        "events": events,
    }
    _write_report(args, payload)
    print(json.dumps({"accepted": bool(accepted_indices), "accepted_faces": len(accepted_indices), "output_root": _rel(_as_path(args.output_root) / str(args.scene))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

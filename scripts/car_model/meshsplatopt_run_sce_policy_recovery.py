#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sce_recovery_policy import (  # noqa: E402
    SCEPolicyConfig,
    decide_sce_policy_action,
    load_json_mapping,
    select_early_stop_candidate,
)


def _metrics_payload(payload: dict) -> dict:
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return dict(payload["metrics"])
    return dict(payload)


def _wrapper_command(args: argparse.Namespace, cfg: SCEPolicyConfig, *, activate_rollback: bool) -> list[str]:
    final_iteration = int(args.final_iteration)
    cmd = [
        args.python,
        "scripts/car_model/meshsplatopt_run_strict_compact_recovery.py",
        "--source_path",
        args.source_path,
        "--images",
        args.images,
        "--resolution",
        str(args.resolution),
        "--output_path",
        args.output_path,
        "--load_iteration",
        str(args.load_iteration),
        "--final_iteration",
        str(final_iteration),
        "--preset",
        "compact_sparse_low_lambda",
        "--sparse_lambda",
        str(cfg.sparse_lambda),
        "--sparse_start_iter",
        str(args.load_iteration),
        "--sparse_warmup_iters",
        str(min(50, max(1, final_iteration - int(args.load_iteration)))),
        "--sparse_min_matches",
        "16",
        "--sparse_sample_mode",
        "mixed_low_error",
        "--sparse_fraction",
        "0.7",
        "--checkpoint_render_normal_anchor_lambda",
        str(cfg.render_normal_anchor_lambda),
        "--checkpoint_render_depth_anchor_lambda",
        str(cfg.render_depth_anchor_lambda),
        "--checkpoint_render_geometry_anchor_start_iter",
        str(args.load_iteration),
        "--checkpoint_render_geometry_anchor_warmup_iters",
        str(min(50, max(1, final_iteration - int(args.load_iteration)))),
        "--lr_triangles_points_init",
        str(cfg.lr_triangles_points_init),
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        args.wandb_group,
        "--wandb_name",
        args.wandb_name,
        "--train_seed",
        str(args.train_seed),
        "--contract_out_dir",
        str(Path(args.output_path).parent / "sce_policy_contract"),
    ]
    if activate_rollback:
        cmd.extend(
            [
                "--sparse_depth_parent_rollback_cache",
                args.sentinel_cache,
                "--sparse_depth_parent_rollback_lambda",
                str(cfg.rollback_lambda_base),
                "--sparse_depth_parent_rollback_start_iter",
                str(args.load_iteration),
                "--sparse_depth_parent_rollback_warmup_iters",
                str(min(50, max(1, final_iteration - int(args.load_iteration)))),
                "--sparse_depth_parent_rollback_max_points_per_view",
                str(args.rollback_max_points_per_view),
                "--sparse_depth_parent_rollback_loss_space",
                cfg.rollback_loss_space,
                "--sparse_depth_parent_rollback_combined_mae_beta",
                str(cfg.rollback_combined_mae_beta),
                "--sparse_depth_parent_rollback_aggregation",
                cfg.rollback_aggregation,
                "--sparse_depth_parent_rollback_cvar_fraction",
                str(cfg.rollback_cvar_fraction),
                "--sparse_depth_parent_rollback_cvar_min_points",
                str(cfg.rollback_cvar_min_points),
                "--sparse_depth_parent_rollback_pixel_radius",
                str(cfg.rollback_pixel_radius),
                "--sparse_depth_parent_rollback_patch_reduce",
                cfg.rollback_patch_reduce,
                "--sparse_depth_parent_rollback_cluster_balance",
            ]
        )
        if bool(cfg.rollback_regressed_only):
            cmd.append("--sparse_depth_parent_rollback_regressed_only")
        if int(cfg.rollback_cluster_top_k) > 0:
            cmd.extend(["--sparse_depth_parent_rollback_cluster_top_k", str(int(cfg.rollback_cluster_top_k))])
    if bool(args.execute):
        cmd.append("--execute")
    return cmd


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.policy_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SCEPolicyConfig(
        policy_name=args.policy,
        visual_probe_iters=int(args.visual_probe_iters),
        recovery_phase_iters=int(args.recovery_phase_iters),
        rollback_lambda_base=float(args.rollback_lambda_base),
        rollback_loss_space=str(args.rollback_loss_space),
        rollback_combined_mae_beta=float(args.rollback_combined_mae_beta),
        rollback_cluster_top_k=int(args.rollback_cluster_top_k),
        rollback_regressed_only=bool(args.rollback_regressed_only),
        rollback_aggregation=str(args.rollback_aggregation),
        rollback_cvar_fraction=float(args.rollback_cvar_fraction),
        rollback_cvar_min_points=int(args.rollback_cvar_min_points),
        rollback_pixel_radius=int(args.rollback_pixel_radius),
        rollback_patch_reduce=str(args.rollback_patch_reduce),
        sparse_lambda=float(args.sparse_lambda),
        render_normal_anchor_lambda=float(args.render_normal_anchor_lambda),
        render_depth_anchor_lambda=float(args.render_depth_anchor_lambda),
        lr_triangles_points_init=float(args.lr_triangles_points_init),
        parent_tolerance=float(args.parent_tolerance),
        require_sentinel_gate_for_recovery=bool(args.require_sentinel_gate_for_recovery),
        require_measured_candidate_for_recovery=bool(args.require_measured_candidate_for_recovery),
        max_psnr_drop=float(args.max_psnr_drop),
        max_ssim_drop=float(args.max_ssim_drop),
        max_lpips_increase=float(args.max_lpips_increase),
        min_render_score_delta=float(args.min_render_score_delta),
        require_parent_pareto_for_acceptance=bool(args.require_parent_pareto_for_acceptance),
    )
    gate = load_json_mapping(args.sentinel_gate_json) if args.sentinel_gate_json else None
    if gate is not None and "gate" in gate:
        gate = gate["gate"]
    parent_metrics = _metrics_payload(load_json_mapping(args.parent_metrics_json)) if args.parent_metrics_json else {}
    candidate_metrics = _metrics_payload(load_json_mapping(args.candidate_metrics_json)) if args.candidate_metrics_json else {}
    decision = decide_sce_policy_action(
        sentinel_gate=gate,
        cfg=cfg,
        candidate_metrics=candidate_metrics or None,
        parent_metrics=parent_metrics or None,
    )
    history = load_json_mapping(args.candidate_history_json) if args.candidate_history_json else []
    if isinstance(history, dict):
        history = history.get("history", [])
    early = select_early_stop_candidate(history, parent_metrics=parent_metrics, cfg=cfg) if parent_metrics and history else {}
    cmd = _wrapper_command(args, cfg, activate_rollback=bool(decision["activate_rollback"]))
    (out_dir / "sce_policy_decision.json").write_text(
        json.dumps({"decision": decision, "early_stop": early, "config": asdict(cfg)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "exact_train_command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    (out_dir / "policy_report.md").write_text(
        "# SCE Policy Recovery Report\n\n"
        f"- policy: `{cfg.policy_name}`\n"
        f"- action: `{decision['action']}`\n"
        f"- activate_rollback: `{decision['activate_rollback']}`\n"
        f"- execute_recovery: `{decision.get('execute_recovery', True)}`\n"
        f"- reason: `{decision['reason']}`\n"
        f"- command: `{shlex.join(cmd)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "early_stop": early, "command": cmd}, indent=2, sort_keys=True))
    if bool(args.execute) and bool(decision.get("execute_recovery", True)):
        subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scene-agnostic SCE policy recovery runner.")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--load_iteration", type=int, required=True)
    parser.add_argument("--final_iteration", type=int, required=True)
    parser.add_argument("--sentinel_cache", required=True)
    parser.add_argument("--parent_model_path", default="")
    parser.add_argument("--policy", default="sce_v1")
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--sentinel_gate_json", default="")
    parser.add_argument("--parent_metrics_json", default="")
    parser.add_argument("--candidate_metrics_json", default="")
    parser.add_argument("--candidate_history_json", default="")
    parser.add_argument("--visual_probe_iters", type=int, default=500)
    parser.add_argument("--recovery_phase_iters", type=int, default=500)
    parser.add_argument("--rollback_lambda_base", type=float, default=1.0)
    parser.add_argument("--rollback_loss_space", choices=("absrel", "mae", "combined"), default="absrel")
    parser.add_argument("--rollback_combined_mae_beta", type=float, default=1.0)
    parser.add_argument("--rollback_cluster_top_k", type=int, default=0)
    parser.add_argument("--rollback_regressed_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollback_aggregation", choices=("mean", "cvar", "cluster_cvar"), default="mean")
    parser.add_argument("--rollback_cvar_fraction", type=float, default=0.2)
    parser.add_argument("--rollback_cvar_min_points", type=int, default=16)
    parser.add_argument("--rollback_pixel_radius", type=int, default=0)
    parser.add_argument("--rollback_patch_reduce", choices=("center", "max_violation", "mean_violation"), default="center")
    parser.add_argument("--rollback_max_points_per_view", type=int, default=2000)
    parser.add_argument("--sparse_lambda", type=float, default=0.003)
    parser.add_argument("--render_normal_anchor_lambda", type=float, default=0.01)
    parser.add_argument("--render_depth_anchor_lambda", type=float, default=0.0)
    parser.add_argument("--lr_triangles_points_init", type=float, default=0.015)
    parser.add_argument("--parent_tolerance", type=float, default=0.0)
    parser.add_argument("--require_sentinel_gate_for_recovery", action="store_true")
    parser.add_argument("--require_measured_candidate_for_recovery", action="store_true")
    parser.add_argument("--max_psnr_drop", type=float, default=0.0)
    parser.add_argument("--max_ssim_drop", type=float, default=0.0)
    parser.add_argument("--max_lpips_increase", type=float, default=0.0)
    parser.add_argument("--min_render_score_delta", type=float, default=0.0)
    parser.add_argument("--require_parent_pareto_for_acceptance", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet_meshprior")
    parser.add_argument("--wandb_group", default="finalSCE7_policy_recovery")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--policy_out_dir", required=True)
    parser.add_argument("--python", default="/home/peilincai/micromamba/envs/mesh_splatting/bin/python")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

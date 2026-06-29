#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)

DEFAULT_PHASEJ_ROOT = (
    ROOT / "outputs/carnet/meshsplatopt/ecsr_phase_f/policy_val_compaction_ladder_v2_envfix"
)
DEFAULT_EVIDENCE_ROOT = Path("/dev/shm/peilincai_spcarnet_vnext_full9_inputs_20260626")
DEFAULT_V39_EVIDENCE_ROOT = ROOT / "outputs/carnet/meshsplatopt/ecsr_phase_v39_multiscene_ssimaware"
DEFAULT_OUTPUT_ROOT = Path("/tmp/peilincai_spcarnet_v238_full9")


def run_cmd(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> dict[str, Any]:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[cmd] {printable}", flush=True)
    if dry_run:
        return {"command": printable, "returncode": None, "dry_run": True}
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    return {"command": printable, "returncode": int(proc.returncode), "dry_run": False}


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def evidence_view_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    views_dir = path / "views"
    root = views_dir if views_dir.is_dir() else path
    return len(list(root.glob("*.npz")))


def require_evidence_dir(path: Path, label: str) -> Path:
    require_dir(path, label)
    count = evidence_view_count(path)
    if count <= 0:
        raise FileNotFoundError(f"{label} has no npz evidence views: {path}")
    return path


def first_existing_evidence(candidates: list[Path], label: str) -> Path:
    checked: list[str] = []
    for path in candidates:
        checked.append(f"{path} views={evidence_view_count(path)}")
        if evidence_view_count(path) > 0:
            return path
    raise FileNotFoundError(f"no usable {label}; checked: " + "; ".join(checked))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def symlink_dir(src: Path, dst: Path) -> None:
    require_dir(src, "symlink source")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(src, dst, target_is_directory=True)


def check_return(stage: dict[str, Any], *, name: str) -> None:
    rc = stage.get("returncode")
    if rc not in (0, None):
        raise RuntimeError(f"{name} failed with returncode={rc}: {stage.get('command')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed v238 surface-texture Phase-J distillation pipeline on one scene: "
            "native-resolution evidence rebase, no-GT apply, exact Phase-J comparison."
        )
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--phasej_root", type=Path, default=DEFAULT_PHASEJ_ROOT)
    parser.add_argument("--evidence_root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--v39_evidence_root", type=Path, default=DEFAULT_V39_EVIDENCE_ROOT)
    parser.add_argument("--fit_evidence_source_dir", type=Path, default=None)
    parser.add_argument("--target_evidence_source_dir", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phasej_train_method", default="ours_26000_phasej_trainval_gate")
    parser.add_argument("--phasej_test_method", default="ours_26000_phasej_guarded_adaptedge_ela")
    parser.add_argument("--parent_train_method", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--parent_test_method", default="ours_26000_phasef_extra_compact_base")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--steps", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=238)
    parser.add_argument(
        "--variant",
        choices=[
            "v238_surface_texture_unet",
            "v239_residual_debt",
            "v240_phasej_to_phasef_distill",
            "v241_target_dense_teacher",
        ],
        default="v238_surface_texture_unet",
        help=(
            "v238 preserves the frozen Phase-J-plus policy; v239 adds train-fit residual-debt masking/no-op "
            "stabilization; v240 switches to true Phase-J teacher -> Phase-F parent surface distillation; "
            "v241 keeps that teacher target but expands target-visible capacity and uses a bounded dense fallback."
        ),
    )
    parser.add_argument("--residual_debt_quantile", type=float, default=0.70)
    parser.add_argument("--residual_debt_min_l1", type=float, default=1.0 / 255.0)
    parser.add_argument("--residual_debt_dilate", type=int, default=1)
    parser.add_argument("--residual_debt_noop_weight", type=float, default=0.06)
    parser.add_argument(
        "--v240_gt_assist",
        action="store_true",
        help="For v240 only: add train-fit GT losses as a quality-assist ablation. Default v240 is teacher-only.",
    )
    parser.add_argument("--force_rebase", action="store_true")
    parser.add_argument("--skip_rebase", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene = str(args.scene)
    phasej_scene_root = Path(args.phasej_root) / scene / "ratio_0200" / "compact_model"
    phasej_train = require_dir(phasej_scene_root / "train" / str(args.phasej_train_method), "Phase-J train method")
    phasej_test = require_dir(phasej_scene_root / "test" / str(args.phasej_test_method), "Phase-J test method")
    phasej_train_renders = require_dir(phasej_train / "renders", "Phase-J train renders")
    phasej_train_gt = require_dir(phasej_train / "gt", "Phase-J train gt")
    phasej_test_renders = require_dir(phasej_test / "renders", "Phase-J test renders")
    phasej_test_gt = require_dir(phasej_test / "gt", "Phase-J test gt")
    parent_train = None
    parent_test = None
    parent_train_renders = None
    parent_test_renders = None
    distill_variants = {"v240_phasej_to_phasef_distill", "v241_target_dense_teacher"}
    if str(args.variant) in distill_variants:
        parent_train = require_dir(
            phasej_scene_root / "train" / str(args.parent_train_method),
            "distillation parent train method",
        )
        parent_test = require_dir(
            phasej_scene_root / "test" / str(args.parent_test_method),
            "distillation parent test method",
        )
        parent_train_renders = require_dir(parent_train / "renders", "distillation parent train renders")
        parent_test_renders = require_dir(parent_test / "renders", "distillation parent test renders")

    evidence_scene_root = Path(args.evidence_root) / scene
    v39_root = Path(args.v39_evidence_root)
    fit_candidates = [
        evidence_scene_root / "fit_evidence",
        v39_root / "train_visible_bary_images2" / f"{scene}_teacher_surface_evidence_phasej_trainval_alpha1",
        v39_root / "train_visible_bary_images2" / f"{scene}_teacher_surface_evidence_phasej_trainval_resize_alpha1",
        v39_root / "train_visible_bary_images2" / scene,
    ]
    target_candidates = [
        evidence_scene_root / "target_evidence",
        evidence_scene_root / "target_visible_bary_base" / scene,
        v39_root / "target_visible_bary_images2" / scene,
    ]
    fit_evidence = (
        require_evidence_dir(Path(args.fit_evidence_source_dir), "explicit fit evidence")
        if args.fit_evidence_source_dir is not None
        else first_existing_evidence(fit_candidates, "fit evidence")
    )
    target_evidence = (
        require_evidence_dir(Path(args.target_evidence_source_dir), "explicit target evidence")
        if args.target_evidence_source_dir is not None
        else first_existing_evidence(target_candidates, "target visible barycentric evidence")
    )

    variant = str(args.variant)
    if variant == "v238_surface_texture_unet":
        variant_prefix = "v238_surface_texture_unet"
    elif variant == "v239_residual_debt":
        variant_prefix = "v239_residual_debt_surface_texture_unet"
    elif variant == "v241_target_dense_teacher":
        variant_prefix = "v241_target_dense_phasej_to_phasef_surface_texture_unet_teacheronly"
    else:
        variant_prefix = (
            "v240_phasej_to_phasef_surface_texture_unet_gtassist"
            if bool(args.v240_gt_assist)
            else "v240_phasej_to_phasef_surface_texture_unet_teacheronly"
        )
    scene_root = Path(args.output_root) / scene
    run_dir = scene_root / f"{variant_prefix}_native1256"
    if variant in distill_variants:
        train_evidence = scene_root / "teacher_surface_evidence_phasej_to_phasef_native1256_surfacegeom"
        target_no_gt = scene_root / "target_evidence_no_gt_phasef_native1256_surfacegeom"
        no_gt_audit = scene_root / "target_phasef_native1256_surfacegeom_no_gt_verify.json"
    else:
        train_evidence = scene_root / "teacher_surface_evidence_phasej_native1256_surfacegeom"
        target_no_gt = scene_root / "target_evidence_no_gt_phasej_native1256_surfacegeom"
        no_gt_audit = scene_root / "target_phasej_native1256_surfacegeom_no_gt_verify.json"
    method = f"ours_26000_{variant_prefix}_native1256_{scene}"
    exact_model = run_dir / f"{scene}_exact_target_apply"
    phasej_ref_method = "phasej_reference_native1256"
    if variant == "v238_surface_texture_unet":
        result_tag = "v238"
    elif variant == "v239_residual_debt":
        result_tag = "v239"
    elif variant == "v241_target_dense_teacher":
        result_tag = "v241dense"
    else:
        result_tag = "v240gt" if bool(args.v240_gt_assist) else "v240teacher"
    exact_results = scene_root / f"{result_tag}_{scene}_native1256_exact_results.json"
    exact_per_view = scene_root / f"{result_tag}_{scene}_native1256_exact_per_view.json"
    summary_path = scene_root / f"{result_tag}_{scene}_fixed_policy_summary.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["TMPDIR"] = "/tmp"
    env["WANDB_MODE"] = "offline"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = f"/tmp/peilincai_pycache_{result_tag}_full9_{scene}"
    env.setdefault("WANDB_DIR", str(run_dir / "wandb"))

    stages: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": f"spcarnet_{result_tag}_fixed_policy_scene_v1",
        "variant": variant,
        "scene": scene,
        "phasej_train_dir": phasej_train,
        "phasej_test_dir": phasej_test,
        "distillation_parent_train_dir": parent_train,
        "distillation_parent_test_dir": parent_test,
        "fit_evidence_dir": fit_evidence,
        "fit_evidence_view_count": evidence_view_count(fit_evidence),
        "target_evidence_source_dir": target_evidence,
        "target_evidence_source_view_count": evidence_view_count(target_evidence),
        "fit_evidence_candidates": fit_candidates,
        "target_evidence_candidates": target_candidates,
        "train_evidence_dir": train_evidence,
        "target_no_gt_evidence_dir": target_no_gt,
        "run_dir": run_dir,
        "method": method,
        "phasej_reference_method": phasej_ref_method,
        "gpu": str(args.gpu),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "wandb_mode": env.get("WANDB_MODE", ""),
        "stages": stages,
    }

    if not args.skip_rebase and not args.eval_only:
        force = ["--force"] if args.force_rebase or not train_evidence.exists() else []
        if not train_evidence.exists() or force:
            if variant in distill_variants:
                stage = run_cmd(
                    [
                        str(PYTHON),
                        "scripts/car_model/ecsr_build_teacher_surface_evidence_cache.py",
                        "--base_evidence_dir",
                        str(fit_evidence),
                        "--teacher_render_dir",
                        str(phasej_train_renders),
                        "--parent_render_dir",
                        str(parent_train_renders),
                        "--out_dir",
                        str(train_evidence),
                        "--allow_resize",
                        "--copy_mode",
                        "auto_link",
                        "--rewrite_rgb_render_to_parent",
                        "--selection_mode",
                        "better_masked_residual",
                        "--top_support_min_alpha",
                        "0.03",
                        "--top_support_limit",
                        "8192",
                        *force,
                    ],
                    env=env,
                    dry_run=bool(args.dry_run),
                )
                stage["stage"] = "build_phasej_to_phasef_train_teacher_surface_evidence"
                stage["summary_path"] = train_evidence / "teacher_surface_evidence_summary.json"
            else:
                train_audit = scene_root / "teacher_surface_evidence_phasej_native1256_surfacegeom_audit.json"
                stage = run_cmd(
                    [
                        str(PYTHON),
                        "scripts/car_model/ecsr_rebase_evidence_rgb_render_from_renders.py",
                        "--input_evidence_dir",
                        str(fit_evidence),
                        "--render_dir",
                        str(phasej_train_renders),
                        "--gt_render_dir",
                        str(phasej_train_gt),
                        "--output_evidence_dir",
                        str(train_evidence),
                        "--audit_path",
                        str(train_audit),
                        "--allow_resize",
                        "--match_render_resolution",
                        "--minimal_fields",
                        "--recompute_residual_from_gt",
                        *force,
                    ],
                    env=env,
                    dry_run=bool(args.dry_run),
                )
                stage["stage"] = "rebase_train_teacher_surface_evidence"
                stage["audit_path"] = train_audit
            stages.append(stage)
            check_return(stage, name="rebase train evidence")

        force = ["--force"] if args.force_rebase or not target_no_gt.exists() else []
        if not target_no_gt.exists() or force:
            target_audit = (
                scene_root / "target_evidence_no_gt_phasef_native1256_surfacegeom_audit.json"
                if variant in distill_variants
                else scene_root / "target_evidence_no_gt_phasej_native1256_surfacegeom_audit.json"
            )
            target_parent_renders = (
                parent_test_renders if variant in distill_variants else phasej_test_renders
            )
            stage = run_cmd(
                [
                    str(PYTHON),
                    "scripts/car_model/ecsr_rebase_evidence_rgb_render_from_renders.py",
                    "--input_evidence_dir",
                    str(target_evidence),
                    "--render_dir",
                    str(target_parent_renders),
                    "--output_evidence_dir",
                    str(target_no_gt),
                    "--audit_path",
                    str(target_audit),
                    "--allow_resize",
                    "--match_render_resolution",
                    "--minimal_fields",
                    "--strip_target_gt_and_residuals",
                    *force,
                ],
                env=env,
                dry_run=bool(args.dry_run),
            )
            stage["stage"] = "rebase_target_no_gt_surface_evidence"
            stage["audit_path"] = target_audit
            stages.append(stage)
            check_return(stage, name="rebase target no-GT evidence")

    if not args.dry_run:
        verify_stage = run_cmd(
            [
                str(PYTHON),
                "scripts/car_model/ecsr_verify_target_evidence_no_gt.py",
                "--target_evidence_dir",
                str(target_no_gt),
                "--audit_path",
                str(no_gt_audit),
            ],
            env=env,
            dry_run=False,
        )
        verify_stage["stage"] = "verify_target_no_gt"
        verify_stage["audit_path"] = no_gt_audit
        stages.append(verify_stage)
        check_return(verify_stage, name="target no-GT verification")
        summary["target_no_gt_audit"] = read_json(no_gt_audit)

    if not args.skip_train and not args.eval_only:
        teacher_l1_weight = "0.06"
        teacher_ssim_weight = "0.08"
        teacher_lpips_weight = "0.10"
        teacher_grad_weight = "0.03"
        teacher_highfreq_weight = "0.04"
        gt_l1_weight = "0.78"
        gt_ssim_weight = "0.62"
        gt_lpips_weight = "0.58"
        gt_grad_weight = "0.14"
        gt_highfreq_weight = "0.22"
        delta_l1_weight = "0.00055"
        confidence_bias = "-1.2"
        if variant in distill_variants:
            teacher_l1_weight = "0.82"
            teacher_ssim_weight = "0.26"
            teacher_lpips_weight = "0.16"
            teacher_grad_weight = "0.08"
            teacher_highfreq_weight = "0.10"
            delta_l1_weight = "0.00080"
            confidence_bias = "-1.0"
            if bool(args.v240_gt_assist) and variant == "v240_phasej_to_phasef_distill":
                gt_l1_weight = "0.16"
                gt_ssim_weight = "0.18"
                gt_lpips_weight = "0.12"
                gt_grad_weight = "0.05"
                gt_highfreq_weight = "0.08"
            else:
                gt_l1_weight = "0.0"
                gt_ssim_weight = "0.0"
                gt_lpips_weight = "0.0"
                gt_grad_weight = "0.0"
                gt_highfreq_weight = "0.0"
        surface_face_max_unique = "65536" if variant == "v241_target_dense_teacher" else "8192"
        surface_feature_dim = "12" if variant == "v241_target_dense_teacher" else "8"
        base_channels = "32" if variant == "v241_target_dense_teacher" else "24"
        lowrank_min_bin_support = "2" if variant == "v241_target_dense_teacher" else "8"
        train_cmd = [
            str(PYTHON),
            "scripts/car_model/train_surface_conditioned_residual_unet.py",
            "--fit_evidence_dir",
            str(train_evidence),
            "--target_evidence_dir",
            str(target_no_gt),
            "--surface_target_visible_evidence_dir",
            str(target_no_gt),
            "--residual_rgb_key",
            "teacher_residual_rgb",
            "--policy_val_stride",
            "4",
            "--train_max_side",
            "512",
            "--patch_size",
            "256",
            "--steps",
            str(int(args.steps)),
            "--lr",
            "0.00024",
            "--model_type",
            "surface_texture_unet",
            "--surface_texture_size",
            "8",
            "--surface_feature_dim",
            surface_feature_dim,
            "--surface_face_max_unique",
            surface_face_max_unique,
            "--surface_face_min_alpha",
            "0.03",
            "--surface_face_min_residual_l1",
            "0.0",
            "--enable_surface_support_gate",
            "--lowrank_min_bin_support",
            lowrank_min_bin_support,
            "--base_channels",
            base_channels,
            "--max_delta",
            "0.08",
            "--confidence_mode",
            "sigmoid",
            "--confidence_bias",
            confidence_bias,
            "--confidence_min",
            "0.0",
            "--confidence_max",
            "1.0",
            "--alpha_conditioned_residual",
            "--teacher_l1_weight",
            teacher_l1_weight,
            "--teacher_ssim_weight",
            teacher_ssim_weight,
            "--teacher_lpips_weight",
            teacher_lpips_weight,
            "--teacher_grad_weight",
            teacher_grad_weight,
            "--teacher_highfreq_weight",
            teacher_highfreq_weight,
            "--gt_l1_weight",
            gt_l1_weight,
            "--gt_ssim_weight",
            gt_ssim_weight,
            "--gt_lpips_weight",
            gt_lpips_weight,
            "--gt_grad_weight",
            gt_grad_weight,
            "--gt_highfreq_weight",
            gt_highfreq_weight,
            "--lpips_loss_max_side",
            "224",
            "--grad_loss_max_side",
            "256",
            "--highfreq_loss_max_side",
            "256",
            "--highfreq_loss_levels",
            "3",
            "--delta_l1_weight",
            delta_l1_weight,
            "--alpha_grid",
            "0,0.25,0.5,0.75,1",
            "--policy_select_mode",
            "tail_guard",
            "--policy_tail_fraction",
            "0.35",
            "--policy_min_psnr_gain",
            "-0.25",
            "--policy_min_ssim_gain",
            "-0.01",
            "--policy_min_lpips_gain",
            "-0.02",
            "--policy_cvar_psnr_gain",
            "-0.08",
            "--policy_cvar_ssim_gain",
            "-0.003",
            "--policy_cvar_lpips_gain",
            "-0.006",
            "--method_name",
            method,
            "--scene_name",
            scene,
            "--eval_tile",
            "512",
            "--eval_overlap",
            "32",
            "--ssim_max_side",
            "512",
            "--lpips_max_side",
            "256",
            "--compute_lpips",
            "--skip_policy_val_renders",
            "--output_dir",
            str(run_dir),
            "--artifact_prefix",
            f"{variant_prefix}_native1256_{scene}",
            "--seed",
            str(int(args.seed)),
        ]
        if variant == "v241_target_dense_teacher":
            train_cmd.extend(
                [
                    "--surface_support_gate_floor",
                    "0.18",
                    "--surface_support_unknown_gate_floor",
                    "0.06",
                ]
            )
        if variant == "v239_residual_debt":
            train_cmd.extend(
                [
                    "--residual_debt_mask",
                    "--residual_debt_quantile",
                    str(float(args.residual_debt_quantile)),
                    "--residual_debt_min_l1",
                    str(float(args.residual_debt_min_l1)),
                    "--residual_debt_dilate",
                    str(int(args.residual_debt_dilate)),
                    "--residual_debt_noop_weight",
                    str(float(args.residual_debt_noop_weight)),
                ]
            )
        if not bool(args.disable_wandb):
            train_cmd.extend(
                [
                    "--enable_wandb",
                    "--wandb_run_name",
                    f"{variant_prefix.replace('_', '-')}-native1256-{scene}",
                ]
            )
        train_stage = run_cmd(train_cmd, env=env, dry_run=bool(args.dry_run))
        train_stage["stage"] = "train_and_no_gt_apply"
        stages.append(train_stage)
        check_return(train_stage, name="train/apply")

    if not args.dry_run:
        method_dir = exact_model / "test" / method
        renders_dir = method_dir / "renders"
        if renders_dir.is_dir():
            symlink_dir(phasej_test_gt, method_dir / "gt")
            phasej_ref_dir = exact_model / "test" / phasej_ref_method
            symlink_dir(phasej_test_renders, phasej_ref_dir / "renders")
            symlink_dir(phasej_test_gt, phasej_ref_dir / "gt")
            eval_stage = run_cmd(
                [
                    str(PYTHON),
                    "scripts/car_model/evaluate_render_split_metrics.py",
                    "-m",
                    str(exact_model),
                    "--split",
                    "test",
                    "--methods",
                    method,
                    phasej_ref_method,
                    "--output",
                    str(exact_results),
                    "--per_view_output",
                    str(exact_per_view),
                    "--merge_model_results",
                ],
                env=env,
                dry_run=False,
            )
            eval_stage["stage"] = "exact_eval_vs_phasej"
            eval_stage["results_path"] = exact_results
            eval_stage["per_view_path"] = exact_per_view
            stages.append(eval_stage)
            check_return(eval_stage, name="exact eval")
            metrics = read_json(exact_results)
            summary["exact_results"] = metrics
            if method in metrics and phasej_ref_method in metrics:
                ours = metrics[method]
                ref = metrics[phasej_ref_method]
                summary["delta_vs_phasej"] = {
                    "PSNR": float(ours["PSNR"] - ref["PSNR"]),
                    "SSIM": float(ours["SSIM"] - ref["SSIM"]),
                    "LPIPS": float(ours["LPIPS"] - ref["LPIPS"]),
                }
                summary["beats_phasej_all_axis"] = bool(
                    ours["PSNR"] > ref["PSNR"] and ours["SSIM"] > ref["SSIM"] and ours["LPIPS"] < ref["LPIPS"]
                )
        else:
            summary["exact_eval_skipped_reason"] = f"missing method renders: {renders_dir}"

    write_json(summary_path, summary)
    print(f"[summary] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

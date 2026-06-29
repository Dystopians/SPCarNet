#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    build_lpips_model,
    evidence_views,
    image_lpips_chw,
    image_ssim_chw,
    save_image_chw,
)
from scripts.car_model.train_surface_lowrank_residual_texture import _predict_delta  # noqa: E402
from scripts.car_model.train_surface_uv_residual_texture import _mean, _psnr, _quantiles, _write_json  # noqa: E402


FORBIDDEN_TARGET_APPLY_KEYS = {
    "rgb_gt",
    "residual_rgb",
    "residual_l1",
    "teacher_residual_rgb",
    "teacher_residual_l1",
    "teacher_residual_rgb_raw",
    "teacher_better_mask",
    "teacher_gain_l1",
    "teacher_parent_delta_l1",
}


def _scalar_json_from_npz(z: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in z:
        return {}
    value = z[key]
    if hasattr(value, "item"):
        value = value.item()
    return json.loads(str(value))


def _verify_no_gt(evidence_dir: Path) -> dict[str, Any]:
    paths = evidence_views(Path(evidence_dir))
    bad: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            keys = sorted(str(k) for k in z.files)
        present = sorted(set(keys) & FORBIDDEN_TARGET_APPLY_KEYS)
        if idx < 4:
            samples.append({"path": str(path), "keys": keys})
        if present:
            bad.append({"path": str(path), "forbidden_keys": present})
    return {
        "mode": "strict_lowrank_target_no_gt_preflight",
        "evidence_dir": str(evidence_dir),
        "view_count": int(len(paths)),
        "forbidden_keys": sorted(FORBIDDEN_TARGET_APPLY_KEYS),
        "bad_view_count": int(len(bad)),
        "bad_views": bad[:64],
        "sample_keys": samples,
        "target_gt_visible_to_apply": any("rgb_gt" in set(row["forbidden_keys"]) for row in bad),
        "target_residual_visible_to_apply": any(bool(set(row["forbidden_keys"]) - {"rgb_gt"}) for row in bad),
        "passed": not bad and bool(paths),
    }


def _gain_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    return {
        "mean": _mean(values),
        "positive_view_fraction": float(np.mean(np.asarray(values) > 0.0)) if values else 0.0,
        "quantiles": _quantiles(values),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    m = payload["metrics"]
    lines = [
        "# Low-Rank Surface Residual Texture Apply Audit",
        "",
        f"- accepted: `{payload['accepted']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- target evidence: `{payload['target_evidence_dir']}`",
        f"- no-GT preflight passed: `{payload['no_gt_preflight']['passed']}`",
        f"- selected alpha: `{payload['selected_alpha']}`",
        f"- changed fraction: `{payload['target_apply']['changed_fraction']:.8f}`",
        f"- output renders: `{payload['renders_dir']}`",
        "",
        "## Metrics",
        "",
        "| row | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|",
        f"| parent | {m.get('parent_psnr', 0.0):.6f} | {m.get('parent_ssim', 0.0):.6f} | {m.get('parent_lpips', 0.0):.6f} |",
        f"| candidate | {m.get('candidate_psnr', 0.0):.6f} | {m.get('candidate_ssim', 0.0):.6f} | {m.get('candidate_lpips', 0.0):.6f} |",
        f"| gain | {m.get('psnr_gain', 0.0):+.6f} | {m.get('ssim_gain', 0.0):+.6f} | {m.get('lpips_gain', 0.0):+.6f} |",
        "",
        "## Phase-J Flowers Gate",
        "",
        f"- reference: `{payload['phasej_flowers_reference']}`",
        f"- pass: `{payload['phasej_flowers_gate_pass']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a low-rank UV residual texture checkpoint to no-GT target evidence.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_evidence_dir", required=True)
    parser.add_argument("--eval_gt_evidence_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--grid", type=int, default=0)
    parser.add_argument("--basis_mode", default="")
    parser.add_argument("--min_alpha", type=float, default=-1.0)
    parser.add_argument("--min_bin_count", type=float, default=-1.0)
    parser.add_argument("--max_abs_delta", type=float, default=-1.0)
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--ssim_max_side", type=int, default=512)
    parser.add_argument("--lpips_max_side", type=int, default=256)
    parser.add_argument("--phasej_flowers_psnr", type=float, default=20.304358)
    parser.add_argument("--phasej_flowers_ssim", type=float, default=0.557770)
    parser.add_argument("--phasej_flowers_lpips", type=float, default=0.329222)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="spcarnet-lowrank-target-apply")
    parser.add_argument("--wandb_run_name", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = output_dir / "renders"
    gt_dir = output_dir / "gt"
    renders_dir.mkdir(parents=True, exist_ok=True)

    no_gt = _verify_no_gt(Path(args.target_evidence_dir))
    if not bool(no_gt["passed"]):
        payload = {
            "schema": "spcarnet_lowrank_target_apply_audit_v1",
            "accepted": False,
            "reject_reason": "target_evidence_no_gt_preflight_failed",
            "no_gt_preflight": no_gt,
            "checkpoint": str(args.checkpoint),
            "target_evidence_dir": str(args.target_evidence_dir),
        }
        _write_json(output_dir / "lowrank_target_apply_audit.json", payload)
        raise SystemExit("target evidence no-GT preflight failed")

    with np.load(args.checkpoint, allow_pickle=False) as ckpt:
        ckpt_args = _scalar_json_from_npz(ckpt, "args_json")
        candidate_faces = np.asarray(ckpt["candidate_faces"], dtype=np.int64)
        coeff = np.asarray(ckpt["coeff"], dtype=np.float32)
        counts = np.asarray(ckpt["counts"], dtype=np.float32)

    grid = int(args.grid) if int(args.grid) > 0 else int(ckpt_args.get("grid", 4))
    basis_mode = str(args.basis_mode or ckpt_args.get("basis_mode", "dir_uv_v1"))
    min_alpha = float(args.min_alpha) if float(args.min_alpha) >= 0.0 else float(ckpt_args.get("min_alpha", 0.03))
    min_bin_count = (
        float(args.min_bin_count) if float(args.min_bin_count) >= 0.0 else float(ckpt_args.get("min_bin_count", 3.0))
    )
    max_abs_delta = (
        float(args.max_abs_delta) if float(args.max_abs_delta) >= 0.0 else float(ckpt_args.get("max_abs_delta", 0.25))
    )
    alpha = float(args.alpha)
    if alpha < 0.0:
        alpha = 1.0

    eval_dir = Path(args.eval_gt_evidence_dir) if str(args.eval_gt_evidence_dir) else None
    eval_index = {p.stem: p for p in evidence_views(eval_dir)} if eval_dir is not None else {}
    lpips_model = build_lpips_model() if bool(args.compute_lpips) and eval_index else None
    rows: list[dict[str, Any]] = []
    changed_pixels = 0
    total_pixels = 0
    active_pixels = 0
    target_paths = evidence_views(Path(args.target_evidence_dir))
    for path in tqdm(target_paths, desc="apply low-rank residual target"):
        with np.load(path, allow_pickle=False) as z:
            parent = np.asarray(z["rgb_render"], dtype=np.float32)
            delta, active = _predict_delta(
                z,
                candidate_faces,
                coeff,
                counts,
                residual_l1_key=str(args.residual_l1_key),
                min_l1=0.0,
                min_alpha=float(min_alpha),
                min_bin_count=float(min_bin_count),
                grid=int(grid),
                basis_mode=str(basis_mode),
                max_abs_delta=float(max_abs_delta),
            )
            adapted = np.clip(parent + float(alpha) * delta, 0.0, 1.0)
        changed = np.any(np.abs(adapted - parent) > (0.5 / 255.0), axis=0)
        changed_pixels += int(np.count_nonzero(changed))
        total_pixels += int(changed.size)
        active_pixels += int(np.count_nonzero(active))
        save_image_chw(renders_dir / f"{path.stem}.png", adapted)
        row: dict[str, Any] = {
            "view": path.stem,
            "changed_fraction": float(np.mean(changed)),
            "active_fraction": float(np.mean(active)),
        }
        if path.stem in eval_index:
            with np.load(eval_index[path.stem], allow_pickle=False) as gt_z:
                gt = np.asarray(gt_z["rgb_gt"], dtype=np.float32)
            gt_dir.mkdir(parents=True, exist_ok=True)
            save_image_chw(gt_dir / f"{path.stem}.png", gt)
            p_psnr = _psnr(parent, gt)
            c_psnr = _psnr(adapted, gt)
            p_ssim = image_ssim_chw(parent, gt, int(args.ssim_max_side))
            c_ssim = image_ssim_chw(adapted, gt, int(args.ssim_max_side))
            row.update(
                {
                    "parent_psnr": float(p_psnr),
                    "candidate_psnr": float(c_psnr),
                    "psnr_gain": float(c_psnr - p_psnr),
                    "parent_ssim": float(p_ssim),
                    "candidate_ssim": float(c_ssim),
                    "ssim_gain": float(c_ssim - p_ssim),
                }
            )
            if bool(args.compute_lpips):
                p_lp = image_lpips_chw(parent, gt, int(args.lpips_max_side), lpips_model)
                c_lp = image_lpips_chw(adapted, gt, int(args.lpips_max_side), lpips_model)
                row.update(
                    {
                        "parent_lpips": float(p_lp),
                        "candidate_lpips": float(c_lp),
                        "lpips_gain": float(p_lp - c_lp),
                    }
                )
        rows.append(row)

    metric_rows = [r for r in rows if "candidate_psnr" in r]
    metrics: dict[str, Any] = {"view_count": int(len(metric_rows))}
    if metric_rows:
        metrics.update(
            {
                "parent_psnr": _mean([float(r["parent_psnr"]) for r in metric_rows]),
                "candidate_psnr": _mean([float(r["candidate_psnr"]) for r in metric_rows]),
                "psnr_gain": _mean([float(r["psnr_gain"]) for r in metric_rows]),
                "parent_ssim": _mean([float(r["parent_ssim"]) for r in metric_rows]),
                "candidate_ssim": _mean([float(r["candidate_ssim"]) for r in metric_rows]),
                "ssim_gain": _mean([float(r["ssim_gain"]) for r in metric_rows]),
                "psnr_gain_summary": _gain_summary(metric_rows, "psnr_gain"),
                "ssim_gain_summary": _gain_summary(metric_rows, "ssim_gain"),
            }
        )
        if "candidate_lpips" in metric_rows[0]:
            metrics.update(
                {
                    "parent_lpips": _mean([float(r["parent_lpips"]) for r in metric_rows]),
                    "candidate_lpips": _mean([float(r["candidate_lpips"]) for r in metric_rows]),
                    "lpips_gain": _mean([float(r["lpips_gain"]) for r in metric_rows]),
                    "lpips_gain_summary": _gain_summary(metric_rows, "lpips_gain"),
                }
            )
    gate_ref = {
        "psnr": float(args.phasej_flowers_psnr),
        "ssim": float(args.phasej_flowers_ssim),
        "lpips": float(args.phasej_flowers_lpips),
    }
    gate_pass = bool(
        metric_rows
        and float(metrics.get("candidate_psnr", -math.inf)) > gate_ref["psnr"]
        and float(metrics.get("candidate_ssim", -math.inf)) > gate_ref["ssim"]
        and float(metrics.get("candidate_lpips", math.inf)) < gate_ref["lpips"]
    )
    wandb_run = None
    if bool(args.enable_wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project),
                name=str(args.wandb_run_name or output_dir.name),
                config=vars(args),
                dir=str(output_dir),
            )
            wandb_run.log(
                {
                    "apply/changed_fraction": float(changed_pixels / max(1, total_pixels)),
                    "apply/active_fraction": float(active_pixels / max(1, total_pixels)),
                    "metrics/candidate_psnr": float(metrics.get("candidate_psnr", 0.0)),
                    "metrics/candidate_ssim": float(metrics.get("candidate_ssim", 0.0)),
                    "metrics/candidate_lpips": float(metrics.get("candidate_lpips", 0.0)),
                    "metrics/psnr_gain": float(metrics.get("psnr_gain", 0.0)),
                    "metrics/ssim_gain": float(metrics.get("ssim_gain", 0.0)),
                    "metrics/lpips_gain": float(metrics.get("lpips_gain", 0.0)),
                    "gate/phasej_flowers_pass": int(gate_pass),
                }
            )
            wandb_run.finish()
        except Exception as exc:  # pragma: no cover
            print(f"[wandb] disabled after init failure: {type(exc).__name__}: {exc}", flush=True)
    payload: dict[str, Any] = {
        "schema": "spcarnet_lowrank_target_apply_audit_v1",
        "accepted": True,
        "checkpoint": str(args.checkpoint),
        "target_evidence_dir": str(args.target_evidence_dir),
        "eval_gt_evidence_dir": str(eval_dir) if eval_dir is not None else None,
        "renders_dir": str(renders_dir),
        "gt_dir": str(gt_dir) if gt_dir.is_dir() else None,
        "selected_alpha": float(alpha),
        "grid": int(grid),
        "basis_mode": str(basis_mode),
        "min_alpha": float(min_alpha),
        "min_bin_count": float(min_bin_count),
        "max_abs_delta": float(max_abs_delta),
        "no_gt_preflight": no_gt,
        "target_apply": {
            "view_count": int(len(target_paths)),
            "changed_pixels": int(changed_pixels),
            "total_pixels": int(total_pixels),
            "changed_fraction": float(changed_pixels / max(1, total_pixels)),
            "active_pixels": int(active_pixels),
            "active_fraction": float(active_pixels / max(1, total_pixels)),
        },
        "metrics": metrics,
        "per_view": rows,
        "phasej_flowers_reference": gate_ref,
        "phasej_flowers_gate_pass": bool(gate_pass),
        "uses_target_or_test_gt_for_apply": False,
        "uses_target_or_test_gt_for_eval_only": bool(metric_rows),
    }
    _write_json(output_dir / "lowrank_target_apply_audit.json", payload)
    _write_md(output_dir / "lowrank_target_apply_audit.md", payload)
    print(
        json.dumps(
            {
                "output_json": str(output_dir / "lowrank_target_apply_audit.json"),
                "output_md": str(output_dir / "lowrank_target_apply_audit.md"),
                "phasej_flowers_gate_pass": bool(gate_pass),
                "metrics": metrics,
                "target_apply": payload["target_apply"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

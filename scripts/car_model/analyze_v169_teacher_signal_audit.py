#!/usr/bin/env python3
"""Audit v169 teacher-parent residual signal before exact target runs.

This diagnostic reads train-fit / policy-val surface evidence NPZ files and
measures whether the Phase-J teacher residual is real, how much of it survives
masking/clipping, and whether it improves the parent on policy-val GT. It does
not read target/test evidence and does not train or apply a method.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit_evidence_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--min_residual_l1", type=float, default=0.0)
    parser.add_argument("--max_abs_delta_rgb", type=float, default=0.12)
    parser.add_argument("--max_views", type=int, default=0)
    return parser.parse_args()


def evidence_views(evidence_dir: Path) -> list[Path]:
    view_dir = evidence_dir / "views"
    root = view_dir if view_dir.exists() else evidence_dir
    return sorted(root.glob("*.npz"), key=lambda p: p.stem)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def psnr_from_mse(mse: float) -> float | None:
    if not math.isfinite(float(mse)) or float(mse) <= 0.0:
        return None
    return float(-10.0 * math.log10(float(mse)))


def luma(rgb: np.ndarray) -> np.ndarray:
    return (
        0.299 * np.asarray(rgb[0], dtype=np.float32)
        + 0.587 * np.asarray(rgb[1], dtype=np.float32)
        + 0.114 * np.asarray(rgb[2], dtype=np.float32)
    )


def mean_gradient_energy(image: np.ndarray, valid: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if image.ndim != 2 or valid.shape != image.shape:
        return 0.0
    dx = np.abs(image[:, 1:] - image[:, :-1])
    dy = np.abs(image[1:, :] - image[:-1, :])
    vx = valid[:, 1:] & valid[:, :-1]
    vy = valid[1:, :] & valid[:-1, :]
    total = 0.0
    count = 0
    if np.any(vx):
        total += float(np.sum(dx[vx], dtype=np.float64))
        count += int(np.count_nonzero(vx))
    if np.any(vy):
        total += float(np.sum(dy[vy], dtype=np.float64))
        count += int(np.count_nonzero(vy))
    return float(total / max(1, count))


@dataclass
class SplitStats:
    name: str
    views: int = 0
    pixels: int = 0
    visible_pixels: int = 0
    valid_surface_pixels: int = 0
    active_pixels: int = 0
    raw_active_pixels: int = 0
    teacher_better_pixels: int = 0
    clipped_pixels: int = 0
    raw_l1_sum: float = 0.0
    masked_l1_sum: float = 0.0
    raw_l2_sum: float = 0.0
    masked_l2_sum: float = 0.0
    clipped_l2_sum: float = 0.0
    parent_l1_sum: float = 0.0
    teacher_raw_l1_sum: float = 0.0
    teacher_masked_l1_sum: float = 0.0
    parent_mse_sum: float = 0.0
    teacher_raw_mse_sum: float = 0.0
    teacher_masked_mse_sum: float = 0.0
    gt_pixels: int = 0
    raw_grad_sum: float = 0.0
    masked_grad_sum: float = 0.0
    grad_views: int = 0
    pos_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    neg_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    visible_faces: set[int] = field(default_factory=set)
    active_faces: set[int] = field(default_factory=set)
    raw_active_faces: set[int] = field(default_factory=set)
    per_view: list[dict[str, Any]] = field(default_factory=list)

    def add_view(self, row: dict[str, Any]) -> None:
        self.per_view.append(row)

    def summary(self) -> dict[str, Any]:
        raw_l1 = float(self.raw_l1_sum / max(1, self.valid_surface_pixels))
        masked_l1 = float(self.masked_l1_sum / max(1, self.valid_surface_pixels))
        raw_l2 = float(self.raw_l2_sum / max(1, self.valid_surface_pixels))
        masked_l2 = float(self.masked_l2_sum / max(1, self.valid_surface_pixels))
        clipped_l2 = float(self.clipped_l2_sum / max(1, self.valid_surface_pixels))
        sign_total = self.pos_counts + self.neg_counts
        sign_consistency = np.divide(
            np.maximum(self.pos_counts, self.neg_counts),
            np.maximum(1, sign_total),
            dtype=np.float64,
        )
        parent_psnr = psnr_from_mse(float(self.parent_mse_sum / max(1, self.gt_pixels * 3)))
        teacher_raw_psnr = psnr_from_mse(float(self.teacher_raw_mse_sum / max(1, self.gt_pixels * 3)))
        teacher_masked_psnr = psnr_from_mse(float(self.teacher_masked_mse_sum / max(1, self.gt_pixels * 3)))
        return {
            "views": int(self.views),
            "pixels": int(self.pixels),
            "visible_pixels": int(self.visible_pixels),
            "valid_surface_pixels": int(self.valid_surface_pixels),
            "active_pixels": int(self.active_pixels),
            "raw_active_pixels": int(self.raw_active_pixels),
            "visible_fraction": float(self.visible_pixels / max(1, self.pixels)),
            "valid_surface_fraction": float(self.valid_surface_pixels / max(1, self.pixels)),
            "active_fraction_of_valid": float(self.active_pixels / max(1, self.valid_surface_pixels)),
            "raw_active_fraction_of_valid": float(self.raw_active_pixels / max(1, self.valid_surface_pixels)),
            "teacher_better_fraction_of_valid": float(self.teacher_better_pixels / max(1, self.valid_surface_pixels)),
            "clipped_fraction_of_valid": float(self.clipped_pixels / max(1, self.valid_surface_pixels)),
            "mean_raw_residual_l1_on_valid": raw_l1,
            "mean_masked_residual_l1_on_valid": masked_l1,
            "mean_raw_residual_l2_on_valid": raw_l2,
            "mean_masked_residual_l2_on_valid": masked_l2,
            "mean_clipped_raw_residual_l2_on_valid": clipped_l2,
            "mask_l1_retention": float(self.masked_l1_sum / max(1.0e-12, self.raw_l1_sum)),
            "mask_l2_retention": float(self.masked_l2_sum / max(1.0e-12, self.raw_l2_sum)),
            "clip_l2_retention": float(self.clipped_l2_sum / max(1.0e-12, self.raw_l2_sum)),
            "mean_raw_luma_gradient_energy": float(self.raw_grad_sum / max(1, self.grad_views)),
            "mean_masked_luma_gradient_energy": float(self.masked_grad_sum / max(1, self.grad_views)),
            "sign_consistency_rgb": [float(x) for x in sign_consistency.tolist()],
            "sign_consistency_mean": float(np.mean(sign_consistency)),
            "visible_face_count": int(len(self.visible_faces)),
            "raw_active_face_count": int(len(self.raw_active_faces)),
            "active_face_count": int(len(self.active_faces)),
            "active_face_fraction": float(len(self.active_faces) / max(1, len(self.visible_faces))),
            "raw_active_face_fraction": float(len(self.raw_active_faces) / max(1, len(self.visible_faces))),
            "gt_pixels": int(self.gt_pixels),
            "parent_l1": float(self.parent_l1_sum / max(1, self.gt_pixels * 3)),
            "teacher_raw_l1": float(self.teacher_raw_l1_sum / max(1, self.gt_pixels * 3)),
            "teacher_masked_l1": float(self.teacher_masked_l1_sum / max(1, self.gt_pixels * 3)),
            "teacher_raw_l1_gain": float((self.parent_l1_sum - self.teacher_raw_l1_sum) / max(1, self.gt_pixels * 3)),
            "teacher_masked_l1_gain": float((self.parent_l1_sum - self.teacher_masked_l1_sum) / max(1, self.gt_pixels * 3)),
            "parent_psnr": parent_psnr,
            "teacher_raw_psnr": teacher_raw_psnr,
            "teacher_masked_psnr": teacher_masked_psnr,
            "teacher_raw_psnr_gain": (
                float(teacher_raw_psnr - parent_psnr)
                if parent_psnr is not None and teacher_raw_psnr is not None
                else None
            ),
            "teacher_masked_psnr_gain": (
                float(teacher_masked_psnr - parent_psnr)
                if parent_psnr is not None and teacher_masked_psnr is not None
                else None
            ),
        }


def process_view(path: Path, split: SplitStats, args: argparse.Namespace) -> None:
    with np.load(path, allow_pickle=False) as z:
        if "rgb_render" not in z or "face_id" not in z:
            return
        parent = np.asarray(z["rgb_render"], dtype=np.float32)
        face_id = np.asarray(z["face_id"], dtype=np.int64)
        alpha = np.asarray(z["alpha"], dtype=np.float32) if "alpha" in z else np.ones(face_id.shape, dtype=np.float32)
        bary_valid = np.asarray(z["barycentric_valid"], dtype=bool) if "barycentric_valid" in z else face_id >= 0
        visible = face_id >= 0
        valid = visible & bary_valid & (alpha >= float(args.min_alpha))
        raw = np.asarray(z["teacher_residual_rgb_raw"], dtype=np.float32) if "teacher_residual_rgb_raw" in z else np.asarray(z["teacher_residual_rgb"], dtype=np.float32)
        masked = np.asarray(z["teacher_residual_rgb"], dtype=np.float32) if "teacher_residual_rgb" in z else raw
        raw_l1_map = np.mean(np.abs(raw), axis=0)
        masked_l1_map = np.asarray(z["teacher_residual_l1"], dtype=np.float32) if "teacher_residual_l1" in z else np.mean(np.abs(masked), axis=0)
        raw_l2_map = np.sum(raw * raw, axis=0)
        masked_l2_map = np.sum(masked * masked, axis=0)
        clipped = np.clip(raw, -float(args.max_abs_delta_rgb), float(args.max_abs_delta_rgb))
        clipped_l2_map = np.sum(clipped * clipped, axis=0)
        raw_active = valid & (raw_l1_map > float(args.min_residual_l1))
        active = valid & (masked_l1_map > float(args.min_residual_l1))
        teacher_better = (
            (np.asarray(z["teacher_better_mask"], dtype=np.uint8) > 0)
            if "teacher_better_mask" in z
            else active
        )
        clipped_pixels = valid & np.any(np.abs(raw) > float(args.max_abs_delta_rgb), axis=0)

        split.views += 1
        split.pixels += int(face_id.size)
        split.visible_pixels += int(np.count_nonzero(visible))
        split.valid_surface_pixels += int(np.count_nonzero(valid))
        split.raw_active_pixels += int(np.count_nonzero(raw_active))
        split.active_pixels += int(np.count_nonzero(active))
        split.teacher_better_pixels += int(np.count_nonzero(teacher_better & valid))
        split.clipped_pixels += int(np.count_nonzero(clipped_pixels))
        split.raw_l1_sum += float(np.sum(raw_l1_map[valid], dtype=np.float64))
        split.masked_l1_sum += float(np.sum(masked_l1_map[valid], dtype=np.float64))
        split.raw_l2_sum += float(np.sum(raw_l2_map[valid], dtype=np.float64))
        split.masked_l2_sum += float(np.sum(masked_l2_map[valid], dtype=np.float64))
        split.clipped_l2_sum += float(np.sum(clipped_l2_map[valid], dtype=np.float64))
        raw_values = raw[:, active]
        if raw_values.size:
            split.pos_counts += np.sum(raw_values > 0.0, axis=1).astype(np.int64)
            split.neg_counts += np.sum(raw_values < 0.0, axis=1).astype(np.int64)
        if np.any(visible):
            split.visible_faces.update(int(x) for x in np.unique(face_id[visible]) if int(x) >= 0)
        if np.any(raw_active):
            split.raw_active_faces.update(int(x) for x in np.unique(face_id[raw_active]) if int(x) >= 0)
        if np.any(active):
            split.active_faces.update(int(x) for x in np.unique(face_id[active]) if int(x) >= 0)
        split.raw_grad_sum += mean_gradient_energy(luma(raw), valid)
        split.masked_grad_sum += mean_gradient_energy(luma(masked), valid)
        split.grad_views += 1

        row: dict[str, Any] = {
            "view": path.stem,
            "valid_surface_fraction": float(np.count_nonzero(valid) / max(1, face_id.size)),
            "active_fraction_of_valid": float(np.count_nonzero(active) / max(1, np.count_nonzero(valid))),
            "mean_raw_residual_l1_on_valid": float(np.mean(raw_l1_map[valid])) if np.any(valid) else 0.0,
            "mean_masked_residual_l1_on_valid": float(np.mean(masked_l1_map[valid])) if np.any(valid) else 0.0,
            "mask_l1_retention": float(np.sum(masked_l1_map[valid]) / max(1.0e-12, np.sum(raw_l1_map[valid]))) if np.any(valid) else 0.0,
            "clipped_fraction_of_valid": float(np.count_nonzero(clipped_pixels) / max(1, np.count_nonzero(valid))),
        }
        if "rgb_gt" in z:
            gt = np.asarray(z["rgb_gt"], dtype=np.float32)
            teacher_raw = np.clip(parent + raw, 0.0, 1.0)
            teacher_masked = np.clip(parent + masked, 0.0, 1.0)
            parent_abs = np.abs(parent - gt)
            raw_abs = np.abs(teacher_raw - gt)
            masked_abs = np.abs(teacher_masked - gt)
            parent_sq = (parent - gt) ** 2
            raw_sq = (teacher_raw - gt) ** 2
            masked_sq = (teacher_masked - gt) ** 2
            split.gt_pixels += int(gt.shape[1] * gt.shape[2])
            split.parent_l1_sum += float(np.sum(parent_abs, dtype=np.float64))
            split.teacher_raw_l1_sum += float(np.sum(raw_abs, dtype=np.float64))
            split.teacher_masked_l1_sum += float(np.sum(masked_abs, dtype=np.float64))
            split.parent_mse_sum += float(np.sum(parent_sq, dtype=np.float64))
            split.teacher_raw_mse_sum += float(np.sum(raw_sq, dtype=np.float64))
            split.teacher_masked_mse_sum += float(np.sum(masked_sq, dtype=np.float64))
            parent_mse = float(np.mean(parent_sq))
            raw_mse = float(np.mean(raw_sq))
            masked_mse = float(np.mean(masked_sq))
            parent_psnr = psnr_from_mse(parent_mse)
            raw_psnr = psnr_from_mse(raw_mse)
            masked_psnr = psnr_from_mse(masked_mse)
            row.update(
                {
                    "parent_l1": float(np.mean(parent_abs)),
                    "teacher_raw_l1": float(np.mean(raw_abs)),
                    "teacher_masked_l1": float(np.mean(masked_abs)),
                    "teacher_raw_l1_gain": float(np.mean(parent_abs) - np.mean(raw_abs)),
                    "teacher_masked_l1_gain": float(np.mean(parent_abs) - np.mean(masked_abs)),
                    "parent_psnr": parent_psnr,
                    "teacher_raw_psnr": raw_psnr,
                    "teacher_masked_psnr": masked_psnr,
                    "teacher_raw_psnr_gain": (
                        float(raw_psnr - parent_psnr)
                        if raw_psnr is not None and parent_psnr is not None
                        else None
                    ),
                    "teacher_masked_psnr_gain": (
                        float(masked_psnr - parent_psnr)
                        if masked_psnr is not None and parent_psnr is not None
                        else None
                    ),
                }
            )
        split.add_view(row)


def render_markdown(result: dict[str, Any]) -> str:
    def fmt(value: Any, digits: int = 6) -> str:
        if value is None:
            return "missing"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# v169 Teacher Signal Audit",
        "",
        f"- fit evidence: `{result['fit_evidence_dir']}`",
        f"- policy-val stride: `{result['settings']['policy_val_stride']}`",
        f"- target/test GT usage: `{result['target_or_test_gt_usage']}`",
        f"- verdict: `{result['verdict']['reason']}`",
        "",
        "## Summary",
        "",
        "| split | views | raw L1 | masked L1 | mask L1 retention | clip frac | active valid frac | active face frac | parent PSNR | teacher raw PSNR gain | teacher masked PSNR gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all", "train_fit", "policy_val"):
        row = result["splits"][name]["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["views"]),
                    fmt(row["mean_raw_residual_l1_on_valid"]),
                    fmt(row["mean_masked_residual_l1_on_valid"]),
                    fmt(row["mask_l1_retention"]),
                    fmt(row["clipped_fraction_of_valid"]),
                    fmt(row["active_fraction_of_valid"]),
                    fmt(row["active_face_fraction"]),
                    fmt(row["parent_psnr"]),
                    fmt(row["teacher_raw_psnr_gain"], signed_digits := 6),
                    fmt(row["teacher_masked_psnr_gain"], signed_digits),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `raw` means the full teacher-parent residual before the better-mask target is applied.",
        "- `masked` means the residual actually stored under `teacher_residual_rgb`.",
        "- High raw signal with low masked retention means the teacher path is real but current masking removes much of the available correction.",
        "- Policy-val PSNR/L1 gains are measured against GT in this diagnostic only; target/test evidence is not read.",
        "",
        "## Key Fields",
        "",
        "```json",
        json.dumps(result["verdict"], indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    views = evidence_views(args.fit_evidence_dir)
    if args.max_views > 0:
        views = views[: int(args.max_views)]
    if not views:
        raise SystemExit(f"no evidence views found under {args.fit_evidence_dir}")

    splits = {
        "all": SplitStats("all"),
        "train_fit": SplitStats("train_fit"),
        "policy_val": SplitStats("policy_val"),
    }
    stride = max(0, int(args.policy_val_stride))
    for view_index, path in enumerate(views):
        split_name = "policy_val" if stride > 1 and view_index % stride == 0 else "train_fit"
        process_view(path, splits["all"], args)
        process_view(path, splits[split_name], args)

    split_payload = {
        name: {
            "summary": stats.summary(),
            "per_view": stats.per_view,
        }
        for name, stats in splits.items()
    }
    policy_summary = split_payload["policy_val"]["summary"]
    verdict = {
        "teacher_parent_residual_nonzero": bool(policy_summary["mean_raw_residual_l1_on_valid"] > 1.0e-5),
        "masked_residual_nonzero": bool(policy_summary["mean_masked_residual_l1_on_valid"] > 1.0e-5),
        "teacher_raw_improves_policy_val_psnr": bool((policy_summary["teacher_raw_psnr_gain"] or 0.0) > 0.0),
        "teacher_masked_improves_policy_val_psnr": bool((policy_summary["teacher_masked_psnr_gain"] or 0.0) > 0.0),
        "mask_l1_retention": float(policy_summary["mask_l1_retention"]),
        "clip_l2_retention": float(policy_summary["clip_l2_retention"]),
        "reason": "teacher_signal_present_but_masked_projection_must_be_certified"
        if policy_summary["mean_raw_residual_l1_on_valid"] > 1.0e-5
        else "teacher_parent_residual_near_zero_check_paths",
    }
    result = json_safe(
        {
            "operator": "analyze_v169_teacher_signal_audit",
            "test_usage": "none",
            "target_or_test_gt_usage": "none",
            "fit_evidence_dir": str(args.fit_evidence_dir),
            "view_count": int(len(views)),
            "settings": {
                "policy_val_stride": int(args.policy_val_stride),
                "min_alpha": float(args.min_alpha),
                "min_residual_l1": float(args.min_residual_l1),
                "max_abs_delta_rgb": float(args.max_abs_delta_rgb),
                "max_views": int(args.max_views),
            },
            "splits": split_payload,
            "verdict": verdict,
        }
    )
    if args.output_json.resolve() == args.output_md.resolve():
        raise SystemExit("output_json and output_md must differ")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()

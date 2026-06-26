#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    _valid_sample_mask,
    evidence_views,
    fit_atlas,
    load_carrier_faces,
    predict_delta_for_npz,
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _carrier_face_filter(carriers: list[dict[str, Any]], retained: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for carrier in carriers:
        faces = [dict(row) for row in carrier.get("faces", []) if int(row.get("face_id", -1)) in retained]
        if not faces:
            continue
        new_carrier = dict(carrier)
        new_carrier["faces"] = faces
        new_carrier["face_ids"] = [int(row["face_id"]) for row in faces]
        new_carrier["pixels"] = int(sum(int(row.get("pixels", 0)) for row in faces))
        new_carrier["pruned_face_count"] = int(len(carrier.get("face_ids", []) or []) - len(faces))
        out.append(new_carrier)
    for idx, carrier in enumerate(out):
        carrier["carrier_id"] = int(idx)
    return out


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    rows = summary.get("per_view", [])
    lines = [
        "# Policy-Val Pruned Region Carriers",
        "",
        f"- input carrier json: `{summary.get('input_carrier_json', '')}`",
        f"- fit evidence: `{summary.get('fit_evidence_dir', '')}`",
        f"- output carrier json: `{summary.get('out_json', '')}`",
        f"- input carriers: `{summary.get('input_carriers', 0)}`",
        f"- output carriers: `{summary.get('output_carriers', 0)}`",
        f"- candidate faces: `{summary.get('candidate_faces', 0)}`",
        f"- atlas faces: `{summary.get('atlas_faces', 0)}`",
        f"- retained faces: `{summary.get('retained_faces', 0)}`",
        f"- removed faces: `{summary.get('removed_faces', 0)}`",
        f"- prune unit: `{summary.get('prune_unit', 'face')}`",
        f"- input units: `{summary.get('input_units', 0)}`",
        f"- retained units: `{summary.get('retained_units', 0)}`",
        f"- prune alpha: `{summary.get('alpha', 0.0)}`",
        f"- greedy removals: `{summary.get('greedy_removed_faces', 0)}`",
        "",
        "## Retained Policy-Val Relative Gains",
        "",
        "| view | samples | rel gain | sse before | sse gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {view} | {samples} | {rel_gain:.8f} | {sse_before:.8f} | {sse_gain:.8f} |".format(
                view=row.get("view", ""),
                samples=int(row.get("samples", 0)),
                rel_gain=float(row.get("relative_gain", 0.0)),
                sse_before=float(row.get("sse_before", 0.0)),
                sse_gain=float(row.get("sse_gain", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "This file is train-evidence only. The held-out test split is not used for",
            "carrier pruning; final promotion still requires the downstream atlas apply",
            "gate and image metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune residual-region carriers using train policy-val face gains.")
    parser.add_argument("--input_carrier_json", required=True)
    parser.add_argument("--fit_evidence_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--residual_rgb_key", default="teacher_residual_rgb")
    parser.add_argument("--residual_l1_key", default="teacher_residual_l1")
    parser.add_argument("--texture_size", type=int, default=16)
    parser.add_argument("--max_carriers", type=int, default=64)
    parser.add_argument("--max_faces_per_carrier", type=int, default=128)
    parser.add_argument("--max_faces", type=int, default=4096)
    parser.add_argument("--policy_val_stride", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.015625)
    parser.add_argument("--min_l1", type=float, default=0.001)
    parser.add_argument("--min_alpha", type=float, default=0.03)
    parser.add_argument("--min_atlas_bin_count", type=int, default=1)
    parser.add_argument("--min_atlas_face_samples", type=int, default=32)
    parser.add_argument("--max_atlas_bin_rgb_variance", type=float, default=-1.0)
    parser.add_argument("--min_atlas_bin_sign_consistency", type=float, default=0.0)
    parser.add_argument("--atlas_lowpass_passes", type=int, default=1)
    parser.add_argument("--atlas_lowpass_neighbor_min_count", type=int, default=1)
    parser.add_argument(
        "--atlas_empty_bin_fill_mode",
        choices=("zero", "face_mean", "nearest_observed"),
        default="face_mean",
        help="Empty-bin fill mode passed through to the shared atlas fitter. Default preserves legacy pruning behavior.",
    )
    parser.add_argument("--atlas_nearest_fill_max_steps", type=int, default=32)
    parser.add_argument("--atlas_nearest_fill_decay", type=float, default=0.92)
    parser.add_argument("--max_samples_per_view", type=int, default=240000)
    parser.add_argument("--min_face_total_gain", type=float, default=0.0)
    parser.add_argument("--min_view_relative_gain", type=float, default=0.0)
    parser.add_argument("--prune_unit", choices=("face", "carrier"), default="face")
    parser.add_argument("--greedy_repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_json = Path(args.input_carrier_json)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if out_json.exists() and not bool(args.force):
        raise FileExistsError(out_json)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    candidate_faces, carrier_summary = load_carrier_faces(
        input_json,
        max_carriers=int(args.max_carriers),
        max_faces_per_carrier=int(args.max_faces_per_carrier),
        max_faces=int(args.max_faces),
    )
    view_paths = evidence_views(Path(args.fit_evidence_dir))
    atlas, fit_summary, _fit_views, val_views = fit_atlas(
        view_paths,
        candidate_faces=candidate_faces,
        residual_rgb_key=str(args.residual_rgb_key),
        residual_l1_key=str(args.residual_l1_key),
        texture_size=int(args.texture_size),
        policy_val_stride=int(args.policy_val_stride),
        min_l1=float(args.min_l1),
        min_alpha=float(args.min_alpha),
        max_samples_per_view=int(args.max_samples_per_view),
        fill_empty_with_face_mean=True,
        atlas_empty_bin_fill_mode=str(args.atlas_empty_bin_fill_mode),
        atlas_nearest_fill_max_steps=int(args.atlas_nearest_fill_max_steps),
        atlas_nearest_fill_decay=float(args.atlas_nearest_fill_decay),
        atlas_lowpass_passes=int(args.atlas_lowpass_passes),
        atlas_lowpass_neighbor_min_count=int(args.atlas_lowpass_neighbor_min_count),
        surface_multiscale_prior_mode="none",
        surface_multiscale_prior_block_sizes=[2, 4, 8],
        surface_multiscale_prior_min_bin_samples=8,
        surface_multiscale_prior_count_tau=32.0,
        surface_multiscale_prior_blend=0.0,
        surface_multiscale_prior_gate_mode="none",
        surface_multiscale_prior_min_prior_weight=0.0,
        surface_multiscale_prior_min_direct_samples=1,
        surface_multiscale_prior_min_sign_consistency=0.0,
        surface_multiscale_prior_max_mean_variance=-1.0,
        surface_multiscale_prior_min_cosine=0.0,
        view_conditioned_basis_mode="none",
        view_conditioned_basis_min_bin_samples=16,
        view_conditioned_basis_ridge=1.0e-3,
        view_conditioned_basis_ood_mode="none",
        view_conditioned_basis_ood_max_z=2.5,
        view_conditioned_basis_ood_min_std=5.0e-2,
        teacher_distilled_basis_mode="none",
        teacher_distilled_basis_min_face_samples=1024,
        teacher_distilled_basis_ridge=1.0e-2,
        teacher_distilled_basis_ood_max_z=3.0,
        teacher_distilled_basis_ood_min_std=5.0e-2,
        teacher_distilled_basis_apply_mode="blend",
        teacher_distilled_basis_blend=0.5,
    )
    faces = np.array(sorted(atlas.keys()), dtype=np.int64)
    face_to_index = {int(face): idx for idx, face in enumerate(faces.tolist())}
    gains = np.zeros((faces.size, len(val_views)), dtype=np.float64)
    counts = np.zeros((faces.size, len(val_views)), dtype=np.int64)
    view_sse = np.zeros((len(val_views),), dtype=np.float64)
    view_samples = np.zeros((len(val_views),), dtype=np.int64)

    for view_idx, path in enumerate(val_views):
        z = np.load(path)
        mask = _valid_sample_mask(
            z,
            set(int(x) for x in faces.tolist()),
            str(args.residual_l1_key),
            float(args.min_l1),
            float(args.min_alpha),
        )
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        if int(args.max_samples_per_view) > 0 and ys.size > int(args.max_samples_per_view):
            rng = np.random.default_rng(37 + view_idx)
            take = rng.choice(ys.size, size=int(args.max_samples_per_view), replace=False)
            ys = ys[take]
            xs = xs[take]
            mask = np.zeros_like(mask, dtype=bool)
            mask[ys, xs] = True
        residual = np.asarray(z[str(args.residual_rgb_key)], dtype=np.float32)
        target = np.stack([residual[0][mask], residual[1][mask], residual[2][mask]], axis=1)
        pred, _valid = predict_delta_for_npz(
            z,
            atlas,
            float(args.alpha),
            float(args.min_alpha),
            min_atlas_bin_count=int(args.min_atlas_bin_count),
            min_atlas_face_samples=int(args.min_atlas_face_samples),
            max_atlas_bin_rgb_variance=float(args.max_atlas_bin_rgb_variance),
            min_atlas_bin_sign_consistency=float(args.min_atlas_bin_sign_consistency),
        )
        pred_samples = np.stack([pred[0][mask], pred[1][mask], pred[2][mask]], axis=1)
        before = np.sum(target * target, axis=1)
        after = np.sum((target - pred_samples) * (target - pred_samples), axis=1)
        pixel_gain = before - after
        face_ids = np.asarray(z["face_id"], dtype=np.int64)[mask]
        view_sse[view_idx] = float(np.sum(before))
        view_samples[view_idx] = int(target.shape[0])
        for face in np.unique(face_ids):
            idx = face_to_index.get(int(face))
            if idx is None:
                continue
            local = face_ids == int(face)
            gains[idx, view_idx] += float(np.sum(pixel_gain[local]))
            counts[idx, view_idx] += int(np.sum(local))

    total_gain = gains.sum(axis=1)
    min_abs_view_gain = float(args.min_view_relative_gain) * np.maximum(view_sse, 1.0e-12)

    unit_names: list[str]
    unit_face_indices: list[np.ndarray]
    if str(args.prune_unit) == "carrier":
        assigned: set[int] = set()
        unit_names = []
        unit_face_indices = []
        for carrier in list(payload.get("carriers") or [])[: int(args.max_carriers)]:
            raw = carrier.get("face_ids")
            if raw is None:
                raw = [row.get("face_id") for row in carrier.get("faces", []) if "face_id" in row]
            raw_faces = [int(face) for face in raw[: int(args.max_faces_per_carrier)]]
            indices = []
            for face in raw_faces:
                if face in assigned or face not in face_to_index:
                    continue
                indices.append(face_to_index[face])
                assigned.add(face)
            if indices:
                unit_names.append(f"carrier_{carrier.get('carrier_id', len(unit_names))}")
                unit_face_indices.append(np.array(indices, dtype=np.int64))
    else:
        unit_names = [f"face_{int(face)}" for face in faces.tolist()]
        unit_face_indices = [np.array([idx], dtype=np.int64) for idx in range(faces.size)]

    if unit_face_indices:
        unit_gains = np.stack([gains[indices].sum(axis=0) for indices in unit_face_indices], axis=0)
    else:
        unit_gains = np.zeros((0, len(val_views)), dtype=np.float64)
    unit_total_gain = unit_gains.sum(axis=1) if unit_gains.size else np.zeros((0,), dtype=np.float64)
    retained_units = unit_total_gain > float(args.min_face_total_gain)

    greedy_removed = 0
    if bool(args.greedy_repair) and retained_units.any():
        while True:
            current = unit_gains[retained_units].sum(axis=0)
            bad_views = np.where(current < min_abs_view_gain)[0]
            if bad_views.size == 0:
                break
            worst = int(bad_views[np.argmin(current[bad_views] - min_abs_view_gain[bad_views])])
            retained_idx = np.where(retained_units)[0]
            harmful = retained_idx[unit_gains[retained_idx, worst] < 0.0]
            if harmful.size == 0:
                break
            other_loss = np.maximum(unit_total_gain[harmful] - unit_gains[harmful, worst], 0.0)
            score = (-unit_gains[harmful, worst]) / (1.0e-12 + other_loss)
            remove_idx = int(harmful[int(np.argmax(score))])
            retained_units[remove_idx] = False
            greedy_removed += 1

    retained_face_indices: set[int] = set()
    for keep, indices in zip(retained_units.tolist(), unit_face_indices):
        if keep:
            retained_face_indices.update(int(idx) for idx in indices.tolist())
    retained = np.zeros((faces.size,), dtype=bool)
    if retained_face_indices:
        retained[np.array(sorted(retained_face_indices), dtype=np.int64)] = True

    retained_faces = set(int(face) for face in faces[retained].tolist())
    pruned_carriers = _carrier_face_filter(list(payload.get("carriers") or []), retained_faces)
    out_payload = dict(payload)
    out_payload["carriers"] = pruned_carriers
    out_payload["carrier_count"] = int(len(pruned_carriers))
    out_payload["evidence_face_count"] = int(len(retained_faces))
    out_payload["evidence_faces_preview"] = sorted(retained_faces)[:50]

    final_gain = gains[retained].sum(axis=0) if retained.any() else np.zeros_like(view_sse)
    per_view = []
    for idx, path in enumerate(val_views):
        per_view.append(
            {
                "view": path.stem,
                "samples": int(view_samples[idx]),
                "sse_before": float(view_sse[idx]),
                "sse_gain": float(final_gain[idx]),
                "relative_gain": float(final_gain[idx] / max(view_sse[idx], 1.0e-12)),
            }
        )
    summary = {
        "input_carrier_json": str(input_json),
        "fit_evidence_dir": str(args.fit_evidence_dir),
        "out_json": str(out_json),
        "out_md": str(out_md),
        "input_carriers": int(len(payload.get("carriers") or [])),
        "output_carriers": int(len(pruned_carriers)),
        "candidate_faces": int(len(candidate_faces)),
        "atlas_faces": int(len(faces)),
        "retained_faces": int(len(retained_faces)),
        "removed_faces": int(len(faces) - len(retained_faces)),
        "greedy_removed_faces": int(greedy_removed),
        "prune_unit": str(args.prune_unit),
        "input_units": int(len(unit_face_indices)),
        "retained_units": int(np.sum(retained_units)),
        "alpha": float(args.alpha),
        "settings": vars(args),
        "carrier_summary": carrier_summary,
        "fit_summary": fit_summary,
        "per_view": per_view,
    }
    out_payload["policy_val_face_pruning"] = summary
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_to_jsonable(out_payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_md(out_md, summary)
    print(json.dumps(_to_jsonable({k: summary[k] for k in ("output_carriers", "retained_faces", "removed_faces", "greedy_removed_faces", "per_view")}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

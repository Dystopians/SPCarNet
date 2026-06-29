#!/usr/bin/env python3
"""Analyze v169 teacher-signal projection diagnostics from an evidence cache.

This utility is intentionally read-only with respect to the evidence cache.  It
loads per-view NPZ files one at a time, samples image-shaped arrays if requested,
and writes only aggregate JSON/Markdown diagnostics.  It does not launch
training, rendering, or image export.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by runtime environment.
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR: Exception | None = exc
else:
    NUMPY_IMPORT_ERROR = None


DEFAULT_PHASEJ_FLOWERS = {"PSNR": 20.304358, "SSIM": 0.557770, "LPIPS": 0.329222}
RAW_TEACHER_RGB_KEYS = ("teacher_residual_rgb_raw", "teacher_parent_residual_rgb", "teacher_residual_rgb")
FIT_SIGNAL_RGB_KEYS = ("teacher_residual_rgb", "teacher_residual_rgb_raw", "teacher_parent_residual_rgb")
EPS = 1.0e-8
CARRIER_BARY_BINS = 8


class RunningScalar:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def update(self, values: Any) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(np.sum(arr))
        self.sum_sq += float(np.sum(arr * arr))
        local_min = float(np.min(arr))
        local_max = float(np.max(arr))
        self.min_value = local_min if self.min_value is None else min(self.min_value, local_min)
        self.max_value = local_max if self.max_value is None else max(self.max_value, local_max)

    def as_dict(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"count": 0, "mean": None, "variance": None, "min": None, "max": None}
        mean = self.sum / float(self.count)
        variance = max(0.0, self.sum_sq / float(self.count) - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "variance": float(variance),
            "std": float(math.sqrt(variance)),
            "min": self.min_value,
            "max": self.max_value,
        }


class CarrierAccumulator:
    """Same-cache least-squares projection proxy for a residual carrier."""

    def __init__(self, mode: str, description: str) -> None:
        self.mode = mode
        self.description = description
        self.count_by_key: dict[int, int] = {}
        self.signal_sum_by_key: dict[int, list[float]] = {}
        self.gt_sum_by_key: dict[int, list[float]] = {}
        self.sample_pixels = 0
        self.signal_l2_total = 0.0
        self.gt_l2_total = 0.0
        self.direct_signal_sse_total = 0.0
        self.gt_available = False

    def update(self, keys: Any, signal_rgb: Any, gt_correction_rgb: Any | None = None) -> None:
        keys_arr = np.asarray(keys, dtype=np.int64).reshape(-1)
        signal = np.asarray(signal_rgb, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[0] != 3:
            raise ValueError(f"carrier {self.mode} expected signal shape 3xN, got {signal.shape}")
        if signal.shape[1] != keys_arr.shape[0]:
            raise ValueError(
                f"carrier {self.mode} key/signal count mismatch: {keys_arr.shape[0]} vs {signal.shape[1]}"
            )
        finite = np.isfinite(signal).all(axis=0)
        gt = None
        if gt_correction_rgb is not None:
            gt = np.asarray(gt_correction_rgb, dtype=np.float64)
            if gt.ndim != 2 or gt.shape[0] != 3:
                raise ValueError(f"carrier {self.mode} expected GT correction shape 3xN, got {gt.shape}")
            if gt.shape[1] != keys_arr.shape[0]:
                raise ValueError(f"carrier {self.mode} key/GT count mismatch: {keys_arr.shape[0]} vs {gt.shape[1]}")
            finite &= np.isfinite(gt).all(axis=0)

        if not np.any(finite):
            return
        keys_arr = keys_arr[finite]
        signal = signal[:, finite]
        if gt is not None:
            gt = gt[:, finite]

        unique, inverse = np.unique(keys_arr, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique)).astype(np.int64, copy=False)
        signal_sums = np.vstack(
            [np.bincount(inverse, weights=signal[channel], minlength=len(unique)) for channel in range(3)]
        )

        for idx, key in enumerate(unique.tolist()):
            key_int = int(key)
            self.count_by_key[key_int] = self.count_by_key.get(key_int, 0) + int(counts[idx])
            previous = self.signal_sum_by_key.get(key_int)
            if previous is None:
                self.signal_sum_by_key[key_int] = [
                    float(signal_sums[0, idx]),
                    float(signal_sums[1, idx]),
                    float(signal_sums[2, idx]),
                ]
            else:
                previous[0] += float(signal_sums[0, idx])
                previous[1] += float(signal_sums[1, idx])
                previous[2] += float(signal_sums[2, idx])

        self.sample_pixels += int(signal.shape[1])
        self.signal_l2_total += float(np.sum(signal * signal))

        if gt is not None:
            self.gt_available = True
            gt_sums = np.vstack(
                [np.bincount(inverse, weights=gt[channel], minlength=len(unique)) for channel in range(3)]
            )
            for idx, key in enumerate(unique.tolist()):
                key_int = int(key)
                previous_gt = self.gt_sum_by_key.get(key_int)
                if previous_gt is None:
                    self.gt_sum_by_key[key_int] = [
                        float(gt_sums[0, idx]),
                        float(gt_sums[1, idx]),
                        float(gt_sums[2, idx]),
                    ]
                else:
                    previous_gt[0] += float(gt_sums[0, idx])
                    previous_gt[1] += float(gt_sums[1, idx])
                    previous_gt[2] += float(gt_sums[2, idx])
            self.gt_l2_total += float(np.sum(gt * gt))
            diff = gt - signal
            self.direct_signal_sse_total += float(np.sum(diff * diff))

    def finalize(self, *, phasej_flowers_psnr: float) -> dict[str, Any]:
        counts = np.asarray(list(self.count_by_key.values()), dtype=np.float64)
        projection_signal_l2 = 0.0
        gt_cross = 0.0
        for key, count in self.count_by_key.items():
            if count <= 0:
                continue
            signal_sum = np.asarray(self.signal_sum_by_key[key], dtype=np.float64)
            projection_signal_l2 += float(np.dot(signal_sum, signal_sum) / float(count))
            if self.gt_available and key in self.gt_sum_by_key:
                gt_sum = np.asarray(self.gt_sum_by_key[key], dtype=np.float64)
                gt_cross += float(np.dot(signal_sum, gt_sum) / float(count))

        explained_fraction = (
            projection_signal_l2 / self.signal_l2_total if self.signal_l2_total > 0.0 else None
        )
        out: dict[str, Any] = {
            "mode": self.mode,
            "description": self.description,
            "sample_pixels": int(self.sample_pixels),
            "carrier_count": int(len(self.count_by_key)),
            "carrier_pixel_count_stats": _count_stats(counts),
            "signal_l2_total": float(self.signal_l2_total),
            "projected_signal_l2": float(projection_signal_l2),
            "projected_signal_l2_fraction": explained_fraction,
        }
        if self.gt_available and self.sample_pixels > 0:
            channel_count = float(3 * self.sample_pixels)
            parent_mse = self.gt_l2_total / channel_count
            direct_mse = self.direct_signal_sse_total / channel_count
            projected_sse = max(0.0, self.gt_l2_total - 2.0 * gt_cross + projection_signal_l2)
            projected_mse = projected_sse / channel_count
            projected_psnr = _psnr_from_mse(projected_mse)
            parent_psnr = _psnr_from_mse(parent_mse)
            direct_psnr = _psnr_from_mse(direct_mse)
            out["same_cache_quality_proxy"] = {
                "available": True,
                "caveat": "Same-cache fit proxy only; not held-out evaluation and no SSIM/LPIPS estimate.",
                "parent_psnr_proxy": parent_psnr,
                "direct_signal_psnr_proxy": direct_psnr,
                "projected_carrier_psnr_proxy": projected_psnr,
                "projected_minus_parent_psnr": _delta(projected_psnr, parent_psnr),
                "direct_signal_minus_parent_psnr": _delta(direct_psnr, parent_psnr),
                "projected_minus_phasej_flowers_psnr": _delta(projected_psnr, phasej_flowers_psnr),
                "beats_phasej_flowers_psnr_proxy": (
                    bool(projected_psnr > phasej_flowers_psnr) if projected_psnr is not None else None
                ),
                "parent_mse_proxy": float(parent_mse),
                "direct_signal_mse_proxy": float(direct_mse),
                "projected_carrier_mse_proxy": float(projected_mse),
            }
        else:
            out["same_cache_quality_proxy"] = {
                "available": False,
                "reason": "rgb_render and rgb_gt were not both available for the sampled carrier pixels.",
            }
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_evidence_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum sorted NPZ view files to analyze. Set <=0 to analyze all files.",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
        help="Spatial stride applied to HxW arrays before aggregation. Use >1 for lighter diagnostics.",
    )
    parser.add_argument(
        "--phasej_flowers_psnr",
        "--phasej-flowers-psnr",
        type=float,
        default=DEFAULT_PHASEJ_FLOWERS["PSNR"],
    )
    parser.add_argument(
        "--phasej_flowers_ssim",
        "--phasej-flowers-ssim",
        type=float,
        default=DEFAULT_PHASEJ_FLOWERS["SSIM"],
    )
    parser.add_argument(
        "--phasej_flowers_lpips",
        "--phasej-flowers-lpips",
        type=float,
        default=DEFAULT_PHASEJ_FLOWERS["LPIPS"],
    )
    return parser.parse_args()


def _evidence_views(evidence_dir: Path) -> tuple[Path, list[Path]]:
    if not evidence_dir.is_dir():
        raise FileNotFoundError(f"missing --teacher_evidence_dir: {evidence_dir}")
    for name in ("views", "per_view_npz"):
        candidate = evidence_dir / name
        if candidate.is_dir():
            paths = sorted(candidate.glob("*.npz"))
            if paths:
                return candidate, paths
    paths = sorted(evidence_dir.glob("*.npz"))
    if paths:
        return evidence_dir, paths
    raise FileNotFoundError(f"{evidence_dir} has no views/*.npz, per_view_npz/*.npz, or root-level .npz files")


def _load_npz_fields(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as z:
        return set(str(key) for key in z.files)


def _select_first_existing(fields: set[str], keys: tuple[str, ...], *, label: str, path: Path) -> str:
    for key in keys:
        if key in fields:
            return key
    raise KeyError(
        f"{path} missing required {label}; expected one of {list(keys)}, available fields: {sorted(fields)}"
    )


def _require_fields(fields: set[str], required: list[str], *, path: Path) -> None:
    missing = [key for key in required if key not in fields]
    if missing:
        raise KeyError(f"{path} missing required NPZ fields {missing}; available fields: {sorted(fields)}")


def _as_2d(value: Any, *, key: str, path: Path) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path} field {key!r} must be HxW, got shape {arr.shape}")
    return arr


def _as_rgb_chw(value: Any, *, key: str, path: Path, expected_hw: tuple[int, int]) -> Any:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{path} field {key!r} must be CHW RGB, got shape {arr.shape}")
    if tuple(arr.shape[1:]) != tuple(expected_hw):
        raise ValueError(f"{path} field {key!r} shape {arr.shape[1:]} does not match face_id shape {expected_hw}")
    return arr


def _shape_matches_2d(value: Any, *, key: str, path: Path, expected_hw: tuple[int, int]) -> Any:
    arr = _as_2d(value, key=key, path=path)
    if tuple(arr.shape) != tuple(expected_hw):
        raise ValueError(f"{path} field {key!r} shape {arr.shape} does not match face_id shape {expected_hw}")
    return arr


def _update_count_dict(counts_by_key: dict[int, int], keys: Any) -> None:
    keys_arr = np.asarray(keys, dtype=np.int64).reshape(-1)
    if keys_arr.size == 0:
        return
    unique, counts = np.unique(keys_arr, return_counts=True)
    for key, count in zip(unique.tolist(), counts.tolist()):
        key_int = int(key)
        counts_by_key[key_int] = counts_by_key.get(key_int, 0) + int(count)


def _update_sign_consistency(
    sums_by_face: dict[int, list[float]],
    counts_by_face: dict[int, list[int]],
    face_ids: Any,
    residual_rgb: Any,
) -> None:
    faces = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    residual = np.asarray(residual_rgb, dtype=np.float32)
    if residual.ndim != 2 or residual.shape[0] != 3 or residual.shape[1] != faces.shape[0]:
        raise ValueError(f"sign consistency expected residual shape 3xN, got {residual.shape}")
    if faces.size == 0:
        return
    signs = np.sign(residual).astype(np.float64)
    signs[np.abs(residual) <= EPS] = 0.0
    unique, inverse = np.unique(faces, return_inverse=True)
    for channel in range(3):
        nz = (signs[channel] != 0.0).astype(np.float64)
        local_counts = np.bincount(inverse, weights=nz, minlength=len(unique))
        local_sums = np.bincount(inverse, weights=signs[channel], minlength=len(unique))
        for idx, face in enumerate(unique.tolist()):
            count = int(local_counts[idx])
            if count <= 0:
                continue
            face_int = int(face)
            sums = sums_by_face.setdefault(face_int, [0.0, 0.0, 0.0])
            counts = counts_by_face.setdefault(face_int, [0, 0, 0])
            sums[channel] += float(local_sums[idx])
            counts[channel] += count


def _count_stats(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "sum": 0, "min": None, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(arr.size),
        "sum": int(np.sum(arr)),
        "min": int(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": int(np.max(arr)),
    }


def _psnr_from_mse(mse: float | None) -> float | None:
    if mse is None or not math.isfinite(float(mse)) or float(mse) <= 0.0:
        return None
    return float(-10.0 * math.log10(float(mse)))


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def _safe_fraction(num: int | float, den: int | float) -> float | None:
    den_f = float(den)
    if den_f <= 0.0:
        return None
    return float(num) / den_f


def _read_top_support_faces(evidence_dir: Path) -> dict[str, Any]:
    path = evidence_dir / "top_residual_supports.csv"
    if not path.is_file():
        return {"available": False, "path": str(path), "faces": set(), "rows": 0}
    faces: set[int] = set()
    rows = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "face_id" not in (reader.fieldnames or []):
            return {"available": False, "path": str(path), "faces": set(), "rows": 0, "reason": "missing_face_id_column"}
        for row in reader:
            rows += 1
            try:
                faces.add(int(row["face_id"]))
            except (TypeError, ValueError):
                continue
    return {"available": True, "path": str(path), "faces": faces, "rows": int(rows)}


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _carrier_keys_barycentric(face_id: Any, barycentric: Any, *, bins: int) -> Any:
    b0 = np.clip(np.asarray(barycentric[0], dtype=np.float32), 0.0, np.nextafter(1.0, 0.0))
    b1 = np.clip(np.asarray(barycentric[1], dtype=np.float32), 0.0, np.nextafter(1.0, 0.0))
    i = np.floor(b0 * float(bins)).astype(np.int64)
    j = np.floor(b1 * float(bins)).astype(np.int64)
    return np.asarray(face_id, dtype=np.int64) * int(bins * bins) + i * int(bins) + j


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "faces"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value_f = float(value)
            return value_f if math.isfinite(value_f) else None
        if isinstance(value, np.bool_):
            return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.sample_stride) <= 0:
        raise ValueError("--sample_stride must be > 0")
    if int(args.max_files) < 0:
        raise ValueError("--max_files must be >= 0")
    if args.output_json.resolve() == args.output_md.resolve():
        raise ValueError("--output_json and --output_md must be different paths")

    evidence_dir = args.teacher_evidence_dir
    view_dir, all_paths = _evidence_views(evidence_dir)
    view_paths = all_paths[: int(args.max_files)] if int(args.max_files) > 0 else all_paths
    if not view_paths:
        raise FileNotFoundError(f"no NPZ view files selected under {view_dir}")

    first_fields = _load_npz_fields(view_paths[0])
    teacher_parent_rgb_key = _select_first_existing(
        first_fields,
        RAW_TEACHER_RGB_KEYS,
        label="teacher-parent residual RGB field",
        path=view_paths[0],
    )
    fit_signal_rgb_key = _select_first_existing(
        first_fields,
        FIT_SIGNAL_RGB_KEYS,
        label="fit/projection residual RGB field",
        path=view_paths[0],
    )
    has_raw_and_fit = teacher_parent_rgb_key != fit_signal_rgb_key
    has_barycentric_carrier = "barycentric" in first_fields and "barycentric_valid" in first_fields

    top_support = _read_top_support_faces(evidence_dir)
    top_faces = top_support.get("faces", set())
    top_face_array = np.asarray(sorted(top_faces), dtype=np.int64) if top_faces else np.empty((0,), dtype=np.int64)

    raw_mag_stats = RunningScalar()
    fit_mag_stats = RunningScalar()
    delta_l1_field_stats = RunningScalar()
    fit_channel_count = 0
    fit_channel_sum = np.zeros((3,), dtype=np.float64)
    fit_channel_sq_sum = np.zeros((3,), dtype=np.float64)
    edge_abs_sum = 0.0
    edge_sq_sum = 0.0
    edge_pair_count = 0

    face_pixel_counts: dict[int, int] = {}
    active_face_pixel_counts: dict[int, int] = {}
    sign_sums_by_face: dict[int, list[float]] = {}
    sign_counts_by_face: dict[int, list[int]] = {}
    face_carrier = CarrierAccumulator(
        "face_constant",
        "Least-squares projection onto one RGB residual per visible face_id using the same sampled cache.",
    )
    bary_carrier = (
        CarrierAccumulator(
            f"face_barycentric_bin{CARRIER_BARY_BINS}",
            (
                "Least-squares projection onto visible face_id plus coarse barycentric bins "
                f"({CARRIER_BARY_BINS}x{CARRIER_BARY_BINS}) using the same sampled cache."
            ),
        )
        if has_barycentric_carrier
        else None
    )

    total_pixels = 0
    valid_face_pixels = 0
    alpha_available_files = 0
    alpha_positive_pixels = 0
    fit_active_pixels = 0
    raw_active_pixels = 0
    active_signal_l1_sum = 0.0
    top_active_pixels = 0
    top_active_signal_l1_sum = 0.0
    rgb_render_available_files = 0
    rgb_gt_available_files = 0
    clip_channel_count = 0
    clip_pixel_count = 0
    clip_denominator_channels = 0
    clip_denominator_pixels = 0
    raw_fit_changed_channel_count = 0
    raw_fit_changed_pixel_count = 0
    raw_fit_denominator_channels = 0
    raw_fit_denominator_pixels = 0
    residual_saturation_channel_count = 0
    residual_saturation_denominator_channels = 0
    view_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in view_paths:
        fields = _load_npz_fields(path)
        required = ["face_id", teacher_parent_rgb_key, fit_signal_rgb_key]
        _require_fields(fields, sorted(set(required)), path=path)
        if has_barycentric_carrier and not {"barycentric", "barycentric_valid"}.issubset(fields):
            raise KeyError(
                f"{path} missing barycentric carrier fields although {view_paths[0]} had them; "
                "expected both 'barycentric' and 'barycentric_valid'"
            )

        with np.load(path, allow_pickle=False) as z:
            face_full = _as_2d(z["face_id"], key="face_id", path=path)
            hw = tuple(int(x) for x in face_full.shape)
            face = np.asarray(face_full[:: int(args.sample_stride), :: int(args.sample_stride)], dtype=np.int64)
            raw_rgb = _as_rgb_chw(
                z[teacher_parent_rgb_key],
                key=teacher_parent_rgb_key,
                path=path,
                expected_hw=hw,
            )[:, :: int(args.sample_stride), :: int(args.sample_stride)]
            fit_rgb = _as_rgb_chw(
                z[fit_signal_rgb_key],
                key=fit_signal_rgb_key,
                path=path,
                expected_hw=hw,
            )[:, :: int(args.sample_stride), :: int(args.sample_stride)]
            raw_mag = np.mean(np.abs(raw_rgb), axis=0).astype(np.float32)
            fit_mag = np.mean(np.abs(fit_rgb), axis=0).astype(np.float32)
            finite = np.isfinite(raw_mag) & np.isfinite(fit_mag) & np.isfinite(fit_rgb).all(axis=0)
            valid = (face >= 0) & finite

            if "alpha" in fields:
                alpha = _shape_matches_2d(z["alpha"], key="alpha", path=path, expected_hw=hw)[
                    :: int(args.sample_stride), :: int(args.sample_stride)
                ].astype(np.float32)
                alpha_positive = np.isfinite(alpha) & (alpha > 0.0)
                alpha_available_files += 1
                alpha_positive_pixels += int(np.sum(valid & alpha_positive))

            if "teacher_parent_delta_l1" in fields:
                delta_l1 = _shape_matches_2d(
                    z["teacher_parent_delta_l1"],
                    key="teacher_parent_delta_l1",
                    path=path,
                    expected_hw=hw,
                )[:: int(args.sample_stride), :: int(args.sample_stride)]
                delta_l1_field_stats.update(delta_l1[valid])

            gt_correction = None
            if "rgb_render" in fields:
                rgb_render_available_files += 1
                rgb_render = _as_rgb_chw(z["rgb_render"], key="rgb_render", path=path, expected_hw=hw)[
                    :, :: int(args.sample_stride), :: int(args.sample_stride)
                ]
                applied = rgb_render + raw_rgb
                clip_channels = ((applied < 0.0) | (applied > 1.0)) & valid[None, :, :]
                clip_channel_count += int(np.sum(clip_channels))
                clip_pixel_count += int(np.sum(np.any(clip_channels, axis=0)))
                clip_denominator_channels += int(3 * np.sum(valid))
                clip_denominator_pixels += int(np.sum(valid))

                if "rgb_gt" in fields:
                    rgb_gt_available_files += 1
                    rgb_gt = _as_rgb_chw(z["rgb_gt"], key="rgb_gt", path=path, expected_hw=hw)[
                        :, :: int(args.sample_stride), :: int(args.sample_stride)
                    ]
                    gt_correction = (rgb_gt - rgb_render).astype(np.float32)

            if has_raw_and_fit:
                changed = (np.abs(raw_rgb - fit_rgb) > 1.0e-6) & valid[None, :, :]
                raw_fit_changed_channel_count += int(np.sum(changed))
                raw_fit_changed_pixel_count += int(np.sum(np.any(changed, axis=0)))
                raw_fit_denominator_channels += int(3 * np.sum(valid))
                raw_fit_denominator_pixels += int(np.sum(valid))

            residual_saturation = (np.abs(raw_rgb) >= 0.999) & valid[None, :, :]
            residual_saturation_channel_count += int(np.sum(residual_saturation))
            residual_saturation_denominator_channels += int(3 * np.sum(valid))

            total_pixels += int(face.size)
            valid_count = int(np.sum(valid))
            valid_face_pixels += valid_count
            raw_active = valid & (raw_mag > EPS)
            fit_active = valid & (fit_mag > EPS)
            raw_active_pixels += int(np.sum(raw_active))
            fit_active_pixels += int(np.sum(fit_active))
            raw_mag_stats.update(raw_mag[valid])
            fit_mag_stats.update(fit_mag[valid])
            active_signal_l1_sum += float(np.sum(fit_mag[fit_active], dtype=np.float64))

            if valid_count > 0:
                valid_faces_flat = face[valid].reshape(-1)
                fit_valid = fit_rgb[:, valid].reshape(3, -1)
                _update_count_dict(face_pixel_counts, valid_faces_flat)
                _update_sign_consistency(sign_sums_by_face, sign_counts_by_face, valid_faces_flat, fit_valid)
                gt_valid = gt_correction[:, valid].reshape(3, -1) if gt_correction is not None else None
                face_carrier.update(valid_faces_flat, fit_valid, gt_valid)

                fit_channel_count += int(fit_valid.shape[1])
                fit_channel_sum += np.sum(fit_valid.astype(np.float64), axis=1)
                fit_channel_sq_sum += np.sum(fit_valid.astype(np.float64) * fit_valid.astype(np.float64), axis=1)

            if int(np.sum(fit_active)) > 0:
                active_faces = face[fit_active].reshape(-1)
                _update_count_dict(active_face_pixel_counts, active_faces)

            if top_face_array.size > 0 and int(np.sum(fit_active)) > 0:
                top_mask = fit_active & np.isin(face, top_face_array)
                top_active_pixels += int(np.sum(top_mask))
                top_active_signal_l1_sum += float(np.sum(fit_mag[top_mask], dtype=np.float64))

            if has_barycentric_carrier and bary_carrier is not None:
                bary_full = _as_rgb_chw(z["barycentric"], key="barycentric", path=path, expected_hw=hw)
                bary_valid_full = _shape_matches_2d(
                    z["barycentric_valid"],
                    key="barycentric_valid",
                    path=path,
                    expected_hw=hw,
                )
                bary = bary_full[:, :: int(args.sample_stride), :: int(args.sample_stride)]
                bary_valid = bary_valid_full[:: int(args.sample_stride), :: int(args.sample_stride)].astype(bool)
                carrier_valid = valid & bary_valid & np.isfinite(bary).all(axis=0)
                if np.any(carrier_valid):
                    bary_keys = _carrier_keys_barycentric(face[carrier_valid], bary[:, carrier_valid], bins=CARRIER_BARY_BINS)
                    gt_carrier = (
                        gt_correction[:, carrier_valid].reshape(3, -1) if gt_correction is not None else None
                    )
                    bary_carrier.update(
                        bary_keys,
                        fit_rgb[:, carrier_valid].reshape(3, -1),
                        gt_carrier,
                    )

            gx_valid = valid[:, 1:] & valid[:, :-1]
            gy_valid = valid[1:, :] & valid[:-1, :]
            if np.any(gx_valid):
                gx = np.abs(fit_mag[:, 1:] - fit_mag[:, :-1])[gx_valid]
                edge_abs_sum += float(np.sum(gx, dtype=np.float64))
                edge_sq_sum += float(np.sum(gx.astype(np.float64) * gx.astype(np.float64)))
                edge_pair_count += int(gx.size)
            if np.any(gy_valid):
                gy = np.abs(fit_mag[1:, :] - fit_mag[:-1, :])[gy_valid]
                edge_abs_sum += float(np.sum(gy, dtype=np.float64))
                edge_sq_sum += float(np.sum(gy.astype(np.float64) * gy.astype(np.float64)))
                edge_pair_count += int(gy.size)

            view_clip_pixel_fraction = None
            if "rgb_render" in fields:
                view_clip = ((rgb_render + raw_rgb < 0.0) | (rgb_render + raw_rgb > 1.0)) & valid[None, :, :]
                view_clip_pixel_fraction = _safe_fraction(int(np.sum(np.any(view_clip, axis=0))), valid_count)

            view_summaries.append(
                {
                    "view": path.stem,
                    "path": str(path),
                    "sampled_pixels": int(face.size),
                    "valid_face_pixels": valid_count,
                    "valid_face_fraction": _safe_fraction(valid_count, face.size),
                    "raw_active_fraction_of_valid": _safe_fraction(int(np.sum(raw_active)), valid_count),
                    "fit_active_fraction_of_valid": _safe_fraction(int(np.sum(fit_active)), valid_count),
                    "mean_teacher_parent_l1": float(np.mean(raw_mag[valid])) if valid_count else None,
                    "mean_fit_signal_l1": float(np.mean(fit_mag[valid])) if valid_count else None,
                    "clip_pixel_fraction": view_clip_pixel_fraction,
                    "has_rgb_render": bool("rgb_render" in fields),
                    "has_rgb_gt": bool("rgb_gt" in fields),
                    "has_alpha": bool("alpha" in fields),
                }
            )

    if valid_face_pixels <= 0:
        raise RuntimeError("no sampled pixels had valid face_id >= 0 and finite teacher residual values")

    fit_channel_mean = fit_channel_sum / max(float(fit_channel_count), 1.0)
    fit_channel_var = np.maximum(0.0, fit_channel_sq_sum / max(float(fit_channel_count), 1.0) - fit_channel_mean * fit_channel_mean)
    edge_mean_abs = edge_abs_sum / float(edge_pair_count) if edge_pair_count > 0 else None
    edge_rms = math.sqrt(edge_sq_sum / float(edge_pair_count)) if edge_pair_count > 0 else None
    fit_mag_mean = fit_mag_stats.as_dict().get("mean")

    sign_total_weight = 0
    sign_weighted_sum = 0.0
    sign_bin_count = 0
    global_sign_sum = np.zeros((3,), dtype=np.float64)
    global_sign_count = np.zeros((3,), dtype=np.float64)
    for face, counts in sign_counts_by_face.items():
        sums = sign_sums_by_face.get(face, [0.0, 0.0, 0.0])
        for channel in range(3):
            count = int(counts[channel])
            if count <= 0:
                continue
            consistency = abs(float(sums[channel])) / float(count)
            sign_weighted_sum += consistency * float(count)
            sign_total_weight += count
            sign_bin_count += 1
            global_sign_sum[channel] += float(sums[channel])
            global_sign_count[channel] += float(count)

    face_result = face_carrier.finalize(phasej_flowers_psnr=float(args.phasej_flowers_psnr))
    bary_result = bary_carrier.finalize(phasej_flowers_psnr=float(args.phasej_flowers_psnr)) if bary_carrier else None
    selected_carrier = bary_result if bary_result is not None and bary_result.get("sample_pixels", 0) else face_result

    summary_files = {
        "teacher_surface_evidence_summary": _read_json_if_present(evidence_dir / "teacher_surface_evidence_summary.json"),
        "surface_evidence_summary": _read_json_if_present(evidence_dir / "surface_evidence_summary.json"),
    }
    source_summary = next((value for value in summary_files.values() if isinstance(value, dict)), {}) or {}

    result = {
        "operator": "analyze_v169_teacher_signal_projection",
        "test_usage": "none",
        "teacher_evidence_dir": str(evidence_dir),
        "view_dir": str(view_dir),
        "selected_files": int(len(view_paths)),
        "available_files": int(len(all_paths)),
        "max_files": int(args.max_files),
        "sample_stride": int(args.sample_stride),
        "source_keys": {
            "teacher_parent_rgb_key": teacher_parent_rgb_key,
            "fit_signal_rgb_key": fit_signal_rgb_key,
            "teacher_parent_delta_l1_key": "teacher_parent_delta_l1" if delta_l1_field_stats.count > 0 else None,
        },
        "phasej_flowers_gate_reference": {
            "PSNR": float(args.phasej_flowers_psnr),
            "SSIM": float(args.phasej_flowers_ssim),
            "LPIPS": float(args.phasej_flowers_lpips),
            "note": "Only PSNR can be proxied from aggregate RGB NPZ fields; SSIM/LPIPS are reference constants.",
        },
        "teacher_parent_residual_magnitude": {
            "source_rgb_key": teacher_parent_rgb_key,
            "mean_abs_rgb_l1": raw_mag_stats.as_dict(),
            "teacher_parent_delta_l1_field": delta_l1_field_stats.as_dict() if delta_l1_field_stats.count > 0 else None,
            "active_fraction_of_valid": _safe_fraction(raw_active_pixels, valid_face_pixels),
        },
        "fit_signal_residual_magnitude": {
            "source_rgb_key": fit_signal_rgb_key,
            "mean_abs_rgb_l1": fit_mag_stats.as_dict(),
            "active_fraction_of_valid": _safe_fraction(fit_active_pixels, valid_face_pixels),
        },
        "clipping_fraction": {
            "source_rgb_key": teacher_parent_rgb_key,
            "rgb_render_files": int(rgb_render_available_files),
            "channel_fraction_rgb_render_plus_teacher_parent_outside_unit_range": _safe_fraction(
                clip_channel_count,
                clip_denominator_channels,
            ),
            "pixel_fraction_any_channel_rgb_render_plus_teacher_parent_outside_unit_range": _safe_fraction(
                clip_pixel_count,
                clip_denominator_pixels,
            ),
            "residual_channel_fraction_abs_ge_0_999": _safe_fraction(
                residual_saturation_channel_count,
                residual_saturation_denominator_channels,
            ),
            "raw_to_fit_changed_channel_fraction": _safe_fraction(
                raw_fit_changed_channel_count,
                raw_fit_denominator_channels,
            )
            if has_raw_and_fit
            else None,
            "raw_to_fit_changed_pixel_fraction": _safe_fraction(
                raw_fit_changed_pixel_count,
                raw_fit_denominator_pixels,
            )
            if has_raw_and_fit
            else None,
            "raw_to_fit_change_interpretation": "masking_or_clipping_proxy" if has_raw_and_fit else "unavailable",
        },
        "edge_gradient_energy_proxy": {
            "source_rgb_key": fit_signal_rgb_key,
            "pair_count": int(edge_pair_count),
            "mean_abs_l1_gradient": edge_mean_abs,
            "rms_l1_gradient": edge_rms,
            "mean_abs_gradient_over_mean_l1": (
                edge_mean_abs / float(fit_mag_mean) if edge_mean_abs is not None and fit_mag_mean else None
            ),
        },
        "support_count_coverage": {
            "sampled_pixels_total": int(total_pixels),
            "valid_face_pixels": int(valid_face_pixels),
            "valid_face_fraction": _safe_fraction(valid_face_pixels, total_pixels),
            "alpha_available_files": int(alpha_available_files),
            "alpha_positive_fraction_of_valid": _safe_fraction(alpha_positive_pixels, valid_face_pixels)
            if alpha_available_files
            else None,
            "fit_signal_active_pixels": int(fit_active_pixels),
            "fit_signal_active_fraction_of_valid": _safe_fraction(fit_active_pixels, valid_face_pixels),
            "unique_faces_observed": int(len(face_pixel_counts)),
            "unique_faces_with_active_fit_signal": int(len(active_face_pixel_counts)),
            "face_pixel_count_stats": _count_stats(list(face_pixel_counts.values())),
            "active_face_pixel_count_stats": _count_stats(list(active_face_pixel_counts.values())),
            "top_residual_supports_csv": {
                "available": bool(top_support.get("available", False)),
                "path": top_support.get("path"),
                "rows": int(top_support.get("rows", 0)),
                "unique_faces": int(len(top_faces)),
                "active_pixel_fraction_in_top_supports": _safe_fraction(top_active_pixels, fit_active_pixels),
                "active_l1_fraction_in_top_supports": _safe_fraction(top_active_signal_l1_sum, active_signal_l1_sum),
            },
        },
        "sign_consistency_proxy": {
            "source_rgb_key": fit_signal_rgb_key,
            "active_threshold_abs_rgb": EPS,
            "face_channel_bins": int(sign_bin_count),
            "weighted_mean_abs_sign_balance": _safe_fraction(sign_weighted_sum, sign_total_weight),
            "global_channel_abs_sign_balance": [
                _safe_fraction(abs(float(global_sign_sum[channel])), float(global_sign_count[channel]))
                for channel in range(3)
            ],
        },
        "residual_variance": {
            "source_rgb_key": fit_signal_rgb_key,
            "channel_count_per_rgb_channel": int(fit_channel_count),
            "channel_mean": [float(x) for x in fit_channel_mean.tolist()],
            "channel_variance": [float(x) for x in fit_channel_var.tolist()],
            "channel_std": [float(math.sqrt(float(x))) for x in fit_channel_var.tolist()],
            "l1_magnitude_variance": fit_mag_stats.as_dict().get("variance"),
            "mean_l2_per_pixel": float(np.sum(fit_channel_sq_sum) / max(float(fit_channel_count), 1.0)),
        },
        "current_carrier_upper_bound_proxy": {
            "selected": selected_carrier,
            "face_constant": face_result,
            "face_barycentric_bin": bary_result,
            "feasible_from_npz_fields": {
                "face_id": True,
                "barycentric": bool(has_barycentric_carrier),
                "rgb_render": bool(rgb_render_available_files > 0),
                "rgb_gt": bool(rgb_gt_available_files > 0),
            },
        },
        "view_summaries": view_summaries,
        "source_summary": {
            "scene": source_summary.get("scene"),
            "num_views": source_summary.get("num_views"),
            "selection_mode": source_summary.get("selection_mode"),
            "mask_target": source_summary.get("mask_target"),
            "parent_source": source_summary.get("parent_source"),
            "parent_render_dir": source_summary.get("parent_render_dir"),
        },
        "warnings": warnings,
        "assumptions": [
            "All diagnostics are computed from existing NPZ fields; no training or rendering is launched.",
            "Current-carrier quality is a same-cache projection proxy, not held-out test quality.",
            "SSIM and LPIPS gates are recorded as constants only; this storage-light utility does not reconstruct images.",
        ],
    }
    return _json_safe(result)


def _fmt(value: Any, digits: int = 6, signed: bool = False) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if signed:
            return f"{value:+.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    teacher_parent = result["teacher_parent_residual_magnitude"]
    fit_signal = result["fit_signal_residual_magnitude"]
    clip = result["clipping_fraction"]
    edge = result["edge_gradient_energy_proxy"]
    coverage = result["support_count_coverage"]
    sign = result["sign_consistency_proxy"]
    variance = result["residual_variance"]
    carrier = result["current_carrier_upper_bound_proxy"]["selected"]
    quality = carrier.get("same_cache_quality_proxy", {})

    rows = [
        ("teacher-parent mean L1", _fmt(teacher_parent["mean_abs_rgb_l1"].get("mean"))),
        ("fit-signal mean L1", _fmt(fit_signal["mean_abs_rgb_l1"].get("mean"))),
        ("RGB-domain clip pixel fraction", _fmt(clip.get("pixel_fraction_any_channel_rgb_render_plus_teacher_parent_outside_unit_range"))),
        ("raw-to-fit changed pixel fraction", _fmt(clip.get("raw_to_fit_changed_pixel_fraction"))),
        ("edge mean abs L1 gradient", _fmt(edge.get("mean_abs_l1_gradient"))),
        ("valid face coverage", _fmt(coverage.get("valid_face_fraction"))),
        ("active fit-signal coverage", _fmt(coverage.get("fit_signal_active_fraction_of_valid"))),
        ("unique observed faces", _fmt(coverage.get("unique_faces_observed"))),
        ("weighted sign consistency", _fmt(sign.get("weighted_mean_abs_sign_balance"))),
        ("fit-signal L1 variance", _fmt(variance.get("l1_magnitude_variance"))),
        ("carrier mode", carrier.get("mode", "missing")),
        ("carrier projected signal L2 fraction", _fmt(carrier.get("projected_signal_l2_fraction"))),
        ("carrier projected PSNR proxy", _fmt(quality.get("projected_carrier_psnr_proxy"))),
        ("carrier PSNR proxy minus Phase-J flowers", _fmt(quality.get("projected_minus_phasej_flowers_psnr"), signed=True)),
    ]

    lines = [
        "# v169 Teacher Signal Projection Diagnostics",
        "",
        f"- teacher evidence dir: `{result['teacher_evidence_dir']}`",
        f"- selected files: `{result['selected_files']}` / `{result['available_files']}`",
        f"- sample stride: `{result['sample_stride']}`",
        f"- teacher-parent RGB key: `{result['source_keys']['teacher_parent_rgb_key']}`",
        f"- fit/projection RGB key: `{result['source_keys']['fit_signal_rgb_key']}`",
        (
            "- Phase-J flowers gate constants: "
            f"PSNR `{_fmt(result['phasej_flowers_gate_reference']['PSNR'])}`, "
            f"SSIM `{_fmt(result['phasej_flowers_gate_reference']['SSIM'])}`, "
            f"LPIPS `{_fmt(result['phasej_flowers_gate_reference']['LPIPS'])}`"
        ),
        "",
        "## Summary",
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | `{value}` |" for name, value in rows)
    lines.extend(
        [
            "",
            "## Support Coverage",
            "",
            f"- valid face pixels: `{coverage['valid_face_pixels']}` / `{coverage['sampled_pixels_total']}`",
            f"- active fit-signal pixels: `{coverage['fit_signal_active_pixels']}`",
            f"- active faces: `{coverage['unique_faces_with_active_fit_signal']}`",
            (
                "- top support active L1 fraction: "
                f"`{_fmt(coverage['top_residual_supports_csv'].get('active_l1_fraction_in_top_supports'))}`"
            ),
            "",
            "## Carrier Proxy",
            "",
            f"- selected carrier mode: `{carrier.get('mode')}`",
            f"- carrier count: `{carrier.get('carrier_count')}`",
            f"- projected signal L2 fraction: `{_fmt(carrier.get('projected_signal_l2_fraction'))}`",
            f"- same-cache parent PSNR proxy: `{_fmt(quality.get('parent_psnr_proxy'))}`",
            f"- same-cache direct signal PSNR proxy: `{_fmt(quality.get('direct_signal_psnr_proxy'))}`",
            f"- same-cache projected carrier PSNR proxy: `{_fmt(quality.get('projected_carrier_psnr_proxy'))}`",
            f"- PSNR-only proxy beats Phase-J flowers: `{_fmt(quality.get('beats_phasej_flowers_psnr_proxy'))}`",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result.get("assumptions", []))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if NUMPY_IMPORT_ERROR is not None:
        raise SystemExit(f"error: NumPy is required to read NPZ evidence files: {NUMPY_IMPORT_ERROR}")
    try:
        result = analyze(args)
        _write_text_atomic(args.output_json, json.dumps(result, indent=2, sort_keys=True) + "\n")
        _write_text_atomic(args.output_md, render_markdown(result))
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

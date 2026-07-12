from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class CameraRecord:
    idx: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    camera_center: tuple[float, float, float]
    world_view_transform: tuple[tuple[float, float, float, float], ...]

    @property
    def center_tensor(self) -> torch.Tensor:
        return torch.tensor(self.camera_center, dtype=torch.float32)

    @property
    def view_matrix(self) -> torch.Tensor:
        return torch.tensor(self.world_view_transform, dtype=torch.float32)

    @property
    def c2w_matrix(self) -> torch.Tensor:
        return torch.inverse(self.view_matrix)

    def view_direction(self) -> torch.Tensor:
        c2w = self.c2w_matrix
        direction = c2w[2, :3]
        return F.normalize(direction, dim=0, eps=1e-8)

    def scaled_intrinsics(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        sx = float(image_width) / max(float(self.width), 1.0)
        sy = float(image_height) / max(float(self.height), 1.0)
        fx = float(self.fx) * sx
        fy = float(self.fy) * sy
        cx = (float(image_width) - 1.0) * 0.5
        cy = (float(image_height) - 1.0) * 0.5
        return fx, fy, cx, cy


@dataclass
class FrameRecord:
    idx: int
    name: str
    render_path: Path
    gt_path: Path
    depth_path: Path
    camera: CameraRecord
    # Edit-aware ECR (Route A, 2026-07-12): optional per-view evidence
    # validity mask (1 = valid, 0 = stale, e.g. pixels imaging deleted
    # faces). None (the default, all pre-edit caches) = fully valid;
    # behavior is bit-identical when absent.
    mask_path: "Path | None" = None


def _image_key(path: Path) -> str:
    return path.stem


def read_image_tensor(path: Path, device: torch.device | str = "cpu") -> torch.Tensor:
    last_error: Exception | None = None
    for attempt in range(30):
        try:
            with Image.open(path) as raw:
                img = raw.convert("RGB")
                img.load()
            return TF.to_tensor(img).to(device=device, dtype=torch.float32)
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt == 29:
                break
            time.sleep(min(1.0, 0.2 * float(attempt + 1)))
    raise OSError(f"failed to read image after retries: {path}") from last_error


def read_depth_tensor(path: Path, device: torch.device | str = "cpu") -> torch.Tensor:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return torch.from_numpy(depth).to(device=device, dtype=torch.float32)


def _resize_chw(image: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(image.unsqueeze(0), size=size, **kwargs).squeeze(0)


def _resize_hw(image: torch.Tensor, size: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    return _resize_chw(image.unsqueeze(0), size=size, mode=mode).squeeze(0)


def _evidence_size(height: int, width: int, max_side: int) -> tuple[int, int] | None:
    max_side = int(max_side)
    if max_side <= 0:
        return None
    current = max(int(height), int(width))
    if current <= max_side:
        return None
    scale = float(max_side) / max(float(current), 1.0)
    out_h = max(8, int(round(float(height) * scale)))
    out_w = max(8, int(round(float(width) * scale)))
    if out_h >= int(height) and out_w >= int(width):
        return None
    return out_h, out_w


def save_image_tensor(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    TF.to_pil_image(image).save(path)


def save_camera_index(cameras: Sequence[CameraRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for cam in cameras:
        payload.append(
            {
                "idx": int(cam.idx),
                "image_name": cam.image_name,
                "width": int(cam.width),
                "height": int(cam.height),
                "fx": float(cam.fx),
                "fy": float(cam.fy),
                "camera_center": [float(x) for x in cam.camera_center],
                "world_view_transform": [[float(v) for v in row] for row in cam.world_view_transform],
            }
        )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_camera_index(path: Path) -> list[CameraRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras = []
    for row in payload:
        cameras.append(
            CameraRecord(
                idx=int(row["idx"]),
                image_name=str(row.get("image_name", row.get("img_name", row["idx"]))),
                width=int(row["width"]),
                height=int(row["height"]),
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                camera_center=tuple(float(x) for x in row["camera_center"]),
                world_view_transform=tuple(tuple(float(v) for v in line) for line in row["world_view_transform"]),
            )
        )
    return cameras


def load_split_frames(model_path: Path, split: str, method: str) -> list[FrameRecord]:
    method_dir = model_path / split / method
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    depth_dir = method_dir / "depths"
    camera_index_path = method_dir / "camera_index.json"
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Missing render directory: {render_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing gt directory: {gt_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    if not camera_index_path.is_file():
        raise FileNotFoundError(f"Missing camera index: {camera_index_path}")
    cameras = load_camera_index(camera_index_path)
    render_paths = sorted(p for p in render_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    frames: list[FrameRecord] = []
    if len(render_paths) != len(cameras):
        raise RuntimeError(
            f"Split {split}/{method} has {len(render_paths)} renders but {len(cameras)} cameras in {camera_index_path}"
        )
    for idx, render_path in enumerate(render_paths):
        key = _image_key(render_path)
        gt_path = gt_dir / f"{key}.png"
        if not gt_path.is_file():
            gt_candidates = sorted(gt_dir.glob(f"{key}.*"))
            if not gt_candidates:
                raise FileNotFoundError(f"Missing GT for {render_path}")
            gt_path = gt_candidates[0]
        depth_path = depth_dir / f"{key}.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing depth for {render_path}: {depth_path}")
        frames.append(
            FrameRecord(
                idx=idx,
                name=key,
                render_path=render_path,
                gt_path=gt_path,
                depth_path=depth_path,
                camera=cameras[idx],
            )
        )
    return frames


def select_support_frames(
    target: FrameRecord,
    support_frames: Sequence[FrameRecord],
    k: int,
    exclude_names: Iterable[str] = (),
    direction_weight: float = 0.35,
    distance_scale: float | None = None,
) -> list[tuple[FrameRecord, float]]:
    excluded = set(exclude_names)
    target_center = target.camera.center_tensor
    target_dir = target.camera.view_direction()
    centers = torch.stack([frame.camera.center_tensor for frame in support_frames]) if support_frames else torch.empty(0, 3)
    if distance_scale is None:
        if len(support_frames) > 1:
            median = torch.median(torch.linalg.norm(centers - torch.mean(centers, dim=0), dim=1))
            distance_scale = max(float(median.item()), 1e-3)
        else:
            distance_scale = 1.0
    scored: list[tuple[float, FrameRecord, float]] = []
    for frame in support_frames:
        if frame.name in excluded or frame.camera.image_name in excluded:
            continue
        distance = torch.linalg.norm(frame.camera.center_tensor - target_center).item()
        direction = frame.camera.view_direction()
        angle_cost = 1.0 - float(torch.clamp(torch.dot(direction, target_dir), -1.0, 1.0).item())
        score = (distance / max(float(distance_scale), 1e-6)) + float(direction_weight) * angle_cost
        weight = math.exp(-score)
        scored.append((score, frame, weight))
    scored.sort(key=lambda x: x[0])
    return [(frame, weight) for _, frame, weight in scored[: max(int(k), 0)]]


def _make_target_world_grid(
    camera: CameraRecord,
    depth: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    height, width = depth.shape
    fx, fy, cx, cy = camera.scaled_intrinsics(width, height)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth.to(device=device, dtype=torch.float32)
    x = (xs - float(cx)) / max(float(fx), 1e-6) * z
    y = (ys - float(cy)) / max(float(fy), 1e-6) * z
    ones = torch.ones_like(z)
    cam_points = torch.stack([x, y, z, ones], dim=-1).reshape(-1, 4)
    inv_view = torch.inverse(camera.view_matrix.to(device=device, dtype=torch.float32))
    return cam_points @ inv_view


def _frame_valid_mask(loader, frame) -> "torch.Tensor | None":
    """Edit-aware ECR: a support frame's evidence-validity mask via the
    loader (None for all pre-edit frames -> bit-identical behavior)."""
    mask_path = getattr(frame, "mask_path", None)
    if not mask_path:
        return None
    return loader.mask(str(mask_path))


def warp_support_residual(
    target: FrameRecord,
    support: FrameRecord,
    target_depth: torch.Tensor,
    support_depth: torch.Tensor,
    support_residual: torch.Tensor,
    *,
    depth_abs_tol: float = 0.02,
    depth_rel_tol: float = 0.03,
    target_world_grid: torch.Tensor | None = None,
    device: torch.device | str = "cuda",
    support_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    target_depth = target_depth.to(device=device, dtype=torch.float32)
    support_depth = support_depth.to(device=device, dtype=torch.float32)
    support_residual = support_residual.to(device=device, dtype=torch.float32)

    target_h, target_w = target_depth.shape
    support_h, support_w = support_depth.shape
    world_points = (
        target_world_grid.to(device=device, dtype=torch.float32)
        if target_world_grid is not None
        else _make_target_world_grid(target.camera, target_depth, device=device)
    )
    support_view = support.camera.view_matrix.to(device=device, dtype=torch.float32)
    support_cam = (world_points @ support_view).reshape(target_h, target_w, 4)
    z = support_cam[..., 2]
    fx, fy, cx, cy = support.camera.scaled_intrinsics(support_w, support_h)
    u = float(fx) * (support_cam[..., 0] / torch.clamp(z, min=1e-6)) + float(cx)
    v = float(fy) * (support_cam[..., 1] / torch.clamp(z, min=1e-6)) + float(cy)
    x_norm = (u / max(float(support_w - 1), 1.0)) * 2.0 - 1.0
    y_norm = (v / max(float(support_h - 1), 1.0)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    sampled_residual = F.grid_sample(
        support_residual.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    sampled_depth = F.grid_sample(
        support_depth[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    in_bounds = (x_norm >= -1.0) & (x_norm <= 1.0) & (y_norm >= -1.0) & (y_norm <= 1.0)
    valid_depth = (target_depth > 1e-6) & (sampled_depth > 1e-6) & (z > 1e-6)
    depth_error = torch.abs(z - sampled_depth)
    tolerance = float(depth_abs_tol) + float(depth_rel_tol) * torch.clamp(torch.abs(sampled_depth), min=1e-6)
    depth_consistent = depth_error <= tolerance
    confidence = torch.exp(-depth_error / torch.clamp(tolerance, min=1e-6))
    confidence = confidence * (in_bounds & valid_depth & depth_consistent).to(torch.float32)
    if support_valid is not None:
        # Edit-aware ECR: evidence-validity mask sampled with the SAME grid;
        # multiplicative (<= 1) so it can only REMOVE evidence, never inject.
        sampled_valid = F.grid_sample(
            support_valid.to(device=device, dtype=torch.float32)[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0).squeeze(0)
        confidence = confidence * sampled_valid
    return sampled_residual, confidence


class FrameLoader:
    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)

    @lru_cache(maxsize=96)
    def render(self, path: str) -> torch.Tensor:
        return read_image_tensor(Path(path), device=self.device)

    @lru_cache(maxsize=96)
    def gt(self, path: str) -> torch.Tensor:
        return read_image_tensor(Path(path), device=self.device)

    @lru_cache(maxsize=96)
    def depth(self, path: str) -> torch.Tensor:
        return read_depth_tensor(Path(path), device=self.device)

    @lru_cache(maxsize=96)
    def mask(self, path: str) -> torch.Tensor:
        # edit-aware evidence-validity mask: [H,W] float in [0,1]
        return read_image_tensor(Path(path), device=self.device)[0]

    def residual(self, frame: FrameRecord, residual_clip: float) -> torch.Tensor:
        residual = self.gt(str(frame.gt_path)) - self.render(str(frame.render_path))
        if residual_clip > 0:
            residual = torch.clamp(residual, -float(residual_clip), float(residual_clip))
        return residual


@dataclass
class EvidenceSignal:
    base: torch.Tensor
    signal: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    support_names: list[str]
    support_count: torch.Tensor | None = None
    residual_std: torch.Tensor | None = None


def _image_edge_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = torch.mean(image, dim=0, keepdim=True)
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, 1:] = gray[:, :, 1:] - gray[:, :, :-1]
    gy[:, 1:, :] = gray[:, 1:, :] - gray[:, :-1, :]
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _edge_acceptance_mask(
    image: torch.Tensor,
    valid: torch.Tensor,
    *,
    quantile: float = -1.0,
    min_edge: float = 0.0,
    dilate: int = 0,
) -> tuple[torch.Tensor, float]:
    edge = _image_edge_magnitude(image)
    q = float(quantile)
    threshold = float(max(min_edge, 0.0))
    if q >= 0.0:
        q = min(max(q, 0.0), 0.999)
        valid_flat = valid.squeeze(0).bool()
        values = edge.squeeze(0)[valid_flat] if bool(valid_flat.any().item()) else edge.reshape(-1)
        if values.numel() > 0:
            threshold = max(threshold, float(torch.quantile(values.detach().float(), q).item()))
    accept = edge >= float(threshold)
    radius = max(int(dilate), 0)
    if radius > 0:
        kernel = 2 * radius + 1
        accept = (
            F.max_pool2d(
                accept.to(dtype=torch.float32).unsqueeze(0),
                kernel_size=kernel,
                stride=1,
                padding=radius,
            ).squeeze(0)
            > 0.0
        )
    return accept.to(device=image.device), float(threshold)


@dataclass
class BenefitCalibrator:
    confidence_edges: tuple[float, ...]
    magnitude_edges: tuple[float, ...]
    gain_table: object
    count_table: object
    accept_table: object
    min_gain: float = 0.0
    min_bin_count: int = 64
    edge_edges: tuple[float, ...] | None = None
    feature_mode: str = "confidence_magnitude"

    def to_json(self) -> dict[str, object]:
        def _nested_list(value: object) -> object:
            if isinstance(value, tuple):
                return [_nested_list(v) for v in value]
            return value

        def _count_true(value: object) -> int:
            if isinstance(value, tuple):
                return sum(_count_true(v) for v in value)
            return int(bool(value))

        return {
            "feature_mode": str(self.feature_mode),
            "confidence_edges": list(self.confidence_edges),
            "magnitude_edges": list(self.magnitude_edges),
            "edge_edges": list(self.edge_edges) if self.edge_edges is not None else None,
            "gain_table": _nested_list(self.gain_table),
            "count_table": _nested_list(self.count_table),
            "accept_table": _nested_list(self.accept_table),
            "min_gain": float(self.min_gain),
            "min_bin_count": int(self.min_bin_count),
            "accepted_bins": int(_count_true(self.accept_table)),
        }

    def acceptance_mask(
        self,
        confidence: torch.Tensor,
        signal: torch.Tensor,
        *,
        base: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        device = torch.device(device or confidence.device)
        confidence = confidence.to(device=device, dtype=torch.float32)
        signal = signal.to(device=device, dtype=torch.float32)
        conf_feature = torch.log1p(torch.clamp(confidence.squeeze(0), min=0.0))
        mag_feature = torch.linalg.vector_norm(signal, dim=0)
        conf_edges = torch.tensor(self.confidence_edges, device=device, dtype=torch.float32)
        mag_edges = torch.tensor(self.magnitude_edges, device=device, dtype=torch.float32)
        accept = torch.tensor(self.accept_table, device=device, dtype=torch.bool)
        conf_idx = torch.bucketize(conf_feature.reshape(-1), conf_edges[1:-1], right=False)
        mag_idx = torch.bucketize(mag_feature.reshape(-1), mag_edges[1:-1], right=False)
        if self.edge_edges is not None:
            if base is None:
                edge_feature = torch.zeros_like(conf_feature)
            else:
                edge_feature = _image_edge_magnitude(base.to(device=device, dtype=torch.float32)).squeeze(0)
            edge_edges = torch.tensor(self.edge_edges, device=device, dtype=torch.float32)
            edge_idx = torch.bucketize(edge_feature.reshape(-1), edge_edges[1:-1], right=False)
            mask = accept[conf_idx, mag_idx, edge_idx].reshape(conf_feature.shape)
        else:
            mask = accept[conf_idx, mag_idx].reshape(conf_feature.shape)
        return mask.unsqueeze(0)


@dataclass
class AlphaCalibrator:
    confidence_edges: tuple[float, ...]
    magnitude_edges: tuple[float, ...]
    alpha_table: object
    gain_table: object
    count_table: object
    accept_table: object
    tail_gain_table: object | None = None
    negative_fraction_table: object | None = None
    risk_zeroed_table: object | None = None
    region_tail_gain_table: object | None = None
    region_negative_fraction_table: object | None = None
    region_count_table: object | None = None
    region_risk_zeroed_table: object | None = None
    default_alpha: float = 0.0
    min_gain: float = 0.0
    min_bin_count: int = 64
    risk_tail_fraction: float = 0.20
    max_negative_gain_fraction: float = 1.0
    min_tail_gain: float = -math.inf
    holdout_safe_zero: bool = False
    region_risk_enabled: bool = False
    region_risk_json: str = ""
    region_risk_objective_bad_only: bool = False
    region_risk_objective_max_balanced_delta: float = 0.0
    region_risk_objective_max_delta_ssim: float = 0.0
    region_risk_objective_min_delta_lpips: float = 0.0
    region_risk_min_tail_gain: float = 0.0
    region_risk_max_negative_fraction: float = 1.0
    region_risk_min_regions: int = 1
    view_tail_scale: float = 1.0
    view_tail_enabled: bool = False
    view_tail_scale_grid: object | None = None
    view_tail_cvar_fraction: float = 0.25
    view_tail_min_gain: float = -math.inf
    view_tail_max_negative_fraction: float = 1.0
    view_tail_objective: str = "mse"
    view_tail_ssim_weight: float = 20.0
    view_tail_lpips_weight: float = 20.0
    view_tail_compute_lpips: bool = False
    view_tail_metric_max_side: int = 512
    view_tail_mean_gain: float = 0.0
    view_tail_cvar_gain: float = 0.0
    view_tail_negative_fraction: float = 0.0
    view_tail_safe_scale_found: bool = False
    view_tail_fallback_used: bool = False
    view_tail_candidate_stats: object | None = None
    edge_edges: tuple[float, ...] | None = None
    feature_mode: str = "confidence_magnitude"

    def to_json(self) -> dict[str, object]:
        def _nested_list(value: object) -> object:
            if isinstance(value, tuple):
                return [_nested_list(v) for v in value]
            return value

        def _count_true(value: object) -> int:
            if isinstance(value, tuple):
                return sum(_count_true(v) for v in value)
            return int(bool(value))

        return {
            "feature_mode": str(self.feature_mode),
            "confidence_edges": list(self.confidence_edges),
            "magnitude_edges": list(self.magnitude_edges),
            "edge_edges": list(self.edge_edges) if self.edge_edges is not None else None,
            "alpha_table": _nested_list(self.alpha_table),
            "gain_table": _nested_list(self.gain_table),
            "count_table": _nested_list(self.count_table),
            "accept_table": _nested_list(self.accept_table),
            "tail_gain_table": _nested_list(self.tail_gain_table) if self.tail_gain_table is not None else None,
            "negative_fraction_table": (
                _nested_list(self.negative_fraction_table) if self.negative_fraction_table is not None else None
            ),
            "risk_zeroed_table": _nested_list(self.risk_zeroed_table) if self.risk_zeroed_table is not None else None,
            "region_tail_gain_table": (
                _nested_list(self.region_tail_gain_table) if self.region_tail_gain_table is not None else None
            ),
            "region_negative_fraction_table": (
                _nested_list(self.region_negative_fraction_table)
                if self.region_negative_fraction_table is not None
                else None
            ),
            "region_count_table": _nested_list(self.region_count_table) if self.region_count_table is not None else None,
            "region_risk_zeroed_table": (
                _nested_list(self.region_risk_zeroed_table) if self.region_risk_zeroed_table is not None else None
            ),
            "default_alpha": float(self.default_alpha),
            "min_gain": float(self.min_gain),
            "min_bin_count": int(self.min_bin_count),
            "risk_tail_fraction": float(self.risk_tail_fraction),
            "max_negative_gain_fraction": float(self.max_negative_gain_fraction),
            "min_tail_gain": float(self.min_tail_gain) if math.isfinite(float(self.min_tail_gain)) else None,
            "holdout_safe_zero": bool(self.holdout_safe_zero),
            "region_risk_enabled": bool(self.region_risk_enabled),
            "region_risk_json": str(self.region_risk_json),
            "region_risk_objective_bad_only": bool(self.region_risk_objective_bad_only),
            "region_risk_objective_max_balanced_delta": float(self.region_risk_objective_max_balanced_delta),
            "region_risk_objective_max_delta_ssim": float(self.region_risk_objective_max_delta_ssim),
            "region_risk_objective_min_delta_lpips": float(self.region_risk_objective_min_delta_lpips),
            "region_risk_min_tail_gain": float(self.region_risk_min_tail_gain),
            "region_risk_max_negative_fraction": float(self.region_risk_max_negative_fraction),
            "region_risk_min_regions": int(self.region_risk_min_regions),
            "accepted_bins": int(_count_true(self.accept_table)),
            "risk_zeroed_bins": int(_count_true(self.risk_zeroed_table)) if self.risk_zeroed_table is not None else 0,
            "region_risk_zeroed_bins": (
                int(_count_true(self.region_risk_zeroed_table)) if self.region_risk_zeroed_table is not None else 0
            ),
            "view_tail_scale": float(self.view_tail_scale),
            "view_tail_enabled": bool(self.view_tail_enabled),
            "view_tail_scale_grid": _nested_list(self.view_tail_scale_grid)
            if self.view_tail_scale_grid is not None
            else None,
            "view_tail_cvar_fraction": float(self.view_tail_cvar_fraction),
            "view_tail_min_gain": float(self.view_tail_min_gain) if math.isfinite(float(self.view_tail_min_gain)) else None,
            "view_tail_max_negative_fraction": float(self.view_tail_max_negative_fraction),
            "view_tail_objective": str(self.view_tail_objective),
            "view_tail_ssim_weight": float(self.view_tail_ssim_weight),
            "view_tail_lpips_weight": float(self.view_tail_lpips_weight),
            "view_tail_compute_lpips": bool(self.view_tail_compute_lpips),
            "view_tail_metric_max_side": int(self.view_tail_metric_max_side),
            "view_tail_mean_gain": float(self.view_tail_mean_gain),
            "view_tail_cvar_gain": float(self.view_tail_cvar_gain),
            "view_tail_negative_fraction": float(self.view_tail_negative_fraction),
            "view_tail_safe_scale_found": bool(self.view_tail_safe_scale_found),
            "view_tail_fallback_used": bool(self.view_tail_fallback_used),
            "view_tail_candidate_stats": (
                _nested_list(self.view_tail_candidate_stats)
                if self.view_tail_candidate_stats is not None
                else None
            ),
        }

    def alpha_map(
        self,
        confidence: torch.Tensor,
        signal: torch.Tensor,
        *,
        base: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device or confidence.device)
        confidence = confidence.to(device=device, dtype=torch.float32)
        signal = signal.to(device=device, dtype=torch.float32)
        conf_feature = torch.log1p(torch.clamp(confidence.squeeze(0), min=0.0))
        mag_feature = torch.linalg.vector_norm(signal, dim=0)
        conf_edges = torch.tensor(self.confidence_edges, device=device, dtype=torch.float32)
        mag_edges = torch.tensor(self.magnitude_edges, device=device, dtype=torch.float32)
        alpha_table = torch.tensor(self.alpha_table, device=device, dtype=torch.float32)
        accept = torch.tensor(self.accept_table, device=device, dtype=torch.bool)
        conf_idx = torch.bucketize(conf_feature.reshape(-1), conf_edges[1:-1], right=False)
        mag_idx = torch.bucketize(mag_feature.reshape(-1), mag_edges[1:-1], right=False)
        if self.edge_edges is not None:
            if base is None:
                edge_feature = torch.zeros_like(conf_feature)
            else:
                edge_feature = _image_edge_magnitude(base.to(device=device, dtype=torch.float32)).squeeze(0)
            edge_edges = torch.tensor(self.edge_edges, device=device, dtype=torch.float32)
            edge_idx = torch.bucketize(edge_feature.reshape(-1), edge_edges[1:-1], right=False)
            alpha = alpha_table[conf_idx, mag_idx, edge_idx].reshape(conf_feature.shape)
            active = accept[conf_idx, mag_idx, edge_idx].reshape(conf_feature.shape)
        else:
            alpha = alpha_table[conf_idx, mag_idx].reshape(conf_feature.shape)
            active = accept[conf_idx, mag_idx].reshape(conf_feature.shape)
        default = torch.full_like(alpha, float(self.default_alpha))
        alpha = torch.where(active, alpha, default)
        alpha = alpha * float(self.view_tail_scale)
        return alpha.unsqueeze(0), active.unsqueeze(0)


def _calibration_candidates(
    train_frames: Sequence[FrameRecord],
    calib_stride: int,
    calib_max_views: int,
    calib_sampler: str = "stride_first",
) -> list[FrameRecord]:
    if not train_frames:
        return []
    sampler = str(calib_sampler)
    if sampler not in {"stride_first", "uniform"}:
        raise ValueError(f"Unsupported calibration sampler: {calib_sampler}")
    max_views = int(calib_max_views)
    if sampler == "uniform":
        if max_views <= 0 or len(train_frames) <= max_views:
            candidates = list(train_frames)
        elif max_views == 1:
            candidates = [train_frames[len(train_frames) // 2]]
        else:
            raw = torch.linspace(0, len(train_frames) - 1, steps=max_views).round().to(torch.int64).tolist()
            seen: set[int] = set()
            candidates = []
            for idx in raw:
                if idx not in seen:
                    seen.add(idx)
                    candidates.append(train_frames[int(idx)])
    else:
        stride = max(int(calib_stride), 1)
        candidates = list(train_frames[::stride])
        if max_views > 0:
            candidates = candidates[:max_views]
    if not candidates and train_frames:
        candidates = [train_frames[0]]
    return candidates


def _numeric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _objective_row_is_bad(
    row: dict,
    *,
    max_balanced_delta: float = 0.0,
    max_delta_ssim: float = 0.0,
    min_delta_lpips: float = 0.0,
) -> bool:
    if row.get("crop_changed") is False:
        return False
    if bool(row.get("metrics_skipped_equal_crop", False)):
        return False
    balanced = _numeric(row, "core_balanced_delta")
    delta_ssim = _numeric(row, "delta_core_ssim")
    delta_lpips = _numeric(row, "delta_core_lpips")
    has_objective = balanced is not None or delta_ssim is not None or delta_lpips is not None
    if not has_objective:
        return False
    return (
        (balanced is not None and balanced < float(max_balanced_delta))
        or (delta_ssim is not None and delta_ssim < float(max_delta_ssim))
        or (delta_lpips is not None and delta_lpips > float(min_delta_lpips))
    )


def _load_region_risk_bboxes(
    path: str | Path | None,
    *,
    objective_bad_only: bool = False,
    objective_max_balanced_delta: float = 0.0,
    objective_max_delta_ssim: float = 0.0,
    objective_min_delta_lpips: float = 0.0,
) -> dict[str, list[tuple[int, int, int, int]]]:
    if path is None or not str(path).strip():
        return {}
    risk_path = Path(path)
    if not risk_path.is_file():
        return {}
    try:
        payload = json.loads(risk_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    out: dict[str, list[tuple[int, int, int, int]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(objective_bad_only) and not _objective_row_is_bad(
            row,
            max_balanced_delta=objective_max_balanced_delta,
            max_delta_ssim=objective_max_delta_ssim,
            min_delta_lpips=objective_min_delta_lpips,
        ):
            continue
        raw_bbox = row.get("bbox_xyxy")
        view = str(row.get("view", "")).strip()
        if not view or not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
            continue
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in raw_bbox[:4]]
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        out.setdefault(view, []).append((x0, y0, x1, y1))
    return out


def _tail_mean(values: torch.Tensor, fraction: float) -> float:
    if values.numel() <= 0:
        return math.nan
    count = max(1, int(math.ceil(max(min(float(fraction), 1.0), 1e-6) * int(values.numel()))))
    return float(torch.topk(values.to(torch.float64), k=count, largest=False).values.mean().item())


def compute_evidence_signal(
    target: FrameRecord,
    support_frames: Sequence[FrameRecord],
    *,
    k: int = 4,
    mode: str = "residual",
    residual_clip: float = 0.25,
    min_confidence: float = 1e-4,
    depth_abs_tol: float = 0.02,
    depth_rel_tol: float = 0.03,
    direction_weight: float = 0.35,
    evidence_max_side: int = 0,
    loader: FrameLoader | None = None,
    device: torch.device | str = "cuda",
) -> EvidenceSignal:
    loader = loader or FrameLoader(device=device)
    device = torch.device(device)
    base_full = loader.render(str(target.render_path)).to(device)
    target_depth_full = loader.depth(str(target.depth_path)).to(device)
    full_h, full_w = int(target_depth_full.shape[0]), int(target_depth_full.shape[1])
    low_size = _evidence_size(full_h, full_w, int(evidence_max_side))
    if low_size is None:
        base = base_full
        target_depth = target_depth_full
    else:
        base = _resize_chw(base_full, low_size, mode="bilinear")
        target_depth = _resize_hw(target_depth_full, low_size, mode="bilinear")
    if mode not in {"residual", "color"}:
        raise ValueError(f"Unsupported ELA mode: {mode}")
    support = select_support_frames(
        target,
        support_frames,
        k=k,
        exclude_names={target.name, target.camera.image_name},
        direction_weight=direction_weight,
    )
    signal_num = torch.zeros_like(base)
    signal_sq_num = torch.zeros_like(base)
    weight_den = torch.zeros((1, base.shape[1], base.shape[2]), device=device, dtype=torch.float32)
    support_count = torch.zeros((1, base.shape[1], base.shape[2]), device=device, dtype=torch.float32)
    target_world_grid = _make_target_world_grid(target.camera, target_depth, device=device)
    used: list[str] = []
    for support_frame, view_weight in support:
        support_depth = loader.depth(str(support_frame.depth_path))
        if mode == "residual":
            support_signal = loader.residual(support_frame, residual_clip=residual_clip)
        else:
            support_signal = loader.gt(str(support_frame.gt_path))
        if low_size is not None:
            support_size = _evidence_size(
                int(support_depth.shape[0]),
                int(support_depth.shape[1]),
                int(evidence_max_side),
            )
            if support_size is not None:
                support_depth = _resize_hw(support_depth, support_size, mode="bilinear")
                support_signal = _resize_chw(support_signal, support_size, mode="bilinear")
        warped, confidence = warp_support_residual(
            target,
            support_frame,
            target_depth,
            support_depth,
            support_signal,
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
            target_world_grid=target_world_grid,
            device=device,
            support_valid=_frame_valid_mask(loader, support_frame),
        )
        weight = confidence.unsqueeze(0) * float(view_weight)
        if float(weight.mean().item()) <= 0.0:
            continue
        signal_num = signal_num + warped * weight
        signal_sq_num = signal_sq_num + warped.pow(2) * weight
        weight_den = weight_den + weight
        support_count = support_count + (confidence.unsqueeze(0) > float(min_confidence)).to(torch.float32)
        used.append(support_frame.name)
    valid = weight_den > float(min_confidence)
    signal = torch.where(valid, signal_num / torch.clamp(weight_den, min=1e-8), torch.zeros_like(signal_num))
    mean_sq = torch.where(valid, signal_sq_num / torch.clamp(weight_den, min=1e-8), torch.zeros_like(signal_sq_num))
    variance = torch.clamp(mean_sq - signal.pow(2), min=0.0)
    residual_std = torch.sqrt(torch.mean(variance, dim=0, keepdim=True) + 1e-12)
    residual_std = torch.where(valid, residual_std, torch.zeros_like(residual_std))
    if low_size is not None:
        full_size = (full_h, full_w)
        signal = _resize_chw(signal, full_size, mode="bilinear")
        confidence = _resize_chw(weight_den, full_size, mode="bilinear")
        support_count = _resize_chw(support_count, full_size, mode="nearest")
        residual_std = _resize_chw(residual_std, full_size, mode="bilinear")
        valid = confidence > float(min_confidence)
        return EvidenceSignal(
            base=base_full,
            signal=torch.where(valid, signal, torch.zeros_like(signal)),
            confidence=confidence,
            valid=valid,
            support_names=used,
            support_count=support_count,
            residual_std=torch.where(valid, residual_std, torch.zeros_like(residual_std)),
        )
    return EvidenceSignal(
        base=base_full,
        signal=signal,
        confidence=weight_den,
        valid=valid,
        support_names=used,
        support_count=support_count,
        residual_std=residual_std,
    )


def local_trust_acceptance_mask(
    evidence: EvidenceSignal,
    *,
    enabled: bool = False,
    min_supports: int = 2,
    max_residual_std: float = -1.0,
    min_agreement: float = 0.0,
    agreement_scale: float = 0.04,
    confidence_quantile: float = -1.0,
    min_confidence: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    valid = evidence.valid.bool()
    if not bool(enabled):
        return valid, {"enabled": False}
    support_count = (
        evidence.support_count.to(device=valid.device, dtype=torch.float32)
        if evidence.support_count is not None
        else valid.to(torch.float32)
    )
    residual_std = (
        evidence.residual_std.to(device=valid.device, dtype=torch.float32)
        if evidence.residual_std is not None
        else torch.zeros_like(evidence.confidence, dtype=torch.float32)
    )
    accept = valid.clone()
    if int(min_supports) > 0:
        accept = accept & (support_count >= float(min_supports))
    max_std = float(max_residual_std)
    if math.isfinite(max_std) and max_std >= 0.0:
        accept = accept & (residual_std <= max_std)
    agreement_scale = max(float(agreement_scale), 1e-8)
    agreement = torch.exp(-residual_std / agreement_scale)
    if float(min_agreement) > 0.0:
        accept = accept & (agreement >= float(min_agreement))
    confidence_threshold = float(max(min_confidence, 0.0))
    q = float(confidence_quantile)
    if q >= 0.0 and bool(valid.any().item()):
        q = min(max(q, 0.0), 0.999)
        values = evidence.confidence[valid]
        if values.numel() > 0:
            confidence_threshold = max(confidence_threshold, float(torch.quantile(values.detach().float(), q).item()))
    if confidence_threshold > 0.0:
        accept = accept & (evidence.confidence >= confidence_threshold)
    valid_std = residual_std[valid] if bool(valid.any().item()) else residual_std.reshape(-1)[:0]
    valid_support = support_count[valid] if bool(valid.any().item()) else support_count.reshape(-1)[:0]
    valid_agreement = agreement[valid] if bool(valid.any().item()) else agreement.reshape(-1)[:0]
    return accept, {
        "enabled": True,
        "min_supports": int(min_supports),
        "max_residual_std": float(max_residual_std),
        "min_agreement": float(min_agreement),
        "agreement_scale": float(agreement_scale),
        "confidence_quantile": float(confidence_quantile),
        "confidence_threshold": float(confidence_threshold),
        "mean_residual_std": float(valid_std.mean().detach().cpu().item()) if valid_std.numel() else 0.0,
        "mean_support_count": float(valid_support.mean().detach().cpu().item()) if valid_support.numel() else 0.0,
        "mean_agreement": float(valid_agreement.mean().detach().cpu().item()) if valid_agreement.numel() else 0.0,
        "accept_fraction": float(accept.to(torch.float32).mean().detach().cpu().item()),
    }


def local_trust_weight_map(
    evidence: EvidenceSignal,
    *,
    enabled: bool = False,
    min_supports: int = 2,
    max_residual_std: float = -1.0,
    min_agreement: float = 0.0,
    agreement_scale: float = 0.04,
    confidence_quantile: float = -1.0,
    min_confidence: float = 0.0,
    min_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    valid = evidence.valid.bool()
    if not bool(enabled):
        return valid.to(torch.float32), {"enabled": False, "mode": "soft"}
    support_count = (
        evidence.support_count.to(device=valid.device, dtype=torch.float32)
        if evidence.support_count is not None
        else valid.to(torch.float32)
    )
    residual_std = (
        evidence.residual_std.to(device=valid.device, dtype=torch.float32)
        if evidence.residual_std is not None
        else torch.zeros_like(evidence.confidence, dtype=torch.float32)
    )
    agreement_scale = max(float(agreement_scale), 1e-8)
    agreement = torch.exp(-residual_std / agreement_scale)
    confidence_threshold = float(max(min_confidence, 0.0))
    q = float(confidence_quantile)
    if q >= 0.0 and bool(valid.any().item()):
        q = min(max(q, 0.0), 0.999)
        values = evidence.confidence[valid]
        if values.numel() > 0:
            confidence_threshold = max(confidence_threshold, float(torch.quantile(values.detach().float(), q).item()))

    min_supports_f = max(float(min_supports), 1.0)
    support_score = torch.clamp(support_count / min_supports_f, 0.0, 1.0)
    if int(min_supports) <= 1:
        support_score = torch.where(support_count > 0.0, torch.ones_like(support_score), torch.zeros_like(support_score))

    std_score = agreement
    max_std = float(max_residual_std)
    if math.isfinite(max_std) and max_std >= 0.0:
        # Softly decay after the hard v26 threshold instead of zeroing all residuals.
        std_score = std_score * torch.clamp((2.0 * max_std - residual_std) / max(max_std, 1e-8), 0.0, 1.0)

    agreement_score = agreement
    min_agreement_f = float(min_agreement)
    if min_agreement_f > 0.0:
        agreement_score = torch.clamp(
            (agreement - 0.5 * min_agreement_f) / max(1.0 - 0.5 * min_agreement_f, 1e-8),
            0.0,
            1.0,
        )

    if confidence_threshold > 0.0:
        confidence_score = torch.clamp(evidence.confidence / max(confidence_threshold, 1e-8), 0.0, 1.0)
    else:
        confidence_score = valid.to(torch.float32)

    weight = support_score * std_score * agreement_score * confidence_score
    weight = torch.where(valid, torch.clamp(weight, 0.0, 1.0), torch.zeros_like(weight))
    floor = max(float(min_weight), 0.0)
    if floor > 0.0:
        weight = torch.where(weight >= floor, weight, torch.zeros_like(weight))

    active = weight > 0.0
    valid_std = residual_std[valid] if bool(valid.any().item()) else residual_std.reshape(-1)[:0]
    valid_support = support_count[valid] if bool(valid.any().item()) else support_count.reshape(-1)[:0]
    valid_agreement = agreement[valid] if bool(valid.any().item()) else agreement.reshape(-1)[:0]
    valid_weight = weight[valid] if bool(valid.any().item()) else weight.reshape(-1)[:0]
    return weight, {
        "enabled": True,
        "mode": "soft",
        "min_supports": int(min_supports),
        "max_residual_std": float(max_residual_std),
        "min_agreement": float(min_agreement),
        "agreement_scale": float(agreement_scale),
        "confidence_quantile": float(confidence_quantile),
        "confidence_threshold": float(confidence_threshold),
        "min_weight": float(floor),
        "mean_residual_std": float(valid_std.mean().detach().cpu().item()) if valid_std.numel() else 0.0,
        "mean_support_count": float(valid_support.mean().detach().cpu().item()) if valid_support.numel() else 0.0,
        "mean_agreement": float(valid_agreement.mean().detach().cpu().item()) if valid_agreement.numel() else 0.0,
        "mean_weight": float(valid_weight.mean().detach().cpu().item()) if valid_weight.numel() else 0.0,
        "active_fraction": float(active.to(torch.float32).mean().detach().cpu().item()),
        "accept_fraction": float(active.to(torch.float32).mean().detach().cpu().item()),
    }


def _sample_1d(values: torch.Tensor, max_samples: int) -> torch.Tensor:
    values = values.reshape(-1)
    if values.numel() <= int(max_samples):
        return values
    idx = torch.linspace(0, values.numel() - 1, steps=int(max_samples), device=values.device).long()
    return values[idx]


def _strictly_increasing_edges(edges: torch.Tensor) -> torch.Tensor:
    edges = edges.clone()
    for idx in range(1, int(edges.numel())):
        if float(edges[idx].item()) <= float(edges[idx - 1].item()):
            edges[idx] = edges[idx - 1] + 1e-8
    return edges


def fit_benefit_calibrator(
    train_frames: Sequence[FrameRecord],
    *,
    calibration_target_frames: Sequence[FrameRecord] | None = None,
    k: int,
    mode: str,
    calib_stride: int,
    calib_max_views: int,
    residual_clip: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    direction_weight: float,
    calib_sampler: str = "stride_first",
    bins: int = 5,
    min_gain: float = 0.0,
    min_bin_count: int = 64,
    max_pixels_per_view: int = 4096,
    feature_mode: str = "confidence_magnitude",
    local_trust_gate: bool = False,
    local_trust_min_supports: int = 2,
    local_trust_max_residual_std: float = -1.0,
    local_trust_min_agreement: float = 0.0,
    local_trust_agreement_scale: float = 0.04,
    local_trust_confidence_quantile: float = -1.0,
    local_trust_min_confidence: float = 0.0,
    local_trust_mode: str = "hard",
    local_trust_min_weight: float = 0.0,
    device: torch.device | str = "cuda",
) -> BenefitCalibrator:
    if feature_mode not in {"confidence_magnitude", "confidence_magnitude_edge"}:
        raise ValueError(f"Unsupported benefit feature mode: {feature_mode}")
    if mode != "residual" or not train_frames:
        return BenefitCalibrator(
            confidence_edges=(0.0, 1.0),
            magnitude_edges=(0.0, 1.0),
            gain_table=((0.0,),),
            count_table=((0,),),
            accept_table=((False,),),
            min_gain=min_gain,
            min_bin_count=min_bin_count,
            feature_mode=feature_mode,
        )
    device = torch.device(device)
    loader = FrameLoader(device=device)
    target_pool = calibration_target_frames if calibration_target_frames is not None else train_frames
    candidates = _calibration_candidates(target_pool, calib_stride, calib_max_views, calib_sampler)
    conf_values: list[torch.Tensor] = []
    mag_values: list[torch.Tensor] = []
    edge_values: list[torch.Tensor] = []
    gain_values: list[torch.Tensor] = []
    for target in candidates:
        support = [frame for frame in train_frames if frame.name != target.name]
        if not support:
            continue
        gt = loader.gt(str(target.gt_path)).to(device)
        ev = compute_evidence_signal(
            target,
            support,
            k=k,
            mode=mode,
            residual_clip=residual_clip,
            min_confidence=1e-8,
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
            direction_weight=direction_weight,
            loader=loader,
            device=device,
        )
        valid_mask = ev.valid
        signal = ev.signal
        if bool(local_trust_gate):
            local_mode = str(local_trust_mode).strip().lower()
            if local_mode == "soft":
                local_weight, _ = local_trust_weight_map(
                    ev,
                    enabled=True,
                    min_supports=int(local_trust_min_supports),
                    max_residual_std=float(local_trust_max_residual_std),
                    min_agreement=float(local_trust_min_agreement),
                    agreement_scale=float(local_trust_agreement_scale),
                    confidence_quantile=float(local_trust_confidence_quantile),
                    min_confidence=float(local_trust_min_confidence),
                    min_weight=float(local_trust_min_weight),
                )
                valid_mask = valid_mask & (local_weight > 0.0)
                signal = signal * local_weight
                signal = torch.where(valid_mask, signal, torch.zeros_like(signal))
            else:
                local_accept, _ = local_trust_acceptance_mask(
                    ev,
                    enabled=True,
                    min_supports=int(local_trust_min_supports),
                    max_residual_std=float(local_trust_max_residual_std),
                    min_agreement=float(local_trust_min_agreement),
                    agreement_scale=float(local_trust_agreement_scale),
                    confidence_quantile=float(local_trust_confidence_quantile),
                    min_confidence=float(local_trust_min_confidence),
                )
                valid_mask = valid_mask & local_accept
                signal = torch.where(valid_mask, signal, torch.zeros_like(signal))
        candidate = torch.clamp(ev.base + signal, 0.0, 1.0)
        benefit = torch.mean((ev.base - gt) ** 2 - (candidate - gt) ** 2, dim=0)
        valid = valid_mask.squeeze(0)
        if not bool(valid.any().item()):
            continue
        conf = torch.log1p(torch.clamp(ev.confidence.squeeze(0), min=0.0))[valid]
        mag = torch.linalg.vector_norm(signal, dim=0)[valid]
        edge = _image_edge_magnitude(ev.base).squeeze(0)[valid]
        gain = benefit[valid]
        max_samples = max(int(max_pixels_per_view), 1)
        conf_values.append(_sample_1d(conf.detach(), max_samples).cpu())
        mag_values.append(_sample_1d(mag.detach(), max_samples).cpu())
        if feature_mode == "confidence_magnitude_edge":
            edge_values.append(_sample_1d(edge.detach(), max_samples).cpu())
        gain_values.append(_sample_1d(gain.detach(), max_samples).cpu())
    if not conf_values:
        return BenefitCalibrator(
            confidence_edges=(0.0, 1.0),
            magnitude_edges=(0.0, 1.0),
            gain_table=((0.0,),),
            count_table=((0,),),
            accept_table=((False,),),
            min_gain=min_gain,
            min_bin_count=min_bin_count,
            feature_mode=feature_mode,
        )
    conf_all = torch.cat(conf_values).float()
    mag_all = torch.cat(mag_values).float()
    gain_all = torch.cat(gain_values).float()
    bin_count = max(int(bins), 1)
    quantiles = torch.linspace(0.0, 1.0, steps=bin_count + 1)
    conf_edges = torch.quantile(conf_all, quantiles).float()
    mag_edges = torch.quantile(mag_all, quantiles).float()
    conf_edges[0] = min(conf_edges[0], conf_all.min()) - 1e-6
    conf_edges[-1] = max(conf_edges[-1], conf_all.max()) + 1e-6
    mag_edges[0] = min(mag_edges[0], mag_all.min()) - 1e-6
    mag_edges[-1] = max(mag_edges[-1], mag_all.max()) + 1e-6
    conf_edges = _strictly_increasing_edges(conf_edges)
    mag_edges = _strictly_increasing_edges(mag_edges)
    conf_idx = torch.bucketize(conf_all, conf_edges[1:-1], right=False)
    mag_idx = torch.bucketize(mag_all, mag_edges[1:-1], right=False)
    edge_edges_tuple: tuple[float, ...] | None = None
    if feature_mode == "confidence_magnitude_edge":
        edge_all = torch.cat(edge_values).float()
        edge_edges = torch.quantile(edge_all, quantiles).float()
        edge_edges[0] = min(edge_edges[0], edge_all.min()) - 1e-6
        edge_edges[-1] = max(edge_edges[-1], edge_all.max()) + 1e-6
        edge_edges = _strictly_increasing_edges(edge_edges)
        edge_idx = torch.bucketize(edge_all, edge_edges[1:-1], right=False)
        gain_sum = torch.zeros((bin_count, bin_count, bin_count), dtype=torch.float64)
        counts = torch.zeros((bin_count, bin_count, bin_count), dtype=torch.int64)
        for c, m, e, g in zip(conf_idx.tolist(), mag_idx.tolist(), edge_idx.tolist(), gain_all.tolist()):
            gain_sum[c, m, e] += float(g)
            counts[c, m, e] += 1
        edge_edges_tuple = tuple(float(x) for x in edge_edges.tolist())
    else:
        gain_sum = torch.zeros((bin_count, bin_count), dtype=torch.float64)
        counts = torch.zeros((bin_count, bin_count), dtype=torch.int64)
        for c, m, g in zip(conf_idx.tolist(), mag_idx.tolist(), gain_all.tolist()):
            gain_sum[c, m] += float(g)
            counts[c, m] += 1
    mean_gain = torch.where(counts > 0, gain_sum / torch.clamp(counts, min=1), torch.zeros_like(gain_sum))
    accept = (mean_gain > float(min_gain)) & (counts >= int(min_bin_count))
    return BenefitCalibrator(
        confidence_edges=tuple(float(x) for x in conf_edges.tolist()),
        magnitude_edges=tuple(float(x) for x in mag_edges.tolist()),
        edge_edges=edge_edges_tuple,
        gain_table=tuple(
            tuple(tuple(float(v) for v in col) for col in row) if feature_mode == "confidence_magnitude_edge" else tuple(float(v) for v in row)
            for row in mean_gain.tolist()
        ),
        count_table=tuple(
            tuple(tuple(int(v) for v in col) for col in row) if feature_mode == "confidence_magnitude_edge" else tuple(int(v) for v in row)
            for row in counts.tolist()
        ),
        accept_table=tuple(
            tuple(tuple(bool(v) for v in col) for col in row) if feature_mode == "confidence_magnitude_edge" else tuple(bool(v) for v in row)
            for row in accept.tolist()
        ),
        min_gain=float(min_gain),
        min_bin_count=int(min_bin_count),
        feature_mode=feature_mode,
    )


def fit_alpha_calibrator(
    train_frames: Sequence[FrameRecord],
    *,
    calibration_target_frames: Sequence[FrameRecord] | None = None,
    k: int,
    mode: str,
    alpha_grid: Sequence[float],
    calib_stride: int,
    calib_max_views: int,
    residual_clip: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    direction_weight: float,
    calib_sampler: str = "stride_first",
    bins: int = 5,
    min_gain: float = 0.0,
    min_bin_count: int = 64,
    risk_tail_fraction: float = 0.20,
    max_negative_gain_fraction: float = 1.0,
    min_tail_gain: float = -math.inf,
    holdout_safe_zero: bool = False,
    max_pixels_per_view: int = 4096,
    feature_mode: str = "confidence_magnitude_edge",
    default_alpha: float = 0.0,
    region_risk_json: str | Path | None = None,
    region_risk_objective_bad_only: bool = False,
    region_risk_objective_max_balanced_delta: float = 0.0,
    region_risk_objective_max_delta_ssim: float = 0.0,
    region_risk_objective_min_delta_lpips: float = 0.0,
    region_risk_min_tail_gain: float = 0.0,
    region_risk_max_negative_fraction: float = 1.0,
    region_risk_min_regions: int = 1,
    view_tail_scale_grid: Sequence[float] | None = None,
    view_tail_cvar_fraction: float = 0.25,
    view_tail_min_gain: float = -math.inf,
    view_tail_max_negative_fraction: float = 1.0,
    view_tail_objective: str = "mse",
    view_tail_ssim_weight: float = 20.0,
    view_tail_lpips_weight: float = 20.0,
    view_tail_compute_lpips: bool = False,
    view_tail_metric_max_side: int = 512,
    local_trust_gate: bool = False,
    local_trust_min_supports: int = 2,
    local_trust_max_residual_std: float = -1.0,
    local_trust_min_agreement: float = 0.0,
    local_trust_agreement_scale: float = 0.04,
    local_trust_confidence_quantile: float = -1.0,
    local_trust_min_confidence: float = 0.0,
    local_trust_mode: str = "hard",
    local_trust_min_weight: float = 0.0,
    device: torch.device | str = "cuda",
) -> AlphaCalibrator:
    if feature_mode not in {"confidence_magnitude", "confidence_magnitude_edge"}:
        raise ValueError(f"Unsupported alpha feature mode: {feature_mode}")
    view_tail_objective_value = str(view_tail_objective).strip().lower()
    if view_tail_objective_value not in {"mse", "balanced"}:
        raise ValueError(f"Unsupported view-tail objective: {view_tail_objective}")
    if mode != "residual" or not train_frames:
        return AlphaCalibrator(
            confidence_edges=(0.0, 1.0),
            magnitude_edges=(0.0, 1.0),
            alpha_table=((float(default_alpha),),),
            gain_table=((0.0,),),
            count_table=((0,),),
            accept_table=((False,),),
            default_alpha=float(default_alpha),
            min_gain=float(min_gain),
            min_bin_count=int(min_bin_count),
            risk_tail_fraction=float(risk_tail_fraction),
            max_negative_gain_fraction=float(max_negative_gain_fraction),
            min_tail_gain=float(min_tail_gain),
            holdout_safe_zero=bool(holdout_safe_zero),
            region_risk_enabled=False,
            region_risk_json=str(region_risk_json or ""),
            region_risk_objective_bad_only=bool(region_risk_objective_bad_only),
            region_risk_objective_max_balanced_delta=float(region_risk_objective_max_balanced_delta),
            region_risk_objective_max_delta_ssim=float(region_risk_objective_max_delta_ssim),
            region_risk_objective_min_delta_lpips=float(region_risk_objective_min_delta_lpips),
            region_risk_min_tail_gain=float(region_risk_min_tail_gain),
            region_risk_max_negative_fraction=float(region_risk_max_negative_fraction),
            region_risk_min_regions=max(int(region_risk_min_regions), 1),
            view_tail_objective=view_tail_objective_value,
            view_tail_ssim_weight=float(view_tail_ssim_weight),
            view_tail_lpips_weight=float(view_tail_lpips_weight),
            view_tail_compute_lpips=bool(view_tail_compute_lpips),
            view_tail_metric_max_side=max(int(view_tail_metric_max_side), 0),
            feature_mode=feature_mode,
        )
    device = torch.device(device)
    loader = FrameLoader(device=device)
    target_pool = calibration_target_frames if calibration_target_frames is not None else train_frames
    candidates = _calibration_candidates(target_pool, calib_stride, calib_max_views, calib_sampler)
    conf_values: list[torch.Tensor] = []
    mag_values: list[torch.Tensor] = []
    edge_values: list[torch.Tensor] = []
    base_values: list[torch.Tensor] = []
    gt_values: list[torch.Tensor] = []
    signal_values: list[torch.Tensor] = []
    view_id_values: list[torch.Tensor] = []
    for target in candidates:
        support = [frame for frame in train_frames if frame.name != target.name]
        if not support:
            continue
        gt = loader.gt(str(target.gt_path)).to(device)
        ev = compute_evidence_signal(
            target,
            support,
            k=k,
            mode=mode,
            residual_clip=residual_clip,
            min_confidence=1e-8,
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
            direction_weight=direction_weight,
            loader=loader,
            device=device,
        )
        valid_mask = ev.valid
        signal = ev.signal
        if bool(local_trust_gate):
            local_mode = str(local_trust_mode).strip().lower()
            if local_mode == "soft":
                local_weight, _ = local_trust_weight_map(
                    ev,
                    enabled=True,
                    min_supports=int(local_trust_min_supports),
                    max_residual_std=float(local_trust_max_residual_std),
                    min_agreement=float(local_trust_min_agreement),
                    agreement_scale=float(local_trust_agreement_scale),
                    confidence_quantile=float(local_trust_confidence_quantile),
                    min_confidence=float(local_trust_min_confidence),
                    min_weight=float(local_trust_min_weight),
                )
                valid_mask = valid_mask & (local_weight > 0.0)
                signal = signal * local_weight
                signal = torch.where(valid_mask, signal, torch.zeros_like(signal))
            else:
                local_accept, _ = local_trust_acceptance_mask(
                    ev,
                    enabled=True,
                    min_supports=int(local_trust_min_supports),
                    max_residual_std=float(local_trust_max_residual_std),
                    min_agreement=float(local_trust_min_agreement),
                    agreement_scale=float(local_trust_agreement_scale),
                    confidence_quantile=float(local_trust_confidence_quantile),
                    min_confidence=float(local_trust_min_confidence),
                )
                valid_mask = valid_mask & local_accept
                signal = torch.where(valid_mask, signal, torch.zeros_like(signal))
        valid = valid_mask.squeeze(0)
        if not bool(valid.any().item()):
            continue
        conf = torch.log1p(torch.clamp(ev.confidence.squeeze(0), min=0.0))[valid]
        mag = torch.linalg.vector_norm(signal, dim=0)[valid]
        edge = _image_edge_magnitude(ev.base).squeeze(0)[valid]
        base_px = ev.base.permute(1, 2, 0)[valid]
        gt_px = gt.permute(1, 2, 0)[valid]
        sig_px = signal.permute(1, 2, 0)[valid]
        max_samples = max(int(max_pixels_per_view), 1)
        if conf.numel() > max_samples:
            idx = torch.linspace(0, conf.numel() - 1, steps=max_samples, device=device).long()
            conf = conf[idx]
            mag = mag[idx]
            edge = edge[idx]
            base_px = base_px[idx]
            gt_px = gt_px[idx]
            sig_px = sig_px[idx]
        view_id = len(view_id_values)
        conf_values.append(conf.detach().cpu())
        mag_values.append(mag.detach().cpu())
        if feature_mode == "confidence_magnitude_edge":
            edge_values.append(edge.detach().cpu())
        base_values.append(base_px.detach().cpu())
        gt_values.append(gt_px.detach().cpu())
        signal_values.append(sig_px.detach().cpu())
        view_id_values.append(torch.full((int(conf.numel()),), view_id, dtype=torch.int64))
    if not conf_values:
        return AlphaCalibrator(
            confidence_edges=(0.0, 1.0),
            magnitude_edges=(0.0, 1.0),
            alpha_table=((float(default_alpha),),),
            gain_table=((0.0,),),
            count_table=((0,),),
            accept_table=((False,),),
            default_alpha=float(default_alpha),
            min_gain=float(min_gain),
            min_bin_count=int(min_bin_count),
            risk_tail_fraction=float(risk_tail_fraction),
            max_negative_gain_fraction=float(max_negative_gain_fraction),
            min_tail_gain=float(min_tail_gain),
            holdout_safe_zero=bool(holdout_safe_zero),
            region_risk_enabled=False,
            region_risk_json=str(region_risk_json or ""),
            region_risk_objective_bad_only=bool(region_risk_objective_bad_only),
            region_risk_objective_max_balanced_delta=float(region_risk_objective_max_balanced_delta),
            region_risk_objective_max_delta_ssim=float(region_risk_objective_max_delta_ssim),
            region_risk_objective_min_delta_lpips=float(region_risk_objective_min_delta_lpips),
            region_risk_min_tail_gain=float(region_risk_min_tail_gain),
            region_risk_max_negative_fraction=float(region_risk_max_negative_fraction),
            region_risk_min_regions=max(int(region_risk_min_regions), 1),
            view_tail_objective=view_tail_objective_value,
            view_tail_ssim_weight=float(view_tail_ssim_weight),
            view_tail_lpips_weight=float(view_tail_lpips_weight),
            view_tail_compute_lpips=bool(view_tail_compute_lpips),
            view_tail_metric_max_side=max(int(view_tail_metric_max_side), 0),
            feature_mode=feature_mode,
        )
    conf_all = torch.cat(conf_values).float()
    mag_all = torch.cat(mag_values).float()
    base_all = torch.cat(base_values).float()
    gt_all = torch.cat(gt_values).float()
    sig_all = torch.cat(signal_values).float()
    view_id_all = torch.cat(view_id_values).to(torch.int64)
    bin_count = max(int(bins), 1)
    quantiles = torch.linspace(0.0, 1.0, steps=bin_count + 1)
    conf_edges = _strictly_increasing_edges(torch.quantile(conf_all, quantiles).float())
    mag_edges = _strictly_increasing_edges(torch.quantile(mag_all, quantiles).float())
    conf_edges[0] = min(conf_edges[0], conf_all.min()) - 1e-6
    conf_edges[-1] = max(conf_edges[-1], conf_all.max()) + 1e-6
    mag_edges[0] = min(mag_edges[0], mag_all.min()) - 1e-6
    mag_edges[-1] = max(mag_edges[-1], mag_all.max()) + 1e-6
    conf_edges = _strictly_increasing_edges(conf_edges)
    mag_edges = _strictly_increasing_edges(mag_edges)
    conf_idx = torch.bucketize(conf_all, conf_edges[1:-1], right=False)
    mag_idx = torch.bucketize(mag_all, mag_edges[1:-1], right=False)
    edge_edges_tuple: tuple[float, ...] | None = None
    if feature_mode == "confidence_magnitude_edge":
        edge_all = torch.cat(edge_values).float()
        edge_edges = _strictly_increasing_edges(torch.quantile(edge_all, quantiles).float())
        edge_edges[0] = min(edge_edges[0], edge_all.min()) - 1e-6
        edge_edges[-1] = max(edge_edges[-1], edge_all.max()) + 1e-6
        edge_edges = _strictly_increasing_edges(edge_edges)
        edge_idx = torch.bucketize(edge_all, edge_edges[1:-1], right=False)
        flat_idx = conf_idx * bin_count * bin_count + mag_idx * bin_count + edge_idx
        table_shape = (bin_count, bin_count, bin_count)
        edge_edges_tuple = tuple(float(x) for x in edge_edges.tolist())
    else:
        flat_idx = conf_idx * bin_count + mag_idx
        table_shape = (bin_count, bin_count)
    flat_bins = int(np.prod(table_shape))
    counts = torch.bincount(flat_idx, minlength=flat_bins).to(torch.int64)
    base_err = torch.mean((base_all - gt_all) ** 2, dim=1)
    alpha_values = [float(a) for a in alpha_grid]
    if not alpha_values:
        alpha_values = [0.0, 0.5, 0.875, 1.0]
    gain_by_alpha = []
    tail_gain_by_alpha = []
    negative_fraction_by_alpha = []
    safe_by_alpha = []
    tail_fraction = max(min(float(risk_tail_fraction), 1.0), 1e-6)
    for alpha in alpha_values:
        pred = torch.clamp(base_all + float(alpha) * sig_all, 0.0, 1.0)
        gain = base_err - torch.mean((pred - gt_all) ** 2, dim=1)
        gain_sum = torch.zeros(flat_bins, dtype=torch.float64)
        gain_sum.scatter_add_(0, flat_idx, gain.to(torch.float64))
        mean_gain = torch.where(counts > 0, gain_sum / torch.clamp(counts, min=1), torch.zeros_like(gain_sum))
        negative_sum = torch.bincount(
            flat_idx,
            weights=(gain < 0.0).to(torch.float64),
            minlength=flat_bins,
        )
        negative_fraction = torch.where(
            counts > 0,
            negative_sum / torch.clamp(counts, min=1).to(torch.float64),
            torch.ones_like(mean_gain),
        )
        tail_gain = torch.zeros(flat_bins, dtype=torch.float64)
        for bin_idx in range(flat_bins):
            bin_gain = gain[flat_idx == int(bin_idx)]
            if bin_gain.numel() <= 0:
                continue
            tail_count = max(1, int(math.ceil(tail_fraction * int(bin_gain.numel()))))
            tail_gain[int(bin_idx)] = torch.topk(bin_gain.to(torch.float64), k=tail_count, largest=False).values.mean()
        safe = (mean_gain > float(min_gain)) & (counts >= int(min_bin_count))
        if bool(holdout_safe_zero):
            safe = (
                safe
                & (tail_gain >= float(min_tail_gain))
                & (negative_fraction <= float(max_negative_gain_fraction))
            )
        gain_by_alpha.append(mean_gain)
        tail_gain_by_alpha.append(tail_gain)
        negative_fraction_by_alpha.append(negative_fraction)
        safe_by_alpha.append(safe)
    region_tail_gain_by_alpha: list[torch.Tensor] = []
    region_negative_fraction_by_alpha: list[torch.Tensor] = []
    region_count_by_alpha: list[torch.Tensor] = []
    region_safe_by_alpha: list[torch.Tensor] = []
    region_boxes = _load_region_risk_bboxes(
        region_risk_json,
        objective_bad_only=bool(region_risk_objective_bad_only),
        objective_max_balanced_delta=float(region_risk_objective_max_balanced_delta),
        objective_max_delta_ssim=float(region_risk_objective_max_delta_ssim),
        objective_min_delta_lpips=float(region_risk_objective_min_delta_lpips),
    )
    region_risk_enabled = bool(region_boxes)
    if region_risk_enabled:
        gains_by_alpha_bin: list[list[list[float]]] = [[[] for _ in range(flat_bins)] for _ in alpha_values]
        for target in candidates:
            target_keys = {
                str(target.name),
                Path(str(target.name)).stem,
                str(target.camera.image_name),
                Path(str(target.camera.image_name)).stem,
            }
            boxes: list[tuple[int, int, int, int]] = []
            for key in target_keys:
                boxes.extend(region_boxes.get(key, []))
            if not boxes:
                continue
            support = [frame for frame in train_frames if frame.name != target.name]
            if not support:
                continue
            gt = loader.gt(str(target.gt_path)).to(device)
            ev = compute_evidence_signal(
                target,
                support,
                k=k,
                mode=mode,
                residual_clip=residual_clip,
                min_confidence=1e-8,
                depth_abs_tol=depth_abs_tol,
                depth_rel_tol=depth_rel_tol,
                direction_weight=direction_weight,
                loader=loader,
                device=device,
            )
            valid_mask = ev.valid
            signal_full = ev.signal
            if bool(local_trust_gate):
                local_mode = str(local_trust_mode).strip().lower()
                if local_mode == "soft":
                    local_weight, _ = local_trust_weight_map(
                        ev,
                        enabled=True,
                        min_supports=int(local_trust_min_supports),
                        max_residual_std=float(local_trust_max_residual_std),
                        min_agreement=float(local_trust_min_agreement),
                        agreement_scale=float(local_trust_agreement_scale),
                        confidence_quantile=float(local_trust_confidence_quantile),
                        min_confidence=float(local_trust_min_confidence),
                        min_weight=float(local_trust_min_weight),
                    )
                    valid_mask = valid_mask & (local_weight > 0.0)
                    signal_full = signal_full * local_weight
                    signal_full = torch.where(valid_mask, signal_full, torch.zeros_like(signal_full))
                else:
                    local_accept, _ = local_trust_acceptance_mask(
                        ev,
                        enabled=True,
                        min_supports=int(local_trust_min_supports),
                        max_residual_std=float(local_trust_max_residual_std),
                        min_agreement=float(local_trust_min_agreement),
                        agreement_scale=float(local_trust_agreement_scale),
                        confidence_quantile=float(local_trust_confidence_quantile),
                        min_confidence=float(local_trust_min_confidence),
                    )
                    valid_mask = valid_mask & local_accept
                    signal_full = torch.where(valid_mask, signal_full, torch.zeros_like(signal_full))
            height, width = int(ev.base.shape[1]), int(ev.base.shape[2])
            conf_feature = torch.log1p(torch.clamp(ev.confidence.squeeze(0), min=0.0))
            mag_feature = torch.linalg.vector_norm(signal_full, dim=0)
            conf_full_idx = torch.bucketize(conf_feature.reshape(-1), conf_edges[1:-1].to(device), right=False)
            mag_full_idx = torch.bucketize(mag_feature.reshape(-1), mag_edges[1:-1].to(device), right=False)
            if feature_mode == "confidence_magnitude_edge":
                edge_feature = _image_edge_magnitude(ev.base).squeeze(0)
                assert edge_edges_tuple is not None
                edge_edges_tensor = torch.tensor(edge_edges_tuple, device=device, dtype=torch.float32)
                edge_full_idx = torch.bucketize(edge_feature.reshape(-1), edge_edges_tensor[1:-1], right=False)
                flat_full_idx = conf_full_idx * bin_count * bin_count + mag_full_idx * bin_count + edge_full_idx
            else:
                flat_full_idx = conf_full_idx * bin_count + mag_full_idx
            flat_full_idx = flat_full_idx.reshape(height, width)
            valid_full = valid_mask.squeeze(0)
            for raw_x0, raw_y0, raw_x1, raw_y1 in boxes:
                x0 = max(0, min(width, int(raw_x0)))
                x1 = max(0, min(width, int(raw_x1)))
                y0 = max(0, min(height, int(raw_y0)))
                y1 = max(0, min(height, int(raw_y1)))
                if x1 <= x0 or y1 <= y0:
                    continue
                valid_crop = valid_full[y0:y1, x0:x1].reshape(-1)
                if not bool(valid_crop.any().item()):
                    continue
                active_bins = flat_full_idx[y0:y1, x0:x1].reshape(-1)[valid_crop].detach().cpu()
                if active_bins.numel() <= 0:
                    continue
                region_bin = int(torch.bincount(active_bins.to(torch.int64), minlength=flat_bins).argmax().item())
                base_crop = ev.base[:, y0:y1, x0:x1]
                signal_crop = signal_full[:, y0:y1, x0:x1]
                gt_crop = gt[:, y0:y1, x0:x1]
                base_region_mse = torch.mean((base_crop - gt_crop) ** 2)
                for alpha_idx, alpha in enumerate(alpha_values):
                    pred = torch.clamp(base_crop + float(alpha) * signal_crop, 0.0, 1.0)
                    gain = base_region_mse - torch.mean((pred - gt_crop) ** 2)
                    gains_by_alpha_bin[alpha_idx][region_bin].append(float(gain.detach().cpu().item()))
        min_region_count = max(int(region_risk_min_regions), 1)
        for alpha_idx in range(len(alpha_values)):
            region_tail = torch.full((flat_bins,), -math.inf, dtype=torch.float64)
            region_negative = torch.ones((flat_bins,), dtype=torch.float64)
            region_counts = torch.zeros((flat_bins,), dtype=torch.int64)
            for bin_idx, values in enumerate(gains_by_alpha_bin[alpha_idx]):
                if not values:
                    continue
                value_tensor = torch.tensor(values, dtype=torch.float64)
                region_counts[bin_idx] = int(value_tensor.numel())
                region_tail[bin_idx] = _tail_mean(value_tensor, tail_fraction)
                region_negative[bin_idx] = float((value_tensor < 0.0).to(torch.float64).mean().item())
            region_safe = (region_counts < min_region_count) | (
                (region_tail >= float(region_risk_min_tail_gain))
                & (region_negative <= float(region_risk_max_negative_fraction))
            )
            region_tail_gain_by_alpha.append(region_tail)
            region_negative_fraction_by_alpha.append(region_negative)
            region_count_by_alpha.append(region_counts)
            region_safe_by_alpha.append(region_safe)
            if bool(holdout_safe_zero):
                safe_by_alpha[alpha_idx] = safe_by_alpha[alpha_idx] & region_safe
    gain_stack = torch.stack(gain_by_alpha, dim=0)
    safe_stack = torch.stack(safe_by_alpha, dim=0)
    if bool(holdout_safe_zero):
        safe_scores = torch.where(safe_stack, gain_stack, torch.full_like(gain_stack, -math.inf))
        has_safe = safe_stack.any(dim=0)
        safe_best_idx = torch.argmax(safe_scores, dim=0)
        raw_best_idx = torch.argmax(gain_stack, dim=0)
        best_idx = torch.where(has_safe, safe_best_idx, raw_best_idx)
    else:
        best_idx = torch.argmax(gain_stack, dim=0)
        has_safe = torch.ones_like(counts, dtype=torch.bool)
    best_gain = torch.gather(gain_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
    tail_gain_stack = torch.stack(tail_gain_by_alpha, dim=0)
    negative_fraction_stack = torch.stack(negative_fraction_by_alpha, dim=0)
    best_tail_gain = torch.gather(tail_gain_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
    best_negative_fraction = torch.gather(negative_fraction_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
    best_region_tail_gain = None
    best_region_negative_fraction = None
    best_region_count = None
    region_risk_zeroed = None
    if region_risk_enabled:
        region_tail_stack = torch.stack(region_tail_gain_by_alpha, dim=0)
        region_negative_stack = torch.stack(region_negative_fraction_by_alpha, dim=0)
        region_count_stack = torch.stack(region_count_by_alpha, dim=0)
        best_region_tail_gain = torch.gather(region_tail_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
        best_region_negative_fraction = torch.gather(region_negative_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
        best_region_count = torch.gather(region_count_stack, 0, best_idx.unsqueeze(0)).squeeze(0)
        region_risk_zeroed = (
            (best_region_count >= max(int(region_risk_min_regions), 1))
            & (
                (best_region_tail_gain < float(region_risk_min_tail_gain))
                | (best_region_negative_fraction > float(region_risk_max_negative_fraction))
            )
            & (counts >= int(min_bin_count))
            & (best_gain > float(min_gain))
        )
    alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)[best_idx]
    accept = (best_gain > float(min_gain)) & (counts >= int(min_bin_count)) & has_safe & (alpha_tensor > 0.0)
    if bool(holdout_safe_zero):
        accept = (
            accept
            & (best_tail_gain >= float(min_tail_gain))
            & (best_negative_fraction <= float(max_negative_gain_fraction))
        )
    risk_zeroed = (counts >= int(min_bin_count)) & (best_gain > float(min_gain)) & (~accept)
    alpha_tensor = torch.where(accept, alpha_tensor, torch.full_like(alpha_tensor, float(default_alpha)))
    view_tail_scale = 1.0
    view_tail_enabled = False
    view_tail_scale_tuple: tuple[float, ...] | None = None
    view_tail_mean_gain = 0.0
    view_tail_cvar_gain = 0.0
    view_tail_negative_fraction = 0.0
    view_tail_safe_scale_found = False
    view_tail_fallback_used = False
    view_tail_candidate_stats: list[dict[str, object]] = []
    raw_scale_values = [float(v) for v in (view_tail_scale_grid or []) if math.isfinite(float(v))]
    scale_values = sorted({min(max(v, 0.0), 1.0) for v in raw_scale_values}, reverse=True)
    if scale_values:
        view_tail_enabled = True
        if 0.0 not in scale_values:
            scale_values.append(0.0)
        scale_values = sorted(set(scale_values), reverse=True)
        view_tail_scale_tuple = tuple(float(v) for v in scale_values)
        selected_alpha = alpha_tensor[flat_idx].to(torch.float32)
        view_count = torch.bincount(view_id_all, minlength=len(view_id_values)).to(torch.float64)
        valid_views = view_count > 0
        tail_fraction_view = max(min(float(view_tail_cvar_fraction), 1.0), 1e-6)
        best_stats: tuple[float, float, float, float, float] | None = None
        fallback_stats: tuple[float, float, float, float, float] | None = None
        balanced_rows: dict[float, dict[str, list[float]]] = {}
        if view_tail_objective_value == "balanced":
            from utils.loss_utils import ssim

            lpips_model = None
            if bool(view_tail_compute_lpips):
                from lpipsPyTorch.modules.lpips import LPIPS

                lpips_model = LPIPS("vgg").to(device).eval()
                for param in lpips_model.parameters():
                    param.requires_grad_(False)

            def _metric_image(image: torch.Tensor) -> torch.Tensor:
                max_side = max(int(view_tail_metric_max_side), 0)
                if max_side <= 0:
                    return image
                height, width = int(image.shape[-2]), int(image.shape[-1])
                long_side = max(height, width)
                if long_side <= max_side:
                    return image
                scale = float(max_side) / float(max(long_side, 1))
                out_h = max(1, int(round(float(height) * scale)))
                out_w = max(1, int(round(float(width) * scale)))
                return F.interpolate(
                    image.unsqueeze(0),
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            def _psnr_value(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
                mse = float(torch.mean((lhs - rhs) ** 2).detach().cpu().item())
                return mse_to_psnr(mse)

            for scale in scale_values:
                balanced_rows[float(scale)] = {
                    "score": [],
                    "psnr_gain": [],
                    "ssim_gain": [],
                    "lpips_gain": [],
                    "mse_gain": [],
                }

            alpha_lookup = alpha_tensor.to(device=device, dtype=torch.float32)
            conf_edges_device = conf_edges.to(device=device)
            mag_edges_device = mag_edges.to(device=device)
            edge_edges_device = (
                torch.tensor(edge_edges_tuple, device=device, dtype=torch.float32)
                if edge_edges_tuple is not None
                else None
            )
            for target in candidates:
                support = [frame for frame in train_frames if frame.name != target.name]
                if not support:
                    continue
                gt_full = loader.gt(str(target.gt_path)).to(device)
                ev = compute_evidence_signal(
                    target,
                    support,
                    k=k,
                    mode=mode,
                    residual_clip=residual_clip,
                    min_confidence=1e-8,
                    depth_abs_tol=depth_abs_tol,
                    depth_rel_tol=depth_rel_tol,
                    direction_weight=direction_weight,
                    loader=loader,
                    device=device,
                )
                valid_mask_full = ev.valid
                signal_full = ev.signal
                if bool(local_trust_gate):
                    local_mode = str(local_trust_mode).strip().lower()
                    if local_mode == "soft":
                        local_weight, _ = local_trust_weight_map(
                            ev,
                            enabled=True,
                            min_supports=int(local_trust_min_supports),
                            max_residual_std=float(local_trust_max_residual_std),
                            min_agreement=float(local_trust_min_agreement),
                            agreement_scale=float(local_trust_agreement_scale),
                            confidence_quantile=float(local_trust_confidence_quantile),
                            min_confidence=float(local_trust_min_confidence),
                            min_weight=float(local_trust_min_weight),
                        )
                        valid_mask_full = valid_mask_full & (local_weight > 0.0)
                        signal_full = signal_full * local_weight
                        signal_full = torch.where(valid_mask_full, signal_full, torch.zeros_like(signal_full))
                    else:
                        local_accept, _ = local_trust_acceptance_mask(
                            ev,
                            enabled=True,
                            min_supports=int(local_trust_min_supports),
                            max_residual_std=float(local_trust_max_residual_std),
                            min_agreement=float(local_trust_min_agreement),
                            agreement_scale=float(local_trust_agreement_scale),
                            confidence_quantile=float(local_trust_confidence_quantile),
                            min_confidence=float(local_trust_min_confidence),
                        )
                        valid_mask_full = valid_mask_full & local_accept
                        signal_full = torch.where(valid_mask_full, signal_full, torch.zeros_like(signal_full))
                height, width = int(ev.base.shape[1]), int(ev.base.shape[2])
                conf_feature = torch.log1p(torch.clamp(ev.confidence.squeeze(0), min=0.0))
                mag_feature = torch.linalg.vector_norm(signal_full, dim=0)
                conf_full_idx = torch.bucketize(conf_feature.reshape(-1), conf_edges_device[1:-1], right=False)
                mag_full_idx = torch.bucketize(mag_feature.reshape(-1), mag_edges_device[1:-1], right=False)
                if feature_mode == "confidence_magnitude_edge":
                    assert edge_edges_device is not None
                    edge_feature = _image_edge_magnitude(ev.base).squeeze(0)
                    edge_full_idx = torch.bucketize(edge_feature.reshape(-1), edge_edges_device[1:-1], right=False)
                    flat_full_idx = conf_full_idx * bin_count * bin_count + mag_full_idx * bin_count + edge_full_idx
                else:
                    flat_full_idx = conf_full_idx * bin_count + mag_full_idx
                alpha_full = alpha_lookup[flat_full_idx].reshape(1, height, width)
                alpha_full = torch.where(valid_mask_full, alpha_full, torch.zeros_like(alpha_full))
                base_metric = _metric_image(ev.base)
                gt_metric = _metric_image(gt_full)
                delta_metric = _metric_image(alpha_full * signal_full)
                base_psnr = _psnr_value(base_metric, gt_metric)
                base_ssim = float(ssim(base_metric.unsqueeze(0), gt_metric.unsqueeze(0)).detach().cpu().item())
                base_lpips = 0.0
                if lpips_model is not None:
                    with torch.no_grad():
                        base_lpips = float(lpips_model(base_metric.unsqueeze(0), gt_metric.unsqueeze(0)).detach().cpu().item())
                base_mse_metric = float(torch.mean((base_metric - gt_metric) ** 2).detach().cpu().item())
                for scale in scale_values:
                    pred = torch.clamp(base_metric + float(scale) * delta_metric, 0.0, 1.0)
                    pred_psnr = _psnr_value(pred, gt_metric)
                    pred_ssim = float(ssim(pred.unsqueeze(0), gt_metric.unsqueeze(0)).detach().cpu().item())
                    pred_lpips = 0.0
                    if lpips_model is not None:
                        with torch.no_grad():
                            pred_lpips = float(lpips_model(pred.unsqueeze(0), gt_metric.unsqueeze(0)).detach().cpu().item())
                    pred_mse_metric = float(torch.mean((pred - gt_metric) ** 2).detach().cpu().item())
                    psnr_gain = float(pred_psnr - base_psnr)
                    ssim_gain = float(pred_ssim - base_ssim)
                    lpips_gain = float(base_lpips - pred_lpips) if lpips_model is not None else 0.0
                    mse_gain = float(base_mse_metric - pred_mse_metric)
                    score_gain = (
                        psnr_gain
                        + float(view_tail_ssim_weight) * ssim_gain
                        + float(view_tail_lpips_weight) * lpips_gain
                    )
                    row = balanced_rows[float(scale)]
                    row["score"].append(score_gain)
                    row["psnr_gain"].append(psnr_gain)
                    row["ssim_gain"].append(ssim_gain)
                    row["lpips_gain"].append(lpips_gain)
                    row["mse_gain"].append(mse_gain)

        for scale in scale_values:
            extra_stats: dict[str, object] = {"objective": view_tail_objective_value}
            if view_tail_objective_value == "balanced":
                row = balanced_rows.get(float(scale), {})
                view_gain = torch.tensor(row.get("score", []), dtype=torch.float64)
                if view_gain.numel() == 0:
                    continue
                psnr_gain_values = torch.tensor(row.get("psnr_gain", []), dtype=torch.float64)
                ssim_gain_values = torch.tensor(row.get("ssim_gain", []), dtype=torch.float64)
                lpips_gain_values = torch.tensor(row.get("lpips_gain", []), dtype=torch.float64)
                mse_gain_values = torch.tensor(row.get("mse_gain", []), dtype=torch.float64)
                extra_stats.update(
                    {
                        "mean_score": float(view_gain.mean().item()),
                        "mean_psnr_gain": float(psnr_gain_values.mean().item()),
                        "mean_ssim_gain": float(ssim_gain_values.mean().item()),
                        "mean_lpips_gain": float(lpips_gain_values.mean().item()),
                        "mean_mse_gain": float(mse_gain_values.mean().item()),
                        "lpips_regression_fraction": float(
                            (lpips_gain_values < 0.0).to(torch.float64).mean().item()
                        )
                        if lpips_gain_values.numel() > 0
                        else 0.0,
                    }
                )
            else:
                pred = torch.clamp(base_all + float(scale) * selected_alpha.unsqueeze(1) * sig_all, 0.0, 1.0)
                gain = base_err - torch.mean((pred - gt_all) ** 2, dim=1)
                view_gain_sum = torch.zeros(len(view_id_values), dtype=torch.float64)
                view_gain_sum.scatter_add_(0, view_id_all, gain.to(torch.float64))
                view_gain = torch.where(
                    view_count > 0,
                    view_gain_sum / torch.clamp(view_count, min=1.0),
                    torch.zeros_like(view_gain_sum),
                )[valid_views]
                if view_gain.numel() == 0:
                    continue
                extra_stats["mean_mse_gain"] = float(view_gain.mean().item())
            mean_gain_scale = float(view_gain.mean().item())
            tail_count = max(1, int(math.ceil(tail_fraction_view * int(view_gain.numel()))))
            cvar_gain = float(torch.topk(view_gain, k=tail_count, largest=False).values.mean().item())
            negative_fraction = float((view_gain < 0.0).to(torch.float64).mean().item())
            stats = (float(scale), mean_gain_scale, cvar_gain, negative_fraction, float(scale))
            if fallback_stats is None or cvar_gain > fallback_stats[2] or (
                cvar_gain == fallback_stats[2] and mean_gain_scale > fallback_stats[1]
            ):
                fallback_stats = stats
            safe = (
                cvar_gain >= float(view_tail_min_gain)
                and negative_fraction <= float(view_tail_max_negative_fraction)
            )
            view_tail_candidate_stats.append(
                {
                    "scale": float(scale),
                    "mean_gain": mean_gain_scale,
                    "cvar_gain": cvar_gain,
                    "negative_fraction": negative_fraction,
                    "safe": bool(safe),
                    **extra_stats,
                }
            )
            if safe and (
                best_stats is None
                or mean_gain_scale > best_stats[1]
                or (mean_gain_scale == best_stats[1] and scale > best_stats[0])
            ):
                best_stats = stats
        view_tail_safe_scale_found = best_stats is not None
        view_tail_fallback_used = best_stats is None and fallback_stats is not None
        selected_stats = best_stats or fallback_stats
        if selected_stats is not None:
            view_tail_scale = float(selected_stats[0])
            view_tail_mean_gain = float(selected_stats[1])
            view_tail_cvar_gain = float(selected_stats[2])
            view_tail_negative_fraction = float(selected_stats[3])
    return AlphaCalibrator(
        confidence_edges=tuple(float(x) for x in conf_edges.tolist()),
        magnitude_edges=tuple(float(x) for x in mag_edges.tolist()),
        edge_edges=edge_edges_tuple,
        alpha_table=tuple(tuple(tuple(float(v) for v in col) for col in row) for row in alpha_tensor.reshape(table_shape).tolist())
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(float(v) for v in row) for row in alpha_tensor.reshape(table_shape).tolist()),
        gain_table=tuple(tuple(tuple(float(v) for v in col) for col in row) for row in best_gain.reshape(table_shape).tolist())
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(float(v) for v in row) for row in best_gain.reshape(table_shape).tolist()),
        count_table=tuple(tuple(tuple(int(v) for v in col) for col in row) for row in counts.reshape(table_shape).tolist())
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(int(v) for v in row) for row in counts.reshape(table_shape).tolist()),
        accept_table=tuple(tuple(tuple(bool(v) for v in col) for col in row) for row in accept.reshape(table_shape).tolist())
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(bool(v) for v in row) for row in accept.reshape(table_shape).tolist()),
        tail_gain_table=tuple(
            tuple(tuple(float(v) for v in col) for col in row)
            for row in best_tail_gain.reshape(table_shape).tolist()
        )
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(float(v) for v in row) for row in best_tail_gain.reshape(table_shape).tolist()),
        negative_fraction_table=tuple(
            tuple(tuple(float(v) for v in col) for col in row)
            for row in best_negative_fraction.reshape(table_shape).tolist()
        )
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(float(v) for v in row) for row in best_negative_fraction.reshape(table_shape).tolist()),
        risk_zeroed_table=tuple(
            tuple(tuple(bool(v) for v in col) for col in row)
            for row in risk_zeroed.reshape(table_shape).tolist()
        )
        if feature_mode == "confidence_magnitude_edge"
        else tuple(tuple(bool(v) for v in row) for row in risk_zeroed.reshape(table_shape).tolist()),
        region_tail_gain_table=(
            tuple(
                tuple(tuple(float(v) for v in col) for col in row)
                for row in best_region_tail_gain.reshape(table_shape).tolist()
            )
            if feature_mode == "confidence_magnitude_edge"
            else tuple(tuple(float(v) for v in row) for row in best_region_tail_gain.reshape(table_shape).tolist())
        )
        if best_region_tail_gain is not None
        else None,
        region_negative_fraction_table=(
            tuple(
                tuple(tuple(float(v) for v in col) for col in row)
                for row in best_region_negative_fraction.reshape(table_shape).tolist()
            )
            if feature_mode == "confidence_magnitude_edge"
            else tuple(
                tuple(float(v) for v in row) for row in best_region_negative_fraction.reshape(table_shape).tolist()
            )
        )
        if best_region_negative_fraction is not None
        else None,
        region_count_table=(
            tuple(
                tuple(tuple(int(v) for v in col) for col in row)
                for row in best_region_count.reshape(table_shape).tolist()
            )
            if feature_mode == "confidence_magnitude_edge"
            else tuple(tuple(int(v) for v in row) for row in best_region_count.reshape(table_shape).tolist())
        )
        if best_region_count is not None
        else None,
        region_risk_zeroed_table=(
            tuple(
                tuple(tuple(bool(v) for v in col) for col in row)
                for row in region_risk_zeroed.reshape(table_shape).tolist()
            )
            if feature_mode == "confidence_magnitude_edge"
            else tuple(tuple(bool(v) for v in row) for row in region_risk_zeroed.reshape(table_shape).tolist())
        )
        if region_risk_zeroed is not None
        else None,
        default_alpha=float(default_alpha),
        min_gain=float(min_gain),
        min_bin_count=int(min_bin_count),
        risk_tail_fraction=float(risk_tail_fraction),
        max_negative_gain_fraction=float(max_negative_gain_fraction),
        min_tail_gain=float(min_tail_gain),
        holdout_safe_zero=bool(holdout_safe_zero),
        region_risk_enabled=bool(region_risk_enabled),
        region_risk_json=str(region_risk_json or ""),
        region_risk_objective_bad_only=bool(region_risk_objective_bad_only),
        region_risk_objective_max_balanced_delta=float(region_risk_objective_max_balanced_delta),
        region_risk_objective_max_delta_ssim=float(region_risk_objective_max_delta_ssim),
        region_risk_objective_min_delta_lpips=float(region_risk_objective_min_delta_lpips),
        region_risk_min_tail_gain=float(region_risk_min_tail_gain),
        region_risk_max_negative_fraction=float(region_risk_max_negative_fraction),
        region_risk_min_regions=max(int(region_risk_min_regions), 1),
        view_tail_scale=float(view_tail_scale),
        view_tail_enabled=bool(view_tail_enabled),
        view_tail_scale_grid=view_tail_scale_tuple,
        view_tail_cvar_fraction=float(view_tail_cvar_fraction),
        view_tail_min_gain=float(view_tail_min_gain),
        view_tail_max_negative_fraction=float(view_tail_max_negative_fraction),
        view_tail_objective=view_tail_objective_value,
        view_tail_ssim_weight=float(view_tail_ssim_weight),
        view_tail_lpips_weight=float(view_tail_lpips_weight),
        view_tail_compute_lpips=bool(view_tail_compute_lpips),
        view_tail_metric_max_side=max(int(view_tail_metric_max_side), 0),
        view_tail_mean_gain=float(view_tail_mean_gain),
        view_tail_cvar_gain=float(view_tail_cvar_gain),
        view_tail_negative_fraction=float(view_tail_negative_fraction),
        view_tail_safe_scale_found=bool(view_tail_safe_scale_found),
        view_tail_fallback_used=bool(view_tail_fallback_used),
        view_tail_candidate_stats=tuple(view_tail_candidate_stats) if view_tail_candidate_stats else None,
        feature_mode=feature_mode,
    )


def adapt_frame(
    target: FrameRecord,
    support_frames: Sequence[FrameRecord],
    *,
    k: int = 4,
    alpha: float = 1.0,
    mode: str = "residual",
    residual_clip: float = 0.25,
    min_confidence: float = 1e-4,
    depth_abs_tol: float = 0.02,
    depth_rel_tol: float = 0.03,
    direction_weight: float = 0.35,
    benefit_calibrator: BenefitCalibrator | None = None,
    alpha_calibrator: AlphaCalibrator | None = None,
    edge_gate: bool = False,
    edge_gate_quantile: float = -1.0,
    edge_gate_min: float = 0.0,
    edge_gate_dilate: int = 0,
    local_trust_gate: bool = False,
    local_trust_min_supports: int = 2,
    local_trust_max_residual_std: float = -1.0,
    local_trust_min_agreement: float = 0.0,
    local_trust_agreement_scale: float = 0.04,
    local_trust_confidence_quantile: float = -1.0,
    local_trust_min_confidence: float = 0.0,
    local_trust_mode: str = "hard",
    local_trust_min_weight: float = 0.0,
    evidence_max_side: int = 0,
    loader: FrameLoader | None = None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, dict[str, object]]:
    loader = loader or FrameLoader(device=device)
    device = torch.device(device)
    evidence = compute_evidence_signal(
        target,
        support_frames,
        k=k,
        mode=mode,
        residual_clip=residual_clip,
        min_confidence=min_confidence,
        depth_abs_tol=depth_abs_tol,
        depth_rel_tol=depth_rel_tol,
        direction_weight=direction_weight,
        evidence_max_side=int(evidence_max_side),
        loader=loader,
        device=device,
    )
    base = evidence.base
    signal = evidence.signal
    valid = evidence.valid
    local_trust_info: dict[str, float | int | bool] | None = None
    if bool(local_trust_gate) and mode == "residual":
        local_mode = str(local_trust_mode).strip().lower()
        if local_mode == "soft":
            local_weight, local_trust_info = local_trust_weight_map(
                evidence,
                enabled=True,
                min_supports=int(local_trust_min_supports),
                max_residual_std=float(local_trust_max_residual_std),
                min_agreement=float(local_trust_min_agreement),
                agreement_scale=float(local_trust_agreement_scale),
                confidence_quantile=float(local_trust_confidence_quantile),
                min_confidence=float(local_trust_min_confidence),
                min_weight=float(local_trust_min_weight),
            )
            valid = valid & (local_weight > 0.0)
            signal = signal * local_weight
            signal = torch.where(valid, signal, torch.zeros_like(signal))
        else:
            local_accept, local_trust_info = local_trust_acceptance_mask(
                evidence,
                enabled=True,
                min_supports=int(local_trust_min_supports),
                max_residual_std=float(local_trust_max_residual_std),
                min_agreement=float(local_trust_min_agreement),
                agreement_scale=float(local_trust_agreement_scale),
                confidence_quantile=float(local_trust_confidence_quantile),
                min_confidence=float(local_trust_min_confidence),
            )
            local_trust_info["mode"] = "hard"
            local_trust_info["min_weight"] = float(local_trust_min_weight)
            valid = valid & local_accept
            signal = torch.where(valid, signal, torch.zeros_like(signal))
    elif bool(local_trust_gate):
        local_trust_info = {"enabled": False, "requested": True}
    benefit_accept = None
    if benefit_calibrator is not None and mode == "residual":
        benefit_accept = benefit_calibrator.acceptance_mask(evidence.confidence, signal, base=evidence.base, device=device)
        valid = valid & benefit_accept
        signal = torch.where(valid, signal, torch.zeros_like(signal))
    edge_accept_fraction = None
    edge_threshold = None
    if bool(edge_gate):
        edge_accept, edge_threshold = _edge_acceptance_mask(
            base,
            valid,
            quantile=float(edge_gate_quantile),
            min_edge=float(edge_gate_min),
            dilate=int(edge_gate_dilate),
        )
        valid = valid & edge_accept
        signal = torch.where(valid, signal, torch.zeros_like(signal))
        edge_accept_fraction = float(edge_accept.to(torch.float32).mean().detach().cpu().item())
    alpha_mean = None
    alpha_active_fraction = None
    if mode == "residual":
        if alpha_calibrator is not None:
            alpha_map, alpha_active = alpha_calibrator.alpha_map(
                evidence.confidence,
                signal,
                base=evidence.base,
                device=device,
            )
            alpha_map = torch.where(valid, alpha_map, torch.zeros_like(alpha_map))
            alpha_mean = float(alpha_map[valid].mean().detach().cpu().item()) if bool(valid.any().item()) else 0.0
            alpha_active_fraction = float((alpha_active & valid).to(torch.float32).mean().detach().cpu().item())
            adapted = torch.clamp(base + alpha_map * signal, 0.0, 1.0)
        else:
            adapted = torch.clamp(base + float(alpha) * signal, 0.0, 1.0)
    else:
        blend = torch.clamp(signal, 0.0, 1.0)
        adapted = torch.where(valid, torch.clamp(base * (1.0 - float(alpha)) + blend * float(alpha), 0.0, 1.0), base)
    info = {
        "support_count": int(len(evidence.support_names)),
        "support_names": evidence.support_names,
        "mean_confidence": float(evidence.confidence.mean().detach().cpu().item()),
        "covered_fraction": float(valid.to(torch.float32).mean().detach().cpu().item()),
        "evidence_max_side": int(evidence_max_side),
        "evidence_scaled": bool(int(evidence_max_side) > 0 and max(int(base.shape[1]), int(base.shape[2])) > int(evidence_max_side)),
    }
    evidence_valid = evidence.valid.bool()
    if evidence.support_count is not None:
        support_values = evidence.support_count[evidence_valid]
        info["mean_signal_support_count"] = (
            float(support_values.mean().detach().cpu().item()) if support_values.numel() else 0.0
        )
    if evidence.residual_std is not None:
        std_values = evidence.residual_std[evidence_valid]
        info["mean_signal_residual_std"] = (
            float(std_values.mean().detach().cpu().item()) if std_values.numel() else 0.0
        )
    if local_trust_info is not None:
        info["local_trust_enabled"] = bool(local_trust_info.get("enabled", False))
        for key, value in local_trust_info.items():
            if key == "enabled":
                continue
            info[f"local_trust_{key}"] = value
    if benefit_accept is not None:
        info["benefit_accept_fraction"] = float(benefit_accept.to(torch.float32).mean().detach().cpu().item())
    if edge_accept_fraction is not None:
        info["edge_accept_fraction"] = edge_accept_fraction
        info["edge_threshold"] = float(edge_threshold or 0.0)
    if alpha_mean is not None:
        info["alpha_mean"] = alpha_mean
        info["alpha_active_fraction"] = float(alpha_active_fraction or 0.0)
    return adapted, info


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def calibrate_alpha(
    train_frames: Sequence[FrameRecord],
    *,
    calibration_target_frames: Sequence[FrameRecord] | None = None,
    alpha_grid: Sequence[float],
    k: int,
    mode: str,
    calib_stride: int,
    calib_max_views: int,
    residual_clip: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    direction_weight: float,
    benefit_calibrator: BenefitCalibrator | None = None,
    edge_gate: bool = False,
    edge_gate_quantile: float = -1.0,
    edge_gate_min: float = 0.0,
    edge_gate_dilate: int = 0,
    local_trust_gate: bool = False,
    local_trust_min_supports: int = 2,
    local_trust_max_residual_std: float = -1.0,
    local_trust_min_agreement: float = 0.0,
    local_trust_agreement_scale: float = 0.04,
    local_trust_confidence_quantile: float = -1.0,
    local_trust_min_confidence: float = 0.0,
    local_trust_mode: str = "hard",
    local_trust_min_weight: float = 0.0,
    policy_objective: str = "psnr",
    ssim_weight: float = 20.0,
    lpips_weight: float = 20.0,
    compute_lpips: bool = False,
    calib_sampler: str = "stride_first",
    device: torch.device | str = "cuda",
    fd_judge: object | None = None,
    fd_weight: float = 0.0,
    fd_strict: bool = False,
    fd_strict_tol: float = 0.0,
    fd_max_views: int = 32,
    fd_min_views: int = 8,
) -> dict[str, object]:
    if not train_frames:
        return {"alpha": 0.0, "reason": "no_train_frames", "rows": []}
    device = torch.device(device)
    target_pool = calibration_target_frames if calibration_target_frames is not None else train_frames
    candidates = _calibration_candidates(target_pool, calib_stride, calib_max_views, calib_sampler)
    loader = FrameLoader(device=device)
    use_ssim = policy_objective != "psnr"
    lpips_model = None
    if compute_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS

        lpips_model = LPIPS("vgg").to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad_(False)
    fd_use = fd_judge is not None and (float(fd_weight) > 0.0 or bool(fd_strict))
    fd_feats_base: list[torch.Tensor] = []
    fd_feats_gt: list[torch.Tensor] = []
    fd_feats_alpha: dict[float, list[torch.Tensor]] = (
        {float(a): [] for a in alpha_grid} if fd_use else {}
    )
    fd_views_taken = 0
    fd_view_budget = max(int(fd_max_views), 1) if fd_use else 0
    rows = {
        float(alpha): {
            "mse": 0.0,
            "base_mse": 0.0,
            "ssim": 0.0,
            "base_ssim": 0.0,
            "lpips": 0.0,
            "base_lpips": 0.0,
            "count": 0,
        }
        for alpha in alpha_grid
    }
    for target in candidates:
        support = [frame for frame in train_frames if frame.name != target.name]
        if not support:
            continue
        gt = loader.gt(str(target.gt_path))
        base = loader.render(str(target.render_path))
        base_mse = float(torch.mean((base - gt) ** 2).detach().cpu().item())
        base_ssim = 0.0
        base_lpips = 0.0
        if use_ssim:
            from utils.loss_utils import ssim

            base_ssim = float(ssim(base.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
        if lpips_model is not None:
            with torch.no_grad():
                base_lpips = float(lpips_model(base.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
        adapted_residual, _ = adapt_frame(
            target,
            support,
            k=k,
            alpha=1.0,
            mode=mode,
            residual_clip=residual_clip,
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
            direction_weight=direction_weight,
            benefit_calibrator=benefit_calibrator,
            edge_gate=edge_gate,
            edge_gate_quantile=edge_gate_quantile,
            edge_gate_min=edge_gate_min,
            edge_gate_dilate=edge_gate_dilate,
            local_trust_gate=local_trust_gate,
            local_trust_min_supports=local_trust_min_supports,
            local_trust_max_residual_std=local_trust_max_residual_std,
            local_trust_min_agreement=local_trust_min_agreement,
            local_trust_agreement_scale=local_trust_agreement_scale,
            local_trust_confidence_quantile=local_trust_confidence_quantile,
            local_trust_min_confidence=local_trust_min_confidence,
            local_trust_mode=local_trust_mode,
            local_trust_min_weight=local_trust_min_weight,
            loader=loader,
            device=device,
        )
        delta = adapted_residual - base
        fd_take = fd_use and fd_views_taken < fd_view_budget
        fd_preds_this_target: dict[float, torch.Tensor] = {}
        for alpha in alpha_grid:
            pred = torch.clamp(base + float(alpha) * delta, 0.0, 1.0)
            if fd_take:
                fd_preds_this_target[float(alpha)] = pred
            mse = float(torch.mean((pred - gt) ** 2).detach().cpu().item())
            rows[float(alpha)]["mse"] += mse
            rows[float(alpha)]["base_mse"] += base_mse
            if use_ssim:
                rows[float(alpha)]["ssim"] += float(ssim(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
                rows[float(alpha)]["base_ssim"] += base_ssim
            if lpips_model is not None:
                with torch.no_grad():
                    rows[float(alpha)]["lpips"] += float(lpips_model(pred.unsqueeze(0), gt.unsqueeze(0)).detach().cpu().item())
                rows[float(alpha)]["base_lpips"] += base_lpips
            rows[float(alpha)]["count"] += 1
        if fd_take:
            alpha_order = list(alpha_grid)
            nonzero_idx = [
                (i, float(a)) for i, a in enumerate(alpha_order) if abs(float(a)) > 1e-9
            ]
            prep_base = fd_judge.prepare(base)
            prep_gt = fd_judge.prepare(gt)
            prep_preds = [
                fd_judge.prepare(fd_preds_this_target[a]) for _, a in nonzero_idx
            ]
            stack = torch.stack([prep_base, prep_gt] + prep_preds, dim=0)
            feats = fd_judge.encode(stack)
            base_feat = feats[0:1].detach().cpu()
            fd_feats_base.append(base_feat)
            fd_feats_gt.append(feats[1:2].detach().cpu())
            for slot, (i, a) in enumerate(nonzero_idx):
                fd_feats_alpha[a].append(feats[2 + slot : 3 + slot].detach().cpu())
            for a_zero in (float(a) for a in alpha_order if abs(float(a)) <= 1e-9):
                fd_feats_alpha[a_zero].append(base_feat)
            fd_views_taken += 1
    fd_summary: dict[float, dict] = {}
    fd_skipped_reason: str | None = None
    if fd_use:
        if fd_views_taken < max(int(fd_min_views), 2):
            fd_skipped_reason = (
                f"insufficient_calibration_views: got {fd_views_taken}, need >= {max(int(fd_min_views), 2)}"
            )
        elif not fd_feats_gt:
            fd_skipped_reason = "empty_gt_features"
        else:
            from utils.fd_loss import frechet_distance

            base_feats = torch.cat(fd_feats_base, dim=0)
            gt_feats = torch.cat(fd_feats_gt, dim=0)
            fd_base_info = frechet_distance(base_feats, gt_feats)
            for alpha in alpha_grid:
                alpha_feats_list = fd_feats_alpha.get(float(alpha), [])
                if not alpha_feats_list:
                    continue
                alpha_feats = torch.cat(alpha_feats_list, dim=0)
                fd_alpha_info = frechet_distance(alpha_feats, gt_feats)
                fd_summary[float(alpha)] = {
                    "fd": float(fd_alpha_info["fd"]),
                    "base_fd": float(fd_base_info["fd"]),
                    "fd_gain": float(fd_base_info["fd"] - fd_alpha_info["fd"]),
                    "fd_views": int(fd_views_taken),
                }
    row_list = []
    best_alpha = 0.0
    best_score = -float("inf")
    for alpha, row in rows.items():
        count = max(int(row["count"]), 1)
        mean_mse = float(row["mse"]) / count
        mean_base = float(row["base_mse"]) / count
        calib_gain = mse_to_psnr(mean_mse) - mse_to_psnr(mean_base)
        mean_ssim = float(row["ssim"]) / count
        mean_base_ssim = float(row["base_ssim"]) / count
        mean_lpips = float(row["lpips"]) / count
        mean_base_lpips = float(row["base_lpips"]) / count
        ssim_gain = mean_ssim - mean_base_ssim if use_ssim else 0.0
        lpips_gain = mean_base_lpips - mean_lpips if lpips_model is not None else 0.0
        selection_score = calib_gain
        if use_ssim:
            selection_score += float(ssim_weight) * ssim_gain
        if lpips_model is not None:
            selection_score += float(lpips_weight) * lpips_gain
        fd_info = fd_summary.get(float(alpha)) if (fd_use and fd_summary) else None
        fd_value = float(fd_info["fd"]) if fd_info is not None else None
        fd_base_value = float(fd_info["base_fd"]) if fd_info is not None else None
        fd_gain_value = float(fd_info["fd_gain"]) if fd_info is not None else 0.0
        fd_views_value = int(fd_info["fd_views"]) if fd_info is not None else 0
        fd_rejected = False
        if fd_info is not None and float(fd_weight) > 0.0:
            selection_score += float(fd_weight) * fd_gain_value
        if (
            fd_info is not None
            and bool(fd_strict)
            and abs(float(alpha)) > 1e-9
            and fd_gain_value < -float(fd_strict_tol)
        ):
            fd_rejected = True
            selection_score = -float("inf")
        row_out = {
            "alpha": float(alpha),
            "mse": mean_mse,
            "base_mse": mean_base,
            "psnr": mse_to_psnr(mean_mse),
            "base_psnr": mse_to_psnr(mean_base),
            "psnr_gain": calib_gain,
            "ssim": mean_ssim if use_ssim else None,
            "base_ssim": mean_base_ssim if use_ssim else None,
            "ssim_gain": ssim_gain if use_ssim else None,
            "lpips": mean_lpips if lpips_model is not None else None,
            "base_lpips": mean_base_lpips if lpips_model is not None else None,
            "lpips_gain": lpips_gain if lpips_model is not None else None,
            "fd": fd_value,
            "base_fd": fd_base_value,
            "fd_gain": fd_gain_value if fd_info is not None else None,
            "fd_views": fd_views_value,
            "fd_rejected": fd_rejected,
            "selection_score": selection_score,
            "views": count,
        }
        row_list.append(row_out)
        if selection_score > best_score:
            best_score = selection_score
            best_alpha = float(alpha)
    row_list.sort(key=lambda x: float(x["alpha"]))
    zero_row = next((row for row in row_list if abs(float(row["alpha"])) < 1e-9), None)
    if zero_row is not None and best_score <= float(zero_row["selection_score"]):
        best_alpha = 0.0
    return {
        "alpha": best_alpha,
        "rows": row_list,
        "calibration_views": [frame.name for frame in candidates],
        "calibration_sampler": str(calib_sampler),
        "policy_objective": policy_objective,
        "ssim_weight": float(ssim_weight),
        "lpips_weight": float(lpips_weight),
        "compute_lpips": bool(compute_lpips),
        "fd_enabled": bool(fd_use and not fd_skipped_reason),
        "fd_requested": bool(fd_use),
        "fd_weight": float(fd_weight),
        "fd_strict": bool(fd_strict),
        "fd_strict_tol": float(fd_strict_tol),
        "fd_min_views": int(fd_min_views),
        "fd_views": int(fd_views_taken),
        "fd_skipped_reason": fd_skipped_reason,
        "local_trust_gate": bool(local_trust_gate),
        "local_trust_min_supports": int(local_trust_min_supports),
        "local_trust_max_residual_std": float(local_trust_max_residual_std),
        "local_trust_min_agreement": float(local_trust_min_agreement),
        "local_trust_agreement_scale": float(local_trust_agreement_scale),
        "local_trust_confidence_quantile": float(local_trust_confidence_quantile),
        "local_trust_min_confidence": float(local_trust_min_confidence),
        "local_trust_mode": str(local_trust_mode),
        "local_trust_min_weight": float(local_trust_min_weight),
    }

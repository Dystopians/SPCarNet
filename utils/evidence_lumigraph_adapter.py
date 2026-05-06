from __future__ import annotations

import json
import math
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


def _image_key(path: Path) -> str:
    return path.stem


def read_image_tensor(path: Path, device: torch.device | str = "cpu") -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).to(device=device, dtype=torch.float32)


def read_depth_tensor(path: Path, device: torch.device | str = "cpu") -> torch.Tensor:
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return torch.from_numpy(depth).to(device=device, dtype=torch.float32)


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


def warp_support_residual(
    target: FrameRecord,
    support: FrameRecord,
    target_depth: torch.Tensor,
    support_depth: torch.Tensor,
    support_residual: torch.Tensor,
    *,
    depth_abs_tol: float = 0.02,
    depth_rel_tol: float = 0.03,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    target_depth = target_depth.to(device=device, dtype=torch.float32)
    support_depth = support_depth.to(device=device, dtype=torch.float32)
    support_residual = support_residual.to(device=device, dtype=torch.float32)

    target_h, target_w = target_depth.shape
    support_h, support_w = support_depth.shape
    world_points = _make_target_world_grid(target.camera, target_depth, device=device)
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

    def residual(self, frame: FrameRecord, residual_clip: float) -> torch.Tensor:
        residual = self.gt(str(frame.gt_path)) - self.render(str(frame.render_path))
        if residual_clip > 0:
            residual = torch.clamp(residual, -float(residual_clip), float(residual_clip))
        return residual


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
    loader: FrameLoader | None = None,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, dict[str, float | int | list[str]]]:
    loader = loader or FrameLoader(device=device)
    device = torch.device(device)
    base = loader.render(str(target.render_path)).to(device)
    target_depth = loader.depth(str(target.depth_path)).to(device)
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
    weight_den = torch.zeros((1, base.shape[1], base.shape[2]), device=device, dtype=torch.float32)
    used: list[str] = []
    for support_frame, view_weight in support:
        support_depth = loader.depth(str(support_frame.depth_path))
        if mode == "residual":
            support_signal = loader.residual(support_frame, residual_clip=residual_clip)
        else:
            support_signal = loader.gt(str(support_frame.gt_path))
        warped, confidence = warp_support_residual(
            target,
            support_frame,
            target_depth,
            support_depth,
            support_signal,
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
            device=device,
        )
        weight = confidence.unsqueeze(0) * float(view_weight)
        if float(weight.mean().item()) <= 0.0:
            continue
        signal_num = signal_num + warped * weight
        weight_den = weight_den + weight
        used.append(support_frame.name)
    valid = weight_den > float(min_confidence)
    signal = torch.where(valid, signal_num / torch.clamp(weight_den, min=1e-8), torch.zeros_like(signal_num))
    if mode == "residual":
        adapted = torch.clamp(base + float(alpha) * signal, 0.0, 1.0)
    else:
        blend = torch.clamp(signal, 0.0, 1.0)
        adapted = torch.where(valid, torch.clamp(base * (1.0 - float(alpha)) + blend * float(alpha), 0.0, 1.0), base)
    info = {
        "support_count": int(len(used)),
        "support_names": used,
        "mean_confidence": float(weight_den.mean().detach().cpu().item()),
        "covered_fraction": float(valid.to(torch.float32).mean().detach().cpu().item()),
    }
    return adapted, info


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1e-12))


def calibrate_alpha(
    train_frames: Sequence[FrameRecord],
    *,
    alpha_grid: Sequence[float],
    k: int,
    mode: str,
    calib_stride: int,
    calib_max_views: int,
    residual_clip: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    direction_weight: float,
    device: torch.device | str = "cuda",
) -> dict[str, object]:
    if not train_frames:
        return {"alpha": 0.0, "reason": "no_train_frames", "rows": []}
    stride = max(int(calib_stride), 1)
    candidates = list(train_frames[::stride])
    if int(calib_max_views) > 0:
        candidates = candidates[: int(calib_max_views)]
    if not candidates:
        candidates = [train_frames[0]]
    loader = FrameLoader(device=device)
    rows = {float(alpha): {"mse": 0.0, "base_mse": 0.0, "count": 0} for alpha in alpha_grid}
    for target in candidates:
        support = [frame for frame in train_frames if frame.name != target.name]
        gt = loader.gt(str(target.gt_path))
        base = loader.render(str(target.render_path))
        base_mse = float(torch.mean((base - gt) ** 2).detach().cpu().item())
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
            loader=loader,
            device=device,
        )
        delta = adapted_residual - base
        for alpha in alpha_grid:
            pred = torch.clamp(base + float(alpha) * delta, 0.0, 1.0)
            mse = float(torch.mean((pred - gt) ** 2).detach().cpu().item())
            rows[float(alpha)]["mse"] += mse
            rows[float(alpha)]["base_mse"] += base_mse
            rows[float(alpha)]["count"] += 1
    row_list = []
    best_alpha = 0.0
    best_mse = float("inf")
    for alpha, row in rows.items():
        count = max(int(row["count"]), 1)
        mean_mse = float(row["mse"]) / count
        mean_base = float(row["base_mse"]) / count
        calib_gain = mse_to_psnr(mean_mse) - mse_to_psnr(mean_base)
        row_out = {
            "alpha": float(alpha),
            "mse": mean_mse,
            "base_mse": mean_base,
            "psnr": mse_to_psnr(mean_mse),
            "base_psnr": mse_to_psnr(mean_base),
            "psnr_gain": calib_gain,
            "views": count,
        }
        row_list.append(row_out)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = float(alpha)
    row_list.sort(key=lambda x: float(x["alpha"]))
    zero_row = next((row for row in row_list if abs(float(row["alpha"])) < 1e-9), None)
    if zero_row is not None and best_mse >= float(zero_row["mse"]):
        best_alpha = 0.0
    return {
        "alpha": best_alpha,
        "rows": row_list,
        "calibration_views": [frame.name for frame in candidates],
    }

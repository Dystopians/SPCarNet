"""GEMS Stage-4 ECR transport renderer (PROTOCOL v1.2.0 §4E).

Applies the frozen Phase-J evidence transport on top of a test view's BASE
render. Consumes ONLY:
 - the evidence cache built by tools/ecr/build_cache.py (train-view renders,
   train GT copies, train median depths, camera index, frozen transport
   config in manifest.json), and
 - the per-test-view BASE products (render + median depth) plus the target
   camera POSE primitives, handed over in memory by run_eval.py.

D4 structural guarantees (audited by tools/audit_test_path.py --ecr):
 - no Camera object crosses into this module — only a plain dict of pose
   primitives (width/height/fov/center/view-matrix), so test-view ground
   truth is unreachable by construction;
 - every disk read is confined to the cache root (realpath-checked) and
   recorded in a read log the audit compares against the manifest;
 - the target FrameRecord's gt_path is a raising sentinel;
 - transport kwargs are frozen ONCE from the manifest; adapt() rehashes the
   kwargs it actually passes on every call so the audit can prove no
   per-test-view parameter injection.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import torch

from utils.evidence_lumigraph_adapter import (
    CameraRecord,
    FrameRecord,
    adapt_frame,
    load_camera_index,
    read_depth_tensor,
    read_image_tensor,
)
from utils.graphics_utils import fov2focal

TARGET_RENDER_PREFIX = "__ecr_target__/render/"
TARGET_DEPTH_PREFIX = "__ecr_target__/depth/"
TARGET_GT_SENTINEL = "__ecr_target_gt_forbidden__"

# Transport kwargs the manifest may freeze (everything else in the manifest's
# transport block is provenance, not an adapt_frame argument).
_ADAPT_KWARG_KEYS = (
    "k", "alpha", "mode", "residual_clip", "min_confidence",
    "depth_abs_tol", "depth_rel_tol", "direction_weight",
    "edge_gate", "edge_gate_quantile", "edge_gate_min", "edge_gate_dilate",
    "local_trust_gate", "local_trust_min_supports",
    "local_trust_max_residual_std", "local_trust_min_agreement",
    "local_trust_agreement_scale", "local_trust_confidence_quantile",
    "local_trust_min_confidence", "local_trust_mode", "local_trust_min_weight",
    "evidence_max_side",
)

# L2 multi-band transport kwargs (tools/ecr/transport_l2.adapt_frame_l2).
_L2_KWARG_KEYS = (
    "k", "alpha", "residual_clip", "min_confidence",
    "depth_abs_tol", "depth_rel_tol", "direction_weight", "bands",
)

# L3 learned-fusion feature kwargs (tools/ecr/fusion.compute_transport_features).
_L3_KWARG_KEYS = (
    "k", "residual_clip", "min_confidence",
    "depth_abs_tol", "depth_rel_tol", "direction_weight",
    "inner_fuse", "bands",
)


def camera_record_from_pose(idx: int, pose: dict) -> CameraRecord:
    """CameraRecord from plain pose primitives (mirrors the construction in
    tools/gems_train/teacher_factory.py::_camera_record_from_view, but takes
    a dict so callers never pass GT-bearing Camera objects)."""
    width = int(pose["width"])
    height = int(pose["height"])
    return CameraRecord(
        idx=int(idx),
        image_name=str(pose["image_name"]),
        width=width,
        height=height,
        fx=float(fov2focal(float(pose["fovx"]), width)),
        fy=float(fov2focal(float(pose["fovy"]), height)),
        camera_center=tuple(float(x) for x in pose["camera_center"]),
        world_view_transform=tuple(
            tuple(float(v) for v in row) for row in pose["world_view_transform"]
        ),
    )


class ConfinedFrameLoader:
    """Duck-typed FrameLoader (render/gt/depth/residual) with two rules:

    1. every disk read must realpath-resolve under the cache root (else
       PermissionError) and is appended to ``read_log``;
    2. in-memory target tensors are served for registered sentinel keys, so
       the test view's base render/depth never round-trip through disk.
    """

    def __init__(self, cache_root: Path, device: torch.device | str = "cuda",
                 max_cached: int = 96) -> None:
        self.device = torch.device(device)
        self._root = os.path.realpath(str(cache_root))
        self._mem: dict[str, torch.Tensor] = {}
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._max_cached = int(max_cached)
        self.read_log: set[str] = set()

    def register_target(self, name: str, render: torch.Tensor,
                        depth: torch.Tensor) -> FrameRecord:
        """Register in-memory base products; returns the target FrameRecord
        (camera is attached by the caller via dataclasses.replace-like use —
        see EcrRenderer.adapt, which builds the record itself)."""
        self._mem[TARGET_RENDER_PREFIX + name] = render.to(self.device)
        self._mem[TARGET_DEPTH_PREFIX + name] = depth.to(self.device)

    def clear_target(self, name: str) -> None:
        self._mem.pop(TARGET_RENDER_PREFIX + name, None)
        self._mem.pop(TARGET_DEPTH_PREFIX + name, None)

    def _confined(self, path: str) -> str:
        real = os.path.realpath(path)
        if not (real == self._root or real.startswith(self._root + os.sep)):
            raise PermissionError(
                f"ECR transport attempted to read outside the evidence cache: "
                f"{path} (cache root: {self._root})")
        self.read_log.add(real)
        return real

    def _cached_disk(self, key: str, load) -> torch.Tensor:
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        value = load()
        self._cache[key] = value
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return value

    def _image(self, path: str) -> torch.Tensor:
        text = str(path)
        if text in self._mem:
            return self._mem[text]
        real = self._confined(text)
        return self._cached_disk(
            real, lambda: read_image_tensor(Path(real), device=self.device))

    def render(self, path: str) -> torch.Tensor:
        return self._image(path)

    def gt(self, path: str) -> torch.Tensor:
        if TARGET_GT_SENTINEL in str(path):
            raise RuntimeError(
                "ECR transport attempted to read the TARGET view's GT — "
                "forbidden by D4 (sentinel path hit)")
        return self._image(path)

    def depth(self, path: str) -> torch.Tensor:
        text = str(path)
        if text in self._mem:
            return self._mem[text]
        real = self._confined(text)
        return self._cached_disk(
            real, lambda: read_depth_tensor(Path(real), device=self.device))

    def residual(self, frame: FrameRecord, residual_clip: float) -> torch.Tensor:
        residual = self.gt(str(frame.gt_path)) - self.render(str(frame.render_path))
        if residual_clip > 0:
            residual = torch.clamp(residual, -float(residual_clip), float(residual_clip))
        return residual


class EcrRenderer:
    """Frozen-config evidence transport over one evidence cache."""

    def __init__(self, cache_dir: str, device: torch.device | str = "cuda") -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.device = torch.device(device)
        manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = hashlib.sha256(
            manifest_path.read_bytes()).hexdigest()

        cameras = load_camera_index(self.cache_dir / "camera_index.json")
        by_name = {cam.image_name: cam for cam in cameras}
        self.train_frames: list[FrameRecord] = []
        for idx, name in enumerate(self.manifest["train_views"]):
            cam = by_name[name]
            self.train_frames.append(FrameRecord(
                idx=idx,
                name=name,
                render_path=self.cache_dir / "renders" / f"{name}.png",
                gt_path=self.cache_dir / "gt" / f"{name}.png",
                depth_path=self.cache_dir / "depths" / f"{name}.npy",
                camera=cam,
            ))
        if not self.train_frames:
            raise RuntimeError(f"evidence cache lists no train views: {cache_dir}")

        transport = dict(self.manifest["transport"])
        self.fuse = str(transport.get("fuse", "single"))
        self._fusion_net = None
        if self.fuse == "multiband":
            from tools.ecr.transport_l2 import adapt_frame_l2
            self._adapt_fn = adapt_frame_l2
            keys = _L2_KWARG_KEYS
        elif self.fuse == "single":
            self._adapt_fn = adapt_frame
            keys = _ADAPT_KWARG_KEYS
        elif self.fuse == "learned":
            # L3: frozen per-scene fusion net (trained train-only by
            # tools/ecr/train_fusion.py; sha pinned in the manifest).
            from tools.ecr.fusion import FusionNet
            net_path = self.cache_dir / str(transport["fusion_net"])
            net_sha = hashlib.sha256(net_path.read_bytes()).hexdigest()
            if net_sha != transport.get("fusion_net_sha256"):
                raise RuntimeError(
                    f"fusion net sha mismatch: {net_sha} != manifest "
                    f"{transport.get('fusion_net_sha256')}")
            net = FusionNet().to(self.device)
            net.load_state_dict(torch.load(net_path, map_location=self.device,
                                           weights_only=True))
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._fusion_net = net
            self._fusion_net_sha = net_sha
            self._adapt_fn = None
            keys = _L3_KWARG_KEYS
        else:
            raise ValueError(f"unknown transport fuse mode: {self.fuse}")
        self._adapt_kwargs = {
            key: transport[key] for key in keys if key in transport
        }
        if self.fuse != "learned":
            # alpha in the manifest is the train-only calibrated scalar (for
            # the L2 transport, k is likewise the calibrated value, frozen
            # into transport["k"] at cache build). The learned fuse has no
            # global alpha — the frozen net IS the alpha map.
            self._adapt_kwargs["alpha"] = float(self.manifest["alpha"]["alpha"])
        hash_payload = {"fuse": self.fuse, **self._adapt_kwargs}
        if self._fusion_net is not None:
            hash_payload["fusion_net_sha256"] = self._fusion_net_sha
        self.config_hash = self._hash_kwargs(hash_payload)
        self.loader = ConfinedFrameLoader(self.cache_dir, device=self.device)
        if self._fusion_net is not None:
            self.loader.read_log.add(os.path.realpath(str(
                self.cache_dir / str(transport["fusion_net"]))))

    @staticmethod
    def _hash_kwargs(kwargs: dict) -> str:
        return hashlib.sha256(
            json.dumps(kwargs, sort_keys=True).encode("utf-8")).hexdigest()

    def checkpoint_fingerprint(self) -> dict:
        return dict(self.manifest.get("checkpoint", {}))

    def cache_cost(self) -> dict:
        sizes = self.manifest.get("sizes", {})
        return {
            "cache_mb_raw": float(sizes.get("cache_mb_raw", -1.0)),
            "cache_mb_compressed": float(sizes.get("cache_mb_compressed", -1.0)),
            "n_cache_files": int(sizes.get("n_files", -1)),
        }

    def adapt(self, view_name: str, pose: dict, base_render: torch.Tensor,
              base_depth: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Transport-correct one test view.

        base_render: [3,H,W] float in [0,1] (already 8-bit-quantized by the
        caller for parity with the archived PNG path); base_depth: [H,W]
        median surf_depth. Returns (adapted [3,H,W], info dict incl. the
        per-call kwargs hash and wall time).
        """
        name = str(view_name)
        camera = camera_record_from_pose(idx=-1, pose=pose)
        target = FrameRecord(
            idx=-1,
            name=name,
            render_path=Path(TARGET_RENDER_PREFIX + name),
            gt_path=Path(TARGET_GT_SENTINEL),
            depth_path=Path(TARGET_DEPTH_PREFIX + name),
            camera=camera,
        )
        self.loader.register_target(name, base_render, base_depth)
        kwargs = dict(self._adapt_kwargs)
        hash_payload = {"fuse": self.fuse, **kwargs}
        if self._fusion_net is not None:
            hash_payload["fusion_net_sha256"] = self._fusion_net_sha
        call_hash = self._hash_kwargs(hash_payload)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if self._fusion_net is not None:
                from tools.ecr.fusion import (apply_fusion,
                                              compute_transport_features)
                with torch.no_grad():
                    feat_kwargs = dict(kwargs)
                    feat_kwargs["fuse"] = feat_kwargs.pop("inner_fuse", "single")
                    feats = compute_transport_features(
                        target, self.train_frames, loader=self.loader,
                        device=self.device, **feat_kwargs)
                    adapted, alpha_map = apply_fusion(self._fusion_net, feats)
                valid = feats["weight_den"] > float(
                    kwargs.get("min_confidence", 1e-4))
                info = {
                    "support_count": len(feats["support_names"]),
                    "support_names": feats["support_names"],
                    "mean_confidence": float(feats["weight_den"].mean()
                                             .detach().cpu().item()),
                    "covered_fraction": float(valid.to(torch.float32).mean()
                                              .detach().cpu().item()),
                    "alpha_mean": float(alpha_map.mean().detach().cpu().item()),
                }
            else:
                adapted, info = self._adapt_fn(
                    target,
                    self.train_frames,
                    loader=self.loader,
                    device=self.device,
                    **kwargs,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        finally:
            self.loader.clear_target(name)
        info = dict(info)
        info["kwargs_hash"] = call_hash
        info["transport_seconds"] = elapsed
        return adapted, info

    def read_log(self) -> list[str]:
        return sorted(self.loader.read_log)

#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
from scene import Scene
import os
import json
import math
import time
import hashlib
import numpy as np
from tqdm import tqdm
from os import makedirs
from pathlib import Path
from triangle_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel
from utils.graphics_utils import fov2focal
from utils.evidence_lumigraph_adapter import (
    CameraRecord,
    FrameLoader,
    FrameRecord,
    adapt_frame,
    load_split_frames,
    save_camera_index,
    save_image_tensor,
)
from scripts.car_model.benchmark_ela_postprocess_runtime import (
    _alpha_calibrator_from_report,
    _alpha_from_report,
    _benefit_calibrator_from_report,
    _json_safe,
    _policy_from_report,
    _read_report,
    _select_support_frames,
)


def _finite_float(value, default=0.0):
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _mean(values):
    finite = [_finite_float(value, math.nan) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else 0.0


def _camera_record(idx, view):
    width = int(view.image_width)
    height = int(view.image_height)
    return CameraRecord(
        idx=int(idx),
        image_name=str(getattr(view, "image_name", f"{idx:05d}")),
        width=width,
        height=height,
        fx=float(fov2focal(float(view.FoVx), width)),
        fy=float(fov2focal(float(view.FoVy), height)),
        camera_center=tuple(float(x) for x in view.camera_center.detach().cpu().tolist()),
        world_view_transform=tuple(
            tuple(float(v) for v in row)
            for row in view.world_view_transform.detach().cpu().tolist()
        ),
    )


def _camera_from_bank(row):
    return CameraRecord(
        idx=int(row["idx"]),
        image_name=str(row["image_name"]),
        width=int(row["width"]),
        height=int(row["height"]),
        fx=float(row["fx"]),
        fy=float(row["fy"]),
        camera_center=tuple(float(x) for x in row["camera_center"]),
        world_view_transform=tuple(tuple(float(v) for v in line) for line in row["world_view_transform"]),
    )


def _camera_payload(camera):
    return {
        "idx": int(camera.idx),
        "image_name": str(camera.image_name),
        "width": int(camera.width),
        "height": int(camera.height),
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "camera_center": [float(x) for x in camera.camera_center],
        "world_view_transform": [[float(v) for v in row] for row in camera.world_view_transform],
    }


def _assert_camera_matches_bank(camera, row, frame_key):
    if not isinstance(row, dict):
        raise RuntimeError(f"v102 preprojected delta bank missing camera metadata for {frame_key}")
    checks = {
        "width": int(camera.width),
        "height": int(camera.height),
        "image_name": str(camera.image_name),
    }
    for key, value in checks.items():
        if key in row and row[key] != value:
            raise RuntimeError(f"v102 preprojected camera {key} mismatch for {frame_key}: bank={row[key]} render={value}")
    if "camera_center" in row:
        lhs = torch.tensor(row["camera_center"], dtype=torch.float64)
        rhs = torch.tensor(camera.camera_center, dtype=torch.float64)
        if lhs.numel() != rhs.numel() or float((lhs - rhs).abs().max().item()) > 1e-5:
            raise RuntimeError(f"v102 preprojected camera center mismatch for {frame_key}")
    if "world_view_transform" in row:
        lhs = torch.tensor(row["world_view_transform"], dtype=torch.float64)
        rhs = torch.tensor(camera.world_view_transform, dtype=torch.float64)
        if lhs.shape != rhs.shape or float((lhs - rhs).abs().max().item()) > 1e-5:
            raise RuntimeError(f"v102 preprojected world_view_transform mismatch for {frame_key}")


def _frame_from_bank(row):
    name = str(row["name"])
    return FrameRecord(
        idx=int(row["idx"]),
        name=name,
        render_path=Path("__v101_evidence_bank__") / "renders" / f"{name}.png",
        gt_path=Path("__v101_evidence_bank__") / "gt" / f"{name}.png",
        depth_path=Path("__v101_evidence_bank__") / "depths" / f"{name}.npy",
        camera=_camera_from_bank(row["camera"]),
    )


class _EndpointBankFrameLoader(FrameLoader):
    def __init__(self, bank, device):
        super().__init__(device=device)
        self.bank = bank
        self.residuals = bank.get("residuals", {})
        self.depths = bank.get("depths", {})

    @staticmethod
    def _key(path):
        return Path(str(path)).stem

    def depth(self, path):
        path_text = str(path)
        key = self._key(path_text)
        if "__v101_evidence_bank__" in path_text and key in self.depths:
            return self.depths[key].to(device=self.device, dtype=torch.float32)
        return super().depth(path)

    def residual(self, frame, residual_clip):
        frame_paths = f"{frame.render_path} {frame.gt_path} {frame.depth_path}"
        if "__v101_evidence_bank__" in frame_paths and frame.name in self.residuals:
            residual = self.residuals[frame.name].to(device=self.device, dtype=torch.float32)
            if residual_clip > 0:
                residual = torch.clamp(residual, -float(residual_clip), float(residual_clip))
            return residual
        return super().residual(frame, residual_clip)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_dtype(name):
    name = str(name or "").strip().lower()
    if name in {"float32", "fp32", ""}:
        return torch.float32
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"unsupported tensor dtype: {name}")


def _load_preprojected_delta_bank(path, report_path, endpoint_method):
    payload = _torch_load(path)
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0) or 0) < 1:
        raise RuntimeError(f"invalid v102 preprojected delta bank: {path}")
    if str(payload.get("bank_type", "")) != "v102_preprojected_delta_bank":
        raise RuntimeError(f"unexpected preprojected bank type: {payload.get('bank_type')}")
    if str(payload.get("endpoint_method", "")) and str(payload.get("endpoint_method", "")) != str(endpoint_method):
        raise RuntimeError(
            f"preprojected bank endpoint mismatch: bank={payload.get('endpoint_method')} endpoint={endpoint_method}"
        )
    report_sha = str(payload.get("endpoint_report_sha256", "") or "")
    if report_sha and report_sha != _sha256(report_path):
        raise RuntimeError("v102 preprojected delta bank endpoint_report_sha256 mismatch")
    deltas = payload.get("deltas", {})
    if not isinstance(deltas, dict) or not deltas:
        raise RuntimeError("v102 preprojected delta bank has no deltas")
    frames = payload.get("frames", {})
    if isinstance(frames, list):
        frames = {str(row.get("frame", row.get("name", idx))): row for idx, row in enumerate(frames)}
    if not isinstance(frames, dict):
        frames = {}
    missing = [str(key) for key in deltas.keys() if str(key) not in frames]
    if missing:
        raise RuntimeError(f"v102 preprojected delta bank missing frame metadata for {len(missing)} deltas")
    return payload


def _load_endpoint_runtime(
    model_path,
    iteration,
    endpoint_method,
    output_method,
    base_model_override,
    base_method_override,
    evidence_max_side_override,
    bank_path_override="",
    require_bank=False,
    preprojected_bank_path_override="",
    require_preprojected_bank=False,
    write_preprojected_bank_path="",
    preprojected_delta_dtype="float32",
    no_intermediate_outputs=False,
):
    endpoint_method = str(endpoint_method or "").strip()
    if not endpoint_method:
        return None
    model_root = Path(model_path)
    endpoint_dir = (
        model_root
        / "point_cloud"
        / f"iteration_{int(iteration)}"
        / "render_residual_endpoint"
        / endpoint_method
    )
    report_path = endpoint_dir / "ela_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing checkpoint endpoint report: {report_path}")
    report = _read_report(str(report_path))
    policy, policy_source = _policy_from_report(report)
    alpha, alpha_source = _alpha_from_report(report)
    base_model = Path(
        str(base_model_override or report.get("base_model_path") or report.get("base_model") or model_root)
    ).resolve()
    base_method = str(base_method_override or report.get("base_method") or report.get("base_method_name") or "")
    if not base_method:
        raise RuntimeError(f"endpoint report has no base method: {report_path}")
    explicit_preprojected_bank_path = bool(str(preprojected_bank_path_override or "").strip())
    preprojected_bank_path = (
        Path(str(preprojected_bank_path_override)).expanduser() if explicit_preprojected_bank_path else None
    )
    preprojected_bank = None
    preprojected_bank_manifest = {}
    if preprojected_bank_path is not None and preprojected_bank_path.is_file():
        preprojected_bank = _load_preprojected_delta_bank(preprojected_bank_path, report_path, endpoint_method)
        preprojected_bank_manifest = {
            "bank_path": str(preprojected_bank_path),
            "bank_sha256": _sha256(preprojected_bank_path),
            "target_frames": int(len(preprojected_bank.get("deltas", {}))),
            "delta_dtype": str(preprojected_bank.get("delta_dtype", "")),
            "endpoint_report_sha256": str(preprojected_bank.get("endpoint_report_sha256", "") or ""),
            "source_support_source": str(preprojected_bank.get("source_support_source", "") or ""),
            "source_evidence_bank_sha256": str(preprojected_bank.get("source_evidence_bank_sha256", "") or ""),
        }
    elif explicit_preprojected_bank_path or bool(require_preprojected_bank):
        raise FileNotFoundError(f"required v102 preprojected delta bank not found: {preprojected_bank_path}")

    explicit_bank_path = bool(str(bank_path_override or "").strip())
    bank_path = Path(str(bank_path_override)).expanduser() if explicit_bank_path else endpoint_dir / "v101_evidence_bank.pt"
    bank = None
    bank_manifest = {}
    if preprojected_bank is not None:
        source_bank_sha = str(preprojected_bank.get("source_evidence_bank_sha256", "") or "")
        if explicit_bank_path:
            if not bank_path.is_file():
                raise FileNotFoundError(f"v102 source evidence bank check path not found: {bank_path}")
            if source_bank_sha and _sha256(bank_path) != source_bank_sha:
                raise RuntimeError("v102 preprojected delta bank source_evidence_bank_sha256 mismatch")
        support_frames = []
        support_source = f"v102_preprojected_delta_bank:{preprojected_bank_path}"
        missing_support_names = []
    elif bank_path.is_file():
        bank = _torch_load(bank_path)
        if not isinstance(bank, dict) or int(bank.get("schema_version", 0) or 0) < 1:
            raise RuntimeError(f"invalid v101 evidence bank: {bank_path}")
        if str(policy.get("mode", "")) != "residual":
            raise RuntimeError("v101 evidence bank currently supports residual endpoint mode only")
        if str(bank.get("base_method", "")) and str(bank.get("base_method", "")) != str(base_method):
            raise RuntimeError(
                f"v101 evidence bank base_method mismatch: bank={bank.get('base_method')} endpoint={base_method}"
            )
        bank_report_sha = str(bank.get("source_report_sha256", "") or "")
        if bank_report_sha and bank_report_sha != _sha256(report_path):
            raise RuntimeError("v101 evidence bank source_report_sha256 mismatch")
        support_frames = [_frame_from_bank(row) for row in bank.get("frames", [])]
        missing_bank_payload = [
            frame.name
            for frame in support_frames
            if frame.name not in bank.get("residuals", {}) or frame.name not in bank.get("depths", {})
        ]
        if missing_bank_payload:
            raise RuntimeError(f"v101 evidence bank missing tensors for {len(missing_bank_payload)} support frames")
        if len(bank.get("residuals", {})) != len(support_frames) or len(bank.get("depths", {})) != len(support_frames):
            raise RuntimeError("v101 evidence bank tensor/frame count mismatch")
        support_source = f"v101_evidence_bank:{bank_path}"
        missing_support_names = []
        bank_manifest = {
            "bank_path": str(bank_path),
            "support_frames": int(len(support_frames)),
            "tensor_dtype": str(bank.get("tensor_dtype", "")),
            "residual_dtype": str(bank.get("residual_dtype", "")),
            "depth_dtype": str(bank.get("depth_dtype", "")),
            "source_base_method": str(bank.get("base_method", "")),
            "source_report_sha256": bank_report_sha,
        }
    else:
        if explicit_bank_path or bool(require_bank):
            raise FileNotFoundError(f"required v101 evidence bank not found: {bank_path}")
        train_frames = load_split_frames(base_model, "train", base_method)
        support_frames, support_source, missing_support_names = _select_support_frames(train_frames, report)
    if preprojected_bank is None and not support_frames:
        raise RuntimeError(f"endpoint selected no support frames: {report_path}")
    evidence_max_side = int(evidence_max_side_override)
    if evidence_max_side < 0:
        evidence_max_side = int(report.get("evidence_max_side", 0) or 0)
    resolved_output_method = str(output_method or "").strip() or f"{endpoint_method}_renderpy_v101"
    return {
        "endpoint_method": endpoint_method,
        "output_method": resolved_output_method,
        "base_render_method": f"{resolved_output_method}_base",
        "endpoint_dir": endpoint_dir,
        "report_path": report_path,
        "policy": policy,
        "policy_source": policy_source,
        "alpha": float(alpha),
        "alpha_source": alpha_source,
        "benefit_calibrator": _benefit_calibrator_from_report(report),
        "alpha_calibrator": _alpha_calibrator_from_report(report),
        "base_model": base_model,
        "base_method": base_method,
        "support_frames": support_frames,
        "support_source": support_source,
        "missing_support_names": missing_support_names,
        "evidence_max_side": evidence_max_side,
        "bank": bank,
        "bank_manifest": bank_manifest,
        "preprojected_bank": preprojected_bank,
        "preprojected_bank_manifest": preprojected_bank_manifest,
        "write_preprojected_bank_path": str(write_preprojected_bank_path or "").strip(),
        "preprojected_delta_dtype": str(preprojected_delta_dtype or "float32"),
        "no_intermediate_outputs": bool(no_intermediate_outputs),
    }


def _render_endpoint_set(model_path, name, iteration, views, triangles, pipeline, background, endpoint):
    model_root = Path(model_path)
    output_method = endpoint["output_method"]
    base_method = endpoint["base_render_method"]
    endpoint_method_dir = model_root / name / output_method
    base_method_dir = model_root / name / base_method
    render_path = endpoint_method_dir / "renders"
    gts_path = endpoint_method_dir / "gt"
    depth_path = endpoint_method_dir / "depths"
    base_render_path = base_method_dir / "renders"
    base_gts_path = base_method_dir / "gt"
    base_depth_path = base_method_dir / "depths"
    for path in (render_path, gts_path, depth_path, base_render_path, base_gts_path, base_depth_path):
        path.mkdir(parents=True, exist_ok=True)

    policy = endpoint["policy"]
    preprojected_bank = endpoint.get("preprojected_bank")
    use_preprojected_bank = preprojected_bank is not None
    preprojected_frames = preprojected_bank.get("frames", {}) if use_preprojected_bank else {}
    if isinstance(preprojected_frames, list):
        preprojected_frames = {str(row.get("frame", row.get("name", idx))): row for idx, row in enumerate(preprojected_frames)}
    preprojected_deltas = preprojected_bank.get("deltas", {}) if use_preprojected_bank else {}
    write_preprojected_bank_path = str(endpoint.get("write_preprojected_bank_path", "") or "").strip()
    write_preprojected_dtype = _tensor_dtype(endpoint.get("preprojected_delta_dtype", "float32"))
    collected_preprojected_deltas = {} if write_preprojected_bank_path and not use_preprojected_bank else None
    collected_preprojected_frames = {} if write_preprojected_bank_path and not use_preprojected_bank else None
    save_intermediate_outputs = not (use_preprojected_bank and bool(endpoint.get("no_intermediate_outputs", False)))
    loader = (
        _EndpointBankFrameLoader(endpoint["bank"], device=background.device)
        if endpoint.get("bank") is not None
        else FrameLoader(device=background.device)
    )
    support_frames = endpoint["support_frames"]
    camera_records = []
    frame_infos = []
    start = time.time()
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} endpoint {output_method}")):
        pkg = render(view, triangles, pipeline, background)
        rendering = pkg["render"]
        depth = pkg.get("surf_depth", None)
        if depth is None:
            raise RuntimeError("render package did not include surf_depth; endpoint mode requires depth evidence")
        gt = view.original_image[0:3, :, :]
        key = f"{idx:05d}"
        image_name = f"{key}.png"
        camera = _camera_record(idx, view)
        base_render_file = base_render_path / image_name
        base_gt_file = base_gts_path / image_name
        base_depth_file = base_depth_path / f"{key}.npy"
        endpoint_depth_file = depth_path / f"{key}.npy"
        torchvision.utils.save_image(gt, gts_path / image_name)
        if save_intermediate_outputs:
            torchvision.utils.save_image(rendering, base_render_file)
            torchvision.utils.save_image(gt, base_gt_file)
            depth_np = depth[0].detach().float().cpu().numpy().astype(np.float32)
            np.save(base_depth_file, depth_np)
            np.save(endpoint_depth_file, depth_np)
        target = FrameRecord(
            idx=int(idx),
            name=key,
            render_path=base_render_file,
            gt_path=base_gt_file,
            depth_path=base_depth_file,
            camera=camera,
        )
        if use_preprojected_bank:
            if key not in preprojected_deltas:
                raise RuntimeError(f"v102 preprojected delta bank missing target frame {key}")
            _assert_camera_matches_bank(camera, preprojected_frames.get(key, {}).get("target_camera", {}), key)
            if save_intermediate_outputs and str(preprojected_frames.get(key, {}).get("base_render_sha256", "") or ""):
                current_base_sha = _sha256(base_render_file)
                if current_base_sha != str(preprojected_frames[key]["base_render_sha256"]):
                    raise RuntimeError(f"v102 preprojected base render SHA mismatch for {key}")
            if save_intermediate_outputs and str(preprojected_frames.get(key, {}).get("base_depth_sha256", "") or ""):
                current_depth_sha = _sha256(base_depth_file)
                if current_depth_sha != str(preprojected_frames[key]["base_depth_sha256"]):
                    raise RuntimeError(f"v102 preprojected base depth SHA mismatch for {key}")
            delta = preprojected_deltas[key].to(device=rendering.device, dtype=torch.float32)
            if tuple(delta.shape) != tuple(rendering.shape):
                raise RuntimeError(
                    f"v102 preprojected delta shape mismatch for {key}: delta={tuple(delta.shape)} render={tuple(rendering.shape)}"
                )
            adapted = torch.clamp(rendering + delta, 0.0, 1.0)
            info = dict(preprojected_frames.get(key, {}))
            info.update(
                {
                    "frame": key,
                    "preprojected_delta": True,
                    "source_support_frames": int(preprojected_bank.get("source_support_frames", 0) or 0),
                }
            )
            info.setdefault("support_names", [])
            info.setdefault("evidence_max_side", int(preprojected_bank.get("source_evidence_max_side", 0) or 0))
            info.setdefault("evidence_scaled", False)
        else:
            adapted, info = adapt_frame(
                target,
                support_frames,
                k=int(policy["k"]),
                alpha=float(endpoint["alpha"]),
                mode=str(policy["mode"]),
                residual_clip=float(policy["residual_clip"]),
                min_confidence=float(policy["min_confidence"]),
                depth_abs_tol=float(policy["depth_abs_tol"]),
                depth_rel_tol=float(policy["depth_rel_tol"]),
                direction_weight=float(policy["direction_weight"]),
                benefit_calibrator=endpoint["benefit_calibrator"],
                alpha_calibrator=endpoint["alpha_calibrator"],
                edge_gate=bool(policy["edge_gate"]),
                edge_gate_quantile=float(policy["edge_gate_quantile"]),
                edge_gate_min=float(policy["edge_gate_min"]),
                edge_gate_dilate=int(policy["edge_gate_dilate"]),
                local_trust_gate=bool(policy["local_trust_gate"]),
                local_trust_min_supports=int(policy["local_trust_min_supports"]),
                local_trust_max_residual_std=float(policy["local_trust_max_residual_std"]),
                local_trust_min_agreement=float(policy["local_trust_min_agreement"]),
                local_trust_agreement_scale=float(policy["local_trust_agreement_scale"]),
                local_trust_confidence_quantile=float(policy["local_trust_confidence_quantile"]),
                local_trust_min_confidence=float(policy["local_trust_min_confidence"]),
                local_trust_mode=str(policy["local_trust_mode"]),
                local_trust_min_weight=float(policy["local_trust_min_weight"]),
                evidence_max_side=int(endpoint["evidence_max_side"]),
                loader=loader,
                device=background.device,
            )
            if collected_preprojected_deltas is not None:
                collected_preprojected_deltas[key] = (adapted - rendering).detach().cpu().to(
                    dtype=write_preprojected_dtype
                ).contiguous()
        save_image_tensor(adapted, render_path / image_name)
        diff = (adapted - rendering).detach().abs()
        changed = diff.amax(dim=0) > 1e-5
        frame_info = {
            "frame": key,
            **info,
            "target_camera": _camera_payload(camera),
            "base_render_sha256": _sha256(base_render_file) if save_intermediate_outputs else "",
            "base_depth_sha256": _sha256(base_depth_file) if save_intermediate_outputs else "",
            "changed_fraction": float(changed.float().mean().detach().cpu().item()),
            "mean_abs_delta": float(diff.mean().detach().cpu().item()),
            "max_abs_delta": float(diff.max().detach().cpu().item()),
        }
        frame_infos.append(frame_info)
        if collected_preprojected_frames is not None:
            collected_preprojected_frames[key] = _json_safe(frame_info)
        camera_records.append(camera)
        del pkg, rendering, gt, depth, adapted, diff
        if use_preprojected_bank:
            del delta
        if not use_preprojected_bank:
            torch.cuda.empty_cache()

    preprojected_written_manifest = {}
    if collected_preprojected_deltas is not None:
        preprojected_path = Path(write_preprojected_bank_path)
        preprojected_path.parent.mkdir(parents=True, exist_ok=True)
        source_bank_path = str(endpoint.get("bank_manifest", {}).get("bank_path", "") or "")
        source_bank_sha256 = _sha256(source_bank_path) if source_bank_path and Path(source_bank_path).is_file() else ""
        payload = {
            "schema_version": 1,
            "bank_type": "v102_preprojected_delta_bank",
            "created_at_unix": time.time(),
            "split": str(name),
            "iteration": int(iteration),
            "endpoint_method": endpoint["endpoint_method"],
            "source_output_method": output_method,
            "endpoint_report": str(endpoint["report_path"]),
            "endpoint_report_sha256": _sha256(endpoint["report_path"]),
            "source_support_source": endpoint["support_source"],
            "source_support_frames": int(len(support_frames)),
            "source_evidence_max_side": int(endpoint["evidence_max_side"]),
            "source_evidence_bank": endpoint.get("bank_manifest", {}),
            "source_evidence_bank_sha256": source_bank_sha256,
            "delta_dtype": str(endpoint.get("preprojected_delta_dtype", "float32")),
            "target_frames": int(len(collected_preprojected_deltas)),
            "frames": collected_preprojected_frames,
            "deltas": collected_preprojected_deltas,
            "note": (
                "Preprojected target-camera residual deltas generated from the v101 train-evidence endpoint. "
                "This bank stores no held-out target GT."
            ),
        }
        torch.save(payload, preprojected_path)
        preprojected_written_manifest = {
            "bank_path": str(preprojected_path),
            "bank_sha256": _sha256(preprojected_path),
            "target_frames": int(len(collected_preprojected_deltas)),
            "delta_dtype": str(endpoint.get("preprojected_delta_dtype", "float32")),
            "source_support_source": endpoint["support_source"],
            "source_support_frames": int(len(support_frames)),
            "source_evidence_bank_sha256": source_bank_sha256,
            "note": "Generated during online v101 endpoint rendering; contains no held-out target GT.",
        }
        preprojected_path.with_suffix(".manifest.json").write_text(
            json.dumps(preprojected_written_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    save_camera_index(camera_records, base_method_dir / "camera_index.json")
    save_camera_index(camera_records, endpoint_method_dir / "camera_index.json")
    report = {
        "render_py_endpoint_version": 1,
        "mode": "preprojected_delta_endpoint" if use_preprojected_bank else "online_checkpoint_attached_endpoint",
        "split": str(name),
        "iteration": int(iteration),
        "endpoint_method": endpoint["endpoint_method"],
        "output_method": output_method,
        "base_render_method": base_method,
        "endpoint_report": str(endpoint["report_path"]),
        "base_model": str(endpoint["base_model"]),
        "base_method": str(endpoint["base_method"]),
        "policy_source": endpoint["policy_source"],
        "alpha_source": endpoint["alpha_source"],
        "alpha": float(endpoint["alpha"]),
        "policy": policy,
        "evidence_max_side": int(endpoint["evidence_max_side"]),
        "support_source": endpoint["support_source"],
        "support_frames": int(
            preprojected_bank.get("source_support_frames", 0) if use_preprojected_bank else len(support_frames)
        ),
        "missing_report_support_names": endpoint["missing_support_names"],
        "evidence_bank": endpoint.get("bank_manifest", {}),
        "preprojected_delta_bank": endpoint.get("preprojected_bank_manifest", {}),
        "preprojected_delta_bank_written": preprojected_written_manifest,
        "intermediate_outputs_saved": bool(save_intermediate_outputs),
        "target_frames": int(len(camera_records)),
        "elapsed_sec": float(time.time() - start),
        "mean_changed_fraction": _mean([row.get("changed_fraction") for row in frame_infos]),
        "mean_abs_delta": _mean([row.get("mean_abs_delta") for row in frame_infos]),
        "mean_covered_fraction": _mean([row.get("covered_fraction") for row in frame_infos]),
        "mean_alpha_active_fraction": _mean([row.get("alpha_active_fraction") for row in frame_infos]),
        "no_test_gt_used_for_policy": True,
        "claim_boundary": (
            "This render.py endpoint consumes a preprojected target-camera residual bank generated from the "
            "train-evidence endpoint. It does not use held-out target GT, but it is still an endpoint sidecar, "
            "not a vanilla checkpoint-baked representation."
            if use_preprojected_bank
            else "This render.py endpoint consumes a checkpoint-attached train-derived sidecar at render time. "
            "It recomputes target renders online and does not rely on pre-materialized target endpoint images."
        ),
        "frames": frame_infos,
    }
    report = _json_safe(report)
    (endpoint_method_dir / "render_py_endpoint_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (endpoint["endpoint_dir"] / f"render_py_{name}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_set(model_path, name, iteration, views, triangles, pipeline, background, endpoint=None):
    if endpoint is not None:
        _render_endpoint_set(model_path, name, iteration, views, triangles, pipeline, background, endpoint)
        return
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, triangles, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(
    dataset : ModelParams,
    iteration : int,
    pipeline : PipelineParams,
    skip_train : bool,
    skip_test : bool,
    checkpoint_endpoint_method: str = "",
    checkpoint_endpoint_output_method: str = "",
    checkpoint_endpoint_base_model: str = "",
    checkpoint_endpoint_base_method: str = "",
    checkpoint_endpoint_evidence_max_side: int = -1,
    checkpoint_endpoint_bank_path: str = "",
    checkpoint_endpoint_require_bank: bool = False,
    checkpoint_endpoint_preprojected_bank_path: str = "",
    checkpoint_endpoint_require_preprojected_bank: bool = False,
    checkpoint_endpoint_write_preprojected_bank: str = "",
    checkpoint_endpoint_preprojected_delta_dtype: str = "float32",
    checkpoint_endpoint_no_intermediate_outputs: bool = False,
):
    with torch.no_grad():
        triangles = TriangleModel(dataset.sh_degree)
        triangles.scaling = 4
        scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  set_sigma=None,
                  load_iteration=iteration,
                  shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        endpoint = _load_endpoint_runtime(
            dataset.model_path,
            scene.loaded_iter,
            checkpoint_endpoint_method,
            checkpoint_endpoint_output_method,
            checkpoint_endpoint_base_model,
            checkpoint_endpoint_base_method,
            checkpoint_endpoint_evidence_max_side,
            checkpoint_endpoint_bank_path,
            checkpoint_endpoint_require_bank,
            checkpoint_endpoint_preprojected_bank_path,
            checkpoint_endpoint_require_preprojected_bank,
            checkpoint_endpoint_write_preprojected_bank,
            checkpoint_endpoint_preprojected_delta_dtype,
            checkpoint_endpoint_no_intermediate_outputs,
        )

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), triangles, pipeline, background, endpoint=endpoint)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), triangles, pipeline, background, endpoint=endpoint)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--checkpoint_endpoint_method", default="")
    parser.add_argument("--checkpoint_endpoint_output_method", default="")
    parser.add_argument("--checkpoint_endpoint_base_model", default="")
    parser.add_argument("--checkpoint_endpoint_base_method", default="")
    parser.add_argument("--checkpoint_endpoint_evidence_max_side", default=-1, type=int)
    parser.add_argument("--checkpoint_endpoint_bank_path", default="")
    parser.add_argument("--checkpoint_endpoint_require_bank", action="store_true")
    parser.add_argument("--checkpoint_endpoint_preprojected_bank_path", default="")
    parser.add_argument("--checkpoint_endpoint_require_preprojected_bank", action="store_true")
    parser.add_argument("--checkpoint_endpoint_write_preprojected_bank", default="")
    parser.add_argument(
        "--checkpoint_endpoint_preprojected_delta_dtype",
        default="float32",
        choices=("float32", "float16"),
    )
    parser.add_argument("--checkpoint_endpoint_no_intermediate_outputs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.checkpoint_endpoint_method,
        args.checkpoint_endpoint_output_method,
        args.checkpoint_endpoint_base_model,
        args.checkpoint_endpoint_base_method,
        args.checkpoint_endpoint_evidence_max_side,
        args.checkpoint_endpoint_bank_path,
        args.checkpoint_endpoint_require_bank,
        args.checkpoint_endpoint_preprojected_bank_path,
        args.checkpoint_endpoint_require_preprojected_bank,
        args.checkpoint_endpoint_write_preprojected_bank,
        args.checkpoint_endpoint_preprojected_delta_dtype,
        args.checkpoint_endpoint_no_intermediate_outputs,
    )

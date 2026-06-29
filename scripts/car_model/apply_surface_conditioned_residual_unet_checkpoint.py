#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.car_model.ecsr_apply_surface_residual_region_texture_adapter import (  # noqa: E402
    evidence_views,
    save_image_chw,
)
from scripts.car_model.train_surface_conditioned_residual_unet import (  # noqa: E402
    SurfaceConditionedFaceEmbeddingUNet,
    SurfaceConditionedResidualUNet,
    _load_face_ids_tensor,
    _load_input_chw,
    _predict_delta_tiled,
    _to_chw,
    verify_target_no_gt,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_model(checkpoint: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, np.ndarray | None]:
    ckpt_args = dict(checkpoint.get("args") or {})
    in_ch = int(checkpoint.get("input_channels", 16))
    base_ch = int(ckpt_args.get("base_channels", 24))
    max_delta = float(ckpt_args.get("max_delta", 0.2))
    confidence_mode = str(ckpt_args.get("confidence_mode", "none"))
    confidence_bias = float(ckpt_args.get("confidence_bias", 2.0))
    confidence_min = float(ckpt_args.get("confidence_min", 0.0))
    confidence_max = float(ckpt_args.get("confidence_max", 1.0))
    face_lut = checkpoint.get("face_lut", None)
    if face_lut is not None:
        face_lut = np.asarray(face_lut, dtype=np.int64)
    embedding_dim = int(ckpt_args.get("face_embedding_dim", 0))
    if embedding_dim > 0:
        if face_lut is None or face_lut.size == 0:
            raise RuntimeError("checkpoint requests face embeddings but has no face_lut")
        model = SurfaceConditionedFaceEmbeddingUNet(
            in_ch,
            base_ch,
            max_delta,
            int(face_lut.size + 1),
            embedding_dim,
            confidence_mode=confidence_mode,
            confidence_bias=confidence_bias,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
        )
    else:
        model = SurfaceConditionedResidualUNet(
            in_ch,
            base_ch,
            max_delta,
            confidence_mode=confidence_mode,
            confidence_bias=confidence_bias,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, face_lut


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a trained surface-conditioned residual U-Net checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target_evidence_dir", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eval_tile", type=int, default=512)
    parser.add_argument("--eval_overlap", type=int, default=32)
    parser.add_argument("--audit_path", default="")
    args = parser.parse_args()

    target_evidence_dir = Path(args.target_evidence_dir)
    output_model = Path(args.output_model)
    method_name = str(args.method_name)
    no_gt = verify_target_no_gt(target_evidence_dir)
    if not bool(no_gt.get("passed")):
        raise RuntimeError(f"target no-GT verification failed: {no_gt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    model, face_lut = _build_model(checkpoint, device)
    render_dir = output_model / str(args.split) / method_name / "renders"
    parent_dir = output_model / str(args.split) / method_name / "parent"
    render_dir.mkdir(parents=True, exist_ok=True)
    parent_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in tqdm(evidence_views(target_evidence_dir), desc="checkpoint no-GT apply"):
        z = np.load(path)
        features = torch.from_numpy(_load_input_chw(z))
        face_ids = _load_face_ids_tensor(z, face_lut, max_side=-1)
        parent = np.clip(_to_chw(z["rgb_render"])[:3], 0.0, 1.0).astype(np.float32)
        delta = _predict_delta_tiled(
            model,
            features,
            face_ids=face_ids,
            device=device,
            tile=int(args.eval_tile),
            overlap=int(args.eval_overlap),
        ).numpy()
        adapted = np.clip(parent + float(args.alpha) * delta, 0.0, 1.0)
        save_image_chw(render_dir / f"{path.stem}.png", adapted)
        save_image_chw(parent_dir / f"{path.stem}.png", parent)
        rows.append(
            {
                "view": path.stem,
                "changed_fraction": float(np.mean(np.any(np.abs(float(args.alpha) * delta) > (0.5 / 255.0), axis=0))),
            }
        )
    payload = {
        "schema": "spcarnet_surface_conditioned_unet_checkpoint_apply_v1",
        "checkpoint": str(args.checkpoint),
        "target_evidence_dir": str(target_evidence_dir),
        "output_model": str(output_model),
        "method_name": method_name,
        "split": str(args.split),
        "alpha": float(args.alpha),
        "no_gt_verify": no_gt,
        "view_count": int(len(rows)),
        "mean_changed_fraction": float(np.mean([r["changed_fraction"] for r in rows])) if rows else 0.0,
        "per_view": rows,
        "gt_usage_audit": {
            "uses_train_fit_gt": bool(
                sum(
                    float((checkpoint.get("args") or {}).get(key, 0.0))
                    for key in ("gt_l1_weight", "gt_ssim_weight", "gt_lpips_weight", "gt_grad_weight")
                )
                > 0.0
            ),
            "uses_policy_val_gt": True,
            "uses_target_or_test_gt_during_apply": False,
            "target_gt_visible_to_apply": bool(no_gt.get("target_gt_visible_to_apply", False)),
            "target_residual_visible_to_apply": bool(no_gt.get("target_residual_visible_to_apply", False)),
        },
    }
    audit_path = Path(args.audit_path) if str(args.audit_path) else output_model / f"{method_name}_checkpoint_apply_audit.json"
    _write_json(audit_path, payload)
    print("OUT", audit_path, flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_view"}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

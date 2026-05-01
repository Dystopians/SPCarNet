#!/usr/bin/env python3
"""SP-CarNet Stage 3 — posterior encoder eval entrypoint.

Loads a Stage-3 checkpoint, runs the encoder on each held-out object's
``partial_observed_points``, decodes through the (frozen) Stage-2 decoder, and
reports the eight headline metrics plus a comparison block against any
provided v0.x / Stage-2 baselines.

Required metrics (see Stage-3 design §4):
- recon_chamfer_l1
- hidden_chamfer_l1
- visible_preservation_error
- free_space_violation_rate
- mesh_iou_at_0.5             (consumes the fixed GLB-derived voxelisation when available)
- zero_corruption_recon_chamfer_l1
- latent_retrieval_error      (train-split only diagnostic)
- mesh_extraction_success_rate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset  # noqa: E402
from ss3dm_prior.mesh.marching_cubes import (  # noqa: E402
    extract_patch_mesh,
    mesh_iou_at_iso,
)
from ss3dm_prior.models.spcarnet_posterior import (  # noqa: E402
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: bidirectional chamfer
# ---------------------------------------------------------------------------


def _bidirectional_chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    d = torch.cdist(a_t, b_t, p=2)
    fwd = d.min(dim=1).values.mean()
    bwd = d.min(dim=0).values.mean()
    return float(0.5 * (fwd + bwd).item())


# ---------------------------------------------------------------------------
# Voxel volume helpers — mirrors Stage 2 eval; the GLB-fix sub-task patches
# this in-place. Until then we keep both the sparse-point fallback and the
# trimesh-based path so the script does not silently emit garbage.
# ---------------------------------------------------------------------------


def _voxelise_clean_points(
    points: np.ndarray, *, resolution: int, padding: float = 1.05
) -> np.ndarray:
    side = 2.0 * padding
    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    coords = (points + padding) / side
    valid = np.all((coords >= 0.0) & (coords < 1.0), axis=1)
    coords = coords[valid]
    idx = np.minimum((coords * resolution).astype(np.int64), resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid


def _voxelise_glb(
    glb_path: str, *, resolution: int, padding: float = 1.05
) -> np.ndarray | None:
    """Filled-volume voxelisation of the source GLB.

    Returns None if trimesh / pyglet are unavailable, the GLB is missing, or the
    mesh is non-watertight and cannot be filled. The caller falls back to the
    sparse-point path in that case.
    """
    try:
        import trimesh  # type: ignore
    except Exception:
        return None
    if not Path(glb_path).is_file():
        return None
    try:
        loaded = trimesh.load(glb_path, force="mesh")
        if loaded is None or not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
            return None
        # Cache vertices are already in canonical [-1, 1]^3 (Stage 1 audit:
        # patch_center_world == 0 and patch_radius_m == 1 for all records).
        # Apply the same normalisation to the GLB by scaling its bounding sphere
        # to unit radius around the origin.
        verts = np.asarray(loaded.vertices, dtype=np.float32)
        centre = verts.mean(axis=0)
        centred = verts - centre
        radius = float(np.linalg.norm(centred, axis=1).max())
        if radius <= 0:
            return None
        loaded.vertices = centred / radius
        pitch = 2.0 * padding / resolution
        try:
            voxels = loaded.voxelized(pitch=pitch)
            if voxels is None:
                return None
            voxels = voxels.fill()
            mat = np.asarray(voxels.matrix, dtype=np.uint8)
        except Exception:
            return None
        # Resize / pad to (resolution, resolution, resolution) regardless of
        # voxel grid shape.
        out = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
        s = mat.shape
        x = min(s[0], resolution)
        y = min(s[1], resolution)
        z = min(s[2], resolution)
        out[:x, :y, :z] = mat[:x, :y, :z]
        return out
    except Exception:
        return None


def _decode_volume(
    decoder: SPCarShapeFieldDecoder,
    z: torch.Tensor,
    *,
    resolution: int,
    device: torch.device,
    padding: float = 1.05,
    chunk: int = 65536,
) -> np.ndarray:
    axis = torch.linspace(-padding, padding, resolution, dtype=torch.float32, device=device)
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    out = torch.empty(grid.shape[0], dtype=torch.float32, device=device)
    z = z.to(device).unsqueeze(0)
    with torch.no_grad():
        for start in range(0, grid.shape[0], chunk):
            x = grid[start : start + chunk].unsqueeze(0)
            logits = decoder(x, z).squeeze(0)
            out[start : start + chunk] = torch.sigmoid(logits)
    return out.detach().cpu().numpy().reshape(resolution, resolution, resolution)


# ---------------------------------------------------------------------------
# Model construction from checkpoint
# ---------------------------------------------------------------------------


def _build_models(checkpoint: dict, device: torch.device) -> tuple[
    SPCarPosteriorCompletionModel, SPCarShapeFieldDecoder, SPCarPosteriorEncoder
]:
    model_cfg = checkpoint["model_cfg"]
    decoder = SPCarShapeFieldDecoder(
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=384,
        depth=6,
        num_fourier_freqs=32,
        field_kind="occupancy",
        feature_dim=0,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder = SPCarPosteriorEncoder(
        latent_dim=int(model_cfg["latent_dim"]),
        feature_dim=int(model_cfg["encoder_feature_dim"]),
        num_xattn_layers=int(model_cfg["num_xattn_layers"]),
        num_self_attn_layers=int(model_cfg["num_self_attn_layers"]),
        num_latent_queries=int(model_cfg["num_latent_queries"]),
        attention_heads=int(model_cfg["attention_heads"]),
        ffn_dim=int(model_cfg["ffn_dim"]),
        dropout=float(model_cfg["dropout"]),
        posterior_kind=str(model_cfg["posterior_kind"]),
        use_normals=bool(model_cfg["use_normals"]),
        use_conditioning_adapter=bool(model_cfg["use_conditioning_adapter"]),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    completion = SPCarPosteriorCompletionModel(
        encoder=encoder,
        decoder=decoder,
        decoder_finetune_enabled=bool(checkpoint.get("decoder_finetune_enabled", False)),
    ).to(device)
    completion.eval()
    return completion, decoder, encoder


# ---------------------------------------------------------------------------
# Per-object eval
# ---------------------------------------------------------------------------


def _encode_observation(
    completion: SPCarPosteriorCompletionModel,
    partial: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        partial_t = torch.from_numpy(np.asarray(partial, dtype=np.float32)).unsqueeze(0).to(device)
        post = completion.encode(partial_t, sample=False)
    return post.z_mean.squeeze(0)


def _per_object_metrics(
    completion: SPCarPosteriorCompletionModel,
    decoder: SPCarShapeFieldDecoder,
    item: dict[str, Any],
    *,
    device: torch.device,
    mc_resolution: int,
    sample_count: int,
    iso_level: float,
    free_violation_threshold: float,
    z_target_table: torch.Tensor | None,
    object_id_to_row: dict[str, int] | None,
    use_glb_iou: bool,
) -> dict[str, Any]:
    oid = item["object_id"]
    metrics: dict[str, Any] = {"object_id": oid, "split": item.get("split")}

    partial = item.get("partial_observed_points")
    if partial is None or len(partial) == 0:
        partial = item["clean_points_object"]
    z_pred = _encode_observation(completion, partial, device)

    # ------- mesh extraction & sampling -------
    def _occ_fn(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = query.unsqueeze(0).to(device)
            return torch.sigmoid(decoder(q, z_pred.unsqueeze(0))).squeeze(0)

    result = extract_patch_mesh(
        occupancy_fn=_occ_fn,
        device=device,
        patch_radius=1.0,
        resolution=mc_resolution,
        iso_level=iso_level,
    )
    metrics["mesh_extraction_success"] = int(result.mesh is not None and result.face_count > 0)
    metrics["vertex_count"] = int(result.vertex_count)
    metrics["face_count"] = int(result.face_count)

    clean = np.asarray(item["clean_points_object"], dtype=np.float32)

    if result.mesh is not None and result.face_count > 0:
        try:
            sampled = np.asarray(result.mesh.sample(sample_count), dtype=np.float32)
        except Exception:
            sampled = np.asarray(result.mesh.vertices, dtype=np.float32)
    else:
        sampled = np.zeros((0, 3), dtype=np.float32)

    metrics["recon_chamfer_l1"] = (
        _bidirectional_chamfer_l1(sampled, clean) if sampled.shape[0] > 0 else float("nan")
    )

    hidden = item.get("hidden_clean_points")
    metrics["hidden_chamfer_l1"] = (
        _bidirectional_chamfer_l1(sampled, np.asarray(hidden, dtype=np.float32))
        if hidden is not None and len(hidden) > 0 and sampled.shape[0] > 0
        else float("nan")
    )

    # visible_preservation_error: each observed point's distance to nearest mesh sample.
    if sampled.shape[0] > 0:
        d_p_to_m = torch.cdist(
            torch.from_numpy(np.asarray(partial, dtype=np.float32)),
            torch.from_numpy(sampled),
        ).numpy()
        metrics["visible_preservation_error"] = float(np.mean(np.min(d_p_to_m, axis=1)))
    else:
        metrics["visible_preservation_error"] = float("nan")

    # ------- free-space violation -------
    free = item.get("free_space_query_points")
    if free is not None and len(free) > 0:
        with torch.no_grad():
            free_t = torch.from_numpy(np.asarray(free, dtype=np.float32)).unsqueeze(0).to(device)
            free_p = torch.sigmoid(decoder(free_t, z_pred.unsqueeze(0))).squeeze(0)
        violation = (free_p > free_violation_threshold).float().mean().item()
        metrics["free_space_violation_rate"] = float(violation)
    else:
        metrics["free_space_violation_rate"] = float("nan")

    # ------- mesh IoU at 0.5 -------
    pred_vol = _decode_volume(decoder, z_pred, resolution=mc_resolution, device=device)
    gt_vol: np.ndarray | None = None
    if use_glb_iou and item.get("source_mesh_path"):
        gt_vol = _voxelise_glb(item["source_mesh_path"], resolution=mc_resolution)
    if gt_vol is None:
        gt_vol = _voxelise_clean_points(clean, resolution=mc_resolution)
        metrics["iou_gt_source"] = "sparse_points"
    else:
        metrics["iou_gt_source"] = "glb_filled"
    try:
        metrics["mesh_iou_at_0.5"] = float(
            mesh_iou_at_iso(pred_vol, gt_vol.astype(np.float32), iso_level=iso_level)
        )
    except Exception:
        metrics["mesh_iou_at_0.5"] = float("nan")

    # ------- zero-corruption recon chamfer -------
    z_zero = _encode_observation(completion, clean, device)

    def _occ_zero(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = query.unsqueeze(0).to(device)
            return torch.sigmoid(decoder(q, z_zero.unsqueeze(0))).squeeze(0)

    zero_result = extract_patch_mesh(
        occupancy_fn=_occ_zero,
        device=device,
        patch_radius=1.0,
        resolution=mc_resolution,
        iso_level=iso_level,
    )
    if zero_result.mesh is not None and zero_result.face_count > 0:
        try:
            zero_sampled = np.asarray(
                zero_result.mesh.sample(sample_count), dtype=np.float32
            )
            metrics["zero_corruption_recon_chamfer_l1"] = _bidirectional_chamfer_l1(
                zero_sampled, clean
            )
        except Exception:
            metrics["zero_corruption_recon_chamfer_l1"] = float("nan")
    else:
        metrics["zero_corruption_recon_chamfer_l1"] = float("nan")

    # ------- latent retrieval error (train-split only) -------
    if (
        z_target_table is not None
        and object_id_to_row is not None
        and oid in object_id_to_row
    ):
        row = object_id_to_row[oid]
        if 0 <= row < z_target_table.shape[0]:
            z_target = z_target_table[row]
            metrics["latent_retrieval_error"] = float(
                (z_pred.detach().cpu() - z_target).norm(p=2).item()
            )
        else:
            metrics["latent_retrieval_error"] = float("nan")
    else:
        metrics["latent_retrieval_error"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(per_object: list[dict[str, Any]]) -> dict[str, float]:
    successes = [m for m in per_object if m.get("mesh_extraction_success") == 1]

    def _avg(key: str, *, source: list[dict[str, Any]] | None = None) -> float:
        src = source if source is not None else successes
        vals = [
            m[key] for m in src
            if isinstance(m.get(key), (int, float)) and not math.isnan(m[key])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_total": len(per_object),
        "n_extracted": len(successes),
        "mesh_extraction_success_rate": float(len(successes) / max(len(per_object), 1)),
        "recon_chamfer_l1_mean": _avg("recon_chamfer_l1"),
        "hidden_chamfer_l1_mean": _avg("hidden_chamfer_l1"),
        "visible_preservation_error_mean": _avg("visible_preservation_error"),
        "free_space_violation_rate_mean": _avg("free_space_violation_rate", source=per_object),
        "mesh_iou_at_0.5_mean": _avg("mesh_iou_at_0.5"),
        "zero_corruption_recon_chamfer_l1_mean": _avg("zero_corruption_recon_chamfer_l1"),
        "latent_retrieval_error_mean": _avg("latent_retrieval_error", source=per_object),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object_index",
                        default=str(REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"))
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--mc_resolution", type=int, default=32)
    parser.add_argument("--sample_count", type=int, default=4096)
    parser.add_argument("--iso_level", type=float, default=0.5)
    parser.add_argument("--free_violation_threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_glb_iou", action="store_true",
                        help="Voxelise the source GLB (requires trimesh) instead of the sparse-point fallback.")
    parser.add_argument("--baseline_v07", default=None,
                        help="Path to v0.7 residual eval JSON for comparison; can be the eval JSON or a summary block.")
    parser.add_argument("--baseline_v082", default=None,
                        help="Path to v0.8.2 point-flow eval JSON for comparison.")
    parser.add_argument("--baseline_stage2", default=None,
                        help="Path to Stage-2 auto-decoder eval JSON for comparison.")
    args = parser.parse_args(argv)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    completion, decoder, encoder = _build_models(ckpt, device)
    z_target_table = ckpt.get("stage2_latent_table")
    if z_target_table is not None:
        z_target_table = z_target_table.detach()
    stage2_object_id_to_row = ckpt.get("stage2_object_id_to_row")

    dataset = SPCarObjectDataset(args.object_index, splits=tuple(args.splits))
    n_total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))

    per_object: list[dict[str, Any]] = []
    for i in range(n_total):
        item = dataset[i]
        try:
            metrics = _per_object_metrics(
                completion,
                decoder,
                item,
                device=device,
                mc_resolution=args.mc_resolution,
                sample_count=args.sample_count,
                iso_level=args.iso_level,
                free_violation_threshold=args.free_violation_threshold,
                z_target_table=z_target_table,
                object_id_to_row=stage2_object_id_to_row,
                use_glb_iou=args.use_glb_iou,
            )
        except Exception as exc:
            metrics = {
                "object_id": item.get("object_id"),
                "split": item.get("split"),
                "error": f"{type(exc).__name__}: {exc}",
                "mesh_extraction_success": 0,
            }
        per_object.append(metrics)

    summary = _aggregate(per_object)

    # ---------------- baselines ----------------
    baselines: dict[str, dict[str, Any]] = {}
    for tag, path in (
        ("v0.7", args.baseline_v07),
        ("v0.8.2", args.baseline_v082),
        ("stage2_autodecoder", args.baseline_stage2),
    ):
        if path is None:
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
            baselines[tag] = doc.get("summary", doc)
        except Exception as exc:
            baselines[tag] = {"error": f"{type(exc).__name__}: {exc}"}

    out_doc = {
        "summary": summary,
        "per_object": per_object,
        "baselines": baselines,
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
    }

    target = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".eval.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(out_doc, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

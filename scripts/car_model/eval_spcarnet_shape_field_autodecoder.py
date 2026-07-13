#!/usr/bin/env python3
"""SP-CarNet Stage 2 — auto-decoder eval entrypoint.

Loads a trained auto-decoder checkpoint, extracts a Marching-Cubes mesh per
validation object, samples points from the mesh, and reports:

- ``recon_chamfer_l1`` (mesh-sampled vs ``clean_points``)
- ``hidden_chamfer_l1`` (mesh-sampled vs ``hidden_clean_points``)
- ``mesh_iou_at_0.5`` (volume IoU using the auto-decoder field vs the
  voxelised filled GT mesh, in the same canonical ``[-1, 1]`` frame as the
  cache; falls back to ``mesh_iou_at_0.5_shell`` when the GLB is missing or
  trimesh voxelisation fails).
- ``surface_normal_consistency`` (mesh face normals vs ``clean_normals``)
- ``mesh_extraction_success_rate``

Note: the headline auto-decoder run is not yet trained; this script ships as
infrastructure for the upcoming training. Smoke runs use ``--limit 2``.

mesh_iou bug fix (2026-04-29)
-----------------------------
The previous implementation voxelised only the 2048 ``clean_points`` of each
object; that produces a thin sparse shell that intersects the dense filled
volume produced by the decoder at ~50%, biasing the metric to ~0.5 even for
perfect reconstructions. The fix is to load the source GLB, apply the same
canonical normalisation that was baked into the cache (``(v - centroid) /
radius``, with parameters read from the NPZ's ``patch_metadata_json``), and
voxelise the mesh with ``trimesh.voxelized(pitch=2/res).fill()``. When the
GLB cannot be located (manifest path missing / unmappable basename) or when
trimesh voxelisation fails (e.g. non-watertight + degenerate fill), the
script falls back to a shell-IoU at radius ``r=1.5/resolution`` reported
under ``mesh_iou_at_0.5_shell`` so the user can distinguish.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    surface_normal_consistency,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402
from ss3dm_prior.training.spcarnet_autodecoder import (  # noqa: E402
    ShapeFieldLossConfig,
    ShapeFieldTrainConfig,
    assemble_query_batch,
    compute_losses,
)


# Locations searched for the source GLB by basename when the manifest path
# does not resolve. The Stage-1 manifest stores ``./raw/<basename>.<ext>``
# joined with ``dataset_root`` pointing at the cache directory, but the GLBs
# actually live at ``/data/peilincai/car_models/meshfleet_ext_v02/{train,
# test}/raw/``. Both locations are scanned. Symlinks are followed.
_GLB_FALLBACK_DIRS = (
    Path("/data/peilincai/car_models/meshfleet_ext_v02/train/raw"),
    Path("/data/peilincai/car_models/meshfleet_ext_v02/test/raw"),
)


def _bidirectional_chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    d = torch.cdist(a_t, b_t, p=2)
    fwd = d.min(dim=1).values.mean()
    bwd = d.min(dim=0).values.mean()
    return float(0.5 * (fwd + bwd).item())


def _json_sanitize(value: Any) -> Any:
    """Convert tensors/arrays/non-finite floats into strict-JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, torch.Tensor):
        return _json_sanitize(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _voxelise_points(points: np.ndarray, *, resolution: int, padding: float = 1.05) -> np.ndarray:
    """Binary occupancy volume at the given resolution by point binning.

    Retained for the shell-IoU fallback (see ``_voxelise_gt_mesh``).
    """
    side = 2.0 * padding
    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    coords = (points + padding) / side  # normalise to [0, 1]
    valid = np.all((coords >= 0.0) & (coords < 1.0), axis=1)
    coords = coords[valid]
    idx = np.minimum((coords * resolution).astype(np.int64), resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return grid


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read source_mesh_manifest.json once and return ``{car_id: record}``.

    The manifest's ``dataset_root`` field is recorded so the caller can
    optionally try ``dataset_root + local_path`` first before falling back to
    a basename scan in the known on-disk GLB roots.
    """
    if not manifest_path.is_file():
        return {"dataset_root": None, "by_id": {}}
    with manifest_path.open() as f:
        doc = json.load(f)
    by_id: dict[str, dict[str, Any]] = {}
    for rec in doc.get("records", []) or []:
        cid = rec.get("car_id")
        if cid:
            by_id[cid] = rec
    return {"dataset_root": doc.get("dataset_root"), "by_id": by_id}


def _resolve_glb_path(
    object_id: str, manifest: dict[str, Any], npz_metadata: dict[str, Any] | None
) -> Path | None:
    """Locate the source GLB for ``object_id``.

    Resolution order:
    1. ``dataset_root + local_path`` from the manifest record (the contract).
    2. ``patch_metadata_json["source_mesh_path"]`` from the NPZ — sometimes
       points at ``/data2/peilincai/...`` which has been moved to
       ``/data/peilincai/...``.
    3. Basename scan under :data:`_GLB_FALLBACK_DIRS`.

    Returns ``None`` when no candidate exists; callers must fall back to a
    point-cloud IoU in that case.
    """
    rec = manifest.get("by_id", {}).get(object_id)
    candidates: list[Path] = []
    if rec is not None:
        local_path = rec.get("local_path")
        ds_root = manifest.get("dataset_root")
        if local_path and ds_root:
            candidates.append(Path(os.path.normpath(os.path.join(ds_root, local_path))))
        # Also try swapping /data2/peilincai -> /data/peilincai.
        if local_path and ds_root:
            swapped = os.path.normpath(os.path.join(ds_root, local_path)).replace(
                "/data2/peilincai/", "/data/peilincai/"
            )
            candidates.append(Path(swapped))
    if npz_metadata is not None:
        smp = npz_metadata.get("source_mesh_path")
        if smp:
            candidates.append(Path(smp))
            candidates.append(Path(str(smp).replace("/data2/peilincai/", "/data/peilincai/")))
    for cand in candidates:
        if cand.is_file():
            return cand
    # Basename scan.
    bn_candidates: set[str] = set()
    if rec is not None and rec.get("local_path"):
        bn_candidates.add(os.path.basename(rec["local_path"]))
    if npz_metadata is not None and npz_metadata.get("source_mesh_path"):
        bn_candidates.add(os.path.basename(npz_metadata["source_mesh_path"]))
    bn_candidates.add(f"{object_id}.glb")
    bn_candidates.add(f"{object_id}.fbx")
    bn_candidates.add(f"{object_id}.gltf")
    for raw_dir in _GLB_FALLBACK_DIRS:
        if not raw_dir.is_dir():
            continue
        for bn in bn_candidates:
            cand = raw_dir / bn
            if cand.is_file():
                return cand
    return None


def _read_npz_metadata(patch_files: list[str] | None) -> dict[str, Any] | None:
    """Return ``patch_metadata_json`` from the primary NPZ, if available."""
    if not patch_files:
        return None
    path = Path(patch_files[0])
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            if "patch_metadata_json" in data.files:
                raw = data["patch_metadata_json"]
                meta = json.loads(str(raw))
                return meta
    except Exception:
        return None
    return None


def _voxelise_gt_mesh(
    *,
    object_id: str,
    npz_metadata: dict[str, Any] | None,
    manifest: dict[str, Any],
    resolution: int,
    padding: float = 1.05,
) -> tuple[np.ndarray | None, str]:
    """Voxelise the canonical GT mesh into a filled occupancy grid.

    Returns ``(grid, status)``. ``status`` is one of:

    - ``"mesh_filled"``    — trimesh voxel.fill() succeeded.
    - ``"missing_glb"``    — manifest + fallbacks could not locate the GLB.
    - ``"missing_metadata"`` — NPZ lacked ``patch_metadata_json`` for the
      canonical normalisation.
    - ``"voxelisation_failed"`` — load or voxelise raised; the caller should
      use the shell-IoU fallback.
    """
    glb_path = _resolve_glb_path(object_id, manifest, npz_metadata)
    if glb_path is None:
        return None, "missing_glb"
    if npz_metadata is None or "original_centroid_world" not in npz_metadata:
        return None, "missing_metadata"
    centroid = np.asarray(npz_metadata["original_centroid_world"], dtype=np.float64)
    radius = float(npz_metadata.get("original_radius_world", 0.0))
    if not (radius > 0.0 and np.isfinite(radius)):
        return None, "missing_metadata"

    try:
        import trimesh  # local import — only required for the IoU path.
        mesh = trimesh.load(str(glb_path), force="mesh", skip_materials=True)
        if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) == 0:
            return None, "voxelisation_failed"
        verts = (np.asarray(mesh.vertices, dtype=np.float64) - centroid) / max(radius, 1e-12)
        mesh.vertices = verts.astype(np.float32)

        pitch = 2.0 / float(resolution)
        vox = mesh.voxelized(pitch=pitch).fill()
        filled_pts = np.asarray(vox.points, dtype=np.float32)
        if filled_pts.size == 0:
            return None, "voxelisation_failed"
    except Exception:
        return None, "voxelisation_failed"

    side = 2.0 * padding
    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    coords = (filled_pts + padding) / side
    valid = np.all((coords >= 0.0) & (coords < 1.0), axis=1)
    coords = coords[valid]
    if coords.size == 0:
        return None, "voxelisation_failed"
    idx = np.minimum((coords * resolution).astype(np.int64), resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    if int(grid.sum()) < 8:
        return None, "voxelisation_failed"
    return grid, "mesh_filled"


def _shell_iou(
    *,
    pred_volume: np.ndarray,
    clean_points: np.ndarray,
    resolution: int,
    iso_level: float,
    padding: float = 1.05,
) -> float:
    """Shell-IoU: dilate both the predicted level set and the binary point
    shell by ``r = 1.5 / resolution`` cells (i.e. ~1.5 voxel radius) and
    compute the IoU of the dilated shells. This is a fallback metric for the
    case where the GLB is unavailable.
    """
    if pred_volume.shape != (resolution, resolution, resolution):
        return float("nan")
    pred_mask = (np.asarray(pred_volume) >= iso_level).astype(bool)
    gt_shell = _voxelise_points(clean_points, resolution=resolution, padding=padding).astype(bool)
    # Dilate by ~1.5 cells via convolution with a small box kernel.
    try:
        from scipy import ndimage  # type: ignore

        kernel_radius = max(1, int(round(1.5)))  # cells
        struct = ndimage.generate_binary_structure(3, 3)
        pred_dil = ndimage.binary_dilation(pred_mask, structure=struct, iterations=kernel_radius)
        gt_dil = ndimage.binary_dilation(gt_shell, structure=struct, iterations=kernel_radius)
    except Exception:
        # Lightweight fallback dilation: uniform-filter threshold.
        def _box_dilate(vol: np.ndarray, r: int) -> np.ndarray:
            out = vol.astype(np.float32, copy=True)
            for axis in range(3):
                cs = np.cumsum(out, axis=axis)
                pad_lo = np.zeros_like(out.take(indices=range(r), axis=axis))
                # Simple: convolve by adding shifted copies.
                shifted_total = np.zeros_like(out)
                for s in range(-r, r + 1):
                    shifted_total += np.roll(out, shift=s, axis=axis)
                out = (shifted_total > 0).astype(np.float32)
                _ = (cs, pad_lo)
            return out.astype(bool)
        pred_dil = _box_dilate(pred_mask, 1)
        gt_dil = _box_dilate(gt_shell, 1)
    inter = float(np.logical_and(pred_dil, gt_dil).sum())
    union = float(np.logical_or(pred_dil, gt_dil).sum())
    if union <= 0.0:
        return float("nan")
    return inter / union


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


def _make_occupancy_fn(decoder: SPCarShapeFieldDecoder, z: torch.Tensor, device: torch.device):
    def _fn(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = query.unsqueeze(0).to(device)
            logits = decoder(q, z.to(device).unsqueeze(0))
            return torch.sigmoid(logits).squeeze(0)

    return _fn


def _load_resolved_loss_config(path: str | None) -> ShapeFieldLossConfig:
    if not path:
        return ShapeFieldLossConfig()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing resolved config: {p}")
    with p.open() as f:
        doc = json.load(f)
    return ShapeFieldLossConfig.from_dict(doc.get("loss", doc.get("losses", {})))


def _fit_latent_for_item(
    *,
    decoder: SPCarShapeFieldDecoder,
    item: dict[str, Any],
    init_z: torch.Tensor,
    device: torch.device,
    field_kind: str,
    loss_cfg: ShapeFieldLossConfig,
    fit_steps: int,
    fit_lr: float,
    fit_queries_surface: int,
    fit_queries_free: int,
    fit_queries_hard: int,
    fit_queries_mixed: int,
    fit_queries_band: int,
    fit_band_epsilon: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """MAP-fit a held-out latent with the decoder frozen.

    Stage-2 auto-decoders only store train-object latent rows. Held-out objects
    therefore require a z-only clean-shape MAP fit to test decoder capacity; this
    must be reported separately from amortised Stage-3 inference.
    """
    if fit_steps <= 0:
        return init_z.detach(), {"fit_steps": 0, "fit_status": "disabled"}

    was_training = decoder.training
    decoder.eval()
    old_requires_grad = [p.requires_grad for p in decoder.parameters()]
    for p in decoder.parameters():
        p.requires_grad_(False)

    z = init_z.detach().clone().to(device).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=float(fit_lr))
    rng = np.random.default_rng(seed)
    train_cfg = ShapeFieldTrainConfig(
        queries_surface=int(fit_queries_surface),
        queries_free=int(fit_queries_free),
        queries_hard=int(fit_queries_hard),
        queries_mixed=int(fit_queries_mixed),
        queries_band=int(fit_queries_band),
        band_epsilon=float(fit_band_epsilon),
        queries_eikonal=0,
        device=str(device),
    )
    first_loss: float | None = None
    final_loss: float | None = None
    status = "ok"
    for _step in range(int(fit_steps)):
        q = assemble_query_batch(item, cfg=train_cfg, rng=rng, field_kind=field_kind)
        queries = {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in q.items()}
        opt.zero_grad(set_to_none=True)
        loss, _metrics = compute_losses(
            decoder,
            z.unsqueeze(0),
            queries,
            loss_cfg=loss_cfg,
            field_kind=field_kind,
        )
        if not torch.isfinite(loss):
            status = "nonfinite_loss"
            break
        if first_loss is None:
            first_loss = float(loss.detach().item())
        final_loss = float(loss.detach().item())
        loss.backward()
        opt.step()

    for p, req in zip(decoder.parameters(), old_requires_grad):
        p.requires_grad_(req)
    decoder.train(was_training)
    return z.detach(), {
        "fit_steps": int(fit_steps),
        "fit_lr": float(fit_lr),
        "fit_status": status,
        "fit_loss_first": first_loss,
        "fit_loss_final": final_loss,
    }


def evaluate(
    *,
    decoder: SPCarShapeFieldDecoder,
    latent_table: torch.nn.Embedding | torch.nn.Parameter,
    object_id_to_row: dict[str, int],
    dataset: SPCarObjectDataset,
    device: torch.device,
    manifest: dict[str, Any] | None = None,
    mc_resolution: int = 32,
    sample_count: int = 4096,
    iso_level: float = 0.5,
    limit: int = 0,
    fit_missing_latents: bool = False,
    fit_all_latents: bool = False,
    loss_cfg: ShapeFieldLossConfig | None = None,
    fit_steps: int = 0,
    fit_lr: float = 1e-2,
    fit_queries_surface: int = 384,
    fit_queries_free: int = 384,
    fit_queries_hard: int = 128,
    fit_queries_mixed: int = 128,
    fit_queries_band: int = 0,
    fit_band_epsilon: float = 0.02,
    fit_seed: int = 0,
) -> dict[str, Any]:
    metrics_per_object: list[dict[str, float]] = []
    n_total = len(dataset) if limit <= 0 else min(limit, len(dataset))
    n_extracted = 0
    n_iou_filled = 0
    n_iou_shell = 0

    if manifest is None:
        manifest = {"dataset_root": None, "by_id": {}}
    if loss_cfg is None:
        loss_cfg = ShapeFieldLossConfig()
    model_field_kind = getattr(decoder, "field_kind", "occupancy")
    latent_mean = latent_table.detach().mean(dim=0).to(device)

    for i in range(n_total):
        item = dataset[i]
        oid = item["object_id"]
        fit_info: dict[str, Any] = {}
        if oid in object_id_to_row:
            z = latent_table[object_id_to_row[oid]].detach()
            latent_source = "train_table"
            if fit_all_latents:
                z, fit_info = _fit_latent_for_item(
                    decoder=decoder,
                    item=item,
                    init_z=z,
                    device=device,
                    field_kind=model_field_kind,
                    loss_cfg=loss_cfg,
                    fit_steps=fit_steps,
                    fit_lr=fit_lr,
                    fit_queries_surface=fit_queries_surface,
                    fit_queries_free=fit_queries_free,
                    fit_queries_hard=fit_queries_hard,
                    fit_queries_mixed=fit_queries_mixed,
                    fit_queries_band=fit_queries_band,
                    fit_band_epsilon=fit_band_epsilon,
                    seed=fit_seed + i,
                )
                latent_source = "train_table_map_fit"
        elif fit_missing_latents:
            z, fit_info = _fit_latent_for_item(
                decoder=decoder,
                item=item,
                init_z=latent_mean,
                device=device,
                field_kind=model_field_kind,
                loss_cfg=loss_cfg,
                fit_steps=fit_steps,
                fit_lr=fit_lr,
                fit_queries_surface=fit_queries_surface,
                fit_queries_free=fit_queries_free,
                fit_queries_hard=fit_queries_hard,
                fit_queries_mixed=fit_queries_mixed,
                fit_queries_band=fit_queries_band,
                fit_band_epsilon=fit_band_epsilon,
                seed=fit_seed + i,
            )
            latent_source = "heldout_map_fit"
        else:
            metrics_per_object.append(
                {
                    "object_id": oid,
                    "skipped": True,
                    "skip_reason": "missing_train_latent",
                    "latent_source": "missing",
                }
            )
            continue

        result = extract_patch_mesh(
            occupancy_fn=_make_occupancy_fn(decoder, z, device),
            device=device,
            patch_radius=1.0,
            resolution=mc_resolution,
            iso_level=iso_level,
        )
        if result.mesh is None:
            metrics_per_object.append(
                {
                    "object_id": oid,
                    "extraction_success": 0,
                    "vertex_count": result.vertex_count,
                    "latent_source": latent_source,
                    **fit_info,
                }
            )
            continue

        mesh = result.mesh
        try:
            sampled, face_indices = mesh.sample(sample_count, return_index=True)
        except Exception:
            sampled = np.asarray(mesh.vertices, dtype=np.float32)
            face_indices = None

        clean = np.asarray(item["clean_points_object"], dtype=np.float32)
        chamfer_full = _bidirectional_chamfer_l1(np.asarray(sampled, dtype=np.float32), clean)
        hidden = item.get("hidden_clean_points")
        chamfer_hidden = (
            _bidirectional_chamfer_l1(np.asarray(sampled, dtype=np.float32), np.asarray(hidden, dtype=np.float32))
            if hidden is not None and len(hidden) > 0
            else float("nan")
        )

        # Volume IoU (decoder field vs voxelised filled GT mesh, with shell
        # fallback when the GLB is missing). The two metrics live in
        # different keys so the user can distinguish them at aggregation.
        iou_filled = float("nan")
        iou_shell = float("nan")
        iou_status = "skipped"
        try:
            pred_vol = _decode_volume(decoder, z, resolution=mc_resolution, device=device)
            patch_files = (
                dataset.get_object_record(i).get("patch_files") if hasattr(dataset, "get_object_record") else None
            )
            npz_meta = _read_npz_metadata(patch_files)
            gt_vol, status = _voxelise_gt_mesh(
                object_id=oid,
                npz_metadata=npz_meta,
                manifest=manifest,
                resolution=mc_resolution,
            )
            iou_status = status
            if gt_vol is not None:
                iou_filled = mesh_iou_at_iso(pred_vol, gt_vol.astype(np.float32), iso_level=iso_level)
                n_iou_filled += 1
            else:
                iou_shell = _shell_iou(
                    pred_volume=pred_vol,
                    clean_points=clean,
                    resolution=mc_resolution,
                    iso_level=iso_level,
                )
                n_iou_shell += 1
        except Exception as exc:  # noqa: BLE001
            iou_status = f"error: {type(exc).__name__}"

        # Surface normal consistency: mesh face normals vs nearest clean_normals.
        clean_normals = item.get("clean_normals_object")
        normal_cos = float("nan")
        if face_indices is not None and clean_normals is not None and mesh.face_normals is not None:
            face_normals = np.asarray(mesh.face_normals, dtype=np.float32)[face_indices]
            face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-8)
            # Nearest clean point per sampled point.
            sampled32 = np.asarray(sampled, dtype=np.float32)
            d = torch.cdist(torch.from_numpy(sampled32), torch.from_numpy(clean.astype(np.float32))).numpy()
            nearest = np.argmin(d, axis=1)
            target_normals = np.asarray(clean_normals, dtype=np.float32)[nearest]
            target_normals /= np.maximum(np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-8)
            normal_cos = float(np.mean(np.abs(np.sum(face_normals * target_normals, axis=1))))

        metrics_per_object.append(
            {
                "object_id": oid,
                "latent_source": latent_source,
                "extraction_success": 1,
                "vertex_count": result.vertex_count,
                "face_count": result.face_count,
                "watertight": bool(result.watertight),
                "recon_chamfer_l1": chamfer_full,
                "hidden_chamfer_l1": chamfer_hidden,
                "mesh_iou_at_0.5": float(iou_filled),
                "mesh_iou_at_0.5_shell": float(iou_shell),
                "iou_status": iou_status,
                "surface_normal_consistency": normal_cos,
                **fit_info,
            }
        )
        n_extracted += 1

    successes = [m for m in metrics_per_object if m.get("extraction_success") == 1]

    def _avg(key: str) -> float:
        vals = [m[key] for m in successes if isinstance(m.get(key), (int, float)) and not math.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_objects_evaluated": n_total,
        "n_extracted": n_extracted,
        "mesh_extraction_success_rate": float(n_extracted) / max(n_total, 1),
        "recon_chamfer_l1_mean": _avg("recon_chamfer_l1"),
        "hidden_chamfer_l1_mean": _avg("hidden_chamfer_l1"),
        "mesh_iou_at_0.5_mean": _avg("mesh_iou_at_0.5"),
        "mesh_iou_at_0.5_shell_mean": _avg("mesh_iou_at_0.5_shell"),
        "n_iou_filled": int(n_iou_filled),
        "n_iou_shell": int(n_iou_shell),
        "surface_normal_consistency_mean": _avg("surface_normal_consistency"),
        "fit_missing_latents": bool(fit_missing_latents),
        "fit_all_latents": bool(fit_all_latents),
        "fit_steps": int(fit_steps),
        "fit_queries_band": int(fit_queries_band),
        "fit_band_epsilon": float(fit_band_epsilon),
    }
    return {"summary": summary, "per_object": metrics_per_object}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Auto-decoder checkpoint (.pt) with decoder state dict + latent table.")
    parser.add_argument("--object_index", default=str(REPO_ROOT / "outputs/carnet/spcarnet/object_index_v1.json"))
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--mc_resolution", type=int, default=32)
    parser.add_argument("--sample_count", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fit_missing_latents",
        action="store_true",
        help="MAP-fit z for objects not present in the train-time latent table; reports latent_source=heldout_map_fit.",
    )
    parser.add_argument(
        "--fit_all_latents",
        action="store_true",
        help="Also refine train-table latents before extraction. This is an ablation, not default train eval.",
    )
    parser.add_argument("--fit_steps", type=int, default=100)
    parser.add_argument("--fit_lr", type=float, default=1e-2)
    parser.add_argument("--fit_queries_surface", type=int, default=384)
    parser.add_argument("--fit_queries_free", type=int, default=384)
    parser.add_argument("--fit_queries_hard", type=int, default=128)
    parser.add_argument("--fit_queries_mixed", type=int, default=128)
    parser.add_argument("--fit_queries_band", type=int, default=0)
    parser.add_argument("--fit_band_epsilon", type=float, default=0.02)
    parser.add_argument("--fit_seed", type=int, default=0)
    parser.add_argument(
        "--resolved_config",
        default=None,
        help="Optional resolved_config.json used to load training loss weights for z-only MAP fitting.",
    )
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument(
        "--manifest",
        default=str(
            REPO_ROOT / "outputs/ss3dm_prior_car/meshfleet_car_cache_v5/source_mesh_manifest.json"
        ),
        help="source_mesh_manifest.json from the Stage-1 cache; used to resolve GLB paths for the IoU GT volume.",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    decoder_state = state["decoder_state_dict"]
    model_cfg = state.get("model_cfg", {})
    decoder = SPCarShapeFieldDecoder(
        latent_dim=int(model_cfg.get("latent_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 384)),
        depth=int(model_cfg.get("depth", 6)),
        num_fourier_freqs=int(model_cfg.get("num_fourier_freqs", 32)),
        field_kind=str(model_cfg.get("field_kind", "occupancy")),
        feature_dim=int(model_cfg.get("feature_dim", 0)),
    ).to(device)
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    latent_table = state["latent_table"].to(device)
    id_to_row = state["object_id_to_row"]

    dataset = SPCarObjectDataset(args.object_index, splits=tuple(args.splits))
    manifest = _load_manifest(Path(args.manifest)) if args.manifest else {"dataset_root": None, "by_id": {}}
    loss_cfg = _load_resolved_loss_config(args.resolved_config)
    out = evaluate(
        decoder=decoder,
        latent_table=latent_table,
        object_id_to_row=id_to_row,
        dataset=dataset,
        device=device,
        manifest=manifest,
        mc_resolution=args.mc_resolution,
        sample_count=args.sample_count,
        limit=args.limit,
        fit_missing_latents=bool(args.fit_missing_latents),
        fit_all_latents=bool(args.fit_all_latents),
        loss_cfg=loss_cfg,
        fit_steps=int(args.fit_steps),
        fit_lr=float(args.fit_lr),
        fit_queries_surface=int(args.fit_queries_surface),
        fit_queries_free=int(args.fit_queries_free),
        fit_queries_hard=int(args.fit_queries_hard),
        fit_queries_mixed=int(args.fit_queries_mixed),
        fit_queries_band=int(args.fit_queries_band),
        fit_band_epsilon=float(args.fit_band_epsilon),
        fit_seed=int(args.fit_seed),
    )
    target = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".eval.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    out = _json_sanitize(out)
    if args.wandb_project:
        if args.wandb_mode:
            os.environ["WANDB_MODE"] = args.wandb_mode
        try:
            import wandb  # type: ignore

            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                dir=str(target.parent),
                config={
                    "checkpoint": str(args.checkpoint),
                    "object_index": str(args.object_index),
                    "splits": list(args.splits),
                    "mc_resolution": int(args.mc_resolution),
                    "sample_count": int(args.sample_count),
                    "limit": int(args.limit),
                    "fit_missing_latents": bool(args.fit_missing_latents),
                    "fit_all_latents": bool(args.fit_all_latents),
                    "fit_steps": int(args.fit_steps),
                    "fit_lr": float(args.fit_lr),
                    "fit_queries_surface": int(args.fit_queries_surface),
                    "fit_queries_free": int(args.fit_queries_free),
                    "fit_queries_hard": int(args.fit_queries_hard),
                    "fit_queries_mixed": int(args.fit_queries_mixed),
                    "fit_queries_band": int(args.fit_queries_band),
                    "fit_band_epsilon": float(args.fit_band_epsilon),
                    "output": str(target),
                },
                reinit=True,
            )
            summary_metrics = {
                f"eval/{k}": v
                for k, v in out["summary"].items()
                if isinstance(v, (int, float, bool))
            }
            wandb.log(summary_metrics)
            run.summary.update(summary_metrics)
            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[stage2-eval] wandb logging failed: {type(exc).__name__}: {exc}", flush=True)
    with target.open("w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps(out["summary"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

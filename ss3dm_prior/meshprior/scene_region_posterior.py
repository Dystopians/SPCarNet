"""Posterior inference utilities for mined scene mesh regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ss3dm_prior.mesh.marching_cubes import extract_patch_mesh
from ss3dm_prior.models.spcarnet_posterior import SPCarPosteriorCompletionModel, SPCarPosteriorEncoder
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder


def load_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    try:
        import trimesh

        mesh = trimesh.load(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate([g for g in mesh.dump() if isinstance(g, trimesh.Trimesh)])
        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
    except Exception:
        return _load_ascii_ply(path)


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        header = []
        for line in f:
            header.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
        vertex_count = 0
        face_count = 0
        for line in header:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif len(parts) == 3 and parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
        vertices = [[float(x) for x in f.readline().split()[:3]] for _ in range(vertex_count)]
        faces = []
        for _ in range(face_count):
            parts = f.readline().split()
            if int(parts[0]) == 3:
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def sample_region_points(
    mesh: tuple[np.ndarray, np.ndarray],
    region: dict[str, Any],
    n_points: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    vertices, faces = mesh
    face_indices = np.asarray(region.get("face_indices", []), dtype=np.int64)
    if face_indices.size == 0:
        raise ValueError(f"region {region.get('region_id')} has no face_indices")
    region_faces = faces[face_indices]
    tri = vertices[region_faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    if float(areas.sum()) <= 1e-12:
        unique = np.unique(region_faces.reshape(-1))
        pts = vertices[unique]
        idx = rng.choice(len(pts), size=n_points, replace=len(pts) < n_points)
        return pts[idx].astype(np.float32)
    probs = areas / areas.sum()
    chosen = rng.choice(len(region_faces), size=n_points, replace=True, p=probs)
    selected = tri[chosen]
    u = rng.random((n_points, 1))
    v = rng.random((n_points, 1))
    swap = (u + v) > 1.0
    u[swap] = 1.0 - u[swap]
    v[swap] = 1.0 - v[swap]
    pts = selected[:, 0] + u * (selected[:, 1] - selected[:, 0]) + v * (selected[:, 2] - selected[:, 0])
    return pts.astype(np.float32)


def estimate_canonical_transform(points: np.ndarray, mode: str = "bbox_pca") -> dict[str, Any]:
    if mode != "bbox_pca":
        raise ValueError(f"unsupported canonical transform mode: {mode}")
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = centered.T @ centered / max(len(pts) - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    rotation = eigvecs[:, ::-1]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    rotated = centered @ rotation
    radius = float(max(np.linalg.norm(rotated, axis=1).max(), 1e-6))
    ext = rotated.max(axis=0) - rotated.min(axis=0)
    anisotropy = float(ext.max() / max(ext.min(), 1e-6))
    confidence = float(min(1.0, max(0.0, (anisotropy - 1.0) / 3.0)))
    return {
        "mode": mode,
        "center": center.astype(float).tolist(),
        "scale": radius,
        "rotation": rotation.astype(float).tolist(),
        "confidence": confidence,
        "notes": ["front_axis_unknown", "pca_sign_ambiguous"],
    }


def canonicalize_region_points(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    center = np.asarray(transform["center"], dtype=np.float64)
    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    scale = max(float(transform["scale"]), 1e-6)
    return ((pts - center) @ rotation / scale).astype(np.float32)


def load_spcarnet_completion_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> SPCarPosteriorCompletionModel:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-3 posterior checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    required = {"encoder_state_dict", "decoder_state_dict", "model_cfg"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(f"Unsupported posterior checkpoint schema; missing keys: {missing}")
    model_cfg = dict(payload["model_cfg"])
    decoder_checkpoint = Path(str(model_cfg.get("decoder_checkpoint", "")))
    if not decoder_checkpoint.is_file():
        raise FileNotFoundError(f"Stage-2 decoder checkpoint referenced by posterior is missing: {decoder_checkpoint}")
    decoder_payload = torch.load(decoder_checkpoint, map_location="cpu")
    decoder_cfg = dict(decoder_payload.get("model_cfg", {}))
    decoder_cfg.pop("latent_init_std", None)
    encoder = SPCarPosteriorEncoder(
        latent_dim=int(model_cfg.get("latent_dim", 256)),
        feature_dim=int(model_cfg.get("encoder_feature_dim", model_cfg.get("feature_dim", 256))),
        num_xattn_layers=int(model_cfg.get("num_xattn_layers", 4)),
        num_self_attn_layers=int(model_cfg.get("num_self_attn_layers", 2)),
        num_latent_queries=int(model_cfg.get("num_latent_queries", 32)),
        attention_heads=int(model_cfg.get("attention_heads", 8)),
        ffn_dim=int(model_cfg.get("ffn_dim", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        posterior_kind=str(model_cfg.get("posterior_kind", "variational")),
        use_normals=bool(model_cfg.get("use_normals", False)),
        use_conditioning_adapter=bool(model_cfg.get("use_conditioning_adapter", True)),
    )
    decoder = SPCarShapeFieldDecoder(**decoder_cfg)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    model = SPCarPosteriorCompletionModel(
        encoder=encoder,
        decoder=decoder,
        decoder_finetune_enabled=bool(payload.get("decoder_finetune_enabled", False)),
        decoder_finetune_tail_blocks=int(model_cfg.get("decoder_finetune_tail_blocks", 2)),
    )
    model.to(torch.device(device))
    model.eval()
    return model


@torch.no_grad()
def run_spcarnet_posterior(
    checkpoint: str | Path,
    points: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[SPCarPosteriorCompletionModel, dict[str, torch.Tensor | None]]:
    model = load_spcarnet_completion_model(checkpoint, device=device)
    partial = torch.from_numpy(np.asarray(points, dtype=np.float32))[None].to(torch.device(device))
    post = model.encode(partial, sample=False)
    return model, {"z_mean": post.z_mean, "z_logvar": post.z_logvar, "z": post.z}


@torch.no_grad()
def decode_region_field(
    decoder: SPCarShapeFieldDecoder,
    z: torch.Tensor,
    query_points: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 65536,
) -> torch.Tensor:
    dev = torch.device(device)
    q = torch.as_tensor(query_points, dtype=torch.float32, device=dev)
    if q.ndim == 2:
        q = q.unsqueeze(0)
    outs = []
    for start in range(0, q.shape[1], chunk_size):
        outs.append(decoder(q[:, start : start + chunk_size], z.to(dev)))
    return torch.cat(outs, dim=1)


def estimate_posterior_uncertainty(
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor | None,
    K: int = 8,
) -> dict[str, float]:
    mu = z_mean.detach().float().cpu()
    out = {"posterior_mu_norm": float(torch.linalg.norm(mu, dim=-1).mean().item())}
    if z_logvar is None:
        out.update(
            {
                "posterior_logvar_mean": float("nan"),
                "posterior_variance_mean": 0.0,
                "uncertainty_score": 0.0,
                "latent_sample_l2_mean": 0.0,
            }
        )
        return out
    logvar = z_logvar.detach().float().cpu()
    var = torch.exp(logvar)
    std = torch.sqrt(var)
    samples = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(max(1, int(K))):
        samples.append(mu + std * torch.randn(std.shape, generator=gen))
    stack = torch.stack(samples, dim=0)
    spread = torch.linalg.norm(stack - mu.unsqueeze(0), dim=-1).mean()
    out.update(
        {
            "posterior_logvar_mean": float(logvar.mean().item()),
            "posterior_variance_mean": float(var.mean().item()),
            "uncertainty_score": float(torch.sqrt(var.mean()).item()),
            "latent_sample_l2_mean": float(spread.item()),
        }
    )
    return out


def make_grid(resolution: int = 32, bounds: float = 1.05) -> np.ndarray:
    axis = np.linspace(-bounds, bounds, int(resolution), dtype=np.float32)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

"""Loss functions for local patch denoising and quality learning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def pairwise_cdist_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=1)


def chamfer_l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = pairwise_cdist_l1(x, y)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def chamfer_l1_loss_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = pairwise_cdist_l1(x, y)
    return dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)


def recon_normal_loss(
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    clean_points: torch.Tensor,
    clean_normals: torch.Tensor,
) -> torch.Tensor:
    idx = torch.cdist(recon_points, clean_points, p=2).argmin(dim=2)
    matched_clean_normals = torch.gather(
        clean_normals,
        1,
        idx[..., None].expand(-1, -1, clean_normals.shape[-1]),
    )
    cosine = F.cosine_similarity(recon_normals, matched_clean_normals, dim=-1)
    return (1.0 - cosine).mean()


def recon_normal_loss_per_sample(
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    clean_points: torch.Tensor,
    clean_normals: torch.Tensor,
) -> torch.Tensor:
    idx = torch.cdist(recon_points, clean_points, p=2).argmin(dim=2)
    matched_clean_normals = torch.gather(
        clean_normals,
        1,
        idx[..., None].expand(-1, -1, clean_normals.shape[-1]),
    )
    cosine = F.cosine_similarity(recon_normals, matched_clean_normals, dim=-1)
    return (1.0 - cosine).mean(dim=1)


def latent_align_loss(corrupted_latent: torch.Tensor, clean_latent: torch.Tensor | None) -> torch.Tensor:
    if clean_latent is None:
        return corrupted_latent.new_zeros(())
    cosine = F.cosine_similarity(
        F.normalize(corrupted_latent, dim=-1),
        F.normalize(clean_latent, dim=-1),
        dim=-1,
    )
    return (1.0 - cosine).mean()


def retrieval_align_loss(
    retrieval_embedding: torch.Tensor,
    clean_retrieval_embedding: torch.Tensor | None,
) -> torch.Tensor:
    if clean_retrieval_embedding is None:
        return retrieval_embedding.new_zeros(())
    logits = torch.matmul(
        F.normalize(retrieval_embedding, dim=-1),
        F.normalize(clean_retrieval_embedding, dim=-1).transpose(0, 1),
    )
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def compute_patch_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor | Any],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or {}
    recon_chamfer_per_sample = chamfer_l1_loss_per_sample(outputs["recon_points"], batch["clean_points"])
    recon_normal_per_sample = recon_normal_loss_per_sample(
        outputs["recon_points"],
        outputs["recon_normals"],
        batch["clean_points"],
        batch["clean_normals"],
    )
    hard_alpha = float(weights.get("hard_example_reweight", 0.0))
    if hard_alpha > 0.0:
        severity = torch.log1p(batch["corruption_score_target"]).detach()
        severity = severity / torch.clamp(severity.mean(), min=1e-6)
        sample_weights = 1.0 + hard_alpha * severity
        sample_weights = sample_weights / torch.clamp(sample_weights.mean(), min=1e-6)
    else:
        sample_weights = recon_chamfer_per_sample.new_ones(recon_chamfer_per_sample.shape)
    recon_chamfer = (recon_chamfer_per_sample * sample_weights).mean()
    recon_normal = (recon_normal_per_sample * sample_weights).mean()
    defect_target = torch.log1p(batch["point_defect_target"])
    score_target = torch.log1p(batch["corruption_score_target"])
    point_defect = F.mse_loss(outputs["point_defect_pred"], defect_target)
    patch_score = F.mse_loss(outputs["patch_score_pred"], score_target)
    latent_align = latent_align_loss(outputs["fused_latent"], outputs["clean_fused_latent"])
    retrieval_align = retrieval_align_loss(outputs["retrieval_embedding"], outputs["clean_retrieval_embedding"])

    total = (
        float(weights.get("recon_chamfer_loss", 1.0)) * recon_chamfer
        + float(weights.get("recon_normal_loss", 0.5)) * recon_normal
        + float(weights.get("point_defect_loss", 1.0)) * point_defect
        + float(weights.get("patch_score_loss", 0.5)) * patch_score
        + float(weights.get("latent_align_loss", 0.25)) * latent_align
        + float(weights.get("retrieval_align_loss", 0.0)) * retrieval_align
    )
    return {
        "total_loss": total,
        "recon_chamfer_loss": recon_chamfer,
        "recon_normal_loss": recon_normal,
        "point_defect_loss": point_defect,
        "patch_score_loss": patch_score,
        "latent_align_loss": latent_align,
        "retrieval_align_loss": retrieval_align,
    }

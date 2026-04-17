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


def intrinsic_difficulty_pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() < 2:
        return pred.new_zeros(())
    target_diff = target[:, None] - target[None, :]
    pred_diff = pred[:, None] - pred[None, :]
    direction = torch.sign(target_diff)
    pair_mask = torch.triu(torch.abs(target_diff) > 1e-6, diagonal=1)
    if not bool(pair_mask.any()):
        return pred.new_zeros(())
    signed_margin = margin - direction * pred_diff
    return F.relu(signed_margin[pair_mask]).mean()


def occupancy_bce_loss(
    query_occupancy_logits: torch.Tensor | None,
    query_labels_all: torch.Tensor | None,
    query_ignore_mask: torch.Tensor | None,
) -> torch.Tensor:
    if query_occupancy_logits is None or query_labels_all is None:
        device = query_occupancy_logits.device if query_occupancy_logits is not None else "cpu"
        return torch.zeros((), device=device)
    labels = query_labels_all.float()
    if query_ignore_mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        valid_mask = ~query_ignore_mask.bool()
    if not bool(valid_mask.any()):
        return query_occupancy_logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(
        query_occupancy_logits[valid_mask],
        labels[valid_mask],
    )


def free_space_violation_loss(
    query_occupancy_logits: torch.Tensor | None,
    query_labels_all: torch.Tensor | None,
    query_ignore_mask: torch.Tensor | None,
) -> torch.Tensor:
    if query_occupancy_logits is None or query_labels_all is None:
        device = query_occupancy_logits.device if query_occupancy_logits is not None else "cpu"
        return torch.zeros((), device=device)
    labels = query_labels_all.float()
    free_mask = labels <= 0.5
    if query_ignore_mask is not None:
        free_mask &= ~query_ignore_mask.bool()
    if not bool(free_mask.any()):
        return query_occupancy_logits.new_zeros(())
    return F.softplus(query_occupancy_logits[free_mask]).mean()


def prototype_diversity_loss(
    code_indices: torch.Tensor | None,
    *,
    codebook_size: int | None,
) -> torch.Tensor:
    if code_indices is None:
        return torch.zeros(())
    indices = code_indices.reshape(-1)
    if indices.numel() == 0 or codebook_size is None or codebook_size <= 1:
        return indices.new_zeros((), dtype=torch.float32)
    usage = torch.bincount(indices, minlength=int(codebook_size)).float()
    usage_probs = usage / torch.clamp(usage.sum(), min=1.0)
    entropy = -torch.sum(usage_probs * torch.log(torch.clamp(usage_probs, min=1e-8)))
    max_entropy = torch.log(torch.tensor(float(codebook_size), device=entropy.device))
    return 1.0 - entropy / torch.clamp(max_entropy, min=1e-8)


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
    corruption_score = F.mse_loss(outputs["patch_score_pred"], score_target)
    latent_align = latent_align_loss(outputs["fused_latent"], outputs["clean_fused_latent"])
    retrieval_align = retrieval_align_loss(outputs["retrieval_embedding"], outputs["clean_retrieval_embedding"])
    intrinsic_difficulty_target = batch.get("intrinsic_patch_difficulty_target")
    intrinsic_difficulty_pred = outputs.get("intrinsic_difficulty_pred")
    if intrinsic_difficulty_target is not None and intrinsic_difficulty_pred is not None:
        intrinsic_difficulty_regression = F.mse_loss(
            intrinsic_difficulty_pred,
            intrinsic_difficulty_target.float().reshape_as(intrinsic_difficulty_pred),
        )
        intrinsic_difficulty_pairwise = intrinsic_difficulty_pairwise_ranking_loss(
            intrinsic_difficulty_pred,
            intrinsic_difficulty_target.float(),
            margin=float(weights.get("intrinsic_difficulty_pairwise_margin", 0.05)),
        )
        intrinsic_difficulty = intrinsic_difficulty_regression + float(
            weights.get("intrinsic_difficulty_pairwise_weight", 0.5)
        ) * intrinsic_difficulty_pairwise
    else:
        intrinsic_difficulty = recon_chamfer.new_zeros(())
        intrinsic_difficulty_pairwise = recon_chamfer.new_zeros(())
    occupancy_bce = occupancy_bce_loss(
        outputs.get("query_occupancy_logits"),
        batch.get("query_labels_all"),
        batch.get("query_ignore_mask"),
    ).to(recon_chamfer.device)
    free_space_violation = free_space_violation_loss(
        outputs.get("query_occupancy_logits"),
        batch.get("query_labels_all"),
        batch.get("query_ignore_mask"),
    ).to(recon_chamfer.device)
    vq_commitment = outputs.get("vq_commitment_loss")
    if vq_commitment is None:
        vq_commitment = recon_chamfer.new_zeros(())
    codebook_stats = outputs.get("codebook_stats")
    codebook_size = None
    if isinstance(codebook_stats, dict):
        raw_codebook_size = codebook_stats.get("codebook_size")
        if raw_codebook_size is not None:
            codebook_size = int(raw_codebook_size)
    diversity = prototype_diversity_loss(outputs.get("code_indices"), codebook_size=codebook_size).to(recon_chamfer.device)

    corruption_score_weight = float(
        weights.get("corruption_score_loss", weights.get("patch_score_loss", 0.5))
    )

    total = (
        float(weights.get("recon_chamfer_loss", 1.0)) * recon_chamfer
        + float(weights.get("recon_normal_loss", 0.5)) * recon_normal
        + float(weights.get("point_defect_loss", 1.0)) * point_defect
        + corruption_score_weight * corruption_score
        + float(weights.get("intrinsic_difficulty_loss", 0.0)) * intrinsic_difficulty
        + float(weights.get("occupancy_bce_loss", 0.0)) * occupancy_bce
        + float(weights.get("free_space_violation_loss", 0.0)) * free_space_violation
        + float(weights.get("vq_commitment_loss", 0.0)) * vq_commitment
        + float(weights.get("prototype_diversity_loss", 0.0)) * diversity
        + float(weights.get("latent_align_loss", 0.25)) * latent_align
        + float(weights.get("retrieval_align_loss", 0.0)) * retrieval_align
    )
    return {
        "total_loss": total,
        "recon_chamfer_loss": recon_chamfer,
        "recon_normal_loss": recon_normal,
        "point_defect_loss": point_defect,
        "patch_score_loss": corruption_score,
        "corruption_score_loss": corruption_score,
        "intrinsic_difficulty_loss": intrinsic_difficulty,
        "intrinsic_difficulty_pairwise_loss": intrinsic_difficulty_pairwise,
        "occupancy_bce_loss": occupancy_bce,
        "free_space_violation_loss": free_space_violation,
        "vq_commitment_loss": vq_commitment,
        "prototype_diversity_loss": diversity,
        "latent_align_loss": latent_align,
        "retrieval_align_loss": retrieval_align,
    }

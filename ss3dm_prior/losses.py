"""Loss functions for local patch denoising and quality learning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ss3dm_prior.models.symmetry_head import reflect_points


def _cdist_fp32_safe(x: torch.Tensor, y: torch.Tensor, *, p: float) -> torch.Tensor:
    """``torch.cdist`` wrapper that forces fp32 CUDA compute.

    ``cdist_cuda`` has no Half kernel. When called inside the trainer's AMP
    autocast context PyTorch's fp32 promotion rule for cdist covers us, but
    code paths that call these losses outside autocast (e.g. eval) still
    fail. Explicit upcast + ``autocast(enabled=False)`` guarantees correctness
    in both.
    """
    device_type = x.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return torch.cdist(x.float(), y.float(), p=p)


def pairwise_cdist_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _cdist_fp32_safe(x, y, p=1)


def chamfer_l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = pairwise_cdist_l1(x, y)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


def chamfer_l1_loss_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = pairwise_cdist_l1(x, y)
    return dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)


def hidden_completion_chamfer_loss(
    recon_points: torch.Tensor,
    hidden_clean_points: "list | None",
) -> torch.Tensor:
    """Chamfer between the reconstructed cloud and the hidden-only clean
    subset for each sample in the batch. v0.1 kept hidden_completion as a
    metric only, so there was no direct gradient on the occluded side —
    visible points dominate the global Chamfer. This term supplies that
    missing signal.

    Samples whose ``hidden_clean_points[i]`` is empty contribute 0 and are
    excluded from the mean (they have no ground-truth hidden targets).
    Returns a scalar tensor on recon_points' device; may be zero if no
    sample in the batch has hidden targets.
    """
    return _masked_chamfer_against_subset(recon_points, hidden_clean_points)


def visible_recon_chamfer_loss(
    recon_points: torch.Tensor,
    visible_clean_points: "list | None",
) -> torch.Tensor:
    """Chamfer between the reconstructed cloud and the visible-only clean
    subset for each sample. Dual of hidden_completion_chamfer_loss — when
    both are active (with their own weights) the model gets independent
    visible/hidden supervision and can no longer let visible dominate the
    global chamfer's gradient. Added in v0.6 to counter a ~7% normal-cosine
    regression that showed up in v0.4 when per-point losses were boosted
    but visible supervision was not separately reinforced.
    """
    return _masked_chamfer_against_subset(recon_points, visible_clean_points)


def _masked_chamfer_against_subset(
    recon_points: torch.Tensor,
    subset_points: "list | None",
) -> torch.Tensor:
    zero = recon_points.new_zeros(())
    if subset_points is None:
        return zero
    losses: list[torch.Tensor] = []
    for i in range(recon_points.shape[0]):
        if i >= len(subset_points):
            break
        sp = subset_points[i]
        if not isinstance(sp, torch.Tensor) or sp.ndim != 2 or sp.shape[0] < 1 or sp.shape[-1] != 3:
            continue
        sp = sp.to(recon_points.device, dtype=recon_points.dtype).unsqueeze(0)
        losses.append(chamfer_l1_loss(recon_points[i : i + 1], sp))
    if not losses:
        return zero
    return torch.stack(losses).mean()


def recon_normal_loss(
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    clean_points: torch.Tensor,
    clean_normals: torch.Tensor,
) -> torch.Tensor:
    idx = _cdist_fp32_safe(recon_points, clean_points, p=2).argmin(dim=2)
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
    idx = _cdist_fp32_safe(recon_points, clean_points, p=2).argmin(dim=2)
    matched_clean_normals = torch.gather(
        clean_normals,
        1,
        idx[..., None].expand(-1, -1, clean_normals.shape[-1]),
    )
    cosine = F.cosine_similarity(recon_normals, matched_clean_normals, dim=-1)
    return (1.0 - cosine).mean(dim=1)


def nearest_neighbor_l1_loss_per_sample(
    recon_points: torch.Tensor,
    corrupted_points: torch.Tensor,
    clean_points: torch.Tensor,
) -> torch.Tensor:
    """Per-point supervised L1 regression against a fixed teacher target.

    For every corrupted input point the teacher target is its nearest
    neighbour in the clean point set; the model's ``recon_points`` (which
    under ``use_residual_reconstruction`` start life at ``corrupted_points``)
    are pushed toward that target by an L1 loss.

    This **direct** supervision complements the symmetric Chamfer loss
    (which has ``recon = corrupted`` as a local minimum and therefore
    allows identity-collapse). NN-L1 penalises identity maximally because
    the teacher target is the clean-side nearest neighbour, not the
    corrupted-side one.

    The nearest-neighbour index is computed on the *corrupted* inputs and
    detached so the loss only teaches the reconstruction head, not the
    target assignment.

    Returns
    -------
    (B,) float tensor — per-sample mean L1 distance to the teacher target.
    """
    if recon_points.shape[1] != corrupted_points.shape[1]:
        # The residual head only emits one reconstruction per corrupted point,
        # so this should never trigger; if the model switches to a global
        # decoder with N != M outputs, fall back to a safe zero.
        return recon_points.new_zeros(recon_points.shape[0])
    with torch.no_grad():
        nn_idx = _cdist_fp32_safe(corrupted_points, clean_points, p=2).argmin(dim=2)
        targets = torch.gather(
            clean_points,
            1,
            nn_idx[..., None].expand(-1, -1, clean_points.shape[-1]),
        )
    return (recon_points - targets).abs().sum(dim=-1).mean(dim=1)


def nearest_neighbor_l1_loss(
    recon_points: torch.Tensor,
    corrupted_points: torch.Tensor,
    clean_points: torch.Tensor,
) -> torch.Tensor:
    return nearest_neighbor_l1_loss_per_sample(recon_points, corrupted_points, clean_points).mean()


def reverse_nearest_neighbor_l1_loss_per_sample(
    recon_points: torch.Tensor,
    clean_points: torch.Tensor,
) -> torch.Tensor:
    """Clean-side coverage supervision.

    For every clean ground-truth point, find the nearest recon point and
    penalise the L1 distance between them. This is the dual of the forward
    :func:`nearest_neighbor_l1_loss_per_sample` and forces the reconstruction
    to **cover** the clean distribution rather than collapsing many recon
    points onto a small number of clean targets (a failure mode of the
    forward-only supervision observed in v0.1 / 5-step diagnosis).

    The nearest-recon index is detached so the gradient flows back only
    through the selected recon tensor, not the assignment.
    """
    if clean_points.shape[1] == 0:
        return recon_points.new_zeros(recon_points.shape[0])
    with torch.no_grad():
        nn_idx = _cdist_fp32_safe(clean_points, recon_points, p=2).argmin(dim=2)
        selected = torch.gather(
            recon_points,
            1,
            nn_idx[..., None].expand(-1, -1, recon_points.shape[-1]),
        )
    return (selected - clean_points).abs().sum(dim=-1).mean(dim=1)


def reverse_nearest_neighbor_l1_loss(
    recon_points: torch.Tensor,
    clean_points: torch.Tensor,
) -> torch.Tensor:
    return reverse_nearest_neighbor_l1_loss_per_sample(recon_points, clean_points).mean()


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


def _normalize_hard_weight_signal(signal: torch.Tensor) -> torch.Tensor:
    normalized = torch.clamp(signal.float(), min=0.0)
    return normalized / torch.clamp(normalized.mean(), min=1e-6)


def compute_hard_example_sample_weights(
    batch: dict[str, torch.Tensor | Any],
    weights: dict[str, float] | None = None,
    *,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = weights or {}
    hard_alpha = float(weights.get("hard_example_reweight", 0.0))
    if reference is None:
        reference = batch["clean_points"]
    sample_count = int(reference.shape[0])
    base_weights = reference.new_ones((sample_count,), dtype=torch.float32)
    if hard_alpha <= 0.0:
        return base_weights

    source = str(weights.get("hard_weight_source", "blend")).strip().lower()
    blend_alpha = float(weights.get("hard_weight_blend_alpha", 0.5))
    blend_alpha = min(max(blend_alpha, 0.0), 1.0)

    corruption_target = batch.get("corruption_score_target")
    intrinsic_target = batch.get("intrinsic_patch_difficulty_target")
    corruption_signal = (
        _normalize_hard_weight_signal(torch.log1p(corruption_target.detach()))
        if corruption_target is not None
        else None
    )
    intrinsic_signal = (
        _normalize_hard_weight_signal(intrinsic_target.detach())
        if intrinsic_target is not None
        else None
    )

    if source == "corruption":
        hard_signal = corruption_signal
    elif source == "intrinsic":
        hard_signal = intrinsic_signal if intrinsic_signal is not None else corruption_signal
    elif source == "blend":
        if corruption_signal is not None and intrinsic_signal is not None:
            hard_signal = (1.0 - blend_alpha) * corruption_signal + blend_alpha * intrinsic_signal
        else:
            hard_signal = intrinsic_signal if intrinsic_signal is not None else corruption_signal
    else:
        raise ValueError(f"Unsupported hard_weight_source: {source}")

    if hard_signal is None:
        return base_weights

    sample_weights = 1.0 + hard_alpha * hard_signal
    return sample_weights / torch.clamp(sample_weights.mean(), min=1e-6)


def symmetry_consistency_loss(
    recon_points: torch.Tensor,
    plane_normal_pred: torch.Tensor,
    plane_offset_pred: torch.Tensor,
    confidence_pred: torch.Tensor,
    plane_normal_target: torch.Tensor,
    plane_offset_target: torch.Tensor,
    confidence_target: torch.Tensor,
    *,
    plane_weight: float = 0.5,
    confidence_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    """CarNet_v0 / Phase 2 / A2 symmetry loss.

    Three components:
      1. ``L_sym_recon`` — σ · Chamfer(recon, reflect(recon, n_pred, d_pred))
         encourages the model's reconstruction to be self-symmetric about the
         plane it predicts, but only where confidence is high.
      2. ``L_plane`` — supervised alignment of the predicted plane with the
         precomputed GT plane, weighted by confidence target so asymmetric
         / partial patches don't force a bogus plane.
      3. ``L_conf`` — BCE between predicted σ and the GT σ (which itself
         is a soft [0, 1] target derived from the GT Chamfer residual).

    All returned tensors are scalars on the same device/dtype as
    ``recon_points``.
    """
    # Component 1: self-symmetry reconstruction consistency.
    reflected = reflect_points(recon_points, plane_normal_pred, plane_offset_pred)
    # per-sample bidirectional chamfer
    recon_sym_chamfer = chamfer_l1_loss_per_sample(recon_points, reflected)
    conf_for_recon = confidence_pred.detach().clamp(0.0, 1.0)
    sym_recon = (conf_for_recon * recon_sym_chamfer).mean()

    # Component 2: plane regression. Use a sign-invariant cosine loss on the
    # normal (a plane and its flipped normal describe the same reflection).
    n_cos = F.cosine_similarity(plane_normal_pred, plane_normal_target, dim=-1)
    n_loss = (1.0 - torch.abs(n_cos))  # (B,)
    # Offset: if we flipped the sign of n_target, flip d_target too.
    sign = torch.sign(torch.sum(plane_normal_pred * plane_normal_target, dim=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    d_loss = torch.abs(plane_offset_pred - sign * plane_offset_target)
    per_sample_plane = n_loss + 0.5 * d_loss
    plane_weight_mask = confidence_target.clamp(0.0, 1.0)
    plane_loss = (plane_weight_mask * per_sample_plane).mean()

    # Component 3: confidence BCE (σ is already sigmoided).
    eps = 1e-6
    conf_clamped = confidence_pred.clamp(eps, 1.0 - eps)
    conf_target = confidence_target.clamp(0.0, 1.0)
    conf_loss = -(
        conf_target * torch.log(conf_clamped)
        + (1.0 - conf_target) * torch.log(1.0 - conf_clamped)
    ).mean()

    total = sym_recon + float(plane_weight) * plane_loss + float(confidence_weight) * conf_loss
    return {
        "symmetry_consistency_loss": total,
        "symmetry_recon_component": sym_recon.detach(),
        "symmetry_plane_component": plane_loss.detach(),
        "symmetry_confidence_component": conf_loss.detach(),
    }


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
    # Nearest-neighbour L1 supervision (CarNet_v0.1 / identity-collapse fix).
    # For the residual-reconstruction head the model output starts at the
    # corrupted input; Chamfer alone admits ``delta = 0`` as a local minimum
    # because the corrupted point cloud is (on average) already close to
    # clean. Pairing Chamfer with a *direct* per-point regression to the
    # corrupted→clean nearest-neighbour target gives every output point a
    # concrete destination to move to, breaking the identity shortcut.
    nn_l1_per_sample = nearest_neighbor_l1_loss_per_sample(
        outputs["recon_points"],
        batch["corrupted_points"],
        batch["clean_points"],
    )
    # Reverse NN-L1 (clean-side coverage) — dual of the forward NN-L1.
    # Without this, the forward-only supervision admits a cluster-collapse
    # solution where many recon points pile onto a small subset of clean
    # targets: forward loss stays small but the clean distribution is
    # uncovered. Adding this symmetric term fixes the coverage side.
    reverse_nn_l1_per_sample = reverse_nearest_neighbor_l1_loss_per_sample(
        outputs["recon_points"],
        batch["clean_points"],
    )
    sample_weights = compute_hard_example_sample_weights(
        batch,
        weights,
        reference=recon_chamfer_per_sample,
    )
    recon_chamfer = (recon_chamfer_per_sample * sample_weights).mean()
    recon_normal = (recon_normal_per_sample * sample_weights).mean()
    nearest_neighbor = (nn_l1_per_sample * sample_weights).mean()
    reverse_nearest_neighbor = (reverse_nn_l1_per_sample * sample_weights).mean()
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
    hidden_completion_chamfer = hidden_completion_chamfer_loss(
        outputs["recon_points"],
        batch.get("hidden_clean_points"),
    )
    visible_recon_chamfer = visible_recon_chamfer_loss(
        outputs["recon_points"],
        batch.get("visible_clean_points"),
    )
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
        + float(weights.get("nearest_neighbor_l1_loss", 0.0)) * nearest_neighbor
        + float(weights.get("reverse_nearest_neighbor_l1_loss", 0.0)) * reverse_nearest_neighbor
        + float(weights.get("point_defect_loss", 1.0)) * point_defect
        + corruption_score_weight * corruption_score
        + float(weights.get("intrinsic_difficulty_loss", 0.0)) * intrinsic_difficulty
        + float(weights.get("occupancy_bce_loss", 0.0)) * occupancy_bce
        + float(weights.get("free_space_violation_loss", 0.0)) * free_space_violation
        + float(weights.get("hidden_completion_chamfer_loss", 0.0)) * hidden_completion_chamfer
        + float(weights.get("visible_recon_chamfer_loss", 0.0)) * visible_recon_chamfer
        + float(weights.get("vq_commitment_loss", 0.0)) * vq_commitment
        + float(weights.get("prototype_diversity_loss", 0.0)) * diversity
        + float(weights.get("latent_align_loss", 0.25)) * latent_align
        + float(weights.get("retrieval_align_loss", 0.0)) * retrieval_align
        + float(weights.get("latent_flow_matching_loss", 0.0))
        * (
            outputs.get("latent_flow_matching_loss")
            if outputs.get("latent_flow_matching_loss") is not None
            else recon_chamfer.new_zeros(())
        )
    )
    latent_flow_matching = outputs.get("latent_flow_matching_loss")
    if latent_flow_matching is None:
        latent_flow_matching = recon_chamfer.new_zeros(())

    # Symmetry-consistency loss (CarNet_v0 / A2). Only active when the model
    # exposes a symmetry head AND the batch carries GT targets (cache format
    # v3+). A missing head or missing targets silently yields zero loss so
    # the older caches / models remain backward-compatible.
    sym_pack: dict[str, torch.Tensor]
    plane_normal_pred = outputs.get("symmetry_plane_normal_pred")
    plane_offset_pred = outputs.get("symmetry_plane_offset_pred")
    confidence_pred = outputs.get("symmetry_confidence_pred")
    plane_normal_target = batch.get("symmetry_plane_normal")
    plane_offset_target = batch.get("symmetry_plane_offset")
    confidence_target = batch.get("symmetry_target_confidence")
    symmetry_available = all(
        t is not None
        for t in (
            plane_normal_pred,
            plane_offset_pred,
            confidence_pred,
            plane_normal_target,
            plane_offset_target,
            confidence_target,
        )
    )
    if symmetry_available:
        sym_pack = symmetry_consistency_loss(
            recon_points=outputs["recon_points"],
            plane_normal_pred=plane_normal_pred,
            plane_offset_pred=plane_offset_pred,
            confidence_pred=confidence_pred,
            plane_normal_target=plane_normal_target.float().to(recon_chamfer.device),
            plane_offset_target=plane_offset_target.float().to(recon_chamfer.device),
            confidence_target=confidence_target.float().to(recon_chamfer.device),
        )
    else:
        zero = recon_chamfer.new_zeros(())
        sym_pack = {
            "symmetry_consistency_loss": zero,
            "symmetry_recon_component": zero,
            "symmetry_plane_component": zero,
            "symmetry_confidence_component": zero,
        }
    sym_weight = float(weights.get("symmetry_consistency_loss", 0.0))
    total = total + sym_weight * sym_pack["symmetry_consistency_loss"]

    return {
        "total_loss": total,
        "recon_chamfer_loss": recon_chamfer,
        "recon_normal_loss": recon_normal,
        "nearest_neighbor_l1_loss": nearest_neighbor,
        "reverse_nearest_neighbor_l1_loss": reverse_nearest_neighbor,
        "point_defect_loss": point_defect,
        "patch_score_loss": corruption_score,
        "corruption_score_loss": corruption_score,
        "intrinsic_difficulty_loss": intrinsic_difficulty,
        "intrinsic_difficulty_pairwise_loss": intrinsic_difficulty_pairwise,
        "occupancy_bce_loss": occupancy_bce,
        "free_space_violation_loss": free_space_violation,
        "hidden_completion_chamfer_loss": hidden_completion_chamfer,
        "visible_recon_chamfer_loss": visible_recon_chamfer,
        "vq_commitment_loss": vq_commitment,
        "prototype_diversity_loss": diversity,
        "latent_align_loss": latent_align,
        "retrieval_align_loss": retrieval_align,
        "latent_flow_matching_loss": latent_flow_matching,
        "symmetry_consistency_loss": sym_pack["symmetry_consistency_loss"],
        "symmetry_recon_component": sym_pack["symmetry_recon_component"],
        "symmetry_plane_component": sym_pack["symmetry_plane_component"],
        "symmetry_confidence_component": sym_pack["symmetry_confidence_component"],
    }

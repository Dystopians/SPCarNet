#!/usr/bin/env python3
"""SP-CarNet Stage 5 — multi-hypothesis sampling & reranking eval.

Sample K candidates per object from ``q(z | O)``, decode K meshes through the
frozen Stage-2 decoder, score each by ``log p(O | f(·;z_k)) + log p(z_k)``,
report top-1 reranked + oracle-best-of-K + diversity.

Constraints (per Stage-5 design §4.2):
- No backprop; all decoder calls under ``torch.no_grad()``.
- Score function consumes only inference-time fields (no clean target points).
- GT-dependent metrics are split into a ``gt_dependent_metrics`` block.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset  # noqa: E402
from ss3dm_prior.mesh.marching_cubes import extract_patch_mesh  # noqa: E402
from ss3dm_prior.models.spcarnet_posterior import (  # noqa: E402
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    d = torch.cdist(a_t, b_t, p=2)
    return float(0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item())


def _build_models(checkpoint: dict, device: torch.device) -> tuple[
    SPCarPosteriorCompletionModel, SPCarShapeFieldDecoder, SPCarPosteriorEncoder
]:
    cfg = checkpoint["model_cfg"]
    decoder = SPCarShapeFieldDecoder(
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=384, depth=6, num_fourier_freqs=32,
        field_kind="occupancy", feature_dim=0,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder = SPCarPosteriorEncoder(
        latent_dim=int(cfg["latent_dim"]),
        feature_dim=int(cfg["encoder_feature_dim"]),
        num_xattn_layers=int(cfg["num_xattn_layers"]),
        num_self_attn_layers=int(cfg["num_self_attn_layers"]),
        num_latent_queries=int(cfg["num_latent_queries"]),
        attention_heads=int(cfg["attention_heads"]),
        ffn_dim=int(cfg["ffn_dim"]),
        dropout=float(cfg["dropout"]),
        posterior_kind=str(cfg["posterior_kind"]),
        use_normals=bool(cfg["use_normals"]),
        use_conditioning_adapter=bool(cfg["use_conditioning_adapter"]),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    completion = SPCarPosteriorCompletionModel(
        encoder=encoder, decoder=decoder, decoder_finetune_enabled=False,
    ).to(device)
    completion.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return completion, decoder, encoder


# ---------------------------------------------------------------------------
# Score function — log p(O | f) + log p(z), no Huber wrap (likelihood form)
# ---------------------------------------------------------------------------


def _score_candidate(
    decoder: SPCarShapeFieldDecoder,
    z: torch.Tensor,
    *,
    observed_points: torch.Tensor,
    free_points: torch.Tensor | None,
    hard_negatives: torch.Tensor | None,
    mixed_points: torch.Tensor | None,
    mixed_labels: torch.Tensor | None,
    mixed_ignore: torch.Tensor | None,
    weights: dict[str, float],
) -> dict[str, float]:
    with torch.no_grad():
        terms: dict[str, float] = {}
        loss_total = 0.0
        # observed surface (target = 1)
        if observed_points.numel() > 0:
            f_obs = decoder(observed_points.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            l_surf = float(F.binary_cross_entropy_with_logits(
                f_obs, torch.ones_like(f_obs)
            ).item())
            terms["loss_surf"] = l_surf
            loss_total += weights["w_surf"] * l_surf
        if free_points is not None and free_points.numel() > 0:
            f_free = decoder(free_points.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            l_free = float(F.binary_cross_entropy_with_logits(
                f_free, torch.zeros_like(f_free)
            ).item())
            terms["loss_free"] = l_free
            loss_total += weights["w_free"] * l_free
        if hard_negatives is not None and hard_negatives.numel() > 0:
            f_hard = decoder(hard_negatives.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            l_hard = float(F.binary_cross_entropy_with_logits(
                f_hard, torch.zeros_like(f_hard)
            ).item())
            terms["loss_hard"] = l_hard
            loss_total += weights["w_free"] * weights.get("alpha_hard", 2.0) * l_hard
        if mixed_points is not None and mixed_labels is not None and mixed_points.numel() > 0:
            f_mix = decoder(mixed_points.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
            target = mixed_labels.float()
            if mixed_ignore is not None:
                keep = ~mixed_ignore
                f_mix = f_mix[keep]
                target = target[keep]
            if f_mix.numel() > 0:
                l_mix = float(F.binary_cross_entropy_with_logits(f_mix, target).item())
                terms["loss_mixed"] = l_mix
                loss_total += weights["w_mixed"] * l_mix
        # log p(z) ≈ -0.5 ||z||²
        log_prior = float(-0.5 * z.pow(2).sum().item())
        terms["log_prior"] = log_prior
        score = -loss_total + log_prior
        terms["score"] = float(score)
        terms["loss_total"] = float(loss_total)
        return terms


# ---------------------------------------------------------------------------
# Per-object multi-hypothesis evaluation
# ---------------------------------------------------------------------------


def _to_tensor(arr: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if arr is None or arr.size == 0:
        return None
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


def _to_label_tensor(arr: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if arr is None or arr.size == 0:
        return None
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


def _to_bool_tensor(arr: np.ndarray | None, device: torch.device) -> torch.Tensor | None:
    if arr is None or arr.size == 0:
        return None
    return torch.from_numpy(np.asarray(arr, dtype=bool)).to(device)


def _extract_mesh_and_sample(
    decoder: SPCarShapeFieldDecoder,
    z: torch.Tensor,
    *,
    device: torch.device,
    mc_resolution: int,
    iso_level: float,
    sample_count: int,
) -> tuple[bool, int, int, np.ndarray]:
    def _occ_fn(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = query.unsqueeze(0).to(device)
            return torch.sigmoid(decoder(q, z.unsqueeze(0))).squeeze(0)

    result = extract_patch_mesh(
        occupancy_fn=_occ_fn, device=device,
        patch_radius=1.0, resolution=mc_resolution, iso_level=iso_level,
    )
    if result.mesh is None or result.face_count == 0:
        return False, int(result.vertex_count), int(result.face_count), np.zeros((0, 3), dtype=np.float32)
    try:
        sampled = np.asarray(result.mesh.sample(sample_count), dtype=np.float32)
    except Exception:
        sampled = np.asarray(result.mesh.vertices, dtype=np.float32)
    return True, int(result.vertex_count), int(result.face_count), sampled


def _evaluate_one(
    *,
    completion: SPCarPosteriorCompletionModel,
    decoder: SPCarShapeFieldDecoder,
    item: dict[str, Any],
    K: int,
    device: torch.device,
    mc_resolution: int,
    sample_count: int,
    iso_level: float,
    weights: dict[str, float],
    free_violation_threshold: float,
    seed_base: int,
) -> dict[str, Any]:
    partial_np = item.get("partial_observed_points")
    if partial_np is None or partial_np.size == 0:
        partial_np = item["clean_points_object"]
    partial_t = _to_tensor(partial_np, device)
    free_t = _to_tensor(item.get("free_space_query_points"), device)
    hard_t = _to_tensor(item.get("free_space_query_hard_negatives"), device)
    qall_t = _to_tensor(item.get("occupancy_query_points"), device)
    qlab_t = _to_label_tensor(item.get("occupancy_query_labels"), device)
    qign_t = _to_bool_tensor(item.get("occupancy_query_ignore"), device)

    started = time.time()

    # ------- mean baseline (sample=False) -------
    with torch.no_grad():
        post_mean = completion.encode(partial_t.unsqueeze(0), sample=False)
        z_mean = post_mean.z_mean.squeeze(0).clone()
    mean_score = _score_candidate(
        decoder, z_mean,
        observed_points=partial_t, free_points=free_t, hard_negatives=hard_t,
        mixed_points=qall_t, mixed_labels=qlab_t, mixed_ignore=qign_t,
        weights=weights,
    )
    mean_extracted, mean_v, mean_f, mean_samples = _extract_mesh_and_sample(
        decoder, z_mean, device=device, mc_resolution=mc_resolution,
        iso_level=iso_level, sample_count=sample_count,
    )

    # ------- K candidates (sample=True with deterministic per-k seeding) -------
    candidates: list[dict[str, Any]] = []
    z_list: list[torch.Tensor] = []
    sampled_list: list[np.ndarray] = []
    for k in range(K):
        torch.manual_seed(seed_base + k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_base + k)
        with torch.no_grad():
            post = completion.encode(partial_t.unsqueeze(0), sample=True)
            z_k = post.z.squeeze(0).clone()
        z_list.append(z_k)
        score_terms = _score_candidate(
            decoder, z_k,
            observed_points=partial_t, free_points=free_t, hard_negatives=hard_t,
            mixed_points=qall_t, mixed_labels=qlab_t, mixed_ignore=qign_t,
            weights=weights,
        )
        ext, v, f, sampled = _extract_mesh_and_sample(
            decoder, z_k, device=device, mc_resolution=mc_resolution,
            iso_level=iso_level, sample_count=sample_count,
        )
        sampled_list.append(sampled)

        # ------- per-candidate metrics -------
        partial_np_clean = np.asarray(partial_np, dtype=np.float32)
        clean = item.get("clean_points_object")
        cand: dict[str, Any] = {
            "k": k,
            "extracted": int(ext),
            "vertex_count": v,
            "face_count": f,
            "score": score_terms["score"],
            "loss_total": score_terms["loss_total"],
            "log_prior": score_terms["log_prior"],
            "z_norm": float(z_k.norm().item()),
        }
        if ext and sampled.shape[0] > 0:
            d_p = torch.cdist(
                torch.from_numpy(partial_np_clean), torch.from_numpy(sampled)
            ).numpy()
            cand["visible_preservation_error"] = float(np.mean(np.min(d_p, axis=1)))
            if free_t is not None:
                with torch.no_grad():
                    f_free = decoder(free_t.unsqueeze(0), z_k.unsqueeze(0)).squeeze(0)
                cand["free_space_violation_rate"] = float(
                    (torch.sigmoid(f_free) > free_violation_threshold).float().mean().item()
                )
            if clean is not None:
                cand["recon_chamfer_l1"] = _chamfer_l1(
                    sampled, np.asarray(clean, dtype=np.float32)
                )
            hidden = item.get("hidden_clean_points")
            if hidden is not None and len(hidden) > 0:
                cand["hidden_chamfer_l1"] = _chamfer_l1(
                    sampled, np.asarray(hidden, dtype=np.float32)
                )
        candidates.append(cand)

    # ------- diversity -------
    div_latent = float("nan")
    if K >= 2:
        z_stack = torch.stack(z_list, dim=0)
        d_lat = torch.cdist(z_stack, z_stack)
        idx = torch.triu_indices(K, K, offset=1)
        if idx.shape[1] > 0:
            div_latent = float(d_lat[idx[0], idx[1]].mean().item())

    div_chamfer_top3 = float("nan")
    extracted_idx = [i for i, c in enumerate(candidates) if c["extracted"] == 1 and sampled_list[i].shape[0] > 0]
    if len(extracted_idx) >= 2:
        scores = [(i, candidates[i]["score"]) for i in extracted_idx]
        scores.sort(key=lambda x: -x[1])
        top3 = [i for i, _ in scores[:3]]
        if len(top3) >= 2:
            ds: list[float] = []
            for a in range(len(top3)):
                for b in range(a + 1, len(top3)):
                    ds.append(_chamfer_l1(sampled_list[top3[a]], sampled_list[top3[b]]))
            ds = [d for d in ds if not math.isnan(d)]
            if ds:
                div_chamfer_top3 = float(np.mean(ds))

    # ------- top-1 reranked + oracle best-of-K -------
    valid = [c for c in candidates if c["extracted"] == 1]
    if valid:
        best_by_score = max(valid, key=lambda c: c["score"])
    else:
        best_by_score = candidates[0]
    valid_with_chamfer = [c for c in valid if "recon_chamfer_l1" in c and not math.isnan(c["recon_chamfer_l1"])]
    if valid_with_chamfer:
        oracle_best = min(valid_with_chamfer, key=lambda c: c["recon_chamfer_l1"])
    else:
        oracle_best = None

    elapsed = time.time() - started
    return {
        "object_id": item.get("object_id"),
        "split": item.get("split"),
        "K": K,
        "elapsed_s": elapsed,
        "mean_baseline": {
            "extracted": int(mean_extracted),
            "score": mean_score["score"],
            "loss_total": mean_score["loss_total"],
            "log_prior": mean_score["log_prior"],
        },
        "candidates": candidates,
        "top1_score": {k: best_by_score.get(k) for k in (
            "k", "score", "recon_chamfer_l1", "hidden_chamfer_l1",
            "visible_preservation_error", "free_space_violation_rate", "extracted"
        )},
        "oracle_best_of_k": (
            {k: oracle_best.get(k) for k in (
                "k", "score", "recon_chamfer_l1", "hidden_chamfer_l1",
                "visible_preservation_error", "free_space_violation_rate", "extracted"
            )}
            if oracle_best is not None else None
        ),
        "diversity_latent_l2": div_latent,
        "diversity_chamfer_top3": div_chamfer_top3,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(per_object: list[dict[str, Any]]) -> dict[str, Any]:
    def _avg(getter) -> float:
        vals = [getter(m) for m in per_object]
        vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    inference_only = {
        "n_objects": len(per_object),
        "K": per_object[0].get("K") if per_object else None,
        "elapsed_s_per_object_mean": _avg(lambda m: m.get("elapsed_s")),
        "diversity_latent_l2_mean": _avg(lambda m: m.get("diversity_latent_l2")),
        "diversity_chamfer_top3_mean": _avg(lambda m: m.get("diversity_chamfer_top3")),
        "mean_baseline_extracted_rate": _avg(lambda m: m.get("mean_baseline", {}).get("extracted")),
        "top1_score_extracted_rate": _avg(lambda m: m.get("top1_score", {}).get("extracted")),
        "top1_score_visible_preservation_error_mean": _avg(
            lambda m: m.get("top1_score", {}).get("visible_preservation_error")
        ),
        "top1_score_free_space_violation_rate_mean": _avg(
            lambda m: m.get("top1_score", {}).get("free_space_violation_rate")
        ),
    }
    gt_dependent = {
        "top1_score_recon_chamfer_l1_mean": _avg(
            lambda m: m.get("top1_score", {}).get("recon_chamfer_l1")
        ),
        "top1_score_hidden_chamfer_l1_mean": _avg(
            lambda m: m.get("top1_score", {}).get("hidden_chamfer_l1")
        ),
        "oracle_best_of_k_recon_chamfer_l1_mean": _avg(
            lambda m: (m.get("oracle_best_of_k") or {}).get("recon_chamfer_l1")
        ),
        "oracle_best_of_k_hidden_chamfer_l1_mean": _avg(
            lambda m: (m.get("oracle_best_of_k") or {}).get("hidden_chamfer_l1")
        ),
    }
    return {
        "inference_only_metrics": inference_only,
        "gt_dependent_metrics": gt_dependent,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior_checkpoint", required=True)
    parser.add_argument("--shape_field_checkpoint", default=None,
                        help="Override decoder weights with a separate Stage-2 ckpt.")
    parser.add_argument("--object_index", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_objects", type=int, default=50)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--mc_resolution", type=int, default=32)
    parser.add_argument("--sample_count", type=int, default=4096)
    parser.add_argument("--iso_level", type=float, default=0.5)
    parser.add_argument("--free_violation_threshold", type=float, default=0.5)
    parser.add_argument("--w_surf", type=float, default=1.0)
    parser.add_argument("--w_free", type=float, default=1.0)
    parser.add_argument("--w_mixed", type=float, default=0.5)
    parser.add_argument("--alpha_hard", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    posterior_ckpt = torch.load(args.posterior_checkpoint, map_location=device)
    completion, decoder, encoder = _build_models(posterior_ckpt, device)
    if args.shape_field_checkpoint:
        sf_ckpt = torch.load(args.shape_field_checkpoint, map_location=device)
        decoder.load_state_dict(sf_ckpt["decoder_state_dict"])
        for p in decoder.parameters():
            p.requires_grad_(False)
    completion.eval()

    weights = {
        "w_surf": args.w_surf, "w_free": args.w_free, "w_mixed": args.w_mixed,
        "alpha_hard": args.alpha_hard,
    }

    dataset = SPCarObjectDataset(args.object_index, splits=(args.split,))
    n_total = len(dataset) if args.num_objects <= 0 else min(args.num_objects, len(dataset))

    per_object: list[dict[str, Any]] = []
    for i in range(n_total):
        item = dataset[i]
        try:
            entry = _evaluate_one(
                completion=completion, decoder=decoder, item=item,
                K=args.K, device=device,
                mc_resolution=args.mc_resolution, sample_count=args.sample_count,
                iso_level=args.iso_level, weights=weights,
                free_violation_threshold=args.free_violation_threshold,
                seed_base=args.seed * 1024 + i * args.K,
            )
        except Exception as exc:
            entry = {
                "object_id": item.get("object_id"),
                "split": item.get("split"),
                "K": args.K,
                "error": f"{type(exc).__name__}: {exc}",
            }
        per_object.append(entry)

    summary = _aggregate(per_object)
    out_doc = {
        "summary": summary,
        "per_object": per_object,
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"K{args.K}.json"
    with target.open("w") as f:
        json.dump(out_doc, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SP-CarNet Stage 4 — test-time MAP latent refinement.

Per-object refinement of the Stage-3 amortised posterior mean ``μ(O)`` toward a
local optimum of the observation-consistency loss, with the Stage-2 decoder
frozen. Reports before/after metrics and writes a JSON.

Constraints (per Stage-4 design §6):
- No backprop through Marching-Cubes (MC is run after refinement under no_grad).
- Refine ``z`` only; decoder ``requires_grad_(False)`` throughout.
- Refinement loss never sees ``clean_points_object`` or ``hidden_clean_points``.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ss3dm_prior.data.spcarnet_object_dataset import SPCarObjectDataset  # noqa: E402
from ss3dm_prior.losses_spcarnet_observation import (  # noqa: E402
    compute_observation_loss,
    free_space_violation_rate,
)
from ss3dm_prior.mesh.marching_cubes import extract_patch_mesh  # noqa: E402
from ss3dm_prior.models.spcarnet_posterior import (  # noqa: E402
    SPCarPosteriorCompletionModel,
    SPCarPosteriorEncoder,
)
from ss3dm_prior.models.spcarnet_shape_field import SPCarShapeFieldDecoder  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: bidirectional chamfer (eval-only)
# ---------------------------------------------------------------------------


def _chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    d = torch.cdist(a_t, b_t, p=2)
    return float(0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item())


# ---------------------------------------------------------------------------
# Build encoder + decoder from a Stage-3 checkpoint
# ---------------------------------------------------------------------------


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
# Per-object refinement
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


def _split_query_holdout(
    points: np.ndarray | None,
    labels: np.ndarray | None,
    ignore: np.ndarray | None,
    *,
    holdout_frac: float,
    rng: np.random.Generator,
) -> tuple[
    tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
]:
    if points is None or labels is None or len(points) == 0 or holdout_frac <= 0:
        return (points, labels, ignore), (None, None, None)
    n = len(points)
    n_hold = int(round(n * holdout_frac))
    if n_hold <= 0 or n_hold >= n:
        return (points, labels, ignore), (None, None, None)
    perm = rng.permutation(n)
    train_idx = perm[n_hold:]
    hold_idx = perm[:n_hold]
    return (
        (points[train_idx], labels[train_idx], ignore[train_idx] if ignore is not None else None),
        (points[hold_idx], labels[hold_idx], ignore[hold_idx] if ignore is not None else None),
    )


def _extract_mesh(decoder, z: torch.Tensor, *, resolution: int, device: torch.device,
                  iso_level: float) -> Any:
    def _occ_fn(query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = query.unsqueeze(0).to(device)
            return torch.sigmoid(decoder(q, z.unsqueeze(0))).squeeze(0)

    return extract_patch_mesh(
        occupancy_fn=_occ_fn, device=device,
        patch_radius=1.0, resolution=resolution, iso_level=iso_level,
    )


def _compute_eval_metrics(
    *,
    decoder,
    z: torch.Tensor,
    item: dict[str, Any],
    free_t: torch.Tensor | None,
    device: torch.device,
    mc_resolution: int,
    sample_count: int,
    iso_level: float,
    free_violation_threshold: float,
) -> dict[str, float]:
    """Returns inference + GT-dependent metrics for one ``z`` snapshot."""
    metrics: dict[str, float] = {}
    if free_t is not None:
        metrics["free_space_violation_rate"] = free_space_violation_rate(
            decoder, z.unsqueeze(0), free_t.unsqueeze(0),
            threshold=free_violation_threshold,
        )
    else:
        metrics["free_space_violation_rate"] = float("nan")

    result = _extract_mesh(decoder, z, resolution=mc_resolution, device=device, iso_level=iso_level)
    metrics["mesh_extraction_success"] = int(
        result.mesh is not None and result.face_count > 0
    )
    if result.mesh is not None and result.face_count > 0:
        try:
            sampled = np.asarray(result.mesh.sample(sample_count), dtype=np.float32)
        except Exception:
            sampled = np.asarray(result.mesh.vertices, dtype=np.float32)
    else:
        sampled = np.zeros((0, 3), dtype=np.float32)

    partial = item.get("partial_observed_points")
    if partial is not None and len(partial) > 0 and sampled.shape[0] > 0:
        d = torch.cdist(
            torch.from_numpy(np.asarray(partial, dtype=np.float32)),
            torch.from_numpy(sampled),
        ).numpy()
        metrics["visible_preservation_error"] = float(np.mean(np.min(d, axis=1)))
    else:
        metrics["visible_preservation_error"] = float("nan")

    # ---- GT-dependent (eval-only) ----
    clean = item.get("clean_points_object")
    metrics["recon_chamfer_l1"] = (
        _chamfer_l1(sampled, np.asarray(clean, dtype=np.float32))
        if clean is not None and sampled.shape[0] > 0
        else float("nan")
    )
    hidden = item.get("hidden_clean_points")
    metrics["hidden_chamfer_l1"] = (
        _chamfer_l1(sampled, np.asarray(hidden, dtype=np.float32))
        if hidden is not None and len(hidden) > 0 and sampled.shape[0] > 0
        else float("nan")
    )
    return metrics


def refine_one_object(
    *,
    completion: SPCarPosteriorCompletionModel,
    decoder: SPCarShapeFieldDecoder,
    item: dict[str, Any],
    device: torch.device,
    steps: int,
    lr: float,
    weights: dict[str, float],
    deltas: dict[str, float],
    holdout_frac: float,
    free_violation_patience_increase: float,
    z_drift_max_norm: float,
    plateau_patience: int,
    rng: np.random.Generator,
    mc_resolution: int,
    sample_count: int,
    iso_level: float,
    free_violation_threshold: float,
    enable_ray_loss: bool,
    field_kind: str,
) -> dict[str, Any]:
    partial_np = item.get("partial_observed_points")
    if partial_np is None or partial_np.size == 0:
        partial_np = item["clean_points_object"]
    partial_t = _to_tensor(partial_np, device)
    free_t = _to_tensor(item.get("free_space_query_points"), device)
    hard_t = _to_tensor(item.get("free_space_query_hard_negatives"), device)

    qall_pts = item.get("occupancy_query_points")
    qall_lab = item.get("occupancy_query_labels")
    qall_ign = item.get("occupancy_query_ignore")

    (q_train, l_train, ign_train), (q_hold, l_hold, ign_hold) = _split_query_holdout(
        qall_pts, qall_lab, qall_ign, holdout_frac=holdout_frac, rng=rng,
    )
    q_train_t = _to_tensor(q_train, device)
    l_train_t = _to_label_tensor(l_train, device)
    ign_train_t = _to_bool_tensor(ign_train, device)
    q_hold_t = _to_tensor(q_hold, device)
    l_hold_t = _to_label_tensor(l_hold, device)
    ign_hold_t = _to_bool_tensor(ign_hold, device)

    scanner_pose = item.get("scanner_pose")
    scanner_t: torch.Tensor | None = None
    if scanner_pose is not None and enable_ray_loss:
        scanner_t = torch.from_numpy(np.asarray(scanner_pose, dtype=np.float32)).to(device)
        if scanner_t.numel() == 16:
            scanner_t = scanner_t.view(4, 4)[:3, 3]
        if scanner_t.numel() != 3:
            scanner_t = None

    # ----- initial z -----
    with torch.no_grad():
        post = completion.encode(partial_t.unsqueeze(0), sample=False)
        z0 = post.z_mean.squeeze(0).clone()

    z = z0.clone().detach().requires_grad_(True)
    optimiser = torch.optim.Adam([z], lr=lr)

    initial_eval = _compute_eval_metrics(
        decoder=decoder, z=z0, item=item, free_t=free_t,
        device=device, mc_resolution=mc_resolution,
        sample_count=sample_count, iso_level=iso_level,
        free_violation_threshold=free_violation_threshold,
    )
    initial_free_violation = initial_eval.get("free_space_violation_rate", float("nan"))

    best_z = z0.clone()
    best_holdout_score = math.inf
    plateau_count = 0
    early_stop_reason: str | None = None
    refine_started = time.time()
    history: list[dict[str, float]] = []

    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss, train_metrics = compute_observation_loss(
            decoder=decoder,
            z=z.unsqueeze(0),
            observed_points=partial_t.unsqueeze(0),
            free_points=free_t.unsqueeze(0) if free_t is not None else None,
            hard_negatives=hard_t.unsqueeze(0) if hard_t is not None else None,
            mixed_points=q_train_t.unsqueeze(0) if q_train_t is not None else None,
            mixed_labels=l_train_t.unsqueeze(0) if l_train_t is not None else None,
            mixed_ignore=ign_train_t.unsqueeze(0) if ign_train_t is not None else None,
            scanner_pose=scanner_t.unsqueeze(0) if scanner_t is not None else None,
            weights=weights, deltas=deltas,
            field_kind=field_kind,
            enable_ray_loss=enable_ray_loss and scanner_t is not None,
        )
        if not torch.isfinite(loss):
            early_stop_reason = "nonfinite_loss"
            break
        loss.backward()
        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=10.0)
        optimiser.step()

        # ----- held-out score -----
        with torch.no_grad():
            hold_score: float
            if q_hold_t is not None and l_hold_t is not None:
                hold_loss, _ = compute_observation_loss(
                    decoder=decoder,
                    z=z.unsqueeze(0),
                    observed_points=partial_t.unsqueeze(0),
                    free_points=None, hard_negatives=None,
                    mixed_points=q_hold_t.unsqueeze(0),
                    mixed_labels=l_hold_t.unsqueeze(0),
                    mixed_ignore=ign_hold_t.unsqueeze(0) if ign_hold_t is not None else None,
                    weights={"w_surf": 0.0, "w_free": 0.0, "w_mixed": 1.0,
                             "w_ray": 0.0, "w_incidence": 0.0, "lambda_prior": 0.0},
                    deltas=deltas, field_kind=field_kind,
                )
                hold_score = float(hold_loss.detach().item())
            else:
                hold_score = float(loss.detach().item())

            cur_violation = (
                free_space_violation_rate(
                    decoder, z.detach().unsqueeze(0), free_t.unsqueeze(0),
                    threshold=free_violation_threshold,
                )
                if free_t is not None
                else float("nan")
            )
            drift = float(torch.norm(z.detach() - z0).item())

        history.append({
            "step": step,
            "loss_total": float(loss.detach().item()),
            "loss_surf_obs": float(train_metrics.get("loss_surf_obs", float("nan"))),
            "loss_free": float(train_metrics.get("loss_free", float("nan"))),
            "loss_mixed": float(train_metrics.get("loss_mixed", float("nan"))),
            "loss_prior": float(train_metrics.get("loss_prior", float("nan"))),
            "free_violation": cur_violation,
            "z_drift": drift,
            "holdout_score": hold_score,
        })

        if hold_score + 1e-7 < best_holdout_score:
            best_holdout_score = hold_score
            best_z = z.detach().clone()
            plateau_count = 0
        else:
            plateau_count += 1
        if plateau_count >= plateau_patience:
            early_stop_reason = "plateau"
            break
        if (
            not math.isnan(initial_free_violation)
            and not math.isnan(cur_violation)
            and cur_violation > initial_free_violation + free_violation_patience_increase
        ):
            early_stop_reason = "free_space_increase"
            break
        if drift > z_drift_max_norm:
            early_stop_reason = "z_too_far_from_prior"
            break

    elapsed = time.time() - refine_started

    final_eval = _compute_eval_metrics(
        decoder=decoder, z=best_z, item=item, free_t=free_t,
        device=device, mc_resolution=mc_resolution,
        sample_count=sample_count, iso_level=iso_level,
        free_violation_threshold=free_violation_threshold,
    )

    return {
        "object_id": item.get("object_id"),
        "split": item.get("split"),
        "early_stop_reason": early_stop_reason,
        "n_steps_actually_run": len(history),
        "refinement_time_seconds": elapsed,
        "z_drift_final": float(torch.norm(best_z - z0).item()),
        "best_holdout_score": (best_holdout_score if math.isfinite(best_holdout_score) else float("nan")),
        "before_metrics": initial_eval,
        "after_metrics": final_eval,
        "before_minus_after": {
            k: (initial_eval[k] - final_eval[k])
            for k in initial_eval
            if isinstance(initial_eval.get(k), (int, float))
            and isinstance(final_eval.get(k), (int, float))
            and not math.isnan(initial_eval[k]) and not math.isnan(final_eval[k])
        },
        "history": history,
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
        "n_early_stop": sum(1 for m in per_object if m.get("early_stop_reason") is not None),
        "early_stop_reasons": {
            reason: sum(1 for m in per_object if m.get("early_stop_reason") == reason)
            for reason in ("plateau", "free_space_increase", "z_too_far_from_prior", "nonfinite_loss")
        },
        "refinement_time_per_object_seconds": _avg(lambda m: m.get("refinement_time_seconds")),
        "z_drift_final_mean": _avg(lambda m: m.get("z_drift_final")),
        "before_free_space_violation_rate_mean": _avg(
            lambda m: m["before_metrics"].get("free_space_violation_rate")
        ),
        "after_free_space_violation_rate_mean": _avg(
            lambda m: m["after_metrics"].get("free_space_violation_rate")
        ),
        "before_visible_preservation_error_mean": _avg(
            lambda m: m["before_metrics"].get("visible_preservation_error")
        ),
        "after_visible_preservation_error_mean": _avg(
            lambda m: m["after_metrics"].get("visible_preservation_error")
        ),
        "before_mesh_extraction_success_rate": _avg(
            lambda m: m["before_metrics"].get("mesh_extraction_success")
        ),
        "after_mesh_extraction_success_rate": _avg(
            lambda m: m["after_metrics"].get("mesh_extraction_success")
        ),
        "best_holdout_score_mean": _avg(lambda m: m.get("best_holdout_score")),
    }
    gt_dependent = {
        "before_recon_chamfer_l1_mean": _avg(lambda m: m["before_metrics"].get("recon_chamfer_l1")),
        "after_recon_chamfer_l1_mean": _avg(lambda m: m["after_metrics"].get("recon_chamfer_l1")),
        "before_hidden_chamfer_l1_mean": _avg(lambda m: m["before_metrics"].get("hidden_chamfer_l1")),
        "after_hidden_chamfer_l1_mean": _avg(lambda m: m["after_metrics"].get("hidden_chamfer_l1")),
        "delta_recon_chamfer_l1_mean": (
            _avg(lambda m: m["before_metrics"].get("recon_chamfer_l1"))
            - _avg(lambda m: m["after_metrics"].get("recon_chamfer_l1"))
        ),
        "delta_hidden_chamfer_l1_mean": (
            _avg(lambda m: m["before_metrics"].get("hidden_chamfer_l1"))
            - _avg(lambda m: m["after_metrics"].get("hidden_chamfer_l1"))
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
    parser.add_argument("--posterior_checkpoint", required=True,
                        help="Stage-3 posterior encoder checkpoint (.pt).")
    parser.add_argument("--shape_field_checkpoint", default=None,
                        help="Stage-2 decoder checkpoint. If None, decoder weights "
                             "are taken from the Stage-3 checkpoint payload.")
    parser.add_argument("--cache", required=True,
                        help="Path to the SP-CarNet object_index_v1.json (Stage-1 output).")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_objects", type=int, default=50)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mc_resolution", type=int, default=32)
    parser.add_argument("--sample_count", type=int, default=4096)
    parser.add_argument("--iso_level", type=float, default=0.5)
    parser.add_argument("--free_violation_threshold", type=float, default=0.5)
    parser.add_argument("--w_surf", type=float, default=1.0)
    parser.add_argument("--w_free", type=float, default=1.0)
    parser.add_argument("--w_mixed", type=float, default=0.5)
    parser.add_argument("--w_ray", type=float, default=0.5)
    parser.add_argument("--lambda_prior", type=float, default=1e-3)
    parser.add_argument("--delta_huber", type=float, default=0.5)
    parser.add_argument("--enable_ray_loss", action="store_true")
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--plateau_patience", type=int, default=10)
    parser.add_argument("--free_violation_patience_increase", type=float, default=0.10)
    parser.add_argument("--z_drift_max_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rng = np.random.default_rng(args.seed)
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

    dataset = SPCarObjectDataset(args.cache, splits=(args.split,))
    n_total = len(dataset) if args.num_objects <= 0 else min(args.num_objects, len(dataset))

    weights = {
        "w_surf": args.w_surf, "w_free": args.w_free, "w_mixed": args.w_mixed,
        "w_ray": args.w_ray, "lambda_prior": args.lambda_prior,
    }
    deltas = {
        "delta_surf": args.delta_huber,
        "delta_free": args.delta_huber,
        "delta_mixed": args.delta_huber,
    }

    per_object: list[dict[str, Any]] = []
    for i in range(n_total):
        item = dataset[i]
        try:
            entry = refine_one_object(
                completion=completion, decoder=decoder, item=item, device=device,
                steps=args.steps, lr=args.lr,
                weights=weights, deltas=deltas,
                holdout_frac=args.holdout_frac,
                free_violation_patience_increase=args.free_violation_patience_increase,
                z_drift_max_norm=args.z_drift_max_norm,
                plateau_patience=args.plateau_patience,
                rng=rng, mc_resolution=args.mc_resolution,
                sample_count=args.sample_count, iso_level=args.iso_level,
                free_violation_threshold=args.free_violation_threshold,
                enable_ray_loss=args.enable_ray_loss,
                field_kind="occupancy",
            )
        except Exception as exc:
            entry = {
                "object_id": item.get("object_id"),
                "split": item.get("split"),
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
    target = output_dir / "refinement.json"
    with target.open("w") as f:
        json.dump(out_doc, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

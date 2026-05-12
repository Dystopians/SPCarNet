#!/usr/bin/env python3
"""Rescore an existing K-best multi-hypothesis JSON with alternate reranker variants.

Recomputes top1 selection from the per-object candidates without re-running encoder/decoder/MC.
Each variant defines an inference-time score over candidate fields; we re-pick
top1 and re-aggregate chamfer/IoU/etc. GT chamfer is used only by the explicit
``oracle`` variant.

Variants
--------
- ``default``     : ``-loss_total + log_prior``  (the originally reported reranker — Stage 5 design §2)
- ``no_prior``    : ``-loss_total``              (drop the prior term entirely)
- ``norm_penalty``: ``-loss_total - λ · max(0, ||z|| - τ)``   (penalise only outlier-norm samples)
- ``rag_only``    : lowest RAG latent-bank distance
- ``sym_only``    : lowest mesh self-symmetry residual
- ``rag_sym``     : equal-rank fusion of RAG and symmetry
- ``obs_rag_sym`` : equal-rank fusion of observation loss, RAG, and symmetry
- ``first``       : candidate with the smallest sample index, for nested K=1/K=K audits
- ``sym_if_loss_le_first`` : symmetry pick only if observation loss is no worse than ``first``
- ``sym_if_score_ge_first``: symmetry pick only if original score is no worse than ``first``
- ``sym_if_loss_and_score_first``: require both guards before a symmetry pick
- ``oracle``      : pick by ``recon_chamfer_l1`` (GT-dependent — sanity / paper-only)

Usage
-----
    python scripts/car_model/rescore_spcarnet_multihypothesis.py \\
        --input outputs/carnet/spcarnet/multihypothesis/val_50_K8/K8.json \\
        --output outputs/carnet/spcarnet/multihypothesis/val_50_K8/K8_rescored.json \\
        --variants default no_prior norm_penalty
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable


def _score_default(c: dict[str, Any]) -> float:
    return -c["loss_total"] + c["log_prior"]


def _score_no_prior(c: dict[str, Any]) -> float:
    return -c["loss_total"]


def _score_norm_penalty(c: dict[str, Any], lam: float = 0.5, tau: float = 4.0) -> float:
    return -c["loss_total"] - lam * max(0.0, c.get("z_norm", 0.0) - tau)


def _score_oracle(c: dict[str, Any]) -> float:
    chamfer = c.get("recon_chamfer_l1")
    if chamfer is None or not math.isfinite(chamfer):
        return -math.inf
    return -chamfer


SCORE_VARIANTS: dict[str, Callable[[dict[str, Any]], float]] = {
    "default": _score_default,
    "no_prior": _score_no_prior,
    "norm_penalty": _score_norm_penalty,
    "oracle": _score_oracle,
}


RANK_FUSION_VARIANTS: dict[str, tuple[str, ...]] = {
    "rag_only": ("rag_dist_mean",),
    "sym_only": ("sym_residual_norm",),
    "rag_sym": ("rag_dist_mean", "sym_residual_norm"),
    "obs_rag_sym": ("loss_total", "rag_dist_mean", "sym_residual_norm"),
}


SPECIAL_VARIANTS = (
    "first",
    "sym_if_loss_le_first",
    "sym_if_score_ge_first",
    "sym_if_loss_and_score_first",
)


VARIANTS = tuple(SCORE_VARIANTS) + tuple(RANK_FUSION_VARIANTS) + SPECIAL_VARIANTS


def _pick_by_score(candidates: list[dict[str, Any]], score_fn: Callable[[dict[str, Any]], float]) -> dict[str, Any] | None:
    extracted = [c for c in candidates if c.get("extracted")]
    if not extracted:
        return None
    return max(extracted, key=score_fn)


def _finite_metric(c: dict[str, Any], key: str) -> float | None:
    value = c.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _pick_by_rank_fusion(candidates: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any] | None:
    extracted = [c for c in candidates if c.get("extracted")]
    eligible = [c for c in extracted if all(_finite_metric(c, key) is not None for key in keys)]
    if not eligible:
        return None
    if len(eligible) == 1:
        return eligible[0]

    ranks: dict[int, float] = {id(c): 0.0 for c in eligible}
    denom = max(len(eligible) - 1, 1)
    for key in keys:
        ordered = sorted(eligible, key=lambda c: _finite_metric(c, key))
        for rank, c in enumerate(ordered):
            ranks[id(c)] += float(rank) / float(denom)
    return min(eligible, key=lambda c: ranks[id(c)])


def _pick_first(extracted: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not extracted:
        return None
    return min(extracted, key=lambda c: int(c.get("k", 10**9)))


def _pick_sym_guarded(candidates: list[dict[str, Any]], *, require_loss: bool, require_score: bool) -> dict[str, Any] | None:
    extracted = [c for c in candidates if c.get("extracted")]
    first = _pick_first(extracted)
    if first is None:
        return None
    first_loss = _finite_metric(first, "loss_total")
    first_score = _finite_metric(first, "score")
    eligible: list[dict[str, Any]] = []
    for c in extracted:
        if _finite_metric(c, "sym_residual_norm") is None:
            continue
        if require_loss:
            loss = _finite_metric(c, "loss_total")
            if first_loss is None or loss is None or loss > first_loss:
                continue
        if require_score:
            score = _finite_metric(c, "score")
            if first_score is None or score is None or score < first_score:
                continue
        eligible.append(c)
    if not eligible:
        return first
    return min(eligible, key=lambda c: _finite_metric(c, "sym_residual_norm"))


def _pick_top1(candidates: list[dict[str, Any]], variant: str) -> dict[str, Any] | None:
    if variant in SCORE_VARIANTS:
        return _pick_by_score(candidates, SCORE_VARIANTS[variant])
    if variant in RANK_FUSION_VARIANTS:
        return _pick_by_rank_fusion(candidates, RANK_FUSION_VARIANTS[variant])
    if variant == "first":
        return _pick_first([c for c in candidates if c.get("extracted")])
    if variant == "sym_if_loss_le_first":
        return _pick_sym_guarded(candidates, require_loss=True, require_score=False)
    if variant == "sym_if_score_ge_first":
        return _pick_sym_guarded(candidates, require_loss=False, require_score=True)
    if variant == "sym_if_loss_and_score_first":
        return _pick_sym_guarded(candidates, require_loss=True, require_score=True)
    raise ValueError(f"unknown variant: {variant} (choices: {list(VARIANTS)})")


def _mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def rescore(per_object: list[dict[str, Any]], variant: str) -> dict[str, float]:
    chamfer_recon: list[float] = []
    chamfer_hidden: list[float] = []
    vis_pres: list[float] = []
    free_viol: list[float] = []
    extracted_rate: list[int] = []
    z_norms: list[float] = []
    skipped_missing_score = 0
    for obj in per_object:
        cands = obj.get("candidates", [])
        if not cands:
            continue
        pick = _pick_top1(cands, variant)
        if pick is None:
            extracted_rate.append(0)
            skipped_missing_score += 1
            continue
        extracted_rate.append(1)
        chamfer_recon.append(pick.get("recon_chamfer_l1"))
        chamfer_hidden.append(pick.get("hidden_chamfer_l1"))
        vis_pres.append(pick.get("visible_preservation_error"))
        free_viol.append(pick.get("free_space_violation_rate"))
        z_norms.append(pick.get("z_norm"))
    return {
        "n_objects": len(per_object),
        "n_extracted": sum(extracted_rate),
        "extracted_rate": _mean(extracted_rate),
        "top1_recon_chamfer_l1_mean": _mean(chamfer_recon),
        "top1_hidden_chamfer_l1_mean": _mean(chamfer_hidden),
        "top1_visible_preservation_error_mean": _mean(vis_pres),
        "top1_free_space_violation_rate_mean": _mean(free_viol),
        "top1_z_norm_mean": _mean(z_norms),
        "n_missing_variant_score": skipped_missing_score,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="K-best JSON from eval_spcarnet_multihypothesis.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args(argv)

    src = Path(args.input)
    data = json.loads(src.read_text())
    per_object = data.get("per_object", [])

    out: dict[str, Any] = {
        "source": str(src),
        "K": data.get("summary", {}).get("inference_only_metrics", {}).get("K"),
        "variants": {},
    }
    for v in args.variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant: {v} (choices: {list(VARIANTS)})")
        out["variants"][v] = rescore(per_object, v)

    out_path = Path(args.output) if args.output else src.with_suffix(".rescored.json")
    out_path.write_text(json.dumps(out, indent=2))

    # Print compact comparison table.
    print(f"# Rescore — source: {src}  K={out['K']}")
    print(f"{'variant':16s} {'top1_chamfer':>14s} {'hidden_chamfer':>16s} {'free_viol':>12s} {'vis_pres':>12s} {'z_norm':>10s}")
    for v in args.variants:
        m = out["variants"][v]
        print(
            f"{v:16s} "
            f"{m['top1_recon_chamfer_l1_mean']:>14.5f} "
            f"{m['top1_hidden_chamfer_l1_mean']:>16.5f} "
            f"{m['top1_free_space_violation_rate_mean']:>12.5f} "
            f"{m['top1_visible_preservation_error_mean']:>12.5f} "
            f"{m['top1_z_norm_mean']:>10.4f}"
        )
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

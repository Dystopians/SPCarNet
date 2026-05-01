"""Adapters for exporting MeshPrior scores to scene optimizers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_triangle_scores(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"triangle score file not found: {path}")
    payload = np.load(path, allow_pickle=False)
    if "scores" not in payload:
        raise KeyError(f"triangle score file missing `scores` array: {path}")
    return payload["scores"]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if not finite.any():
        return out
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    if hi - lo < 1e-8:
        out[finite] = np.clip(values[finite], 0.0, 1.0)
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def normalize_scores_per_region(scores: np.ndarray) -> np.ndarray:
    """Normalize protect/prune scores independently per region."""
    out = scores.copy()
    for region in np.unique(scores["region_id"]):
        mask = scores["region_id"] == region
        out["protect"][mask] = _normalize(scores["protect"][mask])
        out["prune"][mask] = _normalize(scores["prune"][mask])
    return out


def combine_scores(
    existing_score: np.ndarray,
    meshprior_score: np.ndarray,
    *,
    mode: str = "bounded_add",
    weight: float = 0.25,
) -> np.ndarray:
    existing = np.asarray(existing_score, dtype=np.float32)
    meshprior = np.clip(np.asarray(meshprior_score, dtype=np.float32), 0.0, 1.0)
    if mode != "bounded_add":
        raise ValueError(f"unsupported combine mode: {mode}")
    w = float(np.clip(weight, 0.0, 1.0))
    return existing + w * meshprior


def export_generic_meshprior_score_npz(scores: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, scores=scores)
    return output_path


def export_prism_score_json(
    scores: np.ndarray,
    output_path: str | Path,
    *,
    alpha: float = 0.25,
    beta: float = 0.25,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in scores:
        records.append(
            {
                "region_id": str(row["region_id"]),
                "face_index": int(row["face_index"]),
                "meshprior_protect": float(row["protect"]),
                "meshprior_prune": float(row["prune"]),
                "recommended_keep_delta_max": float(alpha),
                "recommended_prune_delta_max": float(beta),
            }
        )
    payload = {
        "format": "meshprior_prism_score_v1",
        "bounded_add": {"alpha": float(alpha), "beta": float(beta)},
        "records": records,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return output_path


def prism_present(repo_root: str | Path = ".") -> bool:
    root = Path(repo_root)
    return (
        (root / "utils/prism_scoring.py").is_file()
        and (root / "utils/prism_counterfactual.py").is_file()
        and (root / "utils/prism_pipeline.py").is_file()
    )

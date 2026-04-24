"""Ground-truth reflection-symmetry targets for CarNet_v0 (Phase 2 / A2).

A closed-form estimator of the dominant symmetry plane of a point cloud,
plus a soft confidence score. Called once per sample at cache-build time so
the training loop does not pay the cost repeatedly.

Convention
----------
A reflection plane is parameterised by a unit normal ``n`` and a signed
offset ``d`` such that a point ``p`` is reflected to ``p - 2·((p·n) − d)·n``.
The offset ``d`` is the signed distance of the plane from the origin, taken
along ``n``.

Confidence
----------
Symmetry confidence ``σ ∈ [0, 1]`` is computed from the mean one-sided
Chamfer residual between the reflected point set and the original set,
normalised by the point cloud's root-mean-square radius. A small residual
(σ → 1) indicates strong symmetry, a large residual (σ → 0) indicates an
asymmetric / partial shape (e.g. a patch covering a single wheel, or a
LiDAR half-scan).

The returned tuple is deterministic given the input point cloud.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

CONFIDENCE_SIGMA_DEFAULT = 0.04
MIN_POINTS = 32


def _rms_radius(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)) + 1e-12))


def _chamfer_to(points: np.ndarray, reference_tree: cKDTree) -> float:
    if len(points) == 0:
        return 0.0
    dists, _ = reference_tree.query(points, k=1)
    return float(np.mean(np.asarray(dists, dtype=np.float64)))


def _reflect(points: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    # p_reflected = p - 2 * ((p·n) − d) * n
    dot = points @ n.astype(np.float64)
    offsets = (dot - float(d))[:, None] * n.astype(np.float64)[None, :]
    return points - 2.0 * offsets


def _fit_plane_via_pca(points: np.ndarray, *, candidate_axes: Iterable[int]) -> list[tuple[np.ndarray, float]]:
    """Return candidate (normal, offset) pairs from the three PCA axes.

    For a symmetric object the symmetry normal is one of the eigenvectors
    of the (centered) covariance matrix. Any of the three could be the
    correct one, so we return all three and let the chamfer residual pick.
    """
    if len(points) < MIN_POINTS:
        return []
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    candidates: list[tuple[np.ndarray, float]] = []
    for axis in candidate_axes:
        n = np.asarray(eigvecs[:, axis], dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm < 1e-8:
            continue
        n = n / norm
        d = float(np.dot(n, centroid))
        candidates.append((n, d))
    return candidates


def _score_plane(
    points: np.ndarray,
    reference_tree: cKDTree,
    *,
    n: np.ndarray,
    d: float,
    rms_radius: float,
) -> tuple[float, float]:
    """Return (mean_chamfer, confidence σ ∈ [0, 1]) for a plane hypothesis."""
    reflected = _reflect(points.astype(np.float64), n, d)
    chamfer = _chamfer_to(reflected, reference_tree)
    normalised = chamfer / max(rms_radius, 1e-6)
    # Gaussian-shaped confidence: σ(0) = 1, σ(CONFIDENCE_SIGMA_DEFAULT) ≈ 0.61.
    sigma = float(np.exp(-(normalised**2) / (2.0 * CONFIDENCE_SIGMA_DEFAULT**2)))
    return chamfer, sigma


def estimate_symmetry_plane(
    points: np.ndarray,
    *,
    confidence_scale: float = CONFIDENCE_SIGMA_DEFAULT,
    max_sample: int = 4096,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Estimate the dominant reflection plane of ``points``.

    Returns
    -------
    n : (3,) float32
        Unit-normal of the best-fit reflection plane. Falls back to the
        canonical X-axis ``(1, 0, 0)`` for degenerate inputs (too few points
        or numerical failure).
    d : float
        Signed offset along ``n``.
    confidence : float ∈ [0, 1]
        Soft confidence that the object is symmetric about ``(n, d)``.
    chamfer_residual : float
        The raw mean Chamfer distance used to derive confidence (useful for
        diagnostics and downstream logging).
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) < MIN_POINTS:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0, float("inf")

    rng = rng if rng is not None else np.random.default_rng(0)
    if len(points) > max_sample:
        idx = rng.choice(len(points), size=max_sample, replace=False)
        sample_points = points[idx]
    else:
        sample_points = points

    rms_radius = _rms_radius(sample_points)
    if rms_radius < 1e-6:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0, float("inf")

    tree = cKDTree(sample_points.astype(np.float64))

    candidates = _fit_plane_via_pca(sample_points, candidate_axes=(0, 1, 2))
    if not candidates:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0, float("inf")

    best = None
    for n, d in candidates:
        chamfer, sigma = _score_plane(sample_points, tree, n=n, d=d, rms_radius=rms_radius)
        if best is None or chamfer < best[2]:
            best = (n.astype(np.float32), float(d), float(chamfer), float(sigma))

    assert best is not None
    n_best, d_best, chamfer_best, sigma_best = best
    # Override confidence scale if caller wants a different sharpness.
    if not np.isclose(confidence_scale, CONFIDENCE_SIGMA_DEFAULT):
        normalised = chamfer_best / max(rms_radius, 1e-6)
        sigma_best = float(np.exp(-(normalised**2) / (2.0 * confidence_scale**2)))

    # Canonicalise sign: flip so the first non-zero component of n is positive.
    for component in n_best:
        if abs(component) > 1e-6:
            if component < 0.0:
                n_best = (-n_best).astype(np.float32)
                d_best = -d_best
            break

    return n_best, d_best, sigma_best, chamfer_best

"""Paired bootstrap confidence intervals — the ONLY CI implementation (PROTOCOL §5).

Given two paired per-unit metric arrays ``a`` and ``b`` (units: test views for
rendering metrics; sampled segments/points/voxels/trajectories for g/d metrics),
we form the paired per-unit differences

    d_i = a_i - b_i

and bootstrap the mean of ``d`` by resampling units with replacement
(``numpy.random.default_rng(seed)``, 10,000 resamples by default) and taking
the percentile 95% CI.

MEMORY-BOUNDED IMPLEMENTATION (PROTOCOL §5, changelog 1.1.0): protocol unit
counts reach 10^6–10^7 (g/d per-sample arrays), so a full
(n_resamples × n) index matrix would need 80–320 GB. Instead, resamples are
drawn in chunks from a single ``numpy.random.default_rng(seed)`` stream and
their means accumulated incrementally; peak extra memory is bounded by
``_MAX_CHUNK_ELEMENTS`` int64 indices (~1.6 GB) regardless of n.

CANONICAL RANDOM STREAM: the canonical definition is a single
``default_rng(seed)`` stream consumed in resample order — resample ``i`` uses
the ``i``-th block of ``n`` bounded integers from that stream. numpy's
``Generator.integers`` fills bounded integers element-by-element from the bit
stream, so the drawn indices are identical whether the stream is consumed as
one (n_resamples × n) matrix, in chunks of any size, or row by row (verified
by the self-test). Chunk size is therefore a pure implementation detail:
results are bit-identical for any chunking, and identical to the historical
full-matrix implementation. Statistics are unchanged from PROTOCOL 1.0.0.

SIGN CONVENTION: every returned interval is a CI of ``mean(a - b)``.
A strictly positive interval means ``a`` is larger than ``b`` on average;
whether "larger" is better depends on the metric (PSNR: yes; LPIPS: no) and is
the caller's responsibility when phrasing results.

Reporting language (PROTOCOL §5 / D3): "improves/reduces" is allowed ONLY when
the 95% CI excludes 0 AND ``|mean(a - b)|`` clears the pre-registered
effect-size floor. Everything else is "DIAGNOSTIC". ``summarize_pair``
implements exactly that rule.
"""

from __future__ import annotations

import numpy as np

# Default resamples per chunk. The random stream is chunk-size-invariant (see
# module docstring), so this only trades RAM against numpy call overhead.
_CHUNK_RESAMPLES = 200
# Hard bound on the int64 index block: chunk_rows * n <= this many elements
# (2e8 int64 = 1.6 GB), so peak RAM stays bounded even at n ~ 10^7.
_MAX_CHUNK_ELEMENTS = 200_000_000


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                        seed: int = 0, chunk_resamples: int = _CHUNK_RESAMPLES) -> dict:
    """Percentile bootstrap 95% CI of mean(a - b) over paired units.

    Memory-bounded (PROTOCOL §5 / changelog 1.1.0): resample index blocks are
    drawn in chunks of at most ``chunk_resamples`` rows (further shrunk so a
    block never exceeds ``_MAX_CHUNK_ELEMENTS`` int64 entries) and resample
    means are accumulated incrementally. Because numpy fills bounded integers
    element-by-element from the generator's bit stream, the resample indices —
    and therefore every returned number — are bit-identical for any chunk size
    and identical to a full (n_resamples × n) draw from the same seed.

    Args:
        a: per-unit metric values for condition A, shape [n].
        b: per-unit metric values for condition B, shape [n], paired with ``a``
           (a[i] and b[i] come from the same unit).
        n_resamples: number of bootstrap resamples with replacement (default
            10,000 per PROTOCOL §5).
        seed: seed for ``numpy.random.default_rng`` (default 0 per PROTOCOL §5).
        chunk_resamples: implementation detail — resamples drawn per chunk;
            does NOT affect results (chunk-size-invariant canonical stream).

    Returns:
        dict with keys:
            'mean_diff': float, mean(a - b) on the original sample.
            'ci_lo': float, 2.5th percentile of bootstrap means of (a - b).
            'ci_hi': float, 97.5th percentile of bootstrap means of (a - b).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must have equal length, got {a.shape} vs {b.shape}")
    n = a.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 paired units, got {n}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired arrays must be finite (no NaN/inf)")
    n_resamples = int(n_resamples)
    if n_resamples < 1:
        raise ValueError(f"need at least 1 resample, got {n_resamples}")

    d = a - b
    rng = np.random.default_rng(seed)

    # Rows per index block: requested chunk size, shrunk so the int64 block
    # stays <= _MAX_CHUNK_ELEMENTS entries. Never affects the drawn values.
    rows = max(1, min(int(chunk_resamples), _MAX_CHUNK_ELEMENTS // n))

    boot_means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        k = min(rows, n_resamples - done)
        idx = rng.integers(0, n, size=(k, n))
        # Per-row gather+mean keeps the float64 gather at one row (8n bytes)
        # instead of materializing a (k, n) float64 block.
        for j in range(k):
            boot_means[done + j] = d[idx[j]].mean()
        done += k
        del idx

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        'mean_diff': float(d.mean()),
        'ci_lo': float(ci_lo),
        'ci_hi': float(ci_hi),
    }


def summarize_pair(a: np.ndarray, b: np.ndarray, floor: float | None = None,
                   n_resamples: int = 10000, seed: int = 0) -> dict:
    """Run paired_bootstrap_ci and attach the PROTOCOL §5 verdict string.

    Verdict rule (PROTOCOL §5 + D3 floors):
      - 'SIGNIFICANT (a > b)'  : CI excludes 0 with mean(a-b) > 0 and, when a
        floor is given, |mean(a-b)| >= floor.
      - 'SIGNIFICANT (a < b)'  : same with mean(a-b) < 0.
      - 'DIAGNOSTIC'           : everything else (CI contains 0, or effect is
        below the floor).

    The caller maps 'SIGNIFICANT (...)' onto "improves"/"reduces" according to
    the metric's polarity; this helper never claims a direction of goodness.

    Returns:
        dict with 'mean_diff', 'ci_lo', 'ci_hi', 'floor', 'excludes_zero'
        (bool), 'clears_floor' (bool; True when floor is None), and 'verdict'
        (str as above).
    """
    res = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
    excludes_zero = (res['ci_lo'] > 0.0) or (res['ci_hi'] < 0.0)
    clears_floor = True if floor is None else (abs(res['mean_diff']) >= float(floor))
    if excludes_zero and clears_floor:
        verdict = 'SIGNIFICANT (a > b)' if res['mean_diff'] > 0 else 'SIGNIFICANT (a < b)'
    else:
        verdict = 'DIAGNOSTIC'
    res.update({
        'floor': None if floor is None else float(floor),
        'excludes_zero': bool(excludes_zero),
        'clears_floor': bool(clears_floor),
        'verdict': verdict,
    })
    return res

"""Self-test for tools/gems/paired_bootstrap.py (PROTOCOL §5).

Run with:  python -m tools.gems.test_paired_bootstrap   (CPU only, no GPU)

Checks:
  1. Known-effect synthetic paired data: CI excludes 0 and contains the true
     effect.
  2. Null paired data: CI contains 0 and summarize_pair says DIAGNOSTIC.
  3. Coverage: over 500 simulated paired datasets with true effect delta, the
     95% CI must contain delta 95% +- 2% of the time (n_resamples=2000 inside
     the loop for speed; the API default stays 10,000).
  4. Determinism + verdict/floor logic sanity.
  5. Canonical-stream invariance: the chunked implementation must be
     bit-identical to a reference full (n_resamples x n) index-matrix draw
     from the same seed, for several chunk sizes (PROTOCOL 1.1.0 §5:
     chunking is an implementation detail, not a statistic change).
  6. Large-n memory/perf: n = 2,000,000 paired units, the full 10,000
     resamples — wall time must stay under 10 minutes and peak RSS is
     reported via resource.getrusage (must stay well under ~8 GB).
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np

from tools.gems.paired_bootstrap import paired_bootstrap_ci, summarize_pair

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_known_effect():
    rng = np.random.default_rng(123)
    n = 50
    true_delta = 0.3
    b = rng.normal(0.0, 1.0, n)
    a = b + rng.normal(true_delta, 1.0, n)  # paired: d_i ~ N(0.3, 1)
    res = paired_bootstrap_ci(a, b)
    print(f"  known-effect: mean_diff={res['mean_diff']:.4f} "
          f"CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] (true delta={true_delta})")
    check("known-effect CI excludes 0", res['ci_lo'] > 0.0 or res['ci_hi'] < 0.0)
    check("known-effect CI contains true delta",
          res['ci_lo'] <= true_delta <= res['ci_hi'])
    s = summarize_pair(a, b, floor=0.10)
    check("known-effect verdict is SIGNIFICANT (a > b)",
          s['verdict'] == 'SIGNIFICANT (a > b)', s['verdict'])


def test_null_effect():
    rng = np.random.default_rng(456)
    n = 50
    b = rng.normal(0.0, 1.0, n)
    a = b + rng.normal(0.0, 1.0, n)  # true effect 0
    res = paired_bootstrap_ci(a, b)
    print(f"  null: mean_diff={res['mean_diff']:.4f} "
          f"CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
    check("null CI contains 0", res['ci_lo'] <= 0.0 <= res['ci_hi'])
    s = summarize_pair(a, b, floor=0.10)
    check("null verdict is DIAGNOSTIC", s['verdict'] == 'DIAGNOSTIC', s['verdict'])


def test_coverage():
    """95% CI must contain the true effect 95% +- 2% over 500 simulations."""
    n_sims = 500
    n = 50
    true_delta = 0.3
    data_rng = np.random.default_rng(2026)
    hits = 0
    for i in range(n_sims):
        b = data_rng.normal(0.0, 1.0, n)
        a = b + data_rng.normal(true_delta, 1.0, n)
        res = paired_bootstrap_ci(a, b, n_resamples=2000, seed=i)
        if res['ci_lo'] <= true_delta <= res['ci_hi']:
            hits += 1
    coverage = hits / n_sims
    print(f"  coverage over {n_sims} sims (n={n}, delta={true_delta}): "
          f"{coverage:.3f} ({hits}/{n_sims})")
    check("coverage within 95% +- 2%", 0.93 <= coverage <= 0.97,
          f"coverage={coverage:.3f}")


def test_determinism_and_floor():
    rng = np.random.default_rng(789)
    a = rng.normal(1.0, 1.0, 40)
    b = rng.normal(0.0, 1.0, 40)
    r1 = paired_bootstrap_ci(a, b, seed=0)
    r2 = paired_bootstrap_ci(a, b, seed=0)
    check("deterministic for fixed seed", r1 == r2)

    # Significant but tiny effect + high floor -> DIAGNOSTIC.
    d = np.full(30, 0.02) + np.random.default_rng(7).normal(0, 0.005, 30)
    s = summarize_pair(d, np.zeros(30), floor=0.10)
    check("below-floor significant effect is DIAGNOSTIC",
          s['verdict'] == 'DIAGNOSTIC' and s['excludes_zero'] and not s['clears_floor'],
          s['verdict'])

    # Sign convention: a < b gives negative mean_diff.
    s2 = summarize_pair(np.zeros(30), d, floor=None)
    check("sign convention mean(a-b) negative when a < b",
          s2['mean_diff'] < 0 and s2['verdict'] == 'SIGNIFICANT (a < b)',
          f"mean_diff={s2['mean_diff']:.4f}, verdict={s2['verdict']}")

    # Input validation.
    try:
        paired_bootstrap_ci(np.ones(5), np.ones(4))
        check("length-mismatch raises ValueError", False)
    except ValueError:
        check("length-mismatch raises ValueError", True)


def _reference_full_matrix_ci(a, b, n_resamples, seed):
    """The historical (pre-1.1.0) implementation: one full index matrix.
    Used only to prove the chunked stream is bit-identical."""
    d = np.asarray(a, np.float64).ravel() - np.asarray(b, np.float64).ravel()
    n = d.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(int(n_resamples), n))
    boot_means = d[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {'mean_diff': float(d.mean()), 'ci_lo': float(ci_lo),
            'ci_hi': float(ci_hi)}


def test_chunk_invariance():
    """Chunked resampling must reproduce the full-matrix draw bit-for-bit,
    for any chunk size (canonical random stream, PROTOCOL 1.1.0 §5)."""
    rng = np.random.default_rng(11)
    n = 137
    b = rng.normal(0.0, 1.0, n)
    a = b + rng.normal(0.2, 1.0, n)
    ref = _reference_full_matrix_ci(a, b, n_resamples=1000, seed=0)
    for chunk in (1, 7, 200, 999, 1000, 5000):
        res = paired_bootstrap_ci(a, b, n_resamples=1000, seed=0,
                                  chunk_resamples=chunk)
        check(f"chunk={chunk} bit-identical to full-matrix reference",
              res == ref, f"res={res} ref={ref}")


def test_large_n_memory_perf():
    """PROTOCOL-scale run: n = 2,000,000 paired units, full 10,000 resamples.
    Wall < 10 min; peak RSS reported (chunked implementation keeps it far
    below the ~80 GB a full index matrix would need)."""
    n = 2_000_000
    rng = np.random.default_rng(99)
    b = rng.normal(0.0, 1.0, n)
    a = b + rng.normal(0.01, 0.5, n)
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    res = paired_bootstrap_ci(a, b, n_resamples=10000, seed=0)
    wall = time.perf_counter() - t0
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = rss_after_kb / (1024.0 ** 2)  # ru_maxrss is KiB on Linux
    print(f"  large-n: n={n}, 10000 resamples -> wall={wall:.1f}s, "
          f"peak RSS={peak_gb:.2f} GB (was {rss_before_kb / 1024.0**2:.2f} GB "
          f"before the call), CI=[{res['ci_lo']:.6f}, {res['ci_hi']:.6f}], "
          f"mean_diff={res['mean_diff']:.6f}")
    check("large-n wall time < 10 min", wall < 600.0, f"wall={wall:.1f}s")
    check("large-n CI brackets the true effect ~0.01 and excludes 0",
          res['ci_lo'] > 0.0 and res['ci_lo'] <= 0.01 <= res['ci_hi'],
          f"CI=[{res['ci_lo']:.6f}, {res['ci_hi']:.6f}]")
    check("large-n peak RSS < 8 GB", peak_gb < 8.0, f"peak={peak_gb:.2f} GB")


def main():
    print("== tools.gems.paired_bootstrap self-test ==")
    test_known_effect()
    test_null_effect()
    test_coverage()
    test_determinism_and_floor()
    test_chunk_invariance()
    test_large_n_memory_perf()
    if FAILURES:
        print(f"\nSELF-TEST FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("\nSELF-TEST PASSED: all checks green.")


if __name__ == "__main__":
    main()

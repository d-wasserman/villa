"""
Summary statistics and A/B comparison for benchmark results.

``compare`` pairs A and B runs in the order given (ABAB interleaving is
done by the caller) and reports the B/A ratio of a per-run metric with a
t-interval on log(B/A), plus a seeded bootstrap interval for reference.
With few pairs the bootstrap interval is too narrow; the t-interval is the
decision rule.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Dict, Optional, Sequence

from profiling import percentile


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None, "cv": None}
    mean = statistics.fmean(vals)
    cv = statistics.stdev(vals) / mean if len(vals) > 1 and mean else None
    return {
        "n": len(vals),
        "mean": mean,
        "p50": percentile(vals, 50.0),
        "p95": percentile(vals, 95.0),
        "min": min(vals),
        "max": max(vals),
        "cv": cv,
    }


def _t_critical(df: int, confidence: float) -> float:
    try:
        from scipy import stats as st

        return float(st.t.ppf(0.5 + confidence / 2.0, df))
    except ImportError:  # pragma: no cover - scipy is in requirements
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        if confidence != 0.95:
            raise
        return table.get(df, 1.96)


def compare_pairs(a: Sequence[float], b: Sequence[float], confidence: float = 0.95,
                  n_boot: int = 10000, seed: int = 0) -> Dict[str, object]:
    """
    Paired comparison of a per-run metric (lower is better, e.g. seconds per
    tile). Returns the geometric-mean ratio B/A with a t-interval on the log
    ratios, a bootstrap interval, and whether the t-interval excludes 1.0.
    """
    if len(a) != len(b):
        raise ValueError(f"need the same number of A and B runs, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError("need at least 2 pairs")
    logs = [math.log(float(y) / float(x)) for x, y in zip(a, b)]
    n = len(logs)
    mean = statistics.fmean(logs)
    sd = statistics.stdev(logs)
    half = _t_critical(n - 1, confidence) * sd / math.sqrt(n)
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(logs, k=n)) for _ in range(n_boot))
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    t_lo, t_hi = math.exp(mean - half), math.exp(mean + half)
    return {
        "pairs": n,
        "ratio_b_over_a": math.exp(mean),
        "t_interval": [t_lo, t_hi],
        "bootstrap_interval": [math.exp(boots[int(lo_q * (n_boot - 1))]), math.exp(boots[int(hi_q * (n_boot - 1))])],
        "confidence": confidence,
        "significant": not (t_lo <= 1.0 <= t_hi),
        "per_pair_ratios": [math.exp(v) for v in logs],
        "note": "t-interval on log(B/A) is the decision rule; bootstrap is too narrow for small n",
    }


def cm2(pixels: int, pixel_um: float) -> float:
    """Area in cm^2 of ``pixels`` square pixels of side ``pixel_um`` micrometres."""
    return pixels * (pixel_um * 1e-4) ** 2

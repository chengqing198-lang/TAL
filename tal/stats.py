"""Statistics used by TAL: Wilson intervals and paired bootstrap deltas.

No numpy dependency: cohorts here are small (tens to a few thousand prompts),
and pure-Python loops keep installation trivial and deterministic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Sequence


@dataclass
class Interval:
    low: float
    high: float

    def __iter__(self):
        return iter((self.low, self.high))

    def to_dict(self) -> dict[str, float]:
        return {"low": round(self.low, 6), "high": round(self.high, 6)}


@dataclass
class RateStats:
    n: int
    successes: int
    rate: float
    ci: Interval

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "successes": self.successes,
            "rate": round(self.rate, 6),
            "ci95": self.ci.to_dict(),
        }


@dataclass
class DeltaStats:
    n: int
    before_rate: float
    after_rate: float
    drop: float
    ci: Interval
    p_drop_le_zero: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "before_rate": round(self.before_rate, 6),
            "after_rate": round(self.after_rate, 6),
            "drop": round(self.drop, 6),
            "drop_ci95": self.ci.to_dict(),
            "p_drop_le_zero": round(self.p_drop_le_zero, 6),
        }


def _z_for(confidence: float) -> float:
    # Normal approximation; adequate for the 90/95/99% values TAL allows.
    table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    if confidence in table:
        return table[confidence]
    # Rational approximation of the inverse normal CDF (Acklam), good to ~1e-9.
    p = 1.0 - (1.0 - confidence) / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return Interval(0.0, 1.0)
    successes = min(max(int(successes), 0), n)
    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def rate_stats(successes: int, n: int, confidence: float = 0.95) -> RateStats:
    rate = successes / n if n else 0.0
    return RateStats(n=n, successes=successes, rate=rate, ci=wilson_interval(successes, n, confidence))


def paired_bootstrap_drop(
    before: Sequence[bool],
    after: Sequence[bool],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> DeltaStats:
    """Bootstrap the paired drop in refusal rate.

    `drop = mean(before) - mean(after)`, so a positive value means fewer refusals.
    Resampling is over prompts (the pairing unit), not responses.
    """
    if len(before) != len(after):
        raise ValueError("paired_bootstrap_drop requires equal-length sequences")
    n = len(before)
    if n == 0:
        return DeltaStats(0, 0.0, 0.0, 0.0, Interval(0.0, 0.0), 1.0)
    b = [1.0 if x else 0.0 for x in before]
    a = [1.0 if x else 0.0 for x in after]
    before_rate = sum(b) / n
    after_rate = sum(a) / n
    drop = before_rate - after_rate
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(max(200, n_boot)):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += b[i] - a[i]
        boots.append(total / n)
    boots.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = boots[int(math.floor(alpha * (len(boots) - 1)))]
    hi = boots[int(math.ceil((1 - alpha) * (len(boots) - 1)))]
    le_zero = sum(1 for x in boots if x <= 0.0) / len(boots)
    return DeltaStats(n, before_rate, after_rate, drop, Interval(lo, hi), le_zero)


def mean_ci(values: Sequence[float], confidence: float = 0.95, n_boot: int = 2000, seed: int = 0) -> Interval:
    if not values:
        return Interval(0.0, 0.0)
    if len(values) == 1:
        return Interval(float(values[0]), float(values[0]))
    rng = random.Random(seed)
    means = []
    for _ in range(max(200, n_boot)):
        means.append(sum(values[rng.randrange(len(values))] for _ in range(len(values))) / len(values))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        means[int(math.floor(alpha * (len(means) - 1)))],
        means[int(math.ceil((1 - alpha) * (len(means) - 1)))],
    )


def relative_drop(baseline: float, after: float) -> float:
    """Relative drop, guarding against a zero baseline."""
    if baseline == 0:
        return 0.0 if after == 0 else -math.inf
    return (baseline - after) / baseline


def as_dict(obj) -> dict:
    return asdict(obj)

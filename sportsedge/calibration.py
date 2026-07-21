"""Measure whether the model's probabilities are actually any good.

Profit and loss over a few dozen bets is almost pure noise, so it cannot tell
you whether a system works. These scores can, and much sooner.

The one that matters most is ``brier_skill_score`` against the market's own
devigged probability. A model that cannot beat the closing price has no edge,
no matter how confident it sounds or how well it happened to run last month.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# Clamp before taking a log so a confident-and-wrong forecast produces a large
# finite penalty instead of infinity.
_LOG_EPSILON = 1e-15


@dataclass
class ReliabilityBin:
    """One bucket of a reliability (calibration) curve."""

    lower: float
    upper: float
    count: int
    mean_forecast: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Positive means the model was overconfident in this bucket."""
        return self.mean_forecast - self.observed_rate


@dataclass
class CalibrationReport:
    n: int
    brier: float
    log_loss: float
    mean_forecast: float
    base_rate: float
    bins: list[ReliabilityBin] = field(default_factory=list)
    market_brier: float | None = None
    brier_skill_score: float | None = None

    @property
    def bias(self) -> float:
        """Overall over/under-confidence. Positive means overconfident."""
        return self.mean_forecast - self.base_rate

    def summary(self) -> str:
        lines = [
            f"n={self.n}  Brier={self.brier:.4f}  LogLoss={self.log_loss:.4f}",
            f"mean forecast={self.mean_forecast:.3f}  base rate={self.base_rate:.3f}  "
            f"bias={self.bias:+.3f} ({'over' if self.bias > 0 else 'under'}confident)",
        ]
        if self.brier_skill_score is not None:
            verdict = "beats" if self.brier_skill_score > 0 else "LOSES TO"
            lines.append(
                f"vs market: BSS={self.brier_skill_score:+.4f} "
                f"(market Brier={self.market_brier:.4f}) -- model {verdict} the market"
            )
        return "\n".join(lines)


def _validate(forecasts: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(forecasts) != len(outcomes):
        raise ValueError(
            f"forecasts and outcomes must be the same length, got {len(forecasts)} and {len(outcomes)}"
        )
    if not forecasts:
        raise ValueError("need at least one forecast to score")
    for p in forecasts:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"forecast probability out of range: {p}")
    for o in outcomes:
        if o not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {o}")


def brier_score(forecasts: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0 is perfect.

    Always guessing the base rate on a 50/50 market scores 0.25, so anything
    at or above that is worthless.
    """
    _validate(forecasts, outcomes)
    return sum((p - o) ** 2 for p, o in zip(forecasts, outcomes)) / len(forecasts)


def log_loss(forecasts: Sequence[float], outcomes: Sequence[int]) -> float:
    """Log loss. Punishes confident errors far harder than Brier does."""
    _validate(forecasts, outcomes)
    total = 0.0
    for p, o in zip(forecasts, outcomes):
        clamped = min(max(p, _LOG_EPSILON), 1.0 - _LOG_EPSILON)
        total += -(o * math.log(clamped) + (1 - o) * math.log(1.0 - clamped))
    return total / len(forecasts)


def brier_skill_score(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    reference: Sequence[float],
) -> float:
    """How much better the model is than a reference forecast.

    ``1 - brier(model) / brier(reference)``. Positive means the model beats the
    reference; zero means it adds nothing; negative means it actively makes the
    reference worse. Pass the market's devigged probability as ``reference`` to
    get the only verdict that matters.
    """
    model = brier_score(forecasts, outcomes)
    baseline = brier_score(reference, outcomes)
    if baseline == 0.0:
        raise ValueError("reference forecast is perfect; skill score is undefined")
    return 1.0 - model / baseline


def reliability_bins(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bucket forecasts and compare predicted rate to observed rate.

    A well-calibrated model puts roughly 30% of its 0.3-forecasts in the win
    column. Systematic gaps reveal where it is fooling itself -- LLM-derived
    confidences are typically far too high in the top buckets.
    """
    _validate(forecasts, outcomes)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(forecasts, outcomes):
        # Clamp so p == 1.0 lands in the last bucket rather than out of range.
        index = min(int(p * n_bins), n_bins - 1)
        buckets[index].append((p, o))

    out = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        out.append(
            ReliabilityBin(
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                count=len(bucket),
                mean_forecast=sum(p for p, _ in bucket) / len(bucket),
                observed_rate=sum(o for _, o in bucket) / len(bucket),
            )
        )
    return out


def evaluate(
    forecasts: Sequence[float],
    outcomes: Sequence[int],
    market: Sequence[float] | None = None,
    n_bins: int = 10,
) -> CalibrationReport:
    """Full calibration report, optionally scored against the market."""
    _validate(forecasts, outcomes)

    report = CalibrationReport(
        n=len(forecasts),
        brier=brier_score(forecasts, outcomes),
        log_loss=log_loss(forecasts, outcomes),
        mean_forecast=sum(forecasts) / len(forecasts),
        base_rate=sum(outcomes) / len(outcomes),
        bins=reliability_bins(forecasts, outcomes, n_bins),
    )

    if market is not None:
        report.market_brier = brier_score(market, outcomes)
        report.brier_skill_score = brier_skill_score(forecasts, outcomes, market)

    return report

"""Turn a model probability and a price into a staking decision.

This is the module the old systems were missing entirely. Both cricket-kg and
football-kg produced a verdict and an LLM "confidence" and stopped there, which
cannot distinguish a good bet from a bad one: 80% confidence at 1.10 is a
losing proposition, and 30% confidence at 5.00 is a strong one. Nothing here
knows or cares which sport it is being asked about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# A bet must clear this much edge before it is worth taking. Model probabilities
# are noisy, so an edge smaller than the noise is indistinguishable from zero;
# betting it just pays vig to the book. 2pp is a deliberately conservative floor
# for a model with no proven track record.
DEFAULT_MIN_EDGE = 0.02

# Full Kelly maximises long-run growth only if the probability is exactly right.
# It never is, and overbetting is punished far more harshly than underbetting,
# so stake a fraction of it. Quarter Kelly is the standard defensive choice.
DEFAULT_KELLY_FRACTION = 0.25

# Hard ceiling on any single stake, as a fraction of bankroll, regardless of
# what Kelly says. Guards against a mis-parsed price or a runaway probability.
DEFAULT_MAX_STAKE_FRACTION = 0.05

Verdict = Literal["bet", "no_edge", "below_threshold", "capped"]


@dataclass(frozen=True)
class BetSizing:
    """The full decision for one candidate bet, including why it was rejected."""

    model_probability: float
    market_probability: float
    decimal_odds: float
    edge: float
    expected_value: float
    kelly_fraction: float
    stake_fraction: float
    verdict: Verdict
    reason: str

    @property
    def is_bet(self) -> bool:
        return self.verdict == "bet" or self.verdict == "capped"

    def stake(self, bankroll: float) -> float:
        """Money amount to stake given a bankroll."""
        return self.stake_fraction * bankroll


def expected_value(probability: float, decimal_odds: float) -> float:
    """EV per unit staked. 0.05 means +5% expected return on the stake."""
    return probability * decimal_odds - 1.0


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll.

    f* = (p * o - 1) / (o - 1)

    Returns 0 for a non-positive edge rather than a negative number, since a
    negative Kelly would mean laying the bet, which these systems cannot do.
    """
    if decimal_odds <= 1.0:
        return 0.0
    fraction = (probability * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return max(0.0, fraction)


def evaluate_bet(
    model_probability: float,
    market_probability: float,
    decimal_odds: float,
    *,
    min_edge: float = DEFAULT_MIN_EDGE,
    kelly_multiplier: float = DEFAULT_KELLY_FRACTION,
    max_stake_fraction: float = DEFAULT_MAX_STAKE_FRACTION,
) -> BetSizing:
    """Decide whether to bet, and how much.

    ``market_probability`` must already be devigged (see ``odds.devig``).
    Comparing a model probability against the raw 1/odds figure would bake the
    bookmaker's margin into the edge and manufacture edges that do not exist.
    """
    if not 0.0 < model_probability < 1.0:
        raise ValueError(f"model probability must be in (0, 1), got {model_probability}")
    if not 0.0 < market_probability < 1.0:
        raise ValueError(f"market probability must be in (0, 1), got {market_probability}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal odds must be > 1.0, got {decimal_odds}")

    edge = model_probability - market_probability
    ev = expected_value(model_probability, decimal_odds)
    full_kelly = kelly_fraction(model_probability, decimal_odds)
    staked = full_kelly * kelly_multiplier

    if edge <= 0:
        return BetSizing(
            model_probability, market_probability, decimal_odds, edge, ev,
            full_kelly, 0.0, "no_edge",
            f"model is {abs(edge):.1%} below the market's devigged price",
        )

    if edge < min_edge:
        return BetSizing(
            model_probability, market_probability, decimal_odds, edge, ev,
            full_kelly, 0.0, "below_threshold",
            f"edge of {edge:.1%} is under the {min_edge:.1%} minimum",
        )

    if staked > max_stake_fraction:
        return BetSizing(
            model_probability, market_probability, decimal_odds, edge, ev,
            full_kelly, max_stake_fraction, "capped",
            f"Kelly wanted {staked:.1%} of bankroll, capped at {max_stake_fraction:.1%}",
        )

    return BetSizing(
        model_probability, market_probability, decimal_odds, edge, ev,
        full_kelly, staked, "bet",
        f"{edge:.1%} edge, EV {ev:+.1%} per unit",
    )


def blend(
    model_probability: float,
    market_probability: float,
    model_weight: float = 0.25,
) -> float:
    """Shrink a model probability toward the market price.

    The market is a strong, well-calibrated forecast that already aggregates
    far more information than any single model. An unproven model should be
    treated as a small adjustment to it, not a replacement for it. Raise
    ``model_weight`` only once calibration data justifies the confidence --
    ``calibration.brier_skill_score`` is the number that earns that right.
    """
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError(f"model weight must be in [0, 1], got {model_weight}")
    return model_weight * model_probability + (1.0 - model_weight) * market_probability

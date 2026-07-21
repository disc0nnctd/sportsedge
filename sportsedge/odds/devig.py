"""Convert bookmaker odds into true probabilities by removing the vig.

A bookmaker's quoted odds imply probabilities that sum to more than 1. That
excess is the overround (the vig), and it has to be stripped before the price
can be compared to a model's probability. How you strip it matters: the naive
method biases favourites and longshots in opposite directions, which is exactly
the region where a betting model most often thinks it has found an edge.

Three methods are provided, in increasing order of realism:

``multiplicative``
    Scale every implied probability by the same factor. Fast, and wrong in a
    known direction: it assumes the bookmaker applies margin evenly, but real
    books load more margin onto longshots. Use as a baseline only.

``power``
    Solve for ``k`` such that ``sum(p_i ** k) == 1``. Applies proportionally
    more correction to longshots, which matches observed bookmaker behaviour
    better than multiplicative at negligible cost.

``shin``
    Solve Shin's model, which derives the margin from an assumed proportion
    ``z`` of insider money. Generally the best fit on 2- and 3-outcome markets
    and the one to prefer for match-winner lines.

All functions take and return plain sequences so they can be used against any
odds provider.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

Method = Literal["multiplicative", "power", "shin"]

# Numerical solver settings. The bisection searches below converge in well
# under 100 iterations for any realistic overround; the cap only guards against
# pathological input.
_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200


class DevigError(ValueError):
    """Raised when odds cannot be converted into a valid probability vector."""


def implied_probabilities(odds: Sequence[float]) -> list[float]:
    """Raw implied probabilities (1/odds). These sum to more than 1."""
    if len(odds) < 2:
        raise DevigError(f"need at least 2 outcomes to devig, got {len(odds)}")
    for value in odds:
        if value <= 1.0:
            raise DevigError(
                f"decimal odds must be > 1.0, got {value}. "
                "American or fractional odds must be converted first."
            )
    return [1.0 / value for value in odds]


def overround(odds: Sequence[float]) -> float:
    """Bookmaker margin as a fraction. 0.05 means a 5% overround."""
    return sum(implied_probabilities(odds)) - 1.0


def _multiplicative(raw: Sequence[float]) -> list[float]:
    total = sum(raw)
    return [p / total for p in raw]


def _power(raw: Sequence[float]) -> list[float]:
    """Find k with sum(p_i ** k) == 1.

    sum(p_i ** k) decreases monotonically in k for p_i < 1, so a plain
    bisection on k is both safe and fast. k > 1 for a normal overround book.
    """
    if all(p >= 1.0 for p in raw):
        raise DevigError("cannot power-devig: every implied probability is >= 1")

    def total_at(k: float) -> float:
        return sum(p**k for p in raw)

    low, high = 1.0, 2.0
    # Expand the bracket until the upper bound undershoots 1.
    while total_at(high) > 1.0:
        high *= 2.0
        if high > 1e6:
            raise DevigError("power devig failed to bracket a solution")

    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        value = total_at(mid)
        if abs(value - 1.0) < _TOLERANCE:
            break
        if value > 1.0:
            low = mid
        else:
            high = mid
    else:
        mid = (low + high) / 2.0

    result = [p**mid for p in raw]
    total = sum(result)
    # Renormalise to absorb any residual solver error.
    return [p / total for p in result]


def _shin(raw: Sequence[float]) -> list[float]:
    """Shin's model: back out the insider proportion z, then invert.

    Given z, the true probability of outcome i is

        p_i = (sqrt(z**2 + 4*(1-z) * pi_i**2 / S) - z) / (2*(1-z))

    where pi_i is the raw implied probability and S their sum. We bisect on z
    in [0, 1) until the recovered probabilities sum to 1.
    """
    total_raw = sum(raw)
    if total_raw <= 1.0:
        # No overround to remove (or an arbitrage). Fall back to normalising.
        return _multiplicative(raw)

    def recover(z: float) -> list[float]:
        if z <= 0.0:
            return _multiplicative(raw)
        denominator = 2.0 * (1.0 - z)
        out = []
        for pi in raw:
            discriminant = z * z + 4.0 * (1.0 - z) * pi * pi / total_raw
            out.append((math.sqrt(discriminant) - z) / denominator)
        return out

    low, high = 0.0, 0.99
    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        total = sum(recover(mid))
        if abs(total - 1.0) < _TOLERANCE:
            break
        # Larger z removes more margin, lowering the recovered total.
        if total > 1.0:
            low = mid
        else:
            high = mid
    else:
        mid = (low + high) / 2.0

    result = recover(mid)
    total = sum(result)
    return [p / total for p in result]


_METHODS = {
    "multiplicative": _multiplicative,
    "power": _power,
    "shin": _shin,
}


def devig(odds: Sequence[float], method: Method = "shin") -> list[float]:
    """Strip the vig from decimal odds and return true probabilities.

    The result always sums to 1 (to floating-point tolerance) and preserves the
    ordering of the input odds.
    """
    if method not in _METHODS:
        raise DevigError(f"unknown devig method {method!r}, expected one of {sorted(_METHODS)}")
    raw = implied_probabilities(odds)
    probabilities = _METHODS[method](raw)

    if any(p <= 0.0 or p >= 1.0 for p in probabilities):
        raise DevigError(f"{method} devig produced a probability outside (0, 1): {probabilities}")
    return probabilities


def fair_odds(probability: float) -> float:
    """The break-even decimal price for a given probability."""
    if not 0.0 < probability < 1.0:
        raise DevigError(f"probability must be in (0, 1), got {probability}")
    return 1.0 / probability


def consensus_probabilities(
    books: Sequence[Sequence[float]],
    method: Method = "shin",
) -> list[float]:
    """Average devigged probabilities across several bookmakers.

    A single book's line carries that book's own bias and stale-price risk.
    Averaging the devigged view across books gives a sharper estimate of the
    market's true consensus, which is the number a model must actually beat.
    """
    if not books:
        raise DevigError("need at least one book to form a consensus")

    devigged = [devig(book, method=method) for book in books]
    width = len(devigged[0])
    if any(len(row) != width for row in devigged):
        raise DevigError("all books must quote the same number of outcomes")

    averaged = [sum(row[i] for row in devigged) / len(devigged) for i in range(width)]
    total = sum(averaged)
    return [p / total for p in averaged]

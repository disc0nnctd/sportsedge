import math

import pytest

from sportsedge.odds.devig import (
    DevigError,
    consensus_probabilities,
    devig,
    fair_odds,
    implied_probabilities,
    overround,
)

# A realistic 3-way football market with roughly a 5% overround.
FOOTBALL_1X2 = [2.10, 3.40, 3.75]
# A realistic 2-way cricket match-winner market.
CRICKET_H2H = [1.65, 2.35]


def test_overround_is_positive_for_a_real_market():
    assert overround(FOOTBALL_1X2) > 0
    assert 0.01 < overround(CRICKET_H2H) < 0.15


def test_implied_probabilities_sum_above_one():
    assert sum(implied_probabilities(FOOTBALL_1X2)) > 1.0


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
@pytest.mark.parametrize("odds", [FOOTBALL_1X2, CRICKET_H2H, [1.2, 5.5], [4.0, 4.0, 4.0]])
def test_devig_sums_to_one(method, odds):
    probabilities = devig(odds, method=method)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
    assert all(0.0 < p < 1.0 for p in probabilities)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_preserves_ordering(method):
    probabilities = devig(FOOTBALL_1X2, method=method)
    # Shortest price must remain the most likely outcome.
    assert probabilities[0] > probabilities[1]
    assert probabilities[1] > probabilities[2]


def test_devig_always_reduces_every_probability():
    """Removing margin can only lower the implied chance of each outcome."""
    raw = implied_probabilities(FOOTBALL_1X2)
    for method in ("multiplicative", "power", "shin"):
        for fair, r in zip(devig(FOOTBALL_1X2, method=method), raw):
            assert fair < r


def test_power_and_shin_shade_the_longshot_lower_than_multiplicative():
    """The whole point of power/shin: longshots carry more of the margin.

    Multiplicative assumes even margin, so it leaves the longshot with too much
    probability -- exactly the bet a naive model then thinks is +EV.
    """
    multiplicative = devig(FOOTBALL_1X2, method="multiplicative")
    power = devig(FOOTBALL_1X2, method="power")
    shin = devig(FOOTBALL_1X2, method="shin")

    longshot = -1
    assert power[longshot] < multiplicative[longshot]
    assert shin[longshot] < multiplicative[longshot]


def test_symmetric_market_gives_equal_probabilities():
    for method in ("multiplicative", "power", "shin"):
        probabilities = devig([1.9, 1.9], method=method)
        assert math.isclose(probabilities[0], 0.5, abs_tol=1e-9)


def test_fair_odds_round_trips():
    probabilities = devig(CRICKET_H2H, method="shin")
    assert math.isclose(1.0 / fair_odds(probabilities[0]), probabilities[0], abs_tol=1e-12)


def test_zero_margin_market_is_left_alone():
    """A book with no overround should pass through essentially unchanged."""
    exact = [2.0, 2.0]
    assert math.isclose(sum(devig(exact, method="shin")), 1.0, abs_tol=1e-9)


def test_consensus_averages_across_books():
    books = [[2.00, 2.00], [1.80, 2.20]]
    consensus = consensus_probabilities(books, method="multiplicative")
    assert math.isclose(sum(consensus), 1.0, abs_tol=1e-9)
    # Second book leans to the first outcome, so consensus must exceed 0.5.
    assert consensus[0] > 0.5


def test_consensus_rejects_mismatched_widths():
    with pytest.raises(DevigError):
        consensus_probabilities([[2.0, 2.0], [3.0, 3.0, 3.0]])


@pytest.mark.parametrize("bad", [[1.0, 2.0], [0.5, 2.0], [-1.0, 2.0]])
def test_rejects_non_decimal_odds(bad):
    with pytest.raises(DevigError):
        devig(bad)


def test_rejects_single_outcome():
    with pytest.raises(DevigError):
        devig([2.0])


def test_rejects_unknown_method():
    with pytest.raises(DevigError):
        devig(FOOTBALL_1X2, method="nonsense")

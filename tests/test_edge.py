import math

import pytest

from sportsedge.edge import (
    blend,
    evaluate_bet,
    expected_value,
    kelly_fraction,
)


def test_expected_value_is_zero_at_fair_odds():
    assert math.isclose(expected_value(0.5, 2.0), 0.0, abs_tol=1e-12)


def test_expected_value_positive_when_price_beats_probability():
    assert expected_value(0.55, 2.0) > 0
    assert expected_value(0.45, 2.0) < 0


def test_kelly_matches_the_closed_form():
    # p=0.6 at 2.0 -> (0.6*2 - 1)/(2 - 1) = 0.2
    assert math.isclose(kelly_fraction(0.6, 2.0), 0.2, abs_tol=1e-12)


def test_kelly_is_zero_without_an_edge():
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_grows_with_the_edge():
    assert kelly_fraction(0.7, 2.0) > kelly_fraction(0.6, 2.0)


def test_bet_is_taken_when_edge_clears_the_floor():
    sizing = evaluate_bet(0.55, 0.50, 2.10)
    assert sizing.verdict == "bet"
    assert sizing.is_bet
    assert sizing.stake_fraction > 0
    assert math.isclose(sizing.edge, 0.05, abs_tol=1e-12)


def test_a_ten_point_edge_already_trips_the_default_cap():
    """Quarter-Kelly on a 10pp edge at 2.10 wants 5.9% of bankroll.

    The 5% cap binds well before the edges an LLM-driven model typically
    claims, which is the intended safety behaviour rather than an edge case.
    """
    sizing = evaluate_bet(0.60, 0.50, 2.10)
    assert sizing.verdict == "capped"
    assert sizing.stake_fraction == 0.05


def test_no_bet_when_model_agrees_with_the_market():
    sizing = evaluate_bet(0.50, 0.50, 2.10)
    assert sizing.verdict == "no_edge"
    assert not sizing.is_bet
    assert sizing.stake_fraction == 0.0


def test_no_bet_when_model_is_below_the_market():
    sizing = evaluate_bet(0.40, 0.55, 2.10)
    assert sizing.verdict == "no_edge"
    assert sizing.stake_fraction == 0.0


def test_small_edge_is_rejected_as_noise():
    sizing = evaluate_bet(0.51, 0.50, 2.10, min_edge=0.02)
    assert sizing.verdict == "below_threshold"
    assert sizing.stake_fraction == 0.0


def test_stake_is_capped_regardless_of_kelly():
    """A huge claimed edge must not produce a huge stake."""
    sizing = evaluate_bet(0.95, 0.30, 3.00, max_stake_fraction=0.05)
    assert sizing.verdict == "capped"
    assert sizing.stake_fraction == 0.05
    assert sizing.is_bet


def test_fractional_kelly_is_applied():
    sizing = evaluate_bet(0.60, 0.50, 2.00, kelly_multiplier=0.25, max_stake_fraction=1.0)
    # full Kelly = 0.2, quarter Kelly = 0.05
    assert math.isclose(sizing.stake_fraction, 0.05, abs_tol=1e-12)


def test_stake_converts_to_money():
    sizing = evaluate_bet(0.60, 0.50, 2.00, kelly_multiplier=0.25, max_stake_fraction=1.0)
    assert math.isclose(sizing.stake(10_000), 500.0, abs_tol=1e-9)


def test_high_confidence_at_a_short_price_is_still_rejected():
    """The failure mode the old systems could not detect.

    80% confident is meaningless at 1.10 -- the market implies 91%.
    """
    sizing = evaluate_bet(model_probability=0.80, market_probability=0.91, decimal_odds=1.10)
    assert not sizing.is_bet
    assert sizing.expected_value < 0


def test_low_confidence_at_a_long_price_can_be_a_bet():
    """The mirror case: 30% is a strong bet at 5.00."""
    sizing = evaluate_bet(model_probability=0.30, market_probability=0.22, decimal_odds=5.00)
    assert sizing.is_bet
    assert sizing.expected_value > 0


@pytest.mark.parametrize(
    "p, m, o",
    [(0.0, 0.5, 2.0), (1.0, 0.5, 2.0), (0.5, 0.0, 2.0), (0.5, 1.0, 2.0), (0.5, 0.5, 1.0)],
)
def test_rejects_invalid_inputs(p, m, o):
    with pytest.raises(ValueError):
        evaluate_bet(p, m, o)


def test_blend_shrinks_toward_the_market():
    assert math.isclose(blend(0.80, 0.50, model_weight=0.25), 0.575, abs_tol=1e-12)
    assert blend(0.80, 0.50, model_weight=0.0) == 0.50
    assert blend(0.80, 0.50, model_weight=1.0) == 0.80


def test_blend_rejects_a_weight_outside_the_unit_interval():
    with pytest.raises(ValueError):
        blend(0.8, 0.5, model_weight=1.5)

"""Tests for the CLI pipeline seam: _market_view and _model_view.

These two functions connect the market feed and the ratings table to the pick
decision, and they are where both fake-edge bugs actually surfaced -- the
football draw miscalibration and the cricket disconnected-pool ratings both
flowed through _model_view. They had been validated only by live runs. These
tests pin the seam directly, with no network and no credits.
"""

import math

import pytest

from sportsedge.cli import _market_view, _model_view, _draw_nu
from sportsedge.odds.theoddsapi import Bookmaker, Event, Outcome
from sportsedge.ratings.elo import (
    Ratings,
    rating_difference,
    three_way_probabilities,
    two_way_probabilities,
)


def two_way_event(sport_key, home, away, home_prices, away_prices):
    """Build an Event with one h2h market per book."""
    books = []
    for i, (hp, ap) in enumerate(zip(home_prices, away_prices)):
        books.append(
            Bookmaker(
                key=f"book{i}",
                title=f"Book {i}",
                last_update="",
                outcomes=[Outcome(home, hp), Outcome(away, ap)],
            )
        )
    return Event(
        id="e1", sport_key=sport_key, commence_time="2026-07-23T13:00:00Z",
        home_team=home, away_team=away, bookmakers=books,
    )


def three_way_event(sport_key, home, away, prices):
    """prices: dict of outcome name -> list of per-book decimal odds."""
    n = len(next(iter(prices.values())))
    books = []
    for i in range(n):
        outcomes = [Outcome(name, prices[name][i]) for name in prices]
        books.append(Bookmaker(f"book{i}", f"Book {i}", "", outcomes))
    return Event(
        id="e1", sport_key=sport_key, commence_time="2026-07-23T13:00:00Z",
        home_team=home, away_team=away, bookmakers=books,
    )


# -- _market_view --------------------------------------------------------


def test_market_view_devigs_to_a_probability_vector():
    event = two_way_event("cricket_odi", "India", "Zimbabwe", [1.50], [2.60])
    view = _market_view(event)
    assert view is not None
    assert math.isclose(sum(view.values()), 1.0, abs_tol=1e-9)
    # The shorter price is the more likely outcome.
    assert view["India"] > view["Zimbabwe"]


def test_market_view_removes_the_vig():
    """Devigged probabilities must sum to 1, unlike raw 1/odds which sum high."""
    event = two_way_event("cricket_odi", "A", "B", [1.90], [1.90])
    raw = 1 / 1.90 + 1 / 1.90
    assert raw > 1.0
    view = _market_view(event)
    assert math.isclose(sum(view.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(view["A"], 0.5, abs_tol=1e-9)


def test_market_view_builds_consensus_across_books():
    event = two_way_event("cricket_odi", "A", "B", [2.00, 1.80], [2.00, 2.20])
    view = _market_view(event)
    # Second book leans to A, so consensus must exceed an even split.
    assert view["A"] > 0.5


def test_market_view_returns_none_for_a_single_outcome():
    event = Event("e", "cricket_odi", "2026-07-23T13:00:00Z", "A", "B",
                  [Bookmaker("b", "B", "", [Outcome("A", 1.5)])])
    assert _market_view(event) is None


def test_market_view_returns_none_with_no_books():
    event = Event("e", "cricket_odi", "2026-07-23T13:00:00Z", "A", "B", [])
    assert _market_view(event) is None


def test_market_view_handles_a_three_way_market():
    event = three_way_event(
        "soccer_epl", "Home", "Away",
        {"Home": [2.10], "Draw": [3.40], "Away": [3.75]},
    )
    view = _market_view(event)
    assert set(view) == {"Home", "Draw", "Away"}
    assert math.isclose(sum(view.values()), 1.0, abs_tol=1e-9)


# -- _model_view ---------------------------------------------------------


def test_model_view_skips_when_a_team_is_unrated():
    """The core safety property: no rating means no bet, not an invented one."""
    event = two_way_event("cricket_odi", "India", "Nowhere United", [1.5], [2.6])
    ratings = Ratings(values={"India": 1700.0})
    market = _market_view(event)
    assert _model_view(event, ratings, market) is None


def test_model_view_matches_names_through_normalisation():
    """Ratings keyed as 'Arsenal FC' must resolve for a fixture's 'Arsenal'."""
    event = three_way_event("soccer_epl", "Arsenal", "Chelsea",
                            {"Arsenal": [2.0], "Draw": [3.4], "Chelsea": [3.8]})
    ratings = Ratings(values={"Arsenal FC": 1650.0, "Chelsea FC": 1550.0})
    market = _market_view(event)
    view = _model_view(event, ratings, market)
    assert view is not None
    assert "Arsenal" in view and "Chelsea" in view


def test_model_view_favours_the_higher_rated_side():
    event = two_way_event("cricket_odi", "India", "Zimbabwe", [1.5], [2.6])
    ratings = Ratings(values={"India": 1750.0, "Zimbabwe": 1500.0})
    market = _market_view(event)
    view = _model_view(event, ratings, market)
    assert view["India"] > view["Zimbabwe"]
    assert math.isclose(sum(view.values()), 1.0, abs_tol=1e-9)


def test_model_view_two_way_matches_the_elo_formula():
    event = two_way_event("cricket_odi", "India", "Zimbabwe", [1.5], [2.6])
    ratings = Ratings(values={"India": 1750.0, "Zimbabwe": 1500.0})
    market = _market_view(event)
    view = _model_view(event, ratings, market)
    # Cricket internationals are treated as neutral venue.
    diff = rating_difference(1750.0, 1500.0, neutral_venue=True)
    p_home, _ = two_way_probabilities(diff)
    assert math.isclose(view["India"], p_home, abs_tol=1e-9)


def test_model_view_includes_a_draw_only_for_draw_sports():
    ratings = Ratings(values={"India": 1600.0, "England": 1600.0,
                              "Home FC": 1600.0, "Away FC": 1600.0})

    t20 = two_way_event("cricket_international_t20", "India", "England", [1.9], [1.9])
    assert "Draw" not in _model_view(t20, ratings, _market_view(t20))

    soccer = three_way_event("soccer_epl", "Home FC", "Away FC",
                             {"Home FC": [2.6], "Draw": [3.3], "Away FC": [2.9]})
    view = _model_view(soccer, ratings, _market_view(soccer))
    assert "Draw" in view


def test_model_view_draw_uses_the_calibrated_parameter():
    """The model draw must use the fitted draw_nu, not the 1.0 default.

    Checked by deriving the draw both ways at the fixture's actual rating gap
    (which includes home advantage, so it is not the bare parity value) and
    asserting the CLI matches the fitted one, not the default that inflated
    draws to ~33% and manufactured edges.
    """
    ratings = Ratings(values={"Home FC": 1600.0, "Away FC": 1600.0})
    event = three_way_event("soccer_epl", "Home FC", "Away FC",
                            {"Home FC": [2.6], "Draw": [3.3], "Away FC": [2.9]})
    view = _model_view(event, ratings, _market_view(event))

    diff = rating_difference(1600.0, 1600.0, neutral_venue=False)  # home advantage applies
    _, fitted_draw, _ = three_way_probabilities(diff, draw_nu=_draw_nu())
    _, default_draw, _ = three_way_probabilities(diff, draw_nu=1.0)

    assert math.isclose(view["Draw"], fitted_draw, abs_tol=1e-9)
    assert not math.isclose(view["Draw"], default_draw, abs_tol=1e-3)
    assert view["Draw"] < 0.30  # well below the uncalibrated ~33%


def test_model_view_only_returns_outcomes_the_market_quotes():
    """If the market has no draw line, the model must not offer a draw pick."""
    ratings = Ratings(values={"Home FC": 1600.0, "Away FC": 1600.0})
    # Two-way market (no Draw) on a draw-capable sport.
    event = two_way_event("soccer_epl", "Home FC", "Away FC", [2.0], [2.0])
    view = _model_view(event, ratings, _market_view(event))
    assert "Draw" not in view


def test_model_view_returns_none_without_team_names():
    event = Event("e", "cricket_odi", "2026-07-23T13:00:00Z", None, None, [])
    assert _model_view(event, Ratings(values={"A": 1500.0}), {"A": 1.0}) is None

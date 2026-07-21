import math

import pytest

from sportsedge.ratings.elo import (
    DEFAULT_HOME_ADVANTAGE,
    Ratings,
    expected_score,
    fit,
    goal_margin_multiplier,
    rating_difference,
    regress_to_mean,
    three_way_probabilities,
    two_way_probabilities,
    update,
)


def test_equal_ratings_on_a_neutral_pitch_are_a_coin_flip():
    assert math.isclose(expected_score(0.0), 0.5)


def test_four_hundred_points_is_ten_to_one():
    assert math.isclose(expected_score(400.0), 10.0 / 11.0, abs_tol=1e-9)


def test_home_advantage_helps_the_home_side():
    with_home = rating_difference(1500, 1500)
    neutral = rating_difference(1500, 1500, neutral_venue=True)
    assert with_home == DEFAULT_HOME_ADVANTAGE
    assert neutral == 0.0


def test_two_way_probabilities_sum_to_one():
    home, away = two_way_probabilities(120.0)
    assert math.isclose(home + away, 1.0)
    assert home > away


@pytest.mark.parametrize("diff", [-600, -200, 0, 150, 400, 900])
def test_three_way_probabilities_sum_to_one(diff):
    probabilities = three_way_probabilities(diff)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
    assert all(p > 0 for p in probabilities)


def test_three_way_is_symmetric_at_parity():
    home, draw, away = three_way_probabilities(0.0)
    assert math.isclose(home, away, abs_tol=1e-12)
    assert draw > 0


def test_draw_probability_peaks_at_parity_and_falls_with_mismatch():
    """Davidson's key behaviour, and what real football results show."""
    _, draw_even, _ = three_way_probabilities(0.0)
    _, draw_small, _ = three_way_probabilities(150.0)
    _, draw_large, _ = three_way_probabilities(500.0)
    assert draw_even > draw_small > draw_large


def test_default_draw_rate_is_realistic_for_football():
    """Around a quarter of evenly matched top-flight games end level.

    The previous default of nu=1.0 gave 33%, which is what top-5 league data
    (25.4% over 1751 matches) says is far too high. Because draws are priced
    long, that surplus surfaced as a steady stream of fake +EV draw bets.
    """
    _, draw, _ = three_way_probabilities(0.0)
    assert 0.23 < draw < 0.28


def test_a_lower_nu_produces_fewer_draws():
    _, fewer, _ = three_way_probabilities(0.0, draw_nu=0.5)
    _, more, _ = three_way_probabilities(0.0, draw_nu=1.5)
    assert fewer < more


def test_three_way_rejects_a_non_positive_nu():
    with pytest.raises(ValueError):
        three_way_probabilities(0.0, draw_nu=0.0)


def test_a_win_raises_the_winner_and_lowers_the_loser():
    ratings = Ratings()
    home, away = update(ratings, "Arsenal", "Everton", "home")
    assert home > 1500
    assert away < 1500


def test_rating_points_are_conserved():
    """Elo is zero-sum: what one side gains the other loses."""
    ratings = Ratings()
    before = ratings.get("A") + ratings.get("B")
    home, away = update(ratings, "A", "B", "home")
    assert math.isclose(home + away, before, abs_tol=1e-9)


def test_beating_a_stronger_side_earns_more_than_beating_a_weaker_one():
    upset = Ratings(values={"underdog": 1400, "giant": 1800})
    routine = Ratings(values={"favourite": 1800, "minnow": 1400})

    update(upset, "underdog", "giant", "home")
    update(routine, "favourite", "minnow", "home")

    assert (upset.get("underdog") - 1400) > (routine.get("favourite") - 1800)


def test_an_expected_home_draw_costs_the_favourite_rating():
    ratings = Ratings(values={"strong": 1800, "weak": 1400})
    update(ratings, "strong", "weak", "draw")
    assert ratings.get("strong") < 1800


def test_update_rejects_an_invalid_result():
    with pytest.raises(ValueError):
        update(Ratings(), "A", "B", "postponed")


def test_margin_multiplier_is_dampened_not_linear():
    assert goal_margin_multiplier(0) == 1.0
    four = goal_margin_multiplier(4)
    one = goal_margin_multiplier(1)
    assert four > one
    assert four < 4 * one  # log-dampened, not proportional


def test_fit_replays_a_season_and_orders_teams_sensibly():
    matches = [
        {"home": "Winner", "away": "Loser", "result": "home", "goal_difference": 2},
        {"home": "Loser", "away": "Winner", "result": "away", "goal_difference": 1},
        {"home": "Winner", "away": "Loser", "result": "home", "goal_difference": 3},
    ]
    ratings = fit(Ratings(), matches)
    assert ratings.get("Winner") > ratings.get("Loser")
    assert ratings.top(1)[0][0] == "Winner"


def test_regression_pulls_ratings_toward_the_mean():
    ratings = Ratings(values={"strong": 1800, "weak": 1200})
    regress_to_mean(ratings, weight=0.5)
    assert math.isclose(ratings.get("strong"), 1650)
    assert math.isclose(ratings.get("weak"), 1350)


def test_regression_rejects_a_weight_outside_the_unit_interval():
    with pytest.raises(ValueError):
        regress_to_mean(Ratings(), weight=2.0)


def test_ratings_survive_a_save_load_round_trip(tmp_path):
    ratings = Ratings(values={"Arsenal": 1720.5})
    path = tmp_path / "ratings.json"
    ratings.save(path)
    assert math.isclose(Ratings.load(path).get("Arsenal"), 1720.5)


def test_loading_a_missing_file_gives_default_ratings(tmp_path):
    assert Ratings.load(tmp_path / "nope.json").get("Anyone") == 1500.0

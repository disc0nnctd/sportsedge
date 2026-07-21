import math

import pytest

from sportsedge.calibration import (
    brier_score,
    brier_skill_score,
    evaluate,
    log_loss,
    reliability_bins,
)


def test_perfect_forecast_scores_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0


def test_worst_forecast_scores_one():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_coin_flip_scores_a_quarter():
    assert math.isclose(brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.25)


def test_log_loss_penalises_confident_errors_harder_than_brier():
    mild = log_loss([0.6], [0])
    severe = log_loss([0.99], [0])
    assert severe > mild * 5


def test_log_loss_is_finite_for_a_certain_and_wrong_forecast():
    assert math.isfinite(log_loss([1.0], [0]))


def test_skill_score_is_positive_when_the_model_beats_the_reference():
    outcomes = [1, 1, 0, 0]
    model = [0.8, 0.7, 0.3, 0.2]
    market = [0.6, 0.55, 0.45, 0.4]
    assert brier_skill_score(model, outcomes, market) > 0


def test_skill_score_is_negative_when_the_model_is_worse():
    outcomes = [1, 1, 0, 0]
    model = [0.4, 0.45, 0.55, 0.6]
    market = [0.6, 0.55, 0.45, 0.4]
    assert brier_skill_score(model, outcomes, market) < 0


def test_skill_score_is_zero_against_itself():
    outcomes = [1, 0, 1, 0]
    forecasts = [0.7, 0.3, 0.6, 0.2]
    assert math.isclose(brier_skill_score(forecasts, outcomes, forecasts), 0.0, abs_tol=1e-12)


def test_reliability_bins_expose_overconfidence():
    """Twenty 0.9-forecasts that win only half the time."""
    forecasts = [0.9] * 20
    outcomes = [1] * 10 + [0] * 10
    bins = reliability_bins(forecasts, outcomes, n_bins=10)
    assert len(bins) == 1
    assert bins[0].count == 20
    assert math.isclose(bins[0].observed_rate, 0.5)
    assert bins[0].gap > 0.35  # badly overconfident


def test_reliability_bin_edge_case_probability_of_one():
    bins = reliability_bins([1.0], [1], n_bins=10)
    assert len(bins) == 1
    assert bins[0].count == 1


def test_evaluate_reports_bias_and_skill():
    outcomes = [1, 1, 0, 0, 1, 0]
    model = [0.9, 0.85, 0.8, 0.7, 0.75, 0.6]  # systematically too high
    market = [0.6, 0.55, 0.45, 0.4, 0.5, 0.35]

    report = evaluate(model, outcomes, market)
    assert report.n == 6
    assert report.bias > 0  # overconfident
    assert report.brier_skill_score is not None
    assert "vs market" in report.summary()


def test_evaluate_without_market_omits_skill():
    report = evaluate([0.6, 0.4], [1, 0])
    assert report.brier_skill_score is None
    assert "vs market" not in report.summary()


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        brier_score([], [])


def test_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        brier_score([1.5], [1])


def test_rejects_non_binary_outcome():
    with pytest.raises(ValueError):
        brier_score([0.5], [2])

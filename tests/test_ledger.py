import math

import pytest

from sportsedge.ledger import Ledger


def make_pick(ledger, **overrides):
    payload = dict(
        sport="cricket",
        event_id="evt1",
        matchup="India vs Zimbabwe",
        commence_time="2026-07-23T13:00:00Z",
        market="h2h",
        selection="India",
        model_probability=0.72,
        market_probability=0.66,
        decimal_odds=1.52,
        bookmaker="Bet365",
        edge=0.06,
        stake_fraction=0.02,
        stake=200.0,
        rationale="ratings gap plus away side missing two seamers",
    )
    payload.update(overrides)
    return ledger.record(**payload)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "picks.jsonl")


def test_record_then_read_round_trips(ledger):
    pick = make_pick(ledger)
    stored = ledger.get(pick.id)
    assert stored is not None
    assert stored.selection == "India"
    assert stored.result == "pending"
    assert math.isclose(stored.decimal_odds, 1.52)


def test_records_append_rather_than_overwrite(ledger):
    make_pick(ledger)
    make_pick(ledger, selection="Zimbabwe")
    assert len(ledger.all()) == 2


def test_winning_pick_profit_and_payout(ledger):
    pick = make_pick(ledger, stake=100.0, decimal_odds=2.5)
    settled = ledger.settle(pick.id, "won")
    assert math.isclose(settled.payout, 250.0)
    assert math.isclose(settled.profit, 150.0)
    assert settled.outcome_bit == 1


def test_losing_pick_loses_the_stake(ledger):
    pick = make_pick(ledger, stake=100.0)
    settled = ledger.settle(pick.id, "lost")
    assert math.isclose(settled.profit, -100.0)
    assert settled.payout == 0.0
    assert settled.outcome_bit == 0


def test_push_returns_the_stake(ledger):
    pick = make_pick(ledger, stake=100.0)
    settled = ledger.settle(pick.id, "push")
    assert math.isclose(settled.payout, 100.0)
    assert settled.profit == 0.0
    assert settled.outcome_bit is None


def test_pending_picks_are_listed(ledger):
    p1 = make_pick(ledger)
    make_pick(ledger)
    ledger.settle(p1.id, "won")
    assert len(ledger.pending()) == 1


def test_closing_line_value_is_positive_when_the_price_shortens(ledger):
    """Took 1.52 (66%); market closed at 70%. The bet was ahead of the move."""
    pick = make_pick(ledger, market_probability=0.66)
    updated = ledger.set_closing_line(pick.id, closing_odds=1.43, closing_probability=0.70)
    assert math.isclose(updated.closing_line_value, 0.04, abs_tol=1e-9)
    assert updated.beat_the_close is True


def test_closing_line_value_is_negative_when_the_price_drifts(ledger):
    pick = make_pick(ledger, market_probability=0.66, decimal_odds=1.52)
    updated = ledger.set_closing_line(pick.id, closing_odds=1.75, closing_probability=0.57)
    assert updated.closing_line_value < 0
    assert updated.beat_the_close is False


def test_clv_is_none_before_the_close_is_captured(ledger):
    pick = make_pick(ledger)
    assert pick.closing_line_value is None
    assert pick.beat_the_close is None


def test_summary_aggregates_pnl_and_clv(ledger):
    won = make_pick(ledger, stake=100.0, decimal_odds=2.0)
    lost = make_pick(ledger, stake=100.0, decimal_odds=2.0)
    make_pick(ledger, stake=100.0)  # left pending

    ledger.set_closing_line(won.id, 1.80, 0.70)
    ledger.set_closing_line(lost.id, 2.20, 0.40)
    ledger.settle(won.id, "won")
    ledger.settle(lost.id, "lost")

    summary = ledger.summarise()
    assert summary.n_picks == 3
    assert summary.n_settled == 2
    assert summary.pending == 1
    assert math.isclose(summary.profit, 0.0)  # +100 then -100
    assert math.isclose(summary.staked, 200.0)
    assert summary.wins == 1 and summary.losses == 1
    assert summary.clv_measured == 2
    assert summary.beat_close_rate == 0.5


def test_summary_filters_by_sport(ledger):
    make_pick(ledger, sport="cricket")
    make_pick(ledger, sport="football")
    assert ledger.summarise(sport="football").n_picks == 1


def test_summary_reports_when_clv_is_unmeasurable(ledger):
    make_pick(ledger)
    assert "CLV not yet measurable" in ledger.summarise().summary()


def test_calibration_inputs_skip_unsettled_picks(ledger):
    won = make_pick(ledger, model_probability=0.7, market_probability=0.6)
    make_pick(ledger)  # pending, must be excluded
    ledger.settle(won.id, "won")

    forecasts, outcomes, market = ledger.calibration_inputs()
    assert forecasts == [0.7]
    assert outcomes == [1]
    assert market == [0.6]


def test_settle_rejects_an_unknown_result(ledger):
    pick = make_pick(ledger)
    with pytest.raises(ValueError):
        ledger.settle(pick.id, "maybe")


def test_settle_rejects_an_unknown_pick(ledger):
    with pytest.raises(KeyError):
        ledger.settle("does-not-exist", "won")


def test_empty_ledger_summarises_cleanly(ledger):
    summary = ledger.summarise()
    assert summary.n_picks == 0
    assert summary.roi == 0.0
    assert summary.win_rate == 0.0

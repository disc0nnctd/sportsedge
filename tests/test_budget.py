import json

import pytest

from sportsedge.odds.budget import (
    BudgetExceeded,
    CreditBudget,
    current_period,
)


@pytest.fixture
def budget(tmp_path):
    return CreditBudget(tmp_path / "budget.json", limit=100, reserve_fraction=0.25)


def test_starts_with_a_full_allowance(budget):
    assert budget.state.spent == 0
    assert budget.state.remaining == 100
    assert budget.state.discretionary == 75  # 25 held in reserve


def test_charging_reduces_the_allowance(budget):
    budget.charge(10)
    assert budget.state.spent == 10
    assert budget.state.remaining == 90
    assert budget.state.discretionary == 65


def test_check_passes_within_the_discretionary_pool(budget):
    budget.check(75)  # must not raise


def test_check_blocks_a_spend_that_eats_the_reserve(budget):
    with pytest.raises(BudgetExceeded, match="discretionary"):
        budget.check(76)


def test_reserve_is_reachable_when_explicitly_allowed(budget):
    budget.check(90, allow_reserve=True)  # must not raise


def test_nothing_can_exceed_the_hard_limit(budget):
    with pytest.raises(BudgetExceeded):
        budget.check(101, allow_reserve=True)


def test_exhausted_budget_blocks_further_spending(budget):
    budget.charge(100)
    assert budget.state.remaining == 0
    with pytest.raises(BudgetExceeded):
        budget.check(1, allow_reserve=True)


def test_spend_persists_across_instances(tmp_path):
    path = tmp_path / "budget.json"
    CreditBudget(path, limit=100).charge(30)
    assert CreditBudget(path, limit=100).state.spent == 30


def test_reconcile_trusts_a_higher_server_count(budget):
    budget.charge(10)
    budget.reconcile(remaining_from_header=80)  # server says 20 used
    assert budget.state.spent == 20


def test_reconcile_ignores_a_lower_server_count(budget):
    """Never revise spending downward -- that would license overspending."""
    budget.charge(30)
    budget.reconcile(remaining_from_header=95)  # server says only 5 used
    assert budget.state.spent == 30


def test_a_new_month_resets_the_counter(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({
        "period": "1999-01",
        "spent": 500,
        "limit": 100,
        "reserved": 25,
        "last_updated": "1999-01-31T00:00:00+00:00",
    }))
    assert CreditBudget(path, limit=100).state.spent == 0


def test_a_corrupt_ledger_does_not_block_the_month(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text("{not valid json")
    assert CreditBudget(path, limit=100).state.spent == 0


def test_current_period_is_year_month():
    period = current_period()
    assert len(period) == 7 and period[4] == "-"


def test_report_mentions_usage(budget):
    budget.charge(40)
    assert "40/100" in budget.report()

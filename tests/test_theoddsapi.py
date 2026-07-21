"""Client tests against a stubbed transport -- no network, no credits spent."""

import json

import pytest

from sportsedge.odds.budget import BudgetExceeded, CreditBudget
from sportsedge.odds.theoddsapi import (
    OddsAPIError,
    OddsClient,
    _parse_event,
    upcoming_within,
)

SAMPLE_EVENT = {
    "id": "abc123",
    "sport_key": "cricket_international_t20",
    "commence_time": "2026-07-23T13:00:00Z",
    "home_team": "India",
    "away_team": "Zimbabwe",
    "bookmakers": [
        {
            "key": "bet365",
            "title": "Bet365",
            "last_update": "2026-07-22T09:00:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "India", "price": 1.50},
                        {"name": "Zimbabwe", "price": 2.60},
                    ],
                }
            ],
        },
        {
            "key": "pinnacle",
            "title": "Pinnacle",
            "last_update": "2026-07-22T09:01:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "India", "price": 1.55},
                        {"name": "Zimbabwe", "price": 2.70},
                    ],
                }
            ],
        },
    ],
}


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeSession:
    """Records every call so tests can assert on request count and params."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return self.response


@pytest.fixture
def client(tmp_path):
    budget = CreditBudget(tmp_path / "budget.json", limit=100, reserve_fraction=0.25)
    return OddsClient("test-key", budget, cache_dir=tmp_path / "cache")


def attach(client, payload, headers=None):
    session = FakeSession(FakeResponse(payload, headers=headers))
    client.session = session
    return session


# -- parsing -------------------------------------------------------------


def test_parses_an_event_and_its_books():
    event = _parse_event(SAMPLE_EVENT)
    assert event.matchup == "India vs Zimbabwe"
    assert len(event.bookmakers) == 2
    assert event.bookmakers[0].outcomes[0].price == 1.50


def test_prices_for_collects_every_book():
    prices = _parse_event(SAMPLE_EVENT).prices_for()
    assert prices["India"] == [1.50, 1.55]
    assert prices["Zimbabwe"] == [2.60, 2.70]


def test_best_price_finds_the_top_of_the_market():
    event = _parse_event(SAMPLE_EVENT)
    price, book = event.best_price("India")
    assert price == 1.55
    assert book == "Pinnacle"


def test_best_price_returns_none_for_an_unknown_selection():
    assert _parse_event(SAMPLE_EVENT).best_price("Australia") is None


def test_parsing_tolerates_an_event_with_no_books():
    event = _parse_event({"id": "x", "sport_key": "s", "commence_time": "2026-07-23T13:00:00Z"})
    assert event.bookmakers == []
    assert event.matchup == "x"


def test_parsing_skips_an_outcome_with_no_price():
    payload = json.loads(json.dumps(SAMPLE_EVENT))
    payload["bookmakers"][0]["markets"][0]["outcomes"][0].pop("price")
    event = _parse_event(payload)
    assert len(event.bookmakers[0].outcomes) == 1


# -- credit accounting ---------------------------------------------------


def test_events_endpoint_is_free(client):
    attach(client, [SAMPLE_EVENT])
    client.list_events("cricket_odi")
    assert client.budget.state.spent == 0


def test_sports_endpoint_is_free(client):
    attach(client, [{"key": "cricket_odi", "title": "ODI"}])
    client.list_sports()
    assert client.budget.state.spent == 0


def test_odds_cost_one_credit_for_one_market_and_region(client):
    attach(client, [SAMPLE_EVENT])
    client.get_odds("cricket_odi", regions=["uk"], markets=["h2h"])
    assert client.budget.state.spent == 1


def test_odds_cost_multiplies_markets_by_regions(client):
    attach(client, [SAMPLE_EVENT])
    client.get_odds("cricket_odi", regions=["uk", "eu", "us"], markets=["h2h", "totals"])
    assert client.budget.state.spent == 6


def test_estimate_cost_matches_what_is_charged(client):
    attach(client, [SAMPLE_EVENT])
    estimate = client.estimate_cost(["uk", "eu"], ["h2h"])
    client.get_odds("cricket_odi", regions=["uk", "eu"], markets=["h2h"])
    assert client.budget.state.spent == estimate


def test_server_reported_cost_overrides_the_local_estimate(client):
    attach(client, [SAMPLE_EVENT], headers={"x-requests-last": "5"})
    client.get_odds("cricket_odi", regions=["uk"], markets=["h2h"])
    assert client.budget.state.spent == 5


def test_remaining_header_reconciles_the_ledger(client):
    attach(client, [SAMPLE_EVENT], headers={"x-requests-last": "1", "x-requests-remaining": "40"})
    client.get_odds("cricket_odi")
    assert client.budget.state.spent == 60  # limit 100 - 40 remaining


def test_budget_blocks_the_call_before_it_is_made(client):
    session = attach(client, [SAMPLE_EVENT])
    client.budget.charge(75)  # exhausts the discretionary pool
    with pytest.raises(BudgetExceeded):
        client.get_odds("cricket_odi")
    assert session.calls == []  # never left the process


def test_scores_may_draw_on_the_reserve(client):
    attach(client, [])
    client.budget.charge(75)
    client.get_scores("cricket_odi")  # allow_reserve defaults to True
    assert client.budget.state.spent == 76


def test_days_from_makes_scores_cost_two(client):
    attach(client, [])
    client.get_scores("cricket_odi", days_from=2)
    assert client.budget.state.spent == 2


# -- caching -------------------------------------------------------------


def test_a_repeat_call_is_served_from_cache_for_free(client):
    session = attach(client, [SAMPLE_EVENT])
    client.get_odds("cricket_odi")
    client.get_odds("cricket_odi")
    assert len(session.calls) == 1
    assert client.budget.state.spent == 1


def test_different_parameters_are_cached_separately(client):
    session = attach(client, [SAMPLE_EVENT])
    client.get_odds("cricket_odi", regions=["uk"])
    client.get_odds("cricket_odi", regions=["eu"])
    assert len(session.calls) == 2


def test_an_expired_cache_entry_is_refetched(client):
    client.cache_ttl = -1  # everything is instantly stale
    session = attach(client, [SAMPLE_EVENT])
    client.get_odds("cricket_odi")
    client.get_odds("cricket_odi")
    assert len(session.calls) == 2


def test_scores_are_never_cached(client):
    session = attach(client, [])
    client.get_scores("cricket_odi")
    client.get_scores("cricket_odi")
    assert len(session.calls) == 2


def test_the_api_key_is_not_part_of_the_cache_filename(client, tmp_path):
    attach(client, [SAMPLE_EVENT])
    client.get_odds("cricket_odi")
    for path in (tmp_path / "cache").iterdir():
        assert "test-key" not in path.name


# -- errors --------------------------------------------------------------


def test_a_missing_key_is_rejected_at_construction(tmp_path):
    budget = CreditBudget(tmp_path / "b.json")
    with pytest.raises(OddsAPIError, match="ODDS_API_KEY"):
        OddsClient("", budget, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "status, expected",
    [(401, "rejected the key"), (422, "parameters"), (429, "quota exhausted"), (500, "500")],
)
def test_http_errors_are_reported_clearly(client, status, expected):
    client.session = FakeSession(FakeResponse({}, status_code=status))
    with pytest.raises(OddsAPIError, match=expected):
        client.get_odds("cricket_odi")


def test_a_failed_request_is_not_charged(client):
    client.session = FakeSession(FakeResponse({}, status_code=500))
    with pytest.raises(OddsAPIError):
        client.get_odds("cricket_odi")
    assert client.budget.state.spent == 0


def test_odds_requires_a_market_and_a_region(client):
    attach(client, [SAMPLE_EVENT])
    with pytest.raises(OddsAPIError):
        client.get_odds("cricket_odi", regions=[], markets=["h2h"])


# -- helpers -------------------------------------------------------------


def test_upcoming_within_filters_by_horizon():
    soon = _parse_event({**SAMPLE_EVENT, "commence_time": "2999-01-01T00:00:00Z"})
    assert upcoming_within([soon], hours=24) == []


def test_upcoming_within_excludes_events_already_started():
    past = _parse_event({**SAMPLE_EVENT, "commence_time": "2000-01-01T00:00:00Z"})
    assert upcoming_within([past], hours=24) == []

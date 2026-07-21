"""The Odds API v4 client, built for a 500-credit monthly allowance.

Two properties drive the design:

* ``/v4/sports`` and ``/v4/sports/{sport}/events`` cost nothing. Fixtures,
  schedules and event IDs are therefore free, and the entire schedule tracker
  runs without touching the budget.
* ``/v4/sports/{sport}/odds`` costs ``len(markets) * len(regions)``. Asking for
  one market in one region is 1 credit; asking for three of each is 9. Defaults
  here are deliberately minimal and widening them is an explicit choice.

Every priced call goes through :class:`CreditBudget`, and responses are cached
on disk so that a re-run inside the TTL is free.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from .budget import CreditBudget

BASE_URL = "https://api.the-odds-api.com/v4"

# Free endpoints, listed explicitly so a typo cannot silently make a call paid.
FREE_PATHS = ("/sports", "/events")

DEFAULT_TIMEOUT = 20

# Cricket and soccer sit in different books; uk+eu carries the best coverage for
# both. Each extra region multiplies cost, so this stays at one by default.
DEFAULT_REGIONS = ("uk",)

# Head-to-head (match winner) only. Totals and spreads each add a full
# multiple of the base cost.
DEFAULT_MARKETS = ("h2h",)


class OddsAPIError(RuntimeError):
    """Raised when the odds API cannot be reached or returns an error."""


@dataclass
class Outcome:
    name: str
    price: float
    point: float | None = None


@dataclass
class Bookmaker:
    key: str
    title: str
    last_update: str
    outcomes: list[Outcome] = field(default_factory=list)


@dataclass
class Event:
    id: str
    sport_key: str
    commence_time: str
    home_team: str | None
    away_team: str | None
    bookmakers: list[Bookmaker] = field(default_factory=list)

    @property
    def starts_at(self) -> datetime:
        return datetime.fromisoformat(self.commence_time.replace("Z", "+00:00"))

    @property
    def matchup(self) -> str:
        if self.home_team and self.away_team:
            return f"{self.home_team} vs {self.away_team}"
        return self.id

    def prices_for(self, market: str = "h2h") -> dict[str, list[float]]:
        """Map each outcome name to the prices offered across all bookmakers.

        Feed the per-book price vectors to ``devig.consensus_probabilities`` to
        get the market's true view rather than one book's.
        """
        collected: dict[str, list[float]] = {}
        for book in self.bookmakers:
            for outcome in book.outcomes:
                collected.setdefault(outcome.name, []).append(outcome.price)
        return collected

    def best_price(self, outcome_name: str) -> tuple[float, str] | None:
        """Highest available price for an outcome, and which book offers it.

        Always bet the best price: on a 2% edge, taking 1.95 instead of 2.00 is
        the difference between a positive and a negative expectation.
        """
        best: tuple[float, str] | None = None
        for book in self.bookmakers:
            for outcome in book.outcomes:
                if outcome.name == outcome_name:
                    if best is None or outcome.price > best[0]:
                        best = (outcome.price, book.title)
        return best


class OddsClient:
    """Budget-enforcing, disk-cached client for The Odds API v4."""

    def __init__(
        self,
        api_key: str,
        budget: CreditBudget,
        cache_dir: Path | str,
        cache_ttl_seconds: int = 900,
        base_url: str = BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise OddsAPIError(
                "no API key. Set ODDS_API_KEY in ~/Keys/.env.keys "
                "(free key: https://the-odds-api.com/#get-access)"
            )
        self.api_key = api_key
        self.budget = budget
        self.cache_dir = Path(cache_dir)
        self.cache_ttl = cache_ttl_seconds
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    # -- caching ---------------------------------------------------------

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        keyed = {k: v for k, v in sorted(params.items()) if k != "apiKey"}
        digest = hashlib.sha256(f"{path}|{json.dumps(keyed)}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{path.strip('/').replace('/', '_')}_{digest}.json"

    def _read_cache(self, cache_path: Path) -> Any | None:
        if not cache_path.exists():
            return None
        if time.time() - cache_path.stat().st_mtime > self.cache_ttl:
            return None
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            return None

    def _write_cache(self, cache_path: Path, payload: Any) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload))

    # -- transport -------------------------------------------------------

    def _is_free(self, path: str) -> bool:
        return any(path.endswith(suffix) for suffix in FREE_PATHS)

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        cost: int = 0,
        allow_reserve: bool = False,
        use_cache: bool = True,
    ) -> Any:
        params = dict(params or {})
        cache_path = self._cache_path(path, params)

        if use_cache:
            cached = self._read_cache(cache_path)
            if cached is not None:
                return cached

        if cost > 0:
            self.budget.check(cost, allow_reserve=allow_reserve)

        params["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OddsAPIError(f"request to {path} failed: {exc}") from exc

        if response.status_code == 401:
            raise OddsAPIError("odds API rejected the key (401). Check ODDS_API_KEY.")
        if response.status_code == 422:
            raise OddsAPIError(f"odds API rejected the parameters (422): {response.text[:200]}")
        if response.status_code == 429:
            raise OddsAPIError("odds API quota exhausted (429). Monthly credits are spent.")
        if not response.ok:
            raise OddsAPIError(f"odds API returned {response.status_code}: {response.text[:200]}")

        # Charge what the server says it charged, not what we predicted.
        actual = response.headers.get("x-requests-last")
        charged = int(actual) if actual and actual.isdigit() else cost
        if charged > 0:
            self.budget.charge(charged)

        remaining = response.headers.get("x-requests-remaining")
        if remaining:
            try:
                self.budget.reconcile(int(float(remaining)))
            except ValueError:
                pass

        try:
            payload = response.json()
        except ValueError as exc:
            raise OddsAPIError(f"non-JSON response from {path}") from exc

        if use_cache:
            self._write_cache(cache_path, payload)
        return payload

    # -- free endpoints --------------------------------------------------

    def list_sports(self, all_sports: bool = False) -> list[dict[str, Any]]:
        """All sport keys. Costs 0 credits."""
        params = {"all": "true"} if all_sports else {}
        return self._get("/sports", params, cost=0)

    def list_events(self, sport_key: str) -> list[Event]:
        """Upcoming fixtures for a sport, without prices. Costs 0 credits.

        This is what the schedule tracker runs on -- a full fixture board for
        every sport, refreshed as often as you like, for nothing.
        """
        payload = self._get(f"/sports/{sport_key}/events", cost=0)
        return [_parse_event(item) for item in payload]

    # -- priced endpoints ------------------------------------------------

    def get_odds(
        self,
        sport_key: str,
        regions: Sequence[str] = DEFAULT_REGIONS,
        markets: Sequence[str] = DEFAULT_MARKETS,
        odds_format: str = "decimal",
        allow_reserve: bool = False,
        use_cache: bool = True,
    ) -> list[Event]:
        """Odds for a sport's upcoming events.

        Costs ``len(markets) * len(regions)`` credits.
        """
        if not regions or not markets:
            raise OddsAPIError("at least one region and one market are required")

        cost = len(markets) * len(regions)
        payload = self._get(
            f"/sports/{sport_key}/odds",
            {
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
            },
            cost=cost,
            allow_reserve=allow_reserve,
            use_cache=use_cache,
        )
        return [_parse_event(item) for item in payload]

    def get_scores(
        self,
        sport_key: str,
        days_from: int | None = None,
        allow_reserve: bool = True,
    ) -> list[dict[str, Any]]:
        """Results for grading settled bets.

        Costs 1 credit, or 2 when ``days_from`` is supplied. Draws on the
        reserve by default, since grading a placed bet matters more than
        another discovery poll.
        """
        params: dict[str, Any] = {}
        cost = 1
        if days_from is not None:
            params["daysFrom"] = days_from
            cost = 2
        # Scores change during play, so never serve them from cache.
        return self._get(
            f"/sports/{sport_key}/scores",
            params,
            cost=cost,
            allow_reserve=allow_reserve,
            use_cache=False,
        )

    def estimate_cost(self, regions: Sequence[str], markets: Sequence[str]) -> int:
        return len(regions) * len(markets)


def _parse_event(item: dict[str, Any]) -> Event:
    bookmakers = []
    for book in item.get("bookmakers", []) or []:
        outcomes: list[Outcome] = []
        for market in book.get("markets", []) or []:
            for outcome in market.get("outcomes", []) or []:
                price = outcome.get("price")
                if price is None:
                    continue
                outcomes.append(
                    Outcome(
                        name=str(outcome.get("name", "")),
                        price=float(price),
                        point=outcome.get("point"),
                    )
                )
        bookmakers.append(
            Bookmaker(
                key=book.get("key", ""),
                title=book.get("title", book.get("key", "")),
                last_update=book.get("last_update", ""),
                outcomes=outcomes,
            )
        )

    return Event(
        id=item.get("id", ""),
        sport_key=item.get("sport_key", ""),
        commence_time=item.get("commence_time", ""),
        home_team=item.get("home_team"),
        away_team=item.get("away_team"),
        bookmakers=bookmakers,
    )


def upcoming_within(events: Iterable[Event], hours: float) -> list[Event]:
    """Filter events starting within the next ``hours``."""
    now = datetime.now(timezone.utc)
    out = []
    for event in events:
        try:
            delta = (event.starts_at - now).total_seconds() / 3600.0
        except ValueError:
            continue
        if 0 <= delta <= hours:
            out.append(event)
    return sorted(out, key=lambda e: e.commence_time)

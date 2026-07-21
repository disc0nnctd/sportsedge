"""A fixture board across every tracked competition.

Runs entirely on ``/v4/sports`` and ``/v4/sports/{sport}/events``, both of
which cost zero credits. The schedule can therefore refresh as often as you
like without touching the odds allowance -- and knowing what is coming is what
lets the priced calls be spent only on matches worth pricing.

Replaces the old blind 48-hour window, which in July shows nothing at all:
European football does not restart until late August, so a 48h lookahead made
both systems appear broken when they were merely early.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from .odds.theoddsapi import Event, OddsClient

# Sport keys on The Odds API. Cricket runs year-round on international tours;
# the European football keys are dormant until their seasons restart.
CRICKET_SPORTS = (
    "cricket_test_match",
    "cricket_odi",
    "cricket_international_t20",
    "cricket_ipl",
    "cricket_big_bash",
    "cricket_t20_blast",
    "cricket_psl",
    "cricket_the_hundred",
    "cricket_caribbean_premier_league",
)

FOOTBALL_SPORTS = (
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
)


@dataclass
class Fixture:
    sport_key: str
    sport_title: str
    event_id: str
    matchup: str
    starts_at: datetime

    @property
    def hours_away(self) -> float:
        return (self.starts_at - datetime.now(timezone.utc)).total_seconds() / 3600.0

    @property
    def is_live_or_past(self) -> bool:
        return self.hours_away < 0


def active_sports(
    client: OddsClient,
    prefixes: Sequence[str] = ("cricket", "soccer"),
) -> list[dict]:
    """Sport keys the API currently reports as in season. Costs 0 credits.

    ``/v4/sports`` only returns sports with upcoming fixtures, so this is the
    cheapest way to discover what is actually worth polling this week rather
    than looping over a hardcoded list of dormant competitions.
    """
    return [
        sport
        for sport in client.list_sports()
        if any(sport.get("key", "").startswith(prefix) for prefix in prefixes)
    ]


def fetch_fixtures(
    client: OddsClient,
    sport_keys: Iterable[str],
    days_ahead: int = 7,
) -> list[Fixture]:
    """Fixture list across several competitions. Costs 0 credits."""
    horizon = datetime.now(timezone.utc) + timedelta(days=days_ahead)
    now = datetime.now(timezone.utc)

    titles = {s.get("key"): s.get("title", s.get("key", "")) for s in client.list_sports()}

    fixtures: list[Fixture] = []
    for key in sport_keys:
        try:
            events = client.list_events(key)
        except Exception:
            # A dormant or misspelled competition should not sink the board.
            continue
        for event in events:
            try:
                starts = event.starts_at
            except ValueError:
                continue
            if now - timedelta(hours=6) <= starts <= horizon:
                fixtures.append(
                    Fixture(
                        sport_key=key,
                        sport_title=titles.get(key, key),
                        event_id=event.id,
                        matchup=event.matchup,
                        starts_at=starts,
                    )
                )

    return sorted(fixtures, key=lambda f: f.starts_at)


def group_by_day(
    fixtures: Sequence[Fixture],
    tz_offset_hours: float = 0.0,
) -> dict[str, list[Fixture]]:
    """Bucket fixtures by calendar day in the viewer's timezone.

    Grouping must use the same offset the times are rendered in. Bucketing by
    UTC while displaying IST files a 00:30 IST match under the previous day.
    """
    offset = timedelta(hours=tz_offset_hours)
    grouped: dict[str, list[Fixture]] = defaultdict(list)
    for fixture in fixtures:
        grouped[(fixture.starts_at + offset).strftime("%Y-%m-%d")].append(fixture)
    return dict(sorted(grouped.items()))


def render_board(
    fixtures: Sequence[Fixture],
    title: str = "Fixture board",
    tz_offset_hours: float = 5.5,
) -> str:
    """Render fixtures as a compact board, times shown in local time.

    Defaults to IST (+5:30).
    """
    if not fixtures:
        return f"**{title}** — no fixtures scheduled in this window."

    offset = timedelta(hours=tz_offset_hours)
    lines = [f"**{title}** — {len(fixtures)} fixtures"]

    for _, day_fixtures in group_by_day(fixtures, tz_offset_hours).items():
        local_day = (day_fixtures[0].starts_at + offset).strftime("%a %d %b")
        lines.append(f"\n__{local_day}__")
        for fixture in day_fixtures:
            local_time = (fixture.starts_at + offset).strftime("%H:%M")
            marker = "🔴" if fixture.is_live_or_past else "  "
            lines.append(
                f"{marker} `{local_time}` {fixture.matchup}  _{fixture.sport_title}_"
            )

    return "\n".join(lines)


def pricing_candidates(
    fixtures: Sequence[Fixture],
    within_hours: float = 36.0,
) -> list[Fixture]:
    """Fixtures close enough to be worth spending credits on.

    Odds far from kickoff are thin and move a lot, so pricing a match six days
    out wastes a credit on a number that will be stale by the time it matters.
    """
    return [f for f in fixtures if 0 <= f.hours_away <= within_hours]


def sports_to_poll(fixtures: Sequence[Fixture]) -> list[str]:
    """Distinct sport keys among the given fixtures, in fixture order.

    The odds endpoint is priced per sport, not per match, so one call covers
    every fixture in a competition. Polling only the competitions that have
    near-term fixtures is the single biggest credit saving available.
    """
    seen: list[str] = []
    for fixture in fixtures:
        if fixture.sport_key not in seen:
            seen.append(fixture.sport_key)
    return seen

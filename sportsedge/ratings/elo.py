"""Elo ratings with an explicit draw model.

Both original systems asked an LLM to produce a confidence number from a bare
prompt. That number has no anchor: nothing in the pipeline knew that a mid-table
home side beats a relegation candidate about 55% of the time, so the LLM was
free to say 85% and nothing contradicted it.

Elo supplies that anchor. It is cheap, needs only results, and is a genuinely
strong baseline -- good enough that the honest use of an LLM is to nudge it for
things ratings cannot see (injuries, rotation, weather) rather than to replace
it.

Two outcome models are provided:

:func:`three_way_probabilities`
    Davidson's model for sports with draws (football). A draw parameter ``nu``
    controls how much probability mass sits on the tie.

:func:`two_way_probabilities`
    Standard logistic Elo for sports decided outright (T20/ODI cricket).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Elo's scale constant. A 400-point gap means 10:1 expected odds.
SCALE = 400.0
LN10 = math.log(10.0)

DEFAULT_RATING = 1500.0

# How fast ratings move. 20 is a reasonable club-football default; sports with
# fewer matches per season warrant a higher K.
DEFAULT_K = 20.0

# Rating points of home advantage. ~65 is the long-run figure for European
# league football; it is much smaller in neutral-venue cricket.
DEFAULT_HOME_ADVANTAGE = 65.0

# Davidson draw parameter. Around 1.0 reproduces the ~25% draw rate seen in top
# European leagues at rating parity. Lower it for competitions that draw less.
DEFAULT_DRAW_NU = 1.0

MatchResult = Literal["home", "away", "draw"]


@dataclass
class Ratings:
    """A table of team ratings, persistable to JSON."""

    values: dict[str, float] = field(default_factory=dict)
    default: float = DEFAULT_RATING

    def get(self, team: str) -> float:
        return self.values.get(team, self.default)

    def set(self, team: str, rating: float) -> None:
        self.values[team] = rating

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.values.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"default": self.default, "values": self.values}, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "Ratings":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls(values=raw.get("values", {}), default=raw.get("default", DEFAULT_RATING))


def rating_difference(
    home_rating: float,
    away_rating: float,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    neutral_venue: bool = False,
) -> float:
    """Effective rating gap, in Elo points, from the home side's perspective."""
    advantage = 0.0 if neutral_venue else home_advantage
    return (home_rating + advantage) - away_rating


def expected_score(rating_diff: float) -> float:
    """Classic Elo expectation in [0, 1], counting a draw as half a win."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / SCALE))


def two_way_probabilities(rating_diff: float) -> tuple[float, float]:
    """(home win, away win) for a sport with no draws."""
    home = expected_score(rating_diff)
    return home, 1.0 - home


def three_way_probabilities(
    rating_diff: float,
    draw_nu: float = DEFAULT_DRAW_NU,
) -> tuple[float, float, float]:
    """(home win, draw, away win) under Davidson's model.

    With ``d`` the rating gap on the natural-log scale:

        P(home) = exp(d/2)          / Z
        P(draw) = nu                / Z
        P(away) = exp(-d/2)         / Z

    where ``Z`` normalises the three. Draw probability peaks at rating parity
    and falls away as the mismatch grows, which is the behaviour real football
    results show.
    """
    if draw_nu <= 0:
        raise ValueError(f"draw_nu must be positive, got {draw_nu}")

    d = rating_diff * LN10 / SCALE
    home_term = math.exp(d / 2.0)
    away_term = math.exp(-d / 2.0)
    total = home_term + away_term + draw_nu
    return home_term / total, draw_nu / total, away_term / total


def update(
    ratings: Ratings,
    home_team: str,
    away_team: str,
    result: MatchResult,
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    neutral_venue: bool = False,
    margin_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Apply one result and return the two new ratings.

    ``margin_multiplier`` scales K for blowouts; see :func:`goal_margin_multiplier`.
    """
    if result not in ("home", "away", "draw"):
        raise ValueError(f"invalid result {result!r}")

    home_rating = ratings.get(home_team)
    away_rating = ratings.get(away_team)
    diff = rating_difference(home_rating, away_rating, home_advantage, neutral_venue)
    expected = expected_score(diff)

    actual = {"home": 1.0, "draw": 0.5, "away": 0.0}[result]
    delta = k * margin_multiplier * (actual - expected)

    new_home = home_rating + delta
    new_away = away_rating - delta
    ratings.set(home_team, new_home)
    ratings.set(away_team, new_away)
    return new_home, new_away


def goal_margin_multiplier(goal_difference: int) -> float:
    """Dampened weight for the margin of victory.

    A 4-0 win is stronger evidence than a 1-0, but not four times as strong,
    and the log keeps a single thrashing from distorting a rating.
    """
    return math.log(abs(goal_difference) + 1.0) if goal_difference else 1.0


def fit(
    ratings: Ratings,
    matches: list[dict],
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    use_margin: bool = True,
) -> Ratings:
    """Replay a chronological list of results to build ratings.

    Each match is a dict with ``home``, ``away``, ``result`` and optionally
    ``goal_difference`` and ``neutral``. Matches must already be in date order:
    Elo is path-dependent, so shuffling them produces different ratings.
    """
    for match in matches:
        multiplier = 1.0
        if use_margin and match.get("goal_difference") is not None:
            multiplier = goal_margin_multiplier(int(match["goal_difference"]))
        update(
            ratings,
            match["home"],
            match["away"],
            match["result"],
            k=k,
            home_advantage=home_advantage,
            neutral_venue=bool(match.get("neutral", False)),
            margin_multiplier=multiplier,
        )
    return ratings


def regress_to_mean(ratings: Ratings, weight: float = 0.25) -> Ratings:
    """Pull ratings toward the mean between seasons.

    Squads change over a summer, so last season's rating overstates what is
    known about this season's team. A 25% pull is a common default.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    for team, rating in ratings.values.items():
        ratings.values[team] = rating + weight * (ratings.default - rating)
    return ratings

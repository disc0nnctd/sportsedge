"""Command line interface for the shared betting pipeline.

    sportsedge budget                  # credits left this month (free)
    sportsedge schedule --days 7       # fixture board (free)
    sportsedge price   --sport KEY     # devigged market probabilities (1 credit)
    sportsedge picks   --sport KEY     # find +EV bets and record them (1 credit)
    sportsedge close                   # capture closing lines for pending picks
    sportsedge settle  --sport KEY     # grade finished matches
    sportsedge report                  # P&L, CLV and calibration

Commands that cost credits print their cost before spending and honour
``--dry-run``. Everything reads ``ODDS_API_KEY`` from ~/Keys/.env.keys; pass
``--test-key`` to use ``ODDS_API_KEY_TEST`` instead and leave production
credits untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import calibration, teams
from .edge import DEFAULT_KELLY_FRACTION, DEFAULT_MIN_EDGE, blend, evaluate_bet
from .ledger import Ledger
from .odds.budget import BudgetExceeded, CreditBudget
from .odds.devig import DevigError, consensus_probabilities
from .odds.theoddsapi import Event, OddsAPIError, OddsClient
from .ratings.elo import (
    DEFAULT_DRAW_NU,
    Ratings,
    rating_difference,
    three_way_probabilities,
    two_way_probabilities,
)
from .schedule import (
    CRICKET_SPORTS,
    FOOTBALL_SPORTS,
    fetch_fixtures,
    pricing_candidates,
    render_board,
    sports_to_poll,
)

DATA_DIR = Path(os.getenv("SPORTSEDGE_HOME", Path.home() / ".sportsedge"))

# Draws exist in football and in Test cricket, but not in limited-overs cricket.
# The 2-way path is used unless a sport is known to produce draws.
DRAW_SPORTS = ("soccer_", "cricket_test")


def _load_keys() -> None:
    keys_file = Path.home() / "Keys" / ".env.keys"
    if not keys_file.exists():
        return
    for line in keys_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _client(args) -> OddsClient:
    _load_keys()
    env_name = "ODDS_API_KEY_TEST" if getattr(args, "test_key", False) else "ODDS_API_KEY"
    api_key = os.getenv(env_name, "")
    if not api_key:
        raise SystemExit(f"{env_name} is not set in ~/Keys/.env.keys")

    suffix = "-test" if getattr(args, "test_key", False) else ""
    budget = CreditBudget(DATA_DIR / f"budget{suffix}.json", limit=args.limit)
    return OddsClient(api_key, budget, cache_dir=DATA_DIR / f"cache{suffix}")


def _ledger() -> Ledger:
    return Ledger(DATA_DIR / "picks.jsonl")


def _has_draws(sport_key: str) -> bool:
    return any(sport_key.startswith(prefix) for prefix in DRAW_SPORTS)


def _ratings_for(sport_key: str, _cache: dict[str, Ratings] = {}) -> Ratings:
    """Ratings for one competition, falling back to the shared table.

    Cricket needs a separate table per format: a side's T20 strength says
    little about its Test strength, so pooling them produces a rating that is
    wrong for both. ``build_cricket_ratings.py`` writes one file per sport key;
    football uses the shared ``ratings.json``.
    """
    if sport_key not in _cache:
        specific = DATA_DIR / f"ratings-{sport_key}.json"
        path = specific if specific.exists() else DATA_DIR / "ratings.json"
        _cache[sport_key] = Ratings.load(path)
    return _cache[sport_key]


def _any_ratings_available() -> bool:
    return bool(list(DATA_DIR.glob("ratings*.json")))


def _draw_nu() -> float:
    """Draw parameter fitted by ``build_ratings.py``, or the package default.

    Using the default when a fitted value exists silently reintroduces the draw
    bias the fit was performed to remove.
    """
    path = DATA_DIR / "draw_nu.json"
    if path.exists():
        try:
            value = float(json.loads(path.read_text())["draw_nu"])
            if value > 0:
                return value
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return DEFAULT_DRAW_NU


def _sport_keys(args) -> list[str]:
    if args.sport:
        return list(args.sport)
    keys: list[str] = []
    if args.cricket or not args.football:
        keys.extend(CRICKET_SPORTS)
    if args.football or not args.cricket:
        keys.extend(FOOTBALL_SPORTS)
    return keys


def _market_view(event: Event) -> dict[str, float] | None:
    """Devigged consensus probability for each outcome, across all books.

    Returns None when the event has too few books or a malformed market.
    """
    prices = event.prices_for("h2h")
    if len(prices) < 2:
        return None

    names = list(prices.keys())
    # Build one price vector per book; only books quoting every outcome count.
    depth = min(len(v) for v in prices.values())
    if depth < 1:
        return None

    books = []
    for i in range(depth):
        books.append([prices[name][i] for name in names])

    try:
        probabilities = consensus_probabilities(books, method="shin")
    except DevigError:
        return None
    return dict(zip(names, probabilities))


def _model_view(event: Event, ratings: Ratings, market: dict[str, float]) -> dict[str, float] | None:
    """Elo prior for the event, or None when either team is unrated.

    An unrated team means Elo has nothing to say, and inventing a number there
    is exactly the failure this replaced. Better to skip the match.
    """
    home, away = event.home_team, event.away_team
    if not home or not away:
        return None

    # Match on normalised names: the odds feed and the results feed spell the
    # same club differently ("Arsenal" vs "Arsenal FC"), and an exact lookup
    # silently skips every fixture.
    home_rating = teams.lookup(home, ratings.values)
    away_rating = teams.lookup(away, ratings.values)
    if home_rating is None or away_rating is None:
        return None

    neutral = _has_draws(event.sport_key) is False  # cricket tours are often neutral
    diff = rating_difference(home_rating, away_rating, neutral_venue=neutral)

    if _has_draws(event.sport_key):
        p_home, p_draw, p_away = three_way_probabilities(diff, draw_nu=_draw_nu())
        view = {home: p_home, away: p_away, "Draw": p_draw}
    else:
        p_home, p_away = two_way_probabilities(diff)
        view = {home: p_home, away: p_away}

    # Only keep outcomes the market actually quotes.
    return {name: p for name, p in view.items() if name in market}


# -- commands ------------------------------------------------------------


def cmd_budget(args) -> int:
    client = _client(args)
    print(client.budget.report())
    return 0


def cmd_schedule(args) -> int:
    client = _client(args)
    keys = _sport_keys(args)
    fixtures = fetch_fixtures(client, keys, days_ahead=args.days)
    print(render_board(fixtures, title=f"Fixtures — next {args.days}d"))
    print()
    near = pricing_candidates(fixtures, within_hours=args.within)
    if near:
        polls = sports_to_poll(near)
        print(
            f"{len(near)} fixture(s) within {args.within:.0f}h across {len(polls)} competition(s) "
            f"— pricing them all costs {len(polls)} credit(s)."
        )
    else:
        print(f"Nothing within {args.within:.0f}h. No credits need spending.")
    print(client.budget.report())
    return 0


def cmd_price(args) -> int:
    client = _client(args)
    keys = _sport_keys(args)

    if args.dry_run:
        print(f"would price {len(keys)} competition(s) at 1 credit each = {len(keys)} credit(s)")
        return 0

    for key in keys:
        try:
            events = client.get_odds(key, regions=[args.region], markets=["h2h"])
        except BudgetExceeded as exc:
            print(f"[budget] {exc}", file=sys.stderr)
            break
        except OddsAPIError as exc:
            print(f"[warn] {key}: {exc}", file=sys.stderr)
            continue

        if not events:
            continue
        print(f"\n=== {key} ===")
        for event in events:
            market = _market_view(event)
            if not market:
                continue
            print(f"\n{event.matchup}  ({event.commence_time})")
            for name, probability in sorted(market.items(), key=lambda kv: -kv[1]):
                best = event.best_price(name)
                price = f"{best[0]:.2f} @ {best[1]}" if best else "n/a"
                print(f"   {name:<28} {probability:6.1%}   best {price}")

    print(f"\n{client.budget.report()}")
    return 0


def cmd_picks(args) -> int:
    """Find +EV bets by comparing an Elo-anchored model to the devigged market."""
    client = _client(args)
    ledger = _ledger()

    if not _any_ratings_available():
        print(
            "No Elo ratings found in ~/.sportsedge/.\n"
            "Without a rated prior there is no model to compare against the market, "
            "so no picks can be made. Build ratings first:\n"
            "  football-kg/build_ratings.py          (football-data.org)\n"
            "  cricket-kg/build_cricket_ratings.py   (Cricsheet, no key needed)",
            file=sys.stderr,
        )
        return 1

    keys = _sport_keys(args)
    if args.dry_run:
        print(f"would price {len(keys)} competition(s) = {len(keys)} credit(s)")
        return 0

    found = 0
    unrated: set[str] = set()
    for key in keys:
        try:
            events = client.get_odds(key, regions=[args.region], markets=["h2h"])
        except BudgetExceeded as exc:
            print(f"[budget] {exc}", file=sys.stderr)
            break
        except OddsAPIError as exc:
            print(f"[warn] {key}: {exc}", file=sys.stderr)
            continue

        ratings = _ratings_for(key)
        for event in events:
            market = _market_view(event)
            if not market:
                continue
            model = _model_view(event, ratings, market)
            if not model:
                # Record which side is unrated so a silent skip is visible.
                for name in (event.home_team, event.away_team):
                    if name and teams.lookup(name, ratings.values) is None:
                        unrated.add(name)
                continue

            for name, raw_probability in model.items():
                market_probability = market[name]
                # Shrink toward the market: an unproven model gets a small vote.
                probability = blend(raw_probability, market_probability, args.model_weight)
                best = event.best_price(name)
                if not best:
                    continue
                price, book = best

                sizing = evaluate_bet(
                    probability,
                    market_probability,
                    price,
                    min_edge=args.min_edge,
                    kelly_multiplier=args.kelly,
                )
                if not sizing.is_bet:
                    continue

                found += 1
                stake = sizing.stake(args.bankroll)
                print(
                    f"\n{event.matchup} — {name} @ {price:.2f} ({book})\n"
                    f"   model {probability:.1%} vs market {market_probability:.1%}"
                    f"  → edge {sizing.edge:+.1%}, EV {sizing.expected_value:+.1%}\n"
                    f"   stake {stake:,.0f} ({sizing.stake_fraction:.2%} of bankroll) — {sizing.reason}"
                )
                if args.record:
                    pick = ledger.record(
                        sport=key,
                        event_id=event.id,
                        matchup=event.matchup,
                        commence_time=event.commence_time,
                        market="h2h",
                        selection=name,
                        model_probability=probability,
                        market_probability=market_probability,
                        decimal_odds=price,
                        bookmaker=book,
                        edge=sizing.edge,
                        stake_fraction=sizing.stake_fraction,
                        stake=stake,
                        rationale=f"elo {raw_probability:.3f} blended at w={args.model_weight}",
                    )
                    print(f"   recorded as {pick.id}")

    if not found:
        print("No bets clear the edge threshold. That is the expected result most days.")

    if unrated:
        print(
            f"\nSkipped {len(unrated)} unrated team(s): {', '.join(sorted(unrated))}\n"
            "Promoted or lower-division sides legitimately have no rating. If a "
            "top-flight club appears here it is a name-matching gap — add it to "
            "sportsedge.teams.ALIASES rather than letting the fixture drop."
        )

    print(f"\n{client.budget.report()}")
    return 0


def cmd_close(args) -> int:
    """Capture closing lines for picks whose match is about to start."""
    client = _client(args)
    ledger = _ledger()
    pending = ledger.pending()
    if not pending:
        print("No pending picks.")
        return 0

    now = datetime.now(timezone.utc)
    due = []
    for pick in pending:
        if pick.closing_odds is not None:
            continue
        try:
            starts = datetime.fromisoformat(pick.commence_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        hours = (starts - now).total_seconds() / 3600.0
        if -1.0 <= hours <= args.within:
            due.append((pick, starts))

    if not due:
        print(f"No picks starting within {args.within:.0f}h need a closing line.")
        return 0

    sports = sorted({pick.sport for pick, _ in due})
    print(f"{len(due)} pick(s) across {len(sports)} competition(s) = {len(sports)} credit(s)")
    if args.dry_run:
        return 0

    for sport in sports:
        try:
            events = client.get_odds(
                sport, regions=[args.region], markets=["h2h"],
                allow_reserve=True, use_cache=False,
            )
        except (BudgetExceeded, OddsAPIError) as exc:
            print(f"[warn] {sport}: {exc}", file=sys.stderr)
            continue

        by_id = {event.id: event for event in events}
        for pick, _ in due:
            if pick.sport != sport:
                continue
            event = by_id.get(pick.event_id)
            if not event:
                continue
            market = _market_view(event)
            best = event.best_price(pick.selection)
            if not market or pick.selection not in market or not best:
                continue
            ledger.set_closing_line(pick.id, best[0], market[pick.selection])
            moved = market[pick.selection] - pick.market_probability
            print(f"{pick.matchup} — {pick.selection}: close {best[0]:.2f}, CLV {moved:+.3f}")

    print(f"\n{client.budget.report()}")
    return 0


def cmd_settle(args) -> int:
    """Grade pending picks against finished-match scores."""
    client = _client(args)
    ledger = _ledger()
    pending = ledger.pending()
    if not pending:
        print("No pending picks.")
        return 0

    sports = sorted({pick.sport for pick in pending})
    print(f"grading {len(pending)} pick(s) across {len(sports)} competition(s)")
    if args.dry_run:
        return 0

    settled = 0
    for sport in sports:
        try:
            scores = client.get_scores(sport, days_from=args.days_from)
        except (BudgetExceeded, OddsAPIError) as exc:
            print(f"[warn] {sport}: {exc}", file=sys.stderr)
            continue

        finished = {s.get("id"): s for s in scores if s.get("completed")}
        for pick in pending:
            if pick.sport != sport or pick.event_id not in finished:
                continue
            game = finished[pick.event_id]
            winner = _winner_from_scores(game)
            if winner is None:
                continue
            result = "won" if winner == pick.selection else "lost"
            ledger.settle(pick.id, result, notes=f"scores: {game.get('scores')}")
            settled += 1
            print(f"{pick.matchup} — {pick.selection}: {result.upper()}")

    print(f"\nsettled {settled} pick(s)")
    print(client.budget.report())
    return 0


def _winner_from_scores(game: dict) -> str | None:
    """Team name with the higher score, or None if it cannot be determined."""
    scores = game.get("scores") or []
    parsed: list[tuple[str, float]] = []
    for entry in scores:
        name = entry.get("name")
        value = entry.get("score")
        if name is None or value is None:
            continue
        try:
            parsed.append((name, float(value)))
        except (TypeError, ValueError):
            continue
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda kv: kv[1], reverse=True)
    if parsed[0][1] == parsed[1][1]:
        return "Draw"
    return parsed[0][0]


def cmd_report(args) -> int:
    ledger = _ledger()
    summary = ledger.summarise(sport=args.filter_sport)
    print("=== Ledger ===")
    print(summary.summary())

    forecasts, outcomes, market = ledger.calibration_inputs(sport=args.filter_sport)
    print("\n=== Calibration ===")
    if len(forecasts) < 2:
        print(f"only {len(forecasts)} settled pick(s) — need more before scoring means anything")
        return 0

    report = calibration.evaluate(forecasts, outcomes, market)
    print(report.summary())

    if report.bins:
        print("\nreliability:")
        for b in report.bins:
            print(
                f"  {b.lower:.1f}-{b.upper:.1f}  n={b.count:<4} "
                f"predicted {b.mean_forecast:.2f}  actual {b.observed_rate:.2f}  gap {b.gap:+.2f}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sportsedge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, sports=True):
        p.add_argument("--limit", type=int, default=500, help="monthly credit allowance")
        p.add_argument("--test-key", action="store_true", help="use ODDS_API_KEY_TEST")
        if sports:
            p.add_argument("--sport", action="append", help="explicit sport key (repeatable)")
            p.add_argument("--cricket", action="store_true", help="cricket only")
            p.add_argument("--football", action="store_true", help="football only")
            p.add_argument("--region", default="uk", help="bookmaker region")

    p = sub.add_parser("budget", help="show remaining credits")
    common(p, sports=False)
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("schedule", help="fixture board (free)")
    common(p)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--within", type=float, default=36.0)
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("price", help="show devigged market probabilities")
    common(p)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("picks", help="find and record +EV bets")
    common(p)
    p.add_argument("--bankroll", type=float, default=10_000.0)
    p.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    p.add_argument("--kelly", type=float, default=DEFAULT_KELLY_FRACTION)
    p.add_argument("--model-weight", type=float, default=0.25)
    p.add_argument("--record", action="store_true", help="write picks to the ledger")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_picks)

    p = sub.add_parser("close", help="capture closing lines")
    common(p)
    p.add_argument("--within", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_close)

    p = sub.add_parser("settle", help="grade finished picks")
    common(p)
    p.add_argument("--days-from", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_settle)

    p = sub.add_parser("report", help="P&L, CLV and calibration")
    p.add_argument("--filter-sport", default=None)
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OddsAPIError, BudgetExceeded) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

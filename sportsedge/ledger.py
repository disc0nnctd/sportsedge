"""An append-only record of every pick, its price, and what happened next.

The old cricket allocator wrote a JSON blob per match with no record of the
market price at the time of the pick, which makes the results unauditable: you
cannot tell afterwards whether a winning bet was skill or a lucky number.

Every pick here stores the model probability, the devigged market probability
and the exact price taken. That makes two things computable later:

* **Closing line value** -- did the price move toward the pick after it was
  made? Beating the closing line is the earliest reliable evidence of edge,
  and it converges in dozens of bets rather than thousands.
* **Calibration** -- were the probabilities honest? Scored by
  ``calibration.evaluate`` against the market as reference.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

Result = Literal["pending", "won", "lost", "push", "void"]


@dataclass
class Pick:
    """One recorded bet."""

    id: str
    sport: str
    event_id: str
    matchup: str
    commence_time: str
    market: str
    selection: str

    model_probability: float
    market_probability: float
    decimal_odds: float
    bookmaker: str
    edge: float
    stake_fraction: float
    stake: float

    placed_at: str
    rationale: str = ""

    # Filled in later.
    closing_odds: float | None = None
    closing_probability: float | None = None
    result: Result = "pending"
    payout: float = 0.0
    settled_at: str | None = None
    notes: str = ""

    @property
    def profit(self) -> float:
        if self.result == "won":
            return self.stake * (self.decimal_odds - 1.0)
        if self.result == "lost":
            return -self.stake
        return 0.0  # push, void, or pending

    @property
    def outcome_bit(self) -> int | None:
        """1 if the pick won, 0 if it lost, None if not settled either way."""
        if self.result == "won":
            return 1
        if self.result == "lost":
            return 0
        return None

    @property
    def closing_line_value(self) -> float | None:
        """Probability points gained against the close.

        Positive means the market moved toward the pick after it was made --
        the price taken was better than the price everyone settled on.
        """
        if self.closing_probability is None:
            return None
        return self.closing_probability - self.market_probability

    @property
    def beat_the_close(self) -> bool | None:
        if self.closing_odds is None:
            return None
        return self.decimal_odds > self.closing_odds


@dataclass
class LedgerSummary:
    n_picks: int
    n_settled: int
    staked: float
    profit: float
    wins: int
    losses: int
    pending: int
    clv_measured: int = 0
    mean_clv: float | None = None
    beat_close_rate: float | None = None

    @property
    def roi(self) -> float:
        return self.profit / self.staked if self.staked else 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    def summary(self) -> str:
        lines = [
            f"{self.n_picks} picks ({self.n_settled} settled, {self.pending} pending)",
            f"staked {self.staked:,.2f}  profit {self.profit:+,.2f}  ROI {self.roi:+.2%}",
            f"record {self.wins}-{self.losses} ({self.win_rate:.1%})",
        ]
        if self.mean_clv is not None:
            verdict = "positive" if self.mean_clv > 0 else "negative"
            lines.append(
                f"CLV {self.mean_clv:+.4f} over {self.clv_measured} picks "
                f"({self.beat_close_rate:.1%} beat the close) -- {verdict}"
            )
        else:
            lines.append("CLV not yet measurable (no closing lines captured)")
        return "\n".join(lines)


class Ledger:
    """JSONL-backed pick store. Append-only by design; edits rewrite in place."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _read_all(self) -> list[Pick]:
        if not self.path.exists():
            return []
        picks = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                picks.append(Pick(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"corrupt ledger line in {self.path}: {exc}") from exc
        return picks

    def _write_all(self, picks: Iterable[Pick]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(asdict(p)) + "\n" for p in picks))

    def all(self) -> list[Pick]:
        return self._read_all()

    def pending(self) -> list[Pick]:
        return [p for p in self._read_all() if p.result == "pending"]

    def get(self, pick_id: str) -> Pick | None:
        for pick in self._read_all():
            if pick.id == pick_id:
                return pick
        return None

    def record(
        self,
        *,
        sport: str,
        event_id: str,
        matchup: str,
        commence_time: str,
        market: str,
        selection: str,
        model_probability: float,
        market_probability: float,
        decimal_odds: float,
        bookmaker: str,
        edge: float,
        stake_fraction: float,
        stake: float,
        rationale: str = "",
    ) -> Pick:
        """Append a new pick and return it."""
        pick = Pick(
            id=uuid.uuid4().hex[:12],
            sport=sport,
            event_id=event_id,
            matchup=matchup,
            commence_time=commence_time,
            market=market,
            selection=selection,
            model_probability=model_probability,
            market_probability=market_probability,
            decimal_odds=decimal_odds,
            bookmaker=bookmaker,
            edge=edge,
            stake_fraction=stake_fraction,
            stake=stake,
            placed_at=datetime.now(timezone.utc).isoformat(),
            rationale=rationale,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(asdict(pick)) + "\n")
        return pick

    def _update(self, pick_id: str, **changes: Any) -> Pick:
        picks = self._read_all()
        for pick in picks:
            if pick.id == pick_id:
                for key, value in changes.items():
                    setattr(pick, key, value)
                self._write_all(picks)
                return pick
        raise KeyError(f"no pick with id {pick_id!r} in {self.path}")

    def set_closing_line(
        self,
        pick_id: str,
        closing_odds: float,
        closing_probability: float,
    ) -> Pick:
        """Attach the closing price. Call as close to kickoff as the budget allows."""
        return self._update(
            pick_id,
            closing_odds=closing_odds,
            closing_probability=closing_probability,
        )

    def settle(self, pick_id: str, result: Result, notes: str = "") -> Pick:
        """Grade a pick once the match has finished."""
        if result not in ("won", "lost", "push", "void"):
            raise ValueError(f"invalid result {result!r}")
        pick = self.get(pick_id)
        if pick is None:
            raise KeyError(f"no pick with id {pick_id!r}")

        if result == "won":
            payout = pick.stake * pick.decimal_odds
        elif result in ("push", "void"):
            payout = pick.stake
        else:
            payout = 0.0

        return self._update(
            pick_id,
            result=result,
            payout=payout,
            settled_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

    def summarise(self, sport: str | None = None) -> LedgerSummary:
        picks = [p for p in self._read_all() if sport is None or p.sport == sport]
        settled = [p for p in picks if p.result != "pending"]
        with_clv = [p for p in picks if p.closing_line_value is not None]

        summary = LedgerSummary(
            n_picks=len(picks),
            n_settled=len(settled),
            staked=sum(p.stake for p in settled),
            profit=sum(p.profit for p in settled),
            wins=sum(1 for p in picks if p.result == "won"),
            losses=sum(1 for p in picks if p.result == "lost"),
            pending=sum(1 for p in picks if p.result == "pending"),
            clv_measured=len(with_clv),
        )

        if with_clv:
            summary.mean_clv = sum(p.closing_line_value for p in with_clv) / len(with_clv)
            beat = sum(1 for p in with_clv if p.beat_the_close)
            summary.beat_close_rate = beat / len(with_clv)

        return summary

    def calibration_inputs(
        self, sport: str | None = None
    ) -> tuple[list[float], list[int], list[float]]:
        """Extract (model probabilities, outcomes, market probabilities).

        Feed straight into ``calibration.evaluate(forecasts, outcomes, market)``
        to find out whether the model beats the price it was betting against.
        """
        forecasts, outcomes, market = [], [], []
        for pick in self._read_all():
            if sport is not None and pick.sport != sport:
                continue
            bit = pick.outcome_bit
            if bit is None:
                continue
            forecasts.append(pick.model_probability)
            outcomes.append(bit)
            market.append(pick.market_probability)
        return forecasts, outcomes, market

"""Credit accounting for a metered odds API.

On The Odds API's free tier the monthly allowance is 500 credits. A single
careless loop -- six sports x three markets x three regions -- burns 54 of them
in one pass, and a cron doing that twice a day exhausts the month in under a
week. So the budget is enforced in code rather than left to discipline: the
client refuses to spend when the ledger says there is nothing left.

Spend is tracked locally and reconciled against the ``x-requests-remaining``
header on every response, so the count self-corrects if requests are made from
somewhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path

# The Odds API resets usage on the 1st of each month (UTC).
FREE_TIER_MONTHLY_CREDITS = 500

# Hold part of the allowance back for grading and closing-line snapshots, which
# happen at fixed times and must not be starved by discovery polling earlier in
# the month.
DEFAULT_RESERVE_FRACTION = 0.25


class BudgetExceeded(RuntimeError):
    """Raised when a request would exceed the remaining credit allowance."""


@dataclass
class BudgetState:
    period: str  # "YYYY-MM"
    spent: int
    limit: int
    reserved: int
    last_updated: str

    @classmethod
    def fresh(cls, limit: int, reserve_fraction: float, period: str) -> "BudgetState":
        return cls(
            period=period,
            spent=0,
            limit=limit,
            reserved=int(limit * reserve_fraction),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def discretionary(self) -> int:
        """Credits available for ordinary polling, ignoring the reserve."""
        return max(0, self.limit - self.reserved - self.spent)


def current_period() -> str:
    today = date.today()
    return f"{today.year:04d}-{today.month:02d}"


class CreditBudget:
    """A persistent monthly credit counter.

    ``reserved`` credits can only be spent by calls that pass
    ``allow_reserve=True`` -- used for grading settled bets and capturing
    closing lines, which are worth more than another round of discovery.
    """

    def __init__(
        self,
        path: Path | str,
        limit: int = FREE_TIER_MONTHLY_CREDITS,
        reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    ) -> None:
        self.path = Path(path)
        self.limit = limit
        self.reserve_fraction = reserve_fraction
        self.state = self._load()

    def _load(self) -> BudgetState:
        period = current_period()
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                state = BudgetState(**raw)
                if state.period == period:
                    # Honour a limit change made since the file was written.
                    state.limit = self.limit
                    state.reserved = int(self.limit * self.reserve_fraction)
                    return state
            except (json.JSONDecodeError, TypeError, ValueError):
                # A corrupt ledger should not block the month; start clean.
                pass
        return BudgetState.fresh(self.limit, self.reserve_fraction, period)

    def save(self) -> None:
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.state), indent=2))

    def check(self, cost: int, allow_reserve: bool = False) -> None:
        """Raise if spending ``cost`` credits is not permitted."""
        available = self.state.remaining if allow_reserve else self.state.discretionary
        if cost > available:
            pool = "total" if allow_reserve else "discretionary"
            raise BudgetExceeded(
                f"request costs {cost} credit(s) but only {available} {pool} "
                f"credit(s) remain this period ({self.state.spent}/{self.state.limit} used). "
                "Wait for the monthly reset, narrow markets/regions, or raise the limit."
            )

    def charge(self, cost: int) -> None:
        self.state.spent += cost
        self.save()

    def reconcile(self, remaining_from_header: int) -> None:
        """Trust the server's count over the local one.

        Keeps the ledger honest when requests are made outside this process.
        """
        server_spent = self.state.limit - remaining_from_header
        if server_spent > self.state.spent:
            self.state.spent = server_spent
            self.save()

    def report(self) -> str:
        s = self.state
        return (
            f"credits {s.spent}/{s.limit} used this period ({s.period}); "
            f"{s.discretionary} discretionary + {min(s.reserved, s.remaining)} reserved remain"
        )

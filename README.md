# sportsedge

Shared betting-analytics core for [`cricket-kg`](https://github.com/disc0nnctd/cricket-kg)
and [`football-kg`](https://github.com/disc0nnctd/football-kg).

## Why this exists

Both sport repos previously ended at *"verdict + LLM confidence"*. That is a
prediction system, not a betting system: it cannot tell an 80%-confident pick at
1.10 (a bad bet — the market implies 91%) from a 30%-confident pick at 5.00 (a
good one). Neither repo fetched a price, so neither could compute an edge.

This package supplies the missing half:

```
fixtures (free)  ->  odds  ->  devig  ->  model probability  ->  edge/Kelly
                                                                    |
                          ledger  <-  closing line  <-  settle  <---+
                              |
                        calibration
```

Nothing here knows which sport it is scoring. Sport-specific logic stays in the
consuming repos.

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Requires `ODDS_API_KEY` in `~/Keys/.env.keys`. `ODDS_API_KEY_TEST` is used by
`--test-key` so experiments never touch the production allowance.

## Credit budget

The Odds API free tier is **500 credits/month**, resetting on the 1st (UTC).
Costs are not uniform, and the design leans hard on that:

| Endpoint | Cost |
|---|---|
| `/sports` | **0** |
| `/sports/{sport}/events` | **0** |
| `/sports/{sport}/odds` | `markets × regions` |
| `/sports/{sport}/scores` | 1 (2 with `daysFrom`) |

The entire schedule tracker runs on the free endpoints, so every credit goes to
actual prices. Defaults are one market (`h2h`) in one region (`uk`) = 1 credit
per competition polled. Widening either multiplies the cost.

`CreditBudget` enforces this in code rather than by discipline: it refuses to
spend when the ledger is empty, reconciles against the `x-requests-remaining`
header, and holds 25% back so grading and closing-line capture cannot be starved
by discovery polling earlier in the month.

## Commands

```bash
sportsedge budget                 # credits remaining          (free)
sportsedge schedule --days 7      # fixture board              (free)
sportsedge price --sport KEY      # devigged market view       (1 credit)
sportsedge picks --record         # find and log +EV bets      (1 credit/comp)
sportsedge close                  # capture closing lines      (1 credit/comp)
sportsedge settle                 # grade finished matches     (2 credits/comp)
sportsedge report                 # P&L, CLV, calibration      (free)
```

Anything that spends prints its cost first and honours `--dry-run`.

## Design notes

**De-vig properly.** Comparing a model probability against raw `1/odds` bakes
the bookmaker's margin into the edge and manufactures edges that do not exist.
Default method is Shin; `power` and `multiplicative` are available. Multiplicative
is provided as a baseline only — it assumes even margin, which leaves longshots
overpriced, and that is precisely where a naive model thinks it has found value.

**Shrink toward the market.** The closing line is a strong, well-calibrated
forecast aggregating far more information than any single model. `blend()`
defaults to giving the model 25% weight. Raise it only when
`calibration.brier_skill_score` against the market earns it.

**Stake defensively.** Quarter Kelly, a 2pp minimum edge, and a hard 5% bankroll
cap. Full Kelly is optimal only if the probability is exactly right; it never is,
and overbetting is punished far more than underbetting. The cap binds at a 10pp
claimed edge — well below what an LLM will routinely assert.

**Skip what you cannot rate.** `picks` refuses to bet a match where either team
is unrated, rather than inventing a number. That was the original failure mode.

**Record the price, not just the pick.** Every ledger entry stores the model
probability, the devigged market probability and the exact price taken, so
results are auditable after the fact. Closing-line value converges in dozens of
bets; P&L needs thousands.

## The number that matters

```bash
sportsedge report
```

`BSS` (Brier skill score vs the market) is the verdict. Positive means the model
beats the price it was betting into. Zero or negative means it does not, however
profitable a given month happened to look. Mean CLV is the leading indicator of
the same thing.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 163 tests, no network
```

The API client is tested against a stubbed transport, so the suite spends no
credits and needs no key.

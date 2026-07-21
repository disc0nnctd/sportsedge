"""sportsedge — shared betting-analytics core for cricket-kg and football-kg.

The pipeline both sports share:

    fixtures (free)  ->  odds  ->  devig  ->  model probability  ->  edge/Kelly
                                                                      |
                            ledger  <-  closing line  <-  settle  <---+
                                |
                          calibration

Nothing in this package knows which sport it is scoring. Sport-specific logic
lives in the consuming repos.
"""

from .edge import BetSizing, blend, evaluate_bet, expected_value, kelly_fraction
from .ledger import Ledger, Pick
from .calibration import evaluate as evaluate_calibration
from .odds.devig import consensus_probabilities, devig, overround

__all__ = [
    "BetSizing",
    "Ledger",
    "Pick",
    "blend",
    "consensus_probabilities",
    "devig",
    "evaluate_bet",
    "evaluate_calibration",
    "expected_value",
    "kelly_fraction",
    "overround",
]

__version__ = "0.1.0"

"""Team-name normalisation across data providers.

Providers disagree about names for the same club. football-data.org says
"Arsenal FC", "Brighton & Hove Albion FC" and "Wolverhampton Wanderers FC";
The Odds API says "Arsenal", "Brighton and Hove Albion" and "Wolves". Ratings
built from one source therefore never match fixtures from the other, and the
pick engine silently skips every match -- a failure that looks exactly like
"no bets today".

:func:`normalise` maps both spellings onto a common key. It is deliberately
conservative: it strips corporate suffixes and punctuation but does not attempt
fuzzy matching, which would eventually pair two genuinely different clubs.
Known irregular cases live in :data:`ALIASES`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Suffixes and prefixes that carry no identifying information.
_AFFIXES = (
    "fc", "afc", "cf", "sc", "ac", "acf", "as", "ss", "ssc", "us", "ud", "cd",
    "rc", "rcd", "sd", "sv", "vfl", "vfb", "tsg", "fsv", "bsc", "sk", "if",
    "bk", "fk", "nk", "club", "calcio", "futbol", "football",
)

# Clubs whose common name is not a substring of their formal name.
ALIASES = {
    "wolverhampton wanderers": "wolves",
    "tottenham hotspur": "tottenham",
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "west ham united": "west ham",
    "newcastle united": "newcastle",
    "leicester city": "leicester",
    "manchester united": "man united",
    "manchester city": "man city",
    "nottingham forest": "nottm forest",
    "sheffield united": "sheffield utd",
    "paris saint germain": "psg",
    "bayern munchen": "bayern munich",
    "borussia monchengladbach": "monchengladbach",
    "borussia dortmund": "dortmund",
    # Alias keys are matched *after* affix stripping, so they must be written
    # in stripped form ("club"/"fc" already removed).
    "internazionale": "inter",
    "internazionale milano": "inter",
    "inter milan": "inter",
    "ac milan": "milan",
    "atletico madrid": "atletico",
    "atletico de madrid": "atletico",
    "real madrid cf": "real madrid",
    "fc barcelona": "barcelona",
    "rb leipzig": "leipzig",
    "bayer 04 leverkusen": "leverkusen",
    "bayer leverkusen": "leverkusen",
    # Spanish clubs whose formal name carries a region or a second surname that
    # the odds feed drops entirely.
    "athletic": "athletic bilbao",
    "deportivo alaves": "alaves",
    "real betis balompie": "real betis",
    "betis": "real betis",
    "real sociedad de": "real sociedad",
    "celta de vigo": "celta vigo",
    "celta": "celta vigo",
    # French clubs where the formal name uses an adjectival form that shares no
    # token with the short name, so subset matching cannot bridge the gap.
    "olympique lyonnais": "lyon",
    "lyonnais": "lyon",
    "stade rennais": "rennes",
    "rennais": "rennes",
    "stade brestois": "brest",
    "brestois": "brest",
}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise(name: str) -> str:
    """Reduce a club name to a comparable key.

    >>> normalise("Arsenal FC") == normalise("Arsenal")
    True
    >>> normalise("Wolverhampton Wanderers FC") == normalise("Wolves")
    True
    """
    if not name:
        return ""

    text = _strip_accents(name).lower()
    text = text.replace("&", " and ")
    # Drop punctuation but keep word boundaries.
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()

    # Remove affixes from either end, repeatedly (e.g. "1. FC Koln").
    words = [w for w in text.split() if w]
    changed = True
    while changed and words:
        changed = False
        if words[0] in _AFFIXES or words[0].isdigit():
            words.pop(0)
            changed = True
        if words and (words[-1] in _AFFIXES or words[-1].isdigit()):
            words.pop()
            changed = True

    text = " ".join(words)
    return ALIASES.get(text, text)


def tokens(name: str) -> frozenset[str]:
    """Normalised word set, used for subset matching."""
    return frozenset(normalise(name).split())


def match(a: str, b: str) -> bool:
    """Whether two provider spellings refer to the same club."""
    return bool(a) and normalise(a) == normalise(b)


def resolve(name: str, candidates: Iterable[str]) -> str | None:
    """Find the candidate key that refers to the same club as ``name``.

    Exact normalised equality wins. Failing that, a candidate matches if one
    side's word set is a subset of the other's -- "Lille" resolves to
    "Lille OSC", "Nice" to "OGC Nice". This generalises far better than an
    ever-growing alias table, because most provider disagreements are just
    extra corporate tokens around a distinctive core.

    A subset match is only accepted when it is *unique*. "Madrid" is a subset
    of both "Real Madrid" and "Atletico Madrid", so it resolves to neither and
    the fixture is skipped -- which is the safe outcome. Betting the wrong team
    is far worse than not betting.
    """
    target = normalise(name)
    if not target:
        return None

    candidates = list(candidates)
    for key in candidates:
        if normalise(key) == target:
            return key

    target_tokens = tokens(name)
    if not target_tokens:
        return None

    subset_hits = [
        key
        for key in candidates
        if (t := tokens(key)) and (t <= target_tokens or target_tokens <= t)
    ]
    if len(subset_hits) == 1:
        return subset_hits[0]
    return None


def lookup(name: str, table: dict[str, float]) -> float | None:
    """Find ``name``'s value in a table keyed by any provider's spelling."""
    key = resolve(name, table.keys())
    return table[key] if key is not None else None

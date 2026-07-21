import pytest

from sportsedge.teams import lookup, match, normalise, resolve, tokens


@pytest.mark.parametrize(
    "a, b",
    [
        ("Arsenal FC", "Arsenal"),
        ("Liverpool FC", "Liverpool"),
        ("AFC Bournemouth", "Bournemouth"),
        ("Wolverhampton Wanderers FC", "Wolves"),
        ("Tottenham Hotspur FC", "Tottenham"),
        ("Brighton & Hove Albion FC", "Brighton"),
        ("West Ham United FC", "West Ham"),
        ("Manchester United FC", "Man United"),
        ("Manchester City FC", "Man City"),
        ("Paris Saint-Germain FC", "PSG"),
        ("FC Bayern München", "Bayern Munich"),
        ("Club Atlético de Madrid", "Atletico"),
        ("FC Internazionale Milano", "Inter"),
        ("Real Madrid CF", "Real Madrid"),
        ("FC Barcelona", "Barcelona"),
        ("Bayer 04 Leverkusen", "Leverkusen"),
        # Spanish and Italian formal names vs the odds feed's short forms.
        ("Deportivo Alavés", "Alavés"),
        ("Athletic Club", "Athletic Bilbao"),
        ("Real Betis Balompié", "Real Betis"),
        ("Real Sociedad de Fútbol", "Real Sociedad"),
        ("ACF Fiorentina", "Fiorentina"),
        ("RC Celta de Vigo", "Celta Vigo"),
    ],
)
def test_provider_spellings_agree(a, b):
    """The exact mismatch that made every football pick silently skip."""
    assert match(a, b), f"{a!r} should match {b!r} (got {normalise(a)!r} vs {normalise(b)!r})"


@pytest.mark.parametrize(
    "a, b",
    [
        ("Manchester United", "Manchester City"),
        ("Real Madrid", "Real Sociedad"),
        ("Arsenal", "Aston Villa"),
        ("Inter", "Milan"),
        ("Nottingham Forest", "Newcastle United"),
    ],
)
def test_different_clubs_do_not_collide(a, b):
    assert not match(a, b)


def test_accents_are_folded():
    assert normalise("Atlético") == normalise("Atletico")


def test_leading_numbers_are_dropped():
    assert normalise("1. FC Köln") == normalise("FC Koln")


def test_punctuation_is_ignored():
    assert normalise("Saint-Étienne") == normalise("Saint Etienne")


def test_ampersand_expands_to_and():
    assert normalise("Brighton & Hove Albion") == normalise("Brighton and Hove Albion")


def test_empty_name_normalises_to_empty():
    assert normalise("") == ""
    assert not match("", "")


def test_lookup_finds_a_differently_spelled_key():
    table = {"Arsenal FC": 1700.0, "Chelsea FC": 1650.0}
    assert lookup("Arsenal", table) == 1700.0


def test_lookup_returns_none_when_absent():
    assert lookup("Barnsley", {"Arsenal FC": 1700.0}) is None


def test_lookup_handles_an_empty_name():
    assert lookup("", {"Arsenal FC": 1700.0}) is None


# -- subset resolution ---------------------------------------------------

LIGUE_1 = [
    "Lille OSC",
    "OGC Nice",
    "Olympique de Marseille",
    "Racing Club de Lens",
    "Angers SCO",
    "AJ Auxerre",
    "Toulouse FC",
]


@pytest.mark.parametrize(
    "short, formal",
    [
        ("Lille", "Lille OSC"),
        ("Nice", "OGC Nice"),
        ("Marseille", "Olympique de Marseille"),
        ("Lens", "Racing Club de Lens"),
        ("Angers", "Angers SCO"),
        ("Auxerre", "AJ Auxerre"),
        ("Toulouse", "Toulouse FC"),
    ],
)
def test_subset_matching_bridges_extra_tokens(short, formal):
    """Most provider disagreements are just extra corporate tokens."""
    assert resolve(short, LIGUE_1) == formal


@pytest.mark.parametrize(
    "bare, candidates",
    [
        ("Rovers", ["Blackburn Rovers", "Bristol Rovers"]),
        ("Wanderers", ["Bolton Wanderers", "Wycombe Wanderers"]),
        ("Athletico", ["Athletico Paranaense", "Athletico Goianiense"]),
    ],
)
def test_an_ambiguous_subset_resolves_to_nothing(bare, candidates):
    """A bare shared token fits several clubs, so it must resolve to none.

    Skipping a fixture costs nothing; betting the wrong team costs the stake.
    """
    assert resolve(bare, candidates) is None


def test_an_exact_match_wins_over_an_ambiguous_subset():
    candidates = ["Blackburn Rovers", "Bristol Rovers"]
    assert resolve("Bristol Rovers", candidates) == "Bristol Rovers"


def test_the_madrid_clubs_stay_distinct():
    """The pair most likely to be confused must never cross-resolve."""
    table = {"Real Madrid CF": 1630.0, "Club Atletico de Madrid": 1570.0}
    assert lookup("Real Madrid", table) == 1630.0
    assert lookup("Atletico Madrid", table) == 1570.0


def test_resolve_returns_none_when_nothing_is_close():
    assert resolve("Monza", LIGUE_1) is None


def test_resolve_handles_an_empty_candidate_list():
    assert resolve("Lille", []) is None


def test_lookup_uses_subset_resolution():
    assert lookup("Nice", {"OGC Nice": 1550.0}) == 1550.0


def test_tokens_splits_the_normalised_name():
    assert tokens("Lille OSC") == frozenset({"lille", "osc"})


def test_relegated_clubs_stay_unmatched():
    """Second-tier sides genuinely have no top-flight rating; do not invent one."""
    for club in ("Frosinone", "SC Paderborn", "Hull City"):
        assert lookup(club, {"Arsenal FC": 1700.0, "Lille OSC": 1500.0}) is None

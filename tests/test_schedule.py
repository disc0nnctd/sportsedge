from datetime import datetime, timedelta, timezone

from sportsedge.schedule import (
    Fixture,
    group_by_day,
    pricing_candidates,
    render_board,
    sports_to_poll,
)


def fixture(hours_from_now: float, sport_key="cricket_odi", matchup="A vs B") -> Fixture:
    return Fixture(
        sport_key=sport_key,
        sport_title=sport_key,
        event_id=f"e{hours_from_now}",
        matchup=matchup,
        starts_at=datetime.now(timezone.utc) + timedelta(hours=hours_from_now),
    )


def at_utc(iso: str) -> Fixture:
    return Fixture(
        sport_key="cricket_odi",
        sport_title="ODI",
        event_id=iso,
        matchup="A vs B",
        starts_at=datetime.fromisoformat(iso).replace(tzinfo=timezone.utc),
    )


def test_hours_away_is_positive_for_a_future_fixture():
    assert 23 < fixture(24).hours_away <= 24


def test_a_started_fixture_is_flagged():
    assert fixture(-1).is_live_or_past
    assert not fixture(1).is_live_or_past


def test_grouping_uses_the_display_timezone():
    """A 19:00 UTC match is 00:30 IST the next day and must group there."""
    late = at_utc("2026-07-21T19:00:00")
    groups_utc = group_by_day([late], tz_offset_hours=0.0)
    groups_ist = group_by_day([late], tz_offset_hours=5.5)

    assert list(groups_utc) == ["2026-07-21"]
    assert list(groups_ist) == ["2026-07-22"]


def test_groups_are_returned_in_date_order():
    fixtures = [at_utc("2026-07-23T10:00:00"), at_utc("2026-07-21T10:00:00")]
    assert list(group_by_day(fixtures)) == ["2026-07-21", "2026-07-23"]


def test_pricing_candidates_respect_the_horizon():
    fixtures = [fixture(2), fixture(20), fixture(100)]
    near = pricing_candidates(fixtures, within_hours=36)
    assert len(near) == 2


def test_pricing_candidates_exclude_matches_already_started():
    assert pricing_candidates([fixture(-3)], within_hours=36) == []


def test_sports_to_poll_deduplicates():
    fixtures = [
        fixture(1, "cricket_odi"),
        fixture(2, "cricket_odi"),
        fixture(3, "soccer_epl"),
    ]
    assert sports_to_poll(fixtures) == ["cricket_odi", "soccer_epl"]


def test_sports_to_poll_is_empty_without_fixtures():
    assert sports_to_poll([]) == []


def test_render_board_handles_an_empty_schedule():
    assert "no fixtures" in render_board([])


def test_render_board_lists_every_fixture():
    board = render_board([fixture(5, matchup="India vs Zimbabwe")])
    assert "India vs Zimbabwe" in board
    assert "1 fixtures" in board

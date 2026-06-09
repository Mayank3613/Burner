import pytest
from datetime import datetime, timezone
from burner.sufficiency import BehaviorEntry, SufficiencyConfig, check_sufficiency, InsufficientEvidence

def some_talk():
    return ["I want to run and stay active."]

def single_day_acts():
    return [
        BehaviorEntry(timestamp="2026-06-01T08:00:00Z", text="Gym session", domain="fitness", intensity=1.0),
        BehaviorEntry(timestamp="2026-06-01T12:00:00Z", text="Lunch run", domain="fitness", intensity=1.0),
        BehaviorEntry(timestamp="2026-06-01T18:00:00Z", text="Brief stretch", domain="fitness", intensity=1.0),
    ]

def run_burner(acts, talk):
    # Setup standard config with a fixed reference date to match sample log times
    cfg = SufficiencyConfig(
        min_behavior_entries=3,
        min_talk_entries=1,
        min_span_days=3,
        stale_threshold_days=14,
        reference_date=datetime(2026, 6, 13, tzinfo=timezone.utc)
    )
    check_sufficiency(talk, acts, "fitness", cfg)

def test_abstention_on_single_day_window():
    """Two data points on the same day must produce abstention."""
    with pytest.raises(InsufficientEvidence, match="observation window"):
        run_burner(acts=single_day_acts(), talk=some_talk())

def test_abstention_on_too_few_entries():
    """Less than min_behavior_entries must raise an exception."""
    acts = [
        BehaviorEntry(timestamp="2026-06-01T08:00:00Z", text="Gym session", domain="fitness", intensity=1.0),
        BehaviorEntry(timestamp="2026-06-05T08:00:00Z", text="Second gym session", domain="fitness", intensity=1.0),
    ] # only 2 entries, min is 3
    with pytest.raises(InsufficientEvidence, match="too few behavioral entries"):
        run_burner(acts=acts, talk=some_talk())

def test_abstention_on_stale_data():
    """Data older than 14 days relative to reference date must raise an exception."""
    acts = [
        BehaviorEntry(timestamp="2026-05-20T08:00:00Z", text="Old gym session", domain="fitness", intensity=1.0),
        BehaviorEntry(timestamp="2026-05-22T08:00:00Z", text="Another old gym session", domain="fitness", intensity=1.0),
        BehaviorEntry(timestamp="2026-05-25T08:00:00Z", text="Third old session", domain="fitness", intensity=1.0),
    ] # max is May 25, ref date is June 13 -> 19 days ago (> 14 days stale)
    with pytest.raises(InsufficientEvidence, match="all data is stale"):
        run_burner(acts=acts, talk=some_talk())

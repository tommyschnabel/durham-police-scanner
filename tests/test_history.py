"""Tests for the dashboard history kept in main.py."""

from datetime import datetime, timedelta, timezone

import pytest

import main
from transcript_storage import TranscriptManager


def stamp(hours_ago):
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(main, "HISTORY_RETENTION_HOURS", 4.0)
    main.latest_transcript[:] = []
    main.latest_summaries[:] = []
    main.latest_summary = None
    yield
    main.latest_transcript[:] = []
    main.latest_summaries[:] = []
    main.latest_summary = None


def test_prune_drops_entries_and_summaries_past_retention():
    main.latest_transcript[:] = [{"text": "old", "timestamp": stamp(5)}, {"text": "new", "timestamp": stamp(3)}]
    main.latest_summaries[:] = [{"summary": "old", "timestamp": stamp(4.5)}, {"summary": "new", "timestamp": stamp(0.1)}]

    main.prune_history()

    assert [e["text"] for e in main.latest_transcript] == ["new"]
    assert [s["summary"] for s in main.latest_summaries] == ["new"]


def test_retention_is_configurable(monkeypatch):
    monkeypatch.setattr(main, "HISTORY_RETENTION_HOURS", 1.0)
    main.latest_transcript[:] = [{"timestamp": stamp(2)}, {"timestamp": stamp(0.5)}]
    main.prune_history()
    assert len(main.latest_transcript) == 1


def test_load_history_splits_feed_and_summaries(tmp_path):
    manager = TranscriptManager(output_file=str(tmp_path / "t.jsonl"))
    manager.write_entry({"type": "transcription", "text": "too old", "timestamp": stamp(6)})
    manager.write_entry({"type": "transcription", "text": "a", "timestamp": stamp(2)})
    manager.write_entry({"type": "summary", "summary": "first", "timestamp": stamp(1.5)})
    manager.write_entry({"type": "summary", "summary": "second", "timestamp": stamp(0.2)})

    main.load_history(manager)

    assert [e["text"] for e in main.latest_transcript] == ["a"]
    assert [s["summary"] for s in main.latest_summaries] == ["first", "second"]
    assert main.latest_summary["summary"] == "second"

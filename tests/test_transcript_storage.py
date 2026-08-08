"""Tests for transcript writing, rotation and retention."""

import gzip
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from transcript_storage import TranscriptManager


@pytest.fixture
def manager(tmp_path):
    return TranscriptManager(output_file=str(tmp_path / "transcript.jsonl"))


class TestWriteEntry:
    def test_writes_one_json_object_per_line(self, manager):
        assert manager.write_entry({"text": "unit 12 en route"}) is True
        assert manager.write_entry({"text": "10-4"}) is True

        lines = manager.output_file.read_text().splitlines()
        assert [json.loads(line)["text"] for line in lines] == ["unit 12 en route", "10-4"]

    def test_non_ascii_is_preserved_not_escaped(self, manager):
        manager.write_entry({"text": "café"})
        assert "café" in manager.output_file.read_text()

    def test_disabled_manager_writes_nothing(self, tmp_path):
        m = TranscriptManager(output_file=str(tmp_path / "off.jsonl"), enabled=False)
        assert m.write_entry({"text": "x"}) is False
        assert not m.output_file.exists()

    def test_unserialisable_entry_is_reported_not_raised(self, manager):
        assert manager.write_entry({"obj": object()}) is False

    def test_parent_directory_is_created(self, tmp_path):
        m = TranscriptManager(output_file=str(tmp_path / "deep" / "nested" / "t.jsonl"))
        assert m.write_entry({"text": "hi"}) is True
        assert m.output_file.exists()


class TestRotation:
    def test_no_rotation_below_the_size_limit(self, manager):
        for _ in range(5):
            manager.write_entry({"text": "short"})
        assert len(list(manager.output_file.parent.iterdir())) == 1

    def test_rotation_compresses_the_old_file(self, tmp_path):
        m = TranscriptManager(
            output_file=str(tmp_path / "transcript.jsonl"),
            max_size_mb=0.0001,  # ~100 bytes
        )
        m.write_entry({"text": "x" * 200})
        m.write_entry({"text": "after rotation"})

        archives = list(tmp_path.glob("transcript_*.jsonl.gz"))
        assert len(archives) == 1
        assert "x" * 200 in gzip.open(archives[0], "rt").read()

        # The live file only holds what was written after the rotation.
        assert json.loads(m.output_file.read_text())["text"] == "after rotation"

    def test_rotation_is_a_no_op_when_there_is_no_file(self, manager):
        manager._rotate_if_needed()  # must not raise
        assert not manager.output_file.exists()


class TestCleanup:
    def test_files_older_than_the_retention_window_are_removed(self, tmp_path):
        m = TranscriptManager(
            output_file=str(tmp_path / "transcript.jsonl"), retention_days=7
        )
        stale = tmp_path / "transcript_20200101_000000.jsonl.gz"
        stale.write_bytes(b"old")
        old = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
        os.utime(stale, (old, old))

        m._cleanup_old_logs()
        assert not stale.exists()

    def test_recent_files_are_kept(self, tmp_path):
        m = TranscriptManager(
            output_file=str(tmp_path / "transcript.jsonl"), retention_days=7
        )
        fresh = tmp_path / "transcript_20260101_000000.jsonl.gz"
        fresh.write_bytes(b"new")
        os.utime(fresh, (time.time(), time.time()))

        m._cleanup_old_logs()
        assert fresh.exists()


class TestGetRecentEntries:
    def test_returns_the_tail_in_write_order(self, manager):
        for i in range(10):
            manager.write_entry({"n": i})
        recent = manager.get_recent_entries(count=3)
        assert [e["n"] for e in recent] == [7, 8, 9]

    def test_corrupt_lines_are_skipped(self, manager):
        manager.write_entry({"n": 1})
        with open(manager.output_file, "a") as f:
            f.write("not json\n")
        manager.write_entry({"n": 2})

        assert [e["n"] for e in manager.get_recent_entries()] == [1, 2]

    def test_missing_file_returns_empty(self, manager):
        assert manager.get_recent_entries() == []

    def test_disabled_manager_returns_empty(self, tmp_path):
        m = TranscriptManager(output_file=str(tmp_path / "off.jsonl"), enabled=False)
        assert m.get_recent_entries() == []

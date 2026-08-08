"""Tests for the Kilo-backed summary generator."""

from datetime import datetime, timedelta, timezone

import pytest
import requests

from summary_generator import SummaryGenerator


def now():
    return datetime.now(timezone.utc).astimezone()


def entry(offset_seconds, text="unit 12 en route"):
    return {
        "timestamp": (now() + timedelta(seconds=offset_seconds)).isoformat(),
        "text": text,
    }


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def chat_response(content=None, reasoning=None):
    message = {}
    if content is not None:
        message["content"] = content
    if reasoning is not None:
        message["reasoning"] = reasoning
    return FakeResponse(payload={"choices": [{"message": message}]})


@pytest.fixture
def gen():
    return SummaryGenerator(api_key="test-key", summary_interval_minutes=5, window_minutes=5)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("summary_generator.time.sleep", lambda s: slept.append(s))
    return slept


class TestEnablement:
    def test_enabled_with_an_api_key(self, gen):
        assert gen.is_enabled() is True

    def test_disabled_without_a_key(self, monkeypatch):
        monkeypatch.delenv("KILO_API_KEY", raising=False)
        assert SummaryGenerator().is_enabled() is False

    def test_key_falls_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("KILO_API_KEY", "from-env")
        assert SummaryGenerator().api_key == "from-env"

    @pytest.mark.asyncio
    async def test_generate_summary_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.delenv("KILO_API_KEY", raising=False)
        assert await SummaryGenerator().generate_summary() is None


class TestHistory:
    @pytest.mark.asyncio
    async def test_entries_are_kept(self, gen):
        await gen.add_transcript(entry(0))
        assert len(gen.transcript_history) == 1

    @pytest.mark.asyncio
    async def test_entries_older_than_the_window_are_evicted(self, gen):
        await gen.add_transcript(entry(-3600))
        await gen.add_transcript(entry(0))
        assert len(gen.transcript_history) == 1


class TestShouldGenerateSummary:
    @pytest.mark.asyncio
    async def test_waits_for_at_least_three_entries(self, gen):
        for offset in (-600, -300):
            await gen.add_transcript(entry(offset))
        assert await gen.should_generate_summary() is False

    @pytest.mark.asyncio
    async def test_fires_once_the_window_is_spanned(self, gen):
        gen.transcript_history = [entry(-360), entry(-180), entry(0)]
        assert await gen.should_generate_summary() is True

    @pytest.mark.asyncio
    async def test_does_not_fire_before_the_window_is_spanned(self, gen):
        gen.transcript_history = [entry(-10), entry(-5), entry(0)]
        assert await gen.should_generate_summary() is False

    @pytest.mark.asyncio
    async def test_after_a_first_summary_it_waits_for_the_interval(self, gen):
        gen.last_summary_time = now()
        assert await gen.should_generate_summary() is False

        gen.last_summary_time = now() - timedelta(minutes=6)
        assert await gen.should_generate_summary() is True


class TestFormatTranscript:
    def test_lines_are_prefixed_with_a_wall_clock_time(self, gen):
        text = gen._format_transcript([{"timestamp": "2026-06-01T13:45:01", "text": "unit 12"}])
        assert text == "[13:45:01] unit 12"

    def test_entries_without_text_are_dropped(self, gen):
        assert gen._format_transcript([{"timestamp": "2026-06-01T13:45:01", "text": ""}]) == ""

    def test_an_unparsable_timestamp_falls_back_to_bare_text(self, gen):
        assert gen._format_transcript([{"timestamp": "whenever", "text": "unit 12"}]) == "unit 12"


class TestCallKiloApi:
    def test_returns_the_message_content(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: chat_response(content="  a summary  "))
        assert gen._call_kilo_api("transcript") == "a summary"

    def test_sends_the_api_key_as_a_bearer_token(self, gen, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json)
            return chat_response(content="ok")

        monkeypatch.setattr(requests, "post", fake_post)
        gen._call_kilo_api("transcript")

        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["json"]["model"] == gen.model

    def test_reasoning_is_used_when_content_is_empty(self, gen, monkeypatch):
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: chat_response(reasoning="thinking hard\nline one\nline two"),
        )
        assert gen._call_kilo_api("transcript") == "line one line two"

    def test_api_error_payload_returns_none(self, gen, monkeypatch):
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(payload={"error": {"message": "bad model"}}),
        )
        assert gen._call_kilo_api("transcript") is None

    def test_missing_choices_returns_none(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload={}))
        assert gen._call_kilo_api("transcript") is None

    def test_rate_limits_are_retried_with_exponential_backoff(self, gen, monkeypatch, no_sleep):
        responses = [FakeResponse(status_code=429)] * 2 + [chat_response(content="ok")]
        calls = []

        def fake_post(*a, **k):
            calls.append(1)
            return responses[len(calls) - 1]

        monkeypatch.setattr(requests, "post", fake_post)

        assert gen._call_kilo_api("transcript") == "ok"
        assert no_sleep == [1.0, 2.0]

    def test_persistent_rate_limiting_gives_up(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status_code=429))
        assert gen._call_kilo_api("transcript") is None

    def test_request_exceptions_are_retried_then_give_up(self, gen, monkeypatch, no_sleep):
        def boom(*a, **k):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(requests, "post", boom)
        assert gen._call_kilo_api("transcript") is None
        assert len(no_sleep) == 3


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_returns_a_summary_envelope(self, gen, monkeypatch):
        gen.transcript_history = [entry(-360), entry(-180), entry(0)]
        monkeypatch.setattr(gen, "_call_kilo_api", lambda text: "three units responding")

        result = await gen.generate_summary()

        assert result["type"] == "summary"
        assert result["summary"] == "three units responding"
        # The 6-minute-old entry is what makes the generator fire, but only the
        # two inside the 5-minute window are summarised.
        assert result["period"]["entry_count"] == 2
        assert gen.last_summary_time is not None

    @pytest.mark.asyncio
    async def test_a_failed_api_call_leaves_the_clock_untouched(self, gen, monkeypatch):
        gen.transcript_history = [entry(-360), entry(-180), entry(0)]
        monkeypatch.setattr(gen, "_call_kilo_api", lambda text: None)

        assert await gen.generate_summary() is None
        assert gen.last_summary_time is None

    @pytest.mark.asyncio
    async def test_returns_none_when_it_is_not_yet_time(self, gen):
        gen.transcript_history = [entry(0)]
        assert await gen.generate_summary() is None

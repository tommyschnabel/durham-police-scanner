"""Tests for the OpenAI-compatible summary generator."""

from datetime import datetime, timedelta, timezone

import pytest
import requests

from summary_generator import SummaryGenerator, clean_summary


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


def chat_response(content=None):
    message = {}
    if content is not None:
        message["content"] = content
    return FakeResponse(payload={"choices": [{"message": message}]})


@pytest.fixture
def gen():
    return SummaryGenerator(
        base_url="http://llm:8080/v1/", api_key="test-key",
        summary_interval_minutes=5, window_minutes=5,
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_DISABLE_THINKING"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("summary_generator.time.sleep", lambda s: slept.append(s))
    return slept


class TestEnablement:
    def test_enabled_with_a_base_url(self, gen):
        assert gen.is_enabled() is True

    def test_disabled_without_a_base_url(self):
        assert SummaryGenerator().is_enabled() is False

    def test_settings_fall_back_to_the_environment(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://env:8080/v1")
        monkeypatch.setenv("LLM_API_KEY", "from-env")
        monkeypatch.setenv("LLM_MODEL", "some-model")
        g = SummaryGenerator()
        assert (g.base_url, g.api_key, g.model) == ("http://env:8080/v1", "from-env", "some-model")

    def test_thinking_is_disabled_unless_turned_back_on(self, monkeypatch):
        assert SummaryGenerator().disable_thinking is True
        monkeypatch.setenv("LLM_DISABLE_THINKING", "false")
        assert SummaryGenerator().disable_thinking is False

    @pytest.mark.asyncio
    async def test_generate_summary_returns_none_when_disabled(self):
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


class TestCallLlm:
    def test_returns_the_message_content(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: chat_response(content="  A summary  "))
        assert gen._call_llm("transcript") == "A summary"

    def test_sends_the_api_key_as_a_bearer_token(self, gen, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json)
            return chat_response(content="ok")

        monkeypatch.setattr(requests, "post", fake_post)
        gen._call_llm("transcript")

        assert captured["url"] == "http://llm:8080/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["json"]["model"] == gen.model
        assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert captured["json"]["messages"][-1] == {
            "role": "user", "content": "<transcript>\ntranscript\n</transcript>",
        }

    def test_no_key_and_thinking_left_on_sends_neither(self, monkeypatch):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(headers=headers, json=json)
            return chat_response(content="ok")

        monkeypatch.setattr(requests, "post", fake_post)
        SummaryGenerator(base_url="http://llm/v1", disable_thinking=False)._call_llm("t")

        assert "Authorization" not in captured["headers"]
        assert "chat_template_kwargs" not in captured["json"]

    def test_empty_content_returns_none(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: chat_response(content=""))
        assert gen._call_llm("transcript") is None

    def test_api_error_payload_returns_none(self, gen, monkeypatch):
        monkeypatch.setattr(
            requests, "post",
            lambda *a, **k: FakeResponse(payload={"error": {"message": "bad model"}}),
        )
        assert gen._call_llm("transcript") is None

    def test_missing_choices_returns_none(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(payload={}))
        assert gen._call_llm("transcript") is None

    def test_rate_limits_are_retried_with_exponential_backoff(self, gen, monkeypatch, no_sleep):
        responses = [FakeResponse(status_code=429)] * 2 + [chat_response(content="Ok")]
        calls = []

        def fake_post(*a, **k):
            calls.append(1)
            return responses[len(calls) - 1]

        monkeypatch.setattr(requests, "post", fake_post)

        assert gen._call_llm("transcript") == "Ok"
        assert no_sleep == [1.0, 2.0]

    def test_persistent_rate_limiting_gives_up(self, gen, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(status_code=429))
        assert gen._call_llm("transcript") is None

    def test_request_exceptions_are_retried_then_give_up(self, gen, monkeypatch, no_sleep):
        def boom(*a, **k):
            raise requests.ConnectionError("no route")

        monkeypatch.setattr(requests, "post", boom)
        assert gen._call_llm("transcript") is None
        assert len(no_sleep) == 3


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_returns_a_summary_envelope(self, gen, monkeypatch):
        gen.transcript_history = [entry(-360), entry(-180), entry(0)]
        monkeypatch.setattr(gen, "_call_llm", lambda text: "three units responding")

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
        monkeypatch.setattr(gen, "_call_llm", lambda text: None)

        assert await gen.generate_summary() is None
        assert gen.last_summary_time is None

    @pytest.mark.asyncio
    async def test_returns_none_when_it_is_not_yet_time(self, gen):
        gen.transcript_history = [entry(0)]
        assert await gen.generate_summary() is None


class TestCleanSummary:
    def test_plain_summary_is_untouched(self):
        assert clean_summary("Medic 31 responds to 540 Main Street.") == "Medic 31 responds to 540 Main Street."

    def test_keeps_only_the_answer_after_a_stray_think_tag(self):
        assert clean_summary("draft one.\n</think>\n\n**Summary:** Final answer.") == "Final answer."

    def test_drops_trailing_notes_and_extra_paragraphs(self):
        text = "A unit stops a car.\n\n*(Note: ambiguous phrasing omitted.)*\n\nWait, let me re-check."
        assert clean_summary(text) == "A unit stops a car."

    def test_drops_lead_in_and_filler_sentences(self):
        text = (
            "Based on the provided transcript, a unit stops a car on Main Street. "
            "No officer safety concerns are mentioned."
        )
        assert clean_summary(text) == "A unit stops a car on Main Street."

    def test_keeps_the_first_sentence_if_everything_looks_like_filler(self):
        assert clean_summary("No incidents are reported. No units are noted.") == "No incidents are reported."

    def test_none_becomes_empty(self):
        assert clean_summary(None) == ""

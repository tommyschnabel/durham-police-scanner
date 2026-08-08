"""Tests for the transcription buffering and segment filtering.

The Whisper model itself is replaced by a fake: what matters here is which
segments survive filtering and how the sliding buffer advances.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import transcriber
from transcriber import TranscriptionSegment, WhisperTranscriber

SAMPLE_RATE = 16000


def word(probability):
    return SimpleNamespace(probability=probability)


def segment(text="unit 12", start=0.0, end=1.0, no_speech_prob=0.1, words=None):
    return SimpleNamespace(
        text=text, start=start, end=end, no_speech_prob=no_speech_prob, words=words
    )


class FakeModel:
    def __init__(self, *args, **kwargs):
        self.segments = []
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"samples": len(audio), "kwargs": kwargs})
        info = SimpleNamespace(language="en", language_probability=0.99)
        return list(self.segments), info


@pytest.fixture
def whisper(monkeypatch):
    monkeypatch.setattr(transcriber, "WhisperModel", FakeModel)
    t = WhisperTranscriber(model_size="tiny")
    return t


def audio(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class TestTranscribeChunk:
    def test_chunks_under_a_second_are_skipped(self, whisper):
        assert whisper.transcribe_chunk(audio(0.5)) == []
        assert whisper.model.calls == []

    def test_segments_are_converted_and_stripped(self, whisper):
        whisper.model.segments = [segment(text="  unit 12 en route  ")]
        result = whisper.transcribe_chunk(audio(2))

        assert len(result) == 1
        assert isinstance(result[0], TranscriptionSegment)
        assert result[0].text == "unit 12 en route"

    def test_silence_is_dropped(self, whisper):
        whisper.model.segments = [segment(no_speech_prob=0.95)]
        assert whisper.transcribe_chunk(audio(2)) == []

    def test_confidence_averages_the_word_probabilities(self, whisper):
        whisper.model.segments = [segment(words=[word(0.8), word(0.6)])]
        assert whisper.transcribe_chunk(audio(2))[0].confidence == pytest.approx(0.7)

    def test_confidence_falls_back_to_the_speech_probability(self, whisper):
        whisper.model.segments = [segment(no_speech_prob=0.25, words=None)]
        assert whisper.transcribe_chunk(audio(2))[0].confidence == pytest.approx(0.75)

    def test_timestamps_are_offset_by_the_buffer_position(self, whisper):
        whisper.buffer_offset = 100.0
        whisper.model.segments = [segment(start=1.0, end=2.0)]

        result = whisper.transcribe_chunk(audio(2))[0]
        assert (result.start_time, result.end_time) == (101.0, 102.0)

    def test_a_model_failure_is_reported_not_raised(self, whisper):
        def boom(*a, **k):
            raise RuntimeError("model exploded")

        whisper.model.transcribe = boom
        assert whisper.transcribe_chunk(audio(2)) == []

    def test_language_auto_is_passed_as_none(self, monkeypatch):
        monkeypatch.setattr(transcriber, "WhisperModel", FakeModel)
        t = WhisperTranscriber(language="auto")
        t.model.segments = [segment()]
        t.transcribe_chunk(audio(2))
        assert t.model.calls[0]["kwargs"]["language"] is None


class TestProcessAudioBuffer:
    def test_short_audio_is_buffered_without_transcribing(self, whisper):
        assert whisper.process_audio_buffer(audio(10)) == []
        assert whisper.model.calls == []
        assert len(whisper.audio_buffer) == 10 * SAMPLE_RATE

    def test_a_full_buffer_is_transcribed(self, whisper):
        whisper.model.segments = [segment()]
        result = whisper.process_audio_buffer(audio(60))
        assert len(result) == 1
        assert len(whisper.model.calls) == 1

    def test_a_five_second_overlap_is_retained(self, whisper):
        whisper.process_audio_buffer(audio(60))
        assert len(whisper.audio_buffer) == 5 * SAMPLE_RATE

    def test_the_offset_advances_by_the_consumed_audio(self, whisper):
        whisper.process_audio_buffer(audio(60))
        assert whisper.buffer_offset == pytest.approx(55.0)

    def test_max_buffer_seconds_can_force_an_earlier_flush(self, whisper):
        whisper.model.segments = [segment()]
        whisper.process_audio_buffer(audio(20), max_buffer_seconds=10.0)
        assert len(whisper.model.calls) == 1


class TestFlushAndReset:
    def test_flush_transcribes_whatever_is_left(self, whisper):
        whisper.model.segments = [segment()]
        whisper.audio_buffer = audio(2)

        assert len(whisper.flush()) == 1
        assert len(whisper.audio_buffer) == 0

    def test_flush_ignores_a_negligible_tail(self, whisper):
        whisper.audio_buffer = audio(0.05)
        assert whisper.flush() == []
        assert whisper.model.calls == []

    def test_reset_clears_buffer_and_offset(self, whisper):
        whisper.audio_buffer = audio(10)
        whisper.buffer_offset = 42.0

        whisper.reset()
        assert len(whisper.audio_buffer) == 0
        assert whisper.buffer_offset == 0.0


def test_segment_to_dict_is_json_friendly(whisper):
    whisper.model.segments = [segment(text="unit 12")]
    payload = whisper.transcribe_chunk(audio(2))[0].to_dict()

    assert payload["text"] == "unit 12"
    assert isinstance(payload["timestamp"], str)

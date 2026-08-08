"""Shared test setup.

`faster_whisper` pulls in ctranslate2 and a CUDA runtime, which is far too
heavy for a unit test run, so it is stubbed before `transcriber` is imported.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "faster_whisper" not in sys.modules:
    stub = types.ModuleType("faster_whisper")

    class WhisperModel:  # pragma: no cover - replaced by fakes in the tests
        def __init__(self, *args, **kwargs):
            raise RuntimeError("the real WhisperModel must never be constructed in tests")

    stub.WhisperModel = WhisperModel
    sys.modules["faster_whisper"] = stub

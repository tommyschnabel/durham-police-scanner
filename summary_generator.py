"""
Summary generator that summarizes transcript history through any
OpenAI-compatible chat completions endpoint (e.g. llama.cpp's llama-server).
Generates summaries of recent transcript activity every N minutes.
"""
import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import requests

from summary_prompt import build_messages

logger = logging.getLogger(__name__)

# Lead-ins and filler sentences small models add despite the prompt, e.g.
# "Based on the provided transcript, ..." or "No officer safety concerns are
# mentioned."
_LEAD_IN = re.compile(
    r'^(based on|according to|from) the (provided |given )?(police scanner )?(transcript|traffic)[^,]*,\s*',
    re.I,
)
_FILLER = re.compile(
    r'^(no |there (are|is) no |the transcript|this transcript|details remain|it is unclear)'
    r'|(not |no )(explicitly )?(mentioned|stated|detailed|evident|described|specified|noted|identified|reported)',
    re.I,
)


def clean_summary(content: Optional[str]) -> str:
    """Reduce a model reply to the bare summary paragraph."""
    text = content or ''
    # With thinking disabled, Nanbeige sometimes still writes a draft, closes
    # an unopened </think>, and then gives its real answer.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    text = re.sub(r'^\s*\**summary:?\**:?\s*', '', text.strip(), flags=re.I)
    text = re.sub(r'\n+\s*\(?\*?\*?note:.*$', '', text, flags=re.I | re.S)
    text = re.split(r'\n\s*\n', text.strip())[0]
    text = _LEAD_IN.sub('', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = [s for s in sentences if not _FILLER.search(s)] or sentences[:1]
    text = ' '.join(kept)
    return (text[:1].upper() + text[1:]).strip()


class SummaryGenerator:
    """
    Generates summaries of transcript activity using an OpenAI-compatible API.
    Maintains a rolling window of transcripts and generates
    summaries at regular intervals.
    """

    DEFAULT_MODEL = "nanbeige-4.2-3b"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        disable_thinking: Optional[bool] = None,
        summary_interval_minutes: int = 5,
        window_minutes: int = 5,
    ):
        """
        Initialize the summary generator.

        Args:
            base_url: OpenAI-compatible API root ending in /v1 (defaults to
                LLM_BASE_URL env var); summaries are disabled without one
            api_key: Bearer token, if the server wants one (defaults to LLM_API_KEY)
            model: Model name to request (defaults to LLM_MODEL, then DEFAULT_MODEL)
            disable_thinking: Send llama.cpp's enable_thinking=false template
                kwarg (defaults to LLM_DISABLE_THINKING, on unless "false")
            summary_interval_minutes: How often to generate summaries (minutes)
            window_minutes: How much transcript history to include in summary (minutes)
        """
        self.base_url = (base_url or os.environ.get('LLM_BASE_URL') or '').rstrip('/')
        self.api_key = api_key or os.environ.get('LLM_API_KEY') or None
        self.model = model or os.environ.get('LLM_MODEL') or self.DEFAULT_MODEL
        if disable_thinking is None:
            disable_thinking = os.environ.get('LLM_DISABLE_THINKING', 'true').lower() != 'false'
        self.disable_thinking = disable_thinking
        self.summary_interval = timedelta(minutes=summary_interval_minutes)
        self.window = timedelta(minutes=window_minutes)

        self.transcript_history: List[dict] = []
        self.last_summary_time: Optional[datetime] = None
        self.lock = asyncio.Lock()

        if not self.base_url:
            logger.warning("No LLM_BASE_URL provided. Summary generation will be disabled.")

        logger.info(
            f"Summary generator initialized: base_url={self.base_url}, model={self.model}, "
            f"interval={summary_interval_minutes}min, window={window_minutes}min"
        )

    def is_enabled(self) -> bool:
        """Check if summary generation is enabled (has an endpoint)."""
        return bool(self.base_url)

    async def add_transcript(self, entry: dict):
        """Add a transcript entry to history."""
        async with self.lock:
            self.transcript_history.append(entry)
            await self._cleanup_old_entries()

    async def _cleanup_old_entries(self):
        """Remove entries older than the window + buffer."""
        cutoff = datetime.now(timezone.utc).astimezone() - self.window - timedelta(minutes=1)
        self.transcript_history = [
            e for e in self.transcript_history
            if datetime.fromisoformat(e.get('timestamp', '2000-01-01')) > cutoff
        ]

    async def should_generate_summary(self) -> bool:
        """Check if it's time to generate a new summary."""
        if not self.last_summary_time:
            # Wait for initial window to fill
            if len(self.transcript_history) < 3:
                return False
            # Check if we have at least window_minutes of data
            if self.transcript_history:
                oldest = datetime.fromisoformat(self.transcript_history[0]['timestamp'])
                newest = datetime.fromisoformat(self.transcript_history[-1]['timestamp'])
                if (newest - oldest) >= self.window:
                    return True
            return False

        return datetime.now(timezone.utc).astimezone() - self.last_summary_time >= self.summary_interval

    async def generate_summary(self, force: bool = False) -> Optional[dict]:
        """
        Generate a summary of recent transcript activity.

        Args:
            force: Skip the interval check. A forced summary leaves
                last_summary_time alone so the regular schedule keeps its
                cadence and still fires on time.

        Returns:
            Summary dict with 'summary', 'timestamp', and 'period' keys,
            or None if generation failed or is disabled.
        """
        if not self.is_enabled():
            return None

        if not force and not await self.should_generate_summary():
            return None

        async with self.lock:
            # Get entries within the window
            cutoff = datetime.now(timezone.utc).astimezone() - self.window
            recent_entries = [
                e for e in self.transcript_history
                if datetime.fromisoformat(e.get('timestamp', '2000-01-01')) >= cutoff
            ]

            if len(recent_entries) < 2:
                logger.debug("Not enough transcript entries for summary")
                return None

            # Build transcript text
            transcript_text = self._format_transcript(recent_entries)

            # Generate summary via API (run in thread pool since requests is blocking)
            try:
                loop = asyncio.get_running_loop()
                summary_text = await loop.run_in_executor(
                    None, self._call_llm, transcript_text
                )
                if summary_text:
                    generated_at = datetime.now(timezone.utc).astimezone()
                    if not force:
                        self.last_summary_time = generated_at

                    # Calculate period
                    start_time = datetime.fromisoformat(recent_entries[0]['timestamp'])
                    end_time = datetime.fromisoformat(recent_entries[-1]['timestamp'])

                    return {
                        'type': 'summary',
                        'summary': summary_text,
                        'timestamp': generated_at.isoformat(),
                        'period': {
                            'start': start_time.isoformat(),
                            'end': end_time.isoformat(),
                            'entry_count': len(recent_entries)
                        }
                    }
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                return None

        return None

    def _format_transcript(self, entries: List[dict]) -> str:
        """Format transcript entries for the LLM."""
        lines = []
        for entry in entries:
            ts = entry.get('timestamp', '')
            text = entry.get('text', '')
            if text:
                try:
                    dt = datetime.fromisoformat(ts)
                    time_str = dt.strftime('%H:%M:%S')
                    lines.append(f"[{time_str}] {text}")
                except:
                    lines.append(text)
        return '\n'.join(lines)

    def _call_llm(self, transcript_text: str) -> Optional[str]:
        """Call the chat completions endpoint (synchronous, runs in thread pool)."""
        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": build_messages(transcript_text),
            "temperature": 0.3,
            "max_tokens": 450,
        }
        if self.disable_thinking:
            # llama.cpp's llama-server forwards these to the chat template.
            # Reasoning costs ~40s per summary on a 3B model for no gain.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("Rate limited, max retries exceeded")
                        return None

                response.raise_for_status()
                data = response.json()

                # Check for API errors
                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    logger.error(f"LLM API error: {error_msg}")
                    return None

                if not data.get("choices"):
                    logger.error("No choices in API response")
                    return None

                message = data["choices"][0].get("message") or {}
                summary = clean_summary(message.get("content"))
                if summary:
                    logger.info(f"Generated summary: {summary[:100]}...")
                    return summary
                logger.warning("Empty summary received from API")
                return None

            except requests.RequestException as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Request failed, retrying in {delay}s... ({e})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Request failed after {max_retries} retries: {e}")
                    return None
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                return None

        return None

    async def get_last_summary(self) -> Optional[dict]:
        """Get the most recent summary."""
        if self.last_summary_time:
            return {
                'timestamp': self.last_summary_time.isoformat(),
                'summary': 'Previous summary available'
            }
        return None

"""
Summary generator using Kilo API to summarize transcript history.
Generates summaries of recent transcript activity every N minutes.
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generates summaries of transcript activity using Kilo API.
    Maintains a rolling window of transcripts and generates
    summaries at regular intervals.
    """

    API_BASE = "https://api.kilo.ai/api/gateway"

    # Free models from Kilo Gateway
    MODEL_PRIMARY = "arcee-ai/trinity-large-preview:free"
    MODEL_SECONDARY = "corethink:free"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        summary_interval_minutes: int = 5,
        window_minutes: int = 5,
    ):
        """
        Initialize the summary generator.

        Args:
            api_key: Kilo API key (defaults to KILO_API_KEY env var)
            model: Kilo model to use for summarization (defaults to free model)
            summary_interval_minutes: How often to generate summaries (minutes)
            window_minutes: How much transcript history to include in summary (minutes)
        """
        self.api_key = api_key or os.environ.get('KILO_API_KEY')
        self.model = model or self.MODEL_PRIMARY
        self.summary_interval = timedelta(minutes=summary_interval_minutes)
        self.window = timedelta(minutes=window_minutes)

        self.transcript_history: List[dict] = []
        self.last_summary_time: Optional[datetime] = None
        self.lock = asyncio.Lock()

        if not self.api_key:
            logger.warning("No KILO_API_KEY provided. Summary generation will be disabled.")

        logger.info(
            f"Summary generator initialized: model={self.model}, "
            f"interval={summary_interval_minutes}min, window={window_minutes}min"
        )

    def is_enabled(self) -> bool:
        """Check if summary generation is enabled (has API key)."""
        return bool(self.api_key)

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

    async def generate_summary(self) -> Optional[dict]:
        """
        Generate a summary of recent transcript activity.

        Returns:
            Summary dict with 'summary', 'timestamp', and 'period' keys,
            or None if generation failed or is disabled.
        """
        if not self.is_enabled():
            return None

        if not await self.should_generate_summary():
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
                    None, self._call_kilo_api, transcript_text
                )
                if summary_text:
                    self.last_summary_time = datetime.now(timezone.utc).astimezone()

                    # Calculate period
                    start_time = datetime.fromisoformat(recent_entries[0]['timestamp'])
                    end_time = datetime.fromisoformat(recent_entries[-1]['timestamp'])

                    return {
                        'type': 'summary',
                        'summary': summary_text,
                        'timestamp': self.last_summary_time.isoformat(),
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

    def _call_kilo_api(self, transcript_text: str) -> Optional[str]:
        """Call Kilo API to generate summary (synchronous, runs in thread pool)."""
        url = f"{self.API_BASE}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = (
            "You are summarizing police scanner radio traffic. "
            "Provide a brief, factual summary of key events, incidents, and locations mentioned. "
            "Focus on: emergency calls, traffic stops, incidents in progress, officer safety concerns, "
            "and any notable locations or descriptions. Be concise - 2-4 sentences maximum. "
            "Use present tense. Do not speculate beyond what's in the transcript."
        )

        # Free models need fewer tokens, low temperature
        max_tokens = 200
        temperature = 0.3

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following police scanner transcript:\n\n{transcript_text}"}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=60
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
                    logger.error(f"Kilo API error: {error_msg}")
                    return None

                # Extract content (handle both normal content and reasoning fields)
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content") or message.get("reasoning") or ""
                    summary = content.strip()
                    if summary:
                        logger.info(f"Generated summary: {summary[:100]}...")
                        return summary
                    else:
                        logger.warning("Empty summary received from API")
                        return None
                else:
                    logger.error("No choices in API response")
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
                logger.error(f"Error calling Kilo API: {e}")
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

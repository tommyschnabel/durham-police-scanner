"""
Main FastAPI application for real-time audio transcription.
Provides WebSocket endpoint for live transcript updates and web dashboard.
"""
import asyncio
import logging
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from audio_streamer import AudioStreamReader
from transcriber import WhisperTranscriber, TranscriptionSegment
from connection_manager import ConnectionManager
from transcript_storage import TranscriptManager, parse_timestamp
from summary_generator import SummaryGenerator
from radio_codes import normalize_radio_codes

# Configuration with defaults (can be overridden via environment variables)
STREAM_URL = os.environ.get('STREAM_URL', 'https://stream.durhampolicescanner.com/')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'small')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
LANGUAGE = os.environ.get('LANGUAGE', 'en')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8000'))
CHUNK_DURATION_MS = int(os.environ.get('CHUNK_DURATION_MS', '60000'))  # Default: 60 seconds for better accuracy with large models
OVERLAP_SECONDS = int(os.environ.get('OVERLAP_SECONDS', '5'))  # 5 second overlap for 60s chunks
SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', '16000'))
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', '')
SUMMARY_INTERVAL_MINUTES = int(os.environ.get('SUMMARY_INTERVAL_MINUTES', '5'))
SUMMARY_WINDOW_MINUTES = int(os.environ.get('SUMMARY_WINDOW_MINUTES', '5'))
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'transcripts/transcript.jsonl')
MAX_LOG_SIZE_MB = float(os.environ.get('MAX_LOG_SIZE_MB', '100'))
LOG_RETENTION_DAYS = int(os.environ.get('LOG_RETENTION_DAYS', '7'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
# How far back the dashboard's live feed and summary list reach, including
# what is reloaded from the transcript file after a restart.
HISTORY_RETENTION_HOURS = float(os.environ.get('HISTORY_RETENTION_HOURS', '4'))
# A chunk normally transcribes in seconds. Past this, assume Whisper is wedged:
# the stuck native call can't be interrupted, so the process exits and the
# container's restart policy brings it back.
TRANSCRIBE_TIMEOUT_SECONDS = int(os.environ.get('TRANSCRIBE_TIMEOUT_SECONDS', '600'))

# Domain-specific vocabulary to improve transcription accuracy
WHISPER_INITIAL_PROMPT = (
    "Durham Police Department, officer, dispatch, suspect, vehicle, license plate, "
    "unit, backup, 10-4, 10-8, 10-20, 10-23, 10-24, 10-50PI, 10-78 code 3, "
    "Signal 20, felony, misdemeanor, traffic stop, "
    "warrant, arrest, incident, location, address, description, "
    "North Carolina, Raleigh, Chapel Hill, Durham, "
    "sheriff, deputy, patrol, sergeant, lieutenant, captain, "
    "emergency, 911, dispatch center, radio check, copy that, "
    "affirmative, negative, stand by, roger"
)

# Common corrections for frequently misrecognized terms
TEXT_CORRECTIONS = {
    "durham police": "Durham Police",
    "durham pd": "Durham PD",
}

# Phrases that should always be skipped (case-insensitive matching)
EXCLUDED_PHRASES = [
    "thanks for watching!",
]

# Important single words that should NOT be filtered
IMPORTANT_SINGLE_WORDS = {
    "stop", "help", "fire", "gun", "shots", "weapon", "knife", "bomb",
    "emergency", "officer", "down", "hurt", "bleeding", "chase", "pursuit",
    "shots fired", "gunshots",
}

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
audio_reader: Optional[AudioStreamReader] = None
transcriber: Optional[WhisperTranscriber] = None
transcript_manager: Optional[TranscriptManager] = None
summary_generator: Optional[SummaryGenerator] = None
connection_manager = ConnectionManager()
transcription_task: Optional[asyncio.Task] = None
summary_task: Optional[asyncio.Task] = None
watchdog_task: Optional[asyncio.Task] = None
latest_transcript: list = []
latest_summary: Optional[dict] = None
latest_summaries: list = []  # History of summaries for new connections
last_broadcast_text: str = ""
transcribe_started_at: Optional[float] = None  # time.monotonic() while a chunk is in Whisper
state_lock = threading.Lock()  # Protects latest_transcript, latest_summary, latest_summaries, and last_broadcast_text

def transcription_stuck() -> bool:
    """True if the chunk currently in Whisper has run past the timeout.

    Only times the Whisper call, so a stream outage (no chunks at all) doesn't count.
    """
    started = transcribe_started_at
    return started is not None and time.monotonic() - started > TRANSCRIBE_TIMEOUT_SECONDS


async def watchdog_worker():
    """Exit the process if transcription wedges; the restart policy recovers it."""
    while True:
        await asyncio.sleep(30)
        if transcription_stuck():
            logger.critical(f"Transcription stuck for over {TRANSCRIBE_TIMEOUT_SECONDS}s, exiting")
            logging.shutdown()
            os._exit(1)


def prune_history():
    """Drop feed entries and summaries older than the retention window.

    Caller must hold state_lock.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HISTORY_RETENTION_HOURS)

    def recent(entry):
        ts = parse_timestamp(entry.get('timestamp'))
        return ts is not None and ts >= cutoff

    latest_transcript[:] = [e for e in latest_transcript if recent(e)]
    latest_summaries[:] = [s for s in latest_summaries if recent(s)]


def load_history(manager: TranscriptManager):
    """Seed the feed and summary list from disk so a restart loses nothing."""
    global latest_summary
    cutoff = datetime.now(timezone.utc) - timedelta(hours=HISTORY_RETENTION_HOURS)
    entries = manager.get_entries_since(cutoff)
    with state_lock:
        latest_transcript[:] = [e for e in entries if e.get('type') == 'transcription']
        latest_summaries[:] = [e for e in entries if e.get('type') == 'summary']
        latest_summary = latest_summaries[-1] if latest_summaries else None
    logger.info(
        f"Loaded {len(latest_transcript)} transcriptions and {len(latest_summaries)} "
        f"summaries from the last {HISTORY_RETENTION_HOURS:g}h"
    )


# Thread-safe message queue for WebSocket broadcasts
broadcast_queue: asyncio.Queue = asyncio.Queue()


async def broadcast_worker():
    """Background task that processes broadcast messages from the queue."""
    logger.info("Starting broadcast worker")
    while True:
        try:
            entry = await broadcast_queue.get()
            if entry is None:  # Shutdown signal
                break
            await connection_manager.broadcast_json(entry)
        except Exception as e:
            logger.error(f"Broadcast worker error: {e}")
    logger.info("Broadcast worker stopped")


async def publish_summary(summary: dict):
    """Record a new summary, persist it, and push it to connected clients."""
    global latest_summary
    with state_lock:
        latest_summary = summary
        latest_summaries.append(summary)
        prune_history()

    logger.info(f"[SUMMARY] {summary['summary'][:200]}...")

    if transcript_manager:
        transcript_manager.write_entry(summary)

    await broadcast_queue.put(summary)


async def summary_worker():
    """Background task that generates summaries periodically."""

    if not summary_generator or not summary_generator.is_enabled():
        logger.info("Summary generation disabled (no LLM_BASE_URL)")
        return

    logger.info(f"Starting summary worker (interval: {SUMMARY_INTERVAL_MINUTES} minutes)")

    while True:
        try:
            # Wait for the interval
            await asyncio.sleep(30)  # Check every 30 seconds

            # Try to generate summary
            summary = await summary_generator.generate_summary()
            if summary:
                await publish_summary(summary)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Summary worker error: {e}")
            await asyncio.sleep(60)  # Wait a minute on error

    logger.info("Summary worker stopped")


def process_audio_chunk_threadsafe(transcriber, audio_chunk, transcript_manager):
    """Process audio chunk in a separate thread."""
    segments = transcriber.process_audio_buffer(audio_chunk)
    return segments


def apply_text_corrections(text: str) -> str:
    """Apply domain-specific text corrections."""
    text_lower = text.lower()
    for wrong, correct in TEXT_CORRECTIONS.items():
        if wrong in text_lower:
            text = text.replace(wrong, correct)
            text_lower = text.lower()
    return normalize_radio_codes(text)


def should_filter_single_word(text: str, confidence: float) -> bool:
    """Determine if a single-word entry should be filtered."""
    words = text.lower().split()
    if len(words) >= 2:
        return False
    single_word = words[0].lower().rstrip('.,!?')
    if single_word in IMPORTANT_SINGLE_WORDS:
        return False
    if confidence >= 0.8:
        return False
    return True


def stream_audio_sync(audio_reader, transcriber, transcript_manager):
    """Synchronous audio streaming and transcription loop."""
    global latest_transcript, last_broadcast_text, transcribe_started_at

    logger.info("Starting audio streaming loop")

    try:
        for audio_chunk in audio_reader.stream_audio():
            # Process audio chunk
            transcribe_started_at = time.monotonic()
            try:
                segments = transcriber.process_audio_buffer(audio_chunk)
            finally:
                transcribe_started_at = None

            logger.debug(f"Got {len(segments)} segments from transcription")
            for segment in segments:
                if not segment.text.strip():
                    logger.debug("Empty segment text, skipping")
                    continue

                # Skip if same as last broadcast
                segment_text = segment.text.strip()
                if segment_text == last_broadcast_text:
                    logger.debug("Duplicate message, skipping broadcast")
                    continue

                # Skip excluded phrases (case-insensitive)
                if segment_text.lower() in EXCLUDED_PHRASES:
                    logger.debug(f"Excluded phrase detected, skipping: {segment_text}")
                    continue

                # Apply domain-specific text corrections
                segment_text = apply_text_corrections(segment_text)

                # Filter low-value single-word entries but keep important ones
                if should_filter_single_word(segment_text, segment.confidence):
                    logger.debug(f"Low-value single word entry, skipping: {segment_text}")
                    continue

                # Create entry
                entry = {
                    'type': 'transcription',
                    'text': segment_text,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'confidence': segment.confidence,
                    'timestamp': segment.timestamp.isoformat()
                }

                # Update last broadcast text
                with state_lock:
                    last_broadcast_text = segment_text

                # Log to console
                logger.info(f"[{segment.timestamp.strftime('%H:%M:%S')}] {segment_text}")

                # Save to file
                transcript_manager.write_entry(entry)

                # Update latest transcript - thread-safe
                with state_lock:
                    latest_transcript.append(entry)
                    prune_history()

                # Add to summary generator (async - fire and forget)
                if summary_generator and summary_generator.is_enabled():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            summary_generator.add_transcript(entry),
                            loop
                        )
                    except Exception as e:
                        logger.debug(f"Failed to add to summary generator: {e}")

                # Add to broadcast queue (thread-safe)
                try:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_queue.put(entry),
                        loop
                    )
                except Exception as e:
                    logger.error(f"Failed to queue broadcast: {e}")

    except Exception as e:
        logger.error(f"Audio streaming error: {e}")
    finally:
        logger.info("Audio streaming stopped")


async def transcription_worker():
    """Background task that reads audio and performs transcription."""
    global latest_transcript, loop, summary_generator, transcript_manager

    logger.info("Starting transcription worker")

    # Get the event loop
    loop = asyncio.get_running_loop()

    # Initialize summary generator
    summary_generator = SummaryGenerator(
        base_url=LLM_BASE_URL,
        summary_interval_minutes=SUMMARY_INTERVAL_MINUTES,
        window_minutes=SUMMARY_WINDOW_MINUTES
    )

    # Initialize audio reader
    audio_reader = AudioStreamReader(
        stream_url=STREAM_URL,
        chunk_duration_ms=CHUNK_DURATION_MS,
        target_sample_rate=16000
    )

    # Initialize transcriber in thread pool to avoid blocking
    logger.info("Loading Whisper model (this may take a moment)...")
    transcriber = await loop.run_in_executor(None, lambda: WhisperTranscriber(
        model_size=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        language=LANGUAGE if LANGUAGE != 'auto' else None,
        vad_filter=True,
        initial_prompt=WHISPER_INITIAL_PROMPT
    ))
    logger.info("Whisper model loaded successfully")

    # Run the synchronous streaming loop in a separate thread
    try:
        await loop.run_in_executor(
            None,
            stream_audio_sync,
            audio_reader,
            transcriber,
            transcript_manager
        )
    except Exception as e:
        logger.error(f"Transcription worker error: {e}")
    finally:
        logger.info("Transcription worker stopped")
        # Flush remaining audio
        if transcriber:
            final_segments = transcriber.flush()
            for segment in final_segments:
                entry = {
                    'type': 'transcription',
                    'text': segment.text,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'confidence': segment.confidence,
                    'timestamp': segment.timestamp.isoformat()
                }
                transcript_manager.write_entry(entry)
                await broadcast_queue.put(entry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global transcription_task, broadcast_task, summary_task, transcript_manager, watchdog_task

    # Startup
    logger.info("Starting up Durham Police Scanner Transcriber")
    logger.info(f"Stream URL: {STREAM_URL}")
    logger.info(f"Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    logger.info(f"Chunk duration: {CHUNK_DURATION_MS}ms")
    logger.info(f"Summary generation: {'enabled' if LLM_BASE_URL else 'disabled (set LLM_BASE_URL)'}")

    transcript_manager = TranscriptManager(
        output_file=OUTPUT_FILE,
        max_size_mb=MAX_LOG_SIZE_MB,
        retention_days=LOG_RETENTION_DAYS
    )
    load_history(transcript_manager)

    # Start broadcast worker
    broadcast_task = asyncio.create_task(broadcast_worker())

    # Start transcription in background
    transcription_task = asyncio.create_task(transcription_worker())

    # Start summary worker (will exit immediately if disabled)
    summary_task = asyncio.create_task(summary_worker())

    watchdog_task = asyncio.create_task(watchdog_worker())

    yield

    # Shutdown
    logger.info("Shutting down")
    if watchdog_task:
        watchdog_task.cancel()
    if broadcast_task:
        await broadcast_queue.put(None)  # Signal broadcast worker to stop
        try:
            await asyncio.wait_for(broadcast_task, timeout=5.0)
        except asyncio.TimeoutError:
            broadcast_task.cancel()
    if summary_task:
        summary_task.cancel()
        try:
            await summary_task
        except asyncio.CancelledError:
            pass
    if transcription_task:
        transcription_task.cancel()
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass


# Create FastAPI app
app = FastAPI(
    title="Durham Police Scanner Transcriber",
    description="Real-time audio transcription from police scanner stream",
    version="1.0.0",
    lifespan=lifespan
)

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "stream_url": STREAM_URL,
            "model": WHISPER_MODEL,
            "retention_hours": HISTORY_RETENTION_HOURS,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcript updates."""
    await connection_manager.connect(websocket)

    try:
        with state_lock:
            prune_history()
            history_snapshot = list(latest_transcript)
            current_summary = latest_summary
            summaries_snapshot = list(latest_summaries)

        # Send transcript history
        await websocket.send_json({
            "type": "history",
            "entries": history_snapshot
        })

        # Send summary history (all past summaries)
        for summary in summaries_snapshot:
            await websocket.send_json(summary)

        # Send current summary if available and not already sent
        if current_summary:
            # Check if it's already in the history
            if not summaries_snapshot or summaries_snapshot[-1].get('timestamp') != current_summary.get('timestamp'):
                await websocket.send_json(current_summary)

        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)

                if data.get('action') == 'ping':
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)


# Forcing a summary costs an LLM call; one per minute is plenty.
FORCE_SUMMARY_COOLDOWN = timedelta(minutes=1)
last_forced_summary: Optional[datetime] = None


@app.post("/api/summary")
async def force_summary():
    """Generate a summary now instead of waiting for the next interval.

    Doesn't move the regular schedule: the interval tick still fires when it
    would have, so this only adds an extra summary in between.
    """
    global last_forced_summary

    if not summary_generator or not summary_generator.is_enabled():
        return JSONResponse(status_code=503, content={"error": "summaries disabled"})

    now = datetime.now(timezone.utc)
    if last_forced_summary and now - last_forced_summary < FORCE_SUMMARY_COOLDOWN:
        retry_after = FORCE_SUMMARY_COOLDOWN - (now - last_forced_summary)
        seconds = max(1, int(retry_after.total_seconds()))
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(seconds)},
            content={"error": f"rate limited, retry in {seconds}s"},
        )
    last_forced_summary = now

    summary = await summary_generator.generate_summary(force=True)
    if not summary:
        return JSONResponse(status_code=409, content={"error": "not enough recent transcript to summarize"})

    await publish_summary(summary)
    return summary


@app.get("/api/status")
async def get_status():
    """Health check endpoint for Docker."""
    if transcription_stuck():
        return JSONResponse(status_code=503, content={"status": "transcription stuck"})
    return {
        "status": "healthy",
        "connected_clients": len(connection_manager.active_connections),
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())

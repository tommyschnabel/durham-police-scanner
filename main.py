"""
Main FastAPI application for real-time audio transcription.
Provides WebSocket endpoint for live transcript updates and web dashboard.
"""
import asyncio
import logging
import json
import os
import threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from audio_streamer import AudioStreamReader
from transcriber import WhisperTranscriber, TranscriptionSegment
from connection_manager import ConnectionManager
from transcript_storage import TranscriptManager
from summary_generator import SummaryGenerator

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
KILO_API_KEY = os.environ.get('KILO_API_KEY', '')
KILO_MODEL = os.environ.get('KILO_MODEL', 'minimax/minimax-m2.5')
SUMMARY_INTERVAL_MINUTES = int(os.environ.get('SUMMARY_INTERVAL_MINUTES', '5'))
SUMMARY_WINDOW_MINUTES = int(os.environ.get('SUMMARY_WINDOW_MINUTES', '5'))
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'transcripts/transcript.jsonl')
MAX_LOG_SIZE_MB = float(os.environ.get('MAX_LOG_SIZE_MB', '100'))
LOG_RETENTION_DAYS = int(os.environ.get('LOG_RETENTION_DAYS', '7'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
WS_HISTORY_LIMIT = int(os.environ.get('WS_HISTORY_LIMIT', '100'))

# Domain-specific vocabulary to improve transcription accuracy
WHISPER_INITIAL_PROMPT = (
    "Durham Police Department, officer, dispatch, suspect, vehicle, license plate, "
    "unit, backup, 10-4, 10-20, 10-code, felony, misdemeanor, traffic stop, "
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
    "10 4": "10-4",
    "10 20": "10-20",
    "ten four": "10-4",
    "ten twenty": "10-20",
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
latest_transcript: list = []
latest_summary: Optional[dict] = None
latest_summaries: list = []  # History of summaries for new connections
last_broadcast_text: str = ""
state_lock = threading.Lock()  # Protects latest_transcript, latest_summary, latest_summaries, and last_broadcast_text

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


async def summary_worker():
    """Background task that generates summaries periodically."""
    global latest_summary

    if not summary_generator or not summary_generator.is_enabled():
        logger.info("Summary generation disabled (no KILO_API_KEY)")
        return

    logger.info(f"Starting summary worker (interval: {SUMMARY_INTERVAL_MINUTES} minutes)")

    while True:
        try:
            # Wait for the interval
            await asyncio.sleep(30)  # Check every 30 seconds

            # Try to generate summary
            summary = await summary_generator.generate_summary()
            if summary:
                # Update latest summary and add to history
                with state_lock:
                    latest_summary = summary
                    latest_summaries.append(summary)
                    # Keep last 50 summaries (about 4 hours at 5-min intervals)
                    if len(latest_summaries) > 50:
                        latest_summaries.pop(0)

                # Log to console
                logger.info(f"[SUMMARY] {summary['summary'][:200]}...")

                # Save to file
                if transcript_manager:
                    transcript_manager.write_entry(summary)

                # Broadcast to all clients
                await broadcast_queue.put(summary)

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
    return text


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
    global latest_transcript, last_broadcast_text

    logger.info("Starting audio streaming loop")

    try:
        for audio_chunk in audio_reader.stream_audio():
            # Process audio chunk
            segments = transcriber.process_audio_buffer(audio_chunk)

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

                # Update latest transcript (keep last 500) - thread-safe
                with state_lock:
                    latest_transcript.append(entry)
                    if len(latest_transcript) > 500:
                        latest_transcript[:] = latest_transcript[-500:]

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
        api_key=KILO_API_KEY,
        model=KILO_MODEL,
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

    # Initialize transcript manager
    transcript_manager = TranscriptManager(
        output_file=OUTPUT_FILE,
        max_size_mb=MAX_LOG_SIZE_MB,
        retention_days=LOG_RETENTION_DAYS
    )

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
    global transcription_task, broadcast_task, summary_task

    # Startup
    logger.info("Starting up Durham Police Scanner Transcriber")
    logger.info(f"Stream URL: {STREAM_URL}")
    logger.info(f"Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    logger.info(f"Chunk duration: {CHUNK_DURATION_MS}ms")
    logger.info(f"Summary generation: {'enabled' if KILO_API_KEY else 'disabled (set KILO_API_KEY)'}")

    # Start broadcast worker
    broadcast_task = asyncio.create_task(broadcast_worker())

    # Start transcription in background
    transcription_task = asyncio.create_task(transcription_worker())

    # Start summary worker (will exit immediately if disabled)
    summary_task = asyncio.create_task(summary_worker())

    yield

    # Shutdown
    logger.info("Shutting down")
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
        context={"stream_url": STREAM_URL, "model": WHISPER_MODEL}
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcript updates."""
    await connection_manager.connect(websocket)

    try:
        with state_lock:
            history_snapshot = list(latest_transcript[-WS_HISTORY_LIMIT:])
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


@app.get("/api/status")
async def get_status():
    """Health check endpoint for Docker."""
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

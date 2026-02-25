"""
Main FastAPI application for real-time audio transcription.
Provides WebSocket endpoint for live transcript updates and web dashboard.
"""
import asyncio
import logging
import json
import os
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

# Configuration with defaults (can be overridden via environment variables)
STREAM_URL = os.environ.get('STREAM_URL', 'https://stream.durhampolicescanner.com/')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
LANGUAGE = os.environ.get('LANGUAGE', 'en')
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', '8000'))
CHUNK_DURATION_MS = int(os.environ.get('CHUNK_DURATION_MS', '5000'))
OVERLAP_MS = int(os.environ.get('OVERLAP_MS', '1000'))
SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', '16000'))
OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'transcripts/transcript.jsonl')
MAX_LOG_SIZE_MB = float(os.environ.get('MAX_LOG_SIZE_MB', '100'))
LOG_RETENTION_DAYS = int(os.environ.get('LOG_RETENTION_DAYS', '7'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
WS_HISTORY_LIMIT = int(os.environ.get('WS_HISTORY_LIMIT', '100'))

# Static list of phrases to exclude from logging
EXCLUDED_PHRASES = [
    "Thanks for watching!",
]

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
connection_manager = ConnectionManager()
transcription_task: Optional[asyncio.Task] = None
latest_transcript: list = []
last_broadcast_text: str = ""

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


def process_audio_chunk_threadsafe(transcriber, audio_chunk, transcript_manager):
    """Process audio chunk in a separate thread."""
    segments = transcriber.process_audio_buffer(audio_chunk)
    return segments


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

                # Skip excluded phrases
                if segment_text in EXCLUDED_PHRASES:
                    logger.debug(f"Excluded phrase detected, skipping: {segment_text}")
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
                last_broadcast_text = segment_text
                
                # Log to console
                logger.info(f"[{segment.timestamp.strftime('%H:%M:%S')}] {segment_text}")
                
                # Save to file
                transcript_manager.write_entry(entry)
                
                # Update latest transcript (keep last 100) - threadsafe
                latest_transcript.append(entry)
                if len(latest_transcript) > 100:
                    latest_transcript[:] = latest_transcript[-100:]
                
                # Add to broadcast queue (thread-safe)
                try:
                    # Use asyncio.run_coroutine_threadsafe to put in queue
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
    global latest_transcript, loop
    
    logger.info("Starting transcription worker")
    
    # Get the event loop
    loop = asyncio.get_running_loop()
    
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
        vad_filter=False
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
    global transcription_task, broadcast_task
    
    # Startup
    logger.info("Starting up Durham Police Scanner Transcriber")
    logger.info(f"Stream URL: {STREAM_URL}")
    logger.info(f"Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}")
    
    # Start broadcast worker
    broadcast_task = asyncio.create_task(broadcast_worker())
    
    # Start transcription in background
    transcription_task = asyncio.create_task(transcription_worker())
    
    yield
    
    # Shutdown
    logger.info("Shutting down")
    if broadcast_task:
        await broadcast_queue.put(None)  # Signal broadcast worker to stop
        try:
            await asyncio.wait_for(broadcast_task, timeout=5.0)
        except asyncio.TimeoutError:
            broadcast_task.cancel()
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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stream_url": STREAM_URL,
        "model": WHISPER_MODEL
    })


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    return {
        "status": "running" if transcription_task and not transcription_task.done() else "stopped",
        "stream_url": STREAM_URL,
        "model": WHISPER_MODEL,
        "connected_clients": len(connection_manager.active_connections),
        "latest_entries": latest_transcript[-10:]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcript updates."""
    await connection_manager.connect(websocket)
    
    try:
        # Send recent history to new client
        await websocket.send_json({
            "type": "history",
            "entries": latest_transcript[-WS_HISTORY_LIMIT:]
        })
        
        # Keep connection alive and handle client messages
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


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())

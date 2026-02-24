# Agent Instructions for Durham Police Scanner Transcriber

## Project Overview
Real-time audio transcription from the Durham Police Scanner using local Whisper speech recognition.

> **Note**: This project was developed with AI assistance. When making changes, please ensure code quality and test thoroughly.

## Commands

### Run the Application

#### Local Development
```bash
source venv/bin/activate
python main.py
```

Or use the startup script:
```bash
./start.sh
```

#### Docker
```bash
# Quick start
docker-compose up -d

# With custom env file
docker-compose --env-file .env.local up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Docker Build
```bash
docker build -t durham-police-transcriber .
```

## Architecture

### Core Modules

1. **main.py** - FastAPI application
   - WebSocket endpoint at `/ws`
   - Background transcription worker (`transcription_worker`)
   - Web dashboard at `/`
   - API endpoints for status and transcript retrieval
   - **Important**: Audio streaming runs in a separate thread to avoid blocking the event loop

2. **audio_streamer.py** - HTTP audio stream capture
   - `AudioStreamReader` class
   - Handles reconnection with `reconnect_delay`
   - Format detection (MP3, AAC, OGG)
   - Audio conversion to 16kHz mono
   - Yields numpy arrays for processing

3. **transcriber.py** - Whisper transcription
   - `WhisperTranscriber` class using faster-whisper
   - `TranscriptionSegment` dataclass for results
   - Sliding window buffer management
   - Processes 5-second chunks with 2-second overlap

4. **connection_manager.py** - WebSocket client management
   - `ConnectionManager` class
   - Broadcasts to multiple connected clients
   - Handles disconnections gracefully

5. **transcript_storage.py** - File logging
   - `TranscriptManager` class
   - JSON Lines format (JSONL)
   - Log rotation at 100MB
   - Automatic compression (gzip)
   - Cleanup after 7 days

### Data Flow
```
HTTP Stream → AudioStreamReader → Audio Chunks → WhisperTranscriber →
TranscriptSegments → ConnectionManager (broadcast) + TranscriptManager (file)
```

## Configuration

### Environment Variables
All configuration uses environment variables with sensible defaults. **No `.env` file is required**.

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_URL` | `https://stream.durhampolicescanner.com/` | Audio stream URL |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `CHUNK_DURATION_MS` | `5000` | Audio chunk size in milliseconds |
| `OVERLAP_MS` | `1000` | Buffer overlap in milliseconds |
| `WHISPER_MODEL` | `base` | Model size: tiny/base/small/medium/large-v1/2/3 |
| `WHISPER_DEVICE` | `cpu` | cpu or cuda |
| `WHISPER_COMPUTE_TYPE` | `int8` | int8, int8_float16, float16, float32 |
| `LANGUAGE` | `en` | Language code (or 'auto' for detection) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `OUTPUT_FILE` | `transcripts/transcript.jsonl` | Transcript output path |
| `MAX_LOG_SIZE_MB` | `100` | Log rotation threshold |
| `LOG_RETENTION_DAYS` | `7` | Cleanup threshold |
| `LOG_LEVEL` | `INFO` | Logging level: DEBUG/INFO/WARNING/ERROR |

### Overriding Settings

**Docker Compose:**
```bash
WHISPER_MODEL=small docker compose up -d
```

**Local Development:**
```bash
WHISPER_MODEL=small python main.py
```

### Optional .env File
While not required, you can create a `.env.local` file for convenience:
```bash
# .env.local
WHISPER_MODEL=small
WHISPER_DEVICE=cuda
LOG_LEVEL=DEBUG
```

Then run with:
```bash
docker compose --env-file .env.local up -d
```

## Key Implementation Details

### Audio Processing
- Target: 16kHz, mono, 16-bit PCM
- Buffer overlap: 2 seconds for continuity
- Auto-reconnect on stream errors (5s delay)

### Transcription
- Uses `faster-whisper` for local inference
- VAD filtering enabled by default
- Beam size: 5 for quality vs speed balance
- Models cached at `/root/.cache/whisper` (Docker) or user cache

### WebSocket
- Broadcasts JSON messages to all clients
- Sends history (last 20 entries) to new connections
- Ping/pong for connection keepalive (30s interval)

### File Output
- Format: JSON Lines (.jsonl)
- Location: `transcripts/` directory
- Rotation: When file exceeds 100MB
- Retention: 7 days (compressed)

## Common Tasks

### Adding Speaker Diarization
1. Install `pyannote.audio`
2. Modify `transcriber.py` to add diarization pipeline
3. Update `TranscriptionSegment` to include speaker ID
4. Update frontend to display speaker labels

### Adding a New API Endpoint
1. Add route in `main.py`
2. Follow existing pattern with type hints
3. Update README.md documentation

### Modifying Web Dashboard
1. Edit `templates/index.html`
2. Uses vanilla JavaScript (no frameworks)
3. WebSocket client in the same file
4. Updates are received via WebSocket and rendered dynamically

### Upgrading Whisper Model
1. Change `WHISPER_MODEL` in `.env`
2. Model downloads automatically on first run
3. Monitor memory usage with larger models
4. For Docker, models are cached in a volume

### Changing Stream URL
1. Update `STREAM_URL` in `.env`
2. Restart the application
3. Audio streamer will auto-detect format

## Docker Development

### Multi-stage Build
The Dockerfile uses multi-stage builds for smaller images:
- Builder stage: Installs build dependencies
- Runtime stage: Contains only runtime dependencies

### Volumes
- `./transcripts` → `/app/transcripts` - Transcript persistence
- `whisper-cache` → Model caching (managed by Docker)

### Health Checks
The container includes a health check that polls the `/api/status` endpoint every 30 seconds.

## Dependencies

Key packages:
- `faster-whisper>=0.10.0` - Local Whisper inference
- `fastapi>=0.104.0` - Web framework
- `pydub>=0.25.1` - Audio processing
- `requests>=2.31.0` - HTTP streaming
- `numpy>=1.24.0` - Audio arrays
- `websockets>=12.0` - WebSocket support

## Performance Considerations

- Model size affects memory and speed significantly
- GPU recommended for medium/large models
- CPU with int8 quantization works for base/small
- 24/7 operation requires log rotation
- Docker resource limits can be set in compose override

## Error Handling

- Stream disconnections: Auto-reconnect with 5s delay
- Transcription errors: Logged, processing continues
- WebSocket errors: Client disconnected, others unaffected
- File I/O errors: Logged, operation continues
- Container health: Auto-restart on failure

## AI Development Notes

When modifying this codebase:
1. Maintain the existing architecture patterns
2. Ensure error handling is robust (24/7 operation requirement)
3. Test both local and Docker deployments
4. Update documentation for any new features
5. Follow the existing code style and type hints

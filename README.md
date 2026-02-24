# Durham Police Scanner Transcriber

Real-time audio transcription from the Durham Police Scanner stream using local Whisper speech recognition.

## Features

- **Live Audio Streaming**: Connects to `https://stream.durhampolicescanner.com/`
- **Local Transcription**: Uses `faster-whisper` for on-device speech recognition (no API costs)
- **Web Dashboard**: Real-time transcript display with live updates via WebSocket
- **File Logging**: Saves transcripts to rotating JSONL files
- **24/7 Operation**: Auto-reconnection, error handling, and memory management
- **Docker Support**: Ready-to-use Docker container for easy deployment

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Police Scanner │────▶│ Audio Stream │────▶│  Whisper    │────▶│  WebSocket  │
│   Stream URL    │     │   Reader     │     │Transcription│     │  Broadcast  │
└─────────────────┘     └──────────────┘     └─────────────┘     └──────┬──────┘
                                                                        │
                                                                   ┌────┴────┐
                                                                   │ Browser │
                                                                   │Dashboard│
                                                                   └─────────┘
```

## Quick Start

Choose your preferred deployment method:

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd durham-police-scanner

# Start with Docker Compose
docker-compose up -d

# Access dashboard at http://localhost:8000
```

### Option 2: Local Python

#### Prerequisites

- Python 3.10+
- ffmpeg installed on your system
- ~500MB disk space for Whisper models

#### Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install ffmpeg (if not already installed):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# Windows (with chocolatey)
choco install ffmpeg
```

#### Running

Start the application:
```bash
./start.sh
```

Or manually:
```bash
source venv/bin/activate
python main.py
```

Access the web dashboard at: **http://localhost:8000**

## Configuration

All configuration is done via environment variables with sensible defaults. No `.env` file is required.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_URL` | `https://stream.durhampolicescanner.com/` | Audio stream URL |
| `WHISPER_MODEL` | `base` | Model size: tiny, base, small, medium, large-v1/2/3 |
| `WHISPER_DEVICE` | `cpu` | Device: cpu or cuda |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute: int8, int8_float16, float16, float32 |
| `LANGUAGE` | `en` | Language code (or 'auto' for detection) |
| `CHUNK_DURATION_MS` | `5000` | Audio chunk size in milliseconds |
| `PORT` | `8000` | Web server port |
| `OUTPUT_FILE` | `transcripts/transcript.jsonl` | Transcript output file |
| `MAX_LOG_SIZE_MB` | `100` | Max log file size before rotation |
| `LOG_RETENTION_DAYS` | `7` | Days to keep old log files |

### Overriding Settings

**Docker Compose:**
```bash
# Override specific settings
WHISPER_MODEL=small WHISPER_DEVICE=cuda docker compose up -d

# Or use an env file (optional)
docker compose --env-file .env.local up -d
```

**Local Development:**
```bash
WHISPER_MODEL=small python main.py
```

### Model Selection Guide

| Model | Size | Speed | Quality | RAM Required |
|-------|------|-------|---------|--------------|
| tiny | ~75MB | Fastest | Basic | ~1GB |
| base | ~150MB | Fast | Good | ~1GB |
| small | ~500MB | Medium | Better | ~2GB |
| medium | ~1.5GB | Slow | Excellent | ~4GB |
| large-v3 | ~3GB | Slowest | Best | ~8GB |

For mid-range hardware, `base` or `small` is recommended.

## API Endpoints

- `GET /` - Web dashboard
- `GET /api/status` - System status and recent entries
- `WS /ws` - WebSocket for real-time updates

## Output Format

Transcripts are saved as JSON Lines (JSONL) with the following format:

```json
{"type": "transcription", "text": "10-4, proceeding to location", "start_time": 12.34, "end_time": 15.67, "confidence": 0.95, "timestamp": "2024-01-15T08:30:00.123456"}
```

## File Structure

```
├── main.py                  # FastAPI application
├── audio_streamer.py        # HTTP audio stream capture
├── transcriber.py          # Whisper transcription logic
├── connection_manager.py    # WebSocket connection handling
├── transcript_storage.py    # File logging with rotation
├── templates/
│   └── index.html          # Web dashboard
├── requirements.txt        # Python dependencies
├── .env                    # Configuration
├── .env.local.example      # Template for local env overrides
├── start.sh                # Startup script
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── docker-compose.override.yml.example  # Docker override template
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git ignore rules
├── transcripts/            # Transcript output directory
│   └── .gitkeep            # Keeps directory in git
├── AGENTS.md               # Development reference for AI agents
└── README.md               # Documentation
```

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the dashboard
open http://localhost:8000

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Docker Configuration

The docker-compose.yml includes:
- **Persistent storage** for transcripts in `./transcripts/`
- **Model caching** to avoid re-downloading Whisper models
- **Auto-restart** on failure
- **Health checks** to ensure service availability

### Environment Variables with Docker

Override any `.env` setting via docker-compose:

```bash
WHISPER_MODEL=small docker-compose up -d
```

Or create a `.env.local` file:
```bash
cp .env.local.example .env.local
# Edit .env.local with your settings
docker-compose --env-file .env.local up -d
```

### Building the Image Manually

```bash
# Build
docker build -t durham-police-transcriber .

# Run
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/transcripts:/app/transcripts \
  -e WHISPER_MODEL=base \
  -e LOG_LEVEL=INFO \
  --name police-scanner \
  durham-police-transcriber
```

### GPU Support (NVIDIA)

Copy the example override file and customize:
```bash
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit docker-compose.override.yml to uncomment GPU settings
docker-compose up -d
```

Or use the provided GPU configuration:
```yaml
version: '3.8'

services:
  transcriber:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - WHISPER_DEVICE=cuda
      - WHISPER_COMPUTE_TYPE=float16
```

## Troubleshooting

### General Issues

**Stream won't connect:**
- Verify the stream URL is accessible: `curl -I https://stream.durhampolicescanner.com/`
- Check your internet connection
- The application auto-reconnects on errors

**Poor transcription quality:**
- Try a larger model: `WHISPER_MODEL=small`
- Ensure `LANGUAGE` matches the audio
- Enable VAD: already enabled by default

**High memory usage:**
- Use a smaller model
- Reduce `CHUNK_DURATION_MS`
- Check transcript file size and enable rotation

**Slow transcription:**
- Use GPU if available: `WHISPER_DEVICE=cuda`
- Use `int8` compute type for faster CPU inference
- Try the `tiny` or `base` model

### Docker-Specific Issues

**Container exits immediately:**
```bash
# Check logs
docker-compose logs

# Verify ffmpeg is installed in container
docker-compose exec transcriber ffmpeg -version
```

**Permission denied on transcript files:**
```bash
# Fix permissions
sudo chown -R $USER:$USER ./transcripts
```

**Model download fails:**
The Whisper model downloads on first run. Ensure you have:
- ~150MB for `base` model
- ~500MB for `small` model
- Sufficient disk space in the container

## Acknowledgments

This project was created with the assistance of AI technology. The codebase, documentation, and Docker configuration were developed using an AI coding assistant.

### Key Technologies Used

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** - Optimized Whisper implementation for local transcription
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern, fast web framework for building APIs
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Open source speech recognition model

## License

MIT License

---

**Disclaimer**: This is an independent project not affiliated with the Durham Police Department. The stream is publicly available and this tool is for informational purposes only.

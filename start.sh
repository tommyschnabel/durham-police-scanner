#!/bin/bash
# Startup script for Durham Police Scanner Transcriber

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please create it first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg not found. Please install it:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  Arch: sudo pacman -S ffmpeg"
    exit 1
fi

echo "Starting Durham Police Scanner Transcriber..."
echo "Web dashboard will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Run the application
python main.py "$@"

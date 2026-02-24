"""
Audio stream capture module - reads audio from HTTP stream and yields processable chunks.
"""
import logging
import io
import requests
import numpy as np
from pydub import AudioSegment
from typing import Optional, Generator

logger = logging.getLogger(__name__)


class AudioStreamReader:
    """
    Reads audio from an HTTP stream (like Icecast/Shoutcast radio streams)
    and yields audio chunks suitable for transcription.
    """
    
    def __init__(
        self,
        stream_url: str,
        chunk_duration_ms: int = 5000,
        target_sample_rate: int = 16000,
        reconnect_delay: float = 5.0
    ):
        self.stream_url = stream_url
        self.chunk_duration_ms = chunk_duration_ms
        self.target_sample_rate = target_sample_rate
        self.reconnect_delay = reconnect_delay
        self._session = requests.Session()
        self._running = False
        
    def stream_audio(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio chunks as numpy arrays.
        Automatically reconnects on connection errors.
        """
        self._running = True
        buffer = io.BytesIO()
        
        while self._running:
            try:
                logger.info(f"Connecting to audio stream: {self.stream_url}")
                
                # Stream with chunked transfer encoding support
                response = self._session.get(
                    self.stream_url,
                    stream=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (compatible; AudioTranscriber/1.0)',
                        'Icy-MetaData': '0'  # Disable metadata for cleaner audio stream
                    },
                    timeout=(10, 60)  # (connect timeout, read timeout)
                )
                response.raise_for_status()
                
                # Detect format from Content-Type or try common formats
                content_type = response.headers.get('Content-Type', '').lower()
                logger.info(f"Stream content-type: {content_type}")
                
                # Read stream in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if not self._running:
                        break
                    
                    if chunk:
                        buffer.write(chunk)
                        
                        # Check if we have enough data for processing
                        current_buffer_ms = self._estimate_buffer_duration(buffer)
                        
                        if current_buffer_ms >= self.chunk_duration_ms:
                            # Extract audio and yield
                            audio_data = self._extract_audio(buffer)
                            if audio_data is not None and len(audio_data) > 0:
                                yield audio_data
                            
                            # Reset buffer, keeping overlap for continuity
                            buffer = self._reset_buffer_with_overlap(buffer)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Stream error: {e}. Reconnecting in {self.reconnect_delay}s...")
                if self._running:
                    import time
                    time.sleep(self.reconnect_delay)
            except Exception as e:
                logger.error(f"Unexpected error in stream: {e}")
                if self._running:
                    import time
                    time.sleep(self.reconnect_delay)
        
        logger.info("Audio stream reader stopped")
    
    def _estimate_buffer_duration(self, buffer: io.BytesIO) -> int:
        """Estimate duration of audio in buffer in milliseconds."""
        try:
            buffer.seek(0)
            # Try to load as MP3 (most common for radio streams)
            audio = AudioSegment.from_mp3(buffer)
            return len(audio)
        except:
            # If we can't parse yet, estimate based on typical bitrate
            # Radio streams are usually 128 kbps = 16 KB/s
            buffer_size = buffer.tell()
            estimated_seconds = buffer_size / (128 * 1024 / 8)  # 128 kbps
            return int(estimated_seconds * 1000)
    
    def _extract_audio(self, buffer: io.BytesIO) -> Optional[np.ndarray]:
        """Extract and convert audio to numpy array at target sample rate."""
        try:
            buffer.seek(0)
            
            # Try common audio formats
            audio = None
            for format_name in ['mp3', 'aac', 'ogg', 'mp4']:
                try:
                    buffer.seek(0)
                    audio = AudioSegment.from_file(buffer, format=format_name)
                    break
                except:
                    continue
            
            if audio is None:
                logger.warning("Could not parse audio format, trying generic load")
                buffer.seek(0)
                audio = AudioSegment.from_file(buffer)
            
            # Convert to target format: mono, 16kHz, 16-bit
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.target_sample_rate)
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Convert to numpy array (float32 for faster-whisper)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1]
            samples = samples / 32768.0
            
            return samples
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def _reset_buffer_with_overlap(self, buffer: io.BytesIO) -> io.BytesIO:
        """Reset buffer keeping overlap for continuity."""
        try:
            buffer.seek(0)
            audio = AudioSegment.from_file(buffer)
            
            # Keep last 1 second as overlap
            overlap_ms = min(1000, len(audio) // 2)
            if len(audio) > overlap_ms:
                overlap_audio = audio[-overlap_ms:]
                new_buffer = io.BytesIO()
                overlap_audio.export(new_buffer, format='mp3')
                return new_buffer
        except:
            pass
        
        return io.BytesIO()
    
    def stop(self):
        """Stop the stream reader."""
        self._running = False
        logger.info("Stop signal received")

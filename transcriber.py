"""
Transcription module using faster-whisper for local speech recognition.
"""
import logging
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single transcription segment with metadata."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class WhisperTranscriber:
    """
    Local transcription using faster-whisper.
    Optimized for real-time streaming with buffering.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
        beam_size: int = 5,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None
    ):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v1/2/3)
            device: "cpu" or "cuda"
            compute_type: "int8", "int8_float16", "float16", "float32"
            language: Language code (e.g., "en", "auto" for detection)
            beam_size: Beam size for decoding (higher = better quality, slower)
            vad_filter: Enable voice activity detection filtering
            initial_prompt: Optional prompt to guide transcription with domain vocabulary
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = None if language == "auto" else language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt
        
        logger.info(f"Loading Whisper model: {model_size} on {device}")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        logger.info("Whisper model loaded successfully")
        
        # Buffer for accumulating audio between transcriptions
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0.0  # Time offset in seconds
        
    def transcribe_chunk(
        self,
        audio: np.ndarray,
        init_prompt: str = ""
    ) -> List[TranscriptionSegment]:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio: Audio samples as float32 numpy array
            init_prompt: Optional prompt to guide transcription
            
        Returns:
            List of TranscriptionSegment objects
        """
        if len(audio) < 16000:  # Less than 1 second
            logger.debug("Audio chunk too short, skipping")
            return []
        
        try:
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                initial_prompt=self.initial_prompt,
                beam_size=self.beam_size,
                word_timestamps=True,
                condition_on_previous_text=True,
                vad_filter=self.vad_filter
            )
            
            logger.debug(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            results = []
            for segment in segments:
                # Skip high no-speech probability segments
                if segment.no_speech_prob > 0.9:
                    continue
                
                # Calculate average word confidence from actual word probabilities
                avg_confidence = 1.0 - segment.no_speech_prob
                if segment.words:
                    word_confidences = []
                    for word in segment.words:
                        if hasattr(word, 'probability'):
                            word_confidences.append(word.probability)
                    if word_confidences:
                        avg_confidence = sum(word_confidences) / len(word_confidences)
                
                results.append(TranscriptionSegment(
                    text=segment.text.strip(),
                    start_time=self.buffer_offset + segment.start,
                    end_time=self.buffer_offset + segment.end,
                    confidence=avg_confidence,
                    timestamp=datetime.now(timezone.utc).astimezone()
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return []
    
    def process_audio_buffer(
        self,
        audio: np.ndarray,
        max_buffer_seconds: float = 30.0
    ) -> List[TranscriptionSegment]:
        """
        Process audio with a sliding buffer for continuous transcription.
        
        Args:
            audio: New audio chunk to add
            max_buffer_seconds: Maximum buffer size before forced flush
            
        Returns:
            List of completed transcription segments
        """
        # Append to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio)
        
        # Check buffer size
        buffer_seconds = len(self.audio_buffer) / 16000  # Assuming 16kHz
        
        results = []
        
        # Process when we have enough audio or buffer is getting full
        if buffer_seconds >= 5.0 or buffer_seconds >= max_buffer_seconds:
            # Get transcription
            segments = self.transcribe_chunk(self.audio_buffer)
            
            # Filter to only return new/updated segments
            if segments:
                results = segments
            
            # Trim buffer - keep overlap for context
            overlap_samples = int(2.0 * 16000)  # Keep last 2 seconds for continuity
            if len(self.audio_buffer) > overlap_samples:
                self.buffer_offset += (len(self.audio_buffer) - overlap_samples) / 16000
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
        
        return results
    
    def flush(self) -> List[TranscriptionSegment]:
        """Flush remaining audio in buffer and return final segments."""
        if len(self.audio_buffer) < 1600:  # Less than 0.1 second
            return []
        
        segments = self.transcribe_chunk(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)
        return segments
    
    def reset(self):
        """Reset the transcriber state."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0.0

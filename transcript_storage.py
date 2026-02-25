"""
Transcript storage and management with log rotation.
"""
import json
import os
import gzip
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)


class TranscriptManager:
    """
    Manages transcript output with:
    - Real-time file writing
    - Log rotation based on size
    - Automatic compression of old logs
    - JSON Lines format for easy parsing
    """
    
    def __init__(
        self,
        output_file: str = "transcript.jsonl",
        max_size_mb: float = 100.0,
        retention_days: int = 7,
        enabled: bool = True
    ):
        self.output_file = Path(output_file)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self.enabled = enabled
        
        if self.enabled:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Transcript output: {self.output_file.absolute()}")
    
    def write_entry(self, entry: dict) -> bool:
        """
        Write a transcript entry to file.
        
        Args:
            entry: Dictionary containing transcript data
            
        Returns:
            True if written successfully
        """
        if not self.enabled:
            return False
        
        try:
            # Check if rotation needed
            self._rotate_if_needed()
            
            # Append entry
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing transcript: {e}")
            return False
    
    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        if not self.output_file.exists():
            return
        
        current_size = self.output_file.stat().st_size
        if current_size >= self.max_size_bytes:
            self._rotate()
    
    def _rotate(self):
        """Perform log rotation."""
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{self.output_file.stem}_{timestamp}{self.output_file.suffix}"
        rotated_path = self.output_file.parent / rotated_name
        
        # Rename current file
        self.output_file.rename(rotated_path)
        logger.info(f"Rotated transcript to: {rotated_path}")
        
        # Compress old rotated files
        self._compress_old_logs()
        
        # Clean up old logs
        self._cleanup_old_logs()
    
    def _compress_old_logs(self):
        """Compress uncompressed log files."""
        for log_file in self.output_file.parent.glob(f"{self.output_file.stem}_*{self.output_file.suffix}"):
            if not log_file.suffix == '.gz':
                try:
                    gz_path = log_file.with_suffix(log_file.suffix + '.gz')
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(gz_path, 'wb') as f_out:
                            f_out.writelines(f_in)
                    log_file.unlink()
                    logger.debug(f"Compressed {log_file.name}")
                except Exception as e:
                    logger.error(f"Error compressing {log_file}: {e}")
    
    def _cleanup_old_logs(self):
        """Remove log files older than retention period."""
        cutoff = datetime.now(timezone.utc).astimezone() - timedelta(days=self.retention_days)
        
        for log_file in self.output_file.parent.glob(f"{self.output_file.stem}_*"):
            try:
                # Extract timestamp from filename
                stat = log_file.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
                
                if mtime < cutoff:
                    log_file.unlink()
                    logger.info(f"Removed old log: {log_file.name}")
            except Exception as e:
                logger.error(f"Error cleaning up {log_file}: {e}")
    
    def get_recent_entries(self, count: int = 100) -> list:
        """Get recent transcript entries from file."""
        if not self.enabled or not self.output_file.exists():
            return []
        
        entries = []
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-count:]:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error reading transcript: {e}")
        
        return entries

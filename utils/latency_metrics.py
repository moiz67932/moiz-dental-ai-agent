"""
Latency tracking and performance monitoring utilities.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional

from config import LATENCY_DEBUG, logger


class TurnMetrics:
    """
    Lightweight latency tracker for voice agent turns.
    Logs structured timing data when LATENCY_DEBUG=1.
    
    Usage:
        metrics = LatencyMetrics()
        metrics.mark("user_eou")  # End of utterance
        metrics.mark("llm_start")
        metrics.mark("llm_first_token")
        metrics.mark("tts_start")
        metrics.mark("audio_start")
        metrics.log_turn()  # Emits structured log line
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._start = time.perf_counter()
        self._marks: Dict[str, float] = {}
        self._filler_info: Dict[str, Any] = {"played": False, "suppressed_reason": None}
    
    def mark(self, label: str):
        """Record a timestamp for a labeled event."""
        self._marks[label] = time.perf_counter() - self._start
    
    def set_filler_info(self, played: bool, reason: Optional[str] = None):
        """Track whether filler was played or suppressed."""
        self._filler_info = {"played": played, "suppressed_reason": reason}
    
    def get_elapsed(self, label: str) -> float:
        """Get elapsed time in ms for a label."""
        return self._marks.get(label, 0) * 1000
    
    def log_turn(self, extra: str = ""):
        """Emit a single structured log line with all latency data."""
        if not LATENCY_DEBUG:
            return
        
        parts = []
        ordered_labels = ["user_eou", "llm_start", "llm_first_token", "llm_done", "tts_start", "audio_start"]
        for label in ordered_labels:
            if label in self._marks:
                parts.append(f"{label}={self._marks[label]*1000:.0f}ms")
        
        filler_str = "played" if self._filler_info["played"] else f"suppressed:{self._filler_info['suppressed_reason'] or 'none'}"
        
        log_line = f"[LATENCY] {' | '.join(parts)} | filler={filler_str}"
        if extra:
            log_line += f" | {extra}"
        
        logger.info(log_line)
        self.reset()


# Global latency tracker (reset per turn)
_turn_metrics = TurnMetrics()

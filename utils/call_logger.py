"""
CallLogger - Centralized Logging & Observability for LiveKit Voice Agent.

This module provides production-grade logging with dual-destination strategy:
- Cloud Logging (stdout): Structured JSON for real-time debugging
- Supabase (Postgres): Persistent storage for analytics, dashboards, and audits

Key Features:
- All events correlated by call_id
- Non-blocking async Supabase inserts
- Buffered batch writes for performance
- Sensitive data sanitization
- Thread-safe with asyncio locks

Usage:
    from utils.call_logger import CallLogger
    
    call_logger = CallLogger(
        call_id=str(uuid.uuid4()),
        agent_id="telephony_agent",
        environment="production"
    )
    call_logger.log_call_start(from_number="+1***1234", to_number="+1***5678")
    call_logger.log_stt(text="Hello", latency_ms=150, audio_duration_ms=1200)
    await call_logger.flush_to_supabase()
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4


# =============================================================================
# CONFIGURATION
# =============================================================================

SUPABASE_LOGGING_ENABLED = os.getenv("SUPABASE_LOGGING", "1") == "1"
LOG_BUFFER_FLUSH_INTERVAL = int(os.getenv("LOG_FLUSH_INTERVAL", "10"))  # seconds
MAX_BUFFER_SIZE = int(os.getenv("LOG_BUFFER_SIZE", "100"))  # max events before auto-flush


# =============================================================================
# STRUCTURED JSON LOGGER
# =============================================================================

class StructuredLogger:
    """
    Google Cloud Logging compatible structured JSON logger.
    All logs include call_id, agent_id, environment, timestamp.
    """
    
    def __init__(self, name: str = "call_logger"):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
    
    def log(
        self,
        level: str,
        message: str,
        call_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        environment: Optional[str] = None,
        job_execution_id: Optional[str] = None,
        **extra
    ):
        """Emit a structured JSON log entry."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": level.upper(),
            "message": message,
            "call_id": call_id,
            "agent_id": agent_id or os.getenv("LIVEKIT_AGENT_NAME", "telephony_agent"),
            "environment": environment or os.getenv("ENVIRONMENT", "development"),
            "job_execution_id": job_execution_id or os.getenv("CLOUD_RUN_EXECUTION", "local"),
        }
        
        # Merge extra fields
        for key, value in extra.items():
            if value is not None:
                log_entry[key] = value
        
        # Remove None values for cleaner logs
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        
        log_line = json.dumps(log_entry, default=str)
        
        if level.upper() == "ERROR":
            self._logger.error(log_line)
        elif level.upper() == "WARNING":
            self._logger.warning(log_line)
        elif level.upper() == "DEBUG":
            self._logger.debug(log_line)
        else:
            self._logger.info(log_line)


# Global structured logger instance
_structured_logger = StructuredLogger()


# =============================================================================
# PHONE SANITIZATION
# =============================================================================

def mask_phone(phone: Optional[str]) -> str:
    """
    Mask phone number for safe logging.
    Example: +13105551234 -> ***1234
    """
    if not phone:
        return "unknown"
    
    # Extract digits only
    digits = re.sub(r"\D", "", phone)
    
    if len(digits) >= 4:
        return f"***{digits[-4:]}"
    return "***"


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove or mask sensitive data from payload before logging.
    """
    sensitive_keys = {
        "api_key", "apikey", "secret", "password", "token", 
        "authorization", "auth", "credential", "key"
    }
    
    sanitized = {}
    for key, value in payload.items():
        key_lower = key.lower()
        
        # Mask sensitive keys
        if any(s in key_lower for s in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        # Mask phone numbers
        elif "phone" in key_lower and isinstance(value, str):
            sanitized[key] = mask_phone(value)
        # Recursively sanitize nested dicts
        elif isinstance(value, dict):
            sanitized[key] = sanitize_payload(value)
        else:
            sanitized[key] = value
    
    return sanitized


# =============================================================================
# CALL LOGGER
# =============================================================================

class CallLogger:
    """
    Centralized call logging with dual-destination strategy:
    - Cloud Logging: Structured JSON to stdout
    - Supabase: Persistent storage with async batch inserts
    
    All log entries include:
    - call_id: Unique identifier for the call
    - agent_id: Agent identity (e.g., "telephony_agent")
    - environment: "production" or "development"
    - job_execution_id: Cloud Run execution ID
    - timestamp: ISO-8601 formatted
    
    Usage:
        logger = CallLogger(call_id="...", agent_id="...", environment="...")
        logger.log_call_start(from_number="+1...", to_number="+1...")
        logger.log_stt(text="Hello", latency_ms=150, audio_duration_ms=1200)
        await logger.flush_to_supabase()
    """
    
    def __init__(
        self,
        call_id: str,
        agent_id: str = "telephony_agent",
        environment: str = "production",
        clinic_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        supabase_client: Optional[Any] = None,
    ):
        self.call_id = call_id
        self.agent_id = agent_id
        self.environment = environment
        self.clinic_id = clinic_id
        self.organization_id = organization_id
        self.job_execution_id = os.getenv("CLOUD_RUN_EXECUTION", "local")
        
        # Metrics tracking
        self._turn_index = 0
        self._utterance_index = 0
        self._call_start_time = time.time()
        
        # Event buffer for batch Supabase inserts
        self._event_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()
        
        # Supabase client (lazy-loaded if not provided)
        self._supabase = supabase_client
        
        # Turn tracking for call_turns table
        self._current_turn: Dict[str, Any] = {}
    
    def _base_log_entry(self) -> Dict[str, Any]:
        """Return base fields for every log entry."""
        return {
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "environment": self.environment,
            "job_execution_id": self.job_execution_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    
    def _log_to_stdout(self, event_type: str, payload: Dict[str, Any]):
        """Emit structured JSON log to stdout for Cloud Logging."""
        log_entry = self._base_log_entry()
        log_entry["type"] = event_type
        log_entry.update(sanitize_payload(payload))
        
        _structured_logger.log(
            level="INFO",
            message=f"[{event_type.upper()}] {json.dumps(sanitize_payload(payload), default=str)}",
            **log_entry
        )
    
    def _buffer_event(self, event_type: str, payload: Dict[str, Any], latency_ms: Optional[int] = None):
        """Buffer event for batch Supabase insert."""
        if not SUPABASE_LOGGING_ENABLED:
            return
        
        event = {
            "call_id": self.call_id,
            "event_type": event_type,
            "payload": sanitize_payload(payload),
            "latency_ms": latency_ms,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        self._event_buffer.append(event)
        
        # Auto-flush if buffer is full
        if len(self._event_buffer) >= MAX_BUFFER_SIZE:
            asyncio.create_task(self._async_flush_events())
    
    async def _async_flush_events(self):
        """Flush buffered events to Supabase (non-blocking)."""
        if not self._event_buffer or not SUPABASE_LOGGING_ENABLED:
            return
        
        async with self._buffer_lock:
            events_to_flush = self._event_buffer.copy()
            self._event_buffer.clear()
        
        if not events_to_flush:
            return
        
        try:
            # Lazy-load Supabase client
            if self._supabase is None:
                from config import supabase
                self._supabase = supabase
            
            # Batch insert to call_events table
            await asyncio.to_thread(
                lambda: self._supabase.table("call_events").insert(events_to_flush).execute()
            )
        except Exception as e:
            # Log error but don't crash the call
            _structured_logger.log(
                level="ERROR",
                message=f"Failed to flush events to Supabase: {e}",
                call_id=self.call_id,
                error=str(e),
                event_count=len(events_to_flush)
            )
    
    # =========================================================================
    # CALL LIFECYCLE LOGS
    # =========================================================================
    
    def log_call_start(self, from_number: Optional[str], to_number: Optional[str]):
        """Log call start event."""
        payload = {
            "from_number": mask_phone(from_number),
            "to_number": mask_phone(to_number),
        }
        
        self._log_to_stdout("call_start", payload)
        self._buffer_event("call_start", payload)
        
        # Create call record in Supabase
        if SUPABASE_LOGGING_ENABLED:
            asyncio.create_task(self._create_call_record(from_number, to_number))
    
    async def _create_call_record(self, from_number: Optional[str], to_number: Optional[str]):
        """Create initial call record in Supabase calls table."""
        try:
            if self._supabase is None:
                from config import supabase
                self._supabase = supabase
            
            call_record = {
                "call_id": self.call_id,
                "from_number": mask_phone(from_number),
                "to_number": mask_phone(to_number),
                "environment": self.environment,
                "job_execution_id": self.job_execution_id,
                "start_time": datetime.utcnow().isoformat() + "Z",
            }
            
            if self.clinic_id:
                call_record["clinic_id"] = self.clinic_id
            if self.organization_id:
                call_record["organization_id"] = self.organization_id
            
            await asyncio.to_thread(
                lambda: self._supabase.table("calls").insert(call_record).execute()
            )
        except Exception as e:
            _structured_logger.log(
                level="ERROR",
                message=f"Failed to create call record: {e}",
                call_id=self.call_id
            )
    
    def log_call_end(self, duration_seconds: int, end_reason: str):
        """Log call end event."""
        payload = {
            "duration_seconds": duration_seconds,
            "end_reason": end_reason,
        }
        
        self._log_to_stdout("call_end", payload)
        self._buffer_event("call_end", payload)
        
        # Update call record with end time
        if SUPABASE_LOGGING_ENABLED:
            asyncio.create_task(self._update_call_record(duration_seconds, end_reason))
    
    async def _update_call_record(self, duration_seconds: int, end_reason: str):
        """Update call record with end time and duration."""
        try:
            if self._supabase is None:
                from config import supabase
                self._supabase = supabase
            
            await asyncio.to_thread(
                lambda: self._supabase.table("calls").update({
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "duration_seconds": duration_seconds,
                    "end_reason": end_reason,
                }).eq("call_id", self.call_id).execute()
            )
        except Exception as e:
            _structured_logger.log(
                level="ERROR",
                message=f"Failed to update call record: {e}",
                call_id=self.call_id
            )
    
    # =========================================================================
    # STT (Speech-to-Text) LOGS
    # =========================================================================
    
    def log_stt(
        self,
        text: str,
        latency_ms: int,
        audio_duration_ms: int = 0,
        confidence: Optional[float] = None,
        is_final: bool = True,
    ):
        """Log STT transcription event."""
        self._utterance_index += 1
        
        payload = {
            "utterance_index": self._utterance_index,
            "text": text,
            "latency_ms": latency_ms,
            "audio_duration_ms": audio_duration_ms,
            "is_final": is_final,
        }
        
        if confidence is not None:
            payload["confidence"] = confidence
        
        self._log_to_stdout("stt", payload)
        self._buffer_event("stt", payload, latency_ms=latency_ms)
        
        # Track for turn aggregation
        self._current_turn["user_text"] = text
        self._current_turn["stt_latency_ms"] = latency_ms
    
    # =========================================================================
    # VAD (Voice Activity Detection) LOGS
    # =========================================================================
    
    def log_vad(self, event: str, duration_ms: Optional[int] = None):
        """
        Log VAD event.
        
        Args:
            event: "speech_start" or "speech_end"
            duration_ms: Speech duration (for speech_end events)
        """
        payload = {
            "event": event,
        }
        
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        
        self._log_to_stdout("vad", payload)
        self._buffer_event("vad", payload, latency_ms=duration_ms)
    
    # =========================================================================
    # LLM LOGS
    # =========================================================================
    
    def log_llm(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        response_text: str = "",
    ):
        """Log LLM request/response event."""
        self._turn_index += 1
        
        payload = {
            "turn_index": self._turn_index,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "response_text": response_text[:500] if response_text else "",  # Truncate long responses
        }
        
        self._log_to_stdout("llm", payload)
        self._buffer_event("llm", payload, latency_ms=latency_ms)
        
        # Track for turn aggregation
        self._current_turn["agent_text"] = response_text
        self._current_turn["llm_latency_ms"] = latency_ms
    
    # =========================================================================
    # TTS LOGS
    # =========================================================================
    
    def log_tts(
        self,
        text: str,
        latency_ms: int,
        audio_duration_ms: int = 0,
        voice: str = "default",
    ):
        """Log TTS generation event."""
        payload = {
            "text": text[:500] if text else "",  # Truncate long text
            "latency_ms": latency_ms,
            "audio_duration_ms": audio_duration_ms,
            "voice": voice,
        }
        
        self._log_to_stdout("tts", payload)
        self._buffer_event("tts", payload, latency_ms=latency_ms)
        
        # Track for turn aggregation
        self._current_turn["tts_latency_ms"] = latency_ms
        
        # Flush turn if we have all components
        if all(k in self._current_turn for k in ["user_text", "llm_latency_ms", "tts_latency_ms"]):
            self._flush_turn()
    
    def _flush_turn(self):
        """Save completed turn to call_turns table."""
        if not self._current_turn or not SUPABASE_LOGGING_ENABLED:
            return
        
        turn = {
            "call_id": self.call_id,
            "turn_index": self._turn_index,
            "user_text": self._current_turn.get("user_text", ""),
            "agent_text": self._current_turn.get("agent_text", ""),
            "stt_latency_ms": self._current_turn.get("stt_latency_ms"),
            "llm_latency_ms": self._current_turn.get("llm_latency_ms"),
            "tts_latency_ms": self._current_turn.get("tts_latency_ms"),
            "total_latency_ms": sum([
                self._current_turn.get("stt_latency_ms", 0),
                self._current_turn.get("llm_latency_ms", 0),
                self._current_turn.get("tts_latency_ms", 0),
            ]),
        }
        
        asyncio.create_task(self._save_turn(turn))
        self._current_turn = {}
    
    async def _save_turn(self, turn: Dict[str, Any]):
        """Save turn to Supabase call_turns table."""
        try:
            if self._supabase is None:
                from config import supabase
                self._supabase = supabase
            
            await asyncio.to_thread(
                lambda: self._supabase.table("call_turns").insert(turn).execute()
            )
        except Exception as e:
            _structured_logger.log(
                level="ERROR",
                message=f"Failed to save turn: {e}",
                call_id=self.call_id,
                turn_index=turn.get("turn_index")
            )
    
    # =========================================================================
    # STATE TRANSITION LOGS
    # =========================================================================
    
    def log_state_change(self, state: str, value: Any):
        """
        Log state transition event.
        
        Args:
            state: State field name (e.g., "phone_confirmed", "date_locked")
            value: New value (will be sanitized)
        """
        # Sanitize phone values
        if "phone" in state.lower() and isinstance(value, str):
            value = mask_phone(value)
        
        payload = {
            "state": state,
            "value": value,
        }
        
        self._log_to_stdout("state_change", payload)
        self._buffer_event("state_change", payload)
    
    # =========================================================================
    # TOOL CALL LOGS
    # =========================================================================
    
    def log_tool_call(
        self,
        tool: str,
        latency_ms: int,
        success: bool,
        args: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
    ):
        """Log tool/function call event."""
        payload = {
            "tool": tool,
            "latency_ms": latency_ms,
            "success": success,
        }
        
        if args:
            payload["args"] = sanitize_payload(args)
        
        if result:
            payload["result"] = result[:200] if len(result) > 200 else result
        
        self._log_to_stdout("tool_call", payload)
        self._buffer_event("tool_call", payload, latency_ms=latency_ms)
    
    # =========================================================================
    # ERROR LOGS
    # =========================================================================
    
    def log_error(
        self,
        component: str,
        error: str,
        recovered: bool,
        stack_trace: Optional[str] = None,
    ):
        """Log error event with optional stack trace."""
        payload = {
            "component": component,
            "error": error,
            "recovered": recovered,
        }
        
        if stack_trace:
            # Truncate long stack traces
            payload["stack_trace"] = stack_trace[:2000] if len(stack_trace) > 2000 else stack_trace
        
        # Log at ERROR level
        _structured_logger.log(
            level="ERROR",
            message=f"[ERROR] {component}: {error}",
            call_id=self.call_id,
            agent_id=self.agent_id,
            environment=self.environment,
            job_execution_id=self.job_execution_id,
            **payload
        )
        
        self._buffer_event("error", payload)
    
    # =========================================================================
    # FLUSH TO SUPABASE
    # =========================================================================
    
    async def flush_to_supabase(self):
        """
        Flush all buffered events to Supabase.
        Call this at call end to ensure all events are persisted.
        """
        await self._async_flush_events()
        
        # Also flush any pending turn
        if self._current_turn:
            self._flush_turn()
    
    # =========================================================================
    # CONTEXT MANAGER SUPPORT
    # =========================================================================
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.flush_to_supabase()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_call_logger(
    clinic_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    supabase_client: Optional[Any] = None,
) -> CallLogger:
    """
    Factory function to create a new CallLogger with auto-generated call_id.
    
    Usage:
        call_logger = create_call_logger(clinic_id="...", organization_id="...")
        call_logger.log_call_start(...)
    """
    return CallLogger(
        call_id=str(uuid4()),
        agent_id=os.getenv("LIVEKIT_AGENT_NAME", "telephony_agent"),
        environment=os.getenv("ENVIRONMENT", "production"),
        clinic_id=clinic_id,
        organization_id=organization_id,
        supabase_client=supabase_client,
    )

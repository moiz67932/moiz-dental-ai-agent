"""
Configuration and constants for the dental AI agent.

Contains all environment variables, API configurations, and tuning parameters.
"""

from __future__ import annotations

import os
import logging
from typing import Dict
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from openai import AsyncOpenAI
from supabase import create_client

from supabase_calendar_store import SupabaseCalendarStore

# =============================================================================
# Load Environment
# =============================================================================

load_dotenv(".env.local")

# =============================================================================
# 🚀 LATENCY OPTIMIZATION CONSTANTS — TUNING KNOBS FOR SNAPPY RESPONSES
# =============================================================================
"""
LATENCY TUNING GUIDE:
- These constants control responsiveness vs. conversation quality tradeoffs
- Adjust based on production metrics; log analysis will show impact
- Set LATENCY_DEBUG=1 env var to enable per-turn latency logging
"""

# Endpointing: How quickly agent detects user finished speaking
# WARNING: Do NOT go below 0.3s unless in controlled low-noise environment
MIN_ENDPOINTING_DELAY = float(os.getenv("MIN_ENDPOINTING_DELAY", "1.0"))  # 1.0s - allow breath without interruption
MAX_ENDPOINTING_DELAY = float(os.getenv("MAX_ENDPOINTING_DELAY", "1.2"))   # 1.5s max wait

# VAD (Voice Activity Detection) tuning
# WARNING: min_silence < 0.25s may cause premature cutoffs on pauses
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH", "0.1"))    # Keep at 0.1 (don't lower)
VAD_MIN_SILENCE_DURATION = float(os.getenv("VAD_MIN_SILENCE", "0.25"))  # 0.25s (was 0.3)

# Filler speech settings
FILLER_ENABLED = os.getenv("FILLER_ENABLED", "1") == "1"
FILLER_MAX_DURATION_MS = int(os.getenv("FILLER_MAX_MS", "700"))  # Hard cap on filler playback
FILLER_PHRASES = ["Okay…", "One moment…", "Got it…", "Hmm…"]  # Short phrases < 400ms spoken

# STT aggressive endpointing (Deepgram-specific)
STT_AGGRESSIVE_ENDPOINTING = os.getenv("STT_AGGRESSIVE", "1") == "1"

# Clinic context TTL cache (seconds) - DO NOT cache availability/schedule conflicts
CLINIC_CONTEXT_CACHE_TTL = int(os.getenv("CLINIC_CACHE_TTL", "60"))  # 60s TTL

# Latency debug mode - logs detailed timing per turn
LATENCY_DEBUG = os.getenv("LATENCY_DEBUG", "0") == "1"

# Mute noisy transport debug logs (reduces log-bloat in production)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

logger = logging.getLogger("snappy_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)

# =============================================================================
# ENVIRONMENT & APPLICATION CONFIG
# =============================================================================

ENVIRONMENT = (os.getenv("ENVIRONMENT") or "development").strip().lower()

# Telephony agent identity (must match SIP trunk dispatch rules)
LIVEKIT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "telephony_agent")

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Karachi")
DEFAULT_MIN = int(os.getenv("DEFAULT_APPT_MINUTES", "60"))
DEFAULT_PHONE_REGION = os.getenv("DEFAULT_PHONE_REGION", "PK")

# =============================================================================
# SUPABASE CONFIGURATION
# =============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================

# Async OpenAI client for RAG embeddings (text-embedding-3-small for speed)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# CALENDAR CONFIGURATION
# =============================================================================

calendar_store = SupabaseCalendarStore(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Google Calendar Config
GOOGLE_CALENDAR_AUTH_MODE = os.getenv("GOOGLE_CALENDAR_AUTH", "oauth")
GOOGLE_OAUTH_TOKEN_PATH = os.getenv("GOOGLE_OAUTH_TOKEN", "./google_token.json")
GOOGLE_CALENDAR_ID_DEFAULT = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# =============================================================================
# DATABASE CONSTANTS
# =============================================================================

BOOKED_STATUSES = ["scheduled", "confirmed"]

# Valid call_sessions.outcome enum values (from Supabase schema)
VALID_CALL_OUTCOMES = {
    "booked",
    "info_only",
    "missed",
    "transferred",
    "voicemail",
}

# Demo/fallback clinic ID for testing and when phone lookup fails
# Override via DEMO_CLINIC_ID environment variable in production
DEMO_CLINIC_ID = os.getenv("DEMO_CLINIC_ID", "5afce5fa-8436-43a3-af65-da29ccad7228")

# =============================================================================
# DEFAULT TREATMENT DURATIONS (minutes)
# =============================================================================

DEFAULT_TREATMENT_DURATIONS: Dict[str, int] = {
    "Teeth whitening": 60,
    "Cleaning": 30,
    "Consultation": 15,
    "Checkup": 30,
    "Check-up": 30,
    "Tooth pain": 30,
    "Extraction": 45,
    "Filling": 45,
    "Crown": 60,
    "Root canal": 90,
    "Emergency": 30,
}

DEFAULT_LUNCH_BREAK: Dict[str, str] = {"start": "13:00", "end": "14:00"}

# =============================================================================
# APPOINTMENT SETTINGS
# =============================================================================

APPOINTMENT_BUFFER_MINUTES = 15


def map_call_outcome(raw_outcome: str | None, booking_made: bool) -> str:
    """
    Maps internal call results to DB-safe call_outcome enum values.
    NEVER returns an invalid enum.
    """
    if booking_made:
        return "booked"

    if raw_outcome in VALID_CALL_OUTCOMES:
        return raw_outcome

    # Fallback: No booking and call ended normally → info_only
    return "info_only"

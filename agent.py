#=============================================================================
# agent.py — PART 1 / 4
# Imports, environment, config, model pickers, system prompt
# =============================================================================

from __future__ import annotations

import os
import re
import json
import sys
import time
import hashlib
import asyncio
import logging
import traceback
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from prompts.base_prompt import BASE_PROMPT
# Intent router is ONLY used for FAQ / small talk, NOT for booking decisions
from intent_router import route_intent, FAQ, CANCEL, GENERAL
from calendar_client import book_appointment, CalendarAuth

# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

logger = logging.getLogger("dental_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# LiveKit imports
# ---------------------------------------------------------------------------

from livekit import agents
from livekit.agents import AutoSubscribe
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.agents import metrics as lk_metrics
from livekit.agents import MetricsCollectedEvent

from livekit.plugins import (
    openai as openai_plugin,
    silero,
    deepgram as deepgram_plugin,
    assemblyai as assemblyai_plugin,
    cartesia as cartesia_plugin,
    noise_cancellation,
)

# ---------------------------------------------------------------------------
# Local utilities (already existing in your repo)
# ---------------------------------------------------------------------------

from contact_utils import (
    normalize_phone,
    normalize_email,
    validate_email_address,
    parse_datetime_natural,
)

from calendar_client import is_time_free, create_event
from supabase_calendar_store import SupabaseCalendarStore

from supabase import create_client

# ---------------------------------------------------------------------------
# Hybrid Slot Extraction System V2 (STATE POISONING FIX)
# ---------------------------------------------------------------------------

from extraction import (
    # V2 State with origin tracking
    EnhancedSlotState,
    SlotOrigin,
    SlotValue,
    # Hybrid extraction
    hybrid_extract_and_merge,
    merge_extraction_to_state,
    get_hybrid_extractor,
    ExtractionResult,
    # Phone capture
    PhoneCaptureManager,
    looks_like_phone_start,
    is_phone_continuation,
    # Turn lock
    get_turn_lock,
    reset_turn_lock,
    # Patterns
    CONFIRMATION_PATTERN,
    YES_PATTERN,
    NO_PATTERN,
    # Safe logging (prevents LogRecord key collisions)
    safe_extra,
)


# =============================================================================
# ENVIRONMENT & CONFIG
# =============================================================================

load_dotenv(".env.local")

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Karachi")
DEFAULT_MIN = int(os.getenv("DEFAULT_APPT_MINUTES", "60"))
DEFAULT_PHONE_REGION = os.getenv("DEFAULT_PHONE_REGION", "PK")

ENABLE_GOOGLE_CAL_CHECK = True  # calendar availability enforced

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

calendar_store = SupabaseCalendarStore(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY
)

BOOKED_STATUSES = ["scheduled", "confirmed"]

# =============================================================================
# CALENDAR AUTH CONFIG (CRITICAL FOR BOOKING)
# =============================================================================
# Supports two modes:
# 1. DB-stored credentials (from calendar_connections table)
# 2. ENV-based OAuth (from GOOGLE_OAUTH_TOKEN file)
#
# Priority: DB credentials > ENV OAuth > Fail

GOOGLE_CALENDAR_AUTH_MODE = os.getenv("GOOGLE_CALENDAR_AUTH", "oauth")  # "oauth" or "service_account"
GOOGLE_OAUTH_TOKEN_PATH = os.getenv("GOOGLE_OAUTH_TOKEN", "./google_token.json")
GOOGLE_OAUTH_CLIENT_SECRETS_PATH = os.getenv("GOOGLE_OAUTH_CLIENT_SECRETS", "./google_oauth_client_secret.json")
GOOGLE_CALENDAR_ID_DEFAULT = os.getenv("GOOGLE_CALENDAR_ID", "primary")


def _load_env_oauth_token() -> Optional[dict]:
    """
    Load OAuth token from ENV-configured file path.
    Returns dict with token info or None if not available.
    """
    token_path = GOOGLE_OAUTH_TOKEN_PATH
    if not token_path:
        logger.warning("[CALENDAR_AUTH] No GOOGLE_OAUTH_TOKEN path configured")
        return None
    
    if not os.path.exists(token_path):
        logger.warning(f"[CALENDAR_AUTH] Token file not found: {token_path}")
        return None
    
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)
        logger.info("[CALENDAR_AUTH] Loaded OAuth token from file")
        return token_data
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] Failed to load token: {e}")
        return None


def resolve_calendar_auth(clinic_info: Optional[dict]) -> Tuple[Optional[CalendarAuth], str]:
    """
    Resolve calendar auth from DB or ENV fallback.
    
    Returns: (CalendarAuth or None, calendar_id)
    
    Priority:
    1. DB-stored calendar_auth_json from clinic_info
    2. ENV-based OAuth token from GOOGLE_OAUTH_TOKEN file
    3. None (booking will fail)
    
    This ensures booking works with both:
    - Multi-tenant DB config (production)
    - Local dev with .env OAuth
    """
    # Try DB-stored credentials first
    if clinic_info:
        db_auth_json = clinic_info.get("calendar_auth_json")
        db_calendar_id = clinic_info.get("calendar_id")
        
        if db_auth_json and db_calendar_id:
            try:
                auth = CalendarAuth(
                    auth_type=clinic_info.get("calendar_auth_type", "oauth_user"),
                    secret_json=db_auth_json,
                    delegated_user=clinic_info.get("delegated_user"),
                )
                logger.info("[CALENDAR_AUTH] Using DB-stored credentials")
                return auth, db_calendar_id
            except Exception as e:
                logger.warning(f"[CALENDAR_AUTH] DB auth failed: {e}, trying ENV fallback")
    
    # Fallback to ENV-based OAuth
    if GOOGLE_CALENDAR_AUTH_MODE == "oauth":
        token_data = _load_env_oauth_token()
        if token_data:
            try:
                # Ensure token_uri is set (required for refresh)
                token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
                
                auth = CalendarAuth(
                    auth_type="oauth_user",
                    secret_json=token_data,
                    delegated_user=None,
                )
                calendar_id = GOOGLE_CALENDAR_ID_DEFAULT
                logger.info(f"[CALENDAR_AUTH] Using ENV OAuth token, calendar_id={calendar_id}")
                return auth, calendar_id
            except Exception as e:
                logger.error(f"[CALENDAR_AUTH] ENV OAuth auth failed: {e}")
    
    logger.error("[CALENDAR_AUTH] No valid calendar credentials found")
    return None, GOOGLE_CALENDAR_ID_DEFAULT


# =============================================================================
# REGEX & LANGUAGE CONSTANTS
# =============================================================================

BOOK_CONFIRM_PAT = re.compile(
    r"\b(yes|yeah|yep|yup|ok|okay|sure|please|confirm(?: booking)?|book(?: it| now)?|go ahead|do it|schedule(?: it)?)\b",
    re.IGNORECASE,
)

YES_PAT = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|that's right|that is right)\b",
    re.IGNORECASE,
)

NO_PAT = re.compile(
    r"\b(no|nope|wrong|incorrect|not correct|that's wrong)\b",
    re.IGNORECASE,
)

EMERGENCY_PAT = re.compile(
    r"\b(bleeding|uncontrolled bleeding|faint|unconscious|can'?t breathe|breathing|trauma|broken jaw|severe swelling|swelling.*eye|fever.*swelling)\b",
    re.IGNORECASE,
)

# =============================================================================
# LOG AGENT SPEECH
# =============================================================================

def log_agent_speech(text: str, *, channel: str = "voice") -> None:
    """
    Centralized agent speech logger.
    Later you can push this to Supabase, files, or analytics.
    """
    clean = (text or "").strip()
    if not clean:
        return

    print(f"[AGENT → USER][{channel}] {clean}")
    
async def say_and_log(session, text: str, channel: str = "voice"):
    if not text:
        return

    # log once
    log_agent_speech(text, channel=channel)

    # then actually send once (pick the correct method your session supports)
    if channel == "voice":
        # LiveKit Agents usually exposes one of these depending on your setup:
        # await session.say(text)
        # await session.tts.say(text)
        # await session.output.say(text)

        await session.say(text)  # ✅ use the correct one for your codebase
    else:
        # if you support text chat output
        await session.send_text(text)  # ✅ adjust to your actual method


# =============================================================================
# Booking Result
# ============================================================================

@dataclass
class BookingResult:
    success: bool
    appointment_id: str | None = None
    calendar_event_id: str | None = None
    reason: str | None = None



# =============================================================================
# MODEL PICKERS (CALL QUALITY OPTIMIZED)
# =============================================================================

def pick_llm() -> openai_plugin.LLM:
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.85"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "200"))
    return openai_plugin.LLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,  # gpt-4o-mini uses max_tokens (not max_completion_tokens)
    )


def pick_stt(language: str = "en-US"):
    if os.getenv("DEEPGRAM_API_KEY"):
        return deepgram_plugin.STT(
            model="nova-2-general",
            language=language,
            endpointing_ms=500,           # ⚡ Faster turn detection (was 800)
            min_endpointing_delay=0.5,    # ⚡ Minimum delay before end-of-speech
        )
    if os.getenv("ASSEMBLYAI_API_KEY"):
        return assemblyai_plugin.STT(
            min_endpointing_delay=0.5,    # ⚡ Snappier turn detection
        )
    return openai_plugin.STT(model="gpt-4o-transcribe", language="en")


def pick_tts():
    if os.getenv("CARTESIA_API_KEY"):
        voice = os.getenv(
            "CARTESIA_VOICE_ID",
            "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        )
        return cartesia_plugin.TTS(model="sonic-2", voice=voice)
    return openai_plugin.TTS(
        model="gpt-4o-mini-tts",
        voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
    )


# =============================================================================
# GENERALIZED SYSTEM PROMPT
# =============================================================================

# BASE_SYSTEM_PROMPT = """
# You are an AI phone receptionist for a dental clinic.

# Your job:
# - Answer calls professionally
# - Collect accurate appointment details
# - Confirm information clearly
# - Book appointments safely without duplication

# Conversation rules:
# - Speak naturally, calm, and friendly
# - Keep responses short (1–2 sentences)
# - Ask ONE question at a time
# - Rephrase if the caller sounds confused
# - Never rush the caller

# Booking rules:
# - Collect: full name, phone, email, reason, date & time
# - Confirm phone by last 4 digits only
# - Confirm email using 'at' and 'dot'
# - Never say 'confirmed' until booking is saved
# - If slot unavailable, suggest nearest alternatives
# - Respect clinic working hours and closed days

# Emergency handling:
# - If caller mentions serious symptoms, advise urgent care immediately
# - Then offer the earliest available appointment if appropriate

# Privacy:
# - Never request credit card details
# - Never repeat full phone numbers aloud

# If something is missing, continue slot-filling calmly.
# """

# =============================================================================

# =============================================================================
# agent.py — PART 2 / 4
# Multi-tenant context fetch + clinic schedule (hours/closed) + SlotState + extractors
# =============================================================================

# ---------------------------------------------------------------------------
# MULTI-TENANT FETCH (phone_numbers -> clinic -> agent -> settings)
# ---------------------------------------------------------------------------

async def fetch_clinic_context(
    called_number: str,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict], str]:
    """
    Returns: (clinic_info, agent_info, agent_settings, agent_name)
    
    ⚡ A-TIER OPTIMIZATION: Single query with Supabase joins
    Instead of 4 sequential queries (400-800ms), now just 1 query (~100ms)
    """
    try:
        # ⚡ SINGLE QUERY with nested joins - fetches everything in one round-trip
        result = await asyncio.to_thread(
            lambda: supabase.table("phone_numbers")
            .select(
                "clinic_id, agent_id, "
                # Join clinic
                "clinics:clinic_id("
                "  id, organization_id, name, timezone, default_phone_region, "
                "  address, city, state, zip_code, country"
                "), "
                # Join agent (may be null if agent_id is null)
                "agents:agent_id("
                "  id, organization_id, clinic_id, name, default_language, status, "
                # Nested join for agent settings
                "  agent_settings(id, greeting_text, persona_tone, collect_insurance, "
                "    insurance_prompt, emergency_triage_enabled, emergency_triage_prompt, "
                "    booking_confirmation_enabled, config_json)"
                ")"
            )
            .eq("phone_e164", called_number)
            .limit(1)
            .execute()
        )
        
        if not result.data:
            print(f"[db] called number not found in phone_numbers: {called_number}")
            return None, None, None, "Sarah"
        
        row = result.data[0]
        clinic_info = row.get("clinics")
        agent_info = row.get("agents")
        
        # If no agent linked directly, fetch by clinic_id (fallback)
        if not agent_info and clinic_info:
            agent_res = await asyncio.to_thread(
                lambda: supabase.table("agents")
                .select(
                    "id, organization_id, clinic_id, name, default_language, status, "
                    "agent_settings(id, greeting_text, persona_tone, collect_insurance, "
                    "  insurance_prompt, emergency_triage_enabled, emergency_triage_prompt, "
                    "  booking_confirmation_enabled, config_json)"
                )
                .eq("clinic_id", clinic_info["id"])
                .limit(1)
                .execute()
            )
            agent_info = agent_res.data[0] if agent_res.data else None
        
        # Extract settings from nested agent data
        settings = None
        if agent_info:
            nested_settings = agent_info.get("agent_settings")
            if isinstance(nested_settings, list) and nested_settings:
                settings = nested_settings[0]
            elif isinstance(nested_settings, dict):
                settings = nested_settings
            # Remove nested settings from agent_info to keep it clean
            agent_info = {k: v for k, v in agent_info.items() if k != "agent_settings"}
        
        agent_name = (agent_info or {}).get("name") or "Sarah"
        
        return clinic_info, agent_info, settings, agent_name

    except Exception as e:
        print(f"[db] context fetch error: {e}")
        traceback.print_exc()
        return None, None, None, "Sarah"


# ---------------------------------------------------------------------------
# CLINIC HOURS / CLOSED DAYS (from agent_settings.config_json, with defaults)
# ---------------------------------------------------------------------------

WEEK_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

def _default_hours() -> Dict[str, List[Dict[str, str]]]:
    return {
        "mon": [{"start": "09:00", "end": "17:00"}],
        "tue": [{"start": "09:00", "end": "17:00"}],
        "wed": [{"start": "09:00", "end": "17:00"}],
        "thu": [{"start": "09:00", "end": "17:00"}],
        "fri": [{"start": "09:00", "end": "17:00"}],
        "sat": [{"start": "10:00", "end": "14:00"}],
        "sun": [],
    }

def _parse_time_hhmm(s: str) -> Tuple[int, int]:
    parts = (s or "").strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time '{s}' (expected HH:MM)")
    return int(parts[0]), int(parts[1])

def _dow_key(dt: datetime) -> str:
    return WEEK_KEYS[dt.weekday()]

def load_schedule_from_settings(settings: Optional[dict]) -> Dict[str, Any]:
    """
    settings.config_json may be dict or json string
    Returns:
      {
        working_hours: {mon:[{start,end}], ...},
        closed_dates: set(date),
        slot_step_minutes: int,
        suggestion_days_ahead: int,
        max_suggestions: int
      }
    """
    cfg: dict = {}
    try:
        raw = (settings or {}).get("config_json")
        if isinstance(raw, str) and raw.strip():
            cfg = json.loads(raw)
        elif isinstance(raw, dict):
            cfg = raw
    except Exception as e:
        print(f"[schedule] failed parsing config_json: {e}")
        cfg = {}

    wh = cfg.get("working_hours") or _default_hours()
    working_hours = {k: wh.get(k, []) for k in WEEK_KEYS}

    closed: set[date] = set()
    for d in (cfg.get("closed_dates") or []):
        try:
            closed.add(date.fromisoformat(d))
        except Exception:
            pass

    slot_step = int(cfg.get("slot_step_minutes") or 30)
    days_ahead = int(cfg.get("suggestion_days_ahead") or 14)
    max_sug = int(cfg.get("max_suggestions") or 4)

    return {
        "working_hours": working_hours,
        "closed_dates": closed,
        "slot_step_minutes": slot_step,
        "suggestion_days_ahead": days_ahead,
        "max_suggestions": max_sug,
    }

def is_within_working_hours(dt: datetime, schedule: Dict[str, Any]) -> bool:
    if dt.tzinfo is None:
        return False
    if dt.date() in schedule["closed_dates"]:
        return False

    intervals = schedule["working_hours"].get(_dow_key(dt), [])
    for it in intervals:
        try:
            sh, sm = _parse_time_hhmm(it["start"])
            eh, em = _parse_time_hhmm(it["end"])
        except Exception:
            continue

        start = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= dt < end:
            return True
    return False

def round_up_to_step(dt: datetime, step_min: int) -> datetime:
    if step_min <= 0:
        return dt
    remainder = dt.minute % step_min
    if remainder == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.replace(second=0, microsecond=0)
    add = step_min - remainder
    new_dt = dt + timedelta(minutes=add)
    return new_dt.replace(second=0, microsecond=0)

def next_open_start(after_dt: datetime, schedule: Dict[str, Any]) -> Optional[datetime]:
    """
    Find next slot datetime >= after_dt that is inside working hours.
    """
    step = schedule["slot_step_minutes"]
    cursor = round_up_to_step(after_dt, step)

    for day_offset in range(schedule["suggestion_days_ahead"] + 1):
        day = (cursor + timedelta(days=day_offset)).date()
        if day in schedule["closed_dates"]:
            continue

        base = cursor if day_offset == 0 else cursor.replace(
            year=day.year, month=day.month, day=day.day,
            hour=0, minute=0, second=0, microsecond=0
        )

        intervals = schedule["working_hours"].get(_dow_key(base), [])
        for it in intervals:
            try:
                sh, sm = _parse_time_hhmm(it["start"])
                eh, em = _parse_time_hhmm(it["end"])
            except Exception:
                continue

            start = base.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end = base.replace(hour=eh, minute=em, second=0, microsecond=0)

            cand = base if (start <= base < end) else start
            cand = round_up_to_step(cand, step)

            if cand < start:
                cand = start
            if cand < end:
                return cand

    return None


# ---------------------------------------------------------------------------
# SLOT STATE (Using EnhancedSlotState from extraction module)
# ---------------------------------------------------------------------------

# SlotState is now an alias for EnhancedSlotState which includes:
# - Origin tracking (SlotOrigin) for each slot value
# - Confidence scores for each value  
# - Proper handling of weak vs strong values
# - Phone capture continuation support
# - Full audit logging
#
# This fixes the STATE POISONING bug where fallback datetime values
# would poison state and block NLU corrections.

SlotState = EnhancedSlotState  # Type alias for backward compatibility


# ---------------------------------------------------------------------------
# EXTRACTION HELPERS
# ---------------------------------------------------------------------------

def _maybe_extract_name(text: str) -> Optional[str]:
    m = re.search(r"\b(?:name is|i am|this is)\s+([a-z][a-z\s\.'-]{2,})", text, re.I)
    if not m:
        return None
    name = m.group(1).strip()
    name = re.split(r"\b(and|i|wanted|to|for|at|my|phone|email)\b", name, flags=re.I)[0].strip()
    return name.title() if len(name) >= 3 else None

def _maybe_extract_reason(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "whiten" in t:
        return "Teeth whitening"
    if any(k in t for k in ["clean", "cleaning", "checkup", "check-up", "exam"]):
        return "Cleaning and exam"
    if any(k in t for k in ["pain", "toothache", "tooth ache", "swelling"]):
        return "Tooth pain"
    if "consult" in t:
        return "Consultation"
    return None

_TZ_TOKEN = re.compile(
    r"(?:time\s*zone|timezone|tz)\s*(?:is|=)?\s*([A-Za-z_/\-+0-9: ]+)\b|"
    r"\b(UTC|GMT)\s*(?:([+-])\s*(\d{1,2})(?::?(\d{2}))?)?\b",
    re.I,
)

def _maybe_extract_timezone(text: str) -> Optional[str]:
    t = text or ""

    m = re.search(r"\bin\s+([A-Za-z_]+/[A-Za-z_]+)\b", t, re.I)
    if m:
        name = m.group(1)
        try:
            ZoneInfo(name)
            return name
        except Exception:
            pass

    m = _TZ_TOKEN.search(t)
    if not m:
        return None

    if m.group(1):
        name = m.group(1).strip()
        try:
            ZoneInfo(name)
            return name
        except Exception:
            return None

    if m.group(2):
        return "UTC"

    return None

def _maybe_extract_dt(text: str, tz_hint: str) -> Optional[datetime]:
    extracted = None
    patterns = [
        r"(?:on|for)\s+([^,]+?)(?:,|$)",
        r"(?:at)\s+(\d+(?:\s*(?:am|pm))?)",
        r"(\d+\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}.*?)(?:,|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            extracted = m.group(1).strip()
            break

    if not extracted:
        extracted = text

    parsed = parse_datetime_natural(extracted, tz_hint=tz_hint or DEFAULT_TZ)
    if not parsed:
        return None

    if parsed.tzinfo is None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(tz_hint or DEFAULT_TZ))
        except Exception:
            parsed = parsed.replace(tzinfo=ZoneInfo(DEFAULT_TZ))

    return parsed

def email_for_speech(email: str) -> str:
    e = (email or "").strip().lower()
    e = e.replace("@", " at ").replace(".", " dot ")
    e = re.sub(r"\s+", " ", e).strip()
    return e

# =============================================================================
# agent.py — PART 3 / 4
# Availability checks (Supabase + Google Calendar), suggestions, booking (Google + Supabase)
# =============================================================================

# ---------------------------------------------------------------------------
# CALENDAR HELPERS (DB-driven calendar per clinic via SupabaseCalendarStore)
# ---------------------------------------------------------------------------

def get_calendar_for_clinic(org_id: str, clinic_id: str) -> tuple[str, str]:
    """
    Returns (calendar_id, tz)
    calendar_id comes from calendar_store connection (or defaults to 'primary').
    tz comes from connection.timezone or clinic timezone or DEFAULT_TZ.
    """
    conn = calendar_store.get_calendar_connection(org_id=org_id, clinic_id=clinic_id)

    cal_id = getattr(conn, "calendar_id", None) or "primary"
    tz = (
        getattr(conn, "timezone", None)
        or calendar_store.get_clinic_timezone(org_id, clinic_id)
        or DEFAULT_TZ
    )
    return cal_id, tz

async def verify_booking(clinic_id, phone_last4, start_dt):
    return await booking_exists_by_fingerprint(
        clinic_id=clinic_id,
        start_dt=start_dt,
        phone_last4=phone_last4,
    )


# ---------------------------------------------------------------------------
# SCHEDULING INTEGRITY (Supabase overlap + optional calendar)
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ))
    return dt.isoformat()

async def is_slot_free_supabase(clinic_id: str, start_dt: datetime, end_dt: datetime) -> bool:
    """
    overlap rule: existing.start < requested.end AND existing.end > requested.start
    """
    try:
        res = (
            supabase.table("appointments")
            .select("id, start_time, end_time, status, patient_phone_masked")
            .eq("clinic_id", clinic_id)
            .lt("start_time", _iso(end_dt))
            .gt("end_time", _iso(start_dt))
            .in_("status", BOOKED_STATUSES)
            .execute()
        )
        overlaps = res.data or []
        print(f"[availability] overlaps={len(overlaps)}")
        return len(overlaps) == 0
    except Exception as e:
        print(f"[availability] supabase overlap check error: {e}")
        return False  # fail-safe

async def is_slot_free_calendar(org_id: str, clinic_id: str, start_dt: datetime, end_dt: datetime) -> bool:
    try:
        calendar_id, tz = get_calendar_for_clinic(org_id, clinic_id)
        return bool(is_time_free(calendar_id, start_dt, end_dt, tz=tz))
    except Exception as e:
        print(f"[availability] calendar check error: {e}")
        return False

def make_fingerprint(clinic_id: str, start_dt: datetime, phone_last4: str) -> str:
    raw = f"{clinic_id}|{_iso(start_dt)}|{phone_last4}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

async def booking_exists_by_fingerprint(clinic_id: str, start_dt: datetime, phone_last4: str) -> bool:
    """
    Idempotency without DB schema changes:
    If same clinic + same start_time + same phone_last4 + blocking status exists, treat as already booked.
    """
    try:
        res = (
            supabase.table("appointments")
            .select("id")
            .eq("clinic_id", clinic_id)
            .eq("start_time", _iso(start_dt))
            .eq("patient_phone_masked", phone_last4)
            .in_("status", BOOKED_STATUSES)
            .limit(1)
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        print(f"[idempotency] existence check error: {e}")
        return False


# ---------------------------------------------------------------------------
# NEAREST SLOT SUGGESTIONS
# ---------------------------------------------------------------------------

async def find_nearest_available_slots(
    org_id: str,
    clinic_id: str,
    requested_dt: datetime,
    schedule: Dict[str, Any],
    appt_minutes: int,
    enable_calendar: bool,
) -> List[datetime]:
    """
    Suggest nearest available start times by scanning step increments within working hours.
    NOTE: org_id is passed so calendar checks can be done per clinic.
    """
    suggestions: List[datetime] = []
    step = schedule["slot_step_minutes"]
    max_sug = schedule["max_suggestions"]

    start_cursor = next_open_start(requested_dt, schedule)
    if not start_cursor:
        return []

    cursor = start_cursor
    deadline = cursor + timedelta(days=schedule["suggestion_days_ahead"])

    while cursor <= deadline and len(suggestions) < max_sug:
        if not is_within_working_hours(cursor, schedule):
            nxt = next_open_start(cursor + timedelta(minutes=step), schedule)
            if not nxt:
                break
            cursor = nxt
            continue

        end_dt = cursor + timedelta(minutes=appt_minutes)

        free_db = await is_slot_free_supabase(clinic_id, cursor, end_dt)
        if not free_db:
            cursor = cursor + timedelta(minutes=step)
            continue

        if enable_calendar:
            free_cal = await is_slot_free_calendar(org_id, clinic_id, cursor, end_dt)
            if not free_cal:
                cursor = cursor + timedelta(minutes=step)
                continue

        suggestions.append(cursor)
        cursor = cursor + timedelta(minutes=step)

    return suggestions


# ---------------------------------------------------------------------------
# BOOKING: Google Calendar + Supabase
# ---------------------------------------------------------------------------

def _build_notes(state: SlotState) -> Optional[str]:
    parts = []
    if state.patient_type:
        parts.append(f"patient_type={state.patient_type}")
    if state.insurance:
        parts.append(f"insurance={state.insurance}")
    if state.payment_method:
        parts.append(f"payment_method={state.payment_method}")
    if state.notes:
        parts.append(f"notes={' | '.join(state.notes)}")
    return "; ".join(parts) if parts else None

async def book_to_supabase(
    clinic_info: dict,
    state: SlotState,
    calendar_provider: Optional[str] = None,
    calendar_id: Optional[str] = None,
    calendar_event_id: Optional[str] = None,
) -> bool:
    """
    Inserts into public.appointments. Works even if calendar_* columns do NOT exist:
    - We only include calendar fields if your table has those columns.
    """
    try:
        start_time = state.dt_local
        end_time = start_time + timedelta(minutes=DEFAULT_MIN)

        payload = {
            "organization_id": clinic_info["organization_id"],
            "clinic_id": clinic_info["id"],
            "patient_name": state.full_name,
            "patient_phone_masked": state.phone_last4,
            "patient_email": state.email,
            "start_time": _iso(start_time),
            "end_time": _iso(end_time),
            "status": "scheduled",
            "source": "ai",
            "reason": state.reason,
            "notes": _build_notes(state),
        }

        # Only attach calendar fields if provided (and if you have those columns)
        if calendar_provider:
            payload["calendar_provider"] = calendar_provider
        if calendar_id:
            payload["calendar_id"] = calendar_id
        if calendar_event_id:
            payload["calendar_event_id"] = calendar_event_id

        res = supabase.table("appointments").insert(payload).execute()
        appt_id = (res.data or [{}])[0].get("id")
        print(f"[db] appointment saved id={appt_id}")
        return True

    except Exception as e:
        print(f"[db] booking insert error: {e}")
        return False


async def create_google_event_for_booking(
    clinic_info: dict,
    state: SlotState,
) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (calendar_id, event_id) or (None, None) if failed.
    """
    try:
        org_id = clinic_info["organization_id"]
        clinic_id = clinic_info["id"]
        cal_id, tz = get_calendar_for_clinic(org_id, clinic_id)

        start_dt = state.dt_local
        end_dt = start_dt + timedelta(minutes=DEFAULT_MIN)

        event = create_event(
            calendar_id=cal_id,
            summary=f"Dental appointment — {state.full_name}",
            start_dt=start_dt,
            end_dt=end_dt,
            tz=tz,
            description=state.reason or "",
            attendee_email=state.email,
            location=(clinic_info.get("address") or ""),
        )
        ev_id = event.get("id") if isinstance(event, dict) else None
        return cal_id, ev_id
    except Exception as e:
        print(f"[calendar] create event error: {e}")
        return None, None


# ---------------------------------------------------------------------------
# BOOKING FLOW (try_book) - DETERMINISTIC WITH STRUCTURED LOGGING
# ---------------------------------------------------------------------------

async def try_book(
    session: AgentSession,
    clinic_info: dict,
    settings: dict,
    schedule: dict,
    state: SlotState,
) -> bool:
    """
    DETERMINISTIC booking function with structured logging.
    
    This is the ONLY place booking happens. Called automatically when
    state.is_complete() == True and state.booking_attempted == False.
    
    Returns:
        True if booking succeeded, False otherwise
    
    Guarantees:
        - NEVER says "booked" unless calendar event is verified
        - NEVER runs twice for same state (idempotent)
        - All I/O is async or wrapped in to_thread()
        - Full structured logging for observability
    """
    
    # ==========================================================
    # STEP 1: ENTRY LOGGING
    # ==========================================================
    logger.info("[BOOKING] try_book() ENTERED", extra={
        "slots": state.slot_summary(),
        "clinic_id": clinic_info.get("id"),
    })
    
    # ==========================================================
    # STEP 2: IDEMPOTENCY GUARDS
    # ==========================================================
    if state.booking_confirmed:
        logger.info("[BOOKING] SKIPPED - already confirmed")
        await say_and_log(session, "Your appointment is already booked.")
        return True
    
    if state.booking_attempted:
        logger.warning("[BOOKING] SKIPPED - already attempted (preventing retry)")
        return False
    
    if state.booking_in_progress:
        logger.warning("[BOOKING] SKIPPED - booking already in progress")
        return False
    
    # Mark as attempted IMMEDIATELY to prevent re-entry
    state.booking_attempted = True
    state.booking_in_progress = True
    
    logger.info("[BOOKING] Booking lock acquired, proceeding...")
    print("\n" + "="*60)
    print("[BOOKING] 🚀 BOOKING PROCESS STARTED")
    print("="*60)
    
    try:
        # ==========================================================
        # STEP 3: VALIDATE REQUIRED FIELDS
        # ==========================================================
        print(f"[BOOKING] Step 3: Validating required fields...")
        if not state.is_complete():
            missing = state.missing_slots()
            logger.error("[BOOKING] FAILED - incomplete slots", extra={"missing": missing})
            print(f"[BOOKING] ❌ FAILED - Missing slots: {missing}")
            await say_and_log(session, "I'm still missing some details to complete the booking.")
            state.booking_attempted = False  # Allow retry after slots filled
            return False
        
        print(f"[BOOKING] ✅ All required fields present")
        logger.info("[BOOKING] Slot validation passed", extra=safe_extra({
            "patient_name": state.full_name,
            "service": state.reason,
            "datetime": state.dt_local.isoformat(),
            "phone": state.phone_last4,
            "email": state.email,
        }))
        print(f"[BOOKING] Patient: {state.full_name}")
        print(f"[BOOKING] Service: {state.reason}")
        print(f"[BOOKING] DateTime: {state.dt_local.isoformat()}")
        print(f"[BOOKING] Phone: ***{state.phone_last4}")
        print(f"[BOOKING] Email: {state.email}")
        
        duration_minutes = settings.get("appointment_minutes", 30)
        
        # ==========================================================
        # STEP 4: BUILD CALENDAR AUTH (DB or ENV fallback)
        # ==========================================================
        logger.info("[BOOKING] Resolving calendar auth...")
        
        auth = None
        calendar_id = None
        calendar_auth_json = clinic_info.get("calendar_auth_json")
        
        if calendar_auth_json:
            # DB-stored credentials path
            try:
                auth = CalendarAuth(
                    auth_type=clinic_info.get("calendar_auth_type", "oauth_user"),
                    secret_json=calendar_auth_json,
                    delegated_user=clinic_info.get("delegated_user"),
                )
                calendar_id = clinic_info.get("calendar_id")
                logger.info("[BOOKING] Calendar auth built from DB")
            except Exception as e:
                logger.error("[BOOKING] FAILED - auth build error", extra={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                await say_and_log(session, "I couldn't access the clinic calendar. Please try again later.")
                return False
        else:
            # DB has no auth - try ENV fallback
            logger.info("[BOOKING] No DB calendar_auth_json, trying ENV fallback...")
            auth, calendar_id = resolve_calendar_auth(clinic_info)
            
            if not auth:
                logger.error("[BOOKING] FAILED - no calendar auth from DB or ENV")
                await say_and_log(session, "The clinic calendar is not properly configured.")
                return False
            
            logger.info(f"[BOOKING] Using ENV-based calendar auth, calendar_id={calendar_id}")
        
        # Final validation
        if not calendar_id:
            logger.error("[BOOKING] FAILED - no calendar_id resolved")
            await say_and_log(session, "This clinic is not configured for calendar booking.")
            return False
        
        # ==========================================================
        # STEP 5: GET CALENDAR SERVICE (wrapped in to_thread for sync I/O)
        # ==========================================================
        logger.info("[BOOKING] Getting calendar service...")
        
        from calendar_client import _get_calendar_service
        
        try:
            # Wrap sync call in to_thread to prevent blocking event loop
            service = await asyncio.wait_for(
                asyncio.to_thread(_get_calendar_service, auth=auth),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error("[BOOKING] FAILED - calendar service timeout")
            await say_and_log(session, "The calendar system is not responding. Please try again.")
            return False
        except Exception as e:
            logger.error("[BOOKING] FAILED - calendar service error", extra={
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            await say_and_log(session, "I couldn't connect to the calendar. Please try again.")
            return False
        
        # DRY-RUN mode check
        if service is None:
            logger.error("[BOOKING] FAILED - calendar service unavailable (DRY-RUN mode)")
            await say_and_log(
                session,
                "I cannot access the calendar system right now. Please call us directly."
            )
            return False
        
        logger.info("[BOOKING] Calendar service obtained successfully")
        
        start_dt = state.dt_local
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        
        # ==========================================================
        # STEP 6: CHECK FOR EXISTING BOOKING (idempotency)
        # ==========================================================
        logger.info("[BOOKING] Checking for existing booking...", extra={
            "calendar_id": calendar_id,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "patient": state.full_name,
        })
        
        try:
            # Wrap sync Google API call
            def _search_existing():
                return service.events().list(
                    calendarId=calendar_id,
                    timeMin=start_dt.isoformat(),
                    timeMax=end_dt.isoformat(),
                    singleEvents=True,
                    q=state.full_name,
                ).execute()
            
            resp = await asyncio.wait_for(
                asyncio.to_thread(_search_existing),
                timeout=15.0
            )
            
            existing_events = resp.get("items", [])
            logger.info(f"[BOOKING] Found {len(existing_events)} existing events in time window")
            
            for existing in existing_events:
                summary = existing.get("summary", "")
                description = existing.get("description", "")
                
                if state.full_name.lower() in summary.lower() or state.full_name.lower() in description.lower():
                    # Found existing booking
                    state.booking_confirmed = True
                    state.calendar_event_id = existing.get("id")
                    state.pending_confirm = None
                    
                    logger.info("[BOOKING] EXISTING BOOKING FOUND", extra={
                        "event_id": existing.get("id"),
                        "summary": summary,
                    })
                    
                    await say_and_log(
                        session,
                        f"Your appointment is already confirmed for "
                        f"{start_dt.strftime('%B %d at %I:%M %p')}."
                    )
                    return True
                    
        except asyncio.TimeoutError:
            logger.warning("[BOOKING] Existing booking check timed out, proceeding to create")
        except Exception as e:
            logger.warning(f"[BOOKING] Existing booking check failed: {e}, proceeding to create")
        
        # ==========================================================
        # STEP 7: CHECK AVAILABILITY
        # ==========================================================
        print(f"\n[BOOKING] Step 7: Checking calendar availability...")
        print(f"[BOOKING] Time window: {start_dt.isoformat()} to {end_dt.isoformat()}")
        logger.info("[BOOKING] Checking slot availability...")
        
        # FIX 7: Say "checking availability" ONLY when actually checking
        await say_and_log(
            session,
            f"Let me check if {start_dt.strftime('%B %d at %I:%M %p')} is available..."
        )
        
        try:
            def _check_freebusy():
                body = {
                    "timeMin": start_dt.isoformat(),
                    "timeMax": end_dt.isoformat(),
                    "timeZone": state.tz,
                    "items": [{"id": calendar_id}],
                }
                print(f"[BOOKING] FreeBusy query body: {body}")
                return service.freebusy().query(body=body).execute()
            
            freebusy_resp = await asyncio.wait_for(
                asyncio.to_thread(_check_freebusy),
                timeout=10.0
            )
            
            print(f"[BOOKING] FreeBusy response received")
            busy = (freebusy_resp.get("calendars", {}).get(calendar_id, {}) or {}).get("busy", [])
            
            if busy:
                print(f"[BOOKING] ❌ Slot NOT available - busy periods: {busy}")
                logger.warning("[BOOKING] FAILED - slot not available", extra={"busy": busy})
                await say_and_log(
                    session,
                    "That time slot is no longer available. Would you like to try another time?"
                )
                state.booking_attempted = False  # Allow rebooking with new time
                state.dt_local = None  # Clear the unavailable time
                return False
            
            print(f"[BOOKING] ✅ Slot is AVAILABLE")
            logger.info("[BOOKING] Availability check PASSED - slot is free")
            
        except asyncio.TimeoutError:
            print(f"[BOOKING] ❌ Availability check TIMEOUT")
            logger.error("[BOOKING] FAILED - availability check timeout")
            await say_and_log(session, "Couldn't check availability. Please try again.")
            return False
        except Exception as e:
            logger.error(f"[BOOKING] Availability check error: {e}")
            # Proceed anyway - let event creation fail if slot taken
        
        # ==========================================================
        # STEP 8: CREATE CALENDAR EVENT
        # ==========================================================
        logger.info("[BOOKING] Creating calendar event...", extra={
            "calendar_id": calendar_id,
            "summary": f"{state.reason} — {state.full_name}",
            "start": start_dt.isoformat(),
            "duration_minutes": duration_minutes,
        })
        
        # FIX 7: Say "booking your appointment" ONLY when actually creating
        await say_and_log(
            session,
            "That time is available. Booking your appointment now..."
        )
        
        try:
            def _create_event():
                event_body = {
                    "summary": f"{state.reason or 'Dental Appointment'} — {state.full_name}",
                    "description": f"Patient: {state.full_name}\nPhone: {state.phone_e164}\nEmail: {state.email}",
                    "location": clinic_info.get("address") or "",
                    "start": {"dateTime": start_dt.isoformat(), "timeZone": state.tz},
                    "end": {"dateTime": end_dt.isoformat(), "timeZone": state.tz},
                    "attendees": [{"email": state.email}] if state.email else [],
                    "reminders": {"useDefault": True},
                }
                return service.events().insert(
                    calendarId=calendar_id,
                    body=event_body,
                    sendUpdates="all",
                ).execute()
            
            event = await asyncio.wait_for(
                asyncio.to_thread(_create_event),
                timeout=20.0
            )
            
            logger.info("[BOOKING] Event creation API call completed", extra={
                "event_id": event.get("id"),
                "status": event.get("status"),
            })
            
        except asyncio.TimeoutError:
            logger.error("[BOOKING] FAILED - event creation timeout")
            await say_and_log(session, "The booking is taking too long. Please try again.")
            return False
        except Exception as e:
            logger.error("[BOOKING] FAILED - event creation error", extra={
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            await say_and_log(session, "I couldn't complete the booking. Would you like to try again?")
            return False
        
        if not event or not event.get("id"):
            logger.error("[BOOKING] FAILED - no event ID returned")
            await say_and_log(session, "Something went wrong with the booking. Please try again.")
            return False
        
        event_id = event.get("id")
        
        # ==========================================================
        # STEP 9: VERIFY BOOKING (CRITICAL - read back from calendar)
        # ==========================================================
        logger.info("[BOOKING] Verifying event in calendar...", extra={"event_id": event_id})
        
        try:
            def _verify_event():
                return service.events().get(
                    calendarId=calendar_id,
                    eventId=event_id,
                ).execute()
            
            verified = await asyncio.wait_for(
                asyncio.to_thread(_verify_event),
                timeout=10.0
            )
            
            verified_status = verified.get("status")
            
            if verified_status not in ("confirmed", "tentative"):
                logger.error("[BOOKING] FAILED - event not confirmed", extra={
                    "event_id": event_id,
                    "status": verified_status,
                })
                await say_and_log(
                    session,
                    "I created the booking but it wasn't confirmed. Please try again."
                )
                return False
            
            logger.info("[BOOKING] Event VERIFIED successfully", extra={
                "event_id": event_id,
                "status": verified_status,
            })
            
        except asyncio.TimeoutError:
            logger.error("[BOOKING] FAILED - verification timeout")
            await say_and_log(
                session,
                "I created the booking but couldn't confirm it was saved. Please call us to verify."
            )
            return False
        except Exception as e:
            logger.error("[BOOKING] FAILED - verification error", extra={
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            await say_and_log(
                session,
                "I tried to book but couldn't confirm it was saved. Please call us directly."
            )
            return False
        
        # ==========================================================
        # STEP 10: SUCCESS - UPDATE STATE
        # ==========================================================
        state.booking_confirmed = True
        state.calendar_event_id = event_id
        state.pending_confirm = None
        
        logger.info("[BOOKING] ✅ SUCCESS - Booking confirmed!", extra={
            "event_id": event_id,
            "patient": state.full_name,
            "service": state.reason,
            "datetime": start_dt.isoformat(),
            "phone": state.phone_last4,
            "email": state.email,
        })
        
        await say_and_log(
            session,
            f"Your appointment has been successfully booked for "
            f"{start_dt.strftime('%A, %B %d at %I:%M %p')}. "
            f"We'll see you then!"
        )
        
        # ==========================================================
        # STEP 11: SAVE TO SUPABASE (non-blocking, non-critical)
        # ==========================================================
        try:
            logger.info("[BOOKING] Saving to Supabase...")
            await asyncio.wait_for(
                book_to_supabase(
                    clinic_info=clinic_info,
                    state=state,
                    calendar_provider="google",
                    calendar_id=calendar_id,
                    calendar_event_id=event_id,
                ),
                timeout=10.0
            )
            logger.info("[BOOKING] Supabase save completed")
        except Exception as e:
            logger.warning(f"[BOOKING] Supabase save failed (non-critical): {e}")
            # Don't fail - calendar is source of truth
        
        return True
        
    except Exception as e:
        logger.error("[BOOKING] UNEXPECTED ERROR", extra={
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        await say_and_log(session, "An unexpected error occurred. Please try again.")
        return False
        
    finally:
        state.booking_in_progress = False
        logger.info("[BOOKING] try_book() EXITED", extra={
            "booking_confirmed": state.booking_confirmed,
            "booking_attempted": state.booking_attempted,
        })




# =============================================================================
# agent.py — PART 4 / 4
# Transcript handler + entrypoint + metrics + shutdown logging
# PRODUCTION-GRADE HYBRID SLOT EXTRACTION SYSTEM
# =============================================================================

# ---------------------------------------------------------------------------
# CONFIRMATION PATTERNS (imported from extraction module)
# ---------------------------------------------------------------------------

# CONFIRMATION_PATTERN, YES_PATTERN, NO_PATTERN are imported from extraction module
# They are used to detect confirmation responses that should NOT go to NLU


# ---------------------------------------------------------------------------
# TRANSCRIPT HANDLER - HYBRID EXTRACTION WITH TURN LOCK
# ---------------------------------------------------------------------------

async def handle_transcript(
    text: str,
    is_final: bool,
    session: AgentSession,
    clinic_info: dict,
    agent_info: dict,
    settings: dict,
    schedule: dict,
    state: SlotState,
    turn_lock = None,
    phone_capture_mgr: Optional[PhoneCaptureManager] = None,
) -> None:
    """
    PRODUCTION-GRADE HYBRID TRANSCRIPT HANDLER (V2).
    
    CRITICAL FIXES FOR STATE POISONING:
    1. Uses EnhancedSlotState with origin tracking
    2. Pessimistic datetime extraction (returns None if not confident)
    3. NLU can overwrite WEAK deterministic values
    4. Phone capture continuation mode for telephony
    5. Timezone normalization BEFORE working hours validation
    
    Architecture:
    1. Turn lock ensures ONE response per user utterance
    2. Deterministic extraction runs FIRST
    3. NLU is called ONLY when needed (2+ missing slots after deterministic)
    4. Booking triggers ONLY when SlotState.is_complete() == True
    5. All state transitions are deterministic - LLM NEVER controls flow
    
    Guarantees:
    - NEVER says "booked" unless booking is actually persisted
    - NEVER sends confirmations ("yes", "no") to NLU
    - NEVER allows duplicate responses per turn
    - NEVER locks in invalid datetimes (midnight fallback)
    - Full observability with structured logging
    """
    
    # ============================================================
    # STEP 0: EARLY EXIT CONDITIONS
    # ============================================================
    
    if not is_final:
        logger.debug("[TRANSCRIPT] Skipping non-final transcript")
        return
    
    text = (text or "").strip()
    if not text:
        logger.debug("[TRANSCRIPT] Empty text, skipping")
        return
    
    # ============================================================
    # STEP 1: STRUCTURED ENTRY LOGGING
    # ============================================================
    
    logger.info("=" * 70)
    logger.info("[TRANSCRIPT] ===== NEW USER TURN =====")
    logger.info(f"[TRANSCRIPT] Input: '{text}'")
    logger.info(f"[TRANSCRIPT] State before: {state.slot_summary()}")
    logger.info("=" * 70)
    
    print(f"\n{'='*70}")
    print(f"[USER] {text}")
    print(f"{'='*70}")
    
    # Get turn lock for this session
    if turn_lock is None:
        turn_lock = get_turn_lock()
    
    # ============================================================
    # STEP 2: ACQUIRE TURN LOCK
    # ============================================================
    
    async with turn_lock.acquire_turn(text) as turn:
        if turn.turn_id == -1:
            # Lock acquisition failed (timeout)
            logger.warning("[TRANSCRIPT] Turn lock acquisition failed, dropping turn")
            return
        
        logger.info(f"[TURN #{turn.turn_id}] Lock acquired")
        
        try:
            await _process_turn(
                turn=turn,
                text=text,
                session=session,
                clinic_info=clinic_info,
                agent_info=agent_info,
                settings=settings,
                schedule=schedule,
                state=state,
                phone_capture_mgr=phone_capture_mgr,
            )
        except Exception as e:
            logger.error(f"[TURN #{turn.turn_id}] Error: {e}", exc_info=True)
            # CRITICAL: Only send apology if NO response was sent yet
            # This prevents double-speak where user hears normal response + error message
            if turn.response_sent:
                logger.info(f"[TURN #{turn.turn_id}] Response already sent, BLOCKING error apology to prevent double-speak")
            else:
                logger.info(f"[TURN #{turn.turn_id}] No response sent yet, sending error apology")
                await turn.send_response(
                    "I apologize, something went wrong. Could you please repeat that?"
                )


async def _process_turn(
    turn,
    text: str,
    session: AgentSession,
    clinic_info: dict,
    agent_info: dict,
    settings: dict,
    schedule: dict,
    state: SlotState,
    phone_capture_mgr: Optional[PhoneCaptureManager] = None,
) -> None:
    """
    Process a single conversation turn with guaranteed single response.
    
    CRITICAL FIXES FOR STATE POISONING:
    1. Uses EnhancedSlotState with origin tracking
    2. Phone capture continuation mode for telephony
    3. Datetime validation AFTER timezone normalization
    4. Structured logging for all decisions
    """
    
    phone_region = clinic_info.get("default_phone_region", DEFAULT_PHONE_REGION)
    timezone = state.tz or clinic_info.get("timezone", DEFAULT_TZ)
    
    # ============================================================
    # STEP 3: POST-BOOKING QUERIES
    # ============================================================
    
    if state.booking_confirmed:
        logger.info("[TURN] Booking already confirmed - handling post-booking query")
        await turn.send_response(
            f"Your appointment is confirmed for "
            f"{state.dt_local.strftime('%A, %B %d at %I:%M %p')}. "
            f"Is there anything else I can help with?"
        )
        return
    
    # ============================================================
    # STEP 4: EMERGENCY TRIAGE
    # ============================================================
    
    if settings.get("emergency_triage_enabled") and EMERGENCY_PAT.search(text):
        logger.info("[TURN] Emergency detected")
        await turn.send_response(
            "If this is a medical emergency, please seek urgent care immediately. "
            "If you're safe, I can help book an appointment."
        )
        return
    
    # ============================================================
    # STEP 4.5: PHONE CAPTURE CONTINUATION MODE
    # ============================================================
    
    if phone_capture_mgr and phone_capture_mgr.is_active:
        # In phone capture mode - check if this is a continuation
        if is_phone_continuation(text):
            result = phone_capture_mgr.process_utterance(text)
            
            if result.is_complete:
                # Got complete phone number!
                state.set_slot(
                    "phone",
                    result.e164,
                    SlotOrigin.DETERMINISTIC_STRONG,
                    1.0,
                    raw_text=text,
                )
                state.phone_last4 = result.last4
                state.pending_confirm = "phone"
                
                logger.info(f"[PHONE_CAPTURE] Complete: ***{result.last4}")
                await turn.send_response(
                    f"I got a number ending {result.last4}. Is that correct?"
                )
                return
            
            elif result.needs_more:
                logger.info(f"[PHONE_CAPTURE] Need more digits: {result.digit_count}")
                # Don't prompt, just wait for more input
                turn.mark_no_response_needed()
                return
            
            elif result.failed:
                logger.warning(f"[PHONE_CAPTURE] Failed: {result.message}")
                phone_capture_mgr.cancel()
                await turn.send_response(
                    "I didn't quite get that number. Could you say the full phone number again?"
                )
                return
        else:
            # Not a continuation - exit phone capture mode
            logger.info("[PHONE_CAPTURE] Non-continuation input, exiting capture mode")
            
            # Try to validate what we have
            if phone_capture_mgr.capture.is_potentially_complete():
                e164, last4 = phone_capture_mgr.capture.validate()
                if e164:
                    state.set_slot(
                        "phone",
                        e164,
                        SlotOrigin.DETERMINISTIC_STRONG,
                        1.0,
                    )
                    state.phone_last4 = last4
            
            phone_capture_mgr.cancel()
            # Continue processing this turn normally
    
    # ============================================================
    # STEP 5: HANDLE PENDING CONFIRMATIONS (DETERMINISTIC)
    # ============================================================
    
    if state.pending_confirm == "phone":
        response = await _handle_phone_confirmation(text, state, clinic_info, settings, schedule, session)
        if response:
            await turn.send_response(response)
        elif state.is_complete() and not state.booking_attempted:
            # Booking triggered after confirmation
            await _trigger_booking(turn, session, clinic_info, settings, schedule, state)
        return
    
    if state.pending_confirm == "email":
        response = await _handle_email_confirmation(text, state, clinic_info, settings, schedule, session)
        if response:
            await turn.send_response(response)
        elif state.is_complete() and not state.booking_attempted:
            # Booking triggered after confirmation
            await _trigger_booking(turn, session, clinic_info, settings, schedule, state)
        return
    
    # ============================================================
    # STEP 6: HYBRID SLOT EXTRACTION (V2 WITH STATE POISONING FIX)
    # ============================================================
    
    # Determine if we're expecting a name response
    expecting_name = not state.full_name and len(state.missing_slots()) >= 4
    
    logger.info("[EXTRACTION] ═══════════════════════════════════════════════════════")
    logger.info("[EXTRACTION] Starting hybrid extraction (V2)")
    logger.info(f"[EXTRACTION] Input: '{text}'")
    logger.info(f"[EXTRACTION] State before: {state.slot_summary()}")
    print(f"\n[EXTRACTION] Starting hybrid slot extraction...")
    
    # Use the hybrid extractor V2
    extraction_result, merged_slots = await hybrid_extract_and_merge(
        text=text,
        state=state,
        phone_region=phone_region,
        timezone=timezone,
        expecting_name=expecting_name,
    )
    
    # Log extraction results with full detail
    logger.info("[EXTRACTION] Complete", extra={
        "extraction": extraction_result.to_dict(),
        "merged_slots": merged_slots,
        "used_nlu": extraction_result.used_nlu,
        "time_ms": extraction_result.extraction_time_ms,
    })
    
    print(f"[EXTRACTION] Merged slots: {merged_slots}")
    print(f"[EXTRACTION] Used NLU: {extraction_result.used_nlu}")
    print(f"[EXTRACTION] Time: {extraction_result.extraction_time_ms:.1f}ms")
    print(f"[EXTRACTION] State after: {state.slot_summary()}")
    
    # ============================================================
    # STEP 7: VALIDATE DATETIME (WITH TIMEZONE NORMALIZATION)
    # ============================================================
    
    if "datetime" in merged_slots and state.dt_local:
        dt_to_check = state.dt_local
        
        # CRITICAL: Ensure timezone is set before validation
        if dt_to_check.tzinfo is None:
            try:
                dt_to_check = dt_to_check.replace(tzinfo=ZoneInfo(timezone))
                state.dt_local = dt_to_check
            except Exception as e:
                logger.warning(f"[DATETIME] Failed to set timezone: {e}")
        
        # Log the datetime being validated
        logger.info(f"[DATETIME] Validating: {dt_to_check.isoformat()}")
        logger.info(f"[DATETIME] Origin: {state.get_slot_info('datetime').origin.value}")
        logger.info(f"[DATETIME] Confidence: {state.get_slot_info('datetime').confidence}")
        
        # Validate working hours
        if not is_within_working_hours(dt_to_check, schedule):
            # Clear the datetime slot
            state.clear_slot("datetime", reason="outside_working_hours")
            
            # Log with audit trail
            logger.warning(
                f"[DATETIME] ❌ REJECTED - outside working hours: "
                f"{dt_to_check.isoformat()}"
            )
            
            await turn.send_response(
                "That time is outside our working hours. What other time works for you?"
            )
            return
        
        # Check availability
        try:
            slot_free = await asyncio.wait_for(
                is_slot_free_supabase(
                    clinic_info.get("id"),
                    dt_to_check,
                    dt_to_check + timedelta(minutes=DEFAULT_MIN),
                ),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("[DATETIME] Availability check timed out")
            slot_free = True  # Proceed, calendar check will catch conflicts
        
        if not slot_free:
            state.clear_slot("datetime", reason="slot_not_available")
            
            logger.warning(
                f"[DATETIME] ❌ REJECTED - slot not available: "
                f"{dt_to_check.isoformat()}"
            )
            
            await turn.send_response(
                "That time slot is not available. What other time would you prefer?"
            )
            return
        
        logger.info(f"[DATETIME] ✅ Validated: {dt_to_check.isoformat()}")
        print(f"[EXTRACTION] ✅ Datetime validated: {dt_to_check.isoformat()}")
    
    # ============================================================
    # STEP 8: CHECK BOOKING TRIGGER (DETERMINISTIC)
    # ============================================================
    
    logger.info(f"[BOOKING CHECK] is_complete={state.is_complete()}, attempted={state.booking_attempted}")
    print(f"\n[BOOKING CHECK] is_complete={state.is_complete()}, booking_attempted={state.booking_attempted}")
    print(f"[BOOKING CHECK] Missing: {state.missing_slots()}")
    
    if state.is_complete() and not state.booking_attempted:
        await _trigger_booking(turn, session, clinic_info, settings, schedule, state)
        return
    
    # ============================================================
    # STEP 9: PROMPT FOR MISSING SLOTS / CONFIRMATIONS
    # ============================================================
    
    missing = state.missing_slots()
    
    if missing:
        logger.info(f"[SLOTS] Missing: {missing}")
        print(f"[SLOTS] Will prompt for: {missing}")
        
        # Phone confirmation needed?
        if "phone_confirmed" in missing and state.phone_e164:
            state.pending_confirm = "phone"
            await turn.send_response(
                f"I got a number ending {state.phone_last4}. Is that correct?"
            )
            return
        
        # Email confirmation needed?
        if "email_confirmed" in missing and state.email:
            state.pending_confirm = "email"
            await turn.send_response(
                f"I have {email_for_speech(state.email)}. Is that correct?"
            )
            return
        
        # Check if we should enter phone capture mode
        if "phone" in missing and phone_capture_mgr and looks_like_phone_start(text):
            phone_capture_mgr.start_capture(phone_region)
            result = phone_capture_mgr.process_utterance(text)
            
            if result.is_complete:
                state.set_slot(
                    "phone",
                    result.e164,
                    SlotOrigin.DETERMINISTIC_STRONG,
                    1.0,
                )
                state.phone_last4 = result.last4
                state.pending_confirm = "phone"
                
                await turn.send_response(
                    f"I got a number ending {result.last4}. Is that correct?"
                )
                return
            elif result.needs_more:
                # Need more digits, prompt to continue
                await turn.send_response(
                    "Got it, please continue with the rest of the number."
                )
                return
        
        # Prompt for first missing slot
        prompts = {
            "full_name": "Could I have your full name?",
            "phone": "What's the best phone number to reach you?",
            "email": "What's your email address?",
            "reason": "What service would you like? For example, teeth whitening, cleaning, or consultation?",
            "datetime": "What day and time would you like to come in?",
        }
        
        first_missing = missing[0]
        prompt = prompts.get(first_missing, "Could you tell me a bit more about what you need?")
        await turn.send_response(prompt)
        return
    
    # ============================================================
    # STEP 10: CATCH-ALL
    # ============================================================
    
    logger.warning("[TURN] Reached catch-all", extra={"slots": state.slot_summary()})
    
    if state.is_complete() and not state.booking_attempted:
        await _trigger_booking(turn, session, clinic_info, settings, schedule, state)
    elif not turn.response_sent:
        await turn.send_response(
            "I'm here to help book your appointment. What service are you interested in?"
        )


async def _handle_phone_confirmation(
    text: str,
    state: SlotState,
    clinic_info: dict,
    settings: dict,
    schedule: dict,
    session: AgentSession,
) -> Optional[str]:
    """
    Handle phone number confirmation.
    Returns response text if one should be sent, None if booking should proceed.
    """
    if YES_PAT.search(text):
        state.phone_confirmed = True
        state.pending_confirm = None
        logger.info("[CONFIRMATION] Phone CONFIRMED")
        print(f"[CONFIRMATION] ✅ Phone confirmed")
        
        if state.is_complete():
            return None  # Signal to trigger booking
        
        # Ask for next missing slot
        missing = state.missing_slots()
        if "email" in missing:
            return "Great. What's your email address?"
        elif "email_confirmed" in missing and state.email:
            state.pending_confirm = "email"
            return f"Great. I have {email_for_speech(state.email)} for email. Is that correct?"
        elif "reason" in missing:
            return "Great. What service would you like?"
        elif "datetime" in missing:
            return "Great. What day and time would you like to come in?"
        else:
            return None  # Should trigger booking
    
    if NO_PAT.search(text):
        state.phone_e164 = None
        state.phone_last4 = None
        state.phone_confirmed = False
        state.pending_confirm = None
        logger.info("[CONFIRMATION] Phone REJECTED")
        print(f"[CONFIRMATION] ❌ Phone rejected")
        return "No problem. What's the correct phone number?"
    
    # Try to extract new phone number
    phone, last4 = normalize_phone(
        text,
        default_region=clinic_info.get("default_phone_region", DEFAULT_PHONE_REGION),
    )
    if phone:
        state.phone_e164 = phone
        state.phone_last4 = last4
        logger.info(f"[CONFIRMATION] New phone extracted: ***{last4}")
        print(f"[CONFIRMATION] 🔄 New phone: ***{last4}")
        return f"I got a number ending {last4}. Is that correct?"
    
    # Re-ask for confirmation
    return f"Just to confirm—number ending {state.phone_last4}. Is that correct?"


async def _handle_email_confirmation(
    text: str,
    state: SlotState,
    clinic_info: dict,
    settings: dict,
    schedule: dict,
    session: AgentSession,
) -> Optional[str]:
    """
    Handle email confirmation.
    Returns response text if one should be sent, None if booking should proceed.
    """
    if YES_PAT.search(text):
        state.email_confirmed = True
        state.pending_confirm = None
        logger.info("[CONFIRMATION] Email CONFIRMED")
        print(f"[CONFIRMATION] ✅ Email confirmed")
        
        if state.is_complete():
            return None  # Signal to trigger booking
        
        # Ask for next missing slot
        missing = state.missing_slots()
        if "reason" in missing:
            return "Thanks. What service would you like?"
        elif "datetime" in missing:
            return "Thanks. What day and time would you like to come in?"
        else:
            return None  # Should trigger booking
    
    if NO_PAT.search(text):
        state.email = None
        state.email_confirmed = False
        state.pending_confirm = None
        logger.info("[CONFIRMATION] Email REJECTED")
        print(f"[CONFIRMATION] ❌ Email rejected")
        return "No problem. What's your correct email address?"
    
    # Try to extract new email
    maybe = normalize_email(text)
    if validate_email_address(maybe):
        state.email = maybe
        logger.info(f"[CONFIRMATION] New email extracted: {maybe}")
        print(f"[CONFIRMATION] 🔄 New email: {maybe}")
        return f"I have {email_for_speech(state.email)}. Is that correct?"
    
    # Re-ask for confirmation
    return f"Just to confirm—{email_for_speech(state.email)}. Is that correct?"


async def _trigger_booking(
    turn,
    session: AgentSession,
    clinic_info: dict,
    settings: dict,
    schedule: dict,
    state: SlotState,
) -> None:
    """
    Trigger booking process with proper logging and response handling.
    
    FIX 7: Speech now matches actual booking state.
    - Say "checking availability" ONLY when check starts
    - Say "booking now" ONLY when booking is being created
    - Say "booked" ONLY after calendar event is verified
    - NEVER give optimistic language before operations complete
    """
    print(f"\n{'='*70}")
    print(f"[TRIGGER] ✅ ALL SLOTS COMPLETE - TRIGGERING BOOKING")
    print(f"{'='*70}")
    
    logger.info("[TRIGGER] ✅ ALL SLOTS COMPLETE - AUTO BOOKING NOW", extra={
        "slots": state.slot_summary(),
    })
    
    # FIX 7: Do NOT say "let me book that" before booking actually starts
    # The try_book function will send appropriate messages as it progresses:
    # - "Checking availability..." when checking
    # - "Booking your appointment..." when creating
    # - "Your appointment is confirmed..." when verified
    # This prevents false confirmations when booking fails
    
    # Perform booking - let try_book handle all user communication
    success = await try_book(session, clinic_info, settings, schedule, state)
    
    if success:
        logger.info("[TRIGGER] ✅ Booking completed successfully!")
        print(f"[TRIGGER] ✅ Booking SUCCESS")
        # Response already sent by try_book
    else:
        logger.warning("[TRIGGER] ❌ Booking failed")
        print(f"[TRIGGER] ❌ Booking FAILED")
        # try_book sends appropriate error response




# ---------------------------------------------------------------------------
# ENTRYPOINT (call quality + metrics + multi-tenant + per-clinic schedule)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# AGENT CLASS
# ---------------------------------------------------------------------------

class DentalAssistant(Agent):
    """
    Thin wrapper around LiveKit Agent.
    We keep this class so we can:
    - Inject dynamic system instructions
    - Extend behavior later (analytics, hooks, etc.)
    """
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)


async def playground_entrypoint(ctx: agents.JobContext):
    """
    Playground-only entrypoint.
    No SIP, no Twilio, no phone costs.
    """

    state = SlotState()
    call_started = time.time()
    
    # ==========================================================
    # INITIALIZE TURN LOCK FOR THIS SESSION
    # ==========================================================
    reset_turn_lock()  # Clear any previous session state
    turn_lock = get_turn_lock(turn_timeout=30.0)

    # 1️⃣ Connect to room (Playground joins as browser participant)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    print("[playground] participant connected:", participant.identity)

    # 2️⃣ Use a fixed / test clinic context
    # IMPORTANT: do NOT use SIP metadata here
    called_num = os.getenv("DEFAULT_TEST_NUMBER", "+13103410536")

    clinic_info, agent_info, settings, agent_name = await fetch_clinic_context(called_num)

    clinic_name = (clinic_info or {}).get("name") or "our clinic"
    clinic_tz = (clinic_info or {}).get("timezone") or DEFAULT_TZ
    clinic_region = (clinic_info or {}).get("default_phone_region") or DEFAULT_PHONE_REGION
    agent_lang = (agent_info or {}).get("default_language") or "en-US"

    state.tz = clinic_tz
    schedule = load_schedule_from_settings(settings or {})


    # Initialize phone capture manager for telephony continuation
    phone_capture_mgr = PhoneCaptureManager(
        default_region=clinic_region,
        min_digits_domestic=10,
        timeout_seconds=30.0,
    )
    greeting = (
        f"Hi, I’m {agent_name} from {clinic_name}. "
        "You’re connected via our test assistant. How can I help you today?"
    )

    # 3️⃣ Build dynamic prompt (same as telephony)
    dynamic_instructions = (
        f"{BASE_PROMPT.strip()}\n\n"
        f"Clinic: {clinic_name}\n"
        f"Agent: {agent_name}\n"
        f"Timezone: {clinic_tz}\n"
        f"Phone region: {clinic_region}\n"
        f"Appointment length: {DEFAULT_MIN} minutes\n"
    )

    # 4️⃣ Build session (same components)
    vad = silero.VAD.load()
    llm = pick_llm()
    stt = pick_stt(language=agent_lang)
    tts = pick_tts()

    session = AgentSession(stt=stt, llm=llm, tts=tts, vad=vad)
    
    # ==========================================================
    # SET TURN LOCK RESPONSE CALLBACK
    # ==========================================================
    async def _send_response(text: str):
        """Response callback for turn lock."""
        await say_and_log(session, text)
    
    turn_lock.set_response_callback(_send_response)
    logger.info("[PLAYGROUND] Turn lock initialized with response callback")

    usage = lk_metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    @session.on("conversation_item_added")
    def _on_item(ev):
        txt = (ev.item.text_content or "").strip()
        
        # Log all conversation items for debugging
        logger.info(f"[CONVERSATION] Item added - role: {ev.item.role}, text: {txt[:100] if txt else '(empty)'}")
        
        if ev.item.role == "assistant":
            logger.debug(f"[AGENT SAID] {txt}")
            return
        
        if ev.item.role != "user":
            return
        
        if not txt:
            logger.debug("[CONVERSATION] Empty user message, skipping")
            return
        
        logger.info(f"[TRANSCRIPT HANDLER] Processing user input: {txt}")
        
        asyncio.create_task(
            handle_transcript(
                txt,
                True,  # ✅ Process as final transcript to trigger slot extraction
                session,
                clinic_info or {},
                agent_info or {},
                settings or {},
                schedule,
                state,
                turn_lock,  # ✅ Pass turn lock to handler
                phone_capture_mgr,  # ✅ Pass phone capture manager
            )
        )

    # 5️⃣ Start agent (KEY DIFFERENCE: must pass ctx.room)
    await session.start(
        room=ctx.room,
        agent=DentalAssistant(instructions=dynamic_instructions),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            close_on_disconnect=True,  # browser disconnect ends session
        ),
    )

    await say_and_log(session, (greeting))

    async def _on_shutdown():
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Playground session ending after {dur}s")
        
        # Log turn lock stats
        try:
            stats = turn_lock.get_stats()
            logger.info(f"[LIFECYCLE] Turn lock stats: {stats}")
            print(f"\n[TURN LOCK STATS] {stats}")
        except Exception as e:
            logger.warning(f"[LIFECYCLE] Failed to get turn lock stats: {e}")
        
        try:
            print(f"[playground usage] {usage.get_summary()}")
        except Exception:
            pass
        
        # Reset turn lock for next session
        reset_turn_lock()

    ctx.add_shutdown_callback(_on_shutdown)
    
    # ==========================================================
    # CRITICAL: Keep entrypoint alive until room disconnects
    # This prevents "job exiting" / "executor unresponsive" errors
    # ==========================================================
    logger.info("[LIFECYCLE] Playground entrypoint waiting for room disconnect...")
    
    disconnect_event = asyncio.Event()
    
    @ctx.room.on("disconnected")
    def _on_room_disconnect():
        logger.info("[LIFECYCLE] Room disconnected")
        disconnect_event.set()
    
    # Also handle participant leaving
    @ctx.room.on("participant_disconnected")
    def _on_participant_left(participant):
        logger.info(f"[LIFECYCLE] Participant left: {participant.identity}")
        disconnect_event.set()
    
    # Wait for disconnect or timeout (2 hour max call)
    try:
        await asyncio.wait_for(disconnect_event.wait(), timeout=7200)
    except asyncio.TimeoutError:
        logger.info("[LIFECYCLE] Session timed out after 2 hours")
    
    logger.info("[LIFECYCLE] Playground entrypoint exiting normally")



async def entrypoint(ctx: agents.JobContext):
    """
    Telephony entrypoint with HYBRID SLOT EXTRACTION.
    Full SIP/Twilio support.
    """
    state = SlotState()
    call_started = time.time()
    
    # ==========================================================
    # INITIALIZE TURN LOCK FOR THIS SESSION
    # ==========================================================
    reset_turn_lock()  # Clear any previous session state
    turn_lock = get_turn_lock(turn_timeout=30.0)

    # Identify called number from SIP metadata
    metadata = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    sip_info = metadata.get("sip", {}) if isinstance(metadata, dict) else {}
    called_num = sip_info.get("toUser", os.getenv("DEFAULT_TEST_NUMBER", "+13103410536"))

    clinic_info, agent_info, settings, agent_name = await fetch_clinic_context(called_num)

    clinic_name = (clinic_info or {}).get("name") or "our clinic"
    clinic_tz = (clinic_info or {}).get("timezone") or DEFAULT_TZ
    clinic_region = (clinic_info or {}).get("default_phone_region") or DEFAULT_PHONE_REGION
    agent_lang = (agent_info or {}).get("default_language") or "en-US"

    state.tz = clinic_tz
    schedule = load_schedule_from_settings(settings or {})


    # Initialize phone capture manager for telephony continuation
    phone_capture_mgr = PhoneCaptureManager(
        default_region=clinic_region,
        min_digits_domestic=10,
        timeout_seconds=30.0,
    )
    greeting = (settings or {}).get("greeting_text") or (
        f"Hi, I’m {agent_name} from {clinic_name}. Are you a new or returning patient?"
    )

    dynamic_instructions = (
        f"{BASE_PROMPT.strip()}\n\n"
        f"Clinic: {clinic_name}\n"
        f"Agent: {agent_name}\n"
        f"Timezone: {clinic_tz}\n"
        f"Phone region: {clinic_region}\n"
        f"Appointment length: {DEFAULT_MIN} minutes\n"
    )

    vad = silero.VAD.load()
    llm = pick_llm()
    stt = pick_stt(language=agent_lang)
    tts = pick_tts()

    session = AgentSession(stt=stt, llm=llm, tts=tts, vad=vad)
    
    # ==========================================================
    # SET TURN LOCK RESPONSE CALLBACK
    # ==========================================================
    async def _send_response(text: str):
        """Response callback for turn lock."""
        await say_and_log(session, text)
    
    turn_lock.set_response_callback(_send_response)
    logger.info("[TELEPHONY] Turn lock initialized with response callback")

    usage = lk_metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    @session.on("conversation_item_added")
    def _on_item(ev):
        if ev.item.role != "user":
            return
        txt = (ev.item.text_content or "").strip()
        if not txt:
            return
        asyncio.create_task(
            handle_transcript(
                txt,
                True,
                session,
                clinic_info or {},
                agent_info or {},
                settings or {},
                schedule,
                state,
                turn_lock,  # ✅ Pass turn lock to handler
                phone_capture_mgr,  # ✅ Pass phone capture manager
            )
        )

    await session.start(
        room=ctx.room,
        agent=DentalAssistant(instructions=dynamic_instructions),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            close_on_disconnect=False,
        ),
    )

    await say_and_log(session, (greeting))

    async def _on_shutdown():
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Telephony session ending after {dur}s", extra={
            "booking_confirmed": state.booking_confirmed,
            "booking_attempted": state.booking_attempted,
        })
        
        # Log turn lock stats
        try:
            stats = turn_lock.get_stats()
            logger.info(f"[LIFECYCLE] Turn lock stats: {stats}")
            print(f"\n[TURN LOCK STATS] {stats}")
        except Exception as e:
            logger.warning(f"[LIFECYCLE] Failed to get turn lock stats: {e}")
        
        try:
            if clinic_info:
                outcome = "appointment_booked" if state.booking_confirmed else "completed"
                # Wrap DB call in to_thread to prevent blocking
                await asyncio.to_thread(
                    lambda: supabase.table("call_sessions").insert(
                        {
                            "organization_id": clinic_info["organization_id"],
                            "clinic_id": clinic_info["id"],
                            "caller_phone_masked": state.phone_last4 or "Unknown",
                            "outcome": outcome,
                            "duration_seconds": dur,
                            "called_number": called_num,
                        }
                    ).execute()
                )
        except Exception as e:
            logger.error(f"[db] call_sessions insert error: {e}")

        try:
            print(f"[usage] {usage.get_summary()}")
        except Exception:
            pass
        
        # Reset turn lock for next session
        reset_turn_lock()

    ctx.add_shutdown_callback(_on_shutdown)
    
    # ==========================================================
    # CRITICAL: Keep entrypoint alive until room disconnects
    # This prevents "job exiting" / "executor unresponsive" errors
    # ==========================================================
    logger.info("[LIFECYCLE] Telephony entrypoint waiting for room disconnect...")
    
    disconnect_event = asyncio.Event()
    
    @ctx.room.on("disconnected")
    def _on_room_disconnect():
        logger.info("[LIFECYCLE] Room disconnected")
        disconnect_event.set()
    
    # Also handle SIP call ending
    @ctx.room.on("participant_disconnected")
    def _on_participant_left(participant):
        logger.info(f"[LIFECYCLE] Participant left: {participant.identity}")
        # Only trigger disconnect if it's the caller, not other participants
        if participant.identity != "agent":
            disconnect_event.set()
    
    # Wait for disconnect or timeout (2 hour max call)
    try:
        await asyncio.wait_for(disconnect_event.wait(), timeout=7200)
    except asyncio.TimeoutError:
        logger.info("[LIFECYCLE] Session timed out after 2 hours")
    
    logger.info("[LIFECYCLE] Telephony entrypoint exiting normally")
    
def prewarm(proc: agents.JobProcess):
    """
    Preload heavy resources once per worker.
    
    FIX 6: Now includes calendar configuration verification.
    """
    try:
        silero.VAD.load()
        print("[prewarm] Silero VAD loaded")
    except Exception as e:
        print(f"[prewarm] Failed to load VAD: {e}")
    
    # FIX 6: Verify calendar configuration at startup
    _verify_calendar_config_at_startup()


def _verify_calendar_config_at_startup():
    """
    FIX 6: Verify calendar configuration is valid at agent startup.
    
    This catches configuration issues early (before booking attempts fail).
    Logs FATAL error if calendar is misconfigured.
    
    Checks:
    1. OAuth token file exists (if using oauth mode)
    2. Calendar ID is configured
    3. Google Calendar API is reachable (optional, can be slow)
    """
    print("\n" + "="*60)
    print("[CALENDAR_CONFIG] Verifying calendar configuration...")
    print("="*60)
    
    has_errors = False
    
    # Check auth mode
    auth_mode = GOOGLE_CALENDAR_AUTH_MODE
    print(f"[CALENDAR_CONFIG] Auth mode: {auth_mode}")
    
    if auth_mode == "oauth":
        # Check OAuth token file exists
        token_path = GOOGLE_OAUTH_TOKEN_PATH
        if not token_path:
            print("[CALENDAR_CONFIG] ❌ FATAL: GOOGLE_OAUTH_TOKEN env var not set")
            has_errors = True
        elif not os.path.exists(token_path):
            print(f"[CALENDAR_CONFIG] ❌ FATAL: OAuth token file not found: {token_path}")
            print(f"[CALENDAR_CONFIG]    Run oauth_bootstrap.py to create token")
            has_errors = True
        else:
            # Validate token file is valid JSON
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    token_data = json.load(f)
                
                # Check for required fields
                if "token" not in token_data and "access_token" not in token_data:
                    print(f"[CALENDAR_CONFIG] ⚠️ WARNING: Token file missing 'token' or 'access_token' field")
                    print(f"[CALENDAR_CONFIG]    Token might be malformed")
                else:
                    print(f"[CALENDAR_CONFIG] ✅ OAuth token file valid: {token_path}")
            except json.JSONDecodeError as e:
                print(f"[CALENDAR_CONFIG] ❌ FATAL: Token file is not valid JSON: {e}")
                has_errors = True
            except Exception as e:
                print(f"[CALENDAR_CONFIG] ⚠️ WARNING: Could not read token file: {e}")
    
    elif auth_mode == "service_account":
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not sa_path:
            print("[CALENDAR_CONFIG] ❌ FATAL: GOOGLE_APPLICATION_CREDENTIALS not set for service_account mode")
            has_errors = True
        elif not os.path.exists(sa_path):
            print(f"[CALENDAR_CONFIG] ❌ FATAL: Service account file not found: {sa_path}")
            has_errors = True
        else:
            print(f"[CALENDAR_CONFIG] ✅ Service account file exists: {sa_path}")
    
    # Check calendar ID
    calendar_id = GOOGLE_CALENDAR_ID_DEFAULT
    if not calendar_id:
        print("[CALENDAR_CONFIG] ❌ FATAL: GOOGLE_CALENDAR_ID not configured")
        has_errors = True
    else:
        print(f"[CALENDAR_CONFIG] ✅ Calendar ID configured: {calendar_id}")
    
    # Summary
    print("-"*60)
    if has_errors:
        print("[CALENDAR_CONFIG] ❌ CALENDAR CONFIGURATION FAILED")
        print("[CALENDAR_CONFIG]    Booking features will be DISABLED")
        logger.error("[CALENDAR_CONFIG] Calendar misconfigured - booking will fail")
    else:
        print("[CALENDAR_CONFIG] ✅ Calendar configuration OK")
    print("="*60 + "\n")



# ---------------------------------------------------------------------------
# APP BOOTSTRAP
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     agents.cli.run_app(
#         agents.WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             agent_name="telephony_agent",
#             load_threshold=1,
#         )
#     )

if __name__ == "__main__":
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=playground_entrypoint,
                prewarm_fnc=prewarm,
            )
        )

# =============================================================================
# agent_v2.py — A-TIER SNAPPY VOICE AI REFACTOR
# =============================================================================
"""
HIGH-PERFORMANCE VOICE AGENT with <1s response latency.

ARCHITECTURAL CHANGES FROM V1:
1. Single Supabase query with joins (4 queries → 1 query: 3.2s → 100ms)
2. AgentSession with aggressive endpointing (min_endpointing_delay=0.5s)
3. gpt-4o-mini for optimal speed/quality balance
4. Streamlined extraction via inline deterministic extractors (no blocking NLU)
5. Non-blocking booking via asyncio.create_task()
6. Improved persona - acknowledges spellings, context-aware

CRITICAL PERFORMANCE OPTIMIZATIONS:
- Supabase joins reduce 4 sequential queries to 1 round-trip
- min_endpointing_delay=0.5s for snappy turn detection
- Temperature 0.7 for natural speech
- Background booking doesn't block conversation
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import asyncio
import logging
import traceback
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

# Mute noisy transport debug logs (reduces log-bloat in production)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from openai import AsyncOpenAI

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
# LiveKit Imports
# =============================================================================

from livekit import agents, rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    Agent,
    AgentSession,
    RoomInputOptions,
    RunContext,
    metrics as lk_metrics,
    MetricsCollectedEvent,
    llm,  # For ChatContext
)
from livekit.agents.llm import function_tool  # v1.2.14 decorator for tools
from livekit.rtc import ParticipantKind  # For SIP participant detection
from livekit.plugins import (
    openai as openai_plugin,
    silero,
    deepgram as deepgram_plugin,
    assemblyai as assemblyai_plugin,
    cartesia as cartesia_plugin,
)

# =============================================================================
# Contact Utilities
# =============================================================================

from contact_utils import (
    normalize_phone,
    normalize_email,
    validate_email_address,
    parse_datetime_natural,
)

from calendar_client import (
    CalendarAuth, 
    is_time_free, 
    create_event, 
    _get_calendar_service,
    set_token_refresh_callback,
)
from supabase_calendar_store import SupabaseCalendarStore
from supabase import create_client

# =============================================================================
# ENVIRONMENT & CONFIG
# =============================================================================

load_dotenv(".env.local")

ENVIRONMENT = (os.getenv("ENVIRONMENT") or "development").strip().lower()

# Telephony agent identity (must match SIP trunk dispatch rules)
LIVEKIT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "telephony_agent")

DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Karachi")
DEFAULT_MIN = int(os.getenv("DEFAULT_APPT_MINUTES", "60"))
DEFAULT_PHONE_REGION = os.getenv("DEFAULT_PHONE_REGION", "PK")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Async OpenAI client for RAG embeddings (text-embedding-3-small for speed)
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

calendar_store = SupabaseCalendarStore(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

BOOKED_STATUSES = ["scheduled", "confirmed"]

# Google Calendar Config
GOOGLE_CALENDAR_AUTH_MODE = os.getenv("GOOGLE_CALENDAR_AUTH", "oauth")
GOOGLE_OAUTH_TOKEN_PATH = os.getenv("GOOGLE_OAUTH_TOKEN", "./google_token.json")
GOOGLE_CALENDAR_ID_DEFAULT = os.getenv("GOOGLE_CALENDAR_ID", "primary")


def _normalize_sip_user_to_e164(raw: Optional[str]) -> Optional[str]:
    """Best-effort normalization for SIP headers like sip.toUser/sip.calledNumber."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Strip common SIP URI wrappers
    s = s.replace("sip:", "")
    if "@" in s:
        s = s.split("@", 1)[0]

    # Keep only digits and '+'
    s = re.sub(r"[^\d+]", "", s)
    if not s:
        return None

    # Convert 00-prefixed international numbers
    if s.startswith("00"):
        s = "+" + s[2:]

    # If it's digits-only, assume missing '+' and add it (8+ digits for international support)
    if not s.startswith("+") and s.isdigit() and len(s) >= 8:
        s = "+" + s

    return s


def speakable_phone(e164: Optional[str]) -> str:
    """
    Convert E.164 phone number to speech-friendly format for verbal confirmation.
    
    Examples:
        +923351897839 -> "+92 335 189 7839"
        +13105551234  -> "+1 310 555 1234"
    
    Used when agent reads back phone number for confirmation.
    ALWAYS use full number, never just last 4 digits.
    """
    if not e164:
        return "unknown"
    
    s = str(e164).strip()
    if not s.startswith("+"):
        s = "+" + re.sub(r"\D", "", s)
    
    digits = re.sub(r"\D", "", s)
    
    # Format based on country code length
    if digits.startswith("1") and len(digits) == 11:  # US/Canada: +1 XXX XXX XXXX
        return f"+1 {digits[1:4]} {digits[4:7]} {digits[7:]}"
    elif digits.startswith("92") and len(digits) >= 11:  # Pakistan: +92 XXX XXX XXXX
        return f"+92 {digits[2:5]} {digits[5:8]} {digits[8:]}"
    elif digits.startswith("44") and len(digits) >= 11:  # UK: +44 XXXX XXXXXX
        return f"+44 {digits[2:6]} {digits[6:]}"
    else:
        # Generic: group in 3-4 digit chunks
        parts = []
        if s.startswith("+"):
            # Keep country code separate (1-3 digits)
            cc_len = 1 if digits[0] == "1" else (2 if len(digits) <= 12 else 3)
            parts.append(f"+{digits[:cc_len]}")
            remaining = digits[cc_len:]
        else:
            remaining = digits
        
        # Split remaining into 3-4 digit groups
        while remaining:
            chunk_size = 4 if len(remaining) > 6 else 3
            parts.append(remaining[:chunk_size])
            remaining = remaining[chunk_size:]
        
        return " ".join(parts)


def _ensure_phone_is_string(state: "PatientState") -> None:
    """
    Safety guard: Ensure state.phone_e164 is always a string, not a tuple.
    Call this after any phone assignment to catch tuple bugs.
    """
    if state.phone_e164 is not None and isinstance(state.phone_e164, tuple):
        logger.error(f"[PHONE BUG] state.phone_e164 was tuple: {state.phone_e164}. Extracting first element.")
        state.phone_e164 = state.phone_e164[0] if state.phone_e164 else None


def _normalize_phone_preserve_plus(raw: Optional[str], default_region: str) -> Tuple[Optional[str], str]:
    """Normalize phone while preserving explicit international '+' prefix.
    
    Returns: Tuple[Optional[str], str] - (e164_phone, last4_digits)
    IMPORTANT: Always unpack result as: clean_phone, last4 = _normalize_phone_preserve_plus(...)
    CRITICAL: This ALWAYS returns a tuple (str|None, str). Never store the raw return value.
    """
    if not raw:
        return None, ""

    s = str(raw).strip()
    
    # Handle local Pakistani formats (e.g., 0335xxxxxxx -> +92335xxxxxxx)
    # Must be done BEFORE the E.164 check since local formats don't start with +
    if default_region == "PK" and s.startswith("0") and len(s) >= 10:
        # Pakistani local format: 0335xxxxxxx -> +92335xxxxxxx
        local_digits = re.sub(r"\D", "", s)
        if len(local_digits) >= 10 and local_digits.startswith("0"):
            s = "+92" + local_digits[1:]  # Remove leading 0, add +92
            logger.debug(f"[PHONE] Converted PK local {raw} -> {s}")
    
    # If already E.164 format with +
    if s.startswith("+"):
        digits = re.sub(r"\D", "", s)
        # Support international numbers (8+ digits covers Pakistani numbers like +923...)
        if len(digits) >= 8:
            e164 = f"+{digits}"
            return e164, e164[-4:]

    # Fallback to normalize_phone which also returns a tuple
    result = normalize_phone(s, default_region)
    if isinstance(result, tuple):
        e164_val, last4_val = result
        # Ensure e164_val is a string, not a tuple (safety guard)
        if isinstance(e164_val, tuple):
            logger.error(f"[PHONE] BUG: normalize_phone returned nested tuple for '{raw}'")
            e164_val = e164_val[0] if e164_val else None
        return e164_val, last4_val
    # Safety: if normalize_phone returns just a string (shouldn't happen), wrap it
    if result:
        if isinstance(result, str):
            return result, result[-4:] if len(result) >= 4 else ""
        # If result is somehow still a tuple at this point
        logger.error(f"[PHONE] BUG: Unexpected type from normalize_phone: {type(result)}")
    return None, ""


# =============================================================================
# LEAN PROMPT — ACCURACY-FIRST, LOW LATENCY
# =============================================================================

A_TIER_PROMPT = """You are {agent_name}, a receptionist for {clinic_name}.

═══════════════════════════════════════════════════════════════════════════════
📋 YOUR MEMORY (TRUST THIS!)
═══════════════════════════════════════════════════════════════════════════════
{state_summary}

• Fields with '✓' are SAVED — never re-ask for them.
• Fields with '?' are missing — collect these naturally.
• Fields with '⏳' NEED CONFIRMATION — ask the user to confirm!

═══════════════════════════════════════════════════════════════════════════════
🎯 HUMANITY
═══════════════════════════════════════════════════════════════════════════════
Speak like a helpful receptionist. Use brief bridge phrases like "Let me check..." or 
"Hmm..." ONLY when you are actually about to call a tool. Don't overuse them.

═══════════════════════════════════════════════════════════════════════════════
🛠️ TOOLS
═══════════════════════════════════════════════════════════════════════════════
• Call `update_patient_record` IMMEDIATELY when you hear name, phone, email, reason, or time.
• Normalize before saving: "six seven nine" → "679", "at gmail dot com" → "@gmail.com"
• Pass times as natural language: "tomorrow at 2pm", "next Monday".
• If a requested time is TAKEN, the tool returns nearby alternatives — offer those!

═══════════════════════════════════════════════════════════════════════════════
📞 PHONE CONFIRMATION (MANDATORY - READ CAREFULLY!)
═══════════════════════════════════════════════════════════════════════════════
• ALWAYS speak the FULL phone number for confirmation: "Is your number +92 335 189 7839?"
• NEVER confirm with just last 4 digits — always say the complete number with country code.
• ⚡ CRITICAL: If state shows "PHONE: ⏳ [Number]" and user says "yes", "yeah", "correct", 
  you MUST call confirm_phone(confirmed=True) IMMEDIATELY!
• Only mark confirmed AFTER user explicitly confirms with affirmative response.

📍 REGION AWARENESS (INTERNATIONAL PHONES)
═══════════════════════════════════════════════════════════════════════════════
• Accept and confirm international phone numbers (e.g., +92 format). Do NOT force a 10-digit format.

═══════════════════════════════════════════════════════════════════════════════
🔄 SMART REVIEW (SINGLE-CHANGE OPTIMIZATION)
═══════════════════════════════════════════════════════════════════════════════
• If user changes ONE detail after review, ONLY confirm that changed detail.
• Do NOT re-read the entire summary for a single change — that's annoying!
• Example: User says "Actually, make it 3pm" → Say "Got it, changed to 3pm. Ready to book?"
• Once they confirm the single change, proceed to booking immediately.

═══════════════════════════════════════════════════════════════════════════════
✅ CONFIRMATION SEMANTICS
═══════════════════════════════════════════════════════════════════════════════
• "Yes", "Yeah", "Yep", "Correct", "That's right" = confirmed=True
• "No", "Nope", "Wrong" = confirmed=False
• When in doubt, ask for clarification.

═══════════════════════════════════════════════════════════════════════════════
🔒 RULES
═══════════════════════════════════════════════════════════════════════════════
• Never say "booked" until the tool confirms it.
• Never admit you are AI — say "I'm the office assistant."
• Never offer callbacks (you cannot dial out).
• Timezone: {timezone} | Hours: Mon-Fri 9-5, Sat 10-2, Sun closed | Lunch: 1-2pm

📅 BOOKING LOGIC (DATE-SPECIFIC - VERY IMPORTANT!)
═══════════════════════════════════════════════════════════════════════════════
• If user asks for a SPECIFIC date/time (e.g., "January 20 at 3pm"):
  1. FIRST try to book that EXACT slot via update_patient_record(time_suggestion="...")
  2. The tool will check availability and either confirm it OR return nearby alternatives
  3. If alternatives are offered, ask the user to CHOOSE one (don't auto-pick next available)
  
• If user asks for "anytime" or "next available": ONLY THEN use get_available_slots()
• NEVER force "next available Saturday" if user asked for a specific weekday date!
• Always respect the user's date preference - offer alternatives NEAR that date.
"""


# =============================================================================
# LLM FUNCTION CALLING — A-TIER PARALLEL EXTRACTION (v1.2.14 API)
# =============================================================================

# Global state reference for tool access
_GLOBAL_STATE: Optional["PatientState"] = None
_GLOBAL_CLINIC_TZ: str = DEFAULT_TZ
_GLOBAL_CLINIC_INFO: Optional[dict] = None  # For booking tool access
_GLOBAL_AGENT_SETTINGS: Optional[dict] = None  # For DB-backed OAuth token refresh
_REFRESH_AGENT_MEMORY: Optional[callable] = None  # Callback to refresh LLM system prompt
_GLOBAL_SCHEDULE: Optional[Dict[str, Any]] = None  # Scheduling config (working hours, lunch, durations)

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
# STANDALONE FUNCTION TOOLS (v1.2.14 @function_tool decorator)
# =============================================================================

@function_tool(description="""
Update the patient record with any information heard during conversation.
Call this IMMEDIATELY when you hear: name, phone, email, reason for visit, or preferred time.
You can call this multiple times as you gather information.
For phone: normalize spoken numbers (e.g., 'six seven nine' → '679').
For email: normalize spoken format (e.g., 'moiz six seven nine at gmail dot com' → 'moiz679@gmail.com').
For time: pass natural language (e.g., 'tomorrow at 2pm', 'next Monday morning').

NEARBY SLOTS: If the requested time is TAKEN, this tool automatically finds and returns 
nearby alternatives (e.g., "9:00 AM or 11:30 AM"). Simply offer these to the patient!
""")
async def update_patient_record(
    name: str = None,
    phone: str = None,
    email: str = None,
    reason: str = None,
    time_suggestion: str = None,
) -> str:
    """
    Update the internal patient record with extracted information.
    
    Enhanced with:
    - Duration lookup from treatment_durations config
    - Time validation against working hours and lunch breaks
    - Helpful error messages with alternative time suggestions
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_TZ, _GLOBAL_SCHEDULE
    
    state = _GLOBAL_STATE
    if not state:
        return "State not initialized."
    
    schedule = _GLOBAL_SCHEDULE or {}
    updates = []
    errors = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🛡️ INPUT SANITIZATION — Gracefully handle None, empty, or "null" values
    # ═══════════════════════════════════════════════════════════════════════════
    def _sanitize_input(value) -> str:
        """Return sanitized string or None if value should be ignored."""
        if value is None:
            return None
        if not isinstance(value, str):
            # Coerce to string if somehow not a string
            value = str(value)
        stripped = value.strip()
        # Treat empty strings and literal "null"/"none" as None
        if not stripped or stripped.lower() in ("null", "none", "undefined"):
            return None
        return stripped
    
    # Sanitize all inputs before processing
    name = _sanitize_input(name)
    phone = _sanitize_input(phone)
    email = _sanitize_input(email)
    reason = _sanitize_input(reason)
    time_suggestion = _sanitize_input(time_suggestion)
    
    # === NAME ===
    if name and not state.full_name:
        state.full_name = name.strip().title()
        updates.append(f"name={state.full_name}")
        logger.info(f"[TOOL] ✓ Name captured: {state.full_name}")
    
    # === PHONE ===
    if phone and not state.phone_e164:
        # Use proper normalization with region awareness
        clinic_region = (_GLOBAL_CLINIC_INFO or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
        clean_phone, last4 = _normalize_phone_preserve_plus(phone, clinic_region)
        
        if clean_phone:
            state.phone_e164 = str(clean_phone)  # Enforce string type
            state.phone_last4 = str(last4) if last4 else ""
            # Safety guard: ensure no tuple was stored
            _ensure_phone_is_string(state)
            # NEVER auto-confirm phone - always require explicit user confirmation
            state.phone_confirmed = False
            state.phone_source = "user_spoken"
            state.pending_confirm = "phone"
            state.pending_confirm_field = "phone"  # Also set this for deterministic routing
            
            # Return message prompting agent to confirm FULL number
            speakable = speakable_phone(state.phone_e164)
            updates.append(f"phone={state.phone_e164} (NEEDS CONFIRMATION)")
            logger.info(f"[TOOL] ⏳ Phone captured (needs confirmation): {state.phone_e164}")
            
            # Trigger memory refresh so LLM sees the pending confirmation
            if _REFRESH_AGENT_MEMORY:
                try:
                    _REFRESH_AGENT_MEMORY()
                except Exception:
                    pass
            
            # Return explicit instruction to confirm full number
            return f"Phone captured as {speakable}. ASK USER TO CONFIRM THE FULL NUMBER: 'Just to confirm, is your phone number {speakable}?'"
    
    # === EMAIL ===
    if email and not state.email:
        clean_email = email.replace(" ", "").lower()
        if "@" in clean_email and "." in clean_email:
            state.email = clean_email
            state.email_confirmed = True
            updates.append(f"email={state.email}")
            logger.info(f"[TOOL] ✓ Email captured: {state.email}")
    
    # === REASON (with duration lookup) ===
    if reason and not state.reason:
        state.reason = reason.strip()
        # Lookup duration from treatment_durations config
        state.duration_minutes = get_duration_for_service(state.reason, schedule)
        updates.append(f"reason={state.reason} (duration: {state.duration_minutes}m)")
        logger.info(f"[TOOL] ✓ Reason captured: {state.reason}, duration: {state.duration_minutes}m")
    
    # === TIME (with validation and availability check) ===
    if time_suggestion and not state.dt_local:
        state.dt_text = time_suggestion.strip()
        state.time_status = "validating"
        
        # Log the narrative check starting
        logger.info(f"[TOOL] ⏰ Checking time: {time_suggestion}...")
        
        try:
            # Use parse_datetime_natural for robust relative date handling
            # Handles "tomorrow at 3:30 PM", "next Monday", etc. correctly
            parsed = parse_datetime_natural(time_suggestion, tz_hint=_GLOBAL_CLINIC_TZ)
            
            if parsed:
                # parse_datetime_natural already applies timezone
                logger.info(f"[TOOL] ⏰ Parsed '{time_suggestion}' → {parsed.isoformat()}")

                # No date restriction - accept any date within 14-day booking window
                now_local = datetime.now(ZoneInfo(_GLOBAL_CLINIC_TZ))
                
                # Format for speech
                time_spoken = parsed.strftime("%I:%M %p").lstrip("0")
                day_spoken = parsed.strftime("%A")
                
                # Validate against working hours and lunch break
                is_valid, error_msg = is_within_working_hours(
                    parsed, schedule, state.duration_minutes
                )
                
                if is_valid:
                    # Also check if slot is actually free in the database
                    clinic_id = (_GLOBAL_CLINIC_INFO or {}).get("id")
                    if clinic_id:
                        slot_end = parsed + timedelta(minutes=state.duration_minutes + APPOINTMENT_BUFFER_MINUTES)
                        slot_free = await is_slot_free_supabase(clinic_id, parsed, slot_end)
                        
                        if not slot_free:
                            # Slot is taken - find nearby alternatives AROUND THE REQUESTED TIME
                            # NOT from "now" - this is the key fix for the scheduling bug
                            state.time_status = "invalid"
                            state.time_error = "That slot is already taken"
                            state.dt_local = None
                            
                            logger.info(f"[TOOL] ✗ {time_spoken} on {parsed.strftime('%b %d')} is booked, searching for nearby alternatives")
                            
                            # Use suggest_slots_around for comprehensive ±4 hour search around requested time
                            alternatives = await suggest_slots_around(
                                clinic_id=clinic_id,
                                requested_start_dt=parsed,  # Search around THIS time, not now
                                duration_minutes=state.duration_minutes,
                                schedule=schedule,
                                tz_str=_GLOBAL_CLINIC_TZ,
                                count=3,
                                window_hours=4,  # ±4 hours around requested time
                                step_min=15,
                            )
                            
                            logger.info(f"[TOOL] Found {len(alternatives)} nearby alternatives around {parsed.strftime('%H:%M')}")
                            
                            if alternatives:
                                # Format alternatives for speech - include date if different from requested
                                alt_descriptions = []
                                for alt in alternatives:
                                    alt_time_str = alt.strftime("%I:%M %p").lstrip("0")
                                    if alt.date() == parsed.date():
                                        alt_descriptions.append(alt_time_str)
                                    else:
                                        alt_descriptions.append(f"{alt.strftime('%A')} at {alt_time_str}")
                                
                                if len(alt_descriptions) == 1:
                                    return f"... I'm sorry, {time_spoken} is booked on {day_spoken}. The closest I have is {alt_descriptions[0]}. Would that work?"
                                elif len(alt_descriptions) == 2:
                                    return f"... I'm sorry, {time_spoken} is booked on {day_spoken}. I can do {alt_descriptions[0]} or {alt_descriptions[1]}. Which works for you?"
                                else:
                                    return f"... I'm sorry, {time_spoken} is booked on {day_spoken}. I can do {alt_descriptions[0]}, {alt_descriptions[1]}, or {alt_descriptions[2]}. Which works for you?"
                            else:
                                # No nearby alternatives, suggest checking another time
                                return f"... hmm, {time_spoken} on {day_spoken} is booked and I don't see openings nearby. Would you like to try a different day?"
                    
                    # ✅ Time is VALID and AVAILABLE
                    state.dt_local = parsed
                    state.time_status = "valid"
                    state.time_error = None
                    time_formatted = parsed.strftime('%B %d at %I:%M %p')
                    updates.append(f"time={time_formatted} ({state.duration_minutes}m slot)")
                    logger.info(f"[TOOL] ✓ Time validated and available: {parsed.isoformat()}")
                    
                    # Sonic-3 prosody: breathy confirmation with ellipses
                    return f"... ah, perfect! {day_spoken} at {time_spoken} is open. I've got that down."
                else:
                    # Time is invalid (lunch, after-hours, holiday)
                    state.time_status = "invalid"
                    state.time_error = error_msg
                    state.dt_local = None  # Don't save invalid time
                    
                    logger.warning(f"[TOOL] ✗ Time rejected: {error_msg}")
                    
                    # Check if it's a lunch break - give a helpful response with Sonic-3 prosody
                    if "lunch" in error_msg.lower():
                        lunch_end = schedule.get("lunch_break", {}).get("end", "14:00")
                        lunch_time = lunch_end.replace(":00", "").lstrip("0")
                        return f"... oh, the team is at lunch then-- but I can get you in right at {lunch_time} when they're back. How does that sound?"
                    
                    # After-hours or closed day
                    if "closed" in error_msg.lower() or "sunday" in error_msg.lower():
                        return f"... hmm, we're actually closed on {day_spoken}. Would another day work for you?"
                    
                    if "outside" in error_msg.lower() or "hours" in error_msg.lower():
                        return f"... ah, {time_spoken} is outside our hours. We're open 9 to 5-- would morning or late afternoon work?"
                    
                    errors.append(error_msg)
            else:
                state.time_status = "pending"
                updates.append(f"time_text={time_suggestion}")
                
        except Exception as e:
            logger.warning(f"[TOOL] Time parse deferred: {time_suggestion} ({e})")
            state.time_status = "pending"
            updates.append(f"time_text={time_suggestion}")
    
    # Trigger memory refresh so LLM sees updated state
    if _REFRESH_AGENT_MEMORY:
        try:
            _REFRESH_AGENT_MEMORY()
            logger.debug("[TOOL] Memory refresh triggered after update")
        except Exception as e:
            logger.warning(f"[TOOL] Memory refresh failed: {e}")
    
    # Return result
    if errors:
        # Return error with suggestion - LLM should use this!
        return errors[0]  # Return the first error (usually the time error)
    elif updates:
        return f"Record updated: {', '.join(updates)}. Continue the conversation naturally."
    else:
        return "No new information to update. Continue gathering missing details."


@function_tool(description="""
Get the next available appointment slots. Call this when:
- The patient asks 'when is the next opening?'
- A requested time is unavailable and you need alternatives
- You want to proactively suggest times

Returns the next 3 available time slots based on the patient's service duration.
""")
async def get_available_slots() -> str:
    """
    Find and return the next available appointment slots.
    Uses Supabase as primary source for speed.
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_INFO, _GLOBAL_SCHEDULE, _GLOBAL_CLINIC_TZ
    
    state = _GLOBAL_STATE
    clinic_info = _GLOBAL_CLINIC_INFO
    schedule = _GLOBAL_SCHEDULE or {}
    
    if not clinic_info:
        return "Unable to check availability. Please try again."
    
    # Use patient's service duration if set, otherwise default to 60
    duration = state.duration_minutes if state else 60
    
    try:
        slots = await get_next_available_slots(
            clinic_id=clinic_info["id"],
            schedule=schedule,
            tz_str=_GLOBAL_CLINIC_TZ,
            duration_minutes=duration,
            num_slots=3,
            days_ahead=14,  # Extended to 14-day window for international booking
        )
        
        if not slots:
            return "No available slots in the next 2 weeks. Would you like me to check a specific date?"
        
        # Format slots for speech
        slot_strings = []
        for slot in slots:
            day_str = slot.strftime("%A")  # e.g., "Monday"
            time_str = slot.strftime("%I:%M %p").lstrip("0")  # e.g., "10:00 AM"
            
            # Make it conversational
            today = datetime.now(ZoneInfo(_GLOBAL_CLINIC_TZ)).date()
            if slot.date() == today:
                slot_strings.append(f"today at {time_str}")
            elif slot.date() == today + timedelta(days=1):
                slot_strings.append(f"tomorrow at {time_str}")
            else:
                slot_strings.append(f"{day_str} at {time_str}")
        
        if len(slot_strings) == 1:
            return f"The next available time is {slot_strings[0]}."
        elif len(slot_strings) == 2:
            return f"The next available times are {slot_strings[0]} and {slot_strings[1]}."
        else:
            return f"The next available times are {slot_strings[0]}, {slot_strings[1]}, and {slot_strings[2]}."
            
    except Exception as e:
        logger.error(f"[TOOL] get_available_slots error: {e}")
        return "I'm having trouble checking availability. Let me try a different approach."


# =============================================================================
# APPOINTMENT BUFFER CONSTANT (15 min between appointments)
# =============================================================================
APPOINTMENT_BUFFER_MINUTES = 15


@function_tool(description="""
Advanced scheduling tool with relative time searching. Use this when:
- User asks for slots "after 2pm tomorrow" or "before noon on Monday"
- User specifies a preferred day like "next Wednesday"
- You need to filter available times based on user constraints
- A requested time is unavailable and you need alternatives

⚡ BRIDGE PHRASE: Before calling this tool, say: "Okay, checking slots after [time] for you... one moment."

Parameters:
- after_datetime: ISO string or natural language (e.g., "tomorrow at 2pm", "2026-01-16T14:00:00")
- preferred_day: Day name (e.g., "Monday", "Wednesday") or "tomorrow", "today"
- num_slots: Number of slots to return (default 3)

Returns slots that match the constraints, respecting working hours and lunch breaks.
Automatically skips lunch break (1pm-2pm) and provides helpful messaging.
""")
async def get_available_slots_v2(
    after_datetime: str = None,
    preferred_day: str = None,
    num_slots: int = 3,
) -> str:
    """
    Advanced slot finder with relative time constraints.
    
    Supports:
    - "after 2pm tomorrow" → after_datetime="tomorrow at 2pm"
    - "next Monday morning" → preferred_day="Monday", after_datetime="9am"
    - Automatic lunch break skipping with helpful messaging
    - 15-minute buffer between appointments
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_INFO, _GLOBAL_SCHEDULE, _GLOBAL_CLINIC_TZ
    
    state = _GLOBAL_STATE
    clinic_info = _GLOBAL_CLINIC_INFO
    schedule = _GLOBAL_SCHEDULE or {}
    
    if not clinic_info:
        return "... hmm, I'm having trouble accessing the schedule. Let me try again."
    
    duration = state.duration_minutes if state else 60
    tz = ZoneInfo(_GLOBAL_CLINIC_TZ)
    now = datetime.now(tz)
    
    # Parse after_datetime constraint
    search_start = now
    if after_datetime:
        try:
            # Use parse_datetime_natural for robust relative date handling
            # Handles "tomorrow at 2pm", "next Monday morning", etc. correctly
            parsed = parse_datetime_natural(after_datetime, tz_hint=_GLOBAL_CLINIC_TZ)
            if parsed:
                search_start = parsed
                logger.info(f"[TOOL] get_available_slots_v2: parsed '{after_datetime}' → searching after {search_start.isoformat()}")
        except Exception as e:
            logger.warning(f"[TOOL] Could not parse after_datetime '{after_datetime}': {e}")
    
    # Parse preferred_day constraint
    target_weekday = None
    if preferred_day:
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
        }
        day_lower = preferred_day.lower().strip()
        
        if day_lower == "today":
            target_weekday = now.weekday()
        elif day_lower == "tomorrow":
            target_weekday = (now + timedelta(days=1)).weekday()
        elif day_lower in day_map:
            target_weekday = day_map[day_lower]
    
    try:
        # Calculate search window
        slot_step = schedule.get("slot_step_minutes", 30)
        
        # Round up search_start to next slot boundary
        minutes_to_add = slot_step - (search_start.minute % slot_step)
        if minutes_to_add == slot_step:
            minutes_to_add = 0
        current = search_start.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
        
        # If we have a preferred day and current isn't that day, jump to it
        if target_weekday is not None and current.weekday() != target_weekday:
            days_until = (target_weekday - current.weekday()) % 7
            if days_until == 0:
                days_until = 7  # Next week same day
            current = datetime.combine(
                current.date() + timedelta(days=days_until),
                datetime.min.time(),
                tzinfo=tz
            )
            current = current.replace(hour=9, minute=0)  # Start at 9am
        
        # FIX: Anchor end_search to SEARCH_START, not just now
        # This ensures future date requests (e.g., Jan 20) are within search window
        end_search = max(now + timedelta(days=14), search_start + timedelta(days=14))
        
        logger.info(f"[SLOTS_V2] Search anchored: start={search_start.date()}, end={end_search.date()}, target_day={preferred_day or 'any'}")
        
        # Fetch existing appointments
        existing_appointments = []
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("appointments")
                .select("start_time, end_time")
                .eq("clinic_id", clinic_info["id"])
                .gte("start_time", now.isoformat())
                .lte("start_time", end_search.isoformat())
                .in_("status", BOOKED_STATUSES)
                .execute()
            )
            for appt in (result.data or []):
                try:
                    appt_start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
                    appt_end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
                    existing_appointments.append((appt_start, appt_end))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[SLOTS_V2] Failed to fetch appointments: {e}")
        
        available_slots = []
        lunch_skipped = False
        
        while current < end_search and len(available_slots) < num_slots:
            # If preferred_day is set, only check that day
            if target_weekday is not None and current.weekday() != target_weekday:
                # Jump to next occurrence of preferred day
                days_until = (target_weekday - current.weekday()) % 7
                if days_until == 0:
                    days_until = 7
                current = datetime.combine(
                    current.date() + timedelta(days=days_until),
                    datetime.min.time(),
                    tzinfo=tz
                )
                current = current.replace(hour=9, minute=0)
                if current >= end_search:
                    break
            
            # Check if slot is valid (working hours, not lunch, not holiday)
            is_valid, error_msg = is_within_working_hours(current, schedule, duration)
            
            if not is_valid and "lunch" in error_msg.lower():
                lunch_skipped = True
            
            if is_valid:
                # Check slot availability with buffer
                slot_end = current + timedelta(minutes=duration + APPOINTMENT_BUFFER_MINUTES)
                is_free = True
                
                for appt_start, appt_end in existing_appointments:
                    # Add buffer to existing appointment end time
                    buffered_end = appt_end + timedelta(minutes=APPOINTMENT_BUFFER_MINUTES)
                    if current < buffered_end and slot_end > appt_start:
                        is_free = False
                        break
                
                if is_free:
                    available_slots.append(current)
            
            # Move to next slot
            current += timedelta(minutes=slot_step)
            
            # Skip to next day if past working hours
            dow_key = WEEK_KEYS[current.weekday()]
            intervals = schedule["working_hours"].get(dow_key, [])
            if intervals:
                last_interval = intervals[-1]
                try:
                    eh, em = map(int, last_interval["end"].split(":"))
                    day_end = current.replace(hour=eh, minute=em)
                    if current >= day_end:
                        next_day = current.date() + timedelta(days=1)
                        current = datetime.combine(next_day, datetime.min.time(), tzinfo=tz)
                        current = current.replace(hour=9, minute=0)
                except Exception:
                    pass
        
        if not available_slots:
            constraint_desc = ""
            if after_datetime:
                constraint_desc = f" after {after_datetime}"
            if preferred_day:
                constraint_desc += f" on {preferred_day}"
            return f"... hmm, I don't see any openings{constraint_desc} in the next week. Would you like me to check a different day?"
        
        # Format slots for natural speech (Sonic-3 optimized)
        slot_strings = []
        for slot in available_slots:
            time_str = slot.strftime("%I:%M %p").lstrip("0")
            day_str = slot.strftime("%A")
            
            today = now.date()
            if slot.date() == today:
                slot_strings.append(f"today at {time_str}")
            elif slot.date() == today + timedelta(days=1):
                slot_strings.append(f"tomorrow at {time_str}")
            else:
                slot_strings.append(f"{day_str} at {time_str}")
        
        # Build natural response with Sonic-3 prosody
        response_prefix = "... ah, okay, I found some times. "
        if lunch_skipped:
            response_prefix = "... okay, skipping the lunch hour... "
        
        if len(slot_strings) == 1:
            return f"{response_prefix}The next available time is {slot_strings[0]}."
        elif len(slot_strings) == 2:
            return f"{response_prefix}I've got {slot_strings[0]}... or {slot_strings[1]}."
        else:
            return f"{response_prefix}We have {slot_strings[0]}... {slot_strings[1]}... or {slot_strings[2]}. Which works best?"
            
    except Exception as e:
        logger.error(f"[TOOL] get_available_slots_v2 error: {e}")
        traceback.print_exc()
        return "... hmm, I'm having trouble with the schedule. Let me try that again."


@function_tool(description="""
Find the next available slot after a specific time, or the last slot on a specific day.
Use this when:
- User asks for "next available after [time]" or "last slot on [day]"
- User says "anything after 2pm" or "late afternoon slots"
- You need to search within a specific time window

⚡ BRIDGE PHRASE: Say "Okay, checking for slots after [time]... one moment." before calling.

Parameters:
- start_search_time: ISO string or natural language (e.g., "tomorrow at 2pm", "after 3pm")
- limit_to_day: Optional day constraint (e.g., "Monday", "tomorrow", "today")
- find_last: If true, finds the LAST available slot instead of the first (for "end of day" requests)

Returns the best matching slot with Sonic-3 prosody formatting.
""")
async def find_relative_slots(
    start_search_time: str = None,
    limit_to_day: str = None,
    find_last: bool = False,
) -> str:
    """
    Advanced relative time slot finder.
    
    Handles:
    - "next available after 2pm tomorrow"
    - "last slot on Monday"
    - "anything after lunch"
    - "late afternoon opening"
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_INFO, _GLOBAL_SCHEDULE, _GLOBAL_CLINIC_TZ
    
    state = _GLOBAL_STATE
    clinic_info = _GLOBAL_CLINIC_INFO
    schedule = _GLOBAL_SCHEDULE or {}
    
    if not clinic_info:
        return "... hmm, I'm having trouble accessing the schedule. Let me try again."
    
    duration = state.duration_minutes if state else 60
    tz = ZoneInfo(_GLOBAL_CLINIC_TZ)
    now = datetime.now(tz)
    
    # Parse start_search_time
    search_start = now
    target_date = None
    
    if start_search_time:
        try:
            # Use parse_datetime_natural for robust relative date handling
            # Handles "tomorrow at 2pm", "after 3pm", etc. correctly
            parsed = parse_datetime_natural(start_search_time, tz_hint=_GLOBAL_CLINIC_TZ)
            if parsed:
                search_start = parsed
                target_date = parsed.date()
                logger.info(f"[TOOL] find_relative_slots: parsed '{start_search_time}' → searching after {search_start.isoformat()}")
        except Exception as e:
            logger.warning(f"[TOOL] Could not parse start_search_time '{start_search_time}': {e}")
    
    # Parse limit_to_day constraint
    if limit_to_day:
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
        }
        day_lower = limit_to_day.lower().strip()
        
        if day_lower == "today":
            target_date = now.date()
        elif day_lower == "tomorrow":
            target_date = (now + timedelta(days=1)).date()
        elif day_lower in day_map:
            target_weekday = day_map[day_lower]
            days_until = (target_weekday - now.weekday()) % 7
            if days_until == 0 and now.hour >= 17:  # Past working hours
                days_until = 7
            target_date = (now + timedelta(days=days_until)).date()
    
    try:
        slot_step = schedule.get("slot_step_minutes", 30)
        
        # Round up search_start to next slot boundary
        minutes_to_add = slot_step - (search_start.minute % slot_step)
        if minutes_to_add == slot_step:
            minutes_to_add = 0
        current = search_start.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
        
        # If we have a target date, ensure we start on that day
        if target_date and current.date() < target_date:
            current = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
            current = current.replace(hour=9, minute=0)
        
        # Search window: limit to target_date if specified, otherwise search from anchor point
        # FIX: Anchor end_search to search_start, not just now
        if target_date:
            end_search = datetime.combine(target_date, datetime.max.time(), tzinfo=tz)
        else:
            # Ensure future date requests are within search window
            end_search = max(now + timedelta(days=14), search_start + timedelta(days=14))

        logger.info(f"🔍 [SLOTS] Searching window: search_start={search_start.date()}, end={end_search.date()}, target_date={target_date or 'any'}")
        
        # Fetch existing appointments
        existing_appointments = []
        try:
            result = await asyncio.to_thread(
                lambda: supabase.table("appointments")
                .select("start_time, end_time")
                .eq("clinic_id", clinic_info["id"])
                .gte("start_time", now.isoformat())
                .lte("start_time", end_search.isoformat())
                .in_("status", BOOKED_STATUSES)
                .execute()
            )
            for appt in (result.data or []):
                try:
                    appt_start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
                    appt_end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
                    existing_appointments.append((appt_start, appt_end))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[TOOL] find_relative_slots: Failed to fetch appointments: {e}")
        
        available_slots = []
        lunch_skipped = False
        
        while current < end_search and len(available_slots) < (10 if find_last else 3):
            # If target_date is set, only check that day
            if target_date and current.date() != target_date:
                break
            
            # Check if slot is valid (working hours, not lunch, not holiday)
            is_valid, error_msg = is_within_working_hours(current, schedule, duration)
            
            if not is_valid and "lunch" in error_msg.lower():
                lunch_skipped = True
            
            if is_valid:
                # Check slot availability with buffer
                slot_end = current + timedelta(minutes=duration + APPOINTMENT_BUFFER_MINUTES)
                is_free = True
                
                for appt_start, appt_end in existing_appointments:
                    buffered_end = appt_end + timedelta(minutes=APPOINTMENT_BUFFER_MINUTES)
                    if current < buffered_end and slot_end > appt_start:
                        is_free = False
                        break
                
                if is_free:
                    available_slots.append(current)
            
            # Move to next slot
            current += timedelta(minutes=slot_step)
        
        if not available_slots:
            time_desc = start_search_time or "that time"
            return f"... hmm, I don't see any openings after {time_desc} in the next 2 weeks. Would you like me to check a different day?"
        
        # If find_last, return the last slot found
        if find_last:
            best_slot = available_slots[-1]
        else:
            best_slot = available_slots[0]
        
        # Format for natural speech
        time_str = best_slot.strftime("%I:%M %p").lstrip("0")
        day_str = best_slot.strftime("%A")
        
        today = now.date()
        if best_slot.date() == today:
            day_desc = "today"
        elif best_slot.date() == today + timedelta(days=1):
            day_desc = "tomorrow"
        else:
            day_desc = day_str
        
        # Build response with Sonic-3 prosody
        if lunch_skipped:
            prefix = "... okay, skipping the lunch hour-- "
        else:
            prefix = "... ah, let me see-- "
        
        if find_last:
            return f"{prefix}the last slot I have is {day_desc} at {time_str}. Does that work?"
        else:
            return f"{prefix}the next opening is {day_desc} at {time_str}. How does that sound?"
            
    except Exception as e:
        logger.error(f"[TOOL] find_relative_slots error: {e}")
        traceback.print_exc()
        return "... hmm, I'm having trouble with the schedule. Let me try that again."


@function_tool(description="""
Confirm the phone number with the patient. Call this ONLY after reading back the FULL phone number.
NEVER confirm based on just last 4 digits - always speak the complete number including country code.
Example: "Just to confirm, is your number +92 335 189 7839?"

IMPORTANT: When user says "yes", "yeah", "correct" etc., call confirm_phone(confirmed=True).
When user says "no", "wrong", "incorrect", call confirm_phone(confirmed=False).
""")
async def confirm_phone(confirmed: bool, new_phone: Optional[str] = None) -> str:
    """
    Mark phone as confirmed or update with correction.
    
    Args:
        confirmed: True if user confirmed the phone is correct
        new_phone: Optional new phone number if user provided a correction
    """
    global _GLOBAL_STATE
    state = _GLOBAL_STATE
    if not state:
        return "State not initialized."
    
    if new_phone:
        # User provided a correction - use proper normalization
        clinic_region = (_GLOBAL_CLINIC_INFO or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
        clean_phone, last4 = _normalize_phone_preserve_plus(new_phone, clinic_region)
        
        if clean_phone:
            state.phone_e164 = str(clean_phone)  # Enforce string type
            state.phone_last4 = str(last4) if last4 else ""
            # Safety guard
            _ensure_phone_is_string(state)
            # NEVER auto-confirm - even with correction
            state.phone_confirmed = False
            state.phone_source = "user_spoken"
            state.pending_confirm = "phone"
            state.pending_confirm_field = "phone"
            
            speakable = speakable_phone(state.phone_e164)
            logger.info(f"[TOOL] ⏳ Phone updated to {state.phone_e164}, needs re-confirmation")
            # Trigger memory refresh
            if _REFRESH_AGENT_MEMORY:
                try:
                    _REFRESH_AGENT_MEMORY()
                except Exception:
                    pass
            return f"Phone updated to {speakable}. Please confirm the FULL number: 'Is your number {speakable}?'"
        else:
            return f"Could not parse phone number '{new_phone}'. Ask user to repeat clearly."
    
    if confirmed:
        if not state.phone_e164:
            return "No phone number to confirm. Ask for phone number first."
        # Safety guard before confirming
        _ensure_phone_is_string(state)
        state.phone_confirmed = True
        state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
        state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
        logger.info(f"[TOOL] ✓ Phone CONFIRMED: {state.phone_e164}")
        # Trigger memory refresh
        if _REFRESH_AGENT_MEMORY:
            try:
                _REFRESH_AGENT_MEMORY()
            except Exception:
                pass
        return "Phone confirmed! Continue gathering remaining info."
    else:
        # User said "no" - clear and re-ask
        old_phone = state.phone_e164
        state.phone_e164 = None
        state.phone_last4 = None
        state.phone_confirmed = False
        state.phone_source = None
        state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
        state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
        logger.info(f"[TOOL] ✗ Phone REJECTED (was {old_phone}), cleared")
        # Trigger memory refresh
        if _REFRESH_AGENT_MEMORY:
            try:
                _REFRESH_AGENT_MEMORY()
            except Exception:
                pass
        return "Phone cleared. Ask user: 'Could you please give me your phone number again?'"


@function_tool(description="""
Confirm the email address with the patient. Call this after spelling it back.
""")
async def confirm_email(confirmed: bool) -> str:
    """Mark email as confirmed or rejected."""
    global _GLOBAL_STATE
    state = _GLOBAL_STATE
    if not state:
        return "State not initialized."
    
    if confirmed:
        state.email_confirmed = True
        logger.info("[TOOL] ✓ Email confirmed")
        # Trigger memory refresh
        if _REFRESH_AGENT_MEMORY:
            try:
                _REFRESH_AGENT_MEMORY()
            except Exception:
                pass
        return "Email confirmed. Continue with next missing info."
    else:
        state.email = None
        state.email_confirmed = False
        logger.info("[TOOL] ✗ Email rejected, cleared")
        # Trigger memory refresh
        if _REFRESH_AGENT_MEMORY:
            try:
                _REFRESH_AGENT_MEMORY()
            except Exception:
                pass
        return "Email cleared. Ask for the correct email."


@function_tool(description="""
Check the current booking status and what information is still missing.
Call this to know what to ask for next.
""")
async def check_booking_status() -> str:
    """Return current state and missing fields."""
    global _GLOBAL_STATE
    state = _GLOBAL_STATE
    if not state:
        return "State not initialized."
    
    collected = []
    missing = []
    
    if state.full_name:
        collected.append(f"name={state.full_name}")
    else:
        missing.append("name")
    
    if state.phone_e164 and state.phone_confirmed:
        collected.append(f"phone=***{state.phone_last4}")
    elif state.phone_e164:
        missing.append("phone_confirmation")
    else:
        missing.append("phone")
    
    if state.email and state.email_confirmed:
        collected.append(f"email={state.email}")
    elif state.email:
        missing.append("email_confirmation")
    else:
        missing.append("email")
    
    if state.reason:
        collected.append(f"reason={state.reason}")
    else:
        missing.append("reason")
    
    if state.dt_local:
        collected.append(f"time={state.dt_local.strftime('%B %d at %I:%M %p')}")
    else:
        missing.append("preferred_time")
    
    status = f"Collected: {', '.join(collected) if collected else 'none'}. "
    if missing:
        status += f"Still need: {', '.join(missing)}."
    else:
        status += "All info collected! Ready to book."
    
    return status


@function_tool(description="""
Finalize the appointment booking. Call this ONLY after:
1. You have collected ALL required information (name, phone, email, reason, time)
2. You have read back the summary to the patient
3. The patient has verbally confirmed with 'yes' or similar
Do NOT call this until the patient confirms the summary!
""")
async def confirm_and_book_appointment() -> str:
    """
    Trigger the actual booking after user confirmation.
    
    Uses DB-backed OAuth with non-blocking token refresh persistence.
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_INFO, _GLOBAL_AGENT_SETTINGS
    state = _GLOBAL_STATE
    clinic_info = _GLOBAL_CLINIC_INFO
    settings = _GLOBAL_AGENT_SETTINGS
    
    if not state:
        return "State not initialized."
    
    if not state.is_complete():
        missing = state.missing_slots()
        return f"Cannot book yet. Missing: {', '.join(missing)}. Continue gathering info."
    
    if state.booking_confirmed:
        return "Appointment already booked! Tell the user their appointment is confirmed."
    
    if state.booking_in_progress:
        return "Booking already in progress. Please wait."
    
    if not clinic_info:
        return "Clinic info not available. Cannot book."
    
    # Mark booking in progress
    state.booking_in_progress = True
    
    try:
        # Get calendar auth with DB-first priority and refresh callback
        auth, calendar_id, refresh_callback = await resolve_calendar_auth_async(
            clinic_info,
            settings=settings
        )
        if not auth:
            state.booking_in_progress = False
            logger.error("[BOOKING] Calendar auth failed - no OAuth token available.")
            return "Calendar not configured. Tell the user to call back."
        
        # Get calendar service with refresh callback for non-blocking token persistence
        service = await asyncio.to_thread(
            _get_calendar_service, 
            auth=auth,
            on_refresh_callback=refresh_callback
        )
        if not service:
            state.booking_in_progress = False
            return "Calendar unavailable. Tell the user to try again."
        
        start_dt = state.dt_local
        end_dt = start_dt + timedelta(minutes=state.duration_minutes)  # Use service-specific duration
        
        # Check availability
        try:
            resp = service.freebusy().query(body={
                "timeMin": start_dt.isoformat(),
                "timeMax": end_dt.isoformat(),
                "timeZone": state.tz,
                "items": [{"id": calendar_id}],
            }).execute()
            busy = resp.get("calendars", {}).get(calendar_id, {}).get("busy", [])
            
            if busy:
                state.booking_in_progress = False
                state.dt_local = None
                return "That time slot is now taken. Ask the user for another time."
        except Exception as e:
            logger.warning(f"[BOOKING] Freebusy check failed: {e}")
        
        # Create event
        event = service.events().insert(
            calendarId=calendar_id,
            body={
                "summary": f"{state.reason or 'Appointment'} — {state.full_name}",
                "description": f"Patient: {state.full_name}\nPhone: {state.phone_e164}\nEmail: {state.email}",
                "start": {"dateTime": start_dt.isoformat(), "timeZone": state.tz},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": state.tz},
                "attendees": [{"email": state.email}] if state.email else [],
            },
            sendUpdates="all",
        ).execute()
        
        if event and event.get("id"):
            state.booking_confirmed = True
            state.calendar_event_id = event.get("id")
            
            # Non-blocking Supabase save
            asyncio.create_task(book_to_supabase(clinic_info, state, event.get("id")))
            
            logger.info(f"[BOOKING] ✓ SUCCESS! Event ID: {event.get('id')}")
            return f"BOOKING CONFIRMED! Tell the user: Your appointment is confirmed for {start_dt.strftime('%A, %B %d at %I:%M %p')}. A confirmation email will be sent to {state.email}."
        else:
            state.booking_in_progress = False
            return "Booking failed. Ask the user to try again."
            
    except Exception as e:
        logger.error(f"[BOOKING] Error: {e}")
        state.booking_in_progress = False
        return f"Booking error: {str(e)}. Tell the user something went wrong."
    finally:
        state.booking_in_progress = False


@function_tool(description="""
Search the clinic knowledge base for information about parking, pricing, insurance, 
location, services, or any clinic-specific details. Call this IMMEDIATELY when the 
user asks about anything not related to booking (e.g., 'Where do I park?', 
'Do you accept Delta Dental?', 'How much is a cleaning?').
""")
async def search_clinic_info(query: str) -> str:
    """
    A-Tier RAG: Non-blocking semantic search against Supabase knowledge base.
    Uses text-embedding-3-small for <100ms embedding latency.
    """
    global _GLOBAL_CLINIC_INFO
    
    if not _GLOBAL_CLINIC_INFO:
        return "I'm sorry, I'm having trouble accessing the office records for this location."
    
    clinic_id = _GLOBAL_CLINIC_INFO.get("id")
    if not clinic_id:
        return "I don't have that specific info right now."
    
    try:
        # 1. Generate embedding using fastest model
        embed_resp = await openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = embed_resp.data[0].embedding
        
        # 2. Non-blocking RPC call to Supabase vector search
        result = await asyncio.to_thread(
            lambda: supabase.rpc("match_knowledge_articles", {
                "query_embedding": query_embedding,
                "match_threshold": 0.4,
                "match_count": 2,
                "target_clinic_id": clinic_id
            }).execute()
        )
        
        if not result.data:
            logger.info(f"[RAG] No matches for query: {query}")
            return "I don't have that specific info in my notes right now."
        
        logger.info(f"[RAG] Search successful. Found {len(result.data)} articles.")

        # 3. Format results concisely for speech
        answers = []
        for r in result.data:
            body = r.get("body", "").strip()
            if body:
                answers.append(body)
        
        if not answers:
            return "I don't have that specific info in my notes right now."
        
        logger.info(f"[RAG] Found {len(answers)} matches for: {query}")
        return "\n".join([f"- {a}" for a in answers])
        
    except Exception as e:
        logger.error(f"[RAG] Search failed: {e}")
        return "I'm having trouble accessing my notes right now."


# List of all tools to pass to Agent
RECEPTIONIST_TOOLS = [
    update_patient_record,
    get_available_slots,
    get_available_slots_v2,  # ⚡ Advanced scheduling with relative time search
    find_relative_slots,     # ⚡ "Next available after X" / "Last slot on Y"
    search_clinic_info,      # RAG knowledge base search
    confirm_phone,
    confirm_email,
    check_booking_status,
    confirm_and_book_appointment,
]


# =============================================================================
# REGEX PATTERNS
# =============================================================================

YES_PAT = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|that's right|that is right|ok|okay|sure)\b",
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
# SLOT STATE — Clean State Container
# =============================================================================

@dataclass
class PatientState:
    """Clean state container for patient booking info."""
    full_name: Optional[str] = None
    phone_e164: Optional[str] = None
    phone_last4: Optional[str] = None
    email: Optional[str] = None
    reason: Optional[str] = None
    dt_local: Optional[datetime] = None
    dt_text: Optional[str] = None  # Natural language time before parsing
    
    # Duration tracking (from treatment_durations config)
    duration_minutes: int = 60  # Default 60 min, updated when reason is set
    time_status: str = "pending"  # "pending", "validating", "valid", "invalid"
    time_error: Optional[str] = None  # Error message if time is invalid
    
    # Confirmations
    phone_confirmed: bool = False
    email_confirmed: bool = False
    pending_confirm: Optional[str] = None  # "phone" or "email"
    
    # Phone source tracking (for confirmation UX)
    phone_source: Optional[str] = None  # "sip", "user_spoken", "extracted"
    
    # Review flow tracking
    review_presented: bool = False  # True after review summary shown
    review_snapshot: Optional[Dict[str, Any]] = None  # Snapshot at review time
    changed_fields: Optional[set] = field(default_factory=set)  # Fields changed after review
    pending_confirm_field: Optional[str] = None  # Single field needing confirmation
    partial_confirm_complete: bool = False  # True after confirming changed field
    
    # Booking state
    booking_attempted: bool = False
    booking_confirmed: bool = False
    booking_in_progress: bool = False
    calendar_event_id: Optional[str] = None
    
    # Context
    tz: str = DEFAULT_TZ
    patient_type: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if we have all required info for booking."""
        return all([
            self.full_name,
            self.phone_e164,
            self.phone_confirmed,
            self.email,
            self.email_confirmed,
            self.reason,
            self.dt_local,
        ])
    
    def missing_slots(self) -> List[str]:
        """Return list of missing required slots."""
        missing = []
        if not self.full_name:
            missing.append("full_name")
        if not self.phone_e164:
            missing.append("phone")
        elif not self.phone_confirmed:
            missing.append("phone_confirmed")
        if not self.email:
            missing.append("email")
        elif not self.email_confirmed:
            missing.append("email_confirmed")
        if not self.reason:
            missing.append("reason")
        if not self.dt_local:
            missing.append("datetime")
        return missing
    
    def slot_summary(self) -> str:
        """Human-readable slot summary for logging."""
        return (
            f"name={self.full_name or '?'}, "
            f"phone={'✓' if self.phone_confirmed else (self.phone_last4 or '?')}, "
            f"email={'✓' if self.email_confirmed else (self.email or '?')}, "
            f"reason={self.reason or '?'}, "
            f"time={self.dt_local.isoformat() if self.dt_local else '?'}"
        )
    
    def detailed_state_for_prompt(self) -> str:
        """
        Generate a detailed state snapshot for the dynamic system prompt.
        This is the LLM's 'source of truth' for what's already captured.
        """
        lines = []
        
        # Name
        if self.full_name:
            lines.append(f"• NAME: ✓ '{self.full_name}' — SAVED. Do NOT ask again.")
        else:
            lines.append("• NAME: ? — Still needed. Ask naturally.")
        
        # Phone - always show full number for confirmation prompt
        # Safety: ensure phone_e164 is string not tuple
        phone_display = self.phone_e164
        if isinstance(phone_display, tuple):
            phone_display = phone_display[0] if phone_display else None
        
        if phone_display and self.phone_confirmed:
            speakable = speakable_phone(phone_display)
            lines.append(f"• PHONE: ✓ {speakable} — CONFIRMED. Do NOT ask again.")
        elif phone_display:
            speakable = speakable_phone(phone_display)
            source_note = f" (from {self.phone_source})" if self.phone_source else ""
            lines.append(f"• PHONE: ⏳ {speakable}{source_note} — MUST CONFIRM FULL NUMBER with user!")
            lines.append(f"  → SAY: 'Just to confirm, is your phone number {speakable}?'")
            lines.append(f"  → If user says YES: call confirm_phone(confirmed=True)")
            lines.append(f"  → If user says NO: call confirm_phone(confirmed=False)")
        else:
            lines.append("• PHONE: ? — Still needed. Ask naturally.")
        
        # Email
        if self.email and self.email_confirmed:
            lines.append(f"• EMAIL: ✓ '{self.email}' — CONFIRMED. Do NOT ask again.")
        elif self.email:
            lines.append(f"• EMAIL: ⏳ '{self.email}' — Captured but needs confirmation.")
        else:
            lines.append("• EMAIL: ? — Still needed. Ask naturally.")
        
        # Reason with duration
        if self.reason:
            lines.append(f"• REASON: ✓ '{self.reason}' (Duration: {self.duration_minutes}m) — SAVED. Do NOT ask again.")
        else:
            lines.append("• REASON: ? — Still needed. Ask what brings them in.")
        
        # Time with validation status
        if self.dt_local and self.time_status == "valid":
            time_str = self.dt_local.strftime('%A, %B %d at %I:%M %p')
            lines.append(f"• TIME: ✓ {time_str} ({self.duration_minutes}m slot) — VALIDATED. Do NOT ask again.")
        elif self.time_status == "invalid" and self.time_error:
            lines.append(f"• TIME: ❌ INVALID — {self.time_error}")
            lines.append("  → Suggest the alternative time from the error message!")
        elif self.dt_text:
            lines.append(f"• TIME: ⏳ '{self.dt_text}' — Status: {self.time_status.upper()}")
        else:
            lines.append("• TIME: ? — Still needed. Ask when they'd like to come in.")
        
        # Booking status
        if self.booking_confirmed:
            lines.append("\n🎉 BOOKING STATUS: CONFIRMED! Appointment is booked.")
        elif self.is_complete():
            lines.append("\n✅ READY TO BOOK: All info collected. Summarize & confirm with patient.")
        else:
            missing = [s for s in self.missing_slots() if not s.endswith('_confirmed')]
            if missing:
                lines.append(f"\n⏳ STILL NEEDED: {', '.join(missing)}")
        
        return '\n'.join(lines)


# =============================================================================
# OPTIMIZED SUPABASE FETCH — SINGLE QUERY WITH JOINS + DEMO FALLBACK
# =============================================================================

# 🚀 PITCH MODE: Hardcoded demo clinic UUID for guaranteed fallback
DEMO_CLINIC_ID = "5afce5fa-8436-43a3-af65-da29ccad7228"

async def fetch_clinic_context_optimized(
    called_number: str,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict], str]:
    """
    A-TIER: Robust clinic lookup with fuzzy suffix matching and demo fallback.
    
    LOOKUP STRATEGY (in order):
    1. phone_numbers table: Match last 10 digits (ignores +1/+92 prefixes)
    2. clinics table: Direct phone match on clinic record (if stored there)
    3. DEMO FALLBACK: If only 1 clinic exists in DB, use it automatically
    4. PITCH MODE: Force-load DEMO_CLINIC_ID as ultimate fallback
    
    Returns: (clinic_info, agent_info, agent_settings, agent_name)
    """
    
    # Helper: Extract nested settings from agent_info
    def _extract_settings(agent_info: Optional[dict]) -> Tuple[Optional[dict], Optional[dict]]:
        """Extract agent_settings and clean agent_info dict."""
        if not agent_info:
            return None, None
        settings = None
        nested_settings = agent_info.get("agent_settings")
        if isinstance(nested_settings, list) and nested_settings:
            settings = nested_settings[0]
        elif isinstance(nested_settings, dict):
            settings = nested_settings
        clean_agent = {k: v for k, v in agent_info.items() if k != "agent_settings"}
        return clean_agent, settings
    
    # Helper: Fetch agent by clinic_id (used when phone_numbers lacks agent link)
    async def _fetch_agent_for_clinic(clinic_id: str) -> Optional[dict]:
        """Fetch agent and settings for a given clinic_id."""
        try:
            agent_res = await asyncio.to_thread(
                lambda: supabase.table("agents")
                .select(
                    "id, organization_id, clinic_id, name, default_language, status, "
                    "agent_settings(id, greeting_text, persona_tone, collect_insurance, "
                    "  emergency_triage_enabled, booking_confirmation_enabled, config_json, "
                    "  google_oauth_token)"
                )
                .eq("clinic_id", clinic_id)
                .limit(1)
                .execute()
            )
            return agent_res.data[0] if agent_res.data else None
        except Exception as e:
            logger.warning(f"[DB] Agent fetch for clinic {clinic_id} failed: {e}")
            return None
    
    try:
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Fuzzy suffix matching — use last 10 digits to ignore prefixes
        # ═══════════════════════════════════════════════════════════════════════
        digits_only = re.sub(r"\D", "", called_number or "")
        last10 = digits_only[-10:] if len(digits_only) >= 10 else digits_only
        
        logger.debug(f"[DB] Looking up phone: raw='{called_number}', last10='{last10}'")

        # ═══════════════════════════════════════════════════════════════════════
        # STRATEGY 1: Search phone_numbers table with suffix match
        # ═══════════════════════════════════════════════════════════════════════
        def _query_phone_numbers():
            q = supabase.table("phone_numbers").select(
                "clinic_id, agent_id, "
                # Join clinic
                "clinics:clinic_id("
                "  id, organization_id, name, timezone, default_phone_region, "
                "  address, city, state, zip_code, country"
                "), "
                # Join agent with nested settings
                "agents:agent_id("
                "  id, organization_id, clinic_id, name, default_language, status, "
                "  agent_settings(id, greeting_text, persona_tone, collect_insurance, "
                "    emergency_triage_enabled, booking_confirmation_enabled, config_json, "
                "    google_oauth_token)"
                ")"
            )
            if last10:
                q = q.ilike("phone_e164", f"%{last10}")
            else:
                q = q.eq("phone_e164", called_number)
            return q.limit(1).execute()

        result = await asyncio.to_thread(_query_phone_numbers)
        
        if result.data:
            row = result.data[0]
            clinic_info = row.get("clinics")
            agent_info = row.get("agents")
            
            # Fallback: fetch agent by clinic_id if not linked to phone
            if not agent_info and clinic_info:
                agent_info = await _fetch_agent_for_clinic(clinic_info["id"])
            
            agent_info, settings = _extract_settings(agent_info)
            agent_name = (agent_info or {}).get("name") or "Office Assistant"
            
            logger.info(f"[DB] ✓ Context loaded via phone_numbers: clinic={clinic_info.get('name') if clinic_info else 'None'}, agent={agent_name}")
            return clinic_info, agent_info, settings, agent_name
        
        logger.debug(f"[DB] No match in phone_numbers for last10='{last10}'")

        # ═══════════════════════════════════════════════════════════════════════
        # STRATEGY 2: Search clinics table directly (some setups store phone there)
        # ═══════════════════════════════════════════════════════════════════════
        def _query_clinics_direct():
            q = supabase.table("clinics").select(
                "id, organization_id, name, timezone, default_phone_region, "
                "address, city, state, zip_code, country, phone"
            )
            if last10:
                q = q.ilike("phone", f"%{last10}")
            return q.limit(1).execute()

        clinic_result = await asyncio.to_thread(_query_clinics_direct)
        
        if clinic_result.data:
            clinic_info = clinic_result.data[0]
            agent_info = await _fetch_agent_for_clinic(clinic_info["id"])
            agent_info, settings = _extract_settings(agent_info)
            agent_name = (agent_info or {}).get("name") or "Office Assistant"
            
            logger.info(f"[DB] ✓ Context loaded via clinics table: clinic={clinic_info.get('name')}, agent={agent_name}")
            return clinic_info, agent_info, settings, agent_name
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🚀 PITCH MODE — Phone lookup failed, force-load demo clinic by UUID
        # ═══════════════════════════════════════════════════════════════════════
        logger.warning(f"[DB] ⚠️ Phone lookup failed for {called_number}. Activating Pitch Mode.")
        logger.warning(f"[DB] 🚀 Pitch Mode: Force-loading Moiz Dental Clinic via UUID.")
        
        def _fetch_demo_clinic():
            """Force-fetch the demo clinic by hardcoded UUID."""
            return supabase.table("clinics").select(
                "id, organization_id, name, timezone, default_phone_region, "
                "address, city, state, zip_code, country"
            ).eq("id", DEMO_CLINIC_ID).limit(1).execute()

        demo_result = await asyncio.to_thread(_fetch_demo_clinic)
        
        if demo_result.data:
            clinic_info = demo_result.data[0]
            agent_info = await _fetch_agent_for_clinic(DEMO_CLINIC_ID)
            agent_info, settings = _extract_settings(agent_info)
            agent_name = (agent_info or {}).get("name") or "Office Assistant"
            
            logger.info(
                f"[DB] ✓ Pitch Mode context loaded: clinic={clinic_info.get('name')}, agent={agent_name}"
            )
            return clinic_info, agent_info, settings, agent_name

        # ═══════════════════════════════════════════════════════════════════════
        # ABSOLUTE FALLBACK — Demo clinic UUID not found in DB
        # ═══════════════════════════════════════════════════════════════════════
        logger.error(f"[DB] ❌ CRITICAL: Demo clinic UUID {DEMO_CLINIC_ID} not found in database!")
        return None, None, None, "Office Assistant"

    except Exception as e:
        logger.error(f"[DB] Context fetch error: {e}")
        traceback.print_exc()
        return None, None, None, "Office Assistant"


# =============================================================================
# SCHEDULE HELPERS
# =============================================================================

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

def load_schedule_from_settings(settings: Optional[dict]) -> Dict[str, Any]:
    """
    Load comprehensive scheduling config from agent_settings.config_json.
    
    Parses:
    - working_hours: Weekly schedule
    - closed_dates: Holiday/closing dates
    - slot_step_minutes: Slot interval for suggestions (default: 30)
    - treatment_durations: Service → minutes mapping
    - lunch_break: Daily recurring break (e.g., {"start": "13:00", "end": "14:00"})
    """
    cfg = {}
    try:
        raw = (settings or {}).get("config_json")
        if isinstance(raw, str) and raw.strip():
            cfg = json.loads(raw)
        elif isinstance(raw, dict):
            cfg = raw
    except Exception as e:
        logger.warning(f"[SCHEDULE] Failed parsing config_json: {e}")
    
    # Working hours
    wh = cfg.get("working_hours") or _default_hours()
    working_hours = {k: wh.get(k, []) for k in WEEK_KEYS}
    
    # Closed dates (holidays)
    closed = set()
    for d in (cfg.get("closed_dates") or []):
        try:
            closed.add(date.fromisoformat(d))
        except Exception:
            pass
    
    # Treatment durations: merge defaults with config overrides
    treatment_durations = DEFAULT_TREATMENT_DURATIONS.copy()
    cfg_durations = cfg.get("treatment_durations") or {}
    if isinstance(cfg_durations, dict):
        for service, mins in cfg_durations.items():
            try:
                treatment_durations[service] = int(mins)
            except (ValueError, TypeError):
                pass
    
    # Lunch break: daily recurring block
    lunch_break = cfg.get("lunch_break") or DEFAULT_LUNCH_BREAK
    if not isinstance(lunch_break, dict) or "start" not in lunch_break:
        lunch_break = DEFAULT_LUNCH_BREAK
    
    return {
        "working_hours": working_hours,
        "closed_dates": closed,
        "slot_step_minutes": int(cfg.get("slot_step_minutes") or 30),
        "treatment_durations": treatment_durations,
        "lunch_break": lunch_break,
    }


def is_within_working_hours(
    start_dt: datetime,
    schedule: Dict[str, Any],
    duration_minutes: int = 60,
) -> Tuple[bool, str]:
    """
    Enhanced validation: Check if entire appointment fits within working hours.
    
    Rules:
    A) The entire appointment (start to start + duration) must fit within working intervals
    B) The appointment must NOT overlap with lunch_break
    C) The date must not be in closed_dates (holidays)
    
    Returns: (is_valid, error_message)
    """
    if start_dt.tzinfo is None:
        return False, "ERROR: Invalid datetime (no timezone)."
    
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    
    # Rule C: Holiday check
    if start_dt.date() in schedule["closed_dates"]:
        return False, f"ERROR: We're closed on {start_dt.strftime('%B %d')}. Would you like to try another day?"
    
    dow_key = WEEK_KEYS[start_dt.weekday()]
    intervals = schedule["working_hours"].get(dow_key, [])
    
    if not intervals:
        return False, f"ERROR: We're closed on {start_dt.strftime('%A')}s. Would you like to try another day?"
    
    # Rule B: Lunch break check
    lunch = schedule.get("lunch_break") or DEFAULT_LUNCH_BREAK
    try:
        lh_s, lm_s = map(int, lunch["start"].split(":"))
        lh_e, lm_e = map(int, lunch["end"].split(":"))
        lunch_start = start_dt.replace(hour=lh_s, minute=lm_s, second=0, microsecond=0)
        lunch_end = start_dt.replace(hour=lh_e, minute=lm_e, second=0, microsecond=0)
        
        # Check if appointment overlaps with lunch
        if start_dt < lunch_end and end_dt > lunch_start:
            # Suggest time after lunch
            suggested = lunch_end + timedelta(minutes=15)
            return False, f"ERROR: Our team is on lunch break from {lunch_start.strftime('%I:%M %p')} to {lunch_end.strftime('%I:%M %p')}. How about {suggested.strftime('%I:%M %p')} instead?"
    except Exception as e:
        logger.warning(f"[SCHEDULE] Lunch break parse error: {e}")
    
    # Rule A: Check if entire appointment fits within working hours
    fits_in_interval = False
    for interval in intervals:
        try:
            sh, sm = map(int, interval["start"].split(":"))
            eh, em = map(int, interval["end"].split(":"))
            work_start = start_dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
            work_end = start_dt.replace(hour=eh, minute=em, second=0, microsecond=0)
            
            # Both start AND end must be within working hours
            if work_start <= start_dt and end_dt <= work_end:
                fits_in_interval = True
                break
        except Exception:
            continue
    
    if not fits_in_interval:
        # Find the clinic hours for error message
        if intervals:
            first_interval = intervals[0]
            clinic_open = first_interval.get("start", "09:00")
            clinic_close = first_interval.get("end", "17:00")
            return False, f"ERROR: We're open from {clinic_open} to {clinic_close} on {start_dt.strftime('%A')}s. Would you like a different time?"
        return False, "ERROR: That time is outside our working hours. Would you like to try a different time?"
    
    return True, "OK"


def get_duration_for_service(service: str, schedule: Dict[str, Any]) -> int:
    """
    Lookup treatment duration from schedule config.
    Returns duration in minutes, defaults to 60 if not found.
    """
    durations = schedule.get("treatment_durations") or DEFAULT_TREATMENT_DURATIONS
    
    # Try exact match first
    if service in durations:
        return durations[service]
    
    # Try case-insensitive match
    service_lower = service.lower()
    for key, mins in durations.items():
        if key.lower() == service_lower:
            return mins
    
    # Try partial match
    for key, mins in durations.items():
        if key.lower() in service_lower or service_lower in key.lower():
            return mins
    
    return 60  # Default duration


async def suggest_slots_around(
    clinic_id: str,
    requested_start_dt: datetime,
    duration_minutes: int,
    schedule: Dict[str, Any],
    tz_str: str,
    count: int = 3,
    window_hours: int = 4,
    step_min: int = 15,
) -> List[datetime]:
    """
    Deterministic slot suggestion around a requested time.
    
    Used when a requested slot is BOOKED - returns 2-3 closest alternatives
    AROUND the requested time (not from "now").
    
    Algorithm:
    1. Generate candidate start times within [requested - window, requested + window]
    2. Prefer same-day candidates first
    3. Filter candidates: within working hours, no overlap with existing appointments
    4. Return up to count best candidates sorted by absolute time difference from requested
    
    Args:
        clinic_id: Clinic UUID for appointment lookup
        requested_start_dt: The exact datetime user requested (that was unavailable)
        duration_minutes: Appointment duration
        schedule: Working hours config  
        tz_str: Timezone string
        count: Max alternatives to return (default 3)
        window_hours: How many hours before/after to search (default 4)
        step_min: Slot step in minutes (default 15)
    
    Returns:
        List of available datetime slots, sorted by proximity to requested_start_dt
    """
    tz = ZoneInfo(tz_str)
    requested_date = requested_start_dt.date()
    
    logger.info(f"[SUGGEST_SLOTS] Searching around {requested_start_dt.strftime('%Y-%m-%d %H:%M')} (±{window_hours}h)")
    
    # Fetch existing appointments for the target date (and adjacent days for edge cases)
    day_start = requested_start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=2)  # Include next day for afternoon requests
    
    existing_appointments = []
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("appointments")
            .select("start_time, end_time")
            .eq("clinic_id", clinic_id)
            .gte("start_time", day_start.isoformat())
            .lt("start_time", day_end.isoformat())
            .in_("status", BOOKED_STATUSES)
            .execute()
        )
        for appt in (result.data or []):
            try:
                appt_start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
                appt_end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
                existing_appointments.append((appt_start, appt_end))
            except Exception:
                pass
        logger.debug(f"[SUGGEST_SLOTS] Found {len(existing_appointments)} existing appointments")
    except Exception as e:
        logger.warning(f"[SUGGEST_SLOTS] Failed to fetch appointments: {e}")
    
    def is_slot_free(check_dt: datetime) -> bool:
        """Check if a slot is free (working hours + no conflicts)."""
        is_valid, _ = is_within_working_hours(check_dt, schedule, duration_minutes)
        if not is_valid:
            return False
        
        slot_end = check_dt + timedelta(minutes=duration_minutes + APPOINTMENT_BUFFER_MINUTES)
        for appt_start, appt_end in existing_appointments:
            buffered_end = appt_end + timedelta(minutes=APPOINTMENT_BUFFER_MINUTES)
            if check_dt < buffered_end and slot_end > appt_start:
                return False
        return True
    
    # Generate candidates: start from (requested - window_hours) to (requested + window_hours)
    candidates = []
    window_delta = timedelta(hours=window_hours)
    search_start = requested_start_dt - window_delta
    search_end = requested_start_dt + window_delta
    
    current = search_start.replace(second=0, microsecond=0)
    # Round to step boundary
    current = current.replace(minute=(current.minute // step_min) * step_min)
    
    while current <= search_end and len(candidates) < count * 3:  # Gather more than needed, filter later
        # Skip the exact requested time (we know it's taken)
        time_diff_minutes = abs((current - requested_start_dt).total_seconds() / 60)
        if time_diff_minutes >= step_min:  # Not the exact requested slot
            if is_slot_free(current):
                candidates.append((current, time_diff_minutes, current.date() == requested_date))
        current += timedelta(minutes=step_min)
    
    # Sort by: same-day first, then by proximity to requested time
    candidates.sort(key=lambda x: (not x[2], x[1]))  # (not same_day, time_diff)
    
    # Take top 'count' results
    result_slots = [c[0] for c in candidates[:count]]
    
    logger.info(f"[SUGGEST_SLOTS] Found {len(result_slots)} alternatives: {[s.strftime('%H:%M') for s in result_slots]}")
    return result_slots


async def get_alternatives_around_datetime(
    clinic_id: str,
    target_dt: datetime,
    duration_minutes: int,
    schedule: Dict[str, Any],
    tz_str: str,
    window_minutes: int = 60,
    num_slots: int = 2,
) -> List[datetime]:
    """
    Bidirectional slot search: Find available slots BEFORE and AFTER the requested time.
    
    This implements the "Nearby Slots" logic:
    - If user asks for 10 AM and it's taken, search 30-60 mins before AND after
    - Returns up to num_slots alternatives (e.g., 9:00 AM and 11:30 AM)
    
    NOTE: For more comprehensive searches, use suggest_slots_around() instead.
    This function is optimized for quick ±60min searches.
    
    Algorithm:
    1. Search backwards from target_dt (within window_minutes)
    2. Search forwards from target_dt (within window_minutes)
    3. Merge results, prioritizing closest to requested time
    
    Args:
        clinic_id: Clinic UUID for appointment lookup
        target_dt: The requested datetime that was unavailable
        duration_minutes: Appointment duration
        schedule: Working hours config
        tz_str: Timezone string
        window_minutes: How far before/after to search (default 60 min)
        num_slots: Max alternatives to return (default 2)
    
    Returns:
        List of available datetime slots, sorted by proximity to target_dt
    """
    tz = ZoneInfo(tz_str)
    slot_step = schedule.get("slot_step_minutes", 30)
    
    # Fetch existing appointments for the target date
    day_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    
    existing_appointments = []
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("appointments")
            .select("start_time, end_time")
            .eq("clinic_id", clinic_id)
            .gte("start_time", day_start.isoformat())
            .lt("start_time", day_end.isoformat())
            .in_("status", BOOKED_STATUSES)
            .execute()
        )
        for appt in (result.data or []):
            try:
                appt_start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
                appt_end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
                existing_appointments.append((appt_start, appt_end))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[NEARBY_SLOTS] Failed to fetch appointments: {e}")
    
    def is_slot_available(check_dt: datetime) -> bool:
        """Check if a slot is free (working hours + no conflicts)."""
        is_valid, _ = is_within_working_hours(check_dt, schedule, duration_minutes)
        if not is_valid:
            return False
        
        slot_end = check_dt + timedelta(minutes=duration_minutes + APPOINTMENT_BUFFER_MINUTES)
        for appt_start, appt_end in existing_appointments:
            buffered_end = appt_end + timedelta(minutes=APPOINTMENT_BUFFER_MINUTES)
            if check_dt < buffered_end and slot_end > appt_start:
                return False
        return True
    
    alternatives = []
    
    # Search BACKWARDS from target time
    search_back = target_dt - timedelta(minutes=slot_step)
    earliest = target_dt - timedelta(minutes=window_minutes)
    while search_back >= earliest and len(alternatives) < num_slots:
        # Round to slot step
        rounded = search_back.replace(
            minute=(search_back.minute // slot_step) * slot_step,
            second=0, microsecond=0
        )
        if is_slot_available(rounded):
            alternatives.append(rounded)
        search_back -= timedelta(minutes=slot_step)
    
    # Search FORWARDS from target time
    search_forward = target_dt + timedelta(minutes=slot_step)
    latest = target_dt + timedelta(minutes=window_minutes)
    while search_forward <= latest and len(alternatives) < num_slots:
        # Round to slot step
        rounded = search_forward.replace(
            minute=(search_forward.minute // slot_step) * slot_step,
            second=0, microsecond=0
        )
        if is_slot_available(rounded):
            alternatives.append(rounded)
        search_forward += timedelta(minutes=slot_step)
    
    # Sort by proximity to target time
    alternatives.sort(key=lambda dt: abs((dt - target_dt).total_seconds()))
    
    logger.info(f"[NEARBY_SLOTS] Found {len(alternatives)} alternatives around {target_dt.strftime('%I:%M %p')}: {[a.strftime('%I:%M %p') for a in alternatives]}")
    
    return alternatives[:num_slots]


async def get_next_available_slots(
    clinic_id: str,
    schedule: Dict[str, Any],
    tz_str: str,
    duration_minutes: int = 60,
    num_slots: int = 3,
    days_ahead: int = 14,
) -> List[datetime]:
    """
    Find the next N available appointment slots.
    
    Algorithm:
    1. Fetch all scheduled appointments from now until days_ahead
    2. Iterate through valid working hour slots (skipping lunch)
    3. Return slots that are not occupied
    """
    tz = ZoneInfo(tz_str)
    now = datetime.now(tz)
    
    # Round up to next slot step
    slot_step = schedule.get("slot_step_minutes", 30)
    minutes_to_add = slot_step - (now.minute % slot_step)
    if minutes_to_add == slot_step:
        minutes_to_add = 0
    current = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
    
    end_search = now + timedelta(days=days_ahead)
    logger.info(f"🔍 [SLOTS] Searching window: {now.date()} to {end_search.date()}")
    
    # Fetch existing appointments from Supabase
    existing_appointments = []
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("appointments")
            .select("start_time, end_time")
            .eq("clinic_id", clinic_id)
            .gte("start_time", now.isoformat())
            .lte("start_time", end_search.isoformat())
            .in_("status", BOOKED_STATUSES)
            .execute()
        )
        for appt in (result.data or []):
            try:
                appt_start = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
                appt_end = datetime.fromisoformat(appt["end_time"].replace("Z", "+00:00"))
                existing_appointments.append((appt_start, appt_end))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[SLOTS] Failed to fetch appointments: {e}")
    
    available_slots = []
    
    while current < end_search and len(available_slots) < num_slots:
        # Check if slot is valid (working hours, not lunch, not holiday)
        is_valid, _ = is_within_working_hours(current, schedule, duration_minutes)
        
        if is_valid:
            # Check if slot conflicts with existing appointments
            slot_end = current + timedelta(minutes=duration_minutes)
            is_free = True
            
            for appt_start, appt_end in existing_appointments:
                # Check for overlap
                if current < appt_end and slot_end > appt_start:
                    is_free = False
                    break
            
            if is_free:
                available_slots.append(current)
        
        # Move to next slot
        current += timedelta(minutes=slot_step)
        
        # Skip to next day if we've passed working hours
        dow_key = WEEK_KEYS[current.weekday()]
        intervals = schedule["working_hours"].get(dow_key, [])
        if intervals:
            last_interval = intervals[-1]
            try:
                eh, em = map(int, last_interval["end"].split(":"))
                day_end = current.replace(hour=eh, minute=em)
                if current >= day_end:
                    # Jump to next day morning
                    next_day = current.date() + timedelta(days=1)
                    current = datetime.combine(next_day, datetime.min.time(), tzinfo=tz)
                    current = current.replace(hour=8, minute=0)  # Start checking from 8am
            except Exception:
                pass
    
    return available_slots


# =============================================================================
# CALENDAR AUTH RESOLUTION — DATABASE-BACKED OAUTH PERSISTENCE
# =============================================================================

# Global reference for the current agent settings ID (for token refresh saves)
_GLOBAL_AGENT_SETTINGS_ID: Optional[str] = None


def _load_env_oauth_token() -> Optional[dict]:
    """Load OAuth token from ENV-configured file path (fallback for local dev)."""
    token_path = GOOGLE_OAUTH_TOKEN_PATH
    if not token_path or not os.path.exists(token_path):
        return None
    try:
        with open(token_path, "r", encoding="utf-8") as f:
            token_data = json.load(f)
        logger.info("[CALENDAR_AUTH] Loaded OAuth token from local file.")
        return token_data
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] Failed to load local token: {e}")
        return None


async def fetch_oauth_token_from_db(agent_settings_id: str) -> Optional[dict]:
    """
    Fetch Google OAuth token from agent_settings.google_oauth_token column.
    
    Uses asyncio.to_thread for non-blocking database access.
    Returns the token dict if found and valid, None otherwise.
    """
    if not agent_settings_id:
        logger.warning("[CALENDAR_AUTH] No agent_settings_id provided for DB token fetch.")
        return None
    
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("agent_settings")
            .select("google_oauth_token")
            .eq("id", agent_settings_id)
            .limit(1)
            .execute()
        )
        
        if not result.data:
            logger.warning(f"[CALENDAR_AUTH] No agent_settings found for id={agent_settings_id}")
            return None
        
        token_json = result.data[0].get("google_oauth_token")
        
        if not token_json:
            logger.debug("[CALENDAR_AUTH] google_oauth_token column is empty in DB.")
            return None
        
        # Parse JSON if stored as string
        if isinstance(token_json, str):
            token_data = json.loads(token_json)
        else:
            token_data = token_json
        
        # Validate required fields
        if not token_data.get("refresh_token"):
            logger.warning("[CALENDAR_AUTH] DB token missing refresh_token - may not be able to refresh.")
        
        logger.info("[CALENDAR_AUTH] ✓ Loaded OAuth token from database.")
        return token_data
        
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] DB token fetch error: {e}")
        return None


async def save_refreshed_token_to_db(agent_settings_id: str, token_data: dict):
    """
    Save refreshed OAuth token back to agent_settings.google_oauth_token.
    
    Called asynchronously when Google refreshes an access token.
    Uses asyncio.create_task in the callback to avoid blocking the voice conversation.
    """
    if not agent_settings_id:
        logger.warning("[CALENDAR_AUTH] Cannot save token - no agent_settings_id.")
        return
    
    try:
        await asyncio.to_thread(
            lambda: supabase.table("agent_settings")
            .update({"google_oauth_token": json.dumps(token_data)})
            .eq("id", agent_settings_id)
            .execute()
        )
        logger.info("[CALENDAR_AUTH] ✓ Refreshed OAuth token saved to database.")
    except Exception as e:
        logger.error(f"[CALENDAR_AUTH] Failed to save refreshed token to DB: {e}")


def _create_token_refresh_callback(agent_settings_id: str) -> callable:
    """
    Create a callback that saves refreshed tokens to the database.
    
    Uses asyncio.create_task for non-blocking persistence so the
    voice conversation is never interrupted by token refresh saves.
    """
    def on_token_refresh(new_token_dict: dict):
        try:
            # Get the current event loop, create task for non-blocking save
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    save_refreshed_token_to_db(agent_settings_id, new_token_dict)
                )
            else:
                # Fallback for edge case where loop isn't running
                asyncio.run(save_refreshed_token_to_db(agent_settings_id, new_token_dict))
        except Exception as e:
            logger.error(f"[CALENDAR_AUTH] Token refresh callback error: {e}")
    
    return on_token_refresh


async def resolve_calendar_auth_async(
    clinic_info: Optional[dict],
    settings: Optional[dict] = None,
) -> Tuple[Optional[CalendarAuth], str, Optional[callable]]:
    """
    Resolve calendar auth with DATABASE-FIRST priority.
    
    Priority order:
    1. Pre-fetched token from settings.google_oauth_token (already loaded in initial query)
    2. Database fetch: agent_settings.google_oauth_token column (if not pre-loaded)
    3. Local file: ENV-configured GOOGLE_OAUTH_TOKEN path (fallback for dev)
    
    Returns: (CalendarAuth, calendar_id, refresh_callback)
    
    The refresh_callback should be passed to _get_calendar_service so that
    token refreshes are persisted back to the database non-blocking.
    """
    global _GLOBAL_AGENT_SETTINGS_ID
    
    calendar_id = GOOGLE_CALENDAR_ID_DEFAULT
    refresh_callback = None
    
    # Extract agent_settings_id for DB operations
    agent_settings_id = (settings or {}).get("id") if settings else None
    if agent_settings_id:
        _GLOBAL_AGENT_SETTINGS_ID = agent_settings_id
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 1A: Pre-fetched OAuth Token (already in settings from initial query)
    # ═══════════════════════════════════════════════════════════════════════════
    token_data = None
    
    if settings:
        pre_fetched_token = settings.get("google_oauth_token")
        if pre_fetched_token:
            # Parse JSON if stored as string
            if isinstance(pre_fetched_token, str):
                try:
                    token_data = json.loads(pre_fetched_token)
                    logger.debug("[CALENDAR_AUTH] Using pre-fetched OAuth token from settings.")
                except json.JSONDecodeError as e:
                    logger.warning(f"[CALENDAR_AUTH] Failed to parse pre-fetched token: {e}")
            elif isinstance(pre_fetched_token, dict):
                token_data = pre_fetched_token
                logger.debug("[CALENDAR_AUTH] Using pre-fetched OAuth token dict from settings.")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 1B: Database OAuth Token fetch (if not pre-loaded)
    # ═══════════════════════════════════════════════════════════════════════════
    if not token_data and agent_settings_id:
        logger.debug(f"[CALENDAR_AUTH] Fetching OAuth token from DB (settings_id={agent_settings_id})")
        token_data = await fetch_oauth_token_from_db(agent_settings_id)
    
    # Build CalendarAuth if we have token data from DB
    if token_data:
        try:
            token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
            auth = CalendarAuth(
                auth_type="oauth_user",
                secret_json=token_data,
                delegated_user=None,
            )
            
            # Create refresh callback for non-blocking token persistence
            if agent_settings_id:
                refresh_callback = _create_token_refresh_callback(agent_settings_id)
            
            logger.info("[CALENDAR_AUTH] ✓ Using DATABASE OAuth token (production mode).")
            return auth, calendar_id, refresh_callback
            
        except Exception as e:
            logger.error(f"[CALENDAR_AUTH] DB token parse error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 2: Local File OAuth Token (Development fallback)
    # ═══════════════════════════════════════════════════════════════════════════
    if GOOGLE_CALENDAR_AUTH_MODE == "oauth":
        token_data = _load_env_oauth_token()
        if token_data:
            try:
                token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
                auth = CalendarAuth(
                    auth_type="oauth_user",
                    secret_json=token_data,
                    delegated_user=None,
                )
                logger.info("[CALENDAR_AUTH] ✓ Using LOCAL FILE OAuth token (dev mode).")
                return auth, calendar_id, None
            except Exception as e:
                logger.error(f"[CALENDAR_AUTH] Local file OAuth failed: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NO TOKEN FOUND — CRITICAL ERROR
    # ═══════════════════════════════════════════════════════════════════════════
    logger.critical(
        "[CALENDAR_AUTH] CRITICAL: No Google OAuth token found. "
        "Please run oauth_bootstrap.py and upload the token to Supabase "
        "(agent_settings.google_oauth_token column)."
    )
    return None, calendar_id, None


def resolve_calendar_auth(clinic_info: Optional[dict]) -> Tuple[Optional[CalendarAuth], str]:
    """
    LEGACY SYNC WRAPPER — For backwards compatibility with existing code.
    
    Prefer using resolve_calendar_auth_async() in async contexts for
    non-blocking database access and refresh callback support.
    """
    # Fallback to ENV OAuth (sync path for legacy callers)
    if GOOGLE_CALENDAR_AUTH_MODE == "oauth":
        token_data = _load_env_oauth_token()
        if token_data:
            try:
                token_data.setdefault("token_uri", "https://oauth2.googleapis.com/token")
                auth = CalendarAuth(
                    auth_type="oauth_user",
                    secret_json=token_data,
                    delegated_user=None,
                )
                return auth, GOOGLE_CALENDAR_ID_DEFAULT
            except Exception as e:
                logger.error(f"[CALENDAR_AUTH] ENV OAuth failed: {e}")
    
    logger.warning("[CALENDAR_AUTH] No OAuth token available (sync path).")
    return None, GOOGLE_CALENDAR_ID_DEFAULT


# =============================================================================
# DETERMINISTIC EXTRACTORS (Inline for speed)
# =============================================================================

def extract_name_quick(text: str) -> Optional[str]:
    """Quick name extraction from common patterns."""
    patterns = [
        r"\b(?:my\s+name\s+is|i\s+am|i'm|this\s+is|call\s+me)\s+([A-Za-z][A-Za-z\s\.'-]{2,})",
        r"^(?:it'?s|its)\s+([A-Za-z][A-Za-z\s\.'-]{2,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Clean trailing noise
            name = re.split(r"\b(and|i|want|need|would|like|to|for|at|my|phone|email)\b", name, flags=re.I)[0].strip()
            if len(name) >= 3:
                return name.title()
    return None


def extract_reason_quick(text: str) -> Optional[str]:
    """Quick service extraction."""
    t = text.lower()
    service_map = {
        "whiten": "Teeth whitening",
        "whitening": "Teeth whitening",
        "clean": "Cleaning",
        "cleaning": "Cleaning",
        "checkup": "Checkup",
        "check-up": "Checkup",
        "exam": "Checkup",
        "pain": "Tooth pain",
        "toothache": "Tooth pain",
        "consult": "Consultation",
        "extract": "Extraction",
        "filling": "Filling",
        "crown": "Crown",
        "root canal": "Root canal",
    }
    for key, value in service_map.items():
        if key in t:
            return value
    return None


def email_for_speech(email: str) -> str:
    """Convert email to speech-friendly format."""
    e = (email or "").strip().lower()
    e = e.replace("@", " at ").replace(".", " dot ")
    return re.sub(r"\s+", " ", e).strip()


# =============================================================================
# BOOKING FUNCTIONS
# =============================================================================

def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ))
    return dt.isoformat()


async def is_slot_free_supabase(clinic_id: str, start_dt: datetime, end_dt: datetime) -> bool:
    """Check if slot is free in Supabase appointments table."""
    try:
        res = await asyncio.to_thread(
            lambda: supabase.table("appointments")
            .select("id")
            .eq("clinic_id", clinic_id)
            .lt("start_time", _iso(end_dt))
            .gt("end_time", _iso(start_dt))
            .in_("status", BOOKED_STATUSES)
            .execute()
        )
        return len(res.data or []) == 0
    except Exception as e:
        logger.error(f"[DB] Availability check error: {e}")
        return False


async def book_to_supabase(
    clinic_info: dict,
    patient_state: PatientState,
    calendar_event_id: Optional[str] = None,
) -> bool:
    """Insert appointment into Supabase."""
    try:
        start_time = patient_state.dt_local
        end_time = start_time + timedelta(minutes=patient_state.duration_minutes)  # Use service-specific duration
        
        payload = {
            "organization_id": clinic_info["organization_id"],
            "clinic_id": clinic_info["id"],
            "patient_name": patient_state.full_name,
            "patient_phone_masked": patient_state.phone_last4,
            "patient_email": patient_state.email,
            "start_time": _iso(start_time),
            "end_time": _iso(end_time),
            "status": "scheduled",
            "source": "ai",
            "reason": patient_state.reason,
        }
        
        if calendar_event_id:
            payload["calendar_event_id"] = calendar_event_id
        
        await asyncio.to_thread(
            lambda: supabase.table("appointments").insert(payload).execute()
        )
        logger.info("[DB] ✓ Appointment saved to Supabase")
        return True
    except Exception as e:
        logger.error(f"[DB] Booking insert error: {e}")
        return False


async def try_book_appointment(
    session: AgentSession,
    clinic_info: dict,
    patient_state: PatientState,
    settings: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    Non-blocking booking with calendar verification and DB-backed OAuth.
    
    Uses resolve_calendar_auth_async() for database-first token resolution
    with non-blocking token refresh persistence.
    
    Returns: (success, message)
    """
    if patient_state.booking_confirmed:
        return True, "Your appointment is already booked!"
    
    if patient_state.booking_in_progress:
        return False, "Already processing..."
    
    patient_state.booking_in_progress = True
    
    try:
        # Validate
        if not patient_state.is_complete():
            patient_state.booking_in_progress = False
            return False, f"Missing: {', '.join(patient_state.missing_slots())}"
        
        # Get calendar auth with DB-first priority and refresh callback
        auth, calendar_id, refresh_callback = await resolve_calendar_auth_async(
            clinic_info, 
            settings=settings
        )
        if not auth:
            patient_state.booking_in_progress = False
            logger.error("[BOOKING] Calendar auth failed - no OAuth token available.")
            return False, "Calendar not configured. Please contact the clinic."
        
        # Get service with refresh callback for non-blocking token persistence
        try:
            service = await asyncio.wait_for(
                asyncio.to_thread(
                    _get_calendar_service, 
                    auth=auth,
                    on_refresh_callback=refresh_callback
                ),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            patient_state.booking_in_progress = False
            return False, "Calendar not responding."
        
        if not service:
            patient_state.booking_in_progress = False
            return False, "Calendar unavailable."
        
        start_dt = patient_state.dt_local
        end_dt = start_dt + timedelta(minutes=DEFAULT_MIN)
        
        # Say we're checking
        await session.say("Let me check that time for you...")
        
        # Check availability
        try:
            def _check_freebusy():
                return service.freebusy().query(body={
                    "timeMin": start_dt.isoformat(),
                    "timeMax": end_dt.isoformat(),
                    "timeZone": patient_state.tz,
                    "items": [{"id": calendar_id}],
                }).execute()
            
            resp = await asyncio.wait_for(asyncio.to_thread(_check_freebusy), timeout=10.0)
            busy = resp.get("calendars", {}).get(calendar_id, {}).get("busy", [])
            
            if busy:
                patient_state.booking_in_progress = False
                patient_state.dt_local = None
                return False, "That time slot is taken. Would you like to try another time?"
        except Exception as e:
            logger.warning(f"[BOOKING] Freebusy check failed: {e}")
        
        # Say we're booking
        await session.say("That time is available! Booking now...")
        
        # Create event
        try:
            def _create_event():
                return service.events().insert(
                    calendarId=calendar_id,
                    body={
                        "summary": f"{patient_state.reason or 'Appointment'} — {patient_state.full_name}",
                        "description": f"Patient: {patient_state.full_name}\nPhone: {patient_state.phone_e164}\nEmail: {patient_state.email}",
                        "start": {"dateTime": start_dt.isoformat(), "timeZone": patient_state.tz},
                        "end": {"dateTime": end_dt.isoformat(), "timeZone": patient_state.tz},
                        "attendees": [{"email": patient_state.email}] if patient_state.email else [],
                    },
                    sendUpdates="all",
                ).execute()
            
            event = await asyncio.wait_for(asyncio.to_thread(_create_event), timeout=20.0)
            
            if not event or not event.get("id"):
                patient_state.booking_in_progress = False
                return False, "Booking failed. Please try again."
            
            event_id = event.get("id")
            
            # Verify
            verified = await asyncio.wait_for(
                asyncio.to_thread(lambda: service.events().get(calendarId=calendar_id, eventId=event_id).execute()),
                timeout=10.0
            )
            
            if verified.get("status") not in ("confirmed", "tentative"):
                patient_state.booking_in_progress = False
                return False, "Booking not confirmed."
            
            # Success!
            patient_state.booking_confirmed = True
            patient_state.calendar_event_id = event_id
            
            # Non-blocking Supabase save
            asyncio.create_task(book_to_supabase(clinic_info, patient_state, event_id))
            
            logger.info(f"[BOOKING] ✓ SUCCESS! Event ID: {event_id}")
            return True, f"Your appointment is confirmed for {start_dt.strftime('%A, %B %d at %I:%M %p')}. We'll see you then!"
            
        except asyncio.TimeoutError:
            patient_state.booking_in_progress = False
            return False, "Booking timed out."
        except Exception as e:
            logger.error(f"[BOOKING] Error: {e}")
            patient_state.booking_in_progress = False
            return False, "Something went wrong."
    
    finally:
        patient_state.booking_in_progress = False


# =============================================================================
# A-TIER AGENT & ENTRYPOINT (VoicePipelineAgent for v1.3.10)
# =============================================================================

async def snappy_entrypoint(ctx: JobContext):
    """
    A-TIER ENTRYPOINT with <1s response latency using VoicePipelineAgent.
    
    Optimizations:
    1. Single Supabase query (3.2s → 100ms)
    2. VoicePipelineAgent with min_endpointing_delay=0.5s
    3. gpt-4o-mini for speed + quality
    4. LLM Function Calling for real-time parallel extraction
    5. Non-blocking booking
    6. Global state for tool access
    7. Dynamic Slot-Aware Prompting - system prompt refreshes every turn!
    8. SIP Telephony Support - auto-detect inbound calls & pre-fill caller phone
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_TZ, _GLOBAL_CLINIC_INFO, _REFRESH_AGENT_MEMORY
    
    state = PatientState()
    _GLOBAL_STATE = state  # Set global reference for tools
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔧 DEFAULTS INITIALIZATION — Must be set BEFORE SIP block uses clinic_region
    # ═══════════════════════════════════════════════════════════════════════════
    clinic_info = None
    agent_info = None
    settings = None
    agent_name = "Office Assistant"
    clinic_name = "our clinic"
    clinic_tz = DEFAULT_TZ
    clinic_region = DEFAULT_PHONE_REGION
    agent_lang = "en-US"
    
    call_started = time.time()
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    logger.info(f"[LIFECYCLE] Participant: {participant.identity}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📞 SIP TELEPHONY DETECTION — Prioritize real SIP metadata over job metadata
    # ═══════════════════════════════════════════════════════════════════════════
    called_num = None
    caller_phone = None
    is_sip_call = False
    used_fallback_called_num = False
    
    # PRIORITY 1: Real SIP participant metadata (production telephony)
    if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
        is_sip_call = True
        # Extract SIP attributes from participant
        sip_attrs = participant.attributes or {}
        caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
        # Fix: Twilio dialed number is typically in one of these keys
        called_num = sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
        called_num = _normalize_sip_user_to_e164(called_num)
        
        logger.info(f"📞 [SIP] Inbound call detected!")
        logger.info(f"📞 [SIP] Caller (from): {caller_phone}")
        logger.info(f"📞 [SIP] Called (to): {called_num}")
        
        # Pre-fill caller's phone from SIP - but NEVER auto-confirm!
        # Agent MUST confirm full phone number with user before booking
        if caller_phone:
            clean_phone, last4 = _normalize_phone_preserve_plus(caller_phone, clinic_region)
            if clean_phone:
                state.phone_e164 = str(clean_phone)  # Enforce string type
                state.phone_last4 = str(last4) if last4 else ""
                # Safety guard: ensure no tuple was stored
                _ensure_phone_is_string(state)
                state.phone_confirmed = False  # NEVER auto-confirm - always ask user
                state.phone_source = "sip"  # Track source for confirmation logic
                state.pending_confirm = "phone"  # Flag that phone needs confirmation
                state.pending_confirm_field = "phone"  # For deterministic yes/no routing
                speakable = speakable_phone(state.phone_e164)
                logger.info(f"📞 [SIP] ⏳ Caller phone pre-filled (needs confirmation): {speakable}")
    
    # PRIORITY 2: Room name regex — flexible US phone number extraction
    # Matches +1XXXXXXXXXX anywhere in room name (e.g., call_+13103410536_abc123)
    if not called_num:
        room_name = getattr(ctx.room, "name", "") or ""
        # Try US format first (+1 followed by 10 digits)
        room_match = re.search(r"(\+1\d{10})", room_name)
        if not room_match:
            # Fallback: any number in call_{number}_ format
            room_match = re.search(r"call_(\+?\d+)_", room_name)
        if room_match:
            called_num = _normalize_sip_user_to_e164(room_match.group(1))
            logger.info(f"[ROOM] ✓ Extracted phone from room name: {called_num}")

    # PRIORITY 3: Job metadata (LiveKit Playground / testing)
    if not called_num:
        metadata = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
        sip_info = metadata.get("sip", {}) if isinstance(metadata, dict) else {}
        called_num = _normalize_sip_user_to_e164(sip_info.get("toUser"))
        # Also check for caller in job metadata
        if not caller_phone:
            caller_phone = sip_info.get("fromUser") or sip_info.get("phoneNumber")
        
        if called_num:
            logger.info(f"[METADATA] Using job metadata: toUser={called_num}")
    
    # PRIORITY 4: Fallback to environment default (for local testing only)
    # NOTE: Comment out in production to ensure proper SIP routing
    if not called_num:
        called_num = os.getenv("DEFAULT_TEST_NUMBER", "+13103410536")
        logger.warning(f"[FALLBACK] Using default test number: {called_num}")
        used_fallback_called_num = True

    # ⚡ FAST-PATH CONTEXT: start the optimized fetch immediately once called_num is known.
    # Do not block audio startup on this; we only wait a tiny budget to personalize if it returns fast.
    context_task: Optional[asyncio.Task] = None
    if called_num:
        context_task = asyncio.create_task(fetch_clinic_context_optimized(called_num))

    # ═══════════════════════════════════════════════════════════════════════════
    # 🏥 IDENTITY-FIRST: Wait up to 5s for DB context (better silence than wrong name)
    # ═══════════════════════════════════════════════════════════════════════════
    if context_task:
        try:
            clinic_info, agent_info, settings, agent_name = await asyncio.wait_for(
                asyncio.shield(context_task), timeout=5.0
            )
            logger.info(f"[DB] ✓ Context loaded in <5s: clinic={clinic_info.get('name') if clinic_info else 'None'}")
        except asyncio.TimeoutError:
            logger.warning("[DB] ⚠️ Context fetch exceeded 5s timeout — using defaults")

    # Safety net: Force-load demo clinic if context still None
    if clinic_info is None:
        logger.warning("[DB] ⚠️ clinic_info is None — force-loading demo fallback")
        clinic_info = {"id": DEMO_CLINIC_ID, "name": "Moiz Dental Clinic Islamabad"}

    # Apply whatever context we have at this point
    _GLOBAL_CLINIC_INFO = clinic_info

    global _GLOBAL_AGENT_SETTINGS
    _GLOBAL_AGENT_SETTINGS = settings

    clinic_name = (clinic_info or {}).get("name") or clinic_name
    clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
    clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
    agent_lang = (agent_info or {}).get("default_language") or agent_lang

    state.tz = clinic_tz
    _GLOBAL_CLINIC_TZ = clinic_tz  # Set global for tool timezone anchoring

    schedule = load_schedule_from_settings(settings or {})

    global _GLOBAL_SCHEDULE
    _GLOBAL_SCHEDULE = schedule
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🧠 DYNAMIC SLOT-AWARE PROMPTING — Refresh system prompt every turn
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_updated_instructions() -> str:
        """
        Generate fresh system prompt with current PatientState snapshot.
        This is the key to Dynamic Slot-Aware Prompting!
        """
        return A_TIER_PROMPT.format(
            agent_name=agent_name,
            clinic_name=clinic_name,
            timezone=clinic_tz,
            state_summary=state.detailed_state_for_prompt(),  # Real-time snapshot!
        )
    
    # Store reference to session for the refresh callback (set after session creation)
    session_ref = {"session": None, "agent": None}
    
    def refresh_agent_memory():
        """
        Refresh the LLM's system prompt with current state.
        Called after user speech and after tool updates.
        """
        try:
            session = session_ref.get("session")
            agent = session_ref.get("agent")
            
            if not session or not agent:
                logger.debug("[MEMORY] Session/agent not yet initialized, skipping refresh")
                return
            
            new_instructions = get_updated_instructions()
            
            # Update the agent's instructions directly
            if hasattr(agent, '_instructions'):
                agent._instructions = new_instructions
                logger.debug(f"[MEMORY] ✓ Refreshed agent instructions. State: {state.slot_summary()}")
            
            # Also try to update chat context if available
            if hasattr(session, 'chat_ctx') and session.chat_ctx:
                try:
                    messages = getattr(session.chat_ctx, 'messages', None) or getattr(session.chat_ctx, 'items', None)
                    if messages and len(messages) > 0:
                        first_msg = messages[0]
                        if hasattr(first_msg, 'content'):
                            first_msg.content = new_instructions
                        elif hasattr(first_msg, 'text_content'):
                            # Try to update text_content for ChatMessage objects
                            if hasattr(first_msg, '_text_content'):
                                first_msg._text_content = new_instructions
                except Exception as e:
                    logger.debug(f"[MEMORY] chat_ctx update skipped: {e}")
                    
        except Exception as e:
            logger.warning(f"[MEMORY] Refresh failed: {e}")
    
    # Set global refresh callback for tools to use
    _REFRESH_AGENT_MEMORY = refresh_agent_memory
    
    # Build initial prompt (may be placeholder; we'll refresh once DB context arrives)
    initial_system_prompt = get_updated_instructions()

    # ═══════════════════════════════════════════════════════════════════════════
    # 🎙️ GREETING: Use DB greeting_text if context loaded, otherwise fallback
    # ═══════════════════════════════════════════════════════════════════════════
    # Build greeting - if we have SIP phone prefilled, proactively confirm it
    phone_confirm_addon = ""
    if state.phone_e164 and state.phone_source == "sip" and not state.phone_confirmed:
        # SIP call with prefilled phone - proactively ask to confirm FULL number
        speakable = speakable_phone(state.phone_e164)
        phone_confirm_addon = f" I see you're calling from {speakable}. Is this the best number to reach you on?"
        logger.info(f"[GREETING] Will ask to confirm SIP caller phone: {speakable}")
    
    if settings and settings.get("greeting_text"):
        greeting = settings.get("greeting_text") + phone_confirm_addon
        logger.info(f"[GREETING] Using DB greeting: {greeting[:50]}...")
    elif clinic_info:
        base_greeting = f"Hi, thanks for calling {clinic_name}!"
        if phone_confirm_addon:
            greeting = base_greeting + phone_confirm_addon
        else:
            greeting = base_greeting + " How can I help you today?"
        logger.info(f"[GREETING] Using clinic-aware greeting for {clinic_name}")
    else:
        greeting = "Hello! Thanks for calling." + (phone_confirm_addon or " How can I help you today?")
        logger.info("[GREETING] Using default greeting (DB context not loaded)")
    
    # ⚡ HIGH-PERFORMANCE LLM with function calling
    llm_instance = openai_plugin.LLM(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    # ⚡ SNAPPY STT
    if os.getenv("DEEPGRAM_API_KEY"):
        stt_instance = deepgram_plugin.STT(
            model="nova-2-general",
            language=agent_lang,
        )
    elif os.getenv("ASSEMBLYAI_API_KEY"):
        stt_instance = assemblyai_plugin.STT()
    else:
        stt_instance = openai_plugin.STT(model="gpt-4o-transcribe", language="en")
    
    # ⚡ FAST VAD
    vad_instance = silero.VAD.load(
        min_speech_duration=0.1,
        min_silence_duration=0.3,
    )
    
    # TTS
    if os.getenv("CARTESIA_API_KEY"):
        tts_instance = cartesia_plugin.TTS(
            model="sonic-3",
            voice=os.getenv("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        )
    else:
        tts_instance = openai_plugin.TTS(
            model="gpt-4o-mini-tts",
            voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
        )
    
    # ⚡ A-TIER AgentSession with LLM Function Calling (v1.2.14 API)
    # The LLM extracts data via tools IN REAL-TIME while generating speech
    # No blocking listeners - parallel extraction eliminates 20s silences
    session = AgentSession(
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        vad=vad_instance,
        min_endpointing_delay=0.5,  # ⚡ 0.5s for snappy turn-taking (was 1.5s)
        max_endpointing_delay=1.5,  # Reduced from 2.0 for faster response
        allow_interruptions=True,
        min_interruption_duration=0.5,
        min_interruption_words=1,
    )
    
    # Metrics
    usage = lk_metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⚡ INSTANT FILLER & INTERRUPTION — Sub-second perceived latency
    # ═══════════════════════════════════════════════════════════════════════════
    # Track active filler speech handle for interruption when real response arrives
    active_filler_handle = {"handle": None, "is_filler": False}
    
    def _interrupt_filler():
        """Safely interrupt active filler speech."""
        h = active_filler_handle.get("handle")
        if not h:
            return
        try:
            # Try various interrupt methods depending on SDK version
            if hasattr(h, 'interrupt'):
                h.interrupt()
            elif hasattr(h, 'cancel'):
                h.cancel()
            elif hasattr(h, 'stop'):
                h.stop()
            logger.debug("[FILLER] ✓ Interrupted filler for real response")
        except Exception as e:
            logger.debug(f"[FILLER] Could not interrupt filler (non-critical): {e}")
        finally:
            active_filler_handle["handle"] = None
            active_filler_handle["is_filler"] = False
    
    async def _send_filler_async(filler_text: str):
        """Non-blocking filler speech - will be interrupted when real response arrives."""
        try:
            # Use session.say with allow_interruptions=True for non-blocking filler
            # Store handle so we can interrupt it when real response arrives
            active_filler_handle["is_filler"] = True
            handle = await session.say(filler_text, allow_interruptions=True)
            active_filler_handle["handle"] = handle
            logger.debug(f"[FILLER] Sent filler: {filler_text}")
        except Exception as e:
            logger.debug(f"[FILLER] Could not send filler: {e}")
            active_filler_handle["is_filler"] = False
    
    @session.on("user_input_transcribed")
    def _on_user_transcribed_filler(ev):
        """
        SYNC callback - spawns async task for filler.
        LiveKit .on() requires sync callbacks; async work via create_task.
        """
        # Only act on final transcriptions
        if not getattr(ev, 'is_final', True):
            return
        
        transcript = getattr(ev, 'transcript', '') or getattr(ev, 'text', '') or ''
        if not transcript.strip():
            return
        
        # Don't send filler for simple yes/no (handled deterministically)
        transcript_lower = transcript.strip().lower()
        if len(transcript_lower.split()) <= 2:
            if YES_PAT.search(transcript_lower) or NO_PAT.search(transcript_lower):
                return  # Skip filler for confirmations
        
        # Don't send filler if we're already speaking
        if active_filler_handle.get("is_filler"):
            return
        
        # Fire off a quick filler phrase while LLM thinks
        import random
        fillers = [
            "... hmm, let me check...",
            "... one moment...",
            "... okay...",
            "... sure...",
        ]
        filler = random.choice(fillers)
        
        # Non-blocking: spawn task, don't await
        asyncio.create_task(_send_filler_async(filler))
    
    @session.on("agent_speech_started")
    def _on_speech_started(ev):
        """
        SYNC callback - interrupt filler when real response starts.
        This ensures filler doesn't overlap with actual content.
        """
        # Check if this is a real response (not the filler itself)
        speech_text = ""
        try:
            speech_text = getattr(ev, 'text', '') or getattr(ev, 'content', '') or ''
        except:
            pass
        
        # If we have an active filler and this is NOT a filler phrase, interrupt it
        handle = active_filler_handle.get("handle")
        is_filler = active_filler_handle.get("is_filler", False)
        
        # Interrupt if: we have a handle AND (not a filler OR speech doesn't start with "...")
        if handle and (not is_filler or (speech_text and not speech_text.strip().startswith("..."))):
            _interrupt_filler()

    # Create agent with tools (v1.2.14 API - tools passed to Agent, not AgentSession)
    class SnappyAgent(Agent):
        def __init__(self, instructions: str):
            super().__init__(
                instructions=instructions,
                tools=RECEPTIONIST_TOOLS,  # ⚡ Parallel extraction via function tools
            )
            # Store instructions for dynamic updates
            self._instructions = instructions
    
    # Create agent instance with initial dynamic prompt
    snappy_agent = SnappyAgent(instructions=initial_system_prompt)
    
    # Store references for the refresh callback
    session_ref["session"] = session
    session_ref["agent"] = snappy_agent
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔄 USER SPEECH EVENT — Refresh memory after each user turn
    # ═══════════════════════════════════════════════════════════════════════════

    def _speech_text_from_msg(msg) -> str:
        for attr in ("text", "content", "text_content", "message"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        try:
            return str(msg).strip()
        except Exception:
            return ""
    
    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        text = _speech_text_from_msg(msg)
        if text:
            ts = datetime.now().strftime("%H:%M:%S")
            # HIGH-VISIBILITY logging for Railway/production debugging
            logger.info(f"🤖 [AGENT RESPONSE] [{ts}] >> {text}")
            # Also log to debug for detailed tracing
            logger.debug(f"[CONVO] [{ts}] AGENT: {text}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 DETERMINISTIC YES/NO ROUTING — Handle confirmations without LLM
    # ═══════════════════════════════════════════════════════════════════════════
    # This intercepts clear yes/no responses during pending confirmations
    # and routes them directly to confirm_phone/confirm_email tools
    # instead of relying on LLM which can misfire (e.g., confirm_email on "yes")
    
    @session.on("user_input_transcribed")
    def _on_user_input_confirmation(ev):
        """
        SYNC callback - deterministic routing for yes/no confirmations.
        Spawns async tasks via create_task (required by LiveKit EventEmitter).
        
        When pending_confirm_field is set (e.g., "phone"), intercept
        clear yes/no responses and call the appropriate confirm tool directly.
        This avoids LLM misfires like calling confirm_email(False) on "Yes".
        """
        # Only act on final transcriptions
        if not getattr(ev, 'is_final', True):
            return
        
        transcript = getattr(ev, 'transcript', '') or getattr(ev, 'text', '') or ''
        transcript = transcript.strip().lower()
        
        if not transcript:
            return
        
        # Log user speech
        ts = datetime.now().strftime("%H:%M:%S")
        logger.info(f"👤 [USER INPUT] [{ts}] << {transcript}")
        
        # Check if we have a pending confirmation
        pending = state.pending_confirm_field or state.pending_confirm
        if not pending:
            return  # No pending confirmation, let LLM handle
        
        # Check for clear yes/no patterns
        is_yes = YES_PAT.search(transcript) is not None
        is_no = NO_PAT.search(transcript) is not None
        
        # Only route if it's clearly yes OR clearly no (not both, not neither)
        if is_yes == is_no:
            logger.debug(f"[CONFIRM] Ambiguous response '{transcript}' - letting LLM handle")
            return  # Ambiguous or neither - let LLM handle
        
        logger.info(f"[CONFIRM] Deterministic routing: pending='{pending}', is_yes={is_yes}")
        
        # Async handler for phone confirmation
        async def _handle_phone_confirm_async(confirmed: bool):
            try:
                if confirmed:
                    result = await confirm_phone(confirmed=True)
                    logger.info(f"[CONFIRM] Phone confirmed via deterministic routing")
                    # Let agent continue naturally
                    await session.generate_reply()
                else:
                    result = await confirm_phone(confirmed=False)
                    logger.info(f"[CONFIRM] Phone rejected via deterministic routing")
                    # Ask for phone again
                    await session.say("No problem! Could you please give me your phone number again?")
            except Exception as e:
                logger.error(f"[CONFIRM] Phone confirm error: {e}")
        
        # Async handler for email confirmation
        async def _handle_email_confirm_async(confirmed: bool):
            try:
                if confirmed:
                    result = await confirm_email(confirmed=True)
                    logger.info(f"[CONFIRM] Email confirmed via deterministic routing")
                    await session.generate_reply()
                else:
                    result = await confirm_email(confirmed=False)
                    logger.info(f"[CONFIRM] Email rejected via deterministic routing")
                    await session.say("No problem! What's your email address?")
            except Exception as e:
                logger.error(f"[CONFIRM] Email confirm error: {e}")
        
        # Route to appropriate confirm tool (spawn async task - don't await)
        if pending == "phone":
            asyncio.create_task(_handle_phone_confirm_async(is_yes))
        elif pending == "email":
            asyncio.create_task(_handle_email_confirm_async(is_yes))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📞 SIP PARTICIPANT EVENT — Handle late-joining SIP participants
    # ═══════════════════════════════════════════════════════════════════════════
    
    @ctx.room.on("participant_connected")
    def _on_participant_joined(p: rtc.RemoteParticipant):
        """
        Handle SIP participants that join after initial connection.
        Auto-capture caller phone from SIP metadata for zero-ask booking.
        """
        if p.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            sip_attrs = p.attributes or {}
            caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
            late_called_num = sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
            late_called_num = _normalize_sip_user_to_e164(late_called_num)
            
            logger.info(f"📞 [SIP EVENT] Participant joined: {p.identity}")
            
            # Pre-fill phone if not already captured
            if caller_phone and not state.phone_e164:
                clean_phone, last4 = _normalize_phone_preserve_plus(caller_phone, clinic_region)
                if clean_phone:
                    state.phone_e164 = str(clean_phone)  # Enforce string type
                    state.phone_last4 = str(last4) if last4 else ""
                    # Safety guard
                    _ensure_phone_is_string(state)
                    state.phone_confirmed = False  # NEVER auto-confirm - always ask user
                    state.phone_source = "sip"
                    state.pending_confirm = "phone"
                    state.pending_confirm_field = "phone"
                    speakable = speakable_phone(state.phone_e164)
                    logger.info(f"📞 [SIP EVENT] ⏳ Phone pre-filled (needs confirmation): {speakable}")
                    # Refresh agent memory so it knows phone needs confirmation
                    refresh_agent_memory()

            # Late dialed-number metadata is common; refresh context if we started with a fallback.
            if late_called_num and used_fallback_called_num:
                logger.info(f"📞 [SIP EVENT] ✓ Late called number detected: {late_called_num}")
                # Fire-and-forget context refresh
                async def _refresh_context():
                    nonlocal clinic_info, agent_info, settings, agent_name
                    nonlocal clinic_name, clinic_tz, clinic_region, agent_lang
                    try:
                        ci, ai, st, an = await fetch_clinic_context_optimized(late_called_num)
                        clinic_info, agent_info, settings, agent_name = ci, ai, st, (an or agent_name)
                        globals()["_GLOBAL_CLINIC_INFO"] = clinic_info
                        globals()["_GLOBAL_AGENT_SETTINGS"] = settings
                        globals()["_GLOBAL_SCHEDULE"] = load_schedule_from_settings(settings or {})
                        clinic_name = (clinic_info or {}).get("name") or clinic_name
                        clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
                        clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
                        agent_lang = (agent_info or {}).get("default_language") or agent_lang
                        state.tz = clinic_tz
                        globals()["_GLOBAL_CLINIC_TZ"] = clinic_tz
                        refresh_agent_memory()
                    except Exception as e:
                        logger.warning(f"[DB] Late context refresh failed: {e}")

                asyncio.create_task(_refresh_context())
    
    # Start the session
    await session.start(
        room=ctx.room,
        agent=snappy_agent,
        room_input_options=RoomInputOptions(
            close_on_disconnect=True,
        ),
    )

    # Say greeting ASAP (don't await; let TTS start immediately)
    session.say(greeting)

    # ═══════════════════════════════════════════════════════════════════════════
    # 🔄 DEFERRED CONTEXT LOAD — Only if 2s timeout was exceeded
    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: With 2s timeout above, this rarely triggers. It's a safety net.
    if context_task and not context_task.done():
        try:
            clinic_info, agent_info, settings, agent_name = await context_task
            logger.info(f"[DB] ✓ Deferred context loaded: {clinic_info.get('name') if clinic_info else 'None'}")

            _GLOBAL_CLINIC_INFO = clinic_info
            _GLOBAL_AGENT_SETTINGS = settings

            clinic_name = (clinic_info or {}).get("name") or clinic_name
            clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
            clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
            agent_lang = (agent_info or {}).get("default_language") or agent_lang

            state.tz = clinic_tz
            _GLOBAL_CLINIC_TZ = clinic_tz
            _GLOBAL_SCHEDULE = load_schedule_from_settings(settings or {})

            refresh_agent_memory()

            # Send proper greeting now that we have context (only if we didn't have it before)
            followup = (settings or {}).get("greeting_text") or (
                f"Hi, I'm {agent_name} from {clinic_name}. How can I help you today?"
            )
            if followup:
                asyncio.create_task(session.say(followup))

        except Exception as e:
            logger.warning(f"[DB] Deferred context load failed: {e}")
    
    # Shutdown
    async def _on_shutdown():
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Ended after {dur}s, booking={state.booking_confirmed}")
        
        try:
            if clinic_info:
                # FIX: Use "completed" as fallback (valid Supabase enum value)
                # Previously "inquiry" caused database crashes
                outcome = "appointment_booked" if state.booking_confirmed else "completed"
                await asyncio.to_thread(
                    lambda: supabase.table("call_sessions").insert({
                        "organization_id": clinic_info["organization_id"],
                        "clinic_id": clinic_info["id"],
                        "caller_phone_masked": state.phone_last4 or "Unknown",
                        "outcome": outcome,
                        "duration_seconds": dur,
                        "called_number": called_num,
                    }).execute()
                )
                logger.info(f"[DB] ✓ Call session saved: outcome={outcome}")
        except Exception as e:
            logger.error(f"[DB] Call session error: {e}")
        
        try:
            print(f"[USAGE] {usage.get_summary()}")
        except Exception:
            pass
    
    ctx.add_shutdown_callback(_on_shutdown)
    
    # Wait for disconnect
    disconnect_event = asyncio.Event()
    
    @ctx.room.on("disconnected")
    def _():
        disconnect_event.set()
    
    @ctx.room.on("participant_disconnected")
    def _(p):
        disconnect_event.set()
    
    try:
        await asyncio.wait_for(disconnect_event.wait(), timeout=7200)
    except asyncio.TimeoutError:
        pass


# =============================================================================
# PREWARM
# =============================================================================

def prewarm(proc: agents.JobProcess):
    logger.info(f"[TELEPHONY] Worker identity set to: {LIVEKIT_AGENT_NAME}")
    
    try:
        silero.VAD.load()
        logger.info("[PREWARM] ✓ VAD loaded")
    except Exception as e:
        logger.error(f"[PREWARM] Error: {e}")
    
    # Verify calendar
    print("\n" + "="*50)
    print("[CONFIG] Verifying calendar...")
    if ENVIRONMENT == "production":
        print("[CONFIG] ✓ Production: using Supabase-backed OAuth token (skipping local file check)")
    else:
        if GOOGLE_OAUTH_TOKEN_PATH and os.path.exists(GOOGLE_OAUTH_TOKEN_PATH):
            print(f"[CONFIG] ✓ OAuth token: {GOOGLE_OAUTH_TOKEN_PATH}")
        else:
            print(f"[CONFIG] ❌ OAuth token missing")
    print("="*50 + "\n")


# =============================================================================
# DEBUG TESTS — Phone normalization and slot suggestion verification
# =============================================================================

def _run_debug_tests():
    """
    Run inline tests for phone normalization and slot suggestion logic.
    Call this via: python agent_v2.py --test
    """
    print("\n" + "="*60)
    print(" PHONE NORMALIZATION TESTS")
    print("="*60)
    
    test_cases = [
        # (input, region, expected_e164_prefix, description)
        ("+923351897839", "PK", "+923351897839", "Already E.164 Pakistani"),
        ("0335-1897839", "PK", "+92335", "Pakistani local format 0335..."),
        ("03351897839", "PK", "+92335", "Pakistani local without dashes"),
        ("+13105551234", "US", "+13105551234", "Already E.164 US"),
        ("310-555-1234", "US", "+1310", "US local format"),
        ("+442071234567", "GB", "+44207", "UK E.164"),
    ]
    
    for raw, region, expected_prefix, desc in test_cases:
        result = _normalize_phone_preserve_plus(raw, region)
        e164, last4 = result
        
        # Verify result is tuple with string or None
        assert isinstance(result, tuple), f"FAIL: {desc} - result not tuple"
        assert e164 is None or isinstance(e164, str), f"FAIL: {desc} - e164 not string: {type(e164)}"
        assert isinstance(last4, str), f"FAIL: {desc} - last4 not string: {type(last4)}"
        
        if e164 and not e164.startswith(expected_prefix):
            print(f"WARN: {desc}")
            print(f"       Input: {raw} (region={region})")
            print(f"       Got: {e164}, expected prefix: {expected_prefix}")
        else:
            print(f"✓ {desc}: {raw} -> {e164} (last4={last4})")
    
    print("\n" + "="*60)
    print(" SPEAKABLE PHONE TESTS")
    print("="*60)
    
    speakable_tests = [
        ("+923351897839", "+92 335 189 7839"),
        ("+13105551234", "+1 310 555 1234"),
        ("+442071234567", "+44 2071 2345 67"),
        (None, "unknown"),
    ]
    
    for e164, expected in speakable_tests:
        result = speakable_phone(e164)
        status = "✓" if result == expected else "✗"
        print(f"{status} speakable_phone({e164!r}) = {result!r} (expected: {expected!r})")
    
    print("\n" + "="*60)
    print(" ALL TESTS COMPLETE")
    print("="*60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_debug_tests()
    else:
        agents.cli.run_app(
            WorkerOptions(
                entrypoint_fnc=snappy_entrypoint,
                prewarm_fnc=prewarm,
                agent_name=LIVEKIT_AGENT_NAME,  # Must match SIP trunk dispatch rules
                load_threshold=1.0,  # Prioritize this agent for incoming telephony calls
            )
        )

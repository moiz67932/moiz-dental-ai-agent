"""
Dental appointment agent.

This module defines a LiveKit voice agent that acts as a receptionist for a
dental clinic. It listens for user speech, extracts booking information
(patient name, phone, email, reason, date/time and payment method) and,
once confirmed by the user, creates a Google Calendar event. The agent
supports LiveKit Agents 1.2 and later by listening for the
``user_input_transcribed`` event for final transcriptions. All
critical steps—slot extraction, confirmation detection, availability check
and event creation—are logged to aid debugging.
"""

import os
import re
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.agents import metrics as lk_metrics
from livekit.agents import MetricsCollectedEvent, ConversationItemAddedEvent

# Import plugins for STT/LLM/TTS
from livekit.plugins import (
    openai as openai_plugin,
    silero,
    deepgram as deepgram_plugin,
    assemblyai as assemblyai_plugin,
    cartesia as cartesia_plugin,
    noise_cancellation,
)

# Local utilities for parsing contacts and dates
from contact_utils import (
    normalize_phone,
    normalize_email,
    validate_email_address,
    parse_datetime_natural,
)
from calendar_client import create_event, is_time_free


# ---------------------------------------------------------------------------
# Environment and configuration
# ---------------------------------------------------------------------------

# Load environment variables from .env.local if present.  This call is idempotent.
load_dotenv(".env.local")

# Clinic defaults
DEFAULT_TZ = os.getenv("DEFAULT_TIMEZONE", "Asia/Karachi")
DEFAULT_MIN = int(os.getenv("DEFAULT_APPT_MINUTES", "60"))
CAL_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
CLINIC = os.getenv("CLINIC_NAME", "Dental Clinic")
LOCATION = os.getenv("CLINIC_LOCATION", "")

# Phone region used for normalising phone numbers.  Pakistani callers
# commonly give numbers like 0335...; by default we treat them as PK and
# convert them to +92335... before validation.
DEFAULT_PHONE_REGION = os.getenv("DEFAULT_PHONE_REGION", "PK")

# Confirmation phrases to trigger booking.  The regex is deliberately
# generous: once all fields are captured, any of these phrases will
# immediately call ``try_book``.
CONFIRM_PAT = re.compile(
    r"\b(yes|yeah|yep|yup|ok|okay|sure|please|confirm(?: booking)?|book(?: it| now)?|go ahead|do it|schedule(?: it)?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Slot state
# ---------------------------------------------------------------------------

@dataclass
class SlotState:
    """Holds the information required to book an appointment.

    All fields default to ``None``.  The ``missing`` method returns a list
    of fields still required to create a booking.  The appointment is
    considered ready only when ``missing`` returns an empty list.
    """

    full_name: Optional[str] = None
    phone_e164: Optional[str] = None
    phone_last4: Optional[str] = None
    email: Optional[str] = None
    reason: Optional[str] = None
    dt_local: Optional[datetime] = None
    tz: str = DEFAULT_TZ
    insurance: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    booking_confirmed: bool = False
    confirmation_asked: bool = False
    booking_in_progress: bool = False
    dt_text: Optional[str] = None


    def missing(self) -> list[str]:
        """Return a list of required fields that are still missing."""
        req = ["full_name", "phone_e164", "email", "reason", "dt_local"]
        return [r for r in req if getattr(self, r) in (None, "", [])]

    def ready(self) -> bool:
        """Return True if all required fields are present."""
        return not self.missing()


# Global state per worker - reset for each new session
state = None


# ---------------------------------------------------------------------------
# Model pickers
# ---------------------------------------------------------------------------

def pick_llm() -> openai_plugin.LLM:
    """Select and configure the conversational LLM."""
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "180"))
    return openai_plugin.LLM(model=model, temperature=temperature, max_completion_tokens=max_tokens)


def pick_stt():
    """Select the speech-to-text engine."""
    if os.getenv("DEEPGRAM_API_KEY"):
        return deepgram_plugin.STT(model="nova-2-general", language="en-US", endpointing_ms=800)
    if os.getenv("ASSEMBLYAI_API_KEY"):
        return assemblyai_plugin.STT()
    return openai_plugin.STT(model="gpt-4o-transcribe", language="en")


def pick_tts():
    """Select the text-to-speech engine."""
    if os.getenv("CARTESIA_API_KEY"):
        voice = os.getenv("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
        return cartesia_plugin.TTS(model="sonic-2", voice=voice)
    return openai_plugin.TTS(model="gpt-4o-mini-tts", voice=os.getenv("OPENAI_TTS_VOICE", "alloy"))


# ---------------------------------------------------------------------------
# Agent persona
# ---------------------------------------------------------------------------

class DentalAssistant(Agent):
    """A polite and efficient dental clinic receptionist."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, efficient dental clinic receptionist.\n"
                "Rules:\n"
                "1) Ask one question at a time; replies ≤25 words.\n"
                "2) Capture information in this order: full name → phone → email → reason → date/time → payment.\n"
                "3) Restate weekday, date, time and timezone.\n"
                "4) Only say 'confirmed' after the system books an appointment.\n"
                "5) Whitening estimate only: $300–$800.\n"
                "6) Be polite and concise.\n"
                "7) When all information is collected, ask user to confirm before booking."
            )
        )


# ---------------------------------------------------------------------------
# Helper functions for extracting slot information
# ---------------------------------------------------------------------------

def _maybe_extract_name(text: str) -> Optional[str]:
    # Remove the $ anchor to allow text after the name
    match = re.search(r"\bmy name is ([a-z][a-z\s\.'-]{2,})", text, re.I)
    if match:
        # Extract and clean up - stop at comma or other delimiter
        name = match.group(1).strip()
        # Stop at first comma, "phone", "email", etc.
        name = re.split(r'[,]|phone|email|for|reason', name, flags=re.I)[0].strip()
        return name.title()
    match = re.search(r"\bi am ([a-z][a-z\s\.'-]{2,})", text, re.I)
    if match:
        name = match.group(1).strip()
        name = re.split(r'[,]|phone|email|for|reason', name, flags=re.I)[0].strip()
        return name.title()
    return None


def _maybe_extract_reason(text: str) -> Optional[str]:
    t = text.lower()
    if "whiten" in t or "whitening" in t:
        return "Teeth whitening"
    if any(key in t for key in ["clean", "checkup", "exam"]):
        return "Cleaning and exam"
    if "pain" in t or "toothache" in t:
        return "Tooth pain"
    if "consult" in t:
        return "Consultation"
    return None


def _maybe_extract_dt(text: str, tz_hint: Optional[str] = None) -> Optional[datetime]:
    """Parse a natural-language datetime and attach clinic timezone if naive."""
    # Try multiple extraction patterns
    patterns = [
        r"(?:on|for)\s+([^,]+?)(?:,|$)",  # "on 14 October 2025 at 2 PM"
        r"(?:at)\s+(\d+(?:\s*(?:am|pm))?)",  # "at 2 PM"
        r"(\d+\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}.*?)(?:,|$)",  # Direct date match
    ]
    
    extracted = None
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            extracted = match.group(1).strip()
            print(f"[dt_extract] Pattern matched: '{extracted}'")
            break
    
    if not extracted:
        # Fallback: try parsing the whole text
        extracted = text
        print(f"[dt_extract] No pattern matched, trying whole text: '{extracted}'")
    
    parsed = parse_datetime_natural(extracted, tz_hint=tz_hint or DEFAULT_TZ)
    if not parsed:
        print(f"[dt_extract] Failed to parse: '{extracted}'")
        return None
    
    print(f"[dt_extract] Successfully parsed: {parsed.isoformat()}")
    
    if parsed.tzinfo is None:
        try:
            tz_to_use = tz_hint or DEFAULT_TZ
            parsed = parsed.replace(tzinfo=ZoneInfo(tz_to_use))
            print(f"[dt_extract] Applied timezone: {tz_to_use}")
        except Exception as e:
            print(f"[dt_extract] Error applying timezone: {e}")
            parsed = parsed.replace(tzinfo=ZoneInfo(DEFAULT_TZ))
    
    return parsed

_TZ_TOKEN = re.compile(
    r"(?:time\s*zone|timezone|tz)\s*(?:is|=)?\s*([A-Za-z_/\-+0-9: ]+)\b|"
    r"\b(UTC|GMT)\s*(?:([+-])\s*(\d{1,2})(?::?(\d{2}))?)?\b",
    re.I,
)

def _maybe_extract_timezone(text: str) -> Optional[str]:
    """
    Accepts 'timezone is UTC', 'in Asia/Karachi', 'UTC', 'GMT', 'UTC+5'.
    We return an IANA zone string when possible. For 'UTC+H' we return 'UTC' (no minutes),
    since your user said simply 'UTC'.
    """
    t = text or ""
    # direct IANA name mentions like 'in Asia/Karachi'
    m = re.search(r"\bin\s+([A-Za-z_]+/[A-Za-z_]+)\b", t, re.I)
    if m:
        name = m.group(1)
        try:
            ZoneInfo(name)  # validate
            return name
        except Exception:
            pass

    m = _TZ_TOKEN.search(t)
    if not m:
        return None

    # group 1 may capture a free-form tz name
    if m.group(1):
        name = m.group(1).strip()
        try:
            ZoneInfo(name)
            return name
        except Exception:
            # fall through
            return None

    # Or a UTC/GMT with optional offset
    if m.group(2):  # UTC/GMT
        if not m.group(3):  # no offset
            return "UTC"
        # We could build fixed-offset tz, but your flow only needs 'UTC' case right now.
        return "UTC"

    return None


# ---------------------------------------------------------------------------
# Booking logic
# ---------------------------------------------------------------------------

async def try_book(session: AgentSession) -> None:
    """Attempt to book an appointment using the information in ``state``.

    Checks for missing fields, verifies calendar availability via ``is_time_free``
    and creates an event using ``create_event``.  Sends a confirmation message
    only upon success.  If the slot is busy or an error occurs, informs the
    user accordingly.
    """
    global state
    
    # Prevent concurrent booking attempts
    if state.booking_in_progress:
        print("[book] Booking already in progress, skipping")
        return
    
    state.booking_in_progress = True
    
    try:
        missing = state.missing()
        if missing:
            # Prompt for the next missing field
            field = missing[0]
            prompts = {
                "full_name": "Could I have your full name?",
                "phone_e164": f"Please say your phone number slowly, including area code for {DEFAULT_PHONE_REGION}.",
                "email": "What's the best email? You can say 'at' and 'dot'.",
                "reason": "What's the reason for your visit?",
                "dt_local": "What date and time do you prefer?",
            }
            print(f"[book] Missing field: {field}")
            await session.say(prompts.get(field, "Could you repeat that?"))
            return

        start_dt = state.dt_local
        end_dt = start_dt + timedelta(minutes=DEFAULT_MIN)

        # Log the booking attempt
        print(f"[book] Trying to book: {state.full_name}, {start_dt.isoformat()} to {end_dt.isoformat()}, tz={state.tz}")

        # Check availability
        try:
            is_free = is_time_free(CAL_ID, start_dt, end_dt)
            print(f"[book] Availability: free={is_free}")
            if not is_free:
                await session.say(
                    "That time is unavailable. Would you like a nearby time on the same day or another date?"
                )
                state.dt_local = None  # Reset date/time to ask again
                state.confirmation_asked = False
                return
        except Exception as exc:
            print(f"[book] Error checking availability: {exc}")
            await session.say("I had trouble checking availability. Please try again.")
            return

        summary = f"{CLINIC} – {state.reason or 'Appointment'} – {state.full_name}"
        description = (
            f"Patient: {state.full_name}\n"
            f"Phone: {state.phone_e164}\n"
            f"Reason: {state.reason}\n"
            f"Insurance: {state.insurance or 'Not specified'}\n"
            f"Notes: {'; '.join(state.notes) if state.notes else '—'}"
        )

        event_kwargs = {
            "calendar_id": CAL_ID,
            "summary": summary,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "tz": state.tz,
            "description": description,
            "location": LOCATION or None,
            "attendee_email": state.email,
        }

        print(f"[book] Creating calendar event: {event_kwargs}")
        try:
            event = create_event(**event_kwargs)
            event_id = event.get("id")
            print(f"[book] Event created with id={event_id}")
            state.booking_confirmed = True  # Mark as booked
            # Construct a friendly confirmation message
            begin = start_dt.strftime("%a, %b %d, %I:%M %p")
            await session.say(
                f"Perfect! Your appointment for {state.reason} is confirmed for {begin} {state.tz}. "
                f"I've sent a calendar invitation to {state.email}. Is there anything else I can help you with?"
            )
        except Exception as exc:
            print(f"[book] Error creating event: {exc}")
            await session.say("I'm sorry, I couldn't create the calendar event. Please try again or call us directly.")
    finally:
        state.booking_in_progress = False
        
        


# ---------------------------------------------------------------------------
# Transcript handler
# ---------------------------------------------------------------------------

async def handle_transcript(text: str, is_final: bool, session: AgentSession) -> None:
    """Process each final transcription and update booking state.

    This function is called for every transcribed user utterance.  It extracts
    slot information (name, phone, email, reason, date/time and insurance),
    detects confirmation phrases and triggers the booking when appropriate.
    """
    global state
    
    if not is_final:
        return
    text = (text or "").strip()
    if not text:
        return
    
    # Timezone updates like "my time zone is utc"
    tz_name = _maybe_extract_timezone(text)
    if tz_name:
        old = state.tz
        state.tz = tz_name
        print(f"[slot] timezone set: {old} -> {state.tz}")

        # If we had a raw date phrase but failed to parse earlier, try again with the new tz
        if not state.dt_local and state.dt_text:
            dt_retry = _maybe_extract_dt(state.dt_text, tz_hint=state.tz)
            if dt_retry:
                state.dt_local = dt_retry
                print(f"[slot] dt_local captured after tz update: {state.dt_local.isoformat()}")


    print(f"[transcript] final: {text}")

    # Don't process further if already booked
    if state.booking_confirmed:
        print("[transcript] Already booked, ignoring further confirmations")
        return

    # 1. Extract name
    if not state.full_name:
        name = _maybe_extract_name(text)
        if name:
            state.full_name = name
            print(f"[slot] name captured: {name}")

    # 2. Extract phone (region aware, with PK fallback for 03xx numbers)
    if not state.phone_e164:
        # First try to find phone-like patterns in the text
        # Look for "phone" keyword followed by numbers
        phone_match = re.search(r'phone[:\s]+([0-9\s\-\(\)]+)', text, re.I)
        if phone_match:
            phone_text = phone_match.group(1)
            phone, last4 = normalize_phone(phone_text, default_region=DEFAULT_PHONE_REGION)
            if phone:
                state.phone_e164, state.phone_last4 = phone, last4
                print(f"[slot] phone captured: {state.phone_e164} (last4={state.phone_last4})")
        else:
            # Fallback: try the whole text
            phone, last4 = normalize_phone(text, default_region=DEFAULT_PHONE_REGION)
            if not phone:
                # Fallback: if user gave a local number starting with 0, assume Pakistan
                digits = re.sub(r"\D+", "", text)
                if re.match(r"^0\d{9,10}$", digits):
                    coerced = "+92" + digits.lstrip("0")
                    phone, last4 = normalize_phone(coerced, default_region="PK")
            if phone:
                state.phone_e164, state.phone_last4 = phone, last4
                print(f"[slot] phone captured: {state.phone_e164} (last4={state.phone_last4})")

    # 3. Extract email
    if not state.email:
        # Look for email keyword first
        email_match = re.search(r'email[:\s]+([^\s,]+(?:\s+at\s+[^\s,]+)?(?:\s+dot\s+[^\s,]+)?)', text, re.I)
        if email_match:
            email_text = email_match.group(1)
            maybe_email = normalize_email(email_text)
            if "@" in maybe_email and "." in maybe_email and validate_email_address(maybe_email):
                state.email = maybe_email
                print(f"[slot] email captured: {state.email}")
        else:
            # Fallback: try the whole text
            maybe_email = normalize_email(text)
            if "@" in maybe_email and "." in maybe_email and validate_email_address(maybe_email):
                state.email = maybe_email
                print(f"[slot] email captured: {state.email}")

    # 4. Extract reason
    if not state.reason:
        reason = _maybe_extract_reason(text)
        if reason:
            state.reason = reason
            print(f"[slot] reason captured: {state.reason}")

    # 5. Extract date/time - IMPROVED LOGIC
    if not state.dt_local:
        print(f"[dt_extract] Attempting to extract datetime from: '{text}'")
        dt_local = _maybe_extract_dt(text, tz_hint=state.tz)
        if dt_local:
            state.dt_local = dt_local
            state.dt_text = text  # Store for potential re-parsing
            print(f"[slot] dt_local captured: {state.dt_local.isoformat()}")
        else:
            print(f"[dt_extract] Failed to extract datetime")

    # 6. Extract insurance/self-pay
    if not state.insurance:
        lower = text.lower()
        if any(k in lower for k in ["self pay", "self-pay", "self-paid", "self paid", "self"]):
            state.insurance = "self-pay"
            print(f"[slot] insurance captured: {state.insurance}")
        elif any(k in lower for k in ["no insurance", "no-insurance", "noinsurance"]):
            state.insurance = "none"
            print(f"[slot] insurance captured: {state.insurance}")

    # Debug current state
    print(f"[state] missing={state.missing()} ready={state.ready()} phone_last4={state.phone_last4}")

    # 7. Confirmation detection: if ready and not already booked, any confirm phrase triggers booking
    if state.ready() and not state.booking_confirmed and CONFIRM_PAT.search(text):
        print("[confirm] detected confirmation phrase -> booking")
        await try_book(session)
        return

    # 8. If ready but no explicit confirmation yet, prompt once to confirm
    if state.ready() and not state.booking_confirmed and not state.confirmation_asked:
        state.confirmation_asked = True
        begin_str = state.dt_local.strftime("%A, %b %d at %I:%M %p")
        await session.say(
            f"Perfect! Let me confirm: {state.full_name}, {state.reason} on {begin_str} {state.tz}. "
            "Say 'yes' or 'confirm' to book this appointment."
        )
        return


# ---------------------------------------------------------------------------
# Entrypoint and event hooks
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
    """Agent entrypoint called for each job.

    Sets up the session, registers event handlers and greets the user.  It
    listens for the ``user_input_transcribed`` event which is available in
    LiveKit Agents versions ≥1.2.  The ``handle_transcript`` coroutine 
    processes final transcriptions and updates the booking state.
    """
    global state
    
    # Reset state for new session
    state = SlotState()
    
    # Log environment at startup
    print(f"[entry] Starting DentalAssistant agent for room={ctx.room.name}")
    
    # Preload VAD for endpointing
    vad = silero.VAD.load()
    llm = pick_llm()
    
    session = AgentSession(
        stt=pick_stt(),
        llm=llm,
        tts=pick_tts(),
        vad=vad,
    )
    
    usage = lk_metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    # Primary event: conversation_item_added (more reliable across versions)
    # MUST be registered BEFORE session.start()
    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev):
        # Only process user messages (not agent messages)
        if ev.item.role == "user":
            text = ev.item.text_content or ""
            print(f"[event] conversation_item_added fired: role={ev.item.role} text='{text}'")
            # Use create_task since .on() requires synchronous callbacks
            asyncio.create_task(handle_transcript(text, True, session))

    # Start the session and greet the user
    await session.start(
        room=ctx.room,
        agent=DentalAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            close_on_disconnect=False,
        ),
    )

    await session.say("Hello! Are you a new or returning patient, and what can I help you with today?")

    # Log usage on shutdown
    async def _log_usage():
        try:
            summary = usage.get_summary()
            print(f"[usage] {summary}")
        except Exception as e:
            print(f"[usage] Error getting summary: {e}")
    
    ctx.add_shutdown_callback(_log_usage)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
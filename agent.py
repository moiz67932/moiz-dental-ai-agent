"""
agent.py — Dental Voice AI Agent (Latency-Optimized, Instrumented)

Target latency:
  Simple turns:           500–900 ms user-EOU → first audio
  Booking happy path:     800–1500 ms
  Conflict/alt-slot:      1200–2200 ms

Latency wins (cumulative):
1. Prompt compressed from ~3600 → ~400 tokens (saves 1-2s LLM latency)
2. Clinic FAQ inline in prompt instead of RAG tool call
3. Google Calendar removed — booking is Supabase-only
4. DB timeout reduced from 5s to 2s; greeting starts immediately
5. Bloat removed (LatencyMetrics, RoomGuard, CallLogger, UsageCollector)
6. [NEW] Pattern B upgraded: phone confirmed → state complete → direct book,
   0 extra LLM hops (was 2 extra LLM roundtrips on the happy path)
7. [NEW] refresh_agent_memory() no longer called on every utterance — only
   called after tool writes that actually change state
8. [NEW] Filler debounce reduced 250ms → FILLER_DEBOUNCE_MS (120ms default)
   so fillers fire earlier and cover more of the LLM gap
9. [NEW] AgentSession gets min/max_endpointing_delay from config (was imported
   but NEVER actually passed to AgentSession — now it is)
10.[NEW] Comprehensive per-turn TurnTimer with structured log output

Preserved patterns (proven latency wins):
A. Filler phrases with smart suppression (hides tool-call latency from user)
B. Deterministic yes/no routing (bypasses LLM for confirmations)
C. Deferred DB loading with 2s timeout (greeting starts immediately)
D. Non-blocking booking (fire-and-forget Supabase writes, confirmed = instant)
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Awaitable, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv(".env.local")

# =============================================================================
# LiveKit imports
# =============================================================================

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    llm,
    metrics as lk_metrics,
    AgentSession,
    Agent,
    room_io,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.rtc import ParticipantKind
from livekit.plugins import silero

try:
    from livekit.plugins import noise_cancellation
    NC_AVAILABLE = True
except ImportError:
    noise_cancellation = None
    NC_AVAILABLE = False

# =============================================================================
# Local imports
# =============================================================================

from config import (
    logger,
    supabase,
    DEFAULT_TZ,
    BOOKING_TZ,
    DEFAULT_PHONE_REGION,
    DEMO_CLINIC_ID,
    FILLER_ENABLED,
    FILLER_MAX_DURATION_MS,
    FILLER_DEBOUNCE_MS,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    MIN_ENDPOINTING_DELAY,
    MAX_ENDPOINTING_DELAY,
    STT_AGGRESSIVE_ENDPOINTING,
    LATENCY_DEBUG,
    TURN_TRACKER_ENABLED,
    DETERMINISTIC_FAST_PATH_ENABLED,
    TURN_SHORT_PAUSE_MS,
    TURN_CONTINUATION_WAIT_MS,
    TURN_LOW_CONFIDENCE_THRESHOLD,
    LOOKUP_FILLER_DELAY_MS,
    EXPECTED_SLOT_CONTINUATION_WAIT_MS,
    EXPECTED_SLOT_WEAK_FRAGMENT_MAX_TOKENS,
    EXPECTED_SLOT_ENABLE_DATE_TIME_FAST_PATH,
)
from models.state import PatientState, YES_PAT, NO_PAT
from pipelines.pipeline_config import get_pipeline_components
from pipelines.urdu_prompt import URDU_SYSTEM_PROMPT
from services.database_service import fetch_clinic_context_optimized
from services.scheduling_service import load_schedule_from_settings, get_duration_for_service
from services.extraction_service import extract_name_quick, extract_reason_quick
from utils.turn_taking import (
    CompletionLabel,
    ExpectedUserSlot,
    PolicyAction,
    StreamingTurnTracker,
    TurnTakingConfig,
    build_policy_decision,
    format_policy_log,
    format_tracker_log,
    preview_turn,
    strip_duplicate_acknowledgement,
)
from utils.agent_flow import (
    has_date_reference,
    has_time_reference,
    looks_like_delivery_follow_up_fragment,
    normalize_patient_name,
    resolve_confirmation_intent,
    resolve_delivery_preference,
    store_detected_phone,
    time_expression_score,
    user_declined_anything_else,
    user_said_goodbye,
)
from tools.assistant_tools import AssistantTools, _delivery_question_text, update_global_clinic_info

# =============================================================================
# Per-turn latency instrumentation
# =============================================================================

import itertools as _itertools
_TURN_COUNTER = _itertools.count(1)


class TurnTimer:
    """
    Lightweight per-turn latency tracker.

    Usage:
        t = TurnTimer()
        t.mark("user_eou")
        t.mark("stt_final")
        t.mark("filler_sent")
        t.mark("llm_response")
        t.mark("speech_started")
        t.mark("speech_committed")
        t.log_summary(user_text)

    All times are in milliseconds relative to the first mark.
    """

    def __init__(self):
        self.turn_id = next(_TURN_COUNTER)
        self._marks: dict[str, float] = {}
        self._t0: float = 0.0

    def mark(self, label: str) -> float:
        """Record timestamp for label. Returns elapsed ms since t0 (or 0 for first mark)."""
        now = time.perf_counter()
        if not self._marks:
            self._t0 = now
        self._marks[label] = now
        return (now - self._t0) * 1000 if self._marks else 0.0

    def elapsed(self, from_label: str, to_label: str) -> Optional[float]:
        """Return ms between two marks. None if either mark is missing."""
        t_from = self._marks.get(from_label)
        t_to = self._marks.get(to_label)
        if t_from is None or t_to is None:
            return None
        return (t_to - t_from) * 1000

    def elapsed_since_start(self, label: str) -> Optional[float]:
        t = self._marks.get(label)
        if t is None or not self._t0:
            return None
        return (t - self._t0) * 1000

    def log_summary(self, user_text: str = "") -> None:
        """Emit one structured log line with all computed deltas."""
        def _ms(a: str, b: str) -> str:
            v = self.elapsed(a, b)
            return f"{v:.0f}ms" if v is not None else "–"

        parts = [
            f"turn={self.turn_id}",
            f"eou→stt={_ms('user_eou', 'stt_final')}",
            f"eou→filler={_ms('user_eou', 'filler_sent')}",
            f"eou→speech={_ms('user_eou', 'speech_started')}",
            f"eou→commit={_ms('user_eou', 'speech_committed')}",
            f"eou→direct={_ms('user_eou', 'direct_say')}",
        ]
        # Include tool time if present
        tool_ms = self.elapsed("tool_start", "tool_end")
        if tool_ms is not None:
            parts.append(f"tool={tool_ms:.0f}ms")

        snippet = (user_text or "")[:60].replace("\n", " ")
        logger.info(f"[TURN METRICS] {' | '.join(parts)} | user='{snippet}'")


# Global turn timer — replaced per user utterance
_current_turn: Optional[TurnTimer] = None


# =============================================================================
# Compressed system prompt — ~400 tokens (down from ~3600)
# =============================================================================

SYSTEM_PROMPT = """You are {agent_name}, receptionist for {clinic_name}. Respond ONLY in English.
Date: {current_date} | Time: {current_time} | Timezone: {timezone}
Hours: Mon-Fri 9am-5pm, Sat 10am-2pm, Sun closed. Lunch 1-2pm unavailable.

PATIENT STATE (YOUR MEMORY — TRUST THIS):
{state_summary}
Fields marked [done] are saved — NEVER re-ask. Fields marked [need] are missing — collect naturally.

CLINIC INFO:
{clinic_context}

WORKFLOW — 1 question at a time, 1-2 sentences max:
1. Greet warmly, ask what they need.
2. Name missing -> call update_patient_record(name=...).
3. Reason missing -> call update_patient_record(reason=...).
4. Time -> call update_patient_record(time_suggestion="...") with natural language like "tomorrow at 2pm".
   - If slot is taken, the tool returns alternatives — offer them immediately.
   - If user says a month without a day (e.g. "February at 2pm") -> ask which day.
5. After name+reason+time captured: ask "Can I use the number you're calling from for your appointment confirmation and reminders?"
   - "yes" / "sure" / similar -> call confirm_phone(confirmed=True) IMMEDIATELY. Do not ask again.
   - "no" or gives different number -> call update_patient_record(phone=...).
6. All required fields captured -> call confirm_and_book_appointment IMMEDIATELY. Don't ask "shall I book?".
7. Read the booking confirmation EXACTLY as the tool returns it. Do not rephrase.
8. If the booking message asks WhatsApp or SMS, ask that exact question and WAIT for the caller's answer.
9. After delivery preference is settled, ask "Is there anything else I can help you with today?"
10. Only after the caller is done, give a brief closing and end the call.

RULES:
- Call update_patient_record IMMEDIATELY when you hear any info. Never wait.
- Normalize spoken input before saving: "three one zero" -> "310", "at gmail dot com" -> "@gmail.com".
- Once caller ID is confirmed, refer to it as "the number you're calling from", "this number", or "your number" — do not repeat the full digits unless the caller asks.
- When asking to confirm caller ID, phrase it naturally around appointment confirmations, booking updates, or reminders.
- CRITICAL PERSPECTIVE RULE: You are the AGENT. The CALLER is on the other end. NEVER say "I'm calling from" or "the number I'm calling from" — that is the caller's perspective. Always say "the number YOU'RE calling from" or "this number".
- NEVER parrot back the caller's own phrasing when it creates a perspective inversion. If the caller says 'use the number I'm calling from', you respond 'Perfect, I'll use this number for your confirmation and reminders.'
- Never say "booked" until the tool confirms it.
- Never admit you are AI — say "I'm the office assistant."
- Never offer callbacks (you cannot dial out).
- Keep every response to 1-2 short sentences. This is a phone call.
- If you need a tiny bridge while waiting, use only a very short acknowledgement like "Sure." or "Of course." Never pad confirmation or slot-capture turns.
- Sound warm and natural: "Of course!", "Perfect!", "Got it!" — not robotic.
- For cancel/reschedule requests: call find_existing_appointment first, confirm details with user, then act.
- For emergencies (severe pain, bleeding, swelling): express concern, direct to ER, offer follow-up booking.
- If user corrects information, update it immediately with the tool.
- After a successful booking and user confirms no more questions, call end_conversation."""

# =============================================================================
# SIP normalization helper
# =============================================================================

def _normalize_sip_user(raw: Optional[str]) -> Optional[str]:
    """Normalize SIP URI fragments to E.164 format."""
    if not raw:
        return None
    s = str(raw).strip().replace("sip:", "")
    if "@" in s:
        s = s.split("@", 1)[0]
    s = re.sub(r"[^\d+]", "", s)
    if not s:
        return None
    if s.startswith("00"):
        s = "+" + s[2:]
    if not s.startswith("+") and s.isdigit() and len(s) >= 8:
        s = "+" + s
    return s


def _normalize_phone_e164(raw: str, region: str = "PK") -> tuple[Optional[str], str]:
    """Normalize phone to E.164. Returns (e164, last4)."""
    from utils.phone_utils import _normalize_phone_preserve_plus
    result = _normalize_phone_preserve_plus(raw, region)
    if isinstance(result, tuple):
        e164, last4 = result
        return (str(e164) if e164 else None), (str(last4) if last4 else "")
    return None, ""


# =============================================================================
# Clinic FAQ fetch (replaces RAG — one query at session start)
# =============================================================================

def _normalize_knowledge_articles(rows: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(rows, list):
        return normalized
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        title_value = row.get("title")
        body_value = row.get("body")
        category_value = row.get("category")
        title = title_value.strip() if isinstance(title_value, str) else ""
        body = " ".join(body_value.split()) if isinstance(body_value, str) else ""
        category = category_value.strip() if isinstance(category_value, str) else ""
        if title or body or category:
            normalized.append({"title": title, "body": body, "category": category})
    return normalized


def _format_clinic_faq(articles: list[dict[str, str]]) -> str:
    if not articles:
        return "No additional clinic information available."

    lines: list[str] = []
    for article in articles:
        category = article.get("category", "").strip()
        title = article.get("title", "").strip()
        body = " ".join(article.get("body", "").split()[:50]).strip()
        label = f"[{category}] " if category else ""
        if title and body:
            lines.append(f"- {label}{title}: {body}")
    return "\n".join(lines) if lines else "No additional clinic information available."


async def _fetch_clinic_knowledge_articles(clinic_id: Optional[str]) -> list[dict[str, str]]:
    """
    Fetch clinic knowledge articles once and keep them in memory for deterministic
    answers to pricing, insurance, hours, parking, and other clinic questions.
    """
    if not clinic_id:
        return []
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("knowledge_articles")
            .select("title, body, category")
            .eq("clinic_id", clinic_id)
            .limit(20)
            .execute()
        )
        return _normalize_knowledge_articles(result.data)
    except Exception as e:
        logger.warning(f"[FAQ] Fetch failed: {e}")
        return []


async def _fetch_clinic_faq(clinic_id: Optional[str]) -> str:
    """
    Fetch clinic FAQ articles in one query and return as a compact text block.
    Injected into the system prompt — no tool call needed per question.
    """
    return _format_clinic_faq(await _fetch_clinic_knowledge_articles(clinic_id))


# =============================================================================
# Filler helpers (Pattern A)
# =============================================================================

SLOT_VALUE_RE = re.compile(
    r"^(?:it'?s|my name is|i'?m|this is)\s+\w+|"
    r"^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$|"
    r"^(?:\+\d{1,3}[-.\s]?)?\d{10,}$|"
    r"^\S+@\S+\.\S+$|"
    r"^(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)?|"
    r"^(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
    re.IGNORECASE,
)


FILLER_THINKING = ["One moment.", "Let me check."]
FILLER_ACKNOWLEDGE = ["Got it.", "Sure thing."]
FILLER_GENERAL = ["Okay.", "Alright."]
QUESTION_HINTS = (
    "what",
    "when",
    "where",
    "how much",
    "how long",
    "do you have",
    "is there",
    "are there",
    "can you",
    "could you",
    "would you",
    "do you take",
    "do you accept",
)
BOOKING_CAPTURE_HINTS = (
    "my name",
    "i'm",
    "i am",
    "this is",
    "it's",
    "it is",
    "i want",
    "i need",
    "book",
    "appointment",
    "schedule",
    "cleaning",
    "consultation",
    "checkup",
    "check-up",
    "whitening",
    "tooth pain",
    "filling",
    "crown",
    "root canal",
)
INCOMPLETE_CAPTURE_RE = re.compile(
    r"(?:\b(?:at|on|for|around|between|this|next)\b|\b(?:uh|um|er|hmm)\b)[\s,.;:!?-]*$",
    re.IGNORECASE,
)
GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|assalam(?:u|o)\s*alaikum)\b",
    re.IGNORECASE,
)
OPENING_REQUEST_RE = re.compile(
    r"\b("
    r"i want(?: to)?|"
    r"i need(?: to)?|"
    r"i'?d like(?: to)?|"
    r"book|"
    r"schedule|"
    r"make an appointment|"
    r"appointment"
    r")\b",
    re.IGNORECASE,
)
INTRO_RE = re.compile(
    r"\b(my name is|this is|i'?m|i am)\b",
    re.IGNORECASE,
)

MICRO_ACK_DELAY_MS = min(FILLER_DEBOUNCE_MS, 180)
MICRO_ACK_MAX_DURATION_MS = max(FILLER_MAX_DURATION_MS, 420)
AUTO_DISCONNECT_SILENCE_MS = 1400
AUTO_DISCONNECT_POST_GOODBYE_MS = 600


def _is_fragmented_turn(text: str) -> bool:
    lower = " ".join((text or "").strip().lower().split())
    if not lower:
        return False
    if INCOMPLETE_CAPTURE_RE.search(lower):
        return True
    return lower.endswith((" at", " on", " for", " around", " between", " this", " next"))


def _is_opening_request_turn(text: str, state: Optional[PatientState] = None) -> bool:
    lower = " ".join((text or "").strip().lower().split())
    if not lower:
        return False
    if has_date_reference(lower) or has_time_reference(lower):
        return False
    if "phone" in lower or "email" in lower or "number" in lower:
        return False
    first_turn = state is not None and len(getattr(state, "recent_user_texts", [])) <= 1
    has_greeting = GREETING_RE.search(lower) is not None
    has_request = OPENING_REQUEST_RE.search(lower) is not None
    has_intro = INTRO_RE.search(lower) is not None
    return first_turn and (has_greeting or has_request or (has_intro and has_request))


def _looks_like_schedule_capture_turn(text: str) -> bool:
    lower = " ".join((text or "").strip().lower().split())
    if not lower:
        return False
    if has_date_reference(lower) or has_time_reference(lower) or time_expression_score(lower) >= 1:
        return True
    if _is_fragmented_turn(lower):
        return True
    if re.search(r"\S+@\S+\.\S+", lower):
        return True
    if re.search(r"(?:\+?\d[\d\s().-]{6,})", lower):
        return True
    if lower.startswith(("my name is", "this is", "i'm", "i am")) and not OPENING_REQUEST_RE.search(lower):
        return True
    return SLOT_VALUE_RE.search(lower) is not None and not _is_opening_request_turn(lower)


def _micro_ack_decision(
    text: str,
    state: Optional[PatientState] = None,
) -> tuple[Optional[str], Optional[str]]:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return None, None

    if state is not None:
        if getattr(state, "booking_in_progress", False):
            return None, "confirmation_turn"
        if getattr(state, "appointment_booked", False) or getattr(state, "booking_confirmed", False):
            return None, "confirmation_turn"
        if getattr(state, "pending_confirm_field", None) or getattr(state, "pending_confirm", None):
            return None, "confirmation_turn"
        if getattr(state, "delivery_preference_pending", False):
            return None, "confirmation_turn"
        if getattr(state, "anything_else_pending", False):
            return None, "confirmation_turn"

    if resolve_confirmation_intent(normalized) is not None and len(normalized.split()) <= 6:
        return None, "confirmation_turn"
    if _is_fragmented_turn(normalized):
        return None, "fragmented_turn"
    if _looks_like_schedule_capture_turn(normalized):
        return None, "capture_turn"
    if _is_opening_request_turn(normalized, state=state):
        if OPENING_REQUEST_RE.search(normalized):
            return "Absolutely.", None
        if INTRO_RE.search(normalized):
            return "Got it.", None
        return "Sure.", None
    if _is_open_question(normalized):
        return "Of course.", None
    return None, None


def _choose_filler(text: str, state: Optional[PatientState] = None) -> str:
    """Deterministically pick a micro-ack based on the utterance intent."""
    ack, _ = _micro_ack_decision(text, state=state)
    return ack or FILLER_GENERAL[0]


def _is_open_question(text: str) -> bool:
    lower = " ".join((text or "").strip().lower().split())
    if not lower:
        return False
    if lower.endswith("?"):
        return True
    return any(lower.startswith(prefix) or prefix in lower for prefix in QUESTION_HINTS)


def _looks_like_capture_turn(text: str) -> bool:
    return _looks_like_schedule_capture_turn(text)


def _needs_filler(text: str, state: Optional[PatientState] = None) -> bool:
    """Preview whether policy would allow a filler for this utterance."""
    if state is not None:
        if getattr(state, "booking_in_progress", False):
            return False
        if getattr(state, "appointment_booked", False) or getattr(state, "booking_confirmed", False):
            return False
        if getattr(state, "pending_confirm_field", None) or getattr(state, "pending_confirm", None):
            return False
        if getattr(state, "delivery_preference_pending", False):
            return False
        if getattr(state, "anything_else_pending", False):
            return False

    snapshot, decision = preview_turn(
        text,
        patient_state=state or PatientState(),
        silence_ms=TURN_SHORT_PAUSE_MS,
        config=TurnTakingConfig(
            short_pause_ms=TURN_SHORT_PAUSE_MS,
            continuation_wait_ms=TURN_CONTINUATION_WAIT_MS,
            low_confidence_threshold=TURN_LOW_CONFIDENCE_THRESHOLD,
            deterministic_fast_path_enabled=DETERMINISTIC_FAST_PATH_ENABLED,
            lookup_filler_delay_ms=LOOKUP_FILLER_DELAY_MS,
            expected_slot_continuation_wait_ms=EXPECTED_SLOT_CONTINUATION_WAIT_MS,
            expected_slot_weak_fragment_max_tokens=EXPECTED_SLOT_WEAK_FRAGMENT_MAX_TOKENS,
            expected_slot_enable_date_time_fast_path=EXPECTED_SLOT_ENABLE_DATE_TIME_FAST_PATH,
        ),
    )
    if snapshot.completion_label in {CompletionLabel.INCOMPLETE, CompletionLabel.LIKELY_CONTINUING}:
        return False
    return decision.filler_text is not None


def _seed_state_from_recent_context(state: PatientState, schedule: dict[str, Any]) -> list[str]:
    """Fill obvious missing slots from recent caller context without waiting on the LLM."""
    updates: list[str] = []
    recent_context = state.recent_user_context(limit=3)
    if not recent_context:
        return updates

    if not state.full_name:
        detected_name = extract_name_quick(recent_context)
        normalized_name = normalize_patient_name(detected_name)
        if normalized_name:
            state.full_name = normalized_name
            updates.append(f"name={normalized_name}")

    if not state.reason:
        detected_reason = extract_reason_quick(recent_context)
        if detected_reason:
            state.reason = detected_reason
            state.duration_minutes = get_duration_for_service(detected_reason, schedule)
            updates.append(f"reason={detected_reason}")

    if state.full_name and state.dt_local:
        state.contact_phase_started = True

    return updates


def _build_missing_slot_prompt(state: PatientState) -> str:
    if state.appointment_booked and state.delivery_preference_pending:
        return _delivery_question_text(state)
    if state.appointment_booked and state.anything_else_pending:
        return "Is there anything else I can help you with today?"
    missing = [slot for slot in state.missing_slots() if slot != "phone_confirmed"]
    if "full_name" in missing and "reason" in missing:
        return "Perfect. What name should I put on the appointment, and what are you coming in for?"
    if "full_name" in missing:
        return "Perfect. What name should I put on the appointment?"
    if "reason" in missing:
        return "Perfect. What brings you in?"
    if "datetime" in missing:
        if state.dt_text and has_date_reference(state.dt_text) and not state.dt_local:
            return "Perfect. What time works best for you?"
        return "Perfect. What day and time would you like?"
    if "phone" in missing:
        return "What number should I use?"
    if state.appointment_booked:
        return "Thanks. I've got that noted."
    return "Perfect. Let me take care of that."


def _build_no_repeat_llm_instruction(state: PatientState, latest_user_text: str) -> Optional[str]:
    if not any([state.full_name, state.reason, state.dt_local, state.dt_text, state.appointment_booked]):
        return None

    guards: list[str] = [
        "Do not greet, welcome, or introduce yourself again. Continue from the current call state.",
    ]
    if state.appointment_booked:
        guards.append(
            "The appointment is already booked. Do not ask for the caller's name, reason, phone number, or appointment time again."
        )
        guards.append(
            "Do not say goodbye or imply the call is over unless the caller explicitly indicates they are done or says goodbye."
        )
        if state.delivery_preference_pending:
            guards.append(
                "Stay in the delivery-preference flow. Answer any brief side question, then ask exactly which delivery method they want for the confirmation."
            )
            guards.append(f"Return to this question after any brief answer: {_delivery_question_text(state)}")
        elif state.anything_else_pending:
            guards.append(
                "Stay in the post-booking follow-up flow. Answer any brief question, then ask if there is anything else you can help with today."
            )
        else:
            guards.append("Keep the booked context intact and do not restart intake.")
    else:
        if state.full_name:
            guards.append(
                f"The caller's name is already saved as {state.full_name}. Do not ask for their name again unless they explicitly correct it."
            )
        if state.reason:
            guards.append(
                f"The reason is already saved as {state.reason}. Do not ask for the reason again unless they explicitly change it."
            )
        if state.dt_local:
            dt_spoken = state.dt_local.strftime("%A, %B %d at %I:%M %p").replace(" 0", " ")
            guards.append(
                f"The appointment time is already captured as {dt_spoken}. Do not restart booking intake."
            )
        elif state.dt_text:
            guards.append(
                f"There is already date/time context saved as '{state.dt_text}'. Do not restart intake or ask for broad booking details again."
            )
        guards.append(
            f"If you need to move the booking forward, only ask the next missing piece: {_build_missing_slot_prompt(state)}"
        )

    if latest_user_text:
        guards.append("If the caller asked a direct question, answer it briefly before asking for anything else.")
    return " ".join(guards)


def _infer_expected_slot_from_response(
    *,
    route: Optional[str],
    spoken_text: str,
    state: Optional[PatientState],
) -> Optional[ExpectedUserSlot]:
    normalized = (spoken_text or "").strip().lower()

    if route in {"booking.ask_service", "booking.reask_service"}:
        return ExpectedUserSlot.SERVICE
    if route in {"booking.ask_date_time", "booking.reask_date_time"}:
        return ExpectedUserSlot.DATE_TIME
    if route in {"booking.ask_date", "booking.reask_date"}:
        return ExpectedUserSlot.DATE
    if route in {"booking.ask_time", "booking.reask_time"}:
        return ExpectedUserSlot.TIME

    if (
        "can i use the number you're calling from" in normalized
        or "is this the right number to send your confirmation to" in normalized
        or bool(state and (state.pending_confirm == "phone" or state.pending_confirm_field == "phone"))
    ):
        return ExpectedUserSlot.PHONE_CONFIRMATION

    if "what time works best" in normalized or "didn't catch the time" in normalized:
        return ExpectedUserSlot.TIME
    if "what day would you like" in normalized or "could you specify the day" in normalized:
        return ExpectedUserSlot.DATE
    if "what day and time would you like" in normalized:
        return ExpectedUserSlot.DATE_TIME

    if route == "booking.capture_date":
        if state and state.dt_text and has_date_reference(state.dt_text) and not state.dt_local:
            return ExpectedUserSlot.TIME
        return ExpectedUserSlot.DATE_TIME

    if route in {"booking.capture_time", "booking.capture_datetime"}:
        if state and state.dt_local:
            return None
        if state and state.dt_text:
            has_date = has_date_reference(state.dt_text)
            has_time = has_time_reference(state.dt_text)
            if has_date and not has_time:
                return ExpectedUserSlot.TIME
            if has_time and not has_date:
                return ExpectedUserSlot.DATE
        if route == "booking.capture_time":
            return ExpectedUserSlot.TIME
        return ExpectedUserSlot.DATE_TIME

    return None


def _caller_number_confirmation_message(state: PatientState) -> str:
    if state.using_caller_number or state.confirmed_contact_number_source == "caller_id":
        return "Perfect, I'll use this number for your confirmation and reminders."
    return "Perfect, I've noted that down."


def _build_post_phone_confirmation_prompt(state: PatientState) -> str:
    follow_up = _build_missing_slot_prompt(state).strip()
    if follow_up.startswith("Perfect. "):
        follow_up = follow_up[len("Perfect. "):]
    return f"{_caller_number_confirmation_message(state)} {follow_up}".strip()


def _final_closing_text() -> str:
    return "Wonderful. You're all set — we'll see you then. Have a great day."


def _non_booking_closing_text() -> str:
    return "Thanks for calling. Have a great day."


def _closing_text_for_state(state: PatientState) -> str:
    if state.appointment_booked or state.booking_confirmed:
        return _final_closing_text()
    return _non_booking_closing_text()


def _session_say(
    session: AgentSession,
    text: str,
    *,
    allow_interruptions: bool = True,
    add_to_chat_ctx: bool = True,
) -> Any:
    say_fn = getattr(session, "say")
    try:
        return say_fn(
            text,
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
        )
    except TypeError:
        return say_fn(text, allow_interruptions=allow_interruptions)


async def _handle_exit_intent_turn(
    *,
    text: str,
    state: PatientState,
    session: AgentSession,
    safe_say: Optional[Callable[..., Any]] = None,
    cancel_scheduled_filler: Callable[[], None],
    interrupt_filler: Callable[..., None],
    refresh_memory_async: Optional[Callable[[], Awaitable[None]]] = None,
    mark_direct_response: Optional[Callable[[], None]] = None,
    schedule_auto_disconnect: Optional[Callable[[Any], None]] = None,
) -> Literal["none", "consumed"]:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized or not user_said_goodbye(normalized):
        return "none"

    resolved_safe_say = safe_say
    if resolved_safe_say is None:
        def _fallback_safe_say(text: str, *, allow_interruptions: bool = True) -> Any:
            return _session_say(
                session,
                text,
                allow_interruptions=allow_interruptions,
            )
        resolved_safe_say = _fallback_safe_say

    cancel_scheduled_filler()
    interrupt_filler(force=True)
    state.call_ended = True
    state.anything_else_pending = False
    state.delivery_preference_pending = False
    state.user_declined_more_help = True
    state.user_goodbye_detected = True
    state.final_goodbye_sent = True
    state.closing_state = "final_goodbye_sent"

    if refresh_memory_async is not None:
        await refresh_memory_async()
    if mark_direct_response is not None:
        mark_direct_response()

    final_handle = resolved_safe_say(_closing_text_for_state(state))
    if schedule_auto_disconnect is not None:
        schedule_auto_disconnect(final_handle)
    return "consumed"


async def _handle_deterministic_confirmation_turn(
    *,
    text: str,
    state: PatientState,
    assistant_tools: AssistantTools,
    session: AgentSession,
    safe_say: Optional[Callable[..., Any]] = None,
    cancel_scheduled_filler: Callable[[], None],
    interrupt_filler: Callable[..., None],
    refresh_memory_async: Optional[Callable[[], Awaitable[None]]] = None,
    mark_direct_response: Optional[Callable[[], None]] = None,
) -> Literal["none", "consumed"]:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return "none"

    resolved_safe_say = safe_say
    if resolved_safe_say is None:
        def _fallback_safe_say(text: str, *, allow_interruptions: bool = True) -> Any:
            return _session_say(
                session,
                text,
                allow_interruptions=allow_interruptions,
            )
        resolved_safe_say = _fallback_safe_say

    pending = state.pending_confirm_field or state.pending_confirm
    if not pending:
        return "none"

    confirm_intent = resolve_confirmation_intent(normalized)
    if confirm_intent is None:
        return "none"

    raw_phone = state.phone_pending or state.phone_e164 or ""
    fingerprint = f"{pending}|{normalized}|{state.dt_local}|{raw_phone}"
    now_pc = time.perf_counter()
    if (
        getattr(state, "last_confirm_fingerprint", None) == fingerprint
        and (now_pc - getattr(state, "last_confirm_ts", 0.0)) < 1.5
    ):
        logger.info("[CONFIRM] Duplicate confirmation ignored (same fingerprint within 1.5s)")
        state.turn_consumed = True
        return "consumed"

    state.last_confirm_fingerprint = fingerprint
    state.last_confirm_ts = now_pc
    state.turn_consumed = True

    logger.info(f"[CONFIRM] Deterministic routing: pending='{pending}', yes={confirm_intent}")
    cancel_scheduled_filler()
    interrupt_filler(force=True)

    if pending == "phone" and state.contact_phase_started:
        t0 = time.perf_counter()
        await assistant_tools.confirm_phone(confirmed=confirm_intent)  # type: ignore[call-arg]
        try:
            if hasattr(session, "interrupt"):
                session.interrupt()
                logger.info("[CONFIRM] Interrupted parallel LLM speech after phone confirmation")
        except Exception as _e:
            logger.debug(f"[CONFIRM] Interrupt attempt: {_e}")
        if refresh_memory_async is not None:
            await refresh_memory_async()

        if confirm_intent and state.is_complete() and not state.appointment_booked:
            logger.info("[CONFIRM] Fast-lane: state complete — booking directly (0 LLM hops)")
            booking_result = await assistant_tools.confirm_and_book_appointment()  # type: ignore[call-arg]
            if refresh_memory_async is not None:
                await refresh_memory_async()
            if mark_direct_response is not None:
                mark_direct_response()
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[CONFIRM] Fast-lane booking complete in {elapsed:.0f}ms")
            # Do NOT prepend the caller-number confirmation message here.
            # confirm_and_book_appointment() already returns the complete booking sentence
            # including the delivery question. Prefixing it again creates a duplicate.
            resolved_safe_say(booking_result)
            return "consumed"

        prompt = (
            _build_post_phone_confirmation_prompt(state)
            if confirm_intent
            else "No problem. What number should I use instead?"
        )
        if mark_direct_response is not None:
            mark_direct_response()
        resolved_safe_say(prompt)
        return "consumed"

    if pending == "email" and not state.email_confirmed:
        await assistant_tools.confirm_email(confirmed=confirm_intent)  # type: ignore[call-arg]
        if refresh_memory_async is not None:
            await refresh_memory_async()

        if confirm_intent and state.is_complete() and not state.appointment_booked:
            booking_result = await assistant_tools.confirm_and_book_appointment()  # type: ignore[call-arg]
            if refresh_memory_async is not None:
                await refresh_memory_async()
            if mark_direct_response is not None:
                mark_direct_response()
            resolved_safe_say(booking_result)
            return "consumed"

        prompt = (
            _build_missing_slot_prompt(state)
            if confirm_intent
            else "No problem. What's the correct email address?"
        )
        if mark_direct_response is not None:
            mark_direct_response()
        resolved_safe_say(prompt)
        return "consumed"

    state.turn_consumed = False
    return "none"


async def _handle_post_booking_turn(
    *,
    text: str,
    state: PatientState,
    assistant_tools: AssistantTools,
    session: AgentSession,
    safe_say: Optional[Callable[..., Any]] = None,
    cancel_scheduled_filler: Callable[[], None],
    interrupt_filler: Callable[..., None],
    refresh_memory_async: Optional[Callable[[], Awaitable[None]]] = None,
    mark_direct_response: Optional[Callable[[], None]] = None,
    schedule_auto_disconnect: Optional[Callable[[Any], None]] = None,
    cancel_auto_disconnect: Optional[Callable[[], None]] = None,
) -> Literal["none", "consumed"]:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized or not state.appointment_booked:
        return "none"

    resolved_safe_say = safe_say
    if resolved_safe_say is None:
        def _fallback_safe_say(text: str, *, allow_interruptions: bool = True) -> Any:
            return _session_say(
                session,
                text,
                allow_interruptions=allow_interruptions,
            )
        resolved_safe_say = _fallback_safe_say

    if state.final_goodbye_sent:
        if user_said_goodbye(normalized) or user_declined_anything_else(normalized):
            if not state.user_goodbye_detected:
                state.user_goodbye_detected = True
                logger.info("user_goodbye_detected")
            if schedule_auto_disconnect is not None:
                schedule_auto_disconnect(None)
            if refresh_memory_async is not None:
                await refresh_memory_async()
            return "consumed"
        if cancel_auto_disconnect is not None:
            cancel_auto_disconnect()
        state.final_goodbye_sent = False
        state.user_declined_more_help = False
        state.user_goodbye_detected = False
        state.closing_state = "open"
        if refresh_memory_async is not None:
            await refresh_memory_async()
        return "none"

    if state.delivery_preference_pending:
        preference = resolve_delivery_preference(normalized)
        if preference is None:
            can_answer_clinic_question = assistant_tools.can_answer_clinic_question(text)
            if can_answer_clinic_question is True:
                cancel_scheduled_filler()
                interrupt_filler(force=True)
                if mark_direct_response is not None:
                    mark_direct_response()
                answer = await assistant_tools.answer_clinic_question(text)
                answer = str(answer or "").strip()
                if answer:
                    state.anything_else_pending = False
                    state.user_declined_more_help = False
                    state.final_goodbye_sent = False
                    state.user_goodbye_detected = False
                    state.closing_state = "delivery_pending"
                    if refresh_memory_async is not None:
                        await refresh_memory_async()
                    resolved_safe_say(f"{answer} {_delivery_question_text(state)}")
                    return "consumed"
            state.delivery_ask_count = getattr(state, "delivery_ask_count", 0) + 1
            if state.delivery_ask_count >= 3:
                logger.info("[DELIVERY] Max retries reached — defaulting to WhatsApp")
                preference = "whatsapp"
            elif looks_like_delivery_follow_up_fragment(normalized):
                cancel_scheduled_filler()
                interrupt_filler(force=True)
                if mark_direct_response is not None:
                    mark_direct_response()
                if refresh_memory_async is not None:
                    await refresh_memory_async()
                resolved_safe_say(_delivery_question_text(state))
                return "consumed"
            else:
                return "none"
        else:
            state.delivery_ask_count = 0
        cancel_scheduled_filler()
        interrupt_filler(force=True)
        if mark_direct_response is not None:
            mark_direct_response()
        acknowledgement = await assistant_tools.set_delivery_preference(channel=preference)  # type: ignore[call-arg]
        if refresh_memory_async is not None:
            await refresh_memory_async()
        resolved_safe_say(acknowledgement)
        return "consumed"

    can_answer_clinic_question = assistant_tools.can_answer_clinic_question(text)
    if can_answer_clinic_question is True:
        cancel_scheduled_filler()
        interrupt_filler(force=True)
        if mark_direct_response is not None:
            mark_direct_response()
        answer = await assistant_tools.answer_clinic_question(
            text,
            include_follow_up=True,
        )
        if answer:
            if refresh_memory_async is not None:
                await refresh_memory_async()
            resolved_safe_say(answer)
            return "consumed"

    if state.anything_else_pending:
        if user_declined_anything_else(normalized) or user_said_goodbye(normalized):
            cancel_scheduled_filler()
            interrupt_filler(force=True)
            if user_said_goodbye(normalized):
                state.user_goodbye_detected = True
                logger.info("user_goodbye_detected")
            state.anything_else_pending = False
            state.user_declined_more_help = True
            state.closing_state = "closing"
            logger.info("closing_state_entered")
            if mark_direct_response is not None:
                mark_direct_response()
            final_handle = resolved_safe_say(_final_closing_text())
            state.final_goodbye_sent = True
            state.closing_state = "final_goodbye_sent"
            logger.info("final_goodbye_sent")
            if refresh_memory_async is not None:
                await refresh_memory_async()
            if schedule_auto_disconnect is not None:
                schedule_auto_disconnect(final_handle)
            return "consumed"

        state.anything_else_pending = False
        state.closing_state = "open"
        if refresh_memory_async is not None:
            await refresh_memory_async()
        return "none"

    return "none"


# =============================================================================
# Entrypoint
# =============================================================================

async def entrypoint(ctx: JobContext):
    """
    Clean entrypoint. ~400ms target latency.

    Pattern A: Filler phrases with smart suppression
    Pattern B: Deterministic yes/no routing (bypass LLM for confirmations)
    Pattern C: Deferred DB loading with 2s timeout (greeting starts immediately)
    Pattern D: Non-blocking booking (fire-and-forget Supabase writes)
    """
    # ── Connect ──────────────────────────────────────────────────────────────
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    logger.info(f"[LIFECYCLE] Participant connected: {participant.identity}")

    # ── Defaults ──────────────────────────────────────────────────────────────
    clinic_info: Optional[dict] = None
    agent_info: Optional[dict] = None
    settings: Optional[dict] = None
    agent_name = "Office Assistant"
    clinic_name = "our clinic"
    clinic_tz = DEFAULT_TZ
    clinic_region = DEFAULT_PHONE_REGION
    agent_lang = "en-US"
    call_started = time.time()

    # ── State ─────────────────────────────────────────────────────────────────
    state = PatientState()
    disconnect_event = asyncio.Event()
    turn_config = TurnTakingConfig(
        short_pause_ms=TURN_SHORT_PAUSE_MS,
        continuation_wait_ms=TURN_CONTINUATION_WAIT_MS,
        low_confidence_threshold=TURN_LOW_CONFIDENCE_THRESHOLD,
        deterministic_fast_path_enabled=DETERMINISTIC_FAST_PATH_ENABLED,
        lookup_filler_delay_ms=LOOKUP_FILLER_DELAY_MS,
        expected_slot_continuation_wait_ms=EXPECTED_SLOT_CONTINUATION_WAIT_MS,
        expected_slot_weak_fragment_max_tokens=EXPECTED_SLOT_WEAK_FRAGMENT_MAX_TOKENS,
        expected_slot_enable_date_time_fast_path=EXPECTED_SLOT_ENABLE_DATE_TIME_FAST_PATH,
    )
    turn_tracker = StreamingTurnTracker(turn_config)
    _turn_runtime: Dict[str, Any] = {
        "last_user_listening_started_at": None,
        "continuation_task": None,
        "planned_filler_text": None,
        "last_policy_decision": None,
    }

    def _set_expected_slot(slot: Optional[str | ExpectedUserSlot], *, reason: str) -> None:
        if not TURN_TRACKER_ENABLED:
            return
        previous = turn_tracker.expected_user_slot or "-"
        if slot is None:
            turn_tracker.clear_expected_user_slot()
            logger.info(f"[EXPECTED SLOT] cleared previous={previous} reason={reason}")
            return
        resolved = slot.value if isinstance(slot, ExpectedUserSlot) else str(slot)
        turn_tracker.set_expected_user_slot(resolved)
        logger.info(f"[EXPECTED SLOT] set={resolved} previous={previous} reason={reason}")

    def _apply_expected_slot_from_output(*, route: Optional[str], spoken_text: str) -> None:
        next_slot = _infer_expected_slot_from_response(
            route=route,
            spoken_text=spoken_text,
            state=state,
        )
        _set_expected_slot(
            next_slot,
            reason=route or ("prompt_text" if next_slot is not None else "response_complete"),
        )

    def _is_session_alive() -> bool:
        """Return False after disconnect has been signalled."""
        return not disconnect_event.is_set()

    def _safe_say(text: str, *, allow_interruptions: bool = True, add_to_chat_ctx: bool = True):
        """Speech wrapper that silently no-ops after disconnect."""
        if not _is_session_alive():
            logger.debug(f"[SAFE_SAY] Suppressed post-disconnect say: '{text[:40]}'")
            return None
        try:
            return _session_say(
                session,
                text,
                allow_interruptions=allow_interruptions,
                add_to_chat_ctx=add_to_chat_ctx,
            )
        except Exception as e:
            logger.debug(f"[SAFE_SAY] session.say failed: {e}")
            return None

    def _cancel_pending_continuation(reason: str) -> None:
        task = _turn_runtime.get("continuation_task")
        if task and not task.done():
            task.cancel()
            logger.info(f"[RESPONSE POLICY] cancelled_pending_continuation reason={reason}")
        _turn_runtime["continuation_task"] = None

    def _log_turn_snapshot() -> None:
        logger.info(f"[TURN TRACKER] {format_tracker_log(turn_tracker.snapshot)}")
        logger.info(
            "[COMPLETION CLASSIFIER] "
            f"label={turn_tracker.snapshot.completion_label.value} "
            f"reasons={turn_tracker.snapshot.completion_reasons}"
        )

    # ── SIP phone extraction ──────────────────────────────────────────────────
    called_num: Optional[str] = None
    caller_phone: Optional[str] = None
    used_fallback = False

    # Priority 1: SIP participant attributes
    if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
        sip_attrs = participant.attributes or {}
        caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
        called_num = _normalize_sip_user(
            sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
        )
        logger.info(f"[SIP] caller={caller_phone} called={called_num}")

    # Priority 2: Room name regex
    if not called_num:
        room_name = getattr(ctx.room, "name", "") or ""
        m = re.search(r"(\+1\d{10})", room_name) or re.search(r"call_(\+?\d+)_", room_name)
        if m:
            called_num = _normalize_sip_user(m.group(1))

    # Priority 3: Job metadata
    if not called_num:
        try:
            meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
            sip_info = meta.get("sip", {}) if isinstance(meta, dict) else {}
            called_num = _normalize_sip_user(sip_info.get("toUser"))
            if not caller_phone:
                caller_phone = sip_info.get("fromUser") or sip_info.get("phoneNumber")
        except Exception:
            pass

    # Priority 4: Env fallback (local testing only)
    if not called_num:
        called_num = os.getenv("DEFAULT_TEST_NUMBER", "+13103410536")
        used_fallback = True
        logger.warning(f"[FALLBACK] Using test number: {called_num}")

    # Pre-fill caller phone silently from SIP
    if caller_phone:
        clean_phone, last4 = _normalize_phone_e164(caller_phone, clinic_region)
        if clean_phone:
            store_detected_phone(state, clean_phone, last4, source="sip")
            state.phone_confirmed = False
            logger.info(f"[SIP] Caller phone pre-filled: ***{last4}")

    # ── Pattern C: Deferred DB load with 2s timeout ──────────────────────────
    context_task: Optional[asyncio.Task] = None
    if called_num:
        context_task = asyncio.create_task(fetch_clinic_context_optimized(called_num))

    if context_task:
        try:
            clinic_info, agent_info, settings, agent_name = await asyncio.wait_for(
                asyncio.shield(context_task), timeout=2.0
            )
            logger.info(f"[DB] Context loaded: clinic={clinic_info.get('name') if clinic_info else None}")
        except asyncio.TimeoutError:
            logger.warning("[DB] 2s timeout — using defaults, will retry after greeting")

    # Safety: demo fallback
    if not clinic_info:
        clinic_info = {"id": DEMO_CLINIC_ID, "name": "Dental Clinic"}

    # Apply context
    clinic_name = (clinic_info or {}).get("name") or clinic_name
    clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
    clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
    agent_lang = (agent_info or {}).get("default_language") or agent_lang
    state.tz = BOOKING_TZ

    # Re-normalize caller ID once the clinic region is known to handle local formats.
    if caller_phone:
        clean_phone, last4 = _normalize_phone_e164(caller_phone, clinic_region)
        if clean_phone:
            store_detected_phone(state, clean_phone, last4, source="sip")
            state.phone_confirmed = False

    # Push globals for tools
    update_global_clinic_info(clinic_info, settings or {})
    logger.info(f"[TZ] Booking timezone locked to: {BOOKING_TZ} (clinic field='{clinic_tz}' ignored for datetime math)")

    schedule = load_schedule_from_settings(settings or {})
    import tools.assistant_tools as _tools_mod

    # Fetch clinic FAQ (replaces RAG — one query, injected into prompt)
    clinic_knowledge_articles = await _fetch_clinic_knowledge_articles((clinic_info or {}).get("id"))
    clinic_faq = _format_clinic_faq(clinic_knowledge_articles)

    # ── System prompt builder ─────────────────────────────────────────────────
    _active_pipeline = os.getenv("ACTIVE_PIPELINE", "english").strip().lower()
    _is_urdu = _active_pipeline == "urdu"

    def get_updated_instructions() -> str:
        now = datetime.now(ZoneInfo(BOOKING_TZ))
        template = URDU_SYSTEM_PROMPT if _is_urdu else SYSTEM_PROMPT
        return template.format(
            agent_name=agent_name,
            clinic_name=clinic_name,
            timezone=BOOKING_TZ,
            current_date=now.strftime("%A, %B %d, %Y"),
            current_time=now.strftime("%I:%M %p"),
            state_summary=state.detailed_state_for_prompt(),
            clinic_context=clinic_faq,
        )

    # ── Pipeline components ───────────────────────────────────────────────────
    pipeline = get_pipeline_components(
        active_pipeline=_active_pipeline,
        agent_lang=agent_lang,
        stt_aggressive=STT_AGGRESSIVE_ENDPOINTING,
        latency_debug=LATENCY_DEBUG,
    )
    stt_instance = pipeline["stt"]
    llm_instance = pipeline["llm"]
    tts_instance = pipeline["tts"]
    logger.info(f"[PIPELINE] Active: {pipeline['pipeline_name']}")

    vad_instance = silero.VAD.load(
        min_speech_duration=VAD_MIN_SPEECH_DURATION,
        min_silence_duration=VAD_MIN_SILENCE_DURATION,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    assistant_tools = AssistantTools(
        state=state,
        clinic_info=clinic_info or {},
        settings=settings or {},
        schedule=schedule,
        clinic_tz=clinic_tz,
        knowledge_articles=clinic_knowledge_articles,
    )
    function_tools = llm.find_function_tools(assistant_tools)
    _session_refs: Dict[str, Any] = {"session": None, "agent": None}

    async def refresh_agent_memory_async() -> None:
        """Refresh the running agent instructions so state changes reach future turns."""
        try:
            ag = _session_refs.get("agent")
            if ag is None:
                return
            updated = get_updated_instructions()
            if hasattr(ag, "update_instructions"):
                await ag.update_instructions(updated)
            elif hasattr(ag, "_instructions"):
                ag._instructions = updated
            logger.debug(f"[MEMORY] Refreshed. State: {state.slot_summary()}")
        except Exception as e:
            logger.warning(f"[MEMORY] Refresh failed: {e}")

    def refresh_agent_memory() -> None:
        asyncio.create_task(refresh_agent_memory_async())

    async def _run_lookup_with_bridge(
        decision: Any,
        *,
        mark_direct_response: Optional[Callable[[], None]] = None,
    ) -> None:
        lookup_tool = getattr(assistant_tools, str(decision.lookup_tool or ""), None)
        if lookup_tool is None:
            logger.warning(f"[FAST PATH] Missing lookup tool: {decision.lookup_tool}")
            return

        lookup_task = asyncio.create_task(lookup_tool())  # type: ignore[misc]
        result_text: Optional[str] = None

        try:
            if decision.filler_text and not turn_tracker.snapshot.filler_spoken_for_turn:
                try:
                    result_text = await asyncio.wait_for(
                        asyncio.shield(lookup_task),
                        timeout=max(1, turn_config.lookup_filler_delay_ms) / 1000.0,
                    )
                except asyncio.TimeoutError:
                    bridge_text = str(decision.filler_text)
                    _filler_state["reason"] = "lookup_bridge"
                    asyncio.create_task(_send_filler(bridge_text))
                    logger.info(f"[FILLER] queued contextual_bridge='{bridge_text}'")

            if result_text is None:
                result_text = await lookup_task

            spoken_text = str(result_text or "").strip()
            if turn_tracker.snapshot.filler_spoken_for_turn:
                spoken_text = strip_duplicate_acknowledgement(spoken_text)

            if not spoken_text:
                return

            if mark_direct_response is not None:
                mark_direct_response()
            turn_tracker.mark_main_response_started()
            logger.info(
                f"[FAST PATH] lookup_complete route={decision.deterministic_route} "
                f"text='{spoken_text[:120]}'"
            )
            _safe_say(spoken_text)
        except Exception as exc:
            logger.warning(f"[FAST PATH] lookup flow failed: {exc}")
            await session.generate_reply(
                instructions=(
                    "The backend lookup was unavailable. Apologize briefly and ask for the "
                    "appointment phone number or name again if needed."
                )
            )

    async def _execute_policy_decision(
        decision: Any,
        *,
        after_continuation_wait: bool = False,
        mark_direct_response: Optional[Callable[[], None]] = None,
    ) -> bool:
        _turn_runtime["last_policy_decision"] = decision
        _turn_runtime["planned_filler_text"] = decision.filler_text
        logger.info(f"[RESPONSE POLICY] {format_policy_log(decision)}")

        if decision.action == PolicyAction.WAIT:
            _turn_runtime["planned_filler_text"] = None
            turn_tracker.mark_waiting_for_continuation(True)
            _cancel_scheduled_filler()
            if turn_tracker.snapshot.expected_user_slot and turn_tracker.snapshot.expected_slot_status == "unsatisfied":
                logger.info(
                    f"[EXPECTED SLOT] unsatisfied wait_for_continuation "
                    f"slot={turn_tracker.snapshot.expected_user_slot} "
                    f"text='{(turn_tracker.snapshot.current_turn_accumulated_text or turn_tracker.snapshot.latest_finalized_text)[:120]}'"
                )

            async def _resume_after_wait(turn_id: int) -> None:
                try:
                    await asyncio.sleep(max(1, decision.wait_ms) / 1000.0)
                    if turn_tracker.snapshot.logical_turn_id != turn_id:
                        return
                    if turn_tracker.snapshot.main_response_started:
                        return
                    turn_tracker.mark_waiting_for_continuation(False)
                    replay_text = (
                        turn_tracker.snapshot.current_turn_accumulated_text
                        or turn_tracker.snapshot.latest_finalized_text
                    )
                    if not replay_text:
                        return
                    turn_tracker.ingest_transcript(
                        replay_text,
                        is_final=True,
                        patient_state=state,
                        silence_ms=max(
                            turn_tracker.snapshot.silence_ms or 0,
                            turn_config.short_pause_ms,
                        )
                        + max(1, decision.wait_ms),
                    )
                    _log_turn_snapshot()
                    resumed = build_policy_decision(
                        turn_tracker.snapshot,
                        state,
                        turn_config,
                        after_continuation_wait=True,
                    )
                    await _execute_policy_decision(
                        resumed,
                        after_continuation_wait=True,
                        mark_direct_response=mark_direct_response,
                    )
                except asyncio.CancelledError:
                    return

            _cancel_pending_continuation("reschedule_wait")
            _turn_runtime["continuation_task"] = asyncio.create_task(
                _resume_after_wait(turn_tracker.snapshot.logical_turn_id)
            )
            return True

        _cancel_pending_continuation("decision_resolved")
        turn_tracker.mark_waiting_for_continuation(False)

        if decision.action == PolicyAction.FAST_PATH:
            _turn_runtime["planned_filler_text"] = None
            _cancel_scheduled_filler()
            _interrupt_filler(force=True)
            route = str(decision.deterministic_route or "")

            if route in {"booking.capture_datetime", "booking.capture_date", "booking.capture_time"}:
                capture_text = (
                    turn_tracker.snapshot.current_turn_accumulated_text
                    or turn_tracker.snapshot.latest_finalized_text
                ).strip()
                logger.info(
                    f"[EXPECTED SLOT] satisfied={turn_tracker.snapshot.expected_user_slot or '-'} "
                    f"status={turn_tracker.snapshot.expected_slot_status or '-'} "
                    f"text='{capture_text[:120]}'"
                )
                logger.info(
                    f"[FAST PATH] route={route} text='{capture_text[:120]}' "
                    f"expected_slot={turn_tracker.snapshot.expected_user_slot or '-'}"
                )
                spoken_text = await assistant_tools.update_patient_record(  # type: ignore[call-arg]
                    time_suggestion=capture_text
                )
                spoken_text = str(spoken_text or "").strip()
                if turn_tracker.snapshot.filler_spoken_for_turn:
                    spoken_text = strip_duplicate_acknowledgement(spoken_text)
                _apply_expected_slot_from_output(route=route, spoken_text=spoken_text)
                if mark_direct_response is not None:
                    mark_direct_response()
                turn_tracker.mark_main_response_started()
                if spoken_text:
                    _safe_say(spoken_text)
                return True

            if route == "clinic_info.answer":
                question_text = (
                    turn_tracker.snapshot.current_turn_accumulated_text
                    or turn_tracker.snapshot.latest_finalized_text
                ).strip()
                logger.info(
                    f"[FAST PATH] route={route} text='{question_text[:120]}' "
                    f"booked={state.appointment_booked}"
                )
                spoken_text = await assistant_tools.answer_clinic_question(
                    question_text,
                    include_follow_up=bool(
                        state.appointment_booked and not state.delivery_preference_pending
                    ),
                )
                spoken_text = str(spoken_text or "").strip()
                if turn_tracker.snapshot.filler_spoken_for_turn:
                    spoken_text = strip_duplicate_acknowledgement(spoken_text)
                _apply_expected_slot_from_output(route=route, spoken_text=spoken_text)
                if mark_direct_response is not None:
                    mark_direct_response()
                turn_tracker.mark_main_response_started()
                if spoken_text:
                    _safe_say(spoken_text)
                return True

            if mark_direct_response is not None:
                mark_direct_response()
            turn_tracker.mark_main_response_started()
            spoken_text = str(decision.response_text or "").strip()
            if turn_tracker.snapshot.filler_spoken_for_turn:
                spoken_text = strip_duplicate_acknowledgement(spoken_text)
            logger.info(
                f"[FAST PATH] route={route or '-'} "
                f"text='{spoken_text[:120]}'"
            )
            if spoken_text:
                _apply_expected_slot_from_output(route=route, spoken_text=spoken_text)
                _safe_say(spoken_text)
            return True

        if decision.action == PolicyAction.LOOKUP:
            _turn_runtime["planned_filler_text"] = None
            _cancel_scheduled_filler()
            _interrupt_filler(force=True)
            await _run_lookup_with_bridge(
                decision,
                mark_direct_response=mark_direct_response,
            )
            return True

        if decision.action == PolicyAction.LLM and decision.llm_instruction:
            _turn_runtime["planned_filler_text"] = None
            logger.info(
                f"[LLM PATH] mode=custom_instruction reasons={decision.reasons} "
                f"filler_spoken={turn_tracker.snapshot.filler_spoken_for_turn}"
            )
            await session.generate_reply(instructions=decision.llm_instruction)
            return True

        logger.info(
            f"[LLM PATH] mode=default reasons={decision.reasons} "
            f"filler={decision.filler_text or '-'}"
        )
        guarded_instruction = _build_no_repeat_llm_instruction(
            state,
            state.last_user_text or "",
        )
        if guarded_instruction:
            _turn_runtime["planned_filler_text"] = None
            logger.info("[LLM PATH] mode=guarded_stateful_fallback")
            await session.generate_reply(instructions=guarded_instruction)
            return True
        return False

    class ReceptionAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
            text = (new_message.text_content or "").strip()
            seeded = _seed_state_from_recent_context(state, schedule)
            if seeded:
                logger.info(f"[STATE PREFILL] {' | '.join(seeded)}")
                refresh_agent_memory()

            def _mark_direct_response() -> None:
                if _current_turn:
                    _current_turn.mark("direct_say")

            result = await _handle_deterministic_confirmation_turn(
                text=text,
                state=state,
                assistant_tools=assistant_tools,
                session=session,
                safe_say=_safe_say,
                cancel_scheduled_filler=_cancel_scheduled_filler,
                interrupt_filler=_interrupt_filler,
                refresh_memory_async=refresh_agent_memory_async,
                mark_direct_response=_mark_direct_response,
            )
            if result == "consumed":
                if TURN_TRACKER_ENABLED:
                    if state.pending_confirm == "phone" or state.pending_confirm_field == "phone":
                        _set_expected_slot(ExpectedUserSlot.PHONE_CONFIRMATION, reason="phone_confirmation_pending")
                    else:
                        _set_expected_slot(None, reason="confirmation_turn_consumed")
                raise llm.StopResponse()

            result = await _handle_post_booking_turn(
                text=text,
                state=state,
                assistant_tools=assistant_tools,
                session=session,
                safe_say=_safe_say,
                cancel_scheduled_filler=_cancel_scheduled_filler,
                interrupt_filler=_interrupt_filler,
                refresh_memory_async=refresh_agent_memory_async,
                mark_direct_response=_mark_direct_response,
                schedule_auto_disconnect=_schedule_auto_disconnect,
                cancel_auto_disconnect=_cancel_auto_disconnect,
            )
            if result == "consumed":
                if TURN_TRACKER_ENABLED:
                    _set_expected_slot(None, reason="post_booking_turn_consumed")
                raise llm.StopResponse()

            result = await _handle_exit_intent_turn(
                text=text,
                state=state,
                session=session,
                safe_say=_safe_say,
                cancel_scheduled_filler=_cancel_scheduled_filler,
                interrupt_filler=_interrupt_filler,
                refresh_memory_async=refresh_agent_memory_async,
                mark_direct_response=_mark_direct_response,
                schedule_auto_disconnect=_schedule_auto_disconnect,
            )
            if result == "consumed":
                if TURN_TRACKER_ENABLED:
                    _set_expected_slot(None, reason="exit_intent_turn_consumed")
                raise llm.StopResponse()

            if TURN_TRACKER_ENABLED:
                if not turn_tracker.snapshot.latest_finalized_text and text:
                    turn_tracker.ingest_transcript(
                        text,
                        is_final=True,
                        patient_state=state,
                        silence_ms=turn_tracker.snapshot.silence_ms,
                    )
                    _log_turn_snapshot()

                decision = build_policy_decision(
                    turn_tracker.snapshot,
                    state,
                    turn_config,
                )
                handled = await _execute_policy_decision(
                    decision,
                    mark_direct_response=_mark_direct_response,
                )
                if handled:
                    raise llm.StopResponse()

    # ── AgentSession + Agent ──────────────────────────────────────────────────
    # NOTE: MIN/MAX_ENDPOINTING_DELAY were previously imported from config but
    # never passed here — they had zero effect. Now they are applied.
    # 0.4s min / 0.7s max for telephony (reduced from 0.7s / 1.0s).
    _session_kwargs: Dict[str, Any] = {
        "vad": vad_instance,
        "stt": stt_instance,
        "llm": llm_instance,
        "tts": tts_instance,
        "max_tool_steps": 10,
    }
    # Apply endpointing delays only if AgentSession accepts them (SDK version guard)
    try:
        import inspect as _inspect
        _session_sig = set(_inspect.signature(AgentSession.__init__).parameters.keys())
        if "min_endpointing_delay" in _session_sig:
            _session_kwargs["min_endpointing_delay"] = MIN_ENDPOINTING_DELAY
            _session_kwargs["max_endpointing_delay"] = MAX_ENDPOINTING_DELAY
            logger.info(f"[SESSION] Endpointing: min={MIN_ENDPOINTING_DELAY}s max={MAX_ENDPOINTING_DELAY}s")
        else:
            logger.info("[SESSION] AgentSession does not support endpointing_delay — skipping")
    except Exception:
        pass
    session = AgentSession(**_session_kwargs)

    agent = ReceptionAgent(
        instructions=get_updated_instructions(),
        tools=function_tools,  # type: ignore[arg-type]
        allow_interruptions=True,
    )

    async def _direct_say(text: str) -> None:
        """
        Speak text directly via TTS, bypassing LLM re-generation.
        Used by Pattern B fast-lane after deterministic booking completion.
        Marks direct_say on the current turn timer.
        """
        if _current_turn:
            _current_turn.mark("direct_say")
        sess = _session_refs.get("session")
        if sess:
            _safe_say(text)
            logger.info(f"[DIRECT_SAY] '{text[:80]}'")

    # Inject refresh callback into tools module
    _tools_mod._REFRESH_AGENT_MEMORY = refresh_agent_memory
    assistant_tools._refresh_memory = refresh_agent_memory
    # Inject direct-say callback so tools can speak terminal responses without LLM
    assistant_tools._direct_say_callback = _direct_say

    # ── Greeting ──────────────────────────────────────────────────────────────
    if _is_urdu:
        greeting = (
            (settings or {}).get("greeting_text_urdu")
            or f"السلام علیکم! {clinic_name} میں کال کرنے کا شکریہ۔ میں آپ کی کیا مدد کر سکتی ہوں؟"
        )
    else:
        greeting = (
            (settings or {}).get("greeting_text")
            or f"Hi, thanks for calling {clinic_name}! How can I help you today?"
        )

    # ── Room options with noise cancellation ─────────────────────────────────
    audio_opts = room_io.AudioInputOptions(
        noise_cancellation=noise_cancellation.BVC() if NC_AVAILABLE and noise_cancellation is not None else None,
    )
    room_opts = room_io.RoomOptions(
        audio_input=audio_opts,
        close_on_disconnect=True,
    )

    # ── Start session ─────────────────────────────────────────────────────────
    await session.start(room=ctx.room, agent=agent, room_options=room_opts)
    _session_refs["session"] = session
    _session_refs["agent"] = agent

    _closing_runtime: Dict[str, Any] = {
        "auto_disconnect_task": None,
    }

    async def _disconnect_call_from_our_side() -> None:
        room_name = getattr(ctx.room, "name", "") or ""
        try:
            from livekit import api as lk_api

            if room_name:
                async with lk_api.LiveKitAPI() as livekit_api:
                    await livekit_api.room.delete_room(lk_api.DeleteRoomRequest(room=room_name))
                    logger.info("auto_disconnect_executed")
                    disconnect_event.set()
                    return
        except Exception as e:
            logger.warning(f"[CLOSE] Room delete fallback to local disconnect: {e}")

        try:
            await ctx.room.disconnect()
            logger.info("auto_disconnect_executed")
        except Exception as e:
            logger.warning(f"[CLOSE] Local room disconnect failed: {e}")
        finally:
            disconnect_event.set()

    def _cancel_auto_disconnect(*, user_resumed: bool = False) -> None:
        task = _closing_runtime.get("auto_disconnect_task")
        if task and not task.done():
            task.cancel()
        _closing_runtime["auto_disconnect_task"] = None
        if user_resumed:
            logger.info("auto_disconnect_cancelled_user_resumed")

    def _schedule_auto_disconnect(final_handle: Any) -> None:
        _cancel_auto_disconnect()
        logger.info("auto_disconnect_scheduled")

        async def _runner() -> None:
            try:
                if final_handle is not None and hasattr(final_handle, "wait_for_playout"):
                    await final_handle.wait_for_playout()
                silence_ms = (
                    AUTO_DISCONNECT_POST_GOODBYE_MS
                    if state.user_goodbye_detected
                    else AUTO_DISCONNECT_SILENCE_MS
                )
                await asyncio.sleep(silence_ms / 1000.0)
                if not state.final_goodbye_sent:
                    return
                await _disconnect_call_from_our_side()
            except asyncio.CancelledError:
                return

        _closing_runtime["auto_disconnect_task"] = asyncio.create_task(_runner())

    # ── Metrics (2 lines, not 40) ─────────────────────────────────────────────
    assistant_tools._schedule_auto_disconnect = _schedule_auto_disconnect

    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
    session.on("metrics_collected", _on_metrics)

    # ── Pattern A: Filler speech helpers ─────────────────────────────────────
    _filler_state: Dict[str, Any] = {
        "handle": None,
        "active": False,
        "sent_at": 0.0,
        "text": "",
        "scheduled_task": None,
        "reason": None,
    }
    filler_debounce_ms = MICRO_ACK_DELAY_MS

    def _clear_filler_state(handle: Optional[Any] = None):
        current = _filler_state.get("handle")
        if handle is not None and current is not handle:
            return
        _filler_state["handle"] = None
        _filler_state["active"] = False
        _filler_state["sent_at"] = 0.0
        _filler_state["text"] = ""
        _filler_state["reason"] = None

    def _clear_scheduled_filler(task: Optional[Any] = None):
        current = _filler_state.get("scheduled_task")
        if task is not None and current is not task:
            return
        _filler_state["scheduled_task"] = None

    def _cancel_scheduled_filler():
        task = _filler_state.get("scheduled_task")
        if task and not task.done():
            task.cancel()
        _clear_scheduled_filler()

    def _interrupt_handle_if_possible(handle: Any) -> None:
        interrupt = getattr(handle, "interrupt", None)
        if not callable(interrupt):
            return
        try:
            interrupt(force=True)
        except TypeError:
            interrupt()

    def _interrupt_filler(force: bool = False):
        """Interrupt a currently active filler. Pass force=True to skip the age guard."""
        sent_at = _filler_state.get("sent_at", 0.0)
        age_ms = (time.perf_counter() - sent_at) * 1000
        if not force and _filler_state.get("active") and age_ms < 300:
            logger.debug(f"[FILLER] Skipping interrupt — filler too young ({age_ms:.0f}ms)")
            return
        h = _filler_state.get("handle")
        if h:
            try:
                if callable(getattr(h, "interrupt", None)):
                    try:
                        h.interrupt(force=True)
                    except TypeError:
                        h.interrupt()
                logger.info(f"[FILLER INTERRUPTED] '{_filler_state.get('text', '')}' age={age_ms:.0f}ms")
            except Exception:
                pass
        _clear_filler_state()

    async def _send_filler(text: str):
        _t0 = time.perf_counter()
        handle = None
        try:
            # Fillers should never block the user or enter the LLM chat context.
            handle = _safe_say(text, add_to_chat_ctx=False)
            _filler_state["handle"] = handle
            _filler_state["active"] = True
            _filler_state["sent_at"] = _t0
            _filler_state["text"] = text

            remaining_ms = max(0, MICRO_ACK_MAX_DURATION_MS)
            await asyncio.sleep(remaining_ms / 1000.0)

            if _filler_state.get("handle") is handle and _filler_state.get("active"):
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info(f"[FILLER TIMEBOX] '{text}' reached {_ms:.0f}ms, interrupting")
                _interrupt_handle_if_possible(handle)
        except asyncio.CancelledError:
            _ms = (time.perf_counter() - _t0) * 1000
            logger.info(f"[FILLER CANCELLED] '{text}' after {_ms:.0f}ms")
            if handle:
                _interrupt_handle_if_possible(handle)
        except Exception as e:
            logger.debug(f"[FILLER ERROR] '{text}': {e}")
        finally:
            _clear_filler_state(handle)

    async def _schedule_filler(planned_text: str):
        task = asyncio.current_task()
        try:
            await asyncio.sleep(filler_debounce_ms / 1000.0)
            if _filler_state.get("scheduled_task") is not task:
                return
            if _filler_state.get("active"):
                return
            filler = (planned_text or "").strip()
            if not filler:
                return
            await _send_filler(filler)
        except asyncio.CancelledError:
            pass
        finally:
            _clear_scheduled_filler(task)

    # ── Pattern A: user_input_transcribed — send filler ───────────────────────
    def _on_user_transcribed(ev):
        global _current_turn
        text = (getattr(ev, "transcript", "") or getattr(ev, "text", "") or "").strip()
        if not text:
            return
        is_final = bool(getattr(ev, "is_final", True))

        if TURN_TRACKER_ENABLED:
            if turn_tracker.snapshot.logical_turn_id == 0 or turn_tracker.snapshot.main_response_started:
                turn_tracker.start_new_turn()
            silence_started = _turn_runtime.get("last_user_listening_started_at")
            silence_ms = None
            if is_final and silence_started:
                silence_ms = max(0, int((time.perf_counter() - silence_started) * 1000))
            turn_tracker.ingest_transcript(
                text,
                is_final=is_final,
                patient_state=state,
                silence_ms=silence_ms,
            )
            if is_final:
                _log_turn_snapshot()

        if not is_final:
            return

        logger.info(f"[USER] {text}")
        state.remember_user_text(text)
        _cancel_scheduled_filler()

        if state.final_goodbye_sent and not user_said_goodbye(text) and not user_declined_anything_else(text):
            _cancel_auto_disconnect(user_resumed=True)
            state.final_goodbye_sent = False
            state.user_declined_more_help = False
            state.user_goodbye_detected = False
            state.closing_state = "open"

        # Start a new per-turn timer
        _current_turn = TurnTimer()
        _current_turn.mark("user_eou")
        _current_turn.mark("stt_final")

        # NOTE: refresh_agent_memory() is intentionally NOT called here.
        # It was previously called on every utterance even when no state changed,
        # wasting CPU and (for some SDK versions) causing a ChatContext rebuild.
        # Memory is now only refreshed inside tool calls that actually mutate state.

        # Reset turn ownership for this utterance before any deterministic routing.
        state.turn_consumed = False

        if TURN_TRACKER_ENABLED:
            decision = build_policy_decision(
                turn_tracker.snapshot,
                state,
                turn_config,
            )
            _turn_runtime["last_policy_decision"] = decision
            _turn_runtime["planned_filler_text"] = decision.filler_text

            if decision.action == PolicyAction.WAIT:
                logger.info(
                    f"[FILLER] suppressed completion={turn_tracker.snapshot.completion_label.value} "
                    f"reasons={turn_tracker.snapshot.completion_reasons}"
                )
                return

            if not FILLER_ENABLED or _filler_state["active"]:
                return

            if decision.filler_text:
                _filler_state["reason"] = "policy_filler"
                _filler_state["scheduled_task"] = asyncio.create_task(
                    _schedule_filler(str(decision.filler_text))
                )
                logger.info(
                    f"[FILLER] scheduled text='{decision.filler_text}' "
                    f"reasons={decision.reasons}"
                )
                return

            logger.info("[FILLER] suppressed_no_helpful_bridge")
            return

        if not FILLER_ENABLED or _filler_state["active"]:
            return
        filler_text, suppress_reason = _micro_ack_decision(text, state=state)
        if filler_text:
            _filler_state["reason"] = "micro_ack"
            _filler_state["scheduled_task"] = asyncio.create_task(_schedule_filler(filler_text))
            return
        if suppress_reason == "capture_turn":
            logger.info("micro_ack_suppressed_capture_turn")
        elif suppress_reason == "confirmation_turn":
            logger.info("micro_ack_suppressed_confirmation_turn")
        elif suppress_reason == "fragmented_turn":
            logger.info("micro_ack_suppressed_fragmented_turn")

    session.on("user_input_transcribed", _on_user_transcribed)

    def _on_user_state_changed(ev):
        new_state = getattr(ev, "new_state", None)
        if new_state == "listening":
            _turn_runtime["last_user_listening_started_at"] = time.perf_counter()
            return
        if new_state != "speaking":
            return
        if state.final_goodbye_sent:
            _cancel_auto_disconnect()
        if TURN_TRACKER_ENABLED and turn_tracker.snapshot.awaiting_continuation:
            logger.info("[RESPONSE POLICY] user_resumed_before_continuation_timeout")
        _cancel_pending_continuation("user_resumed_speaking")
        _cancel_scheduled_filler()
        _interrupt_filler(force=True)

    def _on_agent_state_changed(ev):
        if getattr(ev, "new_state", None) != "speaking":
            return
        current_handle = None
        try:
            current_handle = session.current_speech
        except Exception:
            current_handle = None

        if current_handle is _filler_state.get("handle"):
            if _current_turn:
                _current_turn.mark("filler_sent")
            if TURN_TRACKER_ENABLED:
                turn_tracker.mark_filler_spoken(_filler_state.get("text"))
            logger.info(f"[FILLER] spoken text='{_filler_state.get('text', '')}'")
            return

        scheduled_task = _filler_state.get("scheduled_task")
        if scheduled_task and not scheduled_task.done() and not _filler_state["active"]:
            logger.info("[FILLER] suppressed_fast_main_reply")
        _cancel_scheduled_filler()
        if _filler_state["active"]:
            logger.info("[FILLER] cancelled_main_reply_ready")
            _interrupt_filler(force=True)
        if TURN_TRACKER_ENABLED:
            turn_tracker.mark_main_response_started()
        if _current_turn:
            _current_turn.mark("speech_started")
            _current_turn.log_summary(state.last_user_text or "")

    def _on_conversation_item_added(ev):
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) != "assistant":
            return
        text = getattr(item, "text_content", None)
        if isinstance(text, str) and text.strip():
            spoken = text.strip()
            lower = spoken.lower()

            # Perspective inversion guard
            inversion_patterns = [
                "i'm calling from",
                "i am calling from",
                "the number i'm calling",
                "the number i am calling",
                "number i'm calling from",
                "number i am calling from",
            ]
            if any(pattern in lower for pattern in inversion_patterns):
                logger.warning(f"[AGENT] Perspective inversion detected in: '{spoken[:80]}'")
                corrected = spoken
                for bad, good in [
                    ("the number i'm calling from", "the number you're calling from"),
                    ("the number i am calling from", "the number you're calling from"),
                    ("number i'm calling from", "number you're calling from"),
                    ("number i am calling from", "number you're calling from"),
                    ("i'm calling from", "you're calling from"),
                    ("i am calling from", "you are calling from"),
                ]:
                    corrected = corrected.replace(bad, good).replace(
                        bad.replace("i'm", "I'm").replace("i am", "I am"),
                        good.replace("you're", "You're").replace("you are", "You are"),
                    )
                if corrected != spoken:
                    logger.info(f"[AGENT] Auto-corrected to: '{corrected[:80]}'")
                    try:
                        if hasattr(session, "interrupt"):
                            session.interrupt()
                        _safe_say(corrected)
                        return
                    except Exception as e:
                        logger.warning(f"[AGENT] Correction injection failed: {e}")

            logger.info(f"[AGENT SAID] '{spoken}'")
            if TURN_TRACKER_ENABLED:
                _apply_expected_slot_from_output(route=None, spoken_text=spoken)
            if state.appointment_booked and "whatsapp" in lower and "sms" in lower:
                if not state.delivery_preference_pending:
                    state.delivery_preference_pending = True
                    state.delivery_preference_asked = True
                    state.anything_else_pending = False
                    state.closing_state = "delivery_pending"
                    logger.debug("[STATE] delivery_preference_pending set to True (first occurrence)")
                else:
                    logger.debug("[STATE] delivery_preference_pending already True — skipping re-set on rephrase")
            if "anything else i can help" in lower:
                if not state.anything_else_pending:
                    state.anything_else_pending = True
                    state.anything_else_asked = True
                    state.closing_state = "anything_else_pending"
                    logger.debug("[STATE] anything_else_pending set to True (first occurrence)")
                else:
                    logger.debug("[STATE] anything_else_pending already True — skipping re-set")
            if _current_turn:
                _current_turn.mark("speech_committed")

    session.on("user_state_changed", _on_user_state_changed)  # type: ignore[arg-type]
    session.on("agent_state_changed", _on_agent_state_changed)  # type: ignore[arg-type]
    session.on("conversation_item_added", _on_conversation_item_added)  # type: ignore[arg-type]


    # ── SIP late-join: capture phone from participant metadata ────────────────
    @ctx.room.on("participant_connected")
    def _on_participant_connected(p: rtc.RemoteParticipant):
        if p.kind != ParticipantKind.PARTICIPANT_KIND_SIP:
            return
        sip_attrs = p.attributes or {}
        late_caller = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
        if late_caller and not state.phone_e164:
            clean, last4 = _normalize_phone_e164(late_caller, clinic_region)
            if clean:
                store_detected_phone(state, clean, last4, source="sip")
                state.phone_confirmed = False
                logger.info(f"[SIP LATE] Phone pre-filled: ***{last4}")
                refresh_agent_memory()

        # Refresh clinic context if we used env fallback
        if used_fallback:
            late_called = _normalize_sip_user(
                sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
            )
            if late_called:
                async def _refresh_ctx():
                    nonlocal clinic_info, agent_info, settings, agent_name, clinic_name, clinic_tz, clinic_region, schedule, clinic_faq, clinic_knowledge_articles
                    try:
                        ci, ai, st, an = await fetch_clinic_context_optimized(late_called)
                        if ci:
                            clinic_info = ci
                            agent_info = ai
                            settings = st
                            agent_name = an
                            clinic_name = ci.get("name") or clinic_name
                            clinic_tz = ci.get("timezone") or clinic_tz
                            clinic_region = ci.get("default_phone_region") or clinic_region
                            schedule = load_schedule_from_settings(st or {})
                            update_global_clinic_info(ci, st or {})
                            clinic_knowledge_articles = await _fetch_clinic_knowledge_articles(ci.get("id"))
                            clinic_faq = _format_clinic_faq(clinic_knowledge_articles)
                            assistant_tools.update_clinic_context(
                                clinic_info=ci,
                                settings=st or {},
                                schedule=schedule,
                                clinic_tz=clinic_tz,
                                knowledge_articles=clinic_knowledge_articles,
                            )
                            refresh_agent_memory()
                            logger.info(f"[SIP LATE] Context refreshed: {ci.get('name')}")
                    except Exception as e:
                        logger.warning(f"[SIP LATE] Context refresh failed: {e}")
                asyncio.create_task(_refresh_ctx())

    # ── Say greeting ──────────────────────────────────────────────────────────
    await session.say(greeting)

    # ── Pattern C: Deferred context (if 2s timeout was hit) ──────────────────
    if context_task and not context_task.done():
        try:
            clinic_info, agent_info, settings, agent_name = await context_task
            if clinic_info:
                clinic_name = clinic_info.get("name") or clinic_name
                clinic_tz = clinic_info.get("timezone") or clinic_tz
                clinic_region = clinic_info.get("default_phone_region") or clinic_region
                schedule = load_schedule_from_settings(settings or {})
                state.tz = BOOKING_TZ
                update_global_clinic_info(clinic_info, settings or {})
                clinic_knowledge_articles = await _fetch_clinic_knowledge_articles(clinic_info.get("id"))
                clinic_faq = _format_clinic_faq(clinic_knowledge_articles)
                assistant_tools.update_clinic_context(
                    clinic_info=clinic_info,
                    settings=settings or {},
                    schedule=schedule,
                    clinic_tz=clinic_tz,
                    knowledge_articles=clinic_knowledge_articles,
                )
                refresh_agent_memory()
                logger.info(f"[DB] Deferred context loaded: {clinic_name}")
        except Exception as e:
            logger.warning(f"[DB] Deferred load failed: {e}")

    # ── Shutdown callback ─────────────────────────────────────────────────────
    async def _on_shutdown():
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Call ended. Duration={dur}s, booked={state.booking_confirmed}")
        try:
            if clinic_info and clinic_info.get("organization_id"):
                from config import VALID_CALL_OUTCOMES, map_call_outcome
                outcome = map_call_outcome(None, booking_made=bool(state.booking_confirmed))
                payload = {
                    "organization_id": clinic_info["organization_id"],
                    "clinic_id": clinic_info["id"],
                    "caller_phone_masked": f"***{state.phone_last4}" if state.phone_last4 else "Unknown",
                    "caller_name": state.full_name,
                    "outcome": outcome,
                    "duration_seconds": dur,
                }
                if agent_info and agent_info.get("id"):
                    payload["agent_id"] = agent_info["id"]
                await asyncio.to_thread(
                    lambda: supabase.table("call_sessions").insert(payload).execute()
                )
                logger.info(f"[DB] Call session saved: outcome={outcome}")
        except Exception as e:
            logger.error(f"[DB] Call session error: {e}")

    ctx.add_shutdown_callback(_on_shutdown)

    # ── Wait for disconnect ───────────────────────────────────────────────────
    @ctx.room.on("disconnected")
    def _on_disconnected():
        disconnect_event.set()

    @ctx.room.on("participant_disconnected")
    def _on_participant_disconnected(p):
        disconnect_event.set()

    try:
        await asyncio.wait_for(disconnect_event.wait(), timeout=7200)
    except asyncio.TimeoutError:
        pass


# =============================================================================
# Prewarm
# =============================================================================

def prewarm(proc: JobProcess):
    """Pre-load VAD model once at worker start."""
    try:
        silero.VAD.load()
        logger.info("[PREWARM] VAD loaded")
    except Exception as e:
        logger.error(f"[PREWARM] VAD load failed: {e}")

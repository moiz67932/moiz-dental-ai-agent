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
import random
import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Optional, Dict, Any
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
    DEFAULT_PHONE_REGION,
    DEMO_CLINIC_ID,
    FILLER_ENABLED,
    FILLER_MAX_DURATION_MS,
    FILLER_DEBOUNCE_MS,
    FILLER_PHRASES,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    MIN_ENDPOINTING_DELAY,
    MAX_ENDPOINTING_DELAY,
    STT_AGGRESSIVE_ENDPOINTING,
    LATENCY_DEBUG,
)
from models.state import PatientState, YES_PAT, NO_PAT
from pipelines.pipeline_config import get_pipeline_components
from pipelines.urdu_prompt import URDU_SYSTEM_PROMPT
from services.database_service import fetch_clinic_context_optimized
from services.scheduling_service import load_schedule_from_settings
from utils.agent_flow import (
    has_date_reference,
    has_time_reference,
    is_active_filler_event,
    resolve_confirmation_intent,
    store_detected_phone,
    time_expression_score,
)
from tools.assistant_tools import AssistantTools, update_global_clinic_info

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
        logger.info(f"[TURN] {' | '.join(parts)} | user='{snippet}'")


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
5. After name+reason+time captured: ask "Can I use the number you're calling from?"
   - "yes" / "sure" / similar -> call confirm_phone(confirmed=True) IMMEDIATELY. Do not ask again.
   - "no" or gives different number -> call update_patient_record(phone=...).
6. All required fields captured -> call confirm_and_book_appointment IMMEDIATELY. Don't ask "shall I book?".
7. Read the booking confirmation EXACTLY as the tool returns it. Do not rephrase.
8. Ask "Is there anything else I can help with?" — if no -> say goodbye -> call end_conversation.

RULES:
- Call update_patient_record IMMEDIATELY when you hear any info. Never wait.
- Normalize spoken input before saving: "three one zero" -> "310", "at gmail dot com" -> "@gmail.com".
- Never say "booked" until the tool confirms it.
- Never admit you are AI — say "I'm the office assistant."
- Never offer callbacks (you cannot dial out).
- Keep every response to 1-2 short sentences. This is a phone call.
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

async def _fetch_clinic_faq(clinic_id: Optional[str]) -> str:
    """
    Fetch clinic FAQ articles in one query and return as a compact text block.
    Injected into the system prompt — no tool call needed per question.
    """
    if not clinic_id:
        return "No additional clinic information available."
    try:
        result = await asyncio.to_thread(
            lambda: supabase.table("knowledge_articles")
            .select("title, body")
            .eq("clinic_id", clinic_id)
            .limit(20)
            .execute()
        )
        raw_rows = result.data
        if not isinstance(raw_rows, list) or not raw_rows:
            return "No additional clinic information available."
        lines: list[str] = []
        for row in raw_rows:
            if not isinstance(row, Mapping):
                continue
            title_value = row.get("title")
            body_value = row.get("body")
            title = title_value.strip() if isinstance(title_value, str) else ""
            body = " ".join(body_value.split()[:50]) if isinstance(body_value, str) else ""  # cap at 50 words each
            if title and body:
                lines.append(f"- {title}: {body}")
        return "\n".join(lines) if lines else "No additional clinic information available."
    except Exception as e:
        logger.warning(f"[FAQ] Fetch failed: {e}")
        return "No additional clinic information available."


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


def _choose_filler(text: str) -> str:
    """Pick a contextually appropriate filler based on what the user said."""
    lower = text.strip().lower()
    # Questions → "thinking" filler
    if lower.endswith("?") or any(w in lower for w in [
        "do you have", "is there", "can i", "can you", "what", "when",
        "how much", "available", "price", "cost",
    ]):
        return random.choice(FILLER_THINKING)
    # User giving info (name, time, reason) → acknowledge
    if any(w in lower for w in [
        "my name", "i'm", "i am", "i want", "i need", "book",
        "appointment", "schedule", "cleaning", "whitening",
    ]):
        return random.choice(FILLER_ACKNOWLEDGE)
    # Everything else
    return random.choice(FILLER_GENERAL)


def _needs_filler(text: str) -> bool:
    """Return True if a filler phrase should be sent for this user input."""
    lower = text.strip().lower()
    words = lower.split()
    # Skip for yes/no
    if len(words) <= 2 and (YES_PAT.search(lower) or NO_PAT.search(lower)):
        return False
    # Skip for short non-question answers (micro-confirmations)
    if len(words) <= 2 and not lower.endswith("?"):
        return False
    # Skip for direct date/time answers and slot values
    if len(words) <= 10 and (has_date_reference(text) or has_time_reference(text) or time_expression_score(text) >= 2):
        return False
    if SLOT_VALUE_RE.search(text.strip()):
        return False
    return True


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
    state.tz = clinic_tz

    # Re-normalize caller ID once the clinic region is known to handle local formats.
    if caller_phone:
        clean_phone, last4 = _normalize_phone_e164(caller_phone, clinic_region)
        if clean_phone:
            store_detected_phone(state, clean_phone, last4, source="sip")
            state.phone_confirmed = False

    # Push globals for tools
    update_global_clinic_info(clinic_info, settings or {})

    from tools.assistant_tools import _GLOBAL_SCHEDULE
    schedule = load_schedule_from_settings(settings or {})
    import tools.assistant_tools as _tools_mod
    _tools_mod._GLOBAL_SCHEDULE = schedule
    _tools_mod._GLOBAL_CLINIC_TZ = clinic_tz

    # Fetch clinic FAQ (replaces RAG — one query, injected into prompt)
    clinic_faq = await _fetch_clinic_faq((clinic_info or {}).get("id"))

    # ── System prompt builder ─────────────────────────────────────────────────
    _active_pipeline = os.getenv("ACTIVE_PIPELINE", "english").strip().lower()
    _is_urdu = _active_pipeline == "urdu"

    def get_updated_instructions() -> str:
        now = datetime.now(ZoneInfo(clinic_tz))
        template = URDU_SYSTEM_PROMPT if _is_urdu else SYSTEM_PROMPT
        return template.format(
            agent_name=agent_name,
            clinic_name=clinic_name,
            timezone=clinic_tz,
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
    # Use FILLER_PHRASES from config (single source of truth — no unicode ellipsis)
    filler_phrases = FILLER_PHRASES
    logger.info(f"[PIPELINE] Active: {pipeline['pipeline_name']}")

    vad_instance = silero.VAD.load(
        min_speech_duration=VAD_MIN_SPEECH_DURATION,
        min_silence_duration=VAD_MIN_SILENCE_DURATION,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    assistant_tools = AssistantTools(state)
    function_tools = llm.find_function_tools(assistant_tools)

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

    agent = Agent(
        instructions=get_updated_instructions(),
        tools=function_tools,  # type: ignore[arg-type]
        allow_interruptions=True,
    )

    # Store session ref for memory refresh
    _session_refs: Dict[str, Any] = {"session": None, "agent": None}

    def refresh_agent_memory():
        """Refresh the LLM's system prompt with current state. Called after tool writes."""
        try:
            ag = _session_refs.get("agent")
            if ag and hasattr(ag, "_instructions"):
                ag._instructions = get_updated_instructions()
                logger.debug(f"[MEMORY] Refreshed. State: {state.slot_summary()}")
        except Exception as e:
            logger.warning(f"[MEMORY] Refresh failed: {e}")

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
            sess.say(text)
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

    # ── Metrics (2 lines, not 40) ─────────────────────────────────────────────
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
    }
    # Use config constant (default 120ms, reduced from 250ms)
    filler_debounce_ms = FILLER_DEBOUNCE_MS

    def _clear_filler_state(handle: Optional[Any] = None):
        current = _filler_state.get("handle")
        if handle is not None and current is not handle:
            return
        _filler_state["handle"] = None
        _filler_state["active"] = False
        _filler_state["sent_at"] = 0.0
        _filler_state["text"] = ""

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

    def _interrupt_filler():
        sent_at = _filler_state.get("sent_at", 0.0)
        age_ms = (time.perf_counter() - sent_at) * 1000
        if _filler_state.get("active") and age_ms < 300:
            logger.debug(f"[FILLER] Skipping interrupt — filler too young ({age_ms:.0f}ms)")
            return
        h = _filler_state.get("handle")
        if h:
            try:
                if hasattr(h, "interrupt"):
                    h.interrupt()
            except Exception:
                pass
        _clear_filler_state()

    async def _send_filler(text: str):
        _t0 = time.perf_counter()
        handle = None
        try:
            # session.say() returns SpeechHandle synchronously.
            # Do NOT await, do NOT create_task, do NOT inspect.isawaitable.
            # allow_interruptions=True so LLM response can cut in when ready.
            handle = session.say(text, allow_interruptions=True)
            _filler_state["handle"] = handle
            _filler_state["active"] = True
            _filler_state["sent_at"] = _t0
            _filler_state["text"] = text
            logger.info(f"[FILLER STARTED] '{text}'")

            # Let filler play for up to FILLER_MAX_DURATION_MS, then interrupt
            await asyncio.sleep(FILLER_MAX_DURATION_MS / 1000.0)

            if _filler_state.get("handle") is handle and _filler_state.get("active"):
                _ms = (time.perf_counter() - _t0) * 1000
                logger.info(f"[FILLER TIMEBOX] '{text}' reached {_ms:.0f}ms, interrupting")
                if hasattr(handle, "interrupt"):
                    handle.interrupt()
        except asyncio.CancelledError:
            _ms = (time.perf_counter() - _t0) * 1000
            logger.info(f"[FILLER CANCELLED] '{text}' after {_ms:.0f}ms")
            if handle and hasattr(handle, "interrupt"):
                handle.interrupt()
        except Exception as e:
            logger.debug(f"[FILLER ERROR] '{text}': {e}")
        finally:
            _clear_filler_state(handle)

    async def _schedule_filler():
        task = asyncio.current_task()
        try:
            await asyncio.sleep(filler_debounce_ms / 1000.0)
            if _filler_state.get("scheduled_task") is not task:
                return
            if _filler_state.get("active"):
                return
            last_text = state.last_user_text or ""
            filler = _choose_filler(last_text)
            logger.debug(f"[FILLER] Sending: '{filler}'")
            await _send_filler(filler)
        except asyncio.CancelledError:
            pass
        finally:
            _clear_scheduled_filler(task)

    # ── Pattern A: user_input_transcribed — send filler ───────────────────────
    def _on_user_transcribed(ev):
        global _current_turn
        if not getattr(ev, "is_final", True):
            return
        text = (getattr(ev, "transcript", "") or getattr(ev, "text", "") or "").strip()
        if not text:
            return
        # Log user input
        logger.info(f"[USER] {text}")
        state.remember_user_text(text)
        _cancel_scheduled_filler()

        # Start a new per-turn timer
        _current_turn = TurnTimer()
        _current_turn.mark("user_eou")

        # NOTE: refresh_agent_memory() is intentionally NOT called here.
        # It was previously called on every utterance even when no state changed,
        # wasting CPU and (for some SDK versions) causing a ChatContext rebuild.
        # Memory is now only refreshed inside tool calls that actually mutate state.

        if not FILLER_ENABLED or _filler_state["active"]:
            return
        if _needs_filler(text):
            _filler_state["scheduled_task"] = asyncio.create_task(_schedule_filler())

    session.on("user_input_transcribed", _on_user_transcribed)

    # ── Barge-in: interrupt filler and agent when user speaks ─────────────────
    _agent_speech_handle: Dict[str, Any] = {"handle": None}

    def _interrupt_agent():
        h = _agent_speech_handle.get("handle")
        if h:
            try:
                if hasattr(h, "interrupt"):
                    h.interrupt()
                elif hasattr(h, "cancel"):
                    h.cancel()
            except Exception:
                pass
            _agent_speech_handle["handle"] = None

    def _on_user_speech_started(ev):
        _cancel_scheduled_filler()
        _interrupt_filler()
        _interrupt_agent()

    def _on_agent_speech_started(ev):
        speech_text = getattr(ev, "text", "") or getattr(ev, "content", "") or ""
        speech_handle = getattr(ev, "handle", None) or getattr(ev, "speech_handle", None)
        is_filler = is_active_filler_event(
            speech_text,
            _filler_state.get("text"),
            filler_phrases,
            same_handle=speech_handle is _filler_state.get("handle"),
        )

        # Debug log when a real response fires while filler is active
        if _filler_state["active"]:
            logger.info(f"[FILLER_DEBUG] Speech started while filler active. text='{speech_text}', is_filler={is_filler}")

        # If this IS a filler, don't capture its handle or interrupt it
        if is_filler:
            if _current_turn:
                _current_turn.mark("filler_sent")
            return

        _cancel_scheduled_filler()

        # Mark speech_started on the per-turn timer and emit summary
        if _current_turn:
            _current_turn.mark("speech_started")
            _current_turn.log_summary(state.last_user_text or "")

        # Capture handle for barge-in (real LLM response only)
        h = speech_handle
        if h:
            _agent_speech_handle["handle"] = h

        # Real response started — interrupt any active filler
        if _filler_state["active"]:
            _interrupt_filler()

    session.on("user_speech_started", _on_user_speech_started)  # type: ignore[arg-type]
    session.on("agent_speech_started", _on_agent_speech_started)  # type: ignore[arg-type]
    # Also try alternative event names for SDK compatibility
    try:
        session.on("user_started_speaking", _on_user_speech_started)  # type: ignore[arg-type]
    except Exception:
        pass

    # Agent speech committed (for logging + turn timer)
    def _on_agent_committed(msg):
        for attr in ("text", "content", "text_content"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                is_fil = any(text.startswith(f) for f in filler_phrases)
                prefix = "[FILLER SPOKEN]" if is_fil else "[AGENT SAID]"
                logger.info(f"{prefix} '{text}'")
                if not is_fil and _current_turn:
                    _current_turn.mark("speech_committed")
                break

    try:
        session.on("agent_speech_committed", _on_agent_committed)  # type: ignore[arg-type]
    except Exception:
        pass

    # ── Pattern B: Deterministic yes/no routing for confirmations ─────────────
    def _on_user_input_confirmation(ev):
        """
        Bypass LLM for clear yes/no during pending phone/email confirmations.

        UPGRADED (2026-03-08): After phone confirmation, if all required fields
        are already present (name, reason, time, phone), directly call
        confirm_and_book_appointment() and session.say() the result without any
        extra LLM roundtrip. This eliminates 2 LLM hops from the booking
        happy path (was: generate_reply → LLM decides to book → LLM re-speaks).

        Fast-lane saves 600–1500ms on the happy path booking turn.
        Falls back to generate_reply() for ambiguous/incomplete state.
        """
        if not getattr(ev, "is_final", True):
            return
        text = (getattr(ev, "transcript", "") or getattr(ev, "text", "") or "").strip().lower()
        if not text:
            return

        pending = state.pending_confirm_field or state.pending_confirm
        if not pending:
            return

        confirm_intent = resolve_confirmation_intent(text)
        if confirm_intent is None:
            return

        logger.info(f"[CONFIRM] Deterministic routing: pending='{pending}', yes={confirm_intent}")

        async def _confirm_phone_async(confirmed: bool):
            t0 = time.perf_counter()
            try:
                await assistant_tools.confirm_phone(confirmed=confirmed)  # type: ignore[call-arg]
                if confirmed:
                    # ── FAST LANE: book directly if all info is present ──────
                    # Check: name + reason + confirmed time + phone all present.
                    if state.is_complete():
                        logger.info("[CONFIRM] Fast-lane: state complete — booking directly (0 LLM hops)")
                        if _current_turn:
                            _current_turn.mark("direct_say")
                        booking_result = await assistant_tools.confirm_and_book_appointment()  # type: ignore[call-arg]
                        refresh_agent_memory()  # update prompt so LLM knows booking is done
                        elapsed = (time.perf_counter() - t0) * 1000
                        logger.info(f"[CONFIRM] Fast-lane booking complete in {elapsed:.0f}ms")
                        await session.say(booking_result)
                    else:
                        # State incomplete — missing name, reason, or time.
                        # Let LLM decide what to ask next.
                        logger.info(f"[CONFIRM] Phone confirmed but state incomplete: {state.missing_slots()}")
                        await session.generate_reply()
                else:
                    await session.say("No problem! What number should I use instead?")
            except Exception as e:
                logger.error(f"[CONFIRM] Phone confirm error: {e}")
                # Fallback: let LLM recover
                try:
                    await session.generate_reply()
                except Exception:
                    pass

        async def _confirm_email_async(confirmed: bool):
            try:
                await assistant_tools.confirm_email(confirmed=confirmed)  # type: ignore[call-arg]
                if confirmed:
                    await session.generate_reply()
                else:
                    await session.say("No problem! What's the correct email address?")
            except Exception as e:
                logger.error(f"[CONFIRM] Email confirm error: {e}")

        if pending == "phone" and state.contact_phase_started:
            asyncio.create_task(_confirm_phone_async(confirm_intent))
        elif pending == "email" and not state.email_confirmed:
            asyncio.create_task(_confirm_email_async(confirm_intent))

    session.on("user_input_transcribed", _on_user_input_confirmation)

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
                    try:
                        ci, ai, st, an = await fetch_clinic_context_optimized(late_called)
                        if ci:
                            update_global_clinic_info(ci, st or {})
                            _tools_mod._GLOBAL_SCHEDULE = load_schedule_from_settings(st or {})
                            _tools_mod._GLOBAL_CLINIC_TZ = ci.get("timezone", clinic_tz)
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
                state.tz = clinic_tz
                update_global_clinic_info(clinic_info, settings or {})
                _tools_mod._GLOBAL_SCHEDULE = load_schedule_from_settings(settings or {})
                _tools_mod._GLOBAL_CLINIC_TZ = clinic_tz
                # Fetch FAQ now that we have clinic_id
                clinic_faq = await _fetch_clinic_faq(clinic_info.get("id"))
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
    disconnect_event = asyncio.Event()

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

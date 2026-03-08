"""
Assistant tools for the dental AI agent.
"""

from __future__ import annotations
import re
import time
import asyncio
from typing import Optional, Dict, Any, Callable, cast
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import (
    DEFAULT_TZ,
    BOOKED_STATUSES,
    DEFAULT_PHONE_REGION,
    APPOINTMENT_BUFFER_MINUTES,
    supabase,
    logger,
)

from models.state import PatientState
from livekit.agents import llm

from utils.phone_utils import (
    _normalize_phone_preserve_plus,
    speakable_phone,
)
from utils.agent_flow import (
    build_time_parse_candidates,
    ensure_caller_phone_pending,
    looks_like_phone_input,
)
from services.database_service import is_slot_free_supabase, book_to_supabase
from services.scheduling_service import (
    get_duration_for_service,
    is_within_working_hours,
    get_next_available_slots,
    suggest_slots_around,
    WEEK_KEYS,
)
from services.appointment_management_service import (
    find_appointment_by_phone,
    cancel_appointment,
    reschedule_appointment,
)
from services.extraction_service import _iso
from utils.contact_utils import parse_datetime_natural


# ============================================================================
# Module-level globals (injected by agent.py)
# ============================================================================

_GLOBAL_CLINIC_TZ: str = DEFAULT_TZ
_GLOBAL_CLINIC_INFO: Optional[Dict[str, Any]] = None
_GLOBAL_AGENT_SETTINGS: Optional[Dict[str, Any]] = None
_REFRESH_AGENT_MEMORY: Optional[Callable[[], None]] = None
_GLOBAL_SCHEDULE: Optional[Dict[str, Any]] = None


def update_global_clinic_info(
    info: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    """Called by agent.py to inject the database context including timezone."""
    global _GLOBAL_CLINIC_INFO, _GLOBAL_AGENT_SETTINGS, _GLOBAL_CLINIC_TZ
    _GLOBAL_CLINIC_INFO = info or {}
    if settings:
        _GLOBAL_AGENT_SETTINGS = settings
    if info and info.get("timezone"):
        _GLOBAL_CLINIC_TZ = info["timezone"]
        logger.info(f"[TOOLS] Timezone updated to: {_GLOBAL_CLINIC_TZ}")


# ============================================================================
# Helpers
# ============================================================================

def _sanitize_tool_arg(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    return None if s.lower() in ("null", "none", "") else s


def email_for_speech(email: str) -> str:
    if not email:
        return "unknown"
    return email.replace("@", " at ").replace(".", " dot ")


def contact_phase_allowed(state: PatientState) -> bool:
    safety_check = (
        state.full_name is not None
        and state.time_status == "valid"
        and state.dt_local is not None
    )
    fallback_check = getattr(state, "contact_phase_started", False) is True
    caller_id_accepted = getattr(state, "caller_id_accepted", False) is True
    return safety_check or fallback_check or caller_id_accepted


def _refresh_memory():
    if _REFRESH_AGENT_MEMORY:
        try:
            _REFRESH_AGENT_MEMORY()
        except Exception:
            pass


# ============================================================================
# AssistantTools class
# ============================================================================

class AssistantTools:
    """Tool functions for the dental AI agent."""

    def __init__(self, state: PatientState):
        self.state = state
        self._refresh_memory: Optional[Callable[[], None]] = None
        # Optional async callback for speaking a final response directly from a tool,
        # bypassing the LLM re-generation step. Set by agent.py after session is ready.
        # Signature: async def _direct_say_callback(text: str) -> None
        self._direct_say_callback: Optional[Callable] = None

    @llm.function_tool(
        description=(
            "Save patient info. Handles: name, phone, email, reason, time_suggestion (natural language like 'March 10 at 3pm'). "
            "Checks availability automatically when time is given."
        )
    )
    async def update_patient_record(
        self,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        reason: Optional[str] = None,
        time_suggestion: Optional[str] = None,
    ) -> str:
        _t0 = time.perf_counter()
        state = self.state
        if not state:
            return "State not initialized."

        schedule = _GLOBAL_SCHEDULE or {}
        updates = []

        name = _sanitize_tool_arg(name)
        phone = _sanitize_tool_arg(phone)
        email = _sanitize_tool_arg(email)
        reason = _sanitize_tool_arg(reason)
        time_suggestion = _sanitize_tool_arg(time_suggestion)

        if name and state.full_name and name.strip().lower() == state.full_name.strip().lower():
            name = None
        if reason and state.reason and reason.strip().lower() == state.reason.strip().lower():
            reason = None

        # === NAME ===
        if name:
            state.full_name = name.strip().title()
            updates.append(f"name={state.full_name}")
            logger.info(f"[TOOL] Name: {state.full_name}")

        # === PHONE ===
        if phone and not (state.phone_confirmed and state.phone_e164):
            clinic_region = (_GLOBAL_CLINIC_INFO or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
            clean_phone, last4 = _normalize_phone_preserve_plus(phone, clinic_region)
            if clean_phone:
                state.phone_pending = str(clean_phone)
                state.phone_last4 = str(last4) if last4 else ""
                state.phone_confirmed = False
                state.phone_source = "user_spoken"
                updates.append(f"phone_pending=***{state.phone_last4}")
                logger.info(f"[TOOL] Phone pending: ***{state.phone_last4}")

        # === EMAIL ===
        if email:
            clean_email = email.replace(" ", "").lower()
            if "@" in clean_email and "." in clean_email:
                state.email = clean_email
                state.email_confirmed = False
                updates.append(f"email={state.email}")
                logger.info(f"[TOOL] Email: {state.email}")

        # === REASON ===
        if reason:
            state.reason = reason.strip()
            state.duration_minutes = get_duration_for_service(state.reason, schedule)
            updates.append(f"reason={state.reason} ({state.duration_minutes}m)")
            logger.info(f"[TOOL] Reason: {state.reason}, duration: {state.duration_minutes}m")

        # === TIME ===
        if time_suggestion:
            previous_dt_text = state.dt_text
            state.dt_text = time_suggestion.strip()
            state.time_status = "validating"
            logger.info(f"[TOOL] Checking time: {time_suggestion}")

            try:
                # If we have a saved date and the user gives a time-only string, combine them
                time_only_words = ["am", "pm", "o'clock", "oclock"]
                month_words = [
                    "january","february","march","april","may","june","july",
                    "august","september","october","november","december",
                    "tomorrow","today","monday","tuesday","wednesday",
                    "thursday","friday","saturday","sunday",
                ]
                suggestion_lower = time_suggestion.lower()
                has_date = any(m in suggestion_lower for m in month_words) or bool(re.search(r"\b\d{1,2}[/-]\d{1,2}\b", suggestion_lower))
                has_time = any(w in suggestion_lower for w in time_only_words) or bool(re.search(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b", suggestion_lower, re.IGNORECASE))

                parse_input = time_suggestion
                if not has_date and has_time and state.dt_text and state.dt_text != time_suggestion:
                    # User gave time-only (e.g. "3 PM") — combine with saved date text
                    combined = f"{state.dt_text} at {time_suggestion}"
                    logger.info(f"[TOOL] Combined date+time: '{combined}'")
                    parse_input = combined

                result = parse_datetime_natural(parse_input, tz_hint=_GLOBAL_CLINIC_TZ)
                recent_context = state.recent_user_context() if hasattr(state, "recent_user_context") else ""
                parse_candidates = build_time_parse_candidates(
                    time_suggestion,
                    recent_context=recent_context,
                    previous_text=previous_dt_text,
                )
                for candidate in parse_candidates:
                    if candidate == parse_input:
                        break
                    attempt = parse_datetime_natural(candidate, tz_hint=_GLOBAL_CLINIC_TZ)
                    if (
                        attempt.get("datetime") is not None
                        or attempt.get("date_only")
                        or attempt.get("needs_clarification")
                        or attempt.get("clarification_type")
                    ):
                        parse_input = candidate
                        result = attempt
                        logger.info(f"[TOOL] Time parse override: '{time_suggestion}' -> '{parse_input}'")
                        break

                # Handle date-only result (no time was specified by user)
                if result.get("date_only") or result.get("clarification_type") == "time_missing":
                    parsed_date = result.get("parsed_date") or (result["datetime"].date() if result.get("datetime") else None)
                    if parsed_date:
                        state.dt_text = parse_input
                        state.time_status = "pending"
                        day_spoken = parsed_date.strftime("%A, %B %d")
                        _refresh_memory()
                        return f"Got it, {day_spoken}. What time works best for you?"
                    return result.get("message", "Could you specify the time?")

                if result.get("needs_clarification"):
                    return result.get("message", "Could you specify the day?")

                parsed = result.get("datetime")

                if not parsed:
                    return "I didn't catch that time. Try something like 'tomorrow at 2pm' or 'March 15th at 3:30'."

                logger.info(f"[TOOL] Parsed '{time_suggestion}' → {parsed.isoformat()}")

                time_spoken = parsed.strftime("%I:%M %p").lstrip("0")
                day_spoken = parsed.strftime("%A")

                is_valid, error_msg = is_within_working_hours(parsed, schedule, state.duration_minutes)

                if is_valid:
                    clinic_id = str((_GLOBAL_CLINIC_INFO or {}).get("id") or "")
                    if clinic_id:
                        slot_end = parsed + timedelta(minutes=state.duration_minutes + APPOINTMENT_BUFFER_MINUTES)
                        clinic_info = _GLOBAL_CLINIC_INFO or {}
                        slot_free = await is_slot_free_supabase(
                            clinic_id,
                            parsed,
                            slot_end,
                            clinic_info=clinic_info,
                        )

                        if not slot_free:
                            state.time_status = "invalid"
                            state.time_error = "That slot is already taken"
                            state.add_rejected_slot(parsed, reason="slot_taken")
                            state.dt_local = None
                            state.slot_available = False

                            alternatives = await suggest_slots_around(
                                clinic_id=clinic_id,
                                requested_start_dt=parsed,
                                duration_minutes=state.duration_minutes,
                                schedule=schedule,
                                tz_str=_GLOBAL_CLINIC_TZ,
                                count=3,
                                window_hours=4,
                                step_min=15,
                            )
                            valid_alts = [a for a in alternatives if not state.is_slot_rejected(a)]

                            if valid_alts:
                                alt_strs = []
                                for alt in valid_alts:
                                    t = alt.strftime("%I:%M %p").lstrip("0")
                                    alt_strs.append(t if alt.date() == parsed.date() else f"{alt.strftime('%A')} at {t}")
                                if len(alt_strs) == 1:
                                    return f"Sorry, {time_spoken} is booked. The closest I have is {alt_strs[0]}. Would that work?"
                                elif len(alt_strs) == 2:
                                    return f"Sorry, {time_spoken} is booked. I can do {alt_strs[0]} or {alt_strs[1]}."
                                else:
                                    return f"Sorry, {time_spoken} is booked. I can do {alt_strs[0]}, {alt_strs[1]}, or {alt_strs[2]}."
                            return f"Sorry, {time_spoken} on {day_spoken} is booked and I don't see nearby openings. Try another day?"

                    # Slot is free (or no clinic_id to check)
                    state.dt_local = parsed
                    state.time_status = "valid"
                    state.time_error = None
                    state.slot_available = True
                    updates.append(f"time={parsed.strftime('%A, %B %d at %I:%M %p')}")
                    logger.info(f"[TOOL] Time confirmed available: {parsed.isoformat()}")

                    if state.full_name and state.dt_local:
                        state.contact_phase_started = True

                    # Caller ID flow
                    if contact_phase_allowed(state) and not state.phone_confirmed and not state.caller_id_checked:
                        phone_candidate = ensure_caller_phone_pending(state)
                        if phone_candidate:
                            state.pending_confirm = "phone"
                            state.pending_confirm_field = "phone"
                            state.caller_id_checked = True
                            phone_speak = speakable_phone(phone_candidate)
                            _refresh_memory()
                            return f"Perfect! {day_spoken} at {time_spoken} is open. Should I use the number you're calling from, {phone_speak}?"

                    _refresh_memory()
                    return f"Got it — {day_spoken} at {time_spoken} is available. Continue gathering remaining info."

                else:
                    state.time_status = "invalid"
                    state.time_error = error_msg
                    state.dt_local = None
                    clinic_id = str((_GLOBAL_CLINIC_INFO or {}).get("id") or "")
                    start_date = parsed.date() if parsed else datetime.now(ZoneInfo(_GLOBAL_CLINIC_TZ)).date()
                    alternatives = await get_next_available_slots(
                        clinic_id=clinic_id,
                        schedule=schedule,
                        tz_str=_GLOBAL_CLINIC_TZ,
                        duration_minutes=state.duration_minutes,
                        num_slots=2,
                        days_ahead=7,
                        start_from_date=start_date,
                    )
                    if alternatives:
                        times = [t.strftime("%I:%M %p").lstrip("0") for t in alternatives]
                        return (f"{error_msg} I have {times[0]}"
                                + (f" or {times[1]}" if len(times) > 1 else "")
                                + ". Would you like one of those?")
                    return f"{error_msg} Would you like to try another time?"

            except Exception as e:
                logger.error(f"[TOOL] Time validation error: {e!r}")
                state.time_status = "error"
                state.time_error = "schedule_unavailable"
                state.dt_local = None
                state.slot_available = False
                return "I'm having trouble checking the schedule. Could you try a different time?"

        # Caller ID flow after other updates
        if state.full_name and state.dt_local:
            state.contact_phase_started = True

        if contact_phase_allowed(state) and not state.phone_confirmed and not state.caller_id_checked:
            phone_candidate = ensure_caller_phone_pending(state)
            if phone_candidate and state.pending_confirm != "phone":
                state.pending_confirm = "phone"
                state.pending_confirm_field = "phone"
                state.caller_id_checked = True
                phone_speak = speakable_phone(phone_candidate)
                _refresh_memory()
                return f"Should I use the number you're calling from, {phone_speak}?"

        _refresh_memory()
        _ms = (time.perf_counter() - _t0) * 1000
        logger.info(f"[PERF] update_patient_record: {_ms:.0f}ms")

        return "Noted."

    @llm.function_tool(description="Confirm or reject caller's phone number. confirmed=True saves it, confirmed=False clears and re-asks.")
    async def confirm_phone(self, confirmed: bool, new_phone: Optional[str] = None, phone_number: Optional[str] = None) -> str:
        state = self.state

        if state.phone_confirmed and state.phone_e164 and confirmed and not new_phone and not phone_number:
            if state.full_name and state.dt_local and state.reason:
                return "Phone saved. All info complete. Book now."
            return "Phone saved."

        new_phone = _sanitize_tool_arg(phone_number) or _sanitize_tool_arg(new_phone)

        if new_phone:
            clinic_region = (_GLOBAL_CLINIC_INFO or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
            clean_phone, last4 = _normalize_phone_preserve_plus(new_phone, clinic_region)
            if clean_phone:
                existing_phone = state.phone_pending or state.detected_phone or state.phone_e164
                if (
                    existing_phone
                    and str(clean_phone) == str(existing_phone)
                    and not looks_like_phone_input(getattr(state, "last_user_text", None))
                ):
                    logger.warning("[TOOL] Ignoring synthetic phone update without caller phone input")
                    _refresh_memory()
                    return "Wait for the caller to answer whether to use the caller ID number."
                state.phone_pending = str(clean_phone)
                state.phone_last4 = str(last4) if last4 else ""
                state.phone_confirmed = False
                state.phone_source = "user_spoken"
                state.pending_confirm = "phone"
                state.pending_confirm_field = "phone"
                logger.info(f"[TOOL] Phone updated: ***{state.phone_last4}")
                _refresh_memory()
                return f"I have ***{state.phone_last4}. Is that right?"
            return f"Couldn't parse '{new_phone}'. Could you repeat the number?"

        if confirmed:
            phone_candidate = ensure_caller_phone_pending(state)
            if not phone_candidate:
                return "No phone number to confirm. Ask for the number first."
            state.phone_e164 = str(phone_candidate)
            state.phone_confirmed = True
            state.caller_id_accepted = True
            state.contact_phase_started = True
            state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
            logger.info(f"[TOOL] Phone confirmed: {state.phone_e164}")
            _refresh_memory()
            if state.full_name and state.dt_local and state.reason:
                return "Phone saved. All info complete. Book now."
            return "Phone saved."
        else:
            old = state.phone_pending or state.detected_phone or state.phone_e164
            state.phone_pending = None
            state.detected_phone = None
            state.phone_e164 = None
            state.phone_last4 = None
            state.phone_confirmed = False
            state.phone_source = None
            state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
            logger.info(f"[TOOL] Phone rejected (was {old})")
            _refresh_memory()
            return "Phone cleared. Ask: 'What number should I use instead?'"

    @llm.function_tool(description="Confirm or reject patient's email address.")
    async def confirm_email(self, confirmed: bool, email_address: Optional[str] = None) -> str:
        state = self.state
        if not state:
            return "State not initialized."

        email_address = _sanitize_tool_arg(email_address)
        if email_address:
            state.email = email_address.strip().lower()
            logger.info(f"[TOOL] Email saved: {state.email}")

        if state.email_confirmed:
            return "Email already confirmed."

        if not contact_phase_allowed(state):
            return "Confirm appointment time first."

        if confirmed:
            state.email_confirmed = True
            state.pending_confirm = None if state.pending_confirm == "email" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "email" else state.pending_confirm_field
            logger.info("[TOOL] Email confirmed")
            _refresh_memory()
            return "Email saved."
        else:
            state.email = None
            state.email_confirmed = False
            logger.info("[TOOL] Email rejected")
            _refresh_memory()
            return "Email cleared. Ask for the correct email."

    @llm.function_tool(
        description=(
            "Find open time slots. ONLY use when patient asks 'what times are available' or 'anytime works'. "
            "Do NOT use when patient gives a specific date/time — use update_patient_record(time_suggestion=...) instead, it checks availability automatically."
        )
    )
    async def get_available_slots_v2(
        self,
        after_datetime: Optional[str] = None,
        preferred_day: Optional[str] = None,
        num_slots: int = 3,
    ) -> str:
        _t0 = time.perf_counter()
        state = self.state
        clinic_info = _GLOBAL_CLINIC_INFO
        schedule = _GLOBAL_SCHEDULE or {}

        if not clinic_info:
            return "I'm having trouble accessing the schedule right now."

        duration = state.duration_minutes if state else 60
        tz = ZoneInfo(_GLOBAL_CLINIC_TZ)
        now = datetime.now(tz)

        after_datetime = _sanitize_tool_arg(after_datetime)
        preferred_day = _sanitize_tool_arg(preferred_day)

        search_start = now
        if after_datetime:
            try:
                result = parse_datetime_natural(after_datetime, tz_hint=_GLOBAL_CLINIC_TZ)
                if result.get("success") and result.get("datetime"):
                    search_start = result["datetime"]
            except Exception as e:
                logger.warning(f"[TOOL] Could not parse after_datetime '{after_datetime}': {e}")

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
            slot_step = schedule.get("slot_step_minutes", 30)
            minutes_to_add = slot_step - (search_start.minute % slot_step)
            if minutes_to_add == slot_step:
                minutes_to_add = 0
            current = search_start.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)

            if target_weekday is not None and current.weekday() != target_weekday:
                days_until = (target_weekday - current.weekday()) % 7
                if days_until == 0:
                    days_until = 7
                current = datetime.combine(
                    current.date() + timedelta(days=days_until),
                    datetime.min.time(), tzinfo=tz
                ).replace(hour=9, minute=0)

            end_search = max(now + timedelta(days=14), search_start + timedelta(days=14))

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
                        appt_dict = cast(Dict[str, Any], appt)
                        appt_start = datetime.fromisoformat(str(appt_dict["start_time"]).replace("Z", "+00:00"))
                        appt_end = datetime.fromisoformat(str(appt_dict["end_time"]).replace("Z", "+00:00"))
                        existing_appointments.append((appt_start, appt_end))
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[SLOTS_V2] Failed to fetch appointments: {e}")

            available_slots = []
            lunch_skipped = False

            while current < end_search and len(available_slots) < num_slots:
                if target_weekday is not None and current.weekday() != target_weekday:
                    days_until = (target_weekday - current.weekday()) % 7
                    if days_until == 0:
                        days_until = 7
                    current = datetime.combine(
                        current.date() + timedelta(days=days_until),
                        datetime.min.time(), tzinfo=tz
                    ).replace(hour=9, minute=0)
                    if current >= end_search:
                        break

                is_valid, error_msg = is_within_working_hours(current, schedule, duration)
                if not is_valid and "lunch" in error_msg.lower():
                    lunch_skipped = True

                if is_valid:
                    slot_end = current + timedelta(minutes=duration + APPOINTMENT_BUFFER_MINUTES)
                    is_free = True
                    for appt_start, appt_end in existing_appointments:
                        if current < (appt_end + timedelta(minutes=APPOINTMENT_BUFFER_MINUTES)) and slot_end > appt_start:
                            is_free = False
                            break
                    if is_free:
                        available_slots.append(current)

                current += timedelta(minutes=slot_step)

                dow_key = WEEK_KEYS[current.weekday()]
                intervals = schedule.get("working_hours", {}).get(dow_key, [])
                if intervals:
                    last_interval = intervals[-1]
                    try:
                        eh, em = map(int, last_interval["end"].split(":"))
                        day_end = current.replace(hour=eh, minute=em)
                        if current >= day_end:
                            next_day = current.date() + timedelta(days=1)
                            current = datetime.combine(next_day, datetime.min.time(), tzinfo=tz).replace(hour=9, minute=0)
                    except Exception:
                        pass

            if state:
                available_slots = [s for s in available_slots if not state.is_slot_rejected(s)]

            if not available_slots:
                constraint = f" after {after_datetime}" if after_datetime else ""
                if preferred_day:
                    constraint += f" on {preferred_day}"
                return f"I don't see any openings{constraint} in the next week. Try another day?"

            slot_strings = []
            for slot in available_slots:
                t = slot.strftime("%I:%M %p").lstrip("0")
                today = now.date()
                if slot.date() == today:
                    slot_strings.append(f"today at {t}")
                elif slot.date() == today + timedelta(days=1):
                    slot_strings.append(f"tomorrow at {t}")
                else:
                    slot_strings.append(f"{slot.strftime('%A')} at {t}")

            prefix = "Okay, skipping the lunch hour. " if lunch_skipped else "I found some times. "
            if len(slot_strings) == 1:
                return f"{prefix}The next available is {slot_strings[0]}."
            elif len(slot_strings) == 2:
                return f"{prefix}I have {slot_strings[0]} or {slot_strings[1]}."
            else:
                return f"{prefix}I have {slot_strings[0]}, {slot_strings[1]}, or {slot_strings[2]}. Which works best?"

        except Exception as e:
            logger.error(f"[TOOL] get_available_slots_v2 error: {e}")
            return "I'm having trouble with the schedule. Let me try that again."
        finally:
            _ms = (time.perf_counter() - _t0) * 1000
            logger.info(f"[PERF] get_available_slots_v2: {_ms:.0f}ms")

    @llm.function_tool(description="Book the appointment after all info is confirmed by the patient.")
    async def confirm_and_book_appointment(self) -> str:
        _t0 = time.perf_counter()
        state = self.state
        clinic_info = _GLOBAL_CLINIC_INFO

        if not state or not clinic_info:
            return "Sorry, I'm missing clinic details. Could you call back in a moment?"

        if getattr(state, "appointment_booked", False):
            dt = state.dt_local
            if dt:
                day = dt.strftime("%A, %B %d")
                time_str = dt.strftime("%I:%M %p").lstrip("0")
                return f"You're already booked for {state.reason or 'your appointment'} on {day} at {time_str}."
            return "You're already booked."

        if not state.full_name or not state.dt_local or not state.phone_e164 or not state.phone_confirmed:
            missing = []
            if not state.full_name: missing.append("name")
            if not state.dt_local: missing.append("appointment time")
            if not state.phone_e164 or not state.phone_confirmed: missing.append("confirmed phone number")
            return f"I still need: {', '.join(missing)}. Let me get those first."

        logger.info(f"[BOOK] Starting Supabase insert for {state.full_name}")
        try:
            appt_id = await book_to_supabase(clinic_info, patient_state=state, calendar_event_id=None)
        except Exception as e:
            logger.error(f"[BOOK] Supabase failed: {e!r}")
            appt_id = None

        if not appt_id:
            return "I'm having trouble saving the appointment. Could you try again?"

        state.appointment_booked = True
        state.booking_confirmed = True
        state.appointment_id = appt_id

        dt = state.dt_local
        day = dt.strftime("%A, %B %d")
        time_str = dt.strftime("%I:%M %p").lstrip("0")
        phone_complete = state.phone_e164 or state.phone_pending

        if phone_complete:
            if getattr(state, "prefers_sms", False):
                phone_part = f" We'll send your confirmation via SMS to {speakable_phone(phone_complete)}."
            else:
                phone_part = f" We'll send your confirmation on WhatsApp to {speakable_phone(phone_complete)}. If you don't have WhatsApp on this number, let me know and we'll send an SMS instead."
        else:
            phone_part = ""

        _ms = (time.perf_counter() - _t0) * 1000
        logger.info(f"[PERF] confirm_and_book: {_ms:.0f}ms, appt_id={appt_id}")

        message = (
            f"Perfect, {state.full_name} — you're all set for {state.reason or 'your appointment'} "
            f"on {day} at {time_str}.{phone_part} We look forward to seeing you!"
        )
        return message

    @llm.function_tool(description="Find existing appointment for cancel/reschedule. Call silently when user mentions cancelling or rescheduling.")
    async def find_existing_appointment(self) -> str:
        state = self.state
        clinic_info = _GLOBAL_CLINIC_INFO

        if not clinic_info:
            return "I'm having trouble accessing the system right now."

        phone_to_search = state.phone_e164 or state.phone_pending or state.detected_phone
        if isinstance(phone_to_search, tuple):
            phone_to_search = phone_to_search[0] if phone_to_search else None

        if not phone_to_search:
            return "I don't have a phone number to search with. What number did you use when booking?"

        logger.info(f"[APPT_LOOKUP] Searching for ***{phone_to_search[-4:]}")

        appointment = await find_appointment_by_phone(
            clinic_id=clinic_info["id"],
            phone_number=phone_to_search,
            tz_str=_GLOBAL_CLINIC_TZ
        )

        if appointment:
            state.found_appointment_id = appointment["id"]
            state.found_appointment_details = appointment
            start_time = appointment["start_time"]
            day = start_time.strftime("%A, %B %d")
            time_str = start_time.strftime("%I:%M %p").lstrip("0")
            logger.info(f"[APPT_LOOKUP] Found id={appointment['id']}")
            return f"I found your {appointment['reason']} appointment on {day} at {time_str}. Is this the one you'd like to modify?"
        else:
            logger.info(f"[APPT_LOOKUP] No appointment found for ***{phone_to_search[-4:]}")
            return "I don't see an upcoming appointment with that number. What number did you use when booking?"

    @llm.function_tool(description="Cancel found appointment after explicit user confirmation.")
    async def cancel_appointment_tool(self, confirmed: bool = False) -> str:
        state = self.state
        appointment_id = getattr(state, "found_appointment_id", None)
        appointment = getattr(state, "found_appointment_details", None)

        if not appointment_id or not appointment:
            return "I need to find your appointment first. Let me search using your phone number."

        if not confirmed:
            start_time = appointment["start_time"]
            day = start_time.strftime("%A, %B %d")
            time_str = start_time.strftime("%I:%M %p").lstrip("0")
            return f"Just to confirm — cancel your {appointment['reason']} on {day} at {time_str}?"

        logger.info(f"[CANCEL] Cancelling appointment id={appointment_id}")
        success = await cancel_appointment(appointment_id=appointment_id, reason="user_requested")

        if success:
            start_time = appointment["start_time"]
            day = start_time.strftime("%A, %B %d")
            time_str = start_time.strftime("%I:%M %p").lstrip("0")
            state.found_appointment_id = None
            state.found_appointment_details = None
            logger.info(f"[CANCEL] Cancelled id={appointment_id}")
            return f"Done — your {appointment['reason']} on {day} at {time_str} has been cancelled. Anything else?"
        else:
            return "I'm having trouble cancelling that appointment. Would you like to speak with the office?"

    @llm.function_tool(description="Reschedule found appointment to a new time after explicit user confirmation of both old and new times.")
    async def reschedule_appointment_tool(self, new_time: Optional[str] = None, confirmed: bool = False) -> str:
        state = self.state
        appointment_id = getattr(state, "found_appointment_id", None)
        appointment = getattr(state, "found_appointment_details", None)

        if not appointment_id or not appointment:
            return "I need to find your appointment first."

        if not new_time:
            return "Do you have a specific time in mind, or would you like me to suggest some options?"

        new_time = _sanitize_tool_arg(new_time)
        if not new_time:
            return "What time would work better for you?"

        try:
            parsed_result = parse_datetime_natural(new_time, tz_hint=_GLOBAL_CLINIC_TZ)
            parsed_new_time = parsed_result.get("datetime") if isinstance(parsed_result, dict) else parsed_result

            if not parsed_new_time:
                return f"I couldn't understand '{new_time}'. Could you try again?"

            clinic_info = _GLOBAL_CLINIC_INFO
            if not clinic_info:
                return "I'm having trouble accessing the system."

            schedule = _GLOBAL_SCHEDULE or {}
            duration = appointment.get("duration_minutes", 60)

            is_valid, error_msg = is_within_working_hours(parsed_new_time, schedule, duration)
            if not is_valid:
                return f"{error_msg} Would you like me to suggest available times?"

            end_time = parsed_new_time + timedelta(minutes=duration + APPOINTMENT_BUFFER_MINUTES)
            slot_free = await is_slot_free_supabase(
                clinic_id=clinic_info["id"],
                start_dt=parsed_new_time,
                end_dt=end_time,
                clinic_info=clinic_info
            )

            if not slot_free:
                alternatives = await suggest_slots_around(
                    clinic_id=clinic_info["id"],
                    requested_start_dt=parsed_new_time,
                    duration_minutes=duration,
                    schedule=schedule,
                    tz_str=_GLOBAL_CLINIC_TZ,
                    count=3,
                    window_hours=4,
                    step_min=15,
                )
                if alternatives:
                    alt_strs = []
                    for alt in alternatives:
                        t = alt.strftime("%I:%M %p").lstrip("0")
                        alt_strs.append(t if alt.date() == parsed_new_time.date() else f"{alt.strftime('%A')} at {t}")
                    if len(alt_strs) == 1:
                        return f"That slot is taken. The closest I have is {alt_strs[0]}. Would that work?"
                    elif len(alt_strs) == 2:
                        return f"That slot is booked. I can do {alt_strs[0]} or {alt_strs[1]}."
                    else:
                        return f"That time is taken. I have {alt_strs[0]}, {alt_strs[1]}, or {alt_strs[2]}."
                return "That slot is taken and I don't see nearby openings. Try a different day?"

            old_start = appointment["start_time"]
            old_day = old_start.strftime("%A, %B %d")
            old_time_str = old_start.strftime("%I:%M %p").lstrip("0")
            new_day = parsed_new_time.strftime("%A, %B %d")
            new_time_str = parsed_new_time.strftime("%I:%M %p").lstrip("0")

            if not confirmed:
                return (f"Just to confirm — move your {appointment['reason']} from {old_day} at {old_time_str} "
                        f"to {new_day} at {new_time_str}?")

            end_time = parsed_new_time + timedelta(minutes=duration)
            success = await reschedule_appointment(
                appointment_id=appointment_id,
                new_start_time=parsed_new_time,
                new_end_time=end_time,
            )

            if success:
                state.found_appointment_id = None
                state.found_appointment_details = None
                logger.info(f"[RESCHEDULE] Rescheduled id={appointment_id}")
                return f"All set — moved to {new_day} at {new_time_str}. Anything else?"
            else:
                return "I'm having trouble rescheduling. Would you like to speak with the office?"

        except Exception as e:
            logger.error(f"[RESCHEDULE] Error: {e}")
            return "I'm having trouble with that. Would you like to speak with the office directly?"

    @llm.function_tool(description="End the call when the user says goodbye or conversation is complete.")
    async def end_conversation(self) -> str:
        if self.state:
            self.state.call_ended = True
            if self.state.booking_confirmed:
                logger.info("[CALL_END] Call ending after successful booking")
            else:
                logger.info("[CALL_END] Call ending at user request")
        return "Goodbye! Have a great day."

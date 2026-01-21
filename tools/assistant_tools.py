"""
Assistant tools for the dental AI agent.
"""

from __future__ import annotations
from typing import Optional


# ============================================================================
# Extracted: Global variables
# ============================================================================

# Global state reference for tool access
_GLOBAL_STATE: Optional["PatientState"] = None
_GLOBAL_CLINIC_TZ: str = DEFAULT_TZ
_GLOBAL_CLINIC_INFO: Optional[dict] = None  # For booking tool access
_GLOBAL_AGENT_SETTINGS: Optional[dict] = None  # For DB-backed OAuth token refresh
_REFRESH_AGENT_MEMORY: Optional[Callable[[], None]] = None  # Callback to refresh LLM system prompt
_GLOBAL_SCHEDULE: Optional[Dict[str, Any]] = None  # Scheduling config (working hours, lunch, durations)



# ============================================================================
# Extracted: APPOINTMENT_BUFFER_MINUTES
# ============================================================================




# ============================================================================
# Extracted: AssistantTools class with all methods
# ============================================================================

class AssistantTools(llm.FunctionContext):
    def __init__(self, state: PatientState):
        super().__init__()
        self.state = state
    
    @llm.ai_callable(description="""
    Update the patient record with any information heard during conversation.
    Call this IMMEDIATELY when you hear: name, phone, email, reason for visit, or preferred time.
    You can call this multiple times as you gather information.
    For phone: normalize spoken numbers (e.g., 'six seven nine' → '679').
    For email: normalize spoken format (e.g., 'moiz six seven nine at gmail dot com' → 'moiz679@gmail.com').
    For time: pass natural language (e.g., 'tomorrow at 2pm', 'next Monday morning').
    
    NEARBY SLOTS: If the requested time is TAKEN, this tool automatically finds and returns 
    nearby alternatives (e.g., "9:00 AM or 11:30 AM"). Simply offer these to the patient!
    """)
    async def update_patient_record(self, 
    name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    reason: Optional[str] = None,
    time_suggestion: Optional[str] = None,
) -> str:
        """
        Update the internal patient record with extracted information.
        
        Enhanced with:
        - Duration lookup from treatment_durations config
        - Time validation against working hours and lunch breaks
        - Helpful error messages with alternative time suggestions
        """
        # global state removed, _GLOBAL_CLINIC_TZ, _GLOBAL_SCHEDULE
        
        state = self.state
        if not state:
            return "State not initialized."
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 🔒 IDEMPOTENCY CHECK — Prevent double-execution in same turn
        # ═══════════════════════════════════════════════════════════════════════════
        if state.check_tool_lock("update_patient_record", locals()):
            return "Information already noted."
        
        schedule = _GLOBAL_SCHEDULE or {}
        updates = []
        errors = []
        
        # ═══════════════════════════════════════════════════════════════════════════
        # 🛡️ INPUT SANITIZATION — Gracefully handle None, empty, or "null" values
        # ═══════════════════════════════════════════════════════════════════════════
        def _sanitize_input(value: Optional[str]) -> Optional[str]:
            return _sanitize_tool_arg(value)
        
        # Sanitize all inputs before processing
        name = _sanitize_input(name)
        phone = _sanitize_input(phone)
        email = _sanitize_input(email)
        reason = _sanitize_input(reason)
        time_suggestion = _sanitize_input(time_suggestion)
        
        # === NAME ===
        if name and state.should_update_field("name", state.full_name, name):
            state.full_name = name.strip().title()
            updates.append(f"name={state.full_name}")
            logger.info(f"[TOOL] ✓ Name captured: {state.full_name}")
        
        # === PHONE ===
        # Only update if explicitly corrected or captured for the first time
        if phone:
            # CONTEXT-AWARE VERIFICATION CHECK (Fix #3)
            # If we just captured phone, treat fragments/confirmations as verification, NOT new data
            if state.awaiting_slot_confirmation and state.last_captured_slot == "phone" and state.last_captured_phone:
                action, reason = interpret_followup_for_slot("phone", state.last_captured_phone, phone)
                
                if action in ["FRAGMENT", "CONFIRM"]:
                    state.slot_confirm_turns_left = max(0, state.slot_confirm_turns_left - 1)
                    logger.info(f"[TOOL] 🛡️ Validating phone... Ignoring overwrite '{phone}' ({reason})")
                    state.phone_verification_buffer += f" | {phone}"
                    # If confirmed, we could mark confirmed=True, but let's let confirm_phone tool handle explicit logic if needed.
                    # However, usually we just want to NOT wipe the state.
                    return f"Got it. I have your number ending in {state.phone_last4 or '...'}. Is that right?"
    
            if state.should_update_field("phone", state.phone_pending or state.phone_e164, phone):
                
                # Use proper normalization with region awareness
                clinic_region = (_GLOBAL_CLINIC_INFO or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
                clean_phone, last4 = _normalize_phone_preserve_plus(phone, clinic_region)
                
                if clean_phone:
                    state.phone_pending = str(clean_phone)
                    state.phone_last4 = str(last4) if last4 else ""
                    # Safety guard: ensure no tuple was stored
                    _ensure_phone_is_string(state)
                    # NEVER auto-confirm phone - always require explicit user confirmation
                    state.phone_confirmed = False
                    state.phone_source = "user_spoken"
                    
                    # ENTER VERIFICATION MODE
                    state.awaiting_slot_confirmation = True
                    state.last_captured_slot = "phone"
                    state.last_captured_phone = state.phone_pending or state.phone_e164
                    state.slot_confirm_turns_left = 2
                    state.phone_verification_buffer = ""
                    logger.info(f"[SLOT_CONFIRM] 🏁 Entering PHONE verification mode (2 turns)")
    
                    updates.append(f"phone_pending=***{state.phone_last4}")
                    logger.info(f"[TOOL] ⏳ Phone captured (pending confirmation): ***{state.phone_last4}")
        
        # === EMAIL ===
        if email:
            # CONTEXT-AWARE VERIFICATION CHECK (Fix #3)
            if state.awaiting_slot_confirmation and state.last_captured_slot == "email" and state.last_captured_email:
                action, reason = interpret_followup_for_slot("email", state.last_captured_email, email)
                
                if action in ["FRAGMENT", "CONFIRM"]:
                    state.slot_confirm_turns_left = max(0, state.slot_confirm_turns_left - 1)
                    logger.info(f"[TOOL] 🛡️ Validating email... Ignoring overwrite '{email}' ({reason})")
                    state.email_verification_buffer += f" | {email}"
                    return f"Understood. I have your email as {email_for_speech(state.last_captured_email)}."
    
            if state.should_update_field("email", state.email, email):
                clean_email = email.replace(" ", "").lower()
                if "@" in clean_email and "." in clean_email:
                    state.email = clean_email
                    state.email_confirmed = False  # NEVER auto-confirm - always require explicit confirmation
                    
                    # ENTER VERIFICATION MODE
                    state.awaiting_slot_confirmation = True
                    state.last_captured_slot = "email"
                    state.last_captured_email = state.email
                    state.slot_confirm_turns_left = 2
                    state.email_verification_buffer = ""
                    logger.info(f"[SLOT_CONFIRM] 🏁 Entering EMAIL verification mode (2 turns)")
    
                    updates.append(f"email_pending={state.email}")
                    logger.info(f"[TOOL] ⏳ Email captured (pending confirmation): {state.email}")
                    # Only prompt confirmation if contact phase has started AND not already confirmed
                    if state.contact_phase_started and not state.email_confirmed:
                        state.pending_confirm = "email"
                        state.pending_confirm_field = "email"
                        return f"Email captured as {email_for_speech(state.email)}. Please confirm: 'Is your email {email_for_speech(state.email)}?'"
                # Otherwise, silently store - will confirm later in contact phase
        
        # === REASON (with duration lookup) ===
        if reason and state.should_update_field("reason", state.reason, reason):
            state.reason = reason.strip()
            # Lookup duration from treatment_durations config
            state.duration_minutes = get_duration_for_service(state.reason, schedule)
            updates.append(f"reason={state.reason} (duration: {state.duration_minutes}m)")
            logger.info(f"[TOOL] ✓ Reason captured: {state.reason}, duration: {state.duration_minutes}m")
        
        # === TIME (with validation and availability check) ===
        if time_suggestion and state.should_update_field("time", state.dt_text or state.dt_local, time_suggestion):
            # 1. Capture user intent first
            state.dt_text = time_suggestion.strip()
            state.time_status = "validating"
            
            # Log the narrative check starting
            logger.info(f"[TOOL] ⏰ Checking time: {time_suggestion}...")
            
            try:
                # FIX 4: Date Locking Rule
                # Detect correction intent to allow overrides
                is_correction = has_correction_intent(time_suggestion)
                allow_date_override = not state.date_confirmed or is_correction
    
                # Use parse_datetime_natural for robust relative date handling
                # Handles "tomorrow at 3:30 PM", "next Monday", etc. correctly
                parsed = parse_datetime_natural(time_suggestion, tz_hint=_GLOBAL_CLINIC_TZ)
                
                if parsed:
                    # 2. REJECTION CHECK: If user is repeating a rejected time?
                    # Usually we trust the user if they insist, but let's log it.
                    if state.is_slot_rejected(parsed):
                         logger.warning(f"[TOOL] User requested previously rejected slot: {parsed}")
    
                    # FIX 4: Date Logic
                    date_input = parsed.date()
                    time_input = parsed.time()
                    
                    # Heuristic: Did input contain a date?
                    # If parse_datetime_natural changed the date from TODAY/TOMORROW default, or user used date keywords
                    has_date_keywords = any(w in time_suggestion.lower() for w in [
                        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                        "tomorrow", "today", "next week", "following", "date"
                    ])
                    # Also check digits like "23rd" or "1/23"
                    has_date_digits = bool(re.search(r"\d+(?:st|nd|rd|th)|/\d+", time_suggestion))
                    input_has_date = has_date_keywords or has_date_digits
    
                    # Heuristic: Did input contain a time?
                    input_has_time = bool(re.search(r"\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?|\b(?:noon|midnight|morning|afternoon|evening)\b", time_suggestion, re.IGNORECASE))
    
                    # If date is confirmed and NO correction intent, frame the parsed time on the confirmed date
                    if state.date_confirmed and not allow_date_override:
                         # Check if parsed date implies a change (e.g. user said "tomorrow" but confirm date is "Jan 23")
                         # If input DOES NOT have explicit date, we FORCE the confirmed date
                         if not input_has_date:
                             # Determine the date to use - prefer dt_local, then proposed_date, fallback to parsed date
                             use_date = state.dt_local.date() if state.dt_local else (state.proposed_date or parsed.date())
                             parsed = datetime.combine(use_date, time_input)
                             parsed = parsed.replace(tzinfo=ZoneInfo(_GLOBAL_CLINIC_TZ))
                             logger.info(f"[DATE] Override blocked (date_confirmed=True). Forcing date: {parsed.date()}")
                         else:
                             logger.info(f"[DATE] Date mentioned ({time_suggestion}) but date_confirmed=True. Allowing check.")
                    
                    # Update Candidate logic
                    if input_has_date and input_has_time:
                        # Date + Time together -> Confirm Date
                        state.proposed_date = parsed.date()
                        state.date_confirmed = True
                        state.date_source = "date_time_together"
                        logger.info(f"[DATE] Date confirmed because date+time provided together: {state.proposed_date}")
                    
                    elif input_has_date and not input_has_time:
                        # Date only -> Candidate
                        if allow_date_override:
                            state.proposed_date = parsed.date()
                            state.date_confirmed = False
                            state.date_source = "inferred"
                            logger.info(f"[DATE] Candidate date set (not confirmed): {state.proposed_date}")
                            # If date only, return confirmation question for date
                            day_str = parsed.strftime("%A, %B %d")
                            return f"Just to confirm, you want to come in on {day_str}?"
                    
                    elif not input_has_date and input_has_time:
                        # Time only -> Use existing proposed/confirmed date
                        if state.proposed_date:
                            parsed = datetime.combine(state.proposed_date, time_input)
                            parsed = parsed.replace(tzinfo=ZoneInfo(_GLOBAL_CLINIC_TZ))
                            logger.info(f"[DATE] Using proposed date {state.proposed_date} with new time {time_input}")
                        elif state.dt_local:
                            # Use previous dt_local date
                            parsed = datetime.combine(state.dt_local.date(), time_input)
                            parsed = parsed.replace(tzinfo=ZoneInfo(_GLOBAL_CLINIC_TZ))
    
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
                                # Add rejection to history
                                state.add_rejected_slot(parsed, reason="slot_taken")
                                
                                # PROTECTIVE CLEAR: Clear dt_local but preserve everything else
                                state.dt_local = None
                                state.assert_integrity("time_rejection_taken")
                                
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
                                
                                # Filter out previously rejected slots from alternatives
                                valid_alternatives = [a for a in alternatives if not state.is_slot_rejected(a)]
                                
                                if len(valid_alternatives) < len(alternatives):
                                    logger.info(f"[TOOL] Filtered out {len(alternatives) - len(valid_alternatives)} rejected alternatives")
    
                                logger.info(f"[TOOL] Found {len(valid_alternatives)} valid alternatives around {parsed.strftime('%H:%M')}")
                                
                                if valid_alternatives:
                                    # Format alternatives for speech - include date if different from requested
                                    alt_descriptions = []
                                    for alt in valid_alternatives:
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
                        state.slot_available = True  # CRITICAL: Mark slot as confirmed available
                        time_formatted = parsed.strftime('%B %d at %I:%M %p')
                        updates.append(f"time={time_formatted} ({state.duration_minutes}m slot)")
                        logger.info(f"[TOOL] ✓ Time validated and available: {parsed.isoformat()}")
    
                        # Start contact phase only after name + valid time + slot available
                        if state.full_name and state.dt_local and state.slot_available:
                            state.contact_phase_started = True
    
                        # If we have a detected/pending phone now, ask for confirmation (last 4 only)
                        if contact_phase_allowed(state) and not state.phone_confirmed:
                            if not state.phone_pending and state.detected_phone:
                                state.phone_pending = state.detected_phone
                            if state.phone_pending and state.phone_last4:
                                state.pending_confirm = "phone"
                                state.pending_confirm_field = "phone"
                                return f"... ah, perfect! {day_spoken} at {time_spoken} is open. I have a number ending in {state.phone_last4} — is that okay?"
    
                        # Sonic-3 prosody: breathy confirmation with ellipses
                        # CRITICAL: Phrase forces LLM to understand booking is NOT complete yet
                        return f"... ah, perfect! {day_spoken} at {time_spoken} is open and I've noted that. I'll book it for you once we finish the rest of the details."
                    else:
                        # Time is invalid (lunch, after-hours, holiday)
                        state.time_status = "invalid"
                        state.time_error = error_msg
                        state.dt_local = None  # Don't save invalid time
                        
                        logger.warning(f"[TOOL] ✗ Time rejected: {error_msg}")
                        # Assert integration to catch partial wipe bugs
                        state.assert_integrity("time_invalid_logic")
                        
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
                    # Parsing failed or None result (e.g. user said "No" or "Cancel")
                    state.time_status = "pending"
                    updates.append(f"time_text={time_suggestion}")
                    
            except Exception as e:
                logger.warning(f"[TOOL] Time parse deferred: {time_suggestion} ({e})")
                state.time_status = "pending"
                updates.append(f"time_text={time_suggestion}")
        
        # Start contact phase only after name + valid time + slot available
        if state.full_name and state.dt_local and state.slot_available:
            state.contact_phase_started = True
    
        # If we have a detected/pending phone and contact phase is active, prompt confirmation (last 4 only)
        if contact_phase_allowed(state) and not state.phone_confirmed:
            if not state.phone_pending and state.detected_phone:
                state.phone_pending = state.detected_phone
            if state.phone_pending and state.phone_last4 and state.pending_confirm != "phone":
                state.pending_confirm = "phone"
                state.pending_confirm_field = "phone"
                if _REFRESH_AGENT_MEMORY:
                    try:
                        _REFRESH_AGENT_MEMORY()
                    except Exception:
                        pass
                return f"I have a number ending in {state.phone_last4} — is that okay?"
    
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
    
    

    @llm.ai_callable(description="""
    Get the next available appointment slots. Call this when:
    - The patient asks 'when is the next opening?'
    - A requested time is unavailable and you need alternatives
    - You want to proactively suggest times
    
    Returns the next 3 available time slots based on the patient's service duration.
    """)
    async def get_available_slots(self, ) -> str:
        """
        Find and return the next available appointment slots.
        Uses Supabase as primary source for speed.
        """
        # global state removed, _GLOBAL_CLINIC_INFO, _GLOBAL_SCHEDULE, _GLOBAL_CLINIC_TZ
        
        state = self.state
        
        # IDEMPOTENCY CHECK
        if state and state.check_tool_lock("get_available_slots", locals()):
            return "Checking..."
            
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
    
    

    APPOINTMENT_BUFFER_MINUTES = 15
    @llm.ai_callable(description="""
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
    async def get_available_slots_v2(self, 
    after_datetime: Optional[str] = None,
    preferred_day: Optional[str] = None,
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
        # global state removed, _GLOBAL_CLINIC_INFO, _GLOBAL_SCHEDULE, _GLOBAL_CLINIC_TZ
        
        state = self.state
        
        # IDEMPOTENCY CHECK
        if state and state.check_tool_lock("get_available_slots_v2", locals()):
            return "Checking constraints..."
            
        clinic_info = _GLOBAL_CLINIC_INFO
        schedule = _GLOBAL_SCHEDULE or {}
        
        if not clinic_info:
            return "... hmm, I'm having trouble accessing the schedule. Let me try again."
        
        duration = state.duration_minutes if state else 60
        tz = ZoneInfo(_GLOBAL_CLINIC_TZ)
        now = datetime.now(tz)
        
        # Sanitize optional args
        after_datetime = _sanitize_tool_arg(after_datetime)
        preferred_day = _sanitize_tool_arg(preferred_day)
    
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
            
            # FILTER: Exclude rejected slots
            available_slots = [
                s for s in available_slots if not (state and state.is_slot_rejected(s))
            ]
            if not available_slots:
                 return "... hmm, I don't see any other openings in that time window. Would you like to try a different day?"
    
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
    
    

    @llm.ai_callable(description="""
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
    async def find_relative_slots(self, 
    start_search_time: Optional[str] = None,
    limit_to_day: Optional[str] = None,
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
        # global state removed, _GLOBAL_CLINIC_INFO, _GLOBAL_SCHEDULE, _GLOBAL_CLINIC_TZ
        
        state = self.state
        
        # IDEMPOTENCY CHECK
        if state and state.check_tool_lock("find_relative_slots", locals()):
            return "Searching..."
            
        clinic_info = _GLOBAL_CLINIC_INFO
        schedule = _GLOBAL_SCHEDULE or {}
        
        if not clinic_info:
            return "... hmm, I'm having trouble accessing the schedule. Let me try again."
        
        duration = state.duration_minutes if state else 60
        tz = ZoneInfo(_GLOBAL_CLINIC_TZ)
        now = datetime.now(tz)
        
        # Sanitize optional args
        start_search_time = _sanitize_tool_arg(start_search_time)
        limit_to_day = _sanitize_tool_arg(limit_to_day)
    
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
            
            # FILTER: Exclude rejected slots
            original_count = len(available_slots)
            available_slots = [
                s for s in available_slots if not (state and state.is_slot_rejected(s))
            ]
            if not available_slots:
                 return "... hmm, I don't see any other times that typically work. Maybe try a different day?"
    
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
    
    

    @llm.ai_callable(description="""
    Confirm the phone number with the patient. Call this ONLY after the contact phase is started.
    Example: "I have a number ending in 7839 — is that okay?"
    
    IMPORTANT: When user says "yes", "yeah", "correct" etc., call confirm_phone(confirmed=True).
    When user says "no", "wrong", "incorrect", call confirm_phone(confirmed=False).
    
    SMART CAPTURE: You can also pass a phone_number to save it before confirming.
    Example: confirm_phone(confirmed=True, phone_number="+923351234567") saves AND confirms in one call.
    """)
    async def confirm_phone(self, confirmed: bool, new_phone: Optional[str] = None, phone_number: Optional[str] = None) -> str:
        """
        Mark phone as confirmed or update with correction.
        
        Args:
            confirmed: True if user confirmed the phone is correct
            new_phone: Optional new phone number if user provided a correction
        """
        # global state removed
        state = self.state
        if not state:
            return "State not initialized."
    
        # IDEMPOTENCY CHECK
        if state.check_tool_lock("confirm_phone", locals()):
            return "Phone confirmation noted."
        
        # Gate: Do NOT confirm phone if contact phase hasn't started
        if not contact_phase_allowed(state):
            logger.debug("[TOOL] confirm_phone blocked - contact phase not started")
            return "Continue the conversation. Confirm the appointment time first."
        
        # Smart capture: phone_number param takes priority over new_phone for backwards compat
        new_phone = _sanitize_tool_arg(phone_number) or _sanitize_tool_arg(new_phone)
    
        if new_phone:
            # User provided a correction - use proper normalization
            clinic_region = (_GLOBAL_CLINIC_INFO or {}).get("default_phone_region", DEFAULT_PHONE_REGION)
            clean_phone, last4 = _normalize_phone_preserve_plus(new_phone, clinic_region)
            
            if clean_phone:
                state.phone_pending = str(clean_phone)  # Enforce string type
                state.phone_last4 = str(last4) if last4 else ""
                # Safety guard
                _ensure_phone_is_string(state)
                # NEVER auto-confirm - even with correction
                state.phone_confirmed = False
                state.phone_source = "user_spoken"
                state.pending_confirm = "phone"
                state.pending_confirm_field = "phone"
                
                # ENTER VERIFICATION MODE (Fix #3)
                state.awaiting_slot_confirmation = True
                state.last_captured_slot = "phone"
                state.last_captured_phone = state.phone_pending
                state.slot_confirm_turns_left = 2
                state.phone_verification_buffer = ""
                logger.info(f"[SLOT_CONFIRM] 🏁 Entering PHONE verification mode (via confirm_phone correction)")
                
                logger.info(f"[TOOL] ⏳ Phone updated (pending) ***{state.phone_last4}")
                # Trigger memory refresh
                if _REFRESH_AGENT_MEMORY:
                    try:
                        _REFRESH_AGENT_MEMORY()
                    except Exception:
                        pass
                return f"I have a number ending in {state.phone_last4} — is that okay?"
            else:
                return f"Could not parse phone number '{new_phone}'. Ask user to repeat clearly."
        
        if confirmed:
            if not state.phone_pending and state.detected_phone:
                state.phone_pending = state.detected_phone
            if not state.phone_pending:
                return "No phone number to confirm. Ask for phone number first."
            # Safety guard before confirming
            _ensure_phone_is_string(state)
            state.phone_e164 = str(state.phone_pending)  # Enforce string type - prevents tuple DB/calendar errors
            state.phone_confirmed = True
            state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
            
            # EXIT VERIFICATION MODE (Fix #3)
            state.awaiting_slot_confirmation = False
            state.last_captured_slot = None
            state.slot_confirm_turns_left = 0
            logger.info(f"[SLOT_CONFIRM] ✅ Phone CONFIRMED by tool - Exiting verification mode")
            
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
            old_phone = state.phone_pending or state.detected_phone or state.phone_e164
            state.phone_pending = None
            state.detected_phone = None
            state.phone_e164 = None
            state.phone_last4 = None
            state.phone_confirmed = False
            state.phone_source = None
            state.pending_confirm = None if state.pending_confirm == "phone" else state.pending_confirm
            state.pending_confirm_field = None if state.pending_confirm_field == "phone" else state.pending_confirm_field
            
            # EXIT VERIFICATION MODE (Fix #3)
            state.awaiting_slot_confirmation = False
            state.last_captured_slot = None
            logger.info(f"[SLOT_CONFIRM] ❌ Phone REJECTED by tool - Exiting verification mode")
            
            logger.info(f"[TOOL] ✗ Phone REJECTED (was {old_phone}), cleared")
            # Trigger memory refresh
            if _REFRESH_AGENT_MEMORY:
                try:
                    _REFRESH_AGENT_MEMORY()
                except Exception:
                    pass
            return "Phone cleared. Ask user: 'Could you please give me your phone number again?'"
    
    

    @llm.ai_callable(description="""
    Confirm the email address with the patient. Call this after spelling it back.
    
    SMART CAPTURE: You can also pass an email_address to save it before confirming.
    Example: confirm_email(confirmed=True, email_address="patient@example.com") saves AND confirms in one call.
    """)
    async def confirm_email(self, confirmed: bool, email_address: Optional[str] = None) -> str:
        """Mark email as confirmed or rejected. Optionally saves email_address before confirming."""
        # global state removed
        state = self.state
        if not state:
            return "State not initialized."
    
        # IDEMPOTENCY CHECK
        if state.check_tool_lock("confirm_email", locals()):
            return "Email confirmation noted."
        
        # Smart capture: Save email_address if provided before confirming
        email_address = _sanitize_tool_arg(email_address)
        if email_address:
            state.email = email_address.strip().lower()
            logger.info(f"[TOOL] Email saved via smart capture: {state.email}")
        
        # Idempotent guard: Do NOT re-confirm if already confirmed
        if state.email_confirmed:
            logger.debug("[TOOL] confirm_email skipped - already confirmed")
            # Clear pending state to prevent re-triggering
            if state.pending_confirm == "email":
                state.pending_confirm = None
            if state.pending_confirm_field == "email":
                state.pending_confirm_field = None
            return "Email already confirmed. Continue with next step."
        
        # Gate: Do NOT confirm email if contact phase hasn't started
        if not contact_phase_allowed(state):
            logger.debug("[TOOL] confirm_email blocked - contact phase not started")
            return "Continue the conversation. Confirm the appointment time first."
        
        if confirmed:
            state.email_confirmed = True
            # Clear pending confirmation to prevent re-triggering
            if state.pending_confirm == "email":
                state.pending_confirm = None
            if state.pending_confirm_field == "email":
                state.pending_confirm_field = None
                
            # EXIT VERIFICATION MODE (Fix #3)
            state.awaiting_slot_confirmation = False
            state.last_captured_slot = None
            state.slot_confirm_turns_left = 0
            logger.info(f"[SLOT_CONFIRM] ✅ Email CONFIRMED by tool - Exiting verification mode")
            
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
            
            # EXIT VERIFICATION MODE (Fix #3)
            state.awaiting_slot_confirmation = False
            state.last_captured_slot = None
            logger.info(f"[SLOT_CONFIRM] ❌ Email REJECTED by tool - Exiting verification mode")
            
            logger.info("[TOOL] ✗ Email rejected, cleared")
            # Trigger memory refresh
            if _REFRESH_AGENT_MEMORY:
                try:
                    _REFRESH_AGENT_MEMORY()
                except Exception:
                    pass
            return "Email cleared. Ask for the correct email."
    
    

    @llm.ai_callable(description="""
    Check the current booking status and what information is still missing.
    Call this to know what to ask for next.
    """)
    async def check_booking_status(self, ) -> str:
        """Return current state and missing fields."""
        # global state removed
        state = self.state
        if not state:
            return "State not initialized."
    
        # IDEMPOTENCY CHECK
        if state.check_tool_lock("check_booking_status", locals()):
            # For status checks, duplicate reads are harmless but redundant.
            # However, to reduce improved latency, skipping is fine if result won't change.
            return "Status already checked."
        
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
    
    

    @llm.ai_callable(description="""
    Finalize the appointment booking. Call this ONLY after:
    1. You have collected ALL required information (name, phone, email, reason, time)
    2. You have read back the summary to the patient
    3. The patient has verbally confirmed with 'yes' or similar
    Do NOT call this until the patient confirms the summary!
    """)
    async def confirm_and_book_appointment(self, ) -> str:
        """
        Trigger the actual booking after user confirmation.
        
        Uses DB-backed OAuth with non-blocking token refresh persistence.
        """
        # global state removed, _GLOBAL_CLINIC_INFO, _GLOBAL_AGENT_SETTINGS
        state = self.state
        clinic_info = _GLOBAL_CLINIC_INFO
        settings = _GLOBAL_AGENT_SETTINGS
        
        if not state:
            return "State not initialized."
        
        # IDEMPOTENCY CHECK
        if state.check_tool_lock("confirm_and_book_appointment", locals()):
            return "Booking request already processed."
        
        # FIX 3: Enhanced logging visibility for debugging false bookings
        is_complete = state.is_complete()
        logger.info(f"[BOOKING] Tool triggered. State complete: {is_complete}")
        
        if not is_complete:
            missing = state.missing_slots()
            logger.warning(f"[BOOKING] Cannot book - missing slots: {missing}")
            return f"Missing: {', '.join(missing)}. Continue gathering info before booking."
        
        # CRITICAL SAFETY GATE: Booking REQUIRES confirmed contact details
        if not state.phone_confirmed:
            return "Phone number not confirmed yet. Please confirm phone before booking."
        if not state.email_confirmed:
            return "Email not confirmed yet. Please confirm email before booking."
        
        # DOUBLE BOOKING GUARD
        # Create unique booking key based on time + phone
        booking_key = f"{state.dt_local.isoformat() if state.dt_local else 'None'}:{state.phone_e164}"
        
        if state.booking_confirmed:
            if state.last_booking_key == booking_key:
                 return "Appointment already booked! Tell the user their appointment is confirmed."
            else:
                 return "An appointment is already confirmed. Ask if they want to book another."
                 
        if state.booking_in_progress:
            return "Booking already in progress. Please wait."
        
        if not clinic_info:
            return "Clinic info not available. Cannot book."
        
        # Mark booking in progress
        state.booking_in_progress = True
        state.last_booking_key = booking_key
        
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
            if not start_dt:
                state.booking_in_progress = False
                return "No appointment time set. Please select a time first."
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
                
                # Build warm, human-sounding confirmation for TTS
                spoken_confirmation = build_spoken_confirmation(state)
                return f"BOOKING CONFIRMED! Read this exactly to the user: {spoken_confirmation}"
            else:
                state.booking_in_progress = False
                return "Booking failed. Ask the user to try again."
                
        except Exception as e:
            logger.error(f"[BOOKING] Error: {e}")
            state.booking_in_progress = False
            return f"Booking error: {str(e)}. Tell the user something went wrong."
        finally:
            state.booking_in_progress = False
    
    

    @llm.ai_callable(description="""
    Search the clinic knowledge base for information about parking, pricing, insurance, 
    location, services, or any clinic-specific details. Call this IMMEDIATELY when the 
    user asks about anything not related to booking (e.g., 'Where do I park?', 
    'Do you accept Delta Dental?', 'How much is a cleaning?').
    """)
    async def search_clinic_info(self, query: Optional[str] = None) -> str:
        """
        A-Tier RAG: Non-blocking semantic search against Supabase knowledge base.
        Uses text-embedding-3-small for <100ms embedding latency.
        """
        global _GLOBAL_CLINIC_INFO, _GLOBAL_STATE
        
        # IDEMPOTENCY CHECK
        # We allow repeated searches if query is different, but not same query in same turn
        if _GLOBAL_STATE and self.state.check_tool_lock("search_clinic_info", locals()):
            return "Searching..."
            
        query = _sanitize_tool_arg(query)
    
        if not query:
            return "What would you like to know about the clinic?"
    
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
            
            # Cast result.data to list for type safety
            data_list = cast(List[Dict[str, Any]], result.data)
            logger.info(f"[RAG] Search successful. Found {len(data_list)} articles.")
    
            # 3. Format results concisely for speech
            answers = []
            for r in data_list:
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
    

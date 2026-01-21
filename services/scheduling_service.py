"""
Scheduling service for appointment slot management.

Handles:
- Business hours validation
- Slot availability checking
- Alternative slot suggestion
- Treatment duration lookup

TODO: Extract functions from agent_v2.py lines 2759-3238
See refactoring_guide.md for exact function list and line numbers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

from config import DEFAULT_TREATMENT_DURATIONS, DEFAULT_LUNCH_BREAK, logger, DEFAULT_TZ

# Week day keys for schedule mapping (Monday=0 to Sunday=6)
WEEK_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# TODO: Extract these functions from agent_v2.py:
# - _default_hours() (lines 2759-2768)
# - load_schedule_from_settings() (lines 2770-2825)
# - is_within_working_hours() (lines 2827-2899)
# - get_duration_for_service() (lines 2901-2924)
# - suggest_slots_around() (lines 2926-3031)
# - get_alternatives_around_datetime() (lines 3033-3144)
# - get_next_available_slots() (lines 3146-3238)

# Placeholder for now - use extraction helper script or manual copy


# ============================================================================
# Extracted: _default_hours
# ============================================================================

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


# ============================================================================
# Extracted: load_schedule_from_settings
# ============================================================================

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



# ============================================================================
# Extracted: is_within_working_hours
# ============================================================================

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



# ============================================================================
# Extracted: get_duration_for_service
# ============================================================================

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



# ============================================================================
# Extracted: suggest_slots_around
# ============================================================================

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



# ============================================================================
# Extracted: get_alternatives_around_datetime
# ============================================================================

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



# ============================================================================
# Extracted: get_next_available_slots
# ============================================================================

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



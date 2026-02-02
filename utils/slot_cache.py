"""
Short-lived TTL cache for slot availability checks.
Prevents repeated DB queries for the same slot during a call.
"""

import time
from typing import Dict, Tuple, Optional, List
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)

# Cache structure: {cache_key: (timestamp, is_free)}
_SLOT_CACHE: Dict[str, Tuple[float, bool]] = {}
_CACHE_TTL_SECONDS = 10.0  # Slots don't change that fast during a call

# Day appointments cache: {cache_key: (timestamp, appointments)}
_DAY_CACHE: Dict[str, Tuple[float, List[Tuple[datetime, datetime]]]] = {}
_DAY_CACHE_TTL_SECONDS = 15.0


def _make_cache_key(clinic_id: str, start_dt: datetime) -> str:
    """Create unique cache key for a slot."""
    return f"{clinic_id}:{start_dt.isoformat()}"


def _make_day_cache_key(clinic_id: str, target_date: date) -> str:
    """Create unique cache key for a day's appointments."""
    return f"day:{clinic_id}:{target_date.isoformat()}"


def get_cached_availability(clinic_id: str, start_dt: datetime) -> Optional[bool]:
    """
    Get cached availability if fresh, else None.
    """
    key = _make_cache_key(clinic_id, start_dt)
    if key in _SLOT_CACHE:
        ts, is_free = _SLOT_CACHE[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return is_free
        else:
            del _SLOT_CACHE[key]  # Expired
    return None


def set_cached_availability(clinic_id: str, start_dt: datetime, is_free: bool):
    """Cache slot availability result."""
    key = _make_cache_key(clinic_id, start_dt)
    _SLOT_CACHE[key] = (time.time(), is_free)


def get_cached_day_appointments(
    clinic_id: str, 
    target_date: date
) -> Optional[List[Tuple[datetime, datetime]]]:
    """Get cached day appointments if fresh, else None."""
    key = _make_day_cache_key(clinic_id, target_date)
    if key in _DAY_CACHE:
        ts, appointments = _DAY_CACHE[key]
        if time.time() - ts < _DAY_CACHE_TTL_SECONDS:
            logger.debug(f"[CACHE] Day appointments cache HIT for {target_date}")
            return appointments
        else:
            del _DAY_CACHE[key]  # Expired
    return None


def set_cached_day_appointments(
    clinic_id: str, 
    target_date: date, 
    appointments: List[Tuple[datetime, datetime]]
):
    """Cache day's appointments."""
    key = _make_day_cache_key(clinic_id, target_date)
    _DAY_CACHE[key] = (time.time(), appointments)


def invalidate_slot_cache(clinic_id: Optional[str] = None):
    """
    Invalidate cache for a clinic or all clinics.
    Call this after successful booking to ensure fresh data.
    """
    global _SLOT_CACHE, _DAY_CACHE
    if clinic_id is None:
        _SLOT_CACHE.clear()
        _DAY_CACHE.clear()
        logger.debug("[CACHE] All slot caches cleared")
    else:
        keys_to_delete = [k for k in _SLOT_CACHE if k.startswith(f"{clinic_id}:")]
        for k in keys_to_delete:
            del _SLOT_CACHE[k]
        day_keys_to_delete = [k for k in _DAY_CACHE if f":{clinic_id}:" in k]
        for k in day_keys_to_delete:
            del _DAY_CACHE[k]
        logger.debug(f"[CACHE] Slot cache cleared for clinic {clinic_id}")


def check_slot_against_appointments(
    slot_start: datetime,
    slot_end: datetime,
    appointments: List[Tuple[datetime, datetime]],
    buffer_minutes: int = 15
) -> bool:
    """
    Check if slot is free against a pre-fetched list of appointments.
    Pure function - no DB calls. O(n) where n = appointments that day.
    
    Args:
        slot_start: Start time of the slot to check
        slot_end: End time of the slot to check
        appointments: List of (start, end) tuples of existing appointments
        buffer_minutes: Buffer time between appointments
    
    Returns:
        True if slot is free, False if occupied
    """
    for appt_start, appt_end in appointments:
        buffered_end = appt_end + timedelta(minutes=buffer_minutes)
        # Check for overlap: slot starts before buffered end AND ends after appointment start
        if slot_start < buffered_end and slot_end > appt_start:
            return False
    return True

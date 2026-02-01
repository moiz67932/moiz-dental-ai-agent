"""
Appointment management service for cancellation and rescheduling operations.

Handles:
- Finding existing appointments by phone number
- Cancelling appointments
- Rescheduling appointments
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from zoneinfo import ZoneInfo

from config import supabase, logger, BOOKED_STATUSES, DEFAULT_TZ
from services.extraction_service import _iso


async def find_appointment_by_phone(
    clinic_id: str,
    phone_number: str,
    tz_str: str = DEFAULT_TZ
) -> Optional[Dict[str, Any]]:
    """
    Find an existing appointment using the caller's phone number.
    
    Searches for future appointments (scheduled or confirmed status) matching the phone.
    Returns the most recent upcoming appointment if found.
    
    Args:
        clinic_id: Clinic identifier
        phone_number: Phone number to search for (E164 format)
        tz_str: Timezone string for date comparisons
        
    Returns:
        Dict with appointment details or None if not found
    """
    try:
        # Get current time in clinic timezone
        now = datetime.now(ZoneInfo(tz_str))
        
        # Search for appointments with matching phone that are in the future
        def _query():
            return (
                supabase.table("appointments")
                .select("id, patient_name, reason, start_time, end_time, status, calendar_event_id")
                .eq("clinic_id", clinic_id)
                .eq("patient_phone_masked", phone_number)
                .gte("start_time", _iso(now))
                .in_("status", ["scheduled", "confirmed"])
                .order("start_time", desc=False)
                .limit(1)
                .execute()
            )
        
        result = await asyncio.to_thread(_query)
        
        if result.data and len(result.data) > 0:
            appt = result.data[0]
            logger.info(f"[APPT_MGMT] Found appointment id={appt['id']} for phone={phone_number[-4:]}")
            
            # Parse the datetime for easy formatting
            start_dt = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=ZoneInfo("UTC"))
            
            # Convert to clinic timezone
            start_dt = start_dt.astimezone(ZoneInfo(tz_str))
            
            return {
                "id": appt["id"],
                "patient_name": appt["patient_name"],
                "reason": appt.get("reason", "appointment"),
                "start_time": start_dt,
                "status": appt["status"],
                "calendar_event_id": appt.get("calendar_event_id"),
            }
        
        logger.info(f"[APPT_MGMT] No upcoming appointment found for phone=***{phone_number[-4:] if len(phone_number) >= 4 else phone_number}")
        return None
        
    except Exception as e:
        logger.error(f"[APPT_MGMT] Error finding appointment: {e}")
        return None


async def cancel_appointment(
    appointment_id: str,
    reason: Optional[str] = None
) -> bool:
    """
    Cancel an existing appointment by updating its status to 'cancelled'.
    
    Args:
        appointment_id: Appointment ID to cancel
        reason: Optional cancellation reason for logging
        
    Returns:
        True if cancellation was successful, False otherwise
    """
    try:
        def _update():
            return (
                supabase.table("appointments")
                .update({
                    "status": "cancelled",
                    # "cancelled_at": _iso(datetime.now()), # Column missing in DB
                })
                .eq("id", appointment_id)
                .execute()
            )
        
        result = await asyncio.to_thread(_update)
        
        if result.data:
            logger.info(f"[APPT_MGMT] ✅ Cancelled appointment id={appointment_id}, reason={reason or 'user_request'}")
            return True
        
        logger.error(f"[APPT_MGMT] ❌ Failed to cancel appointment id={appointment_id}")
        return False
        
    except Exception as e:
        logger.error(f"[APPT_MGMT] ❌ Error cancelling appointment: {e}")
        return False


async def reschedule_appointment(
    appointment_id: str,
    new_start_time: datetime,
    new_end_time: datetime,
    new_reason: Optional[str] = None
) -> bool:
    """
    Reschedule an existing appointment to a new time.
    
    Args:
        appointment_id: Appointment ID to reschedule
        new_start_time: New appointment start time
        new_end_time: New appointment end time
        new_reason: Optional updated reason for visit
        
    Returns:
        True if rescheduling was successful, False otherwise
    """
    try:
        update_data = {
            "start_time": _iso(new_start_time),
            "end_time": _iso(new_end_time),
            "status": "scheduled",  # Reset to scheduled after rescheduling
        }
        
        if new_reason:
            update_data["reason"] = new_reason
        
        def _update():
            return (
                supabase.table("appointments")
                .update(update_data)
                .eq("id", appointment_id)
                .execute()
            )
        
        result = await asyncio.to_thread(_update)
        
        if result.data:
            logger.info(
                f"[APPT_MGMT] ✅ Rescheduled appointment id={appointment_id} "
                f"to {new_start_time.isoformat()}"
            )
            return True
        
        logger.error(f"[APPT_MGMT] ❌ Failed to reschedule appointment id={appointment_id}")
        return False
        
    except Exception as e:
        logger.error(f"[APPT_MGMT] ❌ Error rescheduling appointment: {e}")
        return False


async def find_all_appointments_by_phone(
    clinic_id: str,
    phone_number: str,
    tz_str: str = DEFAULT_TZ,
    include_past: bool = False
) -> List[Dict[str, Any]]:
    """
    Find all appointments for a given phone number.
    Useful for disambiguation when multiple appointments exist.
    
    Args:
        clinic_id: Clinic identifier
        phone_number: Phone number to search for
        tz_str: Timezone string
        include_past: Whether to include past appointments
        
    Returns:
        List of appointment dictionaries
    """
    try:
        now = datetime.now(ZoneInfo(tz_str))
        
        def _query():
            q = (
                supabase.table("appointments")
                .select("id, patient_name, reason, start_time, end_time, status, calendar_event_id")
                .eq("clinic_id", clinic_id)
                .eq("patient_phone_masked", phone_number)
                .in_("status", ["scheduled", "confirmed"])
            )
            
            if not include_past:
                q = q.gte("start_time", _iso(now))
            
            return q.order("start_time", desc=False).execute()
        
        result = await asyncio.to_thread(_query)
        
        appointments = []
        for appt in (result.data or []):
            start_dt = datetime.fromisoformat(appt["start_time"].replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=ZoneInfo("UTC"))
            start_dt = start_dt.astimezone(ZoneInfo(tz_str))
            
            appointments.append({
                "id": appt["id"],
                "patient_name": appt["patient_name"],
                "reason": appt.get("reason", "appointment"),
                "start_time": start_dt,
                "status": appt["status"],
                "calendar_event_id": appt.get("calendar_event_id"),
            })
        
        logger.info(f"[APPT_MGMT] Found {len(appointments)} appointments for phone=***{phone_number[-4:]}")
        return appointments
        
    except Exception as e:
        logger.error(f"[APPT_MGMT] Error finding appointments: {e}")
        return []

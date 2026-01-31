"""
Database service for Supabase operations.

Handles:
- Clinic context fetching with optimized joins
- Appointment persistence
- Slot availability checking
"""

from __future__ import annotations

import re
import os
import asyncio
import traceback
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from config import supabase, logger, BOOKED_STATUSES, DEFAULT_MIN, DEMO_CLINIC_ID

# Import from other new modules
from utils.cache import _clinic_cache
from services.extraction_service import _iso

# These will be imported when agent code is available  
# from models.state import PatientState
# from livekit.agents.voice import Agent as VoicePipelineAgent

# For now, use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models.state import PatientState
    from livekit.agents.voice import Agent as VoicePipelineAgent
    

async def fetch_clinic_context_optimized(
    called_number: str,
    use_cache: bool = True,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict], str]:
    """
    A-TIER: Robust clinic lookup with fuzzy suffix matching and demo fallback.
    
    LOOKUP STRATEGY (in order):
    1. phone_numbers table: Match last 10 digits (ignores +1/+92 prefixes)
    2. clinics table: Direct phone match on clinic record (if stored there)
    3. DEMO FALLBACK: If only 1 clinic exists in DB, use it automatically
    4. PITCH MODE: Force-load DEMO_CLINIC_ID as ultimate fallback
    
    CACHING:
    - Static clinic info and agent settings are cached for CLINIC_CONTEXT_CACHE_TTL seconds
    - This avoids repeated DB queries during the same call
    - Cache is keyed by clinic_id:agent_id (stable, no collision)
    - NEVER caches: availability, schedule conflicts, appointment data
    
    Returns: (clinic_info, agent_info, agent_settings, agent_name)
    """
    
    # Helper: Build stable cache key from fetched data
    def _build_cache_key(clinic: Optional[dict], agent: Optional[dict]) -> Optional[str]:
        """Build deterministic cache key from clinic_id:agent_id."""
        clinic_id = (clinic or {}).get("id")
        agent_id = (agent or {}).get("id")
        if clinic_id:
            return f"{clinic_id}:{agent_id or 'no_agent'}"
        return None
    
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
        # Fuzzy suffix matching — use last 10 digits to ignore prefixes
        digits_only = re.sub(r"\D", "", called_number or "")
        last10 = digits_only[-10:] if len(digits_only) >= 10 else digits_only
        
        logger.debug(f"[DB] Looking up phone: raw='{called_number}', last10='{last10}'")

        # STRATEGY 1: Search phone_numbers table with suffix match
        def _query_phone_numbers():
            q = supabase.table("phone_numbers").select(
                "clinic_id, agent_id, "
                "clinics:clinic_id("
                "  id, organization_id, name, timezone, default_phone_region, "
                "  address, city, state, zip_code, country"
                "), "
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
            
            if not agent_info and clinic_info:
                agent_info = await _fetch_agent_for_clinic(clinic_info["id"])
            
            agent_info, settings = _extract_settings(agent_info)
            agent_name = (agent_info or {}).get("name") or "Office Assistant"
            
            logger.info(f"[DB] Context loaded via phone_numbers: clinic={clinic_info.get('name') if clinic_info else 'None'}, agent={agent_name}")
            result = (clinic_info, agent_info, settings, agent_name)
            if use_cache:
                cache_key = _build_cache_key(clinic_info, agent_info)
                if cache_key:
                    _clinic_cache.set(cache_key, result)
            return result
        
        logger.debug(f"[DB] No match in phone_numbers for last10='{last10}'")

        # STRATEGY 2: Search clinics table directly
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
            
            logger.info(f"[DB] Context loaded via clinics table: clinic={clinic_info.get('name')}, agent={agent_name}")
            result = (clinic_info, agent_info, settings, agent_name)
            if use_cache:
                cache_key = _build_cache_key(clinic_info, agent_info)
                if cache_key:
                    _clinic_cache.set(cache_key, result)
            return result
        
        # PITCH MODE — Phone lookup failed, force-load demo clinic by UUID
        logger.warning(f"[DB] Phone lookup failed for {called_number}. Activating Pitch Mode.")
        
        def _fetch_demo_clinic():
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
            
            logger.info(f"[DB] Pitch Mode context loaded: clinic={clinic_info.get('name')}, agent={agent_name}")
            result = (clinic_info, agent_info, settings, agent_name)
            if use_cache:
                cache_key = _build_cache_key(clinic_info, agent_info)
                if cache_key:
                    _clinic_cache.set(cache_key, result)
            return result

        logger.error(f"[DB] CRITICAL: Demo clinic UUID {DEMO_CLINIC_ID} not found in database!")
        return None, None, None, "Office Assistant"

    except Exception as e:
        logger.error(f"[DB] Context fetch error: {e}")
        traceback.print_exc()
        return None, None, None, "Office Assistant"


async def is_slot_free_supabase(clinic_id: str, start_dt: datetime, end_dt: datetime, clinic_info: dict = None) -> bool:
    """Check if slot is free in Supabase appointments table."""
    try:
        # Safety check for closed dates (Fix #2)
        if clinic_info:
            try:
                closed = schedule.get("closed_dates") or set()
            except Exception:
                closed = set()
            if start_dt.strftime("%Y-%m-%d") in closed:
                logger.info(f"[DB] 🛡️ Slot rejected: {start_dt.date()} is a closed date")
                return False

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


# async def book_to_supabase(
#     clinic_info: dict,
#     patient_state: "PatientState",
#     calendar_event_id: Optional[str] = None,
# ) -> bool:
#     """Insert appointment into Supabase."""
#     try:
#         start_time = patient_state.dt_local
#         if not start_time:
#             logger.error("[DB] Cannot book: no start_time set")
#             return False
#         end_time = start_time + timedelta(minutes=patient_state.duration_minutes)
        
#         payload = {
#             "organization_id": clinic_info["organization_id"],
#             "clinic_id": clinic_info["id"],
#             "patient_name": patient_state.full_name,
#             "patient_phone_masked": patient_state.phone_last4,
#             "patient_email": patient_state.email,
#             "start_time": _iso(start_time),
#             "end_time": _iso(end_time),
#             "status": "scheduled",
#             "source": "ai",
#             "reason": patient_state.reason,
#         }
        
#         if calendar_event_id:
#             payload["calendar_event_id"] = calendar_event_id
        
#         await asyncio.to_thread(
#             lambda: supabase.table("appointments").insert(payload).execute()
#         )
#         logger.info("[DB] Appointment saved to Supabase")
#         return True
#     except Exception as e:
#         logger.error(f"[DB] Booking insert error: {e}")
#         return False


# TUNING: Kill DB write if it takes longer than 6 seconds
BOOKING_DB_TIMEOUT_SEC = 6.0 

async def book_to_supabase(clinic_info: dict, patient_state: "PatientState", calendar_event_id: str = None) -> str | None:
    """
    Insert appointment row in Supabase.
    Returns appointment_id on success, None on failure.
    Has hard timeout so calls never hang.
    """
    try:
        start_time = patient_state.dt_local
        if not start_time:
            logger.error("[DB] Cannot book: no start_time set")
            return None

        duration = patient_state.duration_minutes or 30
        end_time = start_time + timedelta(minutes=duration)

        payload = {
            "organization_id": clinic_info.get("organization_id"),
            "clinic_id": clinic_info.get("id"),
            "patient_name": patient_state.full_name,
            "patient_phone_masked": patient_state.phone_last4,
            "patient_email": patient_state.email,
            "start_time": _iso(start_time),
            "end_time": _iso(end_time),
            "status": "scheduled",
            "source": "ai",
            "reason": patient_state.reason or "General Dentistry",
        }

        if calendar_event_id:
            payload["calendar_event_id"] = calendar_event_id

        # Define the sync operation
        def _insert_sync():
            # Supabase Python: don't chain .select() after insert()
            res = supabase.table("appointments").insert(payload, returning="representation").execute()
            data = res.data or []
            return data[0].get("id") if data else None  

        logger.info(f"[DB] Inserting appointment (timeout={BOOKING_DB_TIMEOUT_SEC}s) start={payload['start_time']}")
        
        # Execute with hard timeout
        appt_id = await asyncio.wait_for(asyncio.to_thread(_insert_sync), timeout=BOOKING_DB_TIMEOUT_SEC)

        if not appt_id:
            logger.error("[DB] Insert returned no id (unexpected)")
            return None

        logger.info(f"[DB] ✅ Appointment inserted id={appt_id}")
        return appt_id

    except asyncio.TimeoutError:
        logger.error(f"[DB] ❌ Supabase insert timed out after {BOOKING_DB_TIMEOUT_SEC}s")
        return None
    except Exception as e:
        logger.error(f"[DB] ❌ Booking insert error: {e}")
        return None

async def attach_calendar_event_id(appointment_id: str, calendar_event_id: str) -> bool:
    """
    Updates the appointment row AFTER the call is effectively done for the user.
    """
    if not appointment_id or not calendar_event_id:
        return False

    try:
        def _update_sync():
            res = (
                supabase.table("appointments")
                .update({"calendar_event_id": calendar_event_id}, returning="representation")
                .eq("id", appointment_id)
                .execute()
            )
            return bool(res.data)
        

        ok = await asyncio.to_thread(_update_sync)
        if ok:
            logger.info(f"[DB] ✅ Attached calendar_event_id to appt={appointment_id}")
        else:
            logger.warning(f"[DB] ⚠️ Could not attach calendar_event_id to appt={appointment_id}")
        return ok

    except Exception as e:
        logger.error(f"[DB] ❌ attach_calendar_event_id failed: {e}")
        return False

# NOTE: try_book_appointment requires complex imports from calendar_service and agent
# It's better to keep this in agent.py where it's used, as it needs:
# - VoicePipelineAgent (session param)
# - resolve_calendar_auth_async from calendar_service  
# - _get_calendar_service from calendar_client
# Moving it here would create circular dependencies

# calendar_client.py
# =============================================================================
# Google Calendar Client with OAuth/Service Account Support
# =============================================================================
"""
FIX 6: Enhanced logging when calendar_id is missing or auth fails.
All errors are now logged clearly to help diagnose configuration issues.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

# Try to import Google libs; fall back to dry-run if missing
try:
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    _GOOGLE_LIBS_AVAILABLE = True
except Exception:
    _GOOGLE_LIBS_AVAILABLE = False

_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Cache services by a stable key (so we don't rebuild each call)
_SERVICE_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class CalendarAuth:
    """
    auth_type:
      - 'oauth_user'      => Google OAuth authorized user JSON
      - 'service_account' => Service account JSON
    """
    auth_type: str
    secret_json: dict[str, Any]
    delegated_user: Optional[str] = None  # optional (Workspace domain-wide delegation)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _fingerprint_auth(auth: CalendarAuth) -> str:
    """Stable cache key (never log secrets)."""
    base = {
        "auth_type": auth.auth_type,
        "delegated_user": auth.delegated_user,
        "client_id": auth.secret_json.get("client_id"),
        "sa_client_email": auth.secret_json.get("client_email"),
        "calendar_scopes": tuple(_SCOPES),
    }
    return json.dumps(base, sort_keys=True)


def _build_google_credentials(auth: CalendarAuth):
    if not _GOOGLE_LIBS_AVAILABLE:
        return None

    if auth.auth_type == "oauth_user":
        info = dict(auth.secret_json)
        info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
        return Credentials.from_authorized_user_info(info, scopes=_SCOPES)

    if auth.auth_type == "service_account":
        creds = service_account.Credentials.from_service_account_info(
            auth.secret_json, scopes=_SCOPES
        )
        if auth.delegated_user:
            creds = creds.with_subject(auth.delegated_user)
        return creds

    raise ValueError(f"Unsupported auth_type: {auth.auth_type}")


def _get_calendar_service(auth: Optional[CalendarAuth] = None):
    """
    Return a Google Calendar service or None (dry-run).
    If auth is None, falls back to env-file behavior (local dev).
    """
    if not _GOOGLE_LIBS_AVAILABLE:
        print("[calendar] Google libs not installed → DRY-RUN mode.")
        return None

    # New path: explicit auth (from DB / Vault)
    if auth is not None:
        key = _fingerprint_auth(auth)
        with _CACHE_LOCK:
            if key in _SERVICE_CACHE:
                return _SERVICE_CACHE[key]

            creds = _build_google_credentials(auth)
            service = build("calendar", "v3", credentials=creds, cache_discovery=False)
            _SERVICE_CACHE[key] = service
            return service

    # Old path: env-based local dev
    token_path = os.getenv("GOOGLE_OAUTH_TOKEN")
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    delegated = os.getenv("GCAL_DELEGATED_USER")

    creds = None
    if token_path and os.path.exists(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
        creds = Credentials.from_authorized_user_info(info, scopes=_SCOPES)

    elif sa_path and os.path.exists(sa_path):
        with open(sa_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
        if delegated:
            creds = creds.with_subject(delegated)

    else:
        print("[calendar] No credentials found → DRY-RUN mode.")
        return None

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _to_iso(dt: datetime, tz: Optional[str]) -> tuple[str, str]:
    """Return (iso, tz_name) with timezone applied."""
    tzname = tz or (getattr(dt.tzinfo, "key", None)) or "UTC"
    aware = (
        dt.astimezone(ZoneInfo(tzname))
        if dt.tzinfo
        else dt.replace(tzinfo=ZoneInfo(tzname))
    )
    return aware.isoformat(), tzname


# ---------------------------------------------------------------------
# Public low-level API (kept intact)
# ---------------------------------------------------------------------

def is_time_free(
    calendar_id: str,
    start_dt: datetime,
    end_dt: datetime,
    tz: Optional[str] = None,
    auth: Optional[CalendarAuth] = None,
) -> bool:
    service = _get_calendar_service(auth=auth)
    start_iso, tzname = _to_iso(start_dt, tz)
    end_iso, _ = _to_iso(end_dt, tzname)

    if service is None:
        print(f"[calendar] DRY-RUN freebusy {start_iso} → {end_iso} ({tzname}) → FREE")
        return True

    body = {
        "timeMin": start_iso,
        "timeMax": end_iso,
        "timeZone": tzname,
        "items": [{"id": calendar_id}],
    }
    resp = service.freebusy().query(body=body).execute()
    busy = (resp.get("calendars", {}).get(calendar_id, {}) or {}).get("busy", [])
    return len(busy) == 0


def create_event(
    calendar_id: str,
    summary: str,
    start_dt: datetime,
    end_dt: datetime,
    tz: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendee_email: Optional[str] = None,
    auth: Optional[CalendarAuth] = None,
) -> dict:
    service = _get_calendar_service(auth=auth)
    start_iso, tzname = _to_iso(start_dt, tz)
    end_iso, _ = _to_iso(end_dt, tzname)

    event = {
        "summary": summary,
        "description": description or "",
        "location": location or "",
        "start": {"dateTime": start_iso, "timeZone": tzname},
        "end": {"dateTime": end_iso, "timeZone": tzname},
        "attendees": [{"email": attendee_email}] if attendee_email else [],
        "reminders": {"useDefault": True},
    }

    if service is None:
        fake_id = f"TEST-{int(start_dt.timestamp())}"
        print(f"[calendar] DRY-RUN create → id={fake_id}")
        return {"id": fake_id, "status": "confirmed", **event}

    return service.events().insert(
        calendarId=calendar_id,
        body=event,
        sendUpdates="all",
    ).execute()


# ---------------------------------------------------------------------
# Agent-friendly helper (NEW, SAFE, OPTIONAL)
# ---------------------------------------------------------------------

async def book_appointment(
    *,
    calendar_id: str,
    service_name: str,
    start_dt: datetime,
    duration_minutes: int,
    tz: str,
    patient_name: str,
    patient_phone: str,
    clinic_name: str,
    location: Optional[str] = None,
    auth: Optional[CalendarAuth] = None,
) -> dict:
    """
    High-level helper used by the agent.
    - Checks availability
    - Creates event
    - Returns event dict
    
    FIX 6: Enhanced error logging when calendar_id is missing.
    """
    # FIX 6: Clear logging when calendar_id is not configured
    if not calendar_id:
        error_msg = (
            "[CALENDAR] FAILED - no calendar_id configured. "
            "Set GOOGLE_CALENDAR_ID env var or configure calendar_id in DB."
        )
        print(error_msg)
        raise RuntimeError(error_msg)
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    free = is_time_free(
        calendar_id=calendar_id,
        start_dt=start_dt,
        end_dt=end_dt,
        tz=tz,
        auth=auth,
    )

    if not free:
        raise RuntimeError("Requested time slot is not available")

    summary = f"{service_name} — {patient_name}"
    description = (
        f"Patient: {patient_name}\n"
        f"Phone: {patient_phone}\n"
        f"Clinic: {clinic_name}"
    )

    return create_event(
        calendar_id=calendar_id,
        summary=summary,
        start_dt=start_dt,
        end_dt=end_dt,
        tz=tz,
        description=description,
        location=location,
        auth=auth,
    )

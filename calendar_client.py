# calendar_client.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

# Try to import Google libs; fall back to dry-run if missing
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    _GOOGLE_LIBS_AVAILABLE = True
except Exception:
    _GOOGLE_LIBS_AVAILABLE = False

_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_calendar_service():
    """Return a Google Calendar service or None (dry-run)."""
    if not _GOOGLE_LIBS_AVAILABLE:
        print("[calendar] Google libs not installed → DRY-RUN mode.")
        return None

    from google.oauth2.credentials import Credentials

    token_path = os.getenv("GOOGLE_OAUTH_TOKEN")            # e.g., ./google_token.json
    sa_path    = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")# service-account JSON (optional)
    delegated  = os.getenv("GCAL_DELEGATED_USER")           # optional

    creds = None
    if token_path and os.path.exists(token_path):
        # Use user OAuth token (personal calendar)
        creds = Credentials.from_authorized_user_file(token_path, scopes=_SCOPES)
    elif sa_path and os.path.exists(sa_path):
        # Fallback to service account (Workspace/shared calendar)
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=_SCOPES)
        if delegated:
            creds = creds.with_subject(delegated)
    else:
        print("[calendar] No GOOGLE_OAUTH_TOKEN or GOOGLE_APPLICATION_CREDENTIALS → DRY-RUN mode.")
        return None

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _to_iso(dt: datetime, tz: str | None) -> tuple[str, str]:
    """Return (iso, tz_name) with timezone applied."""
    tzname = tz or (getattr(dt.tzinfo, "key", None)) or "UTC"
    aware = dt.astimezone(ZoneInfo(tzname)) if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(tzname))
    return aware.isoformat(), tzname


def is_time_free(calendar_id: str, start_dt: datetime, end_dt: datetime, tz: Optional[str] = None) -> bool:
    """Return True if the calendar has no busy blocks for the [start, end) window."""
    service = _get_calendar_service()
    start_iso, tzname = _to_iso(start_dt, tz)
    end_iso, _ = _to_iso(end_dt, tzname)

    if service is None:
        print(f"[calendar] DRY-RUN freebusy {start_iso} → {end_iso} ({tzname}) → assuming FREE")
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
) -> dict:
    """Create a calendar event; returns the API response (or a fake one in dry-run)."""
    service = _get_calendar_service()
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

    created = service.events().insert(
        calendarId=calendar_id, body=event, sendUpdates="all"
    ).execute()
    return created

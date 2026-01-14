# supabase_calendar_store.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from supabase import create_client, Client


@dataclass
class CalendarConnection:
    calendar_id: str
    timezone: Optional[str] = None
    provider: str = "google"


class SupabaseCalendarStore:
    def __init__(self, supabase_url: str, service_role_key: str):
        self.supabase: Client = create_client(supabase_url, service_role_key)

    def get_calendar_connection(self, org_id: str, clinic_id: str) -> CalendarConnection:
        resp = (
            self.supabase
            .table("calendar_connections")
            .select("provider, calendar_id, timezone, enabled")
            .eq("organization_id", org_id)
            .eq("clinic_id", clinic_id)
            .eq("provider", "google")
            .eq("enabled", True)
            .limit(1)
            .execute()
        )
        data = (resp.data or [])
        if not data:
            raise RuntimeError(
                "No calendar connection found for this clinic. "
                "Insert a row into public.calendar_connections (provider='google', calendar_id='primary', timezone='Asia/Karachi')."
            )
        row = data[0]
        return CalendarConnection(
            provider=row.get("provider") or "google",
            calendar_id=row["calendar_id"],
            timezone=row.get("timezone"),
        )

    def get_clinic_timezone(self, org_id: str, clinic_id: str) -> Optional[str]:
        resp = (
            self.supabase
            .table("clinics")
            .select("timezone")
            .eq("organization_id", org_id)
            .eq("id", clinic_id)
            .limit(1)
            .execute()
        )
        data = (resp.data or [])
        return (data[0].get("timezone") if data else None)

    def get_clinic_phone_region(self, org_id: str, clinic_id: str) -> Optional[str]:
        resp = (
            self.supabase
            .table("clinics")
            .select("default_phone_region")
            .eq("organization_id", org_id)
            .eq("id", clinic_id)
            .limit(1)
            .execute()
        )
        data = (resp.data or [])
        return (data[0].get("default_phone_region") if data else None)

    def list_appointment_types(self, org_id: str, clinic_id: str) -> list[dict[str, Any]]:
        resp = (
            self.supabase
            .table("appointment_types")
            .select("id,name,description,duration_minutes,active")
            .eq("organization_id", org_id)
            .eq("clinic_id", clinic_id)
            .eq("active", True)
            .order("name")
            .execute()
        )
        return resp.data or []

    def create_appointment(
        self,
        organization_id: str,
        clinic_id: str,
        patient_name: str,
        patient_phone_masked: str,
        patient_email: str,
        start_time: datetime,
        end_time: datetime,
        reason: str,
        calendar_provider: str,
        calendar_id: str,
        calendar_event_id: str,
    ) -> dict[str, Any]:
        payload = {
            "organization_id": organization_id,
            "clinic_id": clinic_id,
            "patient_name": patient_name,
            "patient_phone_masked": patient_phone_masked,
            "patient_email": patient_email,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "reason": reason,
            "calendar_provider": calendar_provider,
            "calendar_id": calendar_id,
            "calendar_event_id": calendar_event_id,
        }

        resp = self.supabase.table("appointments").insert(payload).execute()
        data = resp.data or []
        if not data:
            raise RuntimeError(f"Failed to insert appointment: {resp}")
        return data[0]

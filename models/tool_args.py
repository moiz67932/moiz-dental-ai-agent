"""
Pydantic models for tool function arguments.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class UpdatePatientRecordArgs(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    reason: Optional[str] = None
    time_suggestion: Optional[str] = None


class GetAvailableSlotsV2Args(BaseModel):
    after_datetime: Optional[str] = None
    preferred_day: Optional[str] = None
    num_slots: Optional[str] = None


class FindRelativeSlotsArgs(BaseModel):
    start_search_time: Optional[str] = None
    limit_to_day: Optional[str] = None
    find_last: Optional[str] = None


class SearchClinicInfoArgs(BaseModel):
    query: Optional[str] = None


def _sanitize_tool_arg(value: Optional[str]) -> Optional[str]:
    """Sanitize tool arguments - removes None, empty strings, and 'null' literals."""
    if not value:
        return None
    value = value.strip()
    return value if value.lower() != "null" else None

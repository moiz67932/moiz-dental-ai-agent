"""
Service modules package.

This package contains business logic services for:
- Database operations (Supabase)
- Scheduling and slot management
- Calendar integration
- Data extraction
"""

from .database_service import (
    fetch_clinic_context_optimized,
    is_slot_free_supabase,
    book_to_supabase,
)
from .calendar_service import (
    resolve_calendar_auth,
    resolve_calendar_auth_async,
    fetch_oauth_token_from_db,
    save_refreshed_token_to_db,
)
from .extraction_service import (
    extract_name_quick,
    extract_reason_quick,
    _iso,
)

# scheduling_service imports will be added after cleanup
# from .scheduling_service import (
#     is_within_working_hours,
#     suggest_slots_around,
#     get_duration_for_service,
#     load_schedule_from_settings,
#     get_next_available_slots,
#     get_alternatives_around_datetime,
# )

__all__ = [
    "fetch_clinic_context_optimized",
    "is_slot_free_supabase",
    "book_to_supabase",
    "resolve_calendar_auth",
    "resolve_calendar_auth_async",
    "fetch_oauth_token_from_db",
    "save_refreshed_token_to_db",
    "extract_name_quick",
    "extract_reason_quick",
    "_iso",
]

"""
Patient state management.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Set

from config import DEFAULT_TZ, logger


# =============================================================================
# REGEX PATTERNS
# =============================================================================

YES_PAT = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|that's right|that is right|ok|okay|sure)\b",
    re.IGNORECASE,
)

NO_PAT = re.compile(
    r"\b(no|nope|wrong|incorrect|not correct|that's wrong)\b",
    re.IGNORECASE,
)

EMERGENCY_PAT = re.compile(
    r"\b(bleeding|uncontrolled bleeding|faint|unconscious|can'?t breathe|breathing|trauma|broken jaw|severe swelling|swelling.*eye|fever.*swelling)\b",
    re.IGNORECASE,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_valid_email_strict(text: str) -> bool:
    if not text or "@" not in text or "." not in text:
        return False
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text.strip()))


def is_valid_phone_strict(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    return 7 <= len(digits) <= 15


def contact_phase_allowed(state: "PatientState") -> bool:
    """Contact details can be collected after name + valid time, or explicit flag."""
    safety_check = (
        state.full_name is not None
        and state.time_status == "valid"
        and state.dt_local is not None
    )
    fallback_check = getattr(state, "contact_phase_started", False) is True
    caller_id_accepted = getattr(state, "caller_id_accepted", False) is True
    return safety_check or fallback_check or caller_id_accepted


# =============================================================================
# PATIENT STATE CLASS
# =============================================================================

@dataclass
class PatientState:
    """Clean state container for patient booking info."""
    full_name: Optional[str] = None
    phone_e164: Optional[str] = None
    phone_last4: Optional[str] = None
    email: Optional[str] = None
    reason: Optional[str] = None
    dt_local: Optional[datetime] = None
    dt_text: Optional[str] = None

    last_user_text: Optional[str] = None
    recent_user_texts: List[str] = field(default_factory=list)

    # Rejected slots (to avoid re-suggesting)
    rejected_slots: Set[str] = field(default_factory=set)

    # Duration tracking
    duration_minutes: int = 60
    time_status: str = "pending"  # "pending", "validating", "valid", "invalid", "error"
    time_error: Optional[str] = None
    slot_available: bool = False

    # Phone lifecycle
    detected_phone: Optional[str] = None
    phone_pending: Optional[str] = None

    # Confirmations
    phone_confirmed: bool = False
    email_confirmed: bool = False
    pending_confirm: Optional[str] = None
    pending_confirm_field: Optional[str] = None

    # Phone source tracking
    phone_source: Optional[str] = None

    # Contact phase gating
    contact_phase_started: bool = False

    # SMS preference
    prefers_sms: bool = False
    delivery_channel: Optional[str] = None
    delivery_preference_pending: bool = False
    delivery_preference_asked: bool = False
    anything_else_pending: bool = False
    anything_else_asked: bool = False
    user_declined_more_help: bool = False
    final_goodbye_sent: bool = False
    user_goodbye_detected: bool = False
    closing_state: str = "open"

    # Caller ID flow tracking
    caller_id_checked: bool = False
    caller_id_accepted: bool = False
    using_caller_number: bool = False
    confirmed_contact_number_source: Optional[str] = None

    # Booking state
    booking_confirmed: bool = False
    appointment_booked: bool = False
    appointment_id: Optional[str] = None
    calendar_event_id: Optional[str] = None

    # Appointment management (cancel/reschedule flow)
    found_appointment_id: Optional[str] = None
    found_appointment_details: Optional[Dict[str, Any]] = None

    # Terminal state
    call_ended: bool = False

    # Booking idempotency guards
    booking_in_progress: bool = False
    last_confirm_fingerprint: Optional[str] = None
    last_confirm_ts: float = 0.0
    turn_consumed: bool = False

    # Context
    tz: str = DEFAULT_TZ
    patient_type: Optional[str] = None

    def add_rejected_slot(self, dt: datetime, reason: str = "user_rejected"):
        key = dt.strftime("%Y-%m-%d %H:%M")
        self.rejected_slots.add(key)
        logger.info(f"[STATE] Slot rejected: {key} ({reason})")

    def remember_user_text(self, text: str, max_items: int = 6):
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return
        self.last_user_text = cleaned
        self.recent_user_texts.append(cleaned)
        if len(self.recent_user_texts) > max_items:
            self.recent_user_texts = self.recent_user_texts[-max_items:]

    def recent_user_context(self, limit: int = 4) -> str:
        return " ".join(self.recent_user_texts[-limit:]).strip()

    def is_slot_rejected(self, dt: datetime) -> bool:
        return dt.strftime("%Y-%m-%d %H:%M") in self.rejected_slots

    def is_complete(self) -> bool:
        return all([
            self.full_name,
            self.phone_e164,
            self.phone_confirmed,
            self.reason,
            self.dt_local,
        ])

    def missing_slots(self) -> List[str]:
        missing = []
        if not self.full_name:
            missing.append("full_name")
        if not self.phone_e164:
            missing.append("phone")
        elif not self.phone_confirmed:
            missing.append("phone_confirmed")
        if not self.reason:
            missing.append("reason")
        if not self.dt_local:
            missing.append("datetime")
        return missing

    def slot_summary(self) -> str:
        return (
            f"name={self.full_name or '?'}, "
            f"phone={'confirmed' if self.phone_confirmed else (self.phone_last4 or '?')}, "
            f"reason={self.reason or '?'}, "
            f"time={self.dt_local.isoformat() if self.dt_local else '?'}"
        )

    def detailed_state_for_prompt(self) -> str:
        """Concise state snapshot for the dynamic system prompt."""
        lines = []

        # Name
        if self.full_name:
            lines.append(f"• NAME: {self.full_name}")
        else:
            lines.append("• NAME: ? — Ask naturally")

        # Phone
        if not contact_phase_allowed(self):
            lines.append("• PHONE: — (collect after time confirmed)")
        elif self.phone_e164 and self.phone_confirmed:
            if self.using_caller_number or self.confirmed_contact_number_source == "caller_id":
                lines.append("• PHONE: confirmed caller ID / this number")
            elif self.phone_last4:
                lines.append(f"• PHONE: confirmed ending in {self.phone_last4}")
            else:
                lines.append("• PHONE: confirmed")
        elif self.phone_pending or self.detected_phone:
            lines.append("• PHONE: pending confirmation — Ask: 'Should I save the number you're calling from?'")
        else:
            lines.append("• PHONE: ? — Ask naturally")

        # Email (only show if provided)
        if self.email and self.email_confirmed:
            lines.append(f"• EMAIL: confirmed {self.email}")
        elif self.email:
            lines.append(f"• EMAIL: {self.email} (captured)")

        # Reason
        if self.reason:
            lines.append(f"• REASON: {self.reason} ({self.duration_minutes}m)")
        else:
            lines.append("• REASON: ? — Ask what brings them in")

        # Time — show parsed time even if availability not yet confirmed
        if self.dt_local and self.time_status == "valid":
            time_str = self.dt_local.strftime('%a %b %d @ %I:%M %p')
            lines.append(f"• TIME: confirmed {time_str}")
        elif self.dt_local:
            time_str = self.dt_local.strftime('%a %b %d @ %I:%M %p')
            lines.append(f"• TIME: {time_str} (checking availability)")
        elif self.time_status == "invalid" and self.time_error:
            lines.append(f"• TIME: unavailable — {self.time_error}")
        elif self.dt_text:
            lines.append(f"• TIME: parsing '{self.dt_text}'")
        else:
            lines.append("• TIME: ? — Ask when")

        # Booking status
        if self.booking_confirmed:
            lines.append("BOOKED!")
            if self.delivery_preference_pending:
                lines.append("DELIVERY: Ask WhatsApp or SMS on this number, then wait.")
            elif self.delivery_channel:
                lines.append(f"DELIVERY: {self.delivery_channel}")
            if self.anything_else_pending:
                lines.append("CLOSING: Ask if anything else is needed.")
            elif self.final_goodbye_sent:
                lines.append("CLOSING: Final goodbye sent. Wait briefly, then end call.")
        elif self.is_complete():
            lines.append("READY TO BOOK — Summarize & confirm")
        else:
            missing = [s for s in self.missing_slots() if not s.endswith('_confirmed')]
            if missing:
                lines.append(f"NEED: {', '.join(missing)}")

        return '\n'.join(lines)

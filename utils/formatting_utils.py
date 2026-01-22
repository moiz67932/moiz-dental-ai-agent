"""
General formatting utilities for speech and display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.state import PatientState


def build_spoken_confirmation(state: "PatientState") -> str:
    """
    Build a warm, human-sounding booking confirmation for TTS.
    Pauses are created ONLY via the final join, not inline ellipses.
    
    Called by confirm_and_book_appointment to ensure consistent,
    non-robotic confirmation speech.
    """
    from utils.phone_utils import format_phone_for_speech
    
    parts = []
    
    if state.full_name:
        parts.append(f"Alright {state.full_name}")
    
    if state.reason and state.dt_local:
        day = state.dt_local.strftime('%B %d')
        time_str = state.dt_local.strftime('%I:%M %p').lstrip('0')
        parts.append(
            f"I've officially booked you for {state.reason} "
            f"on {day} at {time_str}."
        )
    
    if state.phone_e164:
        parts.append(
            "I'll send the details to your phone number "
            f"{format_phone_for_speech(state.phone_e164)}."
        )
    
    if state.email:
        # Use this logic for emails:
        local_part, domain = state.email.split('@')
        spelled_local = ", ".join(list(local_part))
        spaced_email = f"{spelled_local} at {domain.replace('.', ' dot ')}"
        parts.append(f"And I've got your email as {spaced_email}.")
    
    parts.append("Is there anything else I can help you with today?")
    
    return " … ".join(parts)


def email_for_speech(email: str) -> str:
    """Format email address for TTS."""
    if not email:
        return "unknown"
    # Spaces in the email string slow down the TTS for clarity
    return email.replace("@", " at ").replace(".", " dot ")

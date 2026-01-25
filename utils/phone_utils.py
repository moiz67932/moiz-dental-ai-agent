"""
Phone number utilities for normalization and formatting.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, TYPE_CHECKING

from .contact_utils import normalize_phone
from config import logger, DEFAULT_PHONE_REGION

if TYPE_CHECKING:
    from models.state import PatientState


def _normalize_sip_user_to_e164(raw: Optional[str]) -> Optional[str]:
    """Best-effort normalization for SIP headers like sip.toUser/sip.calledNumber."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Strip common SIP URI wrappers
    s = s.replace("sip:", "")
    if "@" in s:
        s = s.split("@", 1)[0]

    # Keep only digits and '+'
    s = re.sub(r"[^\d+]", "", s)
    if not s:
        return None

    # Convert 00-prefixed international numbers
    if s.startswith("00"):
        s = "+" + s[2:]

    # If it's digits-only, assume missing '+' and add it (8+ digits for international support)
    if not s.startswith("+") and s.isdigit() and len(s) >= 8:
        s = "+" + s

    return s


def speakable_phone(e164: Optional[str]) -> str:
    """
    Convert E.164 phone number to speech-friendly format for verbal confirmation.
    
    Examples:
        +923351897839 -> "+92 335 189 7839"
        +13105551234  -> "+1 310 555 1234"
    
    Used when agent reads back phone number for confirmation.
    ALWAYS use full number, never just last 4 digits.
    """
    if not e164:
        return "unknown"
    
    s = str(e164).strip()
    if not s.startswith("+"):
        s = "+" + re.sub(r"\D", "", s)
    
    digits = re.sub(r"\D", "", s)
    
    # Format based on country code length
    if digits.startswith("1") and len(digits) == 11:  # US/Canada: +1 XXX XXX XXXX
        return f"+1 {digits[1:4]} {digits[4:7]} {digits[7:]}"
    elif digits.startswith("92") and len(digits) >= 11:  # Pakistan: +92 XXX XXX XXXX
        return f"+92 {digits[2:5]} {digits[5:8]} {digits[8:]}"
    elif digits.startswith("44") and len(digits) >= 11:  # UK: +44 XXXX XXXXXX
        return f"+44 {digits[2:6]} {digits[6:]}"
    else:
        # Generic: group in 3-4 digit chunks
        parts = []
        if s.startswith("+"):
            # Keep country code separate (1-3 digits)
            cc_len = 1 if digits[0] == "1" else (2 if len(digits) <= 12 else 3)
            parts.append(f"+{digits[:cc_len]}")
            remaining = digits[cc_len:]
        else:
            remaining = digits
        
        # Split remaining into 3-4 digit groups
        while remaining:
            chunk_size = 4 if len(remaining) > 6 else 3
            parts.append(remaining[:chunk_size])
            remaining = remaining[chunk_size:]
        
        return " ".join(parts)


def format_phone_for_speech(phone: str) -> str:
    """Format phone number for TTS with proper pacing."""
    if not phone:
        return "unknown"

    digits = [d for d in phone if d.isdigit()]

    # US-style grouping for human pacing
    if len(digits) == 10:
        return (
            f"{digits[0]} {digits[1]} {digits[2]}, "
            f"{digits[3]} {digits[4]} {digits[5]}, "
            f"{digits[6]} {digits[7]} {digits[8]} {digits[9]}"
        )

    # Fallback: spaced digits
    return " ".join(digits)


def _ensure_phone_is_string(state: "PatientState") -> None:
    """
    Safety guard: Ensure phone fields are always strings, not tuples.
    Call this after any phone assignment to catch tuple bugs.
    """
    if state.phone_e164 is not None and isinstance(state.phone_e164, tuple):
        logger.error(f"[PHONE BUG] state.phone_e164 was tuple: {state.phone_e164}. Extracting first element.")
        state.phone_e164 = state.phone_e164[0] if state.phone_e164 else None
    if state.phone_pending is not None and isinstance(state.phone_pending, tuple):
        logger.error(f"[PHONE BUG] state.phone_pending was tuple: {state.phone_pending}. Extracting first element.")
        state.phone_pending = state.phone_pending[0] if state.phone_pending else None
    if state.detected_phone is not None and isinstance(state.detected_phone, tuple):
        logger.error(f"[PHONE BUG] state.detected_phone was tuple: {state.detected_phone}. Extracting first element.")
        state.detected_phone = state.detected_phone[0] if state.detected_phone else None



def parse_spoken_numerals(text: Optional[str]) -> str:
    """
    Parse spoken phone numbers, handling 'double', 'triple' and word-to-digit conversion.
    
    Examples:
        "double two" -> "22"
        "triple three" -> "333"
        "zero three zero zero" -> "0300"
        "plus nine two" -> "+92"
    """
    if not text:
        return ""
        
    s = text.lower().strip()
    
    # word to digit map
    w2d = {
        "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", 
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }
    
    # Handle "plus"
    s = s.replace("plus", "+")
    
    words = s.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        
        # Handle double/triple
        multiplier = 1
        if w in ("double", "triple"):
            multiplier = 2 if w == "double" else 3
            if i + 1 < len(words):
                next_w = words[i+1]
                # If next word is a digit word like "two"
                if next_w in w2d:
                    digit = w2d[next_w]
                    result.append(digit * multiplier)
                    i += 2
                    continue
                # If next word is a digit string like "2"
                elif next_w.isdigit():
                    result.append(next_w * multiplier)
                    i += 2
                    continue
            # If "double" is at end or not followed by digit, ignore it or treat as noise?
            # For safety, just skip it if we can't consume next token
            i += 1
            continue
            
        # Handle digit words
        if w in w2d:
            result.append(w2d[w])
        # Handle existing digits/symbols
        else:
            # clean non-digit chars except +
            cleaned = "".join(c for c in w if c.isdigit() or c == "+")
            if cleaned:
                result.append(cleaned)
                
        i += 1
        
    return "".join(result)


def _normalize_phone_preserve_plus(raw: Optional[str], default_region: str) -> Tuple[Optional[str], str]:
    """Normalize phone while preserving explicit international '+' prefix.
    
    Returns: Tuple[Optional[str], str] - (e164_phone, last4_digits)
    IMPORTANT: Always unpack result as: clean_phone, last4 = _normalize_phone_preserve_plus(...)
    CRITICAL: This ALWAYS returns a tuple (str|None, str). Never store the raw return value.
    """
    if not raw:
        return None, ""

    # Step 1: Parse spoken numerals first (handles "double two" etc.)
    parsed = parse_spoken_numerals(str(raw))
    if not parsed:
        # Fallback to basic string if parsing returns empty allowed chars (unlikely if raw had content)
        parsed = str(raw).strip()

    s = parsed
    
    # Handle local Pakistani formats (e.g., 0335xxxxxxx -> +92335xxxxxxx)
    if default_region == "PK":
        # Remove any non-digits first for clean checking
        clean_digits = re.sub(r"\D", "", s)
        # Check for 03xxxxxxxxx (11 digits, starts with 03)
        if len(clean_digits) == 11 and clean_digits.startswith("03"):
            s = "+92" + clean_digits[1:]  # Remove leading 0, add +92
            logger.debug(f"[PHONE] Converted PK local {raw} -> {s}")
    
    # If already E.164 format with +
    if s.startswith("+"):
        digits = re.sub(r"\D", "", s)
        # Support international numbers (8+ digits covers Pakistani numbers like +923...)
        if len(digits) >= 8:
            e164 = f"+{digits}"
            return e164, e164[-4:]

    # Fallback to normalize_phone which also returns a tuple
    result = normalize_phone(s, default_region)
    if isinstance(result, tuple):
        e164_val, last4_val = result
        # Ensure e164_val is a string, not a tuple (safety guard)
        if isinstance(e164_val, tuple):
            logger.error(f"[PHONE] BUG: normalize_phone returned nested tuple for '{raw}'")
            e164_val = e164_val[0] if e164_val else None
        return e164_val, last4_val
    # Safety: if normalize_phone returns just a string (shouldn't happen), wrap it
    if result:
        if isinstance(result, str):
            return result, result[-4:] if len(result) >= 4 else ""
        # If result is somehow still a tuple at this point
        logger.error(f"[PHONE] BUG: Unexpected type from normalize_phone: {type(result)}")
    return None, ""

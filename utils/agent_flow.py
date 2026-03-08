"""
Lightweight helpers for agent runtime flow.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence


YES_CONFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|that's right|that is right|ok|okay|sure|please do|go ahead)\b",
    re.IGNORECASE,
)
NO_CONFIRM_RE = re.compile(
    r"\b(no|nope|wrong|incorrect|that's wrong|that is wrong|do not|don't)\b",
    re.IGNORECASE,
)
CALLER_ID_AFFIRM_RE = re.compile(
    r"\b("
    r"use (?:the )?(?:same|this|that)? ?number|"
    r"the one i(?: am|'m) calling from|"
    r"number i(?: am|'m) calling from|"
    r"this is the number|"
    r"same number"
    r")\b",
    re.IGNORECASE,
)
MONTH_NAME_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)
WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
QUALIFIED_WEEKDAY_RE = re.compile(
    r"\b(this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
RELATIVE_DAY_RE = re.compile(
    r"\b(day after tomorrow|tomorrow|today)\b",
    re.IGNORECASE,
)
ORDINAL_DAY_RE = re.compile(
    r"\b("
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    r"seventeenth|eighteenth|nineteenth|twentieth|twenty[\s-]first|"
    r"twenty[\s-]second|twenty[\s-]third|twenty[\s-]fourth|"
    r"twenty[\s-]fifth|twenty[\s-]sixth|twenty[\s-]seventh|"
    r"twenty[\s-]eighth|twenty[\s-]ninth|thirtieth|thirty[\s-]first|"
    r"\d{1,2}(?:st|nd|rd|th)"
    r")\b",
    re.IGNORECASE,
)
TIME_REFERENCE_RE = re.compile(
    r"\b("
    r"\d{1,2}:\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?|"
    r"\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)|"
    r"noon|midnight|morning|afternoon|evening|"
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:thirty|fifteen|forty[\s-]?five|forty|twenty|ten|oh five|o five)?\s*"
    r"(?:am|pm|a\.m\.|p\.m\.)?"
    r")\b",
    re.IGNORECASE,
)
PHONE_INPUT_RE = re.compile(
    r"\b(phone|number|digits|calling from|this number|same number|use this number)\b|(?:\+?\d[\d\s().-]{5,})",
    re.IGNORECASE,
)


def store_detected_phone(
    state: Any,
    phone: Optional[str],
    last4: Optional[str] = None,
    *,
    source: str = "sip",
) -> Optional[str]:
    """Seed caller ID into state without marking it confirmed."""
    if not phone:
        return None

    phone_str = str(phone)
    digits = re.sub(r"\D", "", phone_str)
    resolved_last4 = str(last4) if last4 else (digits[-4:] if len(digits) >= 4 else "")

    state.detected_phone = phone_str
    if not getattr(state, "phone_confirmed", False) or not getattr(state, "phone_e164", None):
        if not getattr(state, "phone_pending", None):
            state.phone_pending = phone_str
    if resolved_last4:
        state.phone_last4 = resolved_last4
    state.phone_source = source
    return phone_str


def ensure_caller_phone_pending(state: Any) -> Optional[str]:
    """Promote detected caller ID to pending phone when needed."""
    phone = getattr(state, "phone_pending", None) or getattr(state, "detected_phone", None)
    if not phone:
        return None

    phone_str = str(phone)
    digits = re.sub(r"\D", "", phone_str)
    state.phone_pending = phone_str
    if digits and not getattr(state, "phone_last4", None):
        state.phone_last4 = digits[-4:]
    if getattr(state, "phone_source", None) in (None, "", "sip"):
        state.phone_source = "sip"
    return phone_str


def resolve_confirmation_intent(text: Optional[str]) -> Optional[bool]:
    """
    Return True for affirmative, False for negative, None for ambiguous.

    If both yes and no appear, use the last explicit confirmation marker.
    This fixes mixed utterances like "No. Yep." which should resolve to yes.
    """
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    if not normalized:
        return None

    if CALLER_ID_AFFIRM_RE.search(normalized):
        return True

    yes_positions = [match.start() for match in YES_CONFIRM_RE.finditer(normalized)]
    no_positions = [match.start() for match in NO_CONFIRM_RE.finditer(normalized)]

    if yes_positions and not no_positions:
        return True
    if no_positions and not yes_positions:
        return False
    if yes_positions and no_positions:
        return yes_positions[-1] > no_positions[-1]
    return None


def is_active_filler_event(
    speech_text: Optional[str],
    active_filler_text: Optional[str],
    filler_phrases: Sequence[str],
    *,
    same_handle: bool = False,
) -> bool:
    """Recognize filler speech even when the event does not include text."""
    if same_handle and active_filler_text:
        return True

    text = (speech_text or "").strip()
    if not text:
        return False
    if active_filler_text and text.startswith(active_filler_text):
        return True
    return any(text.startswith(phrase) for phrase in filler_phrases)


def has_date_reference(text: Optional[str]) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False
    return bool(
        RELATIVE_DAY_RE.search(normalized)
        or QUALIFIED_WEEKDAY_RE.search(normalized)
        or WEEKDAY_RE.search(normalized)
        or MONTH_NAME_RE.search(normalized)
        or ORDINAL_DAY_RE.search(normalized)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized)
    )


def has_time_reference(text: Optional[str]) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False
    return TIME_REFERENCE_RE.search(normalized) is not None


def time_expression_score(text: Optional[str]) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0

    score = 0
    if RELATIVE_DAY_RE.search(normalized):
        score += 4 if re.search(r"\bday after tomorrow\b", normalized, re.IGNORECASE) else 3
    if QUALIFIED_WEEKDAY_RE.search(normalized):
        score += 4
    elif WEEKDAY_RE.search(normalized):
        score += 2
    if MONTH_NAME_RE.search(normalized):
        score += 4
    if ORDINAL_DAY_RE.search(normalized):
        score += 3
    if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized):
        score += 4
    if has_time_reference(normalized):
        score += 2
    return score


def build_time_parse_candidates(
    primary_text: Optional[str],
    *,
    recent_context: Optional[str] = None,
    previous_text: Optional[str] = None,
) -> list[str]:
    """
    Order time-parse inputs from most trustworthy to least.

    Recent caller transcript wins when it carries a stronger date signal than the
    model-supplied paraphrase. If the caller gave only a time in the latest turn,
    combine it with the previously captured date phrase first.
    """
    primary = " ".join((primary_text or "").split()).strip()
    recent = " ".join((recent_context or "").split()).strip()
    previous = " ".join((previous_text or "").split()).strip()

    candidates: list[str] = []

    if previous and has_date_reference(previous) and has_time_reference(primary) and not has_date_reference(primary):
        candidates.append(f"{previous} at {primary}")

    primary_score = time_expression_score(primary)
    recent_score = time_expression_score(recent)
    if recent and recent_score > primary_score:
        candidates.append(recent)

    if primary:
        candidates.append(primary)

    if recent:
        candidates.append(recent)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def looks_like_phone_input(text: Optional[str]) -> bool:
    return PHONE_INPUT_RE.search((text or "").strip()) is not None

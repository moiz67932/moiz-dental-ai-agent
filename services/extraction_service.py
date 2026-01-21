"""
Deterministic data extraction from text.

Quick pattern-based extractors for name, reason, and formatting utilities.
"""

from __future__ import annotations

import re
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from config import DEFAULT_TZ


def extract_name_quick(text: str) -> Optional[str]:
    """Quick name extraction from common patterns."""
    patterns = [
        r"\b(?:my\s+name\s+is|i\s+am|i'm|this\s+is|call\s+me)\s+([A-Za-z][A-Za-z\s\.'-]{2,})",
        r"^(?:it'?s|its)\s+([A-Za-z][A-Za-z\s\.'-]{2,})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            # Clean trailing noise
            name = re.split(r"\b(and|i|want|need|would|like|to|for|at|my|phone|email)\b", name, flags=re.I)[0].strip()
            if len(name) >= 3:
                return name.title()
    return None


def extract_reason_quick(text: str) -> Optional[str]:
    """Quick service extraction."""
    t = text.lower()
    service_map = {
        "whiten": "Teeth whitening",
        "whitening": "Teeth whitening",
        "clean": "Cleaning",
        "cleaning": "Cleaning",
        "checkup": "Checkup",
        "check-up": "Checkup",
        "exam": "Checkup",
        "pain": "Tooth pain",
        "toothache": "Tooth pain",
        "consult": "Consultation",
        "extract": "Extraction",
        "filling": "Filling",
        "crown": "Crown",
        "root canal": "Root canal",
    }
    for key, value in service_map.items():
        if key in t:
            return value
    return None


def _iso(dt: datetime) -> str:
    """Convert datetime to ISO format string with timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(DEFAULT_TZ))
    return dt.isoformat()

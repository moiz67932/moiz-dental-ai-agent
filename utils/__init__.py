"""
Utility modules for the dental AI agent.
"""

from .cache import TTLCache, _clinic_cache
from .latency_metrics import TurnMetrics, _turn_metrics
from .phone_utils import (
    _normalize_sip_user_to_e164,
    speakable_phone,
    format_phone_for_speech,
    _ensure_phone_is_string,
    _normalize_phone_preserve_plus,
)
from .formatting_utils import build_spoken_confirmation, email_for_speech
from .call_logger import CallLogger, create_call_logger

__all__ = [
    "TTLCache",
    "_clinic_cache",
    "TurnMetrics",
    "_turn_metrics",
    "_normalize_sip_user_to_e164",
    "speakable_phone",
    "format_phone_for_speech",
    "_ensure_phone_is_string",
    "_normalize_phone_preserve_plus",
    "build_spoken_confirmation",
    "email_for_speech",
    "CallLogger",
    "create_call_logger",
]

# contact_utils.py
import re
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from dateutil import parser as dtparser
from datetime import datetime
from zoneinfo import ZoneInfo

_WORD_DIGITS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "to": "2", "too": "2", "for": "4",
}

def words_to_digits(text: str) -> str:
    toks = re.split(r"[^\w@.]+", (text or "").lower())
    out = []
    for t in toks:
        if t.isdigit():
            out.append(t)
        elif t in _WORD_DIGITS:
            out.append(_WORD_DIGITS[t])
        else:
            out.append(''.join(_WORD_DIGITS.get(x, '') for x in t.split('-')))
    return ''.join(out)

def normalize_phone(spoken: str, default_region: str = "US") -> tuple[str | None, str]:
    raw = re.sub(r"[^\w+]", "", spoken or "")
    digits = words_to_digits(raw)
    candidates = [digits, "+" + digits if not digits.startswith("+") else digits]
    for cand in candidates:
        try:
            num = phonenumbers.parse(cand, default_region)
            if phonenumbers.is_possible_number(num) and phonenumbers.is_valid_number(num):
                e164 = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)
                return e164, e164[-4:]
        except Exception:
            continue
    return None, digits[-4:] if digits else ""

def normalize_email(spoken: str) -> str:
    s = (spoken or "").strip().lower()
    s = s.replace(" at the rate ", " @ ")
    s = s.replace(" at ", " @ ")
    s = s.replace(" dot ", " . ")
    s = re.sub(r"\s+", "", s)
    s = s.replace("gmailcom", "gmail.com").replace("yahoocom", "yahoo.com").replace("outlookcom","outlook.com")
    return s

def validate_email_address(addr: str) -> bool:
    try:
        validate_email(addr, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False

def parse_datetime_natural(spoken: str, tz_hint: str | None = None) -> datetime | None:
    """Parse natural language datetime and optionally apply timezone."""
    try:
        parsed = dtparser.parse(spoken, fuzzy=True)
        # If a timezone hint is provided and the parsed datetime is naive, apply it
        if tz_hint and parsed.tzinfo is None:
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(tz_hint))
            except Exception:
                pass  # If timezone is invalid, keep as-is
        return parsed
    except Exception:
        return None
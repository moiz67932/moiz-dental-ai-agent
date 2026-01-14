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

# =============================================================================
# EMAIL INTRODUCER STRIPPING
# =============================================================================
# 
# WHY THIS EXISTS:
# When users speak their email address in a voice conversation, they naturally
# prefix it with intent phrases like "my email is", "email address is", etc.
# 
# THE BUG:
# If we apply spoken-email normalization (digit conversion, "at the rate"→@)
# to the ENTIRE utterance, these introducer phrases become part of the email:
#   "my email is moiz679@gmail.com" → "myemailismoiz679@gmail.com" (WRONG)
# 
# THE FIX:
# Strip introducer phrases BEFORE any normalization, so only the actual email
# payload gets processed:
#   "my email is moiz679@gmail.com" → "moiz679@gmail.com" (CORRECT)
# 
# WHY DETERMINISTIC (not NLU):
# This is a spoken-language preprocessing concern, not an extraction issue.
# The email regex and validation are correct - the input is polluted.
# Deterministic extraction remains authoritative; NLU is still fallback only.
# =============================================================================

# Spoken digit words to numeric digits (for email local parts with numbers)
_EMAIL_WORD_DIGITS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Common phrases users say before stating their email address
# These are removed BEFORE normalization to prevent pollution
_EMAIL_INTRODUCER_PATTERNS = [
    r"\bmy\s+email\s+(?:address\s+)?is\b",
    r"\bemail\s+(?:address\s+)?is\b",
    r"\byou\s+can\s+(?:reach|email|contact)\s+me\s+at\b",
    r"\bsend\s+(?:it\s+)?to\b",
    r"\bcontact\s+me\s+at\b",
    r"\breach\s+me\s+at\b",
    r"\bit'?s\b",  # "it's john@gmail.com"
]


def _strip_email_introducer(text: str) -> str:
    """
    Remove spoken email introducer phrases so that normalization applies
    only to the actual email payload.
    
    This MUST be called BEFORE:
    - Digit normalization (six → 6)
    - "at the rate" replacement
    - "dot" replacement
    - Space removal
    
    Args:
        text: Raw user utterance containing email
        
    Returns:
        Text with introducer phrases removed, preserving the email payload
    """
    if not text:
        return ""
    
    result = text.strip().lower()
    
    # Remove each introducer pattern
    for pattern in _EMAIL_INTRODUCER_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()
    
    return result


def _convert_spoken_digits_for_email(text: str) -> str:
    """
    Convert spoken digit words to numeric digits for email local parts.
    
    "moiz six seven nine three two" → "moiz67932"
    
    Only converts standalone digit words, not words containing digit words.
    """
    if not text:
        return ""
    
    tokens = text.split()
    result = []
    
    for token in tokens:
        if token in _EMAIL_WORD_DIGITS:
            result.append(_EMAIL_WORD_DIGITS[token])
        else:
            result.append(token)
    
    return " ".join(result)


def normalize_email(spoken: str) -> str:
    """
    Normalize spoken email to standard email format.
    
    Processing order (CRITICAL - DO NOT REORDER):
    1. Strip email introducer phrases ("my email is", etc.)
    2. Convert spoken digits to numbers ("six" → "6")
    3. Replace spoken symbols ("at the rate" → "@", "dot" → ".")
    4. Remove spaces
    5. Fix common domain typos
    
    Args:
        spoken: User's spoken email utterance
        
    Returns:
        Normalized email string (may or may not be valid)
    """
    if not spoken:
        return ""
    
    # STEP 1: Strip introducer phrases FIRST
    # This prevents "my email is" from becoming part of the local part
    s = _strip_email_introducer(spoken)
    
    # STEP 2: Convert spoken digits
    # "moiz six seven nine" → "moiz 6 7 9"
    s = _convert_spoken_digits_for_email(s)
    
    # STEP 3: Replace spoken symbols
    s = s.replace(" at the rate ", " @ ")
    s = s.replace(" at ", " @ ")
    s = s.replace(" dot ", " . ")
    s = s.replace(" underscore ", " _ ")
    s = s.replace(" dash ", " - ")
    s = s.replace(" hyphen ", " - ")
    
    # STEP 4: Remove all spaces
    s = re.sub(r"\s+", "", s)
    
    # STEP 5: Fix common domain typos
    s = s.replace("gmailcom", "gmail.com")
    s = s.replace("yahoocom", "yahoo.com")
    s = s.replace("outlookcom", "outlook.com")
    s = s.replace("hotmailcom", "hotmail.com")
    
    return s

def validate_email_address(addr: str) -> bool:
    try:
        validate_email(addr, check_deliverability=False)
        return True
    except EmailNotValidError:
        return False

def parse_datetime_natural(spoken: str, tz_hint: str | None = None) -> datetime | None:
    """
    Parse natural language datetime and optionally apply timezone if parsed datetime is naive.
    """
    try:
        parsed = dtparser.parse(spoken, fuzzy=True)
        if tz_hint and parsed.tzinfo is None:
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(tz_hint))
            except Exception:
                pass
        return parsed
    except Exception:
        return None

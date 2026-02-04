# contact_utils.py
import re
import phonenumbers
from email_validator import validate_email, EmailNotValidError
from dateutil import parser as dtparser
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any
from config import logger

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

def parse_datetime_natural(spoken: str, tz_hint: str | None = None) -> Dict[str, Any]:
    """
    Parse natural language datetime with proper relative date handling.

    Handles:
    - "tomorrow at 3:30 PM" → next day at 15:30
    - "next Monday at 2pm" → upcoming Monday at 14:00
    - "this Friday afternoon" → upcoming Friday at 14:00
    - Absolute dates like "January 20 at 10am"

    Returns:
        Dict[str, Any]: {
            "success": bool,
            "datetime": Optional[datetime],
            "needs_clarification": bool,
            "clarification_type": str,
            "message": str
        }
    """
    if not spoken:
        return {"success": False, "datetime": None, "needs_clarification": False, "clarification_type": "", "message": "empty_input"}

    try:
        tz = ZoneInfo(tz_hint) if tz_hint else None
        now = datetime.now(tz) if tz else datetime.now()

        spoken_lower = spoken.lower().strip()

        # ═══════════════════════════════════════════════════════════════════════════
        # 🛡️ INCOMPLETE DATE DETECTION (Issue B)
        # ═══════════════════════════════════════════════════════════════════════════
        month_names = ["january","february","march","april","may","june",
                       "july","august","september","october","november","december"]
        has_month = any(m in spoken_lower for m in month_names)
        # Check for day numbers (1-31)
        has_day_number = bool(re.search(r'\b([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?\b', spoken_lower))
        # Check for relative days or weekday names
        relative_words = ["today","tomorrow","monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        has_relative = any(w in spoken_lower for w in relative_words)

        if has_month and not has_day_number and not has_relative:
            mentioned = [m for m in month_names if m in spoken_lower][0].title()
            return {
                "success": False,
                "datetime": None,
                "needs_clarification": True,
                "clarification_type": "missing_day",
                "month_mentioned": mentioned,
                "message": f"Which day in {mentioned} would you prefer?"
            }

        # Original parsing logic continues, but wrapping results in success dict
        # ... (rest of function will need to be wrapped or updated) ...

        # Clean up possessive forms (e.g., "feb's 10th" → "feb 10th")
        spoken_lower = re.sub(r"(\w+)'s\b", r"\1", spoken_lower)

        # ═══════════════════════════════════════════════════════════════════════════
        # 🛠️ FILLER WORD REMOVAL — Clean words that confuse dateutil parser
        # ═══════════════════════════════════════════════════════════════════════════
        _FILLER_PATTERN = r'\b(uh|um|uhh|umm|uhm|er|ah|hmm|like|you know)\b'
        spoken_clean = re.sub(_FILLER_PATTERN, '', spoken_lower, flags=re.IGNORECASE)
        spoken_clean = re.sub(r'\s+', ' ', spoken_clean).strip()
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ⏰ PRE-EXTRACT TIME COMPONENT — Handle spoken times before ordinal matching
        # ═══════════════════════════════════════════════════════════════════════════
        # This fixes: "third of feb at three thirty pm" → 9:00 AM (WRONG)
        # Now correctly parses to 15:30 PM
        
        _WORD_TO_HOUR = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
            'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
        }
        _WORD_TO_MIN = {
            'thirty': 30, 'fifteen': 15, 'forty-five': 45, 'forty five': 45,
            'forty': 40, 'twenty': 20, 'ten': 10, 'oh five': 5, 'o five': 5,
        }
        
        _extracted_time = None
        
        # Pattern 1: Word-based times like "three thirty pm", "two pm"
        word_time_pattern = re.compile(
            r'(?:at\s+)?'  # Optional "at"
            r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*'
            r'(thirty|fifteen|forty-five|forty five|forty|twenty|ten|oh five|o five)?'
            r'\s*(am|pm|a\.m\.|p\.m\.)?',
            re.IGNORECASE
        )
        word_match = word_time_pattern.search(spoken_clean)
        
        # Pattern 2: Numeric times like "3:30 pm", "2pm", "at 10 am", "15:30"
        # IMPORTANT: Require either "at" prefix, colon for minutes, or am/pm suffix
        # to avoid matching day numbers like "6th" in "february 6th"
        numeric_time_pattern = re.compile(
            r'(?:at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?'  # "at 10 am" pattern (requires "at")
            r'|(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?'              # "3:30 pm" pattern (requires colon)
            r'|(\d{1,2})\s+(am|pm|a\.m\.|p\.m\.))',                     # "2pm" pattern (requires am/pm)
            re.IGNORECASE
        )
        numeric_match = numeric_time_pattern.search(spoken_clean)
        
        # Prefer word-based match first (more likely from speech recognition)
        if word_match and word_match.group(1):
            hour_word = word_match.group(1).lower()
            min_word = (word_match.group(2) or '').lower().replace('-', ' ')
            ampm = (word_match.group(3) or '').lower().replace('.', '')
            
            hour = _WORD_TO_HOUR.get(hour_word)
            minute = _WORD_TO_MIN.get(min_word, 0)
            
            if hour is not None:
                # Apply AM/PM conversion
                if ampm == 'pm' and hour < 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
                # Infer PM for business hour times (1-6) without explicit AM/PM
                elif not ampm and 1 <= hour <= 6:
                    hour += 12  # Assume 1-6 without AM/PM means PM for appointments
                
                try:
                    _extracted_time = datetime.min.time().replace(hour=hour, minute=minute)
                except ValueError:
                    pass  # Invalid time, skip
        
        # Fallback to numeric match
        # New pattern has 3 alternatives with different group positions:
        # Alt 1 (at N am): groups 1,2,3 = hour, minute, ampm
        # Alt 2 (N:MM am): groups 4,5,6 = hour, minute, ampm  
        # Alt 3 (N am):    groups 7,8 = hour, ampm (no minute)
        elif numeric_match:
            # Find which alternative matched
            if numeric_match.group(1):  # "at 10 am" pattern
                hour = int(numeric_match.group(1))
                minute = int(numeric_match.group(2)) if numeric_match.group(2) else 0
                ampm = (numeric_match.group(3) or '').lower().replace('.', '')
            elif numeric_match.group(4):  # "3:30 pm" pattern
                hour = int(numeric_match.group(4))
                minute = int(numeric_match.group(5)) if numeric_match.group(5) else 0
                ampm = (numeric_match.group(6) or '').lower().replace('.', '')
            elif numeric_match.group(7):  # "2pm" pattern
                hour = int(numeric_match.group(7))
                minute = 0
                ampm = (numeric_match.group(8) or '').lower().replace('.', '')
            else:
                hour, minute, ampm = None, 0, ''
            
            if hour is not None and hour <= 24 and minute < 60:
                # Apply AM/PM conversion
                if ampm == 'pm' and hour < 12:
                    hour += 12
                elif ampm == 'am' and hour == 12:
                    hour = 0
                # Infer PM for business hour times (1-6) without explicit AM/PM
                elif not ampm and 1 <= hour <= 6:
                    hour += 12
                
                if hour < 24:
                    try:
                        _extracted_time = datetime.min.time().replace(hour=hour, minute=minute)
                    except ValueError:
                        pass
        
        # === RELATIVE DAY HANDLING ===
        # These MUST be handled before dateutil.parse which ignores them
        
        # Handle "tomorrow"
        if "tomorrow" in spoken_lower:
            base_date = (now + timedelta(days=1)).date()
            
            # Use pre-extracted time if available (more reliable)
            if _extracted_time:
                result = datetime.combine(base_date, _extracted_time)
                if tz:
                    result = result.replace(tzinfo=tz)
                return {"success": True, "datetime": result}
            
            # Fallback: Extract time component from the rest of the string
            time_str = re.sub(r"\btomorrow\b", "", spoken_clean, flags=re.IGNORECASE).strip()
            if time_str:
                try:
                    time_parsed = dtparser.parse(time_str, fuzzy=True)
                    if time_parsed:
                        result = datetime.combine(base_date, time_parsed.time())
                        if tz:
                            result = result.replace(tzinfo=tz)
                        return {"success": True, "datetime": result}
                except Exception:
                    pass
            # Default to 9am if no time specified
            result = datetime.combine(base_date, datetime.min.time().replace(hour=9))
            if tz:
                result = result.replace(tzinfo=tz)
            return {"success": True, "datetime": result}
        
        # Handle "today"
        if "today" in spoken_lower:
            base_date = now.date()
            
            # Use pre-extracted time if available (more reliable)
            if _extracted_time:
                result = datetime.combine(base_date, _extracted_time)
                if tz:
                    result = result.replace(tzinfo=tz)
                return {"success": True, "datetime": result}
            
            # Fallback to dateutil parsing
            time_str = re.sub(r"\btoday\b", "", spoken_clean, flags=re.IGNORECASE).strip()
            if time_str:
                try:
                    time_parsed = dtparser.parse(time_str, fuzzy=True)
                    if time_parsed:
                        result = datetime.combine(base_date, time_parsed.time())
                        if tz:
                            result = result.replace(tzinfo=tz)
                        return {"success": True, "datetime": result}
                except Exception:
                    pass
            # Default to next hour if no time specified
            result = datetime.combine(base_date, now.time().replace(minute=0, second=0, microsecond=0))
            if tz:
                result = result.replace(tzinfo=tz)
            return {"success": True, "datetime": result}
        
        # Handle "next [weekday]" or standalone weekday names
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        
        for day_name, weekday_num in day_map.items():
            if day_name in spoken_lower or f"next {day_name}" in spoken_lower:
                # Calculate days until that weekday
                days_until = (weekday_num - now.weekday()) % 7
                if days_until == 0:  # Same day - go to next week
                    days_until = 7
                # If "next" is explicit, always go to next week occurrence
                if f"next {day_name}" in spoken_lower and days_until < 7:
                    days_until = 7 if days_until == 0 else days_until
                
                base_date = (now + timedelta(days=days_until)).date()
                
                # Use pre-extracted time if available (more reliable)
                if _extracted_time:
                    result = datetime.combine(base_date, _extracted_time)
                    if tz:
                        result = result.replace(tzinfo=tz)
                    return {"success": True, "datetime": result}
                
                # Fallback: Extract time from remaining string
                time_str = re.sub(rf"\b(next\s+)?{day_name}\b", "", spoken_clean, flags=re.IGNORECASE).strip()
                if time_str:
                    try:
                        time_parsed = dtparser.parse(time_str, fuzzy=True)
                        if time_parsed:
                            result = datetime.combine(base_date, time_parsed.time())
                            if tz:
                                result = result.replace(tzinfo=tz)
                            return {"success": True, "datetime": result}
                    except Exception:
                        pass
                
                # Default to 9am if no time specified
                result = datetime.combine(base_date, datetime.min.time().replace(hour=9))
                if tz:
                    result = result.replace(tzinfo=tz)
                return {"success": True, "datetime": result}
        
        # === MONTH + DAY MAPPINGS ===
        # Define these FIRST as they're used by multiple pattern matchers
        month_map = {
            "january": 1, "jan": 1,
            "february": 2, "feb": 2,
            "march": 3, "mar": 3,
            "april": 4, "apr": 4,
            "may": 5,
            "june": 6, "jun": 6,
            "july": 7, "jul": 7,
            "august": 8, "aug": 8,
            "september": 9, "sept": 9, "sep": 9,
            "october": 10, "oct": 10,
            "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        
        # Ordinal number words
        ordinal_map = {
            "first": 1, "1st": 1,
            "second": 2, "2nd": 2,
            "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4,
            "fifth": 5, "5th": 5,
            "sixth": 6, "6th": 6,
            "seventh": 7, "7th": 7,
            "eighth": 8, "8th": 8,
            "ninth": 9, "9th": 9,
            "tenth": 10, "10th": 10,
            "eleventh": 11, "11th": 11,
            "twelfth": 12, "12th": 12,
            "thirteenth": 13, "13th": 13,
            "fourteenth": 14, "14th": 14,
            "fifteenth": 15, "15th": 15,
            "sixteenth": 16, "16th": 16,
            "seventeenth": 17, "17th": 17,
            "eighteenth": 18, "18th": 18,
            "nineteenth": 19, "19th": 19,
            "twentieth": 20, "20th": 20,
            "twenty-first": 21, "twenty first": 21, "21st": 21,
            "twenty-second": 22, "twenty second": 22, "22nd": 22,
            "twenty-third": 23, "twenty third": 23, "23rd": 23,
            "twenty-fourth": 24, "twenty fourth": 24, "24th": 24,
            "twenty-fifth": 25, "twenty fifth": 25, "25th": 25,
            "twenty-sixth": 26, "twenty sixth": 26, "26th": 26,
            "twenty-seventh": 27, "twenty seventh": 27, "27th": 27,
            "twenty-eighth": 28, "twenty eighth": 28, "28th": 28,
            "twenty-ninth": 29, "twenty ninth": 29, "29th": 29,
            "thirtieth": 30, "30th": 30,
            "thirty-first": 31, "thirty first": 31, "31st": 31,
        }
        
        # === NUMERIC-FIRST PATTERNS ===
        # Handle patterns like "10 february", "21 march", "3 jan"
        # Match: number (optionally with ordinal suffix) + month name
        numeric_first_pattern = r'\b(\d+)(?:st|nd|rd|th)?\s+(' + '|'.join(month_map.keys()) + r')\b'
        numeric_match = re.search(numeric_first_pattern, spoken_lower)
        if numeric_match:
            day_num = int(numeric_match.group(1))
            month_name = numeric_match.group(2).lower()
            
            if 1 <= day_num <= 31 and month_name in month_map:
                month_num = month_map[month_name]
                
                # Determine the year
                year = now.year
                if month_num < now.month or (month_num == now.month and day_num < now.day):
                    year += 1
                
                try:
                    base_date = datetime(year, month_num, day_num).date()
                    
                    # Use pre-extracted time if available
                    if _extracted_time:
                        result = datetime.combine(base_date, _extracted_time)
                        if tz:
                            result = result.replace(tzinfo=tz)
                        return {"success": True, "datetime": result}
                    
                    # Fallback: Extract time from remaining string
                    after_pattern = spoken_clean[numeric_match.end():].strip()
                    if after_pattern:
                        try:
                            time_parsed = dtparser.parse(after_pattern, fuzzy=True)
                            if time_parsed:
                                result = datetime.combine(base_date, time_parsed.time())
                                if tz:
                                    result = result.replace(tzinfo=tz)
                                return {"success": True, "datetime": result}
                        except Exception:
                            pass
                    
                    # Default to 9am if no time specified
                    result = datetime.combine(base_date, datetime.min.time().replace(hour=9))
                    if tz:
                        result = result.replace(tzinfo=tz)
                    return {"success": True, "datetime": result}
                except ValueError:
                    pass
        
        # === PATTERN 1A: "ordinal + of + month" (e.g., "fourth of February") ===
        # Sort by length (longest first) to match "twenty-first" before "first"
        sorted_ordinals = sorted(ordinal_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        for ordinal_word, day_num in sorted_ordinals:
            # Match "fourth of february"
            pattern_with_of = rf"\b{ordinal_word}\s+of\s+(\w+)\b"
            match = re.search(pattern_with_of, spoken_lower)
            
            # === PATTERN 1B: "ordinal + month" (e.g., "fourth February") ===
            # Match "fourth february" (no 'of')
            pattern_no_of = rf"\b{ordinal_word}\s+(\w+)\b"
            match_no_of = re.search(pattern_no_of, spoken_lower)
            
            final_match = match or match_no_of
            
            if final_match:
                potential_month = final_match.group(1).lower()
                # Check if the word is a valid month
                if potential_month in month_map:
                    month_num = month_map[potential_month]
                    
                    # Determine the year
                    year = now.year
                    if month_num < now.month or (month_num == now.month and day_num < now.day):
                        year += 1
                    
                    try:
                        base_date = datetime(year, month_num, day_num).date()
                        
                        # Use pre-extracted time if available (THE KEY FIX!)
                        if _extracted_time:
                            result = datetime.combine(base_date, _extracted_time)
                            if tz:
                                result = result.replace(tzinfo=tz)
                            return {"success": True, "datetime": result}
                        
                        # Fallback: Extract time from remaining string
                        after_pattern = spoken_clean[final_match.end():].strip()
                        if after_pattern:
                            try:
                                time_parsed = dtparser.parse(after_pattern, fuzzy=True)
                                if time_parsed:
                                    result = datetime.combine(base_date, time_parsed.time())
                                    if tz:
                                        result = result.replace(tzinfo=tz)
                                    return {"success": True, "datetime": result}
                            except Exception:
                                pass
                        
                        # Default to 9am
                        result = datetime.combine(base_date, datetime.min.time().replace(hour=9))
                        if tz:
                            result = result.replace(tzinfo=tz)
                        return {"success": True, "datetime": result}
                    except ValueError:
                        pass
        
        # === PATTERN 2: "month + ordinal" (e.g., "February fourth", "March 3rd") ===
        # Try to match "Month + Day" pattern
        for month_name, month_num in month_map.items():
            if month_name in spoken_lower:
                # Try to find a day number after the month
                # Pattern: "february third" or "february 3" or "feb 20th"
                month_pattern = rf"\b{month_name}\b"
                match = re.search(month_pattern, spoken_lower)
                if match:
                    # Get text after the month name
                    after_month = spoken_lower[match.end():].strip()
                    
                    # Try to extract day number (ordinal word or digit)
                    day_num = None
                    
                    # Check for ordinal words first
                    for ordinal_word, day_value in ordinal_map.items():
                        if after_month.startswith(ordinal_word):
                            day_num = day_value
                            # Remove the ordinal from the string to extract time
                            after_month = after_month[len(ordinal_word):].strip()
                            break
                    
                    # If no ordinal word, try to extract numeric day
                    if day_num is None:
                        day_match = re.match(r"(\d+)(?:st|nd|rd|th)?\b", after_month)
                        if day_match:
                            day_num = int(day_match.group(1))
                            after_month = after_month[day_match.end():].strip()
                    
                    if day_num and 1 <= day_num <= 31:
                        # Determine the year - if month has passed this year, use next year
                        year = now.year
                        if month_num < now.month or (month_num == now.month and day_num < now.day):
                            year += 1
                        
                        try:
                            base_date = datetime(year, month_num, day_num).date()
                            
                            # Use pre-extracted time if available
                            if _extracted_time:
                                result = datetime.combine(base_date, _extracted_time)
                                if tz:
                                    result = result.replace(tzinfo=tz)
                                return {"success": True, "datetime": result}
                            
                            # Fallback: Extract time from remaining string
                            # Clean filler words from the after_month portion
                            after_month_clean = re.sub(_FILLER_PATTERN, '', after_month, flags=re.IGNORECASE)
                            after_month_clean = re.sub(r'\s+', ' ', after_month_clean).strip()
                            if after_month_clean:
                                try:
                                    time_parsed = dtparser.parse(after_month_clean, fuzzy=True)
                                    if time_parsed:
                                        result = datetime.combine(base_date, time_parsed.time())
                                        if tz:
                                            result = result.replace(tzinfo=tz)
                                        return {"success": True, "datetime": result}
                                except Exception:
                                    pass
                            
                            # Default to 9am if no time specified
                            result = datetime.combine(base_date, datetime.min.time().replace(hour=9))
                            if tz:
                                result = result.replace(tzinfo=tz)
                            return {"success": True, "datetime": result}
                        except ValueError:
                            # Invalid date (e.g., Feb 30)
                            pass
                
                # If we found the month but couldn't parse day, break to avoid false matches
                break

        
        # ═══════════════════════════════════════════════════════════════════════════
        # 🛡️ INCOMPLETE DATE DETECTION — Prevent dateutil from guessing the day
        # ═══════════════════════════════════════════════════════════════════════════
        # If user mentions a month but no day number or relative day, return None
        # to let the caller ask for clarification instead of guessing.
        month_names_full = ["january", "february", "march", "april", "may", "june",
                           "july", "august", "september", "october", "november", "december",
                           "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
        relative_days = ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        
        has_month = any(m in spoken_lower for m in month_names_full)
        has_day_number = bool(re.search(r'\b(\d{1,2})(st|nd|rd|th)?\b', spoken_lower))
        has_ordinal_word = any(ord_word in spoken_lower for ord_word in ordinal_map.keys())
        has_relative_day = any(rday in spoken_lower for rday in relative_days)
        
        if has_month and not has_day_number and not has_ordinal_word and not has_relative_day:
            # User said something like "February at 3:30 PM" without specifying the day
            # Return failure dict to signal incomplete date; caller should ask for clarification
            logger.debug(f"[DATE_PARSE] Incomplete date: month mentioned but no day. Input: '{spoken}'")
            return {
                "success": False, 
                "datetime": None, 
                "needs_clarification": True, 
                "clarification_type": "missing_day", 
                "message": "Please specify the day."
            }
        
        
        # === FALLBACK: Use dateutil for absolute dates ===
        parsed = dtparser.parse(spoken, fuzzy=True)
        if parsed:
            if tz and parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=tz)
            return {"success": True, "datetime": parsed}
        
        return {"success": False, "datetime": None, "needs_clarification": False, "clarification_type": "", "message": "parse_failed"}
        
    except Exception as e:
        logger.error(f"[DATE_PARSE] Error parsing '{spoken}': {e}")
        return {"success": False, "datetime": None, "needs_clarification": False, "clarification_type": "", "message": "exception"}

"""
Patient state management and slot interpretation logic.
"""

from __future__ import annotations

import re
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Tuple, Dict, Any, List, Set

from config import DEFAULT_TZ, logger


# =============================================================================
# REGEX PATTERNS FOR CONFIRMATION DETECTION
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

def has_correction_intent(text: str) -> bool:
    """Detect if user is trying to correct a previous value."""
    if not text:
        return False
    # Common correction phrases
    patterns = [
        r"\bactually\b",
        r"\bno\b",
        r"\bnope\b",
        r"\bnot\s+that\b",
        r"\bsorry\b",
        r"\bi\s+mean\b",
        r"\bchange\s+it\b",
        r"\binstead\b",
        r"\bmake\s+it\b",
        r"\brather\b",
        r"\bdifferent\s+day\b",
        r"\banother\s+day\b",
        r"\bwrong\b",
        r"\bmistake\b"
    ]
    regex = re.compile("|".join(patterns), re.IGNORECASE)
    return bool(regex.search(text))


def is_valid_email_strict(text: str) -> bool:
    """Robust email validation requiring domain parts."""
    if not text or "@" not in text or "." not in text:
        return False
    # Basic check: contains @, dot, no spaces
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", text.strip()))


def is_valid_phone_strict(text: str) -> bool:
    """Strict phone check: requires sufficient digits for a real number."""
    digits = re.sub(r"\D", "", text)
    # Allow 7 digits (local w/o area code) to 15 digits (E.164)
    return 7 <= len(digits) <= 15


def is_fragment_of(fragment: str, full_value: str, slot_type: str) -> bool:
    """
    Check if 'fragment' is a substring or digit-subset of 'full_value'.
    Used to detect when user is SPELLING out a value rather than replacing it.
    """
    if not full_value:
        return False
        
    frag_clean = fragment.strip().lower()
    full_clean = full_value.strip().lower()

    if slot_type == "phone":
        # Phone: compare digit sequences
        d_frag = re.sub(r"\D", "", frag_clean)
        d_full = re.sub(r"\D", "", full_clean)
        # If fragment digits appear in full phone, it's a verification
        return (len(d_frag) >= 2 and d_frag in d_full)
    
    if slot_type == "email":
        # Email: check string containment or username part
        # Example: "moiz" in "moiz123@gmail.com"
        if frag_clean in full_clean:
            return True
        # Check digit parts (user often reads out digits in email)
        d_frag = re.sub(r"\D", "", frag_clean)
        d_full = re.sub(r"\D", "", full_clean)
        if len(d_frag) >= 3 and d_frag in d_full:
            return True
            
    return False


def interpret_followup_for_slot(
    slot_type: str, 
    current_value: str, 
    new_input: str
) -> Tuple[str, Optional[str]]:
    """
    Decide action when a new value arrives while awaiting confirmation.
    
    Returns: (Action, Reason/Message)
    Actions: "CONFIRM", "FRAGMENT", "CORRECTION", "OTHER"
    """
    new_clean = new_input.strip().lower()
    
    # 1. Detection Confirmation Keywords (explicit)
    if YES_PAT.search(new_clean):
        return "CONFIRM", "User explicitly confirmed"

    # 2. Detect Fragments (Verification)
    if is_fragment_of(new_input, current_value, slot_type):
        return "FRAGMENT", f"Input '{new_input}' is a fragment of '{current_value}'"

    # 3. Detect Corrections vs Noise
    if slot_type == "email":
        if is_valid_email_strict(new_input):
            return "CORRECTION", "Valid new email provided"
        # Invalid email and not a fragment -> likely noise or partial capture
        return "FRAGMENT", "Invalid email format during verification - treating as fragment/noise"
        
    if slot_type == "phone":
        if is_valid_phone_strict(new_input):
            return "CORRECTION", "Valid new phone provided"
        # Short digits not matching current -> likely noise?
        # But if it's "no, 555" (and 555 not in current), it might be a partial correction.
        # Safest: Treat as OTHER or FRAGMENT to avoid overwrite.
        return "FRAGMENT", "Partial/Invalid phone during verification - treating as fragment"

    return "OTHER", "Unrelated input"


def contact_phase_allowed(state: "PatientState") -> bool:
    """
    SINGLE SOURCE OF TRUTH: Contact details can only be collected/confirmed
    AFTER a valid time slot has been confirmed AND is available.
    
    Returns True when:
    - A datetime has been set (state.dt_local exists)
    - The time has been validated as available (state.time_status == "valid")
    - The slot availability has been confirmed (state.slot_available == True)
    
    OR when contact_phase_started flag is explicitly set (handles edge cases)
    """
    # Primary check: time validated and slot available
    primary_check = (
        state.time_status == "valid"
        and state.dt_local is not None
        and getattr(state, "slot_available", False) is True
    )
    
    # Fallback: contact phase explicitly started (handles edge cases)
    fallback_check = getattr(state, "contact_phase_started", False) is True
    
    return primary_check or fallback_check


# =============================================================================
# PATIENT STATE CLASS
# =============================================================================

@dataclass
class PatientState:
    """Clean state container for patient booking info."""
    full_name: Optional[str] = None
    phone_e164: Optional[str] = None  # Final confirmed phone
    phone_last4: Optional[str] = None
    email: Optional[str] = None
    reason: Optional[str] = None
    dt_local: Optional[datetime] = None
    dt_text: Optional[str] = None  # Natural language time before parsing
    
    # IDEMPOTENCY & LOCKING
    # Tracks unique turn ID to prevent duplicate tool execution
    turn_id: Optional[str] = None
    turn_count: int = 0
    last_user_text: Optional[str] = None
    # executed_tools: Dict[turn_id, Set[tool_signature]]
    executed_tools: Dict[str, Set[str]] = field(default_factory=dict)
    # Booking lock to prevent duplicate bookings
    last_booking_key: Optional[str] = None 
    
    # REJECTION HANDLING
    rejected_slots: Set[str] = field(default_factory=set)  # Set of rejected time strings/keys
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT-AWARE SLOT CONFIRMATION (Fix #3)
    # ═══════════════════════════════════════════════════════════════════════════
    # Tracks short-term verification mode to prevent "67932" overwriting full email
    awaiting_slot_confirmation: bool = False
    last_captured_slot: Optional[str] = None  # "email" | "phone"
    slot_confirm_turns_left: int = 0
    
    # Snapshots for reversion logic
    last_captured_email: Optional[str] = None
    last_captured_phone: Optional[str] = None
    
    # Buffers to hold fragments (logging only)
    email_verification_buffer: str = ""
    phone_verification_buffer: str = ""
    
    # Duration tracking (from treatment_durations config)
    duration_minutes: int = 60  # Default 60 min, updated when reason is set
    time_status: str = "pending"  # "pending", "validating", "valid", "invalid"
    time_error: Optional[str] = None  # Error message if time is invalid
    slot_available: bool = False  # True ONLY when slot is confirmed available
    
    # Phone lifecycle: detected_phone → phone_pending → phone_e164 (confirmed)
    detected_phone: Optional[str] = None  # From SIP, never spoken aloud
    phone_pending: Optional[str] = None   # Waiting for user confirmation
    
    # Confirmations
    phone_confirmed: bool = False
    email_confirmed: bool = False
    pending_confirm: Optional[str] = None  # "phone" or "email"
    
    # Phone source tracking (for confirmation UX)
    phone_source: Optional[str] = None  # "sip", "user_spoken", "extracted"
    
    # Contact phase gating - phone MUST NOT be mentioned until this is True
    contact_phase_started: bool = False
    
    # NEW: WhatsApp/SMS preference tracking
    prefers_sms: bool = False
    
    # NEW: Caller ID flow tracking
    caller_id_checked: bool = False
    
    # Review flow tracking
    review_presented: bool = False  # True after review summary shown
    review_snapshot: Optional[Dict[str, Any]] = None  # Snapshot at review time
    changed_fields: Optional[set] = field(default_factory=set)  # Fields changed after review
    pending_confirm_field: Optional[str] = None  # Single field needing confirmation
    partial_confirm_complete: bool = False  # True after confirming changed field
    
    # Booking state
    booking_attempted: bool = False
    booking_confirmed: bool = False
    booking_in_progress: bool = False
    calendar_event_id: Optional[str] = None
    
    # FIX 4: Date Locking Rule
    date_confirmed: bool = False
    date_source: Optional[str] = None  # "explicit_confirmed" | "date_time_together" | "inferred" | None
    proposed_date: Optional[date] = None  # Tentative date candidate
    confirmation_pending_for_date: bool = False
    last_date_candidate: Optional[date] = None
    
    # FIX 5: One Filler Per Turn
    filler_active: bool = False
    filler_turn_id: Optional[str] = None
    filler_task: Any = None  # asyncio.Task
    current_turn_id: Optional[str] = None
    last_filler_scheduled_at: Optional[float] = None
    real_response_started: bool = False

    # FIX 1: Semantic Buffering
    transcript_buffer: List[str] = field(default_factory=list)

    # FIX 2: Terminal State
    call_ended: bool = False

    # Context
    tz: str = DEFAULT_TZ

    patient_type: Optional[str] = None
    
    def add_rejected_slot(self, dt: datetime, reason: str = "user_rejected"):
        """Track rejected slots to avoid re-suggesting them."""
        key = dt.strftime("%Y-%m-%d %H:%M")
        self.rejected_slots.add(key)
        logger.info(f"[STATE] 🚫 Slot rejected: {key} ({reason})")

    def is_slot_rejected(self, dt: datetime) -> bool:
        """Check if a slot was previously rejected."""
        key = dt.strftime("%Y-%m-%d %H:%M")
        return key in self.rejected_slots

    def assert_integrity(self, context: str):
        """
        Verify state hasn't been accidentally wiped.
        Logs WARNING if critical fields disappear.
        """
        if self.full_name is None and self.turn_count > 2:
             logger.warning(f"[INTEGRITY] ⚠️ Name text lost during {context}!")
        if self.phone_e164 and not isinstance(self.phone_e164, str):
             logger.warning(f"[INTEGRITY] ⚠️ Phone type corruption in {context}: {type(self.phone_e164)}")

    def start_new_turn(self, user_text: str):
        """Start a new turn tracking context."""
        self.turn_count += 1
        
        # FIX #3: Context-Aware Verification Countdown
        if self.awaiting_slot_confirmation:
            if self.slot_confirm_turns_left > 0:
                self.slot_confirm_turns_left -= 1
                logger.debug(f"[SLOT_CONFIRM] ⏳ Verification window active for {self.last_captured_slot} (turns left: {self.slot_confirm_turns_left})")
            else:
                self.awaiting_slot_confirmation = False
                self.last_captured_slot = None
                logger.debug(f"[SLOT_CONFIRM] ⏹️ Verification window expired")

        # Create unique ID for this turn
        # Hash user text + timestamp + turn count
        raw = f"{self.turn_count}:{user_text[:50]}:{time.time()}"
        self.turn_id = hashlib.md5(raw.encode()).hexdigest()[:8]
        self.last_user_text = user_text
        # Clean up old turn history to prevent memory leak (keep last 5 turns)
        if len(self.executed_tools) > 5:
            sorted_keys = sorted(self.executed_tools.keys())
            for k in sorted_keys[:-5]:
                del self.executed_tools[k]
        # Initialize set for this turn
        self.executed_tools[self.turn_id] = set()
        logger.debug(f"[STATE] 🔄 New Turn {self.turn_count} (ID: {self.turn_id})")

    def check_tool_lock(self, tool_name: str, args: dict) -> bool:
        """
        Check if tool was already executed with same args in this turn.
        Returns TRUE if locked (should skip), FALSE if allowed.
        """
        if not self.turn_id:
            return False  # No turn context, allow execution
            
        # Create deterministic signature of relevant args
        # Filter out None values to handle optional args consistently
        clean_args = {k: str(v) for k, v in args.items() if v is not None and k != 'self'}
        
        # Sort keys for stability
        sig_str = f"{tool_name}:" + ",".join(f"{k}={clean_args[k]}" for k in sorted(clean_args.keys()))
        
        sig_hash = hashlib.md5(sig_str.encode()).hexdigest()[:8]
        
        lock_key = f"{tool_name}:{sig_hash}"
        
        if lock_key in self.executed_tools.get(self.turn_id, set()):
            logger.warning(f"[IDEMPOTENCY] 🔒 Skipping duplicate {tool_name} for turn {self.turn_id}")
            return True
            
        # Mark as executed
        if self.turn_id not in self.executed_tools:
            self.executed_tools[self.turn_id] = set()
        self.executed_tools[self.turn_id].add(lock_key)
        return False

    def should_update_field(self, field_name: str, current_value: Any, new_value: Any) -> bool:
        """
        Smart field update guard with context-aware logic.
        Returns True if update allowed, False if should be skipped.
        
        ALLOWS updates when:
        1. Field is empty (first-time population)
        2. User explicitly confirms/accepts (yes, okay, sure, etc.)
        3. User provides explicit correction markers
        4. New value is significantly different (not a fragment)
        
        BLOCKS updates when:
        1. New value is empty/None
        2. Values are identical
        3. New value appears to be a fragment/verification of current value
        """
        # 1. New value is empty -> prevent clearing
        if not new_value:
            return False
            
        # Normalization for comparison
        curr_str = str(current_value).strip().lower() if current_value else ""
        new_str = str(new_value).strip().lower() if new_value else ""
        
        # 2. Values identical -> No-op
        if curr_str == new_str:
            return False
            
        # 3. Current is empty -> Update allowed (first-time population)
        if not current_value:
            logger.info(f"[UPDATE] ✅ Setting {field_name}: '{new_value}' (first time)")
            return True
        
        # 4. Check user intent from last utterance
        user_text = (self.last_user_text or "").lower()
        
        # 4a. Explicit confirmation/acceptance
        confirmation_patterns = [
            "yes", "yeah", "yep", "yup", "correct", "right", "that's right",
            "ok", "okay", "sure", "sounds good", "perfect", "great",
            "that works", "that's fine", "book it", "confirm"
        ]
        has_confirmation = any(pattern in user_text for pattern in confirmation_patterns)
        
        if has_confirmation:
            logger.info(f"[UPDATE] ✅ Updating {field_name}: '{current_value}' -> '{new_value}' (User confirmed)")
            return True
        
        # 4b. Explicit correction markers
        correction_markers = [
            "actually", "no", "nope", "sorry", "change", "instead", 
            "i mean", "not that", "not", "wrong", "mistake", 
            "correction", "it's", "it is", "my name is", 
            "my phone", "my email", "make it", "rather"
        ]
        has_correction = any(marker in user_text for marker in correction_markers)
        
        if has_correction:
            logger.info(f"[UPDATE] ✏️ Overwriting {field_name}: '{current_value}' -> '{new_value}' (Correction detected)")
            return True
        
        # 4c. Check if new value is a fragment/verification of current value
        # (e.g., user saying "67932" when email is "moiz67932@gmail.com")
        if field_name in ["phone", "email"]:
            # For phone: check if new value is subset of current digits
            if field_name == "phone":
                curr_digits = re.sub(r"\D", "", curr_str)
                new_digits = re.sub(r"\D", "", new_str)
                
                # 🛡️ INTENT-BASED SLOT ISOLATION: 
                # If we are currently capturing an email that contains these same digits,
                # and the user hasn't expressed clear intent to give a phone number, block it.
                if self.awaiting_slot_confirmation and self.last_captured_slot == "email":
                    email_digits = re.sub(r"\D", "", self.last_captured_email or "")
                    if new_digits and new_digits in email_digits and not has_correction:
                        logger.info(f"[UPDATE] 🛡️ Blocked phone update '{new_value}' - likely digits from spelling email '{self.last_captured_email}'")
                        return False

                if new_digits and new_digits in curr_digits and len(new_digits) < len(curr_digits):
                    logger.info(f"[UPDATE] 🛡️ Ignoring {field_name} fragment: '{new_value}' is part of '{current_value}'")
                    return False
            
            # For email: check if new value is substring of current
            if field_name == "email":
                if new_str in curr_str and len(new_str) < len(curr_str):
                    logger.info(f"[UPDATE] 🛡️ Ignoring {field_name} fragment: '{new_value}' is part of '{current_value}'")
                    return False
        
        # 5. For time field: allow updates more liberally (agent often suggests alternatives)
        if field_name == "time":
            logger.info(f"[UPDATE] ⏰ Updating {field_name}: '{current_value}' -> '{new_value}' (Time update allowed)")
            return True
        
        # 6. Default: Block update without explicit intent
        logger.info(f"[UPDATE] 🛡️ Ignoring {field_name} change: '{current_value}' -> '{new_value}' (No clear intent detected)")
        return False

    def is_complete(self) -> bool:
        """Check if we have all required info for booking (email NOT required)."""
        return all([
            self.full_name,
            self.phone_e164,
            self.phone_confirmed,
            # EMAIL SUPPRESSED — kept for future use but not required for booking
            # self.email,
            # self.email_confirmed,
            self.reason,
            self.dt_local,
        ])
    
    def missing_slots(self) -> List[str]:
        """Return list of missing required slots (email excluded)."""
        missing = []
        if not self.full_name:
            missing.append("full_name")
        if not self.phone_e164:
            missing.append("phone")
        elif not self.phone_confirmed:
            missing.append("phone_confirmed")
        # EMAIL SUPPRESSED — kept for future use but not required for booking
        # if not self.email:
        #     missing.append("email")
        # elif not self.email_confirmed:
        #     missing.append("email_confirmed")
        if not self.reason:
            missing.append("reason")
        if not self.dt_local:
            missing.append("datetime")
        return missing
    
    def slot_summary(self) -> str:
        """Human-readable slot summary for logging."""
        return (
            f"name={self.full_name or '?'}, "
            f"phone={'✓' if self.phone_confirmed else (self.phone_last4 or '?')}, "
            f"email={'✓' if self.email_confirmed else (self.email or '?')}, "
            f"reason={self.reason or '?'}, "
            f"time={self.dt_local.isoformat() if self.dt_local else '?'}"
        )
    
    def detailed_state_for_prompt(self) -> str:
        """
        Generate a concise state snapshot for the dynamic system prompt.
        This is the LLM's 'source of truth' for what's already captured.
        OPTIMIZED: Reduced redundancy to minimize prompt tokens while preserving semantics.
        """
        lines = []
        
        # Name (concise)
        if self.full_name:
            lines.append(f"• NAME: ✓ {self.full_name}")
        else:
            lines.append("• NAME: ? — Ask naturally")
        
        # Phone - only show if contact phase started (prevents early confirmation)
        # Use pending phone if available, otherwise detected phone
        phone_display = self.phone_e164 or self.phone_pending or self.detected_phone
        if isinstance(phone_display, tuple):
            phone_display = phone_display[0] if phone_display else None
        
        if not contact_phase_allowed(self):
            # Contact phase not started - hide phone from prompt to prevent early mention
            lines.append("• PHONE: — (collect after time confirmed)")
        elif self.phone_e164 and self.phone_confirmed:
            lines.append(f"• PHONE: ✓ {self.phone_e164}") # Show full confirmed number to LLM
        elif self.phone_pending or self.detected_phone:
            # PENDING implies we have a number but need confirmation
            # If source is caller_id (inferred if detected_phone matches), we ask "save this number?"
            # If source is user_spoken, we ask "I have [full number], is that right?"
            # The prompt doesn't need to be micro-managed, just indicate status.
            lines.append(f"• PHONE: ⏳ Pending Confirmation — Ask: 'Should I save the number you're calling from?'")
        else:
            lines.append("• PHONE: ? — Ask naturally")
        
        # Email - SUPPRESSED: Do NOT ask for email
        # Only show email if user voluntarily provided it (keep code intact for future)
        if self.email and self.email_confirmed:
            lines.append(f"• EMAIL: ✓ {self.email}")
        elif self.email:
            # Email captured but not confirmed - just note it, don't prompt confirmation
            lines.append(f"• EMAIL: {self.email} (captured)")
        # DO NOT add "? — Ask naturally" line - email collection is suppressed
        
        # Reason (concise)
        if self.reason:
            lines.append(f"• REASON: ✓ {self.reason} ({self.duration_minutes}m)")
        else:
            lines.append("• REASON: ? — Ask what brings them in")
        
        # Time with validation status (concise)
        if self.dt_local and self.time_status == "valid":
            time_str = self.dt_local.strftime('%a %b %d @ %I:%M %p')
            lines.append(f"• TIME: ✓ {time_str}")
        elif self.time_status == "invalid" and self.time_error:
            lines.append(f"• TIME: ❌ {self.time_error}")
        elif self.dt_text:
            lines.append(f"• TIME: ⏳ '{self.dt_text}' — {self.time_status}")
        else:
            lines.append("• TIME: ? — Ask when")
        
        # Booking status (concise)
        if self.booking_confirmed:
            lines.append("🎉 BOOKED!")
        elif self.is_complete():
            lines.append("✅ READY TO BOOK — Summarize & confirm")
        else:
            missing = [s for s in self.missing_slots() if not s.endswith('_confirmed')]
            if missing:
                lines.append(f"⏳ NEED: {', '.join(missing)}")
        
        return '\n'.join(lines)

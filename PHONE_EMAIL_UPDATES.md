# Phone and Email Handling Updates

## Summary
Updated phone and email confirmation system to show complete information and added repeat functionality.

## Changes Made

### 1. Phone Confirmation Messages
**Changed from:** "I have a number ending in {last 4 digits} — is that okay?"
**Changed to:** "Should I save the number you called from for appointment details?"

**Rationale:** More natural, conversational, and faster than reading out the complete number. The user already knows they called from that number.

#### Files Modified:
- `tools/assistant_tools.py`
  - Line 488: Updated time slot confirmation message
  - Line 556: Updated contact phase confirmation message  
  - Line 1106: Updated tool description example
  - Line 1182: Updated confirm_phone correction message

### 2. Database Storage
**Changed:** Save complete phone number instead of just last 4 digits

#### Files Modified:
- `services/database_service.py`
  - Line 302: Changed `patient_phone_masked` field to save `phone_e164 or phone_pending` (complete number) instead of `phone_last4`

### 3. Booking Confirmation
**Changed:** Show complete phone number in final confirmation

#### Files Modified:
- `tools/assistant_tools.py`
  - Line 1788-1791: Updated final booking confirmation to show complete number using `speakable_phone()` function

### 4. New Repeat Functionality
Added two new tools that only work when explicitly requested by the user:

#### `repeat_phone` Tool
- **Purpose:** Repeat the phone number when user asks
- **Triggers:** "can you repeat the number?", "what's the phone number?", "say that again"
- **Returns:** Complete phone number in speakable format
- **Location:** `tools/assistant_tools.py`, lines 1319-1346

#### `repeat_email` Tool  
- **Purpose:** Repeat the email address when user asks
- **Triggers:** "can you repeat the email?", "what's the email?", "say that again"
- **Returns:** Email address in speakable format (e.g., "moiz at gmail dot com")
- **Location:** `tools/assistant_tools.py`, lines 1349-1373

## Technical Details

### Phone Number Format
- **Storage:** E.164 format (e.g., "+923351234567")
- **Speech:** Human-readable format using `speakable_phone()` function
- **Logging:** Last 4 digits only (for privacy in logs)

### Email Format
- **Storage:** Lowercase, normalized (e.g., "user@example.com")
- **Speech:** Spoken format using `email_for_speech()` function (e.g., "user at example dot com")

## Behavior Changes

### Before:
1. Agent: "I have a number ending in 1234 — is that okay?"
2. Database stored only last 4 digits
3. No way to repeat the information

### After:
1. Agent: "Should I save the number you called from for appointment details?"
2. Database stores complete phone number
3. User can ask "Can you repeat my phone number?" or "Can you repeat my email?"
4. Agent will read full number when explicitly requested via repeat_phone tool

## Safety Features

1. **Idempotency:** Repeat tools check for duplicate calls
2. **Validation:** Only repeat if data exists in state
3. **Privacy:** Logs mask sensitive data (e.g., ***1234)
4. **Explicit Triggering:** Repeat tools ONLY activate when user explicitly requests

## Testing Recommendations

1. Test phone confirmation with various formats
2. Verify database stores complete phone number
3. Test repeat tools with explicit requests
4. Ensure repeat tools don't trigger automatically
5. Verify final booking confirmation shows complete number

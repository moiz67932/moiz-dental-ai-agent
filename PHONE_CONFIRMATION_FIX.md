# Phone Confirmation Error Fix

## Problem
When the agent asked "Should I use the number you're calling from for appointment details?" and the user said "yes", the `confirm_phone` tool was blocked with the error:

```
[TOOL] confirm_phone blocked - contact phase not started
```

This prevented the booking from completing.

## Root Cause
The agent was asking about the phone number in its **natural language response** instead of calling the `update_patient_record` tool. This meant that:

1. `contact_phase_started` was never set to `True` (this flag is set inside `update_patient_record` when name + time + slot availability are all confirmed)
2. When the user said "yes" to confirm the phone, the `confirm_phone` tool checked if `contact_phase_started == True`
3. Since it was still `False`, the tool was blocked

## The Fix
Updated `prompts/agent_prompts.py` with two key changes:

### 1. Explicit instruction to call tool after time confirmation (Lines 36-38)
```python
• CRITICAL: After suggesting a time and user confirms it (says "yes", "that works", etc.), 
  you MUST call update_patient_record(time_suggestion="<the confirmed time>") to finalize it.
  DO NOT just respond naturally - the tool MUST be called to trigger contact phase.
```

This ensures the agent calls `update_patient_record` after the user confirms a suggested time, which:
- Sets `contact_phase_started = True`
- Sets `slot_available = True`
- Returns the phone confirmation message

### 2. Clarified phone confirmation flow (Lines 47-49)
```python
• ⚡ IMPORTANT: The update_patient_record tool will AUTOMATICALLY ask about phone when ready.
  DO NOT manually ask "Should I use the number you're calling from?" in your response.
  The tool will return this question when the time is confirmed.
```

This prevents the agent from asking about the phone in its own response, ensuring the tool's return value is used instead.

## How It Works Now
1. User requests appointment: "I want to book a teeth whitening"
2. Agent suggests time: "How about February 15th at 2 PM?"
3. User confirms: "Yes, that works"
4. **Agent calls `update_patient_record(time_suggestion="February 15th at 2 PM")`**
5. Tool validates time, sets `contact_phase_started = True`, and returns: "... ah, perfect! February 15th at 2 PM is open. Should I save the number you called from for appointment details?"
6. User confirms: "Yes"
7. **Agent calls `confirm_phone(confirmed=True)`** - now allowed because `contact_phase_started == True`
8. Booking proceeds successfully

## Testing
After restarting the worker with these changes:
1. Test the full booking flow
2. Verify that after confirming a time, the agent calls `update_patient_record`
3. Verify that the phone confirmation works without blocking
4. Check logs for `[TOOL] confirm_phone blocked` - should not appear

## Files Modified
- `prompts/agent_prompts.py`: Updated tool usage instructions and phone confirmation flow

# Quick Reference: Cancellation & Rescheduling Agent Behavior

## Intent Keywords (Auto-Trigger Appointment Lookup)

When the user says any of these phrases, the agent automatically calls `find_existing_appointment()`:

### Cancellation Intent
- "cancel"
- "cancel my appointment"
- "cancel booking"
- "I need to cancel"
- "want to cancel"
- "have to cancel"

### Rescheduling Intent
- "reschedule"
- "reschedule my appointment"
- "move my appointment"
- "change my appointment"
- "change the time"
- "different day"
- "different time"

## Agent Response Templates

### 1. Appointment Found
```
"I found your appointment for [REASON] on [DAY, DATE] at [TIME]. 
Is this the appointment you'd like to [cancel/reschedule]?"
```

### 2. Appointment Not Found (Caller's Phone)
```
"I don't see an upcoming appointment with that number. 
What phone number did you use when you booked?"
```

### 3. Cancellation Confirmation Request
```
"Just to confirm - you want to cancel your [REASON] appointment 
on [DAY, DATE] at [TIME]. Is that correct?"
```

### 4. Cancellation Success
```
"All done — your [REASON] appointment on [DAY, DATE] at [TIME] 
has been cancelled. Is there anything else I can help you with?"
```

### 5. Rescheduling - Ask Preference
```
"Do you have a specific day or time in mind, 
or would you like me to suggest some available options?"
```

### 6. Rescheduling - Offer Slots (Max 3)
```
"I have Tuesday at 10 AM, Wednesday at 2 PM, or Thursday at 9 AM."
```

### 7. Rescheduling - Slot Taken with Alternatives
```
"That time slot is taken. The closest I have is [TIME]. Would that work?"
```
OR
```
"That slot is booked. I can do [TIME1] or [TIME2]. Which works for you?"
```

### 8. Rescheduling Confirmation Request
```
"Perfect! Just to confirm — I'll move your [REASON] from [OLD DAY, DATE] 
at [OLD TIME] to [NEW DAY, DATE] at [NEW TIME]. Is that correct?"
```

### 9. Rescheduling Success
```
"All set — your [REASON] appointment has been moved to [DAY, DATE] 
at [TIME]. Is there anything else I can help you with?"
```

### 10. Error/System Issue
```
"I'm having trouble [cancelling/rescheduling] that appointment right now. 
Could you try again in a moment, or would you like to speak with the office?"
```

## Critical Rules

### ❌ NEVER Do This:
- Cancel or reschedule without explicit user confirmation
- Assume which appointment the user wants to modify
- Ask for phone number BEFORE attempting automatic lookup
- Use technical language or system error messages
- Be judgmental about cancellations

### ✅ ALWAYS Do This:
- Try automatic phone lookup first
- Confirm appointment details before action
- Offer alternatives when requested time is unavailable
- Use reassuring, non-judgmental tone
- End with "Is there anything else I can help you with?"

## Conversation Flow Examples

### Fast Cancellation (2 turns)
```
User: "Cancel my appointment"
Agent: "I found your cleaning on Monday, Feb 3 at 2 PM. Is this the one you'd like to cancel?"
User: "Yes"
Agent: "All done — your cleaning appointment on Monday, Feb 3 at 2 PM has been cancelled. 
        Is there anything else I can help you with?"
```

### Fast Rescheduling with Specific Time (3 turns)
```
User: "Reschedule to Tuesday at 10"
Agent: "I found your cleaning on Monday at 2 PM. Is this the one you'd like to reschedule?"
User: "Yes"
Agent: "Perfect! Just to confirm — I'll move your cleaning from Monday at 2 PM to Tuesday at 10 AM. 
        Is that correct?"
User: "Yes"
Agent: "All set — your cleaning appointment has been moved to Tuesday at 10 AM. 
        Is there anything else I can help you with?"
```

### Rescheduling with Suggestions (5 turns)
```
User: "I need to reschedule"
Agent: "I found your cleaning on Monday at 2 PM. Is this the one you'd like to reschedule?"
User: "Yes"
Agent: "Do you have a specific day or time in mind, or would you like me to suggest some options?"
User: "What do you have?"
Agent: "I have Tuesday at 10 AM, Wednesday at 2 PM, or Thursday at 9 AM."
User: "Wednesday at 2"
Agent: "Perfect! Just to confirm — I'll move your cleaning from Monday at 2 PM to Wednesday at 2 PM. 
        Is that correct?"
User: "Yes"
Agent: "All set — your cleaning has been moved to Wednesday at 2 PM. 
        Is there anything else I can help you with?"
```

## State Management

### State Variables Used:
- `found_appointment_id` - Stored appointment ID for operations
- `found_appointment_details` - Full appointment object with dates, reason, etc.
- `detected_phone` / `phone_e164` / `phone_pending` - Phone sources for lookup

### State Cleanup:
After successful cancellation or rescheduling:
- `found_appointment_id` is deleted
- `found_appointment_details` is deleted

## Time Handling

### Supported Time Formats:
- "tomorrow at 3pm"
- "next Monday at 10am"
- "Tuesday at 2:30"
- "February 5th at 10am"
- "next week Wednesday morning"

### Validation:
1. ✅ Working hours check (uses clinic schedule)
2. ✅ Slot availability check (database query)
3. ✅ Buffer time check (15 minutes between appointments)

### If Slot Taken:
- Automatically suggest up to 3 nearby alternatives
- Within ±4 hours of requested time
- 15-minute intervals

## Error Scenarios

### No Phone Number Available
→ Ask: "What phone number did you use when you booked?"

### Appointment Not Found After Manual Phone
→ "I don't see an upcoming appointment with that number."

### Invalid Time Format
→ "I couldn't understand '[TIME]'. Could you try a different way?"

### Slot Unavailable (No Alternatives)
→ "That time slot is taken and I don't see openings nearby. Would you like to try a different day?"

### Database Error
→ "I'm having trouble [action] right now. Could you try again in a moment, or would you like to speak with the office?"

## Pro Tips for Natural Flow

1. **Parallel Processing**: Appointment lookup happens while agent is speaking initial response
2. **No Filler Words**: Removed "um", "let me check" — just direct answers
3. **Confirmation First**: Always confirm appointment details before asking about changes
4. **Options Limit**: Never offer more than 3 time slots (prevents choice paralysis)
5. **Human Language**: "Monday, Feb 3" not "2026-02-03" or "next month"
6. **Reassuring Tone**: "All done" and "All set" not "Completed" or "Success"

## Integration Points

### Database Tables:
- `appointments` - Main storage
  - Status: `scheduled` → `cancelled`
  - Updates: `start_time`, `end_time`, `cancelled_at`

### Related Tools:
- `get_available_slots()` - For suggesting times
- `suggest_slots_around()` - For nearby alternatives
- `is_within_working_hours()` - For validation
- `is_slot_free_supabase()` - For availability check

### External Systems:
- Google Calendar (future): Event deletion/update
- SMS/Email (future): Confirmation notifications

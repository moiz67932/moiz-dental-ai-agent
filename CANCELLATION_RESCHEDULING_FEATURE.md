# Cancellation and Rescheduling Feature

## Overview
This feature implements a fast, human, and mistake-proof flow for canceling and rescheduling dental appointments via voice conversation.

## Key Features

### 1. **Instant Intent Detection**
The agent immediately detects cancellation/rescheduling intent when users say phrases like:
- "I need to cancel"
- "Cancel my appointment"
- "Reschedule my appointment"
- "Move my appointment"
- "Change my appointment time"

### 2. **Automatic Appointment Lookup**
Upon detecting the intent, the agent **silently and automatically**:
- Uses the caller's phone number from the call context
- Searches the database for existing appointments
- Retrieves appointment details without asking the user for their phone number first

### 3. **Mistake-Proof Confirmation**
The agent ALWAYS confirms before taking action:
- **For cancellation**: Reads back the appointment details and asks "Is this the appointment you'd like to cancel?"
- **For rescheduling**: Confirms both the original appointment AND the new time before making changes

### 4. **Graceful Fallback**
If no appointment is found using the caller's number:
- Agent politely asks: "What phone number did you use when booking?"
- Re-checks the database with the provided phone number
- Informs the user calmly if still not found

### 5. **Human-Centric Tone**
Throughout the process, the agent:
- Uses reassuring, non-judgmental language
- Avoids technical or system-level explanations
- Handles errors gracefully without blame

## Implementation Details

### New Service: `appointment_management_service.py`

Located at: `services/appointment_management_service.py`

**Functions:**
- `find_appointment_by_phone()` - Search for appointments by phone number
- `cancel_appointment()` - Update appointment status to 'cancelled'
- `reschedule_appointment()` - Update appointment to new time slot
- `find_all_appointments_by_phone()` - Get all appointments for disambiguation

### New Tools in `assistant_tools.py`

#### 1. `find_existing_appointment()`
- Automatically called when cancellation/rescheduling intent is detected
- Uses caller's phone number from state (detected_phone, phone_e164, or phone_pending)
- Returns appointment details if found, or requests phone number if not available

#### 2. `cancel_appointment_tool(confirmed: bool)`
- Requires explicit user confirmation before canceling
- Updates appointment status to 'cancelled' in database
- Provides reassuring confirmation message
- Clears appointment from state after successful cancellation

#### 3. `reschedule_appointment_tool(new_time: str, confirmed: bool)`
- Asks if user has a specific time or wants suggestions
- Validates new time against working hours and availability
- Offers up to 3 alternative slots if requested time is taken
- Requires confirmation for both original appointment AND new time
- Updates appointment in database with new times

## Workflow Examples

### Cancellation Flow
```
User: "I need to cancel my appointment"
Agent: [Calls find_existing_appointment silently]
Agent: "I found your appointment for cleaning on Monday, February 3rd at 2 PM. Is this the appointment you'd like to cancel?"
User: "Yes"
Agent: [Calls cancel_appointment_tool with confirmed=True]
Agent: "All done — your cleaning appointment on Monday, February 3rd at 2 PM has been cancelled. Is there anything else I can help you with?"
```

### Rescheduling Flow
```
User: "I need to reschedule my appointment"
Agent: [Calls find_existing_appointment silently]
Agent: "I found your cleaning on Monday, February 3rd at 2 PM. Is this the one you'd like to reschedule?"
User: "Yes"
Agent: "Do you have a specific day or time in mind, or would you like me to suggest some available options?"
User: "What do you have available?"
Agent: [Calls get_available_slots]
Agent: "I have Tuesday at 10 AM, Wednesday at 2 PM, or Thursday at 9 AM."
User: "Tuesday at 10 works"
Agent: [Validates slot availability]
Agent: "Perfect! Just to confirm — I'll move your cleaning from Monday, February 3rd at 2 PM to Tuesday, February 4th at 10 AM. Is that correct?"
User: "Yes"
Agent: [Calls reschedule_appointment_tool with new_time="Tuesday at 10 AM", confirmed=True]
Agent: "All set — your cleaning appointment has been moved to Tuesday, February 4th at 10 AM. Is there anything else I can help you with?"
```

### No Appointment Found Flow
```
User: "I want to cancel my appointment"
Agent: [Calls find_existing_appointment silently]
Agent: "I don't see an upcoming appointment with that number. What phone number did you use when you booked?"
User: "555-123-4567"
Agent: [Re-searches with provided number]
Agent: "I found your appointment for cleaning on Monday at 2 PM. Is this the one you'd like to cancel?"
```

## Database Schema

The feature uses the existing `appointments` table with the following fields:
- `id` - Unique appointment identifier
- `clinic_id` - Clinic identifier
- `patient_name` - Patient name
- `patient_phone_masked` - Phone number (E164 format)
- `patient_email` - Email address
- `start_time` - Appointment start time (ISO format)
- `end_time` - Appointment end time (ISO format)
- `status` - Appointment status (scheduled, confirmed, cancelled)
- `reason` - Reason for visit
- `calendar_event_id` - Google Calendar event ID (if synced)

## Safety Features

1. **Idempotency Protection**: Prevents duplicate cancellations/reschedules within the same conversation turn
2. **Explicit Confirmation Required**: Never cancels or reschedules without user confirmation
3. **State Cleanup**: Clears appointment data from state after successful operation
4. **Availability Validation**: Always checks if new time slot is available before rescheduling
5. **Working Hours Validation**: Ensures rescheduled times are within clinic operating hours

## Error Handling

The implementation includes graceful error handling for:
- Database connection issues
- Missing phone numbers
- Appointment not found
- Time slot unavailable
- Invalid time formats
- System failures

All errors are presented to the user in a calm, non-technical manner.

## Benefits

### For the User:
✅ **Fast**: Appointment found automatically, no need to provide details  
✅ **Human**: Natural conversation flow, reassuring tone  
✅ **Mistake-proof**: Double confirmation prevents accidental cancellations  
✅ **Easier than calling**: No hold times, instant processing  

### For the Clinic:
✅ **Reduced no-shows**: Easy cancellation encourages advance notice  
✅ **Better scheduling**: Freed slots can be immediately rebooked  
✅ **Staff efficiency**: Reduces receptionist workload  
✅ **Professional experience**: Enhances clinic reputation  

## Future Enhancements

Potential improvements for future versions:
- [ ] Support for multiple upcoming appointments (disambiguation)
- [ ] Cancellation reason tracking for analytics
- [ ] SMS/Email confirmation after cancellation/rescheduling
- [ ] Integration with Google Calendar to update/delete events
- [ ] Waitlist notification when slots become available
- [ ] Support for recurring appointment patterns

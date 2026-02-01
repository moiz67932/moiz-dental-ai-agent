# Testing Checklist: Cancellation & Rescheduling

## Pre-Testing Setup

- [ ] Database has test appointments with various phone numbers
- [ ] Test clinic has working hours configured in schedule
- [ ] Agent has access to clinic context and schedule
- [ ] Logging is enabled for debugging

## Unit Tests

Run: `pytest test_cancellation_rescheduling.py -v`

- [ ] All appointment lookup tests pass
- [ ] All cancellation tests pass
- [ ] All rescheduling tests pass
- [ ] All tool integration tests pass
- [ ] All edge case tests pass

## Integration Tests

### 1. Cancellation Flow - Happy Path

**Setup**: Create appointment for phone +1-917-555-1234 on Monday at 2 PM

**Test Steps**:
1. [ ] Call agent from +1-917-555-1234
2. [ ] Say: "I need to cancel my appointment"
3. [ ] Verify: Agent finds appointment and asks for confirmation
4. [ ] Say: "Yes"
5. [ ] Verify: Agent confirms cancellation
6. [ ] Check: Database shows appointment status = 'cancelled'

**Expected Time**: < 10 seconds (2 conversation turns)

---

### 2. Cancellation Flow - Phone Mismatch

**Setup**: Create appointment for phone +1-917-555-5678

**Test Steps**:
1. [ ] Call agent from different number
2. [ ] Say: "Cancel my appointment"
3. [ ] Verify: Agent asks for phone number
4. [ ] Say: "555-5678"
5. [ ] Verify: Agent finds appointment and asks for confirmation
6. [ ] Say: "Yes"
7. [ ] Verify: Cancellation confirmed

---

### 3. Rescheduling Flow - Specific Time

**Setup**: Create appointment for Monday at 2 PM, ensure Tuesday at 10 AM is available

**Test Steps**:
1. [ ] Say: "Reschedule to Tuesday at 10"
2. [ ] Verify: Agent identifies appointment
3. [ ] Say: "Yes"
4. [ ] Verify: Agent checks availability and asks for final confirmation
5. [ ] Say: "Yes"
6. [ ] Verify: Agent confirms new time
7. [ ] Check: Database shows new start_time and end_time

---

### 4. Rescheduling Flow - Request Suggestions

**Setup**: Create appointment for Monday at 2 PM

**Test Steps**:
1. [ ] Say: "I need to reschedule"
2. [ ] Verify: Agent identifies appointment
3. [ ] Say: "Yes"
4. [ ] Verify: Agent asks for preference
5. [ ] Say: "What do you have available?"
6. [ ] Verify: Agent offers 3 time slots
7. [ ] Say: "The first one"  # Or specific time
8. [ ] Verify: Agent asks for final confirmation
9. [ ] Say: "Yes"
10. [ ] Verify: Appointment rescheduled

---

### 5. Rescheduling Flow - Requested Time Unavailable

**Setup**: Create appointment, ensure requested time is blocked

**Test Steps**:
1. [ ] Say: "Reschedule to Tuesday at 2 PM"
2. [ ] Verify: Agent identifies appointment
3. [ ] Say: "Yes"
4. [ ] Verify: Agent checks availability
5. [ ] Verify: Agent says time is taken and offers alternatives
6. [ ] Say: "The second option"  # Or specific time
7. [ ] Verify: Appointment rescheduled to alternative

---

### 6. Error Handling - No Appointment Found

**Test Steps**:
1. [ ] Call from phone with no appointments
2. [ ] Say: "Cancel my appointment"
3. [ ] Verify: Agent asks for booking phone number
4. [ ] Say: "555-9999"  # Non-existent
5. [ ] Verify: Agent politely says no appointment found
6. [ ] Verify: No errors or crashes

---

### 7. Error Handling - Invalid Time Format

**Setup**: Create appointment for Monday at 2 PM

**Test Steps**:
1. [ ] Say: "Reschedule my appointment"
2. [ ] Verify: Agent identifies appointment
3. [ ] Say: "Yes"
4. [ ] Say: "blahblah"  # Nonsense time
5. [ ] Verify: Agent asks for clarification
6. [ ] Say: "Tuesday at 10"
7. [ ] Verify: Flow continues normally

---

### 8. Edge Case - Multiple Confirmations

**Setup**: Create appointment

**Test Steps**:
1. [ ] Say: "Cancel"
2. [ ] Verify: Agent finds appointment
3. [ ] Say: "No, actually yes"  # Confusing input
4. [ ] Verify: Agent handles gracefully (asks for clarification or processes based on final intent)

---

### 9. Edge Case - Change Mind Mid-Flow

**Setup**: Create appointment

**Test Steps**:
1. [ ] Say: "Cancel my appointment"
2. [ ] Verify: Agent finds appointment
3. [ ] Say: "Actually, no, I want to reschedule instead"
4. [ ] Verify: Agent switches to rescheduling flow
5. [ ] Complete rescheduling
6. [ ] Verify: Appointment rescheduled, not cancelled

---

### 10. State Cleanup Verification

**Test Steps**:
1. [ ] Complete a cancellation
2. [ ] Verify: `found_appointment_id` is cleared from state
3. [ ] Verify: `found_appointment_details` is cleared from state
4. [ ] Say: "Book a new appointment"
5. [ ] Verify: Booking flow works normally (no interference)

---

## Performance Tests

### Load Test
- [ ] 10 concurrent cancellation requests
- [ ] All complete within expected time
- [ ] No database deadlocks
- [ ] Logging shows no errors

### Database Test
- [ ] Appointment lookup completes in < 200ms
- [ ] Cancellation update completes in < 100ms
- [ ] Rescheduling update completes in < 100ms
- [ ] No orphaned records in database

---

## User Experience Tests

### Natural Language Understanding
Test various phrasings:
- [ ] "I gotta cancel"
- [ ] "Need to move my appointment"
- [ ] "Change my appointment time"
- [ ] "Cancel the appointment"
- [ ] "Can I reschedule?"
- [ ] "Different day"

### Time Parsing
Test various time formats:
- [ ] "tomorrow at 3"
- [ ] "next Monday morning"
- [ ] "February 10th at 10am"
- [ ] "two weeks from now"
- [ ] "3:30 PM next Tuesday"

### Confirmation Variations
Test various confirmation phrases:
- [ ] "yes"
- [ ] "yeah"
- [ ] "correct"
- [ ] "that's right"
- [ ] "yup"
- [ ] "sure"

---

## Logging Verification

Check logs for:
- [ ] `[APPT_LOOKUP]` entries show phone number search
- [ ] `[APPT_LOOKUP] ✅ Found appointment` on success
- [ ] `[APPT_LOOKUP] ❌ No appointment found` on miss
- [ ] `[CANCEL]` entries show cancellation process
- [ ] `[RESCHEDULE]` entries show rescheduling process
- [ ] No stack traces or unhandled exceptions

---

## Database Verification

After each test, verify:
- [ ] Appointment record exists
- [ ] Status is correct ('cancelled' for cancellation)
- [ ] Times are correct (for rescheduling)
- [ ] No duplicate records created
- [ ] All timestamps are accurate

---

## Regression Tests

Ensure existing functionality still works:
- [ ] New appointment booking
- [ ] Phone number confirmation
- [ ] Email confirmation
- [ ] Available slots lookup
- [ ] End conversation tool

---

## Production Readiness Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Error messages are user-friendly
- [ ] Logging is comprehensive
- [ ] No hardcoded values
- [ ] Database queries are optimized
- [ ] Timezone handling is correct
- [ ] Phone number normalization works
- [ ] State management is clean
- [ ] Documentation is complete

---

## Known Limitations (Document for future)

- [ ] Currently handles single appointment per phone only
- [ ] Past appointments not shown (only future)
- [ ] No Google Calendar sync yet (event still exists after cancellation)
- [ ] No SMS/Email confirmations sent
- [ ] No cancellation reason tracking

---

## Bug Tracking Template

If issues found during testing:

**Bug**: [Brief description]
**Severity**: [Critical/High/Medium/Low]
**Steps to Reproduce**:
1. 
2. 
3. 

**Expected Behavior**:

**Actual Behavior**:

**Logs**:
```
[Paste relevant logs]
```

**Fix Applied**:

---

## Sign-Off

Once all checkboxes are complete:

- [ ] Feature tested and verified by: ________________
- [ ] Date: ________________
- [ ] Ready for production deployment: YES / NO
- [ ] Issues found: _____ (count)
- [ ] Issues resolved: _____ (count)
- [ ] Outstanding issues documented in: ________________

---

## Quick Test Commands

```bash
# Run all cancellation/rescheduling tests
pytest test_cancellation_rescheduling.py -v

# Run with coverage
pytest test_cancellation_rescheduling.py --cov=services.appointment_management_service --cov=tools.assistant_tools

# Run specific test class
pytest test_cancellation_rescheduling.py::TestCancellation -v

# Run with detailed logging
pytest test_cancellation_rescheduling.py -v -s --log-cli-level=INFO
```

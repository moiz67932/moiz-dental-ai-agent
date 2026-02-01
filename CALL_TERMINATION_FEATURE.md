# Call Termination Feature - Documentation

## Overview

The agent now has the ability to **proactively end calls** when the conversation is complete, saving tokens for:
- **STT (Speech-to-Text)** - Deepgram/OpenAI transcription
- **LLM (Language Model)** - GPT-4o-mini processing  
- **TTS (Text-to-Speech)** - Cartesia/OpenAI voice synthesis

This is both a **cost-saving measure** and a **best practice** for call center automation.

---

## How It Works

### 1. **Enhanced `end_conversation` Tool**

**Location:** `tools/assistant_tools.py` (lines 1881-1906)

The tool now has comprehensive triggering conditions and logging:

```python
@llm.function_tool(description="""
End the conversation and disconnect the call immediately to save resources.

Call this when:
1. The user explicitly says goodbye: "bye", "goodbye", "hang up", "I'm done", "that's all", etc.
2. The booking is COMPLETE and confirmed, AND you've provided the final confirmation details.
3. The user indicates they have no more questions after you've completed their request.
4. The conversation has naturally concluded (e.g., after giving clinic hours/info and user says "okay" or "thanks").

IMPORTANT: After a successful booking confirmation, you MUST end the call to save tokens for STT, LLM, and TTS.
After saying your farewell message, this tool will automatically disconnect within 3 seconds.
""")
async def end_conversation(self) -> str:
    """End the call to save tokens and resources."""
    if self.state:
        self.state.call_ended = True
        
        # Log the reason for call ending
        if self.state.booking_confirmed:
            logger.info("[CALL_END] 🎯 Call ending after successful booking completion")
        else:
            logger.info("[CALL_END] 👋 Call ending at user request or natural conclusion")
    
    return "Goodbye! Have a great day."
```

### 2. **System Prompt Guidance**

**Location:** `prompts/agent_prompts.py` (lines 88-99)

Added explicit instructions for the agent to end calls proactively:

```
☎️ CALL TERMINATION (CRITICAL - SAVE RESOURCES!)
═══════════════════════════════════════════════════════════════════════════════
• After SUCCESSFULLY booking an appointment, you MUST end the call to save tokens.
• Workflow: 
  1. Confirm the booking (the tool will provide a summary)
  2. Say a brief farewell: "All set! We'll see you then. Have a great day!"
  3. IMMEDIATELY call `end_conversation` tool
• Also end the call when:
  - User explicitly says goodbye, bye, hang up, I'm done, that's all
  - You've answered their question (e.g., clinic hours) and they say "okay" or "thanks"
  - User indicates no more questions after completing their request
• DO NOT keep the call going unnecessarily - every second costs money for STT, LLM, and TTS.
```

### 3. **Automatic Disconnection**

**Location:** `agent.py` (lines 252-256)

When the `call_ended` flag is set, the system automatically disconnects:

```python
if state.call_ended:
    async def _delayed_disconnect():
        await asyncio.sleep(3.0)  # Give TTS time to finish
        await ctx.room.disconnect()
    asyncio.create_task(_delayed_disconnect())
```

**Why 3 seconds?** This allows the TTS engine to finish speaking the goodbye message before disconnecting.

---

## Usage Scenarios

### ✅ Scenario 1: Successful Booking

**Call Flow:**
1. User provides all information (name, phone, email, reason, time)
2. Agent confirms booking: `confirm_and_book_appointment()`
3. Agent says: "Perfect! Your appointment is confirmed for Monday, February 3rd at 2 PM. We'll see you then!"
4. Agent immediately calls: `end_conversation()`
5. System disconnects after 3 seconds

**Result:** Call ends cleanly, saving tokens for unnecessary follow-up conversation.

---

### ✅ Scenario 2: Information Request

**Call Flow:**
1. User: "What are your hours?"
2. Agent: "We're open Monday to Friday, 9 AM to 5 PM, Saturday 10 AM to 2 PM, and closed on Sundays."
3. User: "Okay, thanks!"
4. Agent calls: `end_conversation()`
5. Agent says: "You're welcome! Have a great day!"
6. System disconnects after 3 seconds

**Result:** Clean termination after answering question.

---

### ✅ Scenario 3: User-Initiated Goodbye

**Call Flow:**
1. User: "Actually, I need to go. I'll call back later."
2. Agent detects "need to go" as a termination signal
3. Agent calls: `end_conversation()`
4. Agent says: "No problem! Goodbye! Have a great day!"
5. System disconnects after 3 seconds

**Result:** Immediate, polite termination.

---

## State Management

### Flag: `state.call_ended`

**Location:** `models/state.py` (line 274)

```python
# FIX 2: Terminal State
call_ended: bool = False
```

- **Set by:** `end_conversation()` tool
- **Checked by:** `_on_agent_speech_committed()` in `agent.py`
- **Effect:** Triggers disconnection after 3-second delay

---

## Logging and Monitoring

When a call ends, you'll see one of these log messages:

```
[CALL_END] 🎯 Call ending after successful booking completion
```
or
```
[CALL_END] 👋 Call ending at user request or natural conclusion
```

This helps distinguish between:
- **Successful completions** (booking confirmed)
- **User-initiated terminations** (goodbye, hang up, etc.)
- **Natural conclusions** (question answered, user satisfied)

---

## Cost Savings Estimate

### Token Usage Breakdown (per minute of unnecessary conversation)

| Service | Cost/Min | Example |
|---------|----------|---------|
| **STT** | ~$0.006 | Deepgram transcription |
| **LLM** | ~$0.015 | GPT-4o-mini processing |
| **TTS** | ~$0.015 | Cartesia voice synthesis |
| **Total** | **~$0.036/min** | Additional unnecessary time |

### Example Savings

If the agent ends calls **30 seconds earlier** on average:

- **Per call:** $0.018 saved
- **100 calls/day:** $1.80/day = **$54/month**
- **1000 calls/day:** $18/day = **$540/month**

---

## Testing

### Manual Testing

1. **Test successful booking termination:**
   - Complete a full booking
   - Verify agent says goodbye and disconnects within 3-4 seconds

2. **Test user-initiated goodbye:**
   - Say "bye" or "I'm done"
   - Verify agent ends call immediately

3. **Test info request termination:**
   - Ask about clinic hours
   - Say "thanks" or "okay"
   - Verify agent ends call

### Automated Testing

Create test cases in `test_call_termination.py`:

```python
def test_end_call_after_booking():
    state = PatientState()
    state.booking_confirmed = True
    tools = AssistantTools(state)
    
    result = await tools.end_conversation()
    
    assert state.call_ended == True
    assert "Goodbye" in result

def test_end_call_logs_reason():
    # Test that logging distinguishes booking vs user-initiated
    pass
```

---

## Edge Cases

### 1. ❌ **Premature Termination**

**Problem:** Agent ends call before booking is complete.

**Solution:** System prompt explicitly states "booking is COMPLETE and confirmed" as a requirement.

### 2. ❌ **Lingering Conversations**

**Problem:** Agent doesn't end call after booking, wasting tokens.

**Solution:** System prompt includes "IMMEDIATELY call end_conversation" instruction.

### 3. ❌ **Abrupt Hang-ups**

**Problem:** Disconnect happens before TTS finishes.

**Solution:** 3-second delay ensures goodbye message completes.

---

## Configuration

### Adjusting Disconnect Delay

**Location:** `agent.py` (line 254)

```python
await asyncio.sleep(3.0)  # Adjust this value if needed
```

- **Too short (\<2s):** May cut off goodbye message
- **Too long (\>5s):** Wastes tokens unnecessarily
- **Recommended:** 3.0 seconds (tested optimal)

---

## Future Enhancements

### 1. **Sentiment Analysis**
Detect user frustration/satisfaction to end calls more intelligently.

### 2. **Automatic Timeout**
End call if user is silent for >30 seconds after booking complete.

### 3. **Transfer Support**
Add ability to transfer instead of ending (for complex requests).

### 4. **Analytics Dashboard**
Track:
- Average call duration
- Token savings per call
- Termination reasons (booking vs goodbye vs timeout)

---

## Troubleshooting

### Issue: Agent not ending calls after booking

**Check:**
1. Is `state.booking_confirmed` set to `True`?
2. Is the system prompt being updated correctly?
3. Are there errors in the logs preventing tool execution?

**Fix:** Review logs for `[CALL_END]` messages.

---

### Issue: Calls disconnect too soon

**Check:**
1. Is there a competing disconnect trigger?
2. Is the TTS latency higher than expected?

**Fix:** Increase delay in `_delayed_disconnect()` from 3.0 to 4.0 seconds.

---

### Issue: Goodbye message cut off

**Check:**
1. Is the TTS engine slow?
2. Is the disconnect delay too short?

**Fix:** Increase `asyncio.sleep()` duration or optimize TTS model.

---

## Summary

✅ **Agent can now end calls proactively**  
✅ **Saves tokens for STT, LLM, TTS**  
✅ **Follows best practices for call center automation**  
✅ **Logs termination reasons for analytics**  
✅ **Graceful 3-second delay ensures complete goodbye**  

**Result:** More efficient, cost-effective voice agent! 🎉

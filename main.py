"""
Main entry point for the Dental AI Agent.

Initializes the LiveKit worker and starts the voice agent.
"""

from __future__ import annotations

import os
import sys

# LiveKit imports
from livekit import agents
from livekit.agents import cli, WorkerOptions
from livekit.plugins import silero

# Local imports
from config import (
    logger,
    LIVEKIT_AGENT_NAME,
    ENVIRONMENT,
    GOOGLE_OAUTH_TOKEN_PATH,
)
from agent import entrypoint
from models.state import PatientState
from utils.phone_utils import _normalize_phone_preserve_plus, speakable_phone


# ============================================================================
# Extracted: prewarm function
# ============================================================================

# =============================================================================

def prewarm(proc: agents.JobProcess):
    logger.info(f"[TELEPHONY] Worker identity set to: {LIVEKIT_AGENT_NAME}")
    
    try:
        silero.VAD.load()
        logger.info("[PREWARM] ✓ VAD loaded")
    except Exception as e:
        logger.error(f"[PREWARM] Error: {e}")
    
    # Verify calendar
    print("\n" + "="*50)
    print("[CONFIG] Verifying calendar...")
    if ENVIRONMENT == "production":
        print("[CONFIG] ✓ Production: using Supabase-backed OAuth token (skipping local file check)")
    else:
        if GOOGLE_OAUTH_TOKEN_PATH and os.path.exists(GOOGLE_OAUTH_TOKEN_PATH):
            print(f"[CONFIG] ✓ OAuth token: {GOOGLE_OAUTH_TOKEN_PATH}")
        else:
            print(f"[CONFIG] ❌ OAuth token missing")
    print("="*50 + "\n")



# ============================================================================
# Extracted: _run_debug_tests
# ============================================================================

def _run_debug_tests():
    """
    Run inline tests for phone normalization and slot suggestion logic.
    Call this via: python agent_v2.py --test
    """
    print("\n" + "="*60)
    print(" IDEMPOTENCY & LOCKING TESTS")
    print("="*60)
    
    state = PatientState()
    state.start_new_turn("Hello, my name is John")
    
    # Test 1: Tool Locking
    locked1 = state.check_tool_lock("test_tool", {"arg": 1})
    locked2 = state.check_tool_lock("test_tool", {"arg": 1})
    locked3 = state.check_tool_lock("test_tool", {"arg": 2})
    
    print(f"Lock 1 (First call): {locked1} (Expected: False)")
    print(f"Lock 2 (Same turn, same args): {locked2} (Expected: True)")
    print(f"Lock 3 (Same turn, diff args): {locked3} (Expected: False)")
    
    assert locked1 is False
    assert locked2 is True
    assert locked3 is False
    
    # Test 2: Field Correction
    state.full_name = "John"
    state.last_user_text = "No, it's Jon"
    should_update = state.should_update_field("name", "John", "Jon")
    print(f"Field Update (Correction): {should_update} (Expected: True)")
    assert should_update is True
    
    state.last_user_text = "My name is Jon"
    should_update_dup = state.should_update_field("name", "John", "John")
    print(f"Field Update (Duplicate): {should_update_dup} (Expected: False)")
    assert should_update_dup is False
    
    # Test 3: Field Change Without Correction
    state.last_user_text = "I like blue"
    should_update_hallucination = state.should_update_field("name", "John", "Jim")
    print(f"Field Update (No Correction): {should_update_hallucination} (Expected: False)")
    assert should_update_hallucination is False
    
    print("\n" + "="*60)
    print(" PHONE NORMALIZATION TESTS")
    print("="*60)
    
    test_cases = [
        # (input, region, expected_e164_prefix, description)
        ("+923351897839", "PK", "+923351897839", "Already E.164 Pakistani"),
        ("0335-1897839", "PK", "+92335", "Pakistani local format 0335..."),
        ("03351897839", "PK", "+92335", "Pakistani local without dashes"),
        ("+13105551234", "US", "+13105551234", "Already E.164 US"),
        ("310-555-1234", "US", "+1310", "US local format"),
        ("+442071234567", "GB", "+44207", "UK E.164"),
    ]
    
    for raw, region, expected_prefix, desc in test_cases:
        result = _normalize_phone_preserve_plus(raw, region)
        e164, last4 = result
        
        # Verify result is tuple with string or None
        assert isinstance(result, tuple), f"FAIL: {desc} - result not tuple"
        assert e164 is None or isinstance(e164, str), f"FAIL: {desc} - e164 not string: {type(e164)}"
        assert isinstance(last4, str), f"FAIL: {desc} - last4 not string: {type(last4)}"
        
        if e164 and not e164.startswith(expected_prefix):
            print(f"WARN: {desc}")
            print(f"       Input: {raw} (region={region})")
            print(f"       Got: {e164}, expected prefix: {expected_prefix}")
        else:
            print(f"✓ {desc}: {raw} -> {e164} (last4={last4})")
    
    print("\n" + "="*60)
    print(" SPEAKABLE PHONE TESTS")
    print("="*60)
    
    speakable_tests = [
        ("+923351897839", "+92 335 189 7839"),
        ("+13105551234", "+1 310 555 1234"),
        ("+442071234567", "+44 2071 2345 67"),
        (None, "unknown"),
    ]
    
    for e164, expected in speakable_tests:
        result = speakable_phone(e164)
        status = "✓" if result == expected else "✗"
        print(f"{status} speakable_phone({e164!r}) = {result!r} (expected: {expected!r})")
    
    print("\n" + "="*60)
    print(" ALL TESTS COMPLETE")
    print("="*60 + "\n")


# ============================================================================
# Extracted: main CLI block
# ============================================================================

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_debug_tests()
    else:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
                agent_name=LIVEKIT_AGENT_NAME,  # Must match SIP trunk dispatch rules
                load_threshold=1.0,  # Prioritize this agent for incoming telephony calls
            )
        )

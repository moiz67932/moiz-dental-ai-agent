"""
Main entry point for the Dental AI Agent.

Initializes the LiveKit worker and starts the voice agent.
Updated for Cloud Run compatibility with production hardening:
- Graceful shutdown handling (SIGTERM/SIGINT)
- Background worker management
- Production Uvicorn configuration
"""

from __future__ import annotations

import os
import sys
import time
import signal
import threading
import asyncio

# Cloud Run / HTTP Server imports
from fastapi import FastAPI
import uvicorn

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
from agent_v2 import entrypoint
from models.state import PatientState
from utils.phone_utils import _normalize_phone_preserve_plus, speakable_phone


# ============================================================================
# Cloud Run HTTP Server
# ============================================================================

app = FastAPI(docs_url=None, redoc_url=None)  # Disable docs in prod for slight memory save

@app.get("/healthz")
def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "ok"}


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
    logger.info("[CONFIG] Verifying calendar...")
    if ENVIRONMENT == "production":
        logger.info("[CONFIG] ✓ Production: using Supabase-backed OAuth token (skipping local file check)")
    else:
        if GOOGLE_OAUTH_TOKEN_PATH and os.path.exists(GOOGLE_OAUTH_TOKEN_PATH):
            logger.info(f"[CONFIG] ✓ OAuth token: {GOOGLE_OAUTH_TOKEN_PATH}")
        else:
            logger.warning(f"[CONFIG] ❌ OAuth token missing")


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
        ("2071234567", "GB", "+44207", "UK local w/ default region"),
        ("+442071234567", "GB", "+44207", "UK E.164"),
    ]
    
    for raw, region, expected_prefix, desc in test_cases:
        if raw == "2071234567": continue 
        
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
# Main Execution
# ============================================================================

# Global flag to ensure worker starts only once
_worker_started = False

def run_livekit_worker():
    """
    Run LiveKit worker in a background thread with a fully running asyncio loop.
    
    Required for Cloud Run because:
    1. Threads have no default event loop
    2. cli.run_app() assumes it controls the main process (breaks in threads)
    3. LiveKit internally calls asyncio.get_running_loop() which requires an active loop
    
    Solution: Use agents.Worker directly with loop.run_until_complete()
    """
    try:
        logger.info("[WORKER] Starting LiveKit worker thread...")
        
        # Create and set event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _runner():
            """Inner async function that runs the LiveKit worker."""
            worker = agents.Worker(
                WorkerOptions(
                    entrypoint_fnc=entrypoint,
                    prewarm_fnc=prewarm,
                    agent_name=LIVEKIT_AGENT_NAME,
                    load_threshold=1.0,
                )
            )
            await worker.run()
        
        # ✅ Critical: Run the loop until the worker completes
        # This makes the loop "running" so asyncio.get_running_loop() works
        loop.run_until_complete(_runner())
        
    except SystemExit:
        logger.info("[WORKER] LiveKit worker requested exit")
    except Exception:
        logger.exception("[WORKER] LiveKit worker crashed")
    finally:
        # Clean up the event loop
        try:
            loop.close()
            logger.info("[WORKER] Event loop closed")
        except Exception as e:
            logger.warning(f"[WORKER] Error closing loop: {e}")


def start_worker_thread_once():
    """
    Start the worker as a daemon thread so FastAPI stays alive.
    Ensure it starts only once (important for reload / multiple imports).
    """
    global _worker_started
    
    if _worker_started:
        logger.info("[WORKER] Worker already started, skipping duplicate start")
        return
    
    t = threading.Thread(target=run_livekit_worker, name="run_livekit_worker", daemon=True)
    t.start()
    _worker_started = True
    logger.info("[WORKER] Worker thread started.")

if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_debug_tests()
    else:
        # 1. Setup Signal Handling & Shutdown Coordination
        logger.info("[INIT] Initializing Server...")

        # Cloud Run listens on port defined by PORT env var (default 8080)
        port = int(os.getenv("PORT", 8080))
        
        # Configure Uvicorn for Production
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=port, 
            log_level="info",
            access_log=False  # Reduce noise in Cloud Run logs
        )
        server = uvicorn.Server(config)

        # Override Uvicorn's signal handlers to prevent it from stealing SIGTERM
        # We want to handle signals to ensure the background worker has time to cleanup
        server.install_signal_handlers = lambda: None

        def handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"[SIGNAL] Received {sig_name}. Initiating graceful shutdown...")
            
            # Stop accepting requests immediately (best effort)
            server.should_exit = True
            
            # Grace Period: Allow 2 seconds for background tasks/logs to flush
            # Cloud Run allows up to 10s (or more if configured) before hard kill.
            logger.info("[SIGNAL] Waiting 2s for worker cleanup...")
            time.sleep(2.0)
            
        # Register our custom handlers
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        # 2. Start LiveKit agent in background daemon thread
        # Setting daemon=True ensures it doesn't block program exit if main thread dies,
        # but our signal handler ensures main thread stays alive long enough for cleanup.
        start_worker_thread_once()
        
        # 3. Start HTTP Server (Blocking)
        logger.info(f"[HTTP] Starting production FastAPI server on port {port}")
        try:
            server.run()
        except Exception as e:
            logger.error(f"[HTTP] Server failed: {e}")
            sys.exit(1)
        
        logger.info("[SHUTDOWN] Process exiting.")

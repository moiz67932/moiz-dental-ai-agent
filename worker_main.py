"""
LiveKit Worker Entry Point for Cloud Run Jobs.

WHY THIS ARCHITECTURE:
======================
Cloud Run Services are request-driven and will SIGTERM containers
with no active HTTP traffic. LiveKit agents maintain WebSocket
connections to LiveKit Cloud, not HTTP requests, so Services will
always kill the container after idle timeout.

Cloud Run Jobs are task-driven and run until the task completes.
This is the correct primitive for a long-running voice agent.

WHY NO CLI HELPERS:
===================
cli.run_app() is designed for standalone development:
- Manages its own signal handlers (conflicts with Job lifecycle)
- Creates internal event loops (unpredictable in containers)
- Expects to own the process (breaks in multi-component setups)

We use agents.AgentServer directly with asyncio.run() for:
- Single, clean event loop
- Predictable shutdown behavior
- Full control over lifecycle

USAGE:
======
Cloud Run Job: CMD ["python", "worker_main.py"]
Local dev:     python worker_main.py

API CHANGES (livekit-agents >= 1.3.x):
======================================
- agents.Worker → agents.AgentServer
- Uses @server.rtc_session(agent_name=...) decorator pattern
- agent_name is set in the rtc_session decorator, NOT the constructor
- setup_fnc property replaces prewarm_fnc constructor param
"""

from __future__ import annotations

import os
import sys
import signal
import asyncio
import logging

# Configure logging for LiveKit SDK
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)

# FORCE PORT to avoid conflict if shell env has PORT=8080
os.environ["PORT"] = "8080"

from livekit import agents
from livekit.agents import AgentServer, JobContext, JobProcess
from livekit.plugins import silero

# Local imports
from config import (
    logger,
    LIVEKIT_AGENT_NAME,
    ENVIRONMENT,
    GOOGLE_OAUTH_TOKEN_PATH,
)
from agent import entrypoint

# =============================================================================
# SIGNAL HANDLING — Graceful shutdown for Cloud Run
# =============================================================================

_shutdown_event = asyncio.Event()


def _handle_sigterm(signum, frame):
    """
    Handle SIGTERM from Cloud Run.
    
    Cloud Run sends SIGTERM when:
    - Job timeout is reached
    - Job is cancelled
    - Container is being replaced
    
    We set an event to allow graceful shutdown rather than hard exit.
    """
    sig_name = signal.Signals(signum).name
    logger.info(f"[WORKER] Received {sig_name}, initiating graceful shutdown...")
    _shutdown_event.set()


# =============================================================================
# PREWARM — Called once when worker starts
# =============================================================================

def prewarm(proc: JobProcess):
    """
    Prewarm function called once when the worker process starts.
    
    Use this to:
    - Load models (VAD, etc.)
    - Verify external service connections
    - Log startup diagnostics
    """
    logger.info(f"[PREWARM] Worker identity: {LIVEKIT_AGENT_NAME}")
    
    # Load VAD model
    try:
        silero.VAD.load()
        logger.info("[PREWARM] ✓ Silero VAD loaded")
    except Exception as e:
        logger.error(f"[PREWARM] ✗ VAD load failed: {e}")
    
    # Verify calendar credentials
    if ENVIRONMENT == "production":
        logger.info("[PREWARM] ✓ Production mode: using Supabase OAuth tokens")
    else:
        if GOOGLE_OAUTH_TOKEN_PATH and os.path.exists(GOOGLE_OAUTH_TOKEN_PATH):
            logger.info(f"[PREWARM] ✓ OAuth token found: {GOOGLE_OAUTH_TOKEN_PATH}")
        else:
            logger.warning("[PREWARM] ⚠ OAuth token not found (calendar features may fail)")


# =============================================================================
# CREATE AGENT SERVER
# =============================================================================

# Create the AgentServer instance (replaces the old Worker class)
# NOTE: agent_name is NOT a constructor parameter in 1.3.x
# It's set via @server.rtc_session(agent_name=...) decorator
server = AgentServer(
    load_threshold=1.0,
    setup_fnc=prewarm,
    port=8080,
    host="0.0.0.0",
)


# =============================================================================
# RTC SESSION — Register entrypoint using decorator pattern
# =============================================================================

@server.rtc_session(agent_name=LIVEKIT_AGENT_NAME)
async def session_entrypoint(ctx: JobContext):
    """
    RTC session entrypoint - delegates to the actual agent entrypoint.
    
    This decorator pattern is required for livekit-agents >= 1.3.x.
    The @server.rtc_session() decorator registers this function as the
    handler for incoming LiveKit room connections.
    
    The agent_name parameter here is what enables explicit dispatch.
    """
    logger.info("[RTC] session_entrypoint invoked")
    await entrypoint(ctx)


# =============================================================================
# MAIN — Clean asyncio entry point
# =============================================================================

async def main():
    """
    Main async entry point for the LiveKit worker.
    
    WHY asyncio.run() IS CORRECT:
    - Creates a single event loop for the process
    - Runs until the coroutine completes or is cancelled
    - Cleans up properly on exit
    
    WHY NOT cli.run_app():
    - Designed for development, not containerized production
    - Manages its own signals (conflicts with Cloud Run)
    - Not needed when we have full control over the process
    """
    logger.info("[WORKER] Starting LiveKit worker...")
    logger.info(f"[WORKER] Agent name: {LIVEKIT_AGENT_NAME}")
    logger.info(f"[WORKER] Environment: {ENVIRONMENT}")
    
    # Run the server
    # This blocks until the server is shut down (SIGTERM or error)
    try:
        await server.run()
    except asyncio.CancelledError:
        logger.info("[WORKER] Worker cancelled, shutting down...")
    except Exception:
        logger.exception("[WORKER] Worker crashed with exception")
        raise
    finally:
        logger.info("[WORKER] Worker stopped")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Register signal handlers before starting async code
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)
    
    logger.info("=" * 60)
    logger.info(" LIVEKIT WORKER — CLOUD RUN JOB MODE")
    logger.info("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[WORKER] Interrupted by user")
    except SystemExit as e:
        logger.info(f"[WORKER] System exit: {e.code}")
    except Exception:
        logger.exception("[WORKER] Fatal error")
        sys.exit(1)
    
    logger.info("[WORKER] Process exiting cleanly")

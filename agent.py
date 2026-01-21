"""
Main agent entrypoint and event handlers for the Dental AI Voice Agent.

This module contains the core voice agent logic including:
- LiveKit VoicePipelineAgent setup
- SIP telephony detection and phone capture
- Dynamic slot-aware prompting
- Filler speech management
- Barge-in support
- Yes/No confirmation routing
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

# LiveKit imports
from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe
from livekit.agents.voice import Agent as VoicePipelineAgent
from livekit.agents import llm
from livekit.agents.metrics import UsageCollector
from livekit.plugins import openai as openai_plugin
from livekit.plugins import deepgram as deepgram_plugin
from livekit.plugins import cartesia as cartesia_plugin
from livekit.plugins import silero
from livekit.rtc import ParticipantKind

# Metrics
from livekit.agents import metrics as lk_metrics
from livekit.agents.metrics import MetricsCollectedEvent

# Local config imports
from config import (
    logger,
    DEFAULT_TZ,
    DEFAULT_PHONE_REGION,
    DEMO_CLINIC_ID,
    FILLER_ENABLED,
    FILLER_MAX_DURATION_MS,
    FILLER_PHRASES,
    STT_AGGRESSIVE_ENDPOINTING,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    LATENCY_DEBUG,
    map_call_outcome,
    supabase,
)

# Models
from models.state import PatientState

# Services
from services.database_service import fetch_clinic_context_optimized
from services.scheduling_service import load_schedule_from_settings

# Tools
from tools.assistant_tools import AssistantTools

# Prompts
from prompts.agent_prompts import A_TIER_PROMPT

# Utilities
from utils.phone_utils import (
    _normalize_phone_preserve_plus,
    _normalize_sip_user_to_e164,
    _ensure_phone_is_string,
    speakable_phone,
)
from utils.latency_metrics import TurnMetrics

# Yes/No patterns for deterministic confirmation routing
YES_PAT = re.compile(r"\b(yes|yeah|yep|yup|correct|right|sure|okay|ok|affirmative|absolutely|definitely|that'?s right|that'?s correct)\b", re.IGNORECASE)
NO_PAT = re.compile(r"\b(no|nope|nah|wrong|incorrect|negative|not right|that'?s wrong|that'?s not)\b", re.IGNORECASE)

# Global state references for cross-module access
_GLOBAL_STATE: Optional[PatientState] = None
_GLOBAL_CLINIC_TZ: str = DEFAULT_TZ
_GLOBAL_CLINIC_INFO: Optional[dict] = None
_GLOBAL_AGENT_SETTINGS: Optional[dict] = None
_GLOBAL_SCHEDULE: Optional[dict] = None
_REFRESH_AGENT_MEMORY: Optional[callable] = None

# Turn metrics for latency tracking
_turn_metrics = TurnMetrics()


# ============================================================================
# Extracted: entrypoint function and all event handlers
# ============================================================================

async def entrypoint(ctx: JobContext):
    """
    A-TIER ENTRYPOINT with <1s response latency using VoicePipelineAgent.
    
    Optimizations:
    1. Single Supabase query (3.2s → 100ms)
    2. VoicePipelineAgent with min_endpointing_delay=0.6s
    3. gpt-4o-mini for speed + quality
    4. LLM Function Calling for real-time parallel extraction
    5. Non-blocking booking
    6. Global state for tool access
    7. Dynamic Slot-Aware Prompting - system prompt refreshes every turn!
    8. SIP Telephony Support - auto-detect inbound calls & pre-fill caller phone
    """
    global _GLOBAL_STATE, _GLOBAL_CLINIC_TZ, _GLOBAL_CLINIC_INFO, _REFRESH_AGENT_MEMORY
    
    state = PatientState()
    _GLOBAL_STATE = state  # Set global reference for tools

    active_filler_handle = {"handle": None, "is_filler": False, "start_time": None}
    active_agent_handle = {"handle": None}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔧 DEFAULTS INITIALIZATION — Must be set BEFORE SIP block uses clinic_region
    # ═══════════════════════════════════════════════════════════════════════════
    clinic_info = None
    agent_info = None
    settings = None
    agent_name = "Office Assistant"
    clinic_name = "our clinic"
    clinic_tz = DEFAULT_TZ
    clinic_region = DEFAULT_PHONE_REGION
    agent_lang = "en-US"
    
    call_started = time.time()
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    participant = await ctx.wait_for_participant()
    logger.info(f"[LIFECYCLE] Participant: {participant.identity}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📞 SIP TELEPHONY DETECTION — Prioritize real SIP metadata over job metadata
    # ═══════════════════════════════════════════════════════════════════════════
    called_num = None
    caller_phone = None
    is_sip_call = False
    used_fallback_called_num = False
    
    # PRIORITY 1: Real SIP participant metadata (production telephony)
    if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
        is_sip_call = True
        # Extract SIP attributes from participant
        sip_attrs = participant.attributes or {}
        caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
        # Fix: Twilio dialed number is typically in one of these keys
        called_num = sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
        called_num = _normalize_sip_user_to_e164(called_num)
        
        logger.info(f"📞 [SIP] Inbound call detected!")
        logger.info(f"📞 [SIP] Caller (from): {caller_phone}")
        logger.info(f"📞 [SIP] Called (to): {called_num}")
        
        # Pre-fill caller's phone from SIP - SILENTLY store in detected_phone
        # Agent will confirm later after name + time are captured
        if caller_phone:
            clean_phone, last4 = _normalize_phone_preserve_plus(caller_phone, clinic_region)
            if clean_phone:
                state.detected_phone = str(clean_phone)  # Silent detection - never spoken
                state.phone_last4 = str(last4) if last4 else ""
                # Safety guard: ensure no tuple was stored
                _ensure_phone_is_string(state)
                state.phone_confirmed = False  # NEVER auto-confirm - always ask user
                state.phone_source = "sip"  # Track source for confirmation logic
                # DO NOT set pending_confirm here - contact phase hasn't started
                logger.info(f"📞 [SIP] ✓ Caller phone detected silently: ***{state.phone_last4}")
    
    # PRIORITY 2: Room name regex — flexible US phone number extraction
    # Matches +1XXXXXXXXXX anywhere in room name (e.g., call_+13103410536_abc123)
    if not called_num:
        room_name = getattr(ctx.room, "name", "") or ""
        # Try US format first (+1 followed by 10 digits)
        room_match = re.search(r"(\+1\d{10})", room_name)
        if not room_match:
            # Fallback: any number in call_{number}_ format
            room_match = re.search(r"call_(\+?\d+)_", room_name)
        if room_match:
            called_num = _normalize_sip_user_to_e164(room_match.group(1))
            logger.info(f"[ROOM] ✓ Extracted phone from room name: {called_num}")

    # PRIORITY 3: Job metadata (LiveKit Playground / testing)
    if not called_num:
        metadata = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
        sip_info = metadata.get("sip", {}) if isinstance(metadata, dict) else {}
        called_num = _normalize_sip_user_to_e164(sip_info.get("toUser"))
        # Also check for caller in job metadata
        if not caller_phone:
            caller_phone = sip_info.get("fromUser") or sip_info.get("phoneNumber")
        
        if called_num:
            logger.info(f"[METADATA] Using job metadata: toUser={called_num}")
    
    # PRIORITY 4: Fallback to environment default (for local testing only)
    # NOTE: Comment out in production to ensure proper SIP routing
    if not called_num:
        called_num = os.getenv("DEFAULT_TEST_NUMBER", "+13103410536")
        logger.warning(f"[FALLBACK] Using default test number: {called_num}")
        used_fallback_called_num = True

    # ⚡ FAST-PATH CONTEXT: start the optimized fetch immediately once called_num is known.
    # Do not block audio startup on this; we only wait a tiny budget to personalize if it returns fast.
    context_task: Optional[asyncio.Task] = None
    if called_num:
        context_task = asyncio.create_task(fetch_clinic_context_optimized(called_num))

    # ═══════════════════════════════════════════════════════════════════════════
    # 🏥 IDENTITY-FIRST: Wait up to 5s for DB context (better silence than wrong name)
    # ═══════════════════════════════════════════════════════════════════════════
    if context_task:
        try:
            clinic_info, agent_info, settings, agent_name = await asyncio.wait_for(
                asyncio.shield(context_task), timeout=5.0
            )
            logger.info(f"[DB] ✓ Context loaded in <5s: clinic={clinic_info.get('name') if clinic_info else 'None'}")
        except asyncio.TimeoutError:
            logger.warning("[DB] ⚠️ Context fetch exceeded 5s timeout — using defaults")

    # Safety net: Force-load demo clinic if context still None
    if clinic_info is None:
        logger.warning("[DB] ⚠️ clinic_info is None — force-loading demo fallback")
        clinic_info = {"id": DEMO_CLINIC_ID, "name": "Moiz Dental Clinic Islamabad"}

    # Apply whatever context we have at this point
    _GLOBAL_CLINIC_INFO = clinic_info

    global _GLOBAL_AGENT_SETTINGS
    _GLOBAL_AGENT_SETTINGS = settings

    clinic_name = (clinic_info or {}).get("name") or clinic_name
    clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
    clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
    agent_lang = (agent_info or {}).get("default_language") or agent_lang

    state.tz = clinic_tz
    _GLOBAL_CLINIC_TZ = clinic_tz  # Set global for tool timezone anchoring

    schedule = load_schedule_from_settings(settings or {})

    global _GLOBAL_SCHEDULE
    _GLOBAL_SCHEDULE = schedule
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🧠 DYNAMIC SLOT-AWARE PROMPTING — Refresh system prompt every turn
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_updated_instructions() -> str:
        """
        Generate fresh system prompt with current PatientState snapshot.
        This is the key to Dynamic Slot-Aware Prompting!
        """
        return A_TIER_PROMPT.format(
            agent_name=agent_name,
            clinic_name=clinic_name,
            timezone=clinic_tz,
            state_summary=state.detailed_state_for_prompt(),  # Real-time snapshot!
        )
    
    # Store reference to session for the refresh callback (set after session creation)
    session_ref: Dict[str, Any] = {"session": None, "agent": None}
    
    def refresh_agent_memory():
        """
        Refresh the LLM's system prompt with current state.
        Called after user speech and after tool updates.
        """
        try:
            session = session_ref.get("session")
            agent = session_ref.get("agent")
            
            if not session or not agent:
                logger.debug("[MEMORY] Session/agent not yet initialized, skipping refresh")
                return
            
            new_instructions = get_updated_instructions()
            
            # Update the agent's instructions directly
            if hasattr(agent, '_instructions'):
                agent._instructions = new_instructions
                logger.debug(f"[MEMORY] ✓ Refreshed agent instructions. State: {state.slot_summary()}")
            
            # Also try to update chat context if available
            if hasattr(session, 'chat_ctx') and session.chat_ctx:
                try:
                    messages = getattr(session.chat_ctx, 'messages', None) or getattr(session.chat_ctx, 'items', None)
                    if messages and len(messages) > 0:
                        first_msg = messages[0]
                        if hasattr(first_msg, 'content'):
                            first_msg.content = new_instructions
                        elif hasattr(first_msg, 'text_content'):
                            # Try to update text_content for ChatMessage objects
                            if hasattr(first_msg, '_text_content'):
                                first_msg._text_content = new_instructions
                except Exception as e:
                    logger.debug(f"[MEMORY] chat_ctx update skipped: {e}")
                    
        except Exception as e:
            logger.warning(f"[MEMORY] Refresh failed: {e}")
    
    # Set global refresh callback for tools to use
    _REFRESH_AGENT_MEMORY = refresh_agent_memory
    
    # Build initial prompt (may be placeholder; we'll refresh once DB context arrives)
    initial_system_prompt = get_updated_instructions()

    # ═══════════════════════════════════════════════════════════════════════════
    # 🎙️ GREETING: Use DB greeting_text if context loaded, otherwise fallback
    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: Do NOT confirm phone in greeting - it feels robotic.
    # Phone confirmation should only happen when contact details are explicitly needed.
    # The phone is captured from SIP but will be confirmed later in the flow.
    
    if settings and settings.get("greeting_text"):
        greeting = settings.get("greeting_text")
        logger.info(f"[GREETING] Using DB greeting: {greeting[:50]}...")
    elif clinic_info:
        greeting = f"Hi, thanks for calling {clinic_name}! How can I help you today?"
        logger.info(f"[GREETING] Using clinic-aware greeting for {clinic_name}")
    else:
        greeting = "Hello! Thanks for calling. How can I help you today?"
        logger.info("[GREETING] Using default greeting (DB context not loaded)")
    
    # ⚡ HIGH-PERFORMANCE LLM with function calling
    llm_instance = openai_plugin.LLM(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    # ⚡ SNAPPY STT with aggressive endpointing for faster turn detection
    if os.getenv("DEEPGRAM_API_KEY"):
        # Deepgram with optimized settings for low-latency
        stt_config = {
            "model": "nova-2-general",
            "language": agent_lang,
        }
        # Enable aggressive endpointing if configured AND provider supports it
        # Guard: Only apply if deepgram_plugin.STT accepts these kwargs
        if STT_AGGRESSIVE_ENDPOINTING:
            # Check if STT class accepts endpointing params (capability guard)
            import inspect
            try:
                stt_sig = inspect.signature(deepgram_plugin.STT.__init__)
                stt_params = set(stt_sig.parameters.keys())
                # Only add if supported by this version of the plugin
                if "endpointing" in stt_params or "kwargs" in str(stt_sig):
                    stt_config["endpointing"] = 300  # 300ms silence triggers end
                    stt_config["utterance_end_ms"] = 1000  # Max wait for utterance end
                    if LATENCY_DEBUG:
                        logger.debug("[STT] Deepgram aggressive endpointing enabled: 300ms")
            except Exception:
                pass  # Silently fall back to default if introspection fails
        stt_instance = deepgram_plugin.STT(**stt_config)
    else:
        stt_instance = openai_plugin.STT(model="gpt-4o-transcribe", language="en")
    
    # ⚡ FAST VAD with tuned silence detection
    # WARNING: min_silence < 0.25s may cause premature cutoffs; min_speech should stay at 0.1
    vad_instance = silero.VAD.load(
        min_speech_duration=VAD_MIN_SPEECH_DURATION,  # 0.1s - don't lower this
        min_silence_duration=VAD_MIN_SILENCE_DURATION,  # 0.25s (was 0.3) - faster end detection
    )
    if LATENCY_DEBUG:
        logger.info(f"[VAD] Loaded with min_silence={VAD_MIN_SILENCE_DURATION}s, min_speech={VAD_MIN_SPEECH_DURATION}s")
    
    # TTS
    if os.getenv("CARTESIA_API_KEY"):
        tts_instance = cartesia_plugin.TTS(
            model="sonic-3",
            voice=os.getenv("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        )
    else:
        tts_instance = openai_plugin.TTS(model="tts-1", voice="alloy")

    # Initialize VoicePipelineAgent (as 'session' for decorators)
    # Create chat context with system prompt
    chat_context = llm.ChatContext()
    chat_context.append(text=initial_system_prompt, role=llm.ChatRole.SYSTEM)
    
    # Create function context for Receptionist tools
    fnc_ctx = AssistantTools(state)
    
    session = VoicePipelineAgent(
        vad=vad_instance,
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        chat_ctx=chat_context,
        fnc_ctx=fnc_ctx,
        interruptible=True,
    )

    # Usage metrics
    usage = lk_metrics.UsageCollector()

    # @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        lk_metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    def _interrupt_filler():
        """Interrupt active filler speech."""
        # FIX 5: One Filler Per Turn - Clear state
        if state.filler_active:
             logger.debug("[FILLER] Cleared due to interruption/completion")

        state.filler_active = False
        state.filler_task = None
        state.filler_turn_id = None
        # state.real_response_started = True # Don't set here potentially? Actually yes, interrupt often means real response
        
        h = active_filler_handle.get("handle")
        if not h:
            return
        try:
            # Try various interrupt methods depending on SDK version
            if hasattr(h, 'interrupt'):
                h.interrupt()
            elif hasattr(h, 'cancel'):
                h.cancel()
            elif hasattr(h, 'stop'):
                h.stop()
            logger.debug("[FILLER] ✓ Interrupted filler")
        except Exception as e:
            logger.debug(f"[FILLER] Could not interrupt filler (non-critical): {e}")
        finally:
            active_filler_handle["handle"] = None
            active_filler_handle["is_filler"] = False
            active_filler_handle["start_time"] = None
    
    async def _send_filler_async(filler_text: str):
        """
        Non-blocking filler speech with hard timeout.
        Will be interrupted when real response arrives OR after FILLER_MAX_DURATION_MS.
        """
        try:
            active_filler_handle["is_filler"] = True
            active_filler_handle["start_time"] = time.perf_counter()
            
            # Use session.say with allow_interruptions=True for non-blocking filler
            handle = await session.say(filler_text, allow_interruptions=True)
            active_filler_handle["handle"] = handle
            
            if LATENCY_DEBUG:
                logger.debug(f"[FILLER] Sent: '{filler_text}'")
            
            # Hard timeout: interrupt filler after FILLER_MAX_DURATION_MS even if TTS is slow
            await asyncio.sleep(FILLER_MAX_DURATION_MS / 1000.0)
            if active_filler_handle.get("is_filler"):
                _interrupt_filler()
                if LATENCY_DEBUG:
                    logger.debug(f"[FILLER] Auto-interrupted after {FILLER_MAX_DURATION_MS}ms timeout")
        
        except asyncio.CancelledError:
            # FIX 5: Ensure cleared on cancel
            logger.debug("[FILLER] Cancelled due to interruption")
            state.filler_active = False
            pass
        except Exception as e:
            logger.debug(f"[FILLER] Could not send filler: {e}")
            state.filler_active = False
        finally:
            active_filler_handle["is_filler"] = False
            # FIX 5: Ensure state is cleared when done
            if state.filler_active:
                logger.debug("[FILLER] Completed and cleared")
                state.filler_active = False
                state.filler_turn_id = None

    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 SLOT VALUE DETECTION — Detect direct answers to skip filler
    # ═══════════════════════════════════════════════════════════════════════════
    # Patterns that indicate user is giving a direct slot value (name, time, phone, email)
    SLOT_VALUE_PATTERNS = [
        r"^(?:it'?s|my name is|i'?m|this is)\s+\w+",  # Name patterns
        r"^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$",  # Phone number
        r"^(?:\+\d{1,3}[-.\s]?)?\d{10,}$",  # International phone
        r"^(?:\+\d{1,3}[-.\s]?)?\d{10,}$",  # International phone
        r"^\S+@\S+\.\S+$",  # Email pattern
        r"^(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?",  # Time patterns
        r"^(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",  # Day patterns
        r"^(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",  # Next day
        r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+",  # Date
    ]
    SLOT_VALUE_RE = re.compile("|".join(SLOT_VALUE_PATTERNS), re.IGNORECASE)
    
    def _is_direct_slot_value(text: str) -> bool:
        """Check if user input looks like a direct answer to a slot question."""
        text = text.strip().lower()
        # Short direct answers (1-3 words) that aren't questions
        words = text.split()
        if len(words) <= 3 and not text.endswith("?"):
            if SLOT_VALUE_RE.search(text):
                return True
        return False
    
    # @session.on("user_input_transcribed")
    def _on_user_transcribed_filler(ev):
        """
        SYNC callback - spawns async task for filler.
        LiveKit .on() requires sync callbacks; async work via create_task.
        
        FILLER SUPPRESSION RULES (for low latency):
        1. Skip for yes/no confirmations (handled deterministically)
        2. Skip for micro-confirmations (< 3 words)
        3. Skip for direct slot values (time, date, phone, email, name)
        4. Skip if filler is disabled via FILLER_ENABLED
        """
        # Mark user end-of-utterance for latency tracking
        _turn_metrics.mark("user_eou")
        
        # Only act on final transcriptions
        if not getattr(ev, 'is_final', True):
            return
        
        transcript = getattr(ev, 'transcript', '') or getattr(ev, 'text', '') or ''
        if not transcript.strip():
            return
            
        current_turn = state.turn_count
        
        # FIX 5: Guard - One filler per turn
        if state.filler_active:
            if LATENCY_DEBUG: logger.debug(f"[FILLER] Skipped: filler already active")
            return
        if state.filler_turn_id == str(current_turn):
            if LATENCY_DEBUG: logger.debug(f"[FILLER] Skipped: filler already scheduled for this turn")
            return
        
        # Check if filler is globally disabled
        if not FILLER_ENABLED:
            _turn_metrics.set_filler_info(False, "disabled")
            return
        
        transcript_lower = transcript.strip().lower()
        word_count = len(transcript_lower.split())
        
        # RULE 1: Skip filler for yes/no confirmations (handled deterministically)
        if word_count <= 2:
            if YES_PAT.search(transcript_lower) or NO_PAT.search(transcript_lower):
                _turn_metrics.set_filler_info(False, "yes_no")
                return
        
        # RULE 2: Skip filler for micro-confirmations (very short responses)
        if word_count <= 2 and not transcript_lower.endswith("?"):
            _turn_metrics.set_filler_info(False, "micro_confirm")
            return
        
        # RULE 3: Skip filler for direct slot values (instant LLM response expected)
        if _is_direct_slot_value(transcript):
            _turn_metrics.set_filler_info(False, "direct_slot")
            return
        
        # RULE 4: Skip if already speaking
        if active_filler_handle.get("is_filler"):
            _turn_metrics.set_filler_info(False, "already_speaking")
            return
            
        # Select a short filler phrase (< 400ms spoken duration)
        import random
        filler = random.choice(FILLER_PHRASES)

        state.filler_active = True
        state.filler_turn_id = str(current_turn)
        logger.debug(f"[FILLER] Triggered for turn {current_turn}: '{filler}'")
        
        _turn_metrics.set_filler_info(True, None)
        
        # Non-blocking: spawn task, don't await
        asyncio.create_task(_send_filler_async(filler))
    
    # @session.on("agent_speech_started")
    def _on_speech_started(ev):
        """
        SYNC callback - interrupt filler when real response starts.
        This ensures filler doesn't overlap with actual content.
        Also marks latency metrics for audio start.
        Captures agent speech handle for true barge-in support.
        """
        # Mark audio start for latency tracking
        _turn_metrics.mark("audio_start")
        
        # Capture agent speech handle for barge-in (try common attribute names)
        speech_handle = getattr(ev, 'handle', None) or getattr(ev, 'speech_handle', None)
        if speech_handle:
            active_agent_handle["handle"] = speech_handle
        
        # Check if this is a real response (not the filler itself)
        speech_text = ""
        try:
            speech_text = getattr(ev, 'text', '') or getattr(ev, 'content', '') or ''
        except:
            pass
        
        # If we have an active filler and this is NOT a filler phrase, interrupt it
        handle = active_filler_handle.get("handle")
        is_filler = active_filler_handle.get("is_filler", False)
        
        # Interrupt if: we have a handle AND (not a filler OR speech doesn't start with filler prefix)
        is_filler_text = speech_text and any(speech_text.strip().startswith(f) for f in FILLER_PHRASES)
        if handle and not is_filler_text:
            _interrupt_filler()
        
        # Log latency metrics for this turn
        if not is_filler_text:
            _turn_metrics.log_turn(extra=f"response_preview='{speech_text[:50]}...'" if len(speech_text) > 50 else f"response='{speech_text}'")

    # ═══════════════════════════════════════════════════════════════════════════
    # 🎤 TRUE BARGE-IN — Interrupt agent immediately when user starts speaking
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _interrupt_agent_speech():
        """Interrupt active agent speech for barge-in."""
        h = active_agent_handle.get("handle")
        if h:
            try:
                if hasattr(h, 'interrupt'):
                    h.interrupt()
                elif hasattr(h, 'cancel'):
                    h.cancel()
                elif hasattr(h, 'stop'):
                    h.stop()
                logger.debug("[BARGE-IN] Agent speech interrupted by user")
            except Exception as e:
                logger.debug(f"[BARGE-IN] Interrupt failed: {e}")
            finally:
                active_agent_handle["handle"] = None
    
    # @session.on("user_speech_started")
    def _on_user_speech_started(ev):
        """True barge-in: stop agent speech when user starts speaking."""
        _interrupt_filler()
        _interrupt_agent_speech()
    
    # Register alternative event names (LiveKit SDK variations)
    try:
        # @session.on("user_started_speaking")
        def _on_user_started_speaking(ev):
            _interrupt_filler()
            _interrupt_agent_speech()
    except Exception:
        pass  # Event may not exist in this SDK version

    # Store references for the refresh callback
    session_ref["session"] = session
    session_ref["agent"] = session
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔄 USER SPEECH EVENT — Refresh memory after each user turn
    # ═══════════════════════════════════════════════════════════════════════════

    def _speech_text_from_msg(msg) -> str:
        for attr in ("text", "content", "text_content", "message"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        try:
            return str(msg).strip()
        except Exception:
            return ""
    
    # @session.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        text = _speech_text_from_msg(msg)
        if text:
            ts = datetime.now().strftime("%H:%M:%S")
            # HIGH-VISIBILITY logging for Railway/production debugging
            logger.info(f"🤖 [AGENT RESPONSE] [{ts}] >> {text}")
            # Also log to debug for detailed tracing
            logger.debug(f"[CONVO] [{ts}] AGENT: {text}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 DETERMINISTIC YES/NO ROUTING — Handle confirmations without LLM
    # ═══════════════════════════════════════════════════════════════════════════
    # This intercepts clear yes/no responses during pending confirmations
    # and routes them directly to confirm_phone/confirm_email tools
    # instead of relying on LLM which can misfire (e.g., confirm_email on "yes")
    
    # @session.on("user_input_transcribed")
    def _on_user_input_confirmation(ev):
        """
        SYNC callback - deterministic routing for yes/no confirmations.
        Spawns async tasks via create_task (required by LiveKit EventEmitter).
        
        When pending_confirm_field is set (e.g., "phone"), intercept
        clear yes/no responses and call the appropriate confirm tool directly.
        This avoids LLM misfires like calling confirm_email(False) on "Yes".
        """
        # Only act on final transcriptions
        if not getattr(ev, 'is_final', True):
            return
        
        transcript = getattr(ev, 'transcript', '') or getattr(ev, 'text', '') or ''
        transcript = transcript.strip().lower()
        
        if not transcript:
            return
        
        # Log user speech
        ts = datetime.now().strftime("%H:%M:%S")
        logger.info(f"👤 [USER INPUT] [{ts}] << {transcript}")
        
        # Check if we have a pending confirmation
        pending = state.pending_confirm_field or state.pending_confirm
        if not pending:
            return  # No pending confirmation, let LLM handle
        
        # Check for clear yes/no patterns
        is_yes = YES_PAT.search(transcript) is not None
        is_no = NO_PAT.search(transcript) is not None
        
        # Only route if it's clearly yes OR clearly no (not both, not neither)
        if is_yes == is_no:
            logger.debug(f"[CONFIRM] Ambiguous response '{transcript}' - letting LLM handle")
            return  # Ambiguous or neither - let LLM handle
        
        logger.info(f"[CONFIRM] Deterministic routing: pending='{pending}', is_yes={is_yes}")
        
        # Async handler for phone confirmation
        async def _handle_phone_confirm_async(confirmed: bool):
            try:
                if confirmed:
                    result = await fnc_ctx.confirm_phone(confirmed=True)
                    logger.info(f"[CONFIRM] Phone confirmed via deterministic routing")
                    # Let agent continue naturally
                    await session.generate_reply()
                else:
                    result = await fnc_ctx.confirm_phone(confirmed=False)
                    logger.info(f"[CONFIRM] Phone rejected via deterministic routing")
                    # Ask for phone again
                    await session.say("No problem! Could you please give me your phone number again?")
            except Exception as e:
                logger.error(f"[CONFIRM] Phone confirm error: {e}")
        
        # Async handler for email confirmation
        async def _handle_email_confirm_async(confirmed: bool):
            try:
                if confirmed:
                    result = await fnc_ctx.confirm_email(confirmed=True)
                    logger.info(f"[CONFIRM] Email confirmed via deterministic routing")
                    await session.generate_reply()
                else:
                    result = await fnc_ctx.confirm_email(confirmed=False)
                    logger.info(f"[CONFIRM] Email rejected via deterministic routing")
                    await session.say("No problem! What's your email address?")
            except Exception as e:
                logger.error(f"[CONFIRM] Email confirm error: {e}")
        
        # Route to appropriate confirm tool (spawn async task - don't await)
        if pending == "phone":
            # Only confirm phone if contact phase has started
            if state.contact_phase_started:
                asyncio.create_task(_handle_phone_confirm_async(is_yes))
            else:
                logger.debug("[CONFIRM] Phone confirm blocked - contact phase not started")
                return
        elif pending == "email":
            # Skip if email already confirmed (idempotent guard)
            if state.email_confirmed:
                logger.debug("[CONFIRM] Email confirm skipped - already confirmed")
                state.pending_confirm = None
                state.pending_confirm_field = None
                return
            asyncio.create_task(_handle_email_confirm_async(is_yes))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📞 SIP PARTICIPANT EVENT — Handle late-joining SIP participants
    # ═══════════════════════════════════════════════════════════════════════════
    
    @ctx.room.on("participant_connected")
    def _on_participant_joined(p: rtc.RemoteParticipant):
        """
        Handle SIP participants that join after initial connection.
        Auto-capture caller phone from SIP metadata for zero-ask booking.
        """
        if p.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            sip_attrs = p.attributes or {}
            caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
            late_called_num = sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
            late_called_num = _normalize_sip_user_to_e164(late_called_num)
            
            logger.info(f"📞 [SIP EVENT] Participant joined: {p.identity}")
            
            # Pre-fill phone if not already captured
            if caller_phone and not state.phone_e164:
                clean_phone, last4 = _normalize_phone_preserve_plus(caller_phone, clinic_region)
                if clean_phone:
                    state.phone_e164 = str(clean_phone)  # Enforce string type
                    state.phone_last4 = str(last4) if last4 else ""
                    # Safety guard
                    _ensure_phone_is_string(state)
                    state.phone_confirmed = False  # NEVER auto-confirm - always ask user
                    state.phone_source = "sip"
                    state.pending_confirm = "phone"
                    state.pending_confirm_field = "phone"
                    speakable = speakable_phone(state.phone_e164)
                    logger.info(f"📞 [SIP EVENT] ⏳ Phone pre-filled (needs confirmation): {speakable}")
                    # Refresh agent memory so it knows phone needs confirmation
                    refresh_agent_memory()

            # Late dialed-number metadata is common; refresh context if we started with a fallback.
            if late_called_num and used_fallback_called_num:
                logger.info(f"📞 [SIP EVENT] ✓ Late called number detected: {late_called_num}")
                # Fire-and-forget context refresh
                async def _refresh_context():
                    nonlocal clinic_info, agent_info, settings, agent_name
                    nonlocal clinic_name, clinic_tz, clinic_region, agent_lang
                    try:
                        ci, ai, st, an = await fetch_clinic_context_optimized(late_called_num)
                        clinic_info, agent_info, settings, agent_name = ci, ai, st, (an or agent_name)
                        globals()["_GLOBAL_CLINIC_INFO"] = clinic_info
                        globals()["_GLOBAL_AGENT_SETTINGS"] = settings
                        globals()["_GLOBAL_SCHEDULE"] = load_schedule_from_settings(settings or {})
                        clinic_name = (clinic_info or {}).get("name") or clinic_name
                        clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
                        clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
                        agent_lang = (agent_info or {}).get("default_language") or agent_lang
                        state.tz = clinic_tz
                        globals()["_GLOBAL_CLINIC_TZ"] = clinic_tz
                        refresh_agent_memory()
                    except Exception as e:
                        logger.warning(f"[DB] Late context refresh failed: {e}")

                asyncio.create_task(_refresh_context())
    
    # Start the agent
    agent = session
    agent.start(ctx.room, participant)

    # Say greeting ASAP (don't await; let TTS start immediately)
    session.say(greeting)

    # ═══════════════════════════════════════════════════════════════════════════
    # 🔄 DEFERRED CONTEXT LOAD — Only if 2s timeout was exceeded
    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: With 2s timeout above, this rarely triggers. It's a safety net.
    if context_task and not context_task.done():
        try:
            clinic_info, agent_info, settings, agent_name = await context_task
            logger.info(f"[DB] ✓ Deferred context loaded: {clinic_info.get('name') if clinic_info else 'None'}")

            _GLOBAL_CLINIC_INFO = clinic_info
            _GLOBAL_AGENT_SETTINGS = settings

            clinic_name = (clinic_info or {}).get("name") or clinic_name
            clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
            clinic_region = (clinic_info or {}).get("default_phone_region") or clinic_region
            agent_lang = (agent_info or {}).get("default_language") or agent_lang

            state.tz = clinic_tz
            _GLOBAL_CLINIC_TZ = clinic_tz
            _GLOBAL_SCHEDULE = load_schedule_from_settings(settings or {})

            refresh_agent_memory()

            # Send proper greeting now that we have context (only if we didn't have it before)
            followup = (settings or {}).get("greeting_text") or (
                f"Hi, I'm {agent_name} from {clinic_name}. How can I help you today?"
            )
            if followup:
                asyncio.create_task(session.say(followup))

        except Exception as e:
            logger.warning(f"[DB] Deferred context load failed: {e}")
    
    # Shutdown
    async def _on_shutdown():
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Ended after {dur}s, booking={state.booking_confirmed}")
        
        try:
            if clinic_info:
                # Map to valid Supabase enum: booked, info_only, missed, transferred, voicemail
                outcome = map_call_outcome(
                    raw_outcome=None,
                    booking_made=state.booking_confirmed,
                )
                
                # Build call session payload with proper schema (no called_number column)
                call_session_payload = {
                    "organization_id": clinic_info["organization_id"],
                    "clinic_id": clinic_info["id"],
                    "caller_phone_masked": f"***{state.phone_last4}" if state.phone_last4 else "Unknown",
                    "caller_name": state.full_name,
                    "outcome": outcome,
                    "duration_seconds": dur,
                }
                
                # Add agent_id if available
                if agent_info and agent_info.get("id"):
                    call_session_payload["agent_id"] = agent_info["id"]
                
                await asyncio.to_thread(
                    lambda: supabase.table("call_sessions").insert(call_session_payload).execute()
                )
                logger.info(f"[DB] ✓ Call session saved: outcome={outcome}")
        except Exception as e:
            logger.error(f"[DB] Call session error: {e}")
        
        try:
            print(f"[USAGE] {usage.get_summary()}")
        except Exception:
            pass
    
    ctx.add_shutdown_callback(_on_shutdown)
    
    # Wait for disconnect
    disconnect_event = asyncio.Event()
    
    @ctx.room.on("disconnected")
    def _():
        disconnect_event.set()
    
    @ctx.room.on("participant_disconnected")
    def _(p):
        disconnect_event.set()
    
    try:
        await asyncio.wait_for(disconnect_event.wait(), timeout=7200)
    except asyncio.TimeoutError:
        pass


# =============================================================================

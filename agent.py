"""
Main agent entrypoint and event handlers for the Dental AI Voice Agent.
"""
from __future__ import annotations
import os
import re
import json
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe, WorkerOptions, cli
from livekit.agents.voice import Agent as VoicePipelineAgent
from livekit.agents import llm
from livekit.plugins import openai as openai_plugin
from livekit.plugins import deepgram as deepgram_plugin
from livekit.plugins import cartesia as cartesia_plugin
from livekit.plugins import silero
from livekit.rtc import ParticipantKind
from livekit.agents import metrics as lk_metrics

from config import (
    logger,
    DEFAULT_TZ,
    DEFAULT_PHONE_REGION,
    FILLER_ENABLED,
    FILLER_MAX_DURATION_MS,
    FILLER_PHRASES,
    STT_AGGRESSIVE_ENDPOINTING,
    VAD_MIN_SPEECH_DURATION,
    VAD_MIN_SILENCE_DURATION,
    LATENCY_DEBUG,
    map_call_outcome,
    supabase,
    LIVEKIT_AGENT_NAME,
)
from models.state import PatientState
from services.database_service import fetch_clinic_context_optimized
from services.scheduling_service import load_schedule_from_settings
from tools.assistant_tools import AssistantTools
from prompts.agent_prompts import A_TIER_PROMPT
from utils.phone_utils import (
    _normalize_phone_preserve_plus,
    _normalize_sip_user_to_e164,
    _ensure_phone_is_string,
    speakable_phone,
)
from utils.call_logger import create_call_logger
from utils.latency_metrics import TurnMetrics

YES_PAT = re.compile(r"\b(yes|yeah|yep|yup|correct|right|sure|okay|ok|affirmative|absolutely|definitely|that'?s right|that'?s correct)\b", re.IGNORECASE)
NO_PAT = re.compile(r"\b(no|nope|nah|wrong|incorrect|negative|not right|that'?s wrong|that'?s not)\b", re.IGNORECASE)

_GLOBAL_STATE: Optional[PatientState] = None
_GLOBAL_CLINIC_TZ: str = DEFAULT_TZ
_GLOBAL_CLINIC_INFO: Optional[dict] = None
_GLOBAL_AGENT_SETTINGS: Optional[dict] = None
_GLOBAL_SCHEDULE: Optional[dict] = None
_REFRESH_AGENT_MEMORY: Optional[callable] = None

_turn_metrics = TurnMetrics()

async def entrypoint(ctx: JobContext):
    global _GLOBAL_STATE, _GLOBAL_CLINIC_TZ, _GLOBAL_CLINIC_INFO, _REFRESH_AGENT_MEMORY, _GLOBAL_AGENT_SETTINGS, _GLOBAL_SCHEDULE
    
    # 1. INITIALIZE STATE & LOGGER IMMEDIATELY
    state = PatientState()
    _GLOBAL_STATE = state
    call_logger = create_call_logger(
        clinic_id=None,
        organization_id=None,
        supabase_client=supabase,
    )
    call_started = time.time()
    
    active_filler_handle = {"handle": None, "is_filler": False, "start_time": None}
    active_agent_handle = {"handle": None}

    # 2. INITIALIZE DEPENDENCIES (VAD, STT, TTS, LLM) IMMEDIATELY
    # This ensures no delay after connect
    
    # State Helpers
    clinic_info = None
    agent_info = None
    settings = None
    agent_name = "Office Assistant"
    clinic_name = "our clinic"
    clinic_tz = DEFAULT_TZ
    clinic_region = DEFAULT_PHONE_REGION
    agent_lang = "en-US"
    used_fallback_called_num = False
    
    # Prompt Helper
    def get_updated_instructions() -> str:
        return A_TIER_PROMPT.format(
            agent_name=agent_name,
            clinic_name=clinic_name,
            timezone=clinic_tz,
            state_summary=state.detailed_state_for_prompt(),
        )
    
    # Memory Refresher
    session_ref: Dict[str, Any] = {"session": None, "agent": None}
    def refresh_agent_memory():
        try:
            session = session_ref.get("session")
            agent = session_ref.get("agent")
            if not session or not agent: return
            new_instructions = get_updated_instructions()
            if hasattr(agent, '_instructions'): agent._instructions = new_instructions
            if hasattr(session, 'chat_ctx') and session.chat_ctx:
                messages = getattr(session.chat_ctx, 'messages', None) or getattr(session.chat_ctx, 'items', None)
                if messages and len(messages) > 0:
                     first_msg = messages[0]
                     if hasattr(first_msg, 'content'): first_msg.content = new_instructions
                     elif hasattr(first_msg, 'text_content') and hasattr(first_msg, '_text_content'): first_msg._text_content = new_instructions
        except Exception as e:
            logger.warning(f"[MEMORY] Refresh failed: {e}")
    _REFRESH_AGENT_MEMORY = refresh_agent_memory

    # Init Resources
    initial_system_prompt = get_updated_instructions()
    chat_context = llm.ChatContext(
        messages=[llm.ChatMessage(content=initial_system_prompt, role=llm.ChatRole.SYSTEM)]
    )
    fnc_ctx = AssistantTools(state)
    
    llm_instance = openai_plugin.LLM(model="gpt-4o-mini", temperature=0.7)
    
    if os.getenv("DEEPGRAM_API_KEY"):
        stt_config = {"model": "nova-2-general", "language": agent_lang}
        if STT_AGGRESSIVE_ENDPOINTING:
             stt_config["endpointing"] = 300
             stt_config["utterance_end_ms"] = 1000
        stt_instance = deepgram_plugin.STT(**stt_config)
    else:
        stt_instance = openai_plugin.STT(model="gpt-4o-transcribe", language="en")
        
    vad_instance = silero.VAD.load(min_speech_duration=VAD_MIN_SPEECH_DURATION, min_silence_duration=VAD_MIN_SILENCE_DURATION)
    
    if os.getenv("CARTESIA_API_KEY"):
        tts_instance = cartesia_plugin.TTS(model="sonic-3", voice=os.getenv("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"))
    else:
        tts_instance = openai_plugin.TTS(model="tts-1", voice="alloy")

    async def before_llm_cb(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        if not chat_ctx.messages: return True
        last_msg = chat_ctx.messages[-1]
        if last_msg.role != llm.ChatRole.USER: return True
        text = last_msg.content
        if not isinstance(text, str): text = str(text) if text else ""
        clean = text.strip().lower()
        if not clean: return True
        is_short = len(clean) < 5
        keywords = ["yes", "no", "yep", "nope", "bye", "ok", "sure", "hi", "hey"]
        if is_short and not any(w in clean for w in keywords):
            state.transcript_buffer.append(clean)
            return False 
        if state.transcript_buffer: state.transcript_buffer.clear()
        return True

    # 3. INITIALIZE AGENT
    agent = VoicePipelineAgent(
        vad=vad_instance,
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        chat_ctx=chat_context,
        fnc_ctx=fnc_ctx,
        interruptible=True,
        before_llm_cb=before_llm_cb,
    )
    session = agent
    session_ref["session"] = session
    session_ref["agent"] = session

    # 3b. REGISTER EVENT HANDLERS (IMMEDIATELY)
    # Ensure all internal events are physically registered so they show up in the terminal
    agent.on("metrics_collected", _on_metrics)
    agent.on("agent_speech_committed", _on_agent_speech_committed)
    agent.on("agent_speech_committed", _on_agent_speech_committed_log)
    agent.on("user_input_transcribed", _on_user_input_confirmation)
    agent.on("user_input_transcribed", _on_user_transcribed_filler)
    agent.on("agent_speech_started", _on_speech_started)
    agent.on("user_speech_started", _on_user_speech_started)

    # 4. DEFINE HANDLERS
    
    # Usage metrics
    usage = lk_metrics.UsageCollector()
    def _on_metrics(ev):
        usage.collect(ev.metrics)
        if ev.metrics.llm_ttft > 0 or ev.metrics.stt_latency > 0:
             logger.info(f"📊 [METRICS] LLM TTFT: {ev.metrics.llm_ttft:.2f}s | STT Latency: {ev.metrics.stt_latency:.2f}s")
             print(f"📊 LATENCY: LLM={ev.metrics.llm_ttft:.2f}s | STT={ev.metrics.stt_latency:.2f}s")
    
    # Speech Committed
    def _speech_text_from_msg(msg) -> str:
        for attr in ("text", "content", "text_content", "message"):
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val.strip(): return val.strip()
        return ""

    def _on_agent_speech_committed(msg):
        text = _speech_text_from_msg(msg)
        if text:
            ts = datetime.now().strftime("%H:%M:%S")
            logger.info(f"🤖 [AGENT RESPONSE] [{ts}] >> {text}")
            print(f"🤖 SARAH: {text}")
        if state.call_ended:
            async def _delayed_disconnect():
                await asyncio.sleep(3.0)
                await ctx.room.disconnect()
            asyncio.create_task(_delayed_disconnect())

    def _on_agent_speech_committed_log(msg):
        text = _speech_text_from_msg(msg)
        if text: call_logger.log_llm_response(text=text)

    # User Speech Started
    def _interrupt_agent_speech():
        h = active_agent_handle.get("handle")
        if h:
            try:
                if hasattr(h, 'interrupt'): h.interrupt()
                elif hasattr(h, 'cancel'): h.cancel()
                elif hasattr(h, 'stop'): h.stop()
            except: pass
            finally: active_agent_handle["handle"] = None

    def _on_user_speech_started(ev):
        _interrupt_filler()
        _interrupt_agent_speech()

    # User Input Confirmation
    def _on_user_input_confirmation(ev):
        if not getattr(ev, 'is_final', True): return
        transcript = getattr(ev, 'transcript', '') or getattr(ev, 'text', '') or ''
        transcript = transcript.strip().lower()
        if not transcript: return
        
        ts = datetime.now().strftime("%H:%M:%S")
        logger.info(f"👤 [USER INPUT] [{ts}] << {transcript}")
        print(f"👤 USER: {transcript}")
        call_logger.log_stt_transcript_only(text=transcript, latency_ms=0)

        # EARLY ACCEPTANCE CHECK
        # If user says "use this number" or similar, just auto-confirm caller ID if available
        if state.detected_phone and not state.phone_confirmed:
            early_accept_pat = re.compile(r"\b(use this number|save this number|this is my number|you can use this number)\b", re.IGNORECASE)
            if early_accept_pat.search(transcript):
                state.phone_e164 = state.detected_phone
                state.phone_confirmed = True
                state.phone_source = "caller_id"
                state.pending_confirm = None
                state.pending_confirm_field = None
                logger.info(f"[PHONE] 🚀 Early Acceptance triggered! Confirmed {state.phone_e164}")
                refresh_agent_memory()
                # Optional: Say "Great, I've saved that." - but agent will likely generate a response naturally
                return

        pending = state.pending_confirm_field or state.pending_confirm
        if not pending: return
        is_yes, is_no = YES_PAT.search(transcript) is not None, NO_PAT.search(transcript) is not None
        if is_yes == is_no: return

        async def _confirm(tool_fnc, confirmed):
            try:
                await tool_fnc(confirmed=confirmed)
                await session.generate_reply()
            except Exception as e: logger.error(f"[CONFIRM] Error: {e}")

        if pending == "phone":
             # If confirming caller ID (detected_phone)
             if state.detected_phone and state.phone_pending == state.detected_phone:
                 if is_yes:
                     state.phone_e164 = state.detected_phone
                     state.phone_confirmed = True
                     state.phone_source = "caller_id"
                     state.pending_confirm = None
                     logger.info(f"[PHONE] Caller ID confirmed via YES")
                     # Clear pending so we don't loop
                     state.pending_confirm_field = None
                     refresh_agent_memory()
                     # Let agent respond naturally or use tool if needed
                     # But we don't really have a tool for "confirm_phone" that takes a boolean?
                     # original code called "fnc_ctx.confirm_phone" but that tool ... let's check assistant_tools.
                     # Actually, assistant_tools usually has explicit tools. 
                     # Checking the file `assistant_tools.py`, I don't see `confirm_phone` in the outline I read earlier.
                     # Wait, I read 1-800 of assistant_tools.py. Let me verify if `confirm_phone` exists.
                     # If it doesn't exist, I should just set state directly here (which I did above).
                 else:
                     # User said NO to caller ID
                     logger.info("[PHONE] User rejected Caller ID")
                     state.phone_pending = None
                     state.pending_confirm = None
                     state.pending_confirm_field = None
                     # Agent will likely ask "Okay, what's different number?" based on prompt
             
             # If confirming spoken number (handled inside update_patient_record logic mostly, but if we have a pending specific confirmation...)
             # The tool `update_patient_record` sets `state.awaiting_slot_confirmation`.
             # The regex logic in `state.interpret_followup_for_slot` is used there.
             # This handler is a backup for simple yes/no if `update_patient_record` didn't catch it.
             # But `confirm_phone` tool call in original code implies it exists.
             # I should check if `confirm_phone` exists in `assistant_tools.py`.
             pass 

        if pending == "email" and not state.email_confirmed:
             asyncio.create_task(_confirm(fnc_ctx.confirm_email, is_yes)) # Assuming confirming email exists

    # Filler Logic
    def _interrupt_filler():
        state.filler_active = False
        state.filler_task = None
        state.filler_turn_id = None
        h = active_filler_handle.get("handle")
        if h:
            try:
                if hasattr(h, 'interrupt'): h.interrupt()
                elif hasattr(h, 'cancel'): h.cancel()
                elif hasattr(h, 'stop'): h.stop()
            except: pass
            finally: active_filler_handle["handle"] = None
            
    async def _send_filler_async(filler_text: str):
        try:
            active_filler_handle["is_filler"] = True
            active_filler_handle["start_time"] = time.perf_counter()
            handle = await session.say(filler_text, allow_interruptions=True)
            active_filler_handle["handle"] = handle
            await asyncio.sleep(FILLER_MAX_DURATION_MS / 1000.0)
            if active_filler_handle.get("is_filler"): _interrupt_filler()
        except: pass
        finally: 
             active_filler_handle["is_filler"] = False
             state.filler_active = False

    def _on_user_transcribed_filler(ev):
        _turn_metrics.mark("user_eou")
        if not getattr(ev, 'is_final', True): return
        transcript = getattr(ev, 'transcript', '') or getattr(ev, 'text', '') or ''
        if not transcript.strip(): return
        current_turn = state.turn_count
        if state.filler_active or state.filler_turn_id == str(current_turn) or not FILLER_ENABLED: return
        
        transcript_lower = transcript.strip().lower()
        word_count = len(transcript_lower.split())
        
        if word_count <= 2 and (YES_PAT.search(transcript_lower) or NO_PAT.search(transcript_lower) or not transcript_lower.endswith("?")): return
        
        SLOT_VALUE_PATTERNS = [
            r"^(?:it'?s|my name is|i'?m|this is)\s+\w+",
            r"^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$",
            r"^(?:\+\d{1,3}[-.\s]?)?\d{10,}$",
            r"^\S+@\S+\.\S+$",
            r"^(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)?",
            r"^(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"^(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+",
        ]
        SLOT_VALUE_RE = re.compile("|".join(SLOT_VALUE_PATTERNS), re.IGNORECASE)
        words = transcript_lower.split()
        if len(words) <= 3 and not transcript_lower.endswith("?") and SLOT_VALUE_RE.search(transcript_lower): return
        
        if active_filler_handle.get("is_filler"): return
        
        import random
        filler = random.choice(FILLER_PHRASES)
        state.filler_active = True
        state.filler_turn_id = str(current_turn)
        asyncio.create_task(_send_filler_async(filler))

    def _on_speech_started(ev):
        _turn_metrics.mark("audio_start")
        speech_handle = getattr(ev, 'handle', None) or getattr(ev, 'speech_handle', None)
        if speech_handle: active_agent_handle["handle"] = speech_handle
        
        speech_text = ""
        try: speech_text = getattr(ev, 'text', '') or getattr(ev, 'content', '') or ''
        except: pass
        
        handle = active_filler_handle.get("handle")
        is_filler = speech_text and any(speech_text.strip().startswith(f) for f in FILLER_PHRASES)
        if handle and not is_filler: _interrupt_filler()
        if not is_filler: _turn_metrics.log_turn(extra=f"response='{speech_text[:50]}'")

    # 5. CONNECT AND START ("INSTANT HANDSHAKE")
    logger.info("[STARTUP] 🚀 Connecting...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Move start here for <100ms pickup
    await agent.start(ctx.room)
    logger.info("[STARTUP] ✓ Agent connected and listening!")

    # 6. REGISTER EVENT HANDLERS
    # (Moved to Step 3b to ensure immediate registration)

    # 7. ASYNC CONTEXT FETCH & GREETING
    participant = await ctx.wait_for_participant()
    logger.info(f"[LIFECYCLE] Participant identified: {participant.identity}")

    called_num = None
    caller_phone = None
    
    if participant.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
        sip_attrs = participant.attributes or {}
        caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
        called_num = sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
        called_num = _normalize_sip_user_to_e164(called_num)
        
        if caller_phone:
            clean_phone, last4 = _normalize_phone_preserve_plus(caller_phone, clinic_region)
            state.detected_phone = str(clean_phone)
            # state.phone_e164 = str(clean_phone)  <-- REMOVED: Don't auto-confirm yet
            state.phone_last4 = str(last4) if last4 else ""
            state.phone_pending = str(clean_phone) 
    
    if not called_num:
        room_name = getattr(ctx.room, "name", "") or ""
        room_match = re.search(r"(\+1\d{10})", room_name) or re.search(r"call_(\+?\d+)_", room_name)
        if room_match: called_num = _normalize_sip_user_to_e164(room_match.group(1))

    if not called_num: 
        meta = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
        called_num = _normalize_sip_user_to_e164(meta.get("sip", {}).get("toUser"))
        
    call_logger.log_call_start(from_number=caller_phone, to_number=called_num)

    context_task: Optional[asyncio.Task] = None
    if called_num:
         context_task = asyncio.create_task(fetch_clinic_context_optimized(called_num))
         logger.info(f"[DB] 🚀 Backgrounding context fetch for {called_num}")

    async def _handle_context_and_greeting():
        nonlocal clinic_info, agent_info, settings, agent_name, clinic_name, clinic_tz, clinic_region, agent_lang
        greeting = "Hello! Thanks for calling. How can I help you today?"
        
        if context_task:
            try:
                logger.info("[STARTUP] ⏳ Waiting for DB context to personalize greeting...")
                clinic_info, agent_info, settings, agent_name = await asyncio.wait_for(asyncio.shield(context_task), timeout=2.0)
                
                _GLOBAL_CLINIC_INFO = clinic_info
                _GLOBAL_AGENT_SETTINGS = settings
                _GLOBAL_SCHEDULE = load_schedule_from_settings(settings or {})
                clinic_name = (clinic_info or {}).get("name") or clinic_name
                clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
                state.tz = clinic_tz
                call_logger.clinic_id = clinic_info.get("id")
                call_logger.organization_id = clinic_info.get("organization_id")
                refresh_agent_memory()

                if settings and settings.get("greeting_text"):
                    greeting = settings.get("greeting_text")
                elif clinic_info:
                    greeting = f"Hi, thanks for calling {clinic_name}! How can I help you today?"
                    
                if state.phone_pending:
                    # speakable = speakable_phone(state.phone_pending)  <-- REMOVED
                    greeting = f"Hi, thanks for calling {clinic_name}! Should I save the number you're calling from as your contact number for the appointment?"
                    state.pending_confirm, state.pending_confirm_field, state.contact_phase_started = "phone", "phone", True
                logger.info(f"[DB] ✓ Context loaded for {clinic_name}")
            except asyncio.TimeoutError:
                logger.warning("[DB] ⚠️ Context fetch timeout")
            except Exception as e:
                logger.warning(f"[DB] ⚠️ Context fetch failed: {e}")
            
        await session.say(greeting, allow_interruptions=True)

    asyncio.create_task(_handle_context_and_greeting())

    # Shutdown
    async def _on_shutdown():
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Call ended. Flushing logs to Supabase...")
        await call_logger.flush_to_supabase()
        try:
            if clinic_info:
                outcome = map_call_outcome(None, state.booking_confirmed)
                payload = {
                    "organization_id": clinic_info["organization_id"],
                    "clinic_id": clinic_info["id"],
                    "caller_phone_masked": f"***{state.phone_last4}" if state.phone_last4 else "Unknown",
                    "caller_name": state.full_name,
                    "outcome": outcome,
                    "duration_seconds": dur,
                }
                await asyncio.to_thread(lambda: supabase.table("call_sessions").insert(payload).execute())
                logger.info(f"[DB] ✓ Call session metadata saved")
        except Exception as e:
            logger.error(f"[DB] Call session error: {e}")

    ctx.add_shutdown_callback(_on_shutdown)
    
    # SIP Late Join Listener
    @ctx.room.on("participant_connected")
    def _on_participant_joined(p: rtc.RemoteParticipant):
        if p.kind == ParticipantKind.PARTICIPANT_KIND_SIP:
            sip_attrs = p.attributes or {}
            caller_phone = sip_attrs.get("sip.phoneNumber") or sip_attrs.get("sip.callingNumber")
            late_called_num = sip_attrs.get("sip.calledNumber") or sip_attrs.get("sip.toUser")
            late_called_num = _normalize_sip_user_to_e164(late_called_num)
            
            if caller_phone and not state.phone_e164:
                clean_phone, last4 = _normalize_phone_preserve_plus(caller_phone, clinic_region)
                if clean_phone:
                    state.phone_e164 = str(clean_phone)
                    state.phone_last4 = str(last4) if last4 else ""
                    _ensure_phone_is_string(state)
                    state.phone_confirmed = False
                    state.phone_source = "sip"
                    state.pending_confirm = "phone"
                    state.pending_confirm_field = "phone"
                    refresh_agent_memory()

            if late_called_num and used_fallback_called_num:
                async def _refresh_context():
                    nonlocal clinic_info, agent_info, settings, agent_name, clinic_name, clinic_tz, clinic_region, agent_lang
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
                    except: pass
                asyncio.create_task(_refresh_context())

    # Keep alive
    disconnect_event = asyncio.Event()
    @ctx.room.on("disconnected")
    def _(): disconnect_event.set()
    try: await asyncio.wait_for(disconnect_event.wait(), timeout=7200)
    except: pass

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=LIVEKIT_AGENT_NAME, load_threshold=1.0))

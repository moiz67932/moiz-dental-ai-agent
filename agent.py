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
from livekit.agents import (
    JobContext, 
    AutoSubscribe, 
    AgentSession, 
    Agent,
    llm
)
from livekit.plugins import openai as openai_plugin
from livekit.plugins import deepgram as deepgram_plugin
from livekit.plugins import cartesia as cartesia_plugin
from livekit.plugins import silero
from livekit.rtc import ParticipantKind
from livekit.agents import metrics as lk_metrics
import inspect

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
    logger.info(f"[DEBUG] Entrypoint started. JobID: {ctx.job.id if ctx.job else 'Unknown'}")
    logger.info(f"[DEBUG] Initializing dependencies...")
    
    active_filler_handle = {"handle": None, "is_filler": False, "start_time": None}
    active_agent_handle = {"handle": None}

    # 2. INITIALIZE DEPENDENCIES (VAD, STT, TTS, LLM) IMMEDIATELY
    
    # State Helpers
    clinic_info = None
    agent_info = None
    settings = None
    agent_name = "Office Assistant"
    clinic_name = "the dental clinic"
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
            agent_inst = session_ref.get("agent")
            if not agent_inst: return
            new_instructions = get_updated_instructions()
            if hasattr(agent_inst, '_instructions'): 
                agent_inst._instructions = new_instructions
            # Also try public property if available
            elif hasattr(agent_inst, 'instructions'):
                agent_inst.instructions = new_instructions
        except Exception as e:
            logger.warning(f"[MEMORY] Refresh failed: {e}")
    _REFRESH_AGENT_MEMORY = refresh_agent_memory

    # Init Resources
    initial_system_prompt = get_updated_instructions()
    
    # Tools
    assistant_tools = AssistantTools(state)
    function_tools = llm.find_function_tools(assistant_tools)
    
    llm_instance = openai_plugin.LLM(model="gpt-4o-mini", temperature=0.7)
    
    if os.getenv("DEEPGRAM_API_KEY"):
        # Introspection for Deepgram config to avoid "endpointing" error
        stt_base_config = {"model": "nova-2-general", "language": agent_lang}
        
        # Check what arguments deepgram_plugin.STT accepts
        stt_sig = inspect.signature(deepgram_plugin.STT.__init__)
        allowed_params = set(stt_sig.parameters.keys())
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in stt_sig.parameters.values())

        stt_kwargs = stt_base_config.copy()
        
        # Only add aggressive endpointing if supported or if kwargs allowed
        if STT_AGGRESSIVE_ENDPOINTING:
            if "endpointing_ms" in allowed_params or has_kwargs:
                stt_kwargs["endpointing_ms"] = 300
            elif "utterance_end_ms" in allowed_params or has_kwargs:
                 # Fallback/Alternative key if endpointing_ms isn't the one
                 stt_kwargs["utterance_end_ms"] = 300
        
        stt_instance = deepgram_plugin.STT(**stt_kwargs)
    else:
        stt_instance = openai_plugin.STT(model="gpt-4o-transcribe", language="en")
        
    vad_instance = silero.VAD.load(min_speech_duration=VAD_MIN_SPEECH_DURATION, min_silence_duration=VAD_MIN_SILENCE_DURATION)
    
    if os.getenv("CARTESIA_API_KEY"):
        tts_instance = cartesia_plugin.TTS(model="sonic-3", voice=os.getenv("CARTESIA_VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"))
    else:
        tts_instance = openai_plugin.TTS(model="tts-1", voice="alloy")


    # 3. INITIALIZE AGENT & SESSION
    session = AgentSession(
        vad=vad_instance,
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
    )
    
    from livekit.agents.llm import StopResponse  # add this import

    class SnappyAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
            text = getattr(new_message, "text_content", None) or ""
            clean = text.strip().lower()
            if not clean:
                raise StopResponse()

            keywords = {"yes","yeah","yep","yup","no","nope","nah","bye","goodbye","ok","okay","sure","hi","hello","hey"}
            is_short = len(clean) < 5

            if is_short and clean not in keywords:
                state.transcript_buffer.append(clean)
                raise StopResponse()

            if state.transcript_buffer:
                state.transcript_buffer.clear()

    agent = SnappyAgent(
        instructions=initial_system_prompt,
        tools=function_tools,
        allow_interruptions=True,
    )

    
    session_ref["session"] = session
    session_ref["agent"] = agent


    # 4. DEFINE HANDLERS
    
    # Usage metrics
    usage = lk_metrics.UsageCollector()
    # def _on_metrics(ev):
    #     usage.collect(ev.metrics)
    #     if ev.metrics.llm_ttft > 0 or ev.metrics.stt_latency > 0:
    #          logger.info(f"📊 [METRICS] LLM TTFT: {ev.metrics.llm_ttft:.2f}s | STT Latency: {ev.metrics.stt_latency:.2f}s")
    #          print(f"📊 LATENCY: LLM={ev.metrics.llm_ttft:.2f}s | STT={ev.metrics.stt_latency:.2f}s")
    
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

        pending = state.pending_confirm_field or state.pending_confirm
        if not pending: return
        is_yes, is_no = YES_PAT.search(transcript) is not None, NO_PAT.search(transcript) is not None
        if is_yes == is_no: return

        async def _confirm(tool_fnc, confirmed):
            try:
                await tool_fnc(confirmed=confirmed)
                await session.generate_reply()
            except Exception as e: logger.error(f"[CONFIRM] Error: {e}")

        if pending == "phone" and state.contact_phase_started:
            asyncio.create_task(_confirm(assistant_tools.confirm_phone, is_yes))
        elif pending == "email" and not state.email_confirmed:
             asyncio.create_task(_confirm(assistant_tools.confirm_email, is_yes))

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


        
    # ================================
    # EVENT WIRING (VERSION-SAFE)
    # ================================
    emitter = session if hasattr(session, "on") else agent
    
    #emitter.on("metrics_collected", _on_metrics)
    
    emitter.on("user_input_transcribed", lambda ev: (
        _on_user_input_confirmation(ev),
        _on_user_transcribed_filler(ev),
    ))
    
    emitter.on("conversation_item_added", lambda ev: (
        _on_agent_speech_committed(ev.item)
        if getattr(ev.item, "role", None) == "assistant"
        else None,
        _on_agent_speech_committed_log(ev.item)
        if getattr(ev.item, "role", None) == "assistant"
        else None,
    ))
    
    emitter.on("speech_created", lambda ev: (
        active_agent_handle.__setitem__("handle", getattr(ev, "speech_handle", None))
    ))
    
    emitter.on("user_state_changed", lambda ev: (
        _on_user_speech_started(ev)
        if getattr(ev, "new_state", None) == "speaking"
        else None
    ))
    

    # 6. REGISTER EVENT HANDLERS
    # Register handlers on the AGENT (not session) where appropriate for Agent events,
    # or Session if they are session level.
    # Agent 1.3.x: agent.on("user_speech_committed", ...) etc.
    # We will use the agent object for events.
    
    # ✅ LiveKit v1: events come from AgentSession, not Agent
    from livekit.agents import (
        MetricsCollectedEvent,
        UserInputTranscribedEvent,
        ConversationItemAddedEvent,
        SpeechCreatedEvent,
        UserStateChangedEvent,
    )
    
    # @session.on("metrics_collected")
    # def _on_metrics_collected(ev: MetricsCollectedEvent):
    #     usage.collect(ev.metrics)
    #     m = ev.metrics
    #     # (keep your latency debug logic here if you want)
    #     try:
    #         if getattr(m, "llm_ttft", 0) > 0 or getattr(m, "stt_latency", 0) > 0:
    #             logger.info(f"📊 [METRICS] LLM TTFT: {getattr(m,'llm_ttft',0):.2f}s | STT Latency: {getattr(m,'stt_latency',0):.2f}s")
    #             print(f"📊 LATENCY: LLM={getattr(m,'llm_ttft',0):.2f}s | STT={getattr(m,'stt_latency',0):.2f}s")
    #     except Exception:
    #         pass
        
        
    # @session.on("user_input_transcribed")
    # def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
    #     # This replaces your old "user_speech_committed" style event
    #     if not getattr(ev, "is_final", True):
    #         return
    
    #     transcript = (getattr(ev, "transcript", "") or "").strip()
    #     if not transcript:
    #         return
    
    #     ts = datetime.now().strftime("%H:%M:%S")
    #     logger.info(f"👤 [USER INPUT] [{ts}] << {transcript}")
    #     print(f"👤 USER: {transcript}")
    #     call_logger.log_stt_transcript_only(text=transcript, latency_ms=0)
    
    #     # run your existing confirmation + filler handlers
    #     class _Tmp: pass
    #     tmp = _Tmp()
    #     tmp.is_final = True
    #     tmp.transcript = transcript
    #     _on_user_input_confirmation(tmp)
    #     _on_user_transcribed_filler(tmp)
    
    
    # @session.on("conversation_item_added")
    # def _on_conversation_item_added(ev: ConversationItemAddedEvent):
    #     # This fires when a message is committed to chat history (user or agent)
    #     item = ev.item
    #     role = getattr(item, "role", None)
    
    #     if role == "assistant":
    #         _on_agent_speech_committed(item)
    #         _on_agent_speech_committed_log(item)
    
    
    # @session.on("speech_created")
    # def _on_speech_created(ev: SpeechCreatedEvent):
    #     # This is the closest equivalent to "agent_speech_started"
    #     speech_handle = getattr(ev, "speech_handle", None)
    #     if speech_handle:
    #         active_agent_handle["handle"] = speech_handle
    
    
    # @session.on("user_state_changed")
    # def _on_user_state_changed(ev: UserStateChangedEvent):
    #     # Use this to detect user started speaking (replaces user_speech_started)
    #     if getattr(ev, "new_state", None) == "speaking":
    #         _on_user_speech_started(ev)
    

    # 5. CONNECT AND START ("INSTANT HANDSHAKE")
    logger.info("[STARTUP] 🚀 Connecting to Room...")
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        # INSTANT HANDSHAKE: Start immediately after connect
        await session.start(room=ctx.room, agent=agent)
        logger.info(f"🚀 SARAH IS NOW LIVE IN ROOM: {ctx.room.name}")
        logger.info("[STARTUP] ✓ AgentSession started!")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Connection/Start failed: {e}")
        return

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
            state.phone_e164 = str(clean_phone)
            state.phone_last4 = str(last4) if last4 else ""
            state.phone_pending = state.phone_e164 
    
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

    async def _update_context_background(task: asyncio.Task):
         global _GLOBAL_CLINIC_INFO, _GLOBAL_AGENT_SETTINGS, _GLOBAL_SCHEDULE
         nonlocal clinic_info, agent_info, settings, agent_name, clinic_name, clinic_tz, clinic_region, agent_lang
         import tools.assistant_tools as atools

         atools._GLOBAL_STATE = state
         atools._GLOBAL_CLINIC_INFO = clinic_info
         atools._GLOBAL_AGENT_SETTINGS = settings
         atools._GLOBAL_SCHEDULE = load_schedule_from_settings(settings or {})
         atools._REFRESH_AGENT_MEMORY = refresh_agent_memory
         
         try:
             logger.info("[DB] ⏳ Background context fetch started...")
             clinic_info, agent_info, settings, agent_name = await task
             
             _GLOBAL_CLINIC_INFO = clinic_info
             _GLOBAL_AGENT_SETTINGS = settings
             _GLOBAL_SCHEDULE = load_schedule_from_settings(settings or {})
             clinic_name = (clinic_info or {}).get("name") or clinic_name
             clinic_tz = (clinic_info or {}).get("timezone") or clinic_tz
             state.tz = clinic_tz
             call_logger.clinic_id = clinic_info.get("id")
             call_logger.organization_id = clinic_info.get("organization_id")
             refresh_agent_memory()
             logger.info(f"[DB] ✓ Context loaded for {clinic_name} (Persona updated)")
         except Exception as e:         
             logger.error(f"[DB] ❌ Background context fetch failed: {e}")

    if context_task:
        asyncio.create_task(_update_context_background(context_task))

    async def _handle_greeting():
    # If we have a DB context task, wait briefly so we can greet with the right clinic + greeting_text
        if context_task:
            try:
                await asyncio.wait_for(context_task, timeout=2.5)
            except asyncio.TimeoutError:
                pass  # we'll greet with fallback if DB is slow

        # If background updater already populated settings, use DB greeting_text first
        greeting = None
        if settings and settings.get("greeting_text"):
            greeting = settings["greeting_text"]
        elif clinic_name:
            greeting = f"Hello! This is {clinic_name}. How can I help you today?"
        else:
            greeting = "Hello! Thanks for calling. How can I help you today?"

        logger.info(f"[STARTUP] 🗣️ Saying greeting: {greeting}")
        await session.say(greeting, allow_interruptions=True)


    asyncio.create_task(_handle_greeting())

    # Shutdown
    async def _on_shutdown():
        nonlocal clinic_info, agent_info
        dur = int(max(0, time.time() - call_started))
        logger.info(f"[LIFECYCLE] Call ended. Flushing logs to Supabase...")
        await call_logger.flush_to_supabase()
        
        try:
            # If clinic_info isn't populated (short call), try one last-ditch fetch
            if not clinic_info and called_num:
                try:
                    logger.info(f"[DB] 🛡️ Short call detected. Last-ditch context fetch for {called_num}...")
                    clinic_info, agent_info, _, _ = await fetch_clinic_context_optimized(called_num)
                except:
                    pass

            org_id = (clinic_info or {}).get("organization_id")
            clinic_id = (clinic_info or {}).get("id")
            agent_id = (agent_info or {}).get("id") or "fallback_agent"

            if org_id and clinic_id:
                outcome = map_call_outcome(None, state.booking_confirmed)
                payload = {
                    "organization_id": org_id,
                    "clinic_id": clinic_id,
                    "agent_id": agent_id,
                    "caller_phone_masked": f"***{state.phone_last4}" if state.phone_last4 else "Unknown",
                    "caller_name": state.full_name,
                    "outcome": outcome,
                    "duration_seconds": dur,
                }
                await asyncio.to_thread(lambda: supabase.table("call_sessions").insert(payload).execute())
                logger.info(f"[DB] ✓ Call session metadata saved (Agent: {agent_id})")
            else:
                logger.warning("[DB] ⚠ Skipping call_sessions insert: organization_id or clinic_id missing")
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
    def _(reason=None): 
        logger.info(f"[LIFECYCLE] Room disconnected: {reason}")
        disconnect_event.set()
    
    # Wait for disconnect
    await disconnect_event.wait()
    
    # Final cleanup
    logger.info("[LIFECYCLE] Job finished.")

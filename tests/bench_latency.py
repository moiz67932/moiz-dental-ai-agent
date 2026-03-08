"""
Instrumented latency benchmark for three canonical turn types.
Run: python tests/bench_latency.py

Measures the tool layer (pure Python, no network) and annotates
the expected end-to-end latency using known pipeline constants
labelled [PROXY] where the real runtime is required.
"""
import asyncio
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")

import tools.assistant_tools as _m
from tools.assistant_tools import AssistantTools
from models.state import PatientState

TZ = ZoneInfo("America/Los_Angeles")
FUTURE = (
    datetime.now(TZ).replace(hour=10, minute=0, second=0, microsecond=0)
    + timedelta(days=3)
)

# ── minimal globals ───────────────────────────────────────────────────────────
_m._GLOBAL_CLINIC_INFO = {"id": "clinic-bench", "default_phone_region": "US"}
_m._GLOBAL_CLINIC_TZ = "America/Los_Angeles"
_m._GLOBAL_SCHEDULE = {
    "working_hours": {
        k: [{"start": "09:00", "end": "17:00"}]
        for k in ["mon", "tue", "wed", "thu", "fri"]
    },
    "closed_dates": set(),
    "slot_step_minutes": 30,
    "treatment_durations": {"Cleaning": 30},
    "lunch_break": {"start": "13:00", "end": "14:00"},
}
_m._REFRESH_AGENT_MEMORY = None

# ── known pipeline constants (PROXY — from config + empirical ranges) ─────────
VAD_PLUS_STT_MS  = 450   # 0.25s VAD silence + 0.20s Deepgram endpointing
LLM_MS           = 550   # gpt-4o-mini typical for short prompts (~400 token prompt)
TTS_MS           = 200   # Cartesia sonic-3 first-audio latency

REPS = 6
SEP  = "-" * 70


async def _run(coro_fn):
    """Return average wall-clock ms over REPS executions."""
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        await coro_fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench(label, coro_fn):
    times = asyncio.get_event_loop().run_until_complete(_run(coro_fn))
    avg = sum(times) / len(times)
    mn  = min(times)
    mx  = max(times)
    print(f"  {label:<48} avg={avg:5.1f}ms  min={mn:4.1f}  max={mx:4.1f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# TURN 1 — Simple acknowledgment (name + reason save)
# ─────────────────────────────────────────────────────────────────────────────

async def _turn1_name():
    s = PatientState()
    await AssistantTools(s).update_patient_record(name="Alice Brown")


async def _turn1_reason():
    s = PatientState(full_name="Alice Brown")
    await AssistantTools(s).update_patient_record(reason="Cleaning")


async def _turn1_both():
    s = PatientState()
    await AssistantTools(s).update_patient_record(name="Alice Brown", reason="Cleaning")


# ─────────────────────────────────────────────────────────────────────────────
# TURN 2 — Booking happy path
# ─────────────────────────────────────────────────────────────────────────────

async def _turn2_time_check_free():
    """Time validation with mocked free slot."""
    s = PatientState(
        full_name="Carol King", reason="Cleaning", duration_minutes=30,
        phone_pending="+13105550002", phone_last4="0002",
    )
    with patch("tools.assistant_tools.is_slot_free_supabase",
               new_callable=AsyncMock, return_value=True), \
         patch("tools.assistant_tools.parse_datetime_natural",
               return_value={"success": True, "datetime": FUTURE}):
        await AssistantTools(s).update_patient_record(time_suggestion="Monday at 10 AM")


async def _turn2_phone_confirm():
    s = PatientState(
        full_name="Carol King", reason="Cleaning", duration_minutes=30,
        dt_local=FUTURE, time_status="valid",
        phone_pending="+13105550002", phone_last4="0002",
        contact_phase_started=True,
    )
    await AssistantTools(s).confirm_phone(confirmed=True)


async def _turn2_direct_book():
    s = PatientState(
        full_name="Carol King", reason="Cleaning", duration_minutes=30,
        dt_local=FUTURE, time_status="valid", slot_available=True,
        phone_pending="+13105550002", phone_e164="+13105550002",
        phone_last4="0002", phone_confirmed=True, contact_phase_started=True,
    )
    with patch("tools.assistant_tools.book_to_supabase",
               new_callable=AsyncMock, return_value="appt-001"):
        await AssistantTools(s).confirm_and_book_appointment()


async def _turn2_pattern_b_fast_lane():
    """Pattern B fast-lane: confirm_phone → state.is_complete() → book directly."""
    s = PatientState(
        full_name="Carol King", reason="Cleaning", duration_minutes=30,
        dt_local=FUTURE, time_status="valid", slot_available=True,
        phone_pending="+13105550002", phone_last4="0002",
        contact_phase_started=True,
    )
    tools = AssistantTools(s)
    await tools.confirm_phone(confirmed=True)
    # After confirm, state.is_complete() → True; book directly (0 LLM hops)
    with patch("tools.assistant_tools.book_to_supabase",
               new_callable=AsyncMock, return_value="appt-001"):
        await tools.confirm_and_book_appointment()


# ─────────────────────────────────────────────────────────────────────────────
# TURN 3 — Conflict / alternative-slot
# ─────────────────────────────────────────────────────────────────────────────

async def _turn3_conflict():
    s = PatientState(full_name="Dave Rivera", reason="Cleaning", duration_minutes=30)
    alt1 = FUTURE + timedelta(minutes=30)
    alt2 = FUTURE + timedelta(minutes=60)
    with patch("tools.assistant_tools.is_slot_free_supabase",
               new_callable=AsyncMock, return_value=False), \
         patch("tools.assistant_tools.suggest_slots_around",
               new_callable=AsyncMock, return_value=[alt1, alt2]), \
         patch("tools.assistant_tools.parse_datetime_natural",
               return_value={"success": True, "datetime": FUTURE}):
        await AssistantTools(s).update_patient_record(time_suggestion="Monday at 10 AM")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print(SEP)
    print("TURN 1 — Simple acknowledgment (name / reason collection)")
    print("  Tool layer only — no DB, no LLM, no TTS")
    print(SEP)
    t1n = bench("update_patient_record(name=...)", _turn1_name)
    t1r = bench("update_patient_record(reason=...)", _turn1_reason)
    t1b = bench("update_patient_record(name+reason)", _turn1_both)
    print()
    eoe1 = VAD_PLUS_STT_MS + LLM_MS + t1b + TTS_MS
    print(f"  Estimated end-to-end  [PROXY]:")
    print(f"    VAD+STT: ~{VAD_PLUS_STT_MS}ms + LLM: ~{LLM_MS}ms "
          f"+ tool: ~{t1b:.0f}ms + TTS: ~{TTS_MS}ms  =  ~{eoe1:.0f}ms")
    print(f"    Expected range: ~{int(eoe1*0.85)}–{int(eoe1*1.25)}ms")

    print()
    print(SEP)
    print("TURN 2 — Booking happy path")
    print("  (time available → phone confirmed → direct book, 0 extra LLM hops)")
    print(SEP)
    t2tc = bench("time check + slot free  (mocked DB)", _turn2_time_check_free)
    t2pc = bench("confirm_phone(True)", _turn2_phone_confirm)
    t2db = bench("confirm_and_book  (mocked DB insert)", _turn2_direct_book)
    t2fl = bench("Pattern B fast-lane: confirm→book", _turn2_pattern_b_fast_lane)
    print()

    # BEFORE: user says "yes" → Pattern B → confirm_phone → generate_reply
    #         → LLM1 decides to call confirm_and_book → LLM2 re-speaks result
    before_ms = VAD_PLUS_STT_MS + (LLM_MS * 2) + t2fl + TTS_MS
    # AFTER: Pattern B → confirm_phone → state.is_complete() → book → session.say
    after_ms  = VAD_PLUS_STT_MS + t2fl + TTS_MS

    print(f"  BEFORE change (2 extra LLM hops on happy path)  [PROXY]:")
    print(f"    VAD+STT: ~{VAD_PLUS_STT_MS}ms + 2×LLM ~{LLM_MS}ms each "
          f"+ tool ~{t2fl:.0f}ms + TTS ~{TTS_MS}ms  =  ~{before_ms:.0f}ms")
    print()
    print(f"  AFTER change (Pattern B fast-lane, 0 LLM hops)  [PROXY]:")
    print(f"    VAD+STT: ~{VAD_PLUS_STT_MS}ms "
          f"+ tool ~{t2fl:.0f}ms + TTS ~{TTS_MS}ms  =  ~{after_ms:.0f}ms")
    savings = before_ms - after_ms
    pct = savings / before_ms * 100
    print(f"    Savings: ~{savings:.0f}ms  ({pct:.0f}% reduction on this turn)")

    print()
    print(SEP)
    print("TURN 3 — Conflict / alternative-slot turn")
    print("  (slot taken → suggest 2 nearby alternatives)")
    print(SEP)
    t3 = bench("conflict + suggest_slots_around (mocked DB)", _turn3_conflict)
    print()
    # 1 LLM hop (LLM calls update_patient_record), then tool, then TTS
    eoe3 = VAD_PLUS_STT_MS + LLM_MS + t3 + TTS_MS
    print(f"  Estimated end-to-end  [PROXY]:")
    print(f"    VAD+STT: ~{VAD_PLUS_STT_MS}ms + 1×LLM: ~{LLM_MS}ms "
          f"+ tool: ~{t3:.0f}ms + TTS: ~{TTS_MS}ms  =  ~{eoe3:.0f}ms")
    print(f"    Real DB add: +100–300ms (slot_free check + alternatives query)")
    print(f"    Realistic range: ~{int(eoe3)}–{int(eoe3+300)}ms")

    print()
    print(SEP)
    print("CONFIG CHANGES SUMMARY")
    print(SEP)
    rows = [
        ("VAD_MIN_SILENCE_DURATION",    "0.30s",      "0.25s",      "-50ms turn detection"),
        ("MIN_ENDPOINTING_DELAY",       "0.70s (dead)","0.40s",     "-300ms (now wired to AgentSession)"),
        ("MAX_ENDPOINTING_DELAY",       "1.00s (dead)","0.70s",     "-300ms cap"),
        ("Deepgram endpointing",        "300ms",       "200ms",     "-100ms STT finalization"),
        ("Deepgram utterance_end_ms",   "1000ms",      "800ms",     "-200ms inter-utterance gap"),
        ("FILLER_DEBOUNCE_MS",          "250ms",       "120ms",     "-130ms (filler fires sooner)"),
        ("FILLER_MAX_DURATION_MS",      "700ms",       "400ms",     "-300ms (filler yields sooner)"),
        ("Pattern B booking fast-lane", "2 LLM hops",  "0 LLM hops",f"-{savings:.0f}ms on happy path"),
        ("refresh_memory on EOU",       "every utt",   "tool writes","reduce CPU/context churn"),
        ("TurnTimer",                   "none",        "per-turn",  "structured latency log"),
    ]
    print(f"  {'Parameter':<32} {'Before':>15} {'After':>12}   Impact")
    print("  " + "─" * 68)
    for name, before, after, impact in rows:
        print(f"  {name:<32} {before:>15} {after:>12}   {impact}")

    print()
    print(SEP)
    print("EXPECTED LATENCY BY TURN TYPE (user-EOU -> first audio)")
    print(SEP)
    rows2 = [
        ("Simple acknowledgment",  "2000–4000ms", f"~{int(eoe1)}ms", f"~{int(eoe1*0.85)}–{int(eoe1*1.25)}ms"),
        ("Booking happy path (yes+book)", "3000–5000ms", f"~{int(after_ms)}ms", f"~{int(after_ms*0.85)}–{int(after_ms*1.25)}ms"),
        ("Conflict / alt-slot",    "3500–6000ms", f"~{int(eoe3)}ms", f"~{int(eoe3)}–{int(eoe3+300)}ms (+DB)"),
    ]
    print(f"  {'Scenario':<30} {'Before':>15} {'Target':>12}  {'Realistic range':>20}")
    print("  " + "─" * 82)
    for name, before, target, rng in rows2:
        print(f"  {name:<30} {before:>15} {target:>12}  {rng:>20}")
    print()


if __name__ == "__main__":
    main()

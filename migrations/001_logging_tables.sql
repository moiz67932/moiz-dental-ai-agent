-- ============================================================================
-- LOGGING & OBSERVABILITY TABLES FOR LIVEKIT VOICE AGENT
-- Run this in Supabase SQL Editor
-- ============================================================================
-- 
-- This schema supports:
-- - Per-call replay and debugging
-- - Latency breakdowns (STT → LLM → TTS)
-- - Cost estimation (token counts)
-- - QA review and analytics
--
-- Tables:
-- 1. calls - One row per voice call
-- 2. call_events - All events for a call (STT, LLM, TTS, VAD, etc.)
-- 3. call_turns - Aggregated conversation turns for easy querying
-- ============================================================================


-- ============================================================================
-- 1. CALLS TABLE - One row per voice call
-- ============================================================================

CREATE TABLE IF NOT EXISTS calls (
    -- Primary key: UUID for global uniqueness
    call_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign keys to existing tables (optional, based on your schema)
    organization_id UUID,  -- REFERENCES organizations(id) if exists
    clinic_id UUID,        -- REFERENCES clinics(id) if exists
    agent_id UUID,         -- REFERENCES agents(id) if exists
    
    -- Call participants (masked for privacy)
    from_number TEXT,      -- Caller phone (masked: ***XXXX)
    to_number TEXT,        -- Called number / clinic DID (masked: ***XXXX)
    
    -- Timestamps
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    
    -- Call metrics
    duration_seconds INTEGER,
    
    -- How the call ended
    -- Possible values: user_hangup, agent_hangup, timeout, error, completed
    end_reason TEXT,
    
    -- Environment tracking
    environment TEXT DEFAULT 'production',
    job_execution_id TEXT,  -- Cloud Run execution ID
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add comment for documentation
COMMENT ON TABLE calls IS 'Voice call records for observability and analytics';
COMMENT ON COLUMN calls.from_number IS 'Caller phone number (masked for privacy)';
COMMENT ON COLUMN calls.to_number IS 'Called/dialed number - clinic DID (masked)';
COMMENT ON COLUMN calls.end_reason IS 'How call ended: user_hangup, agent_hangup, timeout, error, completed';


-- ============================================================================
-- 2. CALL_EVENTS TABLE - All events for a call
-- ============================================================================

CREATE TABLE IF NOT EXISTS call_events (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign key to calls table
    call_id UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
    
    -- Event type for filtering
    -- Possible values: stt, llm, tts, vad, state_change, tool_call, error, call_start, call_end
    event_type TEXT NOT NULL,
    
    -- Event-specific data stored as JSONB for flexibility
    -- Examples:
    --   STT: {"text": "...", "utterance_index": 1, "audio_duration_ms": 1200}
    --   LLM: {"model": "gpt-4o-mini", "prompt_tokens": 500, "completion_tokens": 100}
    --   TTS: {"text": "...", "voice": "cartesia-sonic", "audio_duration_ms": 1500}
    --   VAD: {"event": "speech_start"} or {"event": "speech_end", "duration_ms": 2000}
    --   state_change: {"state": "phone_confirmed", "value": "***1234"}
    --   tool_call: {"tool": "book_appointment", "success": true, "args": {...}}
    --   error: {"component": "stt", "error": "TimeoutError", "recovered": true}
    payload JSONB NOT NULL,
    
    -- Latency in milliseconds (if applicable)
    latency_ms INTEGER,
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add comment for documentation
COMMENT ON TABLE call_events IS 'All events within a call (STT, LLM, TTS, VAD, tool calls, errors)';
COMMENT ON COLUMN call_events.event_type IS 'Event type: stt, llm, tts, vad, state_change, tool_call, error';
COMMENT ON COLUMN call_events.payload IS 'Event-specific data as JSONB';


-- ============================================================================
-- 3. CALL_TURNS TABLE - Aggregated conversation turns
-- ============================================================================

CREATE TABLE IF NOT EXISTS call_turns (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign key to calls table
    call_id UUID NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
    
    -- Turn ordering
    turn_index INTEGER NOT NULL,
    
    -- Conversation content
    user_text TEXT,   -- What the user said (STT output)
    agent_text TEXT,  -- What the agent said (TTS input)
    
    -- Per-component latency breakdown
    stt_latency_ms INTEGER,  -- Time to transcribe user speech
    llm_latency_ms INTEGER,  -- Time for LLM to generate response
    tts_latency_ms INTEGER,  -- Time for TTS to generate audio
    
    -- Total turn latency (sum of above)
    total_latency_ms INTEGER,
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add comment for documentation
COMMENT ON TABLE call_turns IS 'Aggregated conversation turns with latency breakdown';
COMMENT ON COLUMN call_turns.total_latency_ms IS 'Total latency: STT + LLM + TTS';


-- ============================================================================
-- INDEXES FOR FAST QUERYING
-- ============================================================================

-- Calls table indexes
CREATE INDEX IF NOT EXISTS idx_calls_clinic_id ON calls(clinic_id);
CREATE INDEX IF NOT EXISTS idx_calls_organization_id ON calls(organization_id);
CREATE INDEX IF NOT EXISTS idx_calls_start_time ON calls(start_time);
CREATE INDEX IF NOT EXISTS idx_calls_end_reason ON calls(end_reason);
CREATE INDEX IF NOT EXISTS idx_calls_environment ON calls(environment);

-- Call events table indexes
CREATE INDEX IF NOT EXISTS idx_call_events_call_id ON call_events(call_id);
CREATE INDEX IF NOT EXISTS idx_call_events_event_type ON call_events(event_type);
CREATE INDEX IF NOT EXISTS idx_call_events_created_at ON call_events(created_at);

-- JSONB index for payload queries (e.g., find all errors)
CREATE INDEX IF NOT EXISTS idx_call_events_payload ON call_events USING GIN (payload);

-- Call turns table indexes
CREATE INDEX IF NOT EXISTS idx_call_turns_call_id ON call_turns(call_id);
CREATE INDEX IF NOT EXISTS idx_call_turns_turn_index ON call_turns(call_id, turn_index);


-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_turns ENABLE ROW LEVEL SECURITY;

-- Service role has full access (for backend inserts)
-- Note: This policy allows the service role to perform all operations
CREATE POLICY "Service role full access on calls" 
    ON calls FOR ALL 
    USING (true) 
    WITH CHECK (true);

CREATE POLICY "Service role full access on call_events" 
    ON call_events FOR ALL 
    USING (true) 
    WITH CHECK (true);

CREATE POLICY "Service role full access on call_turns" 
    ON call_turns FOR ALL 
    USING (true) 
    WITH CHECK (true);


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get full call replay (all events for a call)
CREATE OR REPLACE FUNCTION get_call_replay(p_call_id UUID)
RETURNS TABLE (
    event_type TEXT,
    payload JSONB,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.event_type,
        ce.payload,
        ce.latency_ms,
        ce.created_at
    FROM call_events ce
    WHERE ce.call_id = p_call_id
    ORDER BY ce.created_at ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to get latency breakdown for a call
CREATE OR REPLACE FUNCTION get_call_latency_breakdown(p_call_id UUID)
RETURNS TABLE (
    turn_index INTEGER,
    user_text TEXT,
    agent_text TEXT,
    stt_latency_ms INTEGER,
    llm_latency_ms INTEGER,
    tts_latency_ms INTEGER,
    total_latency_ms INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ct.turn_index,
        ct.user_text,
        ct.agent_text,
        ct.stt_latency_ms,
        ct.llm_latency_ms,
        ct.tts_latency_ms,
        ct.total_latency_ms
    FROM call_turns ct
    WHERE ct.call_id = p_call_id
    ORDER BY ct.turn_index ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to estimate call cost based on token usage
CREATE OR REPLACE FUNCTION estimate_call_cost(p_call_id UUID)
RETURNS TABLE (
    total_prompt_tokens INTEGER,
    total_completion_tokens INTEGER,
    estimated_cost_usd DECIMAL(10, 6)
) AS $$
DECLARE
    -- GPT-4o-mini pricing (as of 2024)
    prompt_cost_per_1k DECIMAL := 0.00015;  -- $0.15 per 1M tokens
    completion_cost_per_1k DECIMAL := 0.0006;  -- $0.60 per 1M tokens
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM((payload->>'prompt_tokens')::INTEGER), 0) as total_prompt_tokens,
        COALESCE(SUM((payload->>'completion_tokens')::INTEGER), 0) as total_completion_tokens,
        (
            COALESCE(SUM((payload->>'prompt_tokens')::INTEGER), 0) * prompt_cost_per_1k / 1000 +
            COALESCE(SUM((payload->>'completion_tokens')::INTEGER), 0) * completion_cost_per_1k / 1000
        )::DECIMAL(10, 6) as estimated_cost_usd
    FROM call_events
    WHERE call_id = p_call_id AND event_type = 'llm';
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SAMPLE QUERIES FOR ANALYTICS
-- ============================================================================

-- These are example queries you can run for analytics:

-- 1. Get calls with high latency (total turn > 2 seconds)
-- SELECT c.call_id, c.start_time, AVG(ct.total_latency_ms) as avg_latency
-- FROM calls c
-- JOIN call_turns ct ON c.call_id = ct.call_id
-- GROUP BY c.call_id, c.start_time
-- HAVING AVG(ct.total_latency_ms) > 2000
-- ORDER BY avg_latency DESC;

-- 2. Get error rate by component
-- SELECT 
--     payload->>'component' as component,
--     COUNT(*) as error_count,
--     COUNT(*) FILTER (WHERE payload->>'recovered' = 'true') as recovered_count
-- FROM call_events
-- WHERE event_type = 'error'
-- GROUP BY payload->>'component'
-- ORDER BY error_count DESC;

-- 3. Get average latency breakdown
-- SELECT 
--     AVG(stt_latency_ms) as avg_stt,
--     AVG(llm_latency_ms) as avg_llm,
--     AVG(tts_latency_ms) as avg_tts,
--     AVG(total_latency_ms) as avg_total
-- FROM call_turns;

-- 4. Get tool usage statistics
-- SELECT 
--     payload->>'tool' as tool_name,
--     COUNT(*) as call_count,
--     AVG(latency_ms) as avg_latency,
--     COUNT(*) FILTER (WHERE payload->>'success' = 'true') as success_count
-- FROM call_events
-- WHERE event_type = 'tool_call'
-- GROUP BY payload->>'tool'
-- ORDER BY call_count DESC;


-- ============================================================================
-- DONE
-- ============================================================================
-- 
-- After running this script, you will have:
-- 1. `calls` table for call-level data
-- 2. `call_events` table for all events within calls
-- 3. `call_turns` table for conversation turns with latency breakdown
-- 4. Indexes for fast querying
-- 5. RLS policies for security
-- 6. Helper functions for common queries
--
-- To verify, run:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
-- ============================================================================

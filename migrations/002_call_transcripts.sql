-- ============================================================================
-- CALL TRANSCRIPTS TABLE
-- Run this in Supabase SQL Editor
-- ============================================================================
-- 
-- This table stores complete call transcripts with per-turn latency metrics.
-- Each row represents one utterance (either user speech or agent response).
--
-- ============================================================================


-- ============================================================================
-- 1. CALL_TRANSCRIPTS TABLE - Individual transcript entries
-- ============================================================================

-- Drop table if it exists to ensure clean creation (handles previous partial attempts)
DROP TABLE IF EXISTS call_transcripts CASCADE;

CREATE TABLE call_transcripts (
    -- Primary key
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Call reference (can link to call_sessions or calls table)
    call_id UUID NOT NULL,
    
    -- Turn ordering
    turn_index INTEGER NOT NULL DEFAULT 0,
    
    -- Speaker identification
    speaker TEXT NOT NULL CHECK (speaker IN ('user', 'agent')),
    
    -- The actual transcript text
    text TEXT NOT NULL,
    
    -- Latency metrics (milliseconds)
    latency_ms INTEGER,           -- Total end-to-end latency for this turn
    vad_duration_ms INTEGER,      -- VAD speech duration (user only)
    stt_latency_ms INTEGER,       -- STT processing time (user only)
    llm_latency_ms INTEGER,       -- LLM response generation time (agent only)
    tts_latency_ms INTEGER,       -- TTS audio generation time (agent only)
    
    -- Timestamp when this utterance occurred
    utterance_time TIMESTAMPTZ DEFAULT NOW(),
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add comments for documentation
COMMENT ON TABLE call_transcripts IS 'Complete call transcripts with per-turn latency metrics';
COMMENT ON COLUMN call_transcripts.speaker IS 'Speaker: user or agent';
COMMENT ON COLUMN call_transcripts.latency_ms IS 'Total end-to-end latency for this turn';
COMMENT ON COLUMN call_transcripts.vad_duration_ms IS 'VAD speech duration (user utterances)';
COMMENT ON COLUMN call_transcripts.stt_latency_ms IS 'STT processing time (user utterances)';
COMMENT ON COLUMN call_transcripts.llm_latency_ms IS 'LLM response time (agent utterances)';
COMMENT ON COLUMN call_transcripts.tts_latency_ms IS 'TTS generation time (agent utterances)';


-- ============================================================================
-- 2. INDEXES FOR FAST QUERYING
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_call_transcripts_call_id ON call_transcripts(call_id);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_speaker ON call_transcripts(speaker);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_turn ON call_transcripts(call_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_call_transcripts_utterance_time ON call_transcripts(utterance_time);


-- ============================================================================
-- 3. ROW LEVEL SECURITY (RLS)
-- ============================================================================

ALTER TABLE call_transcripts ENABLE ROW LEVEL SECURITY;

-- Service role has full access (for backend inserts)
CREATE POLICY "Service role full access on call_transcripts" 
    ON call_transcripts FOR ALL 
    USING (true) 
    WITH CHECK (true);


-- ============================================================================
-- 4. HELPER FUNCTION - Get full transcript for a call
-- ============================================================================

CREATE OR REPLACE FUNCTION get_call_transcript(p_call_id UUID)
RETURNS TABLE (
    turn_index INTEGER,
    speaker TEXT,
    text TEXT,
    latency_ms INTEGER,
    utterance_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ct.turn_index,
        ct.speaker,
        ct.text,
        ct.latency_ms,
        ct.utterance_time
    FROM call_transcripts ct
    WHERE ct.call_id = p_call_id
    ORDER BY ct.turn_index ASC, ct.utterance_time ASC;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- 5. HELPER FUNCTION - Get transcript with full latency breakdown
-- ============================================================================

CREATE OR REPLACE FUNCTION get_call_transcript_with_metrics(p_call_id UUID)
RETURNS TABLE (
    turn_index INTEGER,
    speaker TEXT,
    text TEXT,
    vad_duration_ms INTEGER,
    stt_latency_ms INTEGER,
    llm_latency_ms INTEGER,
    tts_latency_ms INTEGER,
    total_latency_ms INTEGER,
    utterance_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ct.turn_index,
        ct.speaker,
        ct.text,
        ct.vad_duration_ms,
        ct.stt_latency_ms,
        ct.llm_latency_ms,
        ct.tts_latency_ms,
        ct.latency_ms as total_latency_ms,
        ct.utterance_time
    FROM call_transcripts ct
    WHERE ct.call_id = p_call_id
    ORDER BY ct.turn_index ASC, ct.utterance_time ASC;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- DONE
-- ============================================================================
-- 
-- After running this script, you will have:
-- 1. `call_transcripts` table for storing complete transcripts
-- 2. Indexes for fast querying
-- 3. RLS policies for security
-- 4. Helper functions for transcript retrieval
--
-- To verify, run:
-- SELECT table_name FROM information_schema.tables WHERE table_name = 'call_transcripts';
-- ============================================================================

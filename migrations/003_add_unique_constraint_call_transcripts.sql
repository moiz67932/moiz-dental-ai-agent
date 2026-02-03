-- ============================================================================
-- MIGRATION: 003_add_unique_constraint_call_transcripts.sql
-- Purpose: Add unique constraint to support ON CONFLICT upserts
-- Run this in Supabase SQL Editor
-- ============================================================================

-- Add unique constraint on (call_id, turn_index)
ALTER TABLE call_transcripts 
ADD CONSTRAINT call_transcripts_call_id_turn_index_key 
UNIQUE (call_id, turn_index);

-- Verify it was added
-- SELECT * FROM pg_indexes WHERE tablename = 'call_transcripts';
